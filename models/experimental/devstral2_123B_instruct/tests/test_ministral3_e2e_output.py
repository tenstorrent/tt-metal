# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Free-running (autoregressive) smoke test: TT vs HuggingFace greedy generation.

Uses a **coding instruct prompt** (chat template), matching ``demo/text_demo.py`` — Devstral-2
is trained for code generation, not raw book text.

Each model feeds its **own** argmax prediction as the next input (no teacher forcing).
The first token mismatch ends the matched prefix; everything after compares divergent
contexts — useful as a smoke test, not as a numerical fidelity metric (see
``test_teacher_forced_accuracy.py`` for that).

Self-contained: caches HF greedy output in a ``.refpt`` on first run, then runs TT
free-running decode and compares sequences.

Run::

    pytest models/experimental/devstral2_123B_instruct/tests/test_ministral3_e2e_output.py -v

Environment overrides::

    DEVSTRAL2_E2E_REF             Path to cached HF free-run ``.refpt``
    DEVSTRAL2_E2E_PROMPT           Coding user message (default: linked-list task)
    DEVSTRAL2_E2E_MAX_NEW_TOKENS   Generated tokens to compare (default 32)
    DEVSTRAL2_MIN_PREFIX_MATCH     Minimum matched prefix length (default 1)
    DEVSTRAL2_NUM_LAYERS           Limit decoder layers (default: all 88)
    DEVSTRAL2_WEIGHT_CACHE_SEQ_LEN Tiled-weight cache key (default 262144); reuse ``seq_256k`` weights
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.experimental.devstral2_123B_instruct.demo.text_demo import (
    _current_pos_to_tt,
    _input_ids_host,
    _input_ids_to_tt,
    _logits_to_torch,
    _round_up,
)
from models.experimental.devstral2_123B_instruct.reference.hf_reference_loader import (
    DEVSTRAL2_MODEL_ID,
    load_devstral2_causal_lm,
)
from models.experimental.devstral2_123B_instruct.tests._devstral_weights import (
    devstral2_weight_cache_seq_len,
    model_prefill_weight_keys,
    require_hf_weights,
    require_text_config,
    resolve_devstral2_weight_cache_path,
)
from models.experimental.devstral2_123B_instruct.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_123B_instruct.tt.tt_ministral3_model import TtMinistral3ForCausalLM
from models.experimental.devstral2_123B_instruct.tt.weight_loading import DEVSTRAL2_LARGE_REPO_ID
from models.tt_transformers.tt.ccl import TT_CCL

_TESTS_DIR = Path(__file__).resolve().parent
_DEFAULT_REFERENCE_FILE = _TESTS_DIR / "reference_outputs" / "devstral2_123b_freerun_coding.refpt"

_DEFAULT_CODING_PROMPT = (
    "Write a Python function to reverse a singly linked list. " "Include a brief docstring and handle empty lists."
)
_REFERENCE_MODE = "free_running_greedy_coding"

_MAX_NEW_TOKENS = int(os.environ.get("DEVSTRAL2_E2E_MAX_NEW_TOKENS", "32"))
_MIN_PREFIX_MATCH = int(os.environ.get("DEVSTRAL2_MIN_PREFIX_MATCH", "1"))


def _resolve_coding_prompt() -> str:
    raw = os.environ.get("DEVSTRAL2_E2E_PROMPT") or os.environ.get("DEVSTRAL2_PROMPT")
    return (raw or _DEFAULT_CODING_PROMPT).strip()


def _encode_coding_prompt(tokenizer) -> torch.Tensor:
    """Tokenize a coding user turn with the model's instruct chat template."""
    prompt = _resolve_coding_prompt()
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return encoded["input_ids"][0].to(torch.long)
    return tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(torch.long)


def _resolve_reference_file() -> Path:
    raw = os.environ.get("DEVSTRAL2_E2E_REF", "").strip()
    return Path(raw) if raw else _DEFAULT_REFERENCE_FILE


@torch.no_grad()
def _hf_greedy_freerun(prompt_tokens: torch.Tensor, *, max_new_tokens: int) -> list[int]:
    """HF greedy autoregression: each step feeds the model's own previous argmax."""
    model = load_devstral2_causal_lm()
    model.eval()
    device = next(model.parameters()).device
    input_ids = prompt_tokens.unsqueeze(0).to(device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return output_ids[0, prompt_tokens.shape[0] :].tolist()


def _generate_freerun_reference_file(
    output_file: Path,
    *,
    max_new_tokens: int = _MAX_NEW_TOKENS,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_text = _resolve_coding_prompt()
    logger.info(
        f"Generating HF free-running reference at {output_file} "
        f"(max_new_tokens={max_new_tokens}, prompt={prompt_text!r})"
    )
    tokenizer = AutoTokenizer.from_pretrained(DEVSTRAL2_MODEL_ID, trust_remote_code=True)
    prompt_tokens = _encode_coding_prompt(tokenizer)
    prefill_len = int(prompt_tokens.shape[0])
    hf_generated = _hf_greedy_freerun(prompt_tokens, max_new_tokens=max_new_tokens)
    data = {
        "prompt_tokens": prompt_tokens.unsqueeze(0).cpu(),
        "hf_generated_tokens": torch.tensor(hf_generated, dtype=torch.long).unsqueeze(0),
        "prompt_text": prompt_text,
        "prefill_len": prefill_len,
        "max_new_tokens": max_new_tokens,
        "model_id": DEVSTRAL2_MODEL_ID,
        "mode": _REFERENCE_MODE,
    }
    torch.save(data, output_file)
    logger.info(f"Saved HF free-running reference to {output_file} (prefill_len={prefill_len})")


def _reference_is_current(reference_file: Path) -> bool:
    if not reference_file.is_file():
        return False
    ref = _load_freerun_reference(reference_file)
    if ref.get("mode") != _REFERENCE_MODE:
        return False
    if ref.get("prompt_text") != _resolve_coding_prompt():
        return False
    if int(ref.get("max_new_tokens", _MAX_NEW_TOKENS)) != _MAX_NEW_TOKENS:
        return False
    return True


def _ensure_freerun_reference() -> Path:
    reference_file = _resolve_reference_file()
    if _reference_is_current(reference_file):
        logger.info(f"Using existing free-run reference: {reference_file}")
        return reference_file
    if reference_file.is_file():
        logger.info(f"Stale free-run reference at {reference_file}; regenerating for coding prompt...")
    else:
        logger.info(f"Free-run reference not found at {reference_file}; generating from HF...")
    try:
        _generate_freerun_reference_file(reference_file)
    except Exception as exc:
        pytest.skip(
            "Could not generate HF free-running reference "
            "(set HF_TOKEN if gated, ensure offload disk space, or pre-generate the .refpt). "
            f"Error: {exc}"
        )
    return reference_file


def _load_freerun_reference(reference_file: Path) -> dict:
    data = torch.load(reference_file, weights_only=False)
    for key in ("prompt_tokens", "hf_generated_tokens"):
        if key not in data:
            raise KeyError(f"{reference_file} must contain {key!r}")
    return data


def _mesh_device_param():
    return {
        "N150": (1, 1),
        "N300": (1, 2),
        "N150x4": (1, 4),
        "P150x4": (1, 4),
        "T3K": (1, 8),
        "TG": (8, 4),
    }.get(os.environ.get("MESH_DEVICE"), (1, 8))


def _test_seq_len(prefill_len: int, max_new_tokens: int, kv_block_size: int) -> int:
    """Logical KV budget for this test (coding prompt prefill + decode); unchanged by weight-cache reuse."""
    need = prefill_len + max_new_tokens + 1
    return max(_round_up(need, kv_block_size), kv_block_size)


def _runtime_max_seq_len(test_seq_len: int, kv_block_size: int) -> int:
    """Device ``max_seq_len``: at least the weight-cache key so RoPE/matmul caches hit ``seq_256k``."""
    cache_seq = devstral2_weight_cache_seq_len()
    return max(_round_up(test_seq_len, kv_block_size), cache_seq)


def _build_model(mesh_device, num_layers: int | None, max_seq_len: int):
    text_cfg = require_text_config()
    full_layers = int(text_cfg.num_hidden_layers)
    n_layers = num_layers or full_layers
    assert 0 < n_layers <= full_layers, f"requested {n_layers} layers, model has {full_layers}"

    weight_cache_path = resolve_devstral2_weight_cache_path(mesh_device, text_cfg, n_layers)

    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=max_seq_len,
        max_batch_size=1,
    )

    base_keys = model_prefill_weight_keys(n_layers)
    want_lm_head = not args.tie_word_embeddings
    try:
        weight_keys = base_keys + (["lm_head.weight"] if want_lm_head else [])
        state_dict = require_hf_weights(weight_keys)
    except Exception:
        if want_lm_head:
            logger.warning("lm_head.weight unavailable on the Hub; falling back to tied embeddings.")
            state_dict = require_hf_weights(base_keys)
        else:
            raise

    tt_ccl = TT_CCL(mesh_device)
    model = TtMinistral3ForCausalLM(
        args,
        mesh_device,
        state_dict,
        tt_ccl,
        num_layers=n_layers,
        weight_cache_path=weight_cache_path,
    )
    return model, args, weight_cache_path


def _argmax_from_prefill_logits(
    logits: ttnn.Tensor,
    mesh_device,
    vocab_size: int,
    *,
    prefill_len: int,
    last_chunk_start: int,
) -> int:
    local_pos = (prefill_len - 1) - last_chunk_start
    logits_torch = _logits_to_torch(logits, mesh_device, vocab_size)
    return int(logits_torch[local_pos].argmax().item())


def _argmax_from_decode_logits(logits: ttnn.Tensor, mesh_device, vocab_size: int) -> int:
    logits_torch = _logits_to_torch(logits, mesh_device, vocab_size)
    return int(logits_torch[0].argmax().item())


@torch.no_grad()
def _run_tt_freerun(
    model: TtMinistral3ForCausalLM,
    args: Devstral2Args,
    mesh_device,
    prompt_tokens: torch.Tensor,
    *,
    max_new_tokens: int,
) -> list[int]:
    """TT greedy autoregression: each decode step feeds TT's own previous argmax."""
    prefill_len = int(prompt_tokens.shape[0])
    kv_block_size = args.kv_block_size
    pad_id = args.pad_token_id if args.pad_token_id is not None else 0

    input_ids = prompt_tokens.unsqueeze(0)
    num_prefill_chunks = max(1, (prefill_len + kv_block_size - 1) // kv_block_size)
    padded_prompt_len = num_prefill_chunks * kv_block_size

    input_ids_padded = torch.full((1, padded_prompt_len), int(pad_id), dtype=torch.long)
    input_ids_padded[:, :prefill_len] = input_ids

    prefill_chunk_dev = _input_ids_to_tt(
        torch.full((1, kv_block_size), int(pad_id), dtype=torch.long),
        mesh_device,
    )

    prefill_logits = None
    for chunk_idx in range(num_prefill_chunks):
        chunk_start = chunk_idx * kv_block_size
        chunk_tokens = input_ids_padded[:, chunk_start : chunk_start + kv_block_size]
        ttnn.copy_host_to_device_tensor(
            _input_ids_host(chunk_tokens, mesh_device),
            prefill_chunk_dev,
        )
        tt_out = model(prefill_chunk_dev, mode="prefill", start_pos=chunk_start)
        if chunk_idx == num_prefill_chunks - 1:
            prefill_logits = tt_out
        else:
            tt_out.deallocate(True)

    assert prefill_logits is not None
    ttnn.synchronize_device(mesh_device)

    last_chunk_start = (num_prefill_chunks - 1) * kv_block_size
    next_token = _argmax_from_prefill_logits(
        prefill_logits,
        mesh_device,
        args.vocab_size,
        prefill_len=prefill_len,
        last_chunk_start=last_chunk_start,
    )
    prefill_logits.deallocate(True)

    generated = [next_token]
    decode_tok_dev = _input_ids_to_tt(torch.zeros((1, 1), dtype=torch.long), mesh_device)
    current_pos = prefill_len

    for _ in range(max_new_tokens - 1):
        ttnn.copy_host_to_device_tensor(
            _input_ids_host(torch.tensor([[next_token]], dtype=torch.long), mesh_device),
            decode_tok_dev,
        )
        decode_pos_dev = _current_pos_to_tt(torch.tensor([current_pos], dtype=torch.long), mesh_device)
        decode_logits = model(decode_tok_dev, mode="decode", current_pos=decode_pos_dev)
        next_token = _argmax_from_decode_logits(decode_logits, mesh_device, args.vocab_size)
        decode_logits.deallocate(True)
        generated.append(next_token)
        current_pos += 1

    ttnn.synchronize_device(mesh_device)
    return generated


def _matched_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _log_freerun_comparison(
    tokenizer,
    prompt_tokens: torch.Tensor,
    hf_tokens: list[int],
    tt_tokens: list[int],
    *,
    ref_prompt_text: str | None = None,
) -> None:
    prefix = _matched_prefix_len(hf_tokens, tt_tokens)
    prompt_text = ref_prompt_text or tokenizer.decode(prompt_tokens.tolist())
    logger.info(f"Coding prompt ({len(prompt_tokens)} tokens): {prompt_text!r}")
    logger.info(f"Matched prefix length: {prefix} / {min(len(hf_tokens), len(tt_tokens))}")
    if prefix < len(hf_tokens) and prefix < len(tt_tokens):
        logger.info(
            f"First divergence at step {prefix}: "
            f"HF={hf_tokens[prefix]!r} ({tokenizer.decode([hf_tokens[prefix]])!r}) vs "
            f"TT={tt_tokens[prefix]!r} ({tokenizer.decode([tt_tokens[prefix]])!r})"
        )

    logger.info(f"HF generated tokens ({len(hf_tokens)}): {hf_tokens}")
    logger.info(f"TT generated tokens ({len(tt_tokens)}): {tt_tokens}")
    logger.info("HF generated text: %r", tokenizer.decode(hf_tokens, skip_special_tokens=True))
    logger.info("TT generated text: %r", tokenizer.decode(tt_tokens, skip_special_tokens=True))

    logger.info(f"{'Step':<6}{'Match':<6}{'HF id':<10}{'TT id':<10}{'HF':<20}{'TT':<20}")
    logger.info("-" * 72)
    for i in range(min(len(hf_tokens), len(tt_tokens))):
        match = "x" if hf_tokens[i] == tt_tokens[i] else " "
        hf_text = tokenizer.decode([hf_tokens[i]])[:18]
        tt_text = tokenizer.decode([tt_tokens[i]])[:18]
        logger.info(f"{i:<6}{match:<6}{hf_tokens[i]:<10}{tt_tokens[i]:<10}{hf_text!r:<20}{tt_text!r:<20}")


@pytest.mark.slow
@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30_000_000,
            "num_command_queues": 1,
            "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE,
        }
    ],
    indirect=True,
)
def test_ministral3_e2e_freerun_output(mesh_device):
    """Free-running greedy decode: TT vs cached HF sequence (smoke test, prefix match)."""
    reference_file = _ensure_freerun_reference()
    ref = _load_freerun_reference(reference_file)

    prompt_tokens = ref["prompt_tokens"][0]
    hf_generated = ref["hf_generated_tokens"][0].tolist()
    prefill_len = int(ref.get("prefill_len", prompt_tokens.shape[0]))
    max_new_tokens = int(ref.get("max_new_tokens", len(hf_generated)))
    assert int(prompt_tokens.shape[0]) == prefill_len

    raw_layers = os.environ.get("DEVSTRAL2_NUM_LAYERS", "").strip()
    num_layers = int(raw_layers) if raw_layers else None

    kv_block_size = Devstral2Args.kv_block_size
    test_seq_len = _test_seq_len(prefill_len, max_new_tokens, kv_block_size)
    max_seq_len = _runtime_max_seq_len(test_seq_len, kv_block_size)
    logger.info(
        f"Free-running e2e: prefill={prefill_len}, max_new_tokens={max_new_tokens}, "
        f"test_seq_len={test_seq_len}, max_seq_len={max_seq_len}, reference={reference_file}"
    )

    model, args, weight_cache_path = _build_model(mesh_device, num_layers, max_seq_len)
    logger.info(f"TT weight cache (reuse): {weight_cache_path}")

    tt_generated = _run_tt_freerun(
        model,
        args,
        mesh_device,
        prompt_tokens,
        max_new_tokens=max_new_tokens,
    )

    tokenizer = AutoTokenizer.from_pretrained(DEVSTRAL2_LARGE_REPO_ID, trust_remote_code=True)
    _log_freerun_comparison(
        tokenizer,
        prompt_tokens,
        hf_generated,
        tt_generated,
        ref_prompt_text=ref.get("prompt_text"),
    )

    prefix = _matched_prefix_len(hf_generated, tt_generated)
    logger.info(f"Free-running matched prefix: {prefix}/{max_new_tokens} " f"(minimum required: {_MIN_PREFIX_MATCH})")
    assert prefix >= _MIN_PREFIX_MATCH, (
        f"Matched prefix length {prefix} below smoke-test minimum {_MIN_PREFIX_MATCH}. "
        "Use test_teacher_forced_accuracy.py for per-step fidelity; free-running diverges on first tiebreak."
    )
