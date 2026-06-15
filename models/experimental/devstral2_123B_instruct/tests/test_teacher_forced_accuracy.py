# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced top-1 / top-5 token accuracy vs HuggingFace for Devstral-2-123B.

Self-contained tests generate ``.refpt`` HF references on first run (if missing), then run TT
teacher-forced decode and score against HF top-1 / top-5.

Tests:

- ``test_devstral2_teacher_forced_accuracy`` — 1K reference (512 prefill + 500 eval)
- ``test_devstral2_teacher_forced_accuracy_16k`` — long-context prefill (default 16K) + 500 eval

Run::

    pytest models/experimental/devstral2_123B_instruct/tests/test_teacher_forced_accuracy.py -v
    pytest models/experimental/devstral2_123B_instruct/tests/test_teacher_forced_accuracy.py -k 16k -v

    # 32K prefill (corpus supports up to ~192K tokens):
    DEVSTRAL2_TEACHER_PREFILL_LEN=32768 pytest ... -k 16k -v

Environment overrides::

    DEVSTRAL2_TEACHER_REF          Path to 1K ``.refpt`` (default: tests/reference_outputs/…)
    DEVSTRAL2_TEACHER_16K_REF      Path to long-context ``.refpt``
    DEVSTRAL2_TEACHER_PREFILL_LEN  Long-context prefill tokens (default 16384; max ~192K from corpus)
    DEVSTRAL2_MIN_TOP1_ACC         Minimum top-1 fraction (default 0.90)
    DEVSTRAL2_MIN_TOP5_ACC         Minimum top-5 fraction (default 0.96)
    DEVSTRAL2_NUM_LAYERS           Limit decoder layers (default: all 88)
    DEVSTRAL2_TEACHER_TOTAL_LEN    1K reference token count (default 1024)
    DEVSTRAL2_TEACHER_16K_MAX_EVAL Eval tokens after long prefill (default 500)
    DEVSTRAL2_WEIGHT_CACHE_SEQ_LEN Tiled-weight cache key (default 262144); reuse ``seq_256k`` weights
"""

from __future__ import annotations

import bz2
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
_DEFAULT_REFERENCE_FILE = _TESTS_DIR / "reference_outputs" / "devstral2_123b_instruct.refpt"
_DEFAULT_16K_REFERENCE_FILE = _TESTS_DIR / "reference_outputs" / "devstral2_123b_instruct_16k.refpt"
_TALE_OF_TWO_CITIES = _TESTS_DIR.parents[2] / "tt_transformers" / "tests" / "tale-of-two-cities.txt.bz2"

_DEFAULT_TOTAL_LENGTH = int(os.environ.get("DEVSTRAL2_TEACHER_TOTAL_LEN", "1024"))
_DEFAULT_MAX_EVAL_TOKENS = 500
_DEFAULT_TEACHER_PREFILL_LEN = 16 * 1024

_MIN_TOP1_ACC = float(os.environ.get("DEVSTRAL2_MIN_TOP1_ACC", "0.90"))
_MIN_TOP5_ACC = float(os.environ.get("DEVSTRAL2_MIN_TOP5_ACC", "0.96"))


def _teacher_prefill_len() -> int:
    """Long-context teacher-forcing prefill length (tokens before eval region)."""
    return int(os.environ.get("DEVSTRAL2_TEACHER_PREFILL_LEN", str(_DEFAULT_TEACHER_PREFILL_LEN)))


def _teacher_long_context_max_eval() -> int:
    return int(os.environ.get("DEVSTRAL2_TEACHER_16K_MAX_EVAL", "500"))


def _teacher_long_context_total_length() -> int:
    return _teacher_prefill_len() + _teacher_long_context_max_eval()


def _load_corpus_text() -> str:
    if not _TALE_OF_TWO_CITIES.is_file():
        raise FileNotFoundError(f"Corpus not found: {_TALE_OF_TWO_CITIES}")
    with bz2.open(_TALE_OF_TWO_CITIES, "rt", encoding="utf-8") as f:
        return f.read()


def _corpus_token_count(tokenizer) -> int:
    return len(tokenizer.encode(_load_corpus_text(), add_special_tokens=True))


class TokenAccuracy:
    """Record TT argmax predictions while feeding ground-truth reference tokens on the next step."""

    def __init__(
        self,
        reference_tokens: torch.Tensor,
        top5_tokens: torch.Tensor,
        *,
        max_eval_tokens: int = _DEFAULT_MAX_EVAL_TOKENS,
        split_point: int | None = None,
    ) -> None:
        if reference_tokens.ndim == 2:
            reference_tokens = reference_tokens[0]
        if split_point is None:
            split_point = reference_tokens.shape[-1] // 2
        self.input_prompt = reference_tokens[:split_point]
        self.reference_tokens = reference_tokens[split_point:]
        self.top5_tokens = top5_tokens[split_point - 1 :]
        self.max_eval_tokens = min(max_eval_tokens, len(self.reference_tokens))
        self.gt_pos = -1
        self.store_predicted_tokens: list[int] = []

    @property
    def prefill_len(self) -> int:
        return int(self.input_prompt.shape[-1])

    @property
    def eval_len(self) -> int:
        return self.max_eval_tokens

    def collect_predicted_tokens(self, token_id: int) -> int:
        self.store_predicted_tokens.append(int(token_id))
        self.gt_pos += 1
        idx = min(self.gt_pos, len(self.reference_tokens) - 1)
        return int(self.reference_tokens[idx].item())

    def compute_accuracy(self) -> tuple[float, float, int]:
        count_top1 = 0
        count_top5 = 0
        matching_sz = min(len(self.reference_tokens), len(self.store_predicted_tokens), self.max_eval_tokens)
        for i in range(matching_sz):
            pred = self.store_predicted_tokens[i]
            ref_top1 = int(self.top5_tokens[i, 0].item())
            if pred == ref_top1:
                count_top1 += 1
            if pred in self.top5_tokens[i, :].tolist():
                count_top5 += 1
        if matching_sz == 0:
            return 0.0, 0.0, 0
        return count_top1 / matching_sz, count_top5 / matching_sz, matching_sz


def _resolve_16k_reference_file() -> Path:
    raw = os.environ.get("DEVSTRAL2_TEACHER_16K_REF", "").strip()
    return Path(raw) if raw else _DEFAULT_16K_REFERENCE_FILE


def _generate_reference_file(
    output_file: Path,
    *,
    total_length: int = _DEFAULT_TOTAL_LENGTH,
    chunk_size: int = 1024,
    prefill_len: int | None = None,
    require_full_length: bool = False,
) -> None:
    """Tokenize Tale of Two Cities, run HF forward, save ``reference_tokens`` + ``top5_tokens``."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating teacher-forcing reference at {output_file} (total_length={total_length})")
    tokenizer = AutoTokenizer.from_pretrained(DEVSTRAL2_MODEL_ID, trust_remote_code=True)
    corpus_tokens = _corpus_token_count(tokenizer)
    if total_length > corpus_tokens:
        raise ValueError(
            f"Requested {total_length} tokens but Tale of Two Cities yields {corpus_tokens} "
            f"with the Devstral tokenizer (~192K max). Reduce DEVSTRAL2_TEACHER_PREFILL_LEN or "
            "DEVSTRAL2_TEACHER_16K_MAX_EVAL."
        )

    model = load_devstral2_causal_lm()
    model.eval()
    device = next(model.parameters()).device

    encoded_tokens = tokenizer.encode(_load_corpus_text(), add_special_tokens=True)[:total_length]
    if len(encoded_tokens) < total_length:
        if require_full_length:
            raise ValueError(
                f"Corpus yielded {len(encoded_tokens)} tokens (requested {total_length}). "
                f"Maximum available: {corpus_tokens}."
            )
        logger.warning(f"Corpus yielded {len(encoded_tokens)} tokens (requested {total_length}); using all available.")
        total_length = len(encoded_tokens)

    encoded_tokens_tensor = torch.tensor(encoded_tokens, device=device).unsqueeze(0)
    all_top5_tokens: list[torch.Tensor] = []

    with torch.no_grad():
        for chunk_start in range(0, total_length - 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_length)
            chunk_tokens = encoded_tokens_tensor[:, chunk_start:chunk_end]
            chunk_next_tokens = encoded_tokens[chunk_start + 1 : chunk_end + 1]
            actual_chunk_size = min(len(chunk_tokens[0]), len(chunk_next_tokens))
            chunk_tokens = chunk_tokens[:, :actual_chunk_size]

            logits = model(chunk_tokens).logits
            probs = torch.softmax(logits, dim=-1)
            _, chunk_top5_tokens = torch.topk(probs, k=5, dim=-1)
            all_top5_tokens.append(chunk_top5_tokens.squeeze(0).cpu())

    data = {
        "reference_tokens": encoded_tokens_tensor[:, :total_length].clone().cpu(),
        "top5_tokens": torch.cat(all_top5_tokens, dim=0).cpu(),
        "model_id": DEVSTRAL2_MODEL_ID,
        "total_length": total_length,
        "prefill_len": prefill_len if prefill_len is not None else total_length // 2,
    }
    torch.save(data, output_file)
    logger.info(f"Saved reference outputs to {output_file}")


def _ensure_reference_file(
    reference_file: Path,
    *,
    total_length: int,
    expected_prefill_len: int | None = None,
) -> Path:
    """Return path to ``.refpt``, generating it with HF when absent or stale."""
    if reference_file.is_file():
        data = torch.load(reference_file, weights_only=False)
        cached_len = int(data.get("total_length", 0))
        cached_prefill = data.get("prefill_len")
        if cached_len == total_length and (
            expected_prefill_len is None or int(cached_prefill or 0) == expected_prefill_len
        ):
            logger.info(f"Using existing reference: {reference_file}")
            return reference_file
        logger.info(
            f"Stale reference at {reference_file} "
            f"(cached total_length={cached_len}, want {total_length}); regenerating..."
        )
    else:
        logger.info(f"Reference not found at {reference_file}; generating from HF...")

    try:
        _generate_reference_file(
            reference_file,
            total_length=total_length,
            prefill_len=expected_prefill_len,
            require_full_length=expected_prefill_len is not None,
        )
    except Exception as exc:
        pytest.skip(
            "Could not generate teacher-forcing reference "
            "(set HF_TOKEN if gated, ensure offload disk space, or pre-generate the .refpt). "
            f"Error: {exc}"
        )
    return reference_file


def _load_reference_data(reference_file: Path) -> dict:
    logger.info(f"Loading teacher-forcing reference from {reference_file}")
    data = torch.load(reference_file, weights_only=False)
    if "reference_tokens" not in data or "top5_tokens" not in data:
        raise KeyError(f"{reference_file} must contain 'reference_tokens' and 'top5_tokens'")
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


def _resolve_1k_reference_file() -> Path:
    raw = os.environ.get("DEVSTRAL2_TEACHER_REF", "").strip()
    return Path(raw) if raw else _DEFAULT_REFERENCE_FILE


def _test_seq_len(prefill_len: int, eval_tokens: int, kv_block_size: int) -> int:
    """Logical KV budget (prefill + eval); weight tiles still loaded from ``seq_256k`` cache."""
    need = prefill_len + eval_tokens + 1
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
def _run_teacher_forced_accuracy(
    model: TtMinistral3ForCausalLM,
    args: Devstral2Args,
    mesh_device,
    token_acc: TokenAccuracy,
) -> tuple[float, float, int]:
    prefill_len = token_acc.prefill_len
    eval_tokens = token_acc.eval_len
    kv_block_size = args.kv_block_size

    pad_id = args.pad_token_id if args.pad_token_id is not None else 0
    input_ids = token_acc.input_prompt.unsqueeze(0)
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
    predicted = _argmax_from_prefill_logits(
        prefill_logits,
        mesh_device,
        args.vocab_size,
        prefill_len=prefill_len,
        last_chunk_start=last_chunk_start,
    )
    prefill_logits.deallocate(True)
    token_acc.collect_predicted_tokens(predicted)

    decode_tok_dev = _input_ids_to_tt(torch.zeros((1, 1), dtype=torch.long), mesh_device)
    current_pos = prefill_len

    for step in range(1, eval_tokens):
        ref_tok = int(token_acc.reference_tokens[step - 1].item())
        ttnn.copy_host_to_device_tensor(
            _input_ids_host(torch.tensor([[ref_tok]], dtype=torch.long), mesh_device),
            decode_tok_dev,
        )
        decode_pos_dev = _current_pos_to_tt(torch.tensor([current_pos], dtype=torch.long), mesh_device)
        decode_logits = model(decode_tok_dev, mode="decode", current_pos=decode_pos_dev)
        predicted = _argmax_from_decode_logits(decode_logits, mesh_device, args.vocab_size)
        decode_logits.deallocate(True)
        token_acc.collect_predicted_tokens(predicted)
        current_pos += 1

    ttnn.synchronize_device(mesh_device)
    return token_acc.compute_accuracy()


def _log_accuracy_table(token_acc: TokenAccuracy, tokenizer, *, max_rows: int = 20) -> None:
    rows = min(token_acc.eval_len, len(token_acc.store_predicted_tokens), max_rows)
    logger.info(f"{'Step':<8}{'Top1':<6}{'Top5':<6}{'Ref':<20}{'TT':<20}")
    logger.info("-" * 60)
    for i in range(rows):
        pred = token_acc.store_predicted_tokens[i]
        ref_top1 = int(token_acc.top5_tokens[i, 0].item())
        in_top5 = pred in token_acc.top5_tokens[i, :].tolist()
        ref_text = tokenizer.decode([ref_top1])[:18]
        tt_text = tokenizer.decode([pred])[:18]
        logger.info(
            f"{i:<8}{'x' if pred == ref_top1 else ' ':<6}{'x' if in_top5 else ' ':<6}"
            f"{ref_text!r:<20}{tt_text!r:<20}"
        )


_DEVICE_PARAMS = [
    {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": 30_000_000,
        "num_command_queues": 1,
        "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    }
]


def _execute_teacher_forced_test(
    mesh_device,
    *,
    reference_file: Path,
    total_length: int,
    expected_prefill_len: int | None,
    split_point: int | None,
    max_eval_tokens: int,
    test_label: str,
) -> None:
    reference_file = _ensure_reference_file(
        reference_file,
        total_length=total_length,
        expected_prefill_len=expected_prefill_len,
    )
    reference_data = _load_reference_data(reference_file)
    token_acc = TokenAccuracy(
        reference_data["reference_tokens"],
        reference_data["top5_tokens"],
        max_eval_tokens=max_eval_tokens,
        split_point=split_point,
    )

    raw_layers = os.environ.get("DEVSTRAL2_NUM_LAYERS", "").strip()
    num_layers = int(raw_layers) if raw_layers else None

    kv_block_size = Devstral2Args.kv_block_size
    test_seq_len = _test_seq_len(token_acc.prefill_len, token_acc.eval_len, kv_block_size)
    max_seq_len = _runtime_max_seq_len(test_seq_len, kv_block_size)
    logger.info(
        f"{test_label}: prefill={token_acc.prefill_len}, eval={token_acc.eval_len}, "
        f"test_seq_len={test_seq_len}, max_seq_len={max_seq_len}, reference={reference_file}"
    )

    model, args, weight_cache_path = _build_model(mesh_device, num_layers, max_seq_len)
    logger.info(f"TT weight cache (reuse): {weight_cache_path}")

    top1, top5, n_tokens = _run_teacher_forced_accuracy(model, args, mesh_device, token_acc)

    tokenizer = AutoTokenizer.from_pretrained(DEVSTRAL2_LARGE_REPO_ID, trust_remote_code=True)
    _log_accuracy_table(token_acc, tokenizer)

    logger.info(
        f"{test_label} over {n_tokens} tokens: "
        f"top-1={top1:.4f} ({top1 * 100:.2f}%), top-5={top5:.4f} ({top5 * 100:.2f}%)"
    )
    logger.info(f"Thresholds: top-1 >= {_MIN_TOP1_ACC:.2%}, top-5 >= {_MIN_TOP5_ACC:.2%}")

    assert top1 >= _MIN_TOP1_ACC, f"Top-1 accuracy {top1:.4f} below {_MIN_TOP1_ACC:.4f}"
    assert top5 >= _MIN_TOP5_ACC, f"Top-5 accuracy {top5:.4f} below {_MIN_TOP5_ACC:.4f}"


@pytest.mark.slow
@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(14400)
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
def test_devstral2_teacher_forced_accuracy_16k(mesh_device):
    """Long-context prefill (default 16K, env-configurable) + 500 teacher-forced decode vs HF."""
    prefill_len = _teacher_prefill_len()
    max_eval = _teacher_long_context_max_eval()
    total_length = _teacher_long_context_total_length()
    _execute_teacher_forced_test(
        mesh_device,
        reference_file=_resolve_16k_reference_file(),
        total_length=total_length,
        expected_prefill_len=prefill_len,
        split_point=prefill_len,
        max_eval_tokens=max_eval,
        test_label=f"Teacher-forced accuracy ({prefill_len}-token prefill)",
    )
