# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for full-model logit PCC (chunked prefill + HF-greedy decode).

Prefill uses golden corpus tokens (Tale of Two Cities). Decode follows the
tt-transformers ``test_model.py`` pattern: compare raw logits, then advance both
HF and TT with the **HF greedy** (temperature=0 argmax) token.
"""

from __future__ import annotations

import bz2
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_123B_instruct.demo.text_demo import (
    _current_pos_to_tt,
    _input_ids_host,
    _input_ids_to_tt,
    _logits_to_torch,
)
from models.experimental.devstral2_123B_instruct.reference.hf_reference_loader import (
    DEVSTRAL2_MODEL_ID,
    extract_causal_lm_bf16_state_dict,
    load_devstral2_causal_lm,
    load_devstral2_text_config,
    prepare_causal_lm_for_pcc,
    resolve_hf_input_device,
)
from models.experimental.devstral2_123B_instruct.tests._devstral_weights import (
    devstral2_e2e_sweep_model_max_seq_len,
    model_prefill_weight_keys,
    prepare_e2e_tt_model_budget,
    log_hf_hub_weights_status,
)
from models.experimental.devstral2_123B_instruct.tests.decoder_pcc_common import (
    PREFILL_SANITY_SEQ_LENGTHS,
    PREFILL_SWEEP_SEQ_LENGTHS,
)
from models.experimental.devstral2_123B_instruct.tt.model_args import Devstral2Args
from models.experimental.devstral2_123B_instruct.tt.tt_ministral3_model import TtMinistral3ForCausalLM
from models.tt_transformers.tt.ccl import TT_CCL

PCC_REQUIRED = 0.99
DECODE_GENERATION_LENGTH = 10
_SWEEP_TIMEOUT_MARGIN = 1.25
# Default pytest cap for the full 32 … 262144 logit-PCC sweep (override via env).
LOGIT_PCC_FULL_SWEEP_TIMEOUT_SEC = int(os.environ.get("DEVSTRAL2_LOGIT_PCC_SWEEP_TIMEOUT_SEC", str(12 * 3600)))
# Increment when teacher-forced ``.refpt`` layout or HF reference generation changes.
HF_TEACHER_FORCED_REFERENCE_FORMAT_VERSION = 2

_TESTS_DIR = Path(__file__).resolve().parent
_TALE_OF_TWO_CITIES = _TESTS_DIR.parents[2] / "tt_transformers" / "tests" / "tale-of-two-cities.txt.bz2"

# Reuse one HF + TT build per mesh for sanity/sweep runs in the same pytest process.
_logit_pcc_models_cache: dict[tuple[int, int], tuple] = {}


def mesh_device_param() -> tuple[int, int]:
    return {
        "N150": (1, 1),
        "N300": (1, 2),
        "N150x4": (1, 4),
        "P150x4": (1, 4),
        "T3K": (1, 8),
        "TG": (8, 4),
    }.get(os.environ.get("MESH_DEVICE"), (1, 8))


def device_params(l1_small_size: int) -> dict:
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": 30_000_000,
        "num_command_queues": 1,
        "l1_small_size": l1_small_size,
    }


def _estimate_logit_pcc_sweep_seconds(prefill_lengths: list[int]) -> int:
    """Estimate wall time for TT + HF CPU inference (no pytest margin).

    Per sweep point:
    - **HF prefill**: ``prefill_len`` token forwards (chunked 128, CPU/offload).
    - **HF decode**: ``DECODE_GENERATION_LENGTH`` greedy decode forwards (one per step).
    - **TT prefill**: ``num_chunks × tt_prefill_sec_per_chunk``.
    - **TT decode**: ``DECODE_GENERATION_LENGTH × tt_decode_sec_per_step``.

    Calibrated from BH Loudbox 2026-06-17 (HF-greedy logit PCC, weight cache warm):
    ~10 h wall for 14 points; HF CPU is a large share at 64k+ prefill lengths.
    """
    tt_model_setup_sec = 600
    hf_load_once_sec = 600
    hf_forward_sec_per_token = 0.072
    tt_base_sec = 60
    tt_prefill_sec_per_chunk = 39
    tt_decode_sec_per_step = 0.42
    kv_block_size = Devstral2Args.kv_block_size

    budget = tt_model_setup_sec + hf_load_once_sec
    for prefill_len in prefill_lengths:
        num_chunks = (prefill_len + kv_block_size - 1) // kv_block_size
        hf_prefill_tokens = prefill_len
        hf_decode_tokens = DECODE_GENERATION_LENGTH
        budget += int(hf_prefill_tokens * hf_forward_sec_per_token)
        budget += int(hf_decode_tokens * hf_forward_sec_per_token)
        budget += tt_base_sec + num_chunks * tt_prefill_sec_per_chunk
        budget += int(DECODE_GENERATION_LENGTH * tt_decode_sec_per_step)
    return budget


def sweep_timeout_seconds(prefill_lengths: list[int]) -> int:
    """Pytest timeout for a logit-PCC sweep (includes HF CPU + TT, with margin).

    The full ``PREFILL_SWEEP_SEQ_LENGTHS`` sweep defaults to **12 hours** (env
    ``DEVSTRAL2_LOGIT_PCC_SWEEP_TIMEOUT_SEC``), matching observed ~10 h wall on BH
    Loudbox with warm TT weight cache. Shorter sweeps use the estimated budget × 1.25.
    """
    if prefill_lengths == PREFILL_SWEEP_SEQ_LENGTHS:
        return LOGIT_PCC_FULL_SWEEP_TIMEOUT_SEC
    return int(_estimate_logit_pcc_sweep_seconds(prefill_lengths) * _SWEEP_TIMEOUT_MARGIN)


def e2e_sweep_model_max_seq_len() -> int:
    """Fixed KV/RoPE + weight-cache budget shared by logit-PCC and teacher-forced sweeps."""
    return devstral2_e2e_sweep_model_max_seq_len()


def _skip_hf_load_failure(exc: BaseException) -> None:
    pytest.skip(
        "Could not load Devstral-2-123B via Hugging Face "
        "(set HF_TOKEN if gated, ensure offload disk space, or pre-cache weights). "
        f"Error: {exc}"
    )


def _load_corpus_text() -> str:
    if not _TALE_OF_TWO_CITIES.is_file():
        raise FileNotFoundError(f"Corpus not found: {_TALE_OF_TWO_CITIES}")
    with bz2.open(_TALE_OF_TWO_CITIES, "rt", encoding="utf-8") as f:
        return f.read()


def _base_corpus_tokens(tokenizer) -> list[int]:
    return tokenizer.encode(_load_corpus_text(), add_special_tokens=True)


def build_token_sequence(tokenizer, total_length: int) -> list[int]:
    """Return ``total_length`` tokens from Tale of Two Cities, tiling when needed."""
    if total_length <= 0:
        raise ValueError(f"total_length must be positive, got {total_length}")

    base = _base_corpus_tokens(tokenizer)
    if not base:
        raise ValueError("Corpus tokenization produced an empty sequence")

    if total_length <= len(base):
        return base[:total_length]

    repeat_body = base[1:] if len(base) > 1 else base
    tokens = list(base)
    while len(tokens) < total_length:
        need = total_length - len(tokens)
        tokens.extend(repeat_body[:need])
    return tokens[:total_length]


def build_logit_pcc_models(mesh_device, *, max_seq_len: int | None = None):
    """Load HF causal LM and build ``TtMinistral3ForCausalLM`` from extracted bf16 weights."""
    try:
        text_cfg = load_devstral2_text_config()
    except Exception as exc:
        _skip_hf_load_failure(exc)

    num_layers = int(text_cfg.num_hidden_layers)
    budget_seq_len, weight_cache_path = prepare_e2e_tt_model_budget(mesh_device, text_cfg)
    model_max_seq_len = max_seq_len if max_seq_len is not None else budget_seq_len

    log_hf_hub_weights_status(model_prefill_weight_keys(num_layers))
    try:
        causal_lm = load_devstral2_causal_lm()
    except Exception as exc:
        _skip_hf_load_failure(exc)

    prepare_causal_lm_for_pcc(causal_lm)
    hf_device = resolve_hf_input_device(causal_lm)
    state_dict = extract_causal_lm_bf16_state_dict(causal_lm, num_layers)

    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=model_max_seq_len,
        max_batch_size=1,
    )
    tt_ccl = TT_CCL(mesh_device)
    tt_model = TtMinistral3ForCausalLM(
        args,
        mesh_device,
        state_dict,
        tt_ccl,
        num_layers=num_layers,
        weight_cache_path=weight_cache_path,
    )
    logger.info(
        f"Logit PCC models ready: layers={num_layers}, model_max_seq_len={model_max_seq_len}, "
        f"hf_input_device={hf_device}, weight_cache_path={weight_cache_path}"
    )
    return causal_lm, tt_model, args, text_cfg, weight_cache_path


def get_or_build_logit_pcc_models(mesh_device):
    """Return cached HF + TT models for this mesh (one build per pytest process)."""
    mesh_key = tuple(mesh_device.shape)
    if mesh_key not in _logit_pcc_models_cache:
        logger.info(f"Building logit PCC models once for mesh={mesh_key}")
        _logit_pcc_models_cache[mesh_key] = build_logit_pcc_models(mesh_device)
    else:
        logger.info(f"Reusing in-process logit PCC model for mesh={mesh_key}")
    return _logit_pcc_models_cache[mesh_key]


def _logits_row_for_pcc(logits: torch.Tensor) -> torch.Tensor:
    """Normalize to ``[1, vocab]`` for ``comp_pcc``."""
    if logits.ndim == 1:
        return logits.unsqueeze(0)
    if logits.ndim == 2:
        return logits[:1]
    raise ValueError(f"Expected 1D or 2D logits, got shape {logits.shape}")


def assert_logit_pcc(
    ref_logits: torch.Tensor,
    tt_logits: torch.Tensor,
    *,
    label: str,
) -> None:
    ref_row = _logits_row_for_pcc(ref_logits.float().cpu())
    tt_row = _logits_row_for_pcc(tt_logits.float().cpu())
    passing, msg = comp_pcc(ref_row, tt_row, PCC_REQUIRED)
    logger.info(comp_allclose(ref_row, tt_row))
    if passing:
        logger.info(f"{label}: PASS — {msg}")
    else:
        logger.warning(f"{label}: FAIL — {msg}")
    assert passing, f"{label}: PCC below {PCC_REQUIRED}: {msg}"


def _hf_greedy_token_id(logits: torch.Tensor) -> int:
    """Greedy next token from a logits row (temperature=0), matching ``sample_host(..., temperature=0)``."""
    row = _logits_row_for_pcc(logits.float().cpu())
    return int(row.argmax(dim=-1).item())


@torch.no_grad()
def hf_causal_lm_prefill_chunk(
    causal_lm,
    token_ids: list[int],
    *,
    chunk_start: int,
    cache: DynamicCache,
    input_device: torch.device,
) -> torch.Tensor:
    """One HF prefill chunk via ``DynamicCache`` (O(chunk) host memory, same pattern as decoder PCC).

    Only ``len(token_ids)`` tokens are materialized per forward on ``input_device`` (CPU when no GPU).
    """
    chunk = torch.tensor([token_ids], dtype=torch.long, device=input_device)
    if chunk_start == 0:
        out = causal_lm(chunk, past_key_values=cache, use_cache=True)
    else:
        chunk_len = len(token_ids)
        position_ids = torch.arange(
            chunk_start, chunk_start + chunk_len, dtype=torch.long, device=input_device
        ).unsqueeze(0)
        out = causal_lm(
            chunk,
            past_key_values=cache,
            position_ids=position_ids,
            use_cache=True,
        )
    return out.logits


def _top5_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Top-5 token ids from a logits row ``[..., vocab]``; returns ``[5]`` on CPU."""
    probs = torch.softmax(logits.float(), dim=-1)
    _, top5 = torch.topk(probs, k=5, dim=-1)
    return top5.reshape(-1, 5)[0].cpu()


@torch.no_grad()
def hf_teacher_forced_top5_reference(
    causal_lm,
    token_ids: list[int],
    *,
    kv_block_size: int | None = None,
) -> torch.Tensor:
    """Top-5 HF predictions at positions ``0 .. len(token_ids)-2`` (teacher-forced reference).

    Uses the same incremental ``DynamicCache`` + ``kv_block_size`` chunked prefill as
    logit PCC prefill (O(chunk) host memory, correct global positions).
    Returns ``[len(token_ids)-1, 5]`` int64 on CPU.
    """
    n = len(token_ids)
    if n < 2:
        return torch.empty(0, 5, dtype=torch.long)

    block = kv_block_size if kv_block_size is not None else Devstral2Args.kv_block_size
    input_device = resolve_hf_input_device(causal_lm)
    cache = DynamicCache()
    rows: list[torch.Tensor] = []

    for chunk_start in range(0, n - 1, block):
        chunk_end = min(chunk_start + block, n - 1)
        chunk_tokens = token_ids[chunk_start:chunk_end]
        if not chunk_tokens:
            continue
        logits = hf_causal_lm_prefill_chunk(
            causal_lm,
            chunk_tokens,
            chunk_start=chunk_start,
            cache=cache,
            input_device=input_device,
        )
        for j in range(logits.shape[1]):
            rows.append(_top5_from_logits(logits[:, j, :]))

    return torch.stack(rows, dim=0)


@torch.no_grad()
def hf_chunked_prefill_last_logits(
    causal_lm,
    token_ids: list[int],
    *,
    prefill_len: int,
    kv_block_size: int,
) -> tuple[torch.Tensor, DynamicCache]:
    """Chunked HF prefill; return last-position logits and ``DynamicCache`` for decode."""
    input_device = resolve_hf_input_device(causal_lm)
    if len(token_ids) < prefill_len:
        raise ValueError(f"token_ids length {len(token_ids)} < prefill_len {prefill_len}")

    cache = DynamicCache()
    prefill_last_logits: torch.Tensor | None = None

    for chunk_start in range(0, prefill_len, kv_block_size):
        chunk_end = min(chunk_start + kv_block_size, prefill_len)
        chunk_tokens = token_ids[chunk_start:chunk_end]
        logits = hf_causal_lm_prefill_chunk(
            causal_lm,
            chunk_tokens,
            chunk_start=chunk_start,
            cache=cache,
            input_device=input_device,
        )
        if chunk_end == prefill_len:
            prefill_last_logits = logits[:, -1, :].cpu()

    assert prefill_last_logits is not None
    return prefill_last_logits, cache


@torch.no_grad()
def hf_decode_step_logits(
    causal_lm,
    token_id: int,
    *,
    cache: DynamicCache,
) -> torch.Tensor:
    """Single HF decode forward; returns last-position logits ``[1, vocab]`` on CPU."""
    input_device = resolve_hf_input_device(causal_lm)
    tok = torch.tensor([[token_id]], dtype=torch.long, device=input_device)
    out = causal_lm(tok, past_key_values=cache, use_cache=True)
    return out.logits[:, -1, :].cpu()


@torch.no_grad()
def tt_chunked_prefill_last_logits(
    tt_model: TtMinistral3ForCausalLM,
    args: Devstral2Args,
    mesh_device,
    token_ids: list[int],
    *,
    prefill_len: int,
) -> torch.Tensor:
    """Chunked TT prefill; return last-position logits ``[1, vocab]`` on host."""
    kv_block_size = args.kv_block_size
    vocab_size = args.vocab_size
    pad_id = args.pad_token_id if args.pad_token_id is not None else 0

    num_prefill_chunks = max(1, (prefill_len + kv_block_size - 1) // kv_block_size)
    last_chunk_start = (num_prefill_chunks - 1) * kv_block_size

    prefill_chunk_dev = _input_ids_to_tt(
        torch.full((1, kv_block_size), int(pad_id), dtype=torch.long),
        mesh_device,
    )

    prefill_logits_tt = None
    for chunk_idx in range(num_prefill_chunks):
        chunk_start = chunk_idx * kv_block_size
        chunk_end = min(chunk_start + kv_block_size, prefill_len)
        chunk_len = chunk_end - chunk_start
        chunk_tokens = torch.full((1, kv_block_size), int(pad_id), dtype=torch.long)
        chunk_tokens[0, :chunk_len] = torch.tensor(token_ids[chunk_start:chunk_end], dtype=torch.long)
        ttnn.copy_host_to_device_tensor(
            _input_ids_host(chunk_tokens, mesh_device),
            prefill_chunk_dev,
        )
        tt_out = tt_model(prefill_chunk_dev, mode="prefill", start_pos=chunk_start)
        if chunk_idx == num_prefill_chunks - 1:
            prefill_logits_tt = tt_out
        else:
            tt_out.deallocate(True)

    assert prefill_logits_tt is not None
    local_pos = (prefill_len - 1) - last_chunk_start
    prefill_logits_torch = _logits_to_torch(prefill_logits_tt, mesh_device, vocab_size)
    prefill_last = prefill_logits_torch[local_pos : local_pos + 1]
    prefill_logits_tt.deallocate(True)
    ttnn.synchronize_device(mesh_device)
    return prefill_last


@torch.no_grad()
def tt_decode_step_logits(
    tt_model: TtMinistral3ForCausalLM,
    args: Devstral2Args,
    mesh_device,
    token_id: int,
    *,
    current_pos: int,
    decode_tok_dev: ttnn.Tensor,
) -> torch.Tensor:
    """Single TT decode forward; returns last-position logits ``[1, vocab]`` on host."""
    ttnn.copy_host_to_device_tensor(
        _input_ids_host(torch.tensor([[token_id]], dtype=torch.long), mesh_device),
        decode_tok_dev,
    )
    decode_pos_dev = _current_pos_to_tt(torch.tensor([current_pos], dtype=torch.long), mesh_device)
    decode_logits_tt = tt_model(decode_tok_dev, mode="decode", current_pos=decode_pos_dev)
    decode_logits_torch = _logits_to_torch(decode_logits_tt, mesh_device, args.vocab_size)
    decode_logits_tt.deallocate(True)
    return decode_logits_torch[:1]


@torch.no_grad()
def run_hf_greedy_decode_logit_pcc(
    causal_lm,
    hf_cache: DynamicCache,
    tt_model: TtMinistral3ForCausalLM,
    args: Devstral2Args,
    mesh_device,
    *,
    prefill_len: int,
    first_input_token_id: int,
) -> None:
    """Compare HF vs TT logits for ``DECODE_GENERATION_LENGTH`` HF-greedy decode steps."""
    decode_tok_dev = _input_ids_to_tt(torch.zeros((1, 1), dtype=torch.long), mesh_device)
    next_tok = first_input_token_id
    current_pos = prefill_len

    for step in range(DECODE_GENERATION_LENGTH):
        ref_logits = hf_decode_step_logits(causal_lm, next_tok, cache=hf_cache)
        tt_logits = tt_decode_step_logits(
            tt_model,
            args,
            mesh_device,
            next_tok,
            current_pos=current_pos,
            decode_tok_dev=decode_tok_dev,
        )
        assert_logit_pcc(
            ref_logits,
            tt_logits,
            label=f"decode logits seq_len={prefill_len} step={step} pos={current_pos}",
        )
        next_tok = _hf_greedy_token_id(ref_logits)
        current_pos += 1

    ttnn.synchronize_device(mesh_device)


@torch.no_grad()
def run_logit_pcc_at_prefill_len(
    causal_lm,
    tt_model: TtMinistral3ForCausalLM,
    args: Devstral2Args,
    mesh_device,
    tokenizer,
    *,
    prefill_len: int,
) -> None:
    """Logit PCC: golden chunked prefill, then ``DECODE_GENERATION_LENGTH`` HF-greedy decode steps."""
    token_ids = build_token_sequence(tokenizer, prefill_len)

    logger.info(
        f"Logit PCC: prefill_len={prefill_len}, decode_steps={DECODE_GENERATION_LENGTH}, "
        f"decode_teacher=HF_greedy, chunk={args.kv_block_size}, pcc≥{PCC_REQUIRED}, "
        f"layer_max_seq_len={args.max_seq_len}"
    )

    ref_prefill, hf_cache = hf_chunked_prefill_last_logits(
        causal_lm,
        token_ids,
        prefill_len=prefill_len,
        kv_block_size=args.kv_block_size,
    )
    tt_prefill = tt_chunked_prefill_last_logits(
        tt_model,
        args,
        mesh_device,
        token_ids,
        prefill_len=prefill_len,
    )

    assert_logit_pcc(ref_prefill, tt_prefill, label=f"prefill last logits seq_len={prefill_len}")
    run_hf_greedy_decode_logit_pcc(
        causal_lm,
        hf_cache,
        tt_model,
        args,
        mesh_device,
        prefill_len=prefill_len,
        first_input_token_id=_hf_greedy_token_id(ref_prefill),
    )


@torch.no_grad()
def run_logit_pcc_sweep(mesh_device, prefill_lengths: list[int]) -> None:
    """Run logit PCC for each prefill length (one HF + TT model build; HF-greedy decode)."""
    model_max_seq_len = e2e_sweep_model_max_seq_len()
    logger.info(
        f"Logit PCC sweep: points={prefill_lengths}, model_max_seq_len={model_max_seq_len}, "
        f"decode_steps={DECODE_GENERATION_LENGTH}"
    )

    causal_lm, tt_model, args, _text_cfg, weight_cache_path = get_or_build_logit_pcc_models(mesh_device)
    logger.info(f"Reusing TT weight cache: {weight_cache_path}")

    tokenizer = AutoTokenizer.from_pretrained(DEVSTRAL2_MODEL_ID, trust_remote_code=True)
    failures: list[str] = []

    for prefill_len in prefill_lengths:
        try:
            run_logit_pcc_at_prefill_len(
                causal_lm,
                tt_model,
                args,
                mesh_device,
                tokenizer,
                prefill_len=prefill_len,
            )
        except AssertionError as exc:
            failures.append(f"prefill={prefill_len}: {exc}")

    if failures:
        pytest.fail("Logit PCC sweep failures:\n" + "\n".join(failures))


__all__ = [
    "DECODE_GENERATION_LENGTH",
    "LOGIT_PCC_FULL_SWEEP_TIMEOUT_SEC",
    "PCC_REQUIRED",
    "PREFILL_SANITY_SEQ_LENGTHS",
    "PREFILL_SWEEP_SEQ_LENGTHS",
    "build_logit_pcc_models",
    "device_params",
    "e2e_sweep_model_max_seq_len",
    "get_or_build_logit_pcc_models",
    "hf_teacher_forced_top5_reference",
    "HF_TEACHER_FORCED_REFERENCE_FORMAT_VERSION",
    "mesh_device_param",
    "run_logit_pcc_at_prefill_len",
    "run_logit_pcc_sweep",
    "sweep_timeout_seconds",
]
