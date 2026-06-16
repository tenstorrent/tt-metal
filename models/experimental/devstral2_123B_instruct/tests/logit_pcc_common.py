# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for full-model teacher-forced logit PCC (prefill + decode)."""

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
    _round_up,
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
    devstral2_weight_cache_seq_len,
    resolve_devstral2_weight_cache_path,
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

_TESTS_DIR = Path(__file__).resolve().parent
_TALE_OF_TWO_CITIES = _TESTS_DIR.parents[2] / "tt_transformers" / "tests" / "tale-of-two-cities.txt.bz2"


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


def sweep_timeout_seconds(prefill_lengths: list[int]) -> int:
    """Budget HF load, chunked TT prefill, and teacher-forced decode logit PCC per point."""
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
        hf_tokens = prefill_len + DECODE_GENERATION_LENGTH
        budget += int(hf_tokens * hf_forward_sec_per_token)
        budget += tt_base_sec + num_chunks * tt_prefill_sec_per_chunk
        budget += int(DECODE_GENERATION_LENGTH * tt_decode_sec_per_step)
    return int(budget * _SWEEP_TIMEOUT_MARGIN)


def _test_seq_len(prefill_len: int) -> int:
    need = prefill_len + DECODE_GENERATION_LENGTH + 1
    return max(_round_up(need, Devstral2Args.kv_block_size), Devstral2Args.kv_block_size)


def runtime_max_seq_len(test_seq_len: int) -> int:
    cache_seq = devstral2_weight_cache_seq_len()
    return max(_round_up(test_seq_len, Devstral2Args.kv_block_size), cache_seq)


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


def build_logit_pcc_models(mesh_device, *, max_seq_len: int):
    """Load HF causal LM and build ``TtMinistral3ForCausalLM`` from extracted bf16 weights."""
    try:
        text_cfg = load_devstral2_text_config()
        causal_lm = load_devstral2_causal_lm()
    except Exception as exc:
        _skip_hf_load_failure(exc)

    prepare_causal_lm_for_pcc(causal_lm)
    hf_device = resolve_hf_input_device(causal_lm)
    num_layers = int(text_cfg.num_hidden_layers)
    state_dict = extract_causal_lm_bf16_state_dict(causal_lm, num_layers)
    weight_cache_path = resolve_devstral2_weight_cache_path(mesh_device, text_cfg, num_layers)

    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=max_seq_len,
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
        f"Logit PCC models ready: layers={num_layers}, max_seq_len={max_seq_len}, "
        f"hf_input_device={hf_device}, weight_cache_path={weight_cache_path}"
    )
    return causal_lm, tt_model, args, text_cfg, weight_cache_path


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


@torch.no_grad()
def hf_prefill_and_decode_logits(
    causal_lm,
    token_ids: list[int],
    *,
    prefill_len: int,
    kv_block_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor], DynamicCache]:
    """Chunked HF prefill + teacher-forced decode logits (O(chunk) CPU/host per prefill step)."""
    input_device = resolve_hf_input_device(causal_lm)
    need_len = prefill_len + DECODE_GENERATION_LENGTH
    if len(token_ids) < need_len:
        raise ValueError(f"token_ids length {len(token_ids)} < required {need_len}")

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

    decode_logits: list[torch.Tensor] = []
    for step in range(DECODE_GENERATION_LENGTH):
        pos = prefill_len + step
        tok = torch.tensor([[token_ids[pos]]], dtype=torch.long, device=input_device)
        out = causal_lm(tok, past_key_values=cache, use_cache=True)
        decode_logits.append(out.logits[:, -1, :].cpu())

    return prefill_last_logits, decode_logits, cache


@torch.no_grad()
def tt_prefill_and_decode_logits(
    tt_model: TtMinistral3ForCausalLM,
    args: Devstral2Args,
    mesh_device,
    token_ids: list[int],
    *,
    prefill_len: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Chunked TT prefill + teacher-forced decode; return logits matching HF positions."""
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

    decode_logits: list[torch.Tensor] = []
    decode_tok_dev = _input_ids_to_tt(torch.zeros((1, 1), dtype=torch.long), mesh_device)
    current_pos = prefill_len

    for step in range(DECODE_GENERATION_LENGTH):
        ref_tok = token_ids[prefill_len + step]
        ttnn.copy_host_to_device_tensor(
            _input_ids_host(torch.tensor([[ref_tok]], dtype=torch.long), mesh_device),
            decode_tok_dev,
        )
        decode_pos_dev = _current_pos_to_tt(torch.tensor([current_pos], dtype=torch.long), mesh_device)
        decode_logits_tt = tt_model(decode_tok_dev, mode="decode", current_pos=decode_pos_dev)
        decode_logits_torch = _logits_to_torch(decode_logits_tt, mesh_device, vocab_size)
        decode_logits.append(decode_logits_torch[:1])
        decode_logits_tt.deallocate(True)
        current_pos += 1

    ttnn.synchronize_device(mesh_device)
    return prefill_last, decode_logits


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
    """Teacher-forced logit PCC after prefill and for ``DECODE_GENERATION_LENGTH`` decode steps."""
    total_tokens = prefill_len + DECODE_GENERATION_LENGTH
    token_ids = build_token_sequence(tokenizer, total_tokens)

    logger.info(
        f"Logit PCC: prefill_len={prefill_len}, decode_steps={DECODE_GENERATION_LENGTH}, "
        f"chunk={args.kv_block_size}, pcc≥{PCC_REQUIRED}, layer_max_seq_len={args.max_seq_len}"
    )

    ref_prefill, ref_decode_steps, _ = hf_prefill_and_decode_logits(
        causal_lm,
        token_ids,
        prefill_len=prefill_len,
        kv_block_size=args.kv_block_size,
    )
    tt_prefill, tt_decode_steps = tt_prefill_and_decode_logits(
        tt_model,
        args,
        mesh_device,
        token_ids,
        prefill_len=prefill_len,
    )

    assert_logit_pcc(ref_prefill, tt_prefill, label=f"prefill last logits seq_len={prefill_len}")
    for step, (ref_logits, tt_logits) in enumerate(zip(ref_decode_steps, tt_decode_steps)):
        pos = prefill_len + step
        assert_logit_pcc(
            ref_logits,
            tt_logits,
            label=f"decode logits seq_len={prefill_len} step={step} pos={pos}",
        )


@torch.no_grad()
def run_logit_pcc_sweep(mesh_device, prefill_lengths: list[int]) -> None:
    """Run teacher-forced logit PCC for each prefill length (one HF + TT model build)."""
    worst_prefill = max(prefill_lengths)
    global_test_seq_len = _test_seq_len(worst_prefill)
    global_max_seq_len = runtime_max_seq_len(global_test_seq_len)
    logger.info(
        f"Logit PCC sweep: points={prefill_lengths}, worst_prefill={worst_prefill}, "
        f"global_max_seq_len={global_max_seq_len}, decode_steps={DECODE_GENERATION_LENGTH}"
    )

    causal_lm, tt_model, args, _text_cfg, weight_cache_path = build_logit_pcc_models(
        mesh_device,
        max_seq_len=global_max_seq_len,
    )
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
    "PCC_REQUIRED",
    "PREFILL_SANITY_SEQ_LENGTHS",
    "PREFILL_SWEEP_SEQ_LENGTHS",
    "build_logit_pcc_models",
    "device_params",
    "mesh_device_param",
    "run_logit_pcc_at_prefill_len",
    "run_logit_pcc_sweep",
    "sweep_timeout_seconds",
]
