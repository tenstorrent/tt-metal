# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Text backbone logit PCC (prefill + teacher-forced decode).

Full-stack load and config: ``demo/demo.py``, ``tests/perf/test_e2e_isl_sweep_perf.py``.

Long ISL / tail cases use ``@pytest.mark.timeout(0)`` (no limit). CI skips rows above
512-token prompt length via ``CI=true``.
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN
from models.experimental.voxtraltts.tt.voxtral_tt_args import voxtral_text_logits_pcc_optimizations
from models.experimental.voxtraltts.utils.common import (
    build_voxtral_text_page_table_host,
    build_voxtral_text_page_table_tt,
    create_real_voxtral_text_model_or_skip,
    hf_voxtral_text_reference_or_skip,
    tale_continuation_tokens,
    tale_prompt_tokens,
)

_MAX_SEQ_LEN = DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN
# Mirrors devstral2 / tt_transformers bringup: sweep ISLs + tail prefill at 65504 (65536 budget).
_PREFILL_LOGIT_PCC_ISLS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65280)
_DECODE_32STEP_ISLS = (128, 256, 384, 512, 8192)
_DECODE_32STEP_TAIL_ISL = _MAX_SEQ_LEN - 32  # 65504 + 32 → full KV timeline
_DECODE_256STEP_TAIL_ISL = _MAX_SEQ_LEN - 256  # 65280 + 256 → positions through 65535
_PREFILL_LOGIT_PCC_THRESHOLD = 0.99
_DECODE_LOGIT_PCC_THRESHOLD = 0.98
_PREFILL_CHUNK_SIZE = (
    512  # BH: MinimalMatmulConfig (w2) has CB_end=914KB, free L1=643KB/core=52.7MB; 512-chunk peak ~26MB (2.8x margin)
)


def _is_ci() -> bool:
    return os.environ.get("CI") == "true"


def _use_paged_kv(seq_len: int) -> bool:
    return seq_len > 256


def _build_decode_page_table_tt(model, seq_len: int, *, max_seq_len: int, paged_block_size: int = 32):
    """Full-pool page table for paged-KV decode, or ``None`` when paged KV is disabled.

    When ``seq_len > 256`` the model allocates its KV cache in **paged** layout
    (``[max_num_blocks, n_kv_heads, block_size, head_dim]``), so ``paged_update_cache`` /
    paged SDPA require a page table — mirrors ``VoxtralTTSPipeline._build_page_table``. Built
    once per test and reused across all KV-fill + decode steps (single device allocation).
    """
    if not _use_paged_kv(seq_len):
        return None
    return build_voxtral_text_page_table_tt(
        model.inner.mesh_device,
        max_seq_len=max_seq_len,
        paged_block_size=paged_block_size,
    )


def _create_text_model_for_logit_pcc(device):
    # Enable interleaved-weight 1D mcast for w1/w3 decode (~25% faster per decode step) and
    # L1 activations for 512-token prefill chunks (= BH prefill_len_cutoff, no reshape needed).
    # The w2 MinimalMatmulConfig kernel uses 384KB CBs + 508KB kernel = CB_end at 914KB, leaving
    # only 643KB/core free for L1 tensors (52.7MB across 80 cores).
    # At 1024-chunk the live tensors (residual+x+act_out+w2_out+attn intermediates) exceed that
    # limit by ~22KB/core causing a CB clash.  At 512-chunk peak is ~26MB (2.8x margin).
    # Both vars are scoped here; demo pipeline is unaffected.
    _prev_mlp_1d = os.environ.get("VOXTRAL_MLP_1D")
    _prev_l1_max = os.environ.get("VOXTRAL_TEXT_PREFILL_L1_MAX_SEQ_LEN")
    os.environ["VOXTRAL_MLP_1D"] = "1"
    os.environ["VOXTRAL_TEXT_PREFILL_L1_MAX_SEQ_LEN"] = "512"
    try:
        return create_real_voxtral_text_model_or_skip(
            device,
            max_seq_len=_MAX_SEQ_LEN,
            dtype=ttnn.bfloat16,
            optimizations=voxtral_text_logits_pcc_optimizations,
        )
    finally:
        if _prev_mlp_1d is None:
            os.environ.pop("VOXTRAL_MLP_1D", None)
        else:
            os.environ["VOXTRAL_MLP_1D"] = _prev_mlp_1d
        if _prev_l1_max is None:
            os.environ.pop("VOXTRAL_TEXT_PREFILL_L1_MAX_SEQ_LEN", None)
        else:
            os.environ["VOXTRAL_TEXT_PREFILL_L1_MAX_SEQ_LEN"] = _prev_l1_max


def _tt_decode_step_logits(
    model,
    token: torch.Tensor,
    position: int,
    *,
    page_table_tt: ttnn.Tensor | None = None,
) -> torch.Tensor:
    """Single decode step; return last-position logits via ``process_output_decode``."""
    tt_tokens, tt_current_pos, tt_rope_idxs, tt_page_table = model.prepare_inputs_decode(
        token, torch.tensor([position], dtype=torch.int64)
    )
    page_table = tt_page_table if tt_page_table is not None else page_table_tt
    tt_decode_logits, _ = model.inner.ttnn_decode_forward(
        tt_tokens,
        tt_current_pos,
        rot_mat_idxs=tt_rope_idxs,
        page_table=page_table,
        kv_cache=None,
    )
    return model.inner.process_output_decode(tt_decode_logits, B=1, S=1, is_tokens=False)[0, 0].float()


def _fill_tt_kv_cache(
    model,
    tokens: torch.Tensor,
    *,
    page_table_tt: ttnn.Tensor | None = None,
) -> None:
    """Fill TT KV via per-token decode steps (no logits readback)."""
    for i in range(tokens.shape[1]):
        _tt_decode_step_logits(model, tokens[:, i], i, page_table_tt=page_table_tt)


def _run_tt_prefill_fill_kv(
    model,
    tokens: torch.Tensor,
    *,
    page_table_tt: ttnn.Tensor | None = None,
) -> None:
    """Fill TT KV cache via chunked prefill (fast O(n) path); falls back for non-paged."""
    if page_table_tt is None:
        _fill_tt_kv_cache(model, tokens, page_table_tt=None)
        return
    seq_len = tokens.shape[1]
    padded_len = ((seq_len + _PREFILL_CHUNK_SIZE - 1) // _PREFILL_CHUNK_SIZE) * _PREFILL_CHUNK_SIZE
    if padded_len > seq_len:
        tokens = torch.nn.functional.pad(tokens, (0, padded_len - seq_len))
    page_table_host = build_voxtral_text_page_table_host(max_seq_len=_MAX_SEQ_LEN)
    for chunk_start in range(0, padded_len, _PREFILL_CHUNK_SIZE):
        chunk_end = chunk_start + _PREFILL_CHUNK_SIZE
        chunk_tokens = tokens[:, chunk_start:chunk_end]
        tt_x, rot_mats_global, rot_mats_local, tt_page_table, _, _ = model.prepare_inputs_prefill(
            chunk_tokens,
            start_pos=chunk_start,
            page_table=page_table_host,
        )
        chunk_page_table_host = page_table_host[:, chunk_start // 32 : chunk_end // 32]
        tt_chunk_page_table = ttnn.from_torch(
            chunk_page_table_host,
            device=model.inner.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(model.inner.mesh_device),
        )
        tt_logits = model.inner.ttnn_prefill_forward(
            tt_x,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            page_table=tt_page_table,
            chunk_page_table=tt_chunk_page_table,
            chunk_start_idx=chunk_start,
            get_last_token=-1,
            kv_cache=None,
        )
        if tt_logits.is_allocated():
            ttnn.deallocate(tt_logits)
        if tt_chunk_page_table.is_allocated():
            ttnn.deallocate(tt_chunk_page_table)


def _tt_last_logit_from_prefill_tokens(
    model,
    tokens: torch.Tensor,
    *,
    page_table_tt: ttnn.Tensor | None = None,
) -> torch.Tensor:
    """Run chunked TT prefill and return logits at the final prompt position."""
    seq_len = tokens.shape[1]
    if seq_len == 1:
        return _tt_decode_step_logits(model, tokens[:, 0], 0, page_table_tt=page_table_tt)
    last_token_idx = seq_len - 1
    padded_len = ((seq_len + _PREFILL_CHUNK_SIZE - 1) // _PREFILL_CHUNK_SIZE) * _PREFILL_CHUNK_SIZE
    if padded_len > seq_len:
        tokens = torch.nn.functional.pad(tokens, (0, padded_len - seq_len))
    page_table_host = build_voxtral_text_page_table_host(max_seq_len=_MAX_SEQ_LEN)
    last_chunk_start = (last_token_idx // _PREFILL_CHUNK_SIZE) * _PREFILL_CHUNK_SIZE
    last_logits = None
    for chunk_start in range(0, padded_len, _PREFILL_CHUNK_SIZE):
        chunk_end = chunk_start + _PREFILL_CHUNK_SIZE
        chunk_tokens = tokens[:, chunk_start:chunk_end]
        tt_x, rot_mats_global, rot_mats_local, tt_page_table, _, _ = model.prepare_inputs_prefill(
            chunk_tokens,
            start_pos=chunk_start,
            page_table=page_table_host,
        )
        chunk_page_table_host = page_table_host[:, chunk_start // 32 : chunk_end // 32]
        tt_chunk_page_table = ttnn.from_torch(
            chunk_page_table_host,
            device=model.inner.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(model.inner.mesh_device),
        )
        is_last_chunk = chunk_start == last_chunk_start
        last_token_idx_in_chunk = last_token_idx - chunk_start
        get_last_token = (last_token_idx_in_chunk // 32) * 32 if is_last_chunk else -1
        tt_logits = model.inner.ttnn_prefill_forward(
            tt_x,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            page_table=tt_page_table,
            chunk_page_table=tt_chunk_page_table,
            chunk_start_idx=chunk_start,
            get_last_token=get_last_token,
            kv_cache=None,
        )
        if is_last_chunk:
            tt_logits = ttnn.to_layout(tt_logits, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            last_logits = model.inner.process_output_prefill(
                tt_logits.cpu(), last_token_idx=(last_token_idx_in_chunk % 32)
            ).float()
        if tt_logits.is_allocated():
            ttnn.deallocate(tt_logits)
        if tt_chunk_page_table.is_allocated():
            ttnn.deallocate(tt_chunk_page_table)
    assert last_logits is not None
    return last_logits


def _assert_prefill_last_logit_pcc(model, seq_len: int, *, pcc_threshold: float) -> None:
    tokens = tale_prompt_tokens(seq_len)
    page_table_tt = build_voxtral_text_page_table_tt(
        model.inner.mesh_device,
        max_seq_len=_MAX_SEQ_LEN,
    )
    tt_last_logits = _tt_last_logit_from_prefill_tokens(model, tokens, page_table_tt=page_table_tt)
    hf_ref = hf_voxtral_text_reference_or_skip()
    ref_last_logits = hf_ref(input_ids=tokens).logits[0, -1].float()

    passing, pcc_value = comp_pcc(ref_last_logits, tt_last_logits, pcc=pcc_threshold)
    logger.info(f"prefill_logit_pcc seq_len={seq_len} paged_kv=True PCC={float(pcc_value):.6f}")
    assert passing, f"Prefill logits mismatch vs HF (seq_len={seq_len}): {pcc_value}"


def _assert_decode_multistep_logits_pcc(
    model,
    prompt_tokens: torch.Tensor,
    decode_tokens: torch.Tensor,
    *,
    pcc_threshold: float,
    log_name: str,
    page_table_tt: ttnn.Tensor | None = None,
    early_pcc_threshold: float = 0.99,
    early_steps: int = 25,
) -> None:
    """Teacher-forced decode logits PCC for each step (TT ``process_output_decode`` vs HF cache).

    The first ``early_steps`` steps are held to the stricter ``early_pcc_threshold`` (0.99); later
    steps (where small KV/RoPE differences have had more positions to accumulate) are allowed down to
    ``pcc_threshold`` (0.98).
    """
    seq_len = prompt_tokens.shape[1]
    decode_steps = decode_tokens.shape[1]
    _run_tt_prefill_fill_kv(model, prompt_tokens, page_table_tt=page_table_tt)

    hf_ref = hf_voxtral_text_reference_or_skip()
    hf_past = hf_ref(input_ids=prompt_tokens, use_cache=True).past_key_values

    for step in range(decode_steps):
        current_pos = seq_len + step
        step_token = decode_tokens[:, step]
        tt_tokens, tt_current_pos, tt_rope_idxs, tt_page_table = model.prepare_inputs_decode(
            step_token, torch.tensor([current_pos], dtype=torch.int64)
        )
        tt_decode_logits, _ = model.inner.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mat_idxs=tt_rope_idxs,
            page_table=page_table_tt if page_table_tt is not None else tt_page_table,
            kv_cache=None,
        )
        tt_last_logits = model.inner.process_output_decode(tt_decode_logits, B=1, S=1, is_tokens=False)[0, 0].float()

        hf_step = hf_ref(input_ids=step_token.view(1, 1), past_key_values=hf_past, use_cache=True)
        hf_past = hf_step.past_key_values
        ref_last_logits = hf_step.logits[0, -1].float()

        step_pcc_threshold = early_pcc_threshold if step < early_steps else pcc_threshold
        passing, pcc_value = comp_pcc(ref_last_logits, tt_last_logits, pcc=step_pcc_threshold)
        logger.info(
            f"{log_name} seq_len={seq_len} steps={decode_steps} step={step} "
            f"pos={current_pos} PCC={float(pcc_value):.6f} threshold={step_pcc_threshold}"
        )
        assert passing, (
            f"Step {step} decode logits mismatch vs reference "
            f"(seq_len={seq_len}, pos={current_pos}): {pcc_value} < {step_pcc_threshold}"
        )


def _decode_multistep_logit_pcc_cases():
    cases = [pytest.param(seq_len, 32, id=f"{seq_len}-32_steps") for seq_len in _DECODE_32STEP_ISLS]
    cases.append(pytest.param(_DECODE_32STEP_TAIL_ISL, 32, id="65504-32_steps_tail"))
    if os.environ.get("VOXTRAL_DECODE_256STEP", "1") != "0":
        cases.append(pytest.param(_DECODE_256STEP_TAIL_ISL, 256, id="65280-256_steps"))
    return cases


@torch.no_grad()
@pytest.mark.timeout(0)
@pytest.mark.parametrize("seq_len", _PREFILL_LOGIT_PCC_ISLS, ids=[str(s) for s in _PREFILL_LOGIT_PCC_ISLS])
def test_text_model_prefill_logit_pcc(device, reset_seeds, seq_len):
    """Last-token prefill logits PCC vs HF at increasing ISLs (paged KV, max_seq_len=65536)."""
    if _is_ci() and seq_len > 512:
        pytest.skip("CI runs prefill logit PCC up to 512 only")

    model = _create_text_model_for_logit_pcc(device)
    _assert_prefill_last_logit_pcc(model, seq_len, pcc_threshold=_PREFILL_LOGIT_PCC_THRESHOLD)


@torch.no_grad()
@pytest.mark.timeout(0)
@pytest.mark.parametrize("seq_len,decode_steps", _decode_multistep_logit_pcc_cases())
def test_text_model_decode_multistep_logit_pcc(device, reset_seeds, seq_len, decode_steps):
    """Teacher-forced decode logits PCC vs HF (32-step CI; tail rows on bare metal).

    Set ``VOXTRAL_DECODE_256STEP=0`` to skip the long 65280+256 case locally.
    """
    if _is_ci() and (decode_steps > 32 or seq_len > 512):
        pytest.skip("CI runs 32-step decode logit PCC at prompt ISL ≤512 only")
    if seq_len + decode_steps > _MAX_SEQ_LEN:
        pytest.skip(f"seq_len={seq_len} + decode_steps={decode_steps} exceeds max_seq_len={_MAX_SEQ_LEN}")

    model = _create_text_model_for_logit_pcc(device)
    prompt_tokens = tale_prompt_tokens(seq_len)
    decode_tokens = tale_continuation_tokens(seq_len, decode_steps)

    logger.info(
        f"test_text_model_decode_tail_context_multistep_pcc: "
        f"prompt_len={seq_len}, decode_steps={decode_steps}, max_pos={seq_len + decode_steps - 1}, "
        f"tale_tokens=True"
    )
    page_table_tt = _build_decode_page_table_tt(model, seq_len, max_seq_len=DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN)
    _assert_decode_multistep_logits_pcc(
        model,
        prompt_tokens,
        decode_tokens,
        pcc_threshold=_DECODE_LOGIT_PCC_THRESHOLD,
        log_name=f"decode_multistep_{decode_steps}",
        page_table_tt=page_table_tt,
    )
