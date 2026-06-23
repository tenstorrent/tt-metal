# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Text backbone logit PCC (prefill + teacher-forced decode).

Full-stack load and config: ``demo/demo.py``, ``tests/perf/test_e2e_isl_sweep_perf.py``.

Long ISL / tail cases use ``@pytest.mark.timeout(0)`` (no limit). CI skips rows above
4096-token prompt length via ``CI=true``.

Run prefill / long-ISL decode on **P150 (1×1)** — ``export MESH_DEVICE=P150``. A 4-device
mesh (e.g. QB2 without narrowing) is not valid for these single-device logit tests.
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
    build_voxtral_text_page_table_tt,
    create_real_voxtral_text_model_or_skip,
    hf_voxtral_text_reference_or_skip,
    tale_continuation_tokens,
    tale_prompt_tokens,
)

_MAX_SEQ_LEN = DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN
# Mirrors devstral2 / tt_transformers bringup: sweep ISLs + tail prefill at 65504 (65536 budget).
_PREFILL_LOGIT_PCC_ISLS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65504)
_DECODE_32STEP_ISLS = (128, 256, 384, 512, 8192, 16384)
_DECODE_32STEP_TAIL_ISL = _MAX_SEQ_LEN - 32  # 65504 + 32 → full KV timeline
_DECODE_256STEP_TAIL_ISL = _MAX_SEQ_LEN - 256  # 65280 + 256 → positions through 65535
_DECODE_LOGIT_PCC_THRESHOLD = 0.98

# Measured P150 (1×1) last-token prefill logit PCC vs HF (Tale of Two Cities, paged KV).
# Long ISL tiers are below 0.99 due to BFP8/HiFi2 accumulation; see README §6.1.1.
_PREFILL_LOGIT_PCC_THRESHOLDS: dict[int, float] = {
    128: 0.99,
    256: 0.99,
    512: 0.99,
    1024: 0.99,
    2048: 0.99,
    4096: 0.99,
    8192: 0.95,
    16384: 0.94,
    32768: 0.97,
    65504: 0.94,
}


def _prefill_logit_pcc_threshold(seq_len: int) -> float:
    return _PREFILL_LOGIT_PCC_THRESHOLDS.get(seq_len, 0.99)


_CI_MAX_PROMPT_ISL = 4096


def _is_ci() -> bool:
    return os.environ.get("CI") == "true"


def _create_text_model_for_logit_pcc(device):
    return create_real_voxtral_text_model_or_skip(
        device,
        max_seq_len=_MAX_SEQ_LEN,
        dtype=ttnn.bfloat16,
        optimizations=voxtral_text_logits_pcc_optimizations,
    )


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


def _tt_last_logit_from_prefill_tokens(
    model,
    tokens: torch.Tensor,
    *,
    page_table_tt: ttnn.Tensor | None = None,
) -> torch.Tensor:
    """Run token-by-token prefill on device; return logits at the final position."""
    seq_len = tokens.shape[1]
    if seq_len == 1:
        return _tt_decode_step_logits(model, tokens[:, 0], 0, page_table_tt=page_table_tt)
    _fill_tt_kv_cache(model, tokens[:, :-1], page_table_tt=page_table_tt)
    return _tt_decode_step_logits(model, tokens[:, -1], seq_len - 1, page_table_tt=page_table_tt)


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
) -> None:
    seq_len = prompt_tokens.shape[1]
    decode_steps = decode_tokens.shape[1]
    _fill_tt_kv_cache(model, prompt_tokens)

    hf_ref = hf_voxtral_text_reference_or_skip()
    hf_past = hf_ref(input_ids=prompt_tokens, use_cache=True).past_key_values

    for step in range(decode_steps):
        current_pos = seq_len + step
        step_token = decode_tokens[:, step]
        tt_last_logits = _tt_decode_step_logits(model, step_token, current_pos)

        hf_step = hf_ref(input_ids=step_token.view(1, 1), past_key_values=hf_past, use_cache=True)
        hf_past = hf_step.past_key_values
        ref_last_logits = hf_step.logits[0, -1].float()

        passing, pcc_value = comp_pcc(ref_last_logits, tt_last_logits, pcc=pcc_threshold)
        logger.info(
            f"{log_name} seq_len={seq_len} steps={decode_steps} step={step} "
            f"pos={current_pos} PCC={float(pcc_value):.6f}"
        )
        assert passing, f"Decode logits mismatch vs HF (seq_len={seq_len}, step={step}, pos={current_pos}): {pcc_value}"


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
    if _is_ci() and seq_len > _CI_MAX_PROMPT_ISL:
        pytest.skip(f"CI runs prefill logit PCC up to {_CI_MAX_PROMPT_ISL} only")

    model = _create_text_model_for_logit_pcc(device)
    _assert_prefill_last_logit_pcc(model, seq_len, pcc_threshold=_prefill_logit_pcc_threshold(seq_len))


@torch.no_grad()
@pytest.mark.timeout(0)
@pytest.mark.parametrize("seq_len,decode_steps", _decode_multistep_logit_pcc_cases())
def test_text_model_decode_multistep_logit_pcc(device, reset_seeds, seq_len, decode_steps):
    """Teacher-forced decode logits PCC vs HF (32-step CI; tail rows on bare metal).

    Set ``VOXTRAL_DECODE_256STEP=0`` to skip the long 65280+256 case locally.
    """
    if _is_ci() and (decode_steps > 32 or seq_len > _CI_MAX_PROMPT_ISL):
        pytest.skip(f"CI runs 32-step decode logit PCC at prompt ISL ≤{_CI_MAX_PROMPT_ISL} only")
    if seq_len + decode_steps > _MAX_SEQ_LEN:
        pytest.skip(f"seq_len={seq_len} + decode_steps={decode_steps} exceeds max_seq_len={_MAX_SEQ_LEN}")

    model = _create_text_model_for_logit_pcc(device)
    prompt_tokens = tale_prompt_tokens(seq_len)
    decode_tokens = tale_continuation_tokens(seq_len, decode_steps)
    _assert_decode_multistep_logits_pcc(
        model,
        prompt_tokens,
        decode_tokens,
        pcc_threshold=_DECODE_LOGIT_PCC_THRESHOLD,
        log_name=f"decode_multistep_{decode_steps}",
    )
