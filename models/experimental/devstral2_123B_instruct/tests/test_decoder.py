# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Decoder-layer decode-mode PCC vs HuggingFace ``Ministral3DecoderLayer`` (layer 0).

Feeds random hidden states one token at a time (batch 1, ``hidden_size`` from HF config)
while advancing the KV-cache position. Uses the shared ``seq_262144`` weight cache and KV budget
(same as demo). Runs ``DECODE_GENERATION_LENGTH`` decode steps (default 10) at positions 0 … 9
and asserts PCC ≥ 0.99 on every step.

Run: ``pytest …/test_decoder.py -v``
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache

from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_123B_instruct.tests.decoder_pcc_common import (
    DECODE_GENERATION_LENGTH,
    PCC_BATCH_SIZE,
    PCC_REQUIRED,
    build_decode_pcc_context,
    hf_decode_forward,
    log_pcc_step,
    mesh_device_param,
    tt_decode_forward,
    device_params,
)


@torch.no_grad()
def _run_decode_pcc(mesh_device) -> None:
    ctx = build_decode_pcc_context(mesh_device)
    hidden_size = ctx.args.hidden_size
    kv_budget = ctx.args.max_seq_len
    cache = DynamicCache()
    all_tests_pass = True

    logger.info(
        f"Decode PCC: batch={PCC_BATCH_SIZE}, hidden_size={hidden_size}, kv_budget={kv_budget}, "
        f"steps={DECODE_GENERATION_LENGTH}, pcc≥{PCC_REQUIRED}"
    )

    for step in range(DECODE_GENERATION_LENGTH):
        hidden = (torch.rand(PCC_BATCH_SIZE, 1, hidden_size, dtype=torch.bfloat16) * 2) - 1

        ref_out = hf_decode_forward(ctx.ref, ctx.ref_rope, hidden, position=step, cache=cache)
        tt_torch = tt_decode_forward(ctx.tt_layer, mesh_device, hidden, position=step, hidden_size=hidden_size)

        passing, pcc_message = comp_pcc(ref_out, tt_torch, PCC_REQUIRED)
        logger.info(comp_allclose(ref_out, tt_torch))
        log_pcc_step(f"decode step={step} pos={step}", passing, pcc_message)
        if not passing:
            all_tests_pass = False

    assert (
        all_tests_pass
    ), f"Decode PCC below {PCC_REQUIRED} (hidden_size={hidden_size}, kv_budget={kv_budget}). Check warnings."


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("mesh_device", [mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@pytest.mark.timeout(3600)
def test_decoder_decode_pcc(mesh_device):
    """10 decode steps at positions 0–9, batch 1, shared ``seq_262144`` weight cache."""
    _run_decode_pcc(mesh_device)
