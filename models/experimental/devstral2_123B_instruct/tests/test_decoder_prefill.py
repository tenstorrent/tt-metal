# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Decoder-layer prefill-mode PCC vs HuggingFace ``Ministral3DecoderLayer`` (layer 0).

Chunked prefill (128-token blocks): random hidden states per chunk, batch 1. Avoids host OOM
on long sequences by not materializing ``[1, seq_len, hidden_size]`` at once. TT layer uses
the shared ``seq_262144`` weight cache (same as demo).

Sanity (CI): ``pytest …/test_decoder_prefill.py -k sanity -v``
Full sweep (32 … 256K): ``pytest …/test_decoder_prefill.py -k sweep -v``
"""

from __future__ import annotations

import pytest

from models.experimental.devstral2_123B_instruct.tests.decoder_pcc_common import (
    PREFILL_SANITY_SEQ_LENGTHS,
    PREFILL_SWEEP_SEQ_LENGTHS,
    build_prefill_pcc_context,
    mesh_device_param,
    pcc_layer_max_seq_len,
    run_prefill_pcc_at_seq_len,
    device_params,
)
from loguru import logger


def _run_prefill_pcc_sweep(mesh_device, seq_lengths: list[int]) -> None:
    logger.info(
        f"Prefill PCC sweep: {len(seq_lengths)} lengths, "
        f"layer_max_seq_len={pcc_layer_max_seq_len()} (shared weight cache, chunked HF+TT)"
    )
    for seq_len in seq_lengths:
        # Fresh TT layer per length so paged KV starts empty (matches HF DynamicCache per run).
        ctx = build_prefill_pcc_context(mesh_device)
        run_prefill_pcc_at_seq_len(ctx, mesh_device, seq_len=seq_len)


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("mesh_device", [mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@pytest.mark.timeout(3600)
def test_decoder_prefill_pcc_sanity(mesh_device):
    """Short seq-length gate (32, 128): one prefill forward per length, batch 1."""
    _run_prefill_pcc_sweep(mesh_device, PREFILL_SANITY_SEQ_LENGTHS)


@pytest.mark.slow
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("mesh_device", [mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@pytest.mark.timeout(0)
def test_decoder_prefill_pcc_sweep(mesh_device):
    """Full seq-length sweep (32 … 256K): chunked prefill, shared ``seq_262144`` weight cache."""
    _run_prefill_pcc_sweep(mesh_device, PREFILL_SWEEP_SEQ_LENGTHS)
