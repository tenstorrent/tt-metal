# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, one-length-per-process capacity probe for functional prefill."""

from __future__ import annotations

import os

import pytest
import torch

from models.autoports.qwen_qwen3_32b.tests.test_functional_decoder import (
    EMITTED_BATCH,
    REPRESENTATIVE_LAYER,
    FunctionalDecoder,
    _config,
    _empty_caches,
    _synthetic_state,
    _to_host,
    _tt_tensor,
)

PROBE_LENGTH_ENV = "QWEN3_32B_CONTEXT_PROBE_LEN"
EXPECT_OOM_ENV = "QWEN3_32B_CONTEXT_EXPECT_OOM"


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_batch32_prefill_capacity_probe(mesh_device):
    length_text = os.environ.get(PROBE_LENGTH_ENV)
    if not length_text:
        pytest.skip(f"Set {PROBE_LENGTH_ENV} to run an isolated context-capacity measurement")
    seq_len = int(length_text)

    config = _config()
    state = _synthetic_state(config)
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH,
        max_cache_len=seq_len,
    )
    key_cache, value_cache = _empty_caches(config, mesh_device, max_cache_len=seq_len)
    hidden = torch.zeros((1, EMITTED_BATCH, seq_len, config.hidden_size), dtype=torch.bfloat16)

    expect_oom = os.environ.get(EXPECT_OOM_ENV) == "1"
    try:
        actual = decoder.prefill_forward(_tt_tensor(hidden, mesh_device), key_cache, value_cache)
    except RuntimeError as error:
        if not expect_oom:
            raise
        message = str(error)
        assert "Out of Memory" in message, message
        detail = next(line.strip() for line in message.splitlines() if "Out of Memory" in line)
        print(f"capacity probe EXPECTED OOM: batch={EMITTED_BATCH}, seq_len={seq_len}: {detail}")
        return

    if expect_oom:
        pytest.fail(f"Expected a DRAM allocation failure at batch={EMITTED_BATCH}, seq_len={seq_len}")
    host = _to_host(actual)
    assert tuple(host.shape) == tuple(hidden.shape)
    print(f"capacity probe PASS: batch={EMITTED_BATCH}, seq_len={seq_len}, output_shape={tuple(host.shape)}")
