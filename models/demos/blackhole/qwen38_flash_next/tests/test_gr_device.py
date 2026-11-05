# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Gated-residual write program versus the module's chain on the line's hidden shards, with trace reuse."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.helpers.tp_harness import (
    DEVICE_PARAMS,
    assert_program_cache_stable,
    assert_shards_equal,
    replay_trace,
    upload,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import fused, gr
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import Qwen38MeshContract, TensorPlacement

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held four-die mesh")

BRANCHES = gr.RESIDUAL_BRANCHES


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 2_000_000}], indirect=True)
@pytest.mark.parametrize("rows", [1, 5, 32])
def test_gr_write_against_chain(mesh_device, rows):
    entry = fused.kernel(fused.gr_write.NAME)
    assert entry.tolerance == fused.BITWISE, f"{entry.name}: tolerance {entry.tolerance}"
    contract = Qwen38MeshContract(tuple(int(device_id) for device_id in mesh_device.get_device_ids()))
    cached = None
    for seed in (611, 612):
        rng = torch.Generator().manual_seed(seed)
        owned = []
        try:
            residual = upload(
                mesh_device, torch.randn(1, BRANCHES, rows, gr.HIDDEN_SIZE, generator=rng), ttnn.bfloat16, shard_dim=3
            )
            block = upload(
                mesh_device, torch.randn(1, 1, rows, gr.HIDDEN_SIZE, generator=rng), ttnn.bfloat16, shard_dim=3
            )
            coefficients = 2 * torch.sigmoid(torch.randn(1, 1, rows, BRANCHES, generator=rng))
            coefficients[..., 0, 1] = 0.0  # an exact zero coefficient: 0 * value = 0
            injection = upload(mesh_device, coefficients, ttnn.bfloat16)
            owned += [residual, block, injection]
            expected = entry.composed(block, residual, injection)
            owned.append(expected)

            def check(outputs):
                assert_shards_equal(outputs[0], expected, f"residual, rows={rows}, seed={seed}")

            outputs = (entry.fused(block, residual, injection),)
            owned.extend(outputs)
            check(outputs)
            contract.validate_tensor(outputs[0], placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
            cached = assert_program_cache_stable(mesh_device, cached)
            replay_trace(mesh_device, lambda: (entry.fused(block, residual, injection),), check, owned)
        finally:
            for tensor in owned:
                ttnn.deallocate(tensor)
