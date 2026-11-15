# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 12: the op runs beside L1 tensors the caller keeps alive. A 256 KB width-sharded L1 tensor on every
core of the grid sits below the top of L1; every call must dispatch next to it (its static CB region ends at or below
L1_CB_REGION_END) and return exactly what it returns without it."""

import pytest
import torch

import ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from ttnn.operations.gated_delta_net import gated_delta_net_program_descriptor as gdn_pd
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_perf import _to_device

BLOCKER_BYTES_PER_CORE = 256 * 1024


def _l1_blocker(device):
    grid = device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    shard_w = BLOCKER_BYTES_PER_CORE // (32 * 2)
    spec = ttnn.ShardSpec(cores, (32, shard_w), ttnn.ShardOrientation.ROW_MAJOR)
    config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)
    width = shard_w * grid.x * grid.y
    return ttnn.from_torch(
        torch.ones(1, 1, 32, width), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config
    )


def _call(tensors, device):
    returned = gated_delta_net(*_to_device(tensors, device, ttnn.bfloat16), compute_kernel_config=compute_config())
    return [ttnn.to_torch(t) for t in returned]


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 1, 2560), (1, 12, 128, 128)), id="decode_b1"),
        pytest.param(((1, 16, 2560), (16, 12, 128, 128)), id="decode_b16"),
        pytest.param(((1, 128, 2560), (1, 12, 128, 128)), id="prefill_t128"),
        pytest.param(((1, 2048, 2560), (1, 12, 128, 128)), id="prefill_t2048"),
        pytest.param(((1, 4096, 2560), (1, 12, 128, 128)), id="prefill_t4096"),
    ],
)
def test_beside_caller_l1_tensor(device, inputs):
    torch.manual_seed(42)
    tensors = make_inputs(inputs)
    blocker = _l1_blocker(device)
    assert blocker.buffer_address() >= gdn_pd.L1_CB_REGION_END
    beside = _call(tensors, device)
    ttnn.deallocate(blocker)
    alone = _call(tensors, device)
    for name, got, want in zip(("output", "conv_state_out", "recurrent_state_out"), beside, alone):
        assert torch.equal(got, want), f"{name} differs next to the L1 tensor"


def test_footprint_check_names_role(device, monkeypatch, expect_error):
    """A bound no plan can meet raises on the host, naming the core role and its footprint."""
    monkeypatch.setattr(gdn_pd, "L1_CB_REGION_END", gdn_pd.L1_CB_REGION_END - 900_000)
    tensors = make_inputs(((1, 2048, 2560), (1, 12, 128, 128)))
    with expect_error(RuntimeError, r"cores need \d+ B of static circular buffers"):
        gated_delta_net(*_to_device(tensors, device, ttnn.bfloat16), compute_kernel_config=compute_config())
