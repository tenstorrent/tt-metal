# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for matmul_decode fed by the DRISC tensor prefetcher.

The weight lives in DRAM as an ND-sharded (receiver-contiguous) tensor -- one
[K, N/num_receivers] slab per B core -- and the prefetcher pushes each slab into
the matmul's in1 circular buffer through a DRAM-sender GlobalCircularBuffer.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.prefetcher_common import (
    bank_receivers_strided,
    make_recv_contig_weight,
    tensor_prefetcher_session,
)


def _num_cores_to_rectangle_core_range_set(num_cores, device):
    """A single rectangular ``CoreRangeSet`` of exactly ``num_cores`` cores.

    Mirrors ``num_cores_to_rectangle_core_range_set`` in ``test_matmul_decode.py``:
    finds the widest ``x`` dividing ``num_cores`` that fits the device grid, giving
    an ``(x, num_cores // x)`` rectangle. A single-row ``CoreRangeSet`` (as wide as
    ``num_cores``) assumes the device grid is at least ``num_cores`` cores wide,
    which doesn't hold on every device, so we wrap into a rectangle instead.
    """
    grid = device.compute_with_storage_grid_size()
    x = grid.x
    while x > 0 and num_cores % x != 0:
        x -= 1
    y = num_cores // x if x > 0 else 0
    if x == 0 or y > grid.y:
        raise ValueError(f"cannot form a rectangular grid of {num_cores} cores within a {grid.x}x{grid.y} device grid")
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})


@pytest.fixture(autouse=True)
def _require_tensor_prefetcher(device):
    """Skip unless programmable DRAM cores are available on this device."""
    if not ttnn.experimental.is_tensor_prefetcher_supported(device):
        pytest.skip(
            "programmable DRAM cores unavailable (need Blackhole, firmware >= 19.12.0.0, "
            "and either no harvested DRAM channels or a single device)"
        )


def test_matmul_decode_accepts_global_cb_kwarg(device):
    """The global_cb keyword exists and defaults to None (no behavior change)."""
    m, k, n = 32, 1024, 2048
    num_a_cores = 32
    num_b_cores = n // 64

    torch.manual_seed(0)
    pt_a = torch.randn((m, k), dtype=torch.bfloat16)
    pt_b = torch.randn((k, n), dtype=torch.bfloat16)
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    a_grid = _num_cores_to_rectangle_core_range_set(num_a_cores, device)
    b_grid = _num_cores_to_rectangle_core_range_set(num_b_cores, device)
    a = ttnn.from_torch(
        pt_a,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (m, k // num_a_cores),
            core_grid=a_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )
    b = ttnn.from_torch(
        pt_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (k, n // num_b_cores),
            core_grid=b_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    out = ttnn.experimental.matmul_decode(a, b, global_cb=None)
    assert_with_pcc(ref, ttnn.to_torch(out).float(), 0.99)


def _make_gcb_and_operands(device, m, k, n, num_a_cores, num_slabs=2):
    """Build the activation, the DRAM receiver-contiguous weight, and the GCB.

    The B/receiver grid is the rectangle `_num_cores_to_rectangle_core_range_set`
    picks for `num_b_cores`, anchored at (0, 0). `bank_receivers_strided` maps ring
    position p to core `(p % ring_cols, p // ring_cols)`, so passing that rectangle's
    WIDTH as `ring_cols` makes ring position equal the core's row-major index -- which
    is the order matmul_decode assigns N-columns to B cores, and the order the weight's
    ND shards are laid out in. Passing `num_b_cores` instead is only correct when the
    rectangle happens to be a single row, and silently produces wrong results otherwise.
    """
    torch.manual_seed(0)
    num_dram_banks = device.dram_grid_size().x
    num_b_cores = n // 64
    assert num_b_cores % num_dram_banks == 0, f"{num_b_cores} receivers must divide across {num_dram_banks} banks"
    recv_per_bank = num_b_cores // num_dram_banks

    pt_a = torch.randn((m, k), dtype=torch.bfloat16)
    pt_b = torch.randn((k, n), dtype=torch.bfloat16)

    a_grid = _num_cores_to_rectangle_core_range_set(num_a_cores, device)
    b_grid = _num_cores_to_rectangle_core_range_set(num_b_cores, device)
    # Rectangle width == the row-major stride, hence ring_cols.
    ring_cols = b_grid.bounding_box().grid_size().x
    a = ttnn.from_torch(
        pt_a,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (m, k // num_a_cores),
            core_grid=a_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    weight = make_recv_contig_weight(
        device,
        pt_b.reshape(1, 1, k, n),
        num_dram_banks=num_dram_banks,
        ring_size=num_b_cores,
        dtype=ttnn.bfloat16,
        distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )

    # One GCB page == one receiver's whole [K, N/num_b_cores] slab.
    tile_bytes = 2048  # bfloat16 32x32
    slab_bytes = (k // 32) * ((n // num_b_cores) // 32) * tile_bytes
    gcb_size = num_slabs * slab_bytes

    bank_to_receivers = [
        (b, bank_receivers_strided(b, recv_per_bank, num_dram_banks, ring_cols=ring_cols))
        for b in range(num_dram_banks)
    ]
    gcb = ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher(device, bank_to_receivers, gcb_size)
    return pt_a, pt_b, a, weight, gcb, num_b_cores


def test_matmul_decode_gcb_output_spec(device):
    """With a GCB, the output grid/shard come from the GCB receivers, not from a
    legacy shard spec on the DRAM weight (which has none).

    The prefetcher must run even though this test only checks the output spec: once
    the in1 CB is GCB-backed the reader blocks until its page arrives, and a matmul
    launched without a prefetch request would wedge the device for every later test.
    """
    m, k, n = 32, 1024, 2048
    _, _, a, weight, gcb, num_b_cores = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(weight, 1)], global_cb=gcb)
        out = ttnn.experimental.matmul_decode(a, weight, global_cb=gcb)
        ttnn.synchronize_device(device)

    assert tuple(out.shape) == (m, n)
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert out.memory_config().shard_spec.shape == [m, n // num_b_cores]


def test_matmul_decode_prefetched_weights_pcc(device):
    """Full end-to-end: prefetcher pushes each receiver's weight slab into the GCB,
    matmul_decode consumes it, result matches torch."""
    m, k, n = 32, 1024, 2048
    pt_a, pt_b, a, weight, gcb, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(weight, 1)], global_cb=gcb)
        out = ttnn.experimental.matmul_decode(a, weight, global_cb=gcb)
        result = ttnn.to_torch(out).float()

    assert_with_pcc(ref, result, 0.99)
