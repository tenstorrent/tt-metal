# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
``matmul_decode`` through the ``Sequential`` / ``Parallel`` fusion API, all three
schemes fed by the DRISC tensor prefetcher.

Three independent ``matmul_decode`` ops (A, B, C) -- each its own activation, DRAM
ND-sharded weight and ``GlobalCircularBuffer`` -- are placed on disjoint core rows
(one row per op) so any subset of them can be combined by ``Parallel`` without two
branches contending for the same cores. The three schemes:

  * ``test_matmul_decode_existing_sequential_baseline`` -- the three ops issued
    back-to-back via plain ``ttnn.experimental.matmul_decode`` calls, no fusion API.
    This is what callers (e.g. ``LinearDecode``'s ``use_prefetcher=True`` path) do
    today, and is the reference every fused scheme below is checked against.
  * ``test_matmul_decode_three_parallel_fusion`` -- all three fused into a single
    device program via ``Parallel(a, b, c)``.
  * ``test_matmul_decode_two_sequential_one_parallel_fusion`` -- two of the three
    chained into one branch via ``Sequential(a, b)`` (``b`` consumes ``a``'s output,
    a genuine producer/consumer pipeline on one core row), fused alongside the
    unrelated third op via ``Parallel(Sequential(a, b), c)``.

``Sequential`` fuses same-core pipelines by detecting a real data edge (one op's
output tensor is the next op's input) and threading them into one branch, so the
two-sequential scheme makes ``a``/``b`` a real two-stage pipeline rather than two
independent matmuls. Both stages declare the same CB indices, which is fine: the
fusion CB pool assigns them disjoint hardware slots and rewrites the ``cb_*``
named compile-time args every matmul_decode kernel reads its CB indices from.

All three schemes prefetch every weight through the tensor prefetcher
(``global_cb``), matching ``LinearDecode``'s ``use_prefetcher=True`` residency.
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
from tests.ttnn.nightly.unit_tests.operations.matmul.test_matmul_decode import get_tile_height
from models.experimental.ops.descriptors.fusion import Parallel, Sequential
from models.experimental.ops.descriptors.matmul_decode import matmul_decode


@pytest.fixture(autouse=True)
def _require_tensor_prefetcher(device):
    """Skip unless programmable DRAM cores are available on this device."""
    if not ttnn.experimental.is_tensor_prefetcher_supported(device):
        pytest.skip(
            "programmable DRAM cores unavailable (need Blackhole, firmware >= 19.12.0.0, "
            "and either no harvested DRAM channels or a single device)"
        )


def _shift_core_range_set(core_range_set, dx, dy):
    """Translate every ``CoreRange`` in ``core_range_set`` by ``(dx, dy)``."""
    ranges = []
    for r in core_range_set.ranges():
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(r.start.x + dx, r.start.y + dy),
                ttnn.CoreCoord(r.end.x + dx, r.end.y + dy),
            )
        )
    return ttnn.CoreRangeSet(ranges)


def _row_cores(num_cores, row):
    """A single-row ``CoreRangeSet`` of ``num_cores`` cores at ``(0, row)..(num_cores-1, row)``."""
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, row), ttnn.CoreCoord(num_cores - 1, row))})


def _make_branch_operands(device, row, num_cores, m, k, n, seed):
    """One independent ``matmul_decode``'s activation / prefetched weight / GCB, all on
    core row ``row``.

    A and B share the same row of ``num_cores`` cores -- a rectangle
    ``full_width_sharded`` requires at least 2 of -- so each branch's entire compute
    footprint (and the output it produces) lives in that one row, and rows never
    overlap when several branches are combined by ``Parallel``.
    """
    torch.manual_seed(seed)
    num_dram_banks = device.dram_grid_size().x
    assert num_cores % num_dram_banks == 0, f"{num_cores} receivers must divide across {num_dram_banks} banks"
    recv_per_bank = num_cores // num_dram_banks

    pt_a = torch.randn((m, k), dtype=torch.bfloat16)
    pt_b = torch.randn((k, n), dtype=torch.bfloat16)
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    row_cores = _row_cores(num_cores, row)
    a = ttnn.from_torch(
        pt_a,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((get_tile_height(m), 32)),
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (m, k // num_cores),
            core_grid=row_cores,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    weight = make_recv_contig_weight(
        device,
        pt_b.reshape(1, 1, k, n),
        num_dram_banks=num_dram_banks,
        ring_size=num_cores,
        dtype=ttnn.bfloat16,
    )

    # bank_receivers_strided anchors its ring at (0, 0); shift every bank's receiver set
    # down onto this branch's row so the GCB's receivers are exactly `row_cores`.
    bank_to_receivers = [
        (
            b,
            _shift_core_range_set(
                bank_receivers_strided(b, recv_per_bank, num_dram_banks, ring_cols=num_cores), 0, row
            ),
        )
        for b in range(num_dram_banks)
    ]
    slab_tiles = (k // 32) * ((n // num_cores) // 32)
    slab_bytes = slab_tiles * weight.tile.get_tile_size(weight.dtype)
    # Two slabs deep: the minimum a one-page-per-slab GCB needs to stream (the reader holds
    # one page un-acked while the sender delivers the next).
    gcb = ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher(
        device, bank_to_receivers, 2 * slab_bytes
    )

    return dict(a=a, weight=weight, gcb=gcb, ref=ref, K=k, N=n, row_cores=row_cores)


def _branch_params(device):
    """Three independent ``(m, k, n)`` branches, one per core row, sized off this device's
    DRAM bank count so every shard is tile-aligned regardless of architecture."""
    num_dram_banks = device.dram_grid_size().x
    grid = device.compute_with_storage_grid_size()
    if num_dram_banks < 2 or grid.x < num_dram_banks or grid.y < 3:
        pytest.skip(
            f"device grid {grid.x}x{grid.y} with {num_dram_banks} DRAM banks is too small for "
            f"three disjoint {num_dram_banks}-core rows"
        )
    m = 32
    k = n = num_dram_banks * 32
    return [
        dict(row=0, num_cores=num_dram_banks, m=m, k=k, n=n, seed=100),
        dict(row=1, num_cores=num_dram_banks, m=m, k=k, n=n, seed=200),
        dict(row=2, num_cores=num_dram_banks, m=m, k=k, n=n, seed=300),
    ]


def _make_chained_stage_b_operands(device, row, num_cores, n, stage_a_ref, seed):
    """A second matmul_decode stage on the same row as (and consuming the output of)
    another ``matmul_decode`` -- the weight/GCB half of a real producer/consumer pair.

    ``Sequential`` fuses same-core pipelines by detecting an internal edge (one op's
    output tensor is the next op's input) and threading data between their kernels
    directly; it is not meant to group independent ops with no such edge onto one
    core range. So this builds only the *weight side* of stage 2; the caller
    supplies stage 1's output tensor as stage 2's activation, making the edge real.
    """
    torch.manual_seed(seed)
    num_dram_banks = device.dram_grid_size().x
    assert num_cores % num_dram_banks == 0, f"{num_cores} receivers must divide across {num_dram_banks} banks"
    recv_per_bank = num_cores // num_dram_banks

    pt_weight = torch.randn((n, n), dtype=torch.bfloat16)
    ref = stage_a_ref @ pt_weight.to(torch.float32)

    weight = make_recv_contig_weight(
        device,
        pt_weight.reshape(1, 1, n, n),
        num_dram_banks=num_dram_banks,
        ring_size=num_cores,
        dtype=ttnn.bfloat16,
    )
    bank_to_receivers = [
        (
            b,
            _shift_core_range_set(
                bank_receivers_strided(b, recv_per_bank, num_dram_banks, ring_cols=num_cores), 0, row
            ),
        )
        for b in range(num_dram_banks)
    ]
    slab_tiles = (n // 32) * ((n // num_cores) // 32)
    slab_bytes = slab_tiles * weight.tile.get_tile_size(weight.dtype)
    gcb = ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher(
        device, bank_to_receivers, 2 * slab_bytes
    )
    return dict(weight=weight, gcb=gcb, ref=ref, K=n, N=n)


def _branch_params_two_sequential(device):
    """Sizes/rows for the ``Sequential(a, b)`` + ``c`` scheme: ``a``/``b`` share row 0
    (``b`` consumes ``a``'s output directly), ``c`` gets a disjoint row 1."""
    num_dram_banks = device.dram_grid_size().x
    grid = device.compute_with_storage_grid_size()
    if num_dram_banks < 2 or grid.x < num_dram_banks or grid.y < 2:
        pytest.skip(
            f"device grid {grid.x}x{grid.y} with {num_dram_banks} DRAM banks is too small for two "
            f"disjoint {num_dram_banks}-core rows"
        )
    m = 32
    k = n = num_dram_banks * 32
    return m, k, n, num_dram_banks


def _prefetch_all(device, branches):
    """Queue one prefetch request per branch's (distinct) GCB.

    Each GCB here carries exactly one weight, so there is no cross-branch FIFO
    ordering constraint -- unlike the single shared GCB ``DECODE_GCB_GROUP`` uses in
    ``models/experimental/deepseek_v4_flash``, request order across these three
    doesn't matter.
    """
    for br in branches:
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(br["weight"], 1)], global_cb=br["gcb"])


def test_matmul_decode_existing_sequential_baseline(device):
    """Baseline: three independent prefetched ``matmul_decode`` calls, issued
    back-to-back with the plain (non-fusion) API -- what ``LinearDecode``'s
    ``use_prefetcher=True`` path does today. Every other scheme in this file is
    checked against the same three references."""
    branches = [_make_branch_operands(device, **p) for p in _branch_params(device)]

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        _prefetch_all(device, branches)
        results = [
            ttnn.to_torch(ttnn.experimental.matmul_decode(br["a"], br["weight"], global_cb=br["gcb"])).float()
            for br in branches
        ]

    for br, result in zip(branches, results):
        assert_with_pcc(br["ref"], result, 0.99)


def test_matmul_decode_three_parallel_fusion(device):
    """All three ``matmul_decode`` ops fused into a single device program via
    ``Parallel(a, b, c)``, each still fed by its own prefetched ``global_cb``."""
    branches = [_make_branch_operands(device, **p) for p in _branch_params(device)]

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        _prefetch_all(device, branches)

        descs = [matmul_decode(br["a"], br["weight"], K=br["K"], N=br["N"], global_cb=br["gcb"]) for br in branches]
        Parallel(*descs).run()
        results = [ttnn.to_torch(d.output_tensors[0]).float() for d in descs]

    for br, result in zip(branches, results):
        assert_with_pcc(br["ref"], result, 0.99)


def test_matmul_decode_two_sequential_one_parallel_fusion(device):
    """Two of the three ``matmul_decode`` ops chained via ``Sequential``, fused
    alongside the third via ``Parallel(Sequential(a, b), c)``.

    ``a`` and ``b`` are a genuine two-stage pipeline on one core row -- ``b``'s
    activation *is* ``a``'s output tensor -- so ``Sequential`` has the producer/
    consumer edge it needs to thread them into one branch (see
    ``_make_chained_stage_b_operands``). ``c`` is unrelated and runs
    alongside the (a, b) pipeline via ``Parallel`` on a disjoint row -- the "2 in
    sequential, in parallel with the third" split.
    """
    m, k, n, num_cores = _branch_params_two_sequential(device)

    stage_a = _make_branch_operands(device, row=0, num_cores=num_cores, m=m, k=k, n=n, seed=100)
    stage_b = _make_chained_stage_b_operands(
        device, row=0, num_cores=num_cores, n=n, stage_a_ref=stage_a["ref"], seed=200
    )
    stage_c = _make_branch_operands(device, row=1, num_cores=num_cores, m=m, k=k, n=n, seed=300)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        for br in (stage_a, stage_b, stage_c):
            ttnn.experimental.queue_tensor_prefetcher_request(device, [(br["weight"], 1)], global_cb=br["gcb"])

        a_desc = matmul_decode(
            stage_a["a"], stage_a["weight"], K=stage_a["K"], N=stage_a["N"], global_cb=stage_a["gcb"]
        )
        # Both stages declare the same CB indices and share a core row; the fusion CB pool
        # allocates them disjoint hardware slots and rewrites each kernel's named "cb_*" args,
        # so neither factory needs to know the other is there.
        b_desc = matmul_decode(
            a_desc.output_tensors[0],
            stage_b["weight"],
            K=stage_b["K"],
            N=stage_b["N"],
            global_cb=stage_b["gcb"],
        )
        c_desc = matmul_decode(
            stage_c["a"], stage_c["weight"], K=stage_c["K"], N=stage_c["N"], global_cb=stage_c["gcb"]
        )

        Parallel(Sequential(a_desc, b_desc), c_desc).run()

        result_b = ttnn.to_torch(b_desc.output_tensors[0]).float()
        result_c = ttnn.to_torch(c_desc.output_tensors[0]).float()

    assert_with_pcc(stage_b["ref"], result_b, 0.99)
    assert_with_pcc(stage_c["ref"], result_c, 0.99)
