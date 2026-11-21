# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests the design's Work Distribution requires beyond the acceptance test:

- multi-cluster shapes with ragged per-cluster and per-NEO counts at the default plan;
- non-default thread plans and handshake quanta (under STRIDED rings a NEO that did not run leaves its
  tiles unproduced, so the writer cannot finish: completion + a correct result shows every thread ran);
- the op in one device session after another program and before one of tt-metal's own.
"""

import pytest
import torch

import ttnn
from ttnn.operations.subtract import default_compute_kernel_config, subtract
from ttnn.operations.subtract.subtract_program_artifacts import THREAD_PLAN, create_program_artifacts, split_tiles


def _inputs(device, shape, seed=0, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    torch.manual_seed(seed)
    ta = torch.randn(*shape, dtype=torch.bfloat16)
    tb = torch.randn(*shape, dtype=torch.bfloat16)
    put = lambda t: ttnn.from_torch(  # noqa: E731
        t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=memory_config
    )
    return ta, tb, put(ta), put(tb)


def _expected(ta, tb):
    return (ta.double() - tb.double()).to(torch.bfloat16).float()


def _check(got, ta, tb):
    exp = _expected(ta, tb)
    diff = (got.float() - exp).abs()
    # 1 bfloat16 ulp of the expected value (the spec's grade), with a floor for values near zero.
    ulp = exp.abs().clamp(min=2**-100) * 2**-7
    assert bool((diff <= ulp + 1e-30).all()), f"max diff {diff.max().item()}"


def _run_with(a, b, *, threads, sync_tiles=None):
    out = ttnn.allocate_tensor_on_device(a.spec, a.device())
    spec, run_args, idx = create_program_artifacts(
        a, b, out, default_compute_kernel_config(), threads=threads, sync_tiles=sync_tiles
    )
    nums = {k.unique_id: k.num_threads for k in spec.kernels}
    assert nums == {"reader": threads[0], "compute": threads[1], "writer": threads[2]}, nums
    return ttnn.generic_op([a, b, out], spec, run_args, idx)


# 263 tiles (8 or 9 per cluster), 1,050 tiles (32 or 33), 1,024 tiles (32 each), 2 tiles.
@pytest.mark.parametrize("shape", [(1, 2, 8400), (1, 2, 33600), (2, 512, 1024), (1, 1, 17, 50)],
                         ids=lambda s: "x".join(map(str, s)))
def test_default_plan_multi_cluster(device, shape):
    ta, tb, a, b = _inputs(device, shape)
    _check(ttnn.to_torch(subtract(a, b)), ta, tb)


@pytest.mark.parametrize("threads", [(1, 1, 1), (2, 2, 2), (1, 4, 1), (4, 4, 2), (2, 4, 2)],
                         ids=lambda p: "R%dC%dW%d" % p)
@pytest.mark.parametrize("sync_tiles", [1, 3, 4])
def test_thread_plans_and_quanta(device, threads, sync_tiles):
    # [1, 2, 8400]: 263 tiles, 8 or 9 per cluster: not a multiple of C, and a ragged tail per NEO for every quantum.
    ta, tb, a, b = _inputs(device, (1, 2, 8400), seed=1)
    _check(ttnn.to_torch(_run_with(a, b, threads=threads, sync_tiles=sync_tiles)), ta, tb)


def test_default_plan_constant():
    assert THREAD_PLAN == (1, 1, 1)


@pytest.mark.parametrize("num_tiles", [1, 2, 13, 31, 32, 33, 263, 1050])
@pytest.mark.parametrize("min_tiles_per_core", [1, 4, 8])
def test_split_tiles_partitions_range(device, num_tiles, min_tiles_per_core):
    """The host split covers [0, num_tiles) exactly once in contiguous row-major ranges, engages
    min(grid, ceil(num_tiles / min_tiles_per_core)) clusters and is balanced to within one tile."""
    grid = device.compute_with_storage_grid_size()
    core_set, counts, starts = split_tiles(num_tiles, grid, min_tiles_per_core)
    engaged = min(grid.x * grid.y, -(-num_tiles // min_tiles_per_core))
    assert len(counts) == engaged == core_set.num_cores()
    order = sorted(counts, key=lambda c: starts[c])
    pos = 0
    for core in order:
        assert starts[core] == pos
        pos += counts[core]
    assert pos == num_tiles
    assert max(counts.values()) - min(counts.values()) <= 1


def test_after_another_program_and_before_tt_metal(device):
    """One device session: a tt-metal program, subtract (twice, default and a 4-NEO plan), then tt-metal's
    untilize of the result. Catches a sticky transaction tag left by an implicit-sync kernel."""
    # A tt-metal program first: tilize on device.
    torch.manual_seed(3)
    tr = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
    r = ttnn.from_torch(tr, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
    rt = ttnn.to_layout(r, ttnn.TILE_LAYOUT)

    ta, tb, a, b = _inputs(device, (1, 2, 8400), seed=4)
    out1 = subtract(a, b)
    out2 = _run_with(a, b, threads=(4, 4, 2))
    out3 = subtract(rt, rt)
    # tt-metal programs after ours.
    _check(ttnn.to_torch(ttnn.from_device(ttnn.to_layout(out1, ttnn.ROW_MAJOR_LAYOUT))), ta, tb)
    _check(ttnn.to_torch(ttnn.from_device(ttnn.to_layout(out2, ttnn.ROW_MAJOR_LAYOUT))), ta, tb)
    _check(ttnn.to_torch(ttnn.from_device(ttnn.to_layout(out3, ttnn.ROW_MAJOR_LAYOUT))), tr, tr)


def test_l1_output_placement(device):
    """R9: an L1 operand gives an L1 output; a DRAM operand a DRAM output."""
    ta, tb, a, b = _inputs(device, (1, 2, 400), memory_config=ttnn.L1_MEMORY_CONFIG)
    out = subtract(a, b, dtype=ttnn.bfloat16)
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    _check(ttnn.to_torch(out), ta, tb)
    ta, tb, a, b = _inputs(device, (1, 2, 400))
    out = subtract(a, b)
    assert out.memory_config().buffer_type == ttnn.BufferType.DRAM


def test_refusals(device):
    _, _, a, _ = _inputs(device, (1, 1, 32, 64))
    _, _, b, _ = _inputs(device, (1, 1, 32, 50))
    with pytest.raises(ValueError):
        subtract(a, b)  # X1
    _, _, b1, _ = _inputs(device, (1, 1, 32, 1))
    with pytest.raises(NotImplementedError):
        subtract(a, b1)  # broadcast not in Phase 0
    with pytest.raises(NotImplementedError):
        subtract(a, a, dtype=ttnn.float32)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False)
    with pytest.raises(NotImplementedError):
        subtract(a, a, compute_kernel_config=cfg)
