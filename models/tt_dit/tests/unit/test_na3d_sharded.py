# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sharded 3D neighborhood attention reassembles into the whole-volume answer.

Host-only: no device, no mesh. What is under test is the index arithmetic that a real halo
exchange would feed, so each shard is given the slice of the global tensors that
``neighbor_pad_async`` would deliver — its own rows plus the halo the plan asks for — and the
reassembled output must equal the unsharded one exactly.

The failure this is built to catch is silent. NATTEN's window keeps a constant size and shifts
inward at a volume edge, so a shard that computes bounds on its own extent treats every
interior seam as an edge and produces plausible, wrong values near each one.
``test_local_bounds_planning_is_wrong`` pins that, so the mistake cannot return as a passing
suite.
"""

import math

import pytest
import torch

from ...layers.na3d import AxisShard, na3d_torch, plan_na3d, plan_na3d_sharded, required_halo, window_bounds

# Kernels the DiffVAE decoder actually uses: det stages 1-2, det stages 3-4, diffusion stage.
DIFFVAE_KERNELS = [(3, 7, 7), (3, 5, 5), (11, 11, 11)]


def _shards(length: int, parts: int) -> list[AxisShard]:
    """Contiguous even-ish split of one axis, the layout a mesh dimension produces."""
    edges = [round(i * length / parts) for i in range(parts + 1)]
    return [AxisShard(length=length, start=a, stop=b) for a, b in zip(edges, edges[1:])]


def _buffer(tensor: torch.Tensor, shards: tuple[AxisShard, ...], halos: tuple[tuple[int, int], ...]) -> torch.Tensor:
    """The local view a halo exchange would hand this device: its rows plus neighbour rows."""
    index = [slice(None)]
    for shard, (left, right) in zip(shards, halos):
        index.append(slice(shard.start - left, shard.stop + right))
    return tensor[tuple(index)]


def _reassemble(dims, grids, results, heads, head_dim, dtype):
    out = torch.empty(1, *dims, heads, head_dim, dtype=dtype)
    for shards, piece in zip(grids, results):
        out[
            :, shards[0].start : shards[0].stop, shards[1].start : shards[1].stop, shards[2].start : shards[2].stop
        ] = piece
    return out


def _grid(dims, mesh):
    from itertools import product

    return list(product(*(_shards(d, p) for d, p in zip(dims, mesh))))


@pytest.mark.parametrize("kernel", DIFFVAE_KERNELS)
@pytest.mark.parametrize(
    "dims, mesh",
    [
        ((8, 32, 32), (1, 4, 8)),  # H/W over a 4x8 mesh, T local — the production layout
        ((8, 32, 32), (1, 8, 4)),  # transposed mesh assignment
        ((8, 24, 24), (1, 2, 2)),  # few, fat shards
        ((8, 24, 24), (1, 6, 6)),  # many, thin shards: halo comparable to the shard
        ((12, 16, 16), (2, 2, 2)),  # T sharded too, all three axes at once
    ],
)
@pytest.mark.parametrize("heads, head_dim", [(2, 32)])
def test_sharded_reassembles_to_whole_volume(*, dims, mesh, kernel, heads, head_dim):
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, *dims, heads, head_dim, dtype=torch.float64) for _ in range(3))

    expected = na3d_torch(q, k, v, kernel, scale=1.0)

    grid = _grid(dims, mesh)
    pieces = []
    for shards in grid:
        plan = plan_na3d_sharded(shards, kernel)
        halos = tuple(required_halo(s, min(kk, s.length)) for s, kk in zip(shards, kernel))
        local = [_buffer(x, shards, halos) for x in (q, k, v)]
        assert tuple(local[0].shape[1:4]) == plan.dims, f"buffer {tuple(local[0].shape[1:4])} != plan dims {plan.dims}"
        pieces.append(na3d_torch(*local, kernel, scale=1.0, plan=plan))

    actual = _reassemble(dims, grid, pieces, heads, head_dim, q.dtype)
    # Not bit-exact by construction: a shard's softmax runs over a differently shaped key
    # block, which reassociates the sum. float64 keeps that at rounding scale, so a tolerance
    # this tight still fails on any real window-arithmetic error.
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kernel", DIFFVAE_KERNELS)
@pytest.mark.parametrize("dims, mesh", [((8, 32, 32), (1, 4, 8)), ((12, 16, 16), (2, 2, 2))])
def test_halo_matches_the_window_rule(*, dims, mesh, kernel):
    """Halo is ``k//2`` at an interior seam and zero at a volume edge — never guessed."""
    for shards in _grid(dims, mesh):
        for axis, shard in enumerate(shards):
            k_axis = min(kernel[axis], shard.length)
            left, right = required_halo(shard, k_axis)
            starts, ends = window_bounds(shard.length, k_axis)
            assert left == shard.start - starts[shard.start]
            assert right == ends[shard.stop - 1] - shard.stop
            if shard.start == 0:
                assert left == 0, "no neighbour exists before the volume"
            if shard.stop == shard.length:
                assert right == 0, "no neighbour exists past the volume"
            # ``k//2`` holds only while a shard is at least as wide as the kernel. Narrower and
            # the inward shift clamps the window to the volume edge, so it reaches further past
            # the far side of the shard — up to ``k-1``. A caller sizing a halo exchange off
            # ``k//2`` would under-request exactly there.
            assert left <= k_axis - 1 and right <= k_axis - 1
            if shard.stop - shard.start >= k_axis:
                assert left <= k_axis // 2 and right <= k_axis // 2


def test_local_bounds_planning_is_wrong():
    """Planning a shard on its own extent corrupts interior seams — the bug this guards.

    Asserted as a failure rather than described in a comment: if a future change makes local
    and global planning agree, either the window rule stopped shifting inward or this test
    stopped testing anything, and both deserve to be noticed.
    """
    dims, kernel, mesh = (4, 16, 16), (3, 7, 7), (1, 2, 2)
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, *dims, 2, 16, dtype=torch.float64) for _ in range(3))
    expected = na3d_torch(q, k, v, kernel, scale=1.0)

    grid = _grid(dims, mesh)
    pieces = []
    for shards in grid:
        # The mistake: bounds from the shard's own extent, no halo, as if it were the volume.
        local = [x[:, :, shards[1].start : shards[1].stop, shards[2].start : shards[2].stop] for x in (q, k, v)]
        pieces.append(na3d_torch(*local, kernel, scale=1.0))

    wrong = _reassemble(dims, grid, pieces, 2, 16, q.dtype)
    assert not torch.allclose(wrong, expected, rtol=1e-6, atol=1e-6), "local-bounds planning should not match"


@pytest.mark.parametrize("kernel", DIFFVAE_KERNELS)
def test_halo_inflation_matches_the_window_rule(*, kernel):
    """Buffer inflation is exactly what the halo formula predicts, and it is the memory cost.

    This is the quantity that decides whether the decomposition pays: total buffered volume
    over total query volume is the per-chip memory multiplier a halo exchange imposes. Dense
    score elements are printed alongside because sharding moves them in the *opposite*
    direction — a shard is small enough to tile coarsely, where the whole volume is forced
    small tiles by the score budget.
    """
    dims = (8, 32, 32)
    whole = plan_na3d(dims, kernel)
    for mesh in ((1, 2, 2), (1, 4, 4), (1, 4, 8)):
        grid = _grid(dims, mesh)
        plans = [plan_na3d_sharded(s, kernel) for s in grid]

        buffered = sum(math.prod(p.dims) for p in plans)
        queried = sum(math.prod(p.output_dims) for p in plans)
        assert queried == math.prod(dims), "shards must partition the volume exactly"

        predicted = sum(
            math.prod((s.stop - s.start) + sum(required_halo(s, min(kk, s.length))) for s, kk in zip(shards, kernel))
            for shards in grid
        )
        assert buffered == predicted, f"buffer {buffered} != halo formula {predicted}"

        dense = sum(len(g.query_slices) * g.n_queries * g.n_keys for p in plans for g in p.groups)
        ideal = math.prod(dims) * math.prod(min(kk, d) for kk, d in zip(kernel, dims))
        print(
            f"kernel={kernel} mesh={mesh}: memory {buffered / queried:.2f}x  "
            f"dense {whole.waste_factor:.2f}x -> {dense / ideal:.2f}x"
        )
