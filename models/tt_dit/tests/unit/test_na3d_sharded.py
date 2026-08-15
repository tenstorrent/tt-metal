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

from ...layers.na3d import (
    AxisShard,
    na3d_torch,
    plan_na3d,
    plan_na3d_mesh,
    plan_na3d_sharded,
    required_halo,
    uniform_halo,
    window_bounds,
)

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


def _uniform_buffer(tensor: torch.Tensor, shards: tuple[AxisShard, ...], halo: tuple[int, ...]) -> torch.Tensor:
    """What ``neighbor_pad_async`` hands every device: the same halo width on all of them.

    Edge-replicated where no neighbour exists, matching ``padding_mode="replicate"``. Those
    rows are pad, not data — a correct plan never reads them, which is what the test asserts.
    """
    out = tensor
    for axis, (shard, width) in enumerate(zip(shards, halo)):
        dim = axis + 1
        lo, hi = shard.start - width, shard.stop + width
        pieces = []
        for index in range(lo, hi):
            clamped = min(max(index, 0), shard.length - 1)
            pieces.append(out.narrow(dim, clamped, 1))
        out = torch.cat(pieces, dim=dim)
    return out


@pytest.mark.parametrize("kernel", DIFFVAE_KERNELS)
@pytest.mark.parametrize("dims, mesh", [((8, 32, 32), (1, 4, 8)), ((8, 24, 24), (1, 6, 6))])
def test_uniform_halo_reassembles_to_whole_volume(*, dims, mesh, kernel):
    """One halo width for every device, as the halo-exchange op imposes.

    Edge devices receive replicated pad where a neighbour would be. The result must be
    identical to the whole-volume run, which is only true if the plan's global bounds keep it
    out of that pad.
    """
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, *dims, 2, 32, dtype=torch.float64) for _ in range(3))
    expected = na3d_torch(q, k, v, kernel, scale=1.0)

    halo = tuple(uniform_halo(d, p, kk) for d, p, kk in zip(dims, mesh, kernel))
    grid = _grid(dims, mesh)
    pieces = []
    for shards in grid:
        plan = plan_na3d_sharded(shards, kernel, halo=halo)
        local = [_uniform_buffer(x, shards, halo) for x in (q, k, v)]
        assert tuple(local[0].shape[1:4]) == plan.dims
        pieces.append(na3d_torch(*local, kernel, scale=1.0, plan=plan))

    actual = _reassemble(dims, grid, pieces, 2, 32, q.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


def _shape_signature(plan):
    """What a ttnn program dispatch depends on: buffer, group count, per-group work sizes."""
    return (
        plan.dims,
        len(plan.groups),
        tuple(sorted((g.n_queries, g.n_keys, len(g.query_slices)) for g in plan.groups)),
    )


@pytest.mark.parametrize("kernel", DIFFVAE_KERNELS)
@pytest.mark.parametrize("dims, mesh", [((4, 32, 32), (1, 4, 4)), ((4, 32, 32), (1, 4, 8)), ((4, 24, 24), (1, 2, 2))])
def test_uniform_spans_give_every_device_one_shape(*, dims, mesh, kernel):
    """Without uniform spans a mesh needs several programs; with them, one.

    An edge shard's windows clamp, so its key spans are shorter than an interior shard's.
    Correct, but ttnn dispatches one program across the mesh, so the shapes have to agree.
    Both the collapse to a single shape and the unchanged result are asserted — the surplus
    keys a uniform span pulls in must be masked, not attended.
    """
    halo = tuple(uniform_halo(d, p, kk) for d, p, kk in zip(dims, mesh, kernel))
    grid = _grid(dims, mesh)

    varying = {_shape_signature(plan_na3d_sharded(s, kernel, halo=halo)) for s in grid}
    uniform = {_shape_signature(plan_na3d_sharded(s, kernel, halo=halo, uniform_spans=True)) for s in grid}
    assert len(uniform) == 1, f"uniform spans still produced {len(uniform)} shapes"
    assert len(varying) >= len(uniform), "uniform spans should not increase shape count"

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, *dims, 2, 16, dtype=torch.float64) for _ in range(3))
    expected = na3d_torch(q, k, v, kernel, scale=1.0)
    pieces = []
    for shards in grid:
        plan = plan_na3d_sharded(shards, kernel, halo=halo, uniform_spans=True)
        local = [_uniform_buffer(x, shards, halo) for x in (q, k, v)]
        pieces.append(na3d_torch(*local, kernel, scale=1.0, plan=plan))
    torch.testing.assert_close(_reassemble(dims, grid, pieces, 2, 16, q.dtype), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "dims, mesh, kernel, budget",
    [
        ((4, 32, 32), (1, 4, 4), (3, 7, 7), 2**22),  # single-tile: uniform_spans already sufficed
        ((4, 32, 32), (1, 4, 4), (3, 7, 7), 2**14),  # multi-tile: uniform_spans alone gave 3 shapes
        ((4, 32, 32), (1, 2, 4), (3, 5, 5), 2**12),  # 4x4x2 tiles per shard
        ((4, 24, 24), (1, 2, 2), (11, 11, 11), 2**14),
        ((8, 32, 64), (1, 4, 8), (3, 7, 7), 2**13),
    ],
)
def test_mesh_plans_share_one_shape_and_stay_correct(*, dims, mesh, kernel, budget):
    """The canonical padded group set: one dispatch shape per mesh, whatever the tiling.

    ``uniform_spans`` equalises key counts but not grouping, so a multi-tile shard still
    diverges — an edge device meets window regimes an interior one never does. Padding to the
    mesh-wide union of geometries closes that. The padding is attended and thrown away, so the
    result must be identical to the whole-volume run.
    """
    halo = tuple(uniform_halo(d, p, kk) for d, p, kk in zip(dims, mesh, kernel))
    grid = _grid(dims, mesh)
    plans = plan_na3d_mesh(grid, kernel, halo=halo, budget=budget)

    assert len({_shape_signature(p) for p in plans}) == 1, "mesh plans must agree on one shape"
    # Group i must mean the same geometry on every device: the mesh builder pairs them by index.
    for plan in plans[1:]:
        assert [g.geometry for g in plan.groups] == [g.geometry for g in plans[0].groups]

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, *dims, 2, 16, dtype=torch.float64) for _ in range(3))
    expected = na3d_torch(q, k, v, kernel, scale=1.0)
    pieces = [
        na3d_torch(*[_uniform_buffer(x, shards, halo) for x in (q, k, v)], kernel, scale=1.0, plan=plan)
        for shards, plan in zip(grid, plans)
    ]
    torch.testing.assert_close(_reassemble(dims, grid, pieces, 2, 16, q.dtype), expected, rtol=0, atol=1e-12)


def test_padding_tiles_are_excluded_from_the_output():
    """Padded tiles must be discarded, not written — the restore indexes around them."""
    dims, mesh, kernel = (4, 32, 32), (1, 4, 4), (3, 7, 7)
    halo = tuple(uniform_halo(d, p, kk) for d, p, kk in zip(dims, mesh, kernel))
    plans = plan_na3d_mesh(_grid(dims, mesh), kernel, halo=halo, budget=2**14)

    padded = sum(len(g.query_slices) for g in plans[0].groups)
    real = sum(g.kept_tiles for g in plans[0].groups)
    assert padded > real, "this geometry is supposed to need padding"
    # Every device answers exactly its own queries, no matter how many tiles were dispatched.
    for plan in plans:
        covered = sum(g.kept_tiles * g.n_queries for g in plan.groups)
        assert covered == plan.output_dims[0] * plan.output_dims[1] * plan.output_dims[2]


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
