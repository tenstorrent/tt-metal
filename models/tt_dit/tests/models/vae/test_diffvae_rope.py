# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The DiffVAE's RoPE encodings against one oracle and against each other.

One rotation, three device encodings. The deterministic stages permute each head's lanes at load
into two halves and rotate with contiguous half-width cos/sin in row-major volume form. Stage 5
keeps upstream's interleaved pairs and rotates in TILE with a pair-swap matmul, its table factored
into a frame piece and a time piece -- or fused in bricked site order when the stage keeps bricked.

Until these tests the halves path was pinned only through whole blocks against captured activations,
and the bricked table only through eight blocks of stage 5. Both were block-level PCC numbers, which
a lane-order mistake in one 64-wide head can hide inside. Here every encoding is checked in float32
against a torch oracle written from the definition, at the lane level, and the two natural-order
encodings are checked against each other -- the invariant any merge of the two implementations has
to preserve.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.tt_dit.models.vae.diffvae_ltx import apply_rope, rope_tables
from models.tt_dit.models.vae.diffvae_ltx_stage5 import (
    Grid,
    _apply_rope,
    _build_bricked_rope_tables,
    _build_rope_tables,
)
from models.tt_dit.models.vae.diffvae_ops import retile
from models.tt_dit.models.vae.diffvae_rope import default_rope_dim_split, pair_swap_matrix, rope_permutation
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.tensor import from_torch as sharded_from_torch
from models.tt_dit.utils.tensor import to_torch as gathered_to_torch

HEAD_DIM = 64
HEADS = 4
BASE = 10000.0
SPLIT = default_rope_dim_split(HEAD_DIM)


def _rope_oracle(x: torch.Tensor, positions: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Upstream's rotation from its definition: adjacent-lane pairs, one axis chunk at a time.

    ``x`` is ``(..., head_dim)`` in upstream's interleaved layout; each entry of ``positions`` is that
    axis's coordinate, broadcastable to ``x[..., 0]``. Angles in float64 the way upstream's numpy
    computes them, so the high frequencies do not round differently from the reference.
    """
    out = x.clone()
    offset = 0
    for pos, width in zip(positions, SPLIT):
        exponents = torch.arange(0, width, 2, dtype=torch.float64) / width
        inv_freq = 1.0 / BASE**exponents
        angles = pos.to(torch.float64)[..., None] * inv_freq
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even = x[..., offset : offset + width : 2]
        odd = x[..., offset + 1 : offset + width : 2]
        out[..., offset : offset + width : 2] = even * cos - odd * sin
        out[..., offset + 1 : offset + width : 2] = even * sin + odd * cos
        offset += width
    return out


def _volume_positions(volume: tuple[int, int, int]) -> tuple[torch.Tensor, ...]:
    """Per-axis coordinates over ``(T, H, W, 1)``, the trailing 1 broadcasting over heads."""
    t, h, w = volume
    return (
        torch.arange(t).view(t, 1, 1, 1).expand(t, h, w, 1),
        torch.arange(h).view(1, h, 1, 1).expand(t, h, w, 1),
        torch.arange(w).view(1, 1, w, 1).expand(t, h, w, 1),
    )


def _bricked_torch(x: torch.Tensor, volume: tuple[int, int, int], brick: tuple[int, int, int]) -> torch.Tensor:
    """``(T, H, W, C)`` -> ``(bricked_sites, C)``: brick index outermost, sites time-major inside,
    ghost sites zero. The same torch model of brick order as tests/unit/test_brick_activation.py."""
    bricks = [(extent + step - 1) // step for extent, step in zip(volume, brick)]
    channels = x.shape[-1]
    padded = torch.zeros(*(count * step for count, step in zip(bricks, brick)), channels)
    padded[: volume[0], : volume[1], : volume[2]] = x
    grid = padded.reshape(bricks[0], brick[0], bricks[1], brick[1], bricks[2], brick[2], channels)
    return grid.permute(0, 2, 4, 1, 3, 5, 6).reshape(-1, channels)


def _pair_swap(mesh_device):
    return ttnn.from_torch(pair_swap_matrix(HEAD_DIM), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)


def _fp32_compute(mesh_device):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _check(expected: torch.Tensor, got: torch.Tensor, *, what: str) -> None:
    """PCC for the ledger, plus a max-abs bound a swapped lane or a wrong-axis angle cannot pass."""
    assert tuple(got.shape) == tuple(expected.shape), f"{what}: {tuple(got.shape)} != {tuple(expected.shape)}"
    worst = (got - expected).abs().max().item()
    assert worst < 2e-2, f"{what}: max |diff| {worst:.4f}"
    assert_quality(expected, got, pcc=0.9999)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
@pytest.mark.diffvae_gate
def test_halves_and_pair_swap_rotations_agree(mesh_device: ttnn.MeshDevice):
    """The deterministic stages' halves rotation and stage 5's pair-swap rotation are one rotation.

    The halves path is fed lanes permuted the way the weight loader permutes the q/k projections,
    and its output is un-permuted before the comparison, so what is compared is the rotation upstream
    defines. Both encodings are held to the oracle and then to each other.
    """
    volume = (6, 8, 10)
    t, h, w = volume
    torch.manual_seed(0)
    x = torch.randn(1, t, h, w, HEADS, HEAD_DIM)
    expected = _rope_oracle(x, _volume_positions(volume))

    # Deterministic stages: lanes into halves order (the load-time fold), half-width tables,
    # row-major volume form, back out of halves order.
    perm = rope_permutation(SPLIT)
    inverse = torch.argsort(perm)
    cos, sin = rope_tables(volume, SPLIT, mesh_device=mesh_device, dtype=ttnn.float32)
    tt_halves = ttnn.from_torch(
        x[..., perm].contiguous(), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32
    )
    halves = ttnn.to_torch(apply_rope(tt_halves, cos, sin))[..., inverse]
    _check(expected, halves, what="halves rotation vs oracle")

    # Stage 5: interleaved pairs, factored frame/time table, pair-swap matmul in TILE.
    tables = _build_rope_tables(
        Grid(1, t, h, w), dim_split=SPLIT, base=BASE, num_heads=HEADS, mesh_device=mesh_device, dtype=ttnn.float32
    )
    tt_pairs = ttnn.from_torch(
        x.reshape(1, t, h * w * HEADS, HEAD_DIM).contiguous(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
    )
    rotated = _apply_rope(
        tt_pairs, tables, pair_swap=_pair_swap(mesh_device), compute_kernel_config=_fp32_compute(mesh_device)
    )
    pairs = ttnn.to_torch(retile(rotated, tuple(x.shape)))
    _check(expected, pairs, what="pair-swap rotation vs oracle")

    _check(halves, pairs, what="halves vs pair-swap")


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device, sharded",
    [pytest.param((1, 1), False, id="replicated-1x1"), pytest.param((4, 8), True, id="w_sharded-4x8")],
    indirect=["mesh_device"],
)
@pytest.mark.diffvae_gate
def test_bricked_table_matches_the_rotation(mesh_device: ttnn.MeshDevice, sharded: bool):
    """The fused bricked-order table rotates every real site the way natural order does.

    This is the table production stage 5 runs, and until now its only oracle was eight blocks of
    attention. The volume has ghost frames (T not a whole number of bricks) so the zeroed ghost rows
    are checked too, and the W-sharded case puts a different W offset on every column of the mesh,
    with the table replicated over the other axis exactly as the stage uploads it.
    """
    sp_axis = 1
    grid = Grid(1, 12, 8, 64)
    brick = (8, 2, 2)
    sp = int(list(mesh_device.shape)[sp_axis]) if sharded else 1
    assert grid.w % sp == 0
    w_local = grid.w // sp
    local = (grid.t, grid.h, w_local)

    torch.manual_seed(0)
    x = torch.randn(grid.t, grid.h, grid.w, HEADS, HEAD_DIM)
    expected = _rope_oracle(x, _volume_positions((grid.t, grid.h, grid.w)))

    def bricked_rows(volume_tensor: torch.Tensor, shard: int) -> torch.Tensor:
        """This shard's W-band in bricked site order, one row per (site, head), as stage 5 lays q out."""
        band = volume_tensor[:, :, shard * w_local : (shard + 1) * w_local]
        return _bricked_torch(band.reshape(grid.t, grid.h, w_local, HEADS * HEAD_DIM), local, brick).reshape(
            -1, HEAD_DIM
        )

    stacked_x = torch.stack([bricked_rows(x, p) for p in range(sp)]).reshape(1, 1, -1, HEAD_DIM)
    stacked_expected = torch.stack([bricked_rows(expected, p) for p in range(sp)]).reshape(1, 1, -1, HEAD_DIM)

    tables = _build_bricked_rope_tables(
        grid,
        brick,
        dim_split=SPLIT,
        base=BASE,
        num_heads=HEADS,
        mesh_device=mesh_device,
        dtype=ttnn.float32,
        w_shard=(sp, sp_axis) if sharded else None,
    )
    assert tables.fused is not None
    # A band of the first T-brick is exactly that brick's rows.
    assert tables.frames(0, brick[0]).fused.cos.shape[-2] == tables.sites_per_t_br

    if sharded:
        tt_x = sharded_from_torch(
            stacked_x.contiguous(),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32,
            mesh_axes=[None, None, sp_axis, None],
        )
    else:
        tt_x = ttnn.from_torch(stacked_x.contiguous(), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)

    rotated = _apply_rope(
        tt_x, tables, pair_swap=_pair_swap(mesh_device), compute_kernel_config=_fp32_compute(mesh_device)
    )
    got = gathered_to_torch(rotated, mesh_axes=[None, None, sp_axis, None]) if sharded else ttnn.to_torch(rotated)
    _check(stacked_expected, got, what="bricked rotation vs oracle")

    # Ghost rows carry no site: the table is zero there and so must the rotation be.
    ghost = (stacked_x == 0).all(dim=-1)
    assert ghost.any(), "the volume was meant to have ghost frames"
    assert got[ghost].abs().max().item() == 0.0
