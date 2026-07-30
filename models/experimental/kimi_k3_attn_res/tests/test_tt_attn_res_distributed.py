# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 8: the op on a real 2D mesh, in the mapping `DISTRIBUTION.md` settled on.

Sequence on mesh axis 0, hidden on mesh axis 1, exactly as the residual stream is
already laid out by `tt_prefill_block.py:552-553` and `mla/rope.py:160-161`. Every
read all-reduces `2(S+1)` scalars per token across the TP axis; nothing of width
`d` ever crosses a rank boundary.

Two gates carry this phase, and they fail on different mistakes:

  * PCC against the torch oracle, which catches a missing reduction, a wrong
    global `d`, or an axis swap — all of which are silent-wrong-answer bugs on a
    single device because `tp_factor == 1` makes them no-ops.
  * `max|delta| == 0` between a sequence-sharded run and a sequence-replicated
    one. The SP axis communicates nothing, so that equality is exact rather than
    approximate, and it is the only gate here that distinguishes "reduced on the
    right axis" from "reduced on an axis".
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.kimi_k3_attn_res.torch_functional.attn_res import BLOCK_SIZE, EPS, NUM_LAYERS, attn_res
from models.experimental.kimi_k3_attn_res.tests.test_tt_attn_res_depth import (
    DEPTH_PCC_SLACK,
    _make_stack,
    _walk_device,
    _walk_torch,
)
from models.experimental.kimi_k3_attn_res.tt.attn_res import TtAttnRes

PCC_GATE = 0.9999
STAT_REL_TOL = 2e-2
PROJ_STD = 0.02

# LoudBox: 8 chips, and only (8,1), (4,2), (2,4) are valid meshes. (2,4) is the
# one that exercises both axes, and its TP factor of 4 is Galaxy's.
MESH = (2, 4)
READ_SITES = 24

# `ttnn.all_reduce` needs an initialized fabric context — without this the op dies
# on `control_plane.cpp:2186` rather than returning wrong numbers. FABRIC_1D is
# what the analog's own 2x4 prefill config uses (`test_prefill_block.py:513-517`)
# and it is the right pairing for `Topology.Linear` on a single cluster axis.
on_mesh = pytest.mark.parametrize(
    "mesh_device, device_params",
    [(MESH, {"fabric_config": ttnn.FabricConfig.FABRIC_1D})],
    indirect=["mesh_device", "device_params"],
    ids=["mesh-2x4"],
)


def _pcc(a, b):
    stacked = torch.stack((a.double().reshape(-1), b.double().reshape(-1)))
    return torch.corrcoef(stacked)[0, 1].item()


def _rel_err(got, want):
    got, want = got.double(), want.double()
    return ((got - want).abs().max() / want.abs().max()).item()


def _make_case(num_tokens, hidden_size, num_sealed, seed=0):
    generator = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator)
    return (
        randn(num_tokens, hidden_size),
        randn(num_tokens, num_sealed, hidden_size),
        (1.0 + 0.1 * randn(hidden_size)) * (PROJ_STD * randn(hidden_size)),
    )


def _to_device(op, prefix_sum, block_residual, query, stream_mapper=None):
    """Place a case in the op's layout. `stream_mapper` overrides it so a test can
    replicate the sequence instead of sharding it."""
    mapper = op.stream_mapper if stream_mapper is None else stream_mapper
    to_tt = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=op.mesh_device, mesh_mapper=mapper
    )
    return (
        to_tt(prefix_sum.unsqueeze(0).unsqueeze(0)),
        to_tt(block_residual.permute(1, 0, 2).unsqueeze(0)) if block_residual.shape[1] else None,
        op.to_query(query),
    )


def _from_device(op, tensor, hidden_size):
    return ttnn.to_torch(tensor, mesh_composer=op.stream_composer).reshape(-1, hidden_size)


@on_mesh
@pytest.mark.parametrize("hidden_size", [256, 7168])
@pytest.mark.parametrize("num_sealed", [0, 1, 8])
def test_tp_forward_matches_torch(mesh_device, hidden_size, num_sealed):
    """The read with `d` split 4 ways and tokens split 2 ways.

    `num_sealed=0` is the identity path and communicates nothing, so it is the
    control: if it fails, the failure is placement, not the reduction."""
    num_tokens = 64
    prefix_sum, block_residual, query = _make_case(num_tokens, hidden_size, num_sealed)
    op = TtAttnRes(mesh_device, hidden_size=hidden_size, eps=EPS)
    assert (op.sp_factor, op.tp_factor) == MESH
    assert op.shard_width == hidden_size // MESH[1]

    tt_prefix, tt_block, tt_query = _to_device(op, prefix_sum, block_residual, query)
    out = op.forward(tt_prefix, tt_block, tt_query)
    got = _from_device(op, out, hidden_size)
    ttnn.deallocate(out)

    want = attn_res(prefix_sum, block_residual, query, EPS)
    pcc, rel_err = _pcc(got, want), _rel_err(got, want)
    logger.info(f"d={hidden_size} S={num_sealed}: TP forward PCC {pcc:.7f}, rel err {rel_err:.2e}")
    assert pcc >= PCC_GATE, f"d={hidden_size} S={num_sealed}: TP PCC {pcc:.7f} < {PCC_GATE}"
    assert rel_err <= STAT_REL_TOL, f"d={hidden_size} S={num_sealed}: TP rel err {rel_err:.2e} > {STAT_REL_TOL}"


@on_mesh
def test_tp_split_matches_forward(mesh_device):
    """The split form on the mesh, which is where the collective count is worst.

    For 24 read sites the direct form issues 24 collectives — one paired reduction
    per read, covering the sealed set and the live stream together. The split form
    issues 49: one for the sealed RMS, 24 for the per-site sealed dots, and 24
    paired ones in `merge`. Splitting amortizes the RMS and de-amortizes everything
    else. Phase 7 measured the split form 1.50x faster on one device; on a mesh
    that has to be re-measured, and this test only establishes that it is still
    *correct*."""
    hidden_size, num_tokens = 7168, 64
    prefix_sum, block_residual, query = _make_case(num_tokens, hidden_size, 8)
    op = TtAttnRes(mesh_device, hidden_size=hidden_size, eps=EPS)
    tt_prefix, tt_block, tt_query = _to_device(op, prefix_sum, block_residual, query)

    direct = op.forward(tt_prefix, tt_block, tt_query)
    want = _from_device(op, direct, hidden_size)
    ttnn.deallocate(direct)

    partials, shifts, masses = op.inter_block(tt_block, [tt_query] * READ_SITES)
    for read_site, (partial, shift, mass) in enumerate(zip(partials, shifts, masses)):
        merged = op.merge(partial, shift, mass, tt_prefix, tt_query)
        pcc = _pcc(_from_device(op, merged, hidden_size), want)
        ttnn.deallocate(merged)
        assert pcc >= PCC_GATE, f"read site {read_site}: TP split PCC {pcc:.7f} < {PCC_GATE}"


@on_mesh
def test_sequence_axis_communicates_nothing(mesh_device):
    """The exact gate. Same 32 tokens, two placements, bit-identical outputs.

    Run A shards 64 tokens over the two SP rows, so row 0 holds tokens 0-31. Run B
    replicates 32 tokens, so *both* rows hold tokens 0-31. Under the chosen mapping
    the SP axis carries no traffic, so run A's first 32 output rows must equal run
    B's to the last bit — gated at zero, not at a tolerance.

    This is what separates "reduced on the TP axis" from "reduced on some axis". A
    collective pointed at the SP axis mixes different tokens in run A and doubles
    the statistics in run B; either way the two disagree. A collective spanning
    both axes does the same. Neither shows up in the PCC tests above, because both
    stay self-consistent within one placement."""
    hidden_size, num_tokens = 7168, 64
    prefix_sum, block_residual, query = _make_case(num_tokens, hidden_size, 8)
    op = TtAttnRes(mesh_device, hidden_size=hidden_size, eps=EPS)

    # `stream_mapper` shards dim 2 on the SP axis; dropping that entry replicates it.
    replicated_dims = [None, None]
    replicated_dims[op.tp_axis] = 3
    replicated = ttnn.ShardTensor2dMesh(mesh_device, dims=replicated_dims, mesh_shape=mesh_device.shape)

    shared = num_tokens // op.sp_factor
    outputs = []
    for tokens, mapper in ((num_tokens, None), (shared, replicated)):
        tt_prefix, tt_block, tt_query = _to_device(
            op, prefix_sum[:tokens], block_residual[:tokens], query, stream_mapper=mapper
        )
        out = op.forward(tt_prefix, tt_block, tt_query)
        outputs.append(_from_device(op, out, hidden_size))
        ttnn.deallocate(out)

    sharded, duplicated = outputs
    # Both SP rows of run B ran identical inputs, so they must agree too.
    row_delta = (duplicated[:shared].float() - duplicated[shared : 2 * shared].float()).abs().max().item()
    delta = (sharded[:shared].float() - duplicated[:shared].float()).abs().max().item()
    logger.info(f"SP rows agree to {row_delta:.3e}; sharded-vs-replicated max|delta| {delta:.3e}")
    assert row_delta == 0.0, f"the two SP rows disagree by {row_delta:.3e} on identical inputs"
    assert delta == 0.0, (
        f"sharding {num_tokens} tokens over the SP axis changes the first {shared} outputs "
        f"by up to {delta:.3e} — something is communicating on the sequence axis"
    )


@on_mesh
def test_statistics_reduction_is_load_bearing(mesh_device):
    """Delete the collective and the gate above must fail.

    Without it each rank scores against its own quarter of `d` while still
    dividing by the global `d`, so every rank builds a *different* softmax over
    the same candidates and the composed output is a chimera. Worth a test rather
    than a one-off mutation run: `tp_factor == 1` makes `_reduce_stats` the
    identity, so on a single device nothing in this module can tell the two
    apart."""
    hidden_size, num_tokens = 7168, 64
    prefix_sum, block_residual, query = _make_case(num_tokens, hidden_size, 8)
    op = TtAttnRes(mesh_device, hidden_size=hidden_size, eps=EPS)
    tt_prefix, tt_block, tt_query = _to_device(op, prefix_sum, block_residual, query)
    want = attn_res(prefix_sum, block_residual, query, EPS)

    op._reduce_stats = lambda stats: stats
    out = op.forward(tt_prefix, tt_block, tt_query)
    pcc = _pcc(_from_device(op, out, hidden_size), want)
    ttnn.deallocate(out)

    logger.info(f"rank-local statistics give PCC {pcc:.7f} against a gate of {PCC_GATE}")
    assert pcc < PCC_GATE, f"PCC {pcc:.7f} still passes without the statistics all-reduce — the gate is blind"


@on_mesh
def test_tp_depth_walk(mesh_device):
    """93 layers, 186 reads, 186 collectives, on the mesh.

    The per-read collective is what depth turns into a risk here: a reduction that
    is merely close rather than correct compounds through 186 chained mixtures, and
    a topology mismatch shows up as a hang rather than a number. Gated relatively
    against torch-bf16, like the single-device harness — an absolute PCC at this
    depth measures bf16, not the mapping."""
    hidden_size, num_tokens = 7168, 64
    hidden_states, q_pre, q_post, q_out, weights = _make_stack(num_tokens, hidden_size, NUM_LAYERS)

    reference, _ = _walk_torch(hidden_states, weights, q_pre, q_post, q_out, torch.float32)
    analog, _ = _walk_torch(hidden_states, weights, q_pre, q_post, q_out, torch.bfloat16)

    op = TtAttnRes(mesh_device, hidden_size=hidden_size, eps=EPS)
    sealed_after = []
    device, _ = _walk_device(
        mesh_device,
        hidden_states,
        weights,
        q_pre,
        q_post,
        q_out,
        hidden_size,
        record=lambda _, stream: sealed_after.append(stream.num_sealed),
        op=op,
    )

    assert torch.isfinite(device).all(), "distributed stream diverged"
    assert sealed_after[0] == 1 and sealed_after[-1] == 8
    assert len(sealed_after) == NUM_LAYERS and sealed_after.count(1) == BLOCK_SIZE

    device_pcc, analog_pcc = _pcc(device, reference), _pcc(analog, reference)
    norm_ratio = (device.double().norm() / reference.double().norm()).item()
    logger.info(f"{MESH} depth: device PCC {device_pcc:.7f}, torch-bf16 {analog_pcc:.7f}, norm ratio {norm_ratio:.6f}")
    assert abs(norm_ratio - 1.0) <= 2e-2, f"output norm ratio {norm_ratio:.6f} — gross scale error"
    assert (
        device_pcc >= analog_pcc - DEPTH_PCC_SLACK
    ), f"{MESH} device PCC {device_pcc:.7f} trails torch-bf16 {analog_pcc:.7f} by more than {DEPTH_PCC_SLACK}"
