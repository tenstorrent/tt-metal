# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: the Kimi K3 attention-residuals read (`models/demos/deepseek_v3_d_p/tt/attn_res`)
against the torch oracle, at the per-chip shape prefill actually runs.

Prefill chunks 5120 tokens across the sequence-parallel axis, so every chip sees
`5120 / sp` rows. On the Galaxy `(8, 4)` that is **640**, and 640 is the only row
count this file uses — the op's cost and its collective's algorithm both turn on it, and
a suite parametrized at 64 tokens exercises a reduction production never issues.

Two placements, both sharded:

  * `(2, 4)` — LoudBox. `d` split 4 ways, tokens split 2 ways. TP factor 4 is
    Galaxy's, so this arm covers the reduction Galaxy runs.
  * `(8, 4)` — Galaxy. Same TP factor over a wider sequence axis, which the op is
    indifferent to; it is here to be run on the box, not to add coverage.

No single-device arm. `tp_factor == 1` makes `_reduce_stats` the identity, so a green
`(1, 1)` run certifies a score chain the model never executes — the one thing this
file exists to gate is the reduction, and that arm has none.

`mesh_device` skips a placement asking for more chips than the host has, so this file
is inert rather than failing on a runner that cannot hold it — single-card Blackhole
SKUs collect it and skip both arms. CI runs it on `bh_loudbox`, where `(2, 4)` gates
and `(8, 4)` skips on chip count.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.attn_res.attn_res import EPS, attn_res
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes

PCC_GATE = 0.9999
REL_ERR_GATE = 2e-2

HIDDEN_SIZE = 7168
PER_CHIP_TOKENS = 640
READ_SITES = 24
PROJ_STD = 0.02

# `ttnn.all_reduce` needs an initialized fabric context on a real mesh; without it the
# op dies in the control plane rather than returning wrong numbers.
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}

PLACEMENTS = [
    pytest.param((2, 4), FABRIC, id="mesh-2x4"),
    pytest.param((8, 4), FABRIC, id="mesh-8x4"),
]

on_placements = pytest.mark.parametrize(
    "mesh_device, device_params", PLACEMENTS, indirect=["mesh_device", "device_params"]
)

# The op was brought up and measured only on Blackhole, and its mixture runs on
# `ttnn.experimental.fast_weighted_reduce_nc`, which has no Wormhole coverage.
pytestmark = pytest.mark.skipif(not is_blackhole(), reason="Kimi K3 AttnRes is brought up on Blackhole only")


def _pcc(got, want):
    stacked = torch.stack((got.double().reshape(-1), want.double().reshape(-1)))
    return torch.corrcoef(stacked)[0, 1].item()


def _rel_err(got, want):
    got, want = got.double(), want.double()
    return ((got - want).abs().max() / want.abs().max()).item()


def _make_case(num_tokens, num_sealed, seed=0):
    """One read's inputs: the live stream, `num_sealed` frozen snapshots, one folded query."""
    generator = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator)
    return (
        randn(num_tokens, HIDDEN_SIZE),
        randn(num_tokens, num_sealed, HIDDEN_SIZE),
        (1.0 + 0.1 * randn(HIDDEN_SIZE)) * (PROJ_STD * randn(HIDDEN_SIZE)),
    )


def _to_device(op, prefix_sum, block_residual, query, stream_mapper=None):
    mapper = op.stream_mapper if stream_mapper is None else stream_mapper
    to_tt = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=op.mesh_device, mesh_mapper=mapper
    )
    return (
        to_tt(prefix_sum.unsqueeze(0).unsqueeze(0)),
        to_tt(block_residual.permute(1, 0, 2).unsqueeze(0)) if block_residual.shape[1] else None,
        op.to_query(query),
    )


def _from_device(op, tensor):
    return ttnn.to_torch(tensor, mesh_composer=op.stream_composer).reshape(-1, HIDDEN_SIZE)


@on_placements
@pytest.mark.parametrize("num_sealed", [0, 1, 8], ids=["S0", "S1", "S8"])
def test_read_matches_reference(mesh_device, num_sealed, device_params):
    """The direct read at 640 rows per chip.

    `S = 0` is the identity path and communicates nothing, so it is the control: if it
    fails, the failure is placement rather than the reduction. `S = 8` is the full
    snapshot set, where every candidate-axis kernel appears."""
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    num_tokens = PER_CHIP_TOKENS * op.sp_factor
    assert op.shard_width == HIDDEN_SIZE // op.tp_factor

    prefix_sum, block_residual, query = _make_case(num_tokens, num_sealed)
    tt_prefix, tt_block, tt_query = _to_device(op, prefix_sum, block_residual, query)

    out = op.forward(tt_prefix, tt_block, tt_query)
    got = _from_device(op, out)
    ttnn.deallocate(out)

    want = attn_res(prefix_sum, block_residual, query, EPS)
    pcc, rel_err = _pcc(got, want), _rel_err(got, want)
    logger.info(
        f"{tuple(mesh_device.shape)} S={num_sealed} T={num_tokens} ({PER_CHIP_TOKENS}/chip): "
        f"PCC {pcc:.7f}, rel err {rel_err:.2e}"
    )
    assert pcc >= PCC_GATE, f"S={num_sealed}: PCC {pcc:.7f} < {PCC_GATE}"
    assert rel_err <= REL_ERR_GATE, f"S={num_sealed}: rel err {rel_err:.2e} > {REL_ERR_GATE}"


@on_placements
def test_split_matches_reference(mesh_device, device_params):
    """The split form over a whole 12-layer block — production's schedule.

    `inter_block` computes the sealed set once for all 24 read sites and `merge` folds
    the live stream in per site. Every site gets the same query here, so every site
    must land on the same read as the direct form's oracle."""
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    num_tokens = PER_CHIP_TOKENS * op.sp_factor

    prefix_sum, block_residual, query = _make_case(num_tokens, 8)
    tt_prefix, tt_block, tt_query = _to_device(op, prefix_sum, block_residual, query)
    want = attn_res(prefix_sum, block_residual, query, EPS)

    partials, shifts, masses = op.inter_block(tt_block, [tt_query] * READ_SITES)
    worst_pcc, worst_rel_err = 1.0, 0.0
    for site in range(READ_SITES):
        merged = op.merge(partials, shifts, masses, tt_prefix, tt_query, site)
        got = _from_device(op, merged)
        ttnn.deallocate(merged)
        worst_pcc = min(worst_pcc, _pcc(got, want))
        worst_rel_err = max(worst_rel_err, _rel_err(got, want))

    logger.info(
        f"{tuple(mesh_device.shape)} split x{READ_SITES} sites T={num_tokens}: "
        f"worst PCC {worst_pcc:.7f}, worst rel err {worst_rel_err:.3e}"
    )
    assert worst_pcc >= PCC_GATE, f"split: worst PCC {worst_pcc:.7f} < {PCC_GATE}"
    assert worst_rel_err <= REL_ERR_GATE, f"split: worst rel err {worst_rel_err:.3e} > {REL_ERR_GATE}"


@on_placements
def test_sequence_axis_communicates_nothing(mesh_device, device_params):
    """The exact gate: same tokens, two placements, bit-identical outputs.

    Run A shards `640 * sp` tokens over the SP rows, so row 0 holds the first 640. Run B
    replicates 640, so *every* row holds those same 640. Under this mapping the SP axis
    carries no traffic, so the two must agree to the last bit — gated at zero, not at a
    tolerance.

    This is what separates "reduced on the TP axis" from "reduced on some axis". A
    collective pointed at the SP axis mixes different tokens in run A and multiplies the
    statistics in run B; either way the two disagree. PCC against torch cannot see it,
    because both runs stay self-consistent within their own placement."""
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    num_tokens = PER_CHIP_TOKENS * op.sp_factor
    prefix_sum, block_residual, query = _make_case(num_tokens, 8)

    # `stream_mapper` shards dim 2 on the SP axis; dropping that entry replicates it.
    replicated_dims = [None, None]
    replicated_dims[op.tp_axis] = 3
    replicated = ttnn.ShardTensor2dMesh(mesh_device, dims=replicated_dims, mesh_shape=mesh_device.shape)

    outputs = []
    for tokens, mapper in ((num_tokens, None), (PER_CHIP_TOKENS, replicated)):
        tt_prefix, tt_block, tt_query = _to_device(
            op, prefix_sum[:tokens], block_residual[:tokens], query, stream_mapper=mapper
        )
        out = op.forward(tt_prefix, tt_block, tt_query)
        outputs.append(_from_device(op, out))
        ttnn.deallocate(out)

    sharded, duplicated = outputs
    # Both SP rows of run B ran identical inputs, so they must agree too.
    rows = duplicated[: 2 * PER_CHIP_TOKENS].float()
    row_delta = (rows[:PER_CHIP_TOKENS] - rows[PER_CHIP_TOKENS:]).abs().max().item()
    delta = (sharded[:PER_CHIP_TOKENS].float() - duplicated[:PER_CHIP_TOKENS].float()).abs().max().item()

    logger.info(f"SP rows agree to {row_delta:.3e}; sharded-vs-replicated max|delta| {delta:.3e}")
    assert row_delta == 0.0, f"two SP rows disagree by {row_delta:.3e} on identical inputs"
    assert delta == 0.0, (
        f"sharding {num_tokens} tokens over the SP axis changes the first {PER_CHIP_TOKENS} outputs "
        f"by up to {delta:.3e} — something is communicating on the sequence axis"
    )
