# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The op on a real 2D mesh, at the per-chip shape prefill runs.

Sequence on mesh axis 0, hidden on mesh axis 1, exactly as the residual stream is
already laid out by `tt_prefill_block.py:552-553` and `mla/rope.py:160-161`. Every
read all-reduces `2(S+1)` scalars per token across the TP axis; nothing of width
`d` ever crosses a rank boundary.

Prefill chunks 5120 tokens across the sequence-parallel axis, so **640 rows per
chip** is what the collective sees on the Galaxy, and it is the row count this file
uses wherever shape is what is under test. It is not an arbitrary size: the
collective picks its algorithm from the payload, and below roughly 313 rows/chip
the split form stops paying for itself (`ROOFLINE.md` §5, §7). A suite parametrized
at 64 tokens spends most of its arms on a reduction production never issues.

Two placements, both real boxes:

  * `(2, 4)` — LoudBox. Of the three valid 8-chip meshes it is the one that
    exercises both axes, and its TP factor of 4 is Galaxy's, so it covers the
    reduction Galaxy runs.
  * `(8, 4)` — Galaxy. Same TP factor over a wider sequence axis. The op is
    indifferent to the SP axis by construction and
    `test_sequence_axis_communicates_nothing` gates that at zero, so this arm is
    here to be run on the box rather than to add coverage. It skips on anything
    smaller than 32 chips.

Three kinds of gate carry this file, and they fail on different mistakes:

  * PCC against the torch oracle, which catches a missing reduction, a wrong global
    `d`, or an axis swap — all silent-wrong-answer bugs on a single device, where
    `tp_factor == 1` makes them no-ops.
  * `max|delta| == 0` between two placements of the same tokens. The SP axis
    communicates nothing, so that equality is exact rather than approximate, and it
    is the only gate that distinguishes "reduced on the right axis" from "reduced on
    an axis".
  * A mutation gate, which fails if deleting the collective still passes.
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

HIDDEN_SIZE = 7168
PER_CHIP_TOKENS = 640
READ_SITES = 24  # read sites per 12-layer block, the real per-block count

# `ttnn.all_reduce` needs an initialized fabric context — without this the op dies
# on `control_plane.cpp:2186` rather than returning wrong numbers. FABRIC_1D is
# what the analog's own 2x4 prefill config uses (`test_prefill_block.py:513-517`)
# and it is the right pairing for `Topology.Linear` on a single cluster axis.
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}

on_placements = pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((2, 4), FABRIC, id="mesh-2x4"),
        pytest.param((8, 4), FABRIC, id="mesh-8x4"),
    ],
    indirect=["mesh_device", "device_params"],
)


def _pcc(a, b):
    stacked = torch.stack((a.double().reshape(-1), b.double().reshape(-1)))
    return torch.corrcoef(stacked)[0, 1].item()


def _rel_err(got, want):
    got, want = got.double(), want.double()
    return ((got - want).abs().max() / want.abs().max()).item()


def _make_case(num_tokens, num_sealed, seed=0):
    generator = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator)
    return (
        randn(num_tokens, HIDDEN_SIZE),
        randn(num_tokens, num_sealed, HIDDEN_SIZE),
        (1.0 + 0.1 * randn(HIDDEN_SIZE)) * (PROJ_STD * randn(HIDDEN_SIZE)),
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


def _from_device(op, tensor, num_tokens=-1):
    return ttnn.to_torch(tensor, mesh_composer=op.stream_composer).reshape(num_tokens, HIDDEN_SIZE)


@on_placements
@pytest.mark.parametrize("num_sealed", [0, 1, 8])
def test_forward_matches_torch(mesh_device, num_sealed, device_params):
    """The direct read at 640 rows per chip, `d` split 4 ways.

    `num_sealed=0` is the identity path and communicates nothing, so it is the
    control: if it fails, the failure is placement, not the reduction. `S = 8` is the
    full snapshot set, where every candidate-axis kernel appears.

    `d` stays at 7168 on the mesh. Narrower `d` is a shard-width question, not a
    placement one, and `test_tt_attn_res.py` already walks it on one device."""
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
    assert pcc >= PCC_GATE, f"S={num_sealed}: TP PCC {pcc:.7f} < {PCC_GATE}"
    assert rel_err <= STAT_REL_TOL, f"S={num_sealed}: TP rel err {rel_err:.2e} > {STAT_REL_TOL}"


@on_placements
def test_split_matches_torch(mesh_device, device_params):
    """A whole 12-layer block through the split form — production's schedule.

    The unit here is the block, not the read: `inter_block` computes the sealed half
    once for all 24 sites and `merge` folds the live stream in per site, so pricing a
    single `merge` against a single `forward` would compare 24 sites of shared work
    against one site of total work.

    Every site gets the same query, so every site must land on the same read as the
    direct form's oracle. Which of the two forms is *faster* is a separate question
    and depends on the per-chip shape — `test_perf_block_split_vs_direct` decides it."""
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    num_tokens = PER_CHIP_TOKENS * op.sp_factor

    prefix_sum, block_residual, query = _make_case(num_tokens, 8)
    tt_prefix, tt_block, tt_query = _to_device(op, prefix_sum, block_residual, query)
    want = attn_res(prefix_sum, block_residual, query, EPS)

    partials, shifts, masses = op.inter_block(tt_block, [tt_query] * READ_SITES)
    worst_pcc, worst_rel_err = 1.0, 0.0
    for partial, shift, mass in zip(partials, shifts, masses):
        merged = op.merge(partial, shift, mass, tt_prefix, tt_query)
        got = _from_device(op, merged)
        ttnn.deallocate(merged)
        worst_pcc = min(worst_pcc, _pcc(got, want))
        worst_rel_err = max(worst_rel_err, _rel_err(got, want))

    logger.info(
        f"{tuple(mesh_device.shape)} split T={num_tokens} ({PER_CHIP_TOKENS}/chip) x{READ_SITES} sites: "
        f"worst PCC {worst_pcc:.7f}, worst rel err {worst_rel_err:.3e}"
    )
    assert worst_pcc >= PCC_GATE, f"split: worst PCC {worst_pcc:.7f} < {PCC_GATE}"
    assert worst_rel_err <= STAT_REL_TOL, f"split: worst rel err {worst_rel_err:.3e} > {STAT_REL_TOL}"


@on_placements
@pytest.mark.parametrize("per_chip_tokens", [639, 1000])
def test_ragged_token_count_matches_torch(mesh_device, per_chip_tokens, device_params):
    """Per-chip row counts that are not multiples of 32, which real prompts are not.

    `T` tile-pads to the next multiple of 32, so the padded rows are real tiles
    carrying whatever the allocator left there. Nothing reduces over `T`, so they
    must stay isolated from the live rows; if any of the dim-1 reductions leaked
    across `T` this is where it would show. On a mesh the padding lands on every SP
    row independently, which is the case a single device cannot reach."""
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    num_tokens = per_chip_tokens * op.sp_factor

    prefix_sum, block_residual, query = _make_case(num_tokens, 8)
    tt_prefix, tt_block, tt_query = _to_device(op, prefix_sum, block_residual, query)

    out = op.forward(tt_prefix, tt_block, tt_query)
    got = _from_device(op, out)
    ttnn.deallocate(out)

    want = attn_res(prefix_sum, block_residual, query, EPS)
    pcc, rel_err = _pcc(got, want), _rel_err(got, want)
    logger.info(f"{tuple(mesh_device.shape)} T={num_tokens} ({per_chip_tokens}/chip) ragged: PCC {pcc:.7f}")
    assert pcc >= PCC_GATE, f"{per_chip_tokens}/chip: ragged PCC {pcc:.7f} < {PCC_GATE}"
    assert rel_err <= STAT_REL_TOL, f"{per_chip_tokens}/chip: ragged rel err {rel_err:.2e} > {STAT_REL_TOL}"


@on_placements
def test_sequence_axis_communicates_nothing(mesh_device, device_params):
    """The exact gate. Same tokens, two placements, bit-identical outputs.

    Run A shards `640 * sp` tokens over the SP rows, so row 0 holds the first 640.
    Run B replicates 640, so *every* row holds those same 640. Under the chosen
    mapping the SP axis carries no traffic, so the two must agree to the last bit —
    gated at zero, not at a tolerance.

    This is what separates "reduced on the TP axis" from "reduced on some axis". A
    collective pointed at the SP axis mixes different tokens in run A and multiplies
    the statistics in run B; either way the two disagree. A collective spanning both
    axes does the same. Neither shows up in the PCC tests above, because both stay
    self-consistent within one placement."""
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
    assert row_delta == 0.0, f"the two SP rows disagree by {row_delta:.3e} on identical inputs"
    assert delta == 0.0, (
        f"sharding {num_tokens} tokens over the SP axis changes the first {PER_CHIP_TOKENS} outputs "
        f"by up to {delta:.3e} — something is communicating on the sequence axis"
    )


@on_placements
def test_statistics_reduction_is_load_bearing(mesh_device, device_params):
    """Delete the collective and the gate above must fail.

    Without it each rank scores against its own quarter of `d` while still dividing
    by the global `d`, so every rank builds a *different* softmax over the same
    candidates and the composed output is a chimera. Worth a test rather than a
    one-off mutation run: `tp_factor == 1` makes `_reduce_stats` the identity, so on
    a single device nothing in this module can tell the two apart."""
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    num_tokens = PER_CHIP_TOKENS * op.sp_factor
    prefix_sum, block_residual, query = _make_case(num_tokens, 8)
    tt_prefix, tt_block, tt_query = _to_device(op, prefix_sum, block_residual, query)
    want = attn_res(prefix_sum, block_residual, query, EPS)

    op._reduce_stats = lambda stats: stats
    out = op.forward(tt_prefix, tt_block, tt_query)
    pcc = _pcc(_from_device(op, out), want)
    ttnn.deallocate(out)

    logger.info(f"rank-local statistics give PCC {pcc:.7f} against a gate of {PCC_GATE}")
    assert pcc < PCC_GATE, f"PCC {pcc:.7f} still passes without the statistics all-reduce — the gate is blind"


@on_placements
@pytest.mark.parametrize("fold_stats", [True, False], ids=["folded", "unfolded"])
def test_depth_walk(mesh_device, fold_stats, device_params):
    """93 layers, 186 reads, 186 collectives, on the mesh.

    The per-read collective is what depth turns into a risk: a reduction that is
    merely close rather than correct compounds through 186 chained mixtures, and a
    topology mismatch shows up as a hang rather than a number. Gated relatively
    against torch-bf16, like the single-device harness — an absolute PCC at this
    depth measures bf16, not the mapping.

    Both statistics layouts must clear the same gate. The fold reassociates the
    partial sums that cross the collective, so it changes the arithmetic without
    changing the answer, and 186 chained reads is where that would show."""
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS, fold_stats=fold_stats)
    hidden_states, q_pre, q_post, q_out, weights = _make_stack(PER_CHIP_TOKENS * op.sp_factor, HIDDEN_SIZE, NUM_LAYERS)

    reference, _ = _walk_torch(hidden_states, weights, q_pre, q_post, q_out, torch.float32)
    analog, _ = _walk_torch(hidden_states, weights, q_pre, q_post, q_out, torch.bfloat16)

    sealed_after = []
    device, _ = _walk_device(
        mesh_device,
        hidden_states,
        weights,
        q_pre,
        q_post,
        q_out,
        HIDDEN_SIZE,
        record=lambda _, stream: sealed_after.append(stream.num_sealed),
        op=op,
    )

    assert torch.isfinite(device).all(), "distributed stream diverged"
    assert sealed_after[0] == 1 and sealed_after[-1] == 8
    assert len(sealed_after) == NUM_LAYERS and sealed_after.count(1) == BLOCK_SIZE

    device_pcc, analog_pcc = _pcc(device, reference), _pcc(analog, reference)
    norm_ratio = (device.double().norm() / reference.double().norm()).item()
    logger.info(
        f"{tuple(mesh_device.shape)} depth fold={fold_stats}: device PCC {device_pcc:.7f}, "
        f"torch-bf16 {analog_pcc:.7f}, norm ratio {norm_ratio:.6f}"
    )
    assert abs(norm_ratio - 1.0) <= 2e-2, f"output norm ratio {norm_ratio:.6f} — gross scale error"
    assert (
        device_pcc >= analog_pcc - DEPTH_PCC_SLACK
    ), f"device PCC {device_pcc:.7f} trails torch-bf16 {analog_pcc:.7f} by more than {DEPTH_PCC_SLACK}"


@on_placements
def test_chunked_prefill_carries_no_state(mesh_device, device_params):
    """Chunk position must not change a chunk's result, and does not.

    Prefill is chunked, so a 100k prompt walks this stack twenty times over. The
    claim `API_SPEC.md` makes is that AttnRes is immune to context growth: the op
    holds nothing across a chunk boundary, so chunk 20 costs and returns exactly
    what chunk 1 does. Everything about the op says that must be true — `forward`
    takes no position, length, mask or prior-chunk handle, the stream owns two
    tensors and frees both, and `T` enters no reduction — but "the signature has
    nowhere to put state" is an argument, not a measurement.

    This walks the chunks. Three chunks of *different* tokens go through one
    long-lived `TtAttnRes`, then the last chunk is re-walked alone on a fresh op,
    and the two must agree bit for bit. Identical inputs would prove nothing, so
    the chunks differ; a leak through the op — a cached tensor, a stale shape, a
    global semaphore reused across a boundary — lands as a non-zero delta.

    Run on the mesh because that is where the op has state worth leaking. On one
    device there are no semaphores and no collective; the per-call global
    semaphores the fabric path allocates are the plausible carrier, and they only
    exist here.

    What this does *not* establish: there is no chunked-prefill driver in the tree,
    so the loop is this test's own. It gates the op's contribution to chunking, not
    a KV cache or an attention mask."""
    num_chunks = 3
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    num_tokens = PER_CHIP_TOKENS * op.sp_factor
    _, q_pre, q_post, q_out, weights = _make_stack(num_tokens, HIDDEN_SIZE, NUM_LAYERS)

    # Distinct token content per chunk, shared weights and queries — a real prefill
    # advances the tokens and holds the stack fixed.
    generator = torch.Generator().manual_seed(11)
    chunks = [torch.randn(num_tokens, HIDDEN_SIZE, generator=generator) for _ in range(num_chunks)]
    outputs, curves = [], []
    for hidden_states in chunks:
        curve = []
        walked, _ = _walk_device(
            mesh_device,
            hidden_states,
            weights,
            q_pre,
            q_post,
            q_out,
            HIDDEN_SIZE,
            record=lambda _, stream: curve.append(stream.num_sealed),
            op=op,
        )
        outputs.append(walked)
        curves.append(curve)

    alone, _ = _walk_device(
        mesh_device,
        chunks[-1],
        weights,
        q_pre,
        q_post,
        q_out,
        HIDDEN_SIZE,
        record=lambda *_: None,
    )

    for index, curve in enumerate(curves):
        assert curve == curves[0], f"chunk {index} sealed {curve} against chunk 0's {curves[0]}"
    assert curves[0][-1] == 8 and len(curves[0]) == NUM_LAYERS

    spread = (outputs[-1] - alone).abs().max().item()
    logger.info(
        f"{tuple(mesh_device.shape)} chunk {num_chunks} of {num_chunks} against the same chunk walked "
        f"alone: max|delta| {spread:.3e} over {NUM_LAYERS} layers x {2 * NUM_LAYERS} reads"
    )
    assert spread == 0.0, (
        f"the last chunk differs by up to {spread:.3e} depending on whether {num_chunks - 1} chunks "
        "preceded it — the op carries state across a chunk boundary"
    )

    assert not torch.equal(outputs[0], outputs[-1]), "chunks share content, so the comparison is vacuous"
