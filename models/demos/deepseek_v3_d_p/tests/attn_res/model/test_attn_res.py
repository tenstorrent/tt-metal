# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: the Kimi K3 attention-residuals read (`models/demos/deepseek_v3_d_p/tt/attn_res`)
against the torch oracle, at the per-chip shape prefill actually runs.

One placement, one shape: `(2, 4)` on a LoudBox, 1280 tokens split 2 ways over the
sequence axis and `d` split 4 ways over the tensor axis, so every chip holds **640**
rows and `d/4 = 1792` columns. That row count is prefill's own — it chunks 5120 tokens
across an 8-deep sequence axis on the Galaxy — and the op's cost and its collective's
algorithm both turn on it, so a suite parametrized at 64 tokens would exercise a
reduction production never issues. TP factor 4 is Galaxy's, which is what makes this
box's reduction the same reduction Galaxy runs.

No single-device arm. The read's exchange is what its one dispatch is built around, so
`TtAttnRes` rejects `tp_factor == 1` outright rather than degrading to something the
model never executes. No Galaxy `(8, 4)` arm either: it holds the same TP factor over a
wider sequence axis, which the op is indifferent to, so it costs 32 chips to re-run the
reduction this arm already covers.

The parametrization that remains is over branches, not shapes. `S` selects whether the
statistics cross the tensor axis folded or unfolded, and the walk covers the seal cadence
the single reads cannot.

`mesh_device` skips a placement asking for more chips than the host has, so this file is
inert rather than failing on a runner that cannot hold it — single-card Blackhole SKUs
collect it and skip. CI runs it on `bh_loudbox`.

Queries are random unless something on the box names real ones, and every gate below runs
on whatever it gets. `TT_KIMI_K3_PREFILL_TTNN_CACHE` pointing at a complete tensorbin cache
is enough on its own, which is the state a brought-up model ships in; `KIMI_K3_CKPT` naming
a checkpoint that holds the stack's `res_norm`/`res_proj` weights builds that cache first.
`tests/attn_res/fetch_query_weights.py` stages such a checkpoint, by hand — CI sets neither
variable and takes the random arm.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.attn_res.attn_res import EPS, attn_res, attn_res_stack
from models.demos.deepseek_v3_d_p.reference.attn_res.weights import fold_queries
from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import (
    attn_res_tensor_cache_path,
    load_attn_res_state_dict,
)
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import BLOCK_SIZE, attn_res_stack_split
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import AttnResWeights

PCC_GATE = 0.9999
REL_ERR_GATE = 2e-2

HIDDEN_SIZE = 7168
PER_CHIP_TOKENS = 640
READ_SITES = 24

# On a unit-RMS stream the scores are `N(0, ‖q‖₂²)`, so the folded query's norm is the
# softmax's temperature and the only thing about the weights this op can be sensitive to.
# This lands at `‖q‖₂ ≈ 1.7`; K3's own query weights run 0.07 to 0.23 over a block, which
# is a near-uniform softmax and a milder shift for the online rescale to carry. The scale
# here is kept above the checkpoint's deliberately — it is the harder of the two.
PROJ_STD = 0.02

# One `TtAttnRes` serves the whole stack, so its cache namespace names the op and the
# per-layer part of the name lives inside it.
CACHE_PREFIX = "attn_res"

# The mesh axis `d` splits over. Named here and passed to the op, because a query sharded
# on one axis and reduced on another is a mismatch nothing downstream can detect.
TP_AXIS = 1

# Layer 0 has no sealed snapshot so its pre-attention read is skipped, which is why 93
# layers hold 187 queries and take 186 reads: 92 pre, 93 post, and one model-level read
# after the stack.
LAYERS = 93
READS = 2 * LAYERS

# Scales each module's contribution so 93 rounds of accumulation stay in bf16's range.
MODULE_SCALE = 0.02

# `ttnn.all_reduce` needs an initialized fabric context on a real mesh; without it the
# op dies in the control plane rather than returning wrong numbers.
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}

PLACEMENTS = [pytest.param((2, 4), FABRIC, id="mesh-2x4")]

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
    """One read's inputs: the live stream and `num_sealed` frozen snapshots."""
    generator = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator)
    return randn(num_tokens, HIDDEN_SIZE), randn(num_tokens, num_sealed, HIDDEN_SIZE)


def _queries(op, checkpoint_dir, seed=0):
    """Every folded query the stack holds: 93 pre-attention, 93 post-attention, one final.

    The checkpoint's own when there is one; otherwise whatever the op holds, read back off
    device so a published tensorbin cache with no checkpoint beside it still has an oracle.
    Read back the reference is the op's own bf16 query rather than the fp32 fold of it, so
    that arm gates the read arithmetic alone. Either way the sites do not share a query,
    which is what makes `merge`'s site index observable.
    """
    if checkpoint_dir is not None:
        return fold_queries(load_attn_res_state_dict(checkpoint_dir, LAYERS), LAYERS)

    if op.weights is not None:
        # Composing a TP-sharded row also stacks the SP copies of it; they are identical.
        compose = lambda q: ttnn.to_torch(q, mesh_composer=op.stream_composer).reshape(-1, HIDDEN_SIZE)[0].float()
        return (
            [compose(q) for q in op.weights.pre],
            [compose(q) for q in op.weights.post],
            compose(op.weights.output),
        )

    generator = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator)
    fold = lambda: (1.0 + 0.1 * randn(HIDDEN_SIZE)) * (PROJ_STD * randn(HIDDEN_SIZE))
    return [fold() for _ in range(LAYERS)], [fold() for _ in range(LAYERS)], fold()


def _make_op(mesh_device, checkpoint_dir):
    """The op, holding real queries whenever anything on the box names some.

    A complete tensorbin cache is enough on its own — that is the state a brought-up model
    ships in, and it reads with no checkpoint present. A checkpoint without one builds it
    first. Neither costs measurable time at 5 MB of `[d]` vectors; the cache is here so the
    real-weight path is the one the block hands every other module, rather than one this op
    alone would need special-casing for.
    """
    build = lambda **kwargs: TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS, tp_axis=TP_AXIS, **kwargs)
    cache_path = attn_res_tensor_cache_path(mesh_device, TP_AXIS, checkpoint_dir)
    cache_args = dict(num_layers=LAYERS, tensor_parallel_axis=TP_AXIS)

    if not AttnResWeights.check_cache_complete(cache_path, CACHE_PREFIX, num_layers=LAYERS):
        if checkpoint_dir is None:
            return build()
        AttnResWeights.build_ttnn_cache(
            load_attn_res_state_dict(checkpoint_dir, LAYERS), cache_path, CACHE_PREFIX, mesh_device, **cache_args
        )
    logger.info(f"AttnRes weights from {cache_path}")
    return build(weights=AttnResWeights.from_cache(mesh_device, cache_path, CACHE_PREFIX, **cache_args))


def _device_queries(op, queries):
    """`queries` on device: the ones the op already loaded, or freshly placed copies.

    Cached weights are the same checkpoint the host copies were folded from, placed a
    second time rather than copied across — which is what makes the cache path itself part
    of what the gates below cover.
    """
    if op.weights is not None:
        return list(op.weights.pre), list(op.weights.post), op.weights.output
    pre, post, out = queries
    return [op.to_query(q) for q in pre], [op.to_query(q) for q in post], op.to_query(out)


def _walk_sites(q_pre, q_post):
    """The queries in the order the walk issues them.

    Layer 0 has nothing sealed to read against, so `q_pre[0]` is never issued — in the
    checkpoint that entry is a dead constant against a `1.7e-5` projection, which is the
    architecture agreeing with the driver.
    """
    sites = [q_post[0]]
    for pre, post in zip(q_pre[1:], q_post[1:]):
        sites += [pre, post]
    return sites


def _to_device(op, prefix_sum, block_residual, stream_mapper=None):
    mapper = op.stream_mapper if stream_mapper is None else stream_mapper
    to_tt = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=op.mesh_device, mesh_mapper=mapper
    )
    return (
        to_tt(prefix_sum.unsqueeze(0).unsqueeze(0)),
        to_tt(block_residual.permute(1, 0, 2).unsqueeze(0)) if block_residual.shape[1] else None,
    )


def _from_device(op, tensor):
    return ttnn.to_torch(tensor, mesh_composer=op.stream_composer).reshape(-1, HIDDEN_SIZE)


def _read_sites(op, tt_block, tt_prefix, tt_queries):
    """Every read site of one block, as the walk issues them: one `inter_block`, then folds."""
    partials, shifts, masses = op.inter_block(tt_block, tt_queries)
    try:
        for site, tt_query in enumerate(tt_queries):
            merged = op.merge(partials, shifts, masses, tt_prefix, tt_query, site)
            yield _from_device(op, merged)
            ttnn.deallocate(merged)
    finally:
        ttnn.deallocate(partials)
        ttnn.deallocate(shifts)
        ttnn.deallocate(masses)


@on_placements
@pytest.mark.parametrize("num_sealed", [1, 8], ids=["S1", "S8"])
def test_read_matches_reference(mesh_device, num_sealed, device_params, kimi_k3_checkpoint_dir):
    """A whole 12-layer block's reads at 640 rows per chip — production's schedule.

    `S = 1` is the narrowest sealed set the walk ever reads, and the one where the
    statistics cross unfolded; `S = 8` is where every candidate-axis kernel appears. Each
    site carries its own query and is scored against its own oracle, so a `merge` that
    read the wrong site's statistics would fail here rather than agree with itself."""
    op = _make_op(mesh_device, kimi_k3_checkpoint_dir)
    num_tokens = PER_CHIP_TOKENS * op.sp_factor
    assert op.shard_width == HIDDEN_SIZE // op.tp_factor

    prefix_sum, block_residual = _make_case(num_tokens, num_sealed)
    host = _queries(op, kimi_k3_checkpoint_dir)
    queries = _walk_sites(*host[:2])[:READ_SITES]
    tt_queries = _walk_sites(*_device_queries(op, host)[:2])[:READ_SITES]
    tt_prefix, tt_block = _to_device(op, prefix_sum, block_residual)

    worst_pcc, worst_rel_err = 1.0, 0.0
    for site, got in enumerate(_read_sites(op, tt_block, tt_prefix, tt_queries)):
        want = attn_res(prefix_sum, block_residual, queries[site], EPS)
        worst_pcc = min(worst_pcc, _pcc(got, want))
        worst_rel_err = max(worst_rel_err, _rel_err(got, want))

    logger.info(
        f"{tuple(mesh_device.shape)} S={num_sealed} x{READ_SITES} sites T={num_tokens} "
        f"({PER_CHIP_TOKENS}/chip): worst PCC {worst_pcc:.7f}, worst rel err {worst_rel_err:.3e}"
    )
    assert worst_pcc >= PCC_GATE, f"S={num_sealed}: worst PCC {worst_pcc:.7f} < {PCC_GATE}"
    assert worst_rel_err <= REL_ERR_GATE, f"S={num_sealed}: worst rel err {worst_rel_err:.3e} > {REL_ERR_GATE}"


@on_placements
def test_sequence_axis_communicates_nothing(mesh_device, device_params, kimi_k3_checkpoint_dir):
    """The exact gate: same tokens, two placements, bit-identical outputs.

    Run A shards `640 * sp` tokens over the SP rows, so row 0 holds the first 640. Run B
    replicates 640, so *every* row holds those same 640. Under this mapping the SP axis
    carries no traffic, so the two must agree to the last bit — gated at zero, not at a
    tolerance.

    This is what separates "reduced on the TP axis" from "reduced on some axis". A
    collective pointed at the SP axis mixes different tokens in run A and multiplies the
    statistics in run B; either way the two disagree. PCC against torch cannot see it,
    because both runs stay self-consistent within their own placement."""
    op = _make_op(mesh_device, kimi_k3_checkpoint_dir)
    num_tokens = PER_CHIP_TOKENS * op.sp_factor
    prefix_sum, block_residual = _make_case(num_tokens, 8)
    host = _queries(op, kimi_k3_checkpoint_dir)
    tt_query = _walk_sites(*_device_queries(op, host)[:2])[0]

    # `stream_mapper` shards dim 2 on the SP axis; dropping that entry replicates it.
    replicated_dims = [None, None]
    replicated_dims[op.tp_axis] = 3
    replicated = ttnn.ShardTensor2dMesh(mesh_device, dims=replicated_dims, mesh_shape=mesh_device.shape)

    outputs = []
    for tokens, mapper in ((num_tokens, None), (PER_CHIP_TOKENS, replicated)):
        tt_prefix, tt_block = _to_device(op, prefix_sum[:tokens], block_residual[:tokens], stream_mapper=mapper)
        # One site is enough — the sealed half is what the placement change moves, and
        # the batch over sites does not touch the token axis.
        outputs.extend(_read_sites(op, tt_block, tt_prefix, [tt_query]))

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


def _module_stub(h):
    """Stands in for an attention or MLP block.

    `accumulate` takes ownership of what a module returns and the layer driver frees `h`
    afterwards, so a stub cannot hand back `h` itself without a double free. A scalar
    multiply is the cheapest genuinely-new tensor.
    """
    return ttnn.multiply(h, MODULE_SCALE)


def _make_stack(op, checkpoint_dir, seed=0):
    """Everything the walk consumes, on host: the embeddings and all 187 folded queries."""
    generator = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(PER_CHIP_TOKENS * op.sp_factor, HIDDEN_SIZE, generator=generator)
    return (hidden_states, *_queries(op, checkpoint_dir, seed))


@on_placements
def test_walk_matches_reference(mesh_device, device_params, kimi_k3_checkpoint_dir):
    """All 93 layers driven by `attn_res_stack_split`, against the reference walk.

    A driver that batches the wrong sites, or seals on the wrong layer, is wrong in a way
    the reads above cannot see: they issue one `inter_block` and index its sites by hand,
    so they never exercise the seal cadence or the site bookkeeping across a stack. This
    is the only thing in the repo that drives them, and the only caller of
    `attn_res_stack_split` — but it costs more than the rest of this file put together, so
    CI deselects it and it runs by hand.

    It clears the same PCC gate one read does. 186 rounds of bf16 accumulation cost about
    as much accuracy as a single read, because every read renormalizes the stream against
    the sealed set rather than compounding it.
    """
    op = _make_op(mesh_device, kimi_k3_checkpoint_dir)
    hidden_states, q_pre, q_post, q_out = _make_stack(op, kimi_k3_checkpoint_dir)

    embeddings = ttnn.from_torch(
        hidden_states.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=op.mesh_device,
        mesh_mapper=op.stream_mapper,
    )
    tt_pre, tt_post, tt_out = _device_queries(op, (q_pre, q_post, q_out))

    # The walk takes ownership of the stream it is handed and frees it, so the embeddings
    # go in as a clone and stay available for the deallocation sweep below.
    device_walk = attn_res_stack_split(
        op,
        ttnn.clone(embeddings),
        tt_pre,
        tt_post,
        tt_out,
        [_module_stub] * LAYERS,
        [_module_stub] * LAYERS,
        block_size=BLOCK_SIZE,
    )
    got = ttnn.to_torch(device_walk, mesh_composer=op.stream_composer).reshape(-1, HIDDEN_SIZE)
    ttnn.deallocate(device_walk)

    torch_stub = lambda h: h * MODULE_SCALE
    want = attn_res_stack(
        hidden_states,
        q_pre,
        q_post,
        q_out,
        [torch_stub] * LAYERS,
        [torch_stub] * LAYERS,
        block_size=BLOCK_SIZE,
        eps=EPS,
    )

    pcc = _pcc(got, want)
    logger.info(f"device vs reference over {LAYERS} layers, {READS} reads: PCC {pcc:.7f}")
    assert pcc >= PCC_GATE, f"device walk disagrees with the reference: PCC {pcc:.7f} < {PCC_GATE}"

    ttnn.deallocate(embeddings)
    # Placed queries are this test's to free; loaded ones belong to the op and outlive it.
    if op.weights is None:
        for tensor in (tt_out, *tt_pre, *tt_post):
            ttnn.deallocate(tensor)
