# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: the Kimi K3 attention-residuals read (`models/demos/deepseek_v3_d_p/tt/attn_res`)
against the torch oracle, at the per-chip shape prefill actually runs.

One placement, one shape: `(2, 4)` on a LoudBox, 1280 tokens split 2 ways over the
sequence axis and `d` split 4 ways over the tensor axis, so every chip holds the 640 rows
`harness.py` fixes and `d/4 = 1792` columns. TP factor 4 is Galaxy's, which is what makes
this box's reduction the same reduction Galaxy runs.

No single-device arm. The read's exchange is what its one dispatch is built around, so
`TtAttnRes` rejects `tp_factor == 1` outright rather than degrading to something the
model never executes. No Galaxy `(8, 4)` arm either: it holds the same TP factor over a
wider sequence axis, which the op is indifferent to, so it costs 32 chips to re-run the
reduction this arm already covers.

The parametrization that remains is over branches, not shapes: `S` selects whether the
statistics cross the tensor axis folded or unfolded. The seal cadence across a whole stack
is `test_forward_loop.py`'s subject, not this file's.

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
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import EPS, attn_res
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.weights import fold_queries
from models.demos.deepseek_v3_d_p.tests.attn_res.assertions import assert_accurate, assert_bit_identical, assert_equal
from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import (
    attn_res_tensor_cache_path,
    load_attn_res_state_dict,
)
from models.demos.deepseek_v3_d_p.tests.attn_res.model.harness import (
    FABRIC,
    HIDDEN_SIZE,
    PER_CHIP_TOKENS,
    blackhole_only,
    compose,
    generator,
    place_case,
    random_case,
    random_queries,
    read_block,
)
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import AttnResWeights, walk_sites

PCC_GATE = 0.9999
REL_ERR_GATE = 2e-2

READ_SITES = 24

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

PLACEMENTS = [pytest.param((2, 4), FABRIC, id="mesh-2x4")]

on_placements = pytest.mark.parametrize(
    "mesh_device, device_params", PLACEMENTS, indirect=["mesh_device", "device_params"]
)

pytestmark = blackhole_only


def _rel_err(got, want):
    got, want = got.double(), want.double()
    return ((got - want).abs().max() / want.abs().max()).item()


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
        row = lambda q: compose(op, q)[0].float()
        return (
            [row(q) for q in op.weights.pre],
            [row(q) for q in op.weights.post],
            row(op.weights.output),
        )

    rng = generator(seed)
    return random_queries(rng, LAYERS), random_queries(rng, LAYERS), random_queries(rng, 1)[0]


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

    running_sum, block_residual = random_case(generator(), num_tokens, num_sealed)
    host = _queries(op, kimi_k3_checkpoint_dir)
    queries = walk_sites(*host[:2])[:READ_SITES]
    tt_queries = walk_sites(*_device_queries(op, host)[:2])[:READ_SITES]
    tt_prefix, tt_block = place_case(op, running_sum, block_residual)

    worst_pcc, worst_rel_err = 1.0, 0.0
    for site, got in enumerate(read_block(op, tt_block, tt_prefix, tt_queries)):
        want = attn_res(running_sum, block_residual, queries[site], EPS)
        pcc = assert_accurate(want, got, name=f"S={num_sealed} site {site}", pcc_threshold=PCC_GATE)
        worst_pcc = min(worst_pcc, pcc)
        worst_rel_err = max(worst_rel_err, _rel_err(got, want))

    logger.info(
        f"{tuple(mesh_device.shape)} S={num_sealed} x{READ_SITES} sites T={num_tokens} "
        f"({PER_CHIP_TOKENS}/chip): worst PCC {worst_pcc:.7f}, worst rel err {worst_rel_err:.3e}"
    )
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
    running_sum, block_residual = random_case(generator(), num_tokens, 8)
    host = _queries(op, kimi_k3_checkpoint_dir)
    tt_query = walk_sites(*_device_queries(op, host)[:2])[0]

    # `stream_mapper` shards dim 2 on the SP axis; dropping that entry replicates it.
    replicated_dims = [None, None]
    replicated_dims[op.tp_axis] = 3
    replicated = ttnn.ShardTensor2dMesh(mesh_device, dims=replicated_dims, mesh_shape=mesh_device.shape)

    outputs = []
    for tokens, mapper in ((num_tokens, None), (PER_CHIP_TOKENS, replicated)):
        tt_prefix, tt_block = place_case(op, running_sum[:tokens], block_residual[:tokens], mesh_mapper=mapper)
        # One site is enough — the sealed half is what the placement change moves, and
        # the batch over sites does not touch the token axis.
        outputs.extend(read_block(op, tt_block, tt_prefix, [tt_query]))

    sharded, duplicated = outputs
    # Both SP rows of run B ran identical inputs, so they must agree too.
    rows = duplicated[: 2 * PER_CHIP_TOKENS]
    row_delta = (rows[:PER_CHIP_TOKENS].float() - rows[PER_CHIP_TOKENS:].float()).abs().max().item()
    delta = (sharded[:PER_CHIP_TOKENS].float() - duplicated[:PER_CHIP_TOKENS].float()).abs().max().item()
    logger.info(f"SP rows agree to {row_delta:.3e}; sharded-vs-replicated max|delta| {delta:.3e}")

    # Two chips running one program over one input: exact down to the sign bit.
    assert_bit_identical(rows[:PER_CHIP_TOKENS], rows[PER_CHIP_TOKENS:], name="the two SP rows of the replicated run")
    # The cross-placement gate is values, not bytes. Run A carries 1280 tokens and run B
    # 640, so the two are different programs and a sign bit on a zero is theirs to differ
    # on; what may not differ is a number.
    assert_equal(
        sharded[:PER_CHIP_TOKENS],
        duplicated[:PER_CHIP_TOKENS],
        name=f"first {PER_CHIP_TOKENS} outputs, {num_tokens} tokens sharded over the SP axis vs replicated",
    )
