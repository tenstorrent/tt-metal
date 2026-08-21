# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The shape a K3 transformer plugs `TtAttnResWalk` into, gated against the torch oracle.

`_transformer_forward` mirrors `tt/tt_prefill_transformer.py`'s layer loop and `_StandInBlock`
mirrors `tt/tt_prefill_block.py`'s two residual sites, so what a caller changes to adopt
AttnRes is readable as code rather than described. The two `ttnn.add(x, module_out)` become
`walk.write`, and what feeds each module becomes a `walk.read` rather than the running sum.
Nothing else about a block moves, and nothing here knows whether a module is MLA, KDA or MoE.

Queries are random. Which weights the op holds is `test_attn_res.py`'s subject; what this
file gates is the schedule seen from the caller's side — seal cadence, site order, and who
frees what — and that is the same schedule whatever the queries are.

The oracle is `reference.kimi_k3.attn_res.attn_res.attn_res_stack`, which walks the same seal
schedule on host, so what this gates is the device arithmetic and the caller's bookkeeping
rather than the schedule agreeing with itself.
"""

import pytest
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import EPS, attn_res_stack
from models.demos.deepseek_v3_d_p.tests.attn_res.assertions import assert_accurate
from models.demos.deepseek_v3_d_p.tests.attn_res.model.harness import (
    FABRIC,
    HIDDEN_SIZE,
    PER_CHIP_TOKENS,
    blackhole_only,
    compose,
    generator,
    place,
    random_hidden,
    random_queries,
)
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import BLOCK_SIZE, TtAttnResWalk

PCC_GATE = 0.9999
LAYERS = 93

# Scales each module's contribution so 93 rounds of accumulation stay in bf16's range.
MODULE_SCALE = 0.02

PLACEMENTS = [pytest.param((2, 4), FABRIC, id="mesh-2x4")]

pytestmark = blackhole_only


class _StandInBlock(object):
    """One layer, with a block's two residual sites and nothing else.

    `attention` and `mlp` stand in for everything between two reads, including each
    module's own input norm — that norm is distinct from the `*_res_norm` folded into the
    query, and both exist in the checkpoint. A module borrows the tensor it is handed and
    must not free it; ownership of what it returns passes to the walk.
    """

    def __init__(self, attention, mlp):
        self.attention, self.mlp = attention, mlp

    def __call__(self, hidden, walk):
        walk.write(self.attention(hidden))

        # The block's pre-FFN norm reaches for the running sum; under AttnRes the FFN's
        # input is its own read of the residual instead.
        post_attention = walk.read()
        walk.write(self.mlp(post_attention))
        ttnn.deallocate(post_attention)


def _transformer_forward(op, embeddings, queries, blocks, block_size=BLOCK_SIZE, hook=None):
    """`TtPrefillTransformer.forward`'s layer loop, with the residual on AttnRes.

    The walk is per forward pass and `op` is not: a model holds one `TtAttnRes` for its
    lifetime, holds the placed queries alongside it, and builds one walk per call. The walk
    takes ownership of `embeddings` and frees it; what it returns is the caller's, and is
    what `model.norm` sees.
    """
    q_pre, q_post, q_out = queries
    walk = TtAttnResWalk(op, embeddings, q_pre, q_post, q_out, len(blocks), block_size=block_size)

    for layer_idx, block in enumerate(blocks):
        hidden, borrowed = walk.open_layer(layer_idx)
        block(hidden, walk)
        # Layer 0 has nothing sealed to read, so what it was handed is the live stream
        # itself and the walk still owns it.
        if not borrowed:
            ttnn.deallocate(hidden)
        if hook is not None:
            hook(layer_idx, walk)

    return walk.finish()


def _host_case(op, seed=0):
    """The embeddings entering the stack, and the 187 queries the walk reads them against."""
    rng = generator(seed)
    hidden_states = random_hidden(rng, PER_CHIP_TOKENS * op.sp_factor)
    return hidden_states, (random_queries(rng, LAYERS), random_queries(rng, LAYERS), random_queries(rng, 1)[0])


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=["mesh_device", "device_params"])
def test_transformer_loop_matches_reference(mesh_device, device_params):
    """A caller-owned layer loop over all 93 layers, and what it hands a pipeline boundary.

    This is the integration a block will do, so it fails on what a block can get wrong that
    the op cannot: a read taken before the seal, a module output summed instead of written,
    a boundary off by one. It clears the same PCC gate a single read does, because every
    read renormalizes the stream against the sealed set rather than compounding it.

    The second assertion prices the split. Ranks divide on block boundaries, so a rank owns
    whole seal groups and the walk never resumes mid-batch — but `seal` concatenates and
    nothing evicts, so what crosses is a sealed set one snapshot deeper at every boundary,
    not just the activation.
    """
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS)
    hidden_states, (q_pre, q_post, q_out) = _host_case(op)

    embeddings = place(op, hidden_states.unsqueeze(0).unsqueeze(0))
    tt_queries = ([op.to_query(q) for q in q_pre], [op.to_query(q) for q in q_post], op.to_query(q_out))

    module = lambda h: ttnn.multiply(h, MODULE_SCALE)
    blocks = [_StandInBlock(module, module) for _ in range(LAYERS)]

    boundaries = []
    record = lambda layer_idx, walk: (
        boundaries.append((layer_idx + 1, walk.stream.num_sealed)) if (layer_idx + 1) % BLOCK_SIZE == 0 else None
    )
    device_out = _transformer_forward(op, embeddings, tt_queries, blocks, hook=record)
    got = compose(op, device_out)
    ttnn.deallocate(device_out)

    torch_module = lambda h: h * MODULE_SCALE
    want = attn_res_stack(
        hidden_states,
        q_pre,
        q_post,
        q_out,
        [torch_module] * LAYERS,
        [torch_module] * LAYERS,
        block_size=BLOCK_SIZE,
        eps=EPS,
    )

    snapshot_mib = PER_CHIP_TOKENS * op.shard_width * 2 / 1024**2
    for layers_done, num_sealed in boundaries:
        logger.info(
            f"split after layer {layers_done}: {num_sealed} sealed snapshots, "
            f"{num_sealed * snapshot_mib:.1f} MiB/chip crossing beside the activation"
        )

    pcc = assert_accurate(want, got, name="caller-driven loop", pcc_threshold=PCC_GATE)
    logger.info(f"caller-driven loop over {LAYERS} layers, {2 * LAYERS} reads: PCC {pcc:.7f}")
    assert [depth for _, depth in boundaries] == [1 + index for index in range(len(boundaries))], (
        "a seal fires once per block and nothing evicts one, so the set a rank hands on grows "
        f"by one at every boundary: got {boundaries}"
    )

    for tensor in (tt_queries[2], *tt_queries[0], *tt_queries[1]):
        ttnn.deallocate(tensor)
