# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The worked example: what a K3 block and its transformer change to run on AttnRes.

`_ExampleBlock.forward` and `_ExampleTransformer.forward` are `tt/tt_prefill_block.py` and
`tt/tt_prefill_transformer.py` with the residual moved and nothing else. The block's two
`ttnn.add(x, module_out)` become `walk.write`, what feeds the FFN norm becomes `walk.read`
rather than the running sum, and the block stops returning a hidden state because the walk
holds it. The layer loop keeps its shape: one `open_layer` at the top, one `deallocate` at
the bottom.

Nothing here knows whether a module is MLA, KDA, MoE or a dense FFN, and that is the point
— the walk sees a tensor it takes ownership of. `_module` stands in for all of them, so this
file stays a device gate on the residual rather than a second copy of the model.

Five layers at `block_size=2`, which is the smallest stack that reaches every state the
93-layer schedule does: layer 0 borrowing the live stream with nothing sealed, the first
seal moving ownership rather than concatenating, two later seals growing the sealed set,
the batch being rebuilt at each boundary, a write settled by the following read, a write
landing on an empty stream straight after a seal, and a trailing query group shorter than
a full block. `test_forward_loop.py` runs the production cadence; this one runs the corners.

Queries are random. Production reads them off the checkpoint instead, which changes two
lines and no control flow:

    op = TtAttnRes(mesh_device, hidden_size=..., eps=..., tp_axis=TP_AXIS,
                   weights=AttnResWeights.from_cache(mesh_device, cache_path, "attn_res"))
    walk = TtAttnResWalk(op, embeddings, op.weights.pre, op.weights.post, op.weights.output, LAYERS)

Three things a real block still has to resolve, none of them visible here because this
example has none of them: a `kv_only` layer writes without reading and desynchronizes the
site iterator; the `post_mla_residual` and `on_layer_hidden` taps reach for a running sum
that no longer exists and need their own reads; and a non-last pipeline rank needs the
sealed set handed across, which `finish` frees.
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
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk

PCC_GATE = 0.9999
TP_AXIS = 1

LAYERS = 5
BLOCK_SIZE = 2

# Scales each module's contribution so the accumulation stays in bf16's range.
MODULE_SCALE = 0.02

PLACEMENTS = [pytest.param((2, 4), FABRIC, id="mesh-2x4")]

pytestmark = blackhole_only


class _ExampleBlock(object):
    """`TtPrefillBlock` with its two residual sites on the walk.

    A module borrows the tensor it is handed and must not free it; ownership of what it
    returns passes to `write`, so neither `attention` nor `mlp` deallocates its own output.
    """

    def __init__(self, attention, mlp):
        self.attention, self.mlp = attention, mlp

    def forward(self, hidden, walk):
        # `hidden` is the walk's pre-attention read, standing in for `attn_norm(x)`'s input.
        walk.write(self.attention(hidden))

        # The block's pre-FFN norm reaches for the running sum; under AttnRes the FFN's
        # input is its own read of the residual instead.
        post_attention = walk.read()
        walk.write(self.mlp(post_attention))
        ttnn.deallocate(post_attention)


class _ExampleTransformer(object):
    """`TtPrefillTransformer` with one `TtAttnRes` beside its layers.

    One op serves the whole stack — it holds every layer's query and the exchange scratch
    they share — and one walk serves one forward pass. Building the op per layer would
    place the queries `num_layers` times and give each layer a private scratch.
    """

    def __init__(self, mesh_device, queries, num_layers=LAYERS, block_size=BLOCK_SIZE):
        self.op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE, eps=EPS, tp_axis=TP_AXIS)
        self.num_layers, self.block_size = num_layers, block_size

        q_pre, q_post, q_out = queries
        self.queries = (
            [self.op.to_query(q) for q in q_pre],
            [self.op.to_query(q) for q in q_post],
            self.op.to_query(q_out),
        )

        module = lambda hidden: ttnn.multiply(hidden, MODULE_SCALE)
        self.layers = [_ExampleBlock(module, module) for _ in range(num_layers)]

    def forward(self, embeddings, hook=None):
        """The layer loop, unchanged in shape.

        The walk takes ownership of `embeddings` and frees it; what `finish` returns is the
        caller's, and is what `self.norm` would see.
        """
        q_pre, q_post, q_out = self.queries
        walk = TtAttnResWalk(self.op, embeddings, q_pre, q_post, q_out, self.num_layers, block_size=self.block_size)

        for layer_idx, layer in enumerate(self.layers):
            hidden, borrowed = walk.open_layer(layer_idx)
            layer.forward(hidden, walk)
            # Layer 0 has nothing sealed to read, so what it was handed is the live stream
            # itself and the walk still owns it.
            if not borrowed:
                ttnn.deallocate(hidden)
            if hook is not None:
                hook(layer_idx, borrowed, walk)

        return walk.finish()

    def deallocate(self):
        q_pre, q_post, q_out = self.queries
        for tensor in (q_out, *q_pre, *q_post):
            ttnn.deallocate(tensor)


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=["mesh_device", "device_params"])
def test_example_matches_reference(mesh_device, device_params):
    """The example runs, agrees with the oracle, and reaches every state on the way.

    The PCC gate is what makes this an example rather than pseudocode: a `write` that
    dropped its addend, or a `read` taken against the wrong sealed set, still runs.

    The state assertions are what make five layers a substitute for 93. They pin the two
    facts a reader would otherwise have to trust: only layer 0 borrows, and the sealed set
    grows by exactly one per block boundary and never evicts.
    """
    rng = generator()
    queries = (random_queries(rng, LAYERS), random_queries(rng, LAYERS), random_queries(rng, 1)[0])
    model = _ExampleTransformer(mesh_device, queries)
    hidden_states = random_hidden(rng, PER_CHIP_TOKENS * model.op.sp_factor)

    visited = []
    record = lambda layer_idx, borrowed, walk: visited.append((layer_idx, borrowed, walk.stream.num_sealed))
    device_out = model.forward(place(model.op, hidden_states.unsqueeze(0).unsqueeze(0)), hook=record)
    got = compose(model.op, device_out)
    ttnn.deallocate(device_out)

    torch_module = lambda hidden: hidden * MODULE_SCALE
    want = attn_res_stack(
        hidden_states,
        *queries,
        [torch_module] * LAYERS,
        [torch_module] * LAYERS,
        block_size=BLOCK_SIZE,
        eps=EPS,
    )

    pcc = assert_accurate(want, got, name="example integration", pcc_threshold=PCC_GATE)
    logger.info(
        f"{LAYERS} layers at block_size={BLOCK_SIZE}, {2 * LAYERS} reads, "
        f"T={PER_CHIP_TOKENS * model.op.sp_factor} ({PER_CHIP_TOKENS}/chip): PCC {pcc:.7f}"
    )
    logger.info(f"(layer, borrowed, sealed) per layer: {visited}")

    borrowed_layers = [layer_idx for layer_idx, borrowed, _ in visited if borrowed]
    assert borrowed_layers == [0], (
        "only layer 0 reads nothing, because only layer 0 runs before the first seal: "
        f"layers {borrowed_layers} were handed the live stream"
    )
    depths = [num_sealed for _, _, num_sealed in visited]
    assert depths == [1 + layer_idx // BLOCK_SIZE for layer_idx in range(LAYERS)], (
        "a seal fires once per block and nothing evicts one, so the sealed set is one "
        f"deeper at each boundary and flat between them: got {depths}"
    )

    model.deallocate()
