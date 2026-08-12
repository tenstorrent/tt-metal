# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The GlobalCircularBuffers every prefetched decode weight streams through.

A device gets one main GCB, shared by nearly every :class:`~.layers.LinearDecode` and
:class:`~.layers.BatchedLinearDecode` on it: the attention block's q projections, its
grouped output projection, its compressor's pair, and the MoE block's shared expert. This
module owns the two things that has to be agreed on model-wide -- the weight layouts and the
order the matmuls consume them -- so no single block can size a buffer against a layout
another block does not use.

``kv_proj`` is the one exception, on a second buffer over disjoint receivers so that it and
the Q chain can be fused into one program as parallel branches; ``KV_GCB_GROUP`` covers why
that needs a buffer of its own rather than a share of this one.

Why one buffer rather than one per shape:

* A GCB is a permanent L1 allocation. At bf4 the default ring is 288 KB per receiver core;
  a per-projection buffer would multiply that by ten, and a per-layer one by the layer
  count.
* Each GCB also takes a fixed ~176 B slice of the DRISC senders' 1 KB state zone, which
  caps a device at about six however small they are. Ten would not fit.
* Most of all it is what lets the prefetcher run *ahead*: a 16-page ring holds far more
  than the two pages the matmul waiting on it needs, so the senders keep working through
  later weights while the workers are still on an earlier one, instead of resynchronising
  at every weight boundary.

What makes one buffer possible is streaming. The slabs here range from 32 to 512 tiles, but
they all divide into pages of 32, so each weight is delivered as some number of uniformly
sized pages and the ring's page size never changes between transfers -- which matters
because a ring whose page size *does* change hangs (see
:func:`~.layers.make_shared_decode_gcb`). Every weight also has to want the same number of
B cores (64 here), since a GCB's receiver set is fixed at construction -- ``o_a_proj``'s
batched ``b_blocks x n_blocks`` grid is sized to 64 for exactly this reason, and ``kv_proj``
being cut to 32 is what puts it on a buffer of its own.

The price is a FIFO ordering contract spanning every weight on a buffer, and it is not
checked anywhere: a matmul that runs out of turn pops another weight's page and produces
wrong results rather than an error. ``DECODE_GCB_GROUP`` is that order.
"""

from typing import Optional

import ttnn

from .layers import decode_gcb_page_bytes, make_shared_decode_gcb

# Every prefetched decode weight, by name, with the layout ``decode_weight_layout`` reads.
# Hardcoded rather than derived from the config because the buffer is built before any layer
# exists; each block checks its own entries against the config it was handed, so a config
# these do not describe fails loudly instead of reaching the device mis-sharded. They are the
# same for every layer, which is what lets one buffer serve a whole model.
#
# The compressor entries are keyed by layer type rather than by projection name: its kv/gate
# pair share a shape, and which shape depends on the kind (HCA projects a token to ``Dh``,
# CSA to ``2*Dh``).
DECODE_LAYOUTS = {
    "q_a_proj": {"K": 4096, "N": 1024, "partial_width_sharded": True, "k_blocks": 2, "n_blocks": 32},
    "q_b_proj": {"K": 1024, "N": 32768, "n_blocks": 64},
    # 2x16 rather than the 4x16 the other partial weights use, because kv_proj is the one
    # weight held off the main buffer so it can run *beside* the Q chain: 32 receivers is what
    # fits in the column band left over once the main buffer has its 64 (see KV_GCB_ORIGIN).
    # Halving k_blocks and not n_blocks keeps the reduction on 16 output cores, so everything
    # downstream of kv_proj sees the layout it always did.
    "kv_proj": {"K": 4096, "N": 512, "partial_width_sharded": True, "k_blocks": 2, "n_blocks": 16},
    "o_b_proj": {"K": 8192, "N": 4096},
    "compressed_sparse_attention": {
        "K": 4096,
        "N": 1024,
        "partial_width_sharded": True,
        "k_blocks": 2,
        "n_blocks": 32,
    },
    "heavily_compressed_attention": {
        "K": 4096,
        "N": 512,
        "partial_width_sharded": True,
        "k_blocks": 4,
        "n_blocks": 16,
    },
    # The grouped output projection (o_a of DeepseekV4GroupedLinear): batched over o_groups,
    # folded along both batch and N into a b_blocks x n_blocks grid (see
    # BatchedLinearDecode). 8x8 is what falls out of the model's o_groups=8, o_lora_rank=1024
    # on this device's grid -- the same 64 receivers as everything else, which is what lets
    # it join this buffer at all.
    "o_a_proj": {"K": 4096, "N": 1024, "batch": 8, "b_blocks": 8, "n_blocks": 8},
    # The MoE shared expert. gate and up are laid out so their [T, I] outputs land on the 32
    # cores holding 64 columns each -- exactly the K-sharding down_proj wants of its
    # activation -- so the SwiGLU intermediate feeds down_proj where it already sits.
    "shared_gate_proj": {"K": 4096, "N": 2048, "partial_width_sharded": True, "k_blocks": 2, "n_blocks": 32},
    "shared_up_proj": {"K": 4096, "N": 2048, "partial_width_sharded": True, "k_blocks": 2, "n_blocks": 32},
    "shared_down_proj": {"K": 2048, "N": 4096},
}

# The order one layer's matmuls consume the buffer, which is the order the requests must be
# queued in. Attention runs first: q_a/q_b from ``_qkv``, then the compressor (it runs
# after ``_qkv`` and before ``_attend``), then ``_attend``'s grouped output projection
# (o_a_proj before o_b_proj -- see ``DeepSeekV4Attention._grouped_output``). The MoE block
# follows. ``kv_proj`` is absent because it has a buffer of its own; see KV_GCB_GROUP.
DECODE_GCB_GROUP = (
    "q_a_proj",
    "q_b_proj",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
    "o_a_proj",
    "o_b_proj",
    "shared_gate_proj",
    "shared_up_proj",
    "shared_down_proj",
)

# kv_proj gets a second buffer on its own receivers so that it and the Q chain (q_a_proj ->
# q_a_norm -> q_b_proj) can be the two branches of a fused ``Parallel`` in
# ``DeepSeekV4Attention._qkv``. Two things force the split rather than a shared buffer:
# branches must occupy disjoint cores, and a GCB's receiver set is fixed at construction, so
# a 64-receiver and a 32-receiver weight could not share one anyway.
#
# The cost is small: the receivers are disjoint, so no core pays for two rings, and two
# buffers is well inside the ~6 a device's DRISC state zone holds. Each buffer is its own
# FIFO, so kv_proj's ordering is now independent of everything on the main buffer -- there is
# one weight on it and one consumer of it.
KV_GCB_GROUP = ("kv_proj",)

# The two buffers' receiver placements, as (origin, width) column bands.
#
# A harvested Blackhole exposes an 11x10 compute grid, which will not hold two full
# rectangles of 64 and 32 cores: 64 can only be 8x8 there, and the widest rectangle left
# beside it is 3x10 = 30. Splitting by *column band* instead fits both, because the receiver
# set need only be the first N cores of a width-wide row-major walk and may leave its last
# row partial (see ``layers._receiver_ring_cols``).
#
# So the main buffer takes the leftmost 7 columns -- 64 cores is 9 full rows plus 1 core of
# the tenth -- and kv_proj takes the 4 columns beside them, where 32 cores is exactly 8 full
# rows. Neither band can grow into the other whatever its height, and 7 + 4 = 11 uses the
# grid exactly. Both widths are passed explicitly because the default picks the widest that
# fits, which for the main buffer would be all 11 columns.
#
# The main band's ragged tail has a known cost: 64 cores over 7 columns spans a 7x10 bounding
# box, and ops that require a rectangular shard grid refuse it -- ttnn.reshape falls back to
# INTERLEAVED for the one in DeepSeekV4Attention._qkv, logging as it does so. That is accepted
# to keep kv_proj at 32 cores; the two cannot both be had on an 11-column grid, since a
# rectangular 64 is 8x8 and leaves 3 columns where 32 cores need 4.
DECODE_GCB_ORIGIN = (0, 0)
DECODE_GCB_RING_COLS = 7
KV_GCB_ORIGIN = (7, 0)
KV_GCB_RING_COLS = 4

# Ring depth for kv_proj's buffer, against ``num_prefetch_pages`` for the main one. A GCB is a
# permanent L1 reservation on every receiver, and between them the two bands now cover most of
# the grid -- so what is left has to be enough for the ops that follow, SDPA-decode above all,
# which spreads its own circular buffers over the whole grid.
#
# Depth buys run-ahead, and there is none to buy here: this buffer holds one weight with one
# consumer, so beyond the two pages streaming needs, a deeper ring only reserves L1 that the
# senders can never be far enough ahead to use. The main buffer is the opposite case -- ten
# weights whose senders do work through later ones while the workers are still on the first --
# which is why it keeps its depth.
KV_GCB_PAGES = 2


def decode_prefetch_page_bytes(weight_dtype: ttnn.DataType, name: Optional[str] = None) -> int:
    """The GCB page size a prefetched decode weight is streamed at.

    A pure function of the (fixed) layouts and the weight dtype, so
    :func:`make_decode_prefetch_buffers` and the layers streaming through the buffers it builds
    can each derive it independently and cannot disagree -- which matters because a layer
    streaming at a page size the ring was not built for is a hang.

    ``name`` selects which buffer's page size is wanted, since the two groups size their pages
    independently; it defaults to the main buffer's.
    """
    group = KV_GCB_GROUP if name in KV_GCB_GROUP else DECODE_GCB_GROUP
    return decode_gcb_page_bytes([DECODE_LAYOUTS[n] for n in group], weight_dtype)


def make_decode_prefetch_buffers(
    device: ttnn.MeshDevice, weight_dtype: ttnn.DataType, num_prefetch_pages: int = 16
) -> dict:
    """The GCBs every prefetched decode weight on ``device`` streams through.

    Returns a mapping keyed by the names in ``DECODE_LAYOUTS``, to hand to
    :class:`~.attention.DeepSeekV4Attention` and :class:`~.moe.DeepSeekV4SparseMoeBlock` as
    ``prefetch_buffers``. Every key in ``DECODE_GCB_GROUP`` maps to the main buffer and
    ``kv_proj`` to its own (see ``KV_GCB_GROUP``); the mapping is per-weight so the blocks
    need not know which weight is on which buffer.

    ``num_prefetch_pages`` is the ring depth, and the knob for how far ahead the prefetcher
    may run: at the default it is 16 pages, several weights' worth.

    Build this **once per device and pass it to every layer on that device** -- see the module
    docstring for why, and for the ordering contract that comes with sharing.
    """
    global_cb = make_shared_decode_gcb(
        device,
        [DECODE_LAYOUTS[name] for name in DECODE_GCB_GROUP],
        weight_dtype,
        num_pages=num_prefetch_pages,
        origin=ttnn.CoreCoord(*DECODE_GCB_ORIGIN),
        ring_cols=DECODE_GCB_RING_COLS,
    )
    kv_cb = make_shared_decode_gcb(
        device,
        [DECODE_LAYOUTS[name] for name in KV_GCB_GROUP],
        weight_dtype,
        num_pages=KV_GCB_PAGES,
        origin=ttnn.CoreCoord(*KV_GCB_ORIGIN),
        ring_cols=KV_GCB_RING_COLS,
    )
    buffers = {name: global_cb for name in DECODE_GCB_GROUP}
    buffers.update({name: kv_cb for name in KV_GCB_GROUP})
    return buffers


def check_decode_layout(name: str, K: int, N: int, batch: Optional[int] = None) -> dict:
    """``DECODE_LAYOUTS[name]``, having checked it against the ``K``/``N`` (and, for a batched
    weight, ``batch``) the config wants.

    The layouts are constants (the shared GCB is sized from them before any weight is built),
    so a config they do not describe has to be caught here: left alone it would reach the
    device as a silently mis-sharded weight rather than an error.
    """
    layout = DECODE_LAYOUTS[name]
    if (layout["K"], layout["N"]) != (K, N):
        raise ValueError(
            f"the {name} layout is fixed at K={layout['K']}, N={layout['N']} but this config wants K={K}, N={N}"
        )
    if layout.get("batch") != batch:
        raise ValueError(
            f"the {name} layout is fixed at batch={layout.get('batch')} but this config wants batch={batch}"
        )
    return layout
