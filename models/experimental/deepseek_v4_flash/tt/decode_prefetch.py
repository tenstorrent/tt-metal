# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The one GlobalCircularBuffer every prefetched decode weight streams through.

A device gets a single GCB, shared by every :class:`~.layers.LinearDecode` and
:class:`~.layers.BatchedLinearDecode` on it: the attention block's four projections, its
grouped output projection, its compressor's pair, and the MoE block's shared expert. The
router gate cannot join that ring (16-tile slabs vs a 32-tile page) and streams through
:data:`HC_FN_GCB` instead. This module owns the two things that has to be agreed on
model-wide -- the weight layouts and the order the matmuls consume them -- so no single
block can size a buffer against a layout another block does not use.

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
batched ``b_blocks x n_blocks`` grid is sized to 64 for exactly this reason.

The price is a single FIFO ordering contract spanning all ten weights, and it is not
checked anywhere: a matmul that runs out of turn pops another weight's page and produces
wrong results rather than an error. ``DECODE_GCB_GROUP`` is that order.

The hyper-connections' fused ``fn`` weight gets a *second* buffer on the same mapping
(:data:`HC_FN_GCB`), because its 8-tile slab has no page in common with the 32-tile page
above. TP4 balanced ``q_a`` / ``kv`` and the MoE router gate join that ring: their slabs
are 16 or 8 tiles on 64 receivers, which divide the 8-tile page and cannot join the
shared group. One buffer for all of them, not one each: a GCB also costs ~176 B of the
senders' 1 KB state zone, so two per layer overflows that zone at the third layer.

Tensor-parallel decode *could* attach two further GCBs on this mapping (sequential
``o_a`` at 32 receivers, TP gate/up at ``N/TP`` receivers) via :func:`ensure_named_gcb`.
They are not used on the full decode path, and it is worth being precise about why, because
the obvious reason is not the operative one. Placement looks like the blocker: a GCB is a
permanent L1 allocation, and on the default anchor a narrower ring lands inside the shared
ring's rectangle and on ``(0,0)``, the pipeline socket whose ``fused_hyperconnection``
static CBs it collides with once the shared 24-page GCB is allocated. That part is
solvable -- ``matmul_decode`` takes its receiver grid from the buffer, so the rectangle can
simply be anchored past the shared ring's last column, and sequential ``o_a`` on such a
ring runs and is numerically correct.

It is just slower. Streaming those weights measured a regression against the per-step
DRAM->L1 copy it replaced, so they stay on the copy. The cause was not established; a
32-receiver ring is half the width the senders fill on the shared ring, and its depth is
capped at two pages because its page is a whole slab, but neither was measured to be the
reason. Re-measure rather than re-derive before trying this again.
"""

from typing import Optional

import ttnn

from .layers import decode_gcb_page_bytes, make_shared_decode_gcb
from .system_config import active_system_config

# The hyper-connections' ``fn`` ring, keyed separately from ``DECODE_GCB_GROUP``: its slab
# is 8 tiles against that group's 32-tile page, so it is a second buffer rather than a
# second producer on the shared one. Both hyper-connections of every layer on the device
# stream through it (see :func:`make_decode_prefetch_buffers`).
HC_FN_GCB = "hc_fn"
# Depth of the 8-tile ring. A TP4 layer queues attn_hc (1) + q_a (2) + kv (1) +
# ffn_hc (1) + router_gate (2) = 7 pages, and the next layer on the chip may be
# staged before the current one has drained; 8 leaves a page of slack without
# moving the 4.6 KB page into the same class as the shared 18 KB ring.
HC_FN_GCB_PAGES = 8

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
    "kv_proj": {"K": 4096, "N": 512, "partial_width_sharded": True, "k_blocks": 4, "n_blocks": 16},
    # q_a and kv projected by one matmul over their concatenated weight (see
    # ``DeepSeekV4Attention._qkv``). Deliberately absent from ``DECODE_GCB_GROUP``: a
    # 1536-wide weight cuts into 3-tile rows per B core, and the group's page is 32 tiles,
    # so it has no page size in common with the shared buffer. The prefetched path
    # therefore keeps q_a_proj / kv_proj separate and only the L1 path fuses them.
    "qa_kv_proj": {"K": 4096, "N": 1536, "partial_width_sharded": True, "k_blocks": 4, "n_blocks": 16},
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
    # Router gate: [H, E] = [4096, 256]. A 64-core partial cut is 8x8 of [512, 32]
    # (16-tile) slabs, which have no 32-tile page in common with DECODE_GCB_GROUP, so
    # it streams through HC_FN_GCB. Deliberately absent from DECODE_GCB_GROUP.
    "router_gate": {"K": 4096, "N": 256, "partial_width_sharded": True, "k_blocks": 8, "n_blocks": 8},
    # Both hyper-connections' fused fn projection: K = hc_mult * hidden_size against the
    # 24 pre/post/comb outputs padded to one tile. Large K and a single tile of N, so it
    # cuts K 64 ways onto the same 64 receivers and reduces onto one output core. Its
    # [256, 32] slab is 8 tiles, which is why it needs HC_FN_GCB rather than the group
    # above. Deliberately absent from DECODE_GCB_GROUP.
    HC_FN_GCB: {"K": 16384, "N": 32, "partial_width_sharded": True, "k_blocks": 64, "n_blocks": 1},
}

# The order one layer's matmuls consume the buffer, which is the order the requests must be
# queued in. Attention runs first: q_a/q_b/kv from ``_qkv``, then the compressor (it runs
# after ``_qkv`` and before ``_attend``), then ``_attend``'s grouped output projection
# (o_a_proj before o_b_proj -- see ``DeepSeekV4Attention._grouped_output``). The MoE block
# follows.
DECODE_GCB_GROUP = (
    "q_a_proj",
    "q_b_proj",
    "kv_proj",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
    "o_a_proj",
    "o_b_proj",
    "shared_gate_proj",
    "shared_up_proj",
    "shared_down_proj",
)

# Extra GCBs attached to the per-device prefetch mapping under TP. Not in
# ``DECODE_GCB_GROUP``: different receiver counts, independent FIFOs.
SEQUENTIAL_OA_GCB = "o_a_sequential"
TP_GATE_UP_GCB = "shared_gate_up_tp"
# Those private rings cannot use the shared 16/24-page depth: each has a single
# spec, so the page is the whole slab (72 KB for sequential o_a, 144 KB for TP
# gate/up). 24 such pages is 1.7–3.5 MB per receiver, which does not fit in a
# Blackhole L1 bank after the shared GCB. Two pages is the streaming floor and
# covers the two weights on each FIFO.
TP_PRIVATE_GCB_PAGES = 2


def sequential_oa_layout(K: int, N: int) -> dict:
    """One sequential o_a slot: ``[K, o_lora_rank]`` on 32 cores.

    The batched ``o_a_proj`` layout is the same ``K``/``N`` folded over ``o_groups``;
    a sequential slot is that per-group matrix, which no longer spans 64 receivers.
    """
    oa = DECODE_LAYOUTS["o_a_proj"]
    n_blocks = oa["N"] // ttnn.TILE_SIZE
    if (oa["K"], oa["N"]) != (K, N):
        raise ValueError(f"sequential o_a is fixed at K={oa['K']}, N={oa['N']} but this config wants K={K}, N={N}")
    return {"K": K, "N": N, "n_blocks": n_blocks}


def balanced_qkv_layout(name: str, tp_size: int) -> dict:
    """Per-rank balanced ``q_a`` / ``kv``: column-parallel ``N / tp_size`` on 64 B cores.

    ``n_blocks`` is one tile of local N per core; ``k_blocks`` is whatever fills the
    64-receiver :data:`HC_FN_GCB` ring (the shared decode GCB's page is 32 tiles and
    these slabs are 8 or 16). At TP4 that is ``q_a`` 8x8 ``[512, 32]`` and ``kv``
    16x4 ``[256, 32]``.
    """
    if name not in ("q_a_proj", "kv_proj"):
        raise ValueError(f"balanced_qkv_layout is q_a/kv only, not {name}")
    full = DECODE_LAYOUTS[name]
    if full["N"] % tp_size:
        raise ValueError(f"{name} N={full['N']} is not divisible by tp_size={tp_size}")
    n = full["N"] // tp_size
    n_blocks = n // ttnn.TILE_SIZE
    if n_blocks < 1 or 64 % n_blocks:
        raise ValueError(f"{name} TP{tp_size} local N={n} cannot tile a 64-core ring")
    return {
        "K": full["K"],
        "N": n,
        "partial_width_sharded": True,
        "k_blocks": 64 // n_blocks,
        "n_blocks": n_blocks,
    }


def hc_fn_ring_specs() -> list:
    """Layouts that stream through :data:`HC_FN_GCB`, in any order (page is their gcd).

    Always includes the TP4 q_a/kv cuts and the router gate so whoever builds the ring
    first -- attention before the hyper-connections, or the router before either --
    sizes it at 8 tiles rather than at a 16-tile slab. Harmless at TP1 for the q_a/kv
    cuts: those layouts are not streamed, they only pin the page.
    """
    return [
        DECODE_LAYOUTS[HC_FN_GCB],
        DECODE_LAYOUTS["router_gate"],
        balanced_qkv_layout("q_a_proj", 4),
        balanced_qkv_layout("kv_proj", 4),
    ]


def tp_gate_up_layout(tp_size: int, K: int, N: int) -> dict:
    """Per-rank shared-expert gate/up: column-parallel ``N = I / tp_size``."""
    full = DECODE_LAYOUTS["shared_gate_proj"]
    if full["N"] % tp_size:
        raise ValueError(f"shared expert N={full['N']} is not divisible by tp_size={tp_size}")
    expected = {"K": full["K"], "N": full["N"] // tp_size}
    if (expected["K"], expected["N"]) != (K, N):
        raise ValueError(
            f"TP{tp_size} gate/up is fixed at K={expected['K']}, N={expected['N']} "
            f"but this config wants K={K}, N={N}"
        )
    return expected


def ensure_named_gcb(
    prefetch_buffers: dict,
    key: str,
    device: ttnn.MeshDevice,
    specs: list,
    weight_dtype: ttnn.DataType,
    num_pages: int = TP_PRIVATE_GCB_PAGES,
):
    """Return ``prefetch_buffers[key]``, building that GCB on first use.

    Mutates the mapping so later layers on the same device reuse the buffer. The
    caller must pass the same dict to every layer (see
    :func:`make_decode_prefetch_buffers`). Defaults to :data:`TP_PRIVATE_GCB_PAGES`
    rather than the shared-ring depth: these buffers serve two weights, not ten.
    """
    if key not in prefetch_buffers:
        prefetch_buffers[key] = make_shared_decode_gcb(device, specs, weight_dtype, num_pages=num_pages)
    return prefetch_buffers[key]


def decode_prefetch_page_bytes(weight_dtype: ttnn.DataType) -> int:
    """The GCB page size every prefetched decode weight is streamed at.

    A pure function of the (fixed) layouts and the weight dtype, so
    :func:`make_decode_prefetch_buffers` and the layers streaming through the buffer it builds
    can each derive it independently and cannot disagree -- which matters because a layer
    streaming at a page size the ring was not built for is a hang.
    """
    return decode_gcb_page_bytes([DECODE_LAYOUTS[name] for name in DECODE_GCB_GROUP], weight_dtype)


def hc_fn_page_bytes(weight_dtype: ttnn.DataType) -> int:
    """The page size every weight on :data:`HC_FN_GCB` streams at.

    The gcd of :func:`hc_fn_ring_specs` (8 tiles at bf4), matching the buffer
    :func:`ensure_named_gcb` builds from the same list.
    """
    return decode_gcb_page_bytes(hc_fn_ring_specs(), weight_dtype)


def make_decode_prefetch_buffers(
    device: ttnn.MeshDevice, weight_dtype: ttnn.DataType, num_prefetch_pages: Optional[int] = None
) -> dict:
    """The GCB every prefetched decode weight on ``device`` streams through.

    Returns a mapping keyed by the names in ``DECODE_GCB_GROUP``, to hand to
    :class:`~.attention.DeepSeekV4Attention` and :class:`~.moe.DeepSeekV4SparseMoeBlock` as
    ``prefetch_buffers``. Every key maps to the same buffer; the mapping exists so a caller
    can still be handed per-weight buffers in a test without the blocks caring.

    :data:`HC_FN_GCB` is deliberately *not* built here -- the hyper-connections attach it to
    this mapping on first use (:func:`ensure_named_gcb`), so a caller with no hyper-connection
    does not pay for a second ring's L1. That is also why the same mapping has to reach every
    layer on the device: a fresh dict per layer would build a GCB per layer and overflow the
    senders' state zone.

    ``num_prefetch_pages`` is the ring depth, and the knob for how far ahead the prefetcher
    may run: at the profile default of 16 pages that is several weights' worth. ``None``
    takes it from the active system profile (``prefetcher.num_prefetch_pages``).

    Build this **once per device and pass it to every layer on that device** -- see the module
    docstring for why, and for the ordering contract that comes with sharing.
    """
    if num_prefetch_pages is None:
        num_prefetch_pages = active_system_config().prefetcher.num_prefetch_pages
    global_cb = make_shared_decode_gcb(
        device,
        [DECODE_LAYOUTS[name] for name in DECODE_GCB_GROUP],
        weight_dtype,
        num_pages=num_prefetch_pages,
    )
    return {name: global_cb for name in DECODE_GCB_GROUP}


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
