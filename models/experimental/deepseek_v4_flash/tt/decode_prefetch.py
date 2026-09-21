# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The GlobalCircularBuffer ("GCB") rings every prefetched decode weight streams through.

One *shared* ring per device serves every :class:`~.layers.LinearDecode` and
:class:`~.layers.BatchedLinearDecode` on it: the attention block's q_b and grouped output
projection, and the MoE shared expert's down (:data:`DECODE_GCB_GROUP`). Cuts that do not fit
that ring's 64 receivers get their own -- q_a full-width on 32 (the CSA compressor shares it),
kv on 16 (HCA shares it), the router gate full-width but hub mode on 8, and the
hyper-connections' fused ``fn``: :data:`Q_A_GCB`, :data:`KV_GCB`, :data:`ROUTER_GATE_GCB`,
:data:`HC_FN_GCB`. This module owns the two things agreed model-wide -- the weight layouts and
the order the matmuls consume them -- so no block can size a buffer against a layout another
block does not use.

Why share rather than size a ring per shape:

* A GCB is a permanent L1 allocation: 288 KB per receiver core at the default depth, so a
  per-projection ring would multiply that by ten and a per-layer one by the layer count.
* It also takes a fixed ~176 B of the DRISC senders' 1 KB state zone, which caps a device at
  about six however small the rings are.
* Above all it is what lets the prefetcher run *ahead*: a 16-page ring holds far more than the
  two pages the matmul on it needs, so the senders work through later weights while the workers
  are still on an earlier one, instead of resynchronising at every weight boundary.

Sharing is possible because the weights stream. The slabs here run from 32 to 512 tiles but all
divide into 32-tile pages, so a ring's page size never changes between transfers -- a ring whose
page size *does* change hangs (see :func:`~.layers.make_shared_decode_gcb`) -- and every weight
on a ring wants the same receiver count, because a GCB's receiver set is fixed at construction
(64 on the shared ring; a "B core" is a receiver core of the matmul's second operand, not the
decode batch). ``o_a_proj``'s batched ``b_blocks x n_blocks`` grid is 64 for exactly that
reason.

The price is one FIFO ordering contract per ring, and nothing checks it: a matmul that runs out
of turn pops another weight's page and returns wrong results rather than an error.
:data:`DECODE_GCB_GROUP` is that order for the shared ring.

:data:`HC_FN_GCB` is a *second* buffer on the same prefetch mapping, because the fused ``fn``
weight's ``[256, 32]`` slab is 8 tiles and has no page in common with the shared ring's 32. It
is always built from :func:`hc_fn_ring_specs` -- the ``fn`` layout plus the TP4 q_a/kv layouts,
all three on the same 64 receivers -- so its page (their gcd, 8 tiles) does not depend on which
weight is built first, and only ``fn`` streams through it (the deployed q_a/kv are full-width,
on :data:`Q_A_GCB` / :data:`KV_GCB`). One ring, not one per hyper-connection: two per layer
would overflow the senders' state zone at the third layer.

Letters: ``K``/``N`` are a weight's in/out features -- a ``[K, N]`` weight, cut into one
``[Kc, Nc]`` slab per receiver core; ``D`` = ``hidden_size``, ``I`` = ``moe_intermediate_size``,
``E`` = ``num_local_experts``, ``T`` = token rows of an activation's shard.
"""

from typing import Optional

import ttnn

from .layers import decode_gcb_page_bytes, make_shared_decode_gcb
from .system_config import active_system_config

# The hyper-connections' ``fn`` ring, keyed separately from ``DECODE_GCB_GROUP`` because its
# 8-tile slab shares no page with that group's 32. :func:`ensure_named_gcb` builds it on first
# use; both hyper-connections of every layer on the device stream through it.
HC_FN_GCB = "hc_fn"
# Depth of the 8-tile ring. Both hyper-connections of one layer queue a whole-slab page
# each, and the next layer on the chip may be staged before the current one has drained;
# 8 leaves slack without moving the 4.6 KB page into the same class as the shared 18 KB
# ring.
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
    # Full-width on 32 cores so matmul_decode can fuse q_a's RMSNorm and multicast the
    # normalized full row directly onto q_b's 64-core weight grid.
    "q_a_proj": {"K": 4096, "N": 1024, "n_blocks": 32},
    "q_b_proj": {"K": 1024, "N": 32768, "n_blocks": 64},
    # CSA Lightning Indexer (index_head_dim=128, index_n_heads=64). kv/gate share
    # the Ca/Cb pair at 2*128; q_b is 64*128; weights_proj is one tile of heads.
    # Prefetch rides existing rings where the B-core count matches: kv/gate on the
    # router gate's 8-receiver ring (same ``[4096, 256]`` cut), q_b on the 64-receiver
    # shared ring; weights_proj is 2 cores and gets its own ring via ensure_named_gcb.
    "indexer.kv_proj": {"K": 4096, "N": 256, "n_blocks": 8},
    "indexer.q_b_proj": {"K": 1024, "N": 8192, "n_blocks": 64},
    "indexer.weights_proj": {"K": 4096, "N": 64, "n_blocks": 2},
    "kv_proj": {"K": 4096, "N": 512, "n_blocks": 16},
    "o_b_proj": {"K": 8192, "N": 4096},
    # Full-width hub mode so both projections consume the decode all-gather replica
    # (ROW_MAJOR HEIGHT_SHARDED A). CSA matches q_a (32 cores); HCA matches kv (16).
    "compressed_sparse_attention": {"K": 4096, "N": 1024, "n_blocks": 32},
    "heavily_compressed_attention": {"K": 4096, "N": 512, "n_blocks": 16},
    # The grouped output projection (o_a of DeepseekV4GroupedLinear): batched over o_groups,
    # folded along both batch and N into a b_blocks x n_blocks grid (see
    # BatchedLinearDecode). 8x8 is what falls out of the model's o_groups=8, o_lora_rank=1024
    # on this device's grid -- the same 64 receivers as everything else, which is what lets
    # it join this buffer at all.
    "o_a_proj": {"K": 4096, "N": 1024, "batch": 8, "b_blocks": 8, "n_blocks": 8},
    # The MoE shared expert. gate/up are full-width (N = I) on N//64 B cores so they can
    # consume the decode all-gather replica (ROW_MAJOR HEIGHT_SHARDED A) and emit
    # ROW_MAJOR WIDTH_SHARDED [T, I] there -- 64 columns per core, the K-sharding
    # down_proj wants of its activation. At TP4 the per-rank cut is N = I/tp (8 cores at
    # I = 2048), which matches no ring's receiver set, so these two stay on the per-step
    # DRAM -> L1 copy (use_prefetcher=False). Down stays 64-core partial-width on the
    # shared ring at every TP.
    "shared_gate_proj": {"K": 4096, "N": 2048, "n_blocks": 32},
    "shared_up_proj": {"K": 4096, "N": 2048, "n_blocks": 32},
    "shared_down_proj": {"K": 2048, "N": 4096},
    # Router gate: [D, E] = [4096, 256] -- ``hidden_size`` by ``num_local_experts``. Full
    # width so ``matmul_decode`` runs in hub mode and consumes the decode all-gather replica
    # (ROW_MAJOR HEIGHT_SHARDED A) where it already sits; a K-split cut has to unreplicate it
    # through DRAM and re-shard it first, which is four device ops per step. Deliberately
    # absent from DECODE_GCB_GROUP, and from :data:`HC_FN_GCB` too: hub mode needs a
    # full-width cut on 8 receivers, which is neither ring's fixed receiver set. It streams
    # through :data:`ROUTER_GATE_GCB`.
    "router_gate": {"K": 4096, "N": 256, "n_blocks": 8},
    # Both hyper-connections' fused fn projection: K = hc_mult * hidden_size against the
    # 24 pre/post/comb outputs padded to one tile. Large K and a single tile of N, so it
    # cuts K 64 ways onto the same 64 receivers and reduces onto one output core. Its
    # [256, 32] slab is 8 tiles, which is why it needs HC_FN_GCB rather than the group
    # above. Deliberately absent from DECODE_GCB_GROUP.
    HC_FN_GCB: {"K": 16384, "N": 32, "partial_width_sharded": True, "k_blocks": 64, "n_blocks": 1},
}

# The order one layer's matmuls consume the shared buffer, which is the order the requests must
# be queued in. q_a has a private 32-receiver FIFO (the CSA compressor's kv/gate share it, after
# q_a); kv has a private 16-receiver FIFO (HCA's pair shares it, after kv). This shared FIFO
# starts with q_b from ``_qkv``, then ``_attend``'s grouped output projection (o_a_proj before
# o_b_proj -- see ``DeepSeekV4Attention._grouped_output``). The MoE shared down follows.
DECODE_GCB_GROUP = (
    "q_b_proj",
    "o_a_proj",
    "o_b_proj",
    "shared_down_proj",
)

# Not a consumer: pins the shared ring's page at 32 tiles. Without it the remaining 128- and
# 512-tile slabs gcd to 128 tiles, and 16 pages of that (~1.2 MB/core at bf4) leave no room
# for the 16-core kv ring.
_SHARED_GCB_PAGE_PIN = {
    "K": 4096,
    "N": 512,
    "partial_width_sharded": True,
    "k_blocks": 4,
    "n_blocks": 16,
}


def decode_gcb_group_specs() -> list:
    """The ``decode_weight_layout`` dicts (**in the order the matmuls consume them**) that size
    the shared decode GCB: the :data:`DECODE_GCB_GROUP` consumers plus the 32-tile page pin,
    which is not a consumer but fixes the ring's page (their ``[Kc, Nc]`` slabs' gcd)."""
    return [DECODE_LAYOUTS[name] for name in DECODE_GCB_GROUP] + [_SHARED_GCB_PAGE_PIN]


# Extra GCBs attached to the per-device prefetch mapping. Not in
# ``DECODE_GCB_GROUP``: different receiver counts, independent FIFOs.
Q_A_GCB = "q_a_full"
KV_GCB = "kv_full"
ROUTER_GATE_GCB = "router_gate"
# Those private rings cannot use the shared 16/24-page depth: each has a single
# spec, so the page is the whole 128-tile slab (72 KB at bf4). 24 such pages does
# not fit in a Blackhole L1 bank after the shared GCB. Two pages is the streaming
# floor.
TP_PRIVATE_GCB_PAGES = 2


def balanced_qkv_layout(name: str, tp_size: int) -> dict:
    """A column-parallel ``N / tp_size`` ``q_a`` / ``kv`` layout on 64 receivers, as a
    ``decode_weight_layout`` dict.

    ``n_blocks`` is one tile of local ``N`` per core and ``k_blocks`` takes up the rest of the
    factor so that ``k_blocks * n_blocks == 64``, the receiver count a GCB is fixed to; the
    ``[Kc, Nc]`` slab is then 16 tiles for ``q_a`` and 8 for ``kv`` at TP4 (``[512, 32]`` and
    ``[256, 32]``). Raises for a name that is not q_a/kv, an ``N`` not divisible by ``tp_size``,
    or a local ``N`` that cannot tile 64 cores. Used only to pin :data:`HC_FN_GCB`'s geometry
    (see :func:`hc_fn_ring_specs`): the deployed TP4 q_a/kv weights are full-width (replicated)
    and stream through :data:`Q_A_GCB` / :data:`KV_GCB`.
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
    """The layouts :data:`HC_FN_GCB` is sized from: the fused ``fn`` layout plus the TP4
    q_a/kv layouts.

    The ring's page is the gcd of their slabs -- the ``fn`` layout's ``[256, 32]`` slab, 8 tiles
    (4.6 KB) at bf4 -- which the other two entries do not change; they are listed so that
    :func:`~.layers.make_shared_decode_gcb` asserts the same receiver count (64) for those cuts
    too. Only ``fn`` streams through the ring; q_a/kv weights go through :data:`Q_A_GCB` /
    :data:`KV_GCB`.

    The router gate is *not* here: it is full-width (hub mode), so its B grid is 8
    receivers rather than the 64 a GCB's receiver set is fixed to. It has its own
    ring (:data:`ROUTER_GATE_GCB`).
    """
    return [
        DECODE_LAYOUTS[HC_FN_GCB],
        balanced_qkv_layout("q_a_proj", 4),
        balanced_qkv_layout("kv_proj", 4),
    ]


def tp_gate_up_layout(tp_size: int, K: int, N: int) -> dict:
    """Per-rank shared-expert gate/up: column-parallel ``N = I / tp_size``, full-width (no
    ``n_blocks``), i.e. ``N // 64`` B cores -- 8 of them at TP4's ``I = 2048``.

    Checks that ``K`` and the local ``N`` are the ones that cut implies (a ``[K, N]`` weight),
    since a config it does not describe would otherwise reach the device mis-sharded. That
    8-receiver cut matches no ring, so at TP4 the shared expert's gate/up keep
    ``use_prefetcher=False`` (see :class:`~.moe.DeepSeekV4SparseMoeBlock`).
    """
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

    ``specs`` are ``decode_weight_layout`` dicts whose ``[Kc, Nc]`` slabs fix the ring's page
    size and receiver count; the returned buffer is a ``ttnn.GlobalCircularBuffer``.

    Mutates the mapping so later layers on the same device reuse the buffer. The
    caller must pass the same dict to every layer (see
    :func:`make_decode_prefetch_buffers`). Defaults to :data:`TP_PRIVATE_GCB_PAGES`
    rather than the shared-ring depth: these buffers serve two weights, not ten.
    """
    if key not in prefetch_buffers:
        prefetch_buffers[key] = make_shared_decode_gcb(device, specs, weight_dtype, num_pages=num_pages)
    return prefetch_buffers[key]


def decode_prefetch_page_bytes(weight_dtype: ttnn.DataType) -> int:
    """The GCB page size in bytes every prefetched decode weight on the shared ring is streamed
    at: 32 tiles, i.e. 18 KB of a ``[32, 32]`` tile at bf4.

    A pure function of the (fixed) layouts and the weight dtype, so
    :func:`make_decode_prefetch_buffers` and the layers streaming through the buffer it builds
    can each derive it independently and cannot disagree -- which matters because a layer
    streaming at a page size the ring was not built for is a hang.
    """
    return decode_gcb_page_bytes(decode_gcb_group_specs(), weight_dtype)


def hc_fn_page_bytes(weight_dtype: ttnn.DataType) -> int:
    """Page size in bytes every weight on :data:`HC_FN_GCB` streams at: the gcd of
    :func:`hc_fn_ring_specs` (8 tiles, 4.6 KB at bf4), matching the buffer
    :func:`ensure_named_gcb` builds from the same list. A page is a whole number of rows of a
    ``[Kc, Nc]`` slab, so a weight whose slab is several pages is streamed across them."""
    return decode_gcb_page_bytes(hc_fn_ring_specs(), weight_dtype)


def q_a_page_bytes(weight_dtype: ttnn.DataType) -> int:
    """Page size in bytes for q_a's private 32-receiver full-width ring: its single spec makes
    the ``[4096, 32]`` slab (128 tiles, 72 KB at bf4) one whole page, streamed at
    :data:`TP_PRIVATE_GCB_PAGES` pages of depth."""
    return decode_gcb_page_bytes([DECODE_LAYOUTS["q_a_proj"]], weight_dtype)


def router_gate_page_bytes(weight_dtype: ttnn.DataType) -> int:
    """Page size in bytes for the router gate's private 8-receiver full-width ring: its
    ``[4096, 32]`` slab (128 tiles) is one whole page."""
    return decode_gcb_page_bytes([DECODE_LAYOUTS["router_gate"]], weight_dtype)


def kv_page_bytes(weight_dtype: ttnn.DataType) -> int:
    """Page size in bytes for kv's private 16-receiver full-width ring: its ``[4096, 32]`` slab
    (128 tiles) is one whole page."""
    return decode_gcb_page_bytes([DECODE_LAYOUTS["kv_proj"]], weight_dtype)


def make_decode_prefetch_buffers(
    device: ttnn.MeshDevice, weight_dtype: ttnn.DataType, num_prefetch_pages: Optional[int] = None
) -> dict:
    """The shared decode GCB every prefetched weight on ``device`` streams through, in the
    layouts of :func:`decode_gcb_group_specs`.

    Returns a mapping keyed by the names in :data:`DECODE_GCB_GROUP`, to hand to
    :class:`~.attention.DeepSeekV4Attention` and :class:`~.moe.DeepSeekV4SparseMoeBlock` as
    ``prefetch_buffers``. Every key maps to the same ``ttnn.GlobalCircularBuffer``, whose pages
    are 32 tiles of the ``[Kc, Nc]`` slabs in :func:`decode_gcb_group_specs`; the mapping
    exists so a caller can still be handed per-weight buffers in a test without the blocks
    caring.

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
        decode_gcb_group_specs(),
        weight_dtype,
        num_pages=num_prefetch_pages,
    )
    return {name: global_cb for name in DECODE_GCB_GROUP}


def check_decode_layout(name: str, K: int, N: int, batch: Optional[int] = None) -> dict:
    """``DECODE_LAYOUTS[name]``, having checked it against the ``K``/``N`` (and, for a batched
    weight, ``batch``) the config wants.

    The layouts are constants (the shared GCB is sized from them before any weight is built),
    so a config they do not describe has to be caught here: left alone it would reach the
    device as a silently mis-sharded weight rather than an error. ``batch`` is the number of
    ``o_groups`` folded into :class:`~.layers.BatchedLinearDecode`'s ``[Bc*K, Nc]`` per-core
    block, and is ``None`` for an unbatched weight.
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
