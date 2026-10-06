# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The GlobalCircularBuffer ("GCB") rings every prefetched decode weight streams through.

One *shared* ring per device serves every :class:`~..layers.LinearDecode` and
:class:`~..layers.BatchedLinearDecode` on it: the attention block's q_b and grouped output
projection, and the MoE shared expert's down (:data:`DECODE_GCB_GROUP`). Cuts that do not fit
that ring's 64 receivers get their own -- q_a full-width on 32 (the CSA compressor, the
hyper-connections' fused ``fn`` and the TP1 shared expert's gate/up share it), kv on 16 (HCA
shares it), and the router gate full-width but hub mode on 8: :data:`Q_A_GCB`, :data:`KV_GCB`,
:data:`ROUTER_GATE_GCB`. This module owns the two things agreed model-wide -- the weight layouts
and the order the matmuls consume them -- so no block can size a buffer against a layout another
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
page size *does* change hangs (see :func:`~..layers.make_shared_decode_gcb`) -- and every weight
on a ring wants the same receiver count, because a GCB's receiver set is fixed at construction
(64 on the shared ring; a "B core" is a receiver core of the matmul's second operand, not the
decode batch). ``o_a_proj``'s batched ``b_blocks x n_blocks`` grid is 64 for exactly that
reason.

The price is one FIFO ordering contract per ring, and nothing checks it: a matmul that runs out
of turn pops another weight's page and returns wrong results rather than an error.
:data:`DECODE_GCB_GROUP` is that order for the shared ring.

:data:`Q_A_GCB` is always built from :func:`q_a_ring_specs` -- every layout that streams through
it -- so its page (their gcd, 16 tiles) does not depend on which block builds it first. The
fused ``fn`` is cut 32 ways along K (a ``[512, 32]`` slab, 16 tiles) precisely so that it fits
that ring; on 64 receivers its 8-tile slab shared no page with anything and needed a ring of
its own.

Letters: ``K``/``N`` are a weight's in/out features -- a ``[K, N]`` weight, cut into one
``[Kc, Nc]`` slab per receiver core; ``D`` = ``hidden_size``, ``I`` = ``moe_intermediate_size``,
``E`` = ``num_local_experts``, ``T`` = token rows of an activation's shard.
"""

from typing import Optional

import ttnn

from ..layers import decode_gcb_page_bytes, make_shared_decode_gcb
from ..system_config import active_system_config

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
    # shared ring. weights_proj is 2 cores, which does not divide the 8 DRAM banks,
    # so it stays on the per-step DRAM -> L1 copy.
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
    # absent from DECODE_GCB_GROUP: hub mode needs a full-width cut on 8 receivers, which is
    # not that ring's fixed receiver set. It streams through :data:`ROUTER_GATE_GCB`.
    "router_gate": {"K": 4096, "N": 256, "n_blocks": 8},
    # Both hyper-connections' fused fn projection: K = hc_mult * hidden_size against the
    # 24 pre/post/comb outputs padded to one tile. Large K and a single tile of N, so it
    # cuts K 32 ways and reduces onto one output core: 32 receivers so its [512, 32] slab
    # (16 tiles) streams through q_a's ring, one per K shard of its 32-core activation.
    "hc_fn": {"K": 16384, "N": 32, "partial_width_sharded": True, "k_blocks": 32, "n_blocks": 1},
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
# The kv and router rings cannot use the shared 16/24-page depth: each has a single
# spec, so the page is the whole 128-tile slab (72 KB at bf4). 24 such pages does
# not fit in a Blackhole L1 bank after the shared GCB. Two pages is the streaming
# floor.
TP_PRIVATE_GCB_PAGES = 2
# q_a's ring streams 16-tile pages (see :func:`q_a_ring_specs`); 16 of them hold two
# 128-tile q_a slabs, the same 144 KB at bf4 as two whole-slab pages.
Q_A_GCB_PAGES = 16


def q_a_ring_specs() -> list:
    """The layouts :data:`Q_A_GCB` is sized from, **in the order one layer consumes them**:
    the attention hyper-connection's ``fn``, q_a, CSA's compressor kv/gate, the FFN
    hyper-connection's ``fn``, then the TP1 shared expert's gate/up.

    All are 32 receivers. The page is the gcd of their slabs: the ``fn`` layout's ``[512, 32]``
    slab, 16 tiles (9 KB at bf4), which is a whole number of rows of the 128-tile q_a /
    compressor slabs and of the 256-tile ``[4096, 64]`` gate/up slab. Gate/up stays listed at
    TP4, where it is off the prefetcher, so the ring's page does not depend on the TP width.
    """
    return [
        DECODE_LAYOUTS["hc_fn"],
        DECODE_LAYOUTS["q_a_proj"],
        DECODE_LAYOUTS["compressed_sparse_attention"],
        DECODE_LAYOUTS["shared_gate_proj"],
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


def ensure_q_a_gcb(prefetch_buffers: dict, device: ttnn.MeshDevice, weight_dtype: ttnn.DataType):
    """:data:`Q_A_GCB`, built on first use from :func:`q_a_ring_specs` at :data:`Q_A_GCB_PAGES`.

    The single entry point for that ring, so whichever of its consumers is built first sizes it
    the same way; pair it with :func:`q_a_page_bytes`.
    """
    return ensure_named_gcb(prefetch_buffers, Q_A_GCB, device, q_a_ring_specs(), weight_dtype, Q_A_GCB_PAGES)


def decode_prefetch_page_bytes(weight_dtype: ttnn.DataType) -> int:
    """The GCB page size in bytes every prefetched decode weight on the shared ring is streamed
    at: 32 tiles, i.e. 18 KB of a ``[32, 32]`` tile at bf4.

    A pure function of the (fixed) layouts and the weight dtype, so
    :func:`make_decode_prefetch_buffers` and the layers streaming through the buffer it builds
    can each derive it independently and cannot disagree -- which matters because a layer
    streaming at a page size the ring was not built for is a hang.
    """
    return decode_gcb_page_bytes(decode_gcb_group_specs(), weight_dtype)


def q_a_page_bytes(weight_dtype: ttnn.DataType) -> int:
    """Page size in bytes every weight on :data:`Q_A_GCB` streams at: the gcd of
    :func:`q_a_ring_specs` (16 tiles, 9 KB at bf4), matching the buffer :func:`ensure_q_a_gcb`
    builds from the same list. A page is a whole number of rows of a ``[Kc, Nc]`` slab, so a
    weight whose slab is several pages (q_a's 128 tiles are 8) is streamed across them."""
    return decode_gcb_page_bytes(q_a_ring_specs(), weight_dtype)


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

    The private rings (:data:`Q_A_GCB`, :data:`KV_GCB`, :data:`ROUTER_GATE_GCB`) are deliberately
    *not* built here -- their consumers attach them to this mapping on first use
    (:func:`ensure_named_gcb`), so a caller without those blocks does not pay for their L1. That
    is also why the same mapping has to reach every
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
    ``o_groups`` folded into :class:`~..layers.BatchedLinearDecode`'s ``[Bc*K, Nc]`` per-core
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
