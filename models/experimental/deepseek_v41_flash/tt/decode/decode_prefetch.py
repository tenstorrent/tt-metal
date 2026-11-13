# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash's decode prefetch rings.

V4's rings are sized for ``D = 4096``. At V4.1's ``D = 5120`` only q_b and the grouped o_a still
match the shared 64-receiver ring (q_b's ``[1280, 128]`` slab is five of its 32-tile pages), so
V4.1 adds its own private rings next to it:

* q_a on 8 receivers (``[5120, 160]`` slabs, 800 tiles), streamed in 80-tile pages.
* kv on 16 receivers (``[5120, 32]`` slabs), streamed in 32-tile pages.
* :data:`WIDE_RING` on 80 receivers, which three weights stream through at pages of their own: o_b and
  the shared expert's down (``N = 5120`` over 64-column slabs, both in whole-slab pages at TP4), and
  Engram's wkv (bf8, row-parallel under TP; ``[6144 / TP, 320]`` slabs in 30-tile pages). A page is a
  block of one tensor, which the DRISC senders re-grid the ring to per tensor, so the weights need agree
  on neither page size nor dtype (o_b is bf4, the other two bf8). The ring holds four of the largest
  page, o_b's.

The router gate (12 cores) and the shared expert's gate/up (36 cores, 9 per rank at TP4) do not
divide the 8 DRAM banks a ring spreads over. Their matmuls read the DRAM ND-sharded weight
themselves, as do the indexer projections and the compressor.

The q_a and kv rings' pages are pinned by a spec that is not a consumer, as V4's shared ring is: a
ring's page is the gcd of its specs' slabs, and a whole-slab page would not fit two pages deep.

A ring takes its L1 at one address on every worker core, used there or not. The shared ring's 64
receivers are the 8x8 rectangle at (0, 0), q_b's B grid; q_a's 8 and kv's 16 sit beside it, a 4x2
and a 4x4 block stacked in columns 8-11. With no receiver in common the three are reserved in one
range (:func:`make_prefetch_buffers`), which costs every core the shared ring's 288 KB rather than
288 + 180 + 72 KB. Stacked, q_a's and kv's grids also form the one rectangle the tokens' all-gather
multicasts its replica onto. :data:`WIDE_RING`'s 80 receivers overlap the shared ring's, so it keeps a
range of its own.

One ring is one FIFO, so its weights must be queued in the order they are consumed: per layer Engram's
wkv, o_b, then the shared down, which is the order
:meth:`~.decoder_layer.DeepSeekV41DecoderLayer.prefetch_weights` queues and decode pops them in.
:func:`make_prefetch_buffers` builds :data:`WIDE_RING` for all three; a mapping without it (a test of one
block) gets one sized for whichever weight is built first, which another member then rejects.
"""

from typing import Optional

import ttnn

from models.experimental.deepseek_v4_flash.tt.decode.decode_prefetch import (
    DECODE_RING_GROUP,
    KV_RING,
    Q_A_RING,
    PrefetchRing,
    decode_ring_group_specs,
)
from models.experimental.deepseek_v4_flash.tt.layers import (
    _bank_to_receivers,
    _reserve_prefetch_pipes,
    _slab_bytes,
    make_coallocated_decode_pipes,
    make_multi_page_decode_pipes,
)

# Depth of the shared ring in 32-tile pages: 320 tiles (180 KiB per core at bf4 = 2 x 160-tile q_b slabs),
# eliminating the pad-credit stall between consecutive fused_qkv layers at zero extra L1 because
# Q_A_RING_PAGES = 4 already sizes the co-allocated range to 180 KiB.
SHARED_RING_PAGES = 10
# 4 pages of 80 tiles (16 rows of q_a's 5-tile-wide slab): ~180 KB per core at bf4.
_Q_A_PAGE_PIN = {"K": 512, "N": 1280, "n_blocks": 8}
Q_A_RING_PAGES = 4
# 8 pages of 32 tiles: ~144 KB per core at bf4, within the range q_a's ring sizes.
_KV_PAGE_PIN = {"K": 1024, "N": 512, "n_blocks": 16}
KV_RING_PAGES = 8
# Top-left receiver core (x, y) of each ring: q_a's 4x2 block, then kv's 4x4 below it, right of the
# shared ring's 8x8.
Q_A_RING_ORIGIN = (8, 0)
KV_RING_ORIGIN = (8, 2)
# The 80-receiver ring o_b, Engram's wkv and the shared expert's down stream through, each at its own
# page: the ring is sized by o_b's, 288 KB a core, about what the three took as rings of their own.
WIDE_RING = "wide_full"
# o_b's [2048, 64] slab at TP4 (128 tiles) streams in whole-slab pages (~72 KB at bf4), pinned by a
# spec that is not a consumer (at TP1 a slab is 4 such pages). Its A is one tile row (decode is batch
# 1), which is what lets matmul_decode stream a slab over several pages.
_O_B_PAGE_PIN_K = 2048
O_B_RING_PAGES = 4
# Whole-slab pages: [576, 64] at TP4, ~39 KB per page at bf8.
SHARED_DOWN_RING_PAGES = 2
# A [6144, 320] slab of Engram's wkv (1920 tiles, ~2 MB at bf8) is streamed in pages of 3 tile-rows
# (30 tiles, ~32 KB at bf8), pinned by a spec that is not a consumer. The router gate is not on this
# ring: its matmul reads the DRAM ND-sharded weight itself.
_ENGRAM_WKV_PAGE_PIN = {"K": 96, "N": 25600, "n_blocks": 80}
ENGRAM_WKV_RING_PAGES = 2
# The shared expert's and the Engram's weight dtype (their blocks' default), which their pages on
# WIDE_RING are sized at; attention's o_b takes the model's weight_dtype.
SHARED_EXPERT_ENGRAM_DTYPE = ttnn.bfloat8_b
# attention_qkv_decode's q_a (40 cores at (8, 0), 4 wide) and kv (16 at (0, 8), 8 wide) rings, one [5120, 32] slab
# deep each. They overlap the q_a / kv rings above and WIDE_RING, so the two share a range of their own. Its q_b
# streams through the shared ring, whose 8x8 receivers are q_b's cores.
FUSED_Q_A_RING = "fused_q_a"
FUSED_KV_RING = "fused_kv"
_FUSED_RINGS = ((FUSED_Q_A_RING, 40, 4, (8, 0)), (FUSED_KV_RING, 16, 8, (0, 8)))
_FUSED_SLAB = (5120, 32)


def prefetch_rings(tp_size: int = 1) -> dict:
    """V4.1's rings by weight name, at the per-rank shapes of a 1x``tp_size`` stage; a name that is
    absent is not prefetched."""
    # Deferred: both modules import this one for their blocks' ``prefetch_rings``.
    from .attention import DECODE_LAYOUTS as ATTENTION_LAYOUTS
    from .engram import DECODE_LAYOUTS as ENGRAM_LAYOUTS
    from .moe import DECODE_LAYOUTS as MOE_LAYOUTS

    o_b = ATTENTION_LAYOUTS["o_b_proj"]
    down = MOE_LAYOUTS["shared_down_proj"]
    wkv = ENGRAM_LAYOUTS["wkv"]
    shared = tuple(decode_ring_group_specs())
    return {
        "q_a_proj": PrefetchRing(
            Q_A_RING, (ATTENTION_LAYOUTS["q_a_proj"], _Q_A_PAGE_PIN), Q_A_RING_PAGES, Q_A_RING_ORIGIN
        ),
        "kv_proj": PrefetchRing(KV_RING, (ATTENTION_LAYOUTS["kv_proj"], _KV_PAGE_PIN), KV_RING_PAGES, KV_RING_ORIGIN),
        "q_b_proj": PrefetchRing("q_b_proj", shared),
        "o_a_proj": PrefetchRing("o_a_proj", shared),
        "o_b_proj": PrefetchRing(
            WIDE_RING,
            ({"K": o_b["K"] // tp_size, "N": o_b["N"]}, {"K": _O_B_PAGE_PIN_K, "N": o_b["N"]}),
            O_B_RING_PAGES,
        ),
        "shared_down_proj": PrefetchRing(
            WIDE_RING, ({"K": down["K"] // tp_size, "N": down["N"]},), SHARED_DOWN_RING_PAGES
        ),
        "engram.wkv": PrefetchRing(
            WIDE_RING, ({**wkv, "K": wkv["K"] // tp_size}, _ENGRAM_WKV_PAGE_PIN), ENGRAM_WKV_RING_PAGES
        ),
    }


def make_prefetch_buffers(
    device,
    weight_dtype,
    num_prefetch_pages: int = SHARED_RING_PAGES,
    tp_size: Optional[int] = None,
    fused_qkv: bool = True,
) -> dict:
    """The per-device ``prefetch_buffers`` mapping: V4's shared ring under the :data:`DECODE_RING_GROUP` keys
    (as :func:`~models.experimental.deepseek_v4_flash.tt.decode.decode_prefetch.make_decode_prefetch_buffers`
    builds it) plus the q_a and kv rings, the three reserved in one L1 range, and :data:`WIDE_RING`.
    ``fused_qkv`` adds attention_qkv_decode's :data:`FUSED_Q_A_RING` and :data:`FUSED_KV_RING` on a grid
    that holds the op.

    ``weight_dtype`` is the attention projections'; :data:`WIDE_RING` is sized for o_b at it and for the shared
    expert's down and Engram's wkv at :data:`SHARED_EXPERT_ENGRAM_DTYPE`. ``num_prefetch_pages`` is the shared
    ring's depth. ``tp_size`` is the stage's TP width, ``None``
    taking ``device``'s chip count (a decode stage is a 1 x TP mesh). Build it once per device and hand it to
    every layer there.
    """
    if tp_size is None:
        tp_size = device.get_num_devices()
    rings = prefetch_rings(tp_size)
    q_a, kv = rings["q_a_proj"], rings["kv_proj"]
    shared, q_a_pipes, kv_pipes = make_coallocated_decode_pipes(
        device,
        [
            (decode_ring_group_specs(), num_prefetch_pages, (0, 0)),
            (q_a.specs, q_a.num_pages, q_a.origin),
            (kv.specs, kv.num_pages, kv.origin),
        ],
        weight_dtype,
    )
    o_b, down, wkv = rings["o_b_proj"], rings["shared_down_proj"], rings["engram.wkv"]
    wide = make_multi_page_decode_pipes(
        device,
        [
            (o_b.specs, weight_dtype, o_b.num_pages),
            (down.specs, SHARED_EXPERT_ENGRAM_DTYPE, down.num_pages),
            (wkv.specs, SHARED_EXPERT_ENGRAM_DTYPE, wkv.num_pages),
        ],
    )
    buffers = {
        **{name: shared for name in DECODE_RING_GROUP},
        q_a.key: q_a_pipes,
        kv.key: kv_pipes,
        WIDE_RING: wide,
    }
    grid = device.compute_with_storage_grid_size()
    if fused_qkv and grid.x >= 12 and grid.y >= 10:
        slab = _slab_bytes(weight_dtype, _FUSED_SLAB)
        fused = _reserve_prefetch_pipes(
            device,
            [(_bank_to_receivers(cores, device, cols, origin), slab) for _, cores, cols, origin in _FUSED_RINGS],
        )
        buffers.update({key: pipes for (key, *_), pipes in zip(_FUSED_RINGS, fused)})
    return buffers
