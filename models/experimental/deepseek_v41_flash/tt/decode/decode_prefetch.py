# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash's decode prefetch rings.

V4's rings are sized for ``D = 4096``. At V4.1's ``D = 5120`` only q_b and the grouped o_a still
match the shared 64-receiver GCB (q_b's ``[1280, 128]`` slab is five of its 32-tile pages), so
V4.1 adds its own private rings next to it:

* q_a on 8 receivers (``[5120, 160]`` slabs, 800 tiles), streamed in 80-tile pages.
* kv on 16 receivers (``[5120, 32]`` slabs), streamed in 32-tile pages.
* o_b, and separately the shared expert's down, on 80 receivers each (``N = 5120`` over
  64-column slabs).
* Engram's wkv (replicated, bf8) on 80 receivers of ``[6144, 320]`` slabs, streamed in 60-tile pages.

The router gate (12 cores) and the shared expert's gate/up (36 cores, 9 per rank at TP4) do not
divide the 8 DRAM banks a ring spreads over, so they stay on the per-step DRAM -> L1 copy.

The q_a and kv rings' pages are pinned by a spec that is not a consumer, as V4's shared ring is: a
ring's page is the gcd of its specs' slabs, and a whole-slab page would not fit two pages deep.
"""

from models.experimental.deepseek_v4_flash.tt.decode.decode_prefetch import (
    KV_GCB,
    Q_A_GCB,
    PrefetchRing,
    decode_gcb_group_specs,
)

O_B_GCB = "o_b_full"
# 4 pages of 80 tiles (16 rows of q_a's 5-tile-wide slab): ~180 KB per core at bf4.
_Q_A_PAGE_PIN = {"K": 512, "N": 1280, "n_blocks": 8}
Q_A_GCB_PAGES = 4
# 4 pages of 32 tiles: ~72 KB per core at bf4.
_KV_PAGE_PIN = {"K": 1024, "N": 512, "n_blocks": 16}
KV_GCB_PAGES = 4
# Whole-slab pages: o_b's [2048, 64] slab at TP4 is 128 tiles, ~72 KB per page at bf4.
O_B_GCB_PAGES = 2
# The shared expert is bfloat8_b while attention is bfloat4_b, and a ring's page is a fixed
# byte count, so its down cannot share o_b's ring. [576, 64] at TP4: ~39 KB per page at bf8.
SHARED_DOWN_GCB = "shared_down_full"
SHARED_DOWN_GCB_PAGES = 2
# Engram's wkv on 80 receivers: a [6144, 320] slab (1920 tiles, ~2 MB at bf8) is streamed in pages of
# 6 tile-rows (60 tiles, ~65 KB at bf8), pinned by a spec that is not a consumer.
ENGRAM_WKV_GCB = "engram_wkv"
_ENGRAM_WKV_PAGE_PIN = {"K": 192, "N": 25600, "n_blocks": 80}
ENGRAM_WKV_GCB_PAGES = 2


def prefetch_rings(tp_size: int = 1) -> dict:
    """V4.1's rings by weight name, at the per-rank shapes of a 1x``tp_size`` stage; a name that is
    absent is not prefetched."""
    # Deferred: both modules import this one for their blocks' ``prefetch_rings``.
    from .attention import DECODE_LAYOUTS as ATTENTION_LAYOUTS
    from .engram import DECODE_LAYOUTS as ENGRAM_LAYOUTS
    from .moe import DECODE_LAYOUTS as MOE_LAYOUTS

    o_b = ATTENTION_LAYOUTS["o_b_proj"]
    down = MOE_LAYOUTS["shared_down_proj"]
    shared = tuple(decode_gcb_group_specs())
    return {
        "q_a_proj": PrefetchRing(Q_A_GCB, (ATTENTION_LAYOUTS["q_a_proj"], _Q_A_PAGE_PIN), Q_A_GCB_PAGES),
        "kv_proj": PrefetchRing(KV_GCB, (ATTENTION_LAYOUTS["kv_proj"], _KV_PAGE_PIN), KV_GCB_PAGES),
        "q_b_proj": PrefetchRing("q_b_proj", shared),
        "o_a_proj": PrefetchRing("o_a_proj", shared),
        "o_b_proj": PrefetchRing(O_B_GCB, ({"K": o_b["K"] // tp_size, "N": o_b["N"]},), O_B_GCB_PAGES),
        "shared_down_proj": PrefetchRing(
            SHARED_DOWN_GCB, ({"K": down["K"] // tp_size, "N": down["N"]},), SHARED_DOWN_GCB_PAGES
        ),
        "engram.wkv": PrefetchRing(ENGRAM_WKV_GCB, (ENGRAM_LAYOUTS["wkv"], _ENGRAM_WKV_PAGE_PIN), ENGRAM_WKV_GCB_PAGES),
    }
