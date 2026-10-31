# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Short-prompt prefill on a ONE-slot-per-mesh-row chunk trace (``DSV41_PF_UP1_SHORT=1``, default off).

With ``DSV41_PREFILL_UP = U = 2`` (B=8) every chunk replay runs 2 x 512 row-tokens per mesh row, whatever the number of prompts: a lone short prompt pays for an idle second slot (about 0.17 s of the
0.4-0.5 s device time of a window: B=4, which has one slot, has a c1 TTFT of 434 ms against 607 ms at B=8). A second prefill context with ONE slot per mesh row (own carried tables at a small S_pad, own
input / hand-off buffers, own chunk trace; ``PrefillModel.compile_dyn_up``) serves a call only when it is certain that no carried state is shared with the Up = U context:

  * every user of the call starts from scratch (start 0: no restore, no resumed or carried state), and the call holds the user's WHOLE prompt (end <= chunk and the plugin said the row is final);
  * at most one such user per mesh row, no image prompt, no prefix-cache hit (host tier or paged cache) for any of them.

Everything else keeps the Up = U trace. This module holds the pure selection predicate (CPU tested).
"""

import os


def enabled():
    return os.environ.get("DSV41_PF_UP1_SHORT", "0") == "1"


def small_spad(max_ctx, chunk):
    """S_pad of the one-slot context: the smallest context the single trace supports (a column slab needs >= 512 local key slots for the top-k: 16384), a multiple of the chunk, below the main context."""
    s = int(os.environ.get("DSV41_PF_UP1_SPAD", "16384"))
    s = -(-s // chunk) * chunk
    return s if s < int(max_ctx) else 0


def eligible(items, chunk, rows, U, Up, final, vis=None, has_hit=None, owned=None, max_end=None):
    """True when the whole call may run on the one-slot context.

    ``items`` [(user, tokens, start, end)], ``final`` {user: bool} (None: the plugin does not say -> False), ``vis`` {user: VisionSeq} of image prompts, ``has_hit(user, tokens, end)`` -> True when a prefix
    cache would restore a snapshot for that prompt, ``owned(user, end)`` -> True when the paged prefix cache would commit / serve blocks of that prompt (it works on whole 512-token blocks: a prompt shorter than one block never does).
    """
    if not items or Up != U or U < 2 or final is None:
        return False
    seen = set()
    for b, t, st, e in items:
        b, st, e = int(b), int(st), int(e)
        if st != 0 or not (0 < e <= chunk):
            return False
        if max_end is not None and e > max_end:
            return False
        if not final.get(b):
            return False
        if vis and b in vis:
            return False
        r = b // U
        if r in seen:
            return False
        seen.add(r)
        if owned is not None and owned(b, e):
            return False
        if has_hit is not None and has_hit(b, t, e):
            return False
    return True
