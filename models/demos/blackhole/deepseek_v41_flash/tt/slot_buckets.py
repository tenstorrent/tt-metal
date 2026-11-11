# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Slot-bucketed prefill windows (``DSV41_PREFILL_SLOT_BUCKETS=1,2``, default off), pure selection logic (CPU tested).

With ``DSV41_PREFILL_UP = U`` (B=16: 4) the traced chunk holds U prefill slots per mesh row (slot = in-row index of the decode user) and a slot with no prompt in the window is masked, not skipped: a lone
prompt pays for U slots (0.40 s per 512-token window at Up=4, 0.151 s at Up=1). Extra chunk traces run the first k < U slots only, on the SAME carried per-slot state (halo, index-key FIFO: prefix views of
the main context's tensors, ``DSV41PrefillModel.compile_dyn_slots``): a call picks the smallest captured k that covers the in-row index of every user it prefills. The placement policy
(``SlotBook.claim_packed``: lowest free in-row index among the rows of equal prefill load) keeps the prefilling users in the low slots.
"""

import os


def enabled():
    return bool(os.environ.get("DSV41_PREFILL_SLOT_BUCKETS", "").strip())


def parse(spec, U):
    """'1,2' -> [1, 2]: the slot counts k (1 <= k < U) with an extra trace, ascending; U itself is the main trace. '' / None -> []."""
    out = set()
    for x in (spec or "").replace(";", ",").split(","):
        x = x.strip()
        if not x:
            continue
        k = int(x)
        if 1 <= k < int(U):
            out.add(k)
    return sorted(out)


def need_slots(users, U):
    """slots the call needs: 1 + the highest in-row index of its users (0 when it has none)"""
    return 1 + max((int(b) % int(U) for b in users), default=-1)


def pick(ks, have, need):
    """smallest k of ``ks`` (ascending) that is captured (``have``) and covers ``need`` slots; 0 = the full trace"""
    for k in ks:
        if k >= need and k in have:
            return k
    return 0


def cost_model(t_full, t_k, k, U, n_real):
    """expected window wall time (s) of a call with ``n_real`` real slots in its busiest row: the bucket when one covers it, else the full trace (diagnostics for the report)"""
    return t_k.get(n_real if n_real in t_k else next((x for x in sorted(t_k) if x >= n_real), None), t_full)
