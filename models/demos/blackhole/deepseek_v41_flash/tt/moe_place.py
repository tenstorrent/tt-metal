# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Static expert placement of the decode routed experts (default OFF).

``DSV41_MOE_PLACE=<file.json>`` switches the expert -> chip map of ``moe_compute`` from the shipped sequential layout (expert e on chip ``((e % 48) // 12) * 8 + e // 48``) to a per-layer
permutation chosen from real routing statistics so that the busiest chip streams fewer expert chunks. The file is ``{"<layer>": [chip of expert 0, ..., chip of expert 383], ...}``
(12 experts per chip; layers missing from the file keep the sequential layout). Built by ``dsv4-logs/moe_place/sim2.py`` (``--out``).

What changes (nothing in C++): the ``expert_mapping`` handed to ``TTMoEDecode`` (the table of ``all_to_all_dispatch_metadata`` / ``moe_compute`` / ``moe_tail``), the expert order of the
host weights before they are sharded over the 32 chips (a stable argsort of the mapping, done by ``_TTMoEDecodeExpertState``), and the key of the on-disk packed weight cache
(``layer_<L>`` -> ``layer_<L>_pl<hash of the permutation>``: a cache built for another permutation can never be read).

Numerics: the per-token output is the same sum of the same expert outputs. A placement that keeps every expert in its mesh COLUMN (``e // 48``; "samecol") is bit-identical to the
sequential layout (the tail sums the slots of a column in slot order, whichever chip of the column computed them); a placement that moves experts across columns changes only which
column partial a slot lands in, so the bf16 accumulation order of the column partials / reduce_scatter changes (PCC against the golden unchanged).

The unified (deepseek_prefill) routed-expert prefill reads the decode weights in place and assumes the sequential layout: its router ids are relabelled by ``relabel(layer)`` (a [384] id LUT gather
in ``DSV41UnifiedMoE.forward``; ``DSV41_MOE_PLACE_PF_RELABEL=0`` restores the old refusal, ``check_prefill_compat``).
"""
import hashlib
import json
import os

NE, NCHIP = 384, 32
_CACHE = {}


def path():
    p = os.environ.get("DSV41_MOE_PLACE", "").strip()
    return None if p in ("", "0", "off") else p


def enabled():
    return path() is not None


def _load():
    p = path()
    if p not in _CACHE:
        raw = json.load(open(p))
        raw = raw.get("layers", raw) if isinstance(raw, dict) else raw
        _CACHE[p] = {int(k): [int(x) for x in v] for k, v in raw.items()}
    return _CACHE[p]


def sequential():
    return [((e % 48) // 12) * 8 + e // 48 for e in range(NE)]


def validate(a):
    assert len(a) == NE and all(0 <= c < NCHIP for c in a), "placement must map 384 experts to chips 0..31"
    cnt = [0] * NCHIP
    for c in a:
        cnt[c] += 1
    assert all(n == NE // NCHIP for n in cnt), f"placement must hold 12 experts per chip, got {sorted(set(cnt))}"
    return a


def assignment(layer_id):
    """the chip of every expert of ``layer_id`` (list of 384), or None for the sequential layout"""
    if not enabled():
        return None
    a = _load().get(int(layer_id))
    if a is None or a == sequential():
        return None
    return validate(a)


def tag(a):
    return "pl" + hashlib.sha1(",".join(map(str, a)).encode()).hexdigest()[:10]


def cache_suffix(layer_id):
    a = assignment(layer_id)
    return "" if a is None else "_" + tag(a)


def samecol(a):
    """True if every expert stays in its sequential mesh column (e // 48): bit-identical to the sequential layout"""
    return all(c % 8 == e // 48 for e, c in enumerate(a))


def relabel(layer_id):
    """pi (list of 384) for the unified prefill reading the decode ring weights in place, or None for the sequential layout.

    The weights of expert e sit at chip a[e], slot j = rank of e among the experts of that chip (ascending id). The unified prefill finds the weights of id e' at the chip of the
    SEQUENTIAL layout, slot e' % 12; so the router id e is handed to it as pi(e) = the j-th smallest sequential id of chip a[e]. pi is a permutation (a same-column placement relabels inside its column group).
    """
    a = assignment(layer_id)
    if a is None:
        return None
    seq = sequential()
    seq_set = {d: sorted(e for e in range(NE) if seq[e] == d) for d in range(NCHIP)}
    nxt = {d: 0 for d in range(NCHIP)}
    pi = [0] * NE
    for e in range(NE):  # ascending e: the j-th expert of chip a[e]
        d = a[e]
        pi[e] = seq_set[d][nxt[d]]
        nxt[d] += 1
    assert sorted(pi) == list(range(NE))
    return pi


def prefill_relabel_enabled():
    return os.environ.get("DSV41_MOE_PLACE_PF_RELABEL", "1") == "1"


def check_prefill_compat(layer_ids, ring_prefill):
    """raise if a placement is active for a layer whose decode weights the unified prefill reads in place AND the router-id relabel is switched off (DSV41_MOE_PLACE_PF_RELABEL=0)"""
    if not enabled() or not ring_prefill or prefill_relabel_enabled():
        return
    bad = [L for L in layer_ids if assignment(L) is not None]
    if bad:
        raise RuntimeError(
            f"DSV41_MOE_PLACE is set for layers {bad[:5]}... but the unified prefill reads the decode expert weights in place (DSV41_UNI_RING=1) and assumes the sequential layout, and "
            "the relabel is off (DSV41_MOE_PLACE_PF_RELABEL=0): set DSV41_PREFILL_MOE=off or DSV41_UNI_RING=0 (separate prefill copy), or unset DSV41_MOE_PLACE"
        )
