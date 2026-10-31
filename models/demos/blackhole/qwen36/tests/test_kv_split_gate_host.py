# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the KV-pair split's context gate, slot-parity rule and DRAM budget (tt/kv_split.py; QWEN36_ATTN_KV_SPLIT=user):

* the gate tables (QWEN36_ATTN_KV_SPLIT_MIN_CTX / _MIN_CTX_DRAFT): parsing, defaults, the drafter value following an explicit
  verify value, the three modes (always = the K1 measurement setting, never, gated), the per-bucket decision at its boundary,
  the variant widths and the canonical-trace rule;
* the step context: the largest start + T over the live rows, a pure function of the (runner) start positions;
* slot parity: the owner of engine row u is pair rank u % 2 at every bucket width (the rule the own page tables, selections
  and scatters are built from), and the identity classes a replicated-prompt greedy run must satisfy;
* the DRAM budget: the pair weights of Qwen3.8-27B at TP8 BFP8 (453 MiB, measured 455.2 MiB), one prefix-cache slot (21.75 MiB),
  the slot equivalent and the opt-in slot trade.
No device. Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_kv_split_gate_host.py
"""

import pytest
import torch

from models.demos.blackhole.qwen36.tt import kv_split as ks

MiB = 2**20


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (
        "QWEN36_ATTN_KV_SPLIT_MIN_CTX",
        "QWEN36_ATTN_KV_SPLIT_MIN_CTX_DRAFT",
        "QWEN36_ATTN_KV_SPLIT_SLOT_TRADE",
        "QWEN36_ATTN_KV_SPLIT_UNPACKED",
    ):
        monkeypatch.delenv(k, raising=False)


# --------------------------------------------------------------------------- #
# gate tables
# --------------------------------------------------------------------------- #
def test_parse_min_ctx():
    d = {8: 100, 4: None}
    got = ks.parse_min_ctx("", d)
    assert got == d and got is not d  # a copy: the caller may not mutate the defaults
    assert ks.parse_min_ctx("off", d) == {}
    assert ks.parse_min_ctx("0", d) == {b: 0 for b in range(2, 33, 2)}
    assert ks.parse_min_ctx(" 12288 ", d) == {b: 12288 for b in range(2, 33, 2)}
    assert ks.parse_min_ctx("8:10240,4:24576,2:off", d) == {8: 10240, 4: 24576, 2: None}
    assert ks.parse_min_ctx("8:0", d) == {8: 0}  # buckets not listed never split
    for bad in ("3:100", "8:-1", "x:100", "8:abc", "-5", "nope", "34:100", "0:100"):
        with pytest.raises(ValueError):
            ks.parse_min_ctx(bad, d)


def test_default_gate_and_modes(monkeypatch):
    g = ks.KVSplitGate.from_env()
    assert g.tables == ks.GATE_DEFAULT and g.mode == "gated" and not g.canonical_split()
    # MIN_CTX=0 alone: the drafter follows it -> both uniform 0 -> the K1 always-split setting (split traces only)
    monkeypatch.setenv("QWEN36_ATTN_KV_SPLIT_MIN_CTX", "0")
    g = ks.KVSplitGate.from_env()
    assert g.mode == "always" and g.canonical_split() and g.variant_buckets("verify", (8, 4, 2, 1)) == ()
    assert not g.want(
        "verify", 8, 10**6
    )  # nothing is gated in "always" mode: the canonical traces are the split ones
    # an explicit drafter value no longer follows
    monkeypatch.setenv("QWEN36_ATTN_KV_SPLIT_MIN_CTX_DRAFT", "off")
    g = ks.KVSplitGate.from_env()
    assert g.mode == "gated" and g.tables["draft"] == {} and g.want("verify", 4, 0) and not g.want("draft", 8, 10**6)
    monkeypatch.setenv("QWEN36_ATTN_KV_SPLIT_MIN_CTX", "off")
    monkeypatch.delenv("QWEN36_ATTN_KV_SPLIT_MIN_CTX_DRAFT")
    g = ks.KVSplitGate.from_env()
    assert g.mode == "never" and g.variant_buckets("draft", (8, 4, 2)) == () and not g.canonical_split()
    assert "mode=never" in g.describe()


@pytest.mark.parametrize("part", ks.GATE_PARTS)
def test_want_at_the_threshold(part):
    g = ks.KVSplitGate.from_env()
    for b, t in ks.GATE_DEFAULT[part].items():
        if t is None:
            assert not g.want(part, b, 10**7), (part, b)
            continue
        assert not g.want(part, b, t - 1) and g.want(part, b, t) and g.want(part, b, t + 10**5), (part, b)
    assert not g.want(part, 1, 10**7)  # a 1-row bucket never splits (no pair of users)
    assert not g.want(part, 6, 10**7)  # a width without a threshold
    assert not g.want(part, 8, None)  # no step context (outside a step)


def test_variant_buckets_and_defaults_are_ordered():
    g = ks.KVSplitGate.from_env()
    assert g.variant_buckets("verify", (1, 2, 4, 8)) == (8, 4)
    assert g.variant_buckets("draft", (8,)) == (8,)
    # the break-even falls with more users per rank (each SDPA group has fewer cores to lose), and the drafter's decode SDPA
    # breaks even earlier than the packed verify SDPA at the same width
    v, d = ks.GATE_DEFAULT["verify"], ks.GATE_DEFAULT["draft"]
    assert v[8] < v[4] and d[8] < d[4] and d[8] < v[8] and d[4] < v[4]
    assert v[8] == 10240 and d[8] == 7168  # the measured 8-user break-evens (10.4K / 6.8K), rounded to 1K steps


# --------------------------------------------------------------------------- #
# step context
# --------------------------------------------------------------------------- #
def test_gate_ctx_is_the_longest_live_row():
    sp = torch.tensor([100, 9000, 0, 12000, 5, 0, 0, 0])
    act = torch.tensor([True, True, False, True, True, False, False, False])
    assert ks.kv_split_gate_ctx(sp, act, 4) == 12004
    act[3] = False  # the long row left: the context drops at once (no hysteresis: switching variants is free)
    assert ks.kv_split_gate_ctx(sp, act, 4) == 9004
    assert ks.kv_split_gate_ctx(sp, torch.zeros(8, dtype=torch.bool), 4) == 0
    assert ks.kv_split_gate_ctx([7, 3], None, 1) == 8
    # pure: the same inputs give the same context whatever was decided before
    assert ks.kv_split_gate_ctx(sp, act, 4) == ks.kv_split_gate_ctx(sp.clone(), act.clone(), 4)


# --------------------------------------------------------------------------- #
# slot parity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("b", [2, 4, 6, 8, 16])
def test_owner_is_the_row_parity_at_every_width(b):
    for u in range(b):
        r = ks.kv_split_owner(u)
        assert r == u % 2
        assert u in ks.own_users(b, r) and u not in ks.own_users(b, 1 - r)
    # a row keeps its owner across buckets (the own users of bucket b are a prefix of the B-row list)
    for r in (0, 1):
        assert ks.own_users(16, r)[: len(ks.own_users(b, r))] == ks.own_users(b, r)


def test_identity_classes():
    assert ks.kv_split_identity_classes(8, False) == [list(range(8))]
    assert ks.kv_split_identity_classes(8, True) == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert ks.kv_split_identity_classes(1, True) == [[0]]
    assert ks.kv_split_identity_classes(3, True) == [[0, 2], [1]]


def test_identity_classes_match_the_device_assignment():
    """Rows in one class run on the same device of every pair (the same KV replica) with the same SDPA shape; rows of
    different classes run on the two different devices. Built from the plan's own constants, not from the rule."""
    tp, n, T = 8, 8, 4
    sel = ks.sel_host(n, T, tp, 6).float()  # [tp, 6, Ro, R]
    rows_of = {d: set(torch.nonzero(sel[d, 0].sum(0)).reshape(-1).tolist()) for d in range(tp)}
    seen = []
    for cls in ks.kv_split_identity_classes(n, True):
        devs = {d for d in range(tp) if all(u * T + j in rows_of[d] for u in cls for j in range(T))}
        assert devs in ({0, 2, 4, 6}, {1, 3, 5, 7}), (cls, devs)
        assert not any(u * T in rows_of[d] for u in cls for d in set(range(tp)) - devs)  # no row on the other devices
        seen.append(devs)
    assert seen[0] != seen[1]


# --------------------------------------------------------------------------- #
# DRAM budget against the prefix-cache slots
# --------------------------------------------------------------------------- #
def test_pair_weight_bytes_match_the_measured_increase():
    # Qwen3.8-27B at TP8: dim 5120, 3 local q heads, 1 local KV head, head_dim 256; 16 attention layers + the MTP layer
    w = ks.kv_split_pair_weight_bytes(5120, 3, 1, 256, 17, "bfloat8_b")
    assert abs(w / MiB - 453.4) < 0.1
    measured = 455.2  # d21u20, flag on vs off after capture (K1 evidence): + the plan constants / own tables
    assert 0 < measured - w / MiB < 2.5
    assert ks.kv_split_pair_weight_bytes(5120, 3, 1, 256, 17, "bfloat4_b") < w * 0.55  # QWEN36_BFP4_ATTN


def test_prefix_slot_bytes_and_slot_equivalent():
    # one slot: 48 GDN layers x (fp32 state [1, 6, 128, 128] + bf16 conv carry [1, 3, 1280], 3 rows tile-padded to 32)
    slot = ks.prefix_slot_bytes(48, [1, 6, 128, 128], [1, 3, 1280])
    assert slot == 48 * (384 + 80) * 1024 and abs(slot / MiB - 21.75) < 1e-9
    w = ks.kv_split_pair_weight_bytes(5120, 3, 1, 256, 17)
    assert ks.kv_split_slot_equivalent(w, slot) == 21
    var = ks.kv_split_variant_out_bytes([32, 16], 5120, 151936, 8, topk=True)
    assert abs(var / MiB - 54.0) < 0.5  # V8s ~29.3 + V4s ~24.7 MiB of retained outputs
    # the formula against the measured DRAM increments of the row buckets V4 / V2 / V1 (+24.8 / +21.0 / +21.0 MiB)
    for rows, meas in ((16, 24.8), (8, 21.0), (4, 21.0)):
        assert abs(ks.kv_split_variant_out_bytes([rows], 5120, 151936, 8) / MiB - meas) < 2.5, rows
    # 256 production slots: unchanged by default, 256 - 24 with the trade (pair weights + the two verify variants)
    assert ks.kv_split_slot_equivalent(w + var, slot) == 24
    assert ks.kv_split_slots_after_trade(256, w + var, slot, trade=False) == 256
    assert ks.kv_split_slots_after_trade(256, w + var, slot, trade=True) == 232
    assert ks.kv_split_slots_after_trade(4, w, slot, trade=True) == 1  # never below one slot
    assert ks.kv_split_slots_after_trade(256, 0, slot, trade=True) == 256


def test_slot_trade_flag(monkeypatch):
    assert not ks.kv_split_slot_trade_enabled()
    monkeypatch.setenv("QWEN36_ATTN_KV_SPLIT_SLOT_TRADE", "1")
    assert ks.kv_split_slot_trade_enabled()


def test_prefix_cache_takes_the_trade(monkeypatch):
    """GdnPrefixStateCache with the env slot count: unchanged without the split / without the trade, minus the slot
    equivalent with both (fake model, fake allocator: no device)."""
    import types

    from models.demos.blackhole.qwen36.tt import prefix_cache as pc

    class _T:
        def __init__(self, shape, name):
            self.shape, self.name = shape, name

    class _Model:
        def __init__(self, budget):
            self._budget = budget
            dn = types.SimpleNamespace(mesh=types.SimpleNamespace(get_num_devices=lambda: 8))
            self._gdn_prefill_scratch = [(dn, _T([1, 6, 128, 128], "rec"), None, _T([1, 3, 1280], "carry"))] * 48

        def _ensure_gdn_prefill_scratch(self):
            pass

        def kv_split_dram_budget(self):
            return self._budget

    monkeypatch.setattr(pc, "alloc_zeros_like", lambda t, mesh: ("slot", t.name))
    monkeypatch.setattr(pc, "alloc_empty_like", lambda t, mesh: ("slot", t.name))
    monkeypatch.setenv("QWEN36_PREFIX_CACHE_SLOTS", "64")
    w = ks.kv_split_pair_weight_bytes(5120, 3, 1, 256, 17)
    budget = {"weights": w, "variants": 0, "total": w, "layers": 17}
    assert pc.GdnPrefixStateCache(_Model(None)).num_slots == 64  # split off
    assert pc.GdnPrefixStateCache(_Model(budget)).num_slots == 64  # split on, report only
    monkeypatch.setenv("QWEN36_ATTN_KV_SPLIT_SLOT_TRADE", "1")
    c = pc.GdnPrefixStateCache(_Model(budget))
    assert c.num_slots == 64 - 21 and len(c._slots) == 43
    assert pc.GdnPrefixStateCache(_Model(budget), num_slots=4).num_slots == 4  # an explicit count is never traded
