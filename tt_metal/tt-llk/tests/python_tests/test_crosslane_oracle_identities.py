# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cross-lane arsenal: HOST identity battery (lane FB).

Pure-host algebraic checks over the oracle (helpers/crosslane_oracle.py).
No sim, no compiler: this leg proves the oracle is self-consistent and that
every claimed algebraic identity of the consensus surface holds over the
honest 4x8 lane geometry.  The sim leg (test_crosslane_lane_tracer.py)
separately proves the pinned sim agrees with the oracle.

Battery items (each also a builder-acceptance obligation):
  - permutation-hood of every fixed shuffle pattern
  - rotate(k) . rotate(8-k) = id, ror1^8 = id
  - transp . transp = id (SFPTRANSP involution, both banks)
  - zip . unzip = id and unzip . zip = id
  - butterfly(k) self-inverse
  - copy4 nilpotency (4 applications zero the queue)
  - chained_copy4 telescope (L3 sourcing crosses rows, top row zero-fills)
  - swap-mod direction-mask table vs doc comments; mod9 = mod1 role swap;
    EXCHANGE_SRCB_SRCC flip = mod9; equal-value quirk pinned
  - indexed swap: companions follow the key decision (incl. quirk cases)
  - reduce: tree == serial for association-free ops; FP-add tree order is
    load-bearing (witness where tree != serial)
  - scans: exclusive = shift(inclusive); Hillis-Steele == serial prefix
  - bitonic networks (8 and 32, asc/desc, kv): sorted + permutation over
    adversarial stimuli under the sign-magnitude total order
  - top-k golden consistency with full sort
  - softmax_k: lane-mask formulation == tile-id-predicate formulation
    (the lane-EX dissolution claim, checked for every k)
  - genericity: all of the above on varied (non-sentinel) values too
"""

import pytest

from helpers import crosslane_oracle as co

LANES = co.LANES
ROWW = co.ROWW


def sentinels():
    return co.lane_id_sentinels()


def varied(seed=7):
    return co.varied_stimulus(0, seed)


# ---------------------------------------------------------------------------
# permutation-hood + rotate algebra
# ---------------------------------------------------------------------------


def test_ror1_is_row_local_permutation():
    perm = co.extract_permutation(lambda v: co.subvec_shflror1(v))
    assert co.is_full_permutation(perm)
    for l in range(LANES):
        src = perm[l]
        assert co.lane_row(src) == co.lane_row(l), "ror1 crossed a row"
        assert co.lane_col(src) == (co.lane_col(l) - 1) % ROWW


def test_rotate_inverse_identity_all_k():
    for k in range(1, ROWW):
        for stim in (sentinels(), varied(k)):
            v = co.subvec_rotr(co.subvec_rotr(stim, k), ROWW - k)
            assert v == stim, f"rot({k}) o rot({ROWW - k}) != id"


def test_ror1_eighth_power_identity():
    for stim in (sentinels(), varied(3)):
        v = list(stim)
        for _ in range(ROWW):
            v = co.subvec_shflror1(v)
        assert v == stim


def test_shr1_kills_lane0_and_shifts():
    perm = co.extract_permutation(lambda v: co.subvec_shflshr1(v))
    for l in range(LANES):
        if co.lane_col(l) == 0:
            assert perm[l] is None, "lane 0 of a row must be zero-filled"
        else:
            assert perm[l] == l - 1


def test_shr1_wh_arm_marks_lane0_unpredictable():
    out = co.subvec_shflshr1(sentinels(), arch="wh")
    for l in range(LANES):
        if co.lane_col(l) == 0:
            assert out[l] is None, "WH lane-0 must be flagged unpredictable"
        else:
            assert out[l] == sentinels()[l - 1]


# ---------------------------------------------------------------------------
# transpose / zip / butterfly
# ---------------------------------------------------------------------------


def test_transp8_involution_and_pattern():
    regs = [co.varied_stimulus(r, seed=11) for r in range(8)]
    once = co.sfptransp(regs)
    twice = co.sfptransp(once)
    assert twice == regs, "SFPTRANSP must be an involution"
    # exact movement: within each column, (reg base+i, subvec row j) <->
    # (reg base+j, subvec row i)
    for base in (0, 4):
        for i in range(4):
            for j in range(4):
                for c in range(8):
                    assert once[base + i][j * 8 + c] == \
                        regs[base + j][i * 8 + c]


def test_transp8_banks_independent():
    regs = [co.varied_stimulus(r, seed=13) for r in range(8)]
    once = co.sfptransp(regs)
    # bank 0 result depends only on regs[0:4]
    regs_mut = [list(r) for r in regs]
    for r in range(4, 8):
        regs_mut[r] = co.varied_stimulus(r, seed=99)
    once_mut = co.sfptransp(regs_mut)
    assert once_mut[:4] == once[:4]


def test_zip_unzip_identity():
    a, b = varied(1), varied(2)
    lo, hi = co.zip_rows(a, b)
    a2, b2 = co.unzip_rows(lo, hi)
    assert (a2, b2) == (a, b)
    lo2, hi2 = co.zip_rows(*co.unzip_rows(a, b))
    assert (lo2, hi2) == (a, b)


def test_butterfly_self_inverse():
    for k in range(1, ROWW):
        for stim in (sentinels(), varied(k + 20)):
            assert co.butterfly(co.butterfly(stim, k), k) == stim
        perm = co.extract_permutation(lambda v, k=k: co.butterfly(v, k))
        assert co.is_full_permutation(perm)
        for l in range(LANES):
            assert co.lane_col(perm[l]) == co.lane_col(l) ^ k
            assert co.lane_row(perm[l]) == co.lane_row(l)


def test_copy4_nilpotent_and_lanewise():
    regs = [varied(r) for r in range(4)]
    state = [list(r) for r in regs]
    for step in range(1, 5):
        state = list(co.shft2_copy4(*state))
        for idx in range(4):
            want_src = idx + step
            if want_src < 4:
                assert state[idx] == regs[want_src]
            else:
                assert state[idx] == [0] * LANES
    assert all(s == [0] * LANES for s in state), "copy4^4 must zero the queue"


def test_chained_copy4_crosses_rows():
    regs = [varied(r + 40) for r in range(4)]
    n0, n1, n2, n3 = co.shft2_chained_copy4(*regs)
    assert (n0, n1, n2) == (regs[1], regs[2], regs[3])
    for l in range(LANES):
        if l < 24:
            assert n3[l] == regs[0][l + 8], "L3 must source lane+8 of old L0"
        else:
            assert n3[l] == 0, "top row of L3 must zero-fill"


def test_ror1_and_copy4_l3_is_ror1():
    regs = [varied(r + 60) for r in range(4)]
    vc = varied(77)
    n0, n1, n2, n3 = co.shft2_ror1_and_copy4(*regs, vc)
    assert (n0, n1, n2) == (regs[1], regs[2], regs[3])
    assert n3 == co.subvec_shflror1(vc)


# ---------------------------------------------------------------------------
# swap family
# ---------------------------------------------------------------------------


def _fbits(x):
    return co.f32_to_bits(x)


def test_swap_mod1_minmax_all_lanes():
    a = [_fbits(float(l) - 15.5) for l in range(LANES)]
    b = [_fbits(14.0 - float(l)) for l in range(LANES)]
    nvc, nvd = co.sfpswap(a, b, 1)
    for l in range(LANES):
        lo, hi = sorted((a[l], b[l]), key=co._smkey)
        assert nvd[l] == lo and nvc[l] == hi


def test_swap_mod9_is_role_swapped_mod1():
    a, b = varied(5), varied(6)
    c1, d1 = co.sfpswap(a, b, 1)
    c9, d9 = co.sfpswap(a, b, 9)
    assert c9 == d1 and d9 == c1


def test_swap_rowgroup_masks_match_doc_comments():
    # doc: mod2 mask 0x0000ffff = first 16 lanes VD=min; mod5 0x000000ff =
    # first 8 lanes VD=min, etc.
    expects = {
        2: lambda l: l < 16,
        3: lambda l: l < 8 or (16 <= l < 24),
        4: lambda l: l < 8 or l >= 24,
        5: lambda l: l < 8,
        6: lambda l: 8 <= l < 16,
        7: lambda l: 16 <= l < 24,
        8: lambda l: l >= 24,
    }
    a, b = varied(8), varied(9)
    ref_min = [min(a[l], b[l], key=co._smkey) for l in range(LANES)]
    ref_max = [max(b[l], a[l], key=co._smkey) for l in range(LANES)]
    for mod, vd_gets_min in expects.items():
        nvc, nvd = co.sfpswap(a, b, mod)
        for l in range(LANES):
            if vd_gets_min(l):
                assert nvd[l] == ref_min[l] and nvc[l] == ref_max[l], \
                    f"mod{mod} lane{l}"
            else:
                assert nvd[l] == ref_max[l] and nvc[l] == ref_min[l], \
                    f"mod{mod} lane{l}"


def test_swap_equal_value_quirk_pinned():
    """SFPSWAP.md: max lanes swap equal positive values, min lanes swap
    equal negative values.  Value-invisible, but the QUIRK moves
    companions under ENABLE_DEST_INDEX -- pin it."""
    pos = [_fbits(2.5)] * LANES
    neg = [_fbits(-2.5)] * LANES
    ca = co.lane_id_sentinels(1)
    cb = co.lane_id_sentinels(2)
    # mod1: VD gets min in all lanes.  Equal POSITIVE values: ShouldSwap =
    # smaller(c,d)=False || (equal && negative)=False -> then inverted only
    # in max lanes (none for mod1's VDGetsMin=all-ones) -> no swap.
    _, _, ncc, ncd = co.sfpswap_indexed(pos, pos, ca, cb, 1)
    assert ncc == ca and ncd == cb
    # Equal NEGATIVE values in min lanes: swap fires.
    _, _, ncc, ncd = co.sfpswap_indexed(neg, neg, ca, cb, 1)
    assert ncc == cb and ncd == ca
    # mod9 (all lanes VD=max): equal positives now swap, negatives do not.
    _, _, ncc, ncd = co.sfpswap_indexed(pos, pos, ca, cb, 9)
    assert ncc == cb and ncd == ca
    _, _, ncc, ncd = co.sfpswap_indexed(neg, neg, ca, cb, 9)
    assert ncc == ca and ncd == cb


def test_exchange_srcb_srcc_flip_equals_mod9():
    a, b = varied(10), varied(11)
    st = co.LaneState()
    for l in range(LANES):
        st.lane_config[l] |= co.LC_EXCHANGE_SRCB_SRCC
    flipped = co.sfpswap(a, b, 1, state=st)
    assert flipped == co.sfpswap(a, b, 9)


def test_lane_masked_exchange_flip_per_column():
    """SFPCONFIG MOD1_IMM16_IS_LANE_MASK selects columns via bit
    (lane&7)*2; per-column EXCHANGE flip = topk_xl phase-7 mechanism.
    Imm16 0x4444 -> mask bits at positions 2,6,10,14 -> columns 1,3,5,7."""
    st = co.LaneState()
    l0 = [co.LC_EXCHANGE_SRCB_SRCC] * LANES  # value vector (broadcast source)
    co.sfpconfig_write(st, 15, l0, imm16=0x4444, mod1=8)
    flipped_cols = {1, 3, 5, 7}
    for lane in range(LANES):
        want = co.LC_EXCHANGE_SRCB_SRCC if co.lane_col(lane) in flipped_cols \
            else 0
        assert st.lane_config[lane] == want
    a, b = varied(12), varied(13)
    got = co.sfpswap(a, b, 1, state=st)
    plain = co.sfpswap(a, b, 1)
    inv = co.sfpswap(a, b, 9)
    for l in range(LANES):
        src = inv if co.lane_col(l) in flipped_cols else plain
        assert got[0][l] == src[0][l] and got[1][l] == src[1][l]


def test_sfpconfig_vertical_broadcast():
    l0 = varied(14)
    out = co.sfpconfig_broadcast(l0)
    for lane in range(LANES):
        assert out[lane] == l0[lane & 7]


def test_row_mask_disables_whole_rows_per_column():
    st = co.LaneState()
    # set ROW_MASK bit for subvector row 2 in columns 0..7
    for c in range(8):
        st.lane_config[c] |= (1 << 2) << co.LC_ROW_MASK_SHIFT
    en = st.enabled_vec()
    for lane in range(LANES):
        assert en[lane] == (co.lane_row(lane) != 2)


# ---------------------------------------------------------------------------
# reductions and scans
# ---------------------------------------------------------------------------


def test_reduce_tree_equals_serial_for_association_free_ops():
    v = varied(21)
    for op in ("add", "xor", "min", "max"):
        assert co.subvec_reduce_tree(v, op) == co.subvec_reduce_serial(v, op)


def test_reduce32_totals():
    v = varied(22)
    out = co.reduce32_tree(v, "add")
    total = sum(v) & co.M32
    assert out == [total] * LANES
    out = co.reduce32_tree(v, "max")
    best = max(v, key=co._smkey)
    assert out == [best] * LANES


def test_fadd_tree_order_is_load_bearing():
    """Witness that the pinned tree differs from serial for FP add -- the
    reassociation licence is real, so the contract must pin the tree."""
    vals = [1e30, -1e30, 1.0, -1.0, 3.0e-8, 5.0e7, -5.0e7, 2.0e-8]
    v = []
    for r in range(co.NROWS):
        v.extend(co.f32_to_bits(x) for x in vals)
    tree = co.subvec_reduce_tree(v, "fadd")
    serial = co.subvec_reduce_serial(v, "fadd")
    assert tree != serial, (
        "expected a tree-vs-serial FP difference witness; if this ever "
        "fails, pick a sharper witness -- do NOT weaken the contract")


def test_scan_exclusive_is_shift_of_inclusive():
    v = varied(23)
    incl = co.subvec_scan_incl(v, "add")
    excl = co.subvec_scan_excl(v, "add")
    for l in range(LANES):
        if co.lane_col(l) == 0:
            assert excl[l] == 0
        else:
            assert excl[l] == incl[l - 1]


def test_hillis_steele_equals_serial_prefix():
    v = varied(24)
    for op in ("add", "xor", "min", "max"):
        assert co.subvec_scan_incl(v, op) == co.serial_prefix(v, op)


def test_rowchain_scan_matches_direct_recurrence():
    rows = [varied(30 + r) for r in range(8)]
    got = co.rowchain_scan_incl(rows, "add")
    for l in range(LANES):
        acc = 0
        for i in range(8):
            acc = (acc + rows[i][l]) & co.M32
            assert got[i][l] == acc


# ---------------------------------------------------------------------------
# sort networks / top-k
# ---------------------------------------------------------------------------


ADVERSARIAL_8 = [
    [co.f32_to_bits(x) for x in (3.0, -1.0, 2.0, -7.5, 0.0, 9.0, -2.0, 4.0)],
    [co.f32_to_bits(0.0)] * 8,                      # all equal positive
    [co.f32_to_bits(-0.0)] * 8,                     # all equal negative zero
    [co.f32_to_bits(x) for x in (0.0, -0.0, 1.0, -1.0,
                                 float("inf"), float("-inf"), 2.0, -2.0)],
    [0x7FC00000, 0xFFC00000,                        # +NaN, -NaN
     co.f32_to_bits(1.0), co.f32_to_bits(-1.0),
     co.f32_to_bits(0.5), co.f32_to_bits(-0.5),
     co.f32_to_bits(3.0), co.f32_to_bits(-3.0)],
    [co.splitmix32(i) for i in range(8)],           # raw u32 bit soup
]


def _is_sorted_sm(vals, order):
    keys = [co._smkey(v) for v in vals]
    return keys == sorted(keys, reverse=(order == "desc"))


@pytest.mark.parametrize("order", ["asc", "desc"])
@pytest.mark.parametrize("n", [8, 32])
def test_bitonic_network_sorts_and_permutes(order, n):
    cases = [list(v) * (n // 8) for v in ADVERSARIAL_8]
    cases.append([co.splitmix32(100 + i) for i in range(n)])
    cases.append(list(range(n)))
    cases.append(list(range(n))[::-1])
    for vals in cases:
        out, trace = co.bitonic_sort_trace(vals, order)
        assert sorted(out) == sorted(vals), "not a permutation"
        assert _is_sorted_sm(out, order), f"not sorted ({order})"
        assert len(trace) == len(co.bitonic_network_stages(n))
        assert trace[-1] == out


@pytest.mark.parametrize("order", ["asc", "desc"])
def test_bitonic_kv_pairing_preserved(order):
    keys = [co.splitmix32(500 + i) for i in range(32)]
    # inject duplicates (quirk territory)
    keys[5] = keys[9] = co.f32_to_bits(2.5)
    keys[7] = keys[13] = co.f32_to_bits(-2.5)
    pay = [0xC0DE0000 | i for i in range(32)]
    ks, ps, _ = co.bitonic_sort_kv_trace(keys, pay, order)
    assert _is_sorted_sm(ks, order)
    assert sorted(zip(ks, ps)) == sorted(zip(keys, pay)), \
        "key/payload pairing broken"


def test_topk_matches_full_sort():
    keys = [co.splitmix32(900 + i) for i in range(32)]
    full_sorted, _ = co.bitonic_sort_trace(keys, "desc")
    for k in (1, 2, 4, 8, 32):
        vals, idxs = co.topk_select(keys, k, "desc")
        assert vals == full_sorted[:k]
        assert all(keys[i] == v for i, v in zip(idxs, vals))


# ---------------------------------------------------------------------------
# demand-core equivalences
# ---------------------------------------------------------------------------


def test_softmax_k_mask_equals_tileid_predicate():
    """Lane-EX dissolution claim: the SFPCONFIG lane-mask fold equals the
    computed vConstTileId-predicate fold -- for every k and on both
    sentinel and varied stimuli."""
    for k in range(1, 9):
        for stim_seed in (None, 41, 42):
            v = (sentinels() if stim_seed is None
                 else [co.f32_to_bits(float(co.splitmix32(stim_seed * 64 + l)
                                            % 1000) - 500.0)
                       for l in range(LANES)])
            a, b = co.softmax_k_masked_fold(v, k, "max")
            assert a == b, f"mask != tileid-predicate for k={k}"


def test_ema_contracts_differ_witness():
    """The fma and mul_add contracts are genuinely different -- keep both
    fixtures until the lowering pins one."""
    import random
    rng = random.Random(1234)
    alpha = co.f32_to_bits(0.7)
    x_rows = [[co.f32_to_bits(rng.uniform(-1e3, 1e3)) for _ in range(LANES)]
              for _ in range(8)]
    y0 = [co.f32_to_bits(rng.uniform(-1e3, 1e3)) for _ in range(LANES)]
    outs = co.ema_rowchain(x_rows, alpha, y0)
    assert outs["fma"] != outs["mul_add"], (
        "expected an fma-vs-mul_add witness on random stimuli; sharpen the "
        "stimulus rather than dropping a contract")


def test_cumsum_int_exact():
    rows = [varied(70 + r) for r in range(8)]
    got = co.cumsum_rowchain(rows, "int")
    assert got == co.rowchain_scan_incl(rows, "add")


def test_fraction_fma_rounding_sane():
    from fractions import Fraction
    # exact case
    assert co.fraction_to_f32(Fraction(3, 2)) == 1.5
    # rounding case: 1 + 2^-24 rounds to 1.0 (even), 1 + 3*2^-25 rounds up
    assert co.fraction_to_f32(Fraction(1) + Fraction(1, 2**24)) == 1.0
    got = co.fraction_to_f32(Fraction(1) + Fraction(3, 2**25))
    assert got == co.f32_round(1.0 + 2.0**-23)


def test_swap_tie_modes_divergence_witness():
    """DOC-VS-SIM SFPSWAP tie divergence (lane FB finding, 2026-08-21):
    the pinned sim (craq-sim 9f324140, sfpswap_vd_gets_c) decides ties
    as min-lanes:no-swap / max-lanes:swap; SFPSWAP.md keys ties on SIGN.
    Invisible for plain min/max values, VISIBLE via companions under
    ENABLE_DEST_INDEX.  This witness pins both facts; silicon
    adjudication pending -- fixtures must not depend on tie companion
    movement."""
    pos = [co.f32_to_bits(2.5)] * LANES
    neg = [co.f32_to_bits(-2.5)] * LANES
    ca = co.lane_id_sentinels(1)
    cb = co.lane_id_sentinels(2)
    for keys in (pos, neg):
        # plain values: identical under both tie models
        assert co.sfpswap(keys, keys, 1, tie="doc")[0] == \
            co.sfpswap(keys, keys, 1, tie="sim")[0]
    # companions: equal NEGATIVES diverge (doc swaps in min lanes, sim not)
    d = co.sfpswap_indexed(neg, neg, ca, cb, 1, tie="doc")
    s = co.sfpswap_indexed(neg, neg, ca, cb, 1, tie="sim")
    assert d[2] == cb and s[2] == ca, "tie-divergence witness lost"
    # equal POSITIVES under mod 9 (max lanes): doc swaps, sim swaps too --
    # but under mod 1 max-lane-free the sim swaps NOTHING while doc also
    # doesn't; the mod-9 equal-negative case diverges the other way:
    d9 = co.sfpswap_indexed(neg, neg, ca, cb, 9, tie="doc")
    s9 = co.sfpswap_indexed(neg, neg, ca, cb, 9, tie="sim")
    assert d9[2] == ca and s9[2] == cb, "mod9 tie-divergence witness lost"
