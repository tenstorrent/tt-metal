# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-exact host model of kernels/spec_accept_sampled_fused.cpp (QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED): the fused SAMPLED
accept + drafter-input build of stage S2 (spec_fused_sampled.FusedSampledDevice.body) as one generic_op kernel.

No ttnn, no torch: numpy float32 for the IEEE-754 operations the kernel makes with libgcc soft-float calls (__divsf3,
__subsf3, __mulsf3, __addsf3 are correctly rounded, round-to-nearest-even, gradual underflow; so are numpy's float32
+ - * /), and Python integers for everything the kernel does in integer arithmetic (the exp, the exact sums, the merge,
the walk, the output words). The device test checks the kernel against this model word for word; the host tests check
this model against spec_fused_sampled.fused_sampled_reference (the S2 algorithm the op graph and the engine's audit
implement) and its losslessness (chi-square against the exact target distribution).

The algorithm per engine row r = u*T + j of a SAMPLED user (temperature > 0 path; greedy / inactive users commit their
argmax ids exactly as S1), in the reference's own float32 order:
  1. global top-M of the union: a TP-way merge of the per-shard blocks (each already (value desc, local id asc)), ties to
     the lower shard: the stable descending sort of union_topk_stable, so the same candidates in the same order;
  2. v_i = vals_i / temp (IEEE division; the kernel's fdiv_r: a per-divisor reciprocal, one multiply and one exact
     correction, correctly rounded including denormal results), kth = v_{k-1}, keep = v >= kth (ties kept), overflow_row = v_{M-1} >= kth and
     kth > -inf; the merge stops at the first v_i < kth (v is non-increasing along the merge);
  3. e_i = exp(v_i - v_0): an integer exp (Q7.56 argument, 64-entry 2^(i/64) table, degree-6 polynomial, round to
     nearest even), within ~2^-55 of the exact value before rounding, so it is the correctly rounded float32 except within
     2^-55 of a rounding midpoint (torch's float32 exp is off by one ulp on ~1% of inputs); results below 2^-126 are 0;
  4. s = sum(e) rounded once from an exact 64-bit fixed-point sum (torch's float32 sum is a cascade, so it is within an
     ulp either way); p0_i = e_i / s (IEEE); cum_i = the reverse cumsum of p0, each rounded once from an exact fixed-point
     suffix sum (torch's CPU cumsum accumulates float32 in double and rounds each output: the same value unless a double
     rounding hits, ~2^-29); drop_i = cum_i <= 1 - top_p for i >= 1 (skipped when 1 - top_p <= 0, where it drops only
     zero mass);
  5. s2 = sum of the kept e (exact, rounded once), prob_i = e_i / s2 (IEEE); p_d = prob of the draft (0 when it is not a
     kept candidate); accept = p_d > 0 and p_d >= u_accept;
  6. pick (correction row j < nv: the draft's weight removed, uniform u_corr; bonus row j == nv: uniform u_bonus):
     cdf_i = prefix sums of the weights (exact, rounded once, as torch's double-accumulated cumsum), thr = u * cdf_last
     (IEEE), pick = #{i : cdf_i <= thr} over all M columns, clamped to the last positive weight.
The walk: m = first rejected column (or nv), committed = drafts[0..m-1] + the pick of row m, overflow = any examined row
(columns 0..m) overflowed. Then the S1 drafter tail (pack, reseed page table, result rows), as spec_accept_fused.cpp.
"""

from typing import NamedTuple

import numpy as np

# --------------------------------------------------------------------------------------------------------------------- #
# constants (kernels/spec_accept_sampled_fused.cpp holds the same literals; tests/test_spec_sampled_kernel_host.py checks
# them against 80-digit Decimal values)
# --------------------------------------------------------------------------------------------------------------------- #
L2E = 0x5C551D94AE0BF85E  # round(log2(e) * 2^62)
LN2_64 = 0xB17217F7D1CF79AC  # round(ln(2) * 2^64)
POLY = (  # round(2^62 / k!), k = 0..6
    0x4000000000000000,
    0x4000000000000000,
    0x2000000000000000,
    0x0AAAAAAAAAAAAAAB,
    0x02AAAAAAAAAAAAAB,
    0x0088888888888889,
    0x0016C16C16C16C17,
)
EXP2_TAB = (  # round(2^(i/64) * 2^62), i = 0..63
    0x4000000000000000, 0x40B268F9DE0183BA, 0x4166C34C5615D0EC, 0x421D1461D66F2023,
    0x42D561B3E6243D8A, 0x438FB0CB4F468808, 0x444C0740496D4294, 0x450A6ABAA4B77ECD,
    0x45CAE0F1F545EB73, 0x468D6FADBF2DD4F3, 0x47521CC5A2E6A9E0, 0x4818EE218A3358EE,
    0x48E1E9B9D588E19B, 0x49AD159789F37496, 0x4A7A77D47F7B84B1, 0x4B4A169B900C2D00,
    0x4C1BF828C6DC54B8, 0x4CF022C9905BFD32, 0x4DC69CDCEAA72A9C, 0x4E9F6CD3967FDBA8,
    0x4F7A993048D088D7, 0x50582887DCB8A7E1, 0x513821818624B40C, 0x521A8AD704F3404F,
    0x52FF6B54D8A89C75, 0x53E6C9DA74B29AB5, 0x54D0AD5A753E077C, 0x55BD1CDAD49F699C,
    0x56AC1F752150A563, 0x579DBC56B48521BA, 0x5891FAC0E95612C8, 0x5988E20954889245,
    0x5A827999FCEF3242, 0x5B7EC8F19468BBC9, 0x5C7DD7A3B17DCF75, 0x5D7FAD59099F22FE,
    0x5E8451CFAC061B5F, 0x5F8BCCDB3D398841, 0x6096266533384A2B, 0x61A3666D124BB204,
    0x62B39508AA836D6F, 0x63C6BA6455DCD8AE, 0x64DCDEC3371793D1, 0x65F60A7F79393E2E,
    0x6712460A8FC24072, 0x683199ED779592CA, 0x69540EC8F895722D, 0x6A79AD55E7F6FD10,
    0x6BA27E656B4EB57A, 0x6CCE8AE13C57EBDB, 0x6DFDDBCBED791BAB, 0x6F307A412F074892,
    0x70666F76154A7089, 0x719FC4B95F452D29, 0x72DC8373BE41A454, 0x741CB5281E25EE34,
    0x75606373EE921C97, 0x76A7980F6CCA15C2, 0x77F25CCDEE6D7AE6, 0x7940BB9E2CFFD89D,
    0x7A92BE8A92436616, 0x7BE86FB985689DDC, 0x7D41D96DB915019D, 0x7E9F06067A4360BA,
)  # fmt: skip

E_FRAC = 56  # fixed-point fraction bits of the exp sums (e <= 1, at most 64 terms: < 2^7)
P_FRAC = 62  # fixed-point fraction bits of the probability sums (each <= 1, total < 2)
F_ONE, F_NAN, F_INF, F_NEG_INF = 0x3F800000, 0x7FC00000, 0x7F800000, 0xFF800000

# spec_fused.AUX_* / CONST_* rows and spec_fused_sampled.SAMP_PARAM_ROWS (duplicated: this module imports nothing)
AUX_NUM_VALID, AUX_ACTIVE, AUX_DECLINED, AUX_KEEP, AUX_P0, AUX_CONSTS, AUX_DELTA = range(7)
CONST_SCRATCH, CONST_ROPE_ROWS = 0, 2
SAMP_TEMP, SAMP_TOPK, SAMP_TOPP, SAMP_GREEDY, SAMP_PARAM_ROWS = 0, 1, 2, 3, 4


# --------------------------------------------------------------------------------------------------------------------- #
# float32 words (Python int bit patterns) and the IEEE operations
# --------------------------------------------------------------------------------------------------------------------- #
def f32(b):
    return np.array([b & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0]


def bits(x):
    return int(np.array([x], dtype=np.float32).view(np.uint32)[0])


def is_nan(b):
    return (b & 0x7FFFFFFF) > 0x7F800000


def _ieee(op, a, b):
    with np.errstate(all="ignore"):
        r = bits(op(f32(a), f32(b)))
    return F_NAN if is_nan(r) else r  # RISC-V soft-float returns the canonical quiet NaN


def fdiv(a, b):
    return _ieee(np.divide, a, b)


def fsub(a, b):
    return _ieee(np.subtract, a, b)


def fmul(a, b):
    return _ieee(np.multiply, a, b)


def fadd(a, b):
    return _ieee(np.add, a, b)


def rcp62(b):
    """floor(2^62 / significand(b)) of a positive normal float32 word (the kernel's per-divisor reciprocal); 0 otherwise."""
    ex = (b >> 23) & 0xFF
    if b & 0x80000000 or ex == 0 or ex == 0xFF:
        return 0
    return (1 << 62) // ((b & 0x7FFFFF) | 0x800000)


def fdiv_r(a, b, r):
    """a / b correctly rounded (IEEE binary32, round to nearest even, gradual underflow) from b's reciprocal r = rcp62(b):
    the kernel's fdiv_r. Fast path for a finite normal numerator and a positive normal divisor (one multiply, one
    correction step); any other operand goes to the soft-float division. Equal to fdiv on every input (host-tested)."""
    ea, eb = (a >> 23) & 0xFF, (b >> 23) & 0xFF
    if r == 0 or ea == 0 or ea == 0xFF:
        return fdiv(a, b)
    sign = a & 0x80000000
    ma, mb = (a & 0x7FFFFF) | 0x800000, (b & 0x7FFFFF) | 0x800000
    q = (ma * r) >> 36  # floor(ma * 2^26 / mb) or one less
    rem = (ma << 26) - q * mb
    if rem >= mb:
        q, rem = q + 1, rem - mb
    if q >= 1 << 26:
        e, sh = ea - eb + 127, 3
    else:
        e, sh = ea - eb + 126, 2
    if e >= 1:
        mant, lost, half = q >> sh, q & ((1 << sh) - 1), 1 << (sh - 1)
        if lost > half or (lost == half and (rem or mant & 1)):
            mant += 1
        if mant == 1 << 24:
            mant, e = 1 << 23, e + 1
        if e >= 0xFF:
            return sign | F_INF
        return sign | (e << 23) | (mant & 0x7FFFFF)
    ts = sh + 1 - e  # a denormal result: d * 2^-149
    if ts > 40:
        return sign
    d, lost, half = q >> ts, q & ((1 << ts) - 1), 1 << (ts - 1)
    if lost > half or (lost == half and (rem or d & 1)):
        d += 1
    return sign | d


def fkey(b):
    """Total-order key of a non-NaN float32 word (-0 == +0)."""
    if (b & 0x7FFFFFFF) == 0:
        b = 0
    return (~b & 0xFFFFFFFF) if b & 0x80000000 else (b | 0x80000000)


def fle(a, b):
    return not is_nan(a) and not is_nan(b) and fkey(a) <= fkey(b)


def fge(a, b):
    return fle(b, a)


def fgt(a, b):
    return not is_nan(a) and not is_nan(b) and fkey(a) > fkey(b)


def bf16_key(h):
    """Order key of a bf16 word: the order of its float value (-0 == +0; NaN above +inf, as torch.sort ranks NaN)."""
    h &= 0xFFFF
    if (h & 0x7FFF) == 0:
        h = 0
    if (h & 0x7F80) == 0x7F80 and (h & 0x007F):
        return 0x10000
    return (h ^ 0xFFFF) if h & 0x8000 else (h | 0x8000)


def f2fix(b, frac):
    """floor(value * 2^frac) of a non-negative finite float32 word (zero and denormals -> 0)."""
    ex = (b >> 23) & 0xFF
    if ex == 0:
        return 0
    m = (b & 0x7FFFFF) | 0x800000
    sh = ex - 150 + frac
    if sh >= 0:
        return m << sh
    return 0 if sh <= -24 else m >> -sh


def fix2f(v, frac):
    """The float32 word nearest (ties to even) to v * 2^-frac, v a non-negative integer below 2^64."""
    if v == 0:
        return 0
    nb = v.bit_length()
    e = nb - 1 - frac
    if nb > 24:
        sh = nb - 24
        q = v >> sh
        rem, half = v & ((1 << sh) - 1), 1 << (sh - 1)
        if rem > half or (rem == half and q & 1):
            q += 1
        if q == 1 << 24:
            q, e = 1 << 23, e + 1
    else:
        q = v << (24 - nb)
    return ((e + 127) << 23) | (q & 0x7FFFFF)


def f2int_trunc(b):
    """int(value) toward zero of a float32 word, 0 for negatives / NaN, capped at 2^30 (the top_k word)."""
    if b & 0x80000000 or is_nan(b):
        return 0
    ex = (b >> 23) & 0xFF
    if ex < 127:
        return 0
    if ex >= 127 + 30:
        return 1 << 30
    m = (b & 0x7FFFFF) | 0x800000
    sh = ex - 150
    return m << sh if sh >= 0 else m >> -sh


def _mulhi(a, b, s):
    """(a * b) >> s; the kernel's 64x64 -> 128-bit product (every use keeps the product below 2^(64 + s))."""
    p = a * b
    assert p < 1 << (64 + s), (a, b, s)
    return p >> s


def exp_f32(xb):
    """The kernel's float32 exp of the float32 word xb (kernel: exp_bits)."""
    sgn, ex, mx = xb >> 31, (xb >> 23) & 0xFF, xb & 0x7FFFFF
    if ex == 0xFF:
        return F_NAN if mx else (0 if sgn else F_INF)
    if ex == 0:
        return F_ONE  # +-0 and denormals: exp rounds to 1.0f
    if ex >= 127 + 7:
        return 0 if sgn else F_INF  # |x| >= 128
    m = mx | 0x800000
    sh = ex - 127 + 33  # |x| * 2^56 = m * 2^sh
    X = m << sh if sh >= 0 else (0 if sh <= -24 else m >> -sh)
    if X == 0:
        return F_ONE
    Y = _mulhi(X, L2E, 62)  # |x| * log2(e) * 2^56
    FM = (1 << 56) - 1
    if sgn:
        ip, fr = Y >> 56, Y & FM
        n, f = (-ip, 0) if fr == 0 else (-ip - 1, (1 << 56) - fr)
    else:
        n, f = Y >> 56, Y & FM
    if n < -126:
        return 0
    if n > 127:
        return F_INF
    i, g = f >> 50, f & ((1 << 50) - 1)
    t = _mulhi(g, LN2_64, 58)  # g * ln2, Q0.62
    acc = POLY[6]
    for c in (POLY[5], POLY[4], POLY[3], POLY[2], POLY[1], POLY[0]):
        acc = c + _mulhi(t, acc, 62)
    mant = _mulhi(EXP2_TAB[i], acc, 62)  # 2^f, Q1.62 in [2^62, 2^63)
    assert (1 << 62) <= mant < (1 << 63), mant
    q = mant >> 39
    rem, half = mant & ((1 << 39) - 1), 1 << 38
    if rem > half or (rem == half and q & 1):
        q += 1
    if q == 1 << 24:
        q, n = 1 << 23, n + 1
        if n > 127:
            return F_INF
    return ((n + 127) << 23) | (q & 0x7FFFFF)


# --------------------------------------------------------------------------------------------------------------------- #
# one engine row
# --------------------------------------------------------------------------------------------------------------------- #
class _Merge:
    """The TP-way merge of the union row (block d = columns d*M .. d*M+M-1, each (value desc, local id asc)): the next
    candidate of the global (value desc, union position asc) order on every call (kernel: merge_next)."""

    def __init__(self, vals, ids, TP, M, per_shard):
        self.vals, self.ids, self.TP, self.M, self.per_shard = vals, ids, TP, M, per_shard
        self.head = [0] * TP
        self.key = [bf16_key(int(vals[d * M])) for d in range(TP)]

    def next(self):
        best, bd = -1, 0
        for d in range(self.TP):
            k = self.key[d]
            if k > best:
                best, bd = k, d
        c = bd * self.M + self.head[bd]
        self.head[bd] += 1
        self.key[bd] = bf16_key(int(self.vals[c + 1])) if self.head[bd] < self.M else -1
        return int(self.vals[c]) & 0xFFFF, (int(self.ids[c]) + bd * self.per_shard) & 0xFFFFFFFF


class RowState(NamedTuple):
    n_keep: int  # kept candidates (top-k with ties)
    ovf: int  # overflow_row (the tie at the k-th value reached the M-th candidate)
    n2: int  # candidates left after top-p (a prefix of the kept ones; >= 1)
    gids: list  # global ids of the merged candidates 0 .. len-1 (>= n_keep)
    prob: list  # float32 words of prob, i < n2
    p_d: int  # float32 word of the draft's probability (0 when there is no draft)
    cdf: list  # float32 words of the pick row's cdf, i < n2 (draft removed on a correction row)
    last: int  # last positive weight of the pick (M - 1 when none)
    merge: object  # the merge, to fetch a degenerate pick past n_keep


def row_state(vals, ids, *, TP, M, per_shard, temp_b, k, omp_b, draft, corr):
    """The uniform-independent part of one sampled row. vals: TP*M bf16 words; ids: TP*M local ids; temp_b / omp_b:
    float32 words of the effective temperature and of 1 - top_p; k: top_k in [1, M-1]; draft: the draft id of the row
    (None on a bonus row past the drafts); corr: the row's pick removes the draft (j < nv)."""
    mg = _Merge(vals, ids, TP, M, per_shard)
    gids, v = [], []
    kth = None
    r_t = rcp62(temp_b)
    while len(v) < M:
        h, gid = mg.next()
        vi = fdiv_r((h << 16) & 0xFFFFFFFF, temp_b, r_t)
        if len(v) >= k and not fge(vi, kth):
            gids.append(gid)  # merged but not kept (the merge has advanced past it)
            break
        gids.append(gid)
        v.append(vi)
        if len(v) == k:
            kth = vi
    n_keep = len(v)
    ovf = int(n_keep == M and fgt(kth, F_NEG_INF))
    e = [exp_f32(fsub(v[i], v[0])) for i in range(n_keep)]
    s = fix2f(sum(f2fix(x, E_FRAC) for x in e), E_FRAC)
    n2 = n_keep
    if fgt(omp_b, 0):  # top-p (1 - top_p > 0): drop the tail suffix whose reverse cumsum is <= 1 - top_p, never column 0
        r_s = rcp62(s)
        acc = 0
        for i in range(n_keep - 1, 0, -1):
            acc += f2fix(fdiv_r(e[i], s, r_s), P_FRAC)
            if not fle(fix2f(acc, P_FRAC), omp_b):
                break
            n2 = i
    s2 = s if n2 == n_keep else fix2f(sum(f2fix(e[i], E_FRAC) for i in range(n2)), E_FRAC)
    r_s2 = rcp62(s2)
    prob = [fdiv_r(e[i], s2, r_s2) for i in range(n2)]
    p_d, hit = 0, []
    if draft is not None:
        hit = [i for i in range(n2) if gids[i] == draft]
        for n, i in enumerate(hit):
            p_d = prob[i] if n == 0 else fadd(p_d, prob[i])
    excl = set(hit) if corr else set()
    acc, cdf, last = 0, [], M - 1
    found = False
    for i in range(n2):
        w = 0 if i in excl else prob[i]
        acc += f2fix(w, P_FRAC)
        cdf.append(fix2f(acc, P_FRAC))
        if w:
            last, found = i, True
    if not found:
        last = M - 1
    return RowState(n_keep, ovf, n2, gids, prob, p_d, cdf, last, mg)


def row_pick(st, u_b, M):
    """Column index of the pick of a row state with the uniform word u_b (kernel: the pick block of row_work)."""
    total = st.cdf[st.n2 - 1]
    thr = fmul(u_b, total)
    pick = sum(1 for c in st.cdf if fle(c, thr)) + (M - st.n2) * int(fle(total, thr))
    return min(pick, st.last)


def row_pick_gid(st, pick):
    """Global id of merged candidate ``pick`` (merges further when a degenerate pick lies past the kept ones)."""
    while len(st.gids) <= pick:
        st.gids.append(st.merge.next()[1])
    return st.gids[pick]


def row_accept(st, u_b):
    return int(fgt(st.p_d, 0) and fge(st.p_d, u_b))


# --------------------------------------------------------------------------------------------------------------------- #
# the whole kernel: (pack, rs_pt, result) words
# --------------------------------------------------------------------------------------------------------------------- #
def _s32(w):
    w = int(w) & 0xFFFFFFFF
    return w - (1 << 32) if w >= 1 << 31 else w


def _u32_sat(v):
    return 0 if v < 0 else v


def _rope_row(x, delta, rows_m1):
    v = x + delta
    v = 0 if v < 0 else v
    return v if v < rows_m1 else rows_m1


class UserParams(NamedTuple):
    greedy: bool
    nv: int
    temp_b: int
    k: int
    omp_b: int
    uni: list  # K + 2 float32 words: bonus, accepts 0..K-1, correction


def user_params(aux, samp, u, K, M):
    """Per-user effective parameters as the kernel derives them (aux int32 rows, samp float32 words)."""
    g = (samp[SAMP_GREEDY][u] & 0x7FFFFFFF) != 0 or aux[AUX_ACTIVE][u] == 0
    nv = min(max(aux[AUX_NUM_VALID][u], 0), K)
    t = samp[SAMP_TEMP][u]
    use_one = not is_nan(t) and ((t & 0x7FFFFFFF) == 0 or t & 0x80000000)
    temp_b = F_ONE if use_one else t
    k = min(max(f2int_trunc(samp[SAMP_TOPK][u]), 1), M - 1)
    omp_b = fsub(F_ONE, samp[SAMP_TOPP][u])
    uni = [samp[SAMP_PARAM_ROWS + u][i] for i in range(K + 2)]
    return UserParams(bool(g), nv, temp_b, k, omp_b, uni)


def _words(x, dtype=np.uint32):
    a = np.asarray(x)
    if a.dtype == np.float32:
        a = a.view(np.uint32)
    return a.astype(np.int64) & 0xFFFFFFFF


def sampled_accept_twin(union_vals, union_ids, tok, ids, aux, samp, pt_rows, *, B, K, M, TP, per_shard, W, nb):
    """The words kernels/spec_accept_sampled_fused.cpp writes, from host copies of its inputs:
      union_vals [R, TP*M] bf16 words (uint16), union_ids [R, TP*M] local ids, tok [R] / ids [>= R] uint32 words,
      aux [7, W] int32 words, samp [4 + B, W] float32 (array or words), pt_rows [R, nb] int32.
    Returns (pack [2K+5, W], rs_pt [R, nb], result [3, W]) as int64 arrays of uint32 words, every word of every page
    (zero padding included)."""
    T, R = K + 1, B * (K + 1)
    uv = np.asarray(union_vals).astype(np.int64) & 0xFFFF
    ui = _words(union_ids)
    tk = [int(x) for x in _words(tok).reshape(-1)[:R]]
    idw = [int(x) for x in _words(ids).reshape(-1)[:R]]
    a = [[_s32(x) for x in row] for row in _words(aux).tolist()]
    sp = [[int(x) for x in row] for row in _words(samp).tolist()]
    scratch, rows_m1 = a[AUX_CONSTS][CONST_SCRATCH], a[AUX_CONSTS][CONST_ROPE_ROWS] - 1
    m_of, tokm_of, ovf_of, real_of, greedy_of = [], [], [], [], []
    for u in range(B):
        p = user_params(a, sp, u, K, M)
        r0 = u * T
        if p.greedy:
            m = 0
            while m < K and m < a[AUX_NUM_VALID][u] and idw[r0 + m] == tk[r0 + m + 1]:
                m += 1
            tok_m, ovf = idw[r0 + m], 0
        else:
            acc, pick, ovf_r = [], [], []
            for j in range(p.nv + 1):
                corr = j < p.nv
                st = row_state(
                    uv[r0 + j],
                    ui[r0 + j],
                    TP=TP,
                    M=M,
                    per_shard=per_shard,
                    temp_b=p.temp_b,
                    k=p.k,
                    omp_b=p.omp_b,
                    draft=tk[r0 + j + 1] if corr else None,
                    corr=corr,
                )
                acc.append(row_accept(st, p.uni[1 + j]) if corr else 0)
                pick.append(row_pick_gid(st, row_pick(st, p.uni[K + 1] if corr else p.uni[0], M)))
                ovf_r.append(st.ovf)
            m = 0
            while m < p.nv and acc[m]:
                m += 1
            tok_m = pick[m]
            ovf = int(any(ovf_r[: m + 1]))
        kv = m + a[AUX_DECLINED][u] * a[AUX_KEEP][u]
        m_of.append(m)
        tokm_of.append(tok_m)
        ovf_of.append(ovf)
        real_of.append(0 if kv <= 0 else (T if kv >= T else kv))
        greedy_of.append(p.greedy)
    pk = [[0] * W for _ in range(2 * K + 5)]
    rr = [[0] * W for _ in range(3)]
    G2, G3, G4, G5 = K + 1, K + 3, K + 4, K + 5
    for u in range(B):
        m, ndec, p0, delta = m_of[u], 1 - a[AUX_DECLINED][u], a[AUX_P0][u], a[AUX_DELTA][u]
        r0 = u * T
        comm = [
            (idw[r0 + j] if greedy_of[u] else tk[r0 + j + 1]) if j < m else (tokm_of[u] if j == m else -1)
            for j in range(T)
        ]
        p = (p0 + m) * ndec
        pk[0][u] = _u32_sat(comm[m] * ndec)
        for k in range(K):
            pk[1 + k][u] = _u32_sat(p + k)
            pk[G5 + k][u] = _u32_sat(_rope_row(p + k, delta, rows_m1))
        pk[G3][u] = r0 + m
        for j in range(T):
            r = r0 + j
            real = 1 if j < real_of[u] else 0
            rpos = (p0 + j) * real
            pk[G2][r] = _u32_sat(comm[j] * real)
            pk[G2 + 1][r] = _u32_sat(rpos)
            pk[G4][r] = _u32_sat(_rope_row(rpos, delta, rows_m1))
            rr[0][r] = comm[j]
        rr[1][u] = m + 1
        rr[2][u] = ovf_of[u]
    pt = _words(pt_rows)
    rs = np.stack(
        [pt[r] if r % T < real_of[r // T] else np.full((nb,), scratch & 0xFFFFFFFF, dtype=np.int64) for r in range(R)]
    )
    w = lambda x: np.asarray(x, dtype=np.int64) & 0xFFFFFFFF
    return w(pk), w(rs), w(rr)


# --------------------------------------------------------------------------------------------------------------------- #
# the walk alone on fixed rows (chi-square tests: the row states do not depend on the uniforms)
# --------------------------------------------------------------------------------------------------------------------- #
def user_row_states(vals_rows, ids_rows, drafts, nv, *, K, TP, M, per_shard, temp_b, k, omp_b):
    """The row states of one sampled user's rows 0..nv (vals_rows / ids_rows [T, TP*M]; drafts [K])."""
    out = []
    for j in range(nv + 1):
        corr = j < nv
        out.append(
            row_state(
                vals_rows[j],
                ids_rows[j],
                TP=TP,
                M=M,
                per_shard=per_shard,
                temp_b=temp_b,
                k=k,
                omp_b=omp_b,
                draft=int(drafts[j]) if corr else None,
                corr=corr,
            )
        )
    return out


def user_walk(states, uni, nv, K, M, drafts):
    """(count, committed list, overflow) of one sampled user from its row states and its K + 2 uniform words."""
    m = 0
    while m < nv and row_accept(states[m], uni[1 + m]):
        m += 1
    st = states[m]
    tok = row_pick_gid(st, row_pick(st, uni[K + 1] if m < nv else uni[0], M))
    return m + 1, [int(d) for d in drafts[:m]] + [tok], int(any(s.ovf for s in states[: m + 1]))
