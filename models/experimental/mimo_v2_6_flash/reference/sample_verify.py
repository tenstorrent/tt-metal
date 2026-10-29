# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Speculative acceptance on device (tt/verify_sampler.py, tt/kernels/sample_verify.cpp): the host layouts and an
exact integer mirror of the kernel (no ttnn), shared by the vLLM adapter and the host tests.

A verify pass runs a request's last committed token and its drafts in consecutive decode slots
(reference/spec_decode.py): slot s0 + j holds candidate column j, so its logits are the target's distribution of the
token after column j. The DeviceSampler's steps 1-2 (SampleScan, SampleChipMerge, the row all_gather) give every slot
its chips' sorted top-K lists and normalizers. SampleVerify then walks each request ("group") of the pass the way the
plugin's host accept walk does (vllm_tt_plugin.spec_accept, point-mass drafts, draft_probs None), with uniforms the
plugin drew on the host from the request's own generator (``spec_sampling["uniforms"]``: column 0 the bonus, 1..K the
accept draws, K + 1 the correction):

- a greedy group accepts draft j while it equals slot s0 + j's greedy token (GreedyHead's, the lowest id among tied
  maxima: the verify's argmax ids) and commits that argmax chain, exactly as the argmax_ids walk does;
- a sampled group accepts draft j with probability p_j(d_j): p_j is slot s0 + j's distribution as DeviceSampler
  builds it (softmax(logits / T) over the vocabulary, or over the top_k candidates; the nucleus keeps a candidate
  while the mass ranked above it is below top_p), and the draft is accepted when p_j(d_j) > 0 and
  p_j(d_j) >= u_accept[j]. At the first rejection it commits a token drawn from p_j without d_j (the residual of a
  point mass, renormalised) with u_corr; when every draft is accepted it commits the bonus drawn from p_n, n the
  drafts, with u_bonus.

Draws are by inverse CDF in the order the plugin's table walks a row: (logit descending, id ascending) for a row with
top_k (spec_accept's top-k path), id ascending for a nucleus row without top_k (spec_accept._top_p_only_support), so
the same uniforms draw the same token as the host walk. Fixed point as the sampler (tt/kernels/sample_fixed.hpp):
probabilities relative to the slot's largest logit in Q30, uniforms in Q24 (floor(u * 2^24)). A decision differs from
the host's float walk only when a draft probability or a cumulative probability lies within ~1e-7 of its uniform.

Truncation: a slot's candidates are the row's K largest logits and, past that rank, each chip's K largest. Once a chip's
list is spent, the next candidate the merge yields may not be the row's next largest. A group whose walk took a
candidate after that, or ran out of candidates before the nucleus closed, is flagged ``truncated`` and still accepted
on its candidates: the sampler's policy (it draws from the nucleus's tokens among the candidates); the adapter counts
those groups. Top-k ties at the k-th logit keep exactly top_k candidates (the lowest ids), where vLLM keeps every tie.

Layouts (uint32 words):
- params, per slot (PARAM_WORDS, DeviceSampler's layout): log2(e) / T as fp32 bits, top_p Q24, top_k (0: none), greedy,
  1 / T as fp32 bits, three unused.
- control, per group (GROUP_WORDS): first slot (NO_GROUP: unused), drafts n, greedy, unused, drafts [MAX_DRAFTS], then
  u_bonus, u_accept [MAX_DRAFTS], u_corr (Q24), zero padding.
- output, per group (OUT_WORDS): committed count in [1, 1 + n] (0 for an unused group), the committed tokens
  [1 + MAX_DRAFTS] (NO_TOKEN past the count), truncated, unused.
"""

import math
import struct

from .spec_decode import MAX_DRAFTS, SPEC_SLOTS

PARAM_WORDS = 8
GROUP_WORDS = 16 if MAX_DRAFTS <= 5 else 32  # a control group: 6 + 2 * MAX_DRAFTS words
OUT_WORDS = 8 if MAX_DRAFTS <= 5 else 16  # an output group: 3 + MAX_DRAFTS words
GROUPS = SPEC_SLOTS  # a pass holds at most one request per slot
NO_GROUP = 0xFFFFFFFF
NO_TOKEN = 0xFFFFFFFF  # the plugin's PLACEHOLDER_TOKEN_ID (-1) as uint32
LOG2E = 1.4426950408889634
ONE30 = 1 << 30
LN2_Q30 = 744261118
EXP2_TABLE = (
    1073741824, 1085434106, 1097253708, 1109202018, 1121280436, 1133490379, 1145833280, 1158310587,
    1170923762, 1183674286, 1196563654, 1209593378, 1222764986, 1236080024, 1249540052, 1263146652,
    1276901417, 1290805962, 1304861917, 1319070932, 1333434672, 1347954824, 1362633090, 1377471191,
    1392470869, 1407633882, 1422962010, 1438457051, 1454120821, 1469955159, 1485961921, 1502142985,
    1518500250, 1535035634, 1551751076, 1568648537, 1585730000, 1602997467, 1620452965, 1638098541,
    1655936265, 1673968228, 1692196547, 1710623359, 1729250827, 1748081133, 1767116489, 1786359126,
    1805811301, 1825475297, 1845353420, 1865448001, 1885761398, 1906295993, 1927054196, 1948038440,
    1969251188, 1990694927, 2012372174, 2034285470, 2056437387, 2078830522, 2101467502, 2124350982,
)  # fmt: skip
assert 3 + MAX_DRAFTS <= OUT_WORDS and 6 + 2 * MAX_DRAFTS <= GROUP_WORDS


# ---------------------------------------------------------------------- fixed point (sample_fixed.hpp)
def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def sort_key(bits):
    """sample_topk.hpp's monotone key of fp32 bits."""
    return (bits ^ (0xFFFFFFFF if bits & 0x80000000 else 0x80000000)) & 0xFFFFFFFF


def key_bits(key):
    return key & 0x7FFFFFFF if key & 0x80000000 else ~key & 0xFFFFFFFF


def to_fixed(bits, frac):
    e = (bits >> 23) & 0xFF
    if e == 0:
        return 0
    m = (bits & 0x7FFFFF) | 0x800000
    sh = e - 150 + frac
    if sh >= 0:
        v = 1 << 62 if sh >= 39 else m << sh
    else:
        v = 0 if sh <= -32 else m >> -sh
    return -v if bits & 0x80000000 else v


def exp2_q30(t):
    n = t >> 24
    if n < -31:
        return 0
    f = t & 0xFFFFFF
    j = f >> 18
    g = f & 0x3FFFF
    h = (g * LN2_Q30) >> 24
    h2 = (h * h) >> 30
    h3 = (h2 * h) >> 30
    eh = ONE30 + h + (h2 >> 1) + (h3 & 0xFFFFFFFF) // 6
    return ((EXP2_TABLE[j] * eh) >> 30) >> -n


def mul_q24(a, b):
    return (a >> 24) * b + (((a & 0xFFFFFF) * b) >> 24)


def mul_q30(a, b):
    return (a >> 30) * b + (((a & (ONE30 - 1)) * b) >> 30)


class Rel:
    """2^((v - top) * scale) in Q30 for fp32 bits v <= top (sample::Rel)."""

    def __init__(self, scale_bits, top_bits):
        self.scale_m = (scale_bits & 0x7FFFFF) | 0x800000
        self.scale_e = ((scale_bits >> 23) & 0xFF) - 150
        self.top = to_fixed(top_bits, 24)

    def __call__(self, v_bits):
        d = to_fixed(v_bits, 24) - self.top
        if d <= -(1 << 31):
            return 0 if d < -(1 << 34) else exp2_q30(((d >> 8) * self.scale_m) >> (-self.scale_e - 8))
        prod = d * self.scale_m
        return exp2_q30(prod << self.scale_e if self.scale_e >= 0 else prod >> -self.scale_e)


def uniform_q24(u):
    """A uniform in [0, 1) (the plugin's float32) as the kernel's Q24: floor(u * 2^24), below 2^24."""
    return min(max(int(math.floor(float(u) * (1 << 24))), 0), (1 << 24) - 1)


# ---------------------------------------------------------------------- host layouts
def param_words(temperature, top_k, top_p, vocab):
    """One slot's PARAM_WORDS (DeviceSampler._params' layout). top_k < 1 or >= ``vocab``: none; temperature 0 or
    top_k 1: greedy."""
    k = int(top_k) if 1 <= int(top_k) < vocab else 0
    greedy = float(temperature) == 0.0 or k == 1
    inv = 1.0 if greedy else 1.0 / max(float(temperature), 1e-2)
    inv = struct.unpack("<f", struct.pack("<f", inv))[0]
    return [
        f32_bits(inv * LOG2E),
        int(round(min(float(top_p), 1.0) * (1 << 24))),
        k,
        int(greedy),
        f32_bits(inv),
        0,
        0,
        0,
    ]


def group_words(first_slot, drafts, greedy, u_bonus, u_accept, u_corr):
    """One group's GROUP_WORDS: drafts the request's n valid drafts, uniforms in [0, 1)."""
    n = len(drafts)
    assert n <= MAX_DRAFTS and len(u_accept) >= n
    words = [int(first_slot), n, int(bool(greedy)), 0]
    words += [int(d) & 0xFFFFFFFF for d in drafts] + [0] * (MAX_DRAFTS - n)
    words += [uniform_q24(u_bonus)]
    words += [uniform_q24(u) for u in list(u_accept)[:n]] + [0] * (MAX_DRAFTS - n)
    words += [uniform_q24(u_corr)]
    return words + [0] * (GROUP_WORDS - len(words))


def control_words(groups):
    """The pass's control page: ``groups`` [(first_slot, drafts, greedy, u_bonus, u_accept, u_corr), or None for an
    unused group], at most GROUPS."""
    assert len(groups) <= GROUPS
    words = []
    for g in groups:
        words += [NO_GROUP] + [0] * (GROUP_WORDS - 1) if g is None else group_words(*g)
    for _ in range(GROUPS - len(groups)):
        words += [NO_GROUP] + [0] * (GROUP_WORDS - 1)
    return words


def unpack_out(words, groups=GROUPS):
    """[(count, committed tokens [count], truncated)] per group of the output page (uint32 words)."""
    out = []
    for g in range(groups):
        w = [int(x) & 0xFFFFFFFF for x in words[g * OUT_WORDS : (g + 1) * OUT_WORDS]]
        count = w[0]
        out.append((count, w[1 : 1 + count], bool(w[2 + MAX_DRAFTS])))
    return out


def pass_rows(vp):
    """The requests of a verify pass in the accept page's group order, [(group, row)]: in the order their rows first
    appear, or with draft groups (the pass's ``group_slots`` < its slots) each at its draft group's index, so that the
    chain step finds draft group g's commit in accept group g."""
    rows = list(dict.fromkeys(row for _, row, _ in vp.answers))
    if vp.group_slots == len(vp.positions):
        return list(enumerate(rows))
    first = {row: min(s for s, r, _ in vp.answers if r == row) for row in rows}
    return [(first[row] // vp.group_slots, row) for row in rows]


def pass_inputs(vp, spec_sampling):
    """VerifySampler.host_inputs' (slots, groups) for a verify pass (reference.spec_decode.VerifyPass): one group per
    request of the pass (``pass_rows``' index; None unused), with its first slot, its drafts (the tokens of its later
    slots), and greedy or its uniforms; a sampled request's slots take its (temperature, top_k, top_p).
    ``spec_sampling``: the plugin's per-row tensors (temperature, top_k, top_p, greedy, uniforms [B, K + 2]), or None
    (every row greedy)."""
    by_row = {}
    for s, row, col in vp.answers:
        by_row.setdefault(row, []).append((col, s))
    order = pass_rows(vp)
    slots, groups = {}, [None] * (1 + max((g for g, _ in order), default=-1))
    for g, row in order:
        cells = sorted(by_row[row])
        first = cells[0][1]
        drafts = [int(vp.tokens[s]) for _, s in cells[1:]]
        if spec_sampling is None or bool(spec_sampling["greedy"][row]):
            groups[g] = (first, drafts, True, 0.0, [0.0] * len(drafts), 0.0)
            continue
        u = [float(x) for x in spec_sampling["uniforms"][row]]
        k = len(u) - 2
        groups[g] = (first, drafts, False, u[0], u[1 : 1 + k], u[k + 1])
        params = (
            float(spec_sampling["temperature"][row]),
            int(spec_sampling["top_k"][row]),
            float(spec_sampling["top_p"][row]),
        )
        for _, s in cells:
            slots[s] = params
    if len(groups) > GROUPS:
        raise RuntimeError(f"a verify pass holds {len(groups)} requests, the accept step serves {GROUPS}")
    return slots, groups


def commit_pass(vp, accepted, committed, counts):
    """Writes a pass's accept results (``unpack_out``, groups in ``pass_rows``' order) into the verify's
    ``committed`` [B, 1 + K] and ``counts`` [B] rows (torch int32)."""
    for g, row in pass_rows(vp):
        count, tokens, _ = accepted[g]
        width = sum(1 for _, r, _ in vp.answers if r == row)
        if not 1 <= count <= width or len(tokens) != count:
            raise RuntimeError(f"row {row}: the accept step committed {count} tokens {tokens} of {width} columns")
        for c, t in enumerate(tokens):
            committed[row, c] = t - (1 << 32) if t >= 1 << 31 else t
        counts[row] = count


# ---------------------------------------------------------------------- the device's candidates (steps 1-2)
def chip_lists(logits, temperature, tp, k):
    """Steps 1-2 on the host, for tests: per chip, the slot's k largest fp32 logits of its vocab shard as sorted
    (key, local id) pairs (lowest id first among equal logits) and its normalizer, the sum of
    exp((logit - the chip's largest) / T) in Q30. ``logits``: float [vocab] (one slot). The device sums in fp32 on the
    SFPU per core and adds the cores in fixed point, so its normalizer differs from this one by rounding only."""
    vocab = len(logits)
    local = vocab // tp
    inv = 1.0 if float(temperature) == 0.0 else 1.0 / max(float(temperature), 1e-2)
    out = []
    for ch in range(tp):
        shard = [struct.unpack("<f", struct.pack("<f", float(x)))[0] for x in logits[ch * local : (ch + 1) * local]]
        order = sorted(range(local), key=lambda i: (-shard[i], i))[:k]
        pairs = [(sort_key(f32_bits(shard[i])), i) for i in order]
        top = shard[order[0]]
        total = sum(math.exp((x - top) * inv) for x in shard)
        out.append((pairs, int(total * ONE30)))
    return out


# ---------------------------------------------------------------------- the kernel (sample_verify.cpp)
class _Merge:
    """A slot's candidates in (logit descending, global id ascending) order, merging the chips' lists."""

    def __init__(self, lists, vocab_local):
        self.lists, self.vocab_local = lists, vocab_local
        self.k = len(lists[0][0])
        self.head = [0] * len(lists)
        self.spent = False  # a chip's list ran out: later candidates may not be the slot's next largest

    def _key(self, ch):
        h = self.head[ch]
        if h < self.k:
            key, idx = self.lists[ch][0][h]
            return key, ch * self.vocab_local + idx
        return 0, 0xFFFFFFFF

    def next(self):
        best = 0
        bk, bi = self._key(0)
        for ch in range(1, len(self.lists)):
            key, idx = self._key(ch)
            if key > bk or (key == bk and idx < bi):
                best, bk, bi = ch, key, idx
        if self.head[best] == self.k:
            return None
        self.head[best] += 1
        if self.head[best] == self.k:
            self.spent = True
        return bk, bi


def nucleus(lists, params, vocab_local):
    """One slot's kept candidates as sample_verify.cpp builds them: (probs Q30 relative to the largest, ids, kept
    total, by_id, truncated); probs and ids in (logit descending, id ascending) order."""
    merge = _Merge(lists, vocab_local)
    key0, id0 = merge.next()
    top_p, top_k = params[1], params[2]
    rel = Rel(params[0], key_bits(key0))
    total = sum(mul_q30(norm, rel(key_bits(pairs[0][0]))) for pairs, norm in lists)
    max_cands = len(lists) * merge.k
    has_k = top_k >= 1
    n_max = top_k if has_k and top_k < max_cands else max_cands
    limit = mul_q24(total, top_p)
    probs, ids, cum, truncated = [ONE30], [id0], ONE30, False
    while len(probs) < n_max:
        if not has_k and cum >= limit:
            break
        if merge.spent:
            truncated = True
        nxt = merge.next()
        if nxt is None:
            truncated = True
            break
        q = rel(key_bits(nxt[0]))
        probs.append(q)
        ids.append(nxt[1])
        cum += q
    if not has_k and cum < limit:
        truncated = True  # every candidate taken and the nucleus still open
    kept = len(probs)
    if has_k:
        k_limit = mul_q24(cum, top_p)
        above, kept = 0, 0
        while kept < len(probs) and (kept == 0 or above < k_limit):
            above += probs[kept]
            kept += 1
    return probs[:kept], ids[:kept], sum(probs[:kept]), not has_k, truncated


def draw(probs, ids, total, u_q24, by_id, exclude=None):
    """Inverse CDF over the kept candidates without ``exclude``: the first, in draw order, whose running mass exceeds
    u * (total - the excluded mass); past every one (rounding), the last positive one."""
    order = sorted(range(len(ids)), key=lambda i: ids[i]) if by_id else range(len(ids))
    rest = total - sum(p for p, t in zip(probs, ids) if t == exclude)
    threshold = mul_q24(rest, u_q24)
    run, last = 0, None
    for i in order:
        if ids[i] == exclude or probs[i] == 0:
            continue
        last = ids[i]
        run += probs[i]
        if run > threshold:
            return ids[i]
    return last if last is not None else ids[0]


def verify_group(slot_lists, slot_params, argmax, ctrl, vocab_local):
    """One group's (count, committed tokens, truncated): ``slot_lists`` per slot the chips' (pairs, normalizer),
    ``slot_params`` per slot PARAM_WORDS, ``argmax`` per slot the greedy token, ``ctrl`` the group's GROUP_WORDS."""
    first, n, greedy = ctrl[0], ctrl[1], ctrl[2]
    drafts = ctrl[4 : 4 + n]
    u_bonus = ctrl[4 + MAX_DRAFTS]
    u_accept = ctrl[5 + MAX_DRAFTS : 5 + MAX_DRAFTS + n]
    u_corr = ctrl[5 + 2 * MAX_DRAFTS]
    if greedy:
        tokens = []
        for j in range(n):
            tokens.append(argmax[first + j])
            if argmax[first + j] != drafts[j]:
                return j + 1, tokens, False
        return n + 1, tokens + [argmax[first + n]], False
    tokens, truncated = [], False
    for j in range(n):
        probs, ids, total, by_id, trunc = nucleus(slot_lists[first + j], slot_params[first + j], vocab_local)
        truncated |= trunc
        q = next((p for p, t in zip(probs, ids) if t == drafts[j]), 0)
        if q > 0 and q >= mul_q24(total, u_accept[j]):
            tokens.append(drafts[j])
            continue
        tokens.append(draw(probs, ids, total, u_corr, by_id, exclude=drafts[j]))
        return j + 1, tokens, truncated
    probs, ids, total, by_id, trunc = nucleus(slot_lists[first + n], slot_params[first + n], vocab_local)
    tokens.append(draw(probs, ids, total, u_bonus, by_id))
    return n + 1, tokens, truncated | trunc


def verify_pass(slot_lists, slot_params, argmax, control, vocab_local):
    """The kernel's output page (GROUPS * OUT_WORDS uint32 words) for one pass's control page."""
    words = []
    for g in range(GROUPS):
        ctrl = control[g * GROUP_WORDS : (g + 1) * GROUP_WORDS]
        out = [0] + [NO_TOKEN] * (1 + MAX_DRAFTS) + [0] * (OUT_WORDS - 2 - MAX_DRAFTS)
        if ctrl[0] != NO_GROUP:
            count, tokens, truncated = verify_group(slot_lists, slot_params, argmax, ctrl, vocab_local)
            out[0] = count
            out[1 : 1 + count] = [t & 0xFFFFFFFF for t in tokens]
            out[2 + MAX_DRAFTS] = int(truncated)
        words += out
    return words
