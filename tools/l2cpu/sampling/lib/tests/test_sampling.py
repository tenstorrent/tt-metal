# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of the host build (libx280s_host.so) and the NumPy spec guard."""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import np_ref  # noqa: E402
from x280s_ref import X280S, bf16_bits, bf16_to_f32  # noqa: E402

F32 = np.float32
INF = np.float32(np.inf)


@pytest.fixture(scope="module")
def lib():
    return X280S()


def qwen_like(rng, V=151936, n_big=8):
    x = rng.normal(0.0, 3.0, V).astype(F32)
    big = rng.choice(V, n_big, replace=False)
    x[big] += rng.uniform(8.0, 20.0, n_big).astype(F32)
    return x


# ---------------------------------------------------------------- building blocks
def test_splitmix_known_vector(lib):
    # first output of the reference SplitMix64 generator seeded with 0
    assert lib.splitmix64(0) == 0xE220A8397B1DCDAF
    for x in (1, 12345, 2**63 + 7, 2**64 - 1):
        assert lib.splitmix64(x) == np_ref.splitmix64(x)


def test_expf_specials(lib):
    assert np.isnan(lib.expf(np.nan))
    assert lib.expf(INF) == INF
    assert lib.expf(-INF) == 0
    assert lib.expf(0.0) == 1 and lib.expf(-0.0) == 1
    assert lib.expf(-200.0) == 0 and lib.expf(200.0) == INF
    assert lib.expf(-103.0) > 0  # subnormal result, gradual underflow


def test_expf_accuracy_sample(lib):
    rng = np.random.default_rng(0)
    xs = np.concatenate([rng.uniform(-104, 0, 20000), rng.uniform(-1, 1, 5000), rng.uniform(0, 88.7, 5000)])
    xs = xs.astype(F32)
    got = np.array([lib.expf(x) for x in xs], dtype=F32)
    ref = np.exp(xs.astype(np.float64))
    ulp = np.spacing(ref.astype(F32)).astype(np.float64)
    ulp = np.maximum(ulp, 2.0**-149)
    err = np.abs(got.astype(np.float64) - ref) / ulp
    assert err.max() <= 1.5, err.max()


def test_struct_sizes(lib):
    assert lib.lib.x280s_sizeof_work() == 12288


# ---------------------------------------------------------------- greedy
def test_greedy_ties_lowest_index(lib):
    x = np.zeros(1000, F32)
    x[[17, 400, 999]] = 5.0
    assert lib.argmax(x)[0] == 17
    assert lib.sample(x, 0.0, 50, 0.9, 1)[0] == 17


def test_greedy_signed_zero_tie(lib):
    x = np.full(10, -1.0, F32)
    x[3] = -0.0
    x[5] = 0.0
    assert lib.argmax(x)[0] == 3


def test_greedy_nan(lib):
    x = np.arange(100, dtype=F32)
    x[99] = np.nan
    x[50] = np.nan
    tok, st = lib.argmax(x)
    assert tok == 98 and st.nan_count == 2 and st.greedy == 1


def test_greedy_all_nan_and_all_neg_inf(lib):
    assert lib.argmax(np.full(77, np.nan, F32))[0] == 0
    assert lib.argmax(np.full(77, -INF, F32))[0] == 0


def test_negative_and_nan_temperature_is_greedy(lib):
    x = np.arange(64, dtype=F32)
    for T in (-1.0, -0.0, np.nan):
        tok, st = lib.sample(x, T, 10, 0.9, 3)
        assert tok == 63 and st.greedy == 1


def test_padding_ignored(lib):
    x = np.zeros(1100, F32)
    x[1050] = 100.0
    x[3] = 1.0
    assert lib.argmax(x, vocab=1000)[0] == 3
    assert lib.sample(x, 1.0, 1, 1.0, 0, vocab=1000)[0] == 3


def test_bad_args(lib):
    x = np.zeros(10, F32)
    assert lib.argmax(x, vocab=0)[0] == -1


# ---------------------------------------------------------------- sampling corners
def test_k1_is_argmax_of_scaled(lib):
    rng = np.random.default_rng(1)
    for i in range(50):
        x = qwen_like(rng, V=5000)
        assert lib.sample(x, 0.7, 1, 0.9, i, step=i)[0] == int(np.argmax(x))
    # huge temperature: distinct logits collapse to equal scaled values -> lowest index wins
    x = np.array([1.0, 1.0000001, 0.5], F32)
    s = x / F32(3e38)
    assert s[0] == s[1]
    assert lib.sample(x, 3e38, 1, 1.0, 0)[0] == 0


def test_k1_ties(lib):
    x = np.zeros(300, F32)
    x[[10, 20]] = 2.0
    for step in range(20):
        assert lib.sample(x, 1.0, 1, 1.0, 9, step=step)[0] == 10


def test_ties_in_topk_boundary(lib):
    # 5 tied values, k = 3 keeps the 3 lowest indices
    x = np.zeros(100, F32)
    x[[90, 7, 55, 3, 70]] = 4.0
    picks = {lib.sample(x, 1.0, 3, 1.0, 0, step=s)[0] for s in range(300)}
    assert picks == {3, 7, 55}


def test_all_equal_row_uniform_over_first_k(lib):
    x = np.full(5000, 1.25, F32)
    picks = np.array([lib.sample(x, 1.0, 0, 1.0, 5, step=s)[0] for s in range(4000)])
    assert picks.min() >= 0 and picks.max() < 1024
    assert len(np.unique(picks)) > 900
    _, st = lib.sample(x, 1.0, 0, 1.0, 5)
    assert st.cap_applied == 1 and st.k_eff == 1024 and st.S == 1024.0 and st.n_kept == 1024


def test_all_neg_inf_row_samples_like_all_equal(lib):
    x = np.full(50, -INF, F32)
    picks = {lib.sample(x, 1.0, 4, 1.0, 1, step=s)[0] for s in range(200)}
    assert picks == {0, 1, 2, 3}


def test_nan_in_sampling(lib):
    x = np.full(2000, np.nan, F32)
    x[1500] = 1.0
    x[10] = 0.5
    tok, st = lib.sample(x, 1.0, 50, 1.0, 1)
    assert st.nan_count == 1998
    assert tok in (10, 1500)
    for s in range(200):
        assert lib.sample(x, 1.0, 50, 1.0, 1, step=s)[0] in (10, 1500)


def test_cap_flag(lib):
    x = np.random.default_rng(2).normal(size=3000).astype(F32)
    assert lib.sample(x, 1.0, 0, 1.0, 1)[1].cap_applied == 1
    assert lib.sample(x, 1.0, 5000, 1.0, 1)[1].cap_applied == 1
    assert lib.sample(x, 1.0, 1024, 1.0, 1)[1].cap_applied == 0
    assert lib.sample(x, 1.0, 50, 1.0, 1)[1].cap_applied == 0
    small = x[:600]
    _, st = lib.sample(small, 1.0, 0, 1.0, 1)
    assert st.cap_applied == 0 and st.k_eff == 600


def test_k_greater_than_v(lib):
    x = np.array([0.0, 1.0, 2.0], F32)
    _, st = lib.sample(x, 1.0, 50, 1.0, 1)
    assert st.k_eff == 3
    assert {lib.sample(x, 1.0, 50, 1.0, 1, step=s)[0] for s in range(200)} == {0, 1, 2}


def test_top_p_one_keeps_through_last_nonzero(lib):
    x = np.array([0.0, -1.0, -2.0, -200.0], F32)  # last weight underflows to 0
    _, st = lib.sample(x, 1.0, 0, 1.0, 1)
    assert st.n_kept == 3 and st.kept_sum == st.S


def test_top_p_le_zero_keeps_one(lib):
    rng = np.random.default_rng(3)
    x = qwen_like(rng, V=4000)
    for tp in (0.0, -1.0):
        for s in range(20):
            tok, st = lib.sample(x, 1.0, 50, tp, 1, step=s)
            assert tok == int(np.argmax(x)) and st.n_kept == 1


def test_top_p_gt_one_and_nan_behave_as_one(lib):
    x = qwen_like(np.random.default_rng(4), V=4000)
    for s in range(30):
        a = lib.sample(x, 0.9, 40, 1.0, 7, step=s)
        for tp in (1.5, np.nan):
            b = lib.sample(x, 0.9, 40, tp, 7, step=s)
            assert a[0] == b[0] and a[1].float_bits() == b[1].float_bits()


def test_tiny_temperature_overflow_ties(lib):
    x = np.array([1.0, 3.0, -2.0, 3.0, 2.0], F32)
    # T = 1e-39 (subnormal): 3/T, 2/T and 1/T overflow to +inf -> four +inf candidates tie at weight 1
    picks = {lib.sample(x, 1e-39, 0, 1.0, 1, step=s)[0] for s in range(300)}
    assert picks == {0, 1, 3, 4}
    # k = 2 keeps the two lowest-index +inf entries
    picks = {lib.sample(x, 1e-39, 2, 1.0, 1, step=s)[0] for s in range(300)}
    assert picks == {0, 1}


def test_tiny_temperature_no_overflow_is_near_greedy(lib):
    x = qwen_like(np.random.default_rng(5), V=20000)
    for s in range(20):
        assert lib.sample(x, 1e-4, 50, 0.95, 1, step=s)[0] == int(np.argmax(x))


def test_infinite_temperature(lib):
    x = np.array([np.inf, 1.0, -np.inf, 2.0], F32)
    tok, st = lib.sample(x, np.inf, 0, 1.0, 1)
    assert st.x_max == INF and tok == 0
    y = np.array([5.0, 1.0, -7.0], F32)
    assert {lib.sample(y, np.inf, 0, 1.0, 1, step=s)[0] for s in range(200)} == {0, 1, 2}


def test_pos_inf_logits(lib):
    x = np.zeros(100, F32)
    x[[30, 60]] = np.inf
    assert {lib.sample(x, 0.8, 10, 0.9, 1, step=s)[0] for s in range(200)} == {30, 60}


def test_bf16_equals_f32_of_converted(lib):
    rng = np.random.default_rng(6)
    for i in range(10):
        x = qwen_like(rng, V=151936)
        b = bf16_bits(x)
        f = bf16_to_f32(b)
        for T, k, p in ((0.7, 50, 0.9), (1.0, 0, 1.0), (0.6, 20, 0.95)):
            t1, s1 = lib.sample(b, T, k, p, 11, user=i, step=i)
            t2, s2 = lib.sample(f, T, k, p, 11, user=i, step=i)
            assert t1 == t2 and s1.float_bits() == s2.float_bits()
        assert lib.argmax(b)[0] == lib.argmax(f)[0]


def test_stride(lib):
    rng = np.random.default_rng(7)
    base = rng.normal(0, 3, (1001, 3)).astype(F32)
    col = base[:, 1]
    assert col.strides[0] == 12
    contiguous = np.ascontiguousarray(col)
    for s in range(20):
        a = lib.sample(col, 0.8, 30, 0.9, 2, step=s)
        b = lib.sample(contiguous, 0.8, 30, 0.9, 2, step=s)
        assert a[0] == b[0] and a[1].float_bits() == b[1].float_bits()
    assert lib.argmax(col)[0] == lib.argmax(contiguous)[0]


@pytest.mark.parametrize("V", [1, 2, 7, 1000, 1023, 1024, 1025, 4097, 151937])
def test_odd_vocab_sizes(lib, V):
    rng = np.random.default_rng(V)
    x = rng.normal(0, 3, V).astype(F32)
    for k in (0, 1, 50):
        for s in range(5):
            tok, st = lib.sample(x, 0.8, k, 0.9, 3, step=s)
            ref, _ = np_ref.sample(x, 0.8, k, 0.9, 3, step=s)
            assert 0 <= tok < V and tok == ref


def test_seed_user_step_mixing(lib):
    x = np.zeros(10, F32)
    _, a = lib.sample(x, 1.0, 0, 1.0, seed=5, user=3, step=9)
    assert a.r == np_ref.splitmix64(5 ^ (3 << 32) ^ 9)
    _, b = lib.sample(x, 1.0, 0, 1.0, seed=5, user=4, step=9)
    assert a.r != b.r
    assert a.u == F32(a.r >> 40) * F32(2.0**-24)


def test_draw_distribution(lib):
    # weights exp(0), exp(-1), exp(-2): frequencies close to the softmax
    x = np.array([2.0, 1.0, 0.0], F32)
    n = 30000
    picks = np.array([lib.sample(x, 1.0, 0, 1.0, 99, step=s)[0] for s in range(n)])
    p = np.exp([0.0, -1.0, -2.0])
    p /= p.sum()
    freq = np.bincount(picks, minlength=3) / n
    assert np.all(np.abs(freq - p) < 0.01), (freq, p)


def test_determinism(lib):
    x = qwen_like(np.random.default_rng(8))
    a = lib.sample(x, 0.7, 50, 0.9, 1234, user=5, step=77)
    b = lib.sample(x.copy(), 0.7, 50, 0.9, 1234, user=5, step=77)
    assert a[0] == b[0] and a[1].float_bits() == b[1].float_bits()


# ---------------------------------------------------------------- spec guard (NumPy model)
def _explain_mismatch(info, tok_c, st_c, tok_n):
    """True when the C and NumPy picks are adjacent across a decision boundary within tolerance."""
    w = info["w"].astype(np.float64)
    cum = np.cumsum(w) / w.sum()
    order = list(info["order"])
    jc, jn = order.index(tok_c), order.index(tok_n)
    tol = 1e-5
    # a different top-p cut: the cumulative mass at the cut is within tol of top_p
    if st_c.n_kept != info["n_kept"]:
        j = min(st_c.n_kept, info["n_kept"]) - 1
        if abs(cum[j] - float(info["p"])) < tol:
            return True
    # a different draw position: the boundary between the two picks is within tol of the target
    lo = min(jc, jn)
    kept = cum[info["n_kept"] - 1]
    target_frac = float(info["u"]) * kept
    return abs(cum[lo] - target_frac) < tol and abs(jc - jn) == 1


def test_numpy_reference(lib):
    rng = np.random.default_rng(1234)
    settings = [(0.7, 50, 0.9), (1.0, 0, 1.0), (0.6, 20, 0.95), (1.3, 200, 0.8), (0.9, 1024, 0.99)]
    n, same, explained = 0, 0, 0
    for i in range(60):
        V = int(rng.choice([151936, 32000, 4099]))
        x = qwen_like(rng, V=V, n_big=int(rng.integers(1, 30)))
        if i % 3 == 0:
            x = bf16_to_f32(bf16_bits(x))  # many exact ties
        for T, k, p in settings:
            for step in range(3):
                seed = int(rng.integers(0, 2**63))
                tok_c, st_c = lib.sample(x, T, k, p, seed, user=i % 32, step=step)
                tok_n, info = np_ref.sample(x, T, k, p, seed, user=i % 32, step=step)
                n += 1
                if tok_c == tok_n:
                    same += 1
                    continue
                assert _explain_mismatch(info, tok_c, st_c, tok_n), (i, T, k, p, step, tok_c, tok_n)
                explained += 1
    print("numpy guard: %d cases, %d identical, %d explained boundary differences" % (n, same, explained))
    assert same >= 0.99 * n


# ---------------------------------------------------------------- replay tool
@pytest.mark.parametrize("bf16", [False, True])
def test_replay_tool(lib, tmp_path, bf16):
    import subprocess

    rows_path = tmp_path / "rows.npy"
    out = tmp_path / "exp.npz"
    replay = os.path.join(os.path.dirname(HERE), "replay.py")
    args = ["--n", "6"] + (["--bf16"] if bf16 else [])
    subprocess.check_call([sys.executable, replay, "--make-synthetic", str(rows_path)] + args)
    subprocess.check_call(
        [sys.executable, replay, str(rows_path), "--out", str(out), "--seed", "0x5eed", "--user", "3", "--step0", "100"]
    )
    rows = np.load(rows_path)
    assert rows.shape == (6, 151936) and rows.dtype == (np.uint16 if bf16 else np.float32)
    res = np.load(out)
    for i in range(6):
        assert res["greedy"][i] == lib.argmax(rows[i])[0]
        for s, (T, k, p) in enumerate(((0.7, 50, 0.9), (1.0, 0, 1.0), (0.6, 20, 0.95))):
            assert res["tokens"][s, i] == lib.sample(rows[i], T, k, p, 0x5EED, user=3, step=100 + i)[0]
    lines = open(str(out)[:-4] + ".txt").read().splitlines()
    assert len(lines) == 7
