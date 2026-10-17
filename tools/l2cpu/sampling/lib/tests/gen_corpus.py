# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""Generate the QEMU bit-exactness corpus with the HOST build as the oracle.

Writes <out>.bin (rows + cases + expected results) and <out> (a .S that .incbin's it).
Every case stores the token returned by libx280s_host.so and the raw bytes of x280s_stats_t
(nan count, cap flag, k_eff, n_kept, pick, fallback, x_max, S, threshold, kept_sum, u, target, r);
the firmware compares all of them bit for bit.

Binary layout (little-endian), all offsets from the start of the blob:
  header (64 B): magic 'X2SC', version 1, n_rows, n_cases, rows_off, cases_off, expf_step, pad,
                 u64 expf_hash, pad to 64
  rows table: n_rows x {u64 offset, u32 dtype, u32 n_elems}
  cases: n_cases x 128 B {u32 row, vocab, stride, mode(0 sample, 1 argmax); params (24 B);
                          u32 user, pad; u64 step; i32 token, pad; stats (64 B)}
  row data, each 64-byte aligned

Optional: --npy FILE [--npy-start S --npy-count N] embeds recorded rows (float32 or uint16 bf16)
and adds the three reference settings for each of them.
"""

import argparse
import ctypes
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from x280s_ref import DTYPE_BF16, DTYPE_F32, Params, Stats, X280S, bf16_bits  # noqa: E402

F32 = np.float32
V_QWEN = 151936
SAMPLED_SETTINGS = [(0.7, 50, 0.9), (1.0, 0, 1.0), (0.6, 20, 0.95)]
EXTRA_SETTINGS = [
    (0.0, 50, 0.9),  # greedy through sample_row
    (1.0, 1, 1.0),
    (1.0, 1024, 1.0),
    (0.8, 5000, 0.0),  # k > K_MAX, top_p = 0
    (1.3, 200, 0.8),
    (1e-39, 0, 1.0),  # subnormal T: overflow to +inf ties
    (3e38, 10, 0.5),  # huge T: collapse to ties
    (float("inf"), 0, 1.0),
    (-1.0, 7, 0.5),  # negative T -> greedy
    (0.9, 64, 1.5),  # top_p > 1
    (0.9, 64, float("nan")),
]
EXPF_STEP = 61


def qwen_like(rng, V, n_big=8):
    x = rng.normal(0.0, 3.0, V).astype(F32)
    big = rng.choice(V, n_big, replace=False)
    x[big] += rng.uniform(8.0, 20.0, n_big).astype(F32)
    return x


class Corpus:
    def __init__(self, lib):
        self.lib = lib
        self.rows = []  # (np array, dtype)
        self.cases = []

    def add_row(self, arr):
        arr = np.ascontiguousarray(arr)
        dtype = DTYPE_F32 if arr.dtype == np.float32 else DTYPE_BF16
        assert arr.dtype in (np.float32, np.uint16)
        self.rows.append((arr, dtype))
        return len(self.rows) - 1

    def add_case(self, row_id, vocab, stride, mode, T=0.0, k=0, p=1.0, seed=0, user=0, step=0):
        arr, dtype = self.rows[row_id]
        if mode == 0:
            par = Params(T, k, p, 0, seed)
            st = Stats()
            tok = self.lib.lib.x280s_sample_row(
                arr.ctypes.data,
                dtype,
                vocab,
                stride,
                ctypes.byref(par),
                user,
                step,
                ctypes.byref(self.lib.work),
                ctypes.byref(st),
            )
        else:
            par = Params(0.0, 0, 0.0, 0, 0)
            st = Stats()
            tok = self.lib.lib.x280s_argmax_row(arr.ctypes.data, dtype, vocab, stride, ctypes.byref(st))
        assert tok >= 0, (row_id, vocab, stride, mode, T, k, p)
        blob = struct.pack("<4I", row_id, vocab, stride, mode)
        blob += ctypes.string_at(ctypes.byref(par), ctypes.sizeof(par))
        blob += struct.pack("<IIQiI", user, 0, step, tok, 0)
        blob += ctypes.string_at(ctypes.byref(st), ctypes.sizeof(st))
        assert len(blob) == 128
        self.cases.append(blob)

    def write(self, path_bin, expf_hash):
        hdr_size = 64
        rows_off = hdr_size
        cases_off = rows_off + 16 * len(self.rows)
        data_off = (cases_off + 128 * len(self.cases) + 63) & ~63
        table, data, off = b"", b"", data_off
        for arr, dtype in self.rows:
            raw = arr.tobytes()
            pad = (-len(raw)) % 64
            table += struct.pack("<QII", off, dtype, arr.size)
            data += raw + b"\0" * pad
            off += len(raw) + pad
        hdr = struct.pack(
            "<4s7IQ", b"X2SC", 1, len(self.rows), len(self.cases), rows_off, cases_off, EXPF_STEP, 0, expf_hash
        )
        hdr += b"\0" * (hdr_size - len(hdr))
        body = hdr + table + b"".join(self.cases)
        body += b"\0" * (data_off - len(body))
        with open(path_bin, "wb") as f:
            f.write(body + data)
        return len(body) + len(data)


def build(lib, rng, npy=None, npy_start=0, npy_count=0):
    c = Corpus(lib)
    seeds = lambda n: [int(s) for s in rng.integers(0, 2**63, n)]  # noqa: E731

    def all_settings(row_id, vocab, stride=1, steps=2, settings=None):
        for T, k, p in settings or (SAMPLED_SETTINGS + EXTRA_SETTINGS):
            for s, seed in enumerate(seeds(steps)):
                c.add_case(
                    row_id, vocab, stride, 0, T, k, p, seed, int(rng.integers(0, 32)), int(rng.integers(0, 2**40)) + s
                )
        c.add_case(row_id, vocab, stride, 1)

    # Qwen3-shaped rows: f32, bf16 (bf16 has many exact ties), NaN / inf variants, padded buffer
    for i in range(4):
        x = qwen_like(rng, V_QWEN, n_big=int(rng.integers(1, 40)))
        all_settings(c.add_row(x), V_QWEN, steps=3)
        all_settings(c.add_row(bf16_bits(x * F32(rng.uniform(0.5, 2.0)))), V_QWEN, steps=3)
    x = qwen_like(rng, V_QWEN)
    x[rng.choice(V_QWEN, 500, replace=False)] = np.nan
    all_settings(c.add_row(x), V_QWEN)
    x = qwen_like(rng, V_QWEN)
    x[[17, 90000]] = np.inf
    x[rng.choice(V_QWEN, 100, replace=False)] = -np.inf
    all_settings(c.add_row(x), V_QWEN)
    xp = np.concatenate([qwen_like(rng, V_QWEN), np.full(128, 1e30, F32)])  # padding must be ignored
    all_settings(c.add_row(xp), V_QWEN)
    # a narrow distribution (many candidates matter for top-p), and a very peaked one
    all_settings(c.add_row(rng.normal(0, 0.3, V_QWEN).astype(F32)), V_QWEN)
    all_settings(c.add_row(bf16_bits(rng.normal(0, 0.05, V_QWEN).astype(F32))), V_QWEN)

    # bf16 rows aimed at the RVV fast paths' fallbacks: zero maximum (+0 and -0), a negative NaN,
    # massive integer ties at Qwen size (threshold ties for K = 20/50/1024), ties of the maximum
    z = bf16_bits(-np.abs(rng.normal(0, 3, V_QWEN)).astype(F32) - F32(0.5))
    z[[5000, 7000]] = [0x8000, 0x0000]  # -0 first, then +0
    all_settings(c.add_row(z), V_QWEN)
    z2 = z.copy()
    z2[[100]] = 0x0000
    z2[[7000]] = 0x8000
    all_settings(c.add_row(z2), V_QWEN)
    n = bf16_bits(qwen_like(rng, V_QWEN))
    n[[123, 99999]] = 0xFFC0  # negative NaN
    all_settings(c.add_row(n), V_QWEN)
    qi = bf16_bits(np.round(rng.normal(0, 2, V_QWEN)).astype(F32))
    all_settings(c.add_row(qi), V_QWEN)
    m = bf16_bits(qwen_like(rng, V_QWEN))
    m[[3, 150000]] = bf16_bits(np.array([40.0, 40.0], F32))  # tied maximum at both ends
    all_settings(c.add_row(m), V_QWEN)

    # small synthetic rows
    small = []
    t = np.zeros(1000, F32)
    t[[17, 400, 999]] = 5.0
    small.append(t)
    small.append(np.full(5000, 1.25, F32))
    small.append(np.full(300, -np.inf, F32))
    small.append(np.full(300, np.nan, F32))
    z = np.full(64, -1.0, F32)
    z[[3, 5]] = [-0.0, 0.0]
    small.append(z)
    small.append(np.array([1e-39, -1e-40, 3e-45, 0.0, -0.0], F32))
    small.append(np.array([3e38, -3e38, 3.4e38, 1.0, np.inf, -np.inf], F32))
    small.append(np.array([2.0, 1.0, 0.0, -200.0], F32))
    for V in (1, 2, 7, 1023, 1024, 1025, 4097, 31):
        small.append(rng.normal(0, 3, V).astype(F32))
    q = np.round(rng.normal(0, 2, 3000)).astype(F32)  # integer logits: massive ties
    small.append(q)
    small.append(bf16_bits(q))
    for x in small:
        all_settings(c.add_row(x), x.size, steps=2)

    # strided rows: logical row = buf[::stride]
    for stride, dt in ((3, np.float32), (2, np.uint16), (5, np.float32)):
        V = 3001
        logical = rng.normal(0, 3, V).astype(F32)
        buf = rng.normal(0, 50, V * stride).astype(F32)
        buf[::stride] = logical
        if dt == np.uint16:
            buf = bf16_bits(buf)
        all_settings(c.add_row(buf), V, stride=stride, steps=2)

    if npy:
        arr = np.load(npy, mmap_mode="r")
        for i in range(npy_start, min(arr.shape[0], npy_start + npy_count)):
            row = np.ascontiguousarray(arr[i])
            V = min(row.size, V_QWEN)
            all_settings(c.add_row(row), V, steps=2, settings=SAMPLED_SETTINGS)
    return c


def expf_hash_host(build_dir):
    exe = os.path.join(build_dir, "expf_hash")
    src = [os.path.join(HERE, "expf_hash.c"), os.path.join(os.path.dirname(HERE), "x280s.c")]
    subprocess.check_call(["gcc", "-std=c11", "-O2", "-ffp-contract=off", "-fno-fast-math", "-o", exe] + src)
    return int(subprocess.check_output([exe, str(EXPF_STEP)]).decode().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output .S path (the .bin goes next to it)")
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--npy")
    ap.add_argument("--npy-start", type=int, default=0)
    ap.add_argument("--npy-count", type=int, default=20)
    a = ap.parse_args()
    lib = X280S()
    rng = np.random.default_rng(a.seed)
    c = build(lib, rng, a.npy, a.npy_start, a.npy_count)
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    h = expf_hash_host(out_dir)
    path_bin = os.path.splitext(os.path.abspath(a.out))[0] + ".bin"
    size = c.write(path_bin, h)
    with open(a.out, "w") as f:
        f.write('    .section .rodata.corpus, "a"\n    .balign 64\n    .globl x280s_corpus\nx280s_corpus:\n')
        f.write('    .incbin "%s"\n    .globl x280s_corpus_end\nx280s_corpus_end:\n' % path_bin)
    print(
        "corpus: %d rows, %d cases, %.1f MiB, expf hash %#018x (step %d)"
        % (len(c.rows), len(c.cases), size / 2**20, h, EXPF_STEP)
    )


if __name__ == "__main__":
    main()
