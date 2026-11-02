# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Does ``ternary_matmul`` compute the right thing?

The engine's correctness gate, on small synthetic weights where the float64
reference is cheap and the failure modes are distinguishable:

  random     dense ternary, all scales distinct      catches a wrong scale index
  one_group  G == K, a single scale                  catches a missing scale entirely
  sparse     mostly zeros                            catches tile-exponent mistakes
  wide       N much larger than one core's share     catches the column walk

Reported per case: PCC against float64, and the max absolute error relative to
the largest output, which is what actually exposes a scale applied to the wrong
group (PCC stays high when only a few columns are wrong).

Usage (on a chip)::

    python -m models.demos.blackhole.bonsai2.native.ternary_matmul.test_ternary_matmul
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from ..linear_weight import TernaryWeight


def ref_ternary(codes: np.ndarray, scales: np.ndarray, group: int, x: np.ndarray) -> np.ndarray:
    """float64 ``x @ (codes * scales).T`` -- the definition the kernel implements."""
    n, k = codes.shape
    s = np.repeat(scales.astype(np.float64), group, axis=1)
    w = codes.astype(np.float64) * s
    return x.astype(np.float64) @ w.T


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def make_case(kind: str, n: int, k: int, group: int, seed: int):
    rng = np.random.default_rng(seed)
    if kind == "random":
        codes = rng.integers(-1, 2, size=(n, k)).astype(np.int8)
        scales = rng.uniform(0.001, 0.02, size=(n, k // group)).astype(np.float16)
    elif kind == "one_group":
        group = k
        codes = rng.integers(-1, 2, size=(n, k)).astype(np.int8)
        scales = rng.uniform(0.001, 0.02, size=(n, 1)).astype(np.float16)
    elif kind == "sparse":
        codes = rng.integers(-1, 2, size=(n, k)).astype(np.int8)
        mask = rng.random((n, k)) < 0.75
        codes[mask] = 0
        scales = rng.uniform(0.001, 0.02, size=(n, k // group)).astype(np.float16)
    elif kind == "wide":
        codes = rng.integers(-1, 2, size=(n, k)).astype(np.int8)
        scales = rng.uniform(0.001, 0.02, size=(n, k // group)).astype(np.float16)
    else:
        raise ValueError(kind)
    return codes, scales, group


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="random,one_group,sparse")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--pcc", type=float, default=0.9999)
    ap.add_argument("--timeout", type=float, default=90.0, help="seconds before a case is declared hung")
    args = ap.parse_args()

    import signal

    def on_alarm(signum, frame):
        raise TimeoutError(f"case exceeded {args.timeout}s -- almost certainly a CB deadlock in the kernel")

    signal.signal(signal.SIGALRM, on_alarm)

    import ttnn

    from .op import TernaryMatmul

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    rc = 0
    try:
        nbanks = len(device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0))
        print(f"{nbanks} DRAM banks; N={args.n} K={args.k} G={args.group} tokens={args.tokens}\n")

        for i, kind in enumerate(args.cases.split(",")):
            codes, scales, group = make_case(kind, args.n, args.k, args.group, seed=i)
            w = TernaryWeight(codes, scales, group, nbanks, name=kind)
            op = TernaryMatmul(device, w)

            rng = np.random.default_rng(100 + i)
            x = rng.standard_normal((args.tokens, w.K)).astype(np.float32)
            expected = ref_ternary(codes, scales, group, x)
            signal.alarm(int(args.timeout))
            try:
                got = op(torch.from_numpy(x)).float().numpy()
            except TimeoutError as e:
                signal.alarm(0)
                print(f"  {kind:10s} HUNG: {e}")
                return 2
            finally:
                signal.alarm(0)

            c = pcc(got, expected)
            scale = np.abs(expected).max() + 1e-30
            rel_max = float(np.abs(got - expected).max() / scale)
            ok = c >= args.pcc
            rc |= 0 if ok else 1
            print(
                f"  {kind:10s} N={args.n} K={args.k} G={group:4d}  pcc={c:.7f}  "
                f"max_rel_err={rel_max:.2e}  {'ok' if ok else 'FAIL'}"
            )
            ttnn.deallocate(op.w_tensor)
            ttnn.deallocate(op.s_tensor)
            ttnn.deallocate(op.x_tensor)
            ttnn.deallocate(op.out_tensor)
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
