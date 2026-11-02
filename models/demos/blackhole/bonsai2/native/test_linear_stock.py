# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A1-stock correctness gate: does the block-masked scale fold compute the right thing?

The engine computes ``y[t] = x[t] @ (T * s).T`` as

    X_bd[t, g, :] = x[t, :] * mask[g, :]     # row g keeps only group g's K columns
    P[t, g, :]    = X_bd[t, g, :] @ T.T      # [M, G, N] group partial sums
    y[t, :]       = sum_g P[t, g, :] * s[g, :]

and the claim is that this is *exact*, not approximate: bfloat4_b holds -1, 0
and +1 with zero error (magnitude 1 needs no mantissa), so the codes round-trip,
and the only arithmetic beyond the integer sums is the scale fold.

Each case is built to fail differently if a specific step is wrong:

  random      every group has its own scale          catches a misindexed scale
  one_group   G == 1, one scale for all of K          catches a missing fold
  sparse      mostly zeros                            catches mask leakage
  multi_token M = 4, G > 1                            catches a per-token mix-up
  prefill     M = 32, G > 1                           the other end of the M range
  down_shape  K > N with G = 136 (the ffn_down case)  catches row padding bugs
                (G=136 pads to 160 rows)

The multi-token cases are the important ones: an earlier version reduced over a
flat ``[M*G, N]`` tensor, which sums *every* token's partials together and is
invisible at M=1. They compare against a per-token reference, and additionally
check that the rows are not identical to each other (which a cross-token sum
would make them).

Gates, per the plan: exact codes (checked directly on the device weight) and
PCC >= 0.9999 against a float64 reference.

Usage (on a chip)::

    python -m models.demos.blackhole.bonsai2.native.test_linear_stock
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def ref_ternary(codes: np.ndarray, scales: np.ndarray, group: int, x: np.ndarray) -> np.ndarray:
    """float64 ``x @ (codes * scales).T`` -- the definition being computed."""
    s = np.repeat(scales.astype(np.float64), group, axis=1)
    return x.astype(np.float64) @ (codes.astype(np.float64) * s).T


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def make_case(kind: str, n: int, k: int, group: int, tokens: int, seed: int):
    rng = np.random.default_rng(seed)
    if kind == "one_group":
        group = k
    if kind == "down_shape":
        n, k, group, tokens = 512, 17408, 128, 1
    if kind == "prefill":
        tokens = 32
    codes = rng.integers(-1, 2, size=(n, k)).astype(np.int8)
    if kind == "sparse":
        codes[rng.random((n, k)) < 0.75] = 0
    scales = rng.uniform(1e-3, 2e-2, size=(n, k // group)).astype(np.float16)
    # Distinct rows: a per-token mix-up would otherwise be masked by tokens that
    # happen to agree.
    x = rng.standard_normal((tokens, k)).astype(np.float32)
    return codes, scales, group, x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="random,one_group,sparse,multi_token,prefill,down_shape")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--pcc", type=float, default=0.9999)
    args = ap.parse_args()

    import ttnn

    from .linear_stock import TernaryLinearStock

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    rc = 0
    try:
        for i, kind in enumerate(args.cases.split(",")):
            tokens = 4 if kind == "multi_token" else args.tokens
            codes, scales, group, x = make_case(kind, args.n, args.k, args.group, tokens, seed=i)
            tokens = x.shape[0]  # make_case overrides the count for some cases
            n, k = codes.shape
            n_groups = k // group

            eng = TernaryLinearStock(device, codes, scales, group)

            # Gate 1: the codes survive bfloat4_b exactly.
            back = ttnn.to_torch(eng.w).float().numpy().T  # back to [N, K]
            code_err = int(np.abs(back - codes.astype(np.float32)).max() != 0)

            got = eng(torch.from_numpy(x))
            got = ttnn.to_torch(got).reshape(tokens, n).float().numpy()
            expected = ref_ternary(codes, scales, group, x)
            c = pcc(got, expected)
            rel = float(np.abs(got - expected).max() / (np.abs(expected).max() + 1e-30))
            # A cross-token reduction bug makes every row of the output equal.
            rows_differ = tokens == 1 or float(np.abs(got - got[0]).max()) > 1e-3
            ok = code_err == 0 and rows_differ and c >= args.pcc
            rc |= 0 if ok else 1
            print(
                f"  {kind:12s} N={n:5d} K={k:6d} G={group:4d} ({n_groups:3d} groups, "
                f"{eng.rows:3d} rows) M={tokens}  codes_exact={code_err == 0}  "
                f"rows_differ={rows_differ}  pcc={c:.7f}  max_rel={rel:.2e}  "
                f"{'ok' if ok else 'FAIL'}",
                flush=True,
            )
            if code_err:
                print(f"      FAIL: bfloat4_b did not hold the codes (max err {code_err})")
            if not rows_differ:
                print("      FAIL: every output row is identical - the group reduction is")
                print("            summing across tokens (needs the [M, 1, G, N] reshape)")
            ttnn.deallocate(eng.w)
            ttnn.deallocate(eng.mask)
            ttnn.deallocate(eng.scale)
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
