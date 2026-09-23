# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Fit the piecewise-linear sigmoid tables the SwiGLU epilogue evaluates with one SFPLUTFP32 instruction
(`ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/swiglu_lut.hpp`), and score each table
on the ff1 SwiGLU output against an exact sigmoid.

The hardware evaluates f(|x|) = s_k * |x| + i_k on 3 or 6 segments (fp16 coefficients) and re-applies the sign, so
the table approximates the odd function sigmoid(x) - 0.5. Segments are fitted independently by least squares,
reweighted towards the minimax error; the last segment is the constant 0.5 (sigmoid saturates to 1).

    python fit_sigmoid_lut.py            # prints the tables and PCC / rel-RMSE vs an exact sigmoid on the bench's gate distribution

Gate pre-activations follow the mesh bench (`transformer_op_mesh_bench.py --op ff1`): x ~ N(0, 0.5^2), w ~ N(0, 1/K),
bf16-rounded, so |gate| has standard deviation ~0.5 and almost nothing lands above |x| = 2.
"""
import numpy as np
import torch

M, K, N2 = 13664, 5376, 7168
ROWS = 2048

# The shipped 3-segment 8-bit table of ckernel_sfpu_sigmoid_appx.h (cutoffs 1, 2).
SHIPPED_3SEG = [(0.22656, 0.0), (0.26562, -0.04687), (0.0, 0.5)]


def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def fit(cuts, points=2001, reweights=30):
    """Slope/intercept per segment of sigmoid(t) - 0.5 on [0, cuts[0]), [cuts[0], cuts[1]), ..., [cuts[-1], inf)."""
    edges = [0.0] + list(cuts) + [np.inf]
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isinf(hi):
            table.append((0.0, 0.5))
            continue
        t = np.linspace(lo, hi, points)
        y = sigmoid(t) - 0.5
        w = np.ones_like(t)
        for _ in range(reweights):
            a = np.stack([t * w, w], 1)
            s, i = np.linalg.lstsq(a, y * w, rcond=None)[0]
            err = np.abs(s * t + i - y)
            w = w * (err / err.mean()) ** 0.5 + 1e-9
        table.append((float(s), float(i)))
    return table


def apply(table, cuts, t, coeff_dtype=np.float16):
    """Evaluate the table the way the hardware does: fp16-rounded coefficients, sign retained, + 0.5."""
    edges = [0.0] + list(cuts) + [np.inf]
    a = np.abs(t)
    out = np.zeros_like(t)
    for (s, i), lo, hi in zip(table, edges[:-1], edges[1:]):
        s, i = float(coeff_dtype(s)), float(coeff_dtype(i))
        m = (a >= lo) & (a < hi)
        out[m] = s * a[m] + i
    return np.sign(t) * out + 0.5


def main() -> None:
    torch.manual_seed(0)
    x = (torch.randn(M, K) * 0.5).bfloat16().float()
    w = (torch.randn(K, N2) * (1.0 / K**0.5)).bfloat16().float()
    with torch.no_grad():
        full = x[:ROWS] @ w
        gate, up = torch.chunk(full, 2, dim=-1)
        golden = torch.nn.functional.silu(gate) * up
    g = gate.double().numpy()
    u = up.double().numpy()
    gold = golden.double().numpy()
    print(
        f"gate std {g.std():.3f}  |gate| > 1: {(abs(g) > 1).mean() * 100:.2f}%  > 2: {(abs(g) > 2).mean() * 100:.3f}%"
        f"  > 4: {(abs(g) > 4).mean() * 100:.4f}%"
    )
    grid = np.linspace(-8, 8, 20001)

    def score(name, sig_fn):
        out = torch.tensor(g * sig_fn(g) * u).bfloat16().double().numpy()  # the kernel packs bf16
        rel_rmse = np.sqrt(((out - gold) ** 2).mean()) / np.sqrt((gold**2).mean())
        pcc = np.corrcoef(out.ravel(), gold.ravel())[0, 1]
        max_err = np.max(np.abs(sig_fn(grid) - sigmoid(grid)))
        print(f"{name:40s} pcc {pcc:.6f}  rel-rmse {rel_rmse:.5f}  max |sigmoid err| on [-8, 8]: {max_err:.4f}")

    score("exact sigmoid (bf16 output rounding only)", sigmoid)
    for name, cuts, table in [
        ("shipped 8-bit 3-segment (1, 2)", (1, 2), SHIPPED_3SEG),
        ("fitted fp16 3-segment (1, 2)", (1, 2), fit((1, 2))),
        ("fitted fp16 6-segment HWM3 (.5,1,1.5,2,3)", (0.5, 1, 1.5, 2, 3), fit((0.5, 1, 1.5, 2, 3))),
        ("fitted fp16 6-segment HWM4 (.5,1,1.5,2,4)", (0.5, 1, 1.5, 2, 4), fit((0.5, 1, 1.5, 2, 4))),
    ]:
        score(name, lambda t, table=table, cuts=cuts: apply(table, cuts, t))
        print("    (slope, intercept) per segment:", ", ".join(f"({s:.5f}, {i:.5f})" for s, i in table))


if __name__ == "__main__":
    main()
