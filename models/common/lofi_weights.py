# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Weight transforms for matmuls that run at MathFidelity.LoFi.

A LoFi matmul multiplies with its weight operand (in1, unpacked to srcA) truncated to 5 significant bits; the
activation (srcB) keeps its bits. Truncation is toward zero, so every output shrinks on average (~1.3% on a real
4096-wide projection) and the error grows through a model's layers. A weight that already has at most 5 significant
bits is not changed by the truncation, so rounding the weights to that grid once, before they are converted and
cached, makes the LoFi matmul exact for the rounded weights: LoFi speed, error of one unbiased rounding.

Measured on Blackhole (800x4096 @ 4096x1536, bf8 weights, fp32 accumulation), relative error vs fp64:

    LoFi                                       0.0188   (equal to emulate_lofi_weight() on the host)
    LoFi on round_for_lofi() weights           0.0143   (equal to HiFi4 on the same weights)
    HiFi2 / HiFi4 on the original weights      0.00025

round_for_lofi() rounds to nearest. gptq_round_for_lofi() picks the rounding of each weight with GPTQ (Frantar et al.,
2022): the error of each rounded input column is fed back into the columns not rounded yet, weighted by the inverse of
the layer's input second moment H = E[x^T x] (LinearInputStats). It needs calibration activations and lowers the
layer's output error further (7-22x on the Qwen3.5-9B GDN in-projections).

All functions take weights as [out_features, in_features] (torch.nn.Linear layout) and return the dtype they got.
A value with <= 5 significant bits is exact in bf16 and survives the bf8 block-float conversion with <= 5 bits.
"""

import torch

LOFI_WEIGHT_BITS = 5


def _split(w):
    m, e = torch.frexp(w.double() if w.dtype == torch.float64 else w.float())
    return m, e, float(2**LOFI_WEIGHT_BITS)


def round_for_lofi(w: torch.Tensor) -> torch.Tensor:
    """Round every element to LOFI_WEIGHT_BITS significant bits (round half to even). Zeros stay zero."""
    m, e, s = _split(w)
    return torch.ldexp(torch.round(m * s) / s, e).to(w.dtype)


def emulate_lofi_weight(w: torch.Tensor) -> torch.Tensor:
    """What a LoFi matmul uses of `w`: every element truncated toward zero to LOFI_WEIGHT_BITS significant bits."""
    m, e, s = _split(w)
    return torch.ldexp(torch.trunc(m * s) / s, e).to(w.dtype)


def is_lofi_exact(w: torch.Tensor) -> bool:
    """True when a LoFi matmul uses `w` unchanged (every element has <= LOFI_WEIGHT_BITS significant bits)."""
    return bool(torch.equal(emulate_lofi_weight(w), w))


class LinearInputStats:
    """Second moment H = E[x^T x] of a linear layer's input, accumulated over calibration batches (fp64)."""

    def __init__(self, in_features: int):
        self.sum = torch.zeros(in_features, in_features, dtype=torch.float64)
        self.count = 0

    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).double()
        self.sum.addmm_(x.T, x)
        self.count += x.shape[0]

    def hessian(self) -> torch.Tensor:
        assert self.count > 0, "no calibration rows"
        return self.sum / self.count


def lofi_output_error(w_rounded: torch.Tensor, w: torch.Tensor, hessian: torch.Tensor) -> float:
    """Relative output error of the rounded weight over the calibration inputs: tr(dW H dW^T) / tr(W H W^T)."""
    w64 = w.double()
    d = w_rounded.double() - w64
    return float(((d @ hessian) * d).sum() / ((w64 @ hessian) * w64).sum())


def gptq_round_for_lofi(
    w: torch.Tensor, hessian: torch.Tensor, damp: float = 0.01, block_size: int = 128, act_order: bool = True
) -> torch.Tensor:
    """GPTQ rounding of w [out, in] to the LoFi weight grid, given the input second moment `hessian` [in, in].

    Rows are independent, so any row permutation or split of `w` gives the same rows. Deterministic (fp64 on the
    host). Cost O(in^2 * out): ~10 s for a 4096-input, 12k-output projection."""
    n = w.shape[1]
    assert hessian.shape == (n, n), (hessian.shape, w.shape)
    w0 = w.double()
    h = hessian.double().clone()
    h.diagonal().add_(damp * h.diagonal().mean())
    perm = torch.argsort(h.diagonal(), descending=True) if act_order else torch.arange(n)
    W = w0[:, perm].clone()
    hp = h[perm][:, perm]
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hp)), upper=True)
    q = torch.zeros_like(W)
    for i1 in range(0, n, block_size):
        i2 = min(i1 + block_size, n)
        w1 = W[:, i1:i2].clone()
        e1 = torch.zeros_like(w1)
        hi = hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            col = w1[:, j]
            qc = round_for_lofi(col)
            q[:, i1 + j] = qc
            err = (col - qc) / hi[j, j]
            w1[:, j:] -= err.unsqueeze(1) * hi[j, j:].unsqueeze(0)
            e1[:, j] = err
        W[:, i2:] -= e1 @ hinv[i1:i2, i2:]
    out = torch.empty_like(q)
    out[:, perm] = q
    return out.to(w.dtype)
