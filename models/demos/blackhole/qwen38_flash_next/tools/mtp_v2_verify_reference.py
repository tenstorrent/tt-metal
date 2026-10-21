# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Torch models of the MTP v2 verify-pass arithmetic shared by the step-2 no-device tests.

The accept/select model follows the design's section 2.6 op for op; ``fpu_rounding`` is the
precision model of the FPU reduce stages (TF32 source registers, as in the greedy-resolve tests).
``row_serial_torch`` makes torch's row-blocked kernels row-serial so R-row and per-row oracle
calls can be compared bitwise.
"""

from __future__ import annotations

import contextlib
from unittest import mock

import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next import reference

TF32_SIGNIFICANT_BITS = 11


@contextlib.contextmanager
def row_serial_torch():
    """Run F.linear, F.conv1d, F.softplus and the GDN q/k L2 norm one row at a time.

    torch's bf16 GEMM and reductions block differently for M = 1 and M = 5 rows (and its fp32 softplus takes
    a different vector/tail path), so a batched call is not bitwise the per-row call although each output
    row depends only on its own input row (a 1-ULP flip in a normalized k moves the GDN state by 4e-3).  The
    row-serial forms take that artifact out, so the tests compare the math, which is what the device does
    with per_core_M = 1 tiles.
    """

    linear, conv1d, softplus, l2_norm = F.linear, F.conv1d, F.softplus, reference._l2_norm

    def serial_linear(x, weight, bias=None):
        rows = x.reshape(-1, x.shape[-1])
        return torch.stack([linear(row[None], weight, bias)[0] for row in rows]).reshape(*x.shape[:-1], -1)

    def serial_softplus(x, beta=1.0, threshold=20.0):
        rows = x.reshape(-1, x.shape[-1])
        return torch.stack([softplus(row[None], beta, threshold)[0] for row in rows]).reshape(x.shape)

    def serial_conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        span = (weight.shape[-1] - 1) * dilation + 1  # one output position reads exactly this window
        windows = range(x.shape[-1] - span + 1)
        return torch.cat(
            [conv1d(x[..., t : t + span], weight, bias, stride, padding, dilation, groups) for t in windows], -1
        )

    def serial_l2_norm(x, eps=1e-6):  # [B, T, H, K]: one position at a time
        return torch.cat([l2_norm(x[:, t : t + 1], eps) for t in range(x.shape[1])], dim=1)

    patches = ((F, "linear", serial_linear), (F, "conv1d", serial_conv1d), (F, "softplus", serial_softplus))
    with contextlib.ExitStack() as stack:
        for target, name, serial in (*patches, (reference, "_l2_norm", serial_l2_norm)):
            stack.enter_context(mock.patch.object(target, name, serial))
        yield


def tf32(x: torch.Tensor) -> torch.Tensor:
    """Round to TF32 (11 significant bits, nearest-even): what an FPU reduce reads from srcA."""

    x = x.to(torch.float32)
    _, exponent = torch.frexp(x)
    quantum = torch.ldexp(torch.ones_like(x), exponent - TF32_SIGNIFICANT_BITS)
    return torch.round(x / quantum) * quantum


def ulp_distance(left: torch.Tensor, right: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Elementwise distance in units of ``dtype`` representable values (fp32 or bf16)."""

    integer = torch.int32 if dtype == torch.float32 else torch.int16
    bits = []
    for tensor in (left, right):
        raw = tensor.detach().to(dtype).contiguous().view(integer).to(torch.int64)
        sign_bit = 1 << (31 if dtype == torch.float32 else 15)
        bits.append(torch.where(raw < 0, -(raw + sign_bit), raw))  # monotone in the float value
    return (bits[0] - bits[1]).abs()


def accept_select(
    target_rows: torch.Tensor,
    drafts: torch.Tensor,
    alignment_rows: torch.Tensor,
    *,
    fpu_rounding=tf32,
) -> dict[str, int]:
    """Design 2.6 on fp32 token ids: eq, k-1 multiplies, sum, one-hot, and both select forms.

    ``target_rows`` are the k+1 verify-row argmaxes, ``drafts`` the k draft ids (row j of the
    targets is compared with draft j), ``alignment_rows`` the k+1 MTP alignment argmaxes.
    The ``*_sum`` selects are the design's ``sum(one_hot * rows)`` through an FPU reduce; the
    ``*_gather`` selects are a 32-bit copy at row ``a``.
    """

    k = drafts.numel()
    flags = (target_rows[:k] == drafts).to(torch.float32)  # ttnn.eq, SFPU, exact
    for j in range(1, k):
        flags[j] = flags[j] * flags[j - 1]  # k-1 multiplies: cumulative product
    accepted = fpu_rounding(flags).sum()  # ttnn.sum over 0/1 flags, exact
    one_hot = (torch.arange(k + 1, dtype=torch.float32) == accepted).to(torch.float32)
    row = int(accepted.item())
    return {
        "accepted": row,
        "next_token_sum": int(fpu_rounding(one_hot * target_rows).sum().item()),
        "next_token_gather": int(target_rows[row].item()),
        "first_draft_sum": int(fpu_rounding(one_hot * alignment_rows).sum().item()),
        "first_draft_gather": int(alignment_rows[row].item()),
    }


def chain_alphas(histogram: list[int]) -> list[float]:
    """Conditional acceptance alpha_j|j-1 from an accepted-length histogram (index = accepted drafts)."""

    survivors = [sum(histogram[j:]) for j in range(len(histogram))]  # rounds that accepted at least j
    return [survivors[j] / survivors[j - 1] for j in range(1, len(histogram))]


def expected_tokens_per_pass(alphas: list[float], k: int) -> float:
    """1 + sum_{j<=k} prod_{i<=j} alpha_i; alphas beyond the list repeat the last one."""

    if not alphas or k < 1:
        raise ValueError("expected_tokens_per_pass needs at least one alpha and k >= 1")
    total, chain = 1.0, 1.0
    for j in range(k):
        chain *= alphas[min(j, len(alphas) - 1)]
        total += chain
    return total
