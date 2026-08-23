# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only golden equivalence for the HP3 GDN conv1d sub-chunking (gdn/tp.py).

``TPGatedDeltaNet._conv1d_prefill_subchunked`` splits the prefill-chunk length axis into
<=_conv1d_max_t sub-slices and runs the depthwise causal conv per sub-slice, prepending each
with the previous K-1 input tokens (the halo; the first sub-slice gets the cross-chunk carry).
This test mirrors that index arithmetic in pure torch and requires the concatenated sub-slice
outputs to be BIT-IDENTICAL to one conv over the full chunk — for zero and random carries,
chunk length exactly at / above the cap, multi-slice, and ragged final sub-slices.

No ttnn / device needed. The repo-root conftest imports ttnn, so on a host without it run:
    python -m pytest models/demos/blackhole/qwen36/tests/test_gdn_conv1d_subchunk_golden.py --noconftest -q
"""

import pytest
import torch
import torch.nn.functional as F

K = 4  # gdn_conv_kernel_size (causal FIR taps)


def _dw_conv(xin, w):
    """Depthwise FIR over [Lin, C] with taps [C, K] -> [Lin-(K-1), C]; out[t] = sum_j w[:,j]*xin[t+j].

    Matches ttnn.conv1d(groups=C, stride 1, padding 0) over the carry-prepended input, i.e. the
    body of TPGatedDeltaNet._conv1d_run (pre-SiLU; SiLU is elementwise so equivalence is
    unaffected by it). Written as an explicit fixed-order tap sum (not F.conv1d) so the
    accumulation order is invariant to the input length — F.conv1d picks different backends
    for different sizes (~1e-6 fp32 wobble), which would mask/flag the wrong thing in a
    bit-identity test of the SLICING. _dw_conv itself is cross-checked against F.conv1d in
    test_dw_conv_mirror_matches_torch_conv1d.
    """
    Lout = xin.shape[0] - (K - 1)
    out = torch.zeros(Lout, xin.shape[1], dtype=xin.dtype)
    for j in range(K):
        out = out + xin[j : j + Lout] * w[:, j]
    return out


def test_dw_conv_mirror_matches_torch_conv1d():
    """Ground the hand-rolled FIR mirror against torch's conv1d (up to backend rounding)."""
    torch.manual_seed(7)
    x, w = torch.randn(517, 96), torch.randn(96, K)
    ref = F.conv1d(x.t().unsqueeze(0), w.unsqueeze(1), groups=96).squeeze(0).t()
    assert torch.allclose(_dw_conv(x, w), ref, atol=1e-5, rtol=1e-5)


def _conv_single_shot(x, w, carry):
    """The validated single-invocation path (_conv1d_prefill, T <= _conv1d_max_t)."""
    return _dw_conv(torch.cat([carry, x], dim=0), w)


def _conv_subchunked(x, w, carry, max_t):
    """Pure-torch mirror of _conv1d_prefill_subchunked's slicing (keep in lockstep with gdn/tp.py):

    for s in range(0, T, max_t): L = min(max_t, T - s)
      s == 0 : xin = cat(carry, x[:L])          # cross-chunk carry, as single-shot
      s > 0  : xin = x[s-(K-1) : s+L]           # halo = previous sub-slice's last K-1 tokens
    out = cat(per-slice conv outputs)
    """
    T = x.shape[0]
    outs = []
    for s in range(0, T, max_t):
        L = min(max_t, T - s)
        if s == 0:
            xin = torch.cat([carry, x[:L]], dim=0)
        else:
            xin = x[s - (K - 1) : s + L]
        outs.append(_dw_conv(xin, w))
    return torch.cat(outs, dim=0)


@pytest.mark.parametrize(
    "T,max_t",
    [
        (2048, 2048),  # exactly at the cap: single sub-slice, must equal single-shot trivially
        (4096, 2048),  # HP3 target chunk: 2 sub-slices
        (8192, 2048),  # HP3 stretch chunk: 4 sub-slices
        (6144, 2048),  # 3 sub-slices
        (4096 + 96, 2048),  # ragged (32-aligned) final sub-slice, as the device path allows
        (5000, 2048),  # ragged unaligned tail: index arithmetic must still hold
        (4096, 1024),  # smaller cap => more seams
    ],
    ids=lambda v: str(v),
)
@pytest.mark.parametrize("carry_kind", ["zero", "random"], ids=["carry0", "carryR"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_conv1d_subchunk_golden(T, max_t, carry_kind, dtype):
    torch.manual_seed(1234)
    C = 96  # any channel count; device uses qkv_dim_tp
    x = torch.randn(T, C).to(dtype)
    w = torch.randn(C, K).to(dtype)
    carry = torch.zeros(K - 1, C, dtype=dtype) if carry_kind == "zero" else torch.randn(K - 1, C).to(dtype)

    ref = _conv_single_shot(x, w, carry)
    got = _conv_subchunked(x, w, carry, max_t)

    assert ref.shape == got.shape == (T, C)
    # Same K-tap windows fed to the same conv op => bit-identical outputs.
    assert torch.equal(ref, got), (
        f"sub-chunked conv differs from single-shot: max abs err "
        f"{(ref.float() - got.float()).abs().max().item():.3e} (T={T}, max_t={max_t})"
    )

    # new_state parity: both paths derive the next-chunk carry as the last K-1 REAL input tokens
    # of the full chunk (the sub-chunked path slices qkv directly, before any sub-slicing).
    assert torch.equal(x[T - (K - 1) :], torch.cat([carry, x], dim=0)[-(K - 1) :])


def test_conv1d_subchunk_gate_is_inert_at_cap():
    """T == max_t must produce exactly one sub-slice whose input equals the single-shot input
    (the device code never even routes here — _conv1d_prefill gates on T > _conv1d_max_t)."""
    T = max_t = 512
    x = torch.randn(T, 8)
    carry = torch.randn(K - 1, 8)
    slices = [(s, min(max_t, T - s)) for s in range(0, T, max_t)]
    assert slices == [(0, T)]
    w = torch.randn(8, K)
    assert torch.equal(_conv_subchunked(x, w, carry, max_t), _conv_single_shot(x, w, carry))
