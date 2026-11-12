# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks of models/common/lofi_weights.py against a host emulation of the LoFi weight truncation."""

import pytest
import torch

from models.common.lofi_weights import (
    LinearInputStats,
    emulate_lofi_weight,
    gptq_round_for_lofi,
    is_lofi_exact,
    lofi_output_error,
    round_for_lofi,
)


def _weight(out_f, in_f, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(out_f, in_f, generator=g) * 0.02).to(torch.bfloat16)


def _acts(rows, in_f, seed=1):
    """Correlated activations with a few large channels, as in transformer hidden states."""
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(in_f, in_f, generator=g) / in_f**0.5
    x = torch.randn(rows, in_f, generator=g) @ mix
    x[:, :4] *= 20
    return x


def test_rounded_weights_are_lofi_fixed_points():
    w = _weight(64, 256)
    r = round_for_lofi(w)
    assert r.dtype == w.dtype
    assert is_lofi_exact(r)
    assert torch.equal(emulate_lofi_weight(r), r)
    assert torch.equal(round_for_lofi(r), r)
    assert not is_lofi_exact(w)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_round_is_nearest_on_the_5_bit_grid(dtype):
    w = _weight(32, 128).to(dtype)
    r = round_for_lofi(w).double()
    m, _ = torch.frexp(r)
    assert torch.equal(m * 32, torch.round(m * 32))  # 5 significant bits
    # at most half a grid step from w: |w - r| <= 2^(e - 6) with w = m 2^e, m in [0.5, 1)
    _, e = torch.frexp(w.double())
    assert bool((torch.abs(w.double() - r) <= torch.ldexp(torch.ones_like(r), e - 6)).all())
    assert torch.equal(round_for_lofi(torch.zeros(4, dtype=dtype)), torch.zeros(4, dtype=dtype))


def test_truncation_is_biased_rounding_is_not():
    """LoFi on raw weights shrinks the output (truncation toward zero); on rounded weights it is unbiased and exact."""
    w = _weight(256, 512).float()
    x = _acts(512, 512)
    ref = x.double() @ w.double().T
    lofi = x.double() @ emulate_lofi_weight(w).double().T
    lofi_r = x.double() @ emulate_lofi_weight(round_for_lofi(w)).double().T
    exact_r = x.double() @ round_for_lofi(w).double().T
    assert torch.equal(lofi_r, exact_r)
    scale = float((lofi * ref).sum() / (ref * ref).sum())
    scale_r = float((lofi_r * ref).sum() / (ref * ref).sum())
    assert scale < 0.99
    assert abs(scale_r - 1) < 0.003
    assert torch.linalg.norm(lofi_r - ref) < torch.linalg.norm(lofi - ref)


def test_rounded_weights_survive_bf8_style_block_quantization():
    """bf8_b keeps 7 mantissa bits under a shared exponent per 16 values: a <= 5-bit value stays <= 5 bits."""
    r = round_for_lofi(_weight(16, 256)).double().reshape(-1, 16)
    emax = torch.frexp(r.abs().amax(dim=1, keepdim=True))[1]
    step = torch.ldexp(torch.ones_like(r), emax - 7)
    q = torch.round(r / step) * step
    assert is_lofi_exact(q)


def test_input_stats():
    x = _acts(300, 64)
    st = LinearInputStats(64)
    st.update(x[:100])
    st.update(x[100:].reshape(2, 100, 64))
    torch.testing.assert_close(st.hessian(), x.double().T @ x.double() / 300)


def test_gptq_beats_nearest_and_stays_on_grid():
    w = _weight(96, 256)
    x = _acts(2048, 256)
    st = LinearInputStats(256)
    st.update(x)
    h = st.hessian()
    q = gptq_round_for_lofi(w, h)
    assert q.dtype == w.dtype
    assert is_lofi_exact(q)
    e_rtn = lofi_output_error(round_for_lofi(w), w, h)
    e_gptq = lofi_output_error(q, w, h)
    assert e_gptq < 0.5 * e_rtn, (e_gptq, e_rtn)


def test_gptq_rows_are_independent_and_deterministic():
    w = _weight(64, 128)
    st = LinearInputStats(128)
    st.update(_acts(1024, 128))
    h = st.hessian()
    q = gptq_round_for_lofi(w, h)
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(3))
    assert torch.equal(gptq_round_for_lofi(w[perm], h), q[perm])
    assert torch.equal(gptq_round_for_lofi(w, h), q)
