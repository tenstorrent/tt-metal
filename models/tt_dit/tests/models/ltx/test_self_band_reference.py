# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the temporal-band self-attention mask (band_ltx.temporal_band_mask). No device needed."""

import importlib.util
import pathlib

import pytest
import torch

# band_ltx is torch-only; load it by path so the test does not import ttnn through the package.
_spec = importlib.util.spec_from_file_location(
    "band_ltx",
    pathlib.Path(__file__).resolve().parents[3] / "models" / "transformers" / "ltx" / "band_ltx.py",
)
band_ltx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(band_ltx)


def _brute_force_mask(n_pad, n_real, tpf, window):
    q = torch.arange(n_pad)[:, None]
    k = torch.arange(n_pad)[None, :]
    allowed = ((q // tpf - k // tpf).abs() <= window) & (k < n_real)
    allowed |= (q >= n_real) & (k < n_real)
    return torch.where(allowed, 0.0, float("-inf"))


@pytest.mark.parametrize("n_real, n_pad, tpf, window", [(37, 48, 5, 1), (40, 48, 5, 0), (38, 64, 6, 2), (10, 16, 3, 9)])
def test_mask_matches_definition(n_real, n_pad, tpf, window):
    got = band_ltx.temporal_band_mask(n_pad, n_real, tpf, window).float()
    assert torch.equal(got, _brute_force_mask(n_pad, n_real, tpf, window))


def test_every_row_has_a_key():
    mask = band_ltx.temporal_band_mask(64, 50, 7, 0).float()
    probs = torch.softmax(mask, dim=-1)
    assert torch.isfinite(probs).all()


def test_wide_band_is_dense():
    torch.manual_seed(0)
    n_real, n_pad, tpf, d = 45, 48, 5, 16
    q, k, v = (torch.randn(1, 2, n_pad, d) for _ in range(3))
    dense_mask = torch.zeros(n_pad, n_pad)
    dense_mask[:, n_real:] = float("-inf")
    dense = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask)
    band = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=band_ltx.temporal_band_mask(n_pad, n_real, tpf, n_pad).float()
    )
    torch.testing.assert_close(band, dense)


def test_band_equals_sliced_attention():
    # The kernel-side claim: a query frame's band is one contiguous key range, so attending to
    # that slice alone gives the masked result.
    torch.manual_seed(1)
    n_real, tpf, window, d = 60, 6, 2, 8
    q, k, v = (torch.randn(1, 1, n_real, d) for _ in range(3))
    masked = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=band_ltx.temporal_band_mask(n_real, n_real, tpf, window).float()
    )
    for f in range(n_real // tpf):
        q0, q1 = f * tpf, (f + 1) * tpf
        k0, k1 = max(f - window, 0) * tpf, min((f + window + 1) * tpf, n_real)
        sliced = torch.nn.functional.scaled_dot_product_attention(q[:, :, q0:q1], k[:, :, k0:k1], v[:, :, k0:k1])
        torch.testing.assert_close(masked[:, :, q0:q1], sliced)
