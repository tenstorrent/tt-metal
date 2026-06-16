# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 7 — fp32_dest_acc_en precision axis.

Exercises the (dtype × fp32_dest_acc_en) cells, with the focus on the cell the
refinement fixed: bfloat8_b at fp32_dest_acc_en=False (16-bit DEST register).

Before Refinement 7 this cell produced garbage (PCC ~0.047): the online-softmax
QK^T matmul passed the bf8b in0 buffer as the helper's interm placeholder, so
with bf16 DEST + no L1-acc the packer stayed in bf8b block-float encoding and
wrote cb_qk (bf16) with the wrong format. The fix passes a bf16 interm
placeholder (cb_o_tmp). The reference SDPA op reaches PCC ~0.9996 at the
identical config — this test confirms parity.

Inputs use `fa_rand` (the reference SDPA test's distribution: normal + rare
×10 outliers), exercised directly per the refinement instruction.
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


PCC_BY_DTYPE = {
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}


def fa_rand(*shape):
    """Reference SDPA input distribution: normal base + rare large outliers."""
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def _fp16_dest_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _torch_causal_sdpa(q, k, v, scale=None):
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    s = (q @ k.transpose(-2, -1)) * scale
    s_q, s_kv = q.shape[-2], k.shape[-2]
    mask = torch.triu(torch.full((s_q, s_kv), float("-inf")), diagonal=1)
    s = s + mask
    return torch.softmax(s, dim=-1) @ v


# ---------------------------------------------------------------------------
# The fixed cell: bf8b at fp32_dest_acc_en=False, on fa_rand inputs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize(
    "b,h,s,d",
    [
        (1, 1, 1024, 128),  # the minimal repro shape from Refinement 7
        (1, 2, 512, 64),
        (2, 1, 256, 64),
    ],
)
def test_fp16_dest_acc_causal(dtype, b, h, s, d, device):
    """bf8b and bf16 at fp32_dest_acc_en=False must match the reference."""
    torch.manual_seed(1234)
    q = fa_rand(b, h, s, d)
    k = fa_rand(b, h, s, d)
    v = fa_rand(b, h, s, d)
    expected = _torch_causal_sdpa(q.float(), k.float(), v.float())

    ttnn_q = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_k = ttnn.from_torch(k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_v = ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(
        ttnn_q, ttnn_k, ttnn_v, is_causal=True, compute_kernel_config=_fp16_dest_config()
    )
    result = ttnn.to_torch(out).float()

    assert not torch.isnan(result).any(), "output contains NaN"
    assert_with_pcc(expected, result, PCC_BY_DTYPE[dtype])


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_fp16_dest_acc_no_mask(dtype, device):
    """Same fix, no-mask (full attention) path on fa_rand inputs."""
    torch.manual_seed(7)
    b, h, s, d = 1, 1, 512, 128
    q = fa_rand(b, h, s, d)
    k = fa_rand(b, h, s, d)
    v = fa_rand(b, h, s, d)
    scale = 1.0 / (d**0.5)
    expected = (torch.softmax((q.float() @ k.float().transpose(-2, -1)) * scale, dim=-1)) @ v.float()

    ttnn_q = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_k = ttnn.from_torch(k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_v = ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(ttnn_q, ttnn_k, ttnn_v, compute_kernel_config=_fp16_dest_config())
    result = ttnn.to_torch(out).float()

    assert not torch.isnan(result).any()
    assert_with_pcc(expected, result, PCC_BY_DTYPE[dtype])


# ---------------------------------------------------------------------------
# fp32 + 16-bit DEST is a legal-but-lossy EXCLUSION → NotImplementedError.
# ---------------------------------------------------------------------------


def test_fp32_fp16_dest_acc_excluded(device):
    """fp32 input + fp32_dest_acc_en=False is refused op-side (EXCLUSION)."""
    torch.manual_seed(0)
    q = fa_rand(1, 1, 128, 64)
    k = fa_rand(1, 1, 128, 64)
    v = fa_rand(1, 1, 128, 64)
    ttnn_q = ttnn.from_torch(q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_k = ttnn.from_torch(k, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_v = ttnn.from_torch(v, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    with pytest.raises(NotImplementedError):
        scaled_dot_product_attention(ttnn_q, ttnn_k, ttnn_v, is_causal=True, compute_kernel_config=_fp16_dest_config())


def test_fp32_fp32_dest_acc_supported(device):
    """fp32 + fp32_dest_acc_en=True (default-equivalent) is still supported."""
    torch.manual_seed(0)
    q = fa_rand(1, 1, 128, 64)
    k = fa_rand(1, 1, 128, 64)
    v = fa_rand(1, 1, 128, 64)
    expected = _torch_causal_sdpa(q.float(), k.float(), v.float())
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    ttnn_q = ttnn.from_torch(q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_k = ttnn.from_torch(k, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_v = ttnn.from_torch(v, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(ttnn_q, ttnn_k, ttnn_v, is_causal=True, compute_kernel_config=cfg)
    result = ttnn.to_torch(out).float()
    assert_with_pcc(expected, result, 0.999)
