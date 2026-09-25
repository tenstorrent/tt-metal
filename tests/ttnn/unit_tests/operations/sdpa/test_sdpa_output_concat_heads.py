# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""SDPA with output_concat_heads=True against SDPA followed by nlp_concat_heads."""

import pytest
import torch

import ttnn

# Rows dropped from the logical length in the view case; not a tile multiple, so the last tile is partial.
VIEW_PAD_ROWS = 27


@pytest.mark.parametrize("pad_rows", [0, VIEW_PAD_ROWS], ids=["full", "view"])
@pytest.mark.parametrize(
    "b, nh, s, d",
    [(1, 8, 1024, 64), (1, 32, 1824, 64), (2, 4, 512, 128)],
    ids=["b1_nh8_s1024_d64", "b1_nh32_s1824_d64", "b2_nh4_s512_d128"],
)
def test_sdpa_output_concat_heads(device, b, nh, s, d, pad_rows):
    """The fused layout must be bit-identical to nlp_concat_heads on the valid rows."""
    torch.manual_seed(0)
    valid = s - pad_rows
    dim = nh * d
    padded = ttnn.Shape([b, nh, s, d])
    q, k, v = (
        ttnn.from_torch(torch.randn(b, nh, s, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        for _ in range(3)
    )
    if valid != s:
        logical = ttnn.Shape([b, nh, valid, d])
        q, k, v = (ttnn.reshape(t, logical, padded) for t in (q, k, v))

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    sdpa_kwargs = dict(
        attn_mask=None, is_causal=False, program_config=program_config, compute_kernel_config=compute_kernel_config
    )

    ref = ttnn.transformer.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
    if valid != s:
        ref = ttnn.reshape(ref, padded, padded)
    ref = ttnn.reshape(ttnn.experimental.nlp_concat_heads(ref), (b, s, dim))

    out = ttnn.transformer.scaled_dot_product_attention(q, k, v, output_concat_heads=True, **sdpa_kwargs)
    assert tuple(out.shape) == (b, 1, valid, dim)
    if valid != s:
        full = ttnn.Shape([b, 1, s, dim])
        out = ttnn.reshape(out, full, full)
    out = ttnn.reshape(out, (b, s, dim))

    ref_torch, out_torch = ttnn.to_torch(ref), ttnn.to_torch(out)
    assert out_torch.shape == ref_torch.shape == (b, s, dim)
    n_diff = int((out_torch[:, :valid] != ref_torch[:, :valid]).sum())
    assert torch.equal(out_torch[:, :valid], ref_torch[:, :valid]), f"{n_diff} of the valid elements differ"
