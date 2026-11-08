# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device


def _to_device(device, tensor):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device
    )


def _inputs():
    torch.manual_seed(1)
    q = torch.randn(1, 1, 32, 64).bfloat16()
    k = torch.randn(1, 1, 32, 64).bfloat16()
    v = torch.randn(1, 1, 32, 64).bfloat16()
    mask = torch.randn(1, 1, 32, 32).bfloat16()
    return q, k, v, mask


@pytest.mark.parametrize("scale", [0.0, -0.125])
def test_sdpa_rejects_nonpositive_scale_with_mask(device, expect_error, scale):
    q, k, v, mask = _inputs()
    with expect_error(RuntimeError, "scale must be greater than 0"):
        ttnn.transformer.scaled_dot_product_attention(
            _to_device(device, q),
            _to_device(device, k),
            _to_device(device, v),
            attn_mask=_to_device(device, mask),
            is_causal=False,
            scale=scale,
        )


@pytest.mark.parametrize("scale", [0.125, None])
def test_sdpa_mask_positive_scale_matches_torch(device, scale):
    q, k, v, mask = _inputs()
    out = ttnn.transformer.scaled_dot_product_attention(
        _to_device(device, q),
        _to_device(device, k),
        _to_device(device, v),
        attn_mask=_to_device(device, mask),
        is_causal=False,
        scale=scale,
    )
    golden = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), mask.float(), scale=scale if scale is not None else 64**-0.5
    )
    assert torch.allclose(ttnn.to_torch(out).float(), golden, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("scale", [0.0])
def test_sdpa_nonpositive_scale_without_mask_allowed(device, scale):
    """Without a mask the kernel path is not the broken one; these calls stay accepted."""
    q, k, v, _ = _inputs()
    out = ttnn.transformer.scaled_dot_product_attention(
        _to_device(device, q),
        _to_device(device, k),
        _to_device(device, v),
        is_causal=False,
        scale=scale,
    )
    golden = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale)
    assert torch.allclose(ttnn.to_torch(out).float(), golden, atol=0.05, rtol=0.05)
