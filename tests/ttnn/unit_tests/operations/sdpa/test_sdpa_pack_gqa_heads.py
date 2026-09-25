# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``scaled_dot_product_attention(..., pack_gqa_heads=True)`` schedules the NQH / NKH query heads of a GQA group as
one head of (NQH / NKH) * S rows. For a tile-aligned S that is the same memory as viewing Q as
``[B, NKH, (NQH / NKH) * S, d]``, so the packed op must be bit-identical to the unpacked op run on that view, and
with ``output_heads_concat`` to ``nlp_concat_heads`` of the packed head-major output."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("b, nh, nkv, s, d", [(1, 32, 8, 512, 128), (8, 32, 8, 512, 128), (2, 16, 4, 256, 64)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
# 192 and 160 do not divide S: chunks run into the next query head of the group and the last chunk is padded
@pytest.mark.parametrize("q_chunk, k_chunk", [(256, 256), (256, 512), (128, 256), (192, 512), (160, 256)])
@pytest.mark.timeout(300)
def test_sdpa_pack_gqa_heads(device, b, nh, nkv, s, d, dtype, q_chunk, k_chunk):
    if q_chunk > s or k_chunk > s:
        pytest.skip("chunk larger than the sequence")
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, n, s, d) for n in (nh, nkv, nkv))
    tq, tk, tv = (ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v))
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    kwargs = dict(is_causal=False, program_config=pc, compute_kernel_config=ck)
    g = nh // nkv

    # reference: the unpacked op on the packed view of Q
    ref_packed = ttnn.transformer.scaled_dot_product_attention(ttnn.reshape(tq, [b, nkv, g * s, d]), tk, tv, **kwargs)
    ref = ttnn.to_torch(ref_packed).reshape(b, nh, s, d)

    out = ttnn.transformer.scaled_dot_product_attention(tq, tk, tv, pack_gqa_heads=True, **kwargs)
    assert tuple(out.shape) == (b, nh, s, d)
    got = ttnn.to_torch(out)
    assert torch.equal(got, ref), "packed op must be bit-identical to the unpacked op on the packed view of Q"

    out_cat = ttnn.transformer.scaled_dot_product_attention(
        tq, tk, tv, pack_gqa_heads=True, output_heads_concat=True, **kwargs
    )
    assert tuple(out_cat.shape) == (b, 1, s, nh * d)
    got_cat = ttnn.to_torch(out_cat)
    assert torch.equal(
        got_cat, ttnn.to_torch(ttnn.experimental.nlp_concat_heads(out))
    ), "packed concat layout must be bit-identical to nlp_concat_heads of the packed head-major output"

    kr, vr = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
    torch_out = torch.nn.functional.scaled_dot_product_attention(q, kr, vr, is_causal=False)
    assert_with_pcc(torch_out, got.float(), 0.99 if dtype == ttnn.bfloat8_b else 0.999)


def test_sdpa_pack_gqa_heads_rejects_causal_and_mask(device, expect_error):
    q, k, v = (
        ttnn.from_torch(torch.randn(1, n, 256, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        for n in (8, 2, 2)
    )
    mask = ttnn.from_torch(torch.zeros(1, 1, 256, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "pack_gqa_heads supports non-causal, unmasked"):
        ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, pack_gqa_heads=True)
    with expect_error(RuntimeError, "pack_gqa_heads supports non-causal, unmasked"):
        ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, attn_mask=mask, pack_gqa_heads=True)


@pytest.mark.parametrize("output_heads_concat", [False, True])
def test_sdpa_pack_gqa_heads_toggle_same_shapes(device, output_heads_concat):
    """The flag is part of the program-cache key: unpacked, packed, unpacked on the same tensors each get their own
    program. Every Q row sees the same K chunks either way, so the packed output equals the unpacked one."""
    torch.manual_seed(0)
    q, k, v = (
        ttnn.from_torch(torch.randn(1, n, 512, 128), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        for n in (32, 8, 8)
    )
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(), q_chunk_size=256, k_chunk_size=256
    )
    kwargs = dict(is_causal=False, program_config=pc, output_heads_concat=output_heads_concat)
    outs = [
        ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(q, k, v, pack_gqa_heads=p, **kwargs))
        for p in (False, True, False)
    ]
    assert torch.equal(outs[0], outs[2])
    assert torch.equal(outs[0], outs[1]), "packed and unpacked must agree bit for bit at equal chunk sizes"
