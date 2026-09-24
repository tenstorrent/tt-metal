# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``scaled_dot_product_attention(..., output_heads_concat=True)`` writes the ``[B, 1, S, NQH*d]`` layout that
``nlp_concat_heads`` produces from the ``[B, NQH, S, d]`` output. Same tiles, different tile ids, so the two
must be bit-identical; the reference checks the math against torch as well."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("b, nh, nkv, s, d", [(1, 32, 8, 512, 128), (8, 32, 8, 512, 128), (2, 8, 8, 256, 64)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("q_chunk, k_chunk", [(256, 256), (512, 512)])
@pytest.mark.timeout(300)
def test_sdpa_output_heads_concat(device, b, nh, nkv, s, d, dtype, q_chunk, k_chunk):
    if q_chunk > s:
        pytest.skip("q chunk larger than the sequence")
    if dtype == ttnn.bfloat16 and q_chunk == 512:
        pytest.skip("bf16 with 512-token chunks does not fit L1 on the full grid")
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, n, s, d) for n in (nh, nkv, nkv))
    tq, tk, tv = (ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v))
    grid = device.compute_with_storage_grid_size()
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk, exp_approx_mode=False
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    kwargs = dict(is_causal=False, program_config=pc, compute_kernel_config=ck)
    out_heads = ttnn.transformer.scaled_dot_product_attention(tq, tk, tv, **kwargs)
    ref = ttnn.to_torch(ttnn.experimental.nlp_concat_heads(out_heads))
    out_cat = ttnn.transformer.scaled_dot_product_attention(tq, tk, tv, output_heads_concat=True, **kwargs)
    assert tuple(out_cat.shape) == (b, 1, s, nh * d)
    got = ttnn.to_torch(out_cat)
    assert torch.equal(got, ref), "concat layout must be bit-identical to nlp_concat_heads of the head-major output"

    # and the math itself against torch (GQA: repeat K/V heads)
    rep = nh // nkv
    kr, vr = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    torch_out = torch.nn.functional.scaled_dot_product_attention(q, kr, vr, is_causal=False)
    torch_cat = torch_out.permute(0, 2, 1, 3).reshape(b, 1, s, nh * d)
    assert_with_pcc(torch_cat, got.float(), 0.99 if dtype == ttnn.bfloat8_b else 0.999)
