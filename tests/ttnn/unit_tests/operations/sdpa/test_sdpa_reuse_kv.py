# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``scaled_dot_product_attention(..., reuse_kv=True)`` keeps K/V in a core's CBs across its consecutive Q chunks of
the same (batch, KV head) instead of re-reading them, and builds no K/V chains. Only where K/V come from changes, so
the output must be bit-identical to ``reuse_kv=False`` for every schedule: cores whose Q chunks span one or several
KV heads, packed and unpacked GQA, head-major and heads-concat output."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _run(device, b, nh, nkv, s, d, grid, q_chunk, k_chunk, concat, pack, reuse):
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, n, s, d) for n in (nh, nkv, nkv))
    tq, tk, tv = (ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v))
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid), q_chunk_size=q_chunk, k_chunk_size=k_chunk
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    out = ttnn.transformer.scaled_dot_product_attention(
        tq,
        tk,
        tv,
        is_causal=False,
        program_config=pc,
        compute_kernel_config=ck,
        output_heads_concat=concat,
        pack_gqa_heads=pack,
        reuse_kv=reuse,
    )
    return ttnn.to_torch(out), (q, k, v)


# (b, nh, nkv, s, d): the model's bs16 attention, a smaller batch, a non-4 GQA group
@pytest.mark.parametrize("b, nh, nkv, s, d", [(16, 32, 8, 512, 128), (4, 32, 8, 512, 128), (2, 16, 2, 256, 64)])
# 12x10: 4-5 Q chunks per core over one or two KV heads; 8x8 / 8x4: whole KV heads per core
@pytest.mark.parametrize("grid", [(12, 10), (8, 8), (8, 4)])
@pytest.mark.parametrize("q_chunk", [512, 256, 128])
@pytest.mark.parametrize("concat", [False, True])
@pytest.mark.parametrize("pack", [False, True])
@pytest.mark.timeout(600)
def test_sdpa_reuse_kv(device, b, nh, nkv, s, d, grid, q_chunk, concat, pack):
    if q_chunk > s:
        pytest.skip("q chunk larger than the sequence")
    ref, _ = _run(device, b, nh, nkv, s, d, grid, q_chunk, s, concat, pack, reuse=False)
    got, (q, k, v) = _run(device, b, nh, nkv, s, d, grid, q_chunk, s, concat, pack, reuse=True)
    assert torch.equal(got, ref), "reuse_kv must be bit-identical to reading K/V per Q chunk"
    if not concat and not pack and grid == (8, 8) and q_chunk == 512:
        g = nh // nkv
        torch_out = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1), is_causal=False
        )
        assert_with_pcc(torch_out, got.float(), 0.99)


def test_sdpa_reuse_kv_rejects_unsupported(device, expect_error):
    q, k, v = (
        ttnn.from_torch(torch.randn(1, n, 256, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        for n in (8, 2, 2)
    )
    with expect_error(RuntimeError, "reuse_kv supports non-causal"):
        ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, reuse_kv=True)
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(), q_chunk_size=128, k_chunk_size=128
    )
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False)
    with expect_error(RuntimeError, "reuse_kv needs a single K chunk"):
        ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, program_config=pc, compute_kernel_config=ck, reuse_kv=True
        )
