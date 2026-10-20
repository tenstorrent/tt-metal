# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Non-causal GQA SDPA keys its K/V forwarding chains by KV head while batch x KV heads is at most half the core count
(cores on the query heads of one KV head share its K/V) and by query head above. Chains only change where a core's K/V
bytes come from, so both keys must give bit-identical rows, and the KV-head chains must hold up under masks, sinks,
padded sequences, other head ratios and MLA's separate K and V heads."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _skip_if_too_big(device, grid):
    device_grid = device.compute_with_storage_grid_size()
    if grid[0] > device_grid.x or grid[1] > device_grid.y:
        pytest.skip(f"grid {grid} exceeds the device grid {device_grid.x}x{device_grid.y}")


def _sdpa(device, q, k, v, grid, q_chunk, k_chunk, **kwargs):
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid), q_chunk_size=q_chunk, k_chunk_size=k_chunk
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    tq, tk, tv = (ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v))
    out = ttnn.transformer.scaled_dot_product_attention(
        tq, tk, tv, is_causal=False, program_config=pc, compute_kernel_config=ck, **kwargs
    )
    return ttnn.to_torch(out)


def _torch_ref(q, k, v, attn_mask=None):
    g_k, g_v = q.shape[1] // k.shape[1], q.shape[1] // v.shape[1]
    return torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(g_k, dim=1), v.repeat_interleave(g_v, dim=1), attn_mask=attn_mask, is_causal=False
    )


# q chunks divide S, so the Q schedule (and Q-head crossing) is the same at batch 1 and the larger batch
@pytest.mark.parametrize("grid", [(8, 8), (8, 7), (8, 4), (12, 10), (11, 10)])
@pytest.mark.parametrize("q_chunk, k_chunk", [(128, 256), (256, 256), (256, 512)])
@pytest.mark.timeout(600)
def test_kv_head_chains_match_q_head_chains(device, grid, q_chunk, k_chunk):
    """Batch 1 runs KV-head chains, the smallest batch with more KV heads than half the cores Q-head chains: every
    batched row must equal its batch-1 run bit for bit."""
    _skip_if_too_big(device, grid)
    torch.manual_seed(0)
    nh, nkv, s, d = 32, 8, 512, 128
    b = grid[0] * grid[1] // (2 * nkv) + 1
    q, k, v = (torch.randn(b, n, s, d) for n in (nh, nkv, nkv))
    batched = _sdpa(device, q, k, v, grid, q_chunk, k_chunk)
    for i in range(b):
        single = _sdpa(device, q[i : i + 1], k[i : i + 1], v[i : i + 1], grid, q_chunk, k_chunk)
        assert torch.equal(single, batched[i : i + 1]), f"batch row {i}: KV-head and Q-head chains disagree"
    assert_with_pcc(_torch_ref(q, k, v), batched.float(), 0.99)


@pytest.mark.parametrize("grid", [(8, 8), (8, 7), (12, 10)])
@pytest.mark.parametrize("q_chunk, k_chunk", [(128, 256), (256, 512), (192, 512), (160, 256)])
@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.timeout(600)
def test_kv_head_chains_gqa(device, b, grid, q_chunk, k_chunk):
    _skip_if_too_big(device, grid)
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, n, 512, 128) for n in (32, 8, 8))
    assert_with_pcc(_torch_ref(q, k, v), _sdpa(device, q, k, v, grid, q_chunk, k_chunk).float(), 0.99)


@pytest.mark.parametrize("grid", [(8, 8), (12, 10)])
@pytest.mark.timeout(600)
def test_kv_head_chains_mask(device, grid):
    """Serving pad mask: the last quarter of the keys is padding. 12x10 gives unicast chains, 8x8 multicast ones."""
    _skip_if_too_big(device, grid)
    torch.manual_seed(0)
    b, s = 1, 512
    q, k, v = (torch.randn(b, n, s, 128) for n in (32, 8, 8))
    mask = torch.zeros(b, 1, s, s)
    mask[..., 3 * s // 4 :] = -1e9
    tmask = ttnn.from_torch(mask, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    out = _sdpa(device, q, k, v, grid, 256, 256, attn_mask=tmask)
    assert_with_pcc(_torch_ref(q, k, v, attn_mask=mask), out.float(), 0.99)


@pytest.mark.parametrize("grid", [(8, 8), (12, 10)])
@pytest.mark.timeout(600)
def test_kv_head_chains_sink(device, grid):
    _skip_if_too_big(device, grid)
    torch.manual_seed(0)
    nh, s, d = 32, 512, 128
    q, k, v = (torch.randn(1, n, s, d) for n in (nh, 8, 8))
    sink = torch.rand(1, nh, 1, 1) * 4.0
    tsink = ttnn.from_torch(sink, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    out = _sdpa(device, q, k, v, grid, 128, 256, attention_sink=tsink)
    # the sink joins the softmax denominator, scaled like the scores (test_sdpa_prefill's reference)
    kr, vr = k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)
    scores = torch.cat([(q @ kr.transpose(-1, -2)) / d**0.5, sink.expand(1, nh, s, 1) / d**0.5], dim=-1)
    ref = torch.softmax(scores, dim=-1)[..., :s] @ vr
    assert_with_pcc(ref, out.float(), 0.99)


@pytest.mark.parametrize("grid", [(8, 8), (12, 10)])
@pytest.mark.parametrize("q_chunk", [128, 256])
@pytest.mark.timeout(600)
def test_kv_head_chains_padded_sq(device, grid, q_chunk):
    _skip_if_too_big(device, grid)
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, n, 500, 128) for n in (32, 8, 8))
    out = _sdpa(device, q, k, v, grid, q_chunk, 256)
    assert_with_pcc(_torch_ref(q, k, v), out[:, :, :500].float(), 0.99)


@pytest.mark.parametrize("nh, nkv, s", [(32, 32, 512), (64, 8, 1024), (28, 4, 2048)])
@pytest.mark.parametrize("grid", [(8, 8), (12, 10)])
@pytest.mark.timeout(900)
def test_kv_head_chains_ratios(device, nh, nkv, s, grid):
    _skip_if_too_big(device, grid)
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, n, s, 128) for n in (nh, nkv, nkv))
    assert_with_pcc(_torch_ref(q, k, v), _sdpa(device, q, k, v, grid, 128, 256).float(), 0.99)


# (24, 4, 3): 6 and 8 Q heads per K / V head do not nest, so the chain key falls back to one Q head
@pytest.mark.parametrize("nh, nkv, nvh", [(32, 1, 32), (32, 2, 8), (32, 8, 2), (24, 4, 3)])
@pytest.mark.timeout(600)
def test_kv_head_chains_mla_distinct_kv_heads(device, nh, nkv, nvh):
    """flash_mla_prefill with a separate V: NKH != NVH. Chains must share both the K and the V head."""
    b, sq, sk, dk, dv = 1, 512, 1024, 576, 128
    torch.manual_seed(0)
    q = torch.randn(b, nh, sq, dk)
    k = torch.randn(b, nkv, sk, dk)
    v = torch.randn(b, nvh, sk, dv)
    tq, tk, tv = (ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v))
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(), q_chunk_size=64, k_chunk_size=128
    )
    out = ttnn.to_torch(
        ttnn.transformer.flash_mla_prefill(tq, tk, tv, is_causal=False, scale=dk**-0.5, program_config=pc)
    )
    ref = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(nh // nkv, dim=1),
        v.repeat_interleave(nh // nvh, dim=1),
        is_causal=False,
        scale=dk**-0.5,
    )
    assert_with_pcc(ref, out.float(), 0.99)
