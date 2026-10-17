# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Fast post-commit tests for SDPA decode with a representative subset of parametrizations.
"""

import random

import torch
import numpy as np
import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import (
    num_to_corerange,
    run_test_sdpa_decode_single_iter,
    run_test_sdpa_decode_multi_pos,
    run_test_sdpa_decode_paged_attention,
    run_test_sdpa_decode_broadcast_mask_batch,
)


@pytest.fixture(scope="function")
def reset_seeds():
    torch.manual_seed(213919)
    np.random.seed(213919)
    random.seed(213919)
    yield


def test_sdpa_decode_fp32_half_sync_cross_core_reduction(device):
    """#56171: merging two KV chunks used five DST slots, but FP32 half-sync has four."""
    torch.manual_seed(0)
    heads, kv_heads, head_dim, cache = 8, 2, 128, 128
    q = torch.randn(1, 1, heads, head_dim)
    k = torch.randn(1, kv_heads, cache, head_dim)
    v = torch.randn(1, kv_heads, cache, head_dim)
    tq, tk, tv = (ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v))
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(2, 2),
        q_chunk_size=32,
        k_chunk_size=32,
        max_cores_per_head_batch=2,  # Two cores per KV head force cross-core correction at position 32.
        exp_approx_mode=False,
    )
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )
    # Position 31 stays within one chunk; position 32 activates the second core
    # and previously corrupted the result in the fused softmax correction.
    for cur_pos in (31, 32):
        out = ttnn.transformer.scaled_dot_product_attention_decode(
            tq,
            tk,
            tv,
            cur_pos=[cur_pos],
            is_causal=True,
            program_config=program_config,
            compute_kernel_config=compute_config,
        )
        valid_k = k[:, :, : cur_pos + 1].repeat_interleave(heads // kv_heads, dim=1)
        valid_v = v[:, :, : cur_pos + 1].repeat_interleave(heads // kv_heads, dim=1)
        ref = torch.nn.functional.scaled_dot_product_attention(q.permute(0, 2, 1, 3), valid_k, valid_v)
        actual = ttnn.to_torch(out)[:, :, :heads]
        assert_with_pcc(ref.permute(0, 2, 1, 3), actual, 0.999)
        ttnn.deallocate(out)


@pytest.mark.parametrize("k_chunk", [32, 256], ids=["one-tile-chunks", "multi-tile-chunks"])
def test_sdpa_decode_fp32_dest_large_logits(device, k_chunk):
    """#44295: a shared key offset (Qwen2's k_proj bias) puts raw QK logits in the thousands. Packing the
    scores to bf16 before the running max is subtracted costs up to |score| * 2^-9, several units at 3400,
    which reorders near-tied keys. With fp32_dest_acc_en the scores now stay in fp32."""
    torch.manual_seed(0)
    heads, kv_heads, head_dim, cache = 12, 2, 128, 1024
    q = (torch.randn(1, 1, heads, head_dim) + 1.0).bfloat16().float()
    k = (torch.randn(1, kv_heads, cache, head_dim) * 8.0 + 20.0).bfloat16().float()
    v = torch.randn(1, kv_heads, cache, head_dim).bfloat16().float()
    tq, tk, tv = (ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v))
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=0,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    out = ttnn.transformer.scaled_dot_product_attention_decode(
        tq,
        tk,
        tv,
        cur_pos=[cache - 1],
        is_causal=True,
        program_config=program_config,
        compute_kernel_config=compute_config,
    )
    rk = k.repeat_interleave(heads // kv_heads, dim=1)
    rv = v.repeat_interleave(heads // kv_heads, dim=1)
    raw = q.permute(0, 2, 1, 3).double() @ rk.double().transpose(-1, -2)
    assert raw.abs().max() > 3000, "the test must exercise large raw logits"
    ref = torch.softmax(raw * head_dim**-0.5, dim=-1) @ rv.double()
    actual = ttnn.to_torch(out)[:, :, :heads]
    assert_with_pcc(ref.permute(0, 2, 1, 3).float(), actual, 0.999)


def _sdpa_decode_fp64_reference(q, k, v, cur_pos, scale, sink=None):
    """Exact softmax attention in fp64 on bf16-exact inputs. q (B, nh, d), k/v (B, nkv, S, d)."""
    batch, heads, _ = q.shape
    group = heads // k.shape[1]
    out = torch.zeros(q.shape, dtype=torch.float64)
    for b in range(batch):
        n = cur_pos[b] + 1
        kk = k[b, :, :n].double().repeat_interleave(group, 0)
        vv = v[b, :, :n].double().repeat_interleave(group, 0)
        scores = torch.einsum("hd,hsd->hs", q[b].double(), kk) * scale
        if sink is not None:
            weights = torch.softmax(torch.cat([scores, sink.double().reshape(heads, 1)], 1), -1)[:, :-1]
        else:
            weights = torch.softmax(scores, -1)
        out[b] = torch.einsum("hs,hsd->hd", weights, vv)
    return out


@pytest.mark.parametrize(
    "batch, heads, kv_heads, head_dim, cache, cur_pos, k_chunk, causal, use_sink, max_nl2",
    [
        # Half-tile Q (<=16 heads, causal): P used the approximate exp even with exp_approx_mode=False.
        (1, 8, 1, 128, 32768, 32000, 256, True, False, 0.008),
        # Full-tile Q, many cores per head: bf16 (m, l, O) partials re-rounded at every tree-reduction round.
        (1, 32, 8, 128, 16384, 16000, 128, True, False, 0.006),
        (1, 32, 8, 128, 4096, 4095, 256, False, False, 0.005),
        # Attention sink: BF16 sink CB combined with the fp32 running max/sum.
        (4, 32, 4, 64, 4096, 4000, 256, True, True, 0.005),
        (1, 8, 1, 64, 8192, 8000, 128, True, True, 0.008),
    ],
    ids=["half-tile-deep-tree", "full-tile-deep-tree", "non-causal", "sink-full-tile", "sink-half-tile"],
)
def test_sdpa_decode_fp32_intermediates_vs_fp64(
    device, batch, heads, kv_heads, head_dim, cache, cur_pos, k_chunk, causal, use_sink, max_nl2
):
    """With fp32_dest_acc_en every softmax intermediate stays fp32 (QK/P, running max/sum, partial outputs and the
    cross-core (m, l, O) partials), and exp_approx_mode=False is honored for half-tile Q. Checked against an fp64
    softmax on bf16-exact inputs by normalized L2, which PCC alone does not resolve at this precision."""
    torch.manual_seed(1234)
    q = torch.randn(batch, heads, head_dim).bfloat16().float()
    k = torch.randn(batch, kv_heads, cache, head_dim).bfloat16().float()
    v = torch.randn(batch, kv_heads, cache, head_dim).bfloat16().float()
    scale = head_dim**-0.5
    positions = [cur_pos] * batch if causal else [cache - 1] * batch
    sink = torch.randn(heads) * 4.0 if use_sink else None
    ref = _sdpa_decode_fp64_reference(q, k, v, positions, scale, sink)

    tq = ttnn.from_torch(q.reshape(1, batch, heads, head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kwargs = {"cur_pos": positions} if causal else {}
    if use_sink:
        # The kernel scales the sink with the scores, so it is passed unscaled (as GPT-OSS does).
        sink_tile = torch.nn.functional.pad(sink.reshape(heads, 1) / scale, (0, ttnn.TILE_SIZE - 1))
        kwargs["attention_sink"] = ttnn.from_torch(
            sink_tile, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=0,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    out = ttnn.transformer.scaled_dot_product_attention_decode(
        tq,
        tk,
        tv,
        is_causal=causal,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_config,
        **kwargs,
    )
    actual = ttnn.to_torch(out)[0, :, :heads].double()
    nl2 = float((actual - ref).norm() / ref.norm())
    assert_with_pcc(ref.float(), actual.float(), 0.9999)
    assert nl2 <= max_nl2, f"normalized L2 vs fp64 {nl2:.5f} > {max_nl2}"


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, single_iter, cur_pos_tensor",
    (
        [8, 8, 1, 32768, 128, (8, 6), True, False],  # Llama2-70B
        [4, 32, 8, 8192, 128, (8, 8), True, True],  # llama 3.1 8b
    ),
)
@pytest.mark.timeout(120)
def test_sdpa_decode(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, single_iter, cur_pos_tensor):
    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("nkv > 1 requires q_dtype to be bfloat16")

    if single_iter:
        run_test_sdpa_decode_single_iter(
            device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, cur_pos_tensor, sharded_in=False, sharded_out=False
        )
    else:
        run_test_sdpa_decode_multi_pos(
            device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, cur_pos_tensor, sharded_in=False, sharded_out=False
        )


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, cur_pos_tensor",
    ([2, 20, 20, 512, 64, (8, 8), True],),  # Whisper-large (nh not multiple of 32; grid must give num_cores/b <= nkv)
)
@pytest.mark.timeout(120)
def test_sdpa_decode_non_tile_aligned_heads(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, cur_pos_tensor):
    """Regression test for models with num_heads not a multiple of 32 (e.g. Whisper-large with 20 heads).

    The output logical shape must preserve the unpadded head count so that downstream
    ops like nlp_concat_heads produce the correct hidden dimension.
    """
    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("nkv > 1 requires q_dtype to be bfloat16")

    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, cur_pos_tensor, sharded_in=False, sharded_out=False
    )


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    ([1, 64, 8, 2048, 128, (8, 8)],),  # num q heads greater than 32
)
@pytest.mark.timeout(120)
def test_sdpa_decode_non_causal(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype):
    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("nkv > 1 requires q_dtype to be bfloat16")

    for _ in range(2):
        run_test_sdpa_decode_single_iter(
            device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sharded_in=False, sharded_out=False, causal=False
        )
    assert device.cache_entries_counter.total == 1


@pytest.mark.parametrize("num_chunks", [1, 2, 3], ids=["inactive-core", "one-per-core", "multiple-per-core"])
def test_sdpa_decode_non_causal_chunk_distribution(device, num_chunks):
    """Exercise both sides of the single-local-chunk specialization with two cores per KV head."""
    torch.manual_seed(1234)
    heads, kv_heads, head_dim, chunk_size = 8, 2, 64, 32
    q = torch.randn(1, 1, heads, head_dim)
    k = torch.randn(1, kv_heads, num_chunks * chunk_size, head_dim)
    v = torch.randn_like(k)
    tq, tk, tv = (ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v))
    out = ttnn.transformer.scaled_dot_product_attention_decode(
        tq,
        tk,
        tv,
        is_causal=False,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(2, 2),
            q_chunk_size=32,
            k_chunk_size=chunk_size,
            max_cores_per_head_batch=2,
            exp_approx_mode=False,
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        ),
    )
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.repeat_interleave(heads // kv_heads, dim=1),
        v.repeat_interleave(heads // kv_heads, dim=1),
    )
    assert_with_pcc(ref.permute(0, 2, 1, 3), ttnn.to_torch(out), 0.999)


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat16, ttnn.bfloat16],
    ],
    ids=[
        "all_bfp16",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, single_iter, cur_pos_tensor",
    ([32, 8, 1, 32768, 128, (8, 6), True, True],),  # Llama2-70B
)
def test_sdpa_decode_ignore_users(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, single_iter, cur_pos_tensor):
    # Set odd users to -1 to test skipping users
    start_indices = [100 if bb % 2 == 0 else -1 for bb in range(b)]

    run_test_sdpa_decode_single_iter(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        dtype,
        grid_size,
        q_dtype,
        cur_pos_tensor,
        sharded_in=False,
        sharded_out=False,
        start_indices=start_indices,
    )


@pytest.mark.parametrize(
    "kv_dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8_q_bf16",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, cur_pos_tensor, sliding_window_size",
    ([8, 16, 4, 4096, 128, (8, 2), True, None],),  # llama 3.1 8b N300
    ids=["llama3.1-a"],
)
@pytest.mark.parametrize("block_size", (64,), ids=["paged_64"])
def test_sdpa_decode_paged_attention(
    device, b, nh, nkv, s, d, kv_dtype, grid_size, q_dtype, cur_pos_tensor, sliding_window_size, block_size, reset_seeds
):
    if s == 128 * 1024 and block_size != 64:
        # 128k sequence, block_size 64 tests the sizing of the page table CB
        pytest.skip("Skipping test for seq_len=128k with block_size!=64")
    run_test_sdpa_decode_paged_attention(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        kv_dtype,
        grid_size,
        q_dtype,
        cur_pos_tensor,
        block_size=block_size,
        sharded_in=True,
        sharded_out=False,
        sliding_window_size=sliding_window_size,
    )

    assert device.num_program_cache_entries() == 4


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    [
        [32, 32, 8, 2048, 128, (10, 11)],
    ],
    ids=["blackhole_b32_nkv8"],
)
@pytest.mark.timeout(120)
def test_sdpa_decode_kv_head_core_divisibility(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype):
    """Regression test for github.com/tenstorrent/tt-metal/issues/40978.

    When floor(num_cores_available / B) yields a value that makes ceil(num_kv_heads / uncapped)
    a non-divisor of num_kv_heads, the old core allocation produced inconsistent counts causing
    a TT_FATAL crash at output core indexing.
    """
    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, cur_pos_tensor=True, sharded_in=False, sharded_out=False
    )


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat8_b],
    ],
    ids=[
        "all_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    ([1, 8, 1, 32768, 128, (8, 8)],),
)
def test_sdpa_decode_sharded(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype):
    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sharded_in=True, sharded_out=False
    )
    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sharded_in=True, sharded_out=True
    )
    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sharded_in=False, sharded_out=True
    )


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat8_b],
    ids=["bfp8"],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    ([16, 8, 1, 8192, 128],),  # Llama2-70B
)
def test_sdpa_decode_program_cache(device, b, nh, nkv, s, d, dtype, reset_seeds):
    dummy_tensors = []
    # One cur_pos vector for both outer passes: compute_program_hash includes cur_pos, so resampling
    # per iteration would compile extra programs for cur_pos_tensor=False paths (expected cache size 4).
    start_indices = np.random.randint(0, s - 1, b).tolist()
    start_indices[0] = s - 1
    for i in range(2):
        dummy_tensors.append(
            ttnn.as_tensor(
                torch.zeros(32, 32),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        dummy_tensors.append(
            ttnn.as_tensor(
                torch.zeros(1, 1, 32, 32 * 32),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet({num_to_corerange(32)}),
                        (32, 32),
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
        )
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            (8, 6),
            dtype,
            sharded_in=False,
            sharded_out=False,
            start_indices=start_indices,
            cur_pos_tensor=True,
        )
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            (8, 6),
            dtype,
            sharded_in=True,
            sharded_out=False,
            start_indices=start_indices,
            cur_pos_tensor=False,
        )
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            (8, 6),
            dtype,
            sharded_in=True,
            sharded_out=True,
            start_indices=start_indices,
            cur_pos_tensor=False,
        )
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            (8, 6),
            dtype,
            sharded_in=False,
            sharded_out=True,
            start_indices=start_indices,
            cur_pos_tensor=True,
        )

    assert device.num_program_cache_entries() == 4


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8_q_bf16",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, sliding_window_size",
    [
        [4, 8, 1, 1024, 128, (8, 4), 128],  # Medium window
    ],
)
@pytest.mark.parametrize("cur_pos_tensor", [True], ids=["cur_pos_tensor"])
@pytest.mark.timeout(120)
def test_sdpa_decode_sliding_window(
    device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sliding_window_size, cur_pos_tensor
):
    """Test sliding window attention functionality."""

    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("nkv > 1 requires q_dtype to be bfloat16")

    # Ensure sliding window is smaller than sequence length
    if sliding_window_size >= s:
        pytest.skip(f"Sliding window {sliding_window_size} must be smaller than sequence length {s}")

    # Test different window start positions to ensure all fill tile code paths are hit
    # when generating the sliding window
    k_values = [5, 10, 20, 30]

    test_positions = [
        *[
            (sliding_window_size + offset - 1) + 32 * k
            for k in k_values
            for offset in (15, 16, 17)
            # in first face, in second face (1st face completely filled), in second face (1st face completely filled + 2nd face partially filled)
        ],
        sliding_window_size * 2,
        sliding_window_size // 2,
        sliding_window_size - 1,
        s // 2,
        s - 10,
    ]
    for cur_pos in test_positions:
        if cur_pos >= s:
            continue

        # Test both cur_pos and cur_pos_tensor modes
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            grid_size,
            q_dtype,
            cur_pos_tensor=cur_pos_tensor,
            sharded_in=False,
            sharded_out=False,
            start_indices=[cur_pos + i for i in range(b)],  # test a batch with different start positions
            sliding_window_size=sliding_window_size,
        )


@pytest.mark.parametrize("mask_dtype", [ttnn.bfloat16, ttnn.bfloat4_b])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, dtype, grid_size",
    [
        (32, 8, 1, 2048, 128, ttnn.bfloat8_b, (8, 8)),
    ],
)
def test_sdpa_decode_broadcast_mask_batch(device, b, nh, nkv, s, d, dtype, grid_size, mask_dtype):
    run_test_sdpa_decode_broadcast_mask_batch(
        device,
        b=b,
        nh=nh,
        nkv=nkv,
        s=s,
        d=d,
        dtype=dtype,
        grid_size=grid_size,
        mask_dtype=mask_dtype,
    )
