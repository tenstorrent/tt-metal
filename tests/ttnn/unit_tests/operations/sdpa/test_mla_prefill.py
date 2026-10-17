# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import run_flash_mla_prefill_impl


@pytest.mark.parametrize(
    "batch, seq_len, nh, nkv, kv_lora_rank, d_rope",
    # batch, seq_len, num heads q, num heads kv, kv lora rank, dim rope
    [
        (2, 1024, 128, 1, 512, 64),  # large head count
        (2, 4096, 64, 1, 256, 0),  # zero d_rope
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype, use_paged_attention, block_size",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b, True, 128),
        (ttnn.bfloat8_b, ttnn.bfloat4_b, True, 32),
    ],
)
def test_flash_mla_prefill(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    use_paged_attention,
    block_size,
    function_level_defaults,
    reset_seeds,
):
    run_flash_mla_prefill_impl(
        device,
        batch,
        seq_len,
        nh,
        nkv,
        kv_lora_rank,
        d_rope,
        q_dtype,
        dtype,
        use_paged_attention,
        block_size,
    )


def test_chunked_flash_mla_prefill_partial_single_page_paged_cache(
    device,
    function_level_defaults,
    reset_seeds,
):
    run_flash_mla_prefill_impl(
        device,
        batch=1,
        seq_len=64,
        nh=32,
        nkv=1,
        kv_lora_rank=64,
        d_rope=64,
        q_dtype=ttnn.bfloat16,
        dtype=ttnn.bfloat8_b,
        use_paged_attention=True,
        block_size=64,
    )


@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("chunk", [64, 128])
def test_chunked_flash_mla_prefill_separate_v(device, cache_dtype, chunk, function_level_defaults, reset_seeds):
    """Chunked MLA prefill over separate paged K (192 wide) and V (128 wide) caches, as MiMo's global layers keep
    them: every chunk attends to the earlier chunks and itself through a shuffled page table."""
    import torch

    from models.common.utility_functions import comp_pcc

    nh, dk, dv, block, seq = 16, 192, 128, 64, 512
    num_blocks = seq // block
    scale = dk**-0.5
    q = torch.randn(1, nh, seq, dk)
    k = torch.randn(1, 1, seq, dk)
    v = torch.randn(1, 1, seq, dv)
    perm = torch.randperm(num_blocks)

    def paged(t):
        blocks = t[0, 0].reshape(num_blocks, block, -1)
        cache = torch.zeros(num_blocks, 1, block, t.shape[-1])
        cache[perm, 0] = blocks
        return ttnn.from_torch(cache, device=device, dtype=cache_dtype, layout=ttnn.TILE_LAYOUT)

    k_cache, v_cache = paged(k), paged(v)
    page_table = ttnn.from_torch(perm.reshape(1, -1).to(torch.int32), device=device, dtype=ttnn.int32)
    program = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=chunk,
        k_chunk_size=chunk,
        exp_approx_mode=False,
    )
    compute = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    k_ref = ttnn.to_torch(k_cache).float()[perm, 0].reshape(1, 1, seq, dk)
    v_ref = ttnn.to_torch(v_cache).float()[perm, 0].reshape(1, 1, seq, dv)
    for start in range(0, seq, chunk):
        q_chunk = q[:, :, start : start + chunk]
        out = ttnn.transformer.chunked_flash_mla_prefill(
            ttnn.from_torch(q_chunk, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            k_cache,
            dv,
            page_table,
            start,
            scale=scale,
            program_config=program,
            compute_kernel_config=compute,
            input_tensor_v=v_cache,
        )
        got = ttnn.to_torch(out).float()
        end = start + chunk
        scores = (q_chunk @ k_ref[:, :, :end].transpose(-1, -2)) * scale
        pos_q = torch.arange(start, end)[:, None]
        scores = scores.masked_fill(torch.arange(end)[None, :] > pos_q, float("-inf"))
        expected = torch.softmax(scores, dim=-1) @ v_ref[:, :, :end]
        assert got.shape == expected.shape, (got.shape, expected.shape)
        passing, message = comp_pcc(expected, got, 0.999)
        assert passing, f"chunk at {start}: {message}"
