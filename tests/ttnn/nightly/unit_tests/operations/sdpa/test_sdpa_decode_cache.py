# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import (
    comp_pcc,
    fa_rand,
    flash_decode_sdpa,
    get_chunk_size,
    nearest_n,
    nearest_pow_2,
    num_to_corerange,
)


@pytest.fixture
def device_with_program_cache(device):
    # Enable program cache for the tests and reset device cache on exit.
    device.enable_program_cache()
    device.clear_program_cache()
    try:
        yield device
    finally:
        device.disable_and_clear_program_cache()


def test_mask_dtype_bf16_then_bfp8(device_with_program_cache):
    device = device_with_program_cache

    b, nh, nkv, s, d = 2, 8, 1, 1024, 64
    grid_size = (8, 4)
    grid = device.compute_with_storage_grid_size()
    if grid_size[0] > grid.x or grid_size[1] > grid.y:
        pytest.skip(f"Need compute grid at least {grid_size}, got {grid}")

    torch.manual_seed(20250322)

    padded_heads = nearest_pow_2(nearest_n(nh, n=32))
    dram = ttnn.DRAM_MEMORY_CONFIG
    q_dtype = ttnn.bfloat16
    kv_dtype = ttnn.bfloat8_b
    k_chunk_size = get_chunk_size(s // 4 + 1, s)
    scale = d**-0.5
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_heads,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    k = fa_rand(b, nkv, s, d)
    v = fa_rand(b, nkv, s, d)
    tt_k = ttnn.as_tensor(k, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_v = ttnn.as_tensor(v, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    mask_torch = torch.bernoulli(torch.full((b, nh, 1, s), 0.25)) * torch.finfo(torch.float32).min
    q = fa_rand(1, b, nh, d)
    tt_q = ttnn.as_tensor(
        q[:, :, :nh],
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram,
    )

    q_s = q[:, :, :nh, :].permute(1, 2, 0, 3)
    k_s = k[:, :, :s, :]
    k_s = torch.cat([k_s[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    v_s = v[:, :, :s, :]
    v_s = torch.cat([v_s[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    m_s = mask_torch[:, :nh, :, :]
    ref = (
        torch.nn.functional.scaled_dot_product_attention(q_s, k_s, v_s, m_s, scale=scale, is_causal=False)
        .squeeze(2)
        .unsqueeze(0)
    )

    tt_mask_bf16 = ttnn.as_tensor(
        mask_torch.transpose(1, 2).contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram,
    )
    out_bf16 = ttnn.transformer.scaled_dot_product_attention_decode(
        tt_q,
        tt_k,
        tt_v,
        is_causal=False,
        attn_mask=tt_mask_bf16,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=dram,
    )
    data_bf16 = ttnn.to_torch(out_bf16)[:, :, :nh, :]
    assert (
        device.num_program_cache_entries() == 1
    ), f"Expected 1 cache entry after first decode (BFLOAT16 mask), got {device.num_program_cache_entries()}"

    tt_mask_bfp8 = ttnn.as_tensor(
        mask_torch.transpose(1, 2).contiguous(),
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram,
    )
    out_bfp8 = ttnn.transformer.scaled_dot_product_attention_decode(
        tt_q,
        tt_k,
        tt_v,
        is_causal=False,
        attn_mask=tt_mask_bfp8,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=dram,
    )
    data_bfp8 = ttnn.to_torch(out_bfp8)[:, :, :nh, :]
    assert device.num_program_cache_entries() == 2, (
        "Expected 2 cache entries after BFLOAT8_B mask (compile-time mask dtype differs); "
        f"got {device.num_program_cache_entries()}."
    )

    min_pcc = 0.97
    for label, tt_data in (("BFLOAT16 mask", data_bf16), ("BFLOAT8_B mask", data_bfp8)):
        ok, pcc = comp_pcc(ref, tt_data, min_pcc)
        assert ok, f"{label}: output vs PyTorch PCC failed ({pcc}); possible wrong cached program."


# Share_cache must be part of compute_program_hash
def test_share_cache_false_then_true_program_cache_distinct(device_with_program_cache):
    device = device_with_program_cache

    b, nh, nkv, s, d = 1, 8, 1, 512, 64
    grid_size = (8, 4)
    grid = device.compute_with_storage_grid_size()
    if grid_size[0] > grid.x or grid_size[1] > grid.y:
        pytest.skip(f"Need compute grid at least {grid_size}, got {grid}")

    torch.manual_seed(42)
    dram = ttnn.DRAM_MEMORY_CONFIG
    padded_heads = nearest_pow_2(nearest_n(nh, n=32))
    scale = d**-0.5
    cur_pos = [min(127, s - 1)]
    k_chunk_size = get_chunk_size(cur_pos[0] + 1, s)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_heads,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    k = fa_rand(b, nkv, s, d)
    v = fa_rand(b, nkv, s, d)
    q = fa_rand(1, b, nh, d)

    tt_k = ttnn.as_tensor(k, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_v = ttnn.as_tensor(v, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_q = ttnn.as_tensor(
        q[:, :, :nh],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram,
    )

    ttnn.transformer.scaled_dot_product_attention_decode(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
        cur_pos=cur_pos,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=dram,
        share_cache=False,
    )
    assert (
        device.num_program_cache_entries() == 1
    ), f"Expected 1 cache entry after share_cache=False, got {device.num_program_cache_entries()}"

    ttnn.transformer.scaled_dot_product_attention_decode(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
        cur_pos=cur_pos,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=dram,
        share_cache=True,
    )
    assert device.num_program_cache_entries() == 2, (
        "Expected a second program-cache entry when share_cache toggles (same B=1 K/V shapes). "
        f"Got {device.num_program_cache_entries()}."
    )


def _run_decode_non_causal_with_bkv_variant(device, *, b: int, nh: int, nkv: int, s: int, d: int):
    grid_size = (8, 4)
    grid = device.compute_with_storage_grid_size()
    if grid_size[0] > grid.x or grid_size[1] > grid.y:
        pytest.skip(f"Need compute grid at least {grid_size}, got {grid}")

    padded_heads = nearest_pow_2(nearest_n(nh, n=32))
    dram = ttnn.DRAM_MEMORY_CONFIG
    q_dtype = ttnn.bfloat16
    kv_dtype = ttnn.bfloat8_b
    k_chunk_size = get_chunk_size(s // 4 + 1, s)
    scale = d**-0.5
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_heads,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    k = fa_rand(b, nkv, s, d)
    v = fa_rand(b, nkv, s, d)
    q = fa_rand(1, b, nh, d)
    mask_torch = torch.bernoulli(torch.full((b, nh, 1, s), 0.25)) * torch.finfo(torch.float32).min

    tt_q = ttnn.as_tensor(q[:, :, :nh], device=device, dtype=q_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_k = ttnn.as_tensor(k, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_v = ttnn.as_tensor(v, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_mask = ttnn.as_tensor(
        mask_torch.transpose(1, 2).contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram,
    )

    out = ttnn.transformer.scaled_dot_product_attention_decode(
        tt_q,
        tt_k,
        tt_v,
        is_causal=False,
        attn_mask=tt_mask,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=dram,
    )
    out_torch = ttnn.to_torch(out)[:, :, :nh, :]

    q_s = q[:, :, :nh, :].permute(1, 2, 0, 3)
    k_s = k[:, :, :s, :]
    k_s = torch.cat([k_s[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    v_s = v[:, :, :s, :]
    v_s = torch.cat([v_s[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    ref = (
        torch.nn.functional.scaled_dot_product_attention(
            q_s, k_s, v_s, mask_torch[:, :nh, :, :], scale=scale, is_causal=False
        )
        .squeeze(2)
        .unsqueeze(0)
    )
    return comp_pcc(ref, out_torch, 0.97)


def test_share_cache_nullopt_bkv_relation_change(device_with_program_cache):
    device = device_with_program_cache

    torch.manual_seed(20250325)

    ok1, pcc1 = _run_decode_non_causal_with_bkv_variant(device, b=8, nh=8, nkv=8, s=256, d=64)
    assert ok1, f"Run1 PCC too low: {pcc1}"
    assert (
        device.num_program_cache_entries() == 1
    ), f"Expected 1 cache entry after run1 (B=8,Bkv=8, share_cache=None), got {device.num_program_cache_entries()}"

    ok2, pcc2 = _run_decode_non_causal_with_bkv_variant(device, b=8, nh=8, nkv=1, s=256, d=64)
    assert ok2, f"Run2 PCC too low: {pcc2}"
    assert device.num_program_cache_entries() == 2, (
        "Expected cache split when Bkv relation changes under share_cache=None "
        f"(got {device.num_program_cache_entries()})."
    )


# Operation_attributes.cur_pos must be in compute_program_hash when cur_pos_tensor is None.
def test_cur_pos_list_values_change_program_cache_distinct(device_with_program_cache):
    device = device_with_program_cache

    b, nh, nkv, s, d = 2, 8, 1, 512, 64
    grid_size = (8, 4)
    grid = device.compute_with_storage_grid_size()
    if grid_size[0] > grid.x or grid_size[1] > grid.y:
        pytest.skip(f"Need compute grid at least {grid_size}, got {grid}")

    torch.manual_seed(43)
    dram = ttnn.DRAM_MEMORY_CONFIG
    padded_heads = nearest_pow_2(nearest_n(nh, n=32))
    scale = d**-0.5

    cur_pos_a = [10, 20]
    cur_pos_b = [min(300, s - 1), min(310, s - 1)]
    max_pos = max(max(cur_pos_a), max(cur_pos_b))
    k_chunk_size = get_chunk_size(max_pos + 1, s)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_heads,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    k = fa_rand(b, nkv, s, d)
    v = fa_rand(b, nkv, s, d)
    q = fa_rand(1, b, nh, d)

    tt_k = ttnn.as_tensor(k, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_v = ttnn.as_tensor(v, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_q = ttnn.as_tensor(
        q[:, :, :nh],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram,
    )

    ttnn.transformer.scaled_dot_product_attention_decode(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
        cur_pos=cur_pos_a,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=dram,
    )
    assert device.num_program_cache_entries() == 1, (
        f"Expected 1 cache entry after first causal decode (cur_pos={cur_pos_a}), "
        f"got {device.num_program_cache_entries()}"
    )

    ttnn.transformer.scaled_dot_product_attention_decode(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
        cur_pos=cur_pos_b,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=dram,
    )
    assert device.num_program_cache_entries() == 2, (
        "Causal decode with cur_pos=list (no cur_pos_tensor) must not reuse cache when "
        f"cur_pos values change. Got {device.num_program_cache_entries()} entries; "
        "compute_program_hash may omit operation_attributes.cur_pos."
    )


def test_mla_head_dim_v_change_program_cache_distinct(device_with_program_cache):
    device = device_with_program_cache

    b, nh, nkv, s = 2, 8, 1, 512
    d_qk = 128
    head_dim_v_a, head_dim_v_b = 32, 64
    torch.manual_seed(402)

    grid = device.compute_with_storage_grid_size()
    dram = ttnn.DRAM_MEMORY_CONFIG
    q_dtype = ttnn.bfloat16
    kv_dtype = ttnn.bfloat8_b

    cur_pos = [min(200, s - 1), min(250, s - 1)]
    k_chunk_size = get_chunk_size(max(cur_pos) + 1, s)
    assert k_chunk_size % 32 == 0 and s % k_chunk_size == 0, f"adjust s/k_chunk_size (got k_chunk_size={k_chunk_size})"

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        q_chunk_size=0,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
        max_cores_per_head_batch=4,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    scale = d_qk**-0.5

    q = fa_rand(b, nh, 1, d_qk)
    k = fa_rand(b, nkv, s, d_qk)
    v_a = fa_rand(b, nkv, s, head_dim_v_a)
    v_b = fa_rand(b, nkv, s, head_dim_v_b)

    tt_q = ttnn.as_tensor(
        q.permute(2, 0, 1, 3),
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram,
    )
    tt_k = ttnn.as_tensor(k, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_v_a = ttnn.as_tensor(v_a, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_v_b = ttnn.as_tensor(v_b, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)

    try:
        ttnn.transformer.flash_multi_latent_attention_decode(
            tt_q,
            tt_k,
            tt_v_a,
            head_dim_v_a,
            is_causal=True,
            cur_pos=cur_pos,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=dram,
        )
    except (RuntimeError, TypeError) as e:
        pytest.skip(f"MLA decode baseline not supported in this environment: {e}")

    assert device.num_program_cache_entries() == 1, (
        f"Expected 1 cache entry after first MLA decode (head_dim_v={head_dim_v_a}), "
        f"got {device.num_program_cache_entries()}"
    )

    try:
        ttnn.transformer.flash_multi_latent_attention_decode(
            tt_q,
            tt_k,
            tt_v_b,
            head_dim_v_b,
            is_causal=True,
            cur_pos=cur_pos,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=dram,
        )
    except RuntimeError as e:
        pytest.fail(f"Second MLA decode (head_dim_v={head_dim_v_b}) failed: {e}")

    assert device.num_program_cache_entries() == 2, (
        "MLA decode must not reuse cached program when head_dim_v / V width changes "
        f"(shared Q/K). Got {device.num_program_cache_entries()} entries; "
        "compute_program_hash may omit tensor_args.v, head_dim_v, or use_mla."
    )


def test_q_shard_height_diff_same_logical_program_cache_distinct(device_with_program_cache):
    device = device_with_program_cache

    b, nh, nkv, s, d = 1, 8, 1, 512, 64
    grid_size = (8, 4)
    grid = device.compute_with_storage_grid_size()
    if grid_size[0] > grid.x or grid_size[1] > grid.y:
        pytest.skip(f"Need compute grid at least {grid_size}, got {grid}")

    torch.manual_seed(46)
    dram = ttnn.DRAM_MEMORY_CONFIG
    padded_heads_lo = nearest_pow_2(nearest_n(nh, n=32))
    padded_heads_hi = padded_heads_lo * 2
    logical_q = (1, b, nh, d)
    padded_q_lo = (1, b, padded_heads_lo, d)
    padded_q_hi = (1, b, padded_heads_hi, d)

    cur_pos = [min(127, s - 1)]
    k_chunk_size = get_chunk_size(cur_pos[0] + 1, s)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_heads_hi,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    scale = d**-0.5

    k = fa_rand(b, nkv, s, d)
    v = fa_rand(b, nkv, s, d)
    tt_k = ttnn.as_tensor(k, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_v = ttnn.as_tensor(v, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=dram)

    q = fa_rand(1, b, nh, d)
    q_bf16 = q[:, :, :nh].to(torch.bfloat16)
    q_lo_host = torch.zeros(1, b, padded_heads_lo, d, dtype=torch.bfloat16)
    q_lo_host[:, :, :nh, :] = q_bf16
    q_hi_host = torch.zeros(1, b, padded_heads_hi, d, dtype=torch.bfloat16)
    q_hi_host[:, :, :nh, :] = q_bf16

    tt_q_raw_a = ttnn.as_tensor(
        q_lo_host,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram,
    )
    tt_q_raw_b = ttnn.as_tensor(
        q_hi_host,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram,
    )
    try:
        tt_q_a = ttnn.reshape(tt_q_raw_a, logical_q, padded_shape=padded_q_lo)
        tt_q_b = ttnn.reshape(tt_q_raw_b, logical_q, padded_shape=padded_q_hi)
    except RuntimeError as e:
        pytest.skip(f"reshape with explicit padded_shape not supported for this Q setup: {e}")

    if tt_q_a.shape != tt_q_b.shape:
        pytest.fail("Setup bug: Q logical shapes (.shape) must match for Bug #6 isolation.")
    if tt_q_a.padded_shape == tt_q_b.padded_shape:
        pytest.skip(
            "Q padded_shape still identical after host-padded upload; "
            f"got {tt_q_a.padded_shape} for both — cannot validate Bug #6 runtime path on this build."
        )

    ttnn.transformer.scaled_dot_product_attention_decode(
        tt_q_a,
        tt_k,
        tt_v,
        is_causal=True,
        cur_pos=cur_pos,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=dram,
    )
    assert device.num_program_cache_entries() == 1, (
        f"Expected 1 cache entry after first decode (Q padded heads {padded_heads_lo}), "
        f"got {device.num_program_cache_entries()}"
    )

    ttnn.transformer.scaled_dot_product_attention_decode(
        tt_q_b,
        tt_k,
        tt_v,
        is_causal=True,
        cur_pos=cur_pos,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=dram,
    )
    assert device.num_program_cache_entries() == 2, (
        "Same Q logical + K/V + program_config must not reuse cache when Q padded_shape differs "
        f"({padded_heads_lo} vs {padded_heads_hi} head padding). Got {device.num_program_cache_entries()} entries; "
        "compute_program_hash likely fails to distinguish Q padded geometry (see also qkv_logical_padded_shape_key)."
    )


def test_cur_pos_tensor_rank1_then_rank2_same_dram_same_qkv_program_cache(device_with_program_cache):
    device = device_with_program_cache

    b, nh, nkv, s, d = 2, 4, 1, 64, 64
    grid_size = (4, 2)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need grid {grid_size}, device has {compute_grid_size}")

    torch.manual_seed(204)
    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    dram = ttnn.DRAM_MEMORY_CONFIG

    shard_grid = ttnn.CoreRangeSet({num_to_corerange(b)})
    shard_spec_q = ttnn.ShardSpec(shard_grid, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR)
    height_sharded_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec_q)

    k = fa_rand(b, nkv, s, d)
    v = fa_rand(b, nkv, s, d)
    tt_k = ttnn.as_tensor(k, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_v = ttnn.as_tensor(v, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=dram)

    start_indices = [min(s // 4 + i, s - 1) for i in range(b)]
    max_start_idx = max(start_indices)
    scale = d**-0.5
    k_chunk_size = get_chunk_size(max_start_idx + 1, s)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_num_heads,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    q = fa_rand(1, b, nh, d)
    tt_q = ttnn.as_tensor(
        q[:, :, :nh],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=height_sharded_memcfg,
    )

    pos_1d = torch.tensor(start_indices, dtype=torch.int32)
    tt_pos_1d = ttnn.from_torch(
        pos_1d,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=dram,
    )

    try:
        ttnn.transformer.scaled_dot_product_attention_decode(
            tt_q,
            tt_k,
            tt_v,
            cur_pos_tensor=tt_pos_1d,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=dram,
        )
    except RuntimeError as e:
        pytest.skip(f"cur_pos rank-1 baseline failed: {e}")

    assert (
        device.num_program_cache_entries() == 1
    ), f"Expected 1 cache entry after rank-1 cur_pos_tensor, got {device.num_program_cache_entries()}"

    pos_2d = pos_1d.reshape(b, 1)
    tt_pos_2d = ttnn.from_torch(
        pos_2d,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=dram,
    )

    try:
        ttnn.transformer.scaled_dot_product_attention_decode(
            tt_q,
            tt_k,
            tt_v,
            cur_pos_tensor=tt_pos_2d,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=dram,
        )
    except RuntimeError as e:
        pytest.skip(f"Rank-2 cur_pos on DRAM not supported for this decode path (acceptable): {e}")

    assert device.num_program_cache_entries() == 2, (
        "Expected 2 program-cache entries when cur_pos_tensor logical shape changes 1D [B] -> 2D [B,1] "
        f"with identical Q/K/V (DRAM). Got {device.num_program_cache_entries()}. "
        "If cur_pos_tensor is dropped from compute_program_hash, entries can stay at 1."
    )


# ---------------------------------------------------------------------------
# Optional-operand freshness on the program-cache HIT fast path.
#
# sdpa_decode binds all 8 q/k/v/output/cur_pos/page_table/attn_mask/
# attention_sink buffer addresses as Buffer* so they are re-patched on the
# fast cache-hit path (which skips create_descriptor()). The optional operands
# (attn_mask, page_table, attention_sink) are the ones at risk of freezing:
# if any of their addresses were left as a raw uint32 scalar, a later dispatch
# that reuses the cached program with a FRESHLY-ALLOCATED optional tensor (a
# different DRAM address) would read the first dispatch's stale buffer and
# silently produce a wrong result.
#
# The two tests below re-dispatch the SAME cached program with reallocated
# optional operands (distinct addresses, distinct meaning) and assert both the
# numeric result (PCC) and a genuine cache hit each iteration. They FAIL if an
# optional buffer address froze on the fast path.
# ---------------------------------------------------------------------------


def test_attn_mask_realloc_reused_program_cache_hit(device_with_program_cache):
    """Non-causal decode: re-dispatch the cached program with a freshly-allocated
    attn_mask (distinct DRAM address, distinct masked window) on every iteration.

    Each iteration allows attention over a disjoint quarter of the key sequence,
    so the correct output is markedly different per iteration. If the attn_mask
    buffer address froze on the cache-hit fast path, iterations 1+ would read
    iteration 0's mask and fail PCC against their own reference.
    """
    device = device_with_program_cache

    b, nh, nkv, s, d = 2, 8, 1, 1024, 64
    grid_size = (8, 4)
    grid = device.compute_with_storage_grid_size()
    if grid_size[0] > grid.x or grid_size[1] > grid.y:
        pytest.skip(f"Need compute grid at least {grid_size}, got {grid}")

    torch.manual_seed(20250709)

    padded_heads = nearest_pow_2(nearest_n(nh, n=32))
    dram = ttnn.DRAM_MEMORY_CONFIG
    q_dtype = ttnn.bfloat16
    kv_dtype = ttnn.bfloat8_b
    k_chunk_size = get_chunk_size(s // 4 + 1, s)
    scale = d**-0.5
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_heads,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Fixed Q/K/V across all iterations so only the (optional) mask changes.
    k = fa_rand(b, nkv, s, d)
    v = fa_rand(b, nkv, s, d)
    q = fa_rand(1, b, nh, d)
    tt_q = ttnn.as_tensor(q[:, :, :nh], device=device, dtype=q_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_k = ttnn.as_tensor(k, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_v = ttnn.as_tensor(v, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)

    q_s = q[:, :, :nh, :].permute(1, 2, 0, 3)
    k_s = torch.cat([k[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    v_s = torch.cat([v[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)

    n_iters = 4
    window = s // n_iters
    # Large finite negative (not float32.min): each iteration fully masks entire
    # k-chunks, and float32.min overflows the flash running-max reduction to a
    # zero result. -1e4 fully suppresses masked keys (exp(-1e4)==0) without that.
    neg = -1.0e4

    # Keep every mask tensor alive so the allocator hands out a distinct DRAM
    # address for each fresh mask (a live buffer's address is never reused).
    live_masks = []
    seen_addrs = []
    min_pcc = 0.97

    for it in range(n_iters):
        # Allow only the it-th disjoint window of keys; mask everything else.
        mask_torch = torch.full((b, nh, 1, s), neg)
        mask_torch[:, :, :, it * window : (it + 1) * window] = 0.0

        tt_mask = ttnn.as_tensor(
            mask_torch.transpose(1, 2).contiguous(),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=dram,
        )
        live_masks.append(tt_mask)

        addr = tt_mask.buffer_address()
        assert addr not in seen_addrs, (
            f"Iteration {it}: fresh attn_mask reused a prior DRAM address ({addr}); "
            "cannot prove the cache-hit path re-patched a NEW optional-buffer address."
        )
        seen_addrs.append(addr)

        out = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_q,
            tt_k,
            tt_v,
            is_causal=False,
            attn_mask=tt_mask,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=dram,
        )
        out_torch = ttnn.to_torch(out)[:, :, :nh, :]

        ref = (
            torch.nn.functional.scaled_dot_product_attention(
                q_s, k_s, v_s, mask_torch[:, :nh, :, :], scale=scale, is_causal=False
            )
            .squeeze(2)
            .unsqueeze(0)
        )

        assert device.num_program_cache_entries() == 1, (
            f"Iteration {it}: expected exactly 1 program-cache entry (miss on iter 0, hit after), "
            f"got {device.num_program_cache_entries()}. Only the optional attn_mask address changed, "
            "so every dispatch after the first must be a genuine cache hit on the fast path."
        )

        ok, pcc = comp_pcc(ref, out_torch, min_pcc)
        assert ok, (
            f"Iteration {it}: output vs PyTorch PCC failed ({pcc}). On a cache HIT the attn_mask buffer "
            "address must be re-patched from its live Buffer*; a frozen (stale) address would read an "
            "earlier iteration's mask (a different key window) and produce the wrong result here."
        )


def test_page_table_realloc_reused_program_cache_hit(device_with_program_cache):
    """Paged causal decode: re-dispatch the cached program with a freshly-allocated
    page_table (distinct DRAM address, distinct block permutation) each iteration,
    keeping the LOGICAL K/V identical so the correct output is the same every time.

    Each iteration reshuffles the paged K/V cache with a new permutation and uploads
    the matching page_table. If the page_table buffer address froze on the cache-hit
    fast path, iterations 1+ would resolve blocks through iteration 0's permutation
    against the newly-shuffled cache and fail PCC.
    """
    device = device_with_program_cache

    b, nh, nkv, s, d = 2, 8, 1, 512, 64
    block_size = 64
    grid_size = (8, 4)
    grid = device.compute_with_storage_grid_size()
    if grid_size[0] > grid.x or grid_size[1] > grid.y:
        pytest.skip(f"Need compute grid at least {grid_size}, got {grid}")

    torch.manual_seed(20250710)

    dram = ttnn.DRAM_MEMORY_CONFIG
    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    scale = d**-0.5

    max_num_blocks_per_seq = s // block_size
    assert max_num_blocks_per_seq * block_size == s
    max_num_blocks = b * s // block_size

    # Fixed logical Q/K/V for the whole test; only the page_table (and the
    # physical block layout it indexes) changes across iterations.
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)
    Q = fa_rand(1, b, nh, d)

    # cur_pos supplied as a tensor -> the cur_pos scalar arg is a constant and the
    # position is read on-device, so a fixed cur_pos keeps every dispatch a cache hit.
    cur_pos = min(s - 1, s // 2)
    start_indices = [cur_pos for _ in range(b)]
    padded_layer_len = nearest_n(cur_pos + 1, n=get_chunk_size(cur_pos + 1, s))

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_num_heads,
        k_chunk_size=get_chunk_size(cur_pos + 1, s),
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    def to_paged_cache(cache):
        return (
            cache.reshape(b, nkv, max_num_blocks_per_seq, block_size, d)
            .transpose(1, 2)
            .reshape(b * max_num_blocks_per_seq, nkv, block_size, d)
        )

    paged_k = to_paged_cache(K)
    paged_v = to_paged_cache(V)

    # Causal reference is identical every iteration (logical K/V unchanged).
    Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)
    K_slice = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    V_slice = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    ref_mask = torch.zeros((b, nh, 1, s))
    for i in range(b):
        ref_mask[i, :, :, start_indices[i] + 1 :] = torch.finfo(torch.float32).min
    ref = (
        torch.nn.functional.scaled_dot_product_attention(
            Q_slice, K_slice[:, :, :s, :], V_slice[:, :, :s, :], ref_mask[:, :nh, :, :], scale=scale, is_causal=False
        )
        .squeeze(2)
        .unsqueeze(0)
    )

    tt_q = ttnn.as_tensor(Q[:, :, :nh], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_pos = ttnn.Tensor(torch.tensor(start_indices, dtype=torch.int32), ttnn.int32).to(device)

    n_iters = 3
    min_pcc = 0.93
    # Keep page_table tensors alive so each fresh upload gets a distinct DRAM address.
    live_page_tables = []
    seen_addrs = []

    for it in range(n_iters):
        permutation = torch.randperm(max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(b, max_num_blocks_per_seq)

        # Physically shuffle the cache to match this iteration's page_table.
        tt_k = ttnn.as_tensor(
            paged_k[permutation], device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=dram
        )
        tt_v = ttnn.as_tensor(
            paged_v[permutation], device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=dram
        )
        tt_page_table = ttnn.Tensor(page_table, ttnn.int32).to(device)
        live_page_tables.append(tt_page_table)

        addr = tt_page_table.buffer_address()
        assert addr not in seen_addrs, (
            f"Iteration {it}: fresh page_table reused a prior DRAM address ({addr}); "
            "cannot prove the cache-hit path re-patched a NEW optional-buffer address."
        )
        seen_addrs.append(addr)

        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_q,
            tt_k,
            tt_v,
            tt_page_table,
            cur_pos_tensor=tt_pos,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=dram,
        )
        out_torch = ttnn.to_torch(out)[:, :, :nh, :]

        assert device.num_program_cache_entries() == 1, (
            f"Iteration {it}: expected exactly 1 program-cache entry (miss on iter 0, hit after), "
            f"got {device.num_program_cache_entries()}. Only page_table/K/V addresses changed with a "
            "fixed cur_pos_tensor and program_config, so every later dispatch must be a genuine cache hit."
        )

        ok, pcc = comp_pcc(ref, out_torch, min_pcc)
        assert ok, (
            f"Iteration {it}: output vs PyTorch PCC failed ({pcc}). The logical K/V are unchanged, so the "
            "result must match every iteration; a frozen (stale) page_table address on the cache-hit path "
            "would index the freshly-shuffled cache through an earlier permutation and corrupt the output."
        )


def test_attention_sink_realloc_reused_program_cache_hit(device_with_program_cache):
    """Causal decode: re-dispatch the cached program with a freshly-allocated
    attention_sink (distinct DRAM address, distinct per-head sink values) each
    iteration, keeping Q/K/V fixed.

    The attention sink adds one extra logit per head into the softmax denominator
    (it never contributes to the output value), so its per-head value directly
    scales that head's output. Each iteration rotates which half of the heads get a
    dominant (+) sink -- suppressing those heads' output toward zero -- and which
    get a negligible (-) sink. The correct output therefore has a different set of
    near-zero heads every iteration. If the attention_sink buffer address froze on
    the cache-hit fast path (which skips create_descriptor()), iterations 1+ would
    read iteration 0's sink, suppress the wrong heads, and fail PCC against their
    own reference.
    """
    device = device_with_program_cache

    b, nh, nkv, s, d = 2, 8, 1, 256, 64
    grid_size = (8, 4)
    grid = device.compute_with_storage_grid_size()
    if grid_size[0] > grid.x or grid_size[1] > grid.y:
        pytest.skip(f"Need compute grid at least {grid_size}, got {grid}")

    torch.manual_seed(20250711)

    padded_heads = nearest_pow_2(nearest_n(nh, n=32))
    dram = ttnn.DRAM_MEMORY_CONFIG
    q_dtype = ttnn.bfloat16
    kv_dtype = ttnn.bfloat16
    scale = d**-0.5
    # cur_pos == s - 1: the (single) query at the last position attends to every key,
    # so the reference (which processes the whole K/V cache) matches the device.
    cur_pos = s - 1
    k_chunk_size = get_chunk_size(cur_pos + 1, s)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_heads,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Fixed Q/K/V across all iterations so only the (optional) attention_sink changes.
    q = fa_rand(b, nh, s, d)  # (B, nh, S, D)
    k = fa_rand(b, nkv, s, d)
    v = fa_rand(b, nkv, s, d)

    # Reference layout (see flash_decode_sdpa): heads split as (nkv, q_mult).
    ref_q = q.permute(0, 2, 1, 3).view(b, s, nkv, nh // nkv, d)
    ref_k = k.permute(0, 2, 1, 3)
    ref_v = v.permute(0, 2, 1, 3)

    # Device Q is the single last-position token, packed as (1, B, nh, D).
    tt_q_in = q[:, :, -1:, :].permute(2, 0, 1, 3)
    tt_q = ttnn.from_torch(tt_q_in, device=device, dtype=q_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_k = ttnn.from_torch(k, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_v = ttnn.from_torch(v, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)

    # Fixed cur_pos tensor (reused every iteration, so its address never moves and the
    # position is read on-device): every dispatch stays a genuine cache hit.
    tt_pos = ttnn.from_torch(torch.tensor([cur_pos] * b, dtype=torch.int32), device=device, dtype=ttnn.int32)

    n_iters = 4
    min_pcc = 0.99
    # A dominant (+) sink drives a head's output toward zero (the sink logit swamps the
    # softmax denominator); a very negative sink is a no-op (normal attention). Large,
    # finite magnitudes keep the online-softmax numerically stable.
    hi, lo = 15.0, -15.0

    # Keep every sink tensor alive so the allocator hands out a distinct DRAM address
    # for each fresh sink (a live buffer's address is never reused).
    live_sinks = []
    seen_addrs = []

    for it in range(n_iters):
        # Rotate which half of the heads are suppressed so the set of near-zero heads
        # differs every iteration -- a frozen sink would suppress the wrong heads.
        sink_per_head = torch.full((nh,), lo)
        for j in range(nh // 2):
            sink_per_head[(it + j) % nh] = hi

        # Reference sink: [b, 1, nh, 1, 1], raw (unscaled) per-head logits.
        sink_ref = sink_per_head.reshape(1, 1, nh, 1, 1).repeat(b, 1, 1, 1, 1)

        # Device sink: [nh, 1] padded to a single tile [nh, TILE_SIZE], divided by scale
        # because the kernel re-applies scale inside the exp path (GPT-OSS convention).
        tt_sink_in = sink_per_head.reshape(nh, 1)
        tt_sink_in = torch.nn.functional.pad(tt_sink_in, (0, ttnn.TILE_SIZE - 1), "constant", 0)
        tt_sink_in = tt_sink_in / scale
        tt_sink = ttnn.from_torch(
            tt_sink_in, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram
        )
        live_sinks.append(tt_sink)

        addr = tt_sink.buffer_address()
        assert addr not in seen_addrs, (
            f"Iteration {it}: fresh attention_sink reused a prior DRAM address ({addr}); "
            "cannot prove the cache-hit path re-patched a NEW optional-buffer address."
        )
        seen_addrs.append(addr)

        out = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_q,
            tt_k,
            tt_v,
            cur_pos_tensor=tt_pos,
            attention_sink=tt_sink,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=dram,
        )
        out_torch = ttnn.to_torch(out)[..., :nh, :]

        ref = flash_decode_sdpa(ref_q[:, -1:, ...], ref_k, ref_v, sink_ref, scale, sliding_window=0)

        assert device.num_program_cache_entries() == 1, (
            f"Iteration {it}: expected exactly 1 program-cache entry (miss on iter 0, hit after), "
            f"got {device.num_program_cache_entries()}. Only the optional attention_sink address changed "
            "with fixed Q/K/V/cur_pos, so every dispatch after the first must be a genuine cache hit."
        )

        ok, pcc = comp_pcc(ref, out_torch, min_pcc)
        assert ok, (
            f"Iteration {it}: output vs PyTorch PCC failed ({pcc}). On a cache HIT the attention_sink buffer "
            "address must be re-patched from its live Buffer*; a frozen (stale) address would read an earlier "
            "iteration's sink (a different set of suppressed heads) and produce the wrong result here."
        )
