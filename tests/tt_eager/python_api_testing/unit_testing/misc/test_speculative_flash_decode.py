# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
    comp_and_get_pcc,
)
import ttnn
from loguru import logger
import pytest
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0, skip_for_blackhole
import math
import numpy as np
from tqdm import tqdm


def is_watcher_enabled():
    return os.environ.get("TT_METAL_WATCHER") is not None


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    import math

    power = math.ceil(math.log2(x))
    return 1 << power
    # if (2**math.log2(x) == x):
    #     return x
    # return 2**(int(x).bit_length())


def num_to_corerange(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(num_x - 1, num_y - 1),
    )


def get_chunk_size(max_start_pos, s):
    if max_start_pos <= 32:
        chunk_size = 32
    elif max_start_pos <= 64:
        chunk_size = 32
    elif max_start_pos <= 128:
        chunk_size = 32
    elif max_start_pos <= 1024:
        chunk_size = 128
    else:
        chunk_size = 512
    # find maximum power of 2 divisor of s
    for i in range(1, s):
        if s % (2 ** (i + 1)) != 0:
            break
    chunk_size = min(chunk_size, 2**i)
    return chunk_size


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def prepare_test_config_and_data(
    b, nh, s, d, grid_size, padded_num_heads, speculation_length, max_start_idx, start_indices=None, causal=True
):
    """
    Prepare test configuration and input data for speculative flash decode testing.

    Returns:
        Tuple of (program_config, padded_layer_len, attn_mask, Q)
    """

    # Configure chunk size and program
    k_chunk_size = speculation_length  # 128#get_chunk_size(max_start_idx + 1, s)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_num_heads,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    # Calculate padded layer length
    padded_layer_len = nearest_n(max_start_idx + 1, n=k_chunk_size) if causal else s

    # Log debug information
    logger.debug(f"Testing with max position: {max_start_idx if causal else s}")
    logger.debug(f"Using chunk size: {k_chunk_size}")
    logger.debug(f"Using padded layer length: {padded_layer_len}")
    logger.debug(f"Using padded num heads: {padded_num_heads}")

    # Create attention mask
    if causal:
        attn_mask = torch.zeros((b, nh, 1, padded_layer_len))
        for i in range(b):
            start_idx = start_indices[i]
            attn_mask[i, :, :, start_idx + 1 :] = torch.finfo(torch.float32).min
    else:
        attn_mask = torch.bernoulli(
            torch.full(
                (b, nh, 1, padded_layer_len),
                0.25,
            )
        )
        attn_mask = attn_mask * torch.finfo(torch.float32).min

    # Generate input tensor Q
    Q = fa_rand(1, b, nh, d)

    return program_config, padded_layer_len, attn_mask, Q


def get_speculative_flash_decode_tt(
    tt_Q,
    tt_K,
    tt_V,
    device,
    start_indices,
    nh,
    lambda_,
    speculation_length,
    scale,
    program_config,
    compute_kernel_config,
    memory_config,
    causal=True,
    cur_pos_tensor=False,
    sharded_out=False,
    height_sharded_memcfg=None,
):
    """
    Wrapper function for speculative flash decode tensor operations.
    Returns tuple of (gt, spec, spec_lp_distance, lp_norm_x) tensors.
    """
    if causal:
        if cur_pos_tensor:
            start_indices_tt = ttnn.Tensor(torch.tensor(start_indices), ttnn.int32).to(device)
            outputs = ttnn.experimental.speculative_scaled_dot_product_attention_decode(
                tt_Q,
                tt_K,
                tt_V,
                lambda_=lambda_,
                speculative_chunk_size=speculation_length,
                cur_pos_tensor=start_indices_tt,
                scale=scale,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=height_sharded_memcfg if sharded_out else memory_config,
            )
        else:
            outputs = ttnn.experimental.speculative_scaled_dot_product_attention_decode(
                tt_Q,
                tt_K,
                tt_V,
                lambda_=lambda_,
                speculative_chunk_size=speculation_length,
                cur_pos=start_indices,
                scale=scale,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=height_sharded_memcfg if sharded_out else memory_config,
            )
    else:
        raise NotImplementedError("Non-causal not implemented")

    tt_back_gt, tt_back_spec, tt_back_spec_lp_distance, tt_back_lp_norm_x = outputs

    # Convert to torch and slice to correct number of heads
    tt_back_gt = ttnn.to_torch(tt_back_gt)[:, :, :nh, :]
    tt_back_spec = ttnn.to_torch(tt_back_spec)[:, :, :nh, :]
    tt_back_spec_lp_distance = ttnn.to_torch(tt_back_spec_lp_distance)
    tt_back_lp_norm_x = ttnn.to_torch(tt_back_lp_norm_x)

    return tt_back_gt, tt_back_spec, tt_back_spec_lp_distance, tt_back_lp_norm_x


def get_speculative_flash_decode_expected(
    Q, K, V, attn_mask, start_indices, nh, nkv, speculation_length, scale, max_start_idx, padded_layer_len, lambda_
):
    """
    Calculate ground truth and speculative outputs for flash decode using PyTorch.

    Returns:
        Tuple of (expected_gt, expected_spec, lp_distance, lp_norm_x) tensors
    """
    b = Q.shape[1]
    Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d

    # Prepare K and V slices
    K_slice = K[:, :, :padded_layer_len, :]  # b, nkv, S, d
    K_slice = torch.cat(
        [K_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, d, S

    V_slice = V[:, :, :padded_layer_len, :]  # b, nkv, S, d
    V_slice = torch.cat(
        [V_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, d, S

    # Calculate ground truth
    attn_mask_slice = attn_mask[:, :nh, :, :]  # b, nh, 1, S
    expected_gt = torch.nn.functional.scaled_dot_product_attention(
        Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
    )  # b, nh, 1, d
    expected_gt = expected_gt.squeeze(2).unsqueeze(0)

    # Calculate speculative output
    expected_spec = torch.zeros_like(Q_slice)
    for i in range(b):
        start_idx = start_indices[i]
        padded_start_idx = nearest_n(max_start_idx + 1, n=32)
        spec_last_chunk_start = padded_start_idx - speculation_length

        Q_slice_i = Q_slice[[i]]
        K_slice_i = torch.cat(
            [K_slice[[i], :, :speculation_length, :], K_slice[[i], :, spec_last_chunk_start:padded_start_idx, :]],
            dim=2,
        )
        V_slice_i = torch.cat(
            [V_slice[[i], :, :speculation_length, :], V_slice[[i], :, spec_last_chunk_start:padded_start_idx, :]],
            dim=2,
        )
        attn_mask_slice_i = torch.cat(
            [
                attn_mask_slice[[i], :, :, :speculation_length],
                attn_mask_slice[[i], :, :, spec_last_chunk_start:padded_start_idx],
            ],
            dim=3,
        )

        expected_spec[i] = torch.nn.functional.scaled_dot_product_attention(
            Q_slice_i, K_slice_i, V_slice_i, attn_mask_slice_i, scale=scale, is_causal=False
        )
    expected_spec = expected_spec.squeeze(2).unsqueeze(0)

    # Calculate error metrics
    lp_distance = torch.linalg.vector_norm(expected_gt - expected_spec, ord=2, dim=(-2, -1))
    lp_norm_x = torch.linalg.vector_norm(expected_gt, ord=2, dim=(-2, -1))

    return expected_gt, expected_spec, lp_distance, lp_norm_x


def run_test_sdpa_decode_single_device(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    dtype,
    grid_size,
    q_dtype=ttnn.bfloat16,
    cur_pos_tensor=False,
    speculation_length=128,
    lambda_=0.2,
    sharded_in=False,
    sharded_out=False,
    start_indices=None,
    causal=True,
    single_iter=True,
    debug_mode=False,
    check_nd_pcc_iters=None,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y - 1:
        pytest.skip(
            f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size} with last column dedicated for ccl"
        )

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    torch.manual_seed(1234)

    num_parallel_cores = grid_size[0] * grid_size[1] // b
    if num_parallel_cores == 1:
        min_pcc = 0.90
    else:
        min_pcc = 0.99
        if q_dtype == ttnn.bfloat8_b:
            min_pcc = 0.98
        min_pcc = 0.91 if dtype == ttnn.bfloat4_b else min_pcc

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    shard_grid = ttnn.CoreRangeSet({num_to_corerange(b)})
    shard_spec = ttnn.ShardSpec(shard_grid, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR, False)

    height_sharded_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    tt_K = ttnn.as_tensor(K, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)

    scale = d**-0.5
    min_start_idx = 2 * speculation_length
    max_start_idx = (max(start_indices) if start_indices is not None else s // 2 - 1) if single_iter else min_start_idx

    # for debugging
    error_runs = []

    while max_start_idx < s:
        # Set start indices if not provided or in multi-iteration mode
        if not single_iter or start_indices is None:
            start_indices = [
                max_start_idx for _ in range(b)
            ]  # np.linspace(min_start_idx, max_start_idx, b, dtype=np.int32).tolist() if b > 1 else [max_start_idx]

        ##########################################
        #### Prepare test config and data
        ##########################################
        program_config, padded_layer_len, attn_mask, Q = prepare_test_config_and_data(
            b, nh, s, d, grid_size, padded_num_heads, speculation_length, max_start_idx, start_indices, causal
        )

        ##########################################
        #### TT Calculation ####
        ##########################################

        tt_Q = ttnn.as_tensor(
            Q[:, :, :nh],
            device=device,
            dtype=q_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=height_sharded_memcfg if sharded_in else dram_memcfg,
        )

        tt_back_gt, tt_back_spec, tt_back_spec_lp_distance, tt_back_lp_norm_x = get_speculative_flash_decode_tt(
            tt_Q,
            tt_K,
            tt_V,
            device,
            start_indices,
            nh,
            lambda_,
            speculation_length,
            scale,
            program_config,
            compute_kernel_config,
            dram_memcfg,
            cur_pos_tensor=cur_pos_tensor,
            sharded_out=sharded_out,
            height_sharded_memcfg=height_sharded_memcfg,
        )
        if check_nd_pcc_iters is not None:
            assert check_nd_pcc_iters > 0
            for _ in tqdm(range(check_nd_pcc_iters)):
                (
                    tt_back_gt_new,
                    tt_back_spec_new,
                    tt_back_spec_lp_distance_new,
                    tt_back_lp_norm_x_new,
                ) = get_speculative_flash_decode_tt(
                    tt_Q,
                    tt_K,
                    tt_V,
                    device,
                    start_indices,
                    nh,
                    lambda_,
                    speculation_length,
                    scale,
                    program_config,
                    compute_kernel_config,
                    dram_memcfg,
                    cur_pos_tensor=cur_pos_tensor,
                    sharded_out=sharded_out,
                    height_sharded_memcfg=height_sharded_memcfg,
                )
                assert torch.all(tt_back_gt_new == tt_back_gt)
                assert torch.all(tt_back_spec_new == tt_back_spec)
                # assert torch.all(tt_back_spec_lp_distance_new == tt_back_spec_lp_distance)
                # assert torch.all(tt_back_lp_norm_x_new == tt_back_lp_norm_x)

        ##########################################
        #### Expected Calculation ####
        ##########################################
        expected_gt, expected_spec, lp_distance, lp_norm_x = get_speculative_flash_decode_expected(
            Q,
            K,
            V,
            attn_mask,
            start_indices,
            nh,
            nkv,
            speculation_length,
            scale,
            max_start_idx,
            padded_layer_len,
            lambda_,
        )
        passing = torch.all(lp_distance < lambda_ * lp_norm_x)
        logger.debug(f"gt speculation passing: {passing}")

        ##########################################
        #### Comparison ####
        ##########################################

        non_skip_indices = torch.tensor(start_indices) != -1
        out_pass, out_pcc = comp_pcc(expected_gt[:, non_skip_indices], tt_back_gt[:, non_skip_indices], min_pcc)
        logger.debug(f"gt tt vs pytorch: {out_pcc}")
        if debug_mode:
            if not out_pass:
                logger.warning(f"pcc check failed for {start_indices}")
                error_runs.append((start_indices, "gt", out_pcc))
        else:
            assert out_pass

        out_pass, out_pcc = comp_pcc(expected_spec[:, non_skip_indices], tt_back_spec[:, non_skip_indices], min_pcc)
        logger.debug(f"spec tt vs pytorch: {out_pcc}")
        if debug_mode:
            if not out_pass:
                logger.warning(f"pcc check failed for {start_indices}")
                error_runs.append((start_indices, "spec", out_pcc))
        else:
            assert out_pass

        min_frac_tol = 0.25 if torch.all(lp_distance.squeeze() > 2) else 0.5
        out_pass = torch.allclose(
            lp_distance.squeeze(), tt_back_spec_lp_distance.to(torch.float32).squeeze() ** (0.5), rtol=min_frac_tol
        )
        logger.debug(
            f"lp distance output tt vs pytorch: {lp_distance.squeeze()}, {tt_back_spec_lp_distance.to(torch.float32).squeeze()**(0.5)}"
        )
        assert out_pass

        min_frac_tol = 0.25 if torch.all(lp_norm_x.squeeze() > 2) else 0.5
        out_pass = torch.allclose(
            lp_norm_x.squeeze(), tt_back_lp_norm_x.to(torch.float32).squeeze() ** (0.5), rtol=min_frac_tol
        )
        logger.debug(
            f"lp norm output tt vs pytorch: {lp_norm_x.squeeze()}, {tt_back_lp_norm_x.to(torch.float32).squeeze()**(0.5)}"
        )
        assert out_pass

        if single_iter:
            break

        max_start_idx += 71 if max_start_idx < 4096 else 3001

    if debug_mode:
        print("PRINTING ERROR RUNS")
        for run in error_runs:
            print("--------------------------------")
            print(f"start_indices: {run[0]}")
            print(f"type: {run[1]}")
            print(f"pcc: {run[2]}")
        if len(error_runs) == 0:
            print("No errors found")
        else:
            assert False


@skip_for_blackhole("Unsupported on BH, see #12349")
@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
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
        # [8, 8, 1, 32768, 128, (8, 7), True, False],  # Llama2-70B
        # [4, 32, 8, 8192, 128, (8, 7), True, True],  # llama 3.1 8b
        # [1, 32, 8, 8192, 128, (8, 7), False, True],  # llama 3.1 8b
        [1, 32, 8, 16 * 8192, 128, (8, 4), False, True],  # llama 3.1 8b
        # [2, 32, 8, 256, 64, (8, 4), False, True],
        # [2, 8, 4, 256, 64, (8, 1), True, True],
        # [1, 8, 1, 256, 64, (2, 1), False, True],
        # [2, 8, 4, 256, 64, (8, 4), False, True],
    ),
)
@pytest.mark.parametrize(
    "speculation_length",
    [
        128,
    ],
)
def test_sdpa_decode_single_device(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    dtype,
    grid_size,
    q_dtype,
    single_iter,
    cur_pos_tensor,
    speculation_length,
    use_program_cache,
):
    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("nkv > 1 requires q_dtype to be bfloat16")

    ttnn.device.DisablePersistentKernelCache()

    run_test_sdpa_decode_single_device(
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
        speculation_length,
        sharded_in=False,
        sharded_out=False,
        single_iter=single_iter,
        debug_mode=False,
        # check_nd_pcc_iters=5000,
    )
