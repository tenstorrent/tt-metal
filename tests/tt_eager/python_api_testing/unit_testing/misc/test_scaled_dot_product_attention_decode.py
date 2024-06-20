# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import tt_lib
import ttnn
from loguru import logger
import pytest
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0
import math


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
    return ttnn.experimental.tensor.CoreRange(
        ttnn.experimental.tensor.CoreCoord(0, 0),
        ttnn.experimental.tensor.CoreCoord(num_x - 1, num_y - 1),
    )


def flash_attention_loop(q, K, V, mask, scale, k_chunk_size):
    seqlen = K.shape[-2]
    padded_num_heads = q.shape[-2]
    Tc = seqlen // k_chunk_size
    O = torch.zeros_like(q)
    l = torch.zeros([1, 1, padded_num_heads, 1])
    m = torch.ones([1, 1, padded_num_heads, 1]) * torch.finfo(torch.float32).min
    for t in range(Tc):
        K_chunk = K[:, :, t * k_chunk_size : (t + 1) * k_chunk_size, :]
        V_chunk = V[:, :, t * k_chunk_size : (t + 1) * k_chunk_size, :]
        mask_chunk = mask[:, :, :, t * k_chunk_size : (t + 1) * k_chunk_size]

        attn = torch.matmul(q, K_chunk.transpose(-2, -1)) * scale + mask_chunk
        m_old = m
        m = torch.max(m_old, torch.max(attn, dim=-1, keepdim=True)[0])
        P = torch.exp(attn - m)
        l = torch.exp(m_old - m) * l + torch.sum(P, dim=-1, keepdim=True)
        O = torch.matmul(P, V_chunk) + torch.matmul(torch.eye(padded_num_heads) * torch.exp(m_old - m), O)
    return O, m, l


def scaled_dot_product_attention_simulated(
    tt_Q,
    tt_K,
    tt_V,
    tt_attn_mask,
    is_causal,
    scale,
    program_config,
    valid_seq_len,
    compute_kernel_config,
    output_mem_config,
):
    # inputs
    tt_Q = ttnn.to_torch(tt_Q).to(torch.float32)
    tt_K = ttnn.to_torch(tt_K).to(torch.float32)
    tt_V = ttnn.to_torch(tt_V).to(torch.float32)
    tt_attn_mask = ttnn.to_torch(tt_attn_mask).to(torch.float32)

    # shapes
    k_chunk_size = program_config.k_chunk_size
    batch = tt_Q.shape[-3]
    head_dim = tt_Q.shape[-1]
    padded_num_heads = tt_Q.shape[-2]
    seqlen = tt_K.shape[-2]
    core_grid = program_config.compute_with_storage_grid_size
    num_cores = core_grid.x * core_grid.y

    # split to cores
    num_cores_per_batch = num_cores // batch
    num_active_cores = num_cores_per_batch * batch
    active_cores = [[i + k * num_cores_per_batch for i in range(num_cores_per_batch)] for k in range(batch)]

    # sequence length assignment
    assert valid_seq_len % k_chunk_size == 0
    num_chunks = valid_seq_len // k_chunk_size
    chunks_per_core = math.ceil(num_chunks // num_cores_per_batch)
    chunk_assignment = [[i * chunks_per_core, (i + 1) * chunks_per_core] for i in range(num_cores_per_batch)]
    chunk_assignment[-1][-1] += num_chunks % num_cores_per_batch

    # loop over batches
    output_tensor = torch.zeros_like(tt_Q)
    for b, batch_cores in enumerate(active_cores):
        O_intermed = []
        m_intermed = []
        l_intermed = []
        for i, core in enumerate(batch_cores):
            chunk_start, chunk_end = chunk_assignment[i]
            if chunk_start == chunk_end:
                continue
            O, m, l = flash_attention_loop(
                tt_Q[:, [b]],
                tt_K[:, [b], chunk_start * k_chunk_size : chunk_end * k_chunk_size, :],
                tt_V[:, [b], chunk_start * k_chunk_size : chunk_end * k_chunk_size, :],
                tt_attn_mask[:, [b], :, chunk_start * k_chunk_size : chunk_end * k_chunk_size],
                scale,
                k_chunk_size,
            )
            O_intermed.append(O)
            m_intermed.append(m)
            l_intermed.append(l)
        O, m, l = O_intermed[0], m_intermed[0], l_intermed[0]
        for O_2, m_2, l_2 in zip(O_intermed[1:], m_intermed[1:], l_intermed[1:]):
            O_1, m_1, l_1 = O, m, l
            m = torch.max(m_1, m_2)
            l = torch.exp(m_2 - m) * l_2 + torch.exp(m_1 - m) * l_1
            O = torch.matmul(torch.eye(padded_num_heads) * torch.exp(m_2 - m), O_2) + torch.matmul(
                torch.eye(padded_num_heads) * torch.exp(m_1 - m), O_1
            )
        output_tensor[:, b] = torch.matmul(torch.eye(padded_num_heads) * 1 / l, O)
    return output_tensor


def run_test_sdpa_decode(device, b, nh, nkv, s, d, dtype, grid_size):
    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    torch.manual_seed(1234)

    compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
        math_fidelity=tt_lib.tensor.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.types.MemoryConfig(ttnn.types.TensorMemoryLayout.INTERLEAVED, ttnn.types.BufferType.DRAM)
    # shard_grid = ttnn.experimental.tensor.CoreRangeSet(
    #     {
    #         # ttnn.experimental.tensor.CoreRange(
    #         #     ttnn.experimental.tensor.CoreCoord(0, 0),
    #         #     ttnn.experimental.tensor.CoreCoord(7,1),
    #         # )
    #         num_to_corerange(b)
    #     }
    # )
    # shard_spec = ttnn.experimental.tensor.ShardSpec(
    #     shard_grid, (padded_num_heads, d), ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR, False
    # )

    # height_sharded_memcfg = ttnn.types.MemoryConfig(
    #     ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    # )

    K = torch.randn(nkv, b, s, d)
    # K = torch.eye(s, d).expand(nkv, b, s, d)
    # K = torch.ones(nkv, b, s, d)
    V = torch.randn(nkv, b, s, d)
    # V = torch.eye(s, d).expand(nkv, b, s, d)
    # V = torch.ones(nkv, b, s, d)

    tt_K = ttnn.as_tensor(K, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)

    def get_chunk_size(s):
        # Got to test this!
        if s <= 32:
            return 32
        if s <= 64:
            return 32
        if s <= 128:
            return 32
        if s <= 256:
            return 256
        if s <= 2048:
            return 512
        return 1024

    # start_idx = 32

    # while start_idx < s:
    for start_idx in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        scale = d**-0.5

        k_chunk_size = get_chunk_size(start_idx)
        # k_chunk_size = 32
        program_config = tt_lib.operations.primary.transformers.SDPAMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,  # device.compute_with_storage_grid_size(),
            q_chunk_size=padded_num_heads,
            k_chunk_size=k_chunk_size,
        )

        padded_layer_len = nearest_n(start_idx, n=k_chunk_size)

        # Test various sequence lengths
        logger.info(f"Testing with sequence length: {start_idx}")
        logger.info(f"Using chunk size: {k_chunk_size}")
        logger.info(f"Using padded layer length: {padded_layer_len}")
        logger.info(f"Using padded num heads: {padded_num_heads}")

        attn_mask = torch.zeros((1, b, padded_num_heads, padded_layer_len))
        # Assume all users are at same position
        attn_mask[:, :, :, start_idx:] = torch.finfo(torch.float32).min

        Q = torch.randn(1, b, padded_num_heads, d)
        # Q = torch.eye(padded_num_heads, d).expand(1, b, padded_num_heads, d)
        # Q = torch.ones(1, b, padded_num_heads, d) * 1

        tt_Q = ttnn.as_tensor(
            Q, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg  # height_sharded_memcfg
        )
        # print(f"Q memcfg: {tt_Q.memory_config()}")

        tt_attn_mask = ttnn.as_tensor(
            attn_mask, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg
        )

        # logger.info(f"Q shape: {Q.shape}")
        # logger.info(f"K shape: {K.shape}")
        # logger.info(f"V shape: {V.shape}")
        # logger.info(f"attn_mask shape: {attn_mask.shape}")

        tt_back = tt_lib.operations.primary.transformers.scaled_dot_product_attention_decode(
            tt_Q,
            tt_K,
            tt_V,
            tt_attn_mask,
            scale=scale,
            program_config=program_config,
            valid_seq_len=padded_layer_len,
            compute_kernel_config=compute_kernel_config,
            output_mem_config=dram_memcfg,  # height_sharded_memcfg,
        )

        tt_back = ttnn.to_torch(tt_back)
        # tt_back = scaled_dot_product_attention_simulated(
        #     tt_Q,
        #     tt_K,
        #     tt_V,
        #     tt_attn_mask,
        #     is_causal=False,
        #     scale=scale,
        #     program_config=program_config,
        #     valid_seq_len=padded_layer_len,
        #     compute_kernel_config=compute_kernel_config,
        #     output_mem_config=dram_memcfg,
        # )

        tt_back = tt_back[:, :, :nh, :]

        Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d
        K_slice = K[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
        V_slice = V[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
        attn_mask_slice = attn_mask[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, S
        # logger.info("Pytorch inputs:")
        # logger.info(f"Q: {Q_slice.shape}")
        # logger.info(f"K: {K_slice.shape}")
        # logger.info(f"V: {V_slice.shape}")
        # logger.info(f"attn_mask: {attn_mask_slice.shape}")
        expect = torch.nn.functional.scaled_dot_product_attention(
            Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
        )  # b, nh, 1, d
        expect = expect.squeeze().unsqueeze(0)

        out_pass, out_pcc = comp_pcc(expect, tt_back, 0.99)

        # breakpoint()
        logger.debug(f"python vs pytorch: {out_pcc}")
        # logger.debug(f'python: {expect}')
        # logger.debug(f'tt: {tt_back}')
        # breakpoint()
        assert out_pass

        # start_idx += 32
    # attn_mask = torch.triu(attn_mask, diagonal=1).expand(b, 1, -1, -1)

    # # Print shapes of all inputs along with input names
    # logger.debug(f"Q: {Q.shape}")
    # logger.debug(f"K: {K.shape}")
    # logger.debug(f"V: {V.shape}")
    # logger.debug(f"attn_mask: {attn_mask.shape}")

    # tt_Q = tt_lib.tensor.Tensor(Q, dtype).to(tt_lib.tensor.Layout.TILE).to(device)
    # tt_K = tt_lib.tensor.Tensor(K, dtype).to(tt_lib.tensor.Layout.TILE).to(device)
    # tt_V = tt_lib.tensor.Tensor(V, dtype).to(tt_lib.tensor.Layout.TILE).to(device)
    # tt_attn_mask = tt_lib.tensor.Tensor(attn_mask, dtype).to(tt_lib.tensor.Layout.TILE).to(device)

    # tt_back = tt_lib.operations.primary.transformers.scaled_dot_product_attention(
    #     tt_Q, tt_K, tt_V, tt_attn_mask, is_causal=True, program_config=program_config
    # )
    # tt_back = tt_back.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

    # gt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask, is_causal=False)

    # out_pass, out_pcc = comp_pcc(gt, tt_back, 0.994)
    # logger.debug(f"python vs pytorch: {out_pcc}")
    # assert out_pass


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize(
    "dtype",
    [
        # tt_lib.tensor.DataType.BFLOAT8_B,
        tt_lib.tensor.DataType.BFLOAT16
    ],
    ids=[
        # "bfp8",
        "bf16"
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    (
        # [1, 1, 1, 8192, 128],  # Llama2-70B
        # [1, 8, 1, 32768, 128],  # Llama2-70B
        # [8, 8, 1, 32768, 128, (8,4)],  # Llama2-70B
        # [12, 8, 1, 32768, 128],  # Llama2-70B
        # [16, 8, 1, 32768, 128, (8,6)],  # Llama2-70B
        [16, 8, 1, 32768, 128, (8, 4)],  # Llama2-70B
        # [16, 8, 1, 32768, 128, (8,8)],  # Llama2-70B
        # [1, 8, 1, 2048, 128],  # Llama2-70B
        # [32, 16, 1, 2048, 64],  # Falcon-40B
        # [32, 71, 1, 2048, 64],  # Falcon-7B
        # [8, 8, 1, 2048, 128],  # Llama2-70B large batch
        # [1, 8, 1, 8192, 128],  # Llama2-70B large sequence
    ),
)
def test_sdpa_decode(device, b, nh, nkv, s, d, dtype, grid_size):
    tt_lib.device.DisablePersistentKernelCache()
    run_test_sdpa_decode(device, b, nh, nkv, s, d, dtype, grid_size)
