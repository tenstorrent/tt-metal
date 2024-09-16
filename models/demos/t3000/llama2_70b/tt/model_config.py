# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
import os


MAX_SEQ_LEN = 4096
MAX_SEQ_LEN_LLAMA3 = 8192
MAX_SEQ_LEN_LLAMA3_1 = 128 * 1024


def num_to_corerange_set(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(num_x - 1, num_y - 1),
            ),
        }
    )


def pretty_print_model_config(model_config):
    print_str = []
    for key, val in model_config.items():
        if key.endswith("MEMCFG"):
            print_str.append(f"{key}: {val.buffer_type}")

        elif key.endswith("DTYPE") or key.endswith("BOOL"):
            print_str.append(f"{key}: {val}")

        else:
            raise NotImplementedError("Unknown key: {key}!")

    return "\n".join(print_str)


def get_model_config(llama_version="llama3", seq_len=1, num_devices=8, max_batch_size=32, max_context_len=4096):
    assert num_devices == 8
    assert max_batch_size in (1, 16, 32)
    assert seq_len % 32 == 0 or seq_len == 1, f"seq_len must be multiple of 32 or equal to 1, got {seq_len}"
    assert seq_len <= max_context_len

    # Supported values, TODO update for larger TT chips
    if max_context_len == 8192:
        assert max_batch_size == 16
    elif max_context_len == 128 * 1024:
        assert max_batch_size == 1
    else:
        assert max_batch_size == 32
    assert seq_len <= max_context_len

    L1_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    WIDTH_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)
    HEIGHT_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    BLOCK_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)
    BFLOAT16_DTYPE = ttnn.bfloat16
    BFP8_DTYPE = ttnn.bfloat8_b

    # Set defaults for dtype and mem_config for all ops
    model_config = {
        "NUM_DEVICES": num_devices,
        "MAX_GRID_SIZE": (8, 8),
        "ALL_GATHER_NUM_LINKS": 1,
        "MAX_BATCH_SIZE": max_batch_size,
        "MAX_CONTEXT_LEN": max_context_len,
        "llama3-tg": MAX_SEQ_LEN_LLAMA3,
        "llama3.1-tg": MAX_SEQ_LEN_LLAMA3_1,
        "COMPUTE_KERNEL_CONFIG": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
        "COMPUTE_KERNEL_CONFIG_LOFI": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
        "LN_COMPUTE_KERNEL_CONFIG": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        ),
        "COMPUTE_KERNEL_FP16_ACC_CONFIG": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        ),
        "ROT_MAT_COMPUTE_KERNEL_CONFIG": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,  # Highest fidelity
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
        "SDPA_COMPUTE_KERNEL_CONFIG": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,  # Highest fidelity
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        ),
        "MAX_MM_SEQ_LEN": 1024,  # Used to support seq len greater than 2k
        "HIDDEN_SIZE": model_config_entries["hidden_size"],
    }
    hidden_size = model_config_entries["hidden_size"]
    head_dim = model_config_entries["head_dim"]
    n_heads = model_config_entries["num_attention_heads"]
    n_kv_heads = model_config_entries["num_kv_heads"]

    if llama_version == "llama3" or llama_version == "llama3-tg":
        model_config["FFN_EXPANDED_HIDDEN_SIZE"] = 28 * 1024
    elif llama_version == "llama3-405b":
        model_config["FFN_EXPANDED_HIDDEN_SIZE"] = 52 * 1024

    model_config["PADDED_BATCH_SIZE"] = 32
    shard_height = model_config["PADDED_BATCH_SIZE"]
    if max_batch_size == 1:
        batch_grid_size = [1, 1]
        batch_core_range = num_to_corerange_set(max_batch_size)
    elif max_batch_size == 16:
        batch_grid_size = [8, 2]
        batch_core_range = num_to_corerange_set(max_batch_size)
    elif max_batch_size == 32:
        batch_grid_size = [8, 4]
        batch_core_range = num_to_corerange_set(max_batch_size)

    # Constants based on hidden_dim
    shard_width_hidden_dim_across_32_cores = hidden_size // 32
    shard_width_hidden_dim_per_device_across_32_cores = shard_width_hidden_dim_across_32_cores // num_devices
    shard_width_hidden_dim_across_8_cores = hidden_size // 8

    # Constants based on head_dim
    total_width_per_group_of_qkv_heads = head_dim * ((n_heads // n_kv_heads) + 2)  # 8 q_heads + 1 k_heads + 1 v_heads
    n_local_kv_heads = n_kv_heads // num_devices
    total_width_of_qkv_heads_per_device = total_width_per_group_of_qkv_heads * n_local_kv_heads
    shard_width_qkv_heads_per_device_across_8_cores = total_width_of_qkv_heads_per_device // 8

    # Constants based on padded_mlp_dim
    padded_mlp_dim = model_config_entries["padded_mlp_dim"]

    # Embeddings
    model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            num_to_corerange_set(32),
            [
                shard_height,
                shard_width_hidden_dim_per_device_across_32_cores,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            num_to_corerange_set(32),
            [
                shard_height,
                shard_width_hidden_dim_across_32_cores,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    model_config["LN_F_PROGCFG"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 4],
        subblock_w=8,
        block_h=shard_height // 32,
        block_w=8,
        inplace=True,
    )
    # LM Head
    model_config["LM_HEAD_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=4,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    model_config["LLAMA3_LM_HEAD_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=16,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    max_mm_seq_tiles = model_config["MAX_MM_SEQ_LEN"] // 32
    model_config["PREFILL_LM_HEAD_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,  # how much inner dim you take each time
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=max_mm_seq_tiles // 8,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=16,  # N / TILE_WIDTH / Grid_Size
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    model_config["PREFILL_LM_HEAD_MM_PROGCFG_128"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=1,  # how much inner dim you take each time
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=1,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=16,  # N / TILE_WIDTH / Grid_Size
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    max_mm_seq_tiles = model_config["MAX_MM_SEQ_LEN"] // 32
    model_config["PREFILL_LLAMA3_LM_HEAD_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,  # how much inner dim you take each time
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=max_mm_seq_tiles // 8,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=64,  # 16 * 1024 // 32 // (8 if seq_len == 128 else 4) ,  # N / TILE_WIDTH / Grid_Size
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    model_config["PREFILL_LLAMA3_LM_HEAD_MM_PROGCFG_128"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=1,  # how much inner dim you take each time
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=1,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=64,  # 16 * 1024 // 32 // (8 if seq_len == 128 else 4) ,  # N / TILE_WIDTH / Grid_Size
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    # Llama Decoder Config
    model_config["RESIDUAL_ADD_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            num_to_corerange_set(32),
            [
                shard_height,
                shard_width_hidden_dim_per_device_across_32_cores,  # 8192 // 32 // num_devices
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    # LLama Attention Module
    model_config["FUSED_QKV_MM_INPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            num_to_corerange_set(40),
            [
                shard_height,
                256,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    model_config["ATTN_ALL_GATHER_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            num_to_corerange_set(8),
            [
                shard_height,
                shard_width_hidden_dim_across_8_cores,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            num_to_corerange_set(8),
            [
                shard_height,
                shard_width_qkv_heads_per_device_across_8_cores,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    # Llama Attention Module Prefill 2D Matmul
    # Input shape is [1,1,seq_len,8192]
    # qkv_list shape is [8192,1280]
    max_mm_seq_tiles = model_config["MAX_MM_SEQ_LEN"] // 32
    in0_block_w = 8  # smaller in0_block_w for larger seq_len to fit in L1)
    model_config["PREFILL_FUSED_QKV_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=in0_block_w,  # how much inner dim you take each time
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=max_mm_seq_tiles // 8,  # M / TILE_HEIGHT / Grid_Size
        per_core_N=5,  # N / TILE_WIDTH / Grid_Size
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    in0_block_w = 32
    model_config["PREFILL_FUSED_QKV_MM_PROGCFG_128"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=in0_block_w,  # how much inner dim you take each time
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=1,  # M / TILE_HEIGHT / Grid_Size
        per_core_N=5,  # N / TILE_WIDTH / Grid_Size
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    model_config["FUSED_QKV_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 5),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=shard_height // 32,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    model_config["ROT_MAT_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=batch_grid_size,
        in0_block_w=4,  # 128 // TILE_SIZE (dynamic)
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=4,
    )
    model_config["ROT_MAT_MM_IN1_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            batch_core_range,
            [
                head_dim,
                head_dim,  # head dim
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    model_config["SDPA_DECODE_PROGRAM_CONFIG"] = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=[8, 4],  # Can be increased, but could result in di/dt?
        q_chunk_size=0,  # unused
        k_chunk_size=0,  # unused
    )

    model_config["SDPA_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(batch_core_range, (32, head_dim), ttnn.ShardOrientation.ROW_MAJOR, False),  # padded local heads
    )

    model_config["SELFOUT_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=8,  # (32 x 8k) x (8k x 1k) = (32 x 1k)
        out_subblock_h=1,
        out_subblock_w=1,  # TODO: Maximize
        per_core_M=shard_height // 32,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    max_mm_seq_tiles = model_config["MAX_MM_SEQ_LEN"] // 32
    in0_block_w = 8  # smaller in0_block_w for larger seq_len to fit in L1)
    model_config["PREFILL_SELFOUT_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=8,  # how much inner dim you take each time
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=max_mm_seq_tiles // 8,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=4,  # N / TILE_WIDTH / Grid_Size
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    in0_block_w = 32  # smaller in0_block_w for larger seq_len to fit in L1)
    model_config["PREFILL_SELFOUT_MM_PROGCFG_128"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=8,  # how much inner dim you take each time
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=1,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=4,  # N / TILE_WIDTH / Grid_Size
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    # Llama MLP config
    # Padded MLP 32K config:
    model_config["PADDED_FF3_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=8,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
        per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
        per_core_N=4,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
        fused_activation=None,
    )
    model_config["PADDED_FF2_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=4,  # K = 32768 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
        per_core_M=1,
        per_core_N=8,
        fused_activation=None,
    )

    # Llama MLP Module Prefill
    max_mm_seq_tiles = model_config["MAX_MM_SEQ_LEN"] // 32
    model_config["PREFILL_PADDED_FF1_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=4,  # how much inner dim you take each time
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=max_mm_seq_tiles // 8,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=16,  # N / TILE_WIDTH / Grid_Size
        transpose_mcast=False,
        fused_activation=ttnn.UnaryOpType.SILU,
        fuse_batch=False,
    )

    model_config["PREFILL_PADDED_FF1_MM_PROGCFG_128"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=4,  # how much inner dim you take each time
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=1,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=16,  # N / TILE_WIDTH / Grid_Size
        transpose_mcast=False,
        fused_activation=ttnn.UnaryOpType.SILU,
        fuse_batch=False,
    )

    model_config["PREFILL_PADDED_FF3_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=4,  # how much inner dim you take each time
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=max_mm_seq_tiles // 8,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=16,  # N / TILE_WIDTH / Grid_Size
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    model_config["PREFILL_PADDED_FF3_MM_PROGCFG_128"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=4,  # how much inner dim you take each time
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=1,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=16,  # N / TILE_WIDTH / Grid_Size
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    # input0: [1,32,128,32k]
    # input1: [1,1,32k,1k]
    model_config["PREFILL_PADDED_FF2_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=4,  # how much inner dim you take each time
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=max_mm_seq_tiles // 8,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=32,  # N / TILE_WIDTH / Grid_Size
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    model_config["PREFILL_PADDED_FF2_MM_PROGCFG_128"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=4,  # how much inner dim you take each time
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=1,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=32,  # N / TILE_WIDTH / Grid_Size
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    return model_config


model_config_entries = {
    "hidden_size": 8192,
    "head_dim": 128,
    "num_attention_heads": 64,
    "num_kv_heads": 8,
    "num_layers": 80,
    "weight_cache": True,
    "vocab_size": 32000,
    "padded_vocab_size": 32768,
    "mlp_dim": 28672,
    "padded_mlp_dim": 32768,
    "layer_norm_epsilon": 1e-05,
}
