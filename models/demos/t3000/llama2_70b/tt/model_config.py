# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
import os


MAX_SEQ_LEN = 4096
MAX_SEQ_LEN_LLAMA3 = 8192
MAX_SEQ_LEN_LLAMA3_1 = 128 * 1024


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


def get_model_config(
    llama_version="llama3", batch=32, seq_len=1, num_devices=8, max_batch_size=32, max_context_len=4096
):
    llm_mode = "decode" if seq_len == 1 else "prefill"
    assert num_devices == 8
    assert batch in (1, 16, 32)
    assert seq_len in (1, 32, 128, 256, 2048, 4096, 8192, 32 * 1024, 128 * 1024)

    # Supported values, TODO update for larger TT chips
    if max_context_len == 8192:
        assert max_batch_size == 16
    elif max_context_len == 128 * 1024:
        assert max_batch_size == 1
    else:
        assert max_batch_size == 32
    assert batch <= max_batch_size
    assert seq_len <= max_context_len

    seq_tiles = seq_len // 32

    DRAM_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    L1_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    WIDTH_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)
    HEIGHT_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    BLOCK_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)
    BFLOAT16_DTYPE = ttnn.bfloat16
    BFP8_DTYPE = ttnn.bfloat8_b

    # Set defaults for dtype and mem_config for all ops
    model_config = {
        "LLM_MODE": llm_mode,
        "DEFAULT_DTYPE": BFLOAT16_DTYPE,
        "DEFAULT_MEMCFG": DRAM_MEMCFG,
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
        "L1_MEMCFG": L1_MEMCFG,
        "DRAM_MEMCFG": DRAM_MEMCFG,
        "BFLOAT16_DTYPE": BFLOAT16_DTYPE,
        "BFP8_DTYPE": BFP8_DTYPE,
        "WIDTH_SHARDED_MEMCFG": WIDTH_SHARDED_MEMCFG,
        "HEIGHT_SHARDED_MEMCFG": HEIGHT_SHARDED_MEMCFG,
        "BLOCK_SHARDED_MEMCFG": BLOCK_SHARDED_MEMCFG,
        "MAX_MM_SEQ_LEN": min(seq_len, 1024),  # Used to support seq len greater than 2k
        "CORE_GRID_Y": 4
        if min(seq_len, 1024) // 32 >= 4
        else min(seq_len, 1024) // 32,  # Core grid must be ratio of seq_len // 32
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

    shard_spec_64_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 7),
            ),
        }
    )
    shard_spec_40_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 4),
            ),
        }
    )
    shard_spec_32_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 3),
            ),
        }
    )
    shard_spec_16_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 1),
            ),
        }
    )
    shard_spec_8_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 0),
            ),
        }
    )
    shard_spec_1_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(0, 0),
            ),
        }
    )

    if llm_mode == "decode":
        model_config["PADDED_BATCH_SIZE"] = 32
        shard_height = model_config["PADDED_BATCH_SIZE"]
        if batch == 1:
            batch_grid_size = [1, 1]
            batch_core_range = shard_spec_1_cores_grid
        elif batch == 16:
            batch_grid_size = [8, 2]
            batch_core_range = shard_spec_16_cores_grid
        elif batch == 32:
            batch_grid_size = [8, 4]
            batch_core_range = shard_spec_32_cores_grid
        else:
            raise NotImplementedError(f"Batch size {batch} not supported for decode mode!")
    else:
        shard_height = seq_len

    # Constants based on hidden_dim
    shard_width_hidden_dim_across_32_cores = hidden_size // 32
    shard_width_hidden_dim_per_device_across_32_cores = shard_width_hidden_dim_across_32_cores // num_devices
    shard_width_hidden_dim_across_8_cores = hidden_size // 8
    shard_width_hidden_dim_across_64_cores = hidden_size // 64

    # Constants based on head_dim
    total_width_per_group_of_qkv_heads = head_dim * ((n_heads // n_kv_heads) + 2)  # 8 q_heads + 1 k_heads + 1 v_heads
    n_local_kv_heads = n_kv_heads // num_devices
    total_width_of_qkv_heads_per_device = total_width_per_group_of_qkv_heads * n_local_kv_heads
    shard_width_qkv_heads_per_device_across_8_cores = total_width_of_qkv_heads_per_device // 8

    # Constants based on padded_mlp_dim
    padded_mlp_dim = model_config_entries["padded_mlp_dim"]
    shared_with_padded_mlp_dim_across_32_cores = padded_mlp_dim // 32

    # For Prefill. we can calculate based on the dynamic seqlen for block sharded layernorm.
    # shard_height_slice = 128 for prefill
    shard_height_slice = 512 if seq_len == 2048 else 128
    layernorm_num_cores_x = model_config["MAX_GRID_SIZE"][0]
    layernorm_max_num_cores_y = model_config["MAX_GRID_SIZE"][1]
    for i in range(layernorm_max_num_cores_y, 0, -1):
        if (shard_height_slice // 32) % i == 0:
            layernorm_num_cores_y = i
            break

    num_tiles_per_core_h = shard_height_slice // 32 // layernorm_num_cores_y
    num_tiles_per_core_w = hidden_size // 32 // layernorm_num_cores_x

    layernorm_shard_height_hidden_dim = shard_height_slice // layernorm_num_cores_y
    layernorm_shard_width_hidden_dim = hidden_size // layernorm_num_cores_x

    model_config["layernorm_params"] = {
        "layernorm_num_cores_x": layernorm_num_cores_x,
        "layernorm_num_cores_y": layernorm_num_cores_y,
        "layernorm_max_num_cores_y": layernorm_max_num_cores_y,
        "layernorm_shard_height_hidden_dim": layernorm_shard_height_hidden_dim,
        "layernorm_shard_width_hidden_dim": layernorm_shard_width_hidden_dim,
        "num_tiles_per_core_h": num_tiles_per_core_h,
        "num_tiles_per_core_w": num_tiles_per_core_w,
    }

    core_range_block_sharded_layernorm = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(layernorm_num_cores_x - 1, layernorm_num_cores_y - 1),
            ),
        }
    )

    # Embeddings
    model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_32_cores_grid,
            [
                shard_height,
                shard_width_hidden_dim_per_device_across_32_cores,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    # Model prepare_inputs
    model_config["ATTN_MASK_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_32_cores_grid if llm_mode == "decode" else shard_spec_8_cores_grid,
            [
                shard_height,
                1,  # Dynamic - must set before using this config
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    # Llama2 Model Config
    if llm_mode == "decode":
        model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_32_cores_grid,
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
    else:
        model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                core_range_block_sharded_layernorm,
                [
                    layernorm_shard_height_hidden_dim,
                    layernorm_shard_width_hidden_dim,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["LN_F_PROGCFG"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[layernorm_num_cores_x, layernorm_max_num_cores_y],
            subblock_w=8,
            block_h=num_tiles_per_core_h,
            block_w=num_tiles_per_core_w,
            inplace=True,
        )

        cores_y = 4 if seq_len == 128 else 8
        max_mm_seq_tiles = min(seq_len, model_config["MAX_MM_SEQ_LEN"]) // 32
        model_config["LM_HEAD_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, cores_y),
            in0_block_w=1,  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=max_mm_seq_tiles // cores_y,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=16,  # N / TILE_WIDTH / Grid_Size
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

        cores_y = 4 if seq_len == 128 else 8
        max_mm_seq_tiles = min(seq_len, model_config["MAX_MM_SEQ_LEN"]) // 32
        model_config["LLAMA3_LM_HEAD_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, cores_y),
            in0_block_w=1,  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=max_mm_seq_tiles // cores_y,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=64,  # 16 * 1024 // 32 // (8 if seq_len == 128 else 4) ,  # N / TILE_WIDTH / Grid_Size
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    model_config["LN_F_OUTPUT_MEMCFG"] = model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"]

    # Llama Decoder Config
    model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"] = model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"]
    model_config["LN_ATTN_PROGCFG"] = model_config["LN_F_PROGCFG"]
    model_config["LN_ATTN_OUTPUT_MEMCFG"] = model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"]
    model_config["ATTN_ADD_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_32_cores_grid,
            [
                shard_height,
                shard_width_hidden_dim_per_device_across_32_cores,  # 8192 // 32 // num_devices
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    model_config["LN_MLP_PROGCFG"] = model_config["LN_ATTN_PROGCFG"]
    model_config["LN_MLP_OUTPUT_MEMCFG"] = model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"]
    model_config["MLP_ADD_OUTPUT_MEMCFG"] = model_config["ATTN_ADD_OUTPUT_MEMCFG"]

    # LLama Attention Module
    model_config["FUSED_QKV_MM_INPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_40_cores_grid,
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
            shard_spec_8_cores_grid,
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
            shard_spec_8_cores_grid,
            [
                shard_height,
                shard_width_qkv_heads_per_device_across_8_cores,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    if llm_mode == "prefill":
        # Llama Attention Module Prefill 2D Matmul
        # Input shape is [1,1,seq_len,8192]
        # qkv_list shape is [8192,1280]
        cores_y = 4  # 8 if seq_len_tiles % 8 == 0 else 4
        max_mm_seq_tiles = min(seq_len, model_config["MAX_MM_SEQ_LEN"]) // 32
        in0_block_w = 32 if seq_len == 128 else 8  # smaller in0_block_w for larger seq_len to fit in L1)
        model_config["FUSED_QKV_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, cores_y),
            in0_block_w=in0_block_w,  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=max_mm_seq_tiles // cores_y,  # M / TILE_HEIGHT / Grid_Size
            per_core_N=5,  # N / TILE_WIDTH / Grid_Size
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    else:
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

    model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_1_cores_grid,
            [
                shard_height,
                total_width_per_group_of_qkv_heads,  # Must always be minimum a full group
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    if llm_mode == "decode":
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

    model_config["KV_CACHE_SLICE_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_32_cores_grid if num_devices == 8 else shard_spec_8_cores_grid,
            [
                1,  # Dynamic
                head_dim,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    if llm_mode == "decode":
        model_config["SDPA_DECODE_PROGRAM_CONFIG"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=[8, 4],  # Can be increased, but could result in di/dt?
            q_chunk_size=0,  # unused
            k_chunk_size=0,  # unused
        )

        model_config["SDPA_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                batch_core_range, (32, head_dim), ttnn.ShardOrientation.ROW_MAJOR, False  # padded local heads
            ),
        )

    else:
        q_chunk_size = 128
        k_chunk_size = 64 if seq_len == 128 else 256

        model_config["SDPA_PROGCFG"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=[8, 7],
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
        )

    if llm_mode == "decode":
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
    else:
        cores_y = 4  # 8 if seq_len_tiles % 8 == 0 else 4
        max_mm_seq_tiles = min(seq_len, model_config["MAX_MM_SEQ_LEN"]) // 32
        in0_block_w = 32 if seq_len == 128 else 8  # smaller in0_block_w for larger seq_len to fit in L1)
        model_config["SELFOUT_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, cores_y),
            in0_block_w=8,  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=max_mm_seq_tiles // cores_y,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=4,  # N / TILE_WIDTH / Grid_Size
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    # Llama MLP config
    # Padded MLP 32K config:
    if llm_mode == "decode":
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
    else:
        # Llama MLP Module Prefill
        cores_y = 4  # 8 if seq_tiles % 8 == 0 else 4
        max_mm_seq_tiles = min(seq_len, model_config["MAX_MM_SEQ_LEN"]) // 32
        model_config["PADDED_FF1_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, cores_y),
            in0_block_w=4,  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=max_mm_seq_tiles // cores_y,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=16,  # N / TILE_WIDTH / Grid_Size
            transpose_mcast=False,
            fused_activation=ttnn.UnaryOpType.SILU,
            fuse_batch=False,
        )

        model_config["PADDED_FF3_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, cores_y),
            in0_block_w=4,  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=max_mm_seq_tiles // cores_y,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=16,  # N / TILE_WIDTH / Grid_Size
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

        # input0: [1,32,128,32k]
        # input1: [1,1,32k,1k]
        model_config["PADDED_FF2_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, cores_y),
            in0_block_w=4,  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=max_mm_seq_tiles // cores_y,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=32,  # N / TILE_WIDTH / Grid_Size
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        model_config["MLP_BLOCK_SHARDED_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(7, cores_y - 1),
                        ),
                    }
                ),
                [
                    seq_len // cores_y,
                    4096 // 8,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    model_config["PADDED_MLP_ALL_GATHER_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_32_cores_grid,
            [
                shard_height,
                shared_with_padded_mlp_dim_across_32_cores,  # 32k // 32
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    # uncomment if need to see all the configs
    # logger.debug(f"Llama model config: \n{pretty_print_model_config(model_config)}")

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
