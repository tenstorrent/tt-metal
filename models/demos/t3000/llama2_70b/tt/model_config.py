# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib as ttl
import ttnn
from loguru import logger
import os


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
    assert seq_len in (1, 128, 2048, 8192)

    # Supported values, TODO update for larger TT chips
    if max_context_len > 4096:
        assert max_batch_size == 16
    else:
        assert max_batch_size == 32
    assert batch <= max_batch_size
    assert seq_len <= max_context_len

    seq_tiles = seq_len // 32

    DRAM_MEMCFG = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    L1_MEMCFG = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    WIDTH_SHARDED_MEMCFG = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED, ttl.tensor.BufferType.L1
    )
    HEIGHT_SHARDED_MEMCFG = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1
    )
    BLOCK_SHARDED_MEMCFG = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1
    )
    BFLOAT16_DTYPE = ttl.tensor.DataType.BFLOAT16
    BFP8_DTYPE = ttl.tensor.DataType.BFLOAT8_B

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
        "COMPUTE_KERNEL_CONFIG": ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
        "COMPUTE_KERNEL_CONFIG_LOFI": ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
        "LN_COMPUTE_KERNEL_CONFIG": ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        ),
        "COMPUTE_KERNEL_FP16_ACC_CONFIG": ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        ),
        "ROT_MAT_COMPUTE_KERNEL_CONFIG": ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi4,  # Highest fidelity
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
        "L1_MEMCFG": L1_MEMCFG,
        "DRAM_MEMCFG": DRAM_MEMCFG,
        "BFLOAT16_DTYPE": BFLOAT16_DTYPE,
        "BFP8_DTYPE": BFP8_DTYPE,
        "WIDTH_SHARDED_MEMCFG": WIDTH_SHARDED_MEMCFG,
        "HEIGHT_SHARDED_MEMCFG": HEIGHT_SHARDED_MEMCFG,
        "BLOCK_SHARDED_MEMCFG": BLOCK_SHARDED_MEMCFG,
        "MAX_MM_SEQ_LEN": 1024,  # Used to support seq len greater than 2k
    }
    hidden_size = model_config_entries["hidden_size"]
    head_dim = model_config_entries["head_dim"]
    n_heads = model_config_entries["num_attention_heads"]
    n_kv_heads = model_config_entries["num_kv_heads"]

    if llm_mode == "decode":
        shard_height = batch
    else:
        shard_height = seq_len

    shard_spec_64_cores_grid = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(7, 7),
            ),
        }
    )
    shard_spec_40_cores_grid = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(7, 4),
            ),
        }
    )
    shard_spec_32_cores_grid = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(7, 3),
            ),
        }
    )
    shard_spec_16_cores_grid = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(7, 1),
            ),
        }
    )
    shard_spec_8_cores_grid = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(7, 0),
            ),
        }
    )
    shard_spec_1_cores_grid = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(0, 0),
            ),
        }
    )

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

    core_range_block_sharded_layernorm = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(layernorm_num_cores_x - 1, layernorm_num_cores_y - 1),
            ),
        }
    )

    # Embeddings
    model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            shard_spec_32_cores_grid,
            [
                shard_height,
                shard_width_hidden_dim_per_device_across_32_cores,
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    # Model prepare_inputs
    model_config["ATTN_MASK_MEMCFG"] = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            shard_spec_32_cores_grid if llm_mode == "decode" else shard_spec_8_cores_grid,
            [
                shard_height,
                1,  # Dynamic - must set before using this config
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    # Llama2 Model Config
    if llm_mode == "decode":
        model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    shard_height,
                    shard_width_hidden_dim_across_32_cores,
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["LN_F_PROGCFG"] = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
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
        model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                core_range_block_sharded_layernorm,
                [
                    layernorm_shard_height_hidden_dim,
                    layernorm_shard_width_hidden_dim,
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["LN_F_PROGCFG"] = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[layernorm_num_cores_x, layernorm_max_num_cores_y],
            subblock_w=8,
            block_h=num_tiles_per_core_h,
            block_w=num_tiles_per_core_w,
            inplace=True,
        )
        model_config["LM_HEAD_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=seq_tiles // 4,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=16,  # N / TILE_WIDTH / Grid_Size
            transpose_mcast=False,
            fused_activation=None,
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
    model_config["ATTN_ADD_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            shard_spec_32_cores_grid,
            [
                shard_height,
                shard_width_hidden_dim_per_device_across_32_cores,  # 8192 // 32 // num_devices
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    model_config["LN_MLP_PROGCFG"] = model_config["LN_ATTN_PROGCFG"]
    model_config["LN_MLP_OUTPUT_MEMCFG"] = model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"]
    model_config["MLP_ADD_OUTPUT_MEMCFG"] = model_config["ATTN_ADD_OUTPUT_MEMCFG"]

    # LLama Attention Module
    model_config["FUSED_QKV_MM_INPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            shard_spec_40_cores_grid,
            [
                shard_height,
                256,
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    model_config["ATTN_ALL_GATHER_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            shard_spec_8_cores_grid,
            [
                shard_height,
                shard_width_hidden_dim_across_8_cores,
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            shard_spec_8_cores_grid,
            [
                shard_height,
                shard_width_qkv_heads_per_device_across_8_cores,
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
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

    model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            shard_spec_1_cores_grid,
            [
                shard_height,
                total_width_per_group_of_qkv_heads,  # Must always be minimum a full group
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    model_config["ROT_MAT_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=[8, 4],
        in0_block_w=4,  # 128 // TILE_SIZE (dynamic)
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=4,
    )
    model_config["ROT_MAT_MM_IN1_MEMCFG"] = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            shard_spec_32_cores_grid,
            [
                head_dim,
                head_dim,  # head dim
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    model_config["KV_CACHE_SLICE_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            shard_spec_32_cores_grid if num_devices == 8 else shard_spec_8_cores_grid,
            [
                1,  # Dynamic
                head_dim,
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    if llm_mode == "decode":
        model_config["ATTN_BATCHED_MM_PROGCFG_LAMBDA"] = lambda seq_tiles: ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=[8, 4],
            in0_block_w=head_dim // 32,  # HEAD_DIM // TILE_SIZE
            out_subblock_h=1,  # TODO: Maximize
            out_subblock_w=1,  # TODO: Maximize
            per_core_M=32 // 32,  # N_HEADS_PADDED // TILE_SIZE,
            per_core_N=seq_tiles,  # SEQ_LEN // TILE_SIZE (dynamic)
        )
        model_config["ATTN_BATCHED_MM_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    shard_height,  # Each core has 32 users
                    1,  # Dynamic (padded seqlen)
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["SCORES_BATCHED_MM_PROGCFG_LAMBDA"] = lambda seq_tiles: ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=[8, 4],
            in0_block_w=seq_tiles,  # SEQ_LEN // TILE_SIZE (dynamic)
            out_subblock_h=1,  # TODO: Maximize
            out_subblock_w=1,  # TODO: Maximize
            per_core_M=32 // 32,  # N_HEADS_PADDED // TILE_SIZE,
            per_core_N=head_dim // 32,  # HEAD_DIM // TILE_SIZE
        )
        model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    shard_height,  # Each core has 32 users
                    head_dim,  # head dim
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["SCORES_TRANSPOSED_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_8_cores_grid,  # Volume must match # of attn heads
                [
                    shard_height,  # Each core has 32 users
                    head_dim,  # head dim
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    else:
        q_chunk_size = 128
        k_chunk_size = 64 if seq_len == 128 else 256

        model_config["SDPA_PROGCFG"] = ttl.operations.primary.transformers.SDPAMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 7],
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
        )
    if llm_mode == "decode":
        model_config[
            "BATCHED_SOFTMAX_PROGCFG"
        ] = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 4),  # In-place softmax on 32 cores sharded on batch dim
            subblock_w=1,
            block_h=shard_height // 32,
            block_w=1,  # Dynamic
        )
    else:
        model_config[
            "BATCHED_SOFTMAX_PROGCFG"
        ] = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 4 if seq_len == 128 else 8),
            subblock_w=1,
            block_h=32 // 32,  # 128 * 8 // 32 cores // TILE_SIZE
            block_w=1,  # Dynamic
        )
    model_config["SELFOUT_MM_INPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            shard_spec_32_cores_grid,
            [
                shard_height,
                shard_width_hidden_dim_across_32_cores,
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
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
        model_config["PADDED_FF1_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=4,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 4096 for num_device=8
            fuse_batch=True,
            fused_activation=ttl.tensor.FusibleActivation.SILU,
            mcast_in0=True,
        )
        model_config["PADDED_FF3_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=4,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        model_config["PADDED_FF2_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=32,  # K = 32768 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
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
            fused_activation=ttl.tensor.FusibleActivation.SILU,
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
            per_core_N=4,  # N / TILE_WIDTH / Grid_Size
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        model_config["MLP_BLOCK_SHARDED_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                ttl.tensor.CoreRangeSet(
                    {
                        ttl.tensor.CoreRange(
                            ttl.tensor.CoreCoord(0, 0),
                            ttl.tensor.CoreCoord(7, cores_y - 1),
                        ),
                    }
                ),
                [
                    seq_len // cores_y,
                    4096 // 8,
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    model_config["PADDED_MLP_ALL_GATHER_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            shard_spec_32_cores_grid,
            [
                shard_height,
                shared_with_padded_mlp_dim_across_32_cores,  # 32k // 32
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
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
