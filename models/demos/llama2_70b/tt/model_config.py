# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib as ttl
from loguru import logger
import os
from pathlib import Path
import os

OP_KEYS = (
    # Inputs
    "INPUT",
    "ATTN_MASK",
    "KV_CACHE",
    # Embeddings
    "WORD_EMBEDDING_WEIGHTS",
    "WORD_EMBEDDING_OUTPUT",
    # Decoder
    "LN_ATTN_WEIGHTS",
    "LN_ATTN_OUTPUT",
    "LN_MLP_WEIGHTS",
    "LN_MLP_OUTPUT",
    # Rotary
    "ROT_MAT",
    # Attention
    "FUSED_QKV_MM_OUTPUT",
    "PRE_SOFTMAX_MM_OUTPUT",
    "ATTN_BATCHED_MM_OUTPUT",
    "SOFTMAX_OUTPUT",
    "POST_SOFTMAX_MM_OUTPUT",
    "SCORES_BATCHED_MM_OUTPUT",
    "SELFOUT_MM_OUTPUT",
    # MLP
    "PADDED_FF1_MM_OUTPUT",
    "PADDED_FF3_MM_OUTPUT",
    "PADDED_FF2_MM_OUTPUT",
    # Decoder
    "ATTN_ADD_OUTPUT",
    "MLP_ADD_OUTPUT",
    # Model
    "LN_F_WEIGHTS",
    "LN_F_OUTPUT",
    # LM Head
    "LM_HEAD_MM_OUTPUT",
)

NO_MEMCFG = ("SOFTMAX_OUTPUT",)

NO_DTYPE = (
    # Attention
    "SOFTMAX_OUTPUT",
    # MLP
    # Decoder
    "ATTN_ADD_OUTPUT",
    "MLP_ADD_OUTPUT",
)

ACCEPTABLE_MODEL_CONFIG_STRS = ("BFLOAT16-DRAM", "BFLOAT8_B-DRAM")


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


def get_model_config(model_config_str, num_devices=8, llm_mode="decode"):
    assert model_config_str in ACCEPTABLE_MODEL_CONFIG_STRS
    assert num_devices in (8, 32)
    assert llm_mode in ("decode", "prefill")

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

    # Set default dtype and mem_config based on model_config_str
    if model_config_str in ACCEPTABLE_MODEL_CONFIG_STRS:
        dtype_str, mem_config_str = model_config_str.split("-")
        # TODO: Set default memcfg for BFLOAT16-L1 to L1
        mem_config = DRAM_MEMCFG if mem_config_str == "DRAM" else L1_MEMCFG
        dtype = getattr(ttl.tensor.DataType, dtype_str)
    else:
        raise NotImplementedError(f"Model config {model_config_str} is not supported!")

    # Set defaults for dtype and mem_config for all ops
    model_config = {
        "LLM_MODE": llm_mode,
        "DEFAULT_DTYPE": dtype,
        "DEFAULT_MEMCFG": mem_config,
        "NUM_DEVICES": num_devices,
        "MAX_GRID_SIZE": (8, 4),
        "ALL_GATHER_NUM_LINKS": 1,
        "DEFAULT_CKPT_DIR": os.getenv("LLAMA_CKPT_DIR", "/home/llama-data-repacked-2/llama-2-70b/"),
        "DEFAULT_TOKENIZER_PATH": os.getenv("LLAMA_TOKENIZER_PATH", "/home/llama-data/tokenizer.model"),
        "DEFAULT_CACHE_PATH": Path(os.getenv("LLAMA_CACHE_PATH", "/home/llama-data-cache/weights-cache-2")),
        "COMPUTE_KERNEL_CONFIG": ttl.tensor.WormholeComputeKernelConfig(
            # math_fidelity=ttl.tensor.MathFidelity.LoFi,
            math_fidelity=ttl.tensor.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
        "COMPUTE_KERNEL_FP16_ACC_CONFIG": ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        ),
        "L1_MEMCFG": L1_MEMCFG,
        "DRAM_MEMCFG": DRAM_MEMCFG,
        "BFLOAT16_DTYPE": BFLOAT16_DTYPE,
        "BFP8_DTYPE": BFP8_DTYPE,
        "WIDTH_SHARDED_MEMCFG": WIDTH_SHARDED_MEMCFG,
        "HEIGHT_SHARDED_MEMCFG": HEIGHT_SHARDED_MEMCFG,
        "BLOCK_SHARDED_MEMCFG": BLOCK_SHARDED_MEMCFG,
    }
    model_config.update({f"{key}_MEMCFG": mem_config for key in OP_KEYS if key not in NO_MEMCFG})
    model_config.update({f"{key}_DTYPE": dtype for key in OP_KEYS if key not in NO_DTYPE})

    # Matmul Weights must always be BFP8_B
    # Override defaults for certain configs
    for key in model_config.keys():
        if "MM_WEIGHTS_DTYPE" in key:
            model_config[key] = BFP8_DTYPE
        elif "WEIGHTS_MEMCFG" in key or "BIAS_MEMCFG" in key:
            model_config[key] = DRAM_MEMCFG
        elif "LN" in key and ("WEIGHTS_DTYPE" in key or "BIAS_DTYPE" in key):
            model_config[key] = BFLOAT16_DTYPE

    model_config["KV_CACHE_DTYPE"] = BFP8_DTYPE
    # model_config["KV_CACHE_DTYPE"] = BFLOAT16_DTYPE # HACK:! JUST FOR UPDATE_CACHE TESTING

    hidden_size = model_config_entries["hidden_size"]
    head_dim = model_config_entries["head_dim"]
    n_heads = model_config_entries["num_attention_heads"]
    n_kv_heads = model_config_entries["num_kv_heads"]

    if llm_mode == "decode":
        batch, seq_len = 32, 1
        shard_height = batch
    else:
        batch, seq_len = 1, 128
        shard_height = seq_len

    shard_spec_64_cores_grid = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(7, 7),
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
    shard_spec_32_cores_grid_transposed = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(3, 7),
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
    shard_spec_2_cores_grid = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(1, 0),
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
    shard_width_hidden_dim_across_16_cores = hidden_size // 16
    shard_width_hidden_dim_across_8_cores = hidden_size // 8
    shard_width_hidden_dim_across_64_cores = hidden_size // 64

    # Constants based on head_dim
    total_width_per_group_of_qkv_heads = head_dim * ((n_heads // n_kv_heads) + 2)  # 8 q_heads + 1 k_heads + 1 v_heads
    n_local_kv_heads = n_kv_heads // num_devices
    total_width_of_qkv_heads_per_device = total_width_per_group_of_qkv_heads * n_local_kv_heads
    shard_width_qkv_heads_per_device_across_16_cores = total_width_of_qkv_heads_per_device // 16
    shard_width_qkv_heads_per_device_across_8_cores = total_width_of_qkv_heads_per_device // 8
    shard_width_qkv_heads_per_device_across_32_cores = total_width_of_qkv_heads_per_device // 32

    # Constants based on padded_mlp_dim
    padded_mlp_dim = model_config_entries["padded_mlp_dim"]
    shared_with_padded_mlp_dim_across_32_cores = padded_mlp_dim // 32

    # Embeddings
    if num_devices == 8:
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
    elif num_devices == 32:
        model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
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
    # Model prepare_inputs
    model_config["ATTN_MASK_DTYPE"] = BFP8_DTYPE
    if num_devices == 8:
        if llm_mode == "decode":
            model_config["ATTN_MASK_MEMCFG"] = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.BufferType.L1,
                ttl.tensor.ShardSpec(
                    shard_spec_32_cores_grid,
                    [
                        shard_height,
                        1,  # Dynamic - must set before using this config
                    ],
                    ttl.tensor.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
        else:
            model_config["ATTN_MASK_MEMCFG"] = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.BufferType.L1,
                ttl.tensor.ShardSpec(
                    shard_spec_8_cores_grid,
                    [
                        shard_height,
                        1,  # Dynamic - must set before using this config
                    ],
                    ttl.tensor.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
    elif num_devices == 32:
        model_config["ATTN_MASK_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_8_cores_grid,
                [
                    shard_height,
                    1,  # Dynamic - must set before using this config
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

    model_config["ROT_MAT_MEMCFG"] = DRAM_MEMCFG  # L1_MEMCFG

    # Llama Model Config
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
    model_config["LN_F_OUTPUT_MEMCFG"] = model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"]
    model_config["LN_F_PROGCFG"] = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 4],
        subblock_w=8,
        block_h=1,
        block_w=8,
        inplace=True,
    )
    # LM Head
    if num_devices == 8:
        model_config["LM_HEAD_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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

    # Decoder
    # For Prefill. we can calculate based on the dynamic seqlen for block sharded layernorm.
    # shard_height = seqlen for prefill
    layernorm_num_cores_x = 8
    layernorm_max_num_cores_y = 7
    for i in range(layernorm_max_num_cores_y, 0, -1):
        if (shard_height // 32) % i == 0:
            layernorm_num_cores_y = i
            break

    num_tiles_per_core_h = shard_height // 32 // layernorm_num_cores_y
    num_tiles_per_core_w = hidden_size // 32 // layernorm_num_cores_x

    layernorm_shard_height_hidden_dim = shard_height // layernorm_num_cores_y
    layernorm_shard_width_hidden_dim = hidden_size // layernorm_num_cores_x

    core_range_block_sharded_layernorm = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(layernorm_num_cores_x - 1, layernorm_num_cores_y - 1),
            ),
        }
    )
    if llm_mode == "decode":
        model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
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
    else:
        model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
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

    if num_devices == 8:
        if llm_mode == "decode":
            model_config["LN_ATTN_PROGCFG"] = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[8, 4],
                subblock_w=8,
                block_h=shard_height // 32,
                block_w=8,
                math_fidelity=ttl.tensor.MathFidelity.HiFi4,
                im_data_format=ttl.tensor.DataType.BFLOAT16,
                out_data_format=model_config["LN_ATTN_OUTPUT_DTYPE"],
                inplace=True,
            )
        else:
            model_config["LN_ATTN_PROGCFG"] = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[layernorm_num_cores_x, layernorm_max_num_cores_y],
                subblock_w=8,
                block_h=num_tiles_per_core_h,
                block_w=num_tiles_per_core_w,
                math_fidelity=ttl.tensor.MathFidelity.HiFi4,
                im_data_format=ttl.tensor.DataType.BFLOAT16,
                out_data_format=model_config["LN_ATTN_OUTPUT_DTYPE"],
                inplace=True,
            )
    elif num_devices == 32:
        model_config["LN_ATTN_PROGCFG"] = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 4],
            subblock_w=8,
            block_h=1,
            block_w=8,
            inplace=True,  # TODO: Not Inplace RMSNorm because we need to keep the residual
        )
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
    if num_devices == 8:
        if llm_mode == "decode":
            model_config["LN_MLP_PROGCFG"] = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[8, 4],
                subblock_w=8,
                block_h=shard_height // 32,
                block_w=8,
                math_fidelity=ttl.tensor.MathFidelity.HiFi4,
                im_data_format=ttl.tensor.DataType.BFLOAT16,
                out_data_format=model_config["LN_MLP_OUTPUT_DTYPE"],
                inplace=True,
            )
        else:
            model_config["LN_MLP_PROGCFG"] = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[layernorm_num_cores_x, layernorm_max_num_cores_y],
                subblock_w=8,
                block_h=num_tiles_per_core_h,
                block_w=num_tiles_per_core_w,
                math_fidelity=ttl.tensor.MathFidelity.HiFi4,
                im_data_format=ttl.tensor.DataType.BFLOAT16,
                out_data_format=model_config["LN_MLP_OUTPUT_DTYPE"],
                inplace=True,
            )
    elif num_devices == 32:
        model_config["LN_MLP_PROGCFG"] = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 4],
            subblock_w=8,
            block_h=1,
            block_w=8,
            inplace=False,  # Not Inplace RMSNorm because we need to keep the residual
        )
    model_config["LN_MLP_OUTPUT_MEMCFG"] = model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"]
    model_config["MLP_ADD_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
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
    # LLama2 Attention Module
    if num_devices == 8 or num_devices == 32:
        model_config["FUSED_QKV_MM_INPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
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
        model_config["FUSED_QKV_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 1),
            in0_block_w=32,
            out_subblock_h=1,
            out_subblock_w=1,  # TODO: Maximize for fp32
            per_core_M=shard_height // 32,
            per_core_N=5,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        if llm_mode == "prefill":
            # Llama 2 Attention Module Prefill
            # Input shape is [1,1,128,8192]
            # qkv_list shape is [8192,1280]
            # 2D

            model_config["FUSED_QKV_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=32,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=1,  #
                per_core_N=5,  # N / TILE_WIDTH / Grid_Size
                transpose_mcast=False,
                fused_activation=None,
            )
            # model_config["FUSED_QKV_MM_INPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            #     ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            #     ttl.tensor.BufferType.L1,
            #     ttl.tensor.ShardSpec(
            #         shard_spec_32_cores_grid,
            #         [
            #             shard_height // 4,
            #             shard_width_hidden_dim_across_8_cores,
            #         ],
            #         ttl.tensor.ShardOrientation.ROW_MAJOR,
            #         False,
            #     ),
            # )
    if num_devices == 8 or num_devices == 32:
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
    model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"] = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,  # Highest fidelity
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    # TODO: Remove once confirm we don't need this, using only fallback ops
    if num_devices == 8 or num_devices == 32:
        if llm_mode == "decode":
            model_config["ROT_MAT_Q_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 1),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=1,
                per_core_N=4,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            )
        else:
            # ttnn.Shape([1, 8, 128, 128])
            # ttnn.Shape([8, 128, 128, 128])
            model_config["ROT_MAT_Q_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=[8, 1],
                in0_block_w=4,  # 128 // TILE_SIZE (dynamic)
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=shard_height // 32,
                per_core_N=head_dim // 32,
            )
            # ttnn.Shape([1, 1, 128, 128])
            # ttnn.Shape([1, 128, 128, 128])
            model_config["ROT_MAT_K_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(1, 1),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=1,
                per_core_N=4,
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=False,
            )
        model_config["ROT_MAT_Q_MM_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_8_cores_grid,
                [
                    shard_height,
                    head_dim,  # head dim
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["ROT_MAT_K_MM_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_1_cores_grid,
                [
                    shard_height,  # Each core has 32 users
                    head_dim,  # head dim
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    if num_devices == 8:
        model_config["KV_CACHE_SLICE_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    1,  # Dynamic
                    head_dim,
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    elif num_devices == 32:
        model_config["KV_CACHE_SLICE_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_8_cores_grid,
                [
                    1,  # Dynamic
                    head_dim,
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    if num_devices == 8:
        model_config["Q_TRANSPOSE_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_32_cores_grid,  # Sharded on batch dim
                [
                    shard_height,  # Each core has 32 padded heads
                    head_dim,
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        if llm_mode == "decode":
            model_config[
                "ATTN_BATCHED_MM_PROGCFG_LAMBDA"
            ] = lambda n: ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=[8, 4],
                in0_block_w=head_dim // 32,  # HEAD_DIM // TILE_SIZE
                out_subblock_h=1,  # TODO: Maximize
                out_subblock_w=1,  # TODO: Maximize
                per_core_M=shard_height // 32,  # N_HEADS_PADDED // TILE_SIZE,
                per_core_N=n,  # SEQ_LEN // TILE_SIZE (dynamic)
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
            model_config[
                "SCORES_BATCHED_MM_PROGCFG_LAMBDA"
            ] = lambda in0_block_w: ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=[8, 4],
                in0_block_w=in0_block_w,  # SEQ_LEN // TILE_SIZE (dynamic)
                out_subblock_h=1,  # TODO: Maximize
                out_subblock_w=1,  # TODO: Maximize
                per_core_M=shard_height // 32,  # N_HEADS_PADDED // TILE_SIZE,
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
            # (Pdb) query_layer[0].shape
            # ttnn.Shape([1, 8, 128, 128])
            # (Pdb) key_layer_transposed[0].shape
            # ttnn.Shape([1, 1, 128, 128])
            model_config[
                "ATTN_BATCHED_MM_PROGCFG"
            ] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 1),
                in0_block_w=head_dim // 32,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=shard_height // 32,
                per_core_N=seq_len // 32,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            )

            model_config["ATTN_BATCHED_MM_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.BufferType.L1,
                ttl.tensor.ShardSpec(
                    shard_spec_8_cores_grid,
                    [
                        shard_height,  # Each core has 32 users
                        128,  # Dynamic (padded seqlen)
                    ],
                    ttl.tensor.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
            model_config[
                "SCORES_BATCHED_MM_PROGCFG"
            ] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 1),
                in0_block_w=seq_len // 32,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=shard_height // 32,
                per_core_N=head_dim // 32,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            )
            model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.BufferType.L1,
                ttl.tensor.ShardSpec(
                    shard_spec_8_cores_grid,
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
    elif num_devices == 32:
        model_config["Q_TRANSPOSE_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_8_cores_grid,
                [
                    32,  # Each core has 32 padded heads
                    head_dim,
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

        model_config[
            "ATTN_BATCHED_MM_PROGCFG_LAMBDA"
        ] = lambda n: ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=[8, 1],
            in0_block_w=128 // 32,  # HEAD_DIM // TILE_DIM
            out_subblock_h=1,  # TODO: Maximize
            out_subblock_w=1,  # TODO: Maximize
            per_core_M=32 // 32,  # N_HEADS_PADDED // TILE_SIZE,
            per_core_N=n,  # SEQ_LEN // TILE_SIZE (dynamic)
        )

        model_config["ATTN_BATCHED_MM_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_16_cores_grid,
                [
                    32,  # Each core has 32 users
                    1,  # Dynamic (padded seqlen)
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

        model_config[
            "SCORES_BATCHED_MM_PROGCFG_LAMBDA"
        ] = lambda in0_block_w: ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=[8, 1],
            in0_block_w=in0_block_w,  # SEQ_LEN // TILE_DIM (dynamic)
            out_subblock_h=1,  # TODO: Maximize
            out_subblock_w=1,  # TODO: Maximize
            per_core_M=32 // 32,  # N_HEADS_PADDED // TILE_SIZE,
            per_core_N=128 // 32,  # HEAD_DIM // TILE_SIZE (dynamic)
        )

        model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_8_cores_grid,
                [
                    32,  # Each core has 32 users
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
                shard_spec_8_cores_grid,
                [
                    32,  # Each core has 32 users
                    head_dim,  # head dim
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    if num_devices == 8:
        if llm_mode == "decode":
            model_config[
                "BATCHED_SOFTMAX_PROGCFG"
            ] = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 4),  # In-place softmax on 32 cores sharded on batch dim
                subblock_w=1,
                block_h=shard_height // 32,
                block_w=1,  # Dynamic
                math_fidelity=ttl.tensor.MathFidelity.HiFi4,
                im_data_format=ttl.tensor.DataType.BFLOAT16,
            )
        else:
            model_config[
                "BATCHED_SOFTMAX_PROGCFG"
            ] = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 1),  # TODO: NOT WORKING !!!!!!!!
                subblock_w=1,
                block_h=shard_height // 32,
                block_w=1,  # Dynamic
                math_fidelity=ttl.tensor.MathFidelity.HiFi4,
                im_data_format=ttl.tensor.DataType.BFLOAT16,
            )
    elif num_devices == 32:
        model_config[
            "BATCHED_SOFTMAX_PROGCFG"
        ] = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 1),
            subblock_w=1,
            block_h=1,
            block_w=1,  # Dynamic
        )
    model_config["ATTN_ALL_GATHER_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
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
    if num_devices == 8 or num_devices == 32:
        model_config["SELFOUT_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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

    # Example code of using 2d multicast for MLP matmul
    """
        model_config["MLP_BLOCK_SHARDED"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                ttl.tensor.CoreRangeSet(
                    {
                        ttl.tensor.CoreRange(
                            # Volume must match # of attn heads
                            ttl.tensor.CoreCoord(0, 0),
                            ttl.tensor.CoreCoord(7, 7),
                        ),
                    }
                ),
                [
                    512,
                    1024,
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    """
    # Llama MLP config
    # Padded MLP 32K config:
    if num_devices == 8:
        if llm_mode == "decode":
            model_config["PADDED_FF1_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
            model_config["PADDED_FF3_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
            model_config["PADDED_FF2_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
            # Llama 2 MLP Module Prefill
            model_config[
                "PADDED_FF1_MM_PROGCFG_LAMBDA"
            ] = lambda seq_tiles: ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=8,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=seq_tiles // 4,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=16,  # N / TILE_WIDTH / Grid_Size
                transpose_mcast=False,
                fused_activation=ttl.tensor.FusibleActivation.SILU,
            )

            model_config[
                "PADDED_FF3_MM_PROGCFG_LAMBDA"
            ] = lambda seq_tiles: ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=8,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=seq_tiles // 4,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=16,  # N / TILE_WIDTH / Grid_Size
                transpose_mcast=False,
                fused_activation=None,
            )

            # input0: [1,32,128,32k]
            # input1: [1,1,32k,1k]
            model_config[
                "PADDED_FF2_MM_PROGCFG_LAMBDA"
            ] = lambda seq_tiles: ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=8,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=seq_tiles // 4,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=4,  # N / TILE_WIDTH / Grid_Size
                transpose_mcast=False,
                fused_activation=None,
            )
    elif num_devices == 32:
        model_config["PADDED_FF1_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=2,  # K = 8k / 4 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=4,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 4096 for num_device=8
            fuse_batch=True,
            fused_activation=None,  # Can't use SILU on partial activations
            mcast_in0=True,
        )
        model_config["PADDED_FF3_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=2,  # K = 8k / 4 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=4,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        model_config["PADDED_FF2_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=4,  # K = 32k / 8 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    if num_devices == 8:
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
    elif num_devices == 32:
        model_config["PADDED_MLP_ALL_GATHER_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    shard_height,
                    shared_with_padded_mlp_dim_across_32_cores // 8,  # 32k // 32 // 8: 2D sharding 4x8
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

    # uncomment if need to see all the configs
    # logger.debug(f"Falcon model config: \n{pretty_print_model_config(model_config)}")

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


# def main():
#     model_config = get_model_config("BFLOAT16-DRAM")
#     print(model_config["FF1_MM_WEIGHTS_DTYPE"])
#     print(model_config["FF1_MM_OUTPUT_MEMCFG"])
#     # print(pretty_print_model_config(model_config))


# main()
