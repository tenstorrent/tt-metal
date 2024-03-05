# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib as ttl
from loguru import logger
from pathlib import Path

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
    "LN_ATTN_BIAS",
    "LN_ATTN_OUTPUT",
    "LN_MLP_WEIGHTS",
    "LN_MLP_BIAS",
    "LN_MLP_OUTPUT",
    # Rotary
    "ROT_MAT",
    # Attention
    "FUSED_QKV_MM_WEIGHTS",
    "FUSED_QKV_MM_OUTPUT",
    "CREATE_QKV_HEADS_OUTPUT",
    "ROTARY_EMBEDDING_OUTPUT",
    "K_CACHE_SLICE_OUTPUT",
    "V_CACHE_SLICE_OUTPUT",
    "K_TRANSPOSED_OUTPUT",
    "PRE_SOFTMAX_MM_OUTPUT",
    "ATTN_BATCHED_MM_OUTPUT",
    "PRE_SOFTMAX_SCALE_OUTPUT",
    "PRE_SOFTMAX_MASK_OUTPUT",
    "SOFTMAX_OUTPUT",
    "POST_SOFTMAX_MM_OUTPUT",
    "SCORES_BATCHED_MM_OUTPUT",
    "CONCAT_HEADS_OUTPUT",
    "SELFOUT_MM_WEIGHTS",
    "SELFOUT_MM_OUTPUT",
    # MLP
    "FF1_MM_WEIGHTS",
    "FF1_MM_OUTPUT",
    "FF3_MM_WEIGHTS",
    "FF3_MM_OUTPUT",
    "FF2_MM_WEIGHTS",
    "FF2_MM_OUTPUT",
    # Decoder Cont
    "PARALLEL_ATTN_ADD_OUTPUT",
    "DROPOUT_ADD_OUTPUT",
    # Model
    "LN_F_WEIGHTS",
    "LN_F_BIAS",
    "LN_F_OUTPUT",
    # LM Head
    "LM_HEAD_MM_WEIGHTS",
    "LM_HEAD_MM_OUTPUT",
)

NO_MEMCFG = ("SOFTMAX_OUTPUT",)

NO_DTYPE = (
    # Attention
    "ROTARY_EMBEDDING_OUTPUT",
    "CREATE_QKV_HEADS_OUTPUT",
    "K_TRANSPOSED_OUTPUT",
    "PRE_SOFTMAX_SCALE_OUTPUT",
    "PRE_SOFTMAX_MASK_OUTPUT",
    "SOFTMAX_OUTPUT",
    "CONCAT_HEADS_OUTPUT",
    # MLP
    "MLP_ALL_GATHER_OUTPUT",
    # Decoder Cont
    "PARALLEL_ATTN_ADD_OUTPUT",
    "DROPOUT_ADD_OUTPUT",
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


def get_model_config(model_config_str, num_devices=8):
    assert model_config_str in ACCEPTABLE_MODEL_CONFIG_STRS
    assert num_devices in (4, 8)
    DRAM_MEMCFG = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    L1_MEMCFG = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    WIDTH_SHARDED_MEMCFG = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED, ttl.tensor.BufferType.L1
    )
    HEIGHT_SHARDED_MEMCFG = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1
    )
    BFLOAT16_DTYPE = ttl.tensor.DataType.BFLOAT16
    BFP8_DTYPE = ttl.tensor.DataType.BFLOAT8_B
    # BFP2_DTYPE = ttl.tensor.DataType.BFLOAT2_B

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
        "DEFAULT_DTYPE": dtype,
        "DEFAULT_MEMCFG": mem_config,
        "MOVE_DECODER_OUTPUT_BOOL": False,
        "NUM_DEVICES": num_devices,
        "MAX_GRID_SIZE": (8, 4),
        "ALL_GATHER_NUM_LINKS": 1,
        "DEFAULT_CKPT_DIR": "/home/llama-data-repacked-2/llama-2-70b/",
        "DEFAULT_TOKENIZER_PATH": "/home/llama-data/tokenizer.model",
        "DEFAULT_CACHE_PATH": Path("/home/llama-data-cache/weights-cache-2"),
        "COMPUTE_KERNEL_CONFIG": ttl.tensor.WormholeComputeKernelConfig(
            # math_fidelity=ttl.tensor.MathFidelity.LoFi,
            math_fidelity=ttl.tensor.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
        "L1_MEMCFG": L1_MEMCFG,
        "DRAM_MEMCFG": DRAM_MEMCFG,
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

    model_config["KV_CACHE_MEMCFG"] = DRAM_MEMCFG
    model_config["KV_CACHE_DTYPE"] = BFP8_DTYPE

    hidden_size = model_config_entries["hidden_size"]

    head_dim = model_config_entries["head_dim"]
    n_heads = model_config_entries["num_attention_heads"]
    n_kv_heads = model_config_entries["num_kv_heads"]

    batch, seq_len = 32, 1
    shard_height = batch

    shard_spec_32_cores_grid = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(7, 3),
            ),
        }
    )
    # shard_spec_q_heads_cores_grid
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

    # Constants based on head_dim
    total_width_per_group_of_qkv_heads = head_dim * ((n_heads // n_kv_heads) + 2)  # 8 q_heads + 1 k_heads + 1 v_heads
    n_local_kv_heads = n_kv_heads // num_devices
    total_width_of_qkv_heads_per_device = total_width_per_group_of_qkv_heads * n_local_kv_heads
    shard_width_qkv_heads_per_device_across_16_cores = total_width_of_qkv_heads_per_device // 16
    shard_width_qkv_heads_per_device_across_8_cores = total_width_of_qkv_heads_per_device // 8

    # Constants based on padded_mlp_dim
    padded_mlp_dim = model_config_entries["padded_mlp_dim"]
    shared_with_padded_mlp_dim_across_32_cores = padded_mlp_dim // 32

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
    model_config["ATTN_MASK_DTYPE"] = BFP8_DTYPE
    if num_devices == 4:
        model_config["ATTN_MASK_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_16_cores_grid,  # Sharded on num_local_heads = n_qheads // num_devices
                [
                    shard_height,  # Each core has 32 users
                    1,  # Dynamic - must set before using this config
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    elif num_devices == 8:
        model_config["ATTN_MASK_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_32_cores_grid,  # Sharded on batch dim
                [
                    shard_height,  # Padded qheads up to 32 == batch size
                    1,  # Dynamic - must set before using this config
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    model_config["ROT_MAT_MEMCFG"] = DRAM_MEMCFG  # L1_MEMCFG
    # Decoder
    # TODO: change name to be llama2 specific
    model_config["PARALLEL_ATTN_ADD_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
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
    # TODO: change name to be llama2 specific
    model_config["DROPOUT_ADD_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
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
    model_config["LN_ATTN_PROGCFG"] = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 4],
        subblock_w=8,
        block_h=1,
        block_w=8,
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,
        im_data_format=ttl.tensor.DataType.BFLOAT16,
        out_data_format=model_config["LN_ATTN_OUTPUT_DTYPE"],
        inplace=True,
    )
    model_config["LN_ATTN_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
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
    model_config["LN_MLP_PROGCFG"] = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 4],
        subblock_w=8,
        block_h=1,
        block_w=8,
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,
        im_data_format=ttl.tensor.DataType.BFLOAT16,
        out_data_format=model_config["LN_MLP_OUTPUT_DTYPE"],
        inplace=True,
    )
    model_config["LN_MLP_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
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
    model_config["PADDED_LN_MLP_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
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
    # LLama2 Attention Module
    if num_devices == 4:
        model_config["FUSED_QKV_MM_INPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_16_cores_grid,
                [
                    shard_height,
                    shard_width_hidden_dim_across_16_cores,
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["FP32_FUSED_QKV_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 2),
            in0_block_w=16,
            out_subblock_h=1,
            out_subblock_w=1,  # TODO: Maximize
            per_core_M=1,
            per_core_N=5,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_16_cores_grid,
                [
                    shard_height,
                    shard_width_qkv_heads_per_device_across_16_cores,
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    elif num_devices == 8:
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
        model_config["FP32_FUSED_QKV_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 1),
            in0_block_w=32,
            out_subblock_h=1,
            out_subblock_w=1,  # TODO: Maximize
            per_core_M=1,
            per_core_N=5,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
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
    if num_devices == 4:
        model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_2_cores_grid,
                [
                    shard_height,
                    total_width_per_group_of_qkv_heads,  # Must always be minimum a full group
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    elif num_devices == 8:
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
    model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG
    model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"] = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,  # Highest fidelity
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    # TODO: Remove once confirm we don't need this, using only fallback ops
    model_config["L1_HEADS_INTERLEAVED_MEMCFG"] = L1_MEMCFG
    model_config["ROT_MAT_K_MM_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG
    model_config["ROT_MAT_Q_MM_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG
    if num_devices == 4:
        model_config["ROT_MAT_Q_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 2),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        model_config["ROT_MAT_K_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(2, 1),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
    elif num_devices == 8:
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
        model_config["ROT_MAT_K_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(1, 1),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
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
    model_config["K_TRANSPOSED_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG
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
    model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG
    if num_devices == 4:
        model_config["SOFTMAX_PROGCFG"] = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 2),
            subblock_w=1,
            block_h=1,
            block_w=1,  # Dynamic
            math_fidelity=ttl.tensor.MathFidelity.HiFi4,
            im_data_format=ttl.tensor.DataType.BFLOAT16,
        )
    elif num_devices == 8:
        model_config[
            "BATCHED_SOFTMAX_PROGCFG"
        ] = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 4),  # In-place softmax on 32 cores sharded on batch dim
            subblock_w=1,
            block_h=1,
            block_w=1,  # Dynamic
            math_fidelity=ttl.tensor.MathFidelity.HiFi4,
            im_data_format=ttl.tensor.DataType.BFLOAT16,
        )
    model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG
    model_config["CONCAT_HEADS_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
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
    model_config["SELFOUT_MM_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
    if num_devices == 4:
        model_config["SELFOUT_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,  # (32 x 8k) x (8k x 2k) = (32 x 2k)
            out_subblock_h=1,
            out_subblock_w=2,  # TODO: Maximize
            per_core_M=1,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    elif num_devices == 8:
        model_config["SELFOUT_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,  # (32 x 8k) x (8k x 1k) = (32 x 1k)
            out_subblock_h=1,
            out_subblock_w=1,  # TODO: Maximize
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    # Llama MLP config
    model_config["FF1_MM_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
    model_config["FF3_MM_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
    model_config["FF2_MM_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
    model_config["FF13_MUL_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
    # Padded MLP 32K config:
    if num_devices == 4:
        model_config["PADDED_FF1_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=8,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 8192 for num_device=8
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
            per_core_N=8,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        model_config["PADDED_FF2_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=32,  # K = 32768 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    elif num_devices == 8:
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
    model_config["LN_F_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
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
        block_h=1,
        block_w=8,
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,
        im_data_format=ttl.tensor.DataType.BFLOAT16,
        out_data_format=model_config["LN_F_OUTPUT_DTYPE"],
        inplace=True,
    )

    # LM Head
    model_config["LM_HEAD_MM_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
    # Output = 32 * 8192 x 8192 * 8000 = 32 x 8000
    if num_devices == 4:
        model_config["LM_HEAD_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=8,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    elif num_devices == 8:
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
