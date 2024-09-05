# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
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
    "LN_INPUT",
    "LN_ATTN_WEIGHTS",
    "LN_ATTN_BIAS",
    "LN_ATTN_OUTPUT",
    "LN_MLP_WEIGHTS",
    "LN_MLP_BIAS",
    "LN_MLP_OUTPUT",
    # Rotary
    "SIN_CACHED_WEIGHTS",
    "COS_CACHED_WEIGHTS",
    # Attention
    "ATTN_INPUT",
    "ATTENTION_OUT",
    "FUSED_QKV_MM_WEIGHTS",
    "FUSED_QKV_MM_INPUT",
    "FUSED_QKV_MM_OUTPUT",
    "CREATE_QKV_HEADS_INPUT",
    "CREATE_QKV_HEADS_OUTPUT",
    "CREATE_KV_HEADS_OUTPUT",
    "ROTARY_EMBEDDING_OUTPUT",
    "K_CACHE_SLICE_OUTPUT",
    "V_CACHE_SLICE_OUTPUT",
    "K_TRANSPOSED_OUTPUT",
    "PRE_SOFTMAX_MM_OUTPUT",
    "PRE_SOFTMAX_SCALE_OUTPUT",
    "PRE_SOFTMAX_MASK_OUTPUT",
    "SOFTMAX_OUTPUT",
    "POST_SOFTMAX_MM_OUTPUT",
    "CONCAT_HEADS_OUTPUT",
    "SELFOUT_MM_WEIGHTS",
    "SELFOUT_MM_OUTPUT",
    "DENSE_H_TO_4H_MM_WEIGHTS",
    "DENSE_H_TO_4H_MM_OUTPUT",
    "MLP_ALL_GATHER_OUTPUT",
    "DENSE_4H_TO_H_MM_WEIGHTS",
    "DENSE_4H_TO_H_MM_OUTPUT",
    "ATTN_ALL_GATHER_OUTPUT",
    # Decoder Cont
    "PARALLEL_ATTN_ADD_OUTPUT",
    "DROPOUT_ADD_OUTPUT",
    "DECODER_ALL_GATHER_OUTPUT",
    # Model
    "LN_F_WEIGHTS",
    "LN_F_BIAS",
    "LN_F_OUTPUT",
    # LM Head
    "LM_HEAD_MM_WEIGHTS",
    "LM_HEAD_MM_OUTPUT",
    "FINAL_ALL_GATHER_OUTPUT",
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

ACCEPTABLE_DECODE_MODEL_CONFIG_STRS = ("BFLOAT16-SHARDED", "BFLOAT8_B-SHARDED")
ACCEPTABLE_PREFILL_MODEL_CONFIG_STRS = ("BFLOAT8_B-DRAM", "BFLOAT16-DRAM")

model_config_entries = {
    "_name_or_path": "tiiuae/falcon-40b-instruct",
    "alibi": False,
    "apply_residual_connection_post_layernorm": False,
    "architectures": ["FalconForCausalLM"],
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "configuration_falcon.FalconConfig",
        "AutoModel": "modeling_falcon.FalconModel",
        "AutoModelForCausalLM": "modeling_falcon.FalconForCausalLM",
        "AutoModelForQuestionAnswering": "modeling_falcon.FalconForQuestionAnswering",
        "AutoModelForSequenceClassification": "modeling_falcon.FalconForSequenceClassification",
        "AutoModelForTokenClassification": "modeling_falcon.FalconForTokenClassification",
    },
    "bias": False,
    "bos_token_id": 11,
    "eos_token_id": 11,
    "hidden_dropout": 0.0,
    "hidden_size": 8192,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "falcon",
    "multi_query": True,
    "new_decoder_architecture": True,
    "num_attention_heads": 128,
    "num_hidden_layers": 60,
    "num_kv_heads": 8,
    "parallel_attn": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.28.1",
    "use_cache": True,
    "vocab_size": 65024,
}


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


def get_model_config(model_config_str, llm_mode, input_shape, num_devices):
    assert llm_mode in ("prefill", "decode")
    assert len(input_shape) == 2
    assert num_devices == 8, "Currently only supporting 8 devices"
    if llm_mode == "prefill":
        model_config = get_prefill_model_config(model_config_str, input_shape, num_devices)
    elif llm_mode == "decode":
        model_config = get_decode_model_config(model_config_str, input_shape, num_devices)
    else:
        assert False

    model_config["LLM_MODE"] = llm_mode

    return model_config


def get_decode_model_config(model_config_str, input_shape, num_devices):
    assert model_config_str in ACCEPTABLE_DECODE_MODEL_CONFIG_STRS
    assert len(input_shape) == 2
    assert num_devices == 8, "Decode is currently only supported on 8 devicess"

    DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
    L1_MEMCFG = ttnn.L1_MEMORY_CONFIG
    WIDTH_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)
    HEIGHT_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    BFLOAT16_DTYPE = ttnn.bfloat16
    BFP8_DTYPE = ttnn.bfloat8_b
    BFP4_DTYPE = ttnn.bfloat4_b

    # Set default dtype and mem_config based on model_config_str
    if model_config_str in ACCEPTABLE_DECODE_MODEL_CONFIG_STRS:
        dtype_str, mem_config_str = model_config_str.split("-")
        # TODO: Set default memcfg for BFLOAT16-L1 to L1
        mem_config = DRAM_MEMCFG if mem_config_str == "DRAM" else L1_MEMCFG
        dtype = getattr(ttnn.DataType, dtype_str)
    else:
        raise NotImplementedError(f"Model config {model_config_str} is not supported!")

    # Set defaults for dtype and mem_config for all ops
    model_config = {
        "LLM_MODE": "decode",
        "DEFAULT_DTYPE": dtype,
        "DEFAULT_MEMCFG": mem_config,
        "MOVE_DECODER_OUTPUT_BOOL": False,
        "NUM_DEVICES": num_devices,
        "MAX_GRID_SIZE": (8, 4),
        "ALL_GATHER_NUM_LINKS": 1,
        "DEFAULT_CACHE_PATH": Path(f"models/demos/t3000/falcon40b/datasets/"),
        "COMPUTE_KERNEL_CONFIG": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
        "COMPUTE_KERNEL_FP16_ACC_CONFIG": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        ),
        "BFP4_DTYPE": BFP4_DTYPE,
        "BFP8_DTYPE": BFP8_DTYPE,
        "BFLOAT16_DTYPE": BFLOAT16_DTYPE,
        "DRAM_MEMCFG": DRAM_MEMCFG,
        "L1_MEMCFG": L1_MEMCFG,
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

    # # TODO: use BFLOAT16 for the attention mask!
    # # model_config["KV_CACHE_MEMCFG"] = DRAM_MEMCFG
    # model_config["ATTN_MASK_DTYPE"] = BFLOAT16_DTYPE

    head_dim = 64
    hidden_size = model_config_entries["hidden_size"]
    vocab_size = model_config_entries["vocab_size"]
    num_attention_heads = model_config_entries["num_attention_heads"]
    num_kv_heads = model_config_entries["num_kv_heads"]

    batch, seq_len = input_shape
    assert batch == 32
    row_height = batch
    model_config["row_height"] = row_height
    model_config["BATCH_SIZE"] = batch
    model_config["SEQ_LEN"] = seq_len

    if model_config_str in ("BFLOAT16-L1",):
        model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["K_CACHE_SLICE_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["V_CACHE_SLICE_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["K_TRANSPOSED_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["PRE_SOFTMAX_SCALE_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["PRE_SOFTMAX_MASK_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"] = L1_MEMCFG

    if mem_config_str == "SHARDED":
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
        shard_spec_2_cores_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(1, 0),
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

        # Constants based on hidden_dim
        shard_width_hidden_dim_across_32_cores = hidden_size // 32
        shard_width_4x_hidden_dim_across_32_cores = hidden_size * 4 // 32
        shard_width_hidden_dim_per_device_across_32_cores = shard_width_hidden_dim_across_32_cores // num_devices
        shard_width_hidden_dim_across_8_cores = hidden_size // 8

        # Constants based on head_dim
        num_q_heads_per_kv_heads = num_attention_heads // num_kv_heads
        num_qkv_heads_per_device = num_kv_heads // num_devices
        total_width_per_group_of_qkv_heads = head_dim * (
            num_q_heads_per_kv_heads + 2
        )  # 16 q_heads + 1 k_heads + 1 v_heads
        total_width_of_qkv_heads_per_device = num_qkv_heads_per_device * total_width_per_group_of_qkv_heads
        shard_width_qkv_heads_per_device_across_8_cores = total_width_of_qkv_heads_per_device // 8

        # Embeddings
        model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    row_height,
                    shard_width_hidden_dim_per_device_across_32_cores,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["ATTN_MASK_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    row_height,
                    1,  # Dynamic
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["PARALLEL_ATTN_ADD_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    row_height,
                    shard_width_hidden_dim_per_device_across_32_cores,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["DROPOUT_ADD_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    row_height,
                    shard_width_hidden_dim_per_device_across_32_cores,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

        # Decoder
        model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    row_height,
                    shard_width_hidden_dim_across_32_cores,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["LN_ATTN_INPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    row_height,
                    shard_width_hidden_dim_across_32_cores,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["LN_ATTN_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    row_height,
                    shard_width_hidden_dim_across_32_cores,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["LN_ATTN_PROGCFG"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 4],
            subblock_w=8,
            block_h=1,
            block_w=8,
            inplace=False,
        )
        model_config["LN_MLP_PROGCFG"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 4],
            subblock_w=8,
            block_h=1,
            block_w=8,
            inplace=True,
        )
        model_config["LN_MLP_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    row_height,
                    shard_width_hidden_dim_across_32_cores,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

        # ATTN
        model_config["FUSED_QKV_MM_INPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_8_cores_grid,
                [
                    row_height,
                    shard_width_hidden_dim_across_8_cores,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["QKV_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 1),
            in0_block_w=32,  # TODO: Can this be larger
            out_subblock_h=1,  # TODO: Can this be larger
            out_subblock_w=1,
            per_core_M=row_height // 32,
            per_core_N=5,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_8_cores_grid,
                [
                    row_height,
                    shard_width_qkv_heads_per_device_across_8_cores,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_1_cores_grid,
                [
                    row_height,
                    total_width_per_group_of_qkv_heads,  # Must always be minimum a full group
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG
        model_config["CREATE_Q_HEADS_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_16_cores_grid,
                [
                    row_height,
                    head_dim,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["CREATE_KV_HEADS_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_1_cores_grid,
                [
                    row_height,
                    head_dim,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG
        model_config["KV_CACHE_SLICE_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    1,  # Dynamic
                    head_dim,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["K_TRANSPOSED_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG
        model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG
        model_config["SOFTMAX_PROGCFG"] = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 2),
            subblock_w=1,
            block_h=row_height // 32,
            block_w=1,  # Dynamic
        )
        model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG
        model_config["CONCAT_HEADS_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
        model_config["ATTN_ALL_GATHER_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    row_height,
                    shard_width_hidden_dim_across_32_cores,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["SELFOUT_MM_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
        model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
        model_config["MLP_REDUCE_SCATTER_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    row_height,
                    shard_width_hidden_dim_per_device_across_32_cores,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
        model_config["SELFOUT_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,  # TODO: Can this be larger
            out_subblock_h=1,  # TODO: Can this be larger
            out_subblock_w=1,
            per_core_M=row_height // 32,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        # MLP
        model_config["DENSE_H_TO_4H_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,  # TODO: Can this be larger
            out_subblock_h=1,  # TODO: Can this be larger
            out_subblock_w=4,
            per_core_M=row_height // 32,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=[ttnn.UnaryOpType.GELU, True],
            mcast_in0=True,
        )
        model_config["DENSE_4H_TO_H_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=row_height // 32,
            per_core_N=8,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    row_height,
                    shard_width_hidden_dim_across_32_cores,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["LN_F_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    row_height,
                    shard_width_hidden_dim_across_32_cores,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["LN_F_PROGCFG"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 4],
            subblock_w=8,
            block_h=1,
            block_w=8,
            inplace=True,
        )

        # LM Head
        model_config["LM_HEAD_MM_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
        model_config["LM_HEAD_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=row_height // 32,
            per_core_N=8,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    # uncomment if need to see all the configs
    # logger.debug(f"Falcon model config: \n{pretty_print_model_config(model_config)}")

    return model_config


def get_prefill_model_config(model_config_str, input_shape, num_devices):
    assert model_config_str in ACCEPTABLE_PREFILL_MODEL_CONFIG_STRS
    assert len(input_shape) == 2
    assert num_devices == 8, "Prefill is currently only supported on 8 devicess"

    DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
    L1_MEMCFG = ttnn.L1_MEMORY_CONFIG
    WIDTH_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)
    HEIGHT_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    BFLOAT16_DTYPE = ttnn.bfloat16
    BFP8_DTYPE = ttnn.bfloat8_b
    BFP4_DTYPE = ttnn.bfloat4_b

    # Set default dtype and mem_config based on model_config_str
    if model_config_str in ACCEPTABLE_PREFILL_MODEL_CONFIG_STRS:
        dtype_str, mem_config_str = model_config_str.split("-")
        mem_config = DRAM_MEMCFG if mem_config_str == "DRAM" else L1_MEMCFG
        dtype = getattr(ttnn.DataType, dtype_str)
    else:
        raise NotImplementedError(f"Model config {model_config_str} is not supported!")

    # Set defaults for dtype and mem_config for all ops
    model_config = {
        "LLM_MODE": "prefill",
        "DEFAULT_DTYPE": dtype,
        "DEFAULT_MEMCFG": mem_config,
        "MOVE_DECODER_OUTPUT_BOOL": False,
        "NUM_DEVICES": num_devices,
        "MAX_GRID_SIZE": (8, 4),
        "ALL_GATHER_NUM_LINKS": 2 if num_devices == 4 else 1,
        "DEFAULT_CACHE_PATH": Path(f"models/demos/t3000/falcon40b/datasets/"),
        "COMPUTE_KERNEL_CONFIG": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
        "COMPUTE_KERNEL_FP16_ACC_CONFIG": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        ),
        "DRAM_MEMCFG": DRAM_MEMCFG,
        "L1_MEMCFG": L1_MEMCFG,
        "BFP4_DTYPE": BFP4_DTYPE,
        "BFP8_DTYPE": BFP8_DTYPE,
        "BFLOAT16_DTYPE": BFLOAT16_DTYPE,
        "HEIGHT_SHARDED_MEMCFG": HEIGHT_SHARDED_MEMCFG,
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

    model_config["WORD_EMBEDDING_OUTPUT_DTYPE"] = BFLOAT16_DTYPE  # embeddings output and the residual stream

    # Set input df for AllGathers to bfp8 to save data bandwidth
    model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"] = BFP8_DTYPE  # MLP AllGather
    model_config["ATTENTION_OUT_DTYPE"] = BFP8_DTYPE  # Attention AllGather
    model_config["SELFOUT_MM_OUTPUT_DTYPE"] = BFP8_DTYPE  # AllGather at start of the decoder layer and final AllGather

    hidden_size = model_config_entries["hidden_size"]

    batch_size, seq_len = input_shape[0], input_shape[1]
    row_height = seq_len
    model_config["row_height"] = row_height
    model_config["BATCH_SIZE"] = batch_size
    model_config["SEQ_LEN"] = seq_len
    model_config["MAX_MM_SEQ_LEN"] = 2048
    model_config["MM_SEQ_LEN_BATCHED"] = 1024

    def get_grid_size_and_core_range_based_on_num_cores(num_cores):
        assert num_cores in (16, 32, 64)
        if num_cores == 16:
            attention_mm_grid_size = (8, 2)
            attn_core_range_set = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 1),
                    ),
                }
            )
        elif num_cores == 32:
            attention_mm_grid_size = (8, 4)
            attn_core_range_set = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 3),
                    ),
                }
            )
        else:
            attention_mm_grid_size = (8, 8)
            attn_core_range_set = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 7),
                    ),
                }
            )
        return attention_mm_grid_size, attn_core_range_set

    # Attention in slices: determine number of cores and shard spec
    attention_max_slice_size = 1024
    attention_slice_size = min(attention_max_slice_size, row_height)
    assert row_height % attention_slice_size == 0

    # MLP in slices: determine number of cores and shard spec
    use_mm_2d_start = 512
    model_config["MM_USE_MM2D_START"] = use_mm_2d_start
    mlp_max_slice_size = (
        256 if model_config["DEFAULT_DTYPE"] == BFLOAT16_DTYPE else 1024
    )  # BF16 runs out of L1 otherwise
    mlp_slice_size = min(mlp_max_slice_size, row_height)
    assert row_height % mlp_slice_size == 0

    mlp_num_slices = max(row_height // mlp_slice_size, 1)
    model_config["MLP_NUM_SLICES"] = mlp_num_slices

    if row_height >= use_mm_2d_start:
        mlp_num_cores = min(mlp_slice_size // 32 * 8, 64)
    else:  # use 1d matmuls and width sharding
        mlp_num_cores = 32

    mlpn_mm_grid_size, mlp_core_range_set = get_grid_size_and_core_range_based_on_num_cores(mlp_num_cores)
    model_config["MLP_GRID_SIZE"] = mlpn_mm_grid_size

    # Layernorm sharding (the interleaved OP does not fit in L1 for 40b hidden size)
    layernorm_num_cores_x = 8
    layernorm_max_num_cores_y = 8

    layernorm_slice_size = 1024

    (
        layernorm_block_sharded_mem_config,
        layernorm_block_sharded_prg_config,
        layernorm_block_sharded_prg_config_inplace,
        layernorm_params,
    ) = get_sharded_layernorm_specs_for_seqlen(
        layernorm_num_cores_x,
        layernorm_max_num_cores_y,
        row_height if row_height <= layernorm_slice_size else layernorm_slice_size,
        hidden_size,
        dtype,
    )

    model_config["PARTIAL_LN_MEMCFG"] = layernorm_block_sharded_mem_config
    model_config["PARTIAL_LN_PROGCFG"] = layernorm_block_sharded_prg_config
    model_config["PARTIAL_LN_INPLACE_PROGCFG"] = layernorm_block_sharded_prg_config_inplace

    model_config["layernorm_params"] = layernorm_params

    # MLP sharding specs

    if mlp_slice_size > use_mm_2d_start:
        model_config["MLP_INPUT_SHARD_SPEC"] = [
            row_height // mlp_num_slices // mlpn_mm_grid_size[1],
            hidden_size // mlpn_mm_grid_size[0],
        ]

        model_config["MLP_INPUT_SHARD_LAYOUT"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

        model_config["DENSE_H_TO_4H_MM_OPTIMIZED_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                mlp_core_range_set,
                [
                    row_height // mlp_num_slices // mlpn_mm_grid_size[1],
                    hidden_size // 8 // mlpn_mm_grid_size[0],
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["DENSE_4H_TO_H_MM_OPTIMIZED_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                mlp_core_range_set,
                [
                    row_height // mlp_num_slices // mlpn_mm_grid_size[1],
                    hidden_size // mlpn_mm_grid_size[0],
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    else:  # use 1d matmuls and width sharding
        model_config["MLP_INPUT_SHARD_SPEC"] = [
            row_height // mlp_num_slices,
            hidden_size // mlp_num_cores,
        ]

        model_config["MLP_INPUT_SHARD_LAYOUT"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED

        model_config["DENSE_H_TO_4H_MM_OPTIMIZED_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                mlp_core_range_set,
                [
                    row_height // mlp_num_slices,
                    hidden_size // 8 // mlp_num_cores,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["DENSE_4H_TO_H_MM_OPTIMIZED_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                mlp_core_range_set,
                [
                    row_height // mlp_num_slices,
                    hidden_size // mlp_num_cores,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

    # Attention parameters
    q_chunk_size = min(seq_len, 256)
    k_chunk_size = min(seq_len, 256)

    model_config["SDPA_PROGCFG"] = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=[8, 7],
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )

    return model_config


def get_sharded_layernorm_specs_for_seqlen(
    layernorm_num_cores_x, layernorm_max_num_cores_y, partial_seqlen, hidden_size, dtype
):
    for i in range(layernorm_max_num_cores_y, 0, -1):
        if (partial_seqlen // 32) % i == 0:
            layernorm_num_cores_y = i
            break

    num_tiles_per_core_h = partial_seqlen // 32 // layernorm_num_cores_y
    num_tiles_per_core_w = hidden_size // 32 // layernorm_num_cores_x

    layernorm_shard_height_hidden_dim = partial_seqlen // layernorm_num_cores_y
    layernorm_shard_width_hidden_dim = hidden_size // layernorm_num_cores_x

    core_range_block_sharded_layernorm = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(layernorm_num_cores_x - 1, layernorm_num_cores_y - 1),
            ),
        }
    )

    layernorm_block_sharded_mem_config = ttnn.MemoryConfig(
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

    layernorm_block_sharded_prg_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[layernorm_num_cores_x, layernorm_num_cores_y],
        subblock_w=8,
        block_h=num_tiles_per_core_h,
        block_w=num_tiles_per_core_w,
        inplace=False,
    )
    layernorm_block_sharded_prg_config_inplace = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[layernorm_num_cores_x, layernorm_num_cores_y],
        subblock_w=8,
        block_h=num_tiles_per_core_h,
        block_w=num_tiles_per_core_w,
        inplace=True,
    )

    layernorm_params = {
        "slice_size": partial_seqlen,
        "layernorm_num_cores_x": layernorm_num_cores_x,
        "layernorm_num_cores_y": layernorm_num_cores_y,
        "layernorm_max_num_cores_y": layernorm_max_num_cores_y,
        "layernorm_shard_height_hidden_dim": layernorm_shard_height_hidden_dim,
        "layernorm_shard_width_hidden_dim": layernorm_shard_width_hidden_dim,
        "num_tiles_per_core_h": num_tiles_per_core_h,
        "num_tiles_per_core_w": num_tiles_per_core_w,
    }

    return (
        layernorm_block_sharded_mem_config,
        layernorm_block_sharded_prg_config,
        layernorm_block_sharded_prg_config_inplace,
        layernorm_params,
    )
