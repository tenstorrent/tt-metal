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
    "SIN_CACHED_WEIGHTS",
    "COS_CACHED_WEIGHTS",
    # Attention
    "FUSED_QKV_MM_WEIGHTS",
    "FUSED_QKV_MM_OUTPUT",
    "CREATE_QKV_HEADS_OUTPUT",
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
    "DENSE_4H_TO_H_MM_WEIGHTS",
    "DENSE_4H_TO_H_MM_OUTPUT",
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

ACCEPTABLE_MODEL_CONFIG_STRS = ("BFLOAT16-SHARDED", "BFLOAT8_B-SHARDED")

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
    assert model_config_str in ACCEPTABLE_MODEL_CONFIG_STRS
    assert llm_mode in ("prefill", "decode")
    assert len(input_shape) == 2
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
        "ALL_GATHER_NUM_LINKS": 2 if num_devices == 4 else 1,
        "DEFAULT_CACHE_PATH": Path(f"models/demos/falcon40b/datasets/"),
        "COMPUTE_KERNEL_CONFIG": ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
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
        head_dim = 64
        hidden_size = model_config_entries["hidden_size"]
        vocab_size = model_config_entries["vocab_size"]
        num_attention_heads = model_config_entries["num_attention_heads"]
        num_kv_heads = model_config_entries["num_kv_heads"]

        batch, seq_len = input_shape
        if llm_mode == "prefill":
            shard_height = seq_len
        elif llm_mode == "decode":
            assert batch == 32
            shard_height = batch

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
        model_config["ATTN_MASK_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    shard_height,
                    1,  # Dynamic
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["PARALLEL_ATTN_ADD_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
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
        model_config["DROPOUT_ADD_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
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

        # Decoder
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
        model_config["LN_ATTN_PROGCFG"] = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 4],
            subblock_w=8,
            block_h=1,
            block_w=8,
            math_fidelity=ttl.tensor.MathFidelity.HiFi4,
            im_data_format=ttl.tensor.DataType.BFLOAT16,
            out_data_format=model_config["LN_ATTN_OUTPUT_DTYPE"],
            inplace=False,
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

        # ATTN
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
        if num_devices == 4:
            model_config["QKV_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 1),
                in0_block_w=32,  # TODO: Can this be larger
                out_subblock_h=1,  # TODO: Can this be larger
                out_subblock_w=3,
                per_core_M=shard_height // 32,
                per_core_N=9,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        else:
            model_config["QKV_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 1),
                in0_block_w=32,  # TODO: Can this be larger
                out_subblock_h=1,  # TODO: Can this be larger
                out_subblock_w=1,
                per_core_M=shard_height // 32,
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
        # TODO: Remove this once nlp_create_qkv_heads supports HEIGHT > 32 for sharded
        if num_devices == 4:
            model_config["CREATE_Q_HEADS_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.BufferType.L1,
                ttl.tensor.ShardSpec(
                    shard_spec_32_cores_grid,
                    [
                        shard_height,
                        head_dim,
                    ],
                    ttl.tensor.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
            model_config["CREATE_KV_HEADS_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.BufferType.L1,
                ttl.tensor.ShardSpec(
                    shard_spec_2_cores_grid,
                    [
                        shard_height,
                        head_dim,
                    ],
                    ttl.tensor.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
        elif num_devices == 8:
            model_config["CREATE_Q_HEADS_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.BufferType.L1,
                ttl.tensor.ShardSpec(
                    shard_spec_16_cores_grid,
                    [
                        shard_height,
                        head_dim,
                    ],
                    ttl.tensor.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
            model_config["CREATE_KV_HEADS_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.BufferType.L1,
                ttl.tensor.ShardSpec(
                    shard_spec_1_cores_grid,
                    [
                        shard_height,
                        head_dim,
                    ],
                    ttl.tensor.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
        model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG
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
        model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG
        # TODO: Remove if we can update prefill matmul to directly output sharded
        model_config["PREFILL_4CHIPS_PRE_SOFTMAX_MM_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    shard_height,
                    shard_height,
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["PREFILL_4CHIPS_POST_SOFTMAX_MM_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    shard_height,
                    head_dim,
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        if num_devices == 4:
            model_config["SOFTMAX_PROGCFG"] = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                subblock_w=1,
                block_h=shard_height // 32,
                block_w=1,  # Dynamic
                math_fidelity=ttl.tensor.MathFidelity.HiFi4,
                im_data_format=ttl.tensor.DataType.BFLOAT16,
            )
        elif num_devices == 8:
            model_config["SOFTMAX_PROGCFG"] = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 2),
                subblock_w=1,
                block_h=shard_height // 32,
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
        model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
        model_config["MLP_ALL_GATHER_OUTPUT_MEMCFG"] = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_32_cores_grid,
                [
                    shard_height,
                    shard_width_4x_hidden_dim_across_32_cores,
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
        if num_devices == 4:
            model_config["SELFOUT_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=8,  # TODO: Can this be larger
                out_subblock_h=1,  # TODO: Can this be larger
                out_subblock_w=2,
                per_core_M=shard_height // 32,
                per_core_N=2,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
            # MLP
            model_config[
                "DENSE_H_TO_4H_MM_PROGCFG"
            ] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=8,  # TODO: Can this be larger
                out_subblock_h=1,  # TODO: Can this be larger
                out_subblock_w=4,
                per_core_M=shard_height // 32,
                per_core_N=8,
                fuse_batch=True,
                fused_activation=[ttl.tensor.FusibleActivation.GELU, True],
                mcast_in0=True,
            )
            model_config[
                "DENSE_4H_TO_H_MM_PROGCFG"
            ] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=32,  # TODO: Can this be larger
                out_subblock_h=1,  # TODO: Can this be larger
                out_subblock_w=2,
                per_core_M=shard_height // 32,
                per_core_N=2,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        elif num_devices == 8:
            model_config["SELFOUT_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=8,  # TODO: Can this be larger
                out_subblock_h=1,  # TODO: Can this be larger
                out_subblock_w=1,
                per_core_M=shard_height // 32,
                per_core_N=1,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
            # MLP
            model_config[
                "DENSE_H_TO_4H_MM_PROGCFG"
            ] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=8,  # TODO: Can this be larger
                out_subblock_h=1,  # TODO: Can this be larger
                out_subblock_w=4,
                per_core_M=shard_height // 32,
                per_core_N=4,
                fuse_batch=True,
                fused_activation=[ttl.tensor.FusibleActivation.GELU, True],
                mcast_in0=True,
            )
            model_config[
                "DENSE_4H_TO_H_MM_PROGCFG"
            ] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=32,  # TODO: Can this be larger
                out_subblock_h=1,  # TODO: Can this be larger
                out_subblock_w=1,
                per_core_M=shard_height // 32,
                per_core_N=1,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )

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
        if num_devices == 4:
            model_config["LM_HEAD_MM_PROGCFG"] = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
        elif num_devices == 8:
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

    # uncomment if need to see all the configs
    # logger.debug(f"Falcon model config: \n{pretty_print_model_config(model_config)}")

    return model_config
