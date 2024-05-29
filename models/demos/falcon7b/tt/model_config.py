# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from pathlib import Path
from models.utility_functions import is_grayskull, is_wormhole_b0

OP_KEYS = (
    # Inputs
    "INPUT",
    "ATTN_MASK",
    # Embeddings
    "WORD_EMBEDDING_WEIGHTS",
    "WORD_EMBEDDING_OUTPUT",
    # Decoder
    "INPUT_LAYERNORM_WEIGHTS",
    "INPUT_LAYERNORM_BIAS",
    "INPUT_LAYERNORM_OUTPUT",
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
    "LM_HEAD_MM_INPUT",
    "LM_HEAD_MM_WEIGHTS",
    "LM_HEAD_MM_OUTPUT",
)

NO_MEMCFG = ("SOFTMAX_OUTPUT",)

NO_DTYPE = (
    # Decoder
    "INPUT_LAYERNORM_OUTPUT",
    # Attention
    "ROTARY_EMBEDDING_OUTPUT",
    "CREATE_QKV_HEADS_OUTPUT",
    "K_TRANSPOSED_OUTPUT",
    "PRE_SOFTMAX_SCALE_OUTPUT",
    "PRE_SOFTMAX_MASK_OUTPUT",
    "SOFTMAX_OUTPUT",
    "CONCAT_HEADS_OUTPUT",
    # Decoder Cont
    "PARALLEL_ATTN_ADD_OUTPUT",
    "DROPOUT_ADD_OUTPUT",
    # Model
    "LN_F_OUTPUT",
)

ACCEPTABLE_MODEL_CONFIG_STRS = ("BFLOAT16-DRAM", "BFLOAT16-L1", "BFLOAT16-L1_SHARDED")


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


def get_model_config(model_config_str, prefill_seq_len=0):
    assert model_config_str in ACCEPTABLE_MODEL_CONFIG_STRS
    DRAM_MEMCFG = ttnn.experimental.tensor.MemoryConfig(
        ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
    )
    L1_MEMCFG = ttnn.experimental.tensor.MemoryConfig(
        ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.L1
    )
    BFP8_DTYPE = ttnn.experimental.tensor.DataType.BFLOAT8_B

    # Set default dtype and mem_config based on model_config_str
    if model_config_str in ("BFLOAT16-DRAM", "BFLOAT16-L1", "BFLOAT16-L1_SHARDED"):
        dtype_str, mem_config_str = model_config_str.split("-")
        # TODO: Set default memcfg for BFLOAT16-L1 to L1
        # mem_config = DRAM_MEMCFG if mem_config_str == "DRAM" else L1_MEMCFG
        mem_config = DRAM_MEMCFG
        dtype = (
            ttnn.experimental.tensor.DataType.BFLOAT16
            if dtype_str == "BFLOAT16"
            else ttnn.experimental.tensor.DataType.BFLOAT8_B
        )
    else:
        raise NotImplementedError(f"Model config {model_config_str} is not supported!")

    # Set defaults for dtype and mem_config for all ops
    model_config = {
        "l1_sharded": (mem_config_str == "L1_SHARDED"),
        "DEFAULT_DTYPE": dtype,
        "DEFAULT_MEMCFG": mem_config,
        "MOVE_DECODER_OUTPUT_BOOL": False,
        "DEFAULT_CACHE_PATH": Path(f"models/demos/falcon7b/datasets/"),
    }  # DEFAULT_MEMCFG also used to determine banking for ttl.device.InitializeDevice
    model_config.update({f"{key}_MEMCFG": mem_config for key in OP_KEYS if key not in NO_MEMCFG})
    model_config.update({f"{key}_DTYPE": dtype for key in OP_KEYS if key not in NO_DTYPE})

    # Input ids are UINT32
    model_config["INPUT_DTYPE"] = ttnn.experimental.tensor.DataType.UINT32

    # Matmul Weights must always be BFP8_B
    # Override defaults for certain configs
    for key in model_config.keys():
        if "MM_WEIGHTS_DTYPE" in key:
            model_config[key] = BFP8_DTYPE

    if model_config_str in ("BFLOAT16-L1", "BFLOAT16-L1_SHARDED"):
        model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["K_CACHE_SLICE_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["V_CACHE_SLICE_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["K_TRANSPOSED_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["PRE_SOFTMAX_SCALE_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["PRE_SOFTMAX_MASK_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"] = L1_MEMCFG

    if model_config_str in ("BFLOAT16-L1_SHARDED"):
        # Q, K, V are batch sharded across cores
        model_config[
            "ATTN_BATCH_SHARDED_MEMCFG"
        ] = lambda shard_height, shard_width: ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.experimental.tensor.BufferType.L1,
            ttnn.experimental.tensor.ShardSpec(
                ttnn.experimental.tensor.CoreRangeSet(
                    {
                        ttnn.experimental.tensor.CoreRange(
                            # Volume must match batch size
                            ttnn.experimental.tensor.CoreCoord(0, 0),
                            ttnn.experimental.tensor.CoreCoord(7, 3),
                        ),
                    }
                ),
                [
                    shard_height,
                    shard_width,
                ],
                ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

        model_config[
            "ATTN_BATCHED_MM_PROGCFG"
        ] = lambda block_w, per_core_M, per_core_N: ttnn.experimental.operations.primary.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=[8, 4],
            in0_block_w=block_w,
            out_subblock_h=1,  # TODO: Maximize
            out_subblock_w=1,  # TODO: Maximize
            per_core_M=per_core_M,
            per_core_N=per_core_N,
        )

        if is_wormhole_b0():
            model_config["COMPUTE_KERNEL_CONFIG"] = ttnn.experimental.tensor.WormholeComputeKernelConfig(
                math_fidelity=ttnn.experimental.tensor.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        else:
            model_config["COMPUTE_KERNEL_CONFIG"] = ttnn.experimental.tensor.GrayskullComputeKernelConfig(
                math_fidelity=ttnn.experimental.tensor.MathFidelity.LoFi,
                math_approx_mode=True,
            )

    # uncomment if need to see all the configs
    # logger.debug(f"Falcon model config: \n{pretty_print_model_config(model_config)}")
    set_prefill_config(model_config, prefill_seq_len, DRAM_MEMCFG)

    return model_config


def set_prefill_config(model_config, seq_len, dram_memcfg):
    model_config["PREFILL_OPTIMIZED_MODE"] = not is_grayskull()
    model_config["MLP_SEQ_LEN"] = seq_len
    model_config["MLP_PADDING_VALUE"] = 4608
    model_config["MLP_GRID_SIZE"] = (8, 8)

    if is_wormhole_b0():
        default_kernel_config = ttnn.experimental.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttnn.experimental.tensor.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
    else:
        default_kernel_config = ttnn.experimental.tensor.GrayskullComputeKernelConfig(
            math_fidelity=ttnn.experimental.tensor.MathFidelity.LoFi,
            math_approx_mode=True,
        )
    model_config["MLP_KERNEL_CONFIG"] = default_kernel_config

    mm_h_to_4h_prog_cfg = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=model_config["MLP_GRID_SIZE"],
        in0_block_w=3,
        out_subblock_h=1,
        out_subblock_w=1,  # 8,
        per_core_M=4,
        per_core_N=72,
        transpose_mcast=False,
        fused_activation=[ttnn.experimental.tensor.FusibleActivation.GELU, True],
    )
    model_config["DENSE_H_TO_4H_MM_PROGCFG"] = mm_h_to_4h_prog_cfg

    mm_4h_to_h_prog_cfg = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=model_config["MLP_GRID_SIZE"],
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=1,  # 6,
        per_core_M=4,
        per_core_N=18,
        transpose_mcast=False,
        fused_activation=None,
    )
    model_config["DENSE_4H_TO_H_MM_PROGCFG"] = mm_4h_to_h_prog_cfg
    model_config["MLP_INTERLEAVED_TO_SHARDED_MEM_CFG"] = dram_memcfg

    model_config["FUSED_QKV_MM_OPTIMIZED_MEMCFG"] = dram_memcfg
    model_config[
        "FUSED_QKV_MM_OPTIMIZED_PROGCFG"
    ] = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2,
        per_core_M=8,
        per_core_N=21,
        out_subblock_h=1,
        out_subblock_w=1,  # 7,
        transpose_mcast=False,
        fused_activation=None,
    )
    compute_kernel_config = ttnn.experimental.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttnn.experimental.tensor.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    model_config["FUSED_QKV_MM_OPTIMIZED_KERNEL_CONFIG"] = compute_kernel_config
    model_config["SELFOUT_MM_OPTIMIZED_KERNEL_CONFIG"] = compute_kernel_config
    model_config["ATTN_OPTIMIZED_GRID_SIZE"] = (8, 8)
    model_config["ATTN_OPTIMIZED_MEMCFG"] = dram_memcfg
    model_config[
        "ATTN_OPTIMIZED_ALLOWED_NUM_CORES"
    ] = 57  # We can't use full grid for attention, as it causes di/dt problems. Use 64 cores when issue #8644 is resolved.

    model_config[
        "QKT_OPTIMIZED_PROGCFG"
    ] = lambda tiles_per_shard, seq_len, subblock_h, subblock_w: ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=model_config["ATTN_OPTIMIZED_GRID_SIZE"],
        in0_block_w=2,
        per_core_M=tiles_per_shard,
        per_core_N=seq_len // 32,
        out_subblock_h=1,  # subblock_h,
        out_subblock_w=1,  # subblock_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )
    model_config["QKTV_MM_OPTIMIZED_MEMCFG"] = ttnn.experimental.tensor.MemoryConfig(
        memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.experimental.tensor.BufferType.L1,
    )

    model_config["QKTV_AND_SOFTMAX_OPTIMIZED_KERNEL_CONFIG"] = ttnn.experimental.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttnn.experimental.tensor.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    model_config[
        "SOFTMAX_OPTIMIZED_PROGCFG"
    ] = lambda grid_size, subblock_w, block_h, block_w: ttnn.experimental.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=subblock_w,
        block_h=block_h,
        block_w=block_w,
    )

    model_config[
        "QKTV_MM_OPTIMIZED_PROGCFG"
    ] = lambda tiles_per_shard, seq_len, subblock_h: ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=model_config["ATTN_OPTIMIZED_GRID_SIZE"],
        in0_block_w=seq_len // 32,
        per_core_M=tiles_per_shard,
        per_core_N=2,
        out_subblock_h=1,  # subblock_h,
        out_subblock_w=1,  # 2,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )

    model_config["LM_HEAD_KERNEL_CONFIG"] = default_kernel_config


model_config_entries = {
    "_name_or_path": "tiiuae/falcon-7b-instruct",
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
    "hidden_size": 4544,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "falcon",
    "multi_query": True,
    "new_decoder_architecture": False,
    "num_attention_heads": 71,
    "num_hidden_layers": 32,
    "num_kv_heads": 71,
    "parallel_attn": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.28.1",
    "use_cache": True,
    "vocab_size": 65024,
}
