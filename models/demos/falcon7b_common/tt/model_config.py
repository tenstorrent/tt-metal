# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
from loguru import logger
from pathlib import Path
from transformers import FalconConfig
from models.utility_functions import is_grayskull, is_wormhole_b0

OP_KEYS = (
    # Inputs
    "INPUT",
    "ATTN_MASK",
    "ATTN_MASK_OPTIMIZED_PREFILL",
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

model_config_entries = FalconConfig.from_pretrained("tiiuae/falcon-7b-instruct")


def find_subblock_w(block_w, max_value):
    factors = [i for i in range(1, max_value + 1) if block_w % i == 0]
    return max(factors)


def get_ln_block_sharded_config(height_dim, hidden_dim):
    ln_max_num_cores_y = 8
    for i in range(ln_max_num_cores_y, 0, -1):
        if (height_dim // 32) % i == 0:
            ln_num_cores_y = i
            break

    ln_num_cores_x = 8
    num_tiles_per_core_h = math.ceil(height_dim / ln_num_cores_y / 32)
    num_tiles_per_core_w = math.ceil(hidden_dim / ln_num_cores_x / 32)
    ln_shard_height_hidden_dim = num_tiles_per_core_h * 32
    ln_shard_width_hidden_dim = num_tiles_per_core_w * 32

    core_range_block_sharded_layernorm = ttnn.experimental.tensor.CoreRangeSet(
        {
            ttnn.experimental.tensor.CoreRange(
                ttnn.experimental.tensor.CoreCoord(0, 0),
                ttnn.experimental.tensor.CoreCoord(ln_num_cores_x - 1, ln_num_cores_y - 1),
            ),
        }
    )

    ln_block_sharded_mem_config = ttnn.experimental.tensor.MemoryConfig(
        ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.experimental.tensor.BufferType.L1,
        ttnn.experimental.tensor.ShardSpec(
            core_range_block_sharded_layernorm,
            [
                ln_shard_height_hidden_dim,
                ln_shard_width_hidden_dim,
            ],
            ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    fp32_dest_acc_en = False
    if is_wormhole_b0():
        ln_block_sharded_compute_kernel_config = ttnn.experimental.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttnn.experimental.tensor.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=False,
        )
    else:
        ln_block_sharded_compute_kernel_config = ttnn.experimental.tensor.GrayskullComputeKernelConfig(
            math_fidelity=ttnn.experimental.tensor.MathFidelity.HiFi4,
            math_approx_mode=True,
        )

    max_tiles_in_dest = 8
    if fp32_dest_acc_en and is_wormhole_b0():
        max_tiles_in_dest = 4
    subblock_w = find_subblock_w(num_tiles_per_core_w, max_tiles_in_dest)
    ln_block_sharded_prog_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[ln_num_cores_x, ln_num_cores_y],
        subblock_w=subblock_w,
        block_h=num_tiles_per_core_h,
        block_w=num_tiles_per_core_w,
        inplace=True,
    )

    return ln_block_sharded_mem_config, ln_block_sharded_prog_config, ln_block_sharded_compute_kernel_config


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


def get_model_config(model_config_str, prefill_seq_len=0, decode_batch_size=32):
    assert model_config_str in ACCEPTABLE_MODEL_CONFIG_STRS
    DRAM_MEMCFG = ttnn.experimental.tensor.MemoryConfig(
        ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
    )
    L1_MEMCFG = ttnn.experimental.tensor.MemoryConfig(
        ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.L1
    )
    BFP8_DTYPE = ttnn.experimental.tensor.DataType.BFLOAT8_B
    BFP4_DTYPE = ttnn.experimental.tensor.DataType.BFLOAT4_B

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
        "DEFAULT_CACHE_PATH": Path(f"models/demos/falcon7b_common/datasets/"),
    }  # DEFAULT_MEMCFG also used to determine banking for ttl.device.InitializeDevice
    model_config.update({f"{key}_MEMCFG": mem_config for key in OP_KEYS if key not in NO_MEMCFG})
    model_config.update({f"{key}_DTYPE": dtype for key in OP_KEYS if key not in NO_DTYPE})

    (
        ln_block_sharded_mem_config_decode,
        ln_block_sharded_prog_config_decode,
        ln_compute_kernel_config_decode,
    ) = get_ln_block_sharded_config(decode_batch_size, model_config_entries.hidden_size)

    model_config["LAYERNORM_BLOCK_SHARDED_MEM_CFG"] = {}
    model_config["LAYERNORM_BLOCK_SHARDED_PROG_CFG"] = {}
    model_config["LAYERNORM_BLOCK_SHARDED_COMPUTE_KERNEL_CONFIG"] = {}
    model_config["LAYERNORM_BLOCK_SHARDED_MEM_CFG"][decode_batch_size] = ln_block_sharded_mem_config_decode
    model_config["LAYERNORM_BLOCK_SHARDED_PROG_CFG"][decode_batch_size] = ln_block_sharded_prog_config_decode
    model_config["LAYERNORM_BLOCK_SHARDED_COMPUTE_KERNEL_CONFIG"][decode_batch_size] = ln_compute_kernel_config_decode

    # Input ids are UINT32
    model_config["INPUT_DTYPE"] = ttnn.experimental.tensor.DataType.UINT32

    # Use BFP4_B for attention mask in optimized prefill
    model_config["ATTN_MASK_OPTIMIZED_PREFILL_DTYPE"] = BFP4_DTYPE

    # Matmul Weights must always be BFP8_B
    # Override defaults for certain configs
    for key in model_config.keys():
        if "MM_WEIGHTS_DTYPE" in key:
            model_config[key] = BFP8_DTYPE

    if model_config_str in ("BFLOAT16-L1", "BFLOAT16-L1_SHARDED"):
        if model_config_str == "BFLOAT16-L1_SHARDED":
            model_config["ATTN_MASK_MEMCFG"] = L1_MEMCFG
        model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"] = L1_MEMCFG
        if not model_config_str == "BFLOAT16-L1_SHARDED":
            # Don't send keys to l1 before converting to l1-sharded (after kcache update) to avoid l1 framgentation issues with kv_cache_size=2048
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
        ] = lambda block_w, per_core_M, per_core_N: ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=[8, 4],
            in0_block_w=block_w,
            out_subblock_h=1,  # TODO: Maximize
            out_subblock_w=1,  # TODO: Maximize
            per_core_M=per_core_M,
            per_core_N=per_core_N,
        )

        if is_wormhole_b0():
            model_config["PRE_SOFTMAX_MM_COMPUTE_KERNEL_CONFIG"] = ttnn.experimental.tensor.WormholeComputeKernelConfig(
                math_fidelity=ttnn.experimental.tensor.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
            model_config[
                "POST_SOFTMAX_MM_COMPUTE_KERNEL_CONFIG"
            ] = ttnn.experimental.tensor.WormholeComputeKernelConfig(
                math_fidelity=ttnn.experimental.tensor.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        else:
            gs_compute_kernel_config = ttnn.experimental.tensor.GrayskullComputeKernelConfig(
                math_fidelity=ttnn.experimental.tensor.MathFidelity.LoFi,
                math_approx_mode=True,
            )
            model_config["PRE_SOFTMAX_MM_COMPUTE_KERNEL_CONFIG"] = gs_compute_kernel_config
            model_config["POST_SOFTMAX_MM_COMPUTE_KERNEL_CONFIG"] = gs_compute_kernel_config

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

    mm_h_to_4h_prog_cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=model_config["MLP_GRID_SIZE"],
        in0_block_w=3,
        out_subblock_h=1,
        out_subblock_w=1,  # 8,
        per_core_M=4,
        per_core_N=72,
        transpose_mcast=False,
        fused_activation=[ttnn.UnaryOpType.GELU, True],
    )
    model_config["DENSE_H_TO_4H_MM_PROGCFG"] = mm_h_to_4h_prog_cfg

    mm_4h_to_h_prog_cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
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
    model_config["FUSED_QKV_MM_OPTIMIZED_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
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
    ] = lambda tiles_per_shard, seq_len, subblock_h, subblock_w: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
    ] = lambda grid_size, subblock_w, block_h, block_w: ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=subblock_w,
        block_h=block_h,
        block_w=block_w,
    )

    model_config[
        "QKTV_MM_OPTIMIZED_PROGCFG"
    ] = lambda tiles_per_shard, seq_len, subblock_h: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
    (
        ln_block_sharded_mem_config_prefill,
        ln_block_sharded_prog_config_prefill,
        ln_compute_kernel_config_prefill,
    ) = get_ln_block_sharded_config(seq_len, model_config_entries.hidden_size)
    model_config["LAYERNORM_BLOCK_SHARDED_MEM_CFG"][seq_len] = ln_block_sharded_mem_config_prefill
    model_config["LAYERNORM_BLOCK_SHARDED_PROG_CFG"][seq_len] = ln_block_sharded_prog_config_prefill
    model_config["LAYERNORM_BLOCK_SHARDED_COMPUTE_KERNEL_CONFIG"][seq_len] = ln_compute_kernel_config_prefill
