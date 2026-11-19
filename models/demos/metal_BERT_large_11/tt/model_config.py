# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
from loguru import logger
from transformers import BertConfig, BertForQuestionAnswering

import ttnn
from models.demos.metal_BERT_large_11.tt import grid

NUM_LAYERS_ENV = "BERT_NUM_LAYERS"

OP_MEMCFG_KEYS = (
    # EMBEDDINGS
    "INPUT_EMBEDDINGS_WEIGHTS_MEMCFG",
    "INPUT_EMBEDDINGS_MEMCFG",
    "OUTPUT_EMBEDDINGS_MEMCFG",
    "EMBEDDINGS_LAYERNORM_GAMMA_MEMCFG",
    "EMBEDDINGS_LAYERNORM_BETA_MEMCFG",
    # MHA
    "OP1_FUSED_QKV_MM_INPUT_MEMCFG",
    "OP1_FUSED_QKV_MM_WEIGHTS_MEMCFG",  # Needs to be DRAM
    "OP1_FUSED_QKV_MM_BIAS_MEMCFG",
    "OP1_FUSED_QKV_MM_OUTPUT_MEMCFG",
    "OP2_SPLIT_QKV_HEADS_OUTPUT_MEMCFG",
    "OP3_PRE_SOFTMAX_BMM_OUTPUT_MEMCFG",
    "OP4_SOFTMAX_ATTENTION_MASK_MEMCFG",
    "OP5_POST_SOFTMAX_BMM_OUTPUT_MEMCFG",
    "OP6_CONCATENATE_ATTENTION_HEADS_OUTPUT_MEMCFG",
    # MHA SELFOUT ATTENTION
    "OP7_SELFOUT_WEIGHTS_MEMCFG",
    "OP7_SELFOUT_BIAS_MEMCFG",
    "OP7_SELFOUT_OUTPUT_MEMCFG",
    # MHA LAYERNORM
    "OP8_LAYERNORM_GAMMA_MEMCFG",
    "OP8_LAYERNORM_BETA_MEMCFG",
    "OP8_LAYERNORM_OUTPUT_MEMCFG",
    # FFN
    "OP9_FF1_MM_WEIGHTS_MEMCFG",
    "OP9_FF1_MM_BIAS_MEMCFG",
    "OP9_FF1_MM_OUTPUT_MEMCFG",
    "OP10_FF2_MM_WEIGHTS_MEMCFG",
    "OP10_FF2_MM_BIAS_MEMCFG",
    "OP10_FF2_MM_OUTPUT_MEMCFG",
    # FFN LAYERNORM
    "OP11_LAYERNORM_GAMMA_MEMCFG",
    "OP11_LAYERNORM_BETA_MEMCFG",
    "OP11_LAYERNORM_OUTPUT_MEMCFG",
    # After all encoders
    "QA_LINEAR_WEIGHTS_MEMCFG",
    "QA_LINEAR_BIAS_MEMCFG",
    "QA_LINEAR_OUTPUT_MEMCFG",
)
OP_DTYPE_KEYS = (
    "INPUT_EMBEDDINGS_WEIGHTS_DTYPE",
    "EMBEDDINGS_LAYERNORM_GAMMA_DTYPE",
    "EMBEDDINGS_LAYERNORM_BETA_DTYPE",
    # MHA
    "OP1_FUSED_QKV_MM_INPUT_DTYPE",
    "OP1_FUSED_QKV_MM_WEIGHTS_DTYPE",
    "OP1_FUSED_QKV_MM_BIAS_DTYPE",
    "OP1_FUSED_QKV_MM_OUTPUT_DTYPE",
    "OP3_PRE_SOFTMAX_BMM_OUTPUT_DTYPE",
    "OP4_SOFTMAX_ATTENTION_MASK_DTYPE",
    "OP5_POST_SOFTMAX_BMM_OUTPUT_DTYPE",
    # MHA SELFOUT ATTENTION
    "OP7_SELFOUT_WEIGHTS_DTYPE",
    "OP7_SELFOUT_BIAS_DTYPE",
    "OP7_SELFOUT_OUTPUT_DTYPE",
    # MHA LAYERNORM
    "OP8_LAYERNORM_GAMMA_DTYPE",
    "OP8_LAYERNORM_BETA_DTYPE",
    "OP8_LAYERNORM_OUTPUT_DTYPE",  # Used for ffn sub-graph test, might need in the future with mixed precision
    # FFN
    "OP9_FF1_MM_WEIGHTS_DTYPE",
    "OP9_FF1_MM_BIAS_DTYPE",
    "OP9_FF1_MM_OUTPUT_DTYPE",
    "OP10_FF2_MM_WEIGHTS_DTYPE",
    "OP10_FF2_MM_BIAS_DTYPE",
    "OP10_FF2_MM_OUTPUT_DTYPE",
    # FFN LAYERNORM
    "OP11_LAYERNORM_GAMMA_DTYPE",
    "OP11_LAYERNORM_BETA_DTYPE",
    # After all encoders
    "QA_LINEAR_WEIGHTS_DTYPE",
    "QA_LINEAR_BIAS_DTYPE",
)

ACCEPTABLE_MODEL_CONFIG_STRS = (
    "BFLOAT8_B-DRAM",
    "BFLOAT16-DRAM",
    "BFLOAT8_B-L1",
    "BFLOAT16-L1",
    "MIXED_PRECISION_BATCH9",
    "MIXED_PRECISION_BATCH8",
    "BFLOAT8_B-SHARDED",
)


def pretty_print_model_config(model_config):
    print_str = []
    for key, val in model_config.items():
        if key.endswith("MEMCFG"):
            print_str.append(f"{key}: {val.memory_layout, val.buffer_type}")

        elif key.endswith("DTYPE") or key.endswith("BOOL"):
            print_str.append(f"{key}: {val}")
    return "\n".join(print_str)


def get_model_config(batch, device_grid_size, model_config_str):
    assert model_config_str in ACCEPTABLE_MODEL_CONFIG_STRS
    DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
    L1_MEMCFG = ttnn.L1_MEMORY_CONFIG
    BLOCK_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)
    HEIGHT_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)

    # Set default dtype and mem_config based on model_config_str
    if model_config_str in (
        "BFLOAT8_B-DRAM",
        "BFLOAT16-DRAM",
        "BFLOAT8_B-L1",
        "BFLOAT16-L1",
    ):
        dtype_str, mem_config_str = model_config_str.split("-")
        mem_config = DRAM_MEMCFG if mem_config_str == "DRAM" else L1_MEMCFG
        dtype = ttnn.bfloat16 if dtype_str == "BFLOAT16" else ttnn.bfloat8_b

    elif model_config_str in ("MIXED_PRECISION_BATCH9", "MIXED_PRECISION_BATCH8"):
        dtype = ttnn.bfloat8_b
        mem_config = L1_MEMCFG
    elif model_config_str in ("BFLOAT8_B-SHARDED"):
        dtype = ttnn.bfloat8_b
        mem_config = BLOCK_SHARDED_MEMCFG
    else:
        raise NotImplementedError(f"Model config {model_config_str} is not supported!")

    if mem_config == L1_MEMCFG and not grid.fits_l1_activations(batch, device_grid_size):
        logger.warning(
            f"Grid {device_grid_size} does not hold {model_config_str}'s L1 activations; running them in DRAM"
        )
        mem_config = DRAM_MEMCFG

    # Set defaults for dtype and mem_config for all ops
    model_config = {
        "DEFAULT_DTYPE": dtype,
        "DEFAULT_MEMCFG": mem_config,
        "MOVE_ENCODER_OUTPUT_BOOL": False,
        "DEALLOC_INPUT_EMBEDS_AFTER_POSITION_EMBEDS": False,
    }  # DEFAULT_MEMCFG also used to determine banking for ttnn.experimental.device.InitializeDevice
    model_config.update(dict(zip(OP_MEMCFG_KEYS, [mem_config] * len(OP_MEMCFG_KEYS))))
    model_config.update(dict(zip(OP_DTYPE_KEYS, [dtype] * len(OP_DTYPE_KEYS))))

    # Layernorm Gamma Beta must always be BFLOAT16
    model_config.update(
        {
            "INPUT_EMBEDDINGS_WEIGHTS_DTYPE": ttnn.bfloat16,
            "EMBEDDINGS_LAYERNORM_GAMMA_DTYPE": ttnn.bfloat16,
            "EMBEDDINGS_LAYERNORM_BETA_DTYPE": ttnn.bfloat16,
            "OP8_LAYERNORM_GAMMA_DTYPE": ttnn.bfloat16,
            "OP8_LAYERNORM_BETA_DTYPE": ttnn.bfloat16,
            "OP11_LAYERNORM_GAMMA_DTYPE": ttnn.bfloat16,
            "OP11_LAYERNORM_BETA_DTYPE": ttnn.bfloat16,
        }
    )

    # Weights that must always be DRAM
    model_config.update(
        {
            # Embeddings
            "INPUT_EMBEDDINGS_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            # MHA
            "OP1_FUSED_QKV_MM_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "OP1_FUSED_QKV_MM_BIAS_MEMCFG": DRAM_MEMCFG,
            # MHA SELFOUT ATTENTION
            "OP7_SELFOUT_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "OP7_SELFOUT_BIAS_MEMCFG": DRAM_MEMCFG,
            # FFN
            "OP9_FF1_MM_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "OP9_FF1_MM_BIAS_MEMCFG": DRAM_MEMCFG,
            "OP10_FF2_MM_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "OP10_FF2_MM_BIAS_MEMCFG": DRAM_MEMCFG,
            # After all encoders
            "QA_LINEAR_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "QA_LINEAR_BIAS_MEMCFG": DRAM_MEMCFG,
        }
    )

    # Override defaults for certain configs
    if model_config_str == "BFLOAT16-L1":
        new_config_values = {
            # MHA
            "OP3_PRE_SOFTMAX_BMM_OUTPUT_MEMCFG": DRAM_MEMCFG,
            # FFN
            "OP9_FF1_MM_OUTPUT_MEMCFG": DRAM_MEMCFG,
        }
        model_config.update(new_config_values)

    elif model_config_str in ("BFLOAT8_B-L1", "BFLOAT8_B-DRAM") and grid.fits_matmul_configs(batch, device_grid_size):
        # Every per_core_N below covers its output's width on grid.MIN_MATMUL_COLUMNS columns or more.
        grid_size = ttnn.CoreCoord(grid.matmul_columns(device_grid_size), batch)
        new_config_values = {
            "OP3_PRE_SOFTMAX_BMM_CONFIG": ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=6,
                per_core_M=12,
                per_core_N=12,
            ),
            "OP5_POST_SOFTMAX_BMM_CONFIG": ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=12,
                out_subblock_h=4,
                out_subblock_w=2,
                per_core_M=12,
                per_core_N=2,
            ),
            "OP7_SELFOUT_CONFIG": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=4,
                out_subblock_h=2,
                out_subblock_w=4,
                per_core_M=12,
                per_core_N=4,
                transpose_mcast=False,
                fused_activation=None,
            ),
            "OP9_FF1_MM_CONFIG": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=8,
                per_core_M=12,
                per_core_N=16,
                transpose_mcast=False,
                fused_activation=(ttnn.UnaryOpType.GELU, True),
            ),
            "OP10_FF2_MM_CONFIG": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=16,
                out_subblock_h=2,
                out_subblock_w=4,
                per_core_M=12,
                per_core_N=4,
                transpose_mcast=False,
                fused_activation=None,
            ),
            "OP4_SOFTMAX_CONFIG": ttnn.SoftmaxDefaultProgramConfig(),
            "OP8_LAYERNORM_CONFIG": ttnn.LayerNormDefaultProgramConfig(),
            "OP11_LAYERNORM_CONFIG": ttnn.LayerNormDefaultProgramConfig(),
        }
        model_config.update(new_config_values)

    elif model_config_str == "MIXED_PRECISION_BATCH9":
        new_config_values = {
            # MHA
            "OP3_PRE_SOFTMAX_BMM_OUTPUT_MEMCFG": DRAM_MEMCFG,
            # MHA
            "OP1_FUSED_QKV_MM_INPUT_DTYPE": ttnn.bfloat16,
            "OP3_PRE_SOFTMAX_BMM_OUTPUT_DTYPE": ttnn.bfloat16,
            "OP4_SOFTMAX_ATTENTION_MASK_DTYPE": ttnn.bfloat16,
            # MHA SELFOUT ATTENTION
            "OP7_SELFOUT_OUTPUT_DTYPE": ttnn.bfloat16,
            # MHA LAYERNORM
            "OP8_LAYERNORM_OUTPUT_DTYPE": ttnn.bfloat16,  # Used for ffn sub-graph test, might need in the future with mixed precision
            # FFN
            "OP10_FF2_MM_OUTPUT_DTYPE": ttnn.bfloat16,
            # After all encoders
            "QA_LINEAR_WEIGHTS_DTYPE": ttnn.bfloat16,
            "QA_LINEAR_BIAS_DTYPE": ttnn.bfloat16,
        }
        model_config.update(new_config_values)

    elif model_config_str == "MIXED_PRECISION_BATCH8":
        new_config_values = {
            "DEALLOC_INPUT_EMBEDS_AFTER_POSITION_EMBEDS": True,
            "MOVE_ENCODER_OUTPUT_BOOL": True,
            # MHA
            "OP1_FUSED_QKV_MM_INPUT_DTYPE": ttnn.bfloat16,
            "OP3_PRE_SOFTMAX_BMM_OUTPUT_DTYPE": ttnn.bfloat16,
            "OP4_SOFTMAX_ATTENTION_MASK_DTYPE": ttnn.bfloat16,
            # MHA SELFOUT ATTENTION
            "OP7_SELFOUT_OUTPUT_DTYPE": ttnn.bfloat16,
            # MHA LAYERNORM
            "OP8_LAYERNORM_OUTPUT_DTYPE": ttnn.bfloat16,  # Used for ffn sub-graph test, might need in the future with mixed precision
            # FFN
            "OP10_FF2_MM_OUTPUT_DTYPE": ttnn.bfloat16,
            # After all encoders
            "QA_LINEAR_WEIGHTS_DTYPE": ttnn.bfloat16,
            "QA_LINEAR_BIAS_DTYPE": ttnn.bfloat16,
        }
        model_config.update(new_config_values)

    elif model_config_str == "BFLOAT8_B-SHARDED":
        activation_grid_dim = grid.SHARDED_CORES_PER_IMAGE
        sharded = grid.sharded_grid(batch, device_grid_size)
        if sharded is None:
            logger.warning(f"Grid {device_grid_size} has no 8 x {batch} cores; running BFLOAT8_B-DRAM instead")
            return get_model_config(batch, device_grid_size, "BFLOAT8_B-DRAM")
        grid_size, shard_orientation = sharded
        transpose_mm_mcast = shard_orientation == ttnn.ShardOrientation.COL_MAJOR
        new_config_values = {
            "GRID_SIZE": grid_size,
            "SHARD_SIZE": [384, 128],
            "SHARD_ORIENTATION": shard_orientation,
            "QKV_INTERLEAVED": activation_grid_dim,
            "OP4_SOFTMAX_ATTENTION_MASK_DTYPE": ttnn.bfloat16,
            "OP1_FUSED_QKV_MM_INPUT_SHARDED_MEMCFG": BLOCK_SHARDED_MEMCFG,
            "OP1_FUSED_QKV_MM_INPUT_MEMCFG": L1_MEMCFG,
            "OP2_SPLIT_QKV_HEADS_OUTPUT_MEMCFG": HEIGHT_SHARDED_MEMCFG,
            "OP3_PRE_SOFTMAX_BMM_OUTPUT_MEMCFG": HEIGHT_SHARDED_MEMCFG,
            "OP4_SOFTMAX_ATTENTION_MASK_MEMCFG": L1_MEMCFG,
            "OP5_POST_SOFTMAX_BMM_OUTPUT_MEMCFG": HEIGHT_SHARDED_MEMCFG,
            "INPUT_EMBEDDINGS_MEMCFG": L1_MEMCFG,
            "OUTPUT_EMBEDDINGS_MEMCFG": L1_MEMCFG,
            "QA_LINEAR_OUTPUT_MEMCFG": L1_MEMCFG,
            "EMBEDDINGS_LAYERNORM_GAMMA_MEMCFG": DRAM_MEMCFG,
            "EMBEDDINGS_LAYERNORM_BETA_MEMCFG": DRAM_MEMCFG,
            "OP8_LAYERNORM_GAMMA_MEMCFG": DRAM_MEMCFG,
            "OP8_LAYERNORM_BETA_MEMCFG": DRAM_MEMCFG,
            "OP11_LAYERNORM_GAMMA_MEMCFG": DRAM_MEMCFG,
            "OP11_LAYERNORM_BETA_MEMCFG": DRAM_MEMCFG,
            "RESERVE_SPLIT_HEADS_SHAPE": [1, 1, 1, 153 * 1024 // 2],
            "OP1_FUSED_QKV_MM_CONFIG": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=6,
                per_core_M=12,
                per_core_N=12,
                transpose_mcast=transpose_mm_mcast,
                fused_activation=None,
            ),
            "OP3_PRE_SOFTMAX_BMM_CONFIG": ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=6,
                per_core_M=24,
                per_core_N=12,
            ),
            "OP5_POST_SOFTMAX_BMM_CONFIG": ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=12,
                out_subblock_h=4,
                out_subblock_w=2,
                per_core_M=24,
                per_core_N=2,
            ),
            "OP7_SELFOUT_CONFIG": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=4,
                out_subblock_h=2,
                out_subblock_w=4,
                per_core_M=12,
                per_core_N=4,
                transpose_mcast=transpose_mm_mcast,
                fused_activation=None,
            ),
            "OP9_FF1_MM_CONFIG": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=8,
                per_core_M=12,
                per_core_N=16,
                transpose_mcast=transpose_mm_mcast,
                fused_activation=(ttnn.UnaryOpType.GELU, True),
            ),
            "OP10_FF2_MM_CONFIG": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=16,
                out_subblock_h=2,
                out_subblock_w=4,
                per_core_M=12,
                per_core_N=4,
                transpose_mcast=transpose_mm_mcast,
                fused_activation=None,
            ),
            "OP8_LAYERNORM_CONFIG": ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=grid_size,
                subblock_w=4,
                block_h=12,
                block_w=4,
                inplace=True,
            ),
            "OP11_LAYERNORM_CONFIG": ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=grid_size,
                subblock_w=4,
                block_h=12,
                block_w=4,
                inplace=True,
            ),
            "OP4_SOFTMAX_CONFIG": ttnn.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=grid_size,
                subblock_w=6,
                block_h=24,
                block_w=12,
            ),
        }
        model_config.update(new_config_values)

    logger.debug(f"BERT model config: \n{pretty_print_model_config(model_config)}")

    return model_config


def load_reference_model(model_name, **kwargs):
    """The Hugging Face model with its first ``BERT_NUM_LAYERS`` encoder layers, or all of them when
    it is unset. TtBertBatchDram builds as many encoder layers as the model's config holds."""
    value = os.environ.get(NUM_LAYERS_ENV)
    if value:
        num_layers = int(value)
        max_layers = BertConfig.from_pretrained(model_name).num_hidden_layers
        if not 1 <= num_layers <= max_layers:
            raise ValueError(f"{NUM_LAYERS_ENV} must be between 1 and {max_layers}, got {num_layers}")
        logger.info(f"Running {num_layers} of {max_layers} encoder layers ({NUM_LAYERS_ENV}={num_layers})")
        kwargs["num_hidden_layers"] = num_layers
    return BertForQuestionAnswering.from_pretrained(model_name, **kwargs)


# TODO: Generalize TT tensor caching
def get_tt_cache_path(model_version):
    tt_cache_path = Path("/mnt/MLPerf/tt_dnn-models/tt/Bert") / model_version
    if tt_cache_path.exists():
        return tt_cache_path
    else:
        return None


def skip_unsupported_config(device, model_config_str, batch):
    if device.arch() != ttnn.device.Arch.WORMHOLE_B0:
        return
    if model_config_str != "BFLOAT8_B-SHARDED":
        pytest.skip("Only BFLOAT8_B-SHARDED supported for WH B0")
    # The DRAM fallback for a grid without 8 x batch cores has not been verified on Wormhole.
    if grid.sharded_grid(batch, device.compute_with_storage_grid_size()) is None:
        pytest.skip("Unsupported grid size for sharded test")
