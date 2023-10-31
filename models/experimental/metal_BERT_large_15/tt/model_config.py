# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib
from loguru import logger


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
    "OP2TO6_SPLIT_QKV_HEADS_OUTPUT_MEMCFG",
    "OP7_PRE_SOFTMAX_BMM_OUTPUT_MEMCFG",
    "OP8_SOFTMAX_ATTENTION_MASK_MEMCFG",
    "OP9_POST_SOFTMAX_BMM_OUTPUT_MEMCFG",
    "OP10_CONCATENATE_ATTENTION_HEADS_OUTPUT_MEMCFG",
    # MHA SELFOUT ATTENTION
    "OP11_SELFOUT_WEIGHTS_MEMCFG",
    "OP11_SELFOUT_BIAS_MEMCFG",
    "OP11_SELFOUT_OUTPUT_MEMCFG",
    # MHA LAYERNORM
    "OP12_LAYERNORM_GAMMA_MEMCFG",
    "OP12_LAYERNORM_BETA_MEMCFG",
    "OP12_LAYERNORM_OUTPUT_MEMCFG",
    # FFN
    "OP13_FF1_MM_WEIGHTS_MEMCFG",
    "OP13_FF1_MM_BIAS_MEMCFG",
    "OP13_FF1_MM_OUTPUT_MEMCFG",
    "OP14_FF2_MM_WEIGHTS_MEMCFG",
    "OP14_FF2_MM_BIAS_MEMCFG",
    "OP14_FF2_MM_OUTPUT_MEMCFG",
    # FFN LAYERNORM
    "OP15_LAYERNORM_GAMMA_MEMCFG",
    "OP15_LAYERNORM_BETA_MEMCFG",
    "OP15_LAYERNORM_OUTPUT_MEMCFG",
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
    "OP7_PRE_SOFTMAX_BMM_OUTPUT_DTYPE",
    "OP8_SOFTMAX_ATTENTION_MASK_DTYPE",
    "OP9_POST_SOFTMAX_BMM_OUTPUT_DTYPE",
    # MHA SELFOUT ATTENTION
    "OP11_SELFOUT_WEIGHTS_DTYPE",
    "OP11_SELFOUT_BIAS_DTYPE",
    "OP11_SELFOUT_OUTPUT_DTYPE",
    # MHA LAYERNORM
    "OP12_LAYERNORM_GAMMA_DTYPE",
    "OP12_LAYERNORM_BETA_DTYPE",
    "OP12_LAYERNORM_OUTPUT_DTYPE",  # Used for ffn sub-graph test, might need in the future with mixed precision
    # FFN
    "OP13_FF1_MM_WEIGHTS_DTYPE",
    "OP13_FF1_MM_BIAS_DTYPE",
    "OP13_FF1_MM_OUTPUT_DTYPE",
    "OP14_FF2_MM_WEIGHTS_DTYPE",
    "OP14_FF2_MM_BIAS_DTYPE",
    "OP14_FF2_MM_OUTPUT_DTYPE",
    # FFN LAYERNORM
    "OP15_LAYERNORM_GAMMA_DTYPE",
    "OP15_LAYERNORM_BETA_DTYPE",
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
)


def pretty_print_model_config(model_config):
    print_str = []
    for key, val in model_config.items():
        if key.endswith("MEMCFG"):
            print_str.append(f"{key}: {val.buffer_storage}")

        elif key.endswith("DTYPE") or key.endswith("BOOL"):
            print_str.append(f"{key}: {val}")

        else:
            raise NotImplementedError("Unknown key: {key}!")

    return "\n".join(print_str)


def get_model_config(model_config_str):
    assert model_config_str in ACCEPTABLE_MODEL_CONFIG_STRS
    DRAM_MEMCFG = tt_lib.tensor.MemoryConfig(
        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferStorage.DRAM
    )
    L1_MEMCFG = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferStorage.L1)

    # Set default dtype and mem_config based on model_config_str
    if model_config_str in (
        "BFLOAT8_B-DRAM",
        "BFLOAT16-DRAM",
        "BFLOAT8_B-L1",
        "BFLOAT16-L1",
    ):
        dtype_str, mem_config_str = model_config_str.split("-")
        mem_config = DRAM_MEMCFG if mem_config_str == "DRAM" else L1_MEMCFG
        dtype = tt_lib.tensor.DataType.BFLOAT16 if dtype_str == "BFLOAT16" else tt_lib.tensor.DataType.BFLOAT8_B

    elif model_config_str in ("MIXED_PRECISION_BATCH9", "MIXED_PRECISION_BATCH8"):
        dtype = tt_lib.tensor.DataType.BFLOAT8_B
        mem_config = L1_MEMCFG

    else:
        raise NotImplementedError(f"Model config {model_config_str} is not supported!")

    # Set defaults for dtype and mem_config for all ops
    model_config = {
        "DEFAULT_DTYPE": dtype,
        "DEFAULT_MEMCFG": mem_config,
        "MOVE_ENCODER_OUTPUT_BOOL": False,
    }  # DEFAULT_MEMCFG also used to determine banking for tt_lib.device.InitializeDevice
    model_config.update(dict(zip(OP_MEMCFG_KEYS, [mem_config] * len(OP_MEMCFG_KEYS))))
    model_config.update(dict(zip(OP_DTYPE_KEYS, [dtype] * len(OP_DTYPE_KEYS))))

    # Layernorm Gamma Beta must always be BFLOAT16
    model_config.update(
        {
            "INPUT_EMBEDDINGS_WEIGHTS_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            "EMBEDDINGS_LAYERNORM_GAMMA_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            "EMBEDDINGS_LAYERNORM_BETA_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            "OP12_LAYERNORM_GAMMA_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            "OP12_LAYERNORM_BETA_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            "OP15_LAYERNORM_GAMMA_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            "OP15_LAYERNORM_BETA_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
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
            "OP11_SELFOUT_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "OP11_SELFOUT_BIAS_MEMCFG": DRAM_MEMCFG,
            # FFN
            "OP13_FF1_MM_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "OP13_FF1_MM_BIAS_MEMCFG": DRAM_MEMCFG,
            "OP14_FF2_MM_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "OP14_FF2_MM_BIAS_MEMCFG": DRAM_MEMCFG,
            # After all encoders
            "QA_LINEAR_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "QA_LINEAR_BIAS_MEMCFG": DRAM_MEMCFG,
        }
    )

    # Override defaults for certain configs
    if model_config_str == "BFLOAT16-L1":
        new_config_values = {
            # MHA
            "OP7_PRE_SOFTMAX_BMM_OUTPUT_MEMCFG": DRAM_MEMCFG,
            # FFN
            "OP13_FF1_MM_OUTPUT_MEMCFG": DRAM_MEMCFG,
        }
        model_config.update(new_config_values)

    elif model_config_str == "BFLOAT8_B-L1":
        new_config_values = {}
        model_config.update(new_config_values)

    elif model_config_str == "MIXED_PRECISION_BATCH9":
        new_config_values = {
            # MHA
            "OP7_PRE_SOFTMAX_BMM_OUTPUT_MEMCFG": DRAM_MEMCFG,
            # MHA
            "OP1_FUSED_QKV_MM_INPUT_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            "OP7_PRE_SOFTMAX_BMM_OUTPUT_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            "OP8_SOFTMAX_ATTENTION_MASK_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            # MHA SELFOUT ATTENTION
            "OP11_SELFOUT_OUTPUT_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            # MHA LAYERNORM
            "OP12_LAYERNORM_OUTPUT_DTYPE": tt_lib.tensor.DataType.BFLOAT16,  # Used for ffn sub-graph test, might need in the future with mixed precision
            # FFN
            "OP14_FF2_MM_OUTPUT_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            # After all encoders
            "QA_LINEAR_WEIGHTS_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            "QA_LINEAR_BIAS_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
        }
        model_config.update(new_config_values)

    elif model_config_str == "MIXED_PRECISION_BATCH8":
        new_config_values = {
            "MOVE_ENCODER_OUTPUT_BOOL": True,
            # MHA
            "OP1_FUSED_QKV_MM_INPUT_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            "OP7_PRE_SOFTMAX_BMM_OUTPUT_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            "OP8_SOFTMAX_ATTENTION_MASK_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            # MHA SELFOUT ATTENTION
            "OP11_SELFOUT_OUTPUT_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            # MHA LAYERNORM
            "OP12_LAYERNORM_OUTPUT_DTYPE": tt_lib.tensor.DataType.BFLOAT16,  # Used for ffn sub-graph test, might need in the future with mixed precision
            # FFN
            "OP14_FF2_MM_OUTPUT_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            # After all encoders
            "QA_LINEAR_WEIGHTS_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
            "QA_LINEAR_BIAS_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
        }
        model_config.update(new_config_values)

    logger.debug(f"BERT model config: \n{pretty_print_model_config(model_config)}")

    return model_config
