import tt_lib as ttl


OP_MEMCFG_KEYS = (
    # MHA
    "OP1_FUSED_QKV_MM_INPUT_MEMCFG",
    "OP1_FUSED_QKV_MM_WEIGHTS_MEMCFG",  # Needs to be DRAM
    "OP1_FUSED_QKV_MM_BIAS_MEMCFG",
    "OP1_FUSED_QKV_MM_OUTPUT_MEMCFG",
    "OP2TO6_CREATE_QKV_HEADS_OUTPUT_MEMCFG",
    "OP7_PRE_SOFTMAX_BMM_OUTPUT_MEMCFG",
    "OP8_SOFTMAX_ATTENTION_MASK_MEMCFG",
    "OP9_POST_SOFTMAX_BMM_OUTPUT_MEMCFG",
    "OP10_CONCAT_ATTENTION_HEADS_OUTPUT_MEMCFG",
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
    # MHA
    "OP1_FUSED_QKV_MM_INPUT_DTYPE",
    "OP1_FUSED_QKV_MM_WEIGHTS_DTYPE",
    "OP1_FUSED_QKV_MM_BIAS_DTYPE",
    "OP8_SOFTMAX_SCALE_DTYPE",  # Used if attention_mask is None
    "OP8_SOFTMAX_ATTENTION_MASK_DTYPE",
    # MHA SELFOUT ATTENTION
    "OP11_SELFOUT_WEIGHTS_DTYPE",
    "OP11_SELFOUT_BIAS_DTYPE",
    # MHA LAYERNORM
    "OP12_LAYERNORM_GAMMA_DTYPE",
    "OP12_LAYERNORM_BETA_DTYPE",
    "OP12_LAYERNORM_OUTPUT_DTYPE",  # Used for ffn sub-graph test, might need in the future with mixed precision
    # FFN
    "OP13_FF1_MM_WEIGHTS_DTYPE",
    "OP13_FF1_MM_BIAS_DTYPE",
    "OP14_FF2_MM_WEIGHTS_DTYPE",
    "OP14_FF2_MM_BIAS_DTYPE",
    # FFN LAYERNORM
    "OP15_LAYERNORM_GAMMA_DTYPE",
    "OP15_LAYERNORM_BETA_DTYPE",
    # After all encoders
    "QA_LINEAR_WEIGHTS_DTYPE",
    "QA_LINEAR_BIAS_DTYPE",
)


def get_model_config(dtype, mem_config):
    # TODO: We can also couple dtype-mem_config as one input.
    # This will give us more options for experimentation with different configs.

    # Set defaults for dtype and mem_config for all ops
    model_config = {"DEFAULT_DTYPE": dtype, "DEFAULT_MEMCFG": mem_config}
    model_config.update(dict(zip(OP_MEMCFG_KEYS, [mem_config] * len(OP_MEMCFG_KEYS))))
    model_config.update(dict(zip(OP_DTYPE_KEYS, [dtype] * len(OP_DTYPE_KEYS))))

    # Override defaults for certain configs
    DRAM_MEMCFG = ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM)
    if (
        dtype == ttl.tensor.DataType.BFLOAT16
        and mem_config.buffer_type == ttl.tensor.BufferType.L1
    ):
        # TODO: Bias, gamma, and beta don't all fit in L1 persistently.
        new_config_values = {
            # MHA
            "OP1_FUSED_QKV_MM_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "OP1_FUSED_QKV_MM_BIAS_MEMCFG": DRAM_MEMCFG,
            "OP7_PRE_SOFTMAX_BMM_OUTPUT_MEMCFG": DRAM_MEMCFG,
            # MHA SELFOUT ATTENTION
            "OP11_SELFOUT_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "OP11_SELFOUT_BIAS_MEMCFG": DRAM_MEMCFG,
            # MHA LAYERNORM
            "OP12_LAYERNORM_GAMMA_MEMCFG": DRAM_MEMCFG,
            "OP12_LAYERNORM_BETA_MEMCFG": DRAM_MEMCFG,
            # FFN
            "OP13_FF1_MM_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "OP13_FF1_MM_BIAS_MEMCFG": DRAM_MEMCFG,
            "OP13_FF1_MM_OUTPUT_MEMCFG": DRAM_MEMCFG,
            "OP14_FF2_MM_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "OP14_FF2_MM_BIAS_MEMCFG": DRAM_MEMCFG,
            # FFN LAYERNORM
            "OP15_LAYERNORM_GAMMA_MEMCFG": DRAM_MEMCFG,
            "OP15_LAYERNORM_BETA_MEMCFG": DRAM_MEMCFG,
            # After all encoders
            "QA_LINEAR_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "QA_LINEAR_BIAS_MEMCFG": DRAM_MEMCFG,
        }
        model_config.update(new_config_values)
    elif (
        dtype == ttl.tensor.DataType.BFLOAT8_B
        and mem_config.buffer_type == ttl.tensor.BufferType.L1
    ):
        # TODO: Bias, gamma, and beta don't all fit in L1 persistently.
        new_config_values = {
            # MHA
            "OP1_FUSED_QKV_MM_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "OP1_FUSED_QKV_MM_BIAS_MEMCFG": DRAM_MEMCFG,
            # MHA SELFOUT ATTENTION
            "OP11_SELFOUT_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "OP11_SELFOUT_BIAS_MEMCFG": DRAM_MEMCFG,
            # MHA LAYERNORM
            "OP12_LAYERNORM_GAMMA_MEMCFG": DRAM_MEMCFG,
            "OP12_LAYERNORM_BETA_MEMCFG": DRAM_MEMCFG,
            # FFN
            "OP13_FF1_MM_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "OP13_FF1_MM_BIAS_MEMCFG": DRAM_MEMCFG,
            "OP14_FF2_MM_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "OP14_FF2_MM_BIAS_MEMCFG": DRAM_MEMCFG,
            # FFN LAYERNORM
            "OP15_LAYERNORM_GAMMA_MEMCFG": DRAM_MEMCFG,
            "OP15_LAYERNORM_BETA_MEMCFG": DRAM_MEMCFG,
            # After all encoders
            "QA_LINEAR_WEIGHTS_MEMCFG": DRAM_MEMCFG,
            "QA_LINEAR_BIAS_MEMCFG": DRAM_MEMCFG,
        }
        model_config.update(new_config_values)

    return model_config
