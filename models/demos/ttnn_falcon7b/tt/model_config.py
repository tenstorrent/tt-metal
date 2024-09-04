# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from pathlib import Path

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

ACCEPTABLE_MODEL_CONFIG_STRS = ("BFLOAT16-DRAM", "BFLOAT16-L1")


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


def get_model_config(model_config_str):
    assert model_config_str in ACCEPTABLE_MODEL_CONFIG_STRS
    DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
    L1_MEMCFG = ttnn.L1_MEMORY_CONFIG
    BFP8_DTYPE = ttnn.bfloat8_b

    # Set default dtype and mem_config based on model_config_str
    if model_config_str in ("BFLOAT16-DRAM", "BFLOAT16-L1"):
        dtype_str, mem_config_str = model_config_str.split("-")
        # TODO: Set default memcfg for BFLOAT16-L1 to L1
        # mem_config = DRAM_MEMCFG if mem_config_str == "DRAM" else L1_MEMCFG
        mem_config = DRAM_MEMCFG
        dtype = ttnn.bfloat16 if dtype_str == "BFLOAT16" else ttnn.bfloat8_b
    else:
        raise NotImplementedError(f"Model config {model_config_str} is not supported!")

    # Set defaults for dtype and mem_config for all ops
    model_config = {
        "DEFAULT_DTYPE": dtype,
        "DEFAULT_MEMCFG": mem_config,
        "MOVE_DECODER_OUTPUT_BOOL": False,
    }  # DEFAULT_MEMCFG also used to determine banking for ttnn.experimental.device.InitializeDevice
    model_config.update({f"{key}_MEMCFG": mem_config for key in OP_KEYS if key not in NO_MEMCFG})
    model_config.update({f"{key}_DTYPE": dtype for key in OP_KEYS if key not in NO_DTYPE})

    # Matmul Weights must always be BFP8_B
    # Override defaults for certain configs
    for key in model_config.keys():
        if "MM_WEIGHTS_DTYPE" in key:
            model_config[key] = BFP8_DTYPE

    if model_config_str in ("BFLOAT16-L1",):
        model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["K_CACHE_SLICE_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["V_CACHE_SLICE_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["K_TRANSPOSED_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["PRE_SOFTMAX_SCALE_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["PRE_SOFTMAX_MASK_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"] = L1_MEMCFG

    # uncomment if need to see all the configs
    # logger.debug(f"Falcon model config: \n{pretty_print_model_config(model_config)}")

    return model_config


# TODO: Generalize TT tensor caching
def get_tt_cache_path(model_version):
    tt_cache_path = Path("/mnt/MLPerf/ttnn/models/demos/falcon7b/datasets") / model_version
    if tt_cache_path.exists():
        return tt_cache_path
    else:
        Path(f"models/demos/ttnn_falcon7b/datasets/{model_version}").mkdir(parents=True, exist_ok=True)
        return Path(f"models/demos/ttnn_falcon7b/datasets/{model_version}")


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
