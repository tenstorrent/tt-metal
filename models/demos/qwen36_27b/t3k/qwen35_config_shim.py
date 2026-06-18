# SPDX-License-Identifier: Apache-2.0
"""Transformers config shim for Qwen3.6-27B (arch Qwen3_5ForConditionalGeneration,
model_type qwen3_5 / text backbone qwen3_5_text), which transformers 4.57 does not
recognize. vLLM only needs the config to parse (for engine/KV-cache setup) — the actual
compute is our TT model. So these are config-only classes that just hold the fields;
no modeling code. Import this module before vLLM parses the model config.

vLLM reads text dims via get_hf_text_config (descends into .text_config): num_hidden_layers,
hidden_size, num_attention_heads, num_key_value_heads, head_dim, vocab_size,
max_position_embeddings. Our generator_vllm._hf_config_to_cfg reads the same .text_config.
"""
from transformers import AutoConfig, PretrainedConfig


class Qwen3_5TextConfig(PretrainedConfig):
    model_type = "qwen3_5_text"

    def __init__(self, **kwargs):
        # PretrainedConfig stores any unconsumed kwargs as attributes, so all the
        # text dims from config.json's text_config become attributes.
        super().__init__(**kwargs)


class Qwen3_5Config(PretrainedConfig):
    model_type = "qwen3_5"
    # Tell transformers text_config / vision_config are sub-configs.
    sub_configs = {"text_config": Qwen3_5TextConfig}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(text_config, dict):
            text_config = Qwen3_5TextConfig(**text_config)
        self.text_config = text_config
        # keep vision_config as a plain dict (text-only serving ignores it)
        self.vision_config = vision_config


def register():
    """Idempotently register the qwen3_5 config classes with transformers AutoConfig."""
    for mt, cls in (("qwen3_5_text", Qwen3_5TextConfig), ("qwen3_5", Qwen3_5Config)):
        try:
            AutoConfig.register(mt, cls)
        except ValueError:
            pass  # already registered


register()
