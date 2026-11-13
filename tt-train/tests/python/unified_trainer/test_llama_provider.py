# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The Llama provider's config checks and LoRA targets, and the provider registry. Device-free: nothing here
builds a model."""

import json
import re

import pytest

from ttml.unified_trainer import SFTTrainerConfig, UnsupportedTrainingConfig
from ttml.unified_trainer.configs import LoraConfig
from ttml.unified_trainer.providers import PROVIDERS, LlamaProvider, get_provider
from ttml.unified_trainer.providers.base import AdapterState
from ttml.unified_trainer.providers.llama import LORA_LAYERS, _same_model, hf_target_modules, lora_layers

HF_CONFIG = {
    "model_type": "llama",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 100,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
}
ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
EVERY_PROJECTION = ATTENTION + ["gate_proj", "up_proj", "down_proj"]


def config(targets=ATTENTION, model_name="meta-llama/Llama-3.2-1B", lora=None, **overrides):
    """A LoRA run on ``targets``, or with ``lora`` as given; ``targets=None`` trains every weight."""
    if lora is None and targets is not None:
        lora = {"r": 4, "alpha": 8, "target_modules": list(targets)}
    fields = {
        "dataset_id": "sst2",
        "model": {"name": model_name, "lora": lora},
        "optimizer": {"lr": 1e-4},
        "batch_size": 2,
        "max_length": 64,
    }
    return SFTTrainerConfig.model_validate({**fields, **overrides})


def test_every_projection_maps_to_one_ttml_layer():
    assert lora_layers(EVERY_PROJECTION) == list(LORA_LAYERS)
    assert lora_layers(["o_proj"]) == [(("o_proj",), r"\.attention\.out_linear$")]


@pytest.mark.parametrize(
    "layer, projections",
    [
        ("blocks.3.attention.qkv_linear", ["q_proj", "k_proj", "v_proj"]),
        ("blocks.0.attention.out_linear", ["o_proj"]),
        ("blocks.12.mlp.w_gate_up", ["gate_proj", "up_proj"]),
        ("blocks.1.mlp.w2", ["down_proj"]),
    ],
)
def test_the_patterns_match_the_ttml_layer_names(layer, projections):
    matched = [names for names, pattern in LORA_LAYERS if re.search(pattern, layer)]
    assert matched == [tuple(projections)]


@pytest.mark.parametrize("targets", [["q_proj", "v_proj"], ["k_proj"], ["gate_proj"], ["up_proj", "down_proj"]])
def test_a_partly_targeted_fused_layer_is_rejected(targets, expect_error):
    with expect_error(UnsupportedTrainingConfig, "model.lora.target_modules: .* one fused layer"):
        lora_layers(targets)


def test_unknown_targets_are_rejected(expect_error):
    with expect_error(UnsupportedTrainingConfig, r"\['c_attn'\] are not Llama projections"):
        lora_layers(["c_attn", "q_proj", "k_proj", "v_proj"])


def test_hf_target_modules_are_sorted():
    lora = LoraConfig(r=4, alpha=8, target_modules=["v_proj", "o_proj", "q_proj", "k_proj"])
    assert hf_target_modules(lora) == ["k_proj", "o_proj", "q_proj", "v_proj"]


@pytest.mark.parametrize("targets", [ATTENTION, EVERY_PROJECTION, None], ids=["attention", "every projection", "full"])
def test_a_trainable_config_passes(targets):
    LlamaProvider().check(config(targets), HF_CONFIG)


def test_gradient_checkpointing_passes():
    LlamaProvider().check(config(gradient_checkpointing=True), HF_CONFIG)


@pytest.mark.parametrize(
    "overrides, hf_overrides, error",
    [
        ({"dtype": "float32"}, {}, "dtype: ttml Llama weights"),
        ({"dtype": "torch.bfloat16"}, {}, "dtype: ttml Llama weights"),
        ({"max_length": 48}, {}, "max_length: must be a multiple of 32"),
        ({"max_length": 256}, {}, "max_length: exceeds the model's max_position_embeddings"),
        (
            {"lora": {"r": 4, "alpha": 8, "target_modules": ATTENTION, "task_type": "SEQ_CLS"}},
            {},
            "model.lora.task_type",
        ),
        (
            {"lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj"]}},
            {},
            "model.lora.target_modules: .* one fused layer",
        ),
        ({}, {"hidden_act": "gelu"}, "model.name: .* hidden_act 'gelu'"),
        ({}, {"quantization_config": {"quant_method": "gptq"}}, "model.name: .* quantized"),
        ({}, {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}, "model.name: .* rope type 'yarn'"),
    ],
)
def test_an_untrainable_config_is_rejected(overrides, hf_overrides, error, expect_error):
    with expect_error(UnsupportedTrainingConfig, error):
        LlamaProvider().check(config(**overrides), {**HF_CONFIG, **hf_overrides})


def test_get_provider_reads_the_model_type(tmp_path, expect_error):
    (tmp_path / "config.json").write_text(json.dumps(HF_CONFIG))
    provider, hf_config = get_provider(config(model_name=str(tmp_path)))
    assert provider is PROVIDERS["llama"] and hf_config == HF_CONFIG

    (tmp_path / "config.json").write_text(json.dumps({**HF_CONFIG, "model_type": "qwen3"}))
    with expect_error(UnsupportedTrainingConfig, "model.name: no provider for model_type 'qwen3'"):
        get_provider(config(model_name=str(tmp_path)))


@pytest.mark.parametrize(
    "adapter_base, model_name, same",
    [
        ("meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B", True),
        ("/models/Llama-3.2-1B/", "meta-llama/Llama-3.2-1B", True),
        (None, "meta-llama/Llama-3.2-1B", True),
        ("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-1B", False),
    ],
)
def test_same_model(adapter_base, model_name, same):
    assert _same_model(adapter_base, model_name) is same


def adapter_config(**overrides):
    fields = {
        "peft_type": "LORA",
        "r": 4,
        "lora_alpha": 8,
        "target_modules": ["v_proj", "q_proj", "o_proj", "k_proj"],
        "base_model_name_or_path": "meta-llama/Llama-3.2-1B",
    }
    return {**fields, **overrides}


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"r": 8}, "r: adapter 8, run 4"),
        ({"lora_alpha": 16}, "lora_alpha: adapter 16, run 8"),
        ({"use_rslora": True}, "use_rslora: adapter True, run False"),
        ({"target_modules": ["q_proj", "v_proj"]}, "target_modules"),
        ({"peft_type": "IA3"}, "peft_type"),
        ({"base_model_name_or_path": "meta-llama/Llama-3.1-8B"}, "the adapter is for 'meta-llama/Llama-3.1-8B'"),
    ],
)
def test_an_adapter_for_another_lora_is_rejected_before_it_touches_the_model(overrides, error, expect_error):
    state = AdapterState(tensors={}, config=adapter_config(**overrides))
    with expect_error(UnsupportedTrainingConfig, re.escape(error)):
        LlamaProvider().import_adapter(None, state, config())
