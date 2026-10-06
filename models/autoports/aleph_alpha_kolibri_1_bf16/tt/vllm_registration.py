# SPDX-License-Identifier: Apache-2.0
"""Explicit vLLM general plugin for the pinned Kolibri configuration."""

from transformers import AutoConfig, PretrainedConfig


class Kolibri1Config(PretrainedConfig):
    model_type = "kolibri1"


def register():
    AutoConfig.register("kolibri1", Kolibri1Config, exist_ok=True)
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "TTKolibri1ForCausalLM",
        "models.autoports.aleph_alpha_kolibri_1_bf16.tt.generator_vllm:KolibriForCausalLM",
    )

    ModelRegistry.register_model(
        "Kolibri1ForCausalLM", "models.autoports.aleph_alpha_kolibri_1_bf16.tt.generator_vllm:KolibriForCausalLM"
    )
