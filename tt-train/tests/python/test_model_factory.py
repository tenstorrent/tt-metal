# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for TransformerModelFactory config parsing; no device needed."""

from ttml.common.config import load_config, yaml_deep_update
from ttml.common.model_factory import TransformerModelFactory


def test_model_factory_accepts_model_config():
    # gsm8k_finetune.py and llm_inference.ipynb pass the model config, not the training config that points to it.
    training_config = load_config("training_configs/training_gsm8k_tinyllama.yaml")
    model_config = load_config(training_config["training_config"]["model_config"])

    from_training_config = TransformerModelFactory(training_config).transformer_config
    from_model_config = TransformerModelFactory(model_config).transformer_config

    assert vars(from_model_config) == vars(from_training_config)


def test_model_factory_keeps_overrides_merged_into_model_config():
    # gsm8k_finetune.py merges the dashboard's training_overrides.yaml into the model config, which adds a
    # training_config section without model_config.
    training_config = load_config("training_configs/training_gsm8k_tinyllama.yaml")
    model_config = load_config(training_config["training_config"]["model_config"])
    overrides = {"training_config": {"batch_size": 8}, "transformer_config": {"max_sequence_length": 256}}

    factory = TransformerModelFactory(yaml_deep_update(model_config, overrides))

    assert factory.transformer_config.max_sequence_length == 256


def test_model_factory_prefers_training_config_model_config():
    # A training config is still built from training_config.model_config, even with a top-level transformer_config.
    training_config = load_config("training_configs/training_gsm8k_tinyllama.yaml")
    expected = vars(TransformerModelFactory(training_config).transformer_config)

    training_config["transformer_config"] = {"num_heads": 1}

    assert vars(TransformerModelFactory(training_config).transformer_config) == expected
