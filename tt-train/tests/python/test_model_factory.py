# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for TransformerModelFactory config parsing; no device needed."""

from ttml.common.config import load_config
from ttml.common.model_factory import TransformerModelFactory


def test_model_factory_accepts_model_config():
    # gsm8k_finetune.py and llm_inference.ipynb pass the model config, not the training config that points to it.
    training_config = load_config("training_configs/training_gsm8k_tinyllama.yaml")
    model_config = load_config(training_config["training_config"]["model_config"])

    from_training_config = TransformerModelFactory(training_config).transformer_config
    from_model_config = TransformerModelFactory(model_config).transformer_config

    assert vars(from_model_config) == vars(from_training_config)
