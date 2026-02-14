# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Main entry point for transformer training.

This script orchestrates the training of transformer models (GPT-2, Llama)
using configurations specified in YAML files.
"""
import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')

import ttml
from ttml.common.config import get_config, DeviceConfig, TrainingConfig
from ttml.common.utils import set_seed, initialize_device, create_optimizer
from ttml.common.model_factory import TransformerModelFactory
import click

from ttml.common.data import prepare_data
from trainer import train


@click.command()
@click.option("-c", "--config", type=str, default="training_shakespeare_nanogpt.yaml")
def main(config: str):
    """Main training function.

    Args:
        config: Path to YAML configuration file (relative to configs directory)
    """
    # Load configuration and set seed
    yaml_config = get_config(config)
    set_seed(yaml_config["training_config"].get("seed", 42))

    # Prepare data
    train_ids, val_ids, vocab_size, decode = prepare_data(yaml_config)

    # Use vocab_size from data instead of config
    training_config = yaml_config.setdefault("training_config", {})
    transformer_config = training_config.setdefault("transformer_config", {})
    transformer_config["vocab_size"] = int(vocab_size)

    # Initialize device mesh
    initialize_device(yaml_config)

    # Create model, optimizer, and training configuration
    model_factory = TransformerModelFactory(yaml_config)
    model = model_factory.create_model()
    optimizer = create_optimizer(model, yaml_config)

    training_cfg = TrainingConfig(yaml_config)
    device_config = DeviceConfig(yaml_config)

    # Execute training
    train_losses, val_losses = train(
        training_cfg,
        model,
        optimizer,
        train_ids,
        val_ids,
        device_config.enable_ddp,
        device_config.enable_tp,
    )

    # Cleanup
    ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
