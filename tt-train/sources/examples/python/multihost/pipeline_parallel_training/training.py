# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Main entry point for transformer training.

This script orchestrates the training of transformer models (GPT-2, Llama)
using configurations specified in YAML files.
"""

import sys
import ttml
from ttml.common.config import (
    load_config,
    DeviceConfig,
    TrainingConfig,
    MultiHostConfig,
)
from ttml.common.utils import set_seed, initialize_device, create_optimizer
from ttml.common.model_factory import TransformerModelFactory
import click

from ttml.common.data import prepare_data
from trainer import train


@click.command()
@click.option(
    "-c", "--config", type=str, default="training_shakespeare_llama7b_pp_fabric.yaml"
)
def main(config: str):
    """Main training function.

    Args:
        config: Path to YAML configuration file (relative to configs directory)
    """
    # Load configuration and set seed
    yaml_config = load_config(config)

    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)
    distributed_ctx = autograd_ctx.get_distributed_context()

    import socket

    print(
        f"rank {distributed_ctx.rank()} is assigned to hostname {socket.gethostname()}"
    )

    # Initialize socket manager based on multihost configuration
    multihost_config = MultiHostConfig(
        load_config(yaml_config["training_config"]["multihost_config"])
    )
    socket_type = (
        ttml.core.distributed.SocketType.FABRIC
        if multihost_config.socket_type == "fabric"
        else ttml.core.distributed.SocketType.MPI
    )
    autograd_ctx.initialize_socket_manager(socket_type)
    socket_manager = autograd_ctx.get_socket_manager()

    rank = distributed_ctx.rank()
    world_size = distributed_ctx.size()

    assert multihost_config.enabled, "Multihost is not enabled"
    assert (
        world_size == multihost_config.num_workers
    ), f"World size ({world_size}) must equal multihost_config.num_workers ({multihost_config.num_workers})"
    assert (
        world_size > 1
    ), f"World size must be greater than 1, world size: {world_size}"

    # adjust seed based on worker rank to make sure that each worker has a different seed
    set_seed(yaml_config["training_config"].get("seed", 42) + rank)

    # Prepare data
    # train_ids, val_ids, vocab_size, decode = prepare_data(yaml_config)

    # Use vocab_size from data instead of config
    # training_config = yaml_config.setdefault("training_config", {})
    # transformer_config = training_config.setdefault("transformer_config", {})
    # transformer_config["vocab_size"] = int(vocab_size)

    # Initialize device mesh
    initialize_device(yaml_config)

    # Warm up with round robin communication
    import numpy as np
    import ttnn

    assert (
        world_size > 1
    ), f"World size must be greater than 1, world size: {world_size}"

    shard_dim = 3

    device = ttml.autograd.AutoContext.get_instance().get_device()
    if rank < world_size - 1:
        print(f"Rank {rank} is sending data")
        tensor_data = np.array([i for i in range(32)], dtype=np.float32)
        tensor_data = tensor_data.reshape(1, 1, 1, 32)
        print(f"rank {rank} sending {[i for i in tensor_data.tolist()]}")
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(
            device, shard_dim, cluster_axis=0
        )
        tensor = ttml.autograd.Tensor.from_numpy(
            tensor_data,
            ttnn.Layout.TILE,
            ttnn.DataType.BFLOAT16,
            mapper,
        )

        dest_rank = rank + 1
        socket_manager.send(tensor, distributed_ctx, dest_rank)

    if rank > 0:
        print(f"Rank {rank} is receiving data")
        tensor_data = np.zeros((1, 1, 1, 32), dtype=np.float32)
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(
            device, shard_dim, cluster_axis=0
        )
        tensor = ttml.autograd.Tensor.from_numpy(
            tensor_data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper
        )
        src_rank = rank - 1
        tensor = socket_manager.recv(tensor, distributed_ctx, src_rank)
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(
            device, shard_dim
        )
        tensor_data = tensor.to_numpy(composer=composer).flatten()
        print(f"rank {rank} received {[i for i in tensor_data.tolist()]}")
        # tensor_data = tensor.to_numpy().flatten()
        assert tensor_data.tolist()[:32] == [
            i for i in range(32)
        ], f"Rank {rank} received data: {tensor_data} does not match expected data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]"
        print(f"Rank {rank} received data: {tensor_data}")

    """
    # Create model, optimizer, and training configuration
    model_factory = TransformerModelFactory(yaml_config)
    model = model_factory.create_model()
    optimizer = create_optimizer(model, yaml_config)

    training_cfg = TrainingConfig(yaml_config)
    device_config = DeviceConfig(yaml_config)

    # Execute training
    train_losses, val_losses = train(
        training_cfg,
        model_factory.transformer_config.max_sequence_length,
        model,
        optimizer,
        train_ids,
        val_ids,
        device_config.enable_ddp,
        device_config.enable_tp,
    )
    """

    # Cleanup
    ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
