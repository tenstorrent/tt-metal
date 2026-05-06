# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Main entry point for 3-tier hierarchical parallel transformer training.

This script orchestrates the training of transformer models using a 3-tier architecture:
- Workers: Compute forward/backward passes
- Aggregator: Averages gradients from workers
- Optimizer: Applies optimizer updates
"""
import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')

import click
import ttml
from ttml.common.config import (
    DeviceConfig,
    MultiHostConfig,
    TrainingConfig,
    load_config,
)
from ttml.common.model_factory import TransformerModelFactory
from ttml.common.utils import create_optimizer, initialize_device, set_seed

from ttml.common.data import prepare_data
from trainer import worker, aggregator, optimizer, aggregator_optimizer


@click.command()
@click.option(
    "-c",
    "--config",
    type=str,
    default="training_shakespeare_tinyllama_tensor_parallel_3tier_fabric.yaml",
)
@click.option(
    "--worker-type",
    type=click.Choice(
        ["worker", "aggregator", "optimizer", "aggregator_optimizer"],
        case_sensitive=False,
    ),
    default=None,
    help="Type of worker (auto-detected if not specified)",
)
def main(config: str, worker_type: str):
    """Main training function for hierarchical parallel training.

    Supports two architectures:

    2-tier (world_size == num_workers + 1):
    - Workers (ranks 0 to num_workers-1): Compute forward/backward passes
    - AggregatorOptimizer (rank num_workers): Aggregates gradients and applies optimizer

    3-tier (world_size == num_workers + 2):
    - Workers (ranks 0 to num_workers-1): Compute forward/backward passes
    - Aggregator (rank num_workers): Aggregates gradients from workers
    - Optimizer (rank num_workers+1): Applies optimizer updates

    Args:
        config: Path to YAML configuration file (relative to configs directory)
        worker_type: Type of worker to run (auto-detected from rank if not specified)
    """
    # Load configuration
    yaml_config = load_config(config)

    # Initialize distributed context
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)
    distributed_ctx = autograd_ctx.get_distributed_context()

    rank = distributed_ctx.rank()
    world_size = distributed_ctx.size()

    multihost_config = MultiHostConfig(yaml_config)
    num_workers = multihost_config.num_workers

    # Determine architecture based on world_size and num_workers
    # 2-tier: world_size == num_workers + 1 (workers + aggregator_optimizer)
    # 3-tier: world_size == num_workers + 2 (workers + aggregator + optimizer)
    is_two_tier = world_size == num_workers + 1

    # Auto-detect worker type based on rank if not specified
    if worker_type is None:
        if rank < num_workers:
            worker_type = "worker"
        elif rank == num_workers:
            if is_two_tier:
                worker_type = "aggregator_optimizer"
            else:
                worker_type = "aggregator"
        else:
            worker_type = "optimizer"

    mode = "2-tier" if is_two_tier else "3-tier"
    print(f"Rank {rank}/{world_size}: Running as {worker_type} ({mode} mode)")

    # Initialize socket manager
    socket_type = (
        ttml.core.distributed.SocketType.FABRIC
        if multihost_config.socket_type == "fabric"
        else ttml.core.distributed.SocketType.MPI
    )
    autograd_ctx.initialize_socket_manager(socket_type)

    # Set seed based on rank
    set_seed(yaml_config["training_config"].get("seed", 42) + rank)

    # Prepare data
    train_ids, val_ids, vocab_size, decode = prepare_data(yaml_config)

    # Update config with vocab size
    training_config = yaml_config.setdefault("training_config", {})
    transformer_config = training_config.setdefault("transformer_config", {})
    transformer_config["vocab_size"] = int(vocab_size)

    # Initialize device
    initialize_device(yaml_config)

    # Create model and config
    model_factory = TransformerModelFactory(yaml_config)
    model = model_factory.create_model()
    print(f"Rank {rank}: Model created")

    training_cfg = TrainingConfig(yaml_config)
    device_config = DeviceConfig(yaml_config)

    # Execute appropriate worker function
    if worker_type == "worker":
        # Training worker - computes forward/backward and uses RemoteOptimizer
        train_losses, val_losses = worker(
            training_cfg,
            model,
            train_ids,
            val_ids,
            device_config.enable_ddp,
            device_config.enable_tp,
            num_workers,
        )
        print(f"[Worker {rank}] Completed with {len(train_losses)} loss values")
    elif worker_type == "aggregator":
        # Aggregator - averages gradients from workers and broadcasts weights (3-tier only)
        aggregator(model, training_cfg, device_config.enable_ddp)
    elif worker_type == "optimizer":
        # Optimizer - applies optimizer updates (3-tier only)
        optimizer_instance = create_optimizer(model, yaml_config)
        optimizer(model, training_cfg, optimizer_instance)
    elif worker_type == "aggregator_optimizer":
        # Combined aggregator and optimizer for 2-tier architecture
        optimizer_instance = create_optimizer(model, yaml_config)
        aggregator_optimizer(
            model, training_cfg, optimizer_instance, device_config.enable_ddp
        )

    # Cleanup
    distributed_ctx.barrier()
    autograd_ctx.close_device()
    print(f"Rank {rank}: Finished")


if __name__ == "__main__":
    main()
