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

import time
import torch
import ttnn
from loguru import logger


def run_ttnn_fabric_verification() -> None:
    """
    Optional TTNN fabric verification routine.

    This helper contains the verification logic for send/recv between ranks. It is not invoked by default.
    """

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    # The following must match the shape you define in the MGD!
    num_rows = 4
    num_cols = 8
    mesh_shape = ttnn.MeshShape(num_rows, num_cols)

    device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    if not ttnn.distributed_context_is_initialized():
        raise RuntimeError(f"Distributed context failed to initialize!")

    world_size = int(ttnn.distributed_context_get_size())
    rank = int(ttnn.distributed_context_get_rank())

    logger.info(f"Rank {rank}/{world_size} initialized successfully")
    torch.manual_seed(42)

    socket_connections = [
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
        )
    ]
    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 4096)

    test_shape = (1, 1, 32, 32)
    torch_tensor = torch.randn(test_shape, dtype=torch.bfloat16)

    connections_to_test = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 3),
        (3, 2),
        (2, 1),
        (1, 0),
    ]
    for sender, receiver in connections_to_test:
        ttnn.distributed_context_barrier()
        logger.info(f"Rank {rank} passed barrier, starting transfer test")

        if rank == sender:
            print(f"Rank {rank} is sending data")
            tt_tensor = ttnn.from_torch(
                torch_tensor,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )

            logger.info(f"Rank {rank} trying to send to rank {receiver}")
            socket_config = ttnn.SocketConfig(
                socket_connections,
                socket_mem_config,
                sender_rank=sender,
                receiver_rank=receiver,
            )
            send_socket = ttnn.MeshSocket(device, socket_config)
            logger.info(f"Rank {rank} sending tensor of shape {test_shape}")

            t_start = time.time()
            ttnn.experimental.send_async(tt_tensor, send_socket)
            ttnn.synchronize_device(device)
            t_end = time.time()

            logger.info(
                f"Rank {rank} send completed in {(t_end - t_start)*1000:.2f} ms"
            )
            del send_socket
            del tt_tensor

        if rank == receiver:
            print(f"Rank {rank} is receiving data")

            padded_shape = [1, 1, 32, 32]
            recv_tensor = ttnn.allocate_tensor_on_device(
                ttnn.TensorSpec(padded_shape, ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT),
                device,
            )

            logger.info(f"Rank {rank} trying to receive from rank {sender}")
            socket_config = ttnn.SocketConfig(
                socket_connections,
                socket_mem_config,
                sender_rank=sender,
                receiver_rank=receiver,
            )
            recv_socket = ttnn.MeshSocket(device, socket_config)

            logger.info(f"Rank {rank} waiting to receive tensor")

            t_start = time.time()
            ttnn.experimental.recv_async(recv_tensor, recv_socket)
            ttnn.synchronize_device(device)
            t_end = time.time()

            logger.info(
                f"Rank {rank} receive completed in {(t_end - t_start)*1000:.2f} ms"
            )

            received_torch = ttnn.to_torch(
                recv_tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)
            )[0]

            if torch.allclose(received_torch, torch_tensor, rtol=1e-2, atol=1e-2):
                logger.success(f"Data verification PASSED - tensor matches!")
            else:
                max_diff = (received_torch - torch_tensor).abs().max().item()
                logger.error(f"Data verification FAILED - max diff: {max_diff}")
                logger.error(f"expected {torch_tensor}, received {received_torch}")
                raise ValueError("Data mismatch between sent and received tensors")

            del recv_socket
            del recv_tensor

    ttnn.distributed_context_barrier()
    logger.info(f"Rank {rank} test completed successfully")

    ttnn.close_device(device)

    return


@click.command()
@click.option(
    "-c",
    "--config",
    type=str,
    default="training_shakespeare_llama70b_pp4_tp32_fabric_galaxy.yaml",
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
    train_ids, val_ids, vocab_size, decode = prepare_data(yaml_config)

    # Use vocab_size from data instead of config
    training_config = yaml_config.setdefault("training_config", {})
    transformer_config = training_config.setdefault("transformer_config", {})
    transformer_config["vocab_size"] = int(vocab_size)

    # Initialize device mesh
    initialize_device(yaml_config)

    # Warm up with round robin communication
    import numpy as np

    shard_dim = 3
    device = ttml.autograd.AutoContext.get_instance().get_device()
    ttnn.distributed_context_barrier()

    connections_to_test = [(i, (i + 1) % world_size) for i in range(world_size)]
    connections_to_test += [((i + 1) % world_size, i) for i in range(world_size)]

    for sender, receiver in connections_to_test:
        if rank == receiver:
            print(f"Rank {rank} is receiving data")
            tensor_data = np.ones((1, 1, 1, 32), dtype=np.float32)
            mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(
                device, shard_dim, cluster_axis=1
            )
            tensor = ttml.autograd.Tensor.from_numpy(
                tensor_data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper
            )
            tensor = socket_manager.recv(tensor, distributed_ctx, sender)
            composer = ttml.core.distributed.concat_mesh_to_tensor_composer(
                device, shard_dim
            )
            tensor_data = tensor.to_numpy(composer=composer).flatten()
            assert tensor_data.tolist()[:32] == [
                i for i in range(32)
            ], f"Rank {rank} received data: {tensor_data} does not match expected data: {[i for i in range(32)]}"
        if rank == sender:
            print(f"Rank {rank} is sending data")
            tensor_data = np.array([i for i in range(32)], dtype=np.float32)
            tensor_data = tensor_data.reshape(1, 1, 1, 32)
            mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(
                device, shard_dim, cluster_axis=1
            )
            tensor = ttml.autograd.Tensor.from_numpy(
                tensor_data,
                ttnn.Layout.TILE,
                ttnn.DataType.BFLOAT16,
                mapper,
            )
            socket_manager.send(tensor, distributed_ctx, receiver)
    logger.info("Connections all verified working")

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

    # Cleanup
    ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
