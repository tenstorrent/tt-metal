# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')

import click
import ttnn
import ttml
import numpy as np
from ttml.common.config import load_config, DeviceConfig
from ttml.common.utils import initialize_device


@click.command()
@click.option(
    "-c",
    "--config",
    type=str,
    default="training_shakespeare_tinyllama_tensor_parallel_3tier_fabric.yaml",
)
def main(config: str):
    yaml_config = load_config(config)

    # Initialize distributed context
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)
    distributed_ctx = autograd_ctx.get_distributed_context()

    rank = distributed_ctx.rank()
    world_size = distributed_ctx.size()
    assert (
        world_size > 1
    ), f"World size must be greater than 1, world size: {world_size}"

    # Initialize socket manager
    autograd_ctx.initialize_socket_manager(ttml.core.distributed.SocketType.FABRIC)

    socket_manager = autograd_ctx.get_socket_manager()
    device_config = DeviceConfig(yaml_config)

    # Initialize device
    initialize_device(yaml_config)

    N = 1024
    values = np.ones((1, 1, 1, N), dtype=np.float32) * (rank + 1)
    tt_values = ttml.autograd.Tensor.from_numpy(
        values, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    )

    if rank > 0:
        socket_manager.send(tt_values, distributed_ctx, 0)
    else:
        for other_rank in range(1, world_size):
            recv_from_other_rank = ttml.core.empty_like(tt_values)
            recv_from_other_rank = socket_manager.recv(
                recv_from_other_rank, distributed_ctx, other_rank
            )
            tt_values = tt_values + recv_from_other_rank
        device = autograd_ctx.get_device()
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        values_before_all_reduce = tt_values.to_numpy(composer=composer)
        assert np.all(
            values_before_all_reduce == world_size * (world_size + 1) / 2
        ), f"Values before all reduce do not match expected values: {values_before_all_reduce}"

        tt_values_after_all_reduce = ttml.ops.distributed.all_reduce(tt_values)
        values_after_all_reduce = tt_values_after_all_reduce.to_numpy(composer=composer)
        num_devices = device_config.total_devices()
        assert np.all(
            values_after_all_reduce == world_size * (world_size + 1) / 2 * num_devices
        ), f"Values after all reduce do not match expected values: {values_after_all_reduce}"

    # Cleanup
    distributed_ctx.barrier()
    autograd_ctx.close_device()
    print(f"Rank {rank}: Finished")


if __name__ == "__main__":
    main()
