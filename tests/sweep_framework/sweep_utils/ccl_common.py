# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from itertools import count, takewhile

import ttnn

from loguru import logger
from tests.scripts.common import get_updated_device_params


def mesh_shape_iterator(num_devices, limit=None):
    if num_devices == 1:
        return []

    assert num_devices % 2 == 0
    for r in takewhile(lambda x: x <= num_devices, (2**i for i in (range(limit) if limit else count()))):
        yield (num_devices // r, r)


@contextmanager
def device_context(mesh_shape, fabric_config, device_params=None):
    device_params = device_params or {}
    mesh_device = None
    try:
        logger.info("Setting up device")
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        ttnn.set_fabric_config(fabric_config)
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(mesh_shape), **get_updated_device_params(device_params)
        )
        yield mesh_device, None
    except AssertionError as e:
        logger.error(f"Device error: {e}")
        yield None, f"Device error {e}"
    finally:
        logger.info("Tearing down device")
        if mesh_device:
            ttnn.close_mesh_device(mesh_device)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            del mesh_device


def get_serializable_shard_specs(
    input_shape, input_cores, input_strategy, output_shape, output_cores, output_strategy, valid_tensor_shapes
):
    return {
        "input": {
            "shape": input_shape,
            "cores": input_cores,
            "strategy": input_strategy,
        },
        "output": {"shape": output_shape, "cores": output_cores, "strategy": output_strategy},
        "valid_tensor_shapes": valid_tensor_shapes,
    }


def validate_serializable_shard_spec(input_shape, serializable_shard_specs):
    return serializable_shard_specs is None or tuple(input_shape) in list(
        map(tuple, serializable_shard_specs["valid_tensor_shapes"])
    )


def _parse_serializable_shard_spec(serializable_shard_spec, output_shape, tile_layout=False):
    assert len(serializable_shard_spec) == 3

    shape, cores, strategy = tuple(serializable_shard_spec.values())

    if strategy == "w":
        strategy, layout = ttnn.ShardStrategy.WIDTH, ttnn.TensorMemoryLayout.WIDTH_SHARDED
    elif strategy == "h":
        strategy, layout = ttnn.ShardStrategy.HEIGHT, ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    else:
        raise RuntimeError("Ivalid shard strategy option")

    if shape is None:
        core_grid = ttnn.CoreGrid(**dict(zip(("x", "y"), cores)))
        return (
            ttnn.create_sharded_memory_config_(output_shape, core_grid, strategy, tile_layout=tile_layout).shard_spec,
            layout,
        )
    else:
        core_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores[0] - 1, cores[1] - 1))}
        )
        return ttnn.ShardSpec(core_grid, shape, ttnn.ShardOrientation.ROW_MAJOR), layout


def get_mem_configs(buffer_type, serializable_shard_specs, output_shape, tile_layout=False):
    if serializable_shard_specs is None:
        return ttnn.MemoryConfig(buffer_type=buffer_type), ttnn.MemoryConfig(buffer_type=buffer_type)
    else:
        input_spec, input_layout = _parse_serializable_shard_spec(
            serializable_shard_specs["input"], None, tile_layout=tile_layout
        )
        input_config = ttnn.MemoryConfig(input_layout, buffer_type, input_spec)

        output_spec, output_layout = _parse_serializable_shard_spec(
            serializable_shard_specs["output"], output_shape, tile_layout=tile_layout
        )
        output_config = ttnn.MemoryConfig(output_layout, buffer_type, output_spec)

        assert input_layout == output_layout

    return input_config, output_config
