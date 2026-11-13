# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import struct

import torch
import typing
import pytest
import inspect
import ttnn
import tempfile
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc


# Helper function to write binary file to a path
def write_binary_to_file(binary_data, dir, filename):
    bin_file = dir / f"{filename}.bin"
    logger.info(f"Writing light metal binary {binary_data.size()} bytes to {bin_file}")
    binary_data.save_to_file(str(bin_file))


# Helper function to reset a device, and run light metal binary.
def reset_device_and_replay_binary(reset_device, device, binary_data):
    logger.info("Resetting device, and running Light Metal Binary via LightMetalReplay now...")
    device = reset_device(device)
    lm_replay = ttnn.LightMetalReplay.create(binary_data, device)
    success = lm_replay.run()

    if success == 0:
        logger.info("Light Metal Binary failed to execute or encountered errors.")
    else:
        logger.info("Light Metal Binary executed successfully!")

    assert success == 1, "Light Metal Binary replay failure"


# Simple bringup single op test to see if everything uses host APIs and if it can be light-metal traced.
@pytest.mark.parametrize("shape", [[1, 3, 1024, 1024], (1, 1, 512, 512), (1, 3, 32, 32)])
@pytest.mark.parametrize("blocking", [True])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}])
def test_single_op_test_light_metal_capture(device, reset_device, shape, blocking, tmp_path):
    ttnn.light_metal_begin_capture()

    input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    output_tensor = ttnn.add(input_0_dev, input_1_dev)

    ttnn_torch_output_tensor = ttnn.to_torch(output_tensor)
    # logger.info("ttnn_torch_output_tensor: {}".format(ttnn_torch_output_tensor))

    binary_data = ttnn.light_metal_end_capture()
    write_binary_to_file(binary_data, tmp_path, inspect.currentframe().f_code.co_name)
    reset_device_and_replay_binary(reset_device, device, binary_data)

    # TODO (kmabee) - Add capture vs replay verification check


# Simple bringup, multiple ops in chain
@pytest.mark.parametrize("shape", [[1, 3, 1024, 1024], (1, 1, 512, 512), (1, 3, 32, 32)])
@pytest.mark.parametrize("blocking", [True])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}])
def test_chain_op_test_light_metal_capture(device, reset_device, shape, blocking, tmp_path):
    ttnn.light_metal_begin_capture()

    input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

    # Op chain to be traced
    def run_op_chain(input_0, input_1):
        return ttnn.neg(ttnn.add(ttnn.mul(input_1, ttnn.neg(ttnn.gelu(input_0))), ttnn.relu(input_1)))

    output_tensor = run_op_chain(input_0_dev, input_1_dev)

    ttnn_torch_output_tensor = ttnn.to_torch(output_tensor)
    # logger.info("ttnn_torch_output_tensor: {}".format(ttnn_torch_output_tensor))

    binary_data = ttnn.light_metal_end_capture()
    write_binary_to_file(binary_data, tmp_path, inspect.currentframe().f_code.co_name)
    reset_device_and_replay_binary(reset_device, device, binary_data)

    # TODO (kmabee) - Add capture vs replay verification check


# TODO (kmabee) - Add more tests, including version with metal-trace.


def _binary_with_commands_only():
    """LightMetalBinary root table with an empty commands vector and no trace_descriptors field."""
    vtable_size = 8
    object_size = 12
    table_at = 16
    vec_at = table_at + object_size
    body = bytearray()
    body += struct.pack("<I", table_at)
    body += struct.pack("<HHHH", vtable_size, object_size, 4, 0)
    body += b"\x00" * (table_at - len(body))
    body += struct.pack("<i", table_at - 4)
    body += struct.pack("<I", vec_at - (table_at + 4))
    body += struct.pack("<I", 0)
    body += struct.pack("<I", 0)
    return bytes(body)


def test_replay_missing_trace_descriptors_raises(tmp_path, expect_error):
    capture = tmp_path / "missing_trace_descriptors.bin"
    capture.write_bytes(_binary_with_commands_only())
    binary_data = ttnn._ttnn.core.LightMetalBinary.load_from_file(str(capture))
    replay = ttnn.LightMetalReplay.create(binary_data)
    with expect_error(RuntimeError, "trace_descriptors is required for LightMetalBinary"):
        replay.run()
