# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import struct


def float_to_bfloat16_packed(value):
    """Convert float to packed bfloat16 (two copies in uint32)"""
    # Convert float32 to bytes
    float_bytes = struct.pack("f", value)
    # Extract upper 16 bits (bfloat16 is truncated float32)
    bf16_bytes = float_bytes[2:4]  # upper 16 bits in little-endian layout
    # Pack two copies into uint32 (little endian)
    packed = int.from_bytes(bf16_bytes + bf16_bytes, byteorder="little")
    return packed


def float_to_uint32(value):
    """Convert float to uint32"""
    return int.from_bytes(struct.pack("f", value), byteorder="little")


def generate_mm_weights(shape, dtype):
    import torch

    torch_mm_weights = (torch.randn(shape, dtype=torch.float32) / (shape[-2] ** 0.5)).to(dtype)
    return torch_mm_weights
    # TODO: Review the below, which should provide a similar result
    # torch_mm_weights = torch.empty(shape, dtype=dtype)
    # # This assumes that weights are already pre-transposed, so inner dimension is the first dimension
    # # fan_in assumes the inner dimension is the second dimension, which is why we pass a transposed view
    # # Alternatively, we could pass the original shape and use fan_out
    # torch.nn.init.kaiming_normal_(torch_mm_weights.T, mode="fan_in", nonlinearity="linear")
    # return torch_mm_weights


# Hardcoded optimal DRAM bank to logical worker assignment for Blackhole to avoid differences from harvesting
def get_pinned_optimal_dram_bank_to_logical_worker_assignment(device, noc):
    import ttnn

    assert noc == ttnn.NOC.NOC_0, "Only NOC_0 is supported for now"
    assert device.arch() == ttnn.Arch.BLACKHOLE, "Only Blackhole is supported for now"
    return [
        ttnn.CoreCoord(0, 9),
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(0, 7),
        ttnn.CoreCoord(0, 3),
        ttnn.CoreCoord(7, 9),
        ttnn.CoreCoord(7, 1),
        ttnn.CoreCoord(7, 6),
        ttnn.CoreCoord(7, 4),
    ]
