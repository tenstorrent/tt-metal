# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CompressedTensor class."""

from __future__ import annotations

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.compressed_tensor import (
    CompressedTensor,
    CompressedTensorAssigner,
    bfp_tile_packed_size,
    ttnn_quantize_fn,
)
from models.demos.deepseek_v3_b1.compressed_tensor.metrics import metric_value


def test_from_tensor_round_trip():
    """Create CompressedTensor via from_tensor, unpack, and verify PCC."""
    torch.manual_seed(42)
    x = torch.randn(128, 128)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    ct = CompressedTensor.from_tensor(x, assigner)

    logger.info(f"{ct}")
    logger.info(f"Tile counts: {ct.tile_counts}")

    assert ct.tile_counts["bfp8"] > 0 and ct.tile_counts["bfp4"] > 0, f"Expected mix: {ct.tile_counts}"

    recovered = ct.unpack()
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    logger.info(f"Round-trip PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low"


def test_assignment_stored_correctly():
    """The assignment tensor should round-trip through ttnn uint8 storage."""
    torch.manual_seed(0)
    x = torch.randn(64, 64)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    result = assigner.assign(x, ttnn_quantize_fn)
    ct = CompressedTensor(x, result.assignment)

    recovered_assignment = ct.get_assignment_numpy()
    assert (recovered_assignment == result.assignment).all(), "Assignment round-trip mismatch"


def test_data_bytes_matches_packed_size():
    """data_bytes property should match actual packed tensor size."""
    torch.manual_seed(42)
    x = torch.randn(128, 128)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    ct = CompressedTensor.from_tensor(x, assigner)

    actual_size = ttnn.to_torch(ct.data).numel()
    logger.info(f"data_bytes={ct.data_bytes}, actual tensor size={actual_size}")
    assert ct.data_bytes == actual_size, f"data_bytes {ct.data_bytes} != actual {actual_size}"


def test_packed_size_savings():
    """CompressedTensor with mixed formats should use fewer bytes than uniform bfp8."""
    torch.manual_seed(99)
    x = torch.randn(128, 128)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    ct = CompressedTensor.from_tensor(x, assigner)

    uniform_bfp8_bytes = ct.num_tiles * bfp_tile_packed_size(7)

    logger.info(f"{ct}")
    logger.info(
        f"Savings: {uniform_bfp8_bytes - ct.data_bytes} bytes "
        f"({100 * (1 - ct.data_bytes / uniform_bfp8_bytes):.1f}%)"
    )
    assert (
        ct.data_bytes <= uniform_bfp8_bytes
    ), f"Mixed ({ct.data_bytes}) should be <= uniform bfp8 ({uniform_bfp8_bytes})"


# ---------------------------------------------------------------------------
# On-device tests
# ---------------------------------------------------------------------------


def test_device_round_trip(device):
    """Pack on host, place on device via from_tensor(device=), unpack after read-back."""
    torch.manual_seed(42)
    x = torch.randn(128, 128)

    assigner = CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=["bfp8", "bfp4"])
    ct = CompressedTensor.from_tensor(x, assigner, device=device)

    logger.info(f"{ct}")
    logger.info(f"Tile counts: {ct.tile_counts}")

    assert ct.tile_counts["bfp8"] > 0 and ct.tile_counts["bfp4"] > 0, f"Expected mix: {ct.tile_counts}"

    # unpack() handles from_device internally
    recovered = ct.unpack()
    pcc = metric_value(x.numpy(), recovered.numpy(), "pcc")
    logger.info(f"Device round-trip PCC: {pcc:.6f}")
    assert pcc > 0.98, f"PCC {pcc:.6f} unexpectedly low after device round-trip"
