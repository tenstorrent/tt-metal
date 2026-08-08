#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Shared test helpers for ACE-Step v1.5 bring-up (TTTv2-style, single Blackhole p150).

Mirrors the BGE-M3 pattern (models/demos/wormhole/bge_m3/tests/test_utils.py) so the
PCC tests stay small and consistent. Reference model = HF ACE-Step/acestep-v15-base
loaded with trust_remote_code from the local HF cache.
"""

import pytest
import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.lazy_weight import LazyWeight
from models.common.utility_functions import comp_allclose, comp_pcc

# Real ACE-Step v1.5 base dims (config.json), used to parameterize unit tests with
# the true architecture so we never overfit to toy shapes.
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 6144
NUM_ATTENTION_HEADS = 16
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
RMS_NORM_EPS = 1e-6
ROPE_THETA = 1000000
SLIDING_WINDOW = 128
NUM_HIDDEN_LAYERS = 24
TEXT_HIDDEN_DIM = 1024

HF_MODEL_ID = "ACE-Step/acestep-v15-base"

# Sequence lengths to exercise (tile-aligned; audio latent seqs can be long).
SEQUENCE_LENGTHS = [128, 512, 1024, 2048]


def require_single_device(device) -> None:
    if hasattr(device, "get_num_devices") and device.get_num_devices() != 1:
        pytest.skip("ACE-Step PCC tests currently target single-device (p150) execution")


def make_lazy_weight(
    source: torch.Tensor,
    device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> LazyWeight:
    return LazyWeight(
        source=source,
        dtype=dtype,
        device=device,
        layout=layout,
        memory_config=memory_config,
    )


def to_ttnn_tensor(
    tensor: torch.Tensor,
    device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        device=device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
    )


def to_torch(tt_tensor: ttnn.Tensor, expected_shape: tuple[int, ...]) -> torch.Tensor:
    output = to_torch_auto_compose(tt_tensor).to(torch.float32)
    assert tuple(output.shape) == expected_shape, f"Expected output shape {expected_shape}, got {tuple(output.shape)}"
    return output


def assert_pcc(reference: torch.Tensor, candidate: torch.Tensor, threshold: float) -> None:
    passing, pcc_message = comp_pcc(reference, candidate, threshold)
    allclose, allclose_message = comp_allclose(reference, candidate)
    assert passing, f"PCC check failed: {pcc_message}; {allclose_message}; allclose={allclose}"


def load_hf_acestep_config():
    """Load the real ACE-Step v1.5 config (trust_remote_code) from the local HF cache."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
