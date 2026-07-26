# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Galaxy SP=8, TP=4 serial KDA correctness baseline."""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.reference import kda_forward_reference
from models.experimental.kimi_delta_attention.tests.test_factory import random_weights
from models.experimental.kimi_delta_attention.tt.sp_affine_prefix import SP8AffinePrefixProbe
from models.experimental.kimi_delta_attention.tt.sp_layer import (
    SP8AffineTP4KimiDeltaAttention,
    SP8TP4KimiDeltaAttention,
)


pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_2D}],
        indirect=True,
    ),
]


def _host_shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def _full_convolution_state(layer: SP8TP4KimiDeltaAttention, config: KDAConfig) -> torch.Tensor:
    convolution_shards = _host_shards(layer.layers[-1].convolution_state)
    local_key_width = config.head_k_dim * config.num_heads // 4
    local_value_width = config.head_v_dim * config.num_heads // 4
    return torch.cat(
        (
            torch.cat([shard[..., :local_key_width] for shard in convolution_shards], dim=-1),
            torch.cat([shard[..., local_key_width : 2 * local_key_width] for shard in convolution_shards], dim=-1),
            torch.cat(
                [
                    shard[..., 2 * local_key_width : 2 * local_key_width + local_value_width]
                    for shard in convolution_shards
                ],
                dim=-1,
            ),
        ),
        dim=-1,
    )


def test_sp8_tp4_fabric_tree_barrier_stability(mesh_device: ttnn.MeshDevice) -> None:
    """Exercise all four rank-aligned SP8 atomic trees without KDA payloads."""
    if os.getenv("KDA_SP8TP4_FABRIC_TREE_BARRIER_TEST", "0") != "1":
        pytest.skip("set KDA_SP8TP4_FABRIC_TREE_BARRIER_TEST=1 to run the experimental barrier gate")
    repetitions = int(os.getenv("KDA_SP_FABRIC_TREE_REPS", "10"))
    if repetitions <= 0:
        raise ValueError(f"KDA_SP_FABRIC_TREE_REPS must be positive, got {repetitions}")
    probe = SP8AffinePrefixProbe(mesh_device, tp_size=4)
    for _ in range(repetitions):
        probe.queue_fabric_tree_barrier()
        probe._synchronize()


def test_sp8_tp4_serial_layer_pcc(mesh_device: ttnn.MeshDevice) -> None:
    """The production rank layout preserves seven causal TP4 cache handoffs."""
    config = KDAConfig(
        hidden_size=256,
        num_heads=8,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        norm_eps=1e-5,
        chunk_size=32,
    )
    sequence = 8 * config.chunk_size
    state_dict = random_weights(config)
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(9827)).to(
        torch.bfloat16
    )
    golden_output, golden_state = kda_forward_reference(hidden, state_dict, config)
    layer = SP8TP4KimiDeltaAttention(mesh_device, config, state_dict)
    layer.reset_state(batch_size=1)
    span = sequence // 8
    span_inputs = tuple(
        ttnn.from_torch(
            hidden[:, rank * span : (rank + 1) * span],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=span_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(span_device),
        )
        for rank, span_device in enumerate(layer.span_devices)
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        outputs = layer.forward(*span_inputs)

    actual_output = torch.cat(
        [
            ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(span_device, dim=-1))
            for output, span_device in zip(outputs, layer.span_devices, strict=True)
        ],
        dim=1,
    )
    final_layer = layer.layers[-1]
    assert final_layer.recurrent_state is not None
    assert final_layer.convolution_state is not None
    golden_convolution = torch.cat(
        (golden_state.q_convolution, golden_state.k_convolution, golden_state.v_convolution), dim=-1
    )
    for name, golden, actual in (
        ("output", golden_output, actual_output),
        ("recurrent state", golden_state.recurrent, torch.cat(_host_shards(final_layer.recurrent_state), dim=1)),
        ("convolution state", golden_convolution, _full_convolution_state(layer, config)),
    ):
        assert torch.isfinite(actual).all(), f"SP=8 TP=4 {name} contains non-finite values"
        passed, pcc = comp_pcc(golden, actual, pcc=0.98)
        assert passed, f"SP=8 TP=4 {name} PCC {pcc:.6f} < 0.98"

    for boundary in range(span, sequence, span):
        passed, pcc = comp_pcc(
            golden_output[:, boundary : boundary + 1], actual_output[:, boundary : boundary + 1], pcc=0.98
        )
        assert passed, f"SP=8 TP=4 first post-boundary token {boundary} PCC {pcc:.6f} < 0.98"


def test_sp8_tp4_affine_layer_pcc(mesh_device: ttnn.MeshDevice) -> None:
    """The TP4 affine prefix replaces relay while retaining TP output CCLs."""
    config = KDAConfig(
        hidden_size=256,
        num_heads=8,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        norm_eps=1e-5,
        chunk_size=32,
    )
    sequence = 8 * 128
    state_dict = random_weights(config)
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(8831)).to(
        torch.bfloat16
    )
    golden_output, golden_state = kda_forward_reference(hidden, state_dict, config)
    layer = SP8AffineTP4KimiDeltaAttention(mesh_device, config, state_dict)
    layer.reset_state(batch_size=1)
    span = sequence // 8
    span_inputs = tuple(
        ttnn.from_torch(
            hidden[:, rank * span : (rank + 1) * span],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=span_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(span_device),
        )
        for rank, span_device in enumerate(layer.span_devices)
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        outputs = layer.forward(*span_inputs)

    actual_output = torch.cat(
        [
            ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(span_device, dim=-1))
            for output, span_device in zip(outputs, layer.span_devices, strict=True)
        ],
        dim=1,
    )
    final_layer = layer.layers[-1]
    assert final_layer.recurrent_state is not None
    assert final_layer.convolution_state is not None
    golden_convolution = torch.cat(
        (golden_state.q_convolution, golden_state.k_convolution, golden_state.v_convolution), dim=-1
    )
    for name, golden, actual in (
        ("affine output", golden_output, actual_output),
        ("affine recurrent state", golden_state.recurrent, torch.cat(_host_shards(final_layer.recurrent_state), dim=1)),
        ("affine convolution state", golden_convolution, _full_convolution_state(layer, config)),
    ):
        assert torch.isfinite(actual).all(), f"SP=8 affine TP=4 {name} contains non-finite values"
        passed, pcc = comp_pcc(golden, actual, pcc=0.98)
        assert passed, f"SP=8 affine TP=4 {name} PCC {pcc:.6f} < 0.98"
