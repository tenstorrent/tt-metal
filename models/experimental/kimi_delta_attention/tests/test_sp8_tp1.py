# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end seven-boundary KDA sequence-parallel protocol probe."""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.reference import kda_forward_reference
from models.experimental.kimi_delta_attention.tests.test_factory import random_weights
from models.experimental.kimi_delta_attention.tt.sp_layer import (
    SP8AffineTP1KimiDeltaAttention,
    SP8TP1KimiDeltaAttention,
)

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [
            {
                "l1_small_size": 24576,
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "trace_region_size": 256 * 1024 * 1024,
            }
        ],
        indirect=True,
    ),
]


def _probe_shape() -> tuple[KDAConfig, int]:
    if os.getenv("KDA_SP8_TARGET_SHAPE", "0") == "1":
        return (
            KDAConfig(
                hidden_size=2304,
                num_heads=32,
                head_k_dim=128,
                head_v_dim=128,
                conv_kernel_size=4,
                norm_eps=1e-5,
                chunk_size=32,
            ),
            int(os.getenv("KDA_SP8_TEST_SEQ", "5120")),
        )
    return (
        KDAConfig(
            hidden_size=256,
            num_heads=8,
            head_k_dim=128,
            head_v_dim=128,
            conv_kernel_size=4,
            norm_eps=1e-5,
            chunk_size=32,
        ),
        256,
    )


def test_sp8_tp1_layer_pcc(mesh_device: ttnn.MeshDevice) -> None:
    """All seven fabric boundaries preserve output and both causal caches."""
    config, sequence = _probe_shape()
    if sequence % (8 * config.chunk_size):
        raise ValueError(f"KDA_SP8_TEST_SEQ must be divisible by {8 * config.chunk_size}, got {sequence}")
    state_dict = random_weights(config)
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(8117)).to(
        torch.bfloat16
    )
    golden_output, golden_state = kda_forward_reference(hidden, state_dict, config)

    layer = SP8TP1KimiDeltaAttention(mesh_device, config, state_dict)
    layer.reset_state(batch_size=1)
    span = sequence // 8
    span_inputs = tuple(
        ttnn.from_torch(
            hidden[:, span_index * span : (span_index + 1) * span],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=span_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(span_device),
        )
        for span_index, span_device in enumerate(layer.span_devices)
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
    actual_recurrent = ttnn.to_torch(final_layer.recurrent_state)
    actual_convolution = ttnn.to_torch(final_layer.convolution_state)
    golden_convolution = torch.cat(
        (golden_state.q_convolution, golden_state.k_convolution, golden_state.v_convolution), dim=-1
    )
    for name, golden, actual in (
        ("output", golden_output, actual_output),
        ("recurrent state", golden_state.recurrent, actual_recurrent),
        ("convolution state", golden_convolution, actual_convolution),
    ):
        passed, pcc = comp_pcc(golden, actual, pcc=0.98)
        assert passed, f"SP=8 TP=1 {name} PCC {pcc:.6f} < 0.98"

    for boundary in range(span, sequence, span):
        passed, pcc = comp_pcc(
            golden_output[:, boundary : boundary + 1], actual_output[:, boundary : boundary + 1], pcc=0.98
        )
        assert passed, f"SP=8 TP=1 boundary {boundary} PCC {pcc:.6f} < 0.98"


def test_sp8_tp1_affine_layer_pcc(mesh_device: ttnn.MeshDevice) -> None:
    """The real fabric prefix reproduces eight-span KDA outputs and caches."""
    target_shape = os.getenv("KDA_SP8_AFFINE_TARGET_SHAPE", "0") == "1"
    config = KDAConfig(
        # Eight local heads exactly matches one TP=4 rank's KDA state in the
        # production 32-head model.  The output projection remains a local
        # TP=1 probe, intentionally without Galaxy's TP output CCL.
        hidden_size=2304 if target_shape else 256,
        num_heads=8,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        norm_eps=1e-5,
        chunk_size=32,
    )
    # The default performs two invocations to exercise cache reuse.  The
    # target mode uses the Galaxy-equivalent global T=5120 in one invocation:
    # every rank composes five 128-token affine groups and transfers exactly
    # 1 MiB of A/B state (512 KiB each) per prefix stage.
    chunk_sequence = int(os.getenv("KDA_SP8_AFFINE_TEST_SEQ", "5120" if target_shape else "4096"))
    invocations = 1 if target_shape else 2
    sequence = invocations * chunk_sequence
    state_dict = random_weights(config)
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(9137)).to(
        torch.bfloat16
    )
    golden_output, golden_state = kda_forward_reference(hidden, state_dict, config)
    layer = SP8AffineTP1KimiDeltaAttention(mesh_device, config, state_dict)
    layer.reset_state(batch_size=1)
    span = chunk_sequence // 8

    def span_inputs(chunk_index: int) -> tuple[ttnn.Tensor, ...]:
        chunk_start = chunk_index * chunk_sequence
        return tuple(
            ttnn.from_torch(
                hidden[:, chunk_start + span_index * span : chunk_start + (span_index + 1) * span],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=span_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(span_device),
            )
            for span_index, span_device in enumerate(layer.span_devices)
        )

    with ttnn.manage_config("throw_exception_on_fallback", True):
        outputs = tuple(layer.forward(*span_inputs(chunk_index)) for chunk_index in range(invocations))
    actual_output = torch.cat(
        [
            torch.cat(
                [
                    ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(span_device, dim=-1))
                    for output, span_device in zip(chunk_outputs, layer.span_devices, strict=True)
                ],
                dim=1,
            )
            for chunk_outputs in outputs
        ],
        dim=1,
    )
    final_layer = layer.layers[-1]
    assert final_layer.recurrent_state is not None
    assert final_layer.convolution_state is not None
    actual_convolution = ttnn.to_torch(final_layer.convolution_state)
    golden_convolution = torch.cat(
        (golden_state.q_convolution, golden_state.k_convolution, golden_state.v_convolution), dim=-1
    )
    for name, golden, actual in (
        ("affine output", golden_output, actual_output),
        ("affine recurrent state", golden_state.recurrent, ttnn.to_torch(final_layer.recurrent_state)),
        ("affine convolution state", golden_convolution, actual_convolution),
    ):
        passed, pcc = comp_pcc(golden, actual, pcc=0.98)
        assert passed, f"SP=8 affine TP=1 {name} PCC {pcc:.6f} < 0.98"


def test_sp8_tp1_affine_trace_layer_pcc(mesh_device: ttnn.MeshDevice, monkeypatch) -> None:
    """Capture the complete device-queued SP8 affine scheduler on LoudBox."""
    if os.getenv("KDA_SP8_AFFINE_TRACE_TEST", "0") != "1":
        pytest.skip("set KDA_SP8_AFFINE_TRACE_TEST=1 to run the experimental e2e trace gate")
    monkeypatch.setenv("KDA_SP8_TRACE_SCHEDULE", "1")
    monkeypatch.setenv("KDA_SP_FABRIC_TREE_BARRIER", "1")
    monkeypatch.setenv("KDA_SP8_PIPELINED_HANDOFFS", "1")
    monkeypatch.setenv("KDA_SP_PREFIX_LANES", "1")
    target_shape = os.getenv("KDA_SP8_AFFINE_TRACE_TARGET_SHAPE", "0") == "1"
    config = KDAConfig(
        hidden_size=2304 if target_shape else 256,
        num_heads=8,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        norm_eps=1e-5,
        chunk_size=32,
    )
    sequence = int(os.getenv("KDA_SP8_AFFINE_TRACE_SEQ", "5120" if target_shape else str(8 * 128)))
    if sequence % (8 * 128):
        raise ValueError(f"KDA_SP8_AFFINE_TRACE_SEQ must give 128-token-aligned spans, got {sequence}")
    state_dict = random_weights(config)
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(8411)).to(
        torch.bfloat16
    )
    golden_first_output, golden_first_state = kda_forward_reference(hidden, state_dict, config)
    golden_second_output, golden_second_state = kda_forward_reference(hidden, state_dict, config, golden_first_state)
    layer = SP8AffineTP1KimiDeltaAttention(mesh_device, config, state_dict)
    layer.reset_state(batch_size=1)
    layer.enable_trace_stable_state()
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

    warm_outputs = layer.forward(*span_inputs)
    for span_device in layer.span_devices:
        ttnn.synchronize_device(span_device)
    for output in warm_outputs:
        ttnn.deallocate(output)
    layer.reset_trace_stable_state()

    trace_ids = tuple(ttnn.begin_trace_capture(span_device, cq_id=0) for span_device in layer.span_devices)
    outputs = layer.forward(*span_inputs)
    for span_device, trace_id in zip(layer.span_devices, trace_ids, strict=True):
        ttnn.end_trace_capture(span_device, trace_id, cq_id=0)

    def replay_output() -> torch.Tensor:
        for span_device, trace_id in zip(layer.span_devices, trace_ids, strict=True):
            ttnn.execute_trace(span_device, trace_id, cq_id=0, blocking=False)
        for span_device in layer.span_devices:
            ttnn.synchronize_device(span_device)
        return torch.cat(
            [
                ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(span_device, dim=-1))
                for output, span_device in zip(outputs, layer.span_devices, strict=True)
            ],
            dim=1,
        )

    actual_first_output = replay_output()
    passed, pcc = comp_pcc(golden_first_output, actual_first_output, pcc=0.98)
    assert passed, f"SP=8 affine TP=1 first traced output PCC {pcc:.6f} < 0.98"
    actual_second_output = replay_output()
    final_layer = layer.layers[-1]
    assert final_layer.recurrent_state is not None
    assert final_layer.convolution_state is not None
    golden_second_convolution = torch.cat(
        (
            golden_second_state.q_convolution,
            golden_second_state.k_convolution,
            golden_second_state.v_convolution,
        ),
        dim=-1,
    )
    for name, golden, actual in (
        ("second traced output", golden_second_output, actual_second_output),
        ("second traced recurrent state", golden_second_state.recurrent, ttnn.to_torch(final_layer.recurrent_state)),
        ("second traced convolution state", golden_second_convolution, ttnn.to_torch(final_layer.convolution_state)),
    ):
        passed, pcc = comp_pcc(golden, actual, pcc=0.98)
        assert passed, f"SP=8 affine TP=1 {name} PCC {pcc:.6f} < 0.98"

    for span_device, trace_id in zip(layer.span_devices, trace_ids, strict=True):
        ttnn.release_trace(span_device, trace_id)
    for output in outputs:
        ttnn.deallocate(output)
