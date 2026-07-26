# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness test for the LoudBox SP=2, TP=4 KDA path."""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.reference import kda_forward_reference
from models.experimental.kimi_delta_attention.tests.test_factory import random_weights
from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention
from models.experimental.kimi_delta_attention.tt.sp_layer import SP2TP4KimiDeltaAttention
from models.tt_transformers.tt.ccl import TT_CCL


def _host_shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def _sp_test_shape() -> tuple[KDAConfig, int]:
    """Return the smoke or LB production-rank-equivalent PCC shape.

    Run ``KDA_SP_TARGET_SHAPE=1 pytest .../test_sp2_tp4.py`` after a LoudBox
    reset to check the plan's global-T=1280, TP=4-per-span gate.  The default
    remains a fast full-path smoke test suitable for regular development.
    """
    if os.getenv("KDA_SP_TARGET_SHAPE", "0") == "1":
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
            int(os.getenv("KDA_SP_TEST_SEQ", "1280")),
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
        int(os.getenv("KDA_SP_TEST_SEQ", "64")),
    )


pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_2D}],
        indirect=True,
    ),
]


def test_sp2_tp4_layer_pcc(mesh_device: ttnn.MeshDevice) -> None:
    """The second span must consume the recurrent and short-conv carry on fabric."""
    config, sequence = _sp_test_shape()
    if sequence % 64:
        raise ValueError(f"KDA_SP_TEST_SEQ must be divisible by 64 for two aligned spans, got {sequence}")
    state_dict = random_weights(config)
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(6081)).to(
        torch.bfloat16
    )
    golden_output, golden_state = kda_forward_reference(hidden, state_dict, config)

    layer = SP2TP4KimiDeltaAttention(mesh_device, config, state_dict)
    layer.reset_state(batch_size=1)
    span_inputs = []
    for span, span_device in enumerate(layer.span_devices):
        span_inputs.append(
            ttnn.from_torch(
                hidden[:, span * (sequence // 2) : (span + 1) * (sequence // 2)],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=span_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(span_device),
            )
        )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        first_output, second_output = layer.forward(*span_inputs)

    actual_output = torch.cat(
        (
            ttnn.to_torch(first_output, mesh_composer=ttnn.ConcatMeshToTensor(layer.span_devices[0], dim=-1)),
            ttnn.to_torch(second_output, mesh_composer=ttnn.ConcatMeshToTensor(layer.span_devices[1], dim=-1)),
        ),
        dim=1,
    )
    assert layer.second_layer.recurrent_state is not None
    assert layer.second_layer.convolution_state is not None
    actual_recurrent = torch.cat(_host_shards(layer.second_layer.recurrent_state), dim=1)
    convolution_shards = _host_shards(layer.second_layer.convolution_state)
    local_key_width = config.head_k_dim * config.num_heads // 4
    local_value_width = config.head_v_dim * config.num_heads // 4
    actual_convolution = torch.cat(
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
    golden_convolution = torch.cat(
        (golden_state.q_convolution, golden_state.k_convolution, golden_state.v_convolution), dim=-1
    )
    # PCC comparisons can omit NaNs while accumulating their covariance.  An
    # SP output is not valid if a fused producer leaves even one tile
    # unwritten, so make finiteness an explicit correctness invariant.
    assert torch.isfinite(actual_output).all(), "SP=2 TP=4 output contains non-finite values"
    for name, golden, actual in (
        ("output", golden_output, actual_output),
        ("recurrent state", golden_state.recurrent, actual_recurrent),
        ("convolution state", golden_convolution, actual_convolution),
    ):
        passed, pcc = comp_pcc(golden, actual, pcc=0.98)
        assert passed, f"SP=2 TP=4 {name} PCC {pcc:.6f} < 0.98"

    boundary = sequence // 2
    passed, pcc = comp_pcc(
        golden_output[:, boundary : boundary + 1], actual_output[:, boundary : boundary + 1], pcc=0.98
    )
    assert passed, f"SP=2 TP=4 first post-boundary token PCC {pcc:.6f} < 0.98"


def test_sp2_tp4_direct_output_trace_pcc(mesh_device: ttnn.MeshDevice, monkeypatch) -> None:
    """Validate the no-clone MRS output lifetime across two child-trace replays."""
    if os.getenv("KDA_SP_DIRECT_OUTPUT_TRACE_TEST", "0") != "1":
        pytest.skip("set KDA_SP_DIRECT_OUTPUT_TRACE_TEST=1 to run the direct-output trace gate")
    monkeypatch.setenv("KDA_MRS_DIRECT_OUTPUT", "1")
    monkeypatch.setenv("KDA_SP_SPLIT_AFFINE", "1")
    config = KDAConfig(
        hidden_size=2304,
        num_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        norm_eps=1e-5,
        chunk_size=32,
    )
    sequence = 1280
    state_dict = random_weights(config)
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(2607)).to(
        torch.bfloat16
    )
    golden_first, golden_state = kda_forward_reference(hidden, state_dict, config)
    golden_second, _ = kda_forward_reference(hidden, state_dict, config, golden_state)
    layer = SP2TP4KimiDeltaAttention(mesh_device, config, state_dict)
    layer.reset_state(batch_size=1)
    layer.enable_trace_stable_state()
    span = sequence // 2
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

    # Compile before capture, then reset the fixed state buffers without
    # changing their addresses.  Direct MRS results intentionally stay live:
    # they alias the persistent output buffers reused by the trace.
    layer.forward(*span_inputs)
    for span_device in layer.span_devices:
        ttnn.synchronize_device(span_device)
    layer.reset_trace_stable_state()

    traces = tuple(ttnn.begin_trace_capture(span_device, cq_id=0) for span_device in layer.span_devices)
    outputs = layer.forward(*span_inputs)
    for span_device, trace_id in zip(layer.span_devices, traces, strict=True):
        ttnn.end_trace_capture(span_device, trace_id, cq_id=0)

    for expected_name, expected in (("first", golden_first), ("second", golden_second)):
        for span_device, trace_id in zip(layer.span_devices, traces, strict=True):
            ttnn.execute_trace(span_device, trace_id, cq_id=0, blocking=False)
        for span_device in layer.span_devices:
            ttnn.synchronize_device(span_device)
        actual = torch.cat(
            [
                ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(span_device, dim=-1))
                for output, span_device in zip(outputs, layer.span_devices, strict=True)
            ],
            dim=1,
        )
        assert torch.isfinite(actual).all(), f"direct-output {expected_name} replay contains non-finite values"
        passed, pcc = comp_pcc(expected, actual, pcc=0.98)
        assert passed, f"direct-output {expected_name} trace replay PCC {pcc:.6f} < 0.98"

    for span_device, trace_id in zip(layer.span_devices, traces, strict=True):
        ttnn.release_trace(span_device, trace_id)


def test_tp4_submeshes_execute_without_handoff(mesh_device: ttnn.MeshDevice) -> None:
    """Isolate both local TP=4 paths from the cross-span fabric handoff."""
    config, _ = _sp_test_shape()
    sequence = 64
    state_dict = random_weights(config)
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(9123)).to(
        torch.bfloat16
    )
    layer = SP2TP4KimiDeltaAttention(mesh_device, config, state_dict)
    layer.reset_state(batch_size=1)
    span_inputs = tuple(
        ttnn.from_torch(
            hidden[:, span * (sequence // 2) : (span + 1) * (sequence // 2)],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=span_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(span_device),
        )
        for span, span_device in enumerate(layer.span_devices)
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        first_output = layer.first_layer.forward(span_inputs[0], mode="chunk")
        second_output = layer.second_layer.forward(span_inputs[1], mode="chunk")
    for output, span_device in zip((first_output, second_output), layer.span_devices, strict=True):
        actual = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(span_device, dim=-1))
        assert torch.isfinite(actual).all()


def test_tp4_standalone_submesh_control(mesh_device: ttnn.MeshDevice) -> None:
    """Prove the regular (non-fused-output) TP=4 collective without SP sockets."""
    config = KDAConfig(
        hidden_size=256,
        num_heads=4,
        head_k_dim=128,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
        chunk_size=32,
    )
    sequence = 32
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(1941)).to(
        torch.bfloat16
    )
    state_dict = random_weights(config)
    golden_output, _ = kda_forward_reference(hidden, state_dict, config)
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
    layer = KimiDeltaAttention(submesh, config, state_dict, tt_ccl=TT_CCL(submesh))
    layer.reset_state(batch_size=1)
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output = layer.forward(hidden_tt, mode="chunk")
    actual = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=-1))
    passed, pcc = comp_pcc(golden_output, actual, pcc=0.98)
    assert passed, f"TP=4 standalone output PCC {pcc:.6f} < 0.98"
