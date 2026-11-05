# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device weight-placement contracts for the KDA layer."""

from dataclasses import replace

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_small_kda_test_config, random_weights
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.weights import load_kda_weights
from models.tt_transformers.tt.ccl import TT_CCL
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_equal,
    make_actual_start,
)

pytestmark = run_for_blackhole()


def _host_shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def _tp_rank(physical_index: int, mesh_columns: int, tensor_parallel_axis: int) -> int:
    row, column = divmod(physical_index, mesh_columns)
    return (row, column)[tensor_parallel_axis]


_FABRIC_1D = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params",
    [
        pytest.param((1, 8), 1, _FABRIC_1D, id="tp8-1d"),
        pytest.param((2, 4), 0, _FABRIC_1D, id="tp2-axis0"),
        pytest.param((2, 4), 1, _FABRIC_1D, id="tp4-axis1"),
        pytest.param(
            (8, 4),
            1,
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="tp4-8x4-fabric2d",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
# A bounded gate (Kimi-K3) folds the per-head decay scale into the decay projection and bias at load.
@pytest.mark.parametrize("gate_lower_bound", [None, -5.0], ids=["softplus-gate", "bounded-gate"])
def test_device_weight_placement(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    gate_lower_bound: float | None,
) -> None:
    config = KDAConfig(
        hidden_size=64,
        num_heads=8,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
        gate_lower_bound=gate_lower_bound,
    )
    state_dict = random_weights(config)
    weights = load_kda_weights(
        mesh_device,
        config,
        state_dict,
        tensor_parallel_axis=tensor_parallel_axis,
    )
    tensor_parallel_size = tuple(mesh_device.shape)[tensor_parallel_axis]
    local_heads = config.num_heads // tensor_parallel_size
    gate_projection = state_dict["g_b_proj.weight"].reshape(config.num_heads, config.head_v_dim, config.head_v_dim)
    direct_gate = torch.matmul(gate_projection, state_dict["g_a_proj.weight"]).reshape(config.v_dim, config.hidden_size)
    decay_scale = state_dict["A_log"].float().exp()
    decay_scale = decay_scale if gate_lower_bound is not None else -decay_scale
    decay_scale = decay_scale.expand(-1, -1, -1, config.head_k_dim).reshape(1, 1, config.q_dim)
    decay_bias = state_dict["dt_bias"].float().reshape(1, 1, config.q_dim)
    decay_output = state_dict["f_b_proj.weight"].T
    if gate_lower_bound is not None:
        # The fold runs on device after load with the FP32 scale: the projection is rounded to BF16 once, the
        # offset A * dt_bias stays FP32.
        decay_output = decay_output.to(torch.bfloat16).float() * decay_scale.reshape(1, -1)
        decay_bias = decay_bias * decay_scale

    assert weights.tensor_parallel_size == tensor_parallel_size
    assert weights.tensor_parallel_axis == tensor_parallel_axis
    fields = {
        "input": _host_shards(weights.input_projection),
        "decay_output": _host_shards(weights.decay_output_projection),
        "output": _host_shards(weights.output_projection),
        "decay_scale": _host_shards(weights.decay_scale_flat),
        "decay_bias": _host_shards(weights.decay_bias_flat),
        "norm": _host_shards(weights.norm),
        "tap": _host_shards(weights.convolution_taps[0]),
    }

    for physical_index in range(len(fields["input"])):
        tp_rank = (
            _tp_rank(physical_index, tuple(mesh_device.shape)[1], tensor_parallel_axis)
            if tuple(mesh_device.shape)[0] > 1
            else physical_index
        )
        head_start = tp_rank * local_heads
        head_stop = head_start + local_heads
        key_start, key_stop = head_start * config.head_k_dim, head_stop * config.head_k_dim
        value_start, value_stop = head_start * config.head_v_dim, head_stop * config.head_v_dim
        expected_qkv = torch.cat(
            (
                state_dict["q_proj.weight"][key_start:key_stop],
                state_dict["k_proj.weight"][key_start:key_stop],
                state_dict["v_proj.weight"][value_start:value_stop],
            ),
            dim=0,
        ).T
        expected_auxiliary = torch.cat(
            (
                state_dict["f_a_proj.weight"],
                direct_gate[value_start:value_stop],
                state_dict["b_proj.weight"][head_start:head_stop],
            ),
            dim=0,
        ).T
        expected_tap = torch.cat(
            (
                state_dict["q_conv1d.weight"][key_start:key_stop, 0, 0],
                state_dict["k_conv1d.weight"][key_start:key_stop, 0, 0],
                state_dict["v_conv1d.weight"][value_start:value_stop, 0, 0],
            )
        ).reshape(1, 1, -1)
        expected = {
            "input": torch.cat((expected_qkv, expected_auxiliary), dim=-1),
            "decay_output": decay_output[:, key_start:key_stop],
            "output": state_dict["o_proj.weight"][:, value_start:value_stop].T,
            "decay_scale": decay_scale[..., key_start:key_stop],
            "decay_bias": decay_bias[..., key_start:key_stop],
            "norm": state_dict["o_norm.weight"],
            "tap": expected_tap,
        }
        for name, expected_tensor in expected.items():
            # A bounded gate's scale and offset are FP32 on device; every other weight is BF16.
            fp32 = gate_lower_bound is not None and name in ("decay_scale", "decay_bias")
            dtype = torch.float32 if fp32 else torch.bfloat16
            assert_equal(
                expected_tensor.to(dtype),
                fields[name][physical_index],
                name=f"{name} weight device {physical_index}",
            )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_tp_layer_with_nonsquare_state_matches_reference(mesh_device: ttnn.MeshDevice) -> None:
    config = KDAConfig(
        hidden_size=256,
        num_heads=8,
        head_k_dim=32,
        head_v_dim=256,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )
    state_dict = random_weights(config)
    sequence = 32
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(911)).to(
        torch.bfloat16
    )
    golden_output, golden_state = kda_forward_reference(hidden, state_dict, config)

    layer = ttKDA(mesh_device, config, state_dict, tt_ccl=TT_CCL(mesh_device), active_seq_len=sequence)
    initial_state = layer.allocate_state(batch_size=1)
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output, state = layer.forward(hidden_tt, initial_state, make_actual_start(layer.device))

    actual_output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    recurrent_shards = _host_shards(state.recurrent)
    convolution_shards = _host_shards(state.convolution)
    actual_recurrent = torch.cat(recurrent_shards, dim=1)
    local_key_width = config.head_k_dim
    local_value_width = config.head_v_dim
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
    ).to(torch.bfloat16)

    for name, golden, actual in (
        ("output", golden_output, actual_output),
        ("recurrent state", golden_state.recurrent, actual_recurrent),
        ("convolution state", golden_convolution, actual_convolution),
    ):
        assert_accurate(golden, actual, name=f"TP=8 {name}", pcc_threshold=0.999, nrmse_threshold=0.03)


@pytest.mark.use_module_device
def test_bounded_decay_gate_has_no_systematic_error(device: ttnn.Device) -> None:
    """The device's bounded gate sigmoid(A (f_b r + dt_bias)) is unbiased per channel against FP64.

    A per-channel bias of the gate accumulates over each chunk instead of averaging out (see _FP32_ARTIFACTS in
    tt/kda/weights.py). The bias is measured beyond the gate's single BF16 rounding, against FP64 evaluated from the
    device's own decay rank, so this isolates the gate chain from the input projection. A host emulation of the BF16
    scale-and-offset chain on the same rank must fail the same bar.
    """
    config = replace(make_small_kda_test_config(), num_heads=4, gate_lower_bound=-5.0)
    state_dict = random_weights(config)
    # Offsets of real scale: A dt_bias reaches |logit| ~ 2-10, where a BF16 offset is off by up to 2^-9 of it.
    state_dict["dt_bias"] = 2.0 * torch.randn(config.q_dim, generator=torch.Generator().manual_seed(5))
    tokens = 256
    hidden = torch.randn(1, tokens, config.hidden_size, generator=torch.Generator().manual_seed(7)).to(torch.bfloat16)
    layer = ttKDA(device, config, state_dict, active_seq_len=tokens)
    hidden_tt = ttnn.from_torch(
        hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    inputs = layer._project_inputs(hidden_tt)
    gate = ttnn.to_torch(layer._compute_decay(inputs.projected, inputs.decay_rank_offset)).double()
    gate = gate.reshape(tokens, config.q_dim)
    projected = ttnn.to_torch(inputs.projected).double().reshape(tokens, -1)
    rank = projected[:, inputs.decay_rank_offset : inputs.decay_rank_offset + config.head_k_dim]
    scale = state_dict["A_log"].double().exp().expand(-1, -1, -1, config.head_k_dim).reshape(config.q_dim)
    logits = scale * (rank @ state_dict["f_b_proj.weight"].double().T + state_dict["dt_bias"].double())
    expected = torch.sigmoid(logits)

    relative = (gate - expected) / expected
    # The gate's single BF16 rounding is kept by design, and it does not average out over a channel whose gates all
    # fall inside one BF16 step (near 1 the step is 2^-8): measure the chain's bias against the rounded FP64 gate.
    rounding = (expected.to(torch.bfloat16).double() - expected) / expected
    channel_bias = lambda actual: ((actual - expected) / expected - rounding).mean(dim=0).abs().max()
    assert logits.abs().max() > 8, "the case must reach the offsets where BF16 constants bias the gate"
    assert channel_bias(gate) < 1e-3, f"per-channel mean dG/G beyond rounding up to {channel_bias(gate):.2e}"
    # Negative control: BF16 A, BF16 A f_b and BF16 A dt_bias as the matmul bias, sigmoid rounded to BF16.
    scale_bf16 = scale.to(torch.bfloat16).double()
    projection_bf16 = (state_dict["f_b_proj.weight"].double().T * scale_bf16).to(torch.bfloat16).double()
    offset_bf16 = (state_dict["dt_bias"].to(torch.bfloat16).double() * scale_bf16).to(torch.bfloat16).double()
    bf16_gate = torch.sigmoid(rank @ projection_bf16 + offset_bf16).to(torch.bfloat16).double()
    assert channel_bias(bf16_gate) > 3e-3, f"the BF16 chain's bias {channel_bias(bf16_gate):.2e} must fail the bar"
    assert relative.abs().max() < 2e-2, f"max dG/G {relative.abs().max():.2e}"
