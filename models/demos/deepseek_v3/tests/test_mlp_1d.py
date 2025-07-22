# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP
from models.demos.deepseek_v3.tt.mlp.mlp_1d import MLP1D
from models.demos.deepseek_v3.tt.mlp.mlp_1d_dequant import MLP1DDequant
from models.demos.deepseek_v3.tt.mlp.non_expert import NonExpert
from models.demos.deepseek_v3.tt.mlp.shared_expert import SharedExpert
from models.demos.deepseek_v3.utils.config_helpers import dequantize
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    get_model_config,
    load_reference_io_tensors_for_module,
    load_state_dict,
    run_module_forward,
)
from models.utility_functions import comp_pcc


def test_convert_weights_for_non_dequantized_mlp(hf_config, tmp_path, mesh_row):
    reference_model = DeepseekV3MLP(hf_config)
    reference_state_dict = reference_model.state_dict()
    run_weight_conversion_test(
        MLPClass=MLP1D,
        hf_config=hf_config,
        state_dict=reference_model.state_dict(),
        tmp_path=tmp_path,
        mesh_row=mesh_row,
        reference_w1=reference_state_dict["gate_proj.weight"],
    )


@pytest.mark.parametrize(
    "MLPClass,module_path",
    [(NonExpert, "model.layers.0.mlp"), (SharedExpert, "model.layers.3.mlp.shared_experts")],
)
def test_convert_weights_for_dequantized_mlps(MLPClass, model_path, module_path, hf_config, tmp_path, mesh_row):
    state_dict = load_state_dict(model_path, module_path)
    run_weight_conversion_test(
        MLPClass=MLPClass,
        hf_config=hf_config,
        state_dict=state_dict,
        tmp_path=tmp_path,
        mesh_row=mesh_row,
        reference_w1=dequantize(
            state_dict["gate_proj.weight"],
            state_dict["gate_proj.weight_scale_inv"],
            block_shape=hf_config.quantization_config["weight_block_size"],
        ),
    )


def run_weight_conversion_test(MLPClass, hf_config, state_dict, tmp_path, reference_w1, mesh_row):
    # Convert the weights
    weight_config = MLPClass.convert_weights(hf_config, state_dict, tmp_path, mesh_row)

    # Verify weight_config structure
    assert "w1" in weight_config
    assert "w2" in weight_config
    assert "w3" in weight_config
    assert "input_tensor_b" in weight_config["w1"]
    assert "input_tensor_b" in weight_config["w2"]
    assert "input_tensor_b" in weight_config["w3"]

    # Verify files exist
    assert Path(weight_config["w1"]["input_tensor_b"]).exists()
    assert Path(weight_config["w2"]["input_tensor_b"]).exists()
    assert Path(weight_config["w3"]["input_tensor_b"]).exists()

    # Load and verify a weight
    w1_ttnn = ttnn.load_tensor(weight_config["w1"]["input_tensor_b"], device=mesh_row)
    w1_torch = ttnn.to_torch(
        w1_ttnn,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_row, dims=(-2, -1), mesh_shape=tuple(mesh_row.shape)),
    )

    # Weight should be transposed from PyTorch format
    assert w1_torch.shape == (1,) * (w1_torch.ndim - 2) + (reference_w1.shape[1], reference_w1.shape[0])

    # Verify the values match (accounting for transpose and bfloat8 conversion)
    passing, pcc_msg = comp_pcc(reference_w1.T, w1_torch, 0.99)
    assert passing, f"Weight conversion PCC failed: {pcc_msg}"

    # Cleanup
    ttnn.deallocate(w1_ttnn)


@pytest.mark.parametrize(
    "MLPClass,module_path",
    [
        (MLP1D, None),
        (NonExpert, "model.layers.0.mlp"),
        (SharedExpert, "model.layers.3.mlp.shared_experts"),
    ],
)
@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 32),
        ("prefill", 512),
        ("prefill", 2048),  # Test chunking
    ],
)
def test_forward_pass(
    MLPClass,
    module_path,
    mode,
    seq_len,
    hf_config,
    tmp_path,
    mesh_row,
    model_path,
):
    # Get the reference IO
    if not issubclass(MLPClass, MLP1DDequant):
        reference_model = DeepseekV3MLP(hf_config).eval()
        state_dict = reference_model.state_dict()

        torch_input = torch.randn(1, 1, seq_len, hf_config.hidden_size)
        reference_output = reference_model(torch_input)
    else:
        state_dict = load_state_dict(model_path, module_path)
        torch_input, reference_output = load_reference_io_tensors_for_module(mode, module_path, seq_len)
        torch_input.unsqueeze_(0)
        reference_output.unsqueeze_(0)

    # Generate module configs and state
    weight_config = MLPClass.convert_weights(hf_config, state_dict, tmp_path, mesh_row)
    model_config = get_model_config(MLPClass, mode, hf_config, mesh_row)
    model_state = MLPClass.create_state(hf_config, mesh_device=mesh_row)
    run_config = create_run_config(model_config, weight_config, model_state)

    # Convert input to TTNN
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_row,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_row),
        dtype=ttnn.bfloat16,
        memory_config=run_config["input_memory_config"],
        layout=ttnn.TILE_LAYOUT,
    )

    # TTNN forward pass
    tt_output = run_module_forward(MLPClass, mode, tt_input, run_config)

    # Verify output memory config matches expected
    expected_output_memory_config = run_config["output_memory_config"]
    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    # Convert output back to torch
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_row, dims=(-2, -1), mesh_shape=tuple(mesh_row.shape)),
    )

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    # Check PCC
    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.98)


if __name__ == "__main__":
    pytest.main([__file__])
