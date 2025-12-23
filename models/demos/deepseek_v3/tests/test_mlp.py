# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP
from models.demos.deepseek_v3.tt.mlp.mlp import MLP
from models.demos.deepseek_v3.tt.mlp.mlp_dequant import MLPDequant
from models.demos.deepseek_v3.tt.mlp.non_expert import NonExpert
from models.demos.deepseek_v3.tt.mlp.shared_expert import SharedExpert
from models.demos.deepseek_v3.utils.config_helpers import dequantize, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config, load_weight
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    get_model_config,
    get_test_weight_config,
    load_reference_io_tensors_for_module,
    run_module_forward,
)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_convert_weights_for_non_dequantized_mlp(hf_config, tmp_path, mesh_device):
    reference_model = DeepseekV3MLP(hf_config).eval()
    reference_state_dict = reference_model.to(torch.bfloat16).state_dict()
    run_weight_conversion_test(
        MLPClass=MLP,
        hf_config=hf_config,
        state_dict=reference_model.state_dict(),
        tmp_path=tmp_path
        / "mesh_8x8",  # TODO: dummy mesh shape required until convert_weights no longer relies on this for parsing the absolutem filepaths
        mesh_device=mesh_device,
        reference_w1=reference_state_dict["gate_proj.weight"],
    )


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "MLPClass,module_path",
    [(NonExpert, "model.layers.0.mlp"), (SharedExpert, "model.layers.3.mlp.shared_experts")],
)
def test_convert_weights_for_dequantized_mlps(MLPClass, module_path, hf_config, tmp_path, mesh_device, state_dict):
    state_dict = sub_state_dict(state_dict, module_path + ".")
    run_weight_conversion_test(
        MLPClass=MLPClass,
        hf_config=hf_config,
        state_dict=state_dict,
        tmp_path=tmp_path
        / "mesh_8x8",  # TODO: dummy mesh shape required until convert_weights no longer relies on this for parsing the absolutem filepaths
        mesh_device=mesh_device,
        reference_w1=dequantize(
            state_dict["gate_proj.weight"],
            state_dict["gate_proj.weight_scale_inv"],
            block_shape=hf_config.quantization_config["weight_block_size"],
        ),
    )


def run_weight_conversion_test(MLPClass, hf_config, state_dict, tmp_path, reference_w1, mesh_device):
    num_module_layers, _ = mesh_device.shape

    # Convert the weights
    weight_config = MLPClass.convert_weights(
        hf_config, [state_dict] + [None] * (num_module_layers - 1), tmp_path, mesh_device
    )

    # Verify weight_config structure
    assert "w1" in weight_config
    assert "w2" in weight_config
    assert "w3" in weight_config
    assert "input_tensor_b" in weight_config["w1"]
    assert "input_tensor_b" in weight_config["w2"]
    assert "input_tensor_b" in weight_config["w3"]

    # # Verify files exist # TODO: bring regular tensor saving back once Issue #26763 is resolved
    # assert Path(weight_config["w1"]["input_tensor_b"]).exists()
    # assert Path(weight_config["w2"]["input_tensor_b"]).exists()
    # assert Path(weight_config["w3"]["input_tensor_b"]).exists()

    # Make the path absolute - this is required since load_weight expects an absolute path
    weight_config["w1"]["input_tensor_b"].path = tmp_path / weight_config["w1"]["input_tensor_b"].path

    # Load and verify a weight
    w1_ttnn = load_weight(weight_config["w1"]["input_tensor_b"], device=mesh_device)
    w1_ttnn = ttnn.unsqueeze(w1_ttnn, 0)  # Unsqueeze to collect shards on a separate dim
    w1_torch = ttnn.to_torch(
        w1_ttnn,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    # Weight should be transposed from PyTorch format
    assert w1_torch.shape == (
        num_module_layers,
        *[1 for _ in range(w1_torch.ndim - 3)],
        reference_w1.shape[1],
        reference_w1.shape[0],
    )

    # Verify the values match (accounting for transpose and bfloat8 conversion)
    passing, pcc = comp_pcc(reference_w1.T, w1_torch[0], 0.99)
    logger.info(f"PCC: {pcc}")
    assert passing, f"Weight conversion PCC failed: {pcc}"

    # Cleanup
    ttnn.deallocate(w1_ttnn)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "MLPClass,module_path",
    [
        (MLP, None),
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
    mesh_device,
    ccl,
    model_path,
    tmp_path,
    cache_path,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict,
    test_timer,  # Added timing fixture
    test_cache_manager,  # Unified cache fixture
):
    num_module_layers, _ = mesh_device.shape

    with test_timer.time("1. reference_io_setup"):
        # Get the reference IO
        if test_cache_manager.use_random_weights or not issubclass(MLPClass, MLPDequant):
            # Fast path: random weights (cached)
            cache_key = f"DeepseekV3MLP:random"
            reference_model = test_cache_manager.get_reference_model(DeepseekV3MLP, hf_config, None, cache_key)
            state_dict = reference_model.to(torch.bfloat16).state_dict()
            torch_input = torch.randn(num_module_layers, 1, seq_len, hf_config.hidden_size)

            reference_model = reference_model.to(torch.float32)
            reference_output = reference_model(torch_input)
        else:
            # Real weights path: use cached dequantized weights
            dequantized = test_cache_manager.get_dequantized_state_dict(module_path, state_dict, hf_config)
            torch_input, reference_output = load_reference_io_tensors_for_module(
                mode, module_path, seq_len, num_module_layers
            )
            state_dict = dequantized

    with test_timer.time("2. get_test_weight_config"):
        # Generate module configs and state
        weight_config = get_test_weight_config(
            MLPClass,
            hf_config,
            (state_dict,) * num_module_layers,
            cache_path,
            mesh_device,
            force_recalculate_weight_config,
        )

    with test_timer.time("3. get_model_config"):
        model_config = get_model_config(MLPClass, mode, hf_config, mesh_device)

    with test_timer.time("4. create_state"):
        model_state = MLPClass.create_state(hf_config, mesh_device, ccl)

    with test_timer.time("5. create_run_config"):
        cached_ttnn_weights = test_cache_manager.ttnn_weight_tensors if test_cache_manager else None
        run_config = create_run_config(
            model_config, weight_config, model_state, cached_ttnn_weights=cached_ttnn_weights
        )

    with test_timer.time("6. ttnn_from_torch"):
        # Convert input to TTNN
        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, (0, -1)),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

    with test_timer.time("7. ttnn_forward"):
        # TTNN forward pass
        tt_output = run_module_forward(MLPClass, mode, tt_input, run_config)

    with test_timer.time("8. verify_memory_config"):
        # Verify output memory config matches expected
        expected_output_memory_config = run_config["output_memory_config"]
        actual_output_memory_config = tt_output.memory_config()
        assert (
            actual_output_memory_config == expected_output_memory_config
        ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    with test_timer.time("9. ttnn_to_torch"):
        # Convert output back to torch
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=tuple(mesh_device.shape)),
        )

    with test_timer.time("10. cleanup"):
        # Cleanup
        ttnn.deallocate(tt_input)
        ttnn.deallocate(tt_output)

    with test_timer.time("11. pcc_check"):
        # Check PCC
        assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.975)


if __name__ == "__main__":
    pytest.main([__file__])
