# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.conftest import PREFILL_SEQ_LENS
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

TEST_CHECK_ITERS = 100
CI_ACTIVE = os.getenv("CI") == "true"
_CI_SKIP_MARK = pytest.mark.skipif(
    CI_ACTIVE,
    reason="CI runs traced coverage only.",
)


# TODO: Doesn't work on multi-host - we should figure out why
@pytest.mark.requires_device(["TG"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_convert_weights_for_non_dequantized_mlp(hf_config, tmp_path, mesh_device):
    # Add a skip for mesh device shape 8x8 due to known issue https://github.com/tenstorrent/tt-metal/issues/35375
    if tuple(mesh_device.shape) == (8, 8):
        pytest.skip(
            "Skipping test for mesh device shape 8x8 due to known issue https://github.com/tenstorrent/tt-metal/issues/35375"
        )
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


# TODO: Doesn't work on multi-host - we should figure out why
@pytest.mark.requires_device(["TG"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "MLPClass,module_path",
    [(NonExpert, "model.layers.0.mlp"), (SharedExpert, "model.layers.3.mlp.shared_experts")],
)
def test_convert_weights_for_dequantized_mlps(MLPClass, module_path, hf_config, tmp_path, mesh_device, state_dict):
    if tuple(mesh_device.shape) == (8, 8):
        pytest.skip(
            "Skipping test for mesh device shape 8x8 due to known issue https://github.com/tenstorrent/tt-metal/issues/35375"
        )
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
    if tuple(mesh_device.shape) == (8, 8):
        pytest.skip(
            "Skipping test for mesh device shape 8x8 due to known issue https://github.com/tenstorrent/tt-metal/issues/35375"
        )
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
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape),
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


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 10485760}],
    indirect=True,
)
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
    ]
    + [
        ("prefill", seq_len)
        if seq_len == 128
        else pytest.param(
            "prefill",
            seq_len,
            marks=pytest.mark.skipif(
                CI_ACTIVE,
                reason=(
                    f"Skipping prefilling with seq_len={seq_len} since this would cause us to exceed our available CI workload time"
                ),
            ),
        )
        for seq_len in PREFILL_SEQ_LENS
    ],
)
@pytest.mark.parametrize(
    "trace_mode",
    [
        pytest.param(False, marks=_CI_SKIP_MARK, id="eager"),
        pytest.param(True, id="tracing"),
    ],
)
@pytest.mark.parametrize(
    "use_real_weights",
    [
        pytest.param(True, id="real_weights"),
        pytest.param(False, marks=_CI_SKIP_MARK, id="random_weights"),
    ],
)
def test_forward_pass(
    MLPClass,
    module_path,
    mode,
    seq_len,
    trace_mode,
    use_real_weights,
    hf_config,
    mesh_device,
    ccl,
    model_path,
    tmp_path,
    cache_path,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict,
):
    if trace_mode and mode != "decode":
        pytest.skip("Tracing is only supported for decode mode.")

    num_module_layers, _ = mesh_device.shape

    # Get the reference IO
    if use_real_weights:
        if not issubclass(MLPClass, MLPDequant):
            reference_model = DeepseekV3MLP(hf_config).eval()
            state_dict = reference_model.to(torch.bfloat16).state_dict()
            torch_input = torch.randn(num_module_layers, 1, seq_len, hf_config.hidden_size)

            reference_model = reference_model.to(torch.float32)
            reference_output = reference_model(torch_input)
        else:
            state_dict = sub_state_dict(state_dict, module_path + ".")
            torch_input, reference_output = load_reference_io_tensors_for_module(
                mode, module_path, seq_len, num_module_layers
            )
    else:
        reference_model = DeepseekV3MLP(hf_config).eval()
        torch_input = torch.randn(num_module_layers, 1, seq_len, hf_config.hidden_size)
        random_state_dict = {k: torch.randn_like(v) for k, v in reference_model.state_dict().items()}
        if issubclass(MLPClass, MLPDequant):
            from models.demos.deepseek_v3.utils.test_utils import add_inv_scale_to_state_dict, dequantize_state_dict

            state_dict = add_inv_scale_to_state_dict(
                random_state_dict,
                block_shape=hf_config.quantization_config["weight_block_size"],
            )
            reference_model.load_state_dict(dequantize_state_dict(state_dict, hf_config, dtype=torch.float32))
        else:
            state_dict = {k: v.to(torch.bfloat16) for k, v in random_state_dict.items()}
            reference_model.load_state_dict({k: v.to(torch.float32) for k, v in random_state_dict.items()})
        reference_model = reference_model.to(torch.float32)
        reference_output = reference_model(torch_input)

    # Generate module configs and state
    weight_cache_root = cache_path if use_real_weights else cache_path / "random_weights"
    weight_config = get_test_weight_config(
        MLPClass,
        hf_config,
        (state_dict,) * num_module_layers,
        weight_cache_root,
        mesh_device,
        force_recalculate_weight_config or not use_real_weights,
    )
    model_config = get_model_config(MLPClass, mode, hf_config, mesh_device)
    model_state = MLPClass.create_state(hf_config, mesh_device, ccl)
    run_config = create_run_config(model_config, weight_config, model_state)

    # Convert input to TTNN
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, (0, -1)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    def check_outputs(tt_output: ttnn.Tensor) -> None:
        expected_output_memory_config = run_config["output_memory_config"]
        actual_output_memory_config = tt_output.memory_config()
        assert (
            actual_output_memory_config == expected_output_memory_config
        ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape),
        )
        assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.975)

    if trace_mode:
        tt_output = run_module_forward(MLPClass, mode, tt_input, run_config)
        ttnn.synchronize_device(mesh_device)
        check_outputs(tt_output)
        ttnn.deallocate(tt_output)

        # Reset CCL semaphore counters before trace capture
        ccl.reset_sem_counters()

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace_output = run_module_forward(MLPClass, mode, tt_input, run_config)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        for _ in range(TEST_CHECK_ITERS - 1):
            ttnn.execute_trace(mesh_device, trace_id, blocking=True)
        ttnn.synchronize_device(mesh_device)

        check_outputs(trace_output)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(trace_output)
    else:
        for iter_idx in range(TEST_CHECK_ITERS):
            tt_output = run_module_forward(MLPClass, mode, tt_input, run_config)
            ttnn.synchronize_device(mesh_device)
            if iter_idx in (0, TEST_CHECK_ITERS - 1):
                check_outputs(tt_output)
            ttnn.deallocate(tt_output)

    ttnn.deallocate(tt_input)


if __name__ == "__main__":
    pytest.main([__file__])
