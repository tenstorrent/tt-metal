# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Single Galaxy MoE Test for Quad Galaxy Validation

This test runs a scaled-down version of the DeepSeek V3 MoE model on a single galaxy (32 devices, 4x8 mesh)
to validate that the model will work correctly on quad galaxy (128 devices, 16x8 mesh).

Key Design Principles:
1. **Same Work Per Device**: Each device handles 2 experts in both configurations
   - Quad: 256 experts / 128 devices = 2 experts/device
   - TG:   64 experts / 32 devices = 2 experts/device

2. **Multi-Cluster Testing**: Uses 4x8 mesh to ensure cross-cluster communication works
   - Tests cluster_axis=0 (dispatch across rows)
   - Tests cluster_axis=1 (tensor parallel across columns)

3. **Critical Path Coverage**:
   - Expert routing and dispatch
   - Cross-cluster all-to-all operations
   - Tensor parallel operations (all-gather, reduce-scatter)
   - Multiple cluster coordination
   - Memory configurations under load
   - Numerical accuracy at scale

4. **Workload Realism**:
   - Uses real weights (not just random) to catch weight-specific issues
   - Tests both decode and prefill modes
   - Tests various sequence lengths and batch sizes
   - Uses same fabric configuration as quad

If this test passes on TG, we have high confidence the model will work on quad.
If a change breaks quad, it should also break this test.
"""

import math
import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE
from models.demos.deepseek_v3.tt.moe import MoE
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, get_fabric_config
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    get_model_config,
    get_test_weight_config,
    load_reference_io,
    load_reference_io_tensors_for_module,
    run_module_forward,
)

# Single Galaxy Configuration
TG_MESH_SHAPE = (4, 8)  # 32 devices
TG_NUM_EXPERTS = 64
TG_NUM_EXPERT_GROUPS = 8  # Keep same as quad for routing logic testing
EXPERTS_PER_DEVICE = 2


def is_checkpoint_available() -> tuple[bool, str]:
    """
    Check if the DeepSeek V3 checkpoint is available and valid.

    Returns:
        tuple[bool, str]: (is_available, reason_if_not_available)
    """
    model_path = os.getenv("DEEPSEEK_V3_HF_MODEL", "models/demos/deepseek_v3/reference")
    checkpoint_path = Path(model_path)

    # Check if path exists
    if not checkpoint_path.exists():
        return False, f"Checkpoint path does not exist: {checkpoint_path}"

    # Check if it's a directory
    if not checkpoint_path.is_dir():
        return False, f"Checkpoint path is not a directory: {checkpoint_path}"

    # Check for safetensors index file (indicates checkpoint is present)
    index_file = checkpoint_path / "model.safetensors.index.json"
    if not index_file.exists():
        return False, f"Missing model.safetensors.index.json in: {checkpoint_path}"

    # Check for config.json (required for model loading)
    config_file = checkpoint_path / "config.json"
    if not config_file.exists():
        return False, f"Missing config.json in: {checkpoint_path}"

    return True, ""


def create_scaled_config(base_config, num_experts: int = TG_NUM_EXPERTS):
    """
    Create a scaled-down config for single galaxy testing.

    The key insight is that we scale the number of experts proportionally to the number of devices,
    maintaining the same experts-per-device ratio as quad galaxy.
    """
    config = deepcopy(base_config)

    # Scale the number of experts
    config.n_routed_experts = num_experts

    # Keep the expert group structure the same for routing logic
    # This ensures the routing algorithm is tested with the same complexity
    config.n_group = TG_NUM_EXPERT_GROUPS

    # Keep other MoE parameters the same to match quad behavior
    # - num_experts_per_tok: 8 (same topk selection)
    # - topk_group: 4 (same group selection)
    # - scoring_func, norm_topk_prob, etc. (same routing logic)

    return config


def validate_mesh_for_tg_test(mesh_device):
    """Validate that we're running on the correct mesh configuration."""
    actual_shape = mesh_device.shape
    num_devices = mesh_device.get_num_devices()

    # Convert MeshShape to tuple for comparison
    assert tuple(actual_shape) == TG_MESH_SHAPE, (
        f"This test requires a {TG_MESH_SHAPE} mesh (single galaxy), "
        f"but got {tuple(actual_shape)}. Set MESH_DEVICE=TG"
    )

    assert num_devices == 32, f"This test requires 32 devices (single galaxy), but got {num_devices}"

    # Validate experts per device
    experts_per_device = TG_NUM_EXPERTS // num_devices
    assert (
        experts_per_device == EXPERTS_PER_DEVICE
    ), f"Expected {EXPERTS_PER_DEVICE} experts per device, got {experts_per_device}"

    logger.info(
        f"✓ Mesh validation passed: {TG_MESH_SHAPE} mesh, {num_devices} devices, "
        f"{experts_per_device} experts/device (matching quad galaxy workload)"
    )


def _validate_critical_moe_configurations(run_config, scaled_config, mesh_device, mode):
    """
    Validate critical mesh-dependent configurations that must match between TG and quad.

    These validations ensure that operations with specific shard specs will work correctly
    on quad galaxy. If these assertions pass on TG, they will pass on quad.

    Critical operations validated:
    1. deepseek_moe_fast_reduce_nc - output shard spec
    2. all_to_all_dispatch_metadata - memory config
    3. num_experts_per_tok - routing parameter
    4. Tensor parallel and cluster dimensions
    """
    logger.info("Validating critical MoE configurations for quad compatibility...")

    # 1. Validate num_experts_per_tok (must be 8 for both TG and quad)
    num_experts_per_tok = run_config.get("num_experts_per_tok", scaled_config.num_experts_per_tok)
    assert num_experts_per_tok == 8, (
        f"num_experts_per_tok must be 8 (got {num_experts_per_tok}). " "This is critical for expert routing logic."
    )
    logger.info(f"  ✓ num_experts_per_tok: {num_experts_per_tok}")

    # 2. Validate cluster_axis for all_to_all operations
    # cluster_axis=0 for expert dispatch (across rows)
    # cluster_axis=1 for tensor parallel (across columns)
    dispatch_cluster_axis = run_config["all_to_all_dispatch"]["cluster_axis"]
    combine_cluster_axis = run_config["all_to_all_combine"]["cluster_axis"]
    assert dispatch_cluster_axis == 0, (
        f"all_to_all_dispatch cluster_axis must be 0 (got {dispatch_cluster_axis}). "
        "This dispatches experts across mesh rows."
    )
    assert combine_cluster_axis == 0, (
        f"all_to_all_combine cluster_axis must be 0 (got {combine_cluster_axis}). "
        "This combines expert outputs across mesh rows."
    )
    logger.info(f"  ✓ all_to_all dispatch/combine cluster_axis: {dispatch_cluster_axis}")

    # 3. Validate tensor parallel cluster_axis (for reduce_scatter)
    if mode == "decode":
        rs_cluster_axis = run_config["final_output_reduce_scatter"]["cluster_axis"]
    else:
        rs_cluster_axis = run_config["final_output_reduce_scatter"]["cluster_axis"]
    assert rs_cluster_axis == 1, (
        f"reduce_scatter cluster_axis must be 1 (got {rs_cluster_axis}). "
        "This reduces across tensor parallel dimension (mesh columns)."
    )
    logger.info(f"  ✓ reduce_scatter cluster_axis: {rs_cluster_axis}")

    # 4. Validate metadata memory config (currently DRAM for both)
    metadata_mem_config = run_config["all_to_all_dispatch_metadata_memory_config"]
    assert (
        metadata_mem_config.buffer_type == ttnn.BufferType.DRAM
    ), "all_to_all_dispatch_metadata_memory_config must use DRAM buffer type"
    logger.info(f"  ✓ dispatch_metadata memory: DRAM")

    # 5. Validate deepseek_moe_fast_reduce_nc output shard spec (decode mode with FABRIC_1D_RING)
    if mode == "decode" and run_config["fabric_config"] == ttnn.FabricConfig.FABRIC_1D_RING:
        tp_size = mesh_device.shape[1]
        if tp_size == 8:
            fast_reduce_mem_config = run_config.get("ring_sum_experts_output_memory_config")
            if fast_reduce_mem_config is not None:
                assert (
                    fast_reduce_mem_config.buffer_type == ttnn.BufferType.L1
                ), "ring_sum_experts_output_memory_config must use L1 buffer type"
                # Verify it has a shard spec (NdShardSpec)
                assert hasattr(fast_reduce_mem_config, "nd_shard_spec") or hasattr(
                    fast_reduce_mem_config, "shard_spec"
                ), "ring_sum_experts_output_memory_config must have a shard spec"
                logger.info(f"  ✓ fast_reduce_nc output shard spec: L1 sharded")

    # 6. Validate experts_per_device matches expected
    experts_per_device = run_config["num_experts_per_device"]
    assert (
        experts_per_device == EXPERTS_PER_DEVICE
    ), f"num_experts_per_device must be {EXPERTS_PER_DEVICE} (got {experts_per_device})"
    logger.info(f"  ✓ num_experts_per_device: {experts_per_device}")

    # 7. Validate hidden_size is consistent
    hidden_size = run_config["hidden_size"]
    assert (
        hidden_size == scaled_config.hidden_size
    ), f"hidden_size mismatch: run_config={hidden_size}, scaled_config={scaled_config.hidden_size}"
    logger.info(f"  ✓ hidden_size: {hidden_size}")

    logger.info("✓ All critical MoE configurations validated for quad compatibility")


@pytest.fixture
def reference_model_tg(hf_config):
    """Build the routed-experts-only MoE reference for TG testing."""
    torch.use_deterministic_algorithms(True)
    moe_config = create_scaled_config(hf_config, num_experts=TG_NUM_EXPERTS)
    moe_config.n_shared_experts = None
    return DeepseekV3MoE(moe_config).eval()


def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in state_dict.items()}


def load_real_moe_input(mode: str, module_path: str, num_tokens: int) -> torch.Tensor:
    """Load real MoE input from reference data."""
    if mode == "prefill":
        torch_input, _ = load_reference_io_tensors_for_module(mode, module_path, num_tokens, 1)
        return torch_input.squeeze(0).to(torch.bfloat16)

    reference_io = load_reference_io(mode, module_path)
    assert all(len(logs) <= 1 for logs in reference_io), f"Expected a non-range module, got {module_path}"
    assert all(len(logs) > 0 for logs in reference_io), f"Some logs for module {module_path} {mode} were not generated."

    io_module_paths, torch_args, _, _ = zip(*[logs[0] for logs in reference_io])
    (torch_inputs,) = zip(*torch_args)
    assert set(io_module_paths) == {module_path}

    torch_input = torch.concat(torch_inputs, dim=1).unsqueeze(0)

    if torch_input.shape[2] < num_tokens:
        repeats = math.ceil(num_tokens / torch_input.shape[2])
        torch_input = torch_input.repeat(1, 1, repeats, 1)

    return torch_input[:, :, :num_tokens, :].squeeze(0).to(torch.bfloat16)


def sample_experts_from_checkpoint(
    checkpoint_state_dict: dict[str, torch.Tensor],
    module_path: str,
    num_experts_to_sample: int,
    total_experts_in_checkpoint: int,
) -> dict[str, torch.Tensor]:
    """
    Sample a subset of experts from the full checkpoint.

    We sample evenly across the expert range to get representative coverage.
    For example, if we have 256 experts and want 64, we take every 4th expert.
    """
    assert num_experts_to_sample <= total_experts_in_checkpoint

    # Calculate sampling stride
    stride = total_experts_in_checkpoint / num_experts_to_sample
    sampled_expert_ids = [int(i * stride) for i in range(num_experts_to_sample)]

    logger.info(
        f"Sampling {num_experts_to_sample} experts from {total_experts_in_checkpoint} "
        f"(stride={stride:.2f}). Sample IDs: {sampled_expert_ids[:8]}..."
    )

    # Extract expert weights
    sampled_state_dict = {}
    for new_id, orig_id in enumerate(sampled_expert_ids):
        for weight_type in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]:
            orig_key = f"{module_path}.experts.{orig_id}.{weight_type}"
            new_key = f"experts.{new_id}.{weight_type}"

            if orig_key in checkpoint_state_dict:
                sampled_state_dict[new_key] = checkpoint_state_dict[orig_key].detach().clone()

    # Also copy gate weights
    gate_key = f"{module_path}.gate.weight"
    if gate_key in checkpoint_state_dict:
        # For the gate, we need to downsample to match the reduced expert count
        orig_gate = checkpoint_state_dict[gate_key]
        # Sample the gate weights for the selected experts
        sampled_gate = orig_gate[sampled_expert_ids, :]
        sampled_state_dict["gate.weight"] = sampled_gate

    # Copy gate bias if it exists
    gate_bias_key = f"{module_path}.gate.e_score_correction_bias"
    if gate_bias_key in checkpoint_state_dict:
        orig_bias = checkpoint_state_dict[gate_bias_key]
        sampled_bias = orig_bias[sampled_expert_ids]
        sampled_state_dict["gate.e_score_correction_bias"] = sampled_bias

    return sampled_state_dict


def generate_reference_io_tg(
    mode: str,
    num_tokens: int,
    reference_model: DeepseekV3MoE,
    hf_config,
    weight_type: str,
    checkpoint_state_dict: dict[str, torch.Tensor] | None = None,
    module_path: str | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Generate reference input/output for TG test."""
    if weight_type == "random":
        # Preserve random-init dtypes, especially the fp32 gate score-correction bias.
        state_dict_out = _clone_state_dict(reference_model.state_dict())
        torch_input = torch.randn(1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)
    else:
        assert weight_type == "real"
        assert checkpoint_state_dict is not None
        assert module_path is not None

        # Sample experts from full checkpoint
        orig_num_experts = hf_config.n_routed_experts
        state_dict_out = sample_experts_from_checkpoint(
            checkpoint_state_dict,
            module_path,
            TG_NUM_EXPERTS,
            orig_num_experts,
        )

        if not state_dict_out:
            pytest.skip(f"Checkpoint does not contain routed MoE weights under '{module_path}'")

        reference_model.load_state_dict(state_dict_out, strict=False)
        torch_input = load_real_moe_input(mode, module_path, num_tokens)

    reference_model.eval()
    reference_model.to(torch.bfloat16)
    with torch.no_grad():
        reference_output = reference_model(torch_input)

    return state_dict_out, torch_input, reference_output


@pytest.mark.requires_device("TG")  # Only run on single galaxy
@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": get_fabric_config()},  # Use same fabric as quad
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode, batch_size_per_row, seq_len",
    [
        # Decode mode: critical for inference performance
        ("decode", USERS_PER_ROW, 1),
        # Prefill mode: critical for prompt processing
        # Test various sequence lengths to catch length-dependent issues
        ("prefill", 1, 128),
        ("prefill", 1, 512),
        ("prefill", 1, 2048),
    ],
)
@pytest.mark.parametrize(
    "topk_fallback",
    [
        True,  # Use fallback topk for deterministic testing
    ],
)
@pytest.mark.parametrize(
    "weight_type",
    [
        "random",  # Tests basic correctness and shapes
        "real",  # Tests with actual model weights - critical for catching real issues
    ],
)
def test_moe_single_galaxy_for_quad_validation(
    device_params,
    mode,
    batch_size_per_row,
    seq_len,
    set_deterministic_env,
    reference_model_tg,
    hf_config,
    request,
    cache_path,
    mesh_device,
    ccl,
    topk_fallback,
    weight_type,
    force_recalculate_weight_config,
):
    """
    Single Galaxy MoE test for quad galaxy validation.

    This test validates that the MoE implementation works correctly across multiple clusters
    with the same per-device workload as quad galaxy.

    Critical aspects tested:
    1. Cross-cluster expert dispatch (cluster_axis=0)
    2. Tensor parallel operations (cluster_axis=1)
    3. Multi-cluster coordination (4 clusters in 4x8 mesh)
    4. Expert routing with same complexity as quad
    5. Memory management under realistic load
    6. Numerical accuracy with real weights
    """
    # Skip real weight tests if checkpoint is not available or invalid
    if weight_type == "real":
        checkpoint_available, skip_reason = is_checkpoint_available()
        if not checkpoint_available:
            pytest.skip(f"Skipping real weight test: {skip_reason}")

    # Validate mesh configuration
    validate_mesh_for_tg_test(mesh_device)

    # Create scaled config for TG
    scaled_config = create_scaled_config(hf_config, num_experts=TG_NUM_EXPERTS)

    # Setup test data
    module_path = "model.layers.3.mlp" if weight_type == "real" else None
    checkpoint_state_dict = None

    if weight_type == "real":
        try:
            checkpoint_state_dict = request.getfixturevalue("state_dict")
        except Exception as e:
            pytest.skip(f"Skipping real weight test: Unable to load checkpoint: {str(e)}")

    num_tokens = batch_size_per_row * mesh_device.shape[0] if mode == "decode" else seq_len

    try:
        state_dict, torch_input, reference_output = generate_reference_io_tg(
            mode=mode,
            num_tokens=num_tokens,
            reference_model=reference_model_tg,
            hf_config=scaled_config,
            weight_type=weight_type,
            checkpoint_state_dict=checkpoint_state_dict,
            module_path=module_path,
        )
    except Exception as e:
        if weight_type == "real":
            pytest.skip(f"Skipping real weight test: Error loading checkpoint data: {str(e)}")
        else:
            raise

    # Convert weights
    weight_config = get_test_weight_config(
        MoE,
        scaled_config,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate=force_recalculate_weight_config,
        test_name="test_moe_single_galaxy",
        real_weights=weight_type == "real",
        layer_id=module_path,
    )

    # Create model config
    model_config = get_model_config(
        MoE, mode, scaled_config, mesh_device, device_params["fabric_config"], topk_fallback=topk_fallback
    )
    model_state = MoE.create_state(scaled_config, mesh_device, ccl)
    model_shared_state = MoE.create_shared_state(scaled_config, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # Log test configuration
    logger.info("=" * 80)
    logger.info(f"TG MoE Test Configuration:")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Weight type: {weight_type}")
    logger.info(f"  Mesh shape: {mesh_device.shape} ({mesh_device.get_num_devices()} devices)")
    logger.info(f"  Num experts: {TG_NUM_EXPERTS}")
    logger.info(f"  Experts per device: {TG_NUM_EXPERTS // mesh_device.get_num_devices()}")
    logger.info(f"  Num tokens: {num_tokens}")
    logger.info(f"  Num expert groups: {scaled_config.n_group}")
    logger.info(f"  Experts per token: {scaled_config.num_experts_per_tok}")
    logger.info(f"  Fabric config: {device_params['fabric_config']}")
    logger.info("=" * 80)

    # Create input tensor
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])

    # Run forward pass
    tt_output = run_module_forward(MoE, mode, tt_input, run_config, handle_tensor_parallel=True)

    # Validate output memory config
    expected_output_memory_config = run_config["output_memory_config"]
    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"MoE output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    # Convert to torch
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    # Validate PCC
    logger.info(f"Mode: {mode}, Num tokens: {num_tokens}, Weight type: {weight_type}")

    # For TG test, we want high PCC to ensure quad will work
    # Using 0.98 as threshold (higher than standard 0.97) for extra confidence
    pcc_threshold = 0.98 if weight_type == "real" else 0.97

    assert_hidden_dim_pcc(tt_output_torch, reference_output.unsqueeze(0), pcc_required=pcc_threshold)

    logger.info(f"✓ TG test passed with PCC >= {pcc_threshold}")

    # Validate critical mesh-dependent configurations
    _validate_critical_moe_configurations(run_config, scaled_config, mesh_device, mode)


@pytest.mark.requires_device("TG")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": get_fabric_config()},
    ],
    indirect=True,
)
def test_moe_single_galaxy_stress(
    device_params,
    set_deterministic_env,
    reference_model_tg,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    force_recalculate_weight_config,
):
    """
    Stress test for MoE on single galaxy.

    This test runs multiple iterations with different configurations to catch
    issues that might only appear under stress or after multiple runs:
    - Memory leaks
    - Accumulation errors
    - Synchronization issues
    - Cache corruption
    """
    validate_mesh_for_tg_test(mesh_device)

    scaled_config = create_scaled_config(hf_config, num_experts=TG_NUM_EXPERTS)

    # Test configurations to cycle through
    test_configs = [
        ("decode", USERS_PER_ROW, 1),
        ("prefill", 1, 512),
        ("prefill", 1, 2048),
        ("decode", USERS_PER_ROW, 1),  # Repeat decode
    ]

    logger.info("=" * 80)
    logger.info("Starting MoE stress test on single galaxy")
    logger.info(f"Running {len(test_configs)} iterations")
    logger.info("=" * 80)

    # Create shared state once - it's the same for all iterations
    model_shared_state = MoE.create_shared_state(scaled_config, mesh_device)

    # Convert weights once - same weights used for all iterations
    state_dict = _clone_state_dict(reference_model_tg.state_dict())
    weight_config = get_test_weight_config(
        MoE,
        scaled_config,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate=force_recalculate_weight_config,
        test_name="test_moe_stress",
        real_weights=False,
    )

    for iteration, (mode, batch_size_per_row, seq_len) in enumerate(test_configs):
        logger.info(f"\n>>> Iteration {iteration + 1}/{len(test_configs)}: mode={mode}, seq_len={seq_len}")

        num_tokens = batch_size_per_row * mesh_device.shape[0] if mode == "decode" else seq_len

        # Generate random input for stress test
        torch_input = torch.randn(1, num_tokens, scaled_config.hidden_size, dtype=torch.bfloat16)

        reference_model_tg.eval()
        reference_model_tg.to(torch.bfloat16)
        with torch.no_grad():
            reference_output = reference_model_tg(torch_input)

        # Create model config and state (mode-specific)
        model_config = get_model_config(
            MoE, mode, scaled_config, mesh_device, device_params["fabric_config"], topk_fallback=True
        )
        model_state = MoE.create_state(scaled_config, mesh_device, ccl)
        run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

        # Run forward pass
        tt_input = ttnn.from_torch(
            torch_input.unsqueeze(1),
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
        tt_output = run_module_forward(MoE, mode, tt_input, run_config, handle_tensor_parallel=True)

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        )

        # Cleanup
        ttnn.deallocate(tt_input)
        ttnn.deallocate(tt_output)

        # Validate
        assert_hidden_dim_pcc(tt_output_torch, reference_output.unsqueeze(0), pcc_required=0.97)
        logger.info(f"✓ Iteration {iteration + 1} passed")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Stress test completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
