# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test for MoEBlock supporting both DeepSeek and GPT-OSS backends.
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

# Import fixtures from DeepSeek conftest
from models.demos.deepseek_v3.conftest import *  # noqa: F401,F403
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE
from models.demos.deepseek_v3.tt.moe import MoE  # This now points to MoEBlock
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    assert_hidden_dim_pcc,
    get_model_config,
    get_test_weight_config,
    run_module_forward,
)

# GPT-OSS imports for backend
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tt.ccl import CCLManager as GPTOSSCCLManager


@pytest.fixture
def reference_model(hf_config):
    """Get the actual DeepSeek MLP model using local implementation."""
    torch.use_deterministic_algorithms(True)
    # Note : Running Reference MoE without shared experts
    hf_config.n_shared_experts = None
    return DeepseekV3MoE(hf_config).eval()


@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "backend,device_params",
    [
        ("deepseek", {"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
        pytest.param(
            "gptoss",
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},  # GPT-OSS requires FABRIC_1D_RING
            marks=pytest.mark.skipif(
                not os.path.exists("/data/tt_dnn-models/openai/gpt-oss-120b"),
                reason="GPT-OSS model not available",
            ),
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize(
    "mode,num_tokens",
    [
        ("decode", 128),
        ("prefill", 128),
    ],
)
@pytest.mark.parametrize(
    "topk_fallback",
    [
        True,
    ],
)
def test_forward_pass(
    backend,
    mode,
    num_tokens,
    set_deterministic_env,
    reference_model,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    topk_fallback,
):
    """Test forward pass for both DeepSeek and GPT-OSS backends."""

    # Backend-specific setup
    if backend == "deepseek":
        # Get state dict from actual model - pass directly to convert_weights
        state_dict = add_inv_scale_to_state_dict(
            reference_model.state_dict(),
            block_shape=hf_config.quantization_config["weight_block_size"],
        )
        test_name_suffix = "deepseek"
        expected_pcc_decode = 0.99  # DeepSeek expected PCC
        expected_pcc_prefill = 0.99
        # Use the provided cache_path for DeepSeek
        test_cache_path = cache_path
    else:  # gptoss
        logger.info("Setting up GPT-OSS backend test with real weights")

        # Use a different cache path for GPT-OSS tests with unique suffix
        import tempfile

        gptoss_cache = (
            Path(os.environ.get("TT_CACHE_PATH", tempfile.gettempdir())) / f"gptoss_moe_test_cache_{os.getpid()}"
        )
        gptoss_cache.mkdir(parents=True, exist_ok=True)
        test_cache_path = gptoss_cache

        # Load GPT-OSS model with RANDOM weights (matching test_modules.py approach)
        from transformers import AutoConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer

        # Load real config from disk
        MODEL_PATH = "/data/tt_dnn-models/openai/gpt-oss-120b"
        pt_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

        # Verify GPT-OSS constraint: num_experts must be divisible by num_devices
        num_devices = mesh_device.get_num_devices()
        assert pt_config.num_local_experts % num_devices == 0, (
            f"GPT-OSS ThroughputExperts requires num_experts ({pt_config.num_local_experts}) "
            f"to be divisible by num_devices ({num_devices})"
        )

        # Create reference layer
        layer_idx = 0
        reference_layer = GptOssDecoderLayer(pt_config, layer_idx=layer_idx)

        # Initialize with RANDOM weights (matching test_modules.py)
        # Use a fixed seed for reproducibility between reference and TT
        torch.manual_seed(42)
        with torch.no_grad():
            for name, param in reference_layer.named_parameters():
                if any(proj in name for proj in ["router", "experts", "sinks"]):
                    param.data.normal_(0, 1)

        reference_layer.eval()
        reference_layer = reference_layer.to(torch.bfloat16)

        # Get the SAME state dict for both reference and TT model
        # This is critical - we must use the same weights!
        full_layer_state = reference_layer.state_dict()

        # Use the reference MLP for comparison
        reference_model = reference_layer.mlp

        # Get state dict with THE SAME random weights for TT model
        from models.demos.gpt_oss.utils.substate import substate

        state_dict = substate(full_layer_state, "mlp")

        # Replace hf_config with pt_config for GPT-OSS backend
        hf_config = pt_config
        # Add extra attributes needed by our code
        hf_config.moe_intermediate_size = hf_config.intermediate_size  # Use the actual value
        hf_config.n_routed_experts = hf_config.num_local_experts
        hf_config.swiglu_limit = None
        hf_config.quantization_config = None
        hf_config.v_head_dim = 128
        hf_config.qk_nope_head_dim = 128
        hf_config.qk_rope_head_dim = 64

        test_name_suffix = "gptoss"
        # With real weights and real reference, we should match the reference test PCC
        expected_pcc_decode = 0.91  # Should match reference test_modules.py
        expected_pcc_prefill = 0.91

    # Create input tensor
    # Shape depends on mode and backend:
    # - DeepSeek: [batch=1, seq_len=num_tokens, hidden_size] for both modes
    # - GPT-OSS decode: [batch=num_tokens, seq_len=1, hidden_size] (high throughput)
    # - GPT-OSS prefill: [batch=1, seq_len=num_tokens, hidden_size]
    if backend == "gptoss" and mode == "decode":
        # GPT-OSS decode uses batch dimension for parallelism
        torch_input = torch.randn(num_tokens, 1, hf_config.hidden_size, dtype=torch.bfloat16)
    else:
        # DeepSeek (all modes) and GPT-OSS prefill use standard shape
        torch_input = torch.randn(1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)

    # Reference forward pass
    if reference_model is not None:
        reference_model.eval()
        reference_model.to(torch.bfloat16)
        with torch.no_grad():
            if backend == "deepseek":
                reference_output = reference_model(torch_input)
            else:  # gptoss
                # For GPT-OSS decode, input shape is [batch, 1, hidden]
                # But PyTorch model expects [batch, seq, hidden]
                # Reshape input for reference model
                if mode == "decode":
                    # Reshape from [128, 1, 2880] to [1, 128, 2880] for reference
                    ref_input = torch_input.transpose(0, 1).reshape(1, num_tokens, hf_config.hidden_size)
                else:
                    ref_input = torch_input

                # GPT-OSS MLP returns (output, routing_scores)
                reference_output, _ = reference_model(ref_input)
    else:
        # Fallback for when reference model isn't available
        reference_output = torch.randn(1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)

    # Use the standard test utility to get weight config
    # For now, we need to ensure the backend is properly passed through
    # Since get_test_weight_config doesn't support backend parameter,
    # we'll use it for DeepSeek and handle GPT-OSS separately
    if backend == "deepseek":
        # Use standard weight config for DeepSeek
        weight_config = get_test_weight_config(
            MoE,
            hf_config,
            (state_dict,),
            test_cache_path,
            mesh_device,
            force_recalculate=False,
            test_name=f"test_moe_block_{test_name_suffix}",
            real_weights=False,
        )
    else:  # gptoss
        # For GPT-OSS, we need to handle weight conversion differently
        # Create a wrapper to pass backend parameter
        mesh_shape = f"mesh_{mesh_device.shape[0]}x{mesh_device.shape[1]}"
        test_cache_subdir = test_cache_path / mesh_shape / f"test_moe_block_{test_name_suffix}"
        test_cache_subdir.mkdir(parents=True, exist_ok=True)

        # Create a custom convert_weights wrapper that includes backend
        original_method = MoE.convert_weights

        def convert_with_backend(hf_config, state_dicts, output_path, mesh_device):
            return original_method(hf_config, state_dicts, output_path, mesh_device, backend="gptoss")

        # Temporarily replace the method
        MoE.convert_weights = convert_with_backend

        try:
            weight_config = get_test_weight_config(
                MoE,
                hf_config,
                (state_dict,),
                test_cache_path,
                mesh_device,
                force_recalculate=True,  # Force recalculation for GPT-OSS
                test_name=f"test_moe_block_{test_name_suffix}",
                real_weights=False,
            )
        finally:
            # Always restore the original method
            MoE.convert_weights = original_method

    # Generate appropriate config with backend parameter
    model_config = get_model_config(
        MoE,
        mode,
        hf_config,
        mesh_device,
        topk_fallback=topk_fallback,
        backend=backend,  # This gets passed through to decode_model_config
    )

    # Create CCL manager and mesh config - for GPT-OSS, we need specific setup
    if backend == "gptoss":
        # Create GPT-OSS specific CCL manager
        num_links = 4 if mesh_device.shape[0] > 1 else 1
        ccl_to_use = GPTOSSCCLManager(mesh_device, num_links=num_links)

        # Create mesh config required by ThroughputExperts
        mesh_config = MeshConfig(
            mesh_shape=mesh_device.shape,
            decode=ModeConfig(tp=8, ep=4, sp=1),
            prefill=ModeConfig(tp=8, ep=1, sp=4),
            tp_axis=1,
        )
    else:
        # Use the provided CCL for DeepSeek
        ccl_to_use = ccl
        mesh_config = None  # DeepSeek doesn't use mesh_config

    # Create a new model state with CCL
    model_state = MoE.create_state(hf_config, mesh_device, ccl_to_use)

    # Create a new model shared state with backend parameter
    model_shared_state = MoE.create_shared_state(hf_config, mesh_device, backend=backend)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # For GPT-OSS backend, inject the state dict and mesh_config into the run config
    # since we don't convert weights upfront
    if backend == "gptoss":
        from models.demos.gpt_oss.utils.substate import substate

        router_state_dict = substate(state_dict, "router")
        experts_state_dict = substate(state_dict, "experts")

        # Add the state dicts to the config so they can be used at runtime
        run_config["moe_experts"]["router_state_dict"] = router_state_dict
        run_config["moe_experts"]["experts_state_dict"] = experts_state_dict
        run_config["moe_experts"]["mesh_config"] = mesh_config
        run_config["moe_experts"]["ccl_manager"] = ccl_to_use
        # For testing, use None for cache_path (matches reference test_modules.py)
        run_config["moe_experts"]["cache_path"] = None

    # Convert input to TTNN
    # DeepSeek: TP-sharded (DP=4, TP=8)
    # GPT-OSS: Row-sharded only on batch dimension
    if backend == "deepseek":
        # Shard on both batch and hidden dimensions for DeepSeek
        tt_input = ttnn.from_torch(
            torch_input.unsqueeze(1),
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
    else:  # gptoss
        # GPT-OSS: Row-sharded only for decode (batch > 1), replicated for prefill (batch = 1)
        # Following GPT-OSS reference test pattern from run_full_mlp_pipeline
        if mode == "decode":
            # Decode: use row sharding since batch=128
            is_row_sharded = True
            mesh_mapper = ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        else:
            # Prefill: replicate since batch=1
            is_row_sharded = False
            mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

        tt_input = ttnn.from_torch(
            torch_input.unsqueeze(1),  # Make 4D: [batch, 1, seq, hidden]
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            dtype=ttnn.bfloat8_b,  # Use BFLOAT8_B for GPT-OSS (matches reference test)
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

    # Convert to expected memory config
    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])

    # Run forward
    # DeepSeek uses tensor parallel (needs all_gather/reduce_scatter)
    # GPT-OSS doesn't use tensor parallel (handles all_reduce internally)
    handle_tensor_parallel = backend == "deepseek"
    tt_output = run_module_forward(MoE, mode, tt_input, run_config, handle_tensor_parallel=handle_tensor_parallel)

    # Verify memory configuration
    expected_output_memory_config = run_config["output_memory_config"]
    actual_output_memory_config = tt_output.memory_config()

    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    # Debug: Log output shape before conversion
    logger.info(f"TT output shape before conversion: {tt_output.shape}")

    # Convert back to PyTorch
    # Both backends need mesh_composer but with different dimensions
    if backend == "gptoss":
        # GPT-OSS ThroughputExperts output is distributed: [1, 1, batch_per_device, hidden]
        # The output is row-sharded: each device in a row has part of the batch
        # Use the same conversion as successful standalone test
        tt_output_torch = ttnn.to_torch(
            tt_output,
            # Use dims=(0, -1) as in the working standalone test
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=tuple(mesh_device.shape)),
        )
        logger.info(f"GPT-OSS output converted, shape: {tt_output_torch.shape}")

        # The output shape is [4, 1, 32, 23040] which needs proper reshaping
        # This is [mesh_rows, 1, batch_per_row, hidden*mesh_cols]
        # We need to reshape to [1, batch_total, hidden]
        # First, extract the correct hidden dimension by dividing by mesh columns
        mesh_rows, mesh_cols = mesh_device.shape
        if len(tt_output_torch.shape) == 4:
            # Shape is [mesh_rows, 1, batch_per_row, hidden*mesh_cols]
            _, _, batch_per_row, hidden_concat = tt_output_torch.shape
            hidden_size = hidden_concat // mesh_cols  # Divide by number of columns
            total_batch = batch_per_row * mesh_rows  # Total batch size

            # Reshape to get correct dimensions
            # First, take only the first column's worth of hidden dims
            tt_output_torch = tt_output_torch[:, :, :, :hidden_size]  # [4, 1, 32, 2880]

            # Then reshape to combine batch dimensions
            tt_output_torch = tt_output_torch.permute(1, 0, 2, 3)  # [1, 4, 32, 2880]
            tt_output_torch = tt_output_torch.reshape(1, total_batch, hidden_size)  # [1, 128, 2880]

            logger.info(f"GPT-OSS output reshaped to: {tt_output_torch.shape}")
            # Note: For GPT-OSS decode, we keep the shape as [1, 128, 2880] to match reference
    else:
        # DeepSeek uses TP sharding on both dims
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        )
        # Remove batch dimension and validate
        tt_output_torch = tt_output_torch.squeeze(1)

    logger.info(f"Backend: {backend}, Mode: {mode}, Num tokens: {num_tokens}")

    # For backends with reference models, do full PCC check
    if reference_model is not None:
        # Use the appropriate PCC threshold for each backend and mode
        if mode == "decode":
            pcc_threshold = expected_pcc_decode
        else:
            pcc_threshold = expected_pcc_prefill

        assert_hidden_dim_pcc(
            reference_output,
            tt_output_torch,
            pcc_required=pcc_threshold,
        )

    # Also log the actual PCC value for debugging
    from models.common.utility_functions import comp_pcc

    _, pcc_value = comp_pcc(tt_output_torch, reference_output)
    logger.info(f"PCC: {pcc_value}")

    # Verify PCC meets expected threshold for each backend
    if mode == "decode":
        assert pcc_value >= expected_pcc_decode, f"{backend} decode PCC {pcc_value} < {expected_pcc_decode}"
    else:  # prefill
        assert pcc_value >= expected_pcc_prefill, f"{backend} prefill PCC {pcc_value} < {expected_pcc_prefill}"
