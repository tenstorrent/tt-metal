# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE
from models.demos.deepseek_v3.tests.pytest_utils import DEFAULT_PREFILL_SEQ_LEN
from models.demos.deepseek_v3.tt.moe import MoE
from models.demos.deepseek_v3.tt.moe_gate import MoEGate
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import get_model_config, get_test_weight_config, run_module_forward
from tests.ttnn.utils_for_testing import comp_pcc


@pytest.fixture
def reference_model(hf_config):
    """Get the actual DeepSeek MLP model using local implementation."""
    torch.use_deterministic_algorithms(True)
    # Note : Running Reference MoE without shared experts
    hf_config.n_shared_experts = None
    return DeepseekV3MoE(hf_config).eval()


_max_seq_len_env = os.getenv("DEEPSEEK_MAX_SEQ_LEN_OVERRIDE")
_prefill_seq_len = int(_max_seq_len_env) if _max_seq_len_env is not None else DEFAULT_PREFILL_SEQ_LEN


@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode,num_tokens",
    [
        ("decode", 128),
        ("prefill", _prefill_seq_len),
    ],
)
@pytest.mark.parametrize(
    "topk_fallback",
    [
        True,
    ],
)
def test_forward_pass(
    mode,
    num_tokens,
    set_deterministic_env,
    reference_model,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    topk_fallback,
    state_dict,
):
    """Test forward pass against reference model using loaded dequantized weights."""

    def _to_torch_gate_outputs(
        tt_gate_weights: ttnn.Tensor, tt_gate_indices: ttnn.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_weights_torch = ttnn.to_torch(
            tt_gate_weights,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
        )[0].squeeze(0)
        gate_indices_torch = ttnn.to_torch(
            tt_gate_indices,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
        )[0].squeeze(0)
        return gate_weights_torch, gate_indices_torch

    def _manual_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.reshape(-1).to(torch.float32)
        b = b.reshape(-1).to(torch.float32)
        a_centered = a - a.mean()
        b_centered = b - b.mean()
        denom = a_centered.norm() * b_centered.norm()
        if denom == 0:
            return 0.0
        return float((a_centered * b_centered).sum() / denom)

    # Extract MoE state dict from a MoE layer (using layer 3 as default MoE layer)
    # The state_dict fixture contains dequantized weights from the full model
    moe_layer_idx = hf_config.first_k_dense_replace if hasattr(hf_config, "first_k_dense_replace") else 3
    module_path = f"model.layers.{moe_layer_idx}.mlp"
    logger.info(f"DEBUGGING: Extracting MoE state_dict from layer {moe_layer_idx}, path: {module_path}")
    state_dict = sub_state_dict(state_dict, module_path + ".")

    # Filter out shared_experts keys since reference_model has n_shared_experts=None
    state_dict_keys_before = set(state_dict.keys())
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("shared_experts.")}
    state_dict_keys_after = set(state_dict.keys())
    filtered_keys = state_dict_keys_before - state_dict_keys_after

    logger.info(f"DEBUGGING: State dict keys before filtering: {len(state_dict_keys_before)}")
    logger.info(f"DEBUGGING: State dict keys after filtering: {len(state_dict_keys_after)}")
    if filtered_keys:
        logger.info(f"DEBUGGING: Filtered out keys: {filtered_keys}")
    logger.info(
        f"DEBUGGING: State dict keys: {sorted(state_dict.keys())[:10]}..."
        if len(state_dict) > 10
        else f"DEBUGGING: State dict keys: {sorted(state_dict.keys())}"
    )

    # Log weight statistics and verify dequantization
    logger.info("DEBUGGING: Verifying weights are dequantized (bfloat16, not fp8):")
    for key, tensor in list(state_dict.items())[:5]:  # Log first 5 weights
        is_fp8 = tensor.dtype == torch.float8_e4m3fn
        has_scale_inv = f"{key}_scale_inv" in state_dict
        logger.info(
            f"DEBUGGING: Weight '{key}': shape={tensor.shape}, dtype={tensor.dtype}, "
            f"is_fp8={is_fp8}, has_scale_inv={has_scale_inv}, "
            f"mean={tensor.float().mean().item():.6f}, std={tensor.float().std().item():.6f}"
        )
        if is_fp8:
            logger.error(f"ERROR: Weight '{key}' is still fp8 quantized!")
        if has_scale_inv:
            logger.error(f"ERROR: Weight '{key}' has scale_inv (quantized format)!")

    # Verify no fp8 or scale_inv keys exist
    fp8_keys = [k for k, v in state_dict.items() if v.dtype == torch.float8_e4m3fn]
    scale_inv_keys = [k for k in state_dict.keys() if k.endswith("_scale_inv")]
    if fp8_keys:
        logger.error(f"ERROR: Found {len(fp8_keys)} fp8 quantized weights: {fp8_keys[:5]}...")
    if scale_inv_keys:
        logger.error(f"ERROR: Found {len(scale_inv_keys)} scale_inv keys: {scale_inv_keys[:5]}...")
    logger.info(
        f"DEBUGGING: Weight verification complete - fp8 keys: {len(fp8_keys)}, scale_inv keys: {len(scale_inv_keys)}"
    )

    # Load dequantized weights into reference model
    missing_keys, unexpected_keys = reference_model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        logger.warning(f"DEBUGGING: Missing keys when loading state_dict: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"DEBUGGING: Unexpected keys when loading state_dict: {unexpected_keys}")
    reference_model.eval()
    reference_model.to(torch.bfloat16)

    # Create input tensor
    torch.manual_seed(42)  # For reproducibility
    torch_input = torch.randn(1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)
    logger.info(f"DEBUGGING: Input tensor shape: {torch_input.shape}, dtype: {torch_input.dtype}")
    logger.info(
        f"DEBUGGING: Input stats - mean: {torch_input.float().mean().item():.6f}, std: {torch_input.float().std().item():.6f}, min: {torch_input.float().min().item():.6f}, max: {torch_input.float().max().item():.6f}"
    )

    # Reference forward pass
    logger.info("DEBUGGING: Running reference forward pass...")
    with torch.no_grad():
        reference_output = reference_model(torch_input)
    logger.info(f"DEBUGGING: Reference output shape: {reference_output.shape}, dtype: {reference_output.dtype}")
    logger.info(
        f"DEBUGGING: Reference output stats - mean: {reference_output.float().mean().item():.6f}, std: {reference_output.float().std().item():.6f}"
    )

    # Verify state_dict passed to weight conversion matches what reference model uses
    logger.info("DEBUGGING: Verifying state_dict for weight conversion:")
    logger.info(f"DEBUGGING: State dict has {len(state_dict)} keys")
    logger.info(f"DEBUGGING: Sample expert weight check - 'experts.0.gate_proj.weight':")
    if "experts.0.gate_proj.weight" in state_dict:
        expert_weight = state_dict["experts.0.gate_proj.weight"]
        logger.info(f"  Shape: {expert_weight.shape}, dtype: {expert_weight.dtype}")
        logger.info(f"  First 5 values: {expert_weight.flatten()[:5].tolist()}")
        logger.info(f"  Mean: {expert_weight.float().mean().item():.6f}, Std: {expert_weight.float().std().item():.6f}")

    # Check if reference model loaded the weights correctly
    logger.info("DEBUGGING: Verifying reference model loaded weights correctly:")
    ref_gate_weight = reference_model.gate.weight
    logger.info(f"  Reference gate.weight: shape={ref_gate_weight.shape}, dtype={ref_gate_weight.dtype}")
    if "gate.weight" in state_dict:
        state_gate_weight = state_dict["gate.weight"]
        logger.info(f"  State dict gate.weight: shape={state_gate_weight.shape}, dtype={state_gate_weight.dtype}")
        # Check if they match
        if torch.equal(ref_gate_weight, state_gate_weight):
            logger.info("  ✓ Reference model gate.weight matches state_dict")
        else:
            logger.warning("  ✗ Reference model gate.weight does NOT match state_dict!")
            logger.warning(
                f"    Max diff: {(ref_gate_weight.float() - state_gate_weight.float()).abs().max().item():.6f}"
            )

    # IMPORTANT: Force recalculation since we're now using dequantized weights
    # The cached weights were likely computed with the old quantization method
    logger.info(
        "DEBUGGING: Forcing weight recalculation with dequantized weights (cache may be from old quantization method)"
    )
    weight_config = get_test_weight_config(
        MoE,
        hf_config,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate=True,  # Force recalculation with dequantized weights
        test_name="test_moe",
        real_weights=True,
    )

    # Generate appropriate config using utility function
    model_config = get_model_config(MoE, mode, hf_config, mesh_device, topk_fallback=topk_fallback)

    # Create a new model state with CCL
    model_state = MoE.create_state(hf_config, mesh_device, ccl)

    # Create a new model shared state
    model_shared_state = MoE.create_shared_state(hf_config, mesh_device)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # Convert input to TTNN, DP=4 and Replicated
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # TTNN forward pass - collective operations handled inside forward functions
    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])

    # Operation-level diagnostics for gate path
    # This isolates whether ttnn.linear in gate_proj is the dominant source of routing drift.
    with torch.no_grad():
        reference_topk_indices, reference_topk_weights = reference_model.gate(torch_input)
    reference_topk_indices = reference_topk_indices.to(torch.int32)
    reference_topk_weights = reference_topk_weights.to(torch.float32)

    logger.info("DEBUGGING: Running gate operation-level diagnostics (TT gate vs linear fallback gate)...")
    tt_gate_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_gate_input = ttnn.to_memory_config(tt_gate_input, run_config["moe_gate"]["input_memory_config"])

    # Gate-projection operation-level diagnostics (isolates ttnn.linear error before routing).
    reference_logits = torch.nn.functional.linear(
        torch_input.reshape(-1, hf_config.hidden_size).to(torch.float32), reference_model.gate.weight.to(torch.float32)
    )
    tt_gate_linear_input = ttnn.typecast(tt_gate_input, dtype=ttnn.float32)
    tt_gate_logits = ttnn.linear(tt_gate_linear_input, dtype=ttnn.float32, **run_config["moe_gate"]["gate_proj"])
    ttnn.deallocate(tt_gate_linear_input)
    tt_gate_logits_torch = ttnn.to_torch(
        tt_gate_logits,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )[0].squeeze(0)
    tt_gate_logits_torch = tt_gate_logits_torch.reshape(-1, hf_config.n_routed_experts).to(torch.float32)
    ttnn.deallocate(tt_gate_logits)

    gate_logits_pcc = _manual_pcc(reference_logits, tt_gate_logits_torch)
    gate_scores_pcc = _manual_pcc(torch.sigmoid(reference_logits), torch.sigmoid(tt_gate_logits_torch))
    logger.info(f"DEBUGGING gate_proj op: logits_pcc={gate_logits_pcc:.6f}, scores_pcc={gate_scores_pcc:.6f}")

    k = hf_config.num_experts_per_tok
    ref_sorted_scores, _ = torch.sort(torch.sigmoid(reference_logits), dim=-1, descending=True)
    margin = ref_sorted_scores[:, k - 1] - ref_sorted_scores[:, k]
    near_tie_frac_1e3 = (margin < 1e-3).float().mean().item()
    near_tie_frac_1e4 = (margin < 1e-4).float().mean().item()
    logger.info(
        f"DEBUGGING routing sensitivity: near_tie_frac_1e-3={near_tie_frac_1e3:.4f}, "
        f"near_tie_frac_1e-4={near_tie_frac_1e4:.4f}"
    )

    tt_gate_weights, tt_gate_indices = MoEGate.forward(tt_gate_input, run_config["moe_gate"])
    tt_gate_weights_torch, tt_gate_indices_torch = _to_torch_gate_outputs(tt_gate_weights, tt_gate_indices)
    ttnn.deallocate(tt_gate_weights)
    ttnn.deallocate(tt_gate_indices)

    tt_gate_weights_torch = tt_gate_weights_torch.to(torch.float32)
    tt_gate_indices_torch = tt_gate_indices_torch.to(torch.int32)
    _, gate_weight_msg = comp_pcc(reference_topk_weights, tt_gate_weights_torch, 0.999)
    gate_weight_pcc = _manual_pcc(reference_topk_weights, tt_gate_weights_torch)
    ref_idx_sorted = torch.sort(reference_topk_indices, dim=-1, stable=True)[0]
    tt_idx_sorted = torch.sort(tt_gate_indices_torch, dim=-1, stable=True)[0]
    gate_token_match = torch.all(ref_idx_sorted == tt_idx_sorted, dim=-1).float().mean().item()
    logger.info(
        f"DEBUGGING gate (TT linear): topk weight PCC={gate_weight_msg} "
        f"(manual={gate_weight_pcc:.6f}), token index match={gate_token_match:.4f}"
    )

    gate_cfg_linear = dict(run_config["moe_gate"])
    gate_cfg_linear["linear_fallback"] = True
    tt_gate_weights_linear, tt_gate_indices_linear = MoEGate.forward(tt_gate_input, gate_cfg_linear)
    tt_gate_weights_linear_torch, tt_gate_indices_linear_torch = _to_torch_gate_outputs(
        tt_gate_weights_linear, tt_gate_indices_linear
    )
    ttnn.deallocate(tt_gate_weights_linear)
    ttnn.deallocate(tt_gate_indices_linear)
    ttnn.deallocate(tt_gate_input)

    tt_gate_weights_linear_torch = tt_gate_weights_linear_torch.to(torch.float32)
    tt_gate_indices_linear_torch = tt_gate_indices_linear_torch.to(torch.int32)
    _, gate_weight_linear_msg = comp_pcc(reference_topk_weights, tt_gate_weights_linear_torch, 0.999)
    gate_weight_linear_pcc = _manual_pcc(reference_topk_weights, tt_gate_weights_linear_torch)
    tt_linear_idx_sorted = torch.sort(tt_gate_indices_linear_torch, dim=-1, stable=True)[0]
    gate_linear_token_match = torch.all(ref_idx_sorted == tt_linear_idx_sorted, dim=-1).float().mean().item()
    logger.info(
        f"DEBUGGING gate (linear_fallback): topk weight PCC={gate_weight_linear_msg}, "
        f"(manual={gate_weight_linear_pcc:.6f}), token index match={gate_linear_token_match:.4f}"
    )
    logger.info(
        "DEBUGGING gate diagnosis: "
        f"delta_weight_pcc={gate_weight_linear_pcc - gate_weight_pcc:.6f}, "
        f"delta_token_match={gate_linear_token_match - gate_token_match:.4f}"
    )

    # Run forward pass
    logger.info("DEBUGGING: Running TTNN forward pass...")
    tt_output = run_module_forward(MoE, mode, tt_input, run_config)
    logger.info(f"DEBUGGING: TTNN output shape: {tt_output.shape}, dtype: {tt_output.dtype}")

    # Verify output memory config matches expected
    expected_output_memory_config = run_config["output_memory_config"]
    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"MoE output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    # Convert output back to torch
    logger.info("DEBUGGING: Converting TTNN output to torch...")
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )
    logger.info(f"DEBUGGING: Converted TTNN output shape: {tt_output_torch.shape}, dtype: {tt_output_torch.dtype}")

    # Full-model ablation: swap only gate_proj matmul to torch and keep rest path identical.
    # If this closes most gap, culprit is gate_proj ttnn.linear precision.
    run_config_linear_gate = dict(run_config)
    run_config_linear_gate["moe_gate"] = dict(run_config["moe_gate"])
    run_config_linear_gate["moe_gate"]["linear_fallback"] = True
    tt_input_linear_gate = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_input_linear_gate = ttnn.to_memory_config(tt_input_linear_gate, run_config_linear_gate["input_memory_config"])
    tt_output_linear_gate = run_module_forward(MoE, mode, tt_input_linear_gate, run_config_linear_gate)
    tt_output_linear_gate_torch = ttnn.to_torch(
        tt_output_linear_gate,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )
    pass_linear_gate, pcc_linear_gate_msg = comp_pcc(
        reference_output.unsqueeze(0).to(torch.float32), tt_output_linear_gate_torch.to(torch.float32), 0.98
    )
    logger.info(
        f"DEBUGGING full MoE ablation (linear_fallback gate_proj only): pass={pass_linear_gate}, "
        f"pcc={pcc_linear_gate_msg}"
    )
    ttnn.deallocate(tt_input_linear_gate)
    ttnn.deallocate(tt_output_linear_gate)

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    # Heavy debugging for PCC analysis
    logger.info("=" * 80)
    logger.info("DEBUGGING: Output Comparison Analysis")
    logger.info("=" * 80)

    # Prepare outputs for comparison
    ref_output_for_compare = reference_output.unsqueeze(0) if reference_output.dim() == 3 else reference_output
    tt_output_for_compare = tt_output_torch

    logger.info(f"Reference output shape: {ref_output_for_compare.shape}")
    logger.info(f"TTNN output shape: {tt_output_for_compare.shape}")

    # Convert to float for analysis
    ref_float = ref_output_for_compare.cpu().float()
    tt_float = tt_output_for_compare.cpu().float()

    # Shape alignment
    seq_len = min(ref_float.shape[-2], tt_float.shape[-2])
    ref_float = ref_float[..., :seq_len, :]
    tt_float = tt_float[..., :seq_len, :]

    logger.info(f"Aligned shapes - Reference: {ref_float.shape}, TTNN: {tt_float.shape}")

    # Statistics
    logger.info("Reference output statistics:")
    logger.info(f"  Mean: {ref_float.mean().item():.6f}, Std: {ref_float.std().item():.6f}")
    logger.info(f"  Min: {ref_float.min().item():.6f}, Max: {ref_float.max().item():.6f}")
    logger.info(f"  Abs mean: {ref_float.abs().mean().item():.6f}")

    logger.info("TTNN output statistics:")
    logger.info(f"  Mean: {tt_float.mean().item():.6f}, Std: {tt_float.std().item():.6f}")
    logger.info(f"  Min: {tt_float.min().item():.6f}, Max: {tt_float.max().item():.6f}")
    logger.info(f"  Abs mean: {tt_float.abs().mean().item():.6f}")

    # Difference statistics
    diff = tt_float - ref_float
    abs_diff = diff.abs()
    rel_diff = abs_diff / (ref_float.abs() + 1e-8)

    logger.info("Difference statistics:")
    logger.info(f"  Mean diff: {diff.mean().item():.6f}, Std diff: {diff.std().item():.6f}")
    logger.info(f"  Mean abs diff: {abs_diff.mean().item():.6f}, Max abs diff: {abs_diff.max().item():.6f}")
    logger.info(f"  Mean rel diff: {rel_diff.mean().item():.6f}, Max rel diff: {rel_diff.max().item():.6f}")

    # Per-token statistics
    if ref_float.shape[-2] > 1:
        token_abs_diffs = abs_diff.view(-1, seq_len, ref_float.shape[-1]).mean(dim=-1)  # [batch, seq_len]
        token_std_diffs = abs_diff.view(-1, seq_len, ref_float.shape[-1]).std(dim=-1)  # [batch, seq_len]
        logger.info("Per-token mean absolute differences:")
        for i in range(min(10, seq_len)):
            logger.info(f"  Token {i}: mean={token_abs_diffs[0, i].item():.6f}, std={token_std_diffs[0, i].item():.6f}")
        if seq_len > 10:
            logger.info(f"  ... (showing first 10 of {seq_len} tokens)")
        worst_token_pos = token_abs_diffs.argmax().item()
        logger.info(f"  Worst token: mean={token_abs_diffs.max().item():.6f} at position {worst_token_pos}")
        logger.info(
            f"  Best token: mean={token_abs_diffs.min().item():.6f} at position {token_abs_diffs.argmin().item()}"
        )

    # Per-dimension statistics
    dim_abs_diffs = abs_diff.view(-1, seq_len, ref_float.shape[-1]).mean(dim=(0, 1))  # [hidden_dim]
    logger.info("Per-dimension mean absolute differences (worst 10):")
    worst_dims = dim_abs_diffs.topk(10)
    for idx, val in zip(worst_dims.indices, worst_dims.values):
        logger.info(f"  Dim {idx.item()}: {val.item():.6f}")

    # Check for NaN/Inf
    ref_nan = torch.isnan(ref_float).sum().item()
    tt_nan = torch.isnan(tt_float).sum().item()
    ref_inf = torch.isinf(ref_float).sum().item()
    tt_inf = torch.isinf(tt_float).sum().item()
    logger.info("NaN/Inf check:")
    logger.info(f"  Reference: {ref_nan} NaN, {ref_inf} Inf")
    logger.info(f"  TTNN: {tt_nan} NaN, {tt_inf} Inf")

    # Find worst absolute differences
    worst_abs_diff_idx = abs_diff.argmax()
    worst_abs_diff_val = abs_diff.max().item()
    worst_idx_unraveled = torch.unravel_index(worst_abs_diff_idx, abs_diff.shape)
    logger.info(f"Worst absolute difference:")
    logger.info(f"  Value: {worst_abs_diff_val:.6f}")
    logger.info(f"  Position: {worst_idx_unraveled}")
    logger.info(f"  Ref value: {ref_float[worst_idx_unraveled].item():.6f}")
    logger.info(f"  TT value: {tt_float[worst_idx_unraveled].item():.6f}")

    # Find worst relative differences (excluding near-zero ref values)
    ref_abs_threshold = 1e-6
    valid_mask = ref_float.abs() > ref_abs_threshold
    if valid_mask.sum() > 0:
        valid_rel_diff = rel_diff[valid_mask]
        worst_rel_diff_idx = valid_rel_diff.argmax()
        worst_rel_diff_val = valid_rel_diff.max().item()
        worst_rel_positions = torch.where(valid_mask)
        worst_rel_pos = tuple(pos[worst_rel_diff_idx].item() for pos in worst_rel_positions)
        logger.info(f"Worst relative difference (ref > {ref_abs_threshold}):")
        logger.info(f"  Value: {worst_rel_diff_val:.6f}")
        logger.info(f"  Position: {worst_rel_pos}")
        logger.info(f"  Ref value: {ref_float[worst_rel_pos].item():.6f}")
        logger.info(f"  TT value: {tt_float[worst_rel_pos].item():.6f}")
    else:
        logger.info("No valid relative differences (all ref values too small)")

    # Sample values comparison
    logger.info("Sample value comparison (first 5 tokens, first 10 dims):")
    # Handle both 3D and 4D shapes
    if ref_float.ndim == 4:
        # Shape is [batch, 1, seq_len, hidden_dim] or similar
        for token_idx in range(min(5, seq_len)):
            logger.info(f"  Token {token_idx}:")
            for dim_idx in range(min(10, ref_float.shape[-1])):
                ref_val = ref_float[0, 0, token_idx, dim_idx].item()
                tt_val = tt_float[0, 0, token_idx, dim_idx].item()
                diff_val = abs_diff[0, 0, token_idx, dim_idx].item()
                rel_val = (
                    rel_diff[0, 0, token_idx, dim_idx].item()
                    if ref_float[0, 0, token_idx, dim_idx].abs().item() > ref_abs_threshold
                    else float("inf")
                )
                logger.info(
                    f"    Dim {dim_idx}: ref={ref_val:10.6f}, tt={tt_val:10.6f}, abs_diff={diff_val:10.6f}, rel_diff={rel_val:10.2f}"
                )
    else:
        # Shape is [batch, seq_len, hidden_dim]
        for token_idx in range(min(5, seq_len)):
            logger.info(f"  Token {token_idx}:")
            for dim_idx in range(min(10, ref_float.shape[-1])):
                ref_val = ref_float[0, token_idx, dim_idx].item()
                tt_val = tt_float[0, token_idx, dim_idx].item()
                diff_val = abs_diff[0, token_idx, dim_idx].item()
                rel_val = (
                    rel_diff[0, token_idx, dim_idx].item()
                    if ref_float[0, token_idx, dim_idx].abs().item() > ref_abs_threshold
                    else float("inf")
                )
                logger.info(
                    f"    Dim {dim_idx}: ref={ref_val:10.6f}, tt={tt_val:10.6f}, abs_diff={diff_val:10.6f}, rel_diff={rel_val:10.2f}"
                )

    # Check worst token in detail
    worst_token_idx = token_abs_diffs.argmax().item()
    logger.info(f"Worst token {worst_token_idx} detailed analysis:")
    # Handle both 3D and 4D shapes
    if ref_float.ndim == 4:
        worst_token_ref = ref_float[0, 0, worst_token_idx, :]
        worst_token_tt = tt_float[0, 0, worst_token_idx, :]
        worst_token_diff = abs_diff[0, 0, worst_token_idx, :]
    else:
        worst_token_ref = ref_float[0, worst_token_idx, :]
        worst_token_tt = tt_float[0, worst_token_idx, :]
        worst_token_diff = abs_diff[0, worst_token_idx, :]

    # Find worst dimensions in this token
    worst_dims_in_token = worst_token_diff.topk(10)
    logger.info(f"  Worst 10 dimensions in token {worst_token_idx}:")
    for idx, val in zip(worst_dims_in_token.indices, worst_dims_in_token.values):
        dim_idx = idx.item()
        ref_val = worst_token_ref[dim_idx].item()
        tt_val = worst_token_tt[dim_idx].item()
        logger.info(f"    Dim {dim_idx}: ref={ref_val:10.6f}, tt={tt_val:10.6f}, abs_diff={val.item():10.6f}")

    # Check the specific problematic dimension 4978 across all tokens
    if 4978 < ref_float.shape[-1]:
        logger.info(f"Dimension 4978 analysis across all tokens:")
        # Handle both 3D and 4D shapes
        if ref_float.ndim == 4:
            dim_4978_ref = ref_float[0, 0, :, 4978]
            dim_4978_tt = tt_float[0, 0, :, 4978]
            dim_4978_diff = abs_diff[0, 0, :, 4978]
        else:
            dim_4978_ref = ref_float[0, :, 4978]
            dim_4978_tt = tt_float[0, :, 4978]
            dim_4978_diff = abs_diff[0, :, 4978]
        logger.info(f"  Mean ref: {dim_4978_ref.mean().item():.6f}, Mean TT: {dim_4978_tt.mean().item():.6f}")
        logger.info(
            f"  Mean abs diff: {dim_4978_diff.mean().item():.6f}, Max abs diff: {dim_4978_diff.max().item():.6f}"
        )
        worst_token_for_dim = dim_4978_diff.argmax().item()
        logger.info(f"  Worst token for this dim: {worst_token_for_dim}")
        logger.info(
            f"    Ref: {dim_4978_ref[worst_token_for_dim].item():.6f}, TT: {dim_4978_tt[worst_token_for_dim].item():.6f}"
        )

    # Check token 106, dim 4978 specifically (worst absolute difference)
    logger.info(f"Token 106, Dim 4978 (worst absolute difference) detailed:")
    if 106 < seq_len and 4978 < ref_float.shape[-1]:
        # Handle both 3D and 4D shapes
        if ref_float.ndim == 4:
            ref_val_106_4978 = ref_float[0, 0, 106, 4978].item()
            tt_val_106_4978 = tt_float[0, 0, 106, 4978].item()
            logger.info(
                f"  Ref: {ref_val_106_4978:.6f}, TT: {tt_val_106_4978:.6f}, Diff: {abs(ref_val_106_4978 - tt_val_106_4978):.6f}"
            )
            # Check surrounding dimensions
            logger.info(f"  Surrounding dimensions (4976-4980):")
            for dim in range(max(0, 4978 - 2), min(ref_float.shape[-1], 4978 + 3)):
                ref_v = ref_float[0, 0, 106, dim].item()
                tt_v = tt_float[0, 0, 106, dim].item()
                diff_v = abs_diff[0, 0, 106, dim].item()
                marker = " <-- WORST" if dim == 4978 else ""
                logger.info(f"    Dim {dim}: ref={ref_v:10.6f}, tt={tt_v:10.6f}, diff={diff_v:10.6f}{marker}")
        else:
            ref_val_106_4978 = ref_float[0, 106, 4978].item()
            tt_val_106_4978 = tt_float[0, 106, 4978].item()
            logger.info(
                f"  Ref: {ref_val_106_4978:.6f}, TT: {tt_val_106_4978:.6f}, Diff: {abs(ref_val_106_4978 - tt_val_106_4978):.6f}"
            )
            # Check surrounding dimensions
            logger.info(f"  Surrounding dimensions (4976-4980):")
            for dim in range(max(0, 4978 - 2), min(ref_float.shape[-1], 4978 + 3)):
                ref_v = ref_float[0, 106, dim].item()
                tt_v = tt_float[0, 106, dim].item()
                diff_v = abs_diff[0, 106, dim].item()
                marker = " <-- WORST" if dim == 4978 else ""
                logger.info(f"    Dim {dim}: ref={ref_v:10.6f}, tt={tt_v:10.6f}, diff={diff_v:10.6f}{marker}")

    # Correlation analysis
    ref_flat = ref_float.flatten()
    tt_flat = tt_float.flatten()

    # Compute correlation manually for debugging
    ref_mean = ref_flat.mean()
    tt_mean = tt_flat.mean()
    ref_centered = ref_flat - ref_mean
    tt_centered = tt_flat - tt_mean

    numerator = (ref_centered * tt_centered).sum()
    ref_std = ref_centered.norm()
    tt_std = tt_centered.norm()
    denominator = ref_std * tt_std

    manual_pcc = (numerator / denominator).item() if denominator > 0 else 0.0

    logger.info("Correlation analysis:")
    logger.info(f"  Manual PCC: {manual_pcc:.6f}")
    logger.info(f"  Ref std: {ref_std.item():.6f}, TT std: {tt_std.item():.6f}")

    logger.info("=" * 80)

    # Compare outputs using utility function
    logger.info(f"Mode: {mode}, Num tokens: {num_tokens}")
    final_ok, final_pcc_msg = comp_pcc(
        reference_output.unsqueeze(0).to(torch.float32), tt_output_torch.to(torch.float32), 0.98
    )
    if not final_ok:
        raise AssertionError(
            "MoE PCC below threshold. "
            f"{final_pcc_msg}. "
            f"Diagnostics: gate_logits_pcc={gate_logits_pcc:.6f}, gate_scores_pcc={gate_scores_pcc:.6f}, "
            f"gate_token_match_tt_linear={gate_token_match:.4f}, gate_token_match_linear_fallback={gate_linear_token_match:.4f}, "
            f"full_moe_linear_gate_only={pcc_linear_gate_msg}, near_tie_frac_1e-3={near_tie_frac_1e3:.4f}, "
            f"near_tie_frac_1e-4={near_tie_frac_1e4:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
