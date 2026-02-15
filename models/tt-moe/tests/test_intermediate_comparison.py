#!/usr/bin/env python3
"""
Systematic intermediate tensor comparison between our MoEBlock and reference MoEDecoderBlock2D.
This test captures and compares tensors at each stage to identify where numerical divergence occurs.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

# Add paths
sys.path.append("/home/ntarafdar/tt-moe/tt-metal")
sys.path.append("/home/ntarafdar/tt-moe/tt-metal/models")
sys.path.append("/home/ntarafdar/tt-moe/tt-metal/models/tt-moe")

# Set environment
os.environ["PYTHONPATH"] = "/home/ntarafdar/tt-moe/tt-metal"
os.environ["TT_METAL_HOME"] = "/home/ntarafdar/tt-moe/tt-metal"
os.environ["MESH_DEVICE"] = "TG"

from transformers import AutoConfig

import ttnn

# Import test utilities
from models.demos.deepseek_v3.utils.test_utils import load_state_dict


def compute_pcc(x, y):
    """Compute Pearson Correlation Coefficient"""
    # Convert TTNN tensors to torch if needed
    if str(type(x)).find("ttnn") != -1:
        try:
            # For TTNN tensors, we need to convert properly based on mesh device
            # This is a placeholder - actual conversion depends on tensor properties
            x = ttnn.to_torch(x)
        except:
            logger.warning(f"Could not convert TTNN tensor to torch")
            return 0.0

    if str(type(y)).find("ttnn") != -1:
        try:
            y = ttnn.to_torch(y)
        except:
            logger.warning(f"Could not convert TTNN tensor to torch")
            return 0.0

    # Convert to numpy
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if torch.is_tensor(y):
        y = y.cpu().numpy()

    x_flat = x.flatten().astype(np.float32)
    y_flat = y.flatten().astype(np.float32)

    if len(x_flat) == 0 or len(y_flat) == 0:
        return 0.0

    # Handle constant tensors
    if np.std(x_flat) == 0 or np.std(y_flat) == 0:
        if np.allclose(x_flat, y_flat):
            return 1.0
        return 0.0

    return np.corrcoef(x_flat, y_flat)[0, 1]


def run_our_implementation(input_tensor, mesh_device, ccl, config_path, state_dict, layer_idx=3):
    """
    Run our MoEBlock implementation and return intermediate tensors.
    """
    from moe_block import MoEBlock

    logger.info("Running OUR MoEBlock implementation...")

    # Create MoEBlock
    moe_block = MoEBlock(config_path, mesh_device, ccl)

    # Load weights
    module_path = f"model.layers.{layer_idx}"
    moe_state_dict = {}
    weights_loaded = 0

    # Load gate weights
    gate_weight_key = f"{module_path}.mlp.gate.weight"
    if gate_weight_key in state_dict:
        moe_state_dict["mlp.gate.weight"] = state_dict[gate_weight_key]
        weights_loaded += 1

    gate_bias_key = f"{module_path}.mlp.gate.e_score_correction_bias"
    if gate_bias_key in state_dict:
        moe_state_dict["mlp.gate.e_score_correction_bias"] = state_dict[gate_bias_key]
        weights_loaded += 1

    # Load shared expert weights
    shared_keys = [
        "shared_experts.gate_proj.weight",
        "shared_experts.up_proj.weight",
        "shared_experts.down_proj.weight",
    ]
    for key in shared_keys:
        full_key = f"{module_path}.mlp.{key}"
        if full_key in state_dict:
            new_key = f"mlp.{key}"
            moe_state_dict[new_key] = state_dict[full_key]
            weights_loaded += 1

    # Load expert weights (all 256)
    for i in range(256):
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            key = f"{module_path}.mlp.experts.{i}.{proj}.weight"
            if key in state_dict:
                new_key = f"mlp.experts.{i}.{proj}.weight"
                moe_state_dict[new_key] = state_dict[key]
                weights_loaded += 1

                # Also load scale if exists
                scale_key = key + "_scale_inv"
                if scale_key in state_dict:
                    scale_new_key = new_key + "_scale_inv"
                    moe_state_dict[scale_new_key] = state_dict[scale_key]
                    weights_loaded += 1

    logger.info(f"Loaded {weights_loaded} weights for MoEBlock")
    moe_block.load_weights(moe_state_dict)

    # Use the forward_with_intermediates method we added to moe_block
    output, intermediates = moe_block.forward_with_intermediates(input_tensor, mode="decode")

    return output, intermediates


def run_reference_implementation(input_tensor, mesh_device, ccl, hf_config, state_dict, layer_idx=3):
    """
    Run reference MoEDecoderBlock2D implementation and return intermediate tensors.
    Modified to match our working implementation's approach.
    """
    import os
    import sys

    test_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, test_dir)

    # Import the new functions from modified_reference_decoder
    from modified_reference_decoder import create_test_setup, run_mlp_only_with_intermediates

    logger.info("Running REFERENCE MoEDecoderBlock2D implementation...")

    # Cache path for converted weights
    cache_path = Path(os.environ.get("DEEPSEEK_V3_CACHE", "/tmp/deepseek_cache"))

    # The create_test_setup expects the FULL state dict with complete paths
    # It will extract what it needs internally
    # Just pass the full state dict directly
    run_config = create_test_setup(
        hf_config,
        mesh_device,
        ccl,
        layer_idx,
        state_dict,  # Pass the full state dict - create_test_setup will extract layer weights
        cache_path,
        mode="decode",
    )

    # Run MLP forward with intermediate capture
    # This runs just the MoEDecoderBlock2D.forward_mlp_decode part
    output, intermediates = run_mlp_only_with_intermediates(input_tensor, run_config)

    return output, intermediates


def compare_intermediates(our_intermediates, ref_intermediates, mesh_device=None):
    """
    Compare intermediate tensors and report PCC at each stage.
    Returns a dict with comparison results.
    """
    logger.info("\n" + "=" * 60)
    logger.info("INTERMEDIATE TENSOR COMPARISON")
    logger.info("=" * 60)

    stages = [
        "input",
        "all_gather_output",
        "router_weights",
        "router_indices",
        "weights_prepared",
        "dispatch_output",
        "experts_output",
        "moe_output",
        "shared_output",
        "combined_output",
        "reduce_scatter_output",
        "final_output",
    ]

    results = []
    first_divergence = None

    for stage in stages:
        # Check both with and without prefix for reference
        our_key = stage
        ref_key = stage

        # Handle potential prefix differences
        if stage.startswith("moe_") and stage not in ref_intermediates:
            ref_key = stage[4:]  # Remove "moe_" prefix
        elif f"moe_{stage}" in ref_intermediates and stage not in ref_intermediates:
            ref_key = f"moe_{stage}"

        if our_key in our_intermediates and ref_key in ref_intermediates:
            our_tensor = our_intermediates[our_key]
            ref_tensor = ref_intermediates[ref_key]

            # Handle TTNN tensors - need proper conversion
            if str(type(our_tensor)).find("ttnn") != -1 or str(type(ref_tensor)).find("ttnn") != -1:
                logger.info(f"{stage}:")
                logger.info(f"  Our shape: {our_tensor.shape if hasattr(our_tensor, 'shape') else 'unknown'}")
                logger.info(f"  Ref shape: {ref_tensor.shape if hasattr(ref_tensor, 'shape') else 'unknown'}")

                # Try to compute PCC if we can convert
                try:
                    pcc = compute_pcc(our_tensor, ref_tensor)
                    logger.info(f"  PCC: {pcc:.6f}")

                    results.append(
                        {
                            "stage": stage,
                            "pcc": pcc,
                            "our_shape": str(our_tensor.shape) if hasattr(our_tensor, "shape") else "unknown",
                            "ref_shape": str(ref_tensor.shape) if hasattr(ref_tensor, "shape") else "unknown",
                        }
                    )

                    if pcc < 0.98 and first_divergence is None:
                        first_divergence = stage
                        logger.warning(f"  ⚠️ First PCC divergence below 0.98 threshold at {stage}")
                except Exception as e:
                    logger.warning(f"  Could not compute PCC: {e}")
                    continue

            # Handle torch tensors directly
            elif isinstance(our_tensor, torch.Tensor) and isinstance(ref_tensor, torch.Tensor):
                # Check shapes
                if our_tensor.shape != ref_tensor.shape:
                    logger.error(f"{stage}: Shape mismatch - Ours: {our_tensor.shape}, Ref: {ref_tensor.shape}")
                    continue

                # Compute PCC
                pcc = compute_pcc(our_tensor, ref_tensor)

                # Compute differences
                diff = torch.abs(our_tensor - ref_tensor)
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()

                logger.info(f"{stage}:")
                logger.info(f"  Shape: {our_tensor.shape}")
                logger.info(f"  PCC: {pcc:.6f}")
                logger.info(f"  Max diff: {max_diff:.6e}")
                logger.info(f"  Mean diff: {mean_diff:.6e}")

                results.append(
                    {
                        "stage": stage,
                        "pcc": pcc,
                        "max_diff": max_diff,
                        "mean_diff": mean_diff,
                        "our_shape": str(our_tensor.shape),
                        "ref_shape": str(ref_tensor.shape),
                    }
                )

                if pcc < 0.98 and first_divergence is None:
                    first_divergence = stage
                    logger.warning(f"  ⚠️ First PCC divergence below 0.98 threshold at {stage}")
        else:
            # Log which keys are missing
            if our_key not in our_intermediates:
                logger.debug(f"{stage}: Not in our implementation")
            if ref_key not in ref_intermediates:
                logger.debug(f"{stage}: Not in reference implementation")

    return {
        "results": results,
        "first_divergence": first_divergence,
        "our_keys": list(our_intermediates.keys()),
        "ref_keys": list(ref_intermediates.keys()),
    }


@pytest.mark.parametrize("layer_idx", [3])
def test_intermediate_comparison(layer_idx):
    """
    Pytest test to compare intermediate tensors between our and reference implementations.
    """
    logger.info("=" * 60)
    logger.info("Systematic Intermediate Tensor Comparison")
    logger.info("=" * 60)

    # Configuration
    model_path = Path(
        "/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52"
    )
    config_path = "/home/ntarafdar/tt-moe/tt-metal/models/tt-moe/configs/deepseek_v3.json"

    # Load HF config
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Test parameters
    MODE = "decode"
    BATCH_SIZE = 32
    SEQ_LEN = 1
    HIDDEN_SIZE = hf_config.hidden_size

    # Initialize mesh device
    logger.info("Initializing mesh device...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))

    try:
        from utils.ccl import CCL

        ccl = CCL(mesh_device)

        # Generate test input
        torch.manual_seed(42)  # For reproducibility
        torch_input = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16)
        logger.info(f"Input shape: {torch_input.shape}")

        # Load state dict
        logger.info("Loading state dict...")
        state_dict = load_state_dict(model_path, "")

        # Convert input to TTNN format
        torch_input_reshaped = torch_input.permute(1, 0, 2).unsqueeze(0)
        tt_input = ttnn.from_torch(
            torch_input_reshaped,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        # Run our implementation
        logger.info("\n" + "=" * 40)
        logger.info("Running OUR implementation...")
        logger.info("=" * 40)
        our_output, our_intermediates = run_our_implementation(
            tt_input, mesh_device, ccl, config_path, state_dict, layer_idx
        )

        # Run reference implementation
        logger.info("\n" + "=" * 40)
        logger.info("Running REFERENCE implementation...")
        logger.info("=" * 40)
        ref_output, ref_intermediates = run_reference_implementation(
            tt_input, mesh_device, ccl, hf_config, state_dict, layer_idx
        )

        # Compare intermediates
        comparison_results = compare_intermediates(our_intermediates, ref_intermediates, mesh_device)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)

        results = comparison_results["results"]
        first_divergence = comparison_results["first_divergence"]

        if results:
            # Find worst PCC
            worst_stage = min(results, key=lambda x: x["pcc"])
            logger.info(f"Worst PCC: {worst_stage['pcc']:.6f} at stage '{worst_stage['stage']}'")

            if first_divergence:
                logger.info(f"First divergence (PCC < 0.98) at stage: '{first_divergence}'")
                logger.info("\n⚠️ This is where the numerical divergence begins!")

                # For pytest assertion
                divergent_result = next((r for r in results if r["stage"] == first_divergence), None)
                if divergent_result:
                    pytest.fail(
                        f"PCC divergence at stage '{first_divergence}': "
                        f"PCC={divergent_result['pcc']:.6f} (required >= 0.98)"
                    )
            else:
                logger.info("✅ All stages have PCC >= 0.98")

            # Print all results
            logger.info("\nAll PCCs:")
            for result in results:
                status = "✅" if result["pcc"] >= 0.98 else "❌"
                logger.info(f"  {status} {result['stage']}: {result['pcc']:.6f}")
        else:
            logger.warning("No intermediate stages could be compared!")
            logger.info(f"Our keys: {comparison_results['our_keys']}")
            logger.info(f"Ref keys: {comparison_results['ref_keys']}")

        # Compare final outputs
        logger.info("\n" + "=" * 60)
        logger.info("FINAL OUTPUT COMPARISON")
        logger.info("=" * 60)

        final_pcc = compute_pcc(our_output, ref_output)
        logger.info(f"Final output PCC: {final_pcc:.6f}")

        # Cleanup
        ttnn.deallocate(our_output)
        ttnn.deallocate(ref_output)
        ttnn.deallocate(tt_input)

        # Assert final PCC meets threshold
        assert final_pcc >= 0.98, f"Final PCC {final_pcc:.6f} below threshold 0.98"

    finally:
        # Close device
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    logger.info("\nComparison test complete!")


@pytest.mark.parametrize("layer_idx", [3])
def test_save_reference_intermediates(layer_idx):
    """
    Test that calls run_reference_implementation and stores the intermediates to files.
    """
    logger.info("=" * 60)
    logger.info("Saving Reference Implementation Intermediates")
    logger.info("=" * 60)

    # Configuration
    model_path = Path(
        "/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52"
    )

    # Load HF config
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Test parameters
    MODE = "decode"
    BATCH_SIZE = 32
    SEQ_LEN = 1
    HIDDEN_SIZE = hf_config.hidden_size

    # Output directory for intermediates
    output_dir = Path("/tmp/reference_intermediates")
    output_dir.mkdir(exist_ok=True)

    # Initialize mesh device
    logger.info("Initializing mesh device...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))

    try:
        from utils.ccl import CCL

        ccl = CCL(mesh_device)

        # Generate test input
        torch.manual_seed(42)  # For reproducibility
        torch_input = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16)
        logger.info(f"Input shape: {torch_input.shape}")

        # Save input (convert bfloat16 to float32 for numpy)
        input_path = output_dir / "torch_input.npy"
        np.save(input_path, torch_input.float().cpu().numpy())
        logger.info(f"Saved input to {input_path}")

        # Load state dict
        logger.info("Loading state dict...")
        state_dict = load_state_dict(model_path, "")

        # Convert input to TTNN format
        torch_input_reshaped = torch_input.permute(1, 0, 2).unsqueeze(0)
        tt_input = ttnn.from_torch(
            torch_input_reshaped,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        # Run reference implementation
        logger.info("\nRunning REFERENCE implementation...")
        ref_output, ref_intermediates = run_reference_implementation(
            tt_input, mesh_device, ccl, hf_config, state_dict, layer_idx
        )

        logger.info(f"\nCaptured {len(ref_intermediates)} intermediate tensors")

        # Save all intermediates
        saved_count = 0
        for key, tensor in ref_intermediates.items():
            if tensor is not None:
                filepath = output_dir / f"{key}.npy"
                try:
                    # Convert TTNN tensor if needed
                    if str(type(tensor)).find("ttnn") != -1:
                        try:
                            tensor = ttnn.to_torch(
                                tensor, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1))
                            )
                        except:
                            try:
                                tensor = ttnn.to_torch(
                                    tensor,
                                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                                        mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape
                                    ),
                                )
                            except:
                                tensor = ttnn.to_torch(tensor)

                    # Convert to numpy
                    if torch.is_tensor(tensor):
                        if tensor.dtype == torch.bfloat16:
                            tensor = tensor.float()
                        tensor = tensor.cpu().numpy()

                    # Save
                    np.save(filepath, tensor)
                    logger.info(f"  Saved {key}.npy: shape={tensor.shape}")
                    saved_count += 1
                except Exception as e:
                    logger.error(f"  Failed to save {key}: {e}")

        # Save final output
        if ref_output is not None:
            filepath = output_dir / "final_output.npy"
            try:
                # Convert TTNN tensor if needed
                if str(type(ref_output)).find("ttnn") != -1:
                    ref_output_torch = ttnn.to_torch(
                        ref_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1))
                    )
                else:
                    ref_output_torch = ref_output

                # Convert to numpy
                if ref_output_torch.dtype == torch.bfloat16:
                    ref_output_torch = ref_output_torch.float()
                ref_output_np = ref_output_torch.cpu().numpy()

                np.save(filepath, ref_output_np)
                logger.info(f"  Saved final_output.npy: shape={ref_output_np.shape}")
                saved_count += 1
            except Exception as e:
                logger.error(f"  Failed to save final_output: {e}")

        logger.info(f"\n✅ Successfully saved {saved_count} tensors to {output_dir}")

        # Cleanup
        if ref_output is not None:
            ttnn.deallocate(ref_output)
        ttnn.deallocate(tt_input)

    finally:
        # Close device
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    # List all saved files
    saved_files = list(output_dir.glob("*.npy"))
    logger.info(f"\nTotal files saved: {len(saved_files)}")
    for f in sorted(saved_files):
        arr = np.load(f)
        logger.info(f"  {f.name}: shape={arr.shape}")

    logger.info("\n✅ Reference intermediates saved successfully!")


if __name__ == "__main__":
    # Can run as regular script or with pytest
    pytest.main([__file__, "-xvs"])
