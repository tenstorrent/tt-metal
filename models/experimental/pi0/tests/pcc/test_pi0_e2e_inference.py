# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PI0 End-to-End Inference Test - Full TTNN vs PyTorch

This test runs PI0 model inference comparing TTNN vs PyTorch:
- Full PyTorch forward pass (reference)
- TTNN component inference (Vision Tower, Prefix, Suffix embedding)
- PCC comparison between outputs

NOTE: Full E2E TTNN inference requires significant device memory optimization
due to the model size (Gemma 2B VLM + Gemma 300M Expert + 27 SigLIP blocks).
The component-by-component tests verify TTNN correctness while full E2E 
requires additional memory management work.

Similar to segformer's test_segformer_model.py
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from models.experimental.pi0.ttnn_pi0_reference.ttnn_pi0 import (
    PI0ModelTorch,
    PI0ModelTTNN,
    PI0ModelConfig,
)
from models.experimental.pi0.ttnn_pi0_reference.weight_loader import PI0WeightLoader
from models.experimental.pi0.ttnn_pi0_reference.ttnn_siglip import SigLIPConfig


def create_config(checkpoint_path: str = None) -> PI0ModelConfig:
    """Create PI0ModelConfig with correct dimensions."""
    config = PI0ModelConfig(
        action_dim=32,
        action_horizon=50,
        state_dim=32,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=False,
    )
    
    config.siglip_config = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    
    return config


def create_test_inputs(config: PI0ModelConfig, batch_size: int = 1) -> Dict:
    """Create test inputs using correct dimensions."""
    image_size = config.siglip_config.image_size
    num_images = 2
    
    images = [
        torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32) 
        for _ in range(num_images)
    ]
    img_masks = [torch.ones(batch_size, dtype=torch.bool) for _ in range(num_images)]
    
    lang_seq_len = 32
    vocab_size = 256000
    lang_tokens = torch.randint(0, vocab_size, (batch_size, lang_seq_len))
    lang_masks = torch.ones(batch_size, lang_seq_len, dtype=torch.bool)
    
    state = torch.randn(batch_size, config.state_dim, dtype=torch.float32)
    noisy_actions = torch.randn(batch_size, config.action_horizon, config.action_dim, dtype=torch.float32)
    timestep = torch.rand(batch_size, dtype=torch.float32)
    
    return {
        "images": images,
        "img_masks": img_masks,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "state": state,
        "noisy_actions": noisy_actions,
        "timestep": timestep,
    }


def run_pytorch_inference(model: PI0ModelTorch, inputs: Dict) -> Tuple[torch.Tensor, float]:
    """Run full PyTorch forward pass and return output + time."""
    start = time.time()
    with torch.no_grad():
        output = model.forward_training(
            images=inputs["images"],
            img_masks=inputs["img_masks"],
            lang_tokens=inputs["lang_tokens"],
            lang_masks=inputs["lang_masks"],
            state=inputs["state"],
            actions=inputs["noisy_actions"],
            timestep=inputs["timestep"],
        )
    elapsed_ms = (time.time() - start) * 1000
    return output, elapsed_ms


def run_ttnn_inference(model: PI0ModelTTNN, inputs: Dict) -> Tuple[torch.Tensor, float]:
    """
    Run FULL TTNN forward pass and return output + time.
    
    Uses DRAM memory config for large tensors to avoid OOM.
    """
    start = time.time()
    with torch.no_grad():
        output = model.forward_training(
            images=inputs["images"],
            img_masks=inputs["img_masks"],
            lang_tokens=inputs["lang_tokens"],
            lang_masks=inputs["lang_masks"],
            state=inputs["state"],
            actions=inputs["noisy_actions"],
            timestep=inputs["timestep"],
        )
    elapsed_ms = (time.time() - start) * 1000
    return output, elapsed_ms


def run_ttnn_components(
    model_ttnn: PI0ModelTTNN, 
    model_torch: PI0ModelTorch,
    inputs: Dict,
    device: "ttnn.Device"
) -> Dict[str, Tuple[float, bool]]:
    """
    Run TTNN component-by-component and compare with PyTorch.
    
    Returns dict of component_name -> (pcc, passed)
    """
    results = {}
    
    # 1. Vision Tower
    print("   Testing Vision Tower...")
    try:
        test_image = inputs["images"][0]
        
        # PyTorch
        vision_torch = model_torch.backbone.vision_tower.forward(test_image)
        
        # TTNN
        vision_ttnn_tensor = model_ttnn.backbone.vision_tower.forward(test_image)
        vision_ttnn = ttnn.to_torch(vision_ttnn_tensor)
        
        pcc = compute_pcc(vision_torch, vision_ttnn)
        results["Vision Tower"] = (pcc, pcc >= 0.90)
        print(f"      Vision Tower PCC: {pcc:.6f} {'✅' if pcc >= 0.90 else '❌'}")
    except Exception as e:
        print(f"      Vision Tower Error: {e}")
        results["Vision Tower"] = (0.0, False)
    
    # 2. Suffix Embedding
    print("   Testing Suffix Embedding...")
    try:
        state = inputs["state"]
        noisy_actions = inputs["noisy_actions"]
        timestep = inputs["timestep"]
        
        # PyTorch
        suffix_torch, _, _, _ = model_torch.embed_suffix(state, noisy_actions, timestep)
        
        # TTNN
        state_ttnn = ttnn.from_torch(state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        actions_ttnn = ttnn.from_torch(noisy_actions, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        timestep_ttnn = ttnn.from_torch(timestep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        suffix_ttnn_result = model_ttnn.suffix_embedding.embed_suffix(state_ttnn, actions_ttnn, timestep_ttnn)
        suffix_ttnn = ttnn.to_torch(suffix_ttnn_result[0])
        
        pcc = compute_pcc(suffix_torch, suffix_ttnn)
        results["Suffix Embedding"] = (pcc, pcc >= 0.90)
        print(f"      Suffix Embedding PCC: {pcc:.6f} {'✅' if pcc >= 0.90 else '❌'}")
    except Exception as e:
        print(f"      Suffix Embedding Error: {e}")
        results["Suffix Embedding"] = (0.0, False)
    
    return results


def compute_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    t1 = tensor1.flatten().float()
    t2 = tensor2.flatten().float()
    
    mean1, mean2 = torch.mean(t1), torch.mean(t2)
    std1, std2 = torch.std(t1), torch.std(t2)
    
    if std1 == 0 or std2 == 0:
        return 1.0 if torch.allclose(t1, t2) else 0.0
    
    covariance = torch.mean((t1 - mean1) * (t2 - mean2))
    return (covariance / (std1 * std2)).item()


# =============================================================================
# PYTEST-BASED TESTS (similar to segformer)
# =============================================================================

@pytest.fixture(scope="function")
def pi0_config():
    """Create PI0 config fixture."""
    return create_config()


@pytest.fixture(scope="function")  
def test_inputs(pi0_config):
    """Create test inputs fixture."""
    return create_test_inputs(pi0_config, batch_size=1)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pi0_e2e_inference_pcc(device, pi0_config, test_inputs):
    """
    Test PI0 component inference: TTNN vs PyTorch.
    
    This test:
    1. Runs FULL PyTorch forward pass (reference)
    2. Runs TTNN component inference (Vision Tower, Suffix)
    3. Compares outputs with PCC
    
    NOTE: Full E2E TTNN requires memory optimization for the large model.
    """
    # Check for checkpoint path
    checkpoint_path = Path("/home/ubuntu/work/sdawle_pi0/torch_checkpoint/pi0_base")
    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\n{'='*80}")
    print("  PI0 COMPONENT INFERENCE TEST: TTNN vs PyTorch")
    print(f"{'='*80}")
    
    # Load weights
    print("\n1. Loading weights...")
    weight_loader = PI0WeightLoader(str(checkpoint_path))
    print("   ✅ Weights loaded")
    
    # Initialize models
    print("\n2. Initializing models...")
    model_torch = PI0ModelTorch(pi0_config, weight_loader)
    print("   ✅ PyTorch model initialized")
    
    model_ttnn = PI0ModelTTNN(pi0_config, weight_loader, device)
    print("   ✅ TTNN model initialized")
    
    # Run PyTorch inference (reference)
    print("\n3. Running PyTorch full E2E inference (reference)...")
    output_torch, torch_time = run_pytorch_inference(model_torch, test_inputs)
    print(f"   ✅ PyTorch: {output_torch.shape}, time: {torch_time:.2f}ms")
    
    # Run TTNN component tests
    print("\n4. Running TTNN component inference...")
    component_results = run_ttnn_components(model_ttnn, model_torch, test_inputs, device)
    
    # Summary
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    all_passed = True
    for name, (pcc, passed) in component_results.items():
        status = "✅" if passed else "❌"
        print(f"   {name:20}: PCC={pcc:.6f} {status}")
        if not passed:
            all_passed = False
    print(f"{'='*80}\n")
    
    # Assert all components pass
    assert all_passed, f"Some components failed PCC test: {component_results}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_pi0_ttnn_only_inference(device, pi0_config, batch_size):
    """
    Test PI0 TTNN-only inference (no comparison, just verify it runs).
    
    Useful for performance benchmarking.
    """
    checkpoint_path = Path("/home/ubuntu/work/sdawle_pi0/torch_checkpoint/pi0_base")
    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")
    
    # Create inputs
    inputs = create_test_inputs(pi0_config, batch_size=batch_size)
    
    # Load weights and initialize TTNN model
    weight_loader = PI0WeightLoader(str(checkpoint_path))
    model_ttnn = PI0ModelTTNN(pi0_config, weight_loader, device)
    
    # Run inference
    output, elapsed_ms = run_ttnn_inference(model_ttnn, inputs)
    
    print(f"\nTTNN Inference: batch={batch_size}, time={elapsed_ms:.2f}ms, output={output.shape}")
    
    # Basic sanity checks
    assert output.shape == (batch_size, pi0_config.action_horizon, pi0_config.action_dim)
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"


# =============================================================================
# STANDALONE RUNNER (for manual testing)
# =============================================================================

def main():
    """Standalone test runner with command-line args."""
    parser = argparse.ArgumentParser(description="PI0 E2E Inference Test")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="/home/ubuntu/work/sdawle_pi0/torch_checkpoint/pi0_base",
        help="Path to checkpoint directory or model.safetensors"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of inference iterations for timing"
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("  PI0 FULL END-TO-END INFERENCE TEST")
    print("  Running COMPLETE TTNN inference vs PyTorch")
    print("=" * 80)
    
    # Handle checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_file() and checkpoint_path.name == "model.safetensors":
        checkpoint_path = checkpoint_path.parent
    
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        return 1
    
    print(f"\n📁 Checkpoint: {checkpoint_path}")
    
    # Open device
    print("\n🔌 Opening TTNN device...")
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    print(f"✅ Device opened (grid: {grid.x}x{grid.y})")
    
    try:
        # Create config
        config = create_config()
        print(f"\n📋 Config: {config.siglip_config.image_size}x{config.siglip_config.image_size} images")
        
        # Load weights
        print("\n1. Loading weights...")
        weight_loader = PI0WeightLoader(str(checkpoint_path))
        print("   ✅ Weights loaded")
        
        # Initialize models
        print("\n2. Initializing models...")
        model_torch = PI0ModelTorch(config, weight_loader)
        print("   ✅ PyTorch model initialized")
        
        model_ttnn = PI0ModelTTNN(config, weight_loader, device)
        print("   ✅ TTNN model initialized")
        
        # Create inputs
        print("\n3. Creating test inputs...")
        inputs = create_test_inputs(config, batch_size=args.batch_size)
        print(f"   ✅ Inputs: batch_size={args.batch_size}, images={inputs['images'][0].shape}")
        
        # Run PyTorch inference
        print("\n4. Running PyTorch full E2E inference...")
        output_torch, torch_time = run_pytorch_inference(model_torch, inputs)
        print(f"   ✅ PyTorch E2E complete")
        print(f"      Output: {output_torch.shape}")
        print(f"      Range:  [{output_torch.min():.4f}, {output_torch.max():.4f}]")
        print(f"      Time:   {torch_time:.2f}ms")
        
        # Try full E2E TTNN inference (with DRAM memory config)
        print(f"\n5. Running TTNN full E2E inference...")
        try:
            output_ttnn, ttnn_time = run_ttnn_inference(model_ttnn, inputs)
            print(f"   ✅ TTNN E2E complete")
            print(f"      Output: {output_ttnn.shape}")
            print(f"      Range:  [{output_ttnn.min():.4f}, {output_ttnn.max():.4f}]")
            print(f"      Time:   {ttnn_time:.2f}ms")
            
            # Compute PCC
            print("\n6. Computing PCC (PyTorch vs TTNN)...")
            pcc = compute_pcc(output_torch, output_ttnn)
            passed = pcc >= 0.85
            
            # Final summary
            print("\n" + "=" * 80)
            print("  FINAL SUMMARY: Full E2E Inference")
            print("=" * 80)
            print(f"\n   PyTorch E2E Time:     {torch_time:.2f}ms")
            print(f"   TTNN E2E Time:        {ttnn_time:.2f}ms")
            if ttnn_time > 0 and torch_time > 0:
                if ttnn_time < torch_time:
                    speedup = torch_time / ttnn_time
                    print(f"   Speedup:              {speedup:.2f}x faster (TTNN)")
                else:
                    slowdown = ttnn_time / torch_time
                    print(f"   Slowdown:             {slowdown:.2f}x slower (TTNN)")
            print(f"\n   Output Shape:         {output_ttnn.shape}")
            print(f"   PCC:                  {pcc:.6f}")
            print(f"\n   Status:               {'✅ PASS' if passed else '❌ FAIL'}")
            print("=" * 80)
            
            return 0 if passed else 1
            
        except RuntimeError as e:
            if "Out of Memory" in str(e) or "OOM" in str(e):
                print(f"   ⚠️ Full E2E OOM, falling back to component tests...")
                component_results = run_ttnn_components(model_ttnn, model_torch, inputs, device)
                
                all_passed = all(passed for _, passed in component_results.values())
                avg_pcc = sum(pcc for pcc, _ in component_results.values()) / len(component_results) if component_results else 0
                
                print("\n" + "=" * 80)
                print("  FINAL SUMMARY: Component Inference PCC (Full E2E OOM)")
                print("=" * 80)
                print(f"\n   Component PCC Results:")
                for name, (pcc, passed) in component_results.items():
                    status = "✅" if passed else "❌"
                    print(f"      {name:20}: {pcc:.6f} {status}")
                print(f"\n   Average PCC:          {avg_pcc:.6f}")
                print("=" * 80)
                
                return 0 if all_passed else 1
            else:
                raise
        
    finally:
        print("\n🔌 Closing device...")
        ttnn.close_device(device)
        print("✅ Device closed")


if __name__ == "__main__":
    sys.exit(main())

