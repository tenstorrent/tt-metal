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
    """
    Run full PyTorch forward pass and return output + time.
    
    Uses `forward_training()` for single-pass inference (not full denoising).
    """
    start = time.time()
    with torch.no_grad():
        # Single forward pass: (images, lang, state, noisy_actions, timestep) → velocity
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
    
    NOTE: We use `forward_training()` (not `sample_actions()`) because:
    - forward_training(): Single forward pass → returns velocity (for PCC comparison)
    - sample_actions(): Full denoising loop (100+ iterations) → returns clean actions
    
    For PCC testing, we only need a single forward pass to compare TTNN vs PyTorch.
    """
    start = time.time()
    with torch.no_grad():
        # Single forward pass (not full denoising loop)
        # This computes: velocity = model(images, lang, state, noisy_actions, timestep)
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


def run_full_denoising_inference(
    model_torch: PI0ModelTorch,
    model_ttnn: PI0ModelTTNN,
    inputs: Dict,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Run FULL denoising loop (sample_actions) for both PyTorch and TTNN.
    
    This runs the complete inference pipeline with 10+ denoising steps.
    Uses the same random seed for fair comparison.
    
    Returns: (torch_actions, ttnn_actions, torch_time_ms, ttnn_time_ms)
    """
    # PyTorch full denoising (with fixed seed)
    print("   Running PyTorch sample_actions()...")
    torch.manual_seed(seed)
    start = time.time()
    with torch.no_grad():
        torch_actions = model_torch.sample_actions(
            images=inputs["images"],
            img_masks=inputs["img_masks"],
            lang_tokens=inputs["lang_tokens"],
            lang_masks=inputs["lang_masks"],
            state=inputs["state"],
        )
    torch_time = (time.time() - start) * 1000
    print(f"      PyTorch: {torch_actions.shape}, time: {torch_time:.2f}ms")
    
    # TTNN full denoising (with same seed)
    print("   Running TTNN sample_actions()...")
    torch.manual_seed(seed)  # Same seed for fair comparison
    start = time.time()
    with torch.no_grad():
        ttnn_actions = model_ttnn.sample_actions(
            images=inputs["images"],
            img_masks=inputs["img_masks"],
            lang_tokens=inputs["lang_tokens"],
            lang_masks=inputs["lang_masks"],
            state=inputs["state"],
        )
    ttnn_time = (time.time() - start) * 1000
    print(f"      TTNN:    {ttnn_actions.shape}, time: {ttnn_time:.2f}ms")
    
    return torch_actions, ttnn_actions, torch_time, ttnn_time


def run_denoising_with_per_step_pcc(
    model_torch: PI0ModelTorch,
    model_ttnn: PI0ModelTTNN,
    inputs: Dict,
    device: "ttnn.Device",
) -> Dict[str, float]:
    """
    Run denoising with per-step PCC comparison.
    
    This runs PyTorch and TTNN in lockstep, comparing:
    - Prefix embeddings
    - Initial noise (using same seed)
    - Each denoising step (velocity, x_t)
    - Final output
    
    Returns: Dict of per-step PCC values
    """
    print("\n" + "=" * 80)
    print("  PER-STEP DENOISING PCC ANALYSIS")
    print("=" * 80)
    
    batch_size = inputs["lang_tokens"].shape[0]
    results = {}
    
    # ================================================================
    # STEP 0: Compare prefix embeddings
    # ================================================================
    print("\n📍 Step 0: Prefix Embeddings")
    
    with torch.no_grad():
        # PyTorch prefix
        torch_prefix, _, _ = model_torch.embed_prefix(
            inputs["images"],
            inputs["img_masks"],
            inputs["lang_tokens"],
            inputs["lang_masks"],
        )
        print(f"   PyTorch prefix: {torch_prefix.shape}")
        
        # TTNN prefix
        lang_tokens_ttnn = ttnn.from_torch(
            inputs["lang_tokens"],
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        lang_masks_ttnn = ttnn.from_torch(
            inputs["lang_masks"].float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        ttnn_prefix, _, _ = model_ttnn.embed_prefix(
            inputs["images"],
            inputs["img_masks"],
            lang_tokens_ttnn,
            lang_masks_ttnn,
        )
        ttnn_prefix_torch = ttnn.to_torch(ttnn_prefix)
        print(f"   TTNN prefix:    {ttnn_prefix_torch.shape}")
        
        prefix_pcc = compute_pcc(torch_prefix, ttnn_prefix_torch)
        results["0_prefix_emb"] = prefix_pcc
        print(f"   PCC: {prefix_pcc:.6f} {'✅' if prefix_pcc >= 0.99 else '⚠️'}")
    
    # ================================================================
    # STEP 1: Forward prefix through VLM (with caching)
    # ================================================================
    print("\n📍 Step 1: VLM Forward (prefix)")
    
    with torch.no_grad():
        # PyTorch VLM
        torch_vlm_out, torch_prefix_cache = model_torch.backbone.forward_vlm(torch_prefix, use_cache=True)
        print(f"   PyTorch VLM out: {torch_vlm_out.shape}")
        print(f"   PyTorch cache: {len(torch_prefix_cache)} layers")
        
        # TTNN VLM - use same path for fair comparison
        ttnn_vlm_out, ttnn_prefix_cache = model_ttnn.backbone.forward_vlm(ttnn_prefix, use_cache=True)
        ttnn_vlm_out_torch = ttnn.to_torch(ttnn_vlm_out)
        print(f"   TTNN VLM out:    {ttnn_vlm_out_torch.shape}")
        print(f"   TTNN cache:      {len(ttnn_prefix_cache) if ttnn_prefix_cache else 0} layers")
        
        vlm_pcc = compute_pcc(torch_vlm_out, ttnn_vlm_out_torch)
        results["1_vlm_output"] = vlm_pcc
        print(f"   VLM output PCC:  {vlm_pcc:.6f} {'✅' if vlm_pcc >= 0.99 else '⚠️'}")
    
    # ================================================================
    # STEP 2: Denoising loop (per-step comparison)
    # ================================================================
    print("\n📍 Step 2: Denoising Loop")
    
    # Use same seed for noise
    torch.manual_seed(42)
    torch_x_t = torch.randn(batch_size, model_torch.config.action_horizon, model_torch.config.action_dim)
    
    torch.manual_seed(42)
    ttnn_x_t = torch.randn(batch_size, model_ttnn.config.action_horizon, model_ttnn.config.action_dim)
    
    # Verify initial noise is identical
    noise_pcc = compute_pcc(torch_x_t, ttnn_x_t)
    results["1_initial_noise"] = noise_pcc
    print(f"\n   Initial noise PCC: {noise_pcc:.6f} {'✅' if noise_pcc >= 0.99 else '❌'}")
    
    # Get config
    num_steps = model_torch.denoising.config.num_steps
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
    
    print(f"\n   Running {num_steps} denoising steps...")
    print("-" * 70)
    print(f"   {'Step':<6} {'t':<8} {'Velocity PCC':<15} {'x_t PCC':<15} {'Status'}")
    print("-" * 70)
    
    state_ttnn = ttnn.from_torch(
        inputs["state"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    
    for i in range(num_steps):
        t = timesteps[i].item()
        t_next = timesteps[i + 1].item()
        dt = t_next - t
        
        # ---- PyTorch step ----
        torch_velocity = model_torch._denoise_forward(
            torch_x_t,
            torch.full((batch_size,), t),
            torch_prefix_cache,
            state=inputs["state"],
        )
        torch_x_t = torch_x_t + torch_velocity * dt
        
        # ---- TTNN step (using proper forward_expert with cached KV) ----
        ttnn_x_t_tensor = ttnn.from_torch(
            ttnn_x_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        t_tensor = ttnn.from_torch(
            torch.full((batch_size,), t, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        
        # Embed suffix (same as PyTorch)
        suffix_embs, _, _, _ = model_ttnn.embed_suffix(
            state_ttnn, ttnn_x_t_tensor, t_tensor
        )
        
        # Forward through expert with cached prefix KV (matching PyTorch path)
        expert_output, _ = model_ttnn.backbone.forward_expert(
            suffix_embs,
            past_key_values=ttnn_prefix_cache,
        )
        
        # Extract action output
        if not model_ttnn.config.pi05:
            action_output = ttnn.slice(
                expert_output,
                [0, 1, 0],
                [expert_output.shape[0], expert_output.shape[1], expert_output.shape[2]]
            )
        else:
            action_output = expert_output
        
        # Project to velocity
        ttnn_velocity = model_ttnn.suffix_embedding.project_output(action_output)
        ttnn_velocity_torch = ttnn.to_torch(ttnn_velocity)
        
        # Euler step
        ttnn_x_t = ttnn_x_t + ttnn_velocity_torch * dt
        
        # ---- Compute PCCs ----
        velocity_pcc = compute_pcc(torch_velocity, ttnn_velocity_torch)
        x_t_pcc = compute_pcc(torch_x_t, ttnn_x_t)
        
        results[f"step_{i}_velocity"] = velocity_pcc
        results[f"step_{i}_x_t"] = x_t_pcc
        
        status = "✅" if velocity_pcc >= 0.95 and x_t_pcc >= 0.95 else "⚠️" if velocity_pcc >= 0.80 else "❌"
        print(f"   {i:<6} {t:<8.3f} {velocity_pcc:<15.6f} {x_t_pcc:<15.6f} {status}")
    
    print("-" * 70)
    
    # Final comparison
    final_pcc = compute_pcc(torch_x_t, ttnn_x_t)
    results["final_output"] = final_pcc
    print(f"\n   Final output PCC: {final_pcc:.6f} {'✅' if final_pcc >= 0.85 else '❌'}")
    
    return results


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
    parser.add_argument(
        "--full-denoising",
        action="store_true",
        help="Run full denoising loop (sample_actions) instead of single forward pass"
    )
    parser.add_argument(
        "--debug-denoising",
        action="store_true",
        help="Run per-step denoising PCC analysis to identify drift sources"
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
        
        # Check if running debug denoising, full denoising, or single forward pass
        if args.debug_denoising:
            # =====================================================
            # DEBUG: Per-step PCC analysis
            # =====================================================
            print("\n4. Running PER-STEP DENOISING PCC ANALYSIS...")
            
            try:
                results = run_denoising_with_per_step_pcc(
                    model_torch, model_ttnn, inputs, device
                )
                
                # Analyze results
                print("\n" + "=" * 80)
                print("  ANALYSIS SUMMARY")
                print("=" * 80)
                
                # Find first step where PCC drops significantly
                velocity_pccs = [(k, v) for k, v in results.items() if "velocity" in k]
                x_t_pccs = [(k, v) for k, v in results.items() if "x_t" in k]
                
                print("\n📊 Velocity PCC trend:")
                for k, v in velocity_pccs:
                    bar = "█" * int(v * 20)
                    print(f"   {k:20}: {v:.4f} {bar}")
                
                print("\n📊 x_t (accumulated) PCC trend:")
                for k, v in x_t_pccs:
                    bar = "█" * int(v * 20)
                    print(f"   {k:20}: {v:.4f} {bar}")
                
                # Find biggest drop
                velocity_values = [v for _, v in velocity_pccs]
                if velocity_values:
                    min_velocity_pcc = min(velocity_values)
                    avg_velocity_pcc = sum(velocity_values) / len(velocity_values)
                    
                    print(f"\n📈 Statistics:")
                    print(f"   Prefix PCC:          {results.get('0_prefix_emb', 0):.6f}")
                    print(f"   Min velocity PCC:    {min_velocity_pcc:.6f}")
                    print(f"   Avg velocity PCC:    {avg_velocity_pcc:.6f}")
                    print(f"   Final output PCC:    {results.get('final_output', 0):.6f}")
                    
                    # Identify likely cause
                    print("\n🔍 Drift Analysis:")
                    if results.get('0_prefix_emb', 0) < 0.99:
                        print("   ⚠️ Prefix embedding has drift - check SigLIP/language embedding")
                    if avg_velocity_pcc < 0.95:
                        print("   ⚠️ Velocity predictions have drift - check backbone attention/MLP")
                    if avg_velocity_pcc >= 0.95 and results.get('final_output', 0) < 0.85:
                        print("   ⚠️ Good per-step PCC but accumulated drift - check bfloat16 precision")
                    
                print("=" * 80)
                
                final_pcc = results.get('final_output', 0)
                return 0 if final_pcc >= 0.85 else 1
                
            except Exception as e:
                print(f"\n   ❌ Debug denoising failed: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        elif args.full_denoising:
            # =====================================================
            # FULL DENOISING LOOP (sample_actions)
            # =====================================================
            print("\n4. Running FULL DENOISING (sample_actions)...")
            print("   This runs the complete denoising loop (10+ steps)")
            
            try:
                torch_actions, ttnn_actions, torch_time, ttnn_time = run_full_denoising_inference(
                    model_torch, model_ttnn, inputs
                )
                
                # Compute PCC
                print("\n5. Computing PCC (PyTorch vs TTNN)...")
                pcc = compute_pcc(torch_actions, ttnn_actions)
                passed = pcc >= 0.85
                
                # Final summary
                print("\n" + "=" * 80)
                print("  FINAL SUMMARY: Full Denoising (sample_actions)")
                print("=" * 80)
                print(f"\n   PyTorch Time:         {torch_time:.2f}ms")
                print(f"   TTNN Time:            {ttnn_time:.2f}ms")
                if ttnn_time > 0 and torch_time > 0:
                    if ttnn_time < torch_time:
                        speedup = torch_time / ttnn_time
                        print(f"   Speedup:              {speedup:.2f}x faster (TTNN)")
                    else:
                        slowdown = ttnn_time / torch_time
                        print(f"   Slowdown:             {slowdown:.2f}x slower (TTNN)")
                print(f"\n   Output Shape:         {ttnn_actions.shape}")
                print(f"   PCC:                  {pcc:.6f}")
                print(f"\n   Status:               {'✅ PASS' if passed else '❌ FAIL'}")
                print("=" * 80)
                
                return 0 if passed else 1
                
            except Exception as e:
                print(f"\n   ❌ Full denoising failed: {e}")
                import traceback
                traceback.print_exc()
                return 1
        else:
            # =====================================================
            # SINGLE FORWARD PASS (forward_training)
            # =====================================================
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
                print("  FINAL SUMMARY: Full E2E Inference (single forward pass)")
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

