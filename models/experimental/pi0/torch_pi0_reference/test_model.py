#!/usr/bin/env python3
"""
Test script for PI-Zero PyTorch reference implementation.

This script tests that the modularized model can be imported and instantiated.
"""

import sys
import pathlib
import torch

# Add the openpi source to path
openpi_src = pathlib.Path(__file__).parent.parent.parent.parent / "src"
if str(openpi_src) not in sys.path:
    sys.path.insert(0, str(openpi_src))

# Add the torch_pi0_reference directory to path
ref_dir = pathlib.Path(__file__).parent
if str(ref_dir) not in sys.path:
    sys.path.insert(0, str(ref_dir))

try:
    from openpi.models import pi0_config
    from openpi.models.model import Observation
    
    # Import our reference implementation
    from torch_pi0 import PI0Pytorch
    
    print("=" * 70)
    print("Testing PI-Zero PyTorch Reference Implementation")
    print("=" * 70)
    
    # Create a simple config
    print("\n1. Creating model config...")
    config = pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=50,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="bfloat16",
        pi05=False,
    )
    print(f"   ✓ Config created: action_dim={config.action_dim}, action_horizon={config.action_horizon}")
    
    # Create model
    print("\n2. Creating model...")
    # Use CPU for now to avoid device mismatch issues
    device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    model = PI0Pytorch(config=config)
    # Move model to device - this also moves all submodules
    model = model.to(device)
    model.eval()
    print(f"   ✓ Model created successfully")
    
    # Verify model is on correct device
    first_param_device = next(model.parameters()).device
    print(f"   Model device: {first_param_device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    # Create dummy observation
    print("\n3. Creating dummy observation...")
    batch_size = 1
    image_shape = (batch_size, 3, 224, 224)
    
    dummy_images = {
        "base_0_rgb": torch.zeros(image_shape, device=device, dtype=torch.float32),
        "left_wrist_0_rgb": torch.zeros(image_shape, device=device, dtype=torch.float32),
        "right_wrist_0_rgb": torch.zeros(image_shape, device=device, dtype=torch.float32),
    }
    
    dummy_image_masks = {
        "base_0_rgb": torch.ones(batch_size, device=device, dtype=torch.bool),
        "left_wrist_0_rgb": torch.ones(batch_size, device=device, dtype=torch.bool),
        "right_wrist_0_rgb": torch.ones(batch_size, device=device, dtype=torch.bool),
    }
    
    dummy_observation = Observation(
        images=dummy_images,
        image_masks=dummy_image_masks,
        state=torch.zeros(batch_size, 32, device=device, dtype=torch.float32),
        tokenized_prompt=torch.zeros(batch_size, 48, device=device, dtype=torch.int32),
        tokenized_prompt_mask=torch.ones(batch_size, 48, device=device, dtype=torch.bool),
    )
    print(f"   ✓ Dummy observation created")
    
    # Test forward pass (training)
    print("\n4. Testing forward pass (training)...")
    dummy_actions = torch.zeros(batch_size, 50, 32, device=device, dtype=torch.float32)
    
    try:
        with torch.no_grad():
            loss = model.forward(dummy_observation, dummy_actions)
        print(f"   ✓ Forward pass successful!")
        print(f"   Loss shape: {loss.shape}")
        print(f"   Loss mean: {loss.mean().item():.6f}")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test inference (sample_actions)
    print("\n5. Testing inference (sample_actions)...")
    try:
        with torch.no_grad():
            actions = model.sample_actions(device, dummy_observation, num_steps=5)
        print(f"   ✓ Inference successful!")
        print(f"   Actions shape: {actions.shape}")
        print(f"   Actions mean: {actions.mean().item():.6f}")
        print(f"   Actions std: {actions.std().item():.6f}")
    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nMake sure you're running from the correct directory and that")
    print("the openpi source is available.")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

