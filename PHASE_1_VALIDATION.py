#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Integration validation script for Phase 1 Bridge Extension.

This script validates that all modules work together correctly without
requiring pytest to be installed. Run directly with:

    python3 PHASE_1_VALIDATION.py
"""

import sys
import importlib.util
from pathlib import Path
import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))


def load_module(name: str, path: str):
    """Load a module from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_configuration():
    """Test model configuration system."""
    print("\n" + "="*70)
    print("TEST: Model Configuration System")
    print("="*70)

    model_config = load_module("model_config", "comfyui_bridge/model_config.py")

    # Test 1: List available models
    models = model_config.list_available_models()
    print(f"✓ Available models: {models}")
    assert len(models) >= 4, "Should have at least 4 models"

    # Test 2: Get config for each model
    for model_id in ["sdxl", "sd3.5", "sd1.5", "sd1.4"]:
        config = model_config.get_model_config(model_id)
        model_config.validate_config(config)
        latent_channels = model_config.get_latent_channels(model_id)
        clip_dim = model_config.get_clip_dim(model_id)
        print(f"✓ {model_id}: channels={latent_channels}, clip_dim={clip_dim}")

    # Test 3: Invalid model error handling
    try:
        model_config.get_model_config("invalid")
        assert False, "Should raise error"
    except ValueError as e:
        print(f"✓ Error handling for invalid model: {type(e).__name__}")

    print("\n✅ Model Configuration Tests PASSED")


def test_tensor_conversion():
    """Test tensor format conversion."""
    print("\n" + "="*70)
    print("TEST: Tensor Format Conversion")
    print("="*70)

    format_utils = load_module("format_utils", "comfyui_bridge/format_utils.py")
    model_config = load_module("model_config", "comfyui_bridge/model_config.py")

    # Test for each model type
    test_cases = [
        ("sdxl", 4, (1, 4, 128, 128)),
        ("sd3.5", 16, (1, 16, 64, 64)),
        ("sd1.5", 4, (1, 4, 256, 256)),
    ]

    for model_id, channels, shape in test_cases:
        print(f"\nTesting {model_id}...")

        # Create tensor
        tensor = torch.randn(shape)
        print(f"  Original shape: {tensor.shape}")

        # Convert to TT format
        tt_tensor = format_utils.torch_to_tt_format(tensor, expected_channels=channels)
        print(f"  TT format shape: {tt_tensor.shape}")

        # Validate TT format
        assert tt_tensor.shape[1] == 1, "Dimension 1 should be 1"
        assert tt_tensor.shape[3] == channels, f"Dimension 3 should be {channels}"

        # Convert back
        restored = format_utils.tt_to_torch_format(tt_tensor, expected_channels=channels)
        print(f"  Restored shape: {restored.shape}")

        # Check roundtrip
        is_close = torch.allclose(tensor, restored, atol=1e-6)
        print(f"  Roundtrip match: {is_close}")
        assert is_close, "Roundtrip conversion should match"

        print(f"  ✓ {model_id} conversion verified")

    print("\n✅ Tensor Conversion Tests PASSED")


def test_session_management():
    """Test session management."""
    print("\n" + "="*70)
    print("TEST: Session Management")
    print("="*70)

    session_manager = load_module("session_manager", "comfyui_bridge/session_manager.py")

    manager = session_manager.SessionManager(timeout_seconds=120)

    # Test 1: Create session
    session_id = manager.create_session("sdxl", total_steps=20)
    print(f"✓ Created session: {session_id}")

    # Test 2: Retrieve session
    session = manager.get_session(session_id)
    assert session is not None
    assert session.model_id == "sdxl"
    assert session.total_steps == 20
    print(f"✓ Retrieved session: model_id={session.model_id}, steps={session.total_steps}")

    # Test 3: Validate session
    is_valid = manager.is_session_valid(session_id)
    assert is_valid
    print(f"✓ Session is valid: {is_valid}")

    # Test 4: Update activity
    initial_step = session.current_step
    manager.update_activity(session_id)
    updated_session = manager.get_session(session_id)
    assert updated_session.current_step == initial_step + 1
    print(f"✓ Activity updated: step incremented to {updated_session.current_step}")

    # Test 5: Multiple sessions
    session_ids = []
    for i in range(5):
        sid = manager.create_session(f"model_{i}", total_steps=20)
        session_ids.append(sid)

    count = manager.get_session_count()
    print(f"✓ Created multiple sessions: {count} active")
    assert count == 6  # 1 original + 5 new

    # Test 6: Complete session
    stats = manager.complete_session(session_id)
    assert "duration_seconds" in stats
    assert "total_steps" in stats
    print(f"✓ Completed session: {stats['total_steps']} steps, {stats['duration_seconds']:.3f}s")

    print("\n✅ Session Management Tests PASSED")


def test_per_step_handlers():
    """Test per-step handlers."""
    print("\n" + "="*70)
    print("TEST: Per-Step Handlers")
    print("="*70)

    handlers_module = load_module("handlers_per_step", "comfyui_bridge/handlers_per_step.py")

    handlers = handlers_module.PerStepHandlers(
        model_registry={"sdxl": None, "sd1.5": None},
        scheduler_registry={"euler": None, "karras": None}
    )

    # Test 1: Create session
    create_result = handlers.handle_session_create({
        "model_id": "sdxl",
        "total_steps": 20,
        "seed": 42,
        "cfg_scale": 7.5
    })
    assert create_result["status"] == "created"
    session_id = create_result["session_id"]
    print(f"✓ Session created: {session_id}")

    # Test 2: Query session status
    status_result = handlers.handle_session_status({"session_id": session_id})
    assert status_result["status"] == "active"
    print(f"✓ Session status: {status_result['progress']}")

    # Test 3: Denoise single step
    latents = torch.randn(1, 4, 64, 64).tolist()
    step_result = handlers.handle_denoise_step_single({
        "session_id": session_id,
        "timestep": 500,
        "step_index": 0,
        "total_steps": 20,
        "latents": latents,
        "positive_cond": {"embeddings": None},
        "negative_cond": {"embeddings": None},
        "cfg_scale": 7.5
    })
    assert step_result["status"] == "completed"
    print(f"✓ Step completed: timestep={step_result['step_metadata']['timestep']}")

    # Test 4: Multi-step loop
    for step in range(1, 5):
        result = handlers.handle_denoise_step_single({
            "session_id": session_id,
            "timestep": 1000 - (step * 200),
            "step_index": step,
            "total_steps": 20,
            "latents": latents,
            "positive_cond": {"embeddings": None},
            "negative_cond": {"embeddings": None},
            "cfg_scale": 7.5
        })
        assert result["status"] == "completed"

    print(f"✓ Multi-step loop completed (5 steps)")

    # Test 5: Complete session
    complete_result = handlers.handle_session_complete({"session_id": session_id})
    assert complete_result["status"] == "completed"
    assert complete_result["steps_completed"] == 5
    print(f"✓ Session completed: {complete_result['steps_completed']} steps")

    # Test 6: Error handling - invalid model
    error_result = handlers.handle_session_create({
        "model_id": "invalid_model",
        "total_steps": 20
    })
    assert error_result["status"] == "error"
    print(f"✓ Error handling: Invalid model caught")

    print("\n✅ Per-Step Handlers Tests PASSED")


def test_server_integration():
    """Test server registry integration."""
    print("\n" + "="*70)
    print("TEST: Server Registry Integration")
    print("="*70)

    server_module = load_module("server_per_step", "comfyui_bridge/server_per_step.py")
    handlers_module = load_module("handlers_per_step", "comfyui_bridge/handlers_per_step.py")

    # Create handlers
    handlers = handlers_module.PerStepHandlers(
        model_registry={"sdxl": None},
        scheduler_registry={"euler": None}
    )

    # Register with server
    registry = server_module.register_per_step_operations(
        handlers=handlers,
        models={"sdxl": None},
        schedulers={"euler": None}
    )

    print(f"✓ Registered per-step operations")
    print(f"✓ Operations: {list(registry.operations.keys())}")

    # Test dispatch with handlers
    create_response = server_module.handle_per_step_request("session_create", {
        "model_id": "sdxl",
        "total_steps": 20
    })
    print(f"✓ Dispatch create: {create_response['status']}")

    print("\n✅ Server Integration Tests PASSED")


def test_comfyui_nodes_syntax():
    """Test ComfyUI nodes syntax."""
    print("\n" + "="*70)
    print("TEST: ComfyUI Nodes Syntax")
    print("="*70)

    import py_compile

    node_file = Path("/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/tt_sampler_nodes.py")

    try:
        py_compile.compile(str(node_file), doraise=True)
        print(f"✓ tt_sampler_nodes.py: Syntax valid")
    except py_compile.PyCompileError as e:
        print(f"✗ Syntax error: {e}")
        raise

    print("\n✅ ComfyUI Nodes Tests PASSED")


def test_full_workflow():
    """Test complete workflow integration."""
    print("\n" + "="*70)
    print("TEST: Full Workflow Integration")
    print("="*70)

    model_config = load_module("model_config", "comfyui_bridge/model_config.py")
    format_utils = load_module("format_utils", "comfyui_bridge/format_utils.py")
    handlers_module = load_module("handlers_per_step", "comfyui_bridge/handlers_per_step.py")

    # Workflow: sdxl with format conversion and per-step denoising
    handlers = handlers_module.PerStepHandlers(
        model_registry={"sdxl": None},
        scheduler_registry={"euler": None}
    )

    # Get SDXL config
    config = model_config.get_model_config("sdxl")
    channels = config["latent_channels"]
    print(f"✓ Using model: sdxl (channels={channels})")

    # Create session
    create_result = handlers.handle_session_create({
        "model_id": "sdxl",
        "total_steps": 3
    })
    session_id = create_result["session_id"]
    print(f"✓ Session created: {session_id}")

    # Simulate denoising loop
    latents = torch.randn(1, channels, 64, 64)
    print(f"✓ Initial latents: {latents.shape}")

    for step in range(3):
        # Convert to TT format for bridge API
        latents_tt = format_utils.torch_to_tt_format(latents, expected_channels=channels)
        print(f"  Step {step+1}: TT format {latents_tt.shape}", end="")

        # Call handler
        result = handlers.handle_denoise_step_single({
            "session_id": session_id,
            "timestep": 1000 - (step * 333),
            "step_index": step,
            "total_steps": 3,
            "latents": latents.tolist(),
            "positive_cond": {"embeddings": None},
            "negative_cond": {"embeddings": None},
            "cfg_scale": 7.5
        })

        # Convert result back to torch format
        output_latents = torch.tensor(result["latents"])
        latents_restored = format_utils.tt_to_torch_format(
            format_utils.torch_to_tt_format(output_latents, expected_channels=channels),
            expected_channels=channels
        )
        print(f" -> output {latents_restored.shape} ✓")

    # Complete session
    complete_result = handlers.handle_session_complete({"session_id": session_id})
    print(f"✓ Session complete: {complete_result['steps_completed']} steps executed")

    print("\n✅ Full Workflow Integration Tests PASSED")


def main():
    """Run all integration tests."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " " * 15 + "Phase 1: Bridge Extension Integration Tests" + " " * 11 + "║")
    print("╚" + "="*68 + "╝")

    try:
        test_model_configuration()
        test_tensor_conversion()
        test_session_management()
        test_per_step_handlers()
        test_server_integration()
        test_comfyui_nodes_syntax()
        test_full_workflow()

        print("\n")
        print("╔" + "="*68 + "╗")
        print("║" + " " * 20 + "🎉 ALL TESTS PASSED 🎉" + " " * 24 + "║")
        print("╚" + "="*68 + "╝")
        print("\nPhase 1 Bridge Extension implementation is complete and validated!")
        print("\nModules created:")
        print("  • model_config.py - Centralized model configuration")
        print("  • format_utils.py - Tensor format conversion")
        print("  • session_manager.py - Thread-safe session lifecycle management")
        print("  • handlers_per_step.py - Per-step denoising handlers")
        print("  • server_per_step.py - Server integration and routing")
        print("  • tt_sampler_nodes.py - ComfyUI nodes for per-step sampling")
        print("  • test_per_step.py - Comprehensive pytest test suite")
        print("\nReady for Phase 1.5: Bridge integration with native code")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
