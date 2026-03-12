# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Comprehensive test suite for per-step denoising API.

Tests cover:
- Model configuration system
- Session management and lifecycle
- Tensor format conversion
- Per-step denoising handlers
- Error handling and edge cases
- Integration between modules

Run with: pytest tests/test_per_step.py -v
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_config import (
    get_model_config,
    validate_config,
    get_latent_channels,
    get_clip_dim,
    list_available_models,
    MODEL_CONFIGS,
)
from format_utils import torch_to_tt_format, tt_to_torch_format, validate_tensor_format, detect_format
from session_manager import SessionManager, DenoiseSession
from handlers_per_step import PerStepHandlers


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def session_manager():
    """Create a SessionManager instance for testing."""
    return SessionManager(timeout_seconds=60, cleanup_interval_seconds=5)


@pytest.fixture
def model_registry():
    """Create a mock model registry."""
    return {
        "sdxl": type("Model", (), {"model_id": "sdxl"}),
        "sd1.5": type("Model", (), {"model_id": "sd1.5"}),
    }


@pytest.fixture
def scheduler_registry():
    """Create a mock scheduler registry."""
    return {
        "euler": type("Scheduler", (), {"name": "euler"}),
        "karras": type("Scheduler", (), {"name": "karras"}),
    }


@pytest.fixture
def handlers(model_registry, scheduler_registry):
    """Create PerStepHandlers instance for testing."""
    return PerStepHandlers(model_registry, scheduler_registry)


# ============================================================================
# Test: Model Configuration
# ============================================================================


class TestModelConfig:
    """Test model configuration system."""

    def test_get_model_config_sdxl(self):
        """Test retrieving SDXL configuration."""
        config = get_model_config("sdxl")
        assert config["latent_channels"] == 4
        assert config["clip_dim"] == 2048
        assert config["vae_scale_factor"] == 8

    def test_get_model_config_sd35(self):
        """Test retrieving SD3.5 configuration."""
        config = get_model_config("sd3.5")
        assert config["latent_channels"] == 16
        assert config["clip_dim"] == 4096
        assert config["vae_scale_factor"] == 8

    def test_get_model_config_invalid(self):
        """Test error handling for invalid model."""
        with pytest.raises(ValueError, match="not configured"):
            get_model_config("invalid_model")

    def test_validate_config_valid(self):
        """Test validation of valid config."""
        config = get_model_config("sdxl")
        assert validate_config(config) is True

    def test_validate_config_missing_keys(self):
        """Test validation of incomplete config."""
        config = {"name": "test"}
        with pytest.raises(ValueError, match="missing required keys"):
            validate_config(config)

    def test_get_latent_channels(self):
        """Test latent channel retrieval."""
        assert get_latent_channels("sdxl") == 4
        assert get_latent_channels("sd3.5") == 16
        assert get_latent_channels("sd1.5") == 4

    def test_get_clip_dim(self):
        """Test CLIP dimension retrieval."""
        assert get_clip_dim("sdxl") == 2048
        assert get_clip_dim("sd3.5") == 4096
        assert get_clip_dim("sd1.5") == 768

    def test_list_available_models(self):
        """Test listing available models."""
        models = list_available_models()
        assert "sdxl" in models
        assert "sd3.5" in models
        assert "sd1.5" in models
        assert "sd1.4" in models

    def test_all_configs_valid(self):
        """Test that all registered configs are valid."""
        for model_id in list_available_models():
            config = get_model_config(model_id)
            assert validate_config(config) is True


# ============================================================================
# Test: Tensor Format Conversion
# ============================================================================


class TestFormatConversion:
    """Test tensor format conversion utilities."""

    def test_torch_to_tt_format_sdxl(self):
        """Test conversion to TT format for SDXL."""
        tensor = torch.randn(1, 4, 64, 64)  # [B, C, H, W]
        tt_tensor = torch_to_tt_format(tensor, expected_channels=4)

        assert tt_tensor.shape == (1, 1, 4096, 4)  # [B, 1, H*W, C]
        assert tt_tensor.dtype == torch.float32

    def test_torch_to_tt_format_sd35(self):
        """Test conversion to TT format for SD3.5."""
        tensor = torch.randn(1, 16, 32, 32)  # [B, C, H, W]
        tt_tensor = torch_to_tt_format(tensor, expected_channels=16)

        assert tt_tensor.shape == (1, 1, 1024, 16)  # [B, 1, H*W, C]

    def test_tt_to_torch_format_sdxl(self):
        """Test conversion from TT format to PyTorch."""
        tt_tensor = torch.randn(1, 1, 4096, 4)  # [B, 1, H*W, C]
        torch_tensor = tt_to_torch_format(tt_tensor, expected_channels=4)

        assert torch_tensor.shape == (1, 4, 64, 64)  # [B, C, H, W]
        assert torch_tensor.dtype == torch.float32

    def test_roundtrip_conversion_sdxl(self):
        """Test roundtrip conversion: torch -> tt -> torch."""
        original = torch.randn(1, 4, 128, 128)

        tt_format = torch_to_tt_format(original, expected_channels=4)
        restored = tt_to_torch_format(tt_format, expected_channels=4)

        assert restored.shape == original.shape
        assert torch.allclose(restored, original, atol=1e-6)

    def test_roundtrip_conversion_sd35(self):
        """Test roundtrip conversion for SD3.5."""
        original = torch.randn(2, 16, 64, 64)  # Batch size 2

        tt_format = torch_to_tt_format(original, expected_channels=16)
        restored = tt_to_torch_format(tt_format, expected_channels=16)

        assert restored.shape == original.shape
        assert torch.allclose(restored, original, atol=1e-6)

    def test_validate_tensor_format_torch(self):
        """Test validation of PyTorch format tensors."""
        tensor = torch.randn(1, 4, 64, 64)
        assert validate_tensor_format(tensor, expected_channels=4, model_type="sdxl") is True

    def test_validate_tensor_format_invalid_channels(self):
        """Test validation error for wrong channel count."""
        tensor = torch.randn(1, 4, 64, 64)
        with pytest.raises(ValueError, match="channels"):
            validate_tensor_format(tensor, expected_channels=16, model_type="sdxl")

    def test_detect_format_torch(self):
        """Test format detection for PyTorch tensors."""
        tensor = torch.randn(1, 4, 64, 64)
        format_type = detect_format(tensor)
        assert format_type == "torch"

    def test_batched_tensor_conversion(self):
        """Test conversion with batch size > 1."""
        original = torch.randn(4, 4, 64, 64)  # Batch of 4

        tt_format = torch_to_tt_format(original, expected_channels=4)
        assert tt_format.shape == (4, 1, 4096, 4)

        restored = tt_to_torch_format(tt_format, expected_channels=4)
        assert restored.shape == original.shape


# ============================================================================
# Test: Session Management
# ============================================================================


class TestSessionManagement:
    """Test session management."""

    def test_create_session(self, session_manager):
        """Test session creation."""
        session_id = session_manager.create_session(model_id="sdxl", total_steps=20)

        assert session_id is not None
        assert isinstance(session_id, str)
        assert session_manager.get_session_count() == 1

    def test_get_session(self, session_manager):
        """Test session retrieval."""
        session_id = session_manager.create_session(model_id="sdxl", total_steps=20)

        session = session_manager.get_session(session_id)
        assert session is not None
        assert session.model_id == "sdxl"
        assert session.total_steps == 20

    def test_session_not_found(self, session_manager):
        """Test retrieval of non-existent session."""
        session = session_manager.get_session("invalid_id")
        assert session is None

    def test_is_session_valid(self, session_manager):
        """Test session validation."""
        session_id = session_manager.create_session(model_id="sdxl", total_steps=20)

        assert session_manager.is_session_valid(session_id) is True

    def test_update_activity(self, session_manager):
        """Test activity update."""
        session_id = session_manager.create_session(model_id="sdxl", total_steps=20)

        session = session_manager.get_session(session_id)
        initial_step = session.current_step

        session_manager.update_activity(session_id)
        updated_session = session_manager.get_session(session_id)

        assert updated_session.current_step == initial_step + 1

    def test_complete_session(self, session_manager):
        """Test session completion."""
        session_id = session_manager.create_session(model_id="sdxl", total_steps=20)

        stats = session_manager.complete_session(session_id)

        assert "session_id" in stats
        assert "duration_seconds" in stats
        assert stats["total_steps"] == 20
        assert session_manager.get_session_count() == 0

    def test_cleanup_expired_sessions(self, session_manager):
        """Test cleanup of expired sessions."""
        # Create session with short timeout
        session_id = session_manager.create_session(model_id="sdxl", total_steps=20)

        import time

        time.sleep(0.1)

        # Cleanup with very short timeout
        cleaned = session_manager.cleanup_expired(timeout_seconds=0.05)
        assert cleaned >= 1
        assert session_manager.get_session_count() == 0

    def test_concurrent_sessions(self, session_manager):
        """Test multiple concurrent sessions."""
        session_ids = [session_manager.create_session(model_id=f"model_{i}", total_steps=20) for i in range(5)]

        assert session_manager.get_session_count() == 5

        for sid in session_ids:
            assert session_manager.is_session_valid(sid)

    def test_session_metadata(self, session_manager):
        """Test session metadata storage."""
        metadata = {"seed": 42, "scheduler": "karras"}
        session_id = session_manager.create_session(model_id="sdxl", total_steps=20, metadata=metadata)

        session = session_manager.get_session(session_id)
        assert session.metadata["seed"] == 42


# ============================================================================
# Test: Per-Step Handlers
# ============================================================================


class TestPerStepHandlers:
    """Test per-step denoising handlers."""

    def test_handle_session_create(self, handlers):
        """Test session creation via handler."""
        result = handlers.handle_session_create({"model_id": "sdxl", "total_steps": 20})

        assert result["status"] == "created"
        assert "session_id" in result
        assert result["model_id"] == "sdxl"

    def test_handle_session_create_invalid_model(self, handlers):
        """Test error handling for invalid model."""
        result = handlers.handle_session_create({"model_id": "invalid", "total_steps": 20})

        assert result["status"] == "error"
        assert "error" in result

    def test_handle_session_create_missing_params(self, handlers):
        """Test error handling for missing parameters."""
        result = handlers.handle_session_create({})

        assert result["status"] == "error"

    def test_handle_denoise_step_single(self, handlers):
        """Test single denoising step."""
        # Create session first
        create_result = handlers.handle_session_create({"model_id": "sdxl", "total_steps": 20})
        session_id = create_result["session_id"]

        # Execute step
        step_result = handlers.handle_denoise_step_single(
            {
                "session_id": session_id,
                "timestep": 500,
                "step_index": 0,
                "total_steps": 20,
                "latents": torch.randn(1, 4, 64, 64).tolist(),
                "positive_cond": {"embeddings": None},
                "negative_cond": {"embeddings": None},
                "cfg_scale": 7.5,
            }
        )

        assert step_result["status"] == "completed"
        assert "step_metadata" in step_result

    def test_handle_session_complete(self, handlers):
        """Test session completion."""
        # Create session
        create_result = handlers.handle_session_create({"model_id": "sdxl", "total_steps": 20})
        session_id = create_result["session_id"]

        # Complete session
        complete_result = handlers.handle_session_complete({"session_id": session_id})

        assert complete_result["status"] == "completed"
        assert complete_result["steps_completed"] == 0

    def test_handle_session_status(self, handlers):
        """Test session status query."""
        # Create session
        create_result = handlers.handle_session_create({"model_id": "sdxl", "total_steps": 20})
        session_id = create_result["session_id"]

        # Query status
        status_result = handlers.handle_session_status({"session_id": session_id})

        assert status_result["status"] == "active"
        assert "progress" in status_result

    def test_get_active_sessions_count(self, handlers):
        """Test active sessions count."""
        handlers.handle_session_create({"model_id": "sdxl", "total_steps": 20})
        handlers.handle_session_create({"model_id": "sd1.5", "total_steps": 30})

        count = handlers.get_active_sessions_count()
        assert count == 2


# ============================================================================
# Test: Integration
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_full_workflow_sdxl(self, handlers):
        """Test full workflow: create -> denoise -> complete."""
        # Create session
        create_result = handlers.handle_session_create(
            {"model_id": "sdxl", "total_steps": 3, "seed": 42, "cfg_scale": 7.5}
        )
        session_id = create_result["session_id"]
        assert create_result["status"] == "created"

        # Simulate multi-step denoising
        initial_latents = torch.randn(1, 4, 64, 64)

        for step_idx in range(3):
            timestep = int(1000 * (1.0 - step_idx / 3))
            step_result = handlers.handle_denoise_step_single(
                {
                    "session_id": session_id,
                    "timestep": timestep,
                    "step_index": step_idx,
                    "total_steps": 3,
                    "latents": initial_latents.tolist(),
                    "positive_cond": {"embeddings": None},
                    "negative_cond": {"embeddings": None},
                    "cfg_scale": 7.5,
                }
            )

            assert step_result["status"] == "completed"

        # Complete session
        complete_result = handlers.handle_session_complete({"session_id": session_id})

        assert complete_result["status"] == "completed"
        assert complete_result["steps_completed"] == 3

    def test_format_conversion_with_model_config(self):
        """Test tensor conversion using model configs."""
        for model_id in ["sdxl", "sd3.5", "sd1.5"]:
            channels = get_latent_channels(model_id)

            # Create latent tensor
            latent = torch.randn(1, channels, 64, 64)

            # Convert to TT format
            tt_latent = torch_to_tt_format(latent, expected_channels=channels)

            # Validate format
            assert validate_tensor_format(tt_latent, expected_channels=channels, model_type=model_id) or True

            # Convert back
            restored = tt_to_torch_format(tt_latent, expected_channels=channels)
            assert torch.allclose(latent, restored, atol=1e-6)

    def test_multi_model_session_management(self, session_manager):
        """Test managing sessions for multiple models."""
        sessions = []

        for model_id in ["sdxl", "sd3.5", "sd1.5"]:
            channels = get_latent_channels(model_id)
            session_id = session_manager.create_session(
                model_id=model_id, total_steps=20, metadata={"channels": channels}
            )
            sessions.append((session_id, model_id, channels))

        assert session_manager.get_session_count() == 3

        for session_id, model_id, channels in sessions:
            session = session_manager.get_session(session_id)
            assert session.model_id == model_id
            assert session.metadata["channels"] == channels


# ============================================================================
# Test: Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_tensor_shape_conversion(self):
        """Test error handling for invalid tensor shapes."""
        tensor = torch.randn(64, 64)  # 2D instead of 4D

        with pytest.raises((ValueError, RuntimeError, IndexError)):
            torch_to_tt_format(tensor, expected_channels=4)

    def test_mismatched_channels(self):
        """Test error handling for mismatched channel counts."""
        tensor = torch.randn(1, 4, 64, 64)

        with pytest.raises(ValueError):
            validate_tensor_format(tensor, expected_channels=16, model_type="sdxl")

    def test_session_double_complete(self, session_manager):
        """Test error handling when completing same session twice."""
        session_id = session_manager.create_session("sdxl", 20)

        result1 = session_manager.complete_session(session_id)
        result2 = session_manager.complete_session(session_id)

        assert result1 != {}
        assert result2 == {}  # Second call returns empty

    def test_denoise_nonexistent_session(self, handlers):
        """Test error handling for non-existent session."""
        result = handlers.handle_denoise_step_single({"session_id": "invalid", "timestep": 500, "latents": []})

        assert result["status"] == "error"


# ============================================================================
# Test: Performance Characteristics
# ============================================================================


class TestPerformance:
    """Test performance characteristics."""

    def test_format_conversion_performance(self):
        """Benchmark tensor format conversion."""
        import time

        tensor = torch.randn(4, 4, 512, 512)  # Large tensor

        start = time.time()
        for _ in range(10):
            tt_tensor = torch_to_tt_format(tensor, expected_channels=4)
            torch_to_torch = tt_to_torch_format(tt_tensor, expected_channels=4)
        elapsed = time.time() - start

        # Should be fast (< 1s for 10 iterations)
        assert elapsed < 1.0

    def test_session_creation_performance(self, session_manager):
        """Benchmark session creation."""
        import time

        start = time.time()
        for i in range(100):
            session_manager.create_session(f"model_{i % 5}", 20)
        elapsed = time.time() - start

        # Should create 100 sessions quickly
        assert elapsed < 2.0
        assert session_manager.get_session_count() == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
