#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PCC (Pearson Correlation Coefficient) verification tests for SmolVLA CPU vs TT.

Run with:
    cd /local/ttuser/teja/tt-metal
    PYTHONPATH=$(pwd) python models/tt_transformers/tt/multimodal/test_smol_vla_pcc.py

Or with pytest:
    PYTHONPATH=$(pwd) pytest models/tt_transformers/tt/multimodal/test_smol_vla_pcc.py -v
"""

from typing import Dict

import numpy as np
import pytest
import torch
from PIL import Image


def compute_pcc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson Correlation Coefficient between two arrays."""
    x_flat = torch.as_tensor(x).flatten().float()
    y_flat = torch.as_tensor(y).flatten().float()
    x_centered = x_flat - x_flat.mean()
    y_centered = y_flat - y_flat.mean()
    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum())
    return (numerator / denominator).item() if denominator != 0 else 0.0


def compute_cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine similarity between two arrays."""
    return torch.nn.functional.cosine_similarity(
        torch.as_tensor(x).flatten().unsqueeze(0).float(), torch.as_tensor(y).flatten().unsqueeze(0).float()
    ).item()


def compute_all_metrics(cpu: np.ndarray, tt: np.ndarray) -> Dict[str, float]:
    """Compute all comparison metrics between CPU and TT outputs."""
    return {
        "pcc": compute_pcc(cpu, tt),
        "cosine_sim": compute_cosine_similarity(cpu, tt),
        "max_diff": float(np.abs(cpu - tt).max()),
        "mean_diff": float(np.abs(cpu - tt).mean()),
        "rmse": float(np.sqrt(np.mean((cpu - tt) ** 2))),
    }


def create_test_image(size: int = 512, seed: int = 42) -> Image.Image:
    """Create a deterministic test image."""
    np.random.seed(seed)
    # Create a colorful pattern instead of uniform color
    img_array = np.zeros((size, size, 3), dtype=np.uint8)
    img_array[:, :, 0] = np.tile(np.linspace(50, 200, size), (size, 1)).astype(np.uint8)
    img_array[:, :, 1] = np.tile(np.linspace(100, 150, size), (size, 1)).T.astype(np.uint8)
    img_array[:, :, 2] = 128
    return Image.fromarray(img_array)


class TestSmolVLAPCC:
    """PCC verification tests for SmolVLA CPU vs TT implementation."""

    @pytest.fixture(scope="class")
    def models(self):
        """Load both CPU and TT models once for all tests."""
        import ttnn
        from models.tt_transformers.tt.multimodal.smol_vla import SmolVLAForActionPrediction

        print("\nLoading CPU model...")
        model_cpu = SmolVLAForActionPrediction.from_pretrained("lerobot/smolvla_base", ttnn_device=None)
        model_cpu.processor.image_processor.do_image_splitting = False
        model_cpu.eval()

        print("Loading TT model...")
        device = ttnn.open_device(device_id=0)
        model_tt = SmolVLAForActionPrediction.from_pretrained("lerobot/smolvla_base", ttnn_device=device)
        model_tt.processor.image_processor.do_image_splitting = False
        model_tt.eval()

        yield {
            "cpu": model_cpu,
            "tt": model_tt,
            "device": device,
        }

        print("\nClosing TT device...")
        ttnn.close_device(device)

    def test_end_to_end_pcc_1_step(self, models):
        """Test PCC for 1-step inference."""
        img = create_test_image()
        instruction = "pick up the red block"

        torch.manual_seed(42)
        np.random.seed(42)
        actions_cpu = models["cpu"].predict_action(
            images=[img], instruction=instruction, num_inference_steps=1, action_dim=6
        )

        torch.manual_seed(42)
        np.random.seed(42)
        actions_tt = models["tt"].predict_action(
            images=[img], instruction=instruction, num_inference_steps=1, action_dim=6
        )

        metrics = compute_all_metrics(actions_cpu, actions_tt)

        print(f"\n1-Step Inference Metrics:")
        print(f"  PCC:         {metrics['pcc']:.4f}")
        print(f"  Cosine Sim:  {metrics['cosine_sim']:.4f}")
        print(f"  Max Diff:    {metrics['max_diff']:.4f}")
        print(f"  Mean Diff:   {metrics['mean_diff']:.4f}")
        print(f"  RMSE:        {metrics['rmse']:.4f}")
        print(f"  CPU sample:  [{actions_cpu[0,0]:.4f}, {actions_cpu[0,1]:.4f}, {actions_cpu[0,2]:.4f}, ...]")
        print(f"  TT sample:   [{actions_tt[0,0]:.4f}, {actions_tt[0,1]:.4f}, {actions_tt[0,2]:.4f}, ...]")

        # PCC threshold for bfloat16/bfloat8 precision
        assert metrics["pcc"] >= 0.90, f"PCC {metrics['pcc']:.4f} < 0.90"
        assert metrics["cosine_sim"] >= 0.90, f"Cosine sim {metrics['cosine_sim']:.4f} < 0.90"

    def test_end_to_end_pcc_5_steps(self, models):
        """Test PCC for 5-step inference (more iterations = more drift expected)."""
        img = create_test_image()
        instruction = "move to the left"

        torch.manual_seed(42)
        np.random.seed(42)
        actions_cpu = models["cpu"].predict_action(
            images=[img], instruction=instruction, num_inference_steps=5, action_dim=6
        )

        torch.manual_seed(42)
        np.random.seed(42)
        actions_tt = models["tt"].predict_action(
            images=[img], instruction=instruction, num_inference_steps=5, action_dim=6
        )

        metrics = compute_all_metrics(actions_cpu, actions_tt)

        print(f"\n5-Step Inference Metrics:")
        print(f"  PCC:         {metrics['pcc']:.4f}")
        print(f"  Cosine Sim:  {metrics['cosine_sim']:.4f}")
        print(f"  Max Diff:    {metrics['max_diff']:.4f}")
        print(f"  Mean Diff:   {metrics['mean_diff']:.4f}")
        print(f"  RMSE:        {metrics['rmse']:.4f}")

        # Allow slightly lower threshold for multi-step (drift accumulates)
        assert metrics["pcc"] >= 0.85, f"PCC {metrics['pcc']:.4f} < 0.85"
        assert metrics["cosine_sim"] >= 0.85, f"Cosine sim {metrics['cosine_sim']:.4f} < 0.85"

    def test_instruction_sensitivity(self, models):
        """Verify that different instructions produce different outputs."""
        img = create_test_image()

        torch.manual_seed(42)
        np.random.seed(42)
        actions_a = models["tt"].predict_action(
            images=[img], instruction="pick up the red block", num_inference_steps=1, action_dim=6
        )

        torch.manual_seed(42)
        np.random.seed(42)
        actions_b = models["tt"].predict_action(
            images=[img], instruction="move away from the object", num_inference_steps=1, action_dim=6
        )

        diff = np.abs(actions_a - actions_b).mean()
        print(f"\nInstruction Sensitivity Test:")
        print(f"  Mean diff between instructions: {diff:.4f}")
        print(f"  Actions A sample: [{actions_a[0,0]:.4f}, {actions_a[0,1]:.4f}, {actions_a[0,2]:.4f}, ...]")
        print(f"  Actions B sample: [{actions_b[0,0]:.4f}, {actions_b[0,1]:.4f}, {actions_b[0,2]:.4f}, ...]")

        assert diff > 0.01, f"Instructions should produce different outputs (diff={diff:.4f})"

    def test_determinism(self, models):
        """Verify that same inputs produce consistent outputs (within bfloat16 tolerance)."""
        img = create_test_image()
        instruction = "test determinism"

        results = []
        for i in range(3):
            torch.manual_seed(42)
            np.random.seed(42)
            actions = models["tt"].predict_action(
                images=[img], instruction=instruction, num_inference_steps=1, action_dim=6
            )
            results.append(actions)

        for i in range(1, len(results)):
            max_diff = np.abs(results[0] - results[i]).max()
            pcc = compute_pcc(results[0], results[i])
            print(f"\nDeterminism run {i+1} vs run 1: max_diff = {max_diff:.6f}, PCC = {pcc:.4f}")
            # Allow some variance due to bfloat16 precision and potential hardware non-determinism
            # PCC should still be very high (>0.99) even if absolute values drift slightly
            assert pcc > 0.99, f"Results should be highly correlated (PCC={pcc:.4f})"


def run_quick_verification():
    """Run a quick verification without pytest."""
    import ttnn
    from models.tt_transformers.tt.multimodal.smol_vla import SmolVLAForActionPrediction

    print("=" * 70)
    print("SmolVLA CPU vs TT PCC Verification")
    print("=" * 70)

    # Load models
    print("\n[1/4] Loading CPU model...")
    model_cpu = SmolVLAForActionPrediction.from_pretrained("lerobot/smolvla_base", ttnn_device=None)
    model_cpu.processor.image_processor.do_image_splitting = False
    model_cpu.eval()

    print("[2/4] Loading TT model...")
    device = ttnn.open_device(device_id=0)
    model_tt = SmolVLAForActionPrediction.from_pretrained("lerobot/smolvla_base", ttnn_device=device)
    model_tt.processor.image_processor.do_image_splitting = False
    model_tt.eval()

    # Test image
    img = create_test_image()
    instruction = "pick up the red block"

    # Run inference
    print("[3/4] Running CPU inference...")
    torch.manual_seed(42)
    np.random.seed(42)
    actions_cpu = model_cpu.predict_action(images=[img], instruction=instruction, num_inference_steps=1, action_dim=6)

    print("[4/4] Running TT inference...")
    torch.manual_seed(42)
    np.random.seed(42)
    actions_tt = model_tt.predict_action(images=[img], instruction=instruction, num_inference_steps=1, action_dim=6)

    # Compute metrics
    metrics = compute_all_metrics(actions_cpu, actions_tt)

    print("\n" + "=" * 70)
    print("RESULTS (1-step inference)")
    print("=" * 70)
    print(f"  PCC:           {metrics['pcc']:.4f}  {'✓' if metrics['pcc'] >= 0.90 else '✗'}")
    print(f"  Cosine Sim:    {metrics['cosine_sim']:.4f}  {'✓' if metrics['cosine_sim'] >= 0.90 else '✗'}")
    print(f"  Max Diff:      {metrics['max_diff']:.4f}")
    print(f"  Mean Diff:     {metrics['mean_diff']:.4f}")
    print(f"  RMSE:          {metrics['rmse']:.4f}")
    print()
    print(f"  CPU output:    [{actions_cpu[0,0]:.4f}, {actions_cpu[0,1]:.4f}, {actions_cpu[0,2]:.4f}, ...]")
    print(f"  TT output:     [{actions_tt[0,0]:.4f}, {actions_tt[0,1]:.4f}, {actions_tt[0,2]:.4f}, ...]")
    print()

    # Multi-step test
    print("Running 5-step inference test...")
    torch.manual_seed(42)
    np.random.seed(42)
    actions_cpu_5 = model_cpu.predict_action(images=[img], instruction=instruction, num_inference_steps=5, action_dim=6)

    torch.manual_seed(42)
    np.random.seed(42)
    actions_tt_5 = model_tt.predict_action(images=[img], instruction=instruction, num_inference_steps=5, action_dim=6)

    metrics_5 = compute_all_metrics(actions_cpu_5, actions_tt_5)

    print("\n" + "=" * 70)
    print("RESULTS (5-step inference)")
    print("=" * 70)
    print(f"  PCC:           {metrics_5['pcc']:.4f}  {'✓' if metrics_5['pcc'] >= 0.85 else '✗'}")
    print(f"  Cosine Sim:    {metrics_5['cosine_sim']:.4f}  {'✓' if metrics_5['cosine_sim'] >= 0.85 else '✗'}")
    print(f"  Max Diff:      {metrics_5['max_diff']:.4f}")
    print(f"  Mean Diff:     {metrics_5['mean_diff']:.4f}")
    print(f"  RMSE:          {metrics_5['rmse']:.4f}")

    # Final verdict
    print("\n" + "=" * 70)
    passed = metrics["pcc"] >= 0.90 and metrics_5["pcc"] >= 0.85
    if passed:
        print("✓ VERIFICATION PASSED")
        print("  TT implementation matches CPU within expected precision tolerances")
    else:
        print("✗ VERIFICATION FAILED")
        print("  PCC below threshold - check for implementation bugs")
    print("=" * 70)

    ttnn.close_device(device)
    return passed


if __name__ == "__main__":
    import sys

    # Run quick verification when executed directly
    success = run_quick_verification()
    sys.exit(0 if success else 1)
