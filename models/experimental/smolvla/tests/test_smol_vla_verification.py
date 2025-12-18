#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
SmolVLA Final Verification Script: CPU vs TT PCC + Determinism

This script verifies that the TT implementation of SmolVLA matches
the PyTorch (CPU) implementation within acceptable tolerances.

Tests:
1. CPU vs TT PCC comparison (full trajectory)
2. TT determinism (5 runs with seed set once)

Usage:
    cd /local/ttuser/teja/tt-metal
    PYTHONPATH=$(pwd) python models/experimental/smolvla/tests/test_smol_vla_verification.py
"""

import os
import sys

# Ensure PYTHONPATH includes the project root
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from PIL import Image, ImageDraw
import ttnn


def create_test_workspace():
    """Create a synthetic robot workspace image with colored objects."""
    workspace = Image.new("RGB", (512, 512), color=(200, 180, 160))
    draw = ImageDraw.Draw(workspace)
    # Red cube on left
    draw.rectangle([100, 200, 180, 280], fill=(220, 50, 50))
    # Green block on right
    draw.rectangle([380, 200, 450, 280], fill=(50, 180, 50))
    # Gripper area
    draw.rectangle([220, 350, 290, 450], fill=(100, 100, 100))
    return workspace


def run_verification(pcc_threshold: float = 0.90, num_inference_steps: int = 10):
    """
    Run the full CPU vs TT verification.

    Args:
        pcc_threshold: Minimum PCC required to pass (default: 0.90)
        num_inference_steps: Number of flow matching steps (default: 10)

    Returns:
        tuple: (pcc_pass, determinism_pass, results_dict)
    """
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 80)
    print("SMOLVLA FINAL VERIFICATION: PyTorch (CPU) vs TT")
    print("=" * 80)

    device = None
    try:
        device = ttnn.open_device(device_id=0)

        from models.experimental.smolvla.tt.smol_vla import SmolVLAForActionPrediction

        # Load BOTH models
        print("\n[1/5] Loading CPU model...")
        model_cpu = SmolVLAForActionPrediction.from_pretrained("lerobot/smolvla_base", ttnn_device=None)
        model_cpu.processor.image_processor.do_image_splitting = False
        model_cpu.eval()

        print("[2/5] Loading TT model...")
        model_tt = SmolVLAForActionPrediction.from_pretrained("lerobot/smolvla_base", ttnn_device=device)
        model_tt.processor.image_processor.do_image_splitting = False
        model_tt.eval()

        # Create test inputs
        workspace = create_test_workspace()
        robot_state = torch.zeros(6).float()
        instruction = "pick up the red object"

        # ============================================================
        # WARMUP (discard)
        # ============================================================
        print("[3/5] Warmup (discarded)...")
        torch.manual_seed(42)
        with torch.no_grad():
            _ = model_cpu.predict_action(
                images=[workspace],
                robot_state=robot_state,
                instruction=instruction,
                num_inference_steps=num_inference_steps,
            )
        torch.manual_seed(42)
        with torch.no_grad():
            _ = model_tt.predict_action(
                images=[workspace],
                robot_state=robot_state,
                instruction=instruction,
                num_inference_steps=num_inference_steps,
            )

        # ============================================================
        # TEST A: CPU vs TT PCC
        # ============================================================
        print("[4/5] Running CPU vs TT comparison...")

        torch.manual_seed(42)
        with torch.no_grad():
            action_cpu = model_cpu.predict_action(
                images=[workspace],
                robot_state=robot_state,
                instruction=instruction,
                num_inference_steps=num_inference_steps,
            )

        torch.manual_seed(42)
        with torch.no_grad():
            action_tt = model_tt.predict_action(
                images=[workspace],
                robot_state=robot_state,
                instruction=instruction,
                num_inference_steps=num_inference_steps,
            )

        # Safe conversion to numpy
        action_cpu = np.asarray(action_cpu, dtype=np.float32)
        action_tt = np.asarray(action_tt, dtype=np.float32)

        print(f"\n  action_cpu shape: {action_cpu.shape}, dtype: {action_cpu.dtype}")
        print(f"  action_tt shape:  {action_tt.shape}, dtype: {action_tt.dtype}")

        print("\n" + "=" * 80)
        print("RESULT A: CPU vs TT COMPARISON")
        print("=" * 80)

        # PCC with NaN guard
        pcc_full = np.corrcoef(action_cpu.flatten(), action_tt.flatten())[0, 1]
        if np.isnan(pcc_full):
            pcc_full = 0.0
            print("  ⚠️ PCC was NaN (one side constant?), set to 0.0")

        cos_sim = np.dot(action_cpu.flatten(), action_tt.flatten()) / (
            np.linalg.norm(action_cpu.flatten()) * np.linalg.norm(action_tt.flatten()) + 1e-8
        )
        max_diff = np.abs(action_cpu - action_tt).max()
        mean_diff = np.abs(action_cpu - action_tt).mean()

        print(f"\n  Full trajectory ({action_cpu.shape}):")
        print(f"    PCC:          {pcc_full:.4f}")
        print(f"    Cosine Sim:   {cos_sim:.4f}")
        print(f"    Max Diff:     {max_diff:.4f}")
        print(f"    Mean Diff:    {mean_diff:.4f}")

        # Per-DoF comparison (step 0)
        print(f"\n  Step 0 comparison:")
        print(f"  {'DoF':<8} {'CPU':>10} {'TT':>10} {'Diff':>10}")
        print(f"  {'-'*38}")
        dof_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]

        # Handle both [50, 6] and [1, 50, 6] shapes
        if action_cpu.ndim == 2:
            cpu_step0 = action_cpu[0, :6]
            tt_step0 = action_tt[0, :6]
        else:
            cpu_step0 = action_cpu[0, 0, :6]
            tt_step0 = action_tt[0, 0, :6]

        for i, name in enumerate(dof_names):
            diff = abs(cpu_step0[i] - tt_step0[i])
            print(f"  {name:<8} {cpu_step0[i]:>10.4f} {tt_step0[i]:>10.4f} {diff:>10.4f}")

        # ============================================================
        # TEST B: TT Determinism (stronger - no reseed inside loop)
        # ============================================================
        print("\n" + "=" * 80)
        print("RESULT B: TT DETERMINISM (5 runs, seed set ONCE before loop)")
        print("=" * 80)

        print("[5/5] Running determinism test...")

        tt_actions = []
        torch.manual_seed(42)  # Set ONCE before all runs

        for run in range(5):
            with torch.no_grad():
                action = model_tt.predict_action(
                    images=[workspace],
                    robot_state=robot_state,
                    instruction=instruction,
                    num_inference_steps=num_inference_steps,
                )
            action = np.asarray(action, dtype=np.float32)
            if action.ndim == 2:
                tt_actions.append(action[0, :6])
            else:
                tt_actions.append(action[0, 0, :6])
            print(
                f"  Run {run+1}: [{tt_actions[-1][0]:.4f}, {tt_actions[-1][1]:.4f}, {tt_actions[-1][2]:.4f}, "
                f"{tt_actions[-1][3]:.4f}, {tt_actions[-1][4]:.4f}, {tt_actions[-1][5]:.4f}]"
            )

        # Check if all identical
        tt_arr = np.array(tt_actions)
        max_var = tt_arr.var(axis=0).max()
        all_identical = np.allclose(tt_arr[0], tt_arr[1:], atol=1e-5)

        print(f"\n  Max variance across runs: {max_var:.8f}")
        if all_identical:
            print("  ✅ ALL 5 RUNS IDENTICAL (truly deterministic, no reseed)")
        else:
            print("  ⚠️ Some variance detected (model may have internal randomness)")

        # ============================================================
        # FINAL VERDICT
        # ============================================================
        print("\n" + "=" * 80)
        print("FINAL VERDICT")
        print("=" * 80)

        pcc_pass = pcc_full > pcc_threshold
        det_pass = all_identical

        print(
            f"\n  CPU vs TT PCC:     {pcc_full:.4f}  {'✅ PASS' if pcc_pass else '❌ FAIL'} (threshold: {pcc_threshold})"
        )
        print(f"  TT Deterministic:  {'Yes' if det_pass else 'No'}       {'✅ PASS' if det_pass else '❌ FAIL'}")

        if pcc_pass and det_pass:
            print("\n  🎉 MODEL VERIFIED: TT implementation matches PyTorch!")
        else:
            print("\n  ⚠️ Issues detected - review above results")

        print("=" * 80)

        results = {
            "pcc": pcc_full,
            "cosine_sim": cos_sim,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "deterministic": all_identical,
            "max_variance": max_var,
        }

        return pcc_pass, det_pass, results

    finally:
        if device is not None:
            ttnn.close_device(device)
            print("\nDevice closed cleanly.")


def main():
    """Main entry point."""
    pcc_pass, det_pass, results = run_verification(pcc_threshold=0.90, num_inference_steps=10)

    # Exit with appropriate code
    if pcc_pass and det_pass:
        print("\n✅ All verification tests PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Some verification tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
