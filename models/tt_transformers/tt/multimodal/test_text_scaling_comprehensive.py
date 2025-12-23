"""
Comprehensive text embedding scaling test.

Tests:
1. Multiple scale values (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
2. Multiple images (RED, BLUE, GREEN)
3. Multiple instructions (PICK, PUSH, MOVE, LIFT)
"""
import os

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor

import ttnn
from models.tt_transformers.tt.multimodal.open_vla import OpenVLAConfig, TTOpenVLAForActionPrediction


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
            "num_command_queues": 1,
            "l1_small_size": 81920,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "T3K": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), 8)
    ],
    indirect=True,
)
def test_text_scaling_comprehensive(mesh_device):
    """Comprehensive test of text embedding scaling."""

    # Load weights
    weight_path = os.environ.get("OPENVLA_WEIGHTS")
    merged_tensors = None
    if weight_path and os.path.exists(weight_path):
        shard_files = [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]
        merged_tensors = {}
        for path in shard_files:
            with safe_open(weight_path + path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    merged_tensors[key] = f.get_tensor(key)

    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    # Create model ONCE
    kwargs = {
        "return_unused_kwargs": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": False,
        "name_or_path": "openvla/openvla-7b",
        "pretrained_model_name_or_path": "openvla/openvla-7b",
    }
    config_dict, kwargs = OpenVLAConfig.get_config_dict(**kwargs)
    vla_config, kwargs = OpenVLAConfig.from_dict(config_dict, **kwargs)
    vla = TTOpenVLAForActionPrediction(vla_config, ttnn_device=mesh_device, local_state_dict=merged_tensors).to(
        "cpu", dtype=torch.bfloat16
    )
    vla._debug_trace = False  # Keep output clean

    # Test images
    images = {
        "RED": Image.new("RGB", (224, 224), color=(200, 50, 50)),
        "BLUE": Image.new("RGB", (224, 224), color=(50, 50, 200)),
        "GREEN": Image.new("RGB", (224, 224), color=(50, 200, 50)),
    }

    # Test instructions (short format)
    instructions = {
        "PICK": "In: What action should the robot take to pick up the block?\nOut:",
        "PUSH": "In: What action should the robot take to push the object left?\nOut:",
        "MOVE": "In: What action should the robot take to move right?\nOut:",
        "LIFT": "In: What action should the robot take to lift the gripper?\nOut:",
    }

    # Scales to test (including intermediate values)
    scales_to_test = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    print("\n" + "=" * 100)
    print("COMPREHENSIVE TEXT EMBEDDING SCALING TEST")
    print("=" * 100)

    # Store all results
    all_results = {}

    for scale in scales_to_test:
        print(f"\n{'#'*100}")
        print(f"# SCALE = {scale}")
        print(f"{'#'*100}")

        os.environ["OPENVLA_TEXT_SCALE"] = str(scale)
        all_results[scale] = {}

        for img_name, img in images.items():
            all_results[scale][img_name] = {}

            for instr_name, prompt in instructions.items():
                inputs = processor(prompt, img).to("cpu", dtype=torch.bfloat16)
                action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
                all_results[scale][img_name][instr_name] = np.array(action)

                print(f"  {img_name}_{instr_name}: {[f'{v:.4f}' for v in action]}")

    # ============================================
    # ANALYSIS
    # ============================================
    print("\n" + "=" * 100)
    print("ANALYSIS: INSTRUCTION SENSITIVITY (same image, different instructions)")
    print("=" * 100)

    instruction_pairs = [("PICK", "PUSH"), ("PICK", "MOVE"), ("PICK", "LIFT"), ("PUSH", "MOVE")]

    sensitivity_summary = {}

    for scale in scales_to_test:
        sensitivity_summary[scale] = {"total_diff": 0, "num_different": 0, "num_same": 0}
        print(f"\n--- Scale = {scale} ---")

        for img_name in images.keys():
            for instr1, instr2 in instruction_pairs:
                action1 = all_results[scale][img_name][instr1]
                action2 = all_results[scale][img_name][instr2]
                diff = np.abs(action1 - action2).sum()

                sensitivity_summary[scale]["total_diff"] += diff
                if diff > 0.01:
                    sensitivity_summary[scale]["num_different"] += 1
                    status = "✅ DIFF"
                else:
                    sensitivity_summary[scale]["num_same"] += 1
                    status = "❌ SAME"

                print(f"  {img_name}: {instr1} vs {instr2}: L1={diff:.4f} {status}")

    print("\n" + "=" * 100)
    print("ANALYSIS: IMAGE SENSITIVITY (same instruction, different images)")
    print("=" * 100)

    image_pairs = [("RED", "BLUE"), ("RED", "GREEN"), ("BLUE", "GREEN")]

    image_sensitivity = {}

    for scale in scales_to_test:
        image_sensitivity[scale] = {"total_diff": 0, "num_different": 0}
        print(f"\n--- Scale = {scale} ---")

        for instr_name in instructions.keys():
            for img1, img2 in image_pairs:
                action1 = all_results[scale][img1][instr_name]
                action2 = all_results[scale][img2][instr_name]
                diff = np.abs(action1 - action2).sum()

                image_sensitivity[scale]["total_diff"] += diff
                if diff > 0.01:
                    image_sensitivity[scale]["num_different"] += 1
                    status = "✅ DIFF"
                else:
                    status = "❌ SAME"

                print(f"  {instr_name}: {img1} vs {img2}: L1={diff:.4f} {status}")

    # ============================================
    # FINAL SUMMARY
    # ============================================
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    print(
        f"\n{'Scale':<10} {'Instr Diff Pairs':<20} {'Instr Total L1':<20} {'Image Diff Pairs':<20} {'Image Total L1':<20}"
    )
    print("-" * 90)

    best_scale = None
    best_instr_diff = 0

    for scale in scales_to_test:
        instr_diff = sensitivity_summary[scale]["num_different"]
        instr_total = sensitivity_summary[scale]["total_diff"]
        img_diff = image_sensitivity[scale]["num_different"]
        img_total = image_sensitivity[scale]["total_diff"]

        print(
            f"{scale:<10} {instr_diff}/12 different{'':<5} {instr_total:<20.4f} {img_diff}/12 different{'':<5} {img_total:<20.4f}"
        )

        if instr_diff > best_instr_diff:
            best_instr_diff = instr_diff
            best_scale = scale

    print(f"\n🏆 BEST SCALE FOR INSTRUCTION SENSITIVITY: {best_scale}")
    print(f"   Differentiates {best_instr_diff}/12 instruction pairs")

    # ============================================
    # RECOMMENDATION
    # ============================================
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)

    if best_scale and best_instr_diff > 0:
        print(f"\n✅ Set OPENVLA_TEXT_SCALE={best_scale} for better instruction sensitivity")
        print(f"   This scale differentiates {best_instr_diff}/12 instruction pairs")
    else:
        print("\n⚠️ No scale significantly improved instruction sensitivity")
        print("   The issue may be deeper than text embedding magnitude")

    # Reset
    os.environ["OPENVLA_TEXT_SCALE"] = "1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
