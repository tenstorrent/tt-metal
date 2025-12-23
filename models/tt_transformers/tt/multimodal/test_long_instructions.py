"""
Test if longer instructions produce more variation in OpenVLA actions.

Hypothesis: Short instructions (~20 tokens) are drowned out by 256 visual tokens.
Longer instructions might have more signal relative to visual embeddings.
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
def test_long_vs_short_instructions(mesh_device):
    """Test if longer instructions produce different action outputs."""

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

    # Create model
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

    # Create test image
    red_img = Image.new("RGB", (224, 224), color=(200, 50, 50))

    # Define instructions
    instructions = {
        # SHORT instructions (~20 tokens)
        "short_pick": "In: What action should the robot take to pick up the red block?\nOut:",
        "short_push": "In: What action should the robot take to push the object left?\nOut:",
        # LONGER instructions (~80+ tokens)
        "long_pick": "In: You are controlling a robot arm. The workspace contains a red block on a table surface. Your task is to carefully pick up the red block. Plan a smooth approach trajectory, position the gripper above the object, lower down, and grasp securely. What action should the robot take?\nOut:",
        "long_push": "In: You are controlling a robot arm. The workspace contains a red block on a table surface. Your task is to push the object to the left side of the workspace. Plan the contact point and direction of force. Apply gentle lateral force to slide it smoothly. What action should the robot take?\nOut:",
    }

    # Tokenization analysis
    print("\n" + "=" * 70)
    print("TOKENIZATION ANALYSIS")
    print("=" * 70)
    for name, prompt in instructions.items():
        inputs = processor(prompt, red_img)
        text_len = len(inputs["input_ids"][0])
        print(f"{name:12}: {text_len:3} text tokens | ratio vs 256 visual = {text_len/256*100:.1f}%")

    # Run predictions
    print("\n" + "=" * 70)
    print("RUNNING PREDICTIONS")
    print("=" * 70)

    results = {}
    for name, prompt in instructions.items():
        print(f"\n{name}...")
        inputs = processor(prompt, red_img).to("cpu", dtype=torch.bfloat16)
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        results[name] = action
        print(f"  Action: {[f'{v:.4f}' for v in action]}")

    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print("\n--- SHORT INSTRUCTIONS ---")
    short_pick_val = np.array(results["short_pick"])
    short_push_val = np.array(results["short_push"])
    diff_short = np.abs(short_pick_val - short_push_val).sum()
    print(f"short_pick: {[f'{v:.4f}' for v in results['short_pick']]}")
    print(f"short_push: {[f'{v:.4f}' for v in results['short_push']]}")
    print(f"L1 diff: {diff_short:.6f} {'✅ DIFFERENT' if diff_short > 0.01 else '❌ SAME'}")

    print("\n--- LONG INSTRUCTIONS ---")
    long_pick_val = np.array(results["long_pick"])
    long_push_val = np.array(results["long_push"])
    diff_long = np.abs(long_pick_val - long_push_val).sum()
    print(f"long_pick:  {[f'{v:.4f}' for v in results['long_pick']]}")
    print(f"long_push:  {[f'{v:.4f}' for v in results['long_push']]}")
    print(f"L1 diff: {diff_long:.6f} {'✅ DIFFERENT' if diff_long > 0.01 else '❌ SAME'}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if diff_long > diff_short * 1.5:
        print("✅ Longer instructions produce MORE variation - text signal matters!")
        print(f"   Improvement: {diff_long/max(diff_short, 1e-6):.1f}x more difference")
    elif diff_long > diff_short:
        print("🔶 Longer instructions slightly better")
        print(f"   Improvement: {diff_long/max(diff_short, 1e-6):.1f}x")
    elif diff_long == diff_short == 0:
        print("❌ Both short and long produce IDENTICAL actions")
        print("   -> The model is not processing text instructions properly")
    else:
        print("❌ Longer instructions don't help")

    vla.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
