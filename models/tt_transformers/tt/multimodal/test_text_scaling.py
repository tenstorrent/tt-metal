"""
Test text embedding scaling to increase instruction sensitivity.

Hypothesis: Short instructions (~20 tokens) are drowned out by 256 visual tokens.
Scaling text embeddings might improve instruction differentiation.
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
def test_text_embedding_scaling(mesh_device):
    """Test different text embedding scale factors."""

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
    vla._debug_trace = True  # Enable debug to see scale being applied

    # Create test image
    red_img = Image.new("RGB", (224, 224), color=(200, 50, 50))

    # Test prompts
    pick_prompt = "In: What action should the robot take to pick up the red block?\nOut:"
    push_prompt = "In: What action should the robot take to push the object left?\nOut:"

    # Preprocess inputs
    inputs_pick = processor(pick_prompt, red_img).to("cpu", dtype=torch.bfloat16)
    inputs_push = processor(push_prompt, red_img).to("cpu", dtype=torch.bfloat16)

    print("\n" + "=" * 80)
    print("TEXT EMBEDDING SCALING TEST")
    print("=" * 80)
    print(f"\nTest: Short instructions with RED image")
    print(f"  PICK: 'pick up the red block'")
    print(f"  PUSH: 'push the object left'")

    # Test different scale factors
    scales_to_test = [1.0, 2.0, 4.0, 8.0]
    results = {}

    for scale in scales_to_test:
        print(f"\n{'='*60}")
        print(f"SCALE = {scale}")
        print(f"{'='*60}")

        # Set scale via environment variable (read at runtime in forward pass)
        os.environ["OPENVLA_TEXT_SCALE"] = str(scale)

        # Run predictions
        print(f"\nPICK prediction:")
        action_pick = vla.predict_action(**inputs_pick, unnorm_key="bridge_orig", do_sample=False)
        print(f"  Action: {[f'{v:.4f}' for v in action_pick]}")

        print(f"\nPUSH prediction:")
        action_push = vla.predict_action(**inputs_push, unnorm_key="bridge_orig", do_sample=False)
        print(f"  Action: {[f'{v:.4f}' for v in action_push]}")

        # Calculate L1 diff
        diff = np.abs(np.array(action_pick) - np.array(action_push)).sum()

        results[scale] = {
            "pick": action_pick,
            "push": action_push,
            "diff": diff,
        }

        print(f"\nL1 diff: {diff:.6f} {'✅ DIFFERENT' if diff > 0.01 else '❌ SAME'}")

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\n{'Scale':<10} {'L1 Diff':<15} {'PICK[0]':<12} {'PUSH[0]':<12} {'Status':<10}")
    print("-" * 60)
    for scale, data in results.items():
        status = "✅ DIFF" if data["diff"] > 0.01 else "❌ SAME"
        print(f"{scale:<10} {data['diff']:<15.6f} {data['pick'][0]:<12.4f} {data['push'][0]:<12.4f} {status}")

    # Find best scale
    best_scale = max(results.keys(), key=lambda s: results[s]["diff"])
    print(f"\n🏆 Best scale: {best_scale} (L1 diff = {results[best_scale]['diff']:.6f})")

    # Check if scaling helps
    if results[best_scale]["diff"] > results[1.0]["diff"] * 1.5:
        print("✅ Text embedding scaling HELPS instruction differentiation!")
    else:
        print("⚠️ Text embedding scaling had minimal impact")

    # Reset to default
    os.environ["OPENVLA_TEXT_SCALE"] = "1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
