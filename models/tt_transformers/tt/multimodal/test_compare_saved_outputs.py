"""
Compare TT outputs with saved PyTorch outputs.

Step 1: Run in PyTorch environment (timm==0.9.16):
    source /tmp/openvla_pt_env/bin/activate
    cd /home/ttuser/tvardhineni/METAL/tt-metal/models/tt_transformers/tt/multimodal
    python run_pytorch_openvla.py --output /tmp/pytorch_pick.pt --prompt "In: What action should the robot take to pick up the block?\nOut:"
    python run_pytorch_openvla.py --output /tmp/pytorch_push.pt --prompt "In: What action should the robot take to push the object left?\nOut:"

Step 2: Run this test in TT environment:
    pytest test_compare_saved_outputs.py -v -s
"""
import os

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor

import ttnn
from models.tt_transformers.tt.multimodal.open_vla import (
    OpenVLAConfig,
    TTOpenVLAForActionPrediction,
    ttnn_to_torch_safe,
)


def compare_tensors(name, pt_tensor, tt_tensor):
    """Compare two tensors."""
    pt = pt_tensor.float().flatten()
    tt = tt_tensor.float().flatten()

    # Align lengths if needed
    min_len = min(len(pt), len(tt))
    pt = pt[:min_len]
    tt = tt[:min_len]

    # PCC
    pcc = torch.corrcoef(torch.stack([pt, tt]))[0, 1].item()

    # Stats
    diff = (pt - tt).abs()

    status = "✅" if pcc > 0.99 else "⚠️" if pcc > 0.9 else "❌"
    print(f"  {name}:")
    print(f"    PT: mean={pt_tensor.mean():.6f}, std={pt_tensor.std():.6f}")
    print(f"    TT: mean={tt_tensor.mean():.6f}, std={tt_tensor.std():.6f}")
    print(f"    PCC: {pcc:.6f} {status}, MaxDiff: {diff.max():.6f}")
    return pcc


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
def test_compare_with_saved_pytorch(mesh_device):
    """Compare TT outputs with saved PyTorch outputs."""

    print("\n" + "=" * 100)
    print("COMPARE TT vs SAVED PYTORCH OUTPUTS")
    print("=" * 100)

    # Check for saved PyTorch outputs
    pt_pick_path = "/tmp/pytorch_pick.pt"
    pt_push_path = "/tmp/pytorch_push.pt"

    if not os.path.exists(pt_pick_path) or not os.path.exists(pt_push_path):
        print(f"\n⚠️  PyTorch outputs not found!")
        print(f"   Run these in PyTorch environment first:")
        print(
            f'   python run_pytorch_openvla.py --output {pt_pick_path} --prompt "In: What action should the robot take to pick up the block?\\nOut:"'
        )
        print(
            f'   python run_pytorch_openvla.py --output {pt_push_path} --prompt "In: What action should the robot take to push the object left?\\nOut:"'
        )
        pytest.skip("PyTorch outputs not found")

    # Load PyTorch outputs
    print("\n--- Loading PyTorch Outputs ---")
    pt_pick = torch.load(pt_pick_path)
    pt_push = torch.load(pt_push_path)

    print(f"PT PICK tokens: {pt_pick.get('generated_tokens', 'N/A')}")
    print(f"PT PUSH tokens: {pt_push.get('generated_tokens', 'N/A')}")
    print(f"PT PICK action: {pt_pick.get('action', 'N/A')}")
    print(f"PT PUSH action: {pt_push.get('action', 'N/A')}")

    # Check if PyTorch produces different outputs for PICK vs PUSH
    pt_tokens_pick = pt_pick.get("generated_tokens", torch.tensor([]))
    pt_tokens_push = pt_push.get("generated_tokens", torch.tensor([]))

    if torch.equal(pt_tokens_pick, pt_tokens_push):
        print(f"\n⚠️  PyTorch also produces SAME tokens for PICK and PUSH!")
        print(f"   This may be EXPECTED model behavior, not a TT bug.")
    else:
        print(f"\n✅ PyTorch produces DIFFERENT tokens for PICK and PUSH")

    # ============================================
    # LOAD TT MODEL
    # ============================================
    weight_path = os.environ.get("OPENVLA_WEIGHTS")
    merged_tensors = {}
    if weight_path and os.path.exists(weight_path):
        shard_files = [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]
        for path in shard_files:
            with safe_open(weight_path + path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    merged_tensors[key] = f.get_tensor(key)

    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    kwargs = {
        "return_unused_kwargs": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": False,
        "name_or_path": "openvla/openvla-7b",
        "pretrained_model_name_or_path": "openvla/openvla-7b",
    }
    config_dict, kwargs = OpenVLAConfig.get_config_dict(**kwargs)
    vla_config, kwargs = OpenVLAConfig.from_dict(config_dict, **kwargs)
    tt_model = TTOpenVLAForActionPrediction(vla_config, ttnn_device=mesh_device, local_state_dict=merged_tensors).to(
        "cpu", dtype=torch.bfloat16
    )
    tt_model._debug_trace = False

    # ============================================
    # USE SAME IMAGE AS PYTORCH
    # ============================================
    image_path = pt_pick.get("image_path", "synthetic")
    if image_path != "synthetic" and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        print(f"\n✅ Using same image as PyTorch: {image_path}")
    else:
        image = Image.new("RGB", (224, 224), color=(255, 100, 50))  # Orange
        print(f"\n⚠️ Using synthetic orange image (255, 100, 50)")

    # ============================================
    # RUN TT MODEL
    # ============================================
    prompt_pick = pt_pick.get("prompt", "In: What action should the robot take to pick up the block?\nOut:")
    prompt_push = pt_push.get("prompt", "In: What action should the robot take to push the object left?\nOut:")

    inputs_pick = processor(prompt_pick, image).to("cpu", dtype=torch.bfloat16)
    inputs_push = processor(prompt_push, image).to("cpu", dtype=torch.bfloat16)

    print(f"\n--- Running TT Model ---")
    tt_action_pick = tt_model.predict_action(**inputs_pick, unnorm_key="bridge_orig", do_sample=False)
    tt_action_push = tt_model.predict_action(**inputs_push, unnorm_key="bridge_orig", do_sample=False)

    print(f"TT PICK action: {[f'{v:.4f}' for v in tt_action_pick]}")
    print(f"TT PUSH action: {[f'{v:.4f}' for v in tt_action_push]}")

    # ============================================
    # COMPARE VISION OUTPUTS
    # ============================================
    print("\n" + "=" * 80)
    print("VISION ENCODER COMPARISON")
    print("=" * 80)

    # Get TT vision output
    pixel_values = inputs_pick["pixel_values"]
    tt_vision_out = tt_model.vision_backbone(pixel_values)
    if hasattr(tt_vision_out, "device"):
        tt_vision_torch = ttnn_to_torch_safe(tt_vision_out, mesh_device)
    else:
        tt_vision_torch = tt_vision_out

    pt_vision = pt_pick.get("vision_output")
    if pt_vision is not None:
        compare_tensors("Vision Output", pt_vision, tt_vision_torch)

    # ============================================
    # COMPARE PROJECTOR OUTPUTS
    # ============================================
    print("\n" + "=" * 80)
    print("PROJECTOR COMPARISON")
    print("=" * 80)

    tt_projected = tt_model.projector(tt_vision_out)
    if hasattr(tt_projected, "device"):
        tt_projected_torch = ttnn_to_torch_safe(tt_projected, mesh_device)
    else:
        tt_projected_torch = tt_projected

    if len(tt_projected_torch.shape) == 4:
        tt_projected_torch = tt_projected_torch.squeeze(1)

    pt_projected = pt_pick.get("projector_output")
    if pt_projected is not None:
        compare_tensors("Projector Output", pt_projected, tt_projected_torch)

    # ============================================
    # COMPARE GENERATED TOKENS
    # ============================================
    print("\n" + "=" * 80)
    print("TOKEN COMPARISON")
    print("=" * 80)

    # Get TT tokens from last run
    if hasattr(tt_model.language_model, "_last_generated_tokens"):
        tt_tokens = tt_model.language_model._last_generated_tokens
        print(f"TT PICK tokens: {tt_tokens}")

    print(f"PT PICK tokens: {pt_tokens_pick.tolist()}")
    print(f"PT PUSH tokens: {pt_tokens_push.tolist()}")

    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    tt_diff = np.abs(np.array(tt_action_pick) - np.array(tt_action_push)).sum()
    pt_action_pick = pt_pick.get("action", np.zeros(7))
    pt_action_push = pt_push.get("action", np.zeros(7))
    pt_diff = np.abs(np.array(pt_action_pick) - np.array(pt_action_push)).sum()

    print(
        f"""
    | Model     | PICK Action                    | PUSH Action                    | Different? |
    |-----------|--------------------------------|--------------------------------|------------|
    | PyTorch   | {pt_action_pick} | {pt_action_push} | {'✅ YES' if pt_diff > 0.01 else '❌ NO'} (L1={pt_diff:.4f}) |
    | TT        | {tt_action_pick} | {tt_action_push} | {'✅ YES' if tt_diff > 0.01 else '❌ NO'} (L1={tt_diff:.4f}) |
    """
    )

    if pt_diff < 0.01 and tt_diff < 0.01:
        print("🔵 BOTH PyTorch and TT produce same PICK/PUSH - this is EXPECTED model behavior")
    elif pt_diff > 0.01 and tt_diff < 0.01:
        print("🔴 PyTorch differs but TT doesn't - TT HAS A BUG")
    elif pt_diff < 0.01 and tt_diff > 0.01:
        print("🔵 TT differs but PyTorch doesn't - TT might be more sensitive (good?)")
    else:
        print("✅ Both differ - implementations match in behavior")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
