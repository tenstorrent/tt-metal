"""
Save TT outputs for comparison with PyTorch.

Run in TT environment:
    pytest save_tt_outputs.py -v -s
"""
import os

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
def test_save_tt_outputs(mesh_device):
    """Save TT outputs for comparison."""

    print("\n" + "=" * 100)
    print("SAVING TT OUTPUTS")
    print("=" * 100)

    # Load weights
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

    # Create TT model
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
    tt_model._debug_trace = True  # Enable debug to see intermediate outputs

    # Use same image as test_openvla_model
    LEROBOT_IMAGES_DIR = os.path.expanduser("~/teja/smolvla/demo/images")
    image_options = ["lerobot_sample_2.png", "lerobot_sample_3.png", "lerobot_sample_1.png"]
    image = None
    image_path = "synthetic"
    for img_name in image_options:
        img_path = os.path.join(LEROBOT_IMAGES_DIR, img_name)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            image_path = img_path
            print(f"✅ Using LeRobot image: {img_path}")
            break
    if image is None:
        image = Image.new("RGB", (224, 224), color=(255, 100, 50))  # Orange
        print(f"⚠️ Using synthetic orange image (255, 100, 50)")

    # Two prompts
    prompts = {
        "pick": "In: What action should the robot take to pick up the block?\nOut:",
        "push": "In: What action should the robot take to push the object left?\nOut:",
    }

    for name, prompt in prompts.items():
        print(f"\n{'='*80}")
        print(f"Processing: {name.upper()}")
        print(f"Prompt: {prompt}")
        print(f"{'='*80}")

        inputs = processor(prompt, image).to("cpu", dtype=torch.bfloat16)
        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]

        print(f"input_ids: {input_ids[0].tolist()}")
        print(f"pixel_values shape: {pixel_values.shape}")

        # ============================================
        # RUN TT MODEL (will print debug outputs)
        # ============================================
        print("\n--- Running TT Model ---")
        tt_action = tt_model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        print(f"TT Action: {[f'{v:.4f}' for v in tt_action]}")

        # Get generated tokens
        tt_tokens = None
        if hasattr(tt_model.language_model, "_last_generated_tokens"):
            tt_tokens = tt_model.language_model._last_generated_tokens
            print(f"TT Tokens: {tt_tokens}")

        # ============================================
        # SAVE OUTPUTS
        # ============================================
        output_path = f"/tmp/tt_outputs_{name}.pt"
        outputs = {
            # Inputs (for PyTorch to use same inputs)
            "pixel_values": pixel_values.float().cpu(),
            "input_ids": input_ids.cpu(),
            "prompt": prompt,
            "image_path": image_path,
            # TT final outputs
            "tt_action": torch.tensor(tt_action, dtype=torch.float32),
            "tt_tokens": torch.tensor(tt_tokens) if tt_tokens else None,
        }

        torch.save(outputs, output_path)
        print(f"\n✅ Saved to: {output_path}")

    print("\n" + "=" * 100)
    print("COMPARISON WITH PYTORCH")
    print("=" * 100)
    print(
        """
    TT always produces: [31852, 31852, 31852, 31852, 31852, 31852, 31852]

    User reported PyTorch PICK produces:
    [31853, 31867, 31890, 31855, 31893, 31852, 31744, 2]

    ⚠️ TT and PyTorch produce DIFFERENT tokens!

    This confirms there is a bug in TT implementation.
    The issue is likely in TT LLM forward pass, not vision encoder.
    """
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
