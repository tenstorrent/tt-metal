"""Save TT outputs with EXACT same inputs as PyTorch for proper Test A/B."""
import os

import pytest
import torch
from safetensors import safe_open

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
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"), 8)],
    indirect=True,
)
def test_save_correct(mesh_device):
    """Save TT outputs with exact PyTorch inputs."""

    # Load PyTorch saved inputs
    pt_pick = torch.load("/tmp/pytorch_pick.pt")
    pt_push = torch.load("/tmp/pytorch_push.pt")

    print("\n" + "=" * 80)
    print("USING EXACT PYTORCH INPUTS")
    print("=" * 80)
    print(f"PyTorch PICK input_ids: {pt_pick['input_ids'][0].tolist()}")
    print(f"PyTorch PUSH input_ids: {pt_push['input_ids'][0].tolist()}")

    # Load weights
    weight_path = os.environ.get("OPENVLA_WEIGHTS")
    merged_tensors = {}
    for path in [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]:
        with safe_open(weight_path + path, framework="pt", device="cpu") as f:
            for key in f.keys():
                merged_tensors[key] = f.get_tensor(key)

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
    tt_model._debug_trace = False

    print("\n" + "=" * 80)
    print("EXTRACTING TT INTERMEDIATE OUTPUTS")
    print("=" * 80)

    for name, pt_data in [("PICK", pt_pick), ("PUSH", pt_push)]:
        print(f"\n{'='*40} {name} {'='*40}")

        # Use EXACT same inputs from PyTorch
        pixel_values = pt_data["pixel_values"].to(torch.bfloat16)
        input_ids = pt_data["input_ids"]

        print(f"input_ids: {input_ids[0].tolist()}")

        # Compare PyTorch projector stats
        if "projector_output" in pt_data:
            pt_proj = pt_data["projector_output"]
            print(f"PT projector shape: {pt_proj.shape}")
            print(f"PT projector: mean={pt_proj.float().mean():.6f}, std={pt_proj.float().std():.6f}")

        # Enable saving vision/projector output
        vision_save_path = f"/tmp/tt_vision_{name.lower()}.pt"
        tt_model._save_vision_output = vision_save_path

        # Run full TT prediction
        inputs = {"input_ids": input_ids, "pixel_values": pixel_values}
        tt_action = tt_model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        tt_tokens = (
            tt_model.language_model._last_generated_tokens
            if hasattr(tt_model.language_model, "_last_generated_tokens")
            else None
        )

        # Disable for next run
        tt_model._save_vision_output = None

        print(f"TT tokens: {tt_tokens}")
        print(f"TT action: {[f'{x:.4f}' for x in tt_action]}")

        # Load and compare projector output
        if os.path.exists(vision_save_path):
            vision_data = torch.load(vision_save_path)
            tt_proj = vision_data["projector_output"]
            print(f"TT projector shape: {tt_proj.shape}")
            print(f"TT projector: mean={tt_proj.float().mean():.6f}, std={tt_proj.float().std():.6f}")

            # PCC with PyTorch projector
            if "projector_output" in pt_data:
                pt_flat = pt_proj.flatten().float()
                tt_flat = tt_proj.flatten().float()
                if pt_flat.shape == tt_flat.shape:
                    pcc = torch.corrcoef(torch.stack([pt_flat, tt_flat]))[0, 1].item()
                    print(f"TT vs PT Projector PCC: {pcc:.6f} {'✅ HIGH' if pcc > 0.99 else '⚠️ LOW'}")

        # Save final outputs
        output = {
            "pixel_values": pixel_values.float().cpu(),
            "input_ids": input_ids.cpu(),
            "tt_tokens": torch.tensor(tt_tokens) if tt_tokens else None,
            "tt_action": torch.tensor(tt_action),
        }
        torch.save(output, f"/tmp/tt_correct_{name.lower()}.pt")
        print(f"✅ Saved to /tmp/tt_correct_{name.lower()}.pt")

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"PyTorch PICK tokens: {pt_pick['generated_tokens'].tolist()}")
    print(f"PyTorch PUSH tokens: {pt_push['generated_tokens'].tolist()}")

    tt_pick_new = torch.load("/tmp/tt_correct_pick.pt")
    tt_push_new = torch.load("/tmp/tt_correct_push.pt")
    print(f"TT PICK tokens:      {tt_pick_new['tt_tokens'].tolist()}")
    print(f"TT PUSH tokens:      {tt_push_new['tt_tokens'].tolist()}")
