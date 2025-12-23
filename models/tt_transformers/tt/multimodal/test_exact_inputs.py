"""Run TT with EXACT same inputs as PyTorch"""
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
def test_exact_inputs(mesh_device):
    """Run TT with EXACT same inputs as PyTorch"""

    # Load PyTorch saved inputs
    pt_pick = torch.load("/tmp/pytorch_pick.pt")
    pt_push = torch.load("/tmp/pytorch_push.pt")

    print("\n" + "=" * 80)
    print("Using EXACT PyTorch inputs:")
    print(f"  PICK input_ids: {pt_pick['input_ids'][0].tolist()}")
    print(f"  PUSH input_ids: {pt_push['input_ids'][0].tolist()}")
    print(f"  pixel_values shape: {pt_pick['pixel_values'].shape}")
    print("=" * 80)

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
    tt_model._debug_trace = True

    print("\n" + "=" * 80)
    print("RUNNING TT WITH EXACT PYTORCH INPUTS")
    print("=" * 80)

    results = {}
    for name, pt_data in [("PICK", pt_pick), ("PUSH", pt_push)]:
        print(f"\n{'='*40} {name} {'='*40}")

        # Use EXACT same inputs from PyTorch
        inputs = {
            "input_ids": pt_data["input_ids"],
            "pixel_values": pt_data["pixel_values"].to(torch.bfloat16),
        }

        print(f"input_ids: {inputs['input_ids'][0].tolist()}")

        # Run TT model
        tt_action = tt_model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

        tt_tokens = None
        if hasattr(tt_model.language_model, "_last_generated_tokens"):
            tt_tokens = tt_model.language_model._last_generated_tokens

        results[name] = {"action": tt_action, "tokens": tt_tokens}

        print(f"\nTT Action: {[f'{v:.4f}' for v in tt_action]}")
        print(f"TT Tokens: {tt_tokens}")

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"PyTorch PICK tokens: {pt_pick['generated_tokens'].tolist()}")
    print(f"PyTorch PUSH tokens: {pt_push['generated_tokens'].tolist()}")
    print(f"TT PICK tokens:      {results['PICK']['tokens']}")
    print(f"TT PUSH tokens:      {results['PUSH']['tokens']}")

    print(f"\nPyTorch PICK action: {pt_pick['action'].tolist()}")
    print(f"PyTorch PUSH action: {pt_push['action'].tolist()}")
    print(f"TT PICK action:      {results['PICK']['action']}")
    print(f"TT PUSH action:      {results['PUSH']['action']}")

    # Check if TT now produces different outputs for PICK vs PUSH
    if results["PICK"]["tokens"] == results["PUSH"]["tokens"]:
        print("\n❌ TT still produces SAME tokens for PICK and PUSH")
    else:
        print("\n✅ TT produces DIFFERENT tokens for PICK and PUSH")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
