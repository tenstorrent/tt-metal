"""
Isolate where the issue is by feeding TT outputs to PyTorch LLM.

Test A: TT Vision + PyTorch Text → PyTorch LLM
Test B: TT Multimodal Embeddings → PyTorch LLM

Uses same image/prompt as test_openvla_model.
"""
import os

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoModelForVision2Seq, AutoProcessor

import ttnn
from models.tt_transformers.tt.multimodal.open_vla import (
    OpenVLAConfig,
    TTOpenVLAForActionPrediction,
    ttnn_to_torch_safe,
)


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
def test_tt_vision_to_pytorch_llm(mesh_device):
    """
    Feed TT vision outputs to PyTorch LLM.
    This isolates whether TT vision encoder or TT LLM is the problem.
    """

    print("\n" + "=" * 100)
    print("TEST: TT Vision Outputs → PyTorch LLM")
    print("=" * 100)

    # ============================================
    # LOAD WEIGHTS
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

    # ============================================
    # CREATE TEST IMAGE (same as test_openvla_model)
    # ============================================
    LEROBOT_IMAGES_DIR = os.path.expanduser("~/teja/smolvla/demo/images")
    image_options = ["lerobot_sample_2.png", "lerobot_sample_3.png", "lerobot_sample_1.png"]
    image = None
    for img_name in image_options:
        image_path = os.path.join(LEROBOT_IMAGES_DIR, img_name)
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            print(f"✅ Using LeRobot image: {image_path}")
            break
    if image is None:
        image = Image.new("RGB", (224, 224), color=(255, 100, 50))  # Orange (same as test_openvla_model)
        print(f"⚠️ Using synthetic orange image (255, 100, 50)")

    # ============================================
    # CREATE PROMPTS
    # ============================================
    prompt_pick = "In: What action should the robot take to pick up the block?\nOut:"
    prompt_push = "In: What action should the robot take to push the object left?\nOut:"

    inputs_pick = processor(prompt_pick, image).to("cpu", dtype=torch.bfloat16)
    inputs_push = processor(prompt_push, image).to("cpu", dtype=torch.bfloat16)

    print(f"\nPrompts:")
    print(f"  PICK: {prompt_pick}")
    print(f"  PUSH: {prompt_push}")
    print(f"  PICK tokens: {inputs_pick['input_ids'][0].tolist()}")
    print(f"  PUSH tokens: {inputs_push['input_ids'][0].tolist()}")

    # ============================================
    # LOAD PYTORCH OPENVLA (for reference)
    # ============================================
    print("\n--- Loading PyTorch OpenVLA ---")
    pt_openvla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    # ============================================
    # LOAD TT MODEL
    # ============================================
    print("\n--- Loading TT OpenVLA ---")
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
    # STEP 1: GET PYTORCH BASELINE
    # ============================================
    print("\n" + "=" * 80)
    print("STEP 1: PyTorch Baseline (Full PyTorch)")
    print("=" * 80)

    with torch.no_grad():
        pt_out_pick = pt_openvla(
            input_ids=inputs_pick["input_ids"],
            pixel_values=inputs_pick["pixel_values"],
        )
        pt_out_push = pt_openvla(
            input_ids=inputs_push["input_ids"],
            pixel_values=inputs_push["pixel_values"],
        )

    pt_pred_pick = pt_out_pick.logits[0, -1, :].argmax().item()
    pt_pred_push = pt_out_push.logits[0, -1, :].argmax().item()

    print(f"PyTorch predictions:")
    print(f"  PICK first token: {pt_pred_pick}")
    print(f"  PUSH first token: {pt_pred_push}")
    print(f"  {'✅ DIFFERENT' if pt_pred_pick != pt_pred_push else '❌ SAME'}")

    # ============================================
    # STEP 2: GET TT VISION OUTPUTS
    # ============================================
    print("\n" + "=" * 80)
    print("STEP 2: Extract TT Vision Outputs")
    print("=" * 80)

    # Run TT vision encoder
    pixel_values = inputs_pick["pixel_values"]

    # Get TT vision backbone output
    tt_vision_out = tt_model.vision_backbone(pixel_values)
    if hasattr(tt_vision_out, "device") and hasattr(tt_vision_out.device(), "get_num_devices"):
        # It's a TTNN tensor
        tt_vision_torch = ttnn_to_torch_safe(tt_vision_out, mesh_device)
    else:
        tt_vision_torch = tt_vision_out

    print(f"TT Vision output shape: {tt_vision_torch.shape}")
    print(f"TT Vision stats: mean={tt_vision_torch.mean():.6f}, std={tt_vision_torch.std():.6f}")

    # Get TT projector output
    tt_projected = tt_model.projector(tt_vision_out)
    if hasattr(tt_projected, "device") and hasattr(tt_projected.device(), "get_num_devices"):
        tt_projected_torch = ttnn_to_torch_safe(tt_projected, mesh_device)
    else:
        tt_projected_torch = tt_projected

    # Handle shape: might be [1, 256, 4096] or [1, 1, 256, 4096]
    if len(tt_projected_torch.shape) == 4:
        tt_projected_torch = tt_projected_torch.squeeze(1)  # Remove extra dim

    print(f"TT Projected shape: {tt_projected_torch.shape}")
    print(f"TT Projected stats: mean={tt_projected_torch.mean():.6f}, std={tt_projected_torch.std():.6f}")

    # Compare with PyTorch vision
    with torch.no_grad():
        pt_vision_out = pt_openvla.vision_backbone(pixel_values)
        pt_projected = pt_openvla.projector(pt_vision_out)

    print(f"\nPyTorch Projected shape: {pt_projected.shape}")
    print(f"PyTorch Projected stats: mean={pt_projected.mean():.6f}, std={pt_projected.std():.6f}")

    # PCC between TT and PT projected
    tt_flat = tt_projected_torch.flatten().float()
    pt_flat = pt_projected.flatten().float()
    pcc = torch.corrcoef(torch.stack([tt_flat, pt_flat]))[0, 1].item()
    print(f"\nTT vs PT Projected PCC: {pcc:.6f} {'✅' if pcc > 0.99 else '⚠️' if pcc > 0.9 else '❌'}")

    # ============================================
    # STEP 3: TEST A - TT Vision + PT Text → PT LLM
    # ============================================
    print("\n" + "=" * 80)
    print("STEP 3 (TEST A): TT Vision + PyTorch Text → PyTorch LLM")
    print("=" * 80)

    # Get PyTorch text embeddings
    with torch.no_grad():
        pt_text_emb_pick = pt_openvla.language_model.get_input_embeddings()(inputs_pick["input_ids"])
        pt_text_emb_push = pt_openvla.language_model.get_input_embeddings()(inputs_push["input_ids"])

    # Build multimodal embeddings: [BOS] + [TT Vision 256] + [PT Text]
    mm_emb_pick_hybrid = torch.cat(
        [
            pt_text_emb_pick[:, :1, :],  # BOS from PyTorch
            tt_projected_torch.to(pt_text_emb_pick.dtype),  # Vision from TT
            pt_text_emb_pick[:, 1:, :],  # Text from PyTorch
        ],
        dim=1,
    )

    mm_emb_push_hybrid = torch.cat(
        [
            pt_text_emb_push[:, :1, :],
            tt_projected_torch.to(pt_text_emb_push.dtype),
            pt_text_emb_push[:, 1:, :],
        ],
        dim=1,
    )

    print(f"Hybrid multimodal PICK shape: {mm_emb_pick_hybrid.shape}")
    print(f"Hybrid multimodal PUSH shape: {mm_emb_push_hybrid.shape}")

    # Run through PyTorch LLM
    with torch.no_grad():
        hybrid_out_pick = pt_openvla.language_model(
            inputs_embeds=mm_emb_pick_hybrid,
            output_hidden_states=False,
        )
        hybrid_out_push = pt_openvla.language_model(
            inputs_embeds=mm_emb_push_hybrid,
            output_hidden_states=False,
        )

    hybrid_pred_pick = hybrid_out_pick.logits[0, -1, :].argmax().item()
    hybrid_pred_push = hybrid_out_push.logits[0, -1, :].argmax().item()

    print(f"\nTEST A Results (TT Vision + PT Text → PT LLM):")
    print(f"  PICK first token: {hybrid_pred_pick}")
    print(f"  PUSH first token: {hybrid_pred_push}")
    print(f"  {'✅ DIFFERENT' if hybrid_pred_pick != hybrid_pred_push else '❌ SAME'}")

    # ============================================
    # STEP 4: TEST B - TT Full Multimodal → PT LLM
    # ============================================
    print("\n" + "=" * 80)
    print("STEP 4 (TEST B): TT Full Multimodal Embeddings → PyTorch LLM")
    print("=" * 80)

    # Get TT text embeddings
    tt_input_ids_pick = ttnn.from_torch(
        inputs_pick["input_ids"],
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_input_ids_push = ttnn.from_torch(
        inputs_push["input_ids"],
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_text_emb_pick = tt_model.get_input_embeddings()(tt_input_ids_pick)
    tt_text_emb_push = tt_model.get_input_embeddings()(tt_input_ids_push)

    tt_text_emb_pick_torch = ttnn_to_torch_safe(tt_text_emb_pick, mesh_device)
    tt_text_emb_push_torch = ttnn_to_torch_safe(tt_text_emb_push, mesh_device)

    print(f"TT Text Emb PICK shape: {tt_text_emb_pick_torch.shape}")
    print(f"TT Text Emb PICK stats: mean={tt_text_emb_pick_torch.mean():.6f}, std={tt_text_emb_pick_torch.std():.6f}")

    # Build full TT multimodal embeddings
    tt_mm_pick = torch.cat(
        [
            tt_text_emb_pick_torch[:, :1, :],
            tt_projected_torch.to(tt_text_emb_pick_torch.dtype),
            tt_text_emb_pick_torch[:, 1:, :],
        ],
        dim=1,
    )

    tt_mm_push = torch.cat(
        [
            tt_text_emb_push_torch[:, :1, :],
            tt_projected_torch.to(tt_text_emb_push_torch.dtype),
            tt_text_emb_push_torch[:, 1:, :],
        ],
        dim=1,
    )

    print(f"TT Full multimodal PICK shape: {tt_mm_pick.shape}")

    # Check if TT multimodal embeddings differ for PICK vs PUSH
    mm_diff = (tt_mm_pick - tt_mm_push).abs().sum().item()
    print(f"TT PICK vs PUSH multimodal L1 diff: {mm_diff:.6f} {'✅ DIFFERENT' if mm_diff > 0.01 else '❌ SAME'}")

    # Run through PyTorch LLM
    with torch.no_grad():
        tt_full_out_pick = pt_openvla.language_model(
            inputs_embeds=tt_mm_pick.to(torch.bfloat16),
            output_hidden_states=False,
        )
        tt_full_out_push = pt_openvla.language_model(
            inputs_embeds=tt_mm_push.to(torch.bfloat16),
            output_hidden_states=False,
        )

    tt_full_pred_pick = tt_full_out_pick.logits[0, -1, :].argmax().item()
    tt_full_pred_push = tt_full_out_push.logits[0, -1, :].argmax().item()

    print(f"\nTEST B Results (TT Full Multimodal → PT LLM):")
    print(f"  PICK first token: {tt_full_pred_pick}")
    print(f"  PUSH first token: {tt_full_pred_push}")
    print(f"  {'✅ DIFFERENT' if tt_full_pred_pick != tt_full_pred_push else '❌ SAME'}")

    # ============================================
    # STEP 5: TT MODEL PREDICTION (Reference)
    # ============================================
    print("\n" + "=" * 80)
    print("STEP 5: TT Model Full Prediction (Reference)")
    print("=" * 80)

    tt_action_pick = tt_model.predict_action(**inputs_pick, unnorm_key="bridge_orig", do_sample=False)
    tt_action_push = tt_model.predict_action(**inputs_push, unnorm_key="bridge_orig", do_sample=False)

    print(f"TT Full Model Predictions:")
    print(f"  PICK: {[f'{v:.4f}' for v in tt_action_pick]}")
    print(f"  PUSH: {[f'{v:.4f}' for v in tt_action_push]}")
    tt_diff = np.abs(np.array(tt_action_pick) - np.array(tt_action_push)).sum()
    print(f"  L1 diff: {tt_diff:.6f} {'✅ DIFFERENT' if tt_diff > 0.01 else '❌ SAME'}")

    # ============================================
    # FINAL SUMMARY
    # ============================================
    print("\n" + "=" * 100)
    print("FINAL SUMMARY: Where is the issue?")
    print("=" * 100)

    print(
        f"""
    Test                              | PICK token | PUSH token | Different?
    ----------------------------------|------------|------------|------------
    PyTorch Full                      | {pt_pred_pick:10} | {pt_pred_push:10} | {'✅ YES' if pt_pred_pick != pt_pred_push else '❌ NO'}
    TEST A: TT Vision + PT Text→PT LLM| {hybrid_pred_pick:10} | {hybrid_pred_push:10} | {'✅ YES' if hybrid_pred_pick != hybrid_pred_push else '❌ NO'}
    TEST B: TT Full MM → PT LLM       | {tt_full_pred_pick:10} | {tt_full_pred_push:10} | {'✅ YES' if tt_full_pred_pick != tt_full_pred_push else '❌ NO'}
    TT Full Model                     | (actions)  | (actions)  | {'✅ YES' if tt_diff > 0.01 else '❌ NO'}
    """
    )

    # Diagnosis
    print("DIAGNOSIS:")
    if pt_pred_pick == pt_pred_push:
        print("  ⚠️ PyTorch also predicts same token - this may be EXPECTED model behavior")
    elif hybrid_pred_pick == hybrid_pred_push:
        print("  🔴 TT Vision is the problem - TT vision outputs cause same predictions")
    elif tt_full_pred_pick == tt_full_pred_push:
        print("  🔴 TT Text Embeddings are the problem - differ from PyTorch")
    elif tt_diff < 0.01:
        print("  🔴 TT LLM Forward is the problem - embeddings are correct but TT LLM fails")
    else:
        print("  ✅ All components working correctly!")

    # Cleanup
    del pt_openvla
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
