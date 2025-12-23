"""
Module-level debugging: Compare PyTorch vs TT at each stage.

This test will capture outputs at each module and find where divergence occurs.
"""
import os

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoModelForVision2Seq, AutoProcessor

import ttnn
from models.tt_transformers.tt.multimodal.open_vla import OpenVLAConfig, TTOpenVLAForActionPrediction


def compute_pcc(pt_tensor, tt_tensor):
    """Compute Pearson Correlation Coefficient."""
    pt_flat = pt_tensor.flatten().float()
    tt_flat = tt_tensor.flatten().float()

    # Handle NaN/Inf
    if torch.isnan(pt_flat).any() or torch.isnan(tt_flat).any():
        return float("nan")
    if torch.isinf(pt_flat).any() or torch.isinf(tt_flat).any():
        return float("nan")

    pt_mean = pt_flat.mean()
    tt_mean = tt_flat.mean()

    pt_centered = pt_flat - pt_mean
    tt_centered = tt_flat - tt_mean

    numerator = (pt_centered * tt_centered).sum()
    denominator = torch.sqrt((pt_centered**2).sum() * (tt_centered**2).sum())

    if denominator < 1e-10:
        return 1.0 if (pt_centered.abs().max() < 1e-6 and tt_centered.abs().max() < 1e-6) else 0.0

    return (numerator / denominator).item()


def compare_tensors(name, pt_tensor, tt_tensor):
    """Compare two tensors and print detailed stats."""
    if pt_tensor is None or tt_tensor is None:
        print(f"  {name}: SKIPPED (None)")
        return

    pt = pt_tensor.float()
    tt = tt_tensor.float()

    # Align shapes if needed
    if pt.shape != tt.shape:
        print(f"  {name}: SHAPE MISMATCH pt={pt.shape} vs tt={tt.shape}")
        # Try to compare overlapping region
        min_shape = [min(p, t) for p, t in zip(pt.shape, tt.shape)]
        if len(min_shape) == 3:
            pt = pt[: min_shape[0], : min_shape[1], : min_shape[2]]
            tt = tt[: min_shape[0], : min_shape[1], : min_shape[2]]
        elif len(min_shape) == 4:
            pt = pt[: min_shape[0], : min_shape[1], : min_shape[2], : min_shape[3]]
            tt = tt[: min_shape[0], : min_shape[1], : min_shape[2], : min_shape[3]]

    pcc = compute_pcc(pt, tt)
    diff = (pt - tt).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    pt_stats = f"mean={pt.mean().item():.6f}, std={pt.std().item():.6f}"
    tt_stats = f"mean={tt.mean().item():.6f}, std={tt.std().item():.6f}"

    status = "✅" if pcc > 0.99 else "⚠️" if pcc > 0.9 else "❌"

    print(f"  {name}:")
    print(f"    Shape: {list(pt.shape)}")
    print(f"    PT: {pt_stats}")
    print(f"    TT: {tt_stats}")
    print(f"    PCC: {pcc:.6f} {status}")
    print(f"    Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

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
def test_module_level_comparison(mesh_device):
    """Compare PyTorch vs TT outputs at each module."""

    print("\n" + "=" * 100)
    print("MODULE-LEVEL DEBUGGING: PyTorch vs TT Comparison")
    print("=" * 100)

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

    # ============================================
    # LOAD PYTORCH MODEL
    # ============================================
    print("\n--- Loading PyTorch OpenVLA ---")
    pt_model = AutoModelForVision2Seq.from_pretrained(
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
    # TEST INPUTS
    # ============================================
    # Test with RED image and PICK instruction
    red_img = Image.new("RGB", (224, 224), color=(200, 50, 50))
    pick_prompt = "In: What action should the robot take to pick up the block?\nOut:"
    push_prompt = "In: What action should the robot take to push the object left?\nOut:"

    inputs_pick = processor(pick_prompt, red_img).to("cpu", dtype=torch.bfloat16)
    inputs_push = processor(push_prompt, red_img).to("cpu", dtype=torch.bfloat16)

    print(f"\n--- Test Inputs ---")
    print(f"Image: RED solid (200, 50, 50)")
    print(f"PICK prompt: {pick_prompt[:50]}...")
    print(f"PUSH prompt: {push_prompt[:50]}...")
    print(f"PICK input_ids: {inputs_pick['input_ids'][0].tolist()}")
    print(f"PUSH input_ids: {inputs_push['input_ids'][0].tolist()}")

    # ============================================
    # MODULE 1: VISION ENCODER
    # ============================================
    print("\n" + "=" * 80)
    print("MODULE 1: VISION ENCODER")
    print("=" * 80)

    pixel_values = inputs_pick["pixel_values"]

    # PyTorch vision encoder
    with torch.no_grad():
        pt_vision_output = pt_model.vision_backbone(pixel_values)
    print(f"PT Vision Output shape: {pt_vision_output.shape}")

    # TT vision encoder (need to extract from model)
    # For now, we'll compare at the projector output level

    # ============================================
    # MODULE 2: PROJECTOR
    # ============================================
    print("\n" + "=" * 80)
    print("MODULE 2: PROJECTOR (Vision -> LLM space)")
    print("=" * 80)

    with torch.no_grad():
        pt_projected = pt_model.projector(pt_vision_output)
    print(f"PT Projected shape: {pt_projected.shape}")
    print(f"PT Projected stats: mean={pt_projected.mean():.6f}, std={pt_projected.std():.6f}")

    # ============================================
    # MODULE 3: TEXT EMBEDDINGS
    # ============================================
    print("\n" + "=" * 80)
    print("MODULE 3: TEXT EMBEDDINGS")
    print("=" * 80)

    input_ids_pick = inputs_pick["input_ids"]
    input_ids_push = inputs_push["input_ids"]

    with torch.no_grad():
        pt_text_emb_pick = pt_model.language_model.get_input_embeddings()(input_ids_pick)
        pt_text_emb_push = pt_model.language_model.get_input_embeddings()(input_ids_push)

    print(f"PT Text Emb PICK shape: {pt_text_emb_pick.shape}")
    print(f"PT Text Emb PICK stats: mean={pt_text_emb_pick.mean():.6f}, std={pt_text_emb_pick.std():.6f}")
    print(f"PT Text Emb PUSH stats: mean={pt_text_emb_push.mean():.6f}, std={pt_text_emb_push.std():.6f}")

    # Check if PICK and PUSH embeddings differ
    text_diff = (pt_text_emb_pick - pt_text_emb_push).abs().sum().item()
    print(f"PICK vs PUSH text embedding L1 diff: {text_diff:.6f}")

    # ============================================
    # MODULE 4: MULTIMODAL EMBEDDINGS
    # ============================================
    print("\n" + "=" * 80)
    print("MODULE 4: MULTIMODAL EMBEDDINGS (BOS + Vision + Text)")
    print("=" * 80)

    # PyTorch: Build multimodal embeddings manually
    with torch.no_grad():
        # [BOS] + [256 vision] + [text tokens after BOS]
        pt_mm_pick = torch.cat(
            [
                pt_text_emb_pick[:, :1, :],  # BOS
                pt_projected,  # Vision (256 tokens)
                pt_text_emb_pick[:, 1:, :],  # Text after BOS
            ],
            dim=1,
        )

        pt_mm_push = torch.cat(
            [
                pt_text_emb_push[:, :1, :],
                pt_projected,
                pt_text_emb_push[:, 1:, :],
            ],
            dim=1,
        )

    print(f"PT Multimodal PICK shape: {pt_mm_pick.shape}")
    print(f"PT Multimodal PICK stats: mean={pt_mm_pick.mean():.6f}, std={pt_mm_pick.std():.6f}")

    # Check if PICK and PUSH multimodal embeddings differ
    mm_diff = (pt_mm_pick - pt_mm_push).abs().sum().item()
    print(f"PICK vs PUSH multimodal embedding L1 diff: {mm_diff:.6f}")

    # Check positions: where do they differ?
    seq_len = pt_mm_pick.shape[1]
    print(f"\nPosition-wise difference analysis:")
    print(f"  Position 0 (BOS): diff={((pt_mm_pick[:, 0, :] - pt_mm_push[:, 0, :]).abs().sum().item()):.6f}")
    print(
        f"  Positions 1-256 (Vision): diff={((pt_mm_pick[:, 1:257, :] - pt_mm_push[:, 1:257, :]).abs().sum().item()):.6f}"
    )
    print(f"  Positions 257+ (Text): diff={((pt_mm_pick[:, 257:, :] - pt_mm_push[:, 257:, :]).abs().sum().item()):.6f}")

    # ============================================
    # MODULE 5: RUN PYTORCH LLM FORWARD
    # ============================================
    print("\n" + "=" * 80)
    print("MODULE 5: LLM FORWARD PASS (PyTorch)")
    print("=" * 80)

    with torch.no_grad():
        # Run full forward pass for PICK
        pt_outputs_pick = pt_model(
            input_ids=input_ids_pick,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        # Run full forward pass for PUSH
        pt_outputs_push = pt_model(
            input_ids=input_ids_push,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

    pt_logits_pick = pt_outputs_pick.logits
    pt_logits_push = pt_outputs_push.logits

    print(f"PT Logits PICK shape: {pt_logits_pick.shape}")
    print(f"PT Logits PICK stats: mean={pt_logits_pick.mean():.6f}, std={pt_logits_pick.std():.6f}")
    print(f"PT Logits PUSH stats: mean={pt_logits_push.mean():.6f}, std={pt_logits_push.std():.6f}")

    # Check if logits differ
    logits_diff = (pt_logits_pick - pt_logits_push).abs()
    print(f"\nPICK vs PUSH logits analysis:")
    print(f"  Total L1 diff: {logits_diff.sum().item():.6f}")
    print(f"  Max diff: {logits_diff.max().item():.6f}")
    print(f"  Mean diff: {logits_diff.mean().item():.6f}")

    # Check last position logits (most important for generation)
    last_pos_pick = pt_logits_pick[0, -1, :]
    last_pos_push = pt_logits_push[0, -1, :]
    last_pos_diff = (last_pos_pick - last_pos_push).abs()
    print(f"\nLast position logits:")
    print(f"  PICK: top 5 tokens = {torch.topk(last_pos_pick, 5).indices.tolist()}")
    print(f"  PUSH: top 5 tokens = {torch.topk(last_pos_push, 5).indices.tolist()}")
    print(f"  L1 diff: {last_pos_diff.sum().item():.6f}")

    # Predicted tokens
    pt_pred_pick = torch.argmax(last_pos_pick).item()
    pt_pred_push = torch.argmax(last_pos_push).item()
    print(f"\nPT Predicted first token:")
    print(f"  PICK: {pt_pred_pick}")
    print(f"  PUSH: {pt_pred_push}")
    print(f"  {'✅ DIFFERENT' if pt_pred_pick != pt_pred_push else '❌ SAME'}")

    # ============================================
    # MODULE 6: RUN TT MODEL
    # ============================================
    print("\n" + "=" * 80)
    print("MODULE 6: TT MODEL PREDICTIONS")
    print("=" * 80)

    tt_action_pick = tt_model.predict_action(**inputs_pick, unnorm_key="bridge_orig", do_sample=False)
    tt_action_push = tt_model.predict_action(**inputs_push, unnorm_key="bridge_orig", do_sample=False)

    print(f"TT Action PICK: {[f'{v:.4f}' for v in tt_action_pick]}")
    print(f"TT Action PUSH: {[f'{v:.4f}' for v in tt_action_push]}")

    tt_diff = np.abs(np.array(tt_action_pick) - np.array(tt_action_push)).sum()
    print(f"TT PICK vs PUSH L1 diff: {tt_diff:.6f} {'✅ DIFFERENT' if tt_diff > 0.01 else '❌ SAME'}")

    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "=" * 100)
    print("SUMMARY: WHERE DOES DIVERGENCE OCCUR?")
    print("=" * 100)

    print(
        f"""
    Module                      | PICK vs PUSH Differs?
    ----------------------------|----------------------
    Input Token IDs             | ✅ Yes (by design)
    Text Embeddings (PT)        | {'✅ Yes' if text_diff > 0.01 else '❌ No'} (L1={text_diff:.4f})
    Multimodal Embeddings (PT)  | {'✅ Yes' if mm_diff > 0.01 else '❌ No'} (L1={mm_diff:.4f})
    LLM Logits (PT)             | {'✅ Yes' if logits_diff.sum().item() > 0.01 else '❌ No'} (L1={logits_diff.sum().item():.4f})
    Predicted Token (PT)        | {'✅ Yes' if pt_pred_pick != pt_pred_push else '❌ No'} ({pt_pred_pick} vs {pt_pred_push})
    TT Final Actions            | {'✅ Yes' if tt_diff > 0.01 else '❌ No'} (L1={tt_diff:.4f})
    """
    )

    if pt_pred_pick == pt_pred_push:
        print("⚠️  PyTorch model ALSO predicts same token for PICK and PUSH!")
        print("   This is EXPECTED MODEL BEHAVIOR, not a TT bug!")
    else:
        print("🔴 PyTorch predicts different tokens, but TT predicts same!")
        print("   This indicates a TT implementation bug!")

    # Cleanup
    del pt_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
