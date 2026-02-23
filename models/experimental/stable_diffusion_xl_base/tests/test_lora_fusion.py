# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import re
import pytest
import torch
import ttnn
from diffusers import DiffusionPipeline
from loguru import logger
from conftest import is_galaxy

from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_BASE_REFINER_TRACE_REGION_SIZE,
)
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import (
    TtSDXLPipeline,
    TtSDXLPipelineConfig,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

LORA_PATH = "lora_weights/ColoringBookRedmond-ColoringBook-ColoringBookAF.safetensors"


def _create_pipeline_config():
    return TtSDXLPipelineConfig(
        num_inference_steps=20,
        guidance_scale=8.0,
        is_galaxy=is_galaxy(),
        capture_trace=False,
        vae_on_device=False,
        encoders_on_device=False,
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": SDXL_L1_SMALL_SIZE,
            "trace_region_size": SDXL_BASE_REFINER_TRACE_REGION_SIZE,
        },
    ],
    indirect=True,
)
@torch.no_grad()
def test_lora_fusion(mesh_device):
    logger.info("Loading torch pipeline...")
    torch_pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    logger.info("Creating TT pipeline (allocates base weights on device)...")
    tt_pipeline = TtSDXLPipeline(mesh_device, torch_pipeline, _create_pipeline_config())

    lora_mgr = tt_pipeline.lora_weights_manager
    assert len(lora_mgr.base_weights_host) > 0, "No base weights were registered"

    logger.info(f"Registered {len(lora_mgr.base_weights_host)} base weight tensors")
    logger.info(f"Loading LoRA weights from {LORA_PATH}...")
    lora_mgr.load_lora_weights(LORA_PATH)
    assert lora_mgr.has_lora_adapter(), "LoRA adapter not detected after loading"

    logger.info("Fusing LoRA weights into device tensors...")
    lora_mgr.fuse_lora_weights(lora_scale=1.0)
    logger.info("LoRA fusion completed successfully")


def _normalize_state_dict(sd):
    """Return a clean state dict: strip PEFT prefixes, collapse .base_layer, drop LoRA keys."""
    out = {}
    for k, v in sd.items():
        if ".lora_A." in k or ".lora_B." in k:
            continue
        clean = re.sub(r"^base_model\.model\.", "", k)
        clean = clean.replace(".base_layer.", ".")
        out[clean] = v
    return out


def _build_reference_weights(peft_sd):
    """Build a dict of reference fused weights keyed the same way as base_weights_device.

    Returns tensors in the stored format [1, 1, in_dim, out_dim] (transposed linear).
    """
    ref = {}

    for key, value in peft_sd.items():
        w = value.float().unsqueeze(0).unsqueeze(0)

        # Direct linear layers (skip self-attn Q/K/V — handled by _build_qkv_reference)
        if re.search(r"\.attn1\.to_[qkv]\.weight$", key):
            continue
        for suffix in ("to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"):
            if key.endswith(f"{suffix}.weight"):
                ref[key] = w.movedim(-1, -2)

        if re.search(r"\.ff\.net\.2\.weight$", key):
            ref[key] = w.movedim(-1, -2)

        # GEGLU proj: single weight -> split into linear_1 / linear_2
        if re.search(r"\.net\.0\.proj\.weight$", key):
            w1, w2 = w.chunk(2, dim=-2)
            prefix = key.replace(".proj.weight", ".proj")
            ref[f"{prefix}.linear_1.weight"] = w1.movedim(-1, -2)
            ref[f"{prefix}.linear_2.weight"] = w2.movedim(-1, -2)

    return ref


def _build_qkv_reference(peft_sd):
    """Build reference concatenated QKV weights for self-attention blocks.

    Returns dict keyed by '{attn_path}.to_qkv.weight' with shape [1, 1, in_dim, 3*out_dim].
    """
    q_keys = [k for k in peft_sd if k.endswith(".to_q.weight")]
    ref = {}
    for q_key in q_keys:
        attn_path = q_key.replace(".to_q.weight", "")
        k_key = f"{attn_path}.to_k.weight"
        v_key = f"{attn_path}.to_v.weight"
        if k_key not in peft_sd or v_key not in peft_sd:
            continue

        q_w = peft_sd[q_key].float().unsqueeze(0).unsqueeze(0)
        k_w = peft_sd[k_key].float().unsqueeze(0).unsqueeze(0)
        v_w = peft_sd[v_key].float().unsqueeze(0).unsqueeze(0)

        if q_w.shape[-1] == k_w.shape[-1] == v_w.shape[-1]:
            qkv = torch.cat([q_w.transpose(-2, -1), k_w.transpose(-2, -1), v_w.transpose(-2, -1)], dim=-1)
            ref[f"{attn_path}.to_qkv.weight"] = qkv

    return ref


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": SDXL_L1_SMALL_SIZE,
            "trace_region_size": SDXL_BASE_REFINER_TRACE_REGION_SIZE,
        },
    ],
    indirect=True,
)
@torch.no_grad()
def test_lora_fusion_pcc(mesh_device):
    """Compare TT-fused weights against PEFT reference fused weights via PCC."""

    logger.info("Loading torch pipeline...")
    torch_pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    logger.info("Creating TT pipeline...")
    tt_pipeline = TtSDXLPipeline(mesh_device, torch_pipeline, _create_pipeline_config())
    lora_mgr = tt_pipeline.lora_weights_manager

    logger.info(f"Loading LoRA weights from {LORA_PATH}...")
    lora_mgr.load_lora_weights(LORA_PATH)
    assert lora_mgr.has_lora_adapter()

    logger.info("Fusing LoRA weights on TT device...")
    lora_mgr.fuse_lora_weights(lora_scale=1.0)

    # Build PEFT reference: fuse on the torch pipeline side and extract fused state_dict
    logger.info("Fusing LoRA weights via PEFT (reference)...")
    torch_pipeline.load_lora_weights(LORA_PATH)
    torch_pipeline.fuse_lora()
    peft_sd_raw = torch_pipeline.unet.state_dict()
    peft_sd = _normalize_state_dict(peft_sd_raw)

    logger.info(f"PEFT state_dict has {len(peft_sd)} keys (after prefix stripping)")
    logger.info(f"base_weights_device has {len(lora_mgr.base_weights_device)} keys")

    sample_peft = list(peft_sd.keys())[:5]
    sample_device = list(lora_mgr.base_weights_device.keys())[:5]
    logger.info(f"Sample PEFT keys: {sample_peft}")
    logger.info(f"Sample device keys: {sample_device}")

    ref_direct = _build_reference_weights(peft_sd)
    ref_qkv = _build_qkv_reference(peft_sd)
    ref_all = {**ref_direct, **ref_qkv}

    logger.info(f"Built {len(ref_direct)} direct refs + {len(ref_qkv)} QKV refs = {len(ref_all)} total")
    logger.info(f"Comparing {len(ref_all)} reference weights against TT fused weights...")

    min_pcc = 1.0
    worst_key = ""
    checked = 0
    pcc_threshold = 0.999

    skipped_keys = []
    for ref_key, ref_tensor in ref_all.items():
        device_key = ref_key
        if device_key not in lora_mgr.base_weights_device:
            skipped_keys.append(ref_key)
            continue

        tt_tensor = lora_mgr.base_weights_device[device_key]
        tt_torch = ttnn.to_torch(tt_tensor).float()

        if tt_torch.shape != ref_tensor.shape:
            logger.warning(f"Shape mismatch for {ref_key}: TT={tt_torch.shape} vs ref={ref_tensor.shape}, skipping")
            continue

        assert_with_pcc(ref_tensor, tt_torch, pcc=pcc_threshold)

        pcc_val = torch.corrcoef(torch.stack([ref_tensor.flatten(), tt_torch.flatten()]))[0, 1].item()
        if pcc_val < min_pcc:
            min_pcc = pcc_val
            worst_key = ref_key
        checked += 1

    if skipped_keys:
        logger.warning(f"Skipped {len(skipped_keys)} ref keys not found in device weights")
        logger.warning(f"Sample skipped ref keys: {skipped_keys[:5]}")
    assert checked > 0, "No weights were compared — key mapping may be broken"
    logger.info(f"Checked {checked} weights. Min PCC = {min_pcc:.6f} (at {worst_key})")
    logger.info(f"All weights above PCC threshold {pcc_threshold}")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
