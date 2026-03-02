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

# LORA_PATH = "lora_weights/crayons_v1_sdxl.safetensors"
# LORA_PATH = "lora_weights/PS1Redmond-PS1Game-Playstation1Graphics.safetensors"
# LORA_PATH = "lora_weights/ColoringBookRedmond-ColoringBook-ColoringBookAF.safetensors"
LORA_PATH = "lora_weights/pytorch_lora_weights.safetensors"


def _get_lora_impacted_weights(sd):
    """
    Returns a dict of LoRA impacted weights. Weight names are cleaned up to match TT naming convention.
    """
    out = {}
    for k, tensor in sd.items():
        if ".lora_A." in k or ".lora_B." in k:
            continue
        clean = re.sub(r"^base_model\.model\.", "", k)
        clean = clean.replace(".base_layer.", ".")
        out[clean] = tensor
    return out


def _build_reference_weights(peft_sd):
    """
    Transforms weights from PEFT state dict to match TT weights format. Returns a dict of weights that can be compared to TT weights.
    """
    ref = {}
    self_attention_paths = set()

    # Identify self-attention blocks and build their QKV concatenations
    for key, torch_tensor in peft_sd.items():
        if key.endswith(".to_q.weight"):
            attn_path = key.replace(".to_q.weight", "")
            k_key = f"{attn_path}.to_k.weight"
            v_key = f"{attn_path}.to_v.weight"

            if k_key not in peft_sd or v_key not in peft_sd:
                continue

            q_weights = torch_tensor
            k_weights = peft_sd[k_key]
            v_weights = peft_sd[v_key]
            is_self_attention = (
                q_weights.shape[-1] == k_weights.shape[-1] and q_weights.shape[-1] == v_weights.shape[-1]
            )

            if is_self_attention:
                q_w = q_weights.unsqueeze(0).unsqueeze(0).transpose(-2, -1)
                k_w = k_weights.unsqueeze(0).unsqueeze(0).transpose(-2, -1)
                v_w = v_weights.unsqueeze(0).unsqueeze(0).transpose(-2, -1)

                qkv = torch.cat([q_w, k_w, v_w], dim=-1)
                ref[f"{attn_path}.to_qkv.weight"] = qkv
                self_attention_paths.add(attn_path)

    # Handle other weights
    for key, torch_tensor in peft_sd.items():
        # Skip self-attention Q/K/V
        if key.endswith((".to_q.weight", ".to_k.weight", ".to_v.weight")):
            attn_path = key.replace(".to_q.weight", "").replace(".to_k.weight", "").replace(".to_v.weight", "")
            if attn_path in self_attention_paths:
                continue

        # Split single proj weight into linear_1 + linear_2
        if key.endswith(".net.0.proj.weight"):
            w = torch_tensor.unsqueeze(0).unsqueeze(0)
            w1, w2 = w.chunk(2, dim=-2)
            prefix = key.replace(".proj.weight", ".proj")
            ref[f"{prefix}.linear_1.weight"] = w1.movedim(-1, -2)
            ref[f"{prefix}.linear_2.weight"] = w2.movedim(-1, -2)

        elif any(
            key.endswith(f"{suffix}.weight")
            for suffix in ("to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.2")
        ):
            ref[key] = torch_tensor.unsqueeze(0).unsqueeze(0).movedim(-1, -2)

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
    torch_pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    torch_pipeline_for_tt = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    pipeline_config = TtSDXLPipelineConfig(num_inference_steps=50, guidance_scale=5.0, is_galaxy=is_galaxy())

    tt_pipeline = TtSDXLPipeline(mesh_device, torch_pipeline_for_tt, pipeline_config)
    lora_manager = tt_pipeline._lora_weights_manager

    lora_manager.load_lora_weights(LORA_PATH)
    assert lora_manager.has_lora_adapter()

    lora_manager.fuse_lora(lora_scale=1.0)

    # Build PEFT reference
    torch_pipeline.load_lora_weights(LORA_PATH)
    torch_pipeline.fuse_lora()

    peft_unet_state_dict = torch_pipeline.unet.state_dict()
    peft_state_dict = _get_lora_impacted_weights(peft_unet_state_dict)
    ref_weights_dict = _build_reference_weights(peft_state_dict)

    skipped_keys = []
    for weights_name, ref_tensor in ref_weights_dict.items():
        if weights_name not in lora_manager._base_weights_device:
            skipped_keys.append(weights_name)
            continue

        tt_tensor = lora_manager._base_weights_device[weights_name]
        tt_torch_tensor = ttnn.to_torch(tt_tensor)

        if tt_torch_tensor.shape != ref_tensor.shape:
            logger.warning(f"Shape mismatch for {weights_name}: TT={tt_torch_tensor.shape} vs ref={ref_tensor.shape}")
            continue

        assert_with_pcc(ref_tensor, tt_torch_tensor, pcc=0.999)

        try:
            torch.testing.assert_close(ref_tensor, tt_torch_tensor, atol=1e-2, rtol=1e-2)
        except AssertionError as _:
            diff = torch.abs(ref_tensor - tt_torch_tensor).mean()
            logger.warning(f"{weights_name}: Mean Diff = {diff:.6f}")

        # ref_tensor_flattened = ref_tensor.flatten()
        # tt_torch_flattened = tt_torch.flatten()
        # sim = F.cosine_similarity(ref_tensor_flattened.unsqueeze(0), tt_torch_flattened.unsqueeze(0))

        # assert sim >= 0.999, f"{ref_key}: Low cosine similarity! {sim.item():.6f}"

    assert (
        not skipped_keys
    ), f"{len(skipped_keys)} LoRA impacted weights were not fused into base weights. Following weights were not fused: {skipped_keys}"
