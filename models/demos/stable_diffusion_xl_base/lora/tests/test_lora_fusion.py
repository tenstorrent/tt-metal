# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import re

import pytest
import torch
from diffusers import DiffusionPipeline
from loguru import logger

import ttnn
from models.demos.stable_diffusion_xl_base.conftest import get_device_name, is_galaxy
from models.demos.stable_diffusion_xl_base.lora.tt_lora_weights_manager import load_lora_weights_te_compat
from models.demos.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig
from tests.ttnn.utils_for_testing import assert_allclose, assert_with_pcc


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
        {},
    ],
    indirect=True,
)
@pytest.mark.skipif(
    get_device_name() not in ["n150", "p150"],
    reason="test_lora_fusion runs only on n150 and p150",
)
@torch.no_grad()
def test_lora_fusion_pcc(mesh_device, lora_path):
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

    lora_manager.load_lora_weights(lora_path)
    assert lora_manager.has_lora_adapter(), "No LoRA adapter found"

    lora_manager.fuse_lora(lora_scale=1.0)

    # Build PEFT reference
    torch_pipeline.load_lora_weights(lora_path)
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
        # Tensors on a mesh device are sharded; use mesh_composer to concatenate shards when converting to torch.
        is_mesh_device = isinstance(mesh_device, ttnn._ttnn.multi_device.MeshDevice)
        mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=-1) if is_mesh_device else None
        tt_torch_tensor = ttnn.to_torch(tt_tensor, mesh_composer=mesh_composer)

        if tt_torch_tensor.shape != ref_tensor.shape:
            logger.warning(f"Shape mismatch for {weights_name}: TT={tt_torch_tensor.shape} vs ref={ref_tensor.shape}")
            continue

        assert_with_pcc(ref_tensor, tt_torch_tensor, pcc=0.999)
        assert_allclose(ref_tensor, tt_torch_tensor, atol=1e-2, rtol=1e-2)

    assert (
        not skipped_keys
    ), f"{len(skipped_keys)} LoRA impacted weights were not fused into base weights. Following weights were not fused: {skipped_keys}"


@pytest.mark.parametrize(
    "device_params",
    [
        {},
    ],
    indirect=True,
)
@pytest.mark.skipif(
    get_device_name() not in ["n150", "p150"],
    reason="test_lora_fusion runs only on n150 and p150",
)
@torch.no_grad()
def test_text_encoder_lora_fusion_pcc(mesh_device, te_lora_path):
    """Fuse a text-encoder-impacting LoRA and check the fused CLIP weights match a
    PEFT reference. The default adapter used by `test_lora_fusion_pcc` is UNet-only,
    so this is the only coverage of the text-encoder fuse path (`_fuse_text_encoder_lora`).
    """
    torch_pipeline_for_tt = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    pipeline_config = TtSDXLPipelineConfig(
        num_inference_steps=50,
        guidance_scale=5.0,
        is_galaxy=is_galaxy(),
        encoders_on_device=True,  # TE LoRA is only applied when encoders run on device.
    )
    tt_pipeline = TtSDXLPipeline(mesh_device, torch_pipeline_for_tt, pipeline_config)

    tt_pipeline.load_lora_weights(te_lora_path)
    components = tt_pipeline._lora_weights_manager.text_encoder_components()
    assert components, (
        "Chosen LoRA does not impact any text encoder; pick a LoRA that trains "
        "text_encoder / text_encoder_2 to exercise this path."
    )

    tt_pipeline.fuse_lora(lora_scale=1.0)
    assert tt_pipeline.get_lora_status()["text_encoder"] is True, "text-encoder LoRA was not marked fused"

    # PEFT reference: fuse the same LoRA into the text encoders of a fresh pipeline
    # and compare the merged CLIP weights against the ones the TT pipeline pushed to
    # its (host-side) torch encoders before reloading them on device.
    ref_pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    load_lora_weights_te_compat(ref_pipeline, te_lora_path)
    ref_pipeline.fuse_lora(components=components, lora_scale=1.0)

    for component in components:
        ref_sd = getattr(ref_pipeline, component).state_dict()
        fused_sd = getattr(tt_pipeline.torch_pipeline, component).state_dict()
        for name, ref_tensor in ref_sd.items():
            assert name in fused_sd, f"{component}: missing fused weight {name}"
            assert_with_pcc(ref_tensor, fused_sd[name], pcc=0.999)
