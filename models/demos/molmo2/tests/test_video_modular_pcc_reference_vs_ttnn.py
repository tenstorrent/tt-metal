# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Per-stage PCC for video: HuggingFace ``Molmo2ForConditionalGeneration`` vs TTNN modules.

Follows the architecture in ``PLAN.md`` (ViT → multi-scale concat → pooling → projector → LM):

1. **vit_multiscale** — HF ``vision_backbone.encode_image`` vs TTNN
   ``VisionBackbone.encode_image_from_pixels`` (same pixels as ``preprocess_video``).
   This is the tensor after ViT layers **18+24** are concatenated on the channel axis (2304-dim),
   before pooling / projection.

2. **vision_adapter_output** — HF ``vision_backbone(images, token_pooling)`` vs TTNN
   ``VisionBackbone.forward`` on raw ``pixel_values`` + ``pooled_patches_idx`` (pooling + SwiGLU
   projector, valid visual tokens only).

3. **prefill_logits** (optional) — HF full forward last-position logits vs ``Molmo2Model.forward``
   when ``MOLMO2_VIDEO_MODULAR_PCC_PREFILL=1``. Uses the same multimodal batch as HF; TTNN follows
   the production ``embed_image`` path (including DP video routing when frame count > 1).

**Inputs:** Same as ``test_video_pcc_reference_vs_ttnn`` — ``--molmo2-video``, ``--molmo2-prompt``,
``MOLMO2_TEST_VIDEO``, ``MOLMO2_VIDEO_PROMPT``, ``HF_MODEL``.

**Hardware:** Opens an 8-device mesh (same class of setup as other Molmo2 video tests).
"""

from __future__ import annotations

import os
from typing import Union

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.molmo2.reference.model import Molmo2Reference
from models.demos.molmo2.tests.test_video_pcc_reference_vs_ttnn import (
    _hf_video_batch_from_processor,
    _video_demo_max_fps,
    _video_pcc_num_frames,
)
from models.demos.molmo2.tt.hf_processor import preprocess_video
from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors
from models.demos.molmo2.tt.model_loader import create_model


def _pcc_threshold(name: str, default: float) -> float:
    env_key = f"MOLMO2_VIDEO_MODULAR_PCC_{name.upper()}"
    raw = os.environ.get(env_key, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def compute_pcc(ref: torch.Tensor, test: torch.Tensor) -> float:
    """Pearson correlation on flattened tensors (same shape required)."""
    ref_flat = ref.flatten().float()
    test_flat = test.flatten().float()
    if ref_flat.shape != test_flat.shape:
        raise ValueError(f"PCC shape mismatch: ref {ref_flat.shape} vs test {test_flat.shape}")
    ref_mean = ref_flat.mean()
    test_mean = test_flat.mean()
    rc = ref_flat - ref_mean
    tc = test_flat - test_mean
    num = (rc * tc).sum()
    den = torch.sqrt((rc**2).sum() * (tc**2).sum())
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return (num / den).item()


def _ttnn_to_torch_2d(mesh_device: ttnn.Device, t: Union[ttnn.Tensor, torch.Tensor]) -> torch.Tensor:
    """Normalize device output to float CPU 2D (or 1D vocab). ``VisionBackbone.forward`` returns torch already."""
    if isinstance(t, torch.Tensor):
        x = t
    else:
        is_mesh = mesh_device.__class__.__name__ == "MeshDevice"
        mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None
        if is_mesh:
            x = ttnn.to_torch(t, mesh_composer=mesh_composer)[0]
        else:
            x = ttnn.to_torch(t)
    while x.dim() > 2:
        x = x.squeeze(0)
    return x.float().cpu()


def _maybe_deallocate(t: Union[ttnn.Tensor, torch.Tensor]) -> None:
    """``VisionBackbone.forward`` returns ``torch.Tensor``; only TTNN tensors are deallocated."""
    if isinstance(t, torch.Tensor):
        return
    ttnn.deallocate(t)


def _align_visual_tokens(hf: torch.Tensor, tt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """If row counts differ (padding / masking edge cases), compare the common prefix."""
    if hf.dim() != 2 or tt.dim() != 2:
        hf = hf.reshape(-1, hf.shape[-1])
        tt = tt.reshape(-1, tt.shape[-1])
    n = min(hf.shape[0], tt.shape[0])
    if hf.shape[0] != tt.shape[0]:
        logger.warning(
            "Vision token row mismatch: HF rows={} TTNN rows={}; comparing first {} rows",
            hf.shape[0],
            tt.shape[0],
            n,
        )
    return hf[:n], tt[:n]


@pytest.mark.parametrize("num_frames", [_video_pcc_num_frames()])
def test_video_modular_pcc_hf_vs_ttnn(
    molmo2_video_path: str,
    molmo2_video_prompt: str,
    model_location: str,
    num_frames: int,
):
    """Compare HF vs TTNN at ViT concat, vision adapter output, and optionally prefill logits."""
    t_vit = _pcc_threshold("vit", 0.98)
    t_vis = _pcc_threshold("vision", 0.95)
    t_log = _pcc_threshold("logits", 0.95)
    do_prefill = os.environ.get("MOLMO2_VIDEO_MODULAR_PCC_PREFILL", "").strip() in ("1", "true", "yes")

    max_fps = _video_demo_max_fps()
    video_prompt = """<|video|>
What will the person do next?
\nA. Put down the laptop.
\nB. Take the phone/camera.
\nC. Take the clothes.
\nD. Open the laptop.
\nPlease respond with only the letter of the correct answer.
"""

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_location, trust_remote_code=True)

    hf_batch = _hf_video_batch_from_processor(
        processor,
        molmo2_video_path,
        video_prompt,
        num_frames=num_frames,
        max_fps=max_fps,
    )

    video_inputs = preprocess_video(
        molmo2_video_path,
        video_prompt,
        num_frames=num_frames,
        max_fps=max_fps,
        processor=processor,
    )

    pixel_values = video_inputs["pixel_values"].contiguous()  # [n_frames, 3, H, W]
    n_fr = video_inputs["n_frames"]
    n_tok = video_inputs["n_tokens"]
    k_pool = video_inputs["k_pool"]
    n_out = n_tok // n_fr
    # HF stores patch ids in a **global** flatten over all frames: [0, n_fr * PATCHES_PER_FRAME).
    # VisionBackbone.forward with shape [n_frames, n_out, k_pool] indexes each row into **729** patches
    # for that frame only, so we subtract frame_id * PATCHES_PER_FRAME for nonnegative entries.
    PATCHES_PER_FRAME = 27 * 27  # 378x378 image / 14 patch size → 27² patches per frame
    pool_raw = video_inputs["image_token_pooling"]
    if pool_raw.dim() == 2 and pool_raw.shape[0] == n_tok:
        pooled_patches_idx_global = pool_raw.reshape(n_fr, n_out, k_pool).contiguous()
    else:
        pooled_patches_idx_global = pool_raw
    fr = torch.arange(n_fr, device=pooled_patches_idx_global.device, dtype=torch.long).view(n_fr, 1, 1)
    pooled_patches_idx_local = torch.where(
        pooled_patches_idx_global >= 0,
        pooled_patches_idx_global - fr * PATCHES_PER_FRAME,
        pooled_patches_idx_global,
    )

    ref = Molmo2Reference(model_location, torch_dtype=torch.bfloat16)
    hf_model = ref.model
    hf_model.eval()
    dev = next(hf_model.parameters()).device

    inner = hf_model.model
    hf_batch_dev = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in hf_batch.items()}
    images, token_pooling = inner.merge_visual_inputs(
        input_ids=hf_batch_dev["input_ids"],
        pixel_values_videos=hf_batch_dev["pixel_values_videos"],
        video_token_pooling=hf_batch_dev["video_token_pooling"],
        video_grids=hf_batch_dev["video_grids"],
    )

    # merge_visual_inputs leaves patch pixels in float32; HF vision weights are model dtype (e.g. bfloat16).
    hf_dtype = next(hf_model.parameters()).dtype
    images = images.to(device=dev, dtype=hf_dtype)
    token_pooling = token_pooling.to(dev)

    vb = inner.vision_backbone

    with torch.no_grad():
        enc_hf = vb.encode_image(images)
    # [B, T, N, C] -> [B*T*N, C]
    enc_hf_flat = enc_hf.reshape(-1, enc_hf.shape[-1]).float().cpu()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
    try:
        state_dict = load_state_dict_from_safetensors(model_location)
        molmo2 = create_model(
            mesh_device,
            state_dict,
            num_layers=None,
            max_batch_size=1,
            max_seq_len=65536,
        )
        backbone = molmo2.vision_backbone

        vit_ttnn = backbone.encode_image_from_pixels(pixel_values, num_crops=1)
        ttnn.synchronize_device(mesh_device)
        vit_tt_torch = _ttnn_to_torch_2d(mesh_device, vit_ttnn)
        _maybe_deallocate(vit_ttnn)

        vit_tt_flat = vit_tt_torch.reshape(-1, vit_tt_torch.shape[-1])
        if enc_hf_flat.shape[0] != vit_tt_flat.shape[0]:
            logger.warning(
                "ViT token count mismatch HF={} TTNN={}; PCC uses overlapping prefix",
                enc_hf_flat.shape[0],
                vit_tt_flat.shape[0],
            )
            n = min(enc_hf_flat.shape[0], vit_tt_flat.shape[0])
            enc_hf_flat = enc_hf_flat[:n]
            vit_tt_flat = vit_tt_flat[:n]

        pcc_vit = compute_pcc(enc_hf_flat, vit_tt_flat)
        logger.info("Stage vit_multiscale (ViT layers 18+24 concat): PCC={:.6f} (threshold {:.4f})", pcc_vit, t_vit)
        assert pcc_vit >= t_vit, f"ViT multiscale PCC {pcc_vit:.6f} < {t_vit}"

        with torch.no_grad():
            vis_hf = vb(images, token_pooling).float().cpu()

        vis_ttnn = backbone.forward(
            pixel_values,
            pooled_patches_idx_local,
            num_crops=1,
            use_attention_mask=True,
        )
        ttnn.synchronize_device(mesh_device)
        vis_tt = _ttnn_to_torch_2d(mesh_device, vis_ttnn)
        _maybe_deallocate(vis_ttnn)

        a, b = _align_visual_tokens(vis_hf, vis_tt)
        pcc_vis = compute_pcc(a, b)
        logger.info(
            "Stage vision_adapter (pooling + projector, valid tokens): PCC={:.6f} (threshold {:.4f})",
            pcc_vis,
            t_vis,
        )
        assert pcc_vis >= t_vis, f"Vision adapter output PCC {pcc_vis:.6f} < {t_vis}"

        if do_prefill:
            input_ids = video_inputs["input_ids"]
            tok_type = video_inputs.get("token_type_ids")
            attn_m = video_inputs.get("attention_mask")

            with torch.no_grad():
                hf_out = hf_model(
                    **{
                        **hf_batch_dev,
                        "use_cache": False,
                        "output_hidden_states": False,
                    }
                )
                hf_logits = hf_out.logits[0, -1, :].float().cpu()

            logits_ttnn, _ = molmo2.forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pooled_patches_idx=pooled_patches_idx_global,
                attention_mask=attn_m,
                token_type_ids=tok_type,
            )
            ttnn.synchronize_device(mesh_device)
            mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
            lt_full = ttnn.to_torch(logits_ttnn, mesh_composer=mesh_composer)[0].float().cpu()
            ttnn.deallocate(logits_ttnn)
            while lt_full.dim() > 2:
                lt_full = lt_full.squeeze(0)
            lt_last = lt_full[-1] if lt_full.dim() == 2 else lt_full
            pcc_log = compute_pcc(hf_logits, lt_last)
            logger.info("Stage prefill last-token logits: PCC={:.6f} (threshold {:.4f})", pcc_log, t_log)
            assert pcc_log >= t_log, f"Prefill logits PCC {pcc_log:.6f} < {t_log}"

    finally:
        ttnn.close_mesh_device(mesh_device)
