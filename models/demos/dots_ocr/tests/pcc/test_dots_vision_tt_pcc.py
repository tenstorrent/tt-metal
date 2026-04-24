# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PCC test: reference `DotsVisionTransformer` (torch, HF weights) vs `DotsVisionTransformerTT` (ttnn).

Inputs match the vision tower call in `reference/demo.py` when possible (processor + test image);
otherwise a minimal synthetic `pixel_values` / `grid_thw` consistent with `DotsPatchEmbed`.

Requires local HF weights under `models/demos/dots_ocr/reference/dots_ocr/` (safetensors shards);
`_default_model_dir` walks up from this file to find that folder (works with tests under
`tests/pcc/`). Override with env `DOTS_OCR_MODEL_PATH` if needed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.dots_ocr.tt.dots_visionTT import DotsVisionTransformerTT
from models.tt_transformers.tt.common import Mode


def _default_model_dir() -> Path:
    override = os.environ.get("DOTS_OCR_MODEL_PATH")
    if override:
        return Path(override).resolve()
    p = Path(__file__).resolve().parent
    for _ in range(8):
        candidate = p / "reference" / "dots_ocr"
        if candidate.is_dir():
            return candidate
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parent.parent.parent / "reference" / "dots_ocr"


def _weights_available(model_dir: Path) -> bool:
    if not (model_dir / "config.json").exists():
        return False
    # Sharded weights referenced by index
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        return any(model_dir.glob("model-*-of-*.safetensors"))
    return (model_dir / "model.safetensors").exists()


def _build_inputs_like_demo(model_dir: Path, device: torch.device):
    """
    Prefer `reference/test12.png` + processor (same pattern as `reference/demo.py`).
    Fall back to a tiny grid so patch count is divisible by spatial_merge_size**2 (4).
    """
    test_png = model_dir.parent / "test12.png"
    if test_png.exists():
        try:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(test_png)},
                        {"type": "text", "text": "hi"},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            batch = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            pv = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
            grid = batch.get("image_grid_thw", batch.get("grid_thw"))
            if grid is None:
                raise KeyError("processor batch missing image_grid_thw / grid_thw")
            grid = grid.to(device=device)
            return pv, grid
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Demo-style processor inputs failed ({exc}); using synthetic vision inputs.")

    # Synthetic: T=1, H=2, W=2 patch grid => 4 tokens; merge_size^2 divides 4.
    torch.manual_seed(42)
    patch = 14
    ch = 3
    n_patches = 4
    flat = n_patches * ch * patch * patch
    pv = torch.randn(flat, dtype=torch.bfloat16, device=device)
    grid = torch.tensor([[1, 2, 2]], dtype=torch.int32, device=device)
    return pv, grid


def _ttnn_trunk_to_seq2d(ttn: ttnn.Tensor, seqlen: int, dim: int) -> torch.Tensor:
    t = ttnn.to_torch(ttn)
    if t.dim() == 4:
        o = t[0, 0, :seqlen, :dim]
    elif t.dim() == 3:
        o = t[0, :seqlen, :dim]
    else:
        raise RuntimeError(f"unexpected trunk tensor rank {t.dim()} shape={tuple(t.shape)}")
    return o.float()


def _vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """Match ``DotsVisionTransformer.forward`` / ``DotsVisionTransformerTT.forward`` (2D ``grid_thw``)."""
    cu = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2],
        grid_thw[:, 0],
    ).cumsum(dim=0, dtype=torch.int32)
    return F.pad(cu, (1, 0), value=0)


def _ttnn_merger_to_torch(
    x: ttnn.Tensor,
    *,
    t_images: int,
    seqlen: int,
    spatial_merge_size: int,
    hidden_size: int,
) -> torch.Tensor:
    """Same host layout as ``DotsVisionTransformerTT.forward(..., return_host_torch=True)``."""
    s_merge = seqlen // (spatial_merge_size**2)
    o_full = ttnn.to_torch(x)
    if o_full.dim() == 5:
        o = o_full[:, 0, 0, :s_merge, :hidden_size]
    elif o_full.dim() == 4:
        o = o_full[:, 0, :s_merge, :hidden_size]
    elif o_full.dim() == 3:
        o = o_full[:, :s_merge, :hidden_size]
    else:
        raise RuntimeError(f"Unexpected merger tensor rank {o_full.dim()} shape={tuple(o_full.shape)}")
    out = o.squeeze(0) if t_images == 1 else o.reshape(-1, hidden_size)
    return out.float()


@torch.inference_mode()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_dots_vision_tt_pcc_vs_reference(mesh_device, ensure_gc):
    torch.manual_seed(42)
    model_dir = _default_model_dir()
    if not _weights_available(model_dir):
        pytest.skip(
            f"Dots-OCR HF weights not found under {model_dir} "
            "(need config.json + model-*.safetensors). Set DOTS_OCR_MODEL_PATH or download weights."
        )

    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    hf_model.eval()
    vision_ref = hf_model.vision_tower
    vision_ref.eval()
    # Some vision params can remain fp32 (e.g., newly initialized Conv2d bias),
    # while inputs are bf16. Align module dtype to avoid conv dtype mismatch.
    vision_ref = vision_ref.to(dtype=torch.bfloat16)
    state_dict = hf_model.state_dict()

    pv, grid_thw = _build_inputs_like_demo(model_dir, device=torch.device("cpu"))

    tt_model = DotsVisionTransformerTT(
        vision_config=hf_model.config.vision_config,
        mesh_device=mesh_device,
        state_dict=state_dict,
        state_dict_prefix="vision_tower.",
        weight_cache_path=None,
    )

    pcc_required = float(os.environ.get("DOTS_VISION_PCC_REQUIRED", "0.99"))
    n_blocks = len(vision_ref.blocks)
    d = int(vision_ref.config.embed_dim)
    logger.info(f"Vision trunk PCC: {n_blocks} blocks, embed_dim={d}")

    # Staged PCC vs reference: patch → each vision block → post_trunk_norm → merger.
    grid_2d = grid_thw.unsqueeze(0) if grid_thw.dim() == 1 else grid_thw
    t_images = int(grid_2d.shape[0])
    h_ref = vision_ref.patch_embed(pv.bfloat16(), grid_thw)
    rpe_ref = vision_ref.rot_pos_emb(grid_thw)
    cu_ref = _vision_cu_seqlens(grid_2d)
    ref_trunk_in = h_ref.float()

    hidden_tt = tt_model._patchify(pv, grid_2d)
    grid_thw_list = [tuple(int(v) for v in row) for row in grid_2d.tolist()]
    rotary_tt = tt_model._rot_pos_ttnn(grid_thw_list)
    cu_tt = tt_model._cu_seqlens_ttnn_from_grid(grid_2d)
    seqlen = int(hidden_tt.shape[0])
    x_tt, seqlen_logical, _s_pad = tt_model._prepare_ttnn(hidden_tt, mesh_device)
    assert seqlen_logical == seqlen
    tt_trunk_in = _ttnn_trunk_to_seq2d(x_tt, seqlen, d)

    ok_in, msg_in = comp_pcc(ref_trunk_in.cpu(), tt_trunk_in.cpu(), pcc_required)
    logger.info(f"Trunk input (post-patch, pre-{n_blocks} blocks) PCC: {msg_in}")
    assert ok_in, f"Trunk input PCC failed (required {pcc_required}): {msg_in}"

    for layer_idx, (blk_ref, blk_tt) in enumerate(zip(vision_ref.blocks, tt_model.blocks, strict=True)):
        h_ref = blk_ref(h_ref, cu_seqlens=cu_ref, rotary_pos_emb=rpe_ref)
        x_tt = blk_tt(x_tt, rotary_tt, cu_tt, seqlen)
        tt_layer = _ttnn_trunk_to_seq2d(x_tt, seqlen, d)
        ok_l, msg_l = comp_pcc(h_ref.float().cpu(), tt_layer.cpu(), pcc_required)
        logger.info(f"Vision block {layer_idx} output PCC: {msg_l}")
        assert ok_l, f"Vision block {layer_idx} PCC failed (required {pcc_required}): {msg_l}"

    ref_post = bool(vision_ref.config.post_norm)
    tt_has_post_norm = tt_model.post_norm is not None
    assert (
        ref_post == tt_has_post_norm
    ), "post_norm must match between HF vision_tower.config and DotsVisionTransformerTT"
    if ref_post:
        h_ref = vision_ref.post_trunk_norm(h_ref)
        x_tt = tt_model.post_norm(x_tt, mode=Mode.PREFILL)
        tt_post_host = _ttnn_trunk_to_seq2d(x_tt, seqlen, d)
        ok_post, msg_post = comp_pcc(h_ref.float().cpu(), tt_post_host.cpu(), pcc_required)
        logger.info(f"post_trunk_norm (torch) vs post_norm (ttnn) PCC: {msg_post}")
        assert ok_post, f"Post-trunk norm PCC failed (required {pcc_required}): {msg_post}"
    else:
        logger.info("post_norm disabled; skipping post-trunk norm PCC")

    h_merged_ref = vision_ref.merger(h_ref)
    x_merged_tt = tt_model.merger(x_tt, seqlen)
    hs = int(vision_ref.config.hidden_size)
    sm = int(vision_ref.config.spatial_merge_size)
    tt_merged_host = _ttnn_merger_to_torch(
        x_merged_tt,
        t_images=t_images,
        seqlen=seqlen,
        spatial_merge_size=sm,
        hidden_size=hs,
    )
    ok_merge, msg_merge = comp_pcc(h_merged_ref.cpu().float(), tt_merged_host.cpu(), pcc_required)
    logger.info(f"PatchMerger output PCC: {msg_merge}")

    ttnn.deallocate(x_merged_tt)
    ttnn.deallocate(rotary_tt)
    ttnn.deallocate(cu_tt)

    ref_out = vision_ref(pv, grid_thw)
    logger.info(f"reference vision out shape: {ref_out.shape}")
    tt_out = tt_model(pv, grid_thw, return_host_torch=True)

    assert ref_out.shape == tt_out.shape, f"shape mismatch ref={ref_out.shape} tt={tt_out.shape}"

    passing, pcc_message = comp_pcc(ref_out.cpu().float(), tt_out.cpu().float(), pcc_required)
    logger.info(comp_allclose(ref_out, tt_out))
    logger.info(f"Full vision PCC (required {pcc_required}): {pcc_message}")
    assert passing, (
        f"DotsVisionTransformerTT vs reference PCC failed (required {pcc_required}): {pcc_message}. "
        "Tune DOTS_VISION_PCC_REQUIRED if numerical drift is expected on your mesh."
    )
