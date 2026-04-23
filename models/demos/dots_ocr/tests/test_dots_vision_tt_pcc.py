# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PCC test: reference `DotsVisionTransformer` (torch, HF weights) vs `DotsVisionTransformerTT` (ttnn).

Inputs match the vision tower call in `reference/demo.py` when possible (processor + test image);
otherwise a minimal synthetic `pixel_values` / `grid_thw` consistent with `DotsPatchEmbed`.

Requires local HF weights under `models/demos/dots_ocr/reference/dots_ocr/` (safetensors shards).
Override path with env `DOTS_OCR_MODEL_PATH` if needed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.dots_ocr.tt.dots_visionTT import DotsVisionTransformerTT


def _default_model_dir() -> Path:
    override = os.environ.get("DOTS_OCR_MODEL_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "reference" / "dots_ocr"


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
    state_dict = hf_model.state_dict()

    pv, grid_thw = _build_inputs_like_demo(model_dir, device=torch.device("cpu"))

    ref_out = vision_ref(pv, grid_thw)
    logger.info(f"reference vision out shape: {ref_out.shape}")

    tt_model = DotsVisionTransformerTT(
        vision_config=hf_model.config.vision_config,
        mesh_device=mesh_device,
        state_dict=state_dict,
        state_dict_prefix="vision_tower.",
        weight_cache_path=None,
    )
    tt_out = tt_model(pv, grid_thw, return_host_torch=True)

    assert ref_out.shape == tt_out.shape, f"shape mismatch ref={ref_out.shape} tt={tt_out.shape}"

    pcc_required = float(os.environ.get("DOTS_VISION_PCC_REQUIRED", "0.97"))
    passing, pcc_message = comp_pcc(ref_out.cpu().float(), tt_out.cpu().float(), pcc_required)
    logger.info(comp_allclose(ref_out, tt_out))
    logger.info(f"PCC (required {pcc_required}): {pcc_message}")
    assert passing, (
        f"DotsVisionTransformerTT vs reference PCC failed (required {pcc_required}): {pcc_message}. "
        "Tune DOTS_VISION_PCC_REQUIRED if numerical drift is expected on your mesh."
    )
