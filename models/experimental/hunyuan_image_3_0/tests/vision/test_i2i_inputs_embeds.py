# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Device PCC test for the composed I2I cond-ViT embedding assembly — the "ViT half"
# of the reference bundle prep (gen_image_inputs.build_i2i_inputs_embeds).
#
# Reference golden (host):
#   hidden = wte(input_ids)                                   # <img> placeholders
#   hidden = instantiate_vit_image_tokens(hidden, cond_vit_images, vit_mask,
#                                          kwargs, ref_vision, ref_aligner)
# i.e. SigLIP2 tower + LightProjector over a real preprocessed cond image, scattered
# into the <img> span. Two device paths are checked against that golden:
#
#   1. tt/vision/i2i.build_i2i_inputs_embeds  — fully on-device encode + inject
#      (the device alternative referenced in the port status), and
#   2. instantiate_vit_image_tokens(..., TtVisionModelHostAdapter, TtAlignerHostAdapter)
#      — the exact path demo_i2i.py wires (host scatter, device encode).
#
# Inputs: a real PIL image through the reference SigLIP2 processor; weights are the
# real HunyuanImage checkpoint. Reuses the session `device` + ref fixtures from
# conftest.py (set HY_VIT_NUM_LAYERS=27 for the full vision stack).
#
# Run (fast, 1 encoder layer):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/vision/test_i2i_inputs_embeds.py -v -s
# Full stack:
#   HY_VIT_NUM_LAYERS=27 python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/vision/test_i2i_inputs_embeds.py -v -s

import pytest
import torch
import ttnn
from loguru import logger
from PIL import Image

from models.common.utility_functions import comp_pcc
from models.experimental.hunyuan_image_3_0.ref.image_gen.input_instantiate import instantiate_vit_image_tokens
from models.experimental.hunyuan_image_3_0.ref.vision.preprocess import build_cond_image_processor
from models.experimental.hunyuan_image_3_0.tt.vision.i2i import (
    TtAlignerHostAdapter,
    TtVisionModelHostAdapter,
    build_i2i_inputs_embeds,
)
from models.experimental.hunyuan_image_3_0.tt.vision.siglip2 import (
    HunyuanTtLightProjector,
    HunyuanTtSiglip2Vision,
)

from .conftest import NUM_LAYERS, PCC_THR

H = 4096  # aligner output / LLM hidden size
SPAN_START = 32  # TILE-aligned <img> span start (text_pre present)
TAIL_PAD = 32  # text_post length (keep sequence > span)


def _up(device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.fixture(scope="module")
def cond_processor_out(model_dir):
    """Run a deterministic RGB image through the reference SigLIP2 processor."""
    from models.experimental.hunyuan_image_3_0.ref.vision.preprocess import vit_process_image

    processor = build_cond_image_processor(model_dir)
    torch.manual_seed(0)
    arr = torch.randint(0, 256, (384, 512, 3), dtype=torch.uint8).numpy()
    image = Image.fromarray(arr, mode="RGB")
    pixel_values, spatial_shapes_hw, pixel_attention_mask = vit_process_image(processor, image)
    # Normalize to per-image batched form: [1, S, patch_dim], [1, S].
    if pixel_values.ndim == 2:
        pixel_values = pixel_values.unsqueeze(0)
    if pixel_attention_mask.ndim == 1:
        pixel_attention_mask = pixel_attention_mask.unsqueeze(0)
    return pixel_values.float(), spatial_shapes_hw, pixel_attention_mask


def _ref_inputs(cond_processor_out):
    """Build the reference instantiate_vit_image_tokens inputs from processor output."""
    pixel_values, (th, tw), pixel_attention_mask = cond_processor_out
    n_img = int(pixel_values.shape[1])
    spatial_shapes = torch.tensor([[th, tw]], dtype=torch.long)  # [1, 2]
    kwargs = {"spatial_shapes": spatial_shapes, "attention_mask": pixel_attention_mask}

    seq_len = SPAN_START + n_img + TAIL_PAD
    span = slice(SPAN_START, SPAN_START + n_img)
    torch.manual_seed(1)
    base_hidden = torch.randn(1, seq_len, H, dtype=torch.float32)
    mask = torch.zeros(1, seq_len, dtype=torch.bool)
    mask[:, span] = True
    return base_hidden, pixel_values, mask, kwargs, span, n_img, (th, tw)


def test_build_i2i_inputs_embeds_pcc(
    device, cond_processor_out, ref_vision, ref_aligner, vision_state_dict, aligner_state_dict
):
    """Fully on-device tt/vision/i2i.build_i2i_inputs_embeds vs ref ViT-inject."""
    base_hidden, pixel_values, mask, kwargs, span, n_img, (th, tw) = _ref_inputs(cond_processor_out)

    with torch.no_grad():
        ref = instantiate_vit_image_tokens(base_hidden.clone(), pixel_values, mask, kwargs, ref_vision, ref_aligner)

    from models.experimental.hunyuan_image_3_0.tt.vision.siglip2 import Siglip2VisionInputs

    vision_tt = HunyuanTtSiglip2Vision(device, vision_state_dict, num_layers=NUM_LAYERS)
    vision_tt.prewarm_pos_geometries([(th, tw, n_img)])
    aligner_tt = HunyuanTtLightProjector(device, aligner_state_dict)

    vision_inputs = Siglip2VisionInputs.create(
        _up(device, pixel_values),
        ((th, tw),),
        _up(device, kwargs["attention_mask"]),
    )
    text_tt = _up(device, base_hidden)
    out_tt = build_i2i_inputs_embeds(
        device,
        vision=vision_tt,
        aligner=aligner_tt,
        text_embeds=text_tt,
        vision_inputs=vision_inputs,
        img_slices=[span],
    )
    out = ttnn.to_torch(out_tt).float()
    ttnn.deallocate(out_tt)

    assert tuple(out.shape) == tuple(ref.shape), f"shape {tuple(out.shape)} != {tuple(ref.shape)}"
    passing, pcc = comp_pcc(ref, out, PCC_THR)
    logger.info(f"build_i2i_inputs_embeds n_img={n_img} span=[{span.start}:{span.stop}] PCC={pcc} (>= {PCC_THR})")
    assert passing, f"PCC {pcc} < {PCC_THR}"


def test_host_adapter_instantiate_pcc(
    device, cond_processor_out, ref_vision, ref_aligner, vision_state_dict, aligner_state_dict
):
    """ref instantiate_vit_image_tokens driven by the device host-adapters (demo wiring)."""
    base_hidden, pixel_values, mask, kwargs, span, n_img, _hw = _ref_inputs(cond_processor_out)

    with torch.no_grad():
        ref = instantiate_vit_image_tokens(base_hidden.clone(), pixel_values, mask, kwargs, ref_vision, ref_aligner)

    vision_tt = HunyuanTtSiglip2Vision(device, vision_state_dict, num_layers=NUM_LAYERS)
    aligner_tt = HunyuanTtLightProjector(device, aligner_state_dict)
    vision_ad = TtVisionModelHostAdapter(device, vision_tt)
    aligner_ad = TtAlignerHostAdapter(device, aligner_tt)

    with torch.no_grad():
        out = instantiate_vit_image_tokens(base_hidden.clone(), pixel_values, mask, kwargs, vision_ad, aligner_ad)

    assert tuple(out.shape) == tuple(ref.shape), f"shape {tuple(out.shape)} != {tuple(ref.shape)}"
    passing, pcc = comp_pcc(ref, out, PCC_THR)
    logger.info(f"host-adapter instantiate n_img={n_img} span=[{span.start}:{span.stop}] PCC={pcc} (>= {PCC_THR})")
    assert passing, f"PCC {pcc} < {PCC_THR}"
