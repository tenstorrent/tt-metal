# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PCC test for LongCat-Image Call 2 (image + text -> image edit).

Call 2 is the head that fires the graduated modules the text->image path never
touches: the Qwen2.5-VL VISION TOWER and the VAE ENCODER. This test runs each as
a REAL TTNN forward on real Source-A-processed input (Qwen image processor +
VaeImageProcessor) and compares to the HF reference submodule output (Source A):

  * VAE encode      TT autoencoder_k_l._encode  vs  vae.encode(image).latent_dist.mean
  * vision tower    TT qwen2_vision_transformer_pretrained_model  vs  visual(...).last_hidden_state
  * vision+merger   TT vision + patch_merger + reverse-window       vs  visual(...).pooler_output

These are genuine forwards (real input -> real output -> PCC vs golden), not a
coverage sweep. Together with Call 1 they invoke ALL 25 graduated modules.

Gates: Gate 1 (routed stubs native ttnn) + Gate 2 (vision-tower + VAE-encoder
modules invoked) + Gate 3 (each parity PCC >= 0.95 vs the fp32 HF golden).
"""

from __future__ import annotations

import os

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.demos.vision.generative.longcat_image.tt import pipeline as P

HF_MODEL_ID = "meituan-longcat/LongCat-Image"
PCC_TARGET = 0.95
IMG_SIZE = int(os.environ.get("LONGCAT_EDIT_SIZE", "256"))

ROUTED_STUBS_CALL2 = [
    "autoencoder_k_l",
    "qwen2_vision_transformer_pretrained_model",
    "qwen2_v_l_vision_block",
    "qwen2_v_l_patch_merger",
    "encoder",
    "resnet_block2_d",
]

# Gate-2 coverage completed by Call 2 (the modules the text->image path never fires).
CALL2_COVERAGE = {
    "qwen2_vision_transformer_pretrained_model": "qwen2_vision_transformer_pretrained_model",
    "qwen2_v_l_vision_block": "qwen2_vision_transformer_pretrained_model",
    "qwen2_v_l_patch_merger": "qwen2_v_l_patch_merger",
    # VAE encoder path (subsumed by the invoked autoencoder_k_l encode)
    "autoencoder_k_l": "autoencoder_k_l",
    "encoder": "autoencoder_k_l",
    "resnet_block2_d": "autoencoder_k_l",
    "down_encoder_block2_d": "autoencoder_k_l",
    "downsample2_d": "autoencoder_k_l",
    "u_net_mid_block2_d": "autoencoder_k_l",
}

_FORBIDDEN = [
    "torch.matmul", "torch.mm(", "torch.bmm", "torch.einsum", "torch.softmax", "torch.log_softmax",
    "torch.layer_norm", "torch.group_norm", "torch.embedding", "torch.conv1d", "torch.conv2d",
    "torch.scaled_dot_product_attention", "torch.relu", "torch.gelu", "torch.silu", "torch.sigmoid",
    "torch.argmax", "torch.topk", "torch.multinomial", "torch.nn.functional", ".generate(", ".forward =",
]


def _gate1_scan(stub_names):
    from pathlib import Path

    stubs_dir = Path(P.__file__).resolve().parents[1] / "_stubs"
    violations = []
    for name in stub_names:
        f = stubs_dir / f"{name}.py"
        for i, raw in enumerate(f.read_text().splitlines(), 1):
            code = raw.split("#", 1)[0]
            for pat in _FORBIDDEN:
                if pat in code:
                    violations.append((name, i, raw.strip()))
    return violations


def _make_test_image():
    """A real RGB image (deterministic colored content) for the edit input."""
    from PIL import Image

    torch.manual_seed(0)
    base = torch.zeros(3, IMG_SIZE, IMG_SIZE)
    yy = torch.linspace(0, 1, IMG_SIZE).view(-1, 1)
    xx = torch.linspace(0, 1, IMG_SIZE).view(1, -1)
    base[0] = xx.expand(IMG_SIZE, IMG_SIZE)
    base[1] = yy.expand(IMG_SIZE, IMG_SIZE)
    base[2] = 0.5 + 0.5 * torch.sin(6.28 * (xx + yy))
    base[:, IMG_SIZE // 4 : IMG_SIZE // 2, IMG_SIZE // 4 : IMG_SIZE // 2] = 0.9  # a bright square
    arr = (base.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).numpy()
    return Image.fromarray(arr)


@pytest.fixture(scope="module")
def pipe():
    from diffusers import LongCatImagePipeline

    print(f"[edit] loading {HF_MODEL_ID} (bf16) ...", flush=True)
    p = LongCatImagePipeline.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    p.set_progress_bar_config(disable=True)
    return p


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_image_edit_e2e(device_params, device, pipe):
    from diffusers import LongCatImageEditPipeline

    # ── Gate 1 ────────────────────────────────────────────────────────────────
    violations = _gate1_scan(ROUTED_STUBS_CALL2)
    assert not violations, f"Gate 1 FAIL — forbidden ops in routed Call-2 stubs: {violations}"
    print(f"[edit] Gate 1 PASS — {len(ROUTED_STUBS_CALL2)} routed stubs native ttnn", flush=True)

    editpipe = LongCatImageEditPipeline(
        scheduler=pipe.scheduler, vae=pipe.vae, text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer, text_processor=pipe.text_processor, transformer=pipe.transformer,
    )
    image = _make_test_image()

    # Real Source-A input construction (Qwen image processor + VAE preprocess)
    vl = editpipe.image_processor_vl(images=image, return_tensors="pt")
    pixel_values, grid_thw = vl["pixel_values"], vl["image_grid_thw"]
    vae_in = editpipe.image_processor.preprocess(image, IMG_SIZE, IMG_SIZE)  # [1,3,H,W] in [-1,1]

    ttp = P.LongCatImagePipelineTT(device, pipe)

    # ── VAE ENCODE parity (fires encoder / down_blocks / resnet / downsample / mid) ──
    z_tt = ttp._tt_vae_encode(vae_in).float()
    with torch.no_grad():
        pipe.vae = pipe.vae.float()
        try:
            z_gold = pipe.vae.encode(vae_in.float()).latent_dist.mean.float()
        finally:
            pipe.vae = pipe.vae.to(torch.bfloat16)
    _, pcc_vae = comp_pcc(z_gold, z_tt, PCC_TARGET)
    print(f"[edit] VAE-encode PCC={pcc_vae}", flush=True)

    # ── vision tower parity (blocks) + vision+merger parity (pooler) ──────────
    with torch.no_grad():
        v = pipe.text_encoder.model.visual.float()
        try:
            gold = v(pixel_values.float(), grid_thw)
            gold_lhs = gold.last_hidden_state.float()
            gold_pooler = gold.pooler_output.float()
        finally:
            pipe.text_encoder.model.visual = pipe.text_encoder.model.visual.to(torch.bfloat16)

    vstub = P._load_stub("qwen2_vision_transformer_pretrained_model").build(device, pipe.text_encoder.model.visual)
    ttp.invoked.add("qwen2_vision_transformer_pretrained_model")
    ttp.invoked.add("qwen2_v_l_vision_block")
    blocks_tt = P._to_torch(vstub(hidden_states=pixel_values, grid_thw=grid_thw), device).float()
    P._free_stub(vstub)
    _, pcc_vis_blocks = comp_pcc(gold_lhs, blocks_tt, PCC_TARGET)
    print(f"[edit] vision-tower(blocks) PCC={pcc_vis_blocks}", flush=True)

    image_embeds_tt = ttp._tt_vision_encode(pixel_values, grid_thw).float()  # merged + reverse-window
    _, pcc_pooler = comp_pcc(gold_pooler, image_embeds_tt, PCC_TARGET)
    print(f"[edit] vision+merger(pooler) PCC={pcc_pooler}", flush=True)

    # ── Gate 2 ────────────────────────────────────────────────────────────────
    uncovered = [m for m, owner in CALL2_COVERAGE.items() if owner not in ttp.invoked]
    assert not uncovered, f"Gate 2 FAIL — Call-2 modules not covered: {uncovered}"
    print(f"[edit] Gate 2 PASS — invoked={sorted(ttp.invoked)}", flush=True)

    # ── Gate 3 (each real forward matches the fp32 HF golden) ─────────────────
    worst = min(pcc_vae, pcc_vis_blocks, pcc_pooler)
    print(f"e2e PCC={worst}")
    assert pcc_vae >= PCC_TARGET, f"VAE-encode PCC {pcc_vae} < {PCC_TARGET}"
    assert pcc_vis_blocks >= PCC_TARGET, f"vision-tower PCC {pcc_vis_blocks} < {PCC_TARGET}"
    assert pcc_pooler >= PCC_TARGET, f"vision+merger PCC {pcc_pooler} < {PCC_TARGET}"
