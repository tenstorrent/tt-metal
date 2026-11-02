# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Vision tower (TtVisionTower) vs HF Glm5NextVisionModel with the real ``model.visual.*`` weights.

    TT_VISIBLE_DEVICES=4 pytest models/demos/glm53_flash/tests/test_vision.py -s

Reference: ``Glm5NextVisionModel`` built from the checkpoint config, run in fp32 on the CPU over the
``Glm5NextImageProcessor`` output of four deterministic synthetic 448x448 images; ``get_image_features``
is ``visual(pixel_values, grid_thw).pooler_output`` split by ``grid.prod(-1) // merge**2`` (= 256 tokens
per image), reproduced here without instantiating the 320 B language model.

Images and bars. The tower is chaotic on smooth images: for the plain RGB gradient the fp32 HF model
itself moves to PCC 0.96 under 1e-3 pixel noise and HF run in bf16 scores 0.86 against HF fp32, so no
bf16-weight implementation can meet 0.995 there. The hard bar (PCC >= 0.995 per image) is therefore
applied to the three textured images (seeded noise, textured shapes, stripes), whose fp32 model is
stable (>= 0.9993 under 1e-3 noise) and whose HF-bf16 floor is ~0.998; the gradient image stays in
the set as the documented ill-conditioned case with a relative bar: TT must be at least as close to
HF fp32 as HF's own bf16 run is, minus a margin.

Tests: (1) patch embed, every block on the HF block input (real activations) and the post-norm +
downsample + merger tail; (2) the full tower on device vs the HF per-image features, parametrized
over the weight dtype (bf16 default, bfp8_b option).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image, ImageDraw

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.vision import GLM53VisionConfig, TtVisionTower, load_vision_weights

PCC_BAR = 0.995
# bfp8_b weights are an option, not the default: measured 0.9944 / 0.9968 / 0.9962 on the three textured
# images (bf16 weights: 0.9968 / 0.9982 / 0.9975), i.e. they miss the 0.995 bar on the noise image.
BFP8_BAR = 0.99
CHAOTIC_MARGIN = 0.03  # gradient image: TT vs fp32 >= (HF-bf16 vs fp32) - margin
IMAGE_NAMES = ("noise", "textured_shapes", "stripes", "gradient")
HARD_BAR_IMAGES = ("noise", "textured_shapes", "stripes")


def synthetic_images(size: int = 448) -> dict[str, Image.Image]:
    """Deterministic test images (seeded): uniform noise, flat shapes with grain, sinusoidal stripes
    with grain, and the plain RGB gradient (the ill-conditioned case)."""
    y, x = np.mgrid[0:size, 0:size]
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    shapes = Image.new("RGB", (size, size), (240, 240, 240))
    d = ImageDraw.Draw(shapes)
    d.ellipse((60, 60, 220, 220), fill=(200, 30, 30))
    d.rectangle((250, 80, 420, 300), fill=(30, 60, 200))
    d.polygon([(100, 400), (300, 260), (400, 420)], fill=(20, 160, 60))
    textured = np.clip(np.asarray(shapes, dtype=np.float32) + rng.normal(0, 18, (size, size, 3)), 0, 255)
    a = 128 + 100 * np.sin(x / 7.0) * np.cos(y / 11.0)
    stripes = np.stack([a, np.roll(a, 60, 0), np.roll(a, 120, 1)], -1) + rng.normal(0, 10, (size, size, 3))
    grad = np.stack([x * 255 // (size - 1), y * 255 // (size - 1), (x + y) * 255 // (2 * size - 2)], -1)
    return {
        "noise": Image.fromarray(noise),
        "textured_shapes": Image.fromarray(textured.astype(np.uint8)),
        "stripes": Image.fromarray(np.clip(stripes, 0, 255).astype(np.uint8)),
        "gradient": Image.fromarray(grad.astype(np.uint8)),
    }


def build_hf_vision(snapshot, weights, dtype=torch.float32):
    """``Glm5NextVisionModel`` from the raw tower weights (non-persistent rope buffer recomputed)."""
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel, Glm5NextVisionRotaryEmbedding

    vcfg = AutoConfig.from_pretrained(str(snapshot)).vision_config
    vcfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        hf = Glm5NextVisionModel(vcfg)
    hf = hf.to_empty(device="cpu")
    missing, unexpected = hf.load_state_dict(weights, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    inv_freq, _ = Glm5NextVisionRotaryEmbedding.compute_axial_rope_parameters(vcfg)
    hf.rotary_pos_emb.inv_freq = inv_freq.clone()
    hf.rotary_pos_emb.original_inv_freq = inv_freq.clone()
    return hf.to(dtype).eval()


@pytest.fixture(scope="module")
def ref():
    from transformers import AutoProcessor

    snapshot = default_snapshot_dir()
    cfg = GLM53VisionConfig.from_snapshot(snapshot)
    weights = load_vision_weights(snapshot)
    hf = build_hf_vision(snapshot, weights)
    proc = AutoProcessor.from_pretrained(str(snapshot)).image_processor
    images = synthetic_images()
    feats = proc(images=[images[n] for n in IMAGE_NAMES], return_tensors="pt")
    pv, grid = feats["pixel_values"], feats["image_grid_thw"]
    logger.info(f"processor {type(proc).__name__}: pixel_values {tuple(pv.shape)} grid_thw {grid.tolist()}")

    block_io: list[tuple[torch.Tensor, torch.Tensor]] = [None] * cfg.depth
    captured = {}

    def block_hook(i):
        def hook(_m, args, _kwargs, out):
            block_io[i] = (args[0].detach().clone(), out.detach().clone())

        return hook

    handles = [blk.register_forward_hook(block_hook(i), with_kwargs=True) for i, blk in enumerate(hf.blocks)]
    handles.append(hf.patch_embed.register_forward_hook(lambda _m, _a, out: captured.__setitem__("embed", out)))
    with torch.no_grad():
        out = hf(pv.float(), grid_thw=grid)
    for h in handles:
        h.remove()
    split = (grid.prod(-1) // cfg.merge_group).tolist()
    features = list(torch.split(out.pooler_output, split))  # == Glm5NextModel.get_image_features
    # the model as it ships (bf16 weights, bf16 pixels and activations): the realistic precision floor
    hf16 = build_hf_vision(snapshot, weights, torch.bfloat16)
    with torch.no_grad():
        features16 = list(torch.split(hf16(pv.to(torch.bfloat16), grid_thw=grid).pooler_output.float(), split))
    floor = {n: pcc(a, b) for n, a, b in zip(IMAGE_NAMES, features16, features)}
    logger.info(f"HF-bf16 vs HF-fp32 feature PCC (precision floor of the shipped model): {floor}")
    return dict(
        cfg=cfg,
        weights=weights,
        pv=pv,
        grid=grid,
        block_io=block_io,
        embed=captured["embed"],
        features=features,
        floor=floor,
        split=split,
    )


def _to_dev(device, t: torch.Tensor, dtype) -> ttnn.Tensor:
    return ttnn.from_torch(t.float().reshape(1, 1, *t.shape[-2:]), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _rows(t: ttnn.Tensor, n: int) -> torch.Tensor:
    return ttnn.to_torch(t).float().reshape(-1, t.shape[-1])[:n]


def _check_images(ps: dict[str, float], floor: dict[str, float], bar: float = PCC_BAR):
    for n in HARD_BAR_IMAGES:
        assert ps[n] >= bar, (n, ps[n], bar)
    assert ps["gradient"] >= floor["gradient"] - CHAOTIC_MARGIN, (ps["gradient"], floor["gradient"])


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_vision_blocks(mesh_device, ref):
    """Patch embed, each of the 24 blocks on the HF block input, and the merge tail, PCC vs HF fp32."""
    cfg = ref["cfg"]
    tower = TtVisionTower(mesh_device, cfg, ref["weights"])
    inputs = tower.prepare_inputs(ref["pv"], ref["grid"])
    N = inputs.num_patches
    assert N % 32 == 0 and inputs.patches.shape[-2] == N, (N, inputs.patches.shape)

    p = pcc(_rows(tower.embed(inputs), N), ref["embed"])
    logger.info(f"patch_embed (matmul over unfolded patches): PCC {p:.6f}")
    assert p >= PCC_BAR

    worst = 1.0
    for i, (x_in, x_out) in enumerate(ref["block_io"]):
        y = tower.blocks[i](_to_dev(mesh_device, x_in, tower.act_dtype), inputs.cos, inputs.sin, inputs.cu_window)
        p = pcc(_rows(y, N), x_out)
        ttnn.deallocate(y)
        logger.info(f"block {i:2d}: PCC {p:.6f}")
        worst = min(worst, p)
        assert p >= PCC_BAR, (i, p)
    logger.info(f"blocks worst PCC {worst:.6f}")

    merged = tower.merge(_to_dev(mesh_device, ref["block_io"][-1][1], tower.act_dtype))
    tt_feats = tower.image_embeddings(inputs, merged)
    ps = {n: pcc(a, b) for n, a, b in zip(IMAGE_NAMES, tt_feats, ref["features"])}
    logger.info(f"merge tail (post_layernorm + downsample + merger) on the HF block-23 output: PCC {ps}")
    assert min(ps.values()) >= PCC_BAR


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bfp8"])
def test_vision_tower(mesh_device, ref, weight_dtype):
    """Full tower on device vs HF ``get_image_features`` (fp32), per image; bf16 weights (default) at the
    0.995 bar, bfp8_b weights (option) at their documented bar."""
    cfg = ref["cfg"]
    tower = TtVisionTower(mesh_device, cfg, ref["weights"], weight_dtype=weight_dtype)
    inputs = tower.prepare_inputs(ref["pv"], ref["grid"])
    out = tower(inputs)
    tt_feats = tower.image_embeddings(inputs, out)
    assert [f.shape[0] for f in tt_feats] == ref["split"]
    ps = {n: pcc(a, b) for n, a, b in zip(IMAGE_NAMES, tt_feats, ref["features"])}
    for n, f in zip(IMAGE_NAMES, tt_feats):
        logger.info(
            f"{n:16s}: {tuple(f.shape)} tokens, full-tower PCC {ps[n]:.6f} (HF-bf16 floor {ref['floor'][n]:.6f})"
        )
    logger.info(
        f"tokens per image {inputs.split_sizes} (grid {ref['grid'].tolist()}); decoder placeholders: "
        f"image_token_id {cfg.image_token_id} x {inputs.split_sizes[0]} per image, wrapped by "
        f"image_start {cfg.image_start_token_id} / image_end {cfg.image_end_token_id}"
    )
    _check_images(ps, ref["floor"], PCC_BAR if weight_dtype == ttnn.bfloat16 else BFP8_BAR)
