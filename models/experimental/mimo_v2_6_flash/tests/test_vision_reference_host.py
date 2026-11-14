# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks of reference/vision.py on real weights.

* The reference equals the checkpoint's HF ``MiMoVisionTransformer`` (run from its ``modeling_mimo_v2.py`` on the
  CPU) bit for bit, with the HF merger norm (``merger_norm="layernorm"``), on images and a short video from the
  vLLM processor.
* The same on synthetic pixels and grids, without vLLM (``test_reference_matches_hf_without_vllm``): the bit-for-bit
  claim is exercised wherever the checkpoint is, also where no vLLM is installed for the processor.
* The default (vLLM) merger norm differs from HF's by design (see reference/vision.py): the distance is logged and
  held inside MERGER_NORM_PCC / MERGER_NORM_REL, so a change of either norm shows.
* Query-blocked attention (long sequences) matches the one-shot path to fp32 rounding.
* Token counts from the grid match the vLLM processor's placeholder counts.

Needs the checkpoint (``MIMO_MODEL_PATH``); the tests on processor inputs also need a vLLM install. When the
checkpoint's remote code does not import (a transformers version it was not written for) the HF comparisons skip;
MIMO_REQUIRE_HF_VISION=1 makes that a failure, for a run that must not pass without them. Keep ``OMP_NUM_THREADS``
<= 16.
"""

import importlib
import os
import shutil
import sys
import tempfile

import pytest
import torch
from loguru import logger

from models.experimental.mimo_v2_6_flash.reference import vision as ref
from models.experimental.mimo_v2_6_flash.reference import vision_inputs as vin
from models.experimental.mimo_v2_6_flash.tests.common import model_path, pcc

MISSING_IN_CHECKPOINT = {"merger.ln_q.bias", "merger.mlp.0.bias", "merger.mlp.2.bias"}
# vLLM's RMSNorm merger (the default here) against HF's LayerNorm one, on the processor's images / video: measured
# PCC 0.977 / 0.953, relative L2 0.21 / 0.30 (MOPD, vLLM 0.31).
MERGER_NORM_PCC = 0.93
MERGER_NORM_REL = 0.40


def _import_hf_modeling(path):
    """The checkpoint's modeling_mimo_v2.py as a package (it imports .configuration_mimo_v2), or skip."""
    pkg_root = tempfile.mkdtemp(prefix="mimo_hf_")
    pkg = os.path.join(pkg_root, "mimo_hf_vision")
    os.makedirs(pkg)
    for name in ("modeling_mimo_v2.py", "configuration_mimo_v2.py"):
        shutil.copy(os.path.join(path, name), os.path.join(pkg, name))
    open(os.path.join(pkg, "__init__.py"), "w").close()
    sys.path.insert(0, pkg_root)
    try:
        return importlib.import_module("mimo_hf_vision.modeling_mimo_v2")
    except Exception as e:  # transformers too old / too new for the remote code
        if os.environ.get("MIMO_REQUIRE_HF_VISION") == "1":
            raise
        pytest.skip(f"cannot import the checkpoint's modeling_mimo_v2.py: {e!r}")
    finally:
        sys.path.remove(pkg_root)


@pytest.fixture(scope="module")
def setup():
    torch.set_num_threads(min(16, torch.get_num_threads()))
    path = model_path()
    cfg = ref.VisionConfig.from_model_path(path)
    sd = ref.load_vision_state_dict(path)
    return path, cfg, sd


@pytest.fixture(scope="module")
def inputs(setup):
    path, _, _ = setup
    pytest.importorskip("vllm")  # the processor; the tests on synthetic inputs run without it
    images = vin.process_images(
        path, [vin.sample_image(size=(224, 224)), vin.sample_image("house_in_field_1080p.jpg", size=(288, 160))]
    )
    video = vin.process_video(path, vin.sample_video(num_frames=4, size=(192, 128)))
    return {
        "images": (images["pixel_values"], images["image_grid_thw"], images["num_tokens"]),
        "video": (video["pixel_values_videos"], video["video_grid_thw"], video["num_tokens"]),
    }


@pytest.fixture(scope="module")
def hf_visual(setup):
    path, _, sd = setup
    import json

    modeling = _import_hf_modeling(path)
    with open(os.path.join(path, "config.json")) as f:
        vision_config = json.load(f)["vision_config"]
    model = modeling.MiMoVisionTransformer(modeling._as_namespace(vision_config)).float().eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert set(missing) <= MISSING_IN_CHECKPOINT, missing
    # from_pretrained zero-initialises what the checkpoint lacks (Linear bias, LayerNorm bias).
    with torch.no_grad():
        for name in missing:
            model.get_parameter(name).zero_()
    return model


@pytest.mark.parametrize("item", ["images", "video"])
def test_token_counts_match_processor(inputs, setup, item):
    _, cfg, _ = setup
    _, grid, num_tokens = inputs[item]
    per_item = [ref.num_merged_tokens([g], cfg.spatial_merge_size) for g in ref.as_grid_list(grid)]
    assert per_item == list(num_tokens), (per_item, num_tokens)


@pytest.mark.parametrize("item", ["images", "video"])
def test_reference_matches_hf(inputs, setup, hf_visual, item):
    _, cfg, sd = setup
    pixel_values, grid, _ = inputs[item]
    hf_sd = dict(sd)
    for name in ("merger.mlp.0.bias", "merger.mlp.2.bias"):
        hf_sd[name] = torch.zeros(hf_visual.get_parameter(name).shape)
    model = ref.MiMoVisionReference(cfg, hf_sd, merger_norm="layernorm")
    with torch.no_grad():
        expected = hf_visual(pixel_values, grid)
        got = model(pixel_values, grid)
    assert got.shape == (ref.num_merged_tokens(grid, cfg.spatial_merge_size), cfg.out_hidden_size)
    max_diff = (got - expected).abs().max().item()
    logger.info(f"{item}: grid {ref.as_grid_list(grid)}, max |ref - HF| = {max_diff:.3e}")
    assert torch.equal(got, expected), max_diff


@pytest.mark.parametrize("grids", [[(1, 14, 14), (1, 10, 18)], [(2, 8, 12)], [(1, 2, 2)], [(4, 6, 4), (1, 4, 6)]])
def test_reference_matches_hf_without_vllm(setup, hf_visual, grids):
    """The reference against the checkpoint's HF module bit for bit, on pixel rows drawn here for the given (t, h, w)
    grids (images, a video, the smallest grid, a mix): no processor, so no vLLM."""
    _, cfg, sd = setup
    rows = sum(t * h * w for t, h, w in grids)
    pixel_values = torch.randn(rows, cfg.patch_dim, generator=torch.Generator().manual_seed(rows))
    grid = torch.tensor(grids)
    hf_sd = dict(sd)
    for name in ("merger.mlp.0.bias", "merger.mlp.2.bias"):
        hf_sd[name] = torch.zeros(hf_visual.get_parameter(name).shape)
    model = ref.MiMoVisionReference(cfg, hf_sd, merger_norm="layernorm")
    with torch.no_grad():
        expected = hf_visual(pixel_values, grid)
        got = model(pixel_values, grid)
    assert got.shape == (ref.num_merged_tokens(grid, cfg.spatial_merge_size), cfg.out_hidden_size)
    assert torch.equal(got, expected), (got - expected).abs().max().item()


@pytest.mark.parametrize("item", ["images", "video"])
def test_merger_norm_hf_vs_vllm(inputs, setup, item):
    """How far vLLM's RMSNorm merger (followed here) lands from HF's LayerNorm one: logged, and held inside the
    measured distance (MERGER_NORM_PCC / MERGER_NORM_REL). The two are not the same function (HF's centres the
    mean), so they also must not come out equal."""
    _, cfg, sd = setup
    pixel_values, grid, _ = inputs[item]
    with torch.no_grad():
        out_rms, hidden = ref.MiMoVisionReference(cfg, sd)(pixel_values, grid, return_hidden=True)
        out_ln = ref.MiMoVisionReference(cfg, sd, merger_norm="layernorm").merger(hidden[-1])
    score = pcc(out_rms, out_ln)
    rel = ((out_rms - out_ln).norm() / out_ln.norm()).item()
    logger.info(f"{item}: merger RMSNorm (vLLM) vs LayerNorm (HF): PCC {score:.6f}, relative L2 {rel:.4f}")
    assert out_rms.shape == out_ln.shape
    assert score >= MERGER_NORM_PCC and 0 < rel <= MERGER_NORM_REL, (item, score, rel)


def test_query_blocks_match_one_shot(inputs, setup):
    _, cfg, sd = setup
    pixel_values, grid, _ = inputs["images"]
    with torch.no_grad():
        one_shot = ref.MiMoVisionReference(cfg, sd, query_block=None)(pixel_values, grid)
        blocked = ref.MiMoVisionReference(cfg, sd, query_block=96)(pixel_values, grid)
    torch.testing.assert_close(blocked, one_shot, rtol=1e-4, atol=1e-4)
