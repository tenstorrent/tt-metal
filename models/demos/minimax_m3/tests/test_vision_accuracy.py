# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Accuracy of the MiniMax-M3 vision encoder with the checkpoint's weights, against Hugging Face goldens.

Builds TtMiniMaxM3VisionEncoder with from_weights_path and runs each golden image through the media module's
prepare, then forward and read_embeddings. Each image must pass check_close against the goldens:
  - vision tokens: PCC >= 0.998 and relative L2 error <= 0.10;
  - tower output (forward_tower): PCC no more than 0.002 below Hugging Face's own bf16 run, relative L2 error <= 0.12;
  - no NaN or Inf in either.
The log also gives the PCC margin, the gain and the worst rows' cosine, which are not checked.

Before any device work, it checks on the CPU that:
  - each golden file has the sha256 that manifest.json gives;
  - prepare and patch_rows give the goldens' input bit for bit;
  - HF_MODEL is the goldens' checkpoint;
  - load_vision_state_dict names and reads every vision tensor as transformers does.

Each checkpoint needs its own goldens, made by golden_hf_vision.py. Skips without them or the checkpoint:
    HF_MODEL=<checkpoint> M3_VISION_GOLDEN_DIR=<goldens> pytest models/demos/minimax_m3/tests/test_vision_accuracy.py
M3_VISION_GOLDEN_DIR defaults to $HF_MODEL/golden/vision.
"""
from __future__ import annotations

import json
import os

import numpy as np
import PIL
import pytest
import torch
import torchvision
from loguru import logger
from safetensors.torch import load_file

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.tt.vision.loader import load_vision_state_dict
from models.demos.minimax_m3.tt.vision.preprocess import patch_rows
from models.demos.minimax_m3.vision.encoder import TtMiniMaxM3VisionEncoder, read_embeddings
from models.demos.minimax_m3.vision.media import MiniMaxM3Media

from .golden_hf_vision import (
    GOLDEN_FORMAT,
    IMAGES,
    VISION_PREFIXES,
    gain,
    load_hf_model,
    load_image,
    sha256,
    token_cosine_p01,
)
from .test_factory import parametrize_mesh_with_fabric
from .unit.vision_helpers import check_close, to_host

FINAL_MIN_PCC = 0.998
TOWER_MARGIN = 0.002  # below Hugging Face's own bf16 tower PCC on the same image
# Loose bounds, against scale and offset errors: about twice the largest relative L2 error of the encoder on these
# images in a P150 run, 0.052 (vision tokens) and 0.059 (tower).
FINAL_MAX_REL_L2 = 0.10
TOWER_MAX_REL_L2 = 0.12
# The goldens' checkpoint reproduces their rows up to rounding; on these images MiniMax-M3's weights give 0.83 to 0.90
# on pre_layrnorm and 0.77 to 0.82 on the projector.
CHECKPOINT_MIN_PCC = 0.9999
PROJECTOR_ROWS = 64  # tower rows of the projector check: 16 vision tokens

HF_MODEL = os.getenv("HF_MODEL")
GOLDEN_DIR = os.getenv("M3_VISION_GOLDEN_DIR") or (os.path.join(HF_MODEL, "golden", "vision") if HF_MODEL else None)


def _manifest() -> dict:
    with open(os.path.join(GOLDEN_DIR, "manifest.json")) as f:
        return json.load(f)


def _skip_reason() -> str:
    """Why the test cannot run on this host; empty when the checkpoint and the goldens are there."""
    has_goldens = bool(GOLDEN_DIR) and os.path.isfile(os.path.join(GOLDEN_DIR, "manifest.json"))
    if not (HF_MODEL and os.path.isdir(HF_MODEL)):
        checkpoint = _manifest().get("checkpoint") if has_goldens else None
        checkpoint = checkpoint or "named in their manifest.json"
        return f"set HF_MODEL to the checkpoint the vision goldens were made from ({checkpoint})"
    if not has_goldens:
        return f"no vision goldens (manifest.json) in {GOLDEN_DIR}: set M3_VISION_GOLDEN_DIR"
    return ""


SKIP_REASON = _skip_reason()


def _load_goldens():
    """Return the manifest and the golden tensors of each image; each file must have the manifest's sha256."""
    manifest = _manifest()
    assert manifest.get("format") == GOLDEN_FORMAT, (
        f"{GOLDEN_DIR}/manifest.json has format {manifest.get('format')!r}, this test reads {GOLDEN_FORMAT!r}:"
        " make the goldens again with golden_hf_vision.py"
    )
    missing = [case for case in IMAGES if case not in manifest["cases"]]
    assert not missing, f"{GOLDEN_DIR}/manifest.json has no goldens for {missing}"
    goldens = {}
    for case in IMAGES:
        entry = manifest["cases"][case]
        path = os.path.join(GOLDEN_DIR, entry["file"])
        digest = sha256(path)
        assert digest == entry["sha256"], f"{path} has sha256 {digest}, manifest.json gives {entry['sha256']}"
        goldens[case] = load_file(path)
    return manifest, goldens


def _resized_images(goldens, media: MiniMaxM3Media) -> dict:
    """Each image resized by media's prepare, from the uint8 RGB array an encoder service passes.

    Asserts that patch_rows of each gives the goldens' pixel values and grid, bit for bit.
    """
    images, failed = {}, []
    for case, golden in goldens.items():
        pixels = media.prepare("image", np.array(load_image(case, "RGB")[0])).pixels
        pixel_values, grid_thw = patch_rows(pixels)
        want_values, want_grid = golden["pixel_values"], golden["image_grid_thw"]
        if not torch.equal(grid_thw, want_grid):
            failed.append(f"{case}: grid {grid_thw.tolist()}, the goldens' {want_grid.tolist()}")
        elif not torch.equal(pixel_values, want_values):
            differ = int((pixel_values != want_values).sum())
            failed.append(f"{case}: {differ} of {want_values.numel()} pixel values differ from the goldens'")
        images[case] = pixels
    assert not failed, (
        f"prepare and patch_rows do not give the goldens' input (here torch {torch.__version__}, torchvision"
        f" {torchvision.__version__} and Pillow {PIL.__version__}):\n" + "\n".join(failed)
    )
    return images


def _check_checkpoint(model, goldens, manifest):
    """Check on the CPU that HF_MODEL is the goldens' checkpoint.

    With its weights, Hugging Face's patch embedding and pre_layrnorm must reproduce the goldens' first pre_layrnorm
    rows, and its projector the goldens' first vision tokens.
    """
    tower, projector = model.vision_tower, model.multi_modal_projector
    tokens = PROJECTOR_ROWS // projector.spatial_merge_size**2
    failed = []
    for case, golden in goldens.items():
        rows = golden["pre_layrnorm_rows"].float()
        checks = (
            ("pre_layrnorm", rows, tower.pre_layrnorm(tower.embeddings(golden["pixel_values"][: rows.shape[0]]))),
            ("projector", golden["final"][:tokens], projector(golden["tower_out"][:PROJECTOR_ROWS])),
        )
        for what, ref, out in checks:
            _, pcc = comp_pcc(ref, out, CHECKPOINT_MIN_PCC)
            if not pcc >= CHECKPOINT_MIN_PCC:
                failed.append(f"{what} PCC {pcc:.4f} on {case}")
    assert not failed, (
        f"HF_MODEL={HF_MODEL} is not the checkpoint of the goldens in {GOLDEN_DIR} ({manifest['checkpoint']}): "
        + ", ".join(failed)
    )


def _check_loader(model):
    """Check on the CPU that load_vision_state_dict(HF_MODEL) gives the vision tensors of model.

    model is Hugging Face's. The loader must give exactly its vision tensors, under the same names and with the same
    values in bf16.
    """
    expected = {name: value for name, value in model.state_dict().items() if name.startswith(VISION_PREFIXES)}
    loaded = load_vision_state_dict(HF_MODEL, dtype=torch.bfloat16)
    wrong = sorted(
        name
        for name in expected.keys() | loaded.keys()
        if name not in expected or name not in loaded or not torch.equal(loaded[name], expected[name].bfloat16())
    )
    assert not wrong, (
        f"load_vision_state_dict (the loader's _remap) gives {len(wrong)} of the {len(expected)} vision tensors of"
        f" HF_MODEL={HF_MODEL} otherwise than transformers names them: " + ", ".join(wrong[:8])
    )


def _compare(case, golden, entry, tower, final):
    """Check one image's tower output and vision tokens against the goldens with check_close, and log the results.

    Returns the log lines of the checks that fail.
    """
    hf_bf16 = entry["hf_bf16"]
    checks = (
        ("final", golden["final"], final, FINAL_MIN_PCC, FINAL_MAX_REL_L2),
        ("tower", golden["tower_out"], tower, hf_bf16["tower"]["pcc"] - TOWER_MARGIN, TOWER_MAX_REL_L2),
    )
    failed = []
    for name, ref, out, min_pcc, max_rel_l2 in checks:
        line, passed = check_close(f"{case} {name}", ref, out, min_pcc, max_rel_l2)
        _, pcc = comp_pcc(ref, out, min_pcc)
        hf = hf_bf16[name]
        margin = f"PCC margin {pcc - min_pcc:+.6f}"
        rows = f"1st percentile row cosine {token_cosine_p01(ref, out):.5f}, Hugging Face bf16 {hf['cos_p01']:.5f}"
        if passed:
            line += (
                f"; PCC bar {min_pcc:.6f}, {margin}; relL2 bound {max_rel_l2}, Hugging Face bf16 relL2"
                f" {hf['rel_l2']:.4f}; gain {gain(ref, out):.4f}; {rows}"
            )
        else:
            line += f"; {margin}; Hugging Face bf16 relL2 {hf['rel_l2']:.4f}; {rows}"
        logger.info(line)
        if not passed:
            failed.append(line)
    return failed


@torch.no_grad()
@pytest.mark.skipif(bool(SKIP_REASON), reason=SKIP_REASON)
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_vision_accuracy(mesh_device, device_params):
    """On every golden image, the vision tokens and the tower output pass the checks of the module docstring."""
    manifest, goldens = _load_goldens()
    with open(os.path.join(HF_MODEL, "config.json")) as f:
        config = json.load(f)  # the service's config, which load() passes to from_weights_path
    images = _resized_images(goldens, MiniMaxM3Media(config))
    hf_model = load_hf_model(HF_MODEL)
    _check_checkpoint(hf_model, goldens, manifest)
    _check_loader(hf_model)
    del hf_model  # its fp32 vision weights, which the device run does not need
    cores = mesh_device.compute_with_storage_grid_size()
    logger.info(f"goldens {GOLDEN_DIR} ({manifest['checkpoint']}), HF_MODEL {HF_MODEL}, cores {cores.x}x{cores.y}")

    encoder = TtMiniMaxM3VisionEncoder.from_weights_path(mesh_device, HF_MODEL, config)
    failed = []
    for case, golden in goldens.items():
        pixels = images[case]
        tokens = encoder.forward(pixels)
        embeddings = np.empty(tuple(golden["final"].shape), dtype=np.uint16)  # the bf16 bits, as encode() stores them
        read_embeddings(tokens, embeddings)
        ttnn.deallocate(tokens)
        final = torch.from_numpy(embeddings).view(torch.bfloat16).float()
        tower_tt = encoder.model.forward_tower(*patch_rows(pixels))
        tower = to_host(tower_tt, mesh_device, golden["tower_out"].shape)
        ttnn.deallocate(tower_tt)
        failed += _compare(case, golden, manifest["cases"][case], tower, final)
    assert not failed, f"{len(failed)} checks failed:\n" + "\n".join(failed)
