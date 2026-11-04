# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Accuracy of the MiniMax-M3 vision encoder with the checkpoint's weights, against Hugging Face goldens.

Builds TtMiniMaxM3VisionEncoder from the checkpoint as the encoder service's load() does (from_weights_path: all 32
layers and the projector on one chip) and runs it on the images of golden_hf_vision.py as encode() does: resize_image,
forward(pixels), then read_embeddings into a host buffer. The tower output comes from the same encoder's model
(M3VLVisionModel.forward_tower). Both are compared with Hugging Face's MiniMaxM3VLModel vision tower and projector in
fp32 on the CPU, stored with the goldens, through the vision unit tests' check_close:
  - vision tokens (final, L / 4 x 6144): PCC >= 0.998 and relative L2 error <= 0.10 on every image;
  - tower output (L x 1280): PCC >= Hugging Face's own bf16 PCC on that image minus 0.002, and relative L2 error
    <= 0.12. A bf16 run of the Hugging Face model already loses more on some images than on others (tower PCC
    0.99826 on house, 0.99979 on perf_report), so the PCC bar follows it instead of being one number;
  - no NaN or Inf in either output: comp_pcc counts them as zeros, so a few would barely move the PCC.
PCC cannot see a scale or an offset. The relative L2 bounds catch large ones and are loose on purpose, about twice
the largest error measured on these images. On saved P150 outputs of this encoder, an extra scale passes them up to
about 6% upward or 11% downward on the vision tokens, and about 10% upward or 11% downward on the tower output; those
outputs already had a least-squares gain against the goldens of 1.017 to 1.023 on the vision tokens and 1.004 to
1.008 on the tower output. So each output's log line also gives the PCC margin, Hugging Face bf16's relative L2 error,
the gain (1.0: no scale error) and the 1st percentile of the cosine between its rows and the goldens', beside Hugging
Face bf16's. Those are logged, not checked.

The goldens use MiniMax-M3.1-Flash, the newer release, on which the encoder's accuracy was measured. Its vision tower
and projector are MiniMax-M3's (same architecture, tensor names and image processor) with other weights, so this code
runs both, but each checkpoint needs its own goldens; golden_hf_vision.py makes them for the checkpoint in HF_MODEL.

Needs the checkpoint and the goldens, and skips without either:
  HF_MODEL              the checkpoint the goldens were made from (manifest.json names it), or a directory with
                        its config.json, model.safetensors.index.json and the shards that hold the vision tensors;
  M3_VISION_GOLDEN_DIR  the output of golden_hf_vision.py: manifest.json and one <image>.safetensors per image;
                        default $HF_MODEL/golden/vision.
Before any device work, on the CPU:
  - each golden file must have the sha256 that manifest.json gives, so a half-updated copy fails;
  - resize_image and patch_rows (tt/vision/preprocess.py) must give each image's pixel values and grid as the
    goldens hold them, the image processor's, bit for bit; resize_image must also give the same pixels for the image
    as a uint8 RGBA array, whose alpha channel it drops, as for the PIL RGB image;
  - HF_MODEL must be the goldens' checkpoint: its patch embedding and pre_layrnorm must reproduce the goldens' first
    pre_layrnorm rows, and its projector the first vision tokens from the goldens' tower output. Goldens of another
    checkpoint fail there, not as a low PCC. These are 11 of the checkpoint's 523 vision tensors; weights that differ
    only in the blocks fail as a low PCC.

Run:
    HF_MODEL=<checkpoint> M3_VISION_GOLDEN_DIR=<goldens> pytest models/demos/minimax_m3/tests/test_vision_accuracy.py
"""
from __future__ import annotations

import dataclasses
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
from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionEncoder
from models.demos.minimax_m3.tt.vision.loader import load_vision_state_dict
from models.demos.minimax_m3.tt.vision.preprocess import patch_rows, resize_image
from models.demos.minimax_m3.vision.encoder import TtMiniMaxM3VisionEncoder, read_embeddings, vision_config

from .golden_hf_vision import GOLDEN_FORMAT, IMAGES, gain, load_image, sha256, token_cosine_p01
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
    """The manifest and, for each image of IMAGES, its tensors: pixel_values, image_grid_thw, tower_out, final and
    pre_layrnorm_rows. Each file must have the manifest's sha256."""
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


def _resized_images(goldens) -> dict:
    """Each image as resize_image gives it, after checking that patch_rows of it gives the goldens' pixel values and
    grid bit for bit, and that resize_image gives the same pixels for the image as a uint8 RGBA array."""
    images, failed = {}, []
    for case, golden in goldens.items():
        image, _ = load_image(case)
        pixels = resize_image(image).pixels
        pixel_values, grid_thw = patch_rows(pixels)
        want_values, want_grid = golden["pixel_values"], golden["image_grid_thw"]
        if not torch.equal(grid_thw, want_grid):
            failed.append(f"{case}: grid {grid_thw.tolist()}, the goldens' {want_grid.tolist()}")
        elif not torch.equal(pixel_values, want_values):
            differ = int((pixel_values != want_values).sum())
            failed.append(f"{case}: {differ} of {want_values.numel()} pixel values differ from the goldens'")
        if not np.array_equal(resize_image(np.asarray(load_image(case, "RGBA")[0])).pixels, pixels):
            failed.append(f"{case}: resize_image gives other pixels for the image as an RGBA array")
        images[case] = pixels
    assert not failed, (
        f"resize_image and patch_rows do not give the goldens' input (here torch {torch.__version__}, torchvision"
        f" {torchvision.__version__} and Pillow {PIL.__version__}):\n" + "\n".join(failed)
    )
    return images


def _checkpoint_reference() -> MiniMaxM3VisionEncoder:
    """The Torch reference's patch embedding, pre_layrnorm and projector with HF_MODEL's weights, without the blocks."""
    with open(os.path.join(HF_MODEL, "config.json")) as f:
        config = dataclasses.replace(vision_config(json.load(f)), num_hidden_layers=0)
    return MiniMaxM3VisionEncoder(config).load_hf_state_dict(load_vision_state_dict(HF_MODEL)).eval()


def _check_checkpoint(reference, goldens, manifest):
    """Check that HF_MODEL is the goldens' checkpoint, on the CPU: the reference's patch embedding and pre_layrnorm
    must reproduce the goldens' first pre_layrnorm rows, and its projector the vision tokens of the goldens' first
    PROJECTOR_ROWS tower rows."""
    tokens = PROJECTOR_ROWS // reference.config.spatial_merge_size**2
    failed = []
    for case, golden in goldens.items():
        rows = golden["pre_layrnorm_rows"].float()
        checks = (
            ("pre_layrnorm", rows, reference.pre_layrnorm(reference.embed(golden["pixel_values"][: rows.shape[0]]))),
            ("projector", golden["final"][:tokens], reference.projector(golden["tower_out"][:PROJECTOR_ROWS])),
        )
        for what, ref, out in checks:
            _, pcc = comp_pcc(ref, out, CHECKPOINT_MIN_PCC)
            if not pcc >= CHECKPOINT_MIN_PCC:
                failed.append(f"{what} PCC {pcc:.4f} on {case}")
    assert not failed, (
        f"HF_MODEL={HF_MODEL} is not the checkpoint of the goldens in {GOLDEN_DIR} ({manifest['checkpoint']}), or"
        " load_vision_state_dict reads its patch embedding, pre_layrnorm or projector wrongly: " + ", ".join(failed)
    )


def _compare(case, golden, entry, tower, final):
    """Check the tower output and the vision tokens of one image against the goldens with check_close, and log each
    check's line with the PCC margin, Hugging Face bf16's relative L2 error and the 1st-percentile row cosine beside
    Hugging Face bf16's; a passing line also gets the PCC bar, the relL2 bound and the gain, which check_close's line
    of a failing check already gives. Returns the lines of the checks that fail."""
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
    """On every image, vision tokens at PCC >= 0.998 and relL2 <= 0.10, and the tower output within 0.002 of Hugging
    Face's bf16 PCC and at relL2 <= 0.12, with no NaN or Inf."""
    manifest, goldens = _load_goldens()
    images = _resized_images(goldens)
    _check_checkpoint(_checkpoint_reference(), goldens, manifest)
    cores = mesh_device.compute_with_storage_grid_size()
    logger.info(f"goldens {GOLDEN_DIR} ({manifest['checkpoint']}), HF_MODEL {HF_MODEL}, cores {cores.x}x{cores.y}")

    encoder = TtMiniMaxM3VisionEncoder.from_weights_path(mesh_device, HF_MODEL)
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
