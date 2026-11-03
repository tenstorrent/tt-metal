# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Accuracy of the MiniMax-M3 vision encoder with the checkpoint's weights, against Hugging Face goldens.

Runs M3VLVisionModel (all 32 layers and the projector) on one device for the images of golden_hf_vision.py and
compares with Hugging Face's MiniMaxM3VLModel vision tower and projector in fp32 on the CPU, stored with the goldens:
  - vision tokens (final, L / 4 x 6144): PCC >= 0.998 on every image;
  - tower output (L x 1280): PCC >= Hugging Face's own bf16 PCC on that image minus 0.002. A bf16 run of
    the Hugging Face model already loses more on some images than on others (tower PCC 0.99826 on house,
    0.99979 on perf_report), so the bar follows it instead of being one number;
  - no NaN or Inf in either output: comp_pcc counts them as zeros, so a few would barely move the PCC.

The goldens use MiniMax-M3.1-Flash, the newer release, on which the encoder's accuracy was measured. Its vision tower
and projector are MiniMax-M3's (same architecture, tensor names and image processor) with other weights, so this code
runs both, but each checkpoint needs its own goldens; golden_hf_vision.py makes them for the checkpoint in HF_MODEL.

Needs the checkpoint and the goldens, and skips without either:
  HF_MODEL              the checkpoint the goldens were made from (manifest.json names it), or a directory with
                        its model.safetensors.index.json and the shards that hold the vision tensors;
  M3_VISION_GOLDEN_DIR  the output of golden_hf_vision.py: manifest.json and one <image>.safetensors per image;
                        default $HF_MODEL/golden/vision.
Before the device run, the first rows of the pre_layrnorm output are recomputed on the CPU from HF_MODEL's
weights and compared with the goldens', so goldens of another checkpoint fail there, not as a low PCC.

Run:
    HF_MODEL=<checkpoint> M3_VISION_GOLDEN_DIR=<goldens> pytest models/demos/minimax_m3/tests/test_vision_accuracy.py
"""
from __future__ import annotations

import json
import os

import pytest
import torch
from loguru import logger
from safetensors.torch import load_file

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.tt.vision.loader import build_reference
from models.demos.minimax_m3.tt.vision.model import M3VLVisionModel

from .golden_hf_vision import IMAGES
from .test_factory import parametrize_mesh_with_fabric
from .unit.vision_helpers import to_host

FINAL_MIN_PCC = 0.998
TOWER_MARGIN = 0.002  # below Hugging Face's own bf16 tower PCC on the same image
# pre_layrnorm rows from the goldens' checkpoint match up to bf16 rounding; another checkpoint gives about 0.9.
CHECKPOINT_MIN_PCC = 0.9999

HF_MODEL = os.getenv("HF_MODEL")
GOLDEN_DIR = os.getenv("M3_VISION_GOLDEN_DIR") or (os.path.join(HF_MODEL, "golden", "vision") if HF_MODEL else None)


def _manifest() -> dict:
    with open(os.path.join(GOLDEN_DIR, "manifest.json")) as f:
        return json.load(f)


def _skip_reason() -> str:
    """Why the test cannot run on this host; empty when the checkpoint and the goldens are there."""
    has_goldens = bool(GOLDEN_DIR) and os.path.isfile(os.path.join(GOLDEN_DIR, "manifest.json"))
    if not (HF_MODEL and os.path.isdir(HF_MODEL)):
        checkpoint = _manifest()["checkpoint"] if has_goldens else "named in their manifest.json"
        return f"set HF_MODEL to the checkpoint the vision goldens were made from ({checkpoint})"
    if not has_goldens:
        return f"no vision goldens (manifest.json) in {GOLDEN_DIR}: set M3_VISION_GOLDEN_DIR"
    return ""


SKIP_REASON = _skip_reason()


def _load_goldens():
    """The manifest and, for each image of IMAGES, its tensors: pixel_values, image_grid_thw, tower_out, final and
    pre_layrnorm_rows."""
    manifest = _manifest()
    missing = [case for case in IMAGES if case not in manifest["cases"]]
    assert not missing, f"{GOLDEN_DIR}/manifest.json has no goldens for {missing}"
    return manifest, {case: load_file(os.path.join(GOLDEN_DIR, manifest["cases"][case]["file"])) for case in IMAGES}


def _check_checkpoint(reference, goldens, manifest):
    """Check that HF_MODEL is the goldens' checkpoint: on the CPU, the reference's patch embed and pre_layrnorm
    must reproduce the goldens' first pre_layrnorm rows."""
    for case, golden in goldens.items():
        rows = golden["pre_layrnorm_rows"].float()
        ours = reference.pre_layrnorm(reference.embed(golden["pixel_values"][: rows.shape[0]]))
        _, pcc = comp_pcc(rows, ours, CHECKPOINT_MIN_PCC)
        assert pcc >= CHECKPOINT_MIN_PCC, (
            f"HF_MODEL={HF_MODEL} is not the checkpoint of the goldens in {GOLDEN_DIR} ({manifest['checkpoint']}): "
            f"pre_layrnorm PCC {pcc:.4f} on {case}"
        )


def _compare(case, golden, entry, tower, final):
    """Log the PCC of the tower output and of the vision tokens of one image against the goldens, with the margin to
    the bar; return the checks that fail: a PCC below its bar, or any NaN or Inf in the output."""
    checks = (
        ("final", golden["final"], final, FINAL_MIN_PCC),
        ("tower", golden["tower_out"], tower, entry["hf_bf16_pcc"]["tower"] - TOWER_MARGIN),
    )
    failed = []
    for name, ref, out, min_pcc in checks:
        _, pcc = comp_pcc(ref, out, min_pcc)
        rel_l2 = (torch.linalg.norm(out - ref) / torch.linalg.norm(ref)).item()
        line = f"{case} {name}: PCC {pcc:.6f}, bar {min_pcc:.6f}, margin {pcc - min_pcc:+.6f}, relL2 {rel_l2:.4f}"
        nan, inf = int(out.isnan().sum()), int(out.isinf().sum())
        if nan or inf:
            line += f", {nan} NaN and {inf} Inf in {out.numel()} values"
        logger.info(line)
        if nan or inf or not pcc >= min_pcc:
            failed.append(line)
    return failed


@torch.no_grad()
@pytest.mark.skipif(bool(SKIP_REASON), reason=SKIP_REASON)
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_vision_accuracy(mesh_device, device_params):
    """Vision tokens at PCC >= 0.998 and the tower output within 0.002 of Hugging Face's bf16 PCC, with no NaN or
    Inf, on every image."""
    manifest, goldens = _load_goldens()
    cores = mesh_device.compute_with_storage_grid_size()
    logger.info(f"goldens {GOLDEN_DIR} ({manifest['checkpoint']}), HF_MODEL {HF_MODEL}, cores {cores.x}x{cores.y}")

    reference = build_reference(HF_MODEL)
    _check_checkpoint(reference, goldens, manifest)
    model = M3VLVisionModel.from_torch(mesh_device, reference, with_projector=True, dtype=ttnn.bfloat16)
    del reference

    failed = []
    for case, golden in goldens.items():
        tower_tt = model.forward_tower(golden["pixel_values"], golden["image_grid_thw"])
        tower = to_host(tower_tt, mesh_device, golden["tower_out"].shape)
        final_tt = model.projector(tower_tt)
        final = to_host(final_tt, mesh_device, golden["final"].shape)
        ttnn.deallocate(tower_tt)
        ttnn.deallocate(final_tt)
        failed += _compare(case, golden, manifest["cases"][case], tower, final)
    assert not failed, f"{len(failed)} checks failed:\n" + "\n".join(failed)
