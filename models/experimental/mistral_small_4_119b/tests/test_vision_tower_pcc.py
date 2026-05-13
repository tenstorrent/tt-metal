# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pixtral vision tower PCC test — TTNN vs HF reference.

Runs the same image through HF's ``PixtralVisionModel`` (CPU, bf16) and our
``TtPixtralVisionTower`` (device, bf16), then compares the per-patch hidden
states.

PCC notes:
  - No MoE → numerical agreement should be tight. A healthy run gives ≥0.99
    overall and ≥0.99 at every patch.
  - bf16 mat-mul accumulators on device vs CPU still incur some rounding,
    so we don't expect bit-exact equality.

Run::

    export MISTRAL4_VISION_PCC=1
    export MISTRAL4_VISION_N_LAYERS=24       # optional; default 2
    export MISTRAL4_VISION_IMG_PATCHES=10    # patches per side; default 10
    export MESH_DEVICE=T3K                   # T3K=1x8, P150x4=1x4, single=1x1
    pytest models/experimental/mistral_small_4_119b/tests/test_vision_tower_pcc.py -v -s --timeout=0
"""

from __future__ import annotations

import copy
import gc
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import (
    HF_MODEL_ID,
    VISION_PATCH_SIZE,
    vision_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral4_vision_tower import TtPixtralVisionTower
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

pytest.importorskip("transformers")
pytest.importorskip("transformers.models.pixtral.modeling_pixtral", reason="Pixtral required")


_N_LAYERS = int(os.environ.get("MISTRAL4_VISION_N_LAYERS", "2"))
_IMG_PATCHES = int(os.environ.get("MISTRAL4_VISION_IMG_PATCHES", "10"))
_PCC_FLOOR = 0.98


def _state_dict_prefixes(n_layers: int) -> tuple:
    p = ["vision_tower.patch_conv.", "vision_tower.ln_pre."]
    for i in range(n_layers):
        p.append(vision_layer_state_dict_prefix(i))
    return tuple(p)


def _mesh_params():
    shape = mesh_device_request_param()
    base = {"trace_region_size": 30_000_000, "num_command_queues": 1}
    fabric = ttnn.FabricConfig.DISABLED if shape == (1, 1) else ttnn.FabricConfig.FABRIC_1D
    return [pytest.param(shape, {**base, "fabric_config": fabric}, id=f"mesh{shape[0]}x{shape[1]}")]


def _build_hf_vision_ref(vision_config, state_dict: dict, n_layers: int):
    """
    Build the HF ``PixtralVisionModel`` reference with truncated layer count.

    Uses ``accelerate.init_empty_weights`` so the model shell costs ~0 RAM, then
    streams weights from our filtered state dict (HF keys are already correct
    for vision; strip the ``vision_tower.`` prefix that's on every key).
    """
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel

    cfg = copy.deepcopy(vision_config)
    cfg.num_hidden_layers = n_layers

    with init_empty_weights():
        model = PixtralVisionModel(cfg)

    missing = []
    for name, _ in model.named_parameters():
        key = f"vision_tower.{name}"
        if key not in state_dict:
            missing.append(name)
            continue
        set_module_tensor_to_device(model, name, "cpu", value=state_dict[key].to(torch.bfloat16))

    if missing:
        logger.warning(f"HF vision missing keys (first 5): {missing[:5]}")
    return model.eval()


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.skipif(
    os.environ.get("MISTRAL4_VISION_PCC") != "1",
    reason="Set MISTRAL4_VISION_PCC=1 to run.",
)
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_mistral_small_4_vision_pcc(reset_seeds, mesh_device):
    """Compare TTNN vision tower features to HF PixtralVisionModel reference."""
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")

    vision_cfg = cfg.vision_config

    try:
        state_dict = load_hf_state_dict_filtered(HF_MODEL_ID, _state_dict_prefixes(_N_LAYERS))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")

    # ── Input image ──────────────────────────────────────────────────────
    img_size = _IMG_PATCHES * VISION_PATCH_SIZE
    image = torch.rand(1, 3, img_size, img_size, dtype=torch.bfloat16) * 2 - 1
    expected_patches = _IMG_PATCHES * _IMG_PATCHES
    logger.info(f"Image {img_size}×{img_size} → {expected_patches} patches, {_N_LAYERS} layers")

    # ── HF reference (CPU, bf16) ─────────────────────────────────────────
    logger.info("Building HF PixtralVisionModel reference (CPU, bf16)…")
    hf_model = _build_hf_vision_ref(vision_cfg, state_dict, _N_LAYERS)
    logger.info("Running HF reference forward…")
    hf_out = hf_model(image)
    ref_features = hf_out.last_hidden_state[0].float()  # [num_patches, 1024]
    del hf_model, hf_out
    gc.collect()
    logger.info(f"HF reference features: {tuple(ref_features.shape)}")

    # ── TTNN model ────────────────────────────────────────────────────────
    logger.info(f"Building TtPixtralVisionTower ({_N_LAYERS} layers)…")
    tt_model = TtPixtralVisionTower(
        mesh_device=mesh_device,
        state_dict=state_dict,
        num_layers=_N_LAYERS,
    )

    logger.info("Running TTNN vision tower forward…")
    features_tt, h_p, w_p = tt_model.forward(image)
    assert h_p == _IMG_PATCHES and w_p == _IMG_PATCHES

    # Pull device-0 slice to host: each device holds an identical replica.
    tt_features = ttnn.to_torch(
        features_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[
        0, 0
    ].float()  # [num_patches, 1024]
    ttnn.deallocate(features_tt)

    assert (
        tt_features.shape == ref_features.shape
    ), f"Shape mismatch: tt={tuple(tt_features.shape)} vs ref={tuple(ref_features.shape)}"

    # ── PCC per patch + overall ──────────────────────────────────────────
    pccs = []
    for i in range(expected_patches):
        _, msg = comp_pcc(ref_features[i], tt_features[i], _PCC_FLOOR)
        pcc_val = float(msg.split("=")[-1].strip() if "=" in str(msg) else msg)
        pccs.append(pcc_val)

    mean_pcc = sum(pccs) / len(pccs)
    min_pcc = min(pccs)
    logger.info(f"Per-patch PCC: mean={mean_pcc:.4f}, min={min_pcc:.4f}  (floor {_PCC_FLOOR})")

    passing, overall_msg = comp_pcc(ref_features.flatten(), tt_features.flatten(), _PCC_FLOOR)
    logger.info(f"Overall flattened PCC: {overall_msg}")
    assert passing, (
        f"Vision tower PCC below floor {_PCC_FLOOR}.\n"
        f"mean per-patch PCC={mean_pcc:.4f}, min per-patch PCC={min_pcc:.4f}\n"
        f"{overall_msg}"
    )
    logger.info(f"PASSED — TTNN vision features within PCC≥{_PCC_FLOOR} of HF reference")
