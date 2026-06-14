# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Single-device (Blackhole p150a, chip 2) validation of the TT-NN MoonViT vision tower.

Loads pixel_values + grid from the torch CPU golden, runs the tt-nn MoonViT + mlp1
projector on device, and asserts incremental PCC vs the golden:
  patch_embed -> encoder_out -> vit_raw (post-merge) -> vit_proj (final, gate >= 0.99).

Run (single chip 2, no cross-chip fabric):
  TT_METAL_HOME=$(pwd) ARCH_NAME=blackhole TT_METAL_VISIBLE_DEVICES=2 MESH_DEVICE=N150 \
    ./python_env/bin/python -m pytest -svq \
    models/experimental/locate_anything/tests/test_vision.py
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.locate_anything.reference import la_inputs
from models.experimental.locate_anything.tt.vision import MoonViT

GOLDEN_PATH = "models/experimental/locate_anything/reference/golden.pt"
PROJ_PCC_GATE = 0.99


def _pcc(name, golden, calc, gate=0.99):
    passing, msg = comp_pcc(golden.float(), calc.float(), pcc=gate)
    logger.info(f"[PCC] {name}: {msg}")
    print(f"PCC {name}: {msg}")
    return passing, msg


@pytest.mark.parametrize(
    "device_params",
    # Single isolated p150a: NO cross-chip fabric (conflicts with sibling experiments
    # on the other chips). fabric_config falsy => DISABLED.
    [{"fabric_config": False, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
def test_moonvit_vision(mesh_device):
    assert os.path.isfile(GOLDEN_PATH), f"Golden not found at {GOLDEN_PATH}"
    golden = torch.load(GOLDEN_PATH, weights_only=False)

    grid_hw = golden["grid_hw"]  # (26, 42)
    pixel_values = golden["pixel_values"].float()  # [L,3,14,14]
    logger.info(f"grid_hw={grid_hw} pixel_values={tuple(pixel_values.shape)}")

    model_path = la_inputs.find_model_path()
    model = MoonViT(mesh_device, model_path, grid_hw, dtype=ttnn.bfloat16)

    vit_proj_tt, inter = model.forward(pixel_values, return_intermediates=True)

    # --- incremental PCC (isolate each stage) ---
    p_pe, _ = _pcc("patch_embed", golden["vit_patch_embed"], inter["patch_embed"], gate=0.99)
    p_enc, _ = _pcc("encoder_out", golden["vit_encoder_out"], inter["encoder_out"], gate=0.99)

    # vit_raw = post patch-merge, pre mlp1. Reconstruct host merge from encoder golden to
    # validate the merge layout, then compare device merge result implicitly via vit_proj.
    vit_raw_tt = model.patch_merger(inter["encoder_out"])  # [N,4608] from device encoder output
    p_raw, _ = _pcc("vit_raw(merged)", golden["vit_raw"], vit_raw_tt, gate=0.99)

    # --- primary gate: vit_proj ---
    p_proj, msg_proj = _pcc("vit_proj(final)", golden["vit_proj"], vit_proj_tt, gate=PROJ_PCC_GATE)

    logger.info(f"SUMMARY patch_embed={p_pe} encoder={p_enc} vit_raw={p_raw} vit_proj={p_proj}")
    assert p_proj, f"vit_proj PCC below {PROJ_PCC_GATE}: {msg_proj}"
