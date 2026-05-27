# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN-only smoke test for the Mistral-Small-4 Pixtral vision tower.

Loads the vision tower weights, runs a dummy image through it, and checks the
output shape. No HF reference, no PCC — use this to iterate on the device
forward path without the slow CPU reference.

Run::

    export MISTRAL4_VISION_SMOKE=1
    export MISTRAL4_VISION_N_LAYERS=24       # optional; default 2
    export MISTRAL4_VISION_IMG_PATCHES=10    # patches per side (image = 10*14 = 140 px); default 10
    export MESH_DEVICE=T3K                   # T3K=1x8, P150x4=1x4, single=1x1
    pytest models/experimental/mistral_small_4_119b/tests/test_vision_tower_smoke.py -v -s --timeout=0

Tracy / tt-perf-report (per-op CSV comes from trace *capture*, not replay)::

    python -m tracy -r -m -p -v pytest \\
        models/experimental/mistral_small_4_119b/tests/test_vision_tower_smoke.py -v -s --timeout=0
    tt-perf-report generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv

``execute_trace`` does not emit per-op profiler rows. The signpost sits immediately
before ``capture_forward_trace`` so tt-perf-report sees the captured forward's ops.
Replay after the signpost is only for wall-clock timing.
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import (
    HF_MODEL_ID,
    VISION_HIDDEN_SIZE,
    VISION_PATCH_SIZE,
    vision_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral4_vision_tower import TtPixtralVisionTower
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

pytest.importorskip("transformers")


_N_LAYERS = int(os.environ.get("MISTRAL4_VISION_N_LAYERS", "2"))
_IMG_PATCHES = int(os.environ.get("MISTRAL4_VISION_IMG_PATCHES", "10"))  # 10×14 = 140 px


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


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.skipif(
    os.environ.get("MISTRAL4_VISION_SMOKE") != "1",
    reason="Set MISTRAL4_VISION_SMOKE=1 to run.",
)
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_mistral_small_4_vision_smoke(reset_seeds, mesh_device):
    """Build TtPixtralVisionTower and run one forward — shape check only."""
    try:
        state_dict = load_hf_state_dict_filtered(HF_MODEL_ID, _state_dict_prefixes(_N_LAYERS))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")

    img_size = _IMG_PATCHES * VISION_PATCH_SIZE
    logger.info(f"Building TtPixtralVisionTower ({_N_LAYERS} layers, image {img_size}×{img_size})…")
    model = TtPixtralVisionTower(
        mesh_device=mesh_device,
        state_dict=state_dict,
        num_layers=_N_LAYERS,
    )

    # Dummy image: random uniform in [-1, 1]. Real Pixtral expects normalised pixel values.
    image = torch.rand(1, 3, img_size, img_size, dtype=torch.bfloat16) * 2 - 1
    image_tt, h_patches, w_patches = model.patch_conv.upload_image(image)
    model.cache_rope_for_grid(h_patches, w_patches)

    expected_patches = _IMG_PATCHES * _IMG_PATCHES
    assert (
        h_patches == _IMG_PATCHES and w_patches == _IMG_PATCHES
    ), f"Patch grid {h_patches}×{w_patches} != expected {_IMG_PATCHES}×{_IMG_PATCHES}"

    # JIT compile pass (outside Tracy signpost).
    logger.info("Compile pass (forward_device warmup)…")
    t_compile = time.perf_counter()
    _ = model.forward_device(image_tt, h_patches, w_patches)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"Compile pass wall-clock: {(time.perf_counter() - t_compile) * 1e3:.2f} ms")

    # Signpost before capture: tt-perf-report filters to ops after the last signpost.
    # Trace capture runs a real forward (per-op rows in the CSV). execute_trace replay
    # does not — putting the signpost before replay would yield an empty report.
    signpost("Performance pass")
    logger.info("Capturing vision forward trace (Tracy measured pass)…")
    t_capture = time.perf_counter()
    features_tt = model.capture_forward_trace(image_tt, h_patches, w_patches)
    ttnn.synchronize_device(mesh_device)
    logger.info(
        f"Trace capture wall-clock: {(time.perf_counter() - t_capture) * 1e3:.2f} ms "
        "(per-op data for tt-perf-report)"
    )

    # Fast replay — no per-op profiler rows; wall-clock only.
    logger.info("Trace replay pass…")
    t_replay = time.perf_counter()
    features_tt = model.execute_forward_trace(blocking=False)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"Trace replay wall-clock: {(time.perf_counter() - t_replay) * 1e3:.2f} ms")

    # Pull device-0 slice to host and verify shape + finiteness.
    features = ttnn.to_torch(
        features_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[
        0
    ]  # [1, num_patches, 1024]

    assert features.shape == (
        1,
        expected_patches,
        VISION_HIDDEN_SIZE,
    ), f"Expected ({1}, {expected_patches}, {VISION_HIDDEN_SIZE}), got {tuple(features.shape)}"
    assert torch.isfinite(features.float()).all(), "Vision features contain NaN or Inf"

    logger.info(
        f"PASSED — vision tower produced features {tuple(features.shape)} "
        f"for {_N_LAYERS} layers, {img_size}×{img_size} image"
    )
