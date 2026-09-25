# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-free smoke PCC and one-block profile for the vision tower.
Config-init weights miss outlier quantization; test_wrapped_model.py is the real-weight gate."""

from __future__ import annotations

import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc, run_for_wormhole_b0_or_blackhole
from models.demos.blackhole.qwen36.tt.vision.model import DropInVisionTransformer
from models.demos.blackhole.qwen36.tt.vision.vision_model_config import VisionModelArgs
from models.tt_transformers.tt.ccl import TT_CCL

# (grid, depth, pcc_required, profile). depth=None is the config's full depth.
CASES = [
    ((1, 86, 128), 1, 0.999, True),
    ((1, 86, 128), None, 0.998, False),
]

WEIGHT_DTYPE = ttnn.bfloat8_b


def _mesh_device_param() -> tuple[int, int]:
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    explicit = {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}
    if name in explicit:
        return explicit[name]
    return (1, max(1, min(ttnn.get_num_devices(), 2)))


MESH_SHAPE = _mesh_device_param()
_MULTI = MESH_SHAPE != (1, 1)
DEVICE_PARAMS = [{"l1_small_size": 24576, **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {})}]


@torch.no_grad()
@pytest.mark.timeout(3600)
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "grid, depth, pcc_required, profile",
    CASES,
    # Selector strings avoid pcc and vision_tower so -k cannot match both cases via the module name.
    ids=[
        f"{'oneblock' if prof else 'fulldepth'}_patches{math.prod(g)}_depth{d or 'full'}"
        for g, d, _, prof in CASES  # noqa: B023
    ],
)
def test_vision_tower_pcc(mesh_device, device_params, grid, depth, pcc_required, profile, tmp_path, reset_seeds):
    del device_params
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    mesh_device.enable_program_cache()
    n_patches = math.prod(grid)
    seq_len = ((n_patches // 2048) + 1) * 2048

    model_args = VisionModelArgs(mesh_device, dummy_weights=False, max_batch_size=1, max_seq_len=seq_len)
    vcfg = model_args.hf_config.vision_config
    if depth is not None:
        vcfg.depth = depth
    merge = vcfg.spatial_merge_size
    assert grid[1] % merge == 0 and grid[2] % merge == 0, f"grid h,w must divide by {merge}"

    torch.manual_seed(0)
    reference_model = Qwen3_5VisionModel(vcfg).eval()
    tt_model = DropInVisionTransformer(
        reference_model,
        model_args,
        dtype=WEIGHT_DTYPE,
        debug=False,
        tt_ccl=TT_CCL(mesh_device),
        # Random weights must never land in the production ttnn cache (keyed by name only).
        weight_cache_path=tmp_path / "vision_pcc_weights",
    )

    pixel_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size**2
    grid_thw = torch.tensor([grid], dtype=torch.long)
    pixel_values = torch.randn(n_patches, pixel_dim)

    # The tower pads rows to a multiple of 128; report that length, not the buffer bound.
    tower_rows = -(-n_patches // 128) * 128
    logger.info(
        f"{'PROFILE' if profile else 'PCC'} case: depth={vcfg.depth}, grid={grid} "
        f"({n_patches} patches -> {tower_rows} rows, max_seq_len {seq_len})"
    )
    reference_output = reference_model(pixel_values, grid_thw).pooler_output

    signpost = None
    if profile:
        # Warm up outside the signposts so compile and first-touch allocation are not in the window.
        ttnn.deallocate(tt_model(pixel_values, grid_thw))
        read_profiler = getattr(ttnn, "ReadDeviceProfiler", None)
        if read_profiler is not None:
            read_profiler(mesh_device)
        try:
            from tracy import signpost
        except ImportError:
            logger.info("tracy.signpost unavailable; running without signpost markers.")

    if signpost is not None:
        signpost("start")
    tt_output = tt_model(pixel_values, grid_thw)
    ttnn.synchronize_device(mesh_device)
    if signpost is not None:
        signpost("stop")
        read_profiler = getattr(ttnn, "ReadDeviceProfiler", None)
        if read_profiler is not None:
            read_profiler(mesh_device)

    # The merger output is fractured along dim=3 (out_hidden_size/TP per device).
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    tt_output_torch = tt_output_torch.squeeze(0).squeeze(0)[:, : vcfg.out_hidden_size]
    ttnn.deallocate(tt_output)

    assert tt_output_torch.shape == reference_output.shape, f"{tt_output_torch.shape} != {reference_output.shape}"
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"vision tower depth={vcfg.depth} grid={grid}: PCC {pcc_message} (required {pcc_required})")
    assert passing, f"vision tower PCC below {pcc_required}: {pcc_message}"
