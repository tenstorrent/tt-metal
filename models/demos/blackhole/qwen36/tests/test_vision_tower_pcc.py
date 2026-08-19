# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-free PCC gate for the Qwen3.5 VISION TOWER.

``test_wrapped_model.py`` is the tower's PCC test, but it needs ``dummy_weights=True``, which routes
the CONFIG through ``ModelArgs.LOCAL_HF_PARAMS`` -- and that table has no ``Qwen3.5-9B`` entry, so it
raises ``KeyError: 'Qwen3.5-9B'`` before it reaches the device. That leaves the 9B tower with no
runnable numerical gate.

This test closes that gap by building the HF reference from ``vision_config`` alone
(``Qwen3_5VisionModel(vcfg)``, random weights) and comparing the TT tower against it. Weight *values* do not matter to PCC -- only that both sides use the same ones.

THE SHAPE
---------
``(1, 86, 128)`` / 11008 patches -> 12288 padded: the image ``demo/benchmark_vision.py`` and
``demo/vision_demo.py`` default to, and therefore the shape whose matmul plans
(``VisionModelArgs.vision_mm_plan``) the sweep tuned. Depth is capped at 2 because the torch
reference costs ~0.8 TFLOP per block on the host for this grid.

ONE RUN GIVES BOTH PCC AND THE DEVICE-PERF REPORT
-------------------------------------------------
The measured forward is wrapped in ``start``/``stop`` signposts, with an untimed warmup forward
before it and the profiler drained in between. So under Tracy this test yields a device-op report for
exactly the code whose PCC it just asserted, instead of profiling and PCC-gating in two separate runs
that could disagree::

    python -m tracy -p -v -r --dump-device-data-mid-run -m \\
        pytest "models/demos/blackhole/qwen36/tests/test_vision_tower_pcc.py::test_vision_tower_pcc[wormhole_b0-patches11008_depth2-mesh_device0-device_params0]"
    tt-perf-report --start-signpost start --end-signpost stop <ops_perf_results_*.csv>

Note the report then covers ``depth`` blocks, not one, so a per-block figure is the block ops divided
by ``depth`` (or just read the second instance of each op, which is the cache-warm one).

Run::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B pytest \\
        models/demos/blackhole/qwen36/tests/test_vision_tower_pcc.py -v -s

For the 27B tower, point ``HF_MODEL`` at the LOCAL config dir -- ``ModelArgs`` takes
``CKPT_DIR = HF_MODEL`` and ``model_name`` from its basename, so no checkpoint or hub fetch is
needed (this tower's reference weights are config-init either way)::

    MESH_DEVICE=T3K HF_MODEL=$PWD/models/tt_transformers/model_params/Qwen3.6-27B pytest \\
        models/demos/blackhole/qwen36/tests/test_vision_tower_pcc.py -v -s
"""

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

# (grid, depth, pcc). depth=None means the config's full depth.
# The threshold is set from measured numbers: 0.99853 before this tower's matmuls were tuned,
# 0.99857 after.
CASES = [
    ((1, 86, 128), 2, 0.99),
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
    "grid, depth, pcc_required",
    CASES,
    ids=[f"patches{math.prod(g)}_depth{d or 'full'}" for g, d, _ in CASES],
)
def test_vision_tower_pcc(mesh_device, device_params, grid, depth, pcc_required, tmp_path, reset_seeds):
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

    logger.info(f"reference tower: depth={vcfg.depth}, grid={grid} ({n_patches} patches -> {seq_len} padded)")
    reference_output = reference_model(pixel_values, grid_thw).pooler_output

    # Warmup outside the signposts so kernel compilation and first-touch allocation are not measured,
    # then drain the on-device profiler buffer so those markers neither accumulate nor land in the
    # report's start..stop window. Both are no-ops without a profiler build.
    ttnn.deallocate(tt_model(pixel_values, grid_thw))
    read_profiler = getattr(ttnn, "ReadDeviceProfiler", None)
    if read_profiler is not None:
        read_profiler(mesh_device)

    signpost = None
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
