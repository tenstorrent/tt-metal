# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-performance test for the HunyuanImage-3.0 TTNN SigLIP2 vision tower (tracy).

Two tests, the standard tt-metal device-perf split:

  * ``test_siglip2_perf_device_ops`` — the workload. Runs the full 27-layer vision
    tower at the processor's max_num_patches (1024, 32x32), on the replicated 2x2
    mesh (production I2I conditioning path — see tt/vision/i2i_bundle.py
    ``load_tt_vision_stack``), wrapped in tracy ``signpost("start")`` /
    ``signpost("stop")`` so only the vision-tower ops are measured. The tower is
    built ONCE and run twice; the signposts bracket the SECOND run so
    program-cache-warmed device kernel durations are what's profiled.

  * ``test_siglip2_perf_device`` — the gate. Re-invokes the workload under the
    device profiler (``run_device_perf``), sums the device-kernel duration between
    the signposts across a few iterations, and reports / checks it.

Run the raw op profile (workload only, writes ops_perf_results_*.csv):
    python3 tools/tracy/profile_this.py -n hunyuan_siglip2_vision \
      -c "pytest models/experimental/hunyuan_image_3_0/tests/perf/test_siglip2_perf_tracy.py::test_siglip2_perf_device_ops -s"

Run the device-perf gate (spawns the workload under the profiler for you):
    pytest models/experimental/hunyuan_image_3_0/tests/perf/test_siglip2_perf_tracy.py::test_siglip2_perf_device

Requires a real checkpoint (HUNYUAN_MODEL_DIR or the default HF cache path) — skips
otherwise, since weight-dependent kernel selection can shift device time vs random
weights.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.experimental.hunyuan_image_3_0.ref.vision.siglip2 import load_siglip2_vision
from models.experimental.hunyuan_image_3_0.ref.weights import MODEL_DIR
from models.experimental.hunyuan_image_3_0.tt.vision.siglip2 import (
    HunyuanTtSiglip2Vision,
    Siglip2VisionInputs,
    VIT_CONFIG,
)

FULL_LAYERS = VIT_CONFIG["num_hidden_layers"]  # 27
FULL_S = 1024  # max_num_patches (32x32)
FULL_HW = (32, 32)
PCC_THRESHOLD = 0.99

use_signpost = True
try:
    from tracy import signpost
except ModuleNotFoundError:
    use_signpost = False


def _require_checkpoint():
    index = MODEL_DIR / "model.safetensors.index.json"
    if not index.exists():
        pytest.skip(f"Hunyuan checkpoint not found at {MODEL_DIR} (set HUNYUAN_MODEL_DIR)")


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_siglip2_perf_device_ops(mesh_device):
    """Full 27L SigLIP2 vision tower @ S=1024, replicated 2x2 (production I2I mesh),
    bracketed by signposts for the perf gate."""
    _require_checkpoint()
    mesh_device.enable_program_cache()

    patch_dim = VIT_CONFIG["num_channels"] * VIT_CONFIG["patch_size"] ** 2
    torch.manual_seed(0)
    pixel_values = torch.randn(1, FULL_S, patch_dim, dtype=torch.float32)
    pixel_attention_mask = torch.ones(1, FULL_S, dtype=torch.long)
    pixel_attention_mask[0, FULL_S - 32 :] = 0

    ref_vision = load_siglip2_vision(MODEL_DIR, num_layers=FULL_LAYERS)
    sd = ref_vision.state_dict()

    # Build once so both runs share the program cache (run 1 compiles, run 2 —
    # the profiled one — runs warm).
    vision = HunyuanTtSiglip2Vision(mesh_device, sd, num_layers=FULL_LAYERS, weight_dtype=ttnn.bfloat16)
    vision.prewarm_pos_geometries([(FULL_HW[0], FULL_HW[1], FULL_S)])

    pv = ttnn.from_torch(
        pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    mask = ttnn.from_torch(
        pixel_attention_mask.to(torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    inputs = Siglip2VisionInputs.create(pv, (FULL_HW,), mask)

    def _forward():
        return vision(inputs)

    _forward()  # warmup: compile programs / fill cache

    if use_signpost:
        signpost(header="start")
    out = _forward()  # profiled run
    if use_signpost:
        signpost(header="stop")

    logger.info(f"SigLIP2 vision tower output shape={tuple(out.shape)}")
    assert tuple(out.shape) == (1, FULL_S, VIT_CONFIG["hidden_size"]), f"unexpected output shape {tuple(out.shape)}"

    # Functional validation: the profiled forward must also be numerically correct,
    # so a perf "win" from a broken kernel can't pass the gate. Torch fp32 reference
    # (CPU) for the same input — computed outside the signposts so it isn't profiled.
    with torch.no_grad():
        pt_out = ref_vision(pixel_values, pixel_attention_mask, torch.tensor([list(FULL_HW)], dtype=torch.long))
    tt_out = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].float()
    passing, pcc = comp_pcc(pt_out, tt_out, PCC_THRESHOLD)
    logger.info(f"SigLIP2 vision tower vs reference PCC: {pcc:.6f}")
    assert passing, f"PCC {pcc:.6f} < {PCC_THRESHOLD} — profiled forward is numerically wrong"


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.timeout(1800)
def test_siglip2_perf_device():
    """Device-perf gate: sums DEVICE KERNEL DURATION [ns] between the signposts,
    averaged over a few profiled iterations. No hard-coded expected value yet —
    first run establishes the baseline (see test_vae_decode_perf.py for the
    baseline-update convention once one is recorded)."""
    _require_checkpoint()
    batch_size = 1  # one conditioning image per encode
    subdir = "hunyuan_siglip2_vision"
    num_iterations = 3
    cols = ["DEVICE KERNEL"]

    command = (
        "pytest models/experimental/hunyuan_image_3_0/tests/perf/test_siglip2_perf_tracy.py"
        "::test_siglip2_perf_device_ops"
    )

    duration_key = "AVG DEVICE KERNEL DURATION [ns]"
    post_processed_results = run_device_perf(
        command,
        subdir=subdir,
        num_iterations=num_iterations,
        cols=cols,
        batch_size=batch_size,
        has_signposts=True,
    )
    logger.info(f"SigLIP2 vision tower device kernel duration: {post_processed_results[duration_key] / 1e6:.3f} ms")
    # No baseline expected value yet — margin=1.0 so check_device_perf never fails the
    # first run; it just records post_processed_results for prep_device_perf_report.
    expected_results = check_device_perf(
        post_processed_results,
        margin=1.0,
        expected_perf_cols={duration_key: post_processed_results[duration_key]},
        assert_on_fail=False,
    )
    prep_device_perf_report(
        model_name=subdir,
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="27L_S1024_replicated_2x2",
    )
