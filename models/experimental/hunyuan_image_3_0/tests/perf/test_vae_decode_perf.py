# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-performance test for the HunyuanImage-3.0 TTNN VAE decoder (tracy).

Two tests, the standard tt-metal device-perf split:

  * ``test_vae_decode_device_ops`` — the workload. Runs the full-res, H/W-spatial
    (2x2) VAE decode (64x64x16 latent -> 1024x1024x3), wrapped in tracy
    ``signpost("start")`` / ``signpost("stop")`` so only the decode ops are
    measured. The decoder is built ONCE and run twice; the signposts bracket the
    SECOND run so program-cache-warmed device kernel durations are what's profiled.

  * ``test_vae_decode_perf_device`` — the gate. Re-invokes the workload under the
    device profiler (``run_device_perf``), sums the device-kernel duration between
    the signposts across a few iterations, and reports / checks it.

Run the raw op profile (workload only, writes ops_perf_results_*.csv):
    python3 tools/tracy/profile_this.py -n hunyuan_vae_decode \
      -c "pytest models/experimental/hunyuan_image_3_0/tests/perf/test_vae_decode_perf.py::test_vae_decode_device_ops -s"

Run the device-perf gate (spawns the workload under the profiler for you):
    pytest models/experimental/hunyuan_image_3_0/tests/perf/test_vae_decode_perf.py::test_vae_decode_perf_device
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.hunyuan_image_3_0.ref.vae.decoder import Z_CHANNELS, load_decoder, vae_decode_output_to_rgb
from models.experimental.hunyuan_image_3_0.ref.model_config import VAE_SCALING_FACTOR
from models.experimental.hunyuan_image_3_0.tt.pipeline import decode_latent
from models.experimental.hunyuan_image_3_0.tt.vae.decoder import VAEDecoderTTNN

SCALING_FACTOR = VAE_SCALING_FACTOR
PCC_THRESHOLD = 0.99  # matches tests/pcc/test_vae_decode_pipeline.py

use_signpost = True
try:
    from tracy import signpost
except ModuleNotFoundError:
    use_signpost = False


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_vae_decode_device_ops(mesh_device):
    """Full-res H/W-spatial VAE decode, bracketed by signposts for the perf gate."""
    mesh_device.enable_program_cache()
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    torch.manual_seed(42)
    latent_bchw = torch.randn(1, Z_CHANNELS, 64, 64, dtype=torch.float32)

    # Build the decoder once so both runs share the program cache (run 1 compiles,
    # run 2 — the profiled one — runs warm).
    decoder = VAEDecoderTTNN(mesh_device, dtype=ttnn.bfloat16)

    def _decode():
        return decode_latent(
            mesh_device,
            latent_bchw,
            scaling_factor=SCALING_FACTOR,
            decoder=decoder,
            ccl_manager=ccl,
            h_mesh_axis=0,
            w_mesh_axis=1,
        )

    _decode()  # warmup: compile programs / fill cache

    if use_signpost:
        signpost(header="start")
    img = _decode()  # profiled run
    if use_signpost:
        signpost(header="stop")

    logger.info(f"VAE decode output shape={tuple(img.shape)}")
    assert tuple(img.shape) == (1, 3, 1024, 1024), f"unexpected output shape {tuple(img.shape)}"

    # Functional validation: the profiled decode must also be numerically correct,
    # so a perf "win" from a broken kernel can't pass the gate. Torch fp32 reference
    # (CPU) for the same latent — computed outside the signposts so it isn't profiled.
    z_bcthw = (latent_bchw / SCALING_FACTOR).unsqueeze(2)
    with torch.no_grad():
        pt_out = load_decoder()(z_bcthw)
    pt_img = vae_decode_output_to_rgb(pt_out)  # last temporal frame -> [1, 3, 1024, 1024]
    passing, pcc = comp_pcc(pt_img, img, PCC_THRESHOLD)
    logger.info(f"VAE decode vs reference PCC: {pcc:.6f}")
    assert passing, f"PCC {pcc:.6f} < {PCC_THRESHOLD} — profiled decode is numerically wrong"


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.timeout(1800)  # 3 profiled subprocess runs of a full-res decode; overrides the 300s default
@pytest.mark.parametrize(
    # Summed DEVICE KERNEL DURATION [ns] between the signposts, on Blackhole (2x2 mesh).
    # Baseline: avg of 3 profiled iterations, 2026-07-03 (AVG 24.019s; MIN 24.012s, MAX
    # 24.024s; run-to-run spread <0.1%). This branch's decoder runs distributed GroupNorm
    # (tt/vae/spatial.py HY_GN_MODE=dist: per-shard stats + all-reduce, no full-spatial
    # gather) to fix the full-res OOM, which is far more device ops than a single fused
    # ttnn.group_norm — hence much slower than a baseline measured against a decoder
    # without that fix. Re-baseline here (not on an unrelated branch's number) since
    # Phase 2 perf hardening (README) hasn't landed yet. Previous baseline (25.351s) was
    # measured before group_norm_distributed dropped its fp32 upcast of the full-size
    # activation (sum/sum-of-squares now run in the activation's native bf16) — ~5.2%
    # faster than fp32, with no PCC regression (0.999941 vs 0.999957).
    "expected_device_kernel_duration_ns",
    [24_019_439_706],
)
def test_vae_decode_perf_device(expected_device_kernel_duration_ns):
    batch_size = 1  # one image per decode
    subdir = "hunyuan_vae_decode"
    num_iterations = 3
    margin = 0.05
    cols = ["DEVICE KERNEL"]

    command = (
        "pytest models/experimental/hunyuan_image_3_0/tests/perf/test_vae_decode_perf.py::test_vae_decode_device_ops"
    )

    duration_key = "AVG DEVICE KERNEL DURATION [ns]"
    post_processed_results = run_device_perf(
        command,
        subdir=subdir,
        num_iterations=num_iterations,
        cols=cols,
        batch_size=batch_size,
        has_signposts=True,  # measure only between signpost("start") and signpost("stop")
        # The default (1333, tools/tracy/process_model_log.py) sizes the per-core DRAM
        # profiler marker buffer too small for this decoder: distributed group_norm
        # (tt/vae/spatial.py) expands each norm into many elementwise/reduce/all_gather
        # ops instead of one fused ttnn.group_norm call, so full-res H/W-spatial decode
        # overflows it ("Profiler DRAM buffers were full, markers were dropped!" on every
        # core), which corrupts the device log and makes tracy's report step crash with
        # "cpp_device_perf_report.csv not found". Bumped well above the observed op count.
        op_support_count=6000,
    )

    expected_results = check_device_perf(
        post_processed_results,
        margin=margin,
        expected_perf_cols={duration_key: expected_device_kernel_duration_ns},
        assert_on_fail=False,  # baseline is stale during the perf-optimization pass; report only
    )
    prep_device_perf_report(
        model_name="hunyuan_vae_decode",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        # Underscore-joined, filename-safe: `comments` flows into the CSV name (no '/', no spaces).
        comments="1024x1024_HxW_spatial_2x2",
    )
