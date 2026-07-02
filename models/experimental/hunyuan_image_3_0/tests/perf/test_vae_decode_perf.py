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
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.hunyuan_image_3_0.ref.vae.decoder import Z_CHANNELS
from models.experimental.hunyuan_image_3_0.tt.pipeline import decode_latent
from models.experimental.hunyuan_image_3_0.tt.vae.decoder import VAEDecoderTTNN

SCALING_FACTOR = 0.562679178327931  # config.json vae.scaling_factor

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


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.timeout(1800)  # 3 profiled subprocess runs of a full-res decode; overrides the 300s default
@pytest.mark.parametrize(
    # Placeholder budget — update after the first profiling run on target hardware.
    # This is the summed DEVICE KERNEL DURATION [ns] between the signposts.
    "expected_device_kernel_duration_ns",
    [750_000_000],
)
def test_vae_decode_perf_device(expected_device_kernel_duration_ns):
    batch_size = 1  # one image per decode
    subdir = "hunyuan_vae_decode"
    num_iterations = 3
    margin = 0.10
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
    )

    expected_results = check_device_perf(
        post_processed_results,
        margin=margin,
        expected_perf_cols={duration_key: expected_device_kernel_duration_ns},
        assert_on_fail=False,  # report-only until a measured baseline is set
    )
    prep_device_perf_report(
        model_name="hunyuan_vae_decode",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="1024x1024 H/W-spatial 2x2 decode",
    )
