# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import time

import numpy as np
import pytest

ttnn = pytest.importorskip("ttnn")
pytest.importorskip("PIL")
pytest.importorskip("transformers")
from PIL import Image

from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.dpt_large.tt.config import DPTLargeConfig
from models.experimental.dpt_large.tt.pipeline import DPTTTPipeline
from models.perf.perf_utils import prep_perf_report


def _make_dummy_image(path, size=384):
    rng = np.random.default_rng(seed=1)
    arr = rng.integers(low=0, high=256, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _runtime_device_name() -> str:
    try:
        if hasattr(ttnn, "GetNumAvailableDevices"):
            ndev = int(ttnn.GetNumAvailableDevices())
        else:
            ndev = int(ttnn.get_num_devices())
        return "wormhole_n300" if ndev >= 2 else "wormhole_n150"
    except Exception:
        return "wormhole_n300"


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
def test_tt_pipeline_perf_smoke(tmp_path):
    img_path = tmp_path / "perf_input.png"
    _make_dummy_image(img_path, size=384)

    cfg_tt = DPTLargeConfig(
        image_size=384,
        device=_runtime_device_name(),
        allow_cpu_fallback=False,
        enable_tt_device=True,
        tt_device_reassembly=True,
        tt_device_fusion=True,
        tt_perf_encoder=True,
        tt_perf_neck=True,
    )

    with DPTTTPipeline(config=cfg_tt, pretrained=False, device="cpu") as tt_pipe:
        # First run includes setup/compile overhead.
        t0 = time.perf_counter()
        _ = tt_pipe.forward(str(img_path), normalize=True)
        first_run_s = time.perf_counter() - t0

        # Warmup and timing runs.
        _ = tt_pipe.forward(str(img_path), normalize=True)
        times_s = []
        for _ in range(2):
            t1 = time.perf_counter()
            _ = tt_pipe.forward(str(img_path), normalize=True)
            times_s.append(time.perf_counter() - t1)

        avg_s = float(np.mean(times_s))
        fps = (1.0 / avg_s) if avg_s > 0 else 0.0
        compile_s = max(first_run_s - avg_s, 0.0)

        assert tt_pipe.last_perf is not None
        assert tt_pipe.last_perf.get("mode") == "tt"
        for key in ("backbone_ms", "reassembly_ms", "fusion_head_ms", "total_ms"):
            assert key in tt_pipe.last_perf
        assert fps > 0

    min_fps = float(os.environ.get("DPT_LARGE_MIN_FPS", "0"))
    if min_fps > 0:
        assert fps >= min_fps, f"DPT-Large throughput below target: {fps:.3f} FPS < {min_fps:.3f} FPS"

    # Perf report hook aligned with tt-metal perf utilities.
    prep_perf_report(
        model_name="dpt_large_full_pipeline",
        batch_size=1,
        inference_and_compile_time=first_run_s,
        inference_time=avg_s,
        expected_compile_time=compile_s,
        expected_inference_time=avg_s,
        comments="tt_384_smoke",
    )
