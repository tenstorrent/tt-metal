# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
import json
from pathlib import Path

import numpy as np
import pytest

ttnn = pytest.importorskip("ttnn")
pytest.importorskip("PIL")
pytest.importorskip("transformers")
from PIL import Image

from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.dpt_large.tt.config import DPTLargeConfig
from models.experimental.dpt_large.tt.perf_counters import reset_perf_counters, set_strict_program_config
from models.experimental.dpt_large.tt.pipeline import DPTTTPipeline


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
@pytest.mark.parametrize("execution_mode", ["eager", "trace", "trace_2cq"])
def test_tt_pipeline_perf_smoke(tmp_path, execution_mode):
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
        tt_approx_align_corners=True,
        tt_execution_mode=str(execution_mode),
    )

    # Perf tests should fail fast if a perf-tuned program_config is silently ignored.
    set_strict_program_config(True)
    reset_perf_counters()
    try:
        with DPTTTPipeline(config=cfg_tt, pretrained=False, device="cpu") as tt_pipe:
            pixel_values = tt_pipe.fallback._prepare(str(img_path))

            tt_input_host = None
            if str(execution_mode).lower() in {"trace", "trace_2cq"}:
                tt_input_host = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            def _run_once():
                if tt_input_host is not None:
                    return tt_pipe.forward_tt_host_tensor(tt_input_host, normalize=True)
                return tt_pipe.forward_pixel_values(pixel_values, normalize=True)

            # First run includes setup/compile/trace-capture overhead.
            t0 = time.perf_counter()
            _ = _run_once()
            first_run_s = time.perf_counter() - t0

            # Warmup and timing runs.
            _ = _run_once()
            times_s = []
            for _ in range(2):
                t1 = time.perf_counter()
                _ = _run_once()
                times_s.append(time.perf_counter() - t1)

            avg_s = float(np.mean(times_s))
            fps = (1.0 / avg_s) if avg_s > 0 else 0.0
            compile_s = max(first_run_s - avg_s, 0.0)

            assert tt_pipe.last_perf is not None
            assert tt_pipe.last_perf.get("mode") == "tt"
            counts = tt_pipe.last_perf.get("fallback_counts", {}) or {}
            for key in (
                "vit_backbone_fallback_count",
                "reassembly_readout_fallback_count",
                "upsample_host_fallback_count",
            ):
                assert int(counts.get(key, 0)) == 0, f"Unexpected host fallback in perf run: {counts}"
            assert int(counts.get("program_config_fallback_total", 0)) == 0, (
                "Unexpected TT program_config fallbacks in perf run: " f"{counts}"
            )
            if str(execution_mode).lower() in {"trace", "trace_2cq"}:
                assert tt_pipe.last_perf.get("full_trace_unavailable_reason") is None
                assert tt_pipe.last_perf.get("requested_execution_mode") == str(execution_mode).lower()
                assert tt_pipe.last_perf.get("trace_wall_ms") is not None
            else:
                for key in ("backbone_ms", "reassembly_ms", "fusion_head_ms", "total_ms"):
                    assert key in tt_pipe.last_perf
            assert fps > 0
    finally:
        set_strict_program_config(False)

    min_fps = float(os.environ.get("DPT_LARGE_MIN_FPS", "0"))
    if min_fps > 0:
        assert fps >= min_fps, f"DPT-Large throughput below target: {fps:.3f} FPS < {min_fps:.3f} FPS"

    # Persist a lightweight perf report artifact for manual/CI inspection.
    report = {
        "model_name": "dpt_large_full_pipeline",
        "execution_mode": str(execution_mode).lower(),
        "batch_size": 1,
        "image_size": 384,
        "first_run_s": first_run_s,
        "avg_inference_s": avg_s,
        "compile_s": compile_s,
        "fps": fps,
    }
    out_path = Path(f"generated/dpt_large_tt_perf_smoke_{str(execution_mode).lower()}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
