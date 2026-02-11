# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")
pytest.importorskip("PIL")
pytest.importorskip("transformers")
from PIL import Image

from models.common.utility_functions import comp_pcc, run_for_wormhole_b0
from models.experimental.dpt_large.tt.config import DPTLargeConfig
from models.experimental.dpt_large.tt.fallback import DPTFallbackPipeline
from models.experimental.dpt_large.tt.pipeline import DPTTTPipeline


def _make_dummy_image(path, size=384):
    # Deterministic input so PCC results are stable across runs.
    rng = np.random.default_rng(seed=0)
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
def test_tt_pipeline_pcc_vs_cpu_reference(tmp_path):
    img_path = tmp_path / "pcc_input.png"
    _make_dummy_image(img_path, size=384)

    cfg_cpu = DPTLargeConfig(
        image_size=384,
        device="cpu",
        allow_cpu_fallback=True,
        enable_tt_device=False,
        tt_device_reassembly=False,
        tt_device_fusion=False,
        tt_perf_encoder=False,
        tt_perf_neck=False,
    )
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

    cpu = DPTFallbackPipeline(config=cfg_cpu, pretrained=False, device="cpu")
    with DPTTTPipeline(config=cfg_tt, pretrained=False, device="cpu") as tt_pipe:
        depth_cpu = cpu.forward(str(img_path), normalize=True)
        depth_tt = tt_pipe.forward(str(img_path), normalize=True)

        assert tt_pipe.last_perf is not None
        assert tt_pipe.last_perf.get("mode") == "tt"
        assert tt_pipe.backbone.used_tt_encoder_last_forward

    assert np.isfinite(depth_cpu).all()
    assert np.isfinite(depth_tt).all()

    passing, pcc = comp_pcc(depth_cpu, depth_tt, pcc=0.99)
    assert passing, f"DPT-Large TT full-pipeline PCC below target: got {pcc}"
