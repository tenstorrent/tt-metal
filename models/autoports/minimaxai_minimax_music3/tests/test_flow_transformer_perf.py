# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tracy-signposted performance evidence for ``FlowTransformer``: ONE warmed B=2 forward at the golden T.

Pattern (functional-decoder skill): warm, synchronize, signpost start, run the measured window, synchronize,
signpost end. Marked ``slow`` (excluded from the stage gate); driven by ``scripts/collect_dit_perf.sh`` under
``python -m tracy``. Without Tracy the test still records wall times in ``doc/flow_dit/perf/*.json``.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R
from models.autoports.minimaxai_minimax_music3.tt.flow_transformer import BATCH, FlowTransformer

pytestmark = [pytest.mark.hardware, pytest.mark.slow, pytest.mark.timeout(1800)]
DOC_DIR = Path(__file__).resolve().parents[1] / "doc" / "flow_dit"


def _profiled() -> bool:
    return bool(os.environ.get("TT_METAL_DEVICE_PROFILER"))


def _signpost(name: str) -> None:
    try:
        from tracy import signpost
    except ImportError:
        return
    signpost(name)


def test_dit_forward_perf(mm3_mesh_device):
    root = R.reference_dir()
    if not (root / "chunks.pt").is_file():
        pytest.skip("golden reference missing")
    g = torch.load(root / "chunks.pt")
    noise, cond = g["noises"][0], g["conditions"][0]
    latents = noise.expand(BATCH, -1, -1).contiguous()
    timestep = torch.full((BATCH,), 0.5)
    model = FlowTransformer.from_pretrained(mm3_mesh_device)
    cond_proj = model.prepare_condition(torch.cat([cond, torch.zeros_like(cond)], 0))
    for _ in range(2):  # compile + warm
        model(latents, timestep, cond_proj=cond_proj)
    ttnn.synchronize_device(mm3_mesh_device)
    if _profiled():
        ttnn.ReadDeviceProfiler(mm3_mesh_device)  # drain the warm-up ops
    _signpost("PERF_DIT_FORWARD")
    t0 = time.time()
    out = model(latents, timestep, cond_proj=cond_proj)
    ttnn.synchronize_device(mm3_mesh_device)
    measured = time.time() - t0
    _signpost("PERF_DIT_FORWARD_END")
    if _profiled():
        ttnn.ReadDeviceProfiler(mm3_mesh_device)
    assert torch.isfinite(out).all()
    times = []
    if not _profiled():
        for _ in range(5):
            ttnn.synchronize_device(mm3_mesh_device)
            t0 = time.time()
            model(latents, timestep, cond_proj=cond_proj)
            ttnn.synchronize_device(mm3_mesh_device)
            times.append(time.time() - t0)
        out_dir = DOC_DIR / "perf"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dit_forward_eager.json").write_text(
            json.dumps(
                {
                    "T": noise.shape[-1],
                    "batch": BATCH,
                    "signposted_forward_s": measured,
                    "repeat_s": times,
                    "median_ms": statistics.median(times) * 1e3,
                },
                indent=2,
            )
            + "\n"
        )
    ttnn.deallocate(cond_proj)
    model.release()
