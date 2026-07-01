# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Reliable warm-latency measurement for the LTX Gemma encoder (full encode_prompts).

The single-shot wall-clock in test_gemma_full swings ±50ms run-to-run, so perf deltas
need median-of-N. This runs N warm encodes and reports min/median/max — the measurement
tool for the trace + fusion work. No torch reference, no PCC.

Run: GEMMA_TIME_ITERS=20 python -m pytest <this>::test_time_gemma_ltx -s
"""

import os
import statistics
import time

import pytest
from safetensors import safe_open

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline
from models.tt_dit.tests.encoders.gemma.test_gemma_full import CONNECTOR_PREFIXES, PROMPT, _gemma_path, _ltx_ckpt

_MESH = tuple(int(v) for v in os.environ.get("GEMMA_TIME_MESH", "2x4").split("x"))


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_time_gemma_ltx(mesh_device, device_params):
    gemma = _gemma_path()
    ckpt = _ltx_ckpt()
    if not os.path.isdir(gemma):
        pytest.skip(f"Gemma not found: {gemma}")
    if not ckpt:
        pytest.skip("LTX checkpoint not found")

    # dynamic_load=False keeps the encoder resident so the whole-encode trace is captured once and
    # replayed warm — the latency this harness measures.
    pipe = LTXPipeline.create_pipeline(
        mesh_device, checkpoint_name=None, gemma_path=gemma, mode="av", dynamic_load=False
    )
    pipe.gemma_encoder_pair.load_gemma_encoder(gemma)
    conn_state = {}
    with safe_open(ckpt, "pt") as f:
        for k in f.keys():
            if k.startswith(CONNECTOR_PREFIXES):
                conn_state[k] = f.get_tensor(k)
    pipe.gemma_encoder_pair.load_embeddings_connectors(conn_state, audio_num_blocks=8)

    iters = int(os.environ.get("GEMMA_TIME_ITERS", "20"))
    pipe.encode_prompts([PROMPT], use_cache=False)  # warm program cache
    ttnn.synchronize_device(mesh_device)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        pipe.encode_prompts([PROMPT], use_cache=False)
        ttnn.synchronize_device(mesh_device)
        times.append((time.perf_counter() - t0) * 1e3)

    times.sort()
    print(
        f"\nGEMMA_ENCODE_MS n={iters} min={times[0]:.1f} median={statistics.median(times):.1f} "
        f"max={times[-1]:.1f} mean={statistics.mean(times):.1f}",
        flush=True,
    )
