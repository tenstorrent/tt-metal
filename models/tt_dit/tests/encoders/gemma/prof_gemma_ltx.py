# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device-time profiling for the LTX Gemma text encoder (full encode_prompts:
tokenize → 48 layers → feature extractor → video/audio connectors).

Mirrors prof_vae_wan.py: flushes the on-device profiler after each decoder layer
(and after the feature extractor and each connector) to stay under Tracy's
1000-zone buffer, then reports the warm host wall-clock. Sum DEVICE FW DURATION
from the CSV and compare to GEMMA_ENCODE_HOST_WALL_MS — the gap is host dispatch.

No torch reference and no PCC: pure device timing. Reuses the real pipeline + 12B
weights (op timings are data-independent).

Run: python -m tracy -p -r -v -m pytest <this>::test_prof_gemma_ltx_devicetime -s -v
"""

import os
import time

import pytest
from safetensors import safe_open

import ttnn
from models.tt_dit.encoders.gemma.model_gemma import GemmaEncoderLayer
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline
from models.tt_dit.tests.encoders.gemma.test_gemma_full import CONNECTOR_PREFIXES, PROMPT, _gemma_path, _ltx_ckpt


def _walk(m):
    yield m
    for _, c in m.named_children():
        yield from _walk(c)


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_prof_gemma_ltx_devicetime(mesh_device, device_params):
    gemma = _gemma_path()
    ckpt = _ltx_ckpt()
    if not os.path.isdir(gemma):
        pytest.skip(f"Gemma not found: {gemma}")
    if not ckpt:
        pytest.skip("LTX checkpoint not found")

    pipe = LTXPipeline.create_pipeline(mesh_device, checkpoint_name=None, gemma_path=gemma, mode="av")
    pipe.gemma_encoder_pair.load_gemma_encoder(gemma)

    conn_state = {}
    with safe_open(ckpt, "pt") as f:
        for k in f.keys():
            if k.startswith(CONNECTOR_PREFIXES):
                conn_state[k] = f.get_tensor(k)
    pipe.gemma_encoder_pair.load_embeddings_connectors(conn_state, audio_num_blocks=8)

    # Flush after each decoder layer (≈15 ops) plus the feature extractor and each
    # connector — keeps every flush window well under Tracy's 1000-zone buffer.
    pair = pipe.gemma_encoder_pair

    def _flush_after(mod):
        orig = mod.forward

        def timed(*a, _orig=orig, **k):
            r = _orig(*a, **k)
            ttnn.ReadDeviceProfiler(mesh_device)
            return r

        mod.forward = timed

    for mod in _walk(pair.gemma_encoder):
        if isinstance(mod, GemmaEncoderLayer):
            _flush_after(mod)
    _flush_after(pair.feature_extractor)
    _flush_after(pair.video_connector)
    if pair.audio_connector is not None:
        _flush_after(pair.audio_connector)

    pipe.encode_prompts([PROMPT], use_cache=False)  # warm program cache
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)

    t0 = time.perf_counter()
    pipe.encode_prompts([PROMPT], use_cache=False)
    ttnn.synchronize_device(mesh_device)
    host_wall = (time.perf_counter() - t0) * 1000
    ttnn.ReadDeviceProfiler(mesh_device)
    print(f"\nGEMMA_ENCODE_HOST_WALL_MS={host_wall:.2f}", flush=True)
