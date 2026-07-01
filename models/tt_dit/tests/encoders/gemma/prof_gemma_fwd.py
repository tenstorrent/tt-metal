# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-op device-time profile of the Gemma encoder FORWARD ONLY (no feature-extractor/
connectors) — fewer ops than the full encode, to dodge the profiler zone/merge limits.
Per-layer ReadDeviceProfiler flush. GEMMA_PROF_MESH=1x1 or 2x4 (default 2x4).

Run: python -m tracy -p -r -m pytest <this>::test_prof_gemma_fwd -s
"""

import os

import pytest

import ttnn
from models.tt_dit.encoders.gemma.model_gemma import GemmaEncoderLayer
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline
from models.tt_dit.tests.encoders.gemma.test_gemma_full import PROMPT, _gemma_path


def _walk(m):
    yield m
    for _, c in m.named_children():
        yield from _walk(c)


_MESH = tuple(int(v) for v in os.environ.get("GEMMA_PROF_MESH", "2x4").split("x"))
_DP = (
    {"l1_small_size": 8192}
    if _MESH == (1, 1)
    else {"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D}
)


@pytest.mark.parametrize("device_params", [_DP], indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_prof_gemma_fwd(mesh_device, device_params):
    gemma = _gemma_path()
    if not os.path.isdir(gemma):
        pytest.skip(f"Gemma not found: {gemma}")
    pipe = LTXPipeline.create_pipeline(mesh_device, checkpoint_name=None, gemma_path=gemma, mode="av")
    pair = pipe.gemma_encoder_pair
    pair.load_gemma_encoder(gemma)
    enc = pair.gemma_encoder
    # Limit layers to keep the profiler zone count low (48 layers overflow the source-location
    # hash table). One representative layer's op breakdown × 48 ≈ the forward.
    _nl = int(os.environ.get("GEMMA_PROF_NLAYERS", "0"))
    if _nl:
        enc.layers = enc.layers[:_nl]

    tokens = pair.tokenizer(
        PROMPT, return_tensors="pt", padding="max_length", max_length=pair._sequence_length, truncation=True
    )
    tt_ids = ttnn.from_torch(tokens.input_ids, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_mask = enc.build_attn_mask(tokens.attention_mask, tt_ids.shape[-1])

    for mod in _walk(enc):
        if isinstance(mod, GemmaEncoderLayer):
            orig = mod.forward

            def timed(*a, _orig=orig, **k):
                r = _orig(*a, **k)
                ttnn.ReadDeviceProfiler(mesh_device)
                return r

            mod.forward = timed

    enc(tt_ids, tt_attn_mask=tt_mask)  # warm program cache
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)

    enc(tt_ids, tt_attn_mask=tt_mask)  # measured
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)
