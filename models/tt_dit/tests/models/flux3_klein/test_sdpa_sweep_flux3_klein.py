# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os
import statistics

import pytest
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.blocks.attention_opt import Attention
from models.tt_dit.pipelines.flux3_klein.pipeline_flux3_klein import Flux3KleinPipeline, Flux3KleinRequest
from models.tt_dit.utils.test import ring_params_8k

PROMPT = "A man in a red jacket walks a golden retriever along a snowy forest path at sunrise, photorealistic"
REFERENCES = {"none": None, "1mp": (1024, 1024), "4mp": (2048, 2048)}
CANDIDATES = [(q, k) for q in (64, 128, 256) for k in (128, 256, 512)]
RESULTS_FILE = os.environ.get("SDPA_SWEEP_RESULTS", "sdpa_sweep.jsonl")

_override = {"chunks": None, "lengths": set()}
_original = Attention.get_ring_sdpa_program_config


def _patched(self, per_device_seq_len):
    _override["lengths"].add(per_device_seq_len)
    chunks = _override["chunks"]
    if chunks is None:
        return _original(self, per_device_seq_len)
    cache = self.__dict__.setdefault("_sweep_program_configs", {})
    key = (per_device_seq_len, chunks)
    if key not in cache:
        cache[key] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.get_ring_sdpa_core_grid(per_device_seq_len),
            q_chunk_size=chunks[0],
            k_chunk_size=chunks[1],
            exp_approx_mode=self.ring_sdpa_exp_approx,
        )
    return cache[key]


def _record(**fields):
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(fields) + "\n")


@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [{**ring_params_8k, "l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("size", [1024, 2048])
def test_sdpa_sweep(*, mesh_device, size, monkeypatch):
    monkeypatch.setattr(Attention, "get_ring_sdpa_program_config", _patched)
    device = mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    pipeline = Flux3KleinPipeline.create(device, tp_axis=1, sp_axis=0, topology=ttnn.Topology.Linear)
    pipeline.warmup((size,))

    for reference, ref_size in REFERENCES.items():
        images = () if ref_size is None else (Image.new("RGB", ref_size, (90, 120, 160)),)
        request = Flux3KleinRequest(PROMPT, width=size, height=size, seed=123, images=images)
        for chunks in [None, *CANDIDATES]:
            _override["chunks"], _override["lengths"] = chunks, set()
            try:
                pipeline.generate(request)
                denoise = [pipeline.generate(request).timings["denoise"] for _ in range(2)]
            except Exception as e:
                _record(size=size, reference=reference, chunks=chunks, status=f"{type(e).__name__}: {e}"[:300])
                logger.warning(f"SDPA {size} {reference} {chunks}: {type(e).__name__}: {e}"[:300])
                continue
            med = statistics.median(denoise)
            _record(
                size=size, reference=reference, chunks=chunks, status="ok", denoise_s=round(med, 4),
                runs=[round(d, 4) for d in denoise], lengths=sorted(_override["lengths"]),
            )
            logger.info(f"SDPA {size} {reference} {chunks}: denoise {med:.3f} s, lengths {sorted(_override['lengths'])}")
    _override["chunks"] = None
    pipeline.close()
