# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Served hang reproducer (2026-10-27 smoke r10): the first traced decode step after a prompt whose length is an exact
multiple of the 128-token pool (65536 tokens: the prefill leaves an EMPTY tail pool) hung the 8x4, while prompts ending
inside a pool (65514-token needles, whose decode later crossed 65535/65536 incrementally) and 128-token prompts (dense
flash-MLA range, no tail gather) run. Minimal form on the (2,4) block: 4 layers (0-3, layer 3 = DSA), a vllm-bench-style
ramp prompt of GLM53_BOUNDARY_LEN tokens (2176 = 17 pools + empty tail, above the 2050 dense bound, vs the 2175 control)
prefilled in 1024-token chunks (the KDA SP L1 window), then the eager decode step, the capture and 3 replays at
positions len, len+1, len+2. A hang trips the watchdog (set TT_METAL_OPERATION_TIMEOUT_SECONDS in the stage env).

    GLM53_MESH_SHAPE=2,4 GLM53_MODEL_LAYERS=0-3 GLM53_MODEL_B=4 GLM53_MODEL_TRACE_MB=512 GLM53_BOUNDARY_LEN=2176 \
    pytest models/demos/glm53_flash/tests/test_decode_pool_boundary_mesh.py -s
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
B = int(os.environ.get("GLM53_MODEL_B", "4"))
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
LEN = int(os.environ.get("GLM53_BOUNDARY_LEN", "2176"))
MAX_SEQ_LEN = int(os.environ.get("GLM53_BOUNDARY_MAX_SEQ_LEN", str(LEN + 256)))
PREFILL_CHUNK = int(os.environ.get("GLM53_BOUNDARY_CHUNK", str(512 * MESH[0])))
N_REPLAYS = int(os.environ.get("GLM53_BOUNDARY_REPLAYS", "3"))  # 25+ crosses the next pool boundary incrementally
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/home/ttuser/ashai/glm53/_cache/mesh")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)


def _layer_range() -> list[int]:
    spec = os.environ.get("GLM53_MODEL_LAYERS", "0-3")
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in spec.split(",")]


def _ramp(n: int) -> torch.Tensor:
    """vllm bench "random" prompt: a ramp of consecutive vocab ids (what the served run decoded)."""
    return (torch.arange(n, dtype=torch.long) + 12345) % 150000


@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_decode_after_pool_boundary_prompt(mesh_device):
    cfg = GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        _layer_range(),
        max_batch=B,
        max_seq_len=MAX_SEQ_LEN,
        weight_cache_path=CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    U = model.total_users
    t0 = time.perf_counter()
    model.prefill_chunked(_ramp(LEN), user=0, chunk=PREFILL_CHUNK)
    ttnn.synchronize_device(mesh_device)
    logger.info(
        f"BOUNDARY prefill of {LEN} tokens (tail {LEN % cfg.index_kpool} rows) in {time.perf_counter() - t0:.1f} s"
    )
    pos = [LEN if u % B == 0 else 0 for u in range(U)]  # slot 0 of every row decodes after the prompt; the rest idle
    ids = torch.full((U,), 1000, dtype=torch.long)
    t0 = time.perf_counter()
    model.warmup_decode(ids, pos)  # the eager step at the boundary position (the served hang site)
    logger.info(f"BOUNDARY eager decode step at position {LEN} done in {time.perf_counter() - t0:.1f} s")
    trace_id, outs = model.capture(ids, [p + 1 if p else 0 for p in pos])
    for k in range(N_REPLAYS):
        model.load_ids(ids)
        model.set_positions([p + 2 + k if p else 0 for p in pos])
        t0 = time.perf_counter()
        model.replay(trace_id)
        got = model.ids_to_host(outs[2])
        logger.info(
            f"BOUNDARY replay {k} at position {LEN + 2 + k}: {time.perf_counter() - t0 :.3f} s, slot-0 ids {got[::B].tolist()}"
        )
    logger.info(f"BOUNDARY PASS len {LEN}")
