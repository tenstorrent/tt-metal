# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-layer device-time profile of the FULL model's decode step under the device profiler (``python -m tracy -r -p -v -m
pytest ...``; the queue wrapper ``_logs/mesh/test_perfops_glm53.py`` runs it and ``tests/perf_mesh_report.py``). A 45-layer step
overflows the per-core profiler marker buffers (12 000 entries) if it is replayed as one trace and Tracy's post-processing
asserts on any op that lost its device data, so this test never captures a trace: it sets the decode positions directly
(seed-only caches, no prefill: op timings do not depend on the values), runs the eager warm-up step and then ONE labelled eager
step, draining the profiler after every layer's ``decode_forward`` (labels ``L{i}_{KDA|DSA}``; the step is wrapped in the
``PERF_DECODE`` and ``PERF_EAGER`` windows the report expects). Knobs: GLM53_FULL_POS (decode position, 1024),
GLM53_FULL_MAX_SEQ_LEN (cache capacity: the served 1 M vs POS + 256), the usual GLM53_MESH_SHAPE / GLM53_MODEL_LAYERS /
GLM53_MODEL_B / GLM53_PARTITION_USERS / GLM53_PREFILL_PARTITION.
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

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
B = int(os.environ.get("GLM53_MODEL_B", "4"))
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
POS = int(os.environ.get("GLM53_FULL_POS", "1024"))
MAX_SEQ_LEN = int(os.environ.get("GLM53_FULL_MAX_SEQ_LEN", str(POS + 256)))
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/home/ttuser/ashai/glm53/_cache/mesh")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"


def _layer_range() -> list[int]:
    spec = os.environ.get("GLM53_MODEL_LAYERS", "0-44")
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in spec.split(",")]


def _drain(mesh_device):
    if PROFILING:
        ttnn.ReadDeviceProfiler(mesh_device)


def _signpost(name: str):
    if PROFILING:
        from tracy import signpost

        signpost(name)


@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_full_model_decode_profile(mesh_device):
    layers = _layer_range()
    cfg = GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        layers,
        max_batch=B,
        max_seq_len=MAX_SEQ_LEN,
        weight_cache_path=CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    U = model.total_users
    logger.info(
        f"FULL-PROFILE {len(layers)} layers on {MESH}, B={B}/row ({U} slots), max_seq_len {MAX_SEQ_LEN}, position {POS}"
    )
    walls: dict[str, list[float]] = {}
    originals = []
    _drain(mesh_device)
    # per-layer labels + a profiler drain after every layer (keeps every core's marker buffer under its 12 000 entries)
    for i, dec in zip(layers, model.layers):
        orig = dec.decode_forward
        label = f"L{i}_{'KDA' if dec.is_kda else 'DSA'}"

        def wrapped(*a, _orig=orig, _label=label, **kw):
            _signpost(f"{_label}_BEGIN")
            t0 = time.perf_counter()
            try:
                return _orig(*a, **kw)
            finally:
                if not PROFILING:  # per-layer eager wall (device-synchronised): the profiler-free attribution
                    ttnn.synchronize_device(mesh_device)
                    walls.setdefault(_label, []).append(time.perf_counter() - t0)
                _signpost(f"{_label}_END")
                _drain(mesh_device)

        dec.decode_forward = wrapped
        originals.append((dec, orig))
    ids = torch.randint(0, cfg.vocab_size, (U,), generator=torch.Generator().manual_seed(0))
    pos = [POS] * U
    model.warmup_decode(ids, pos)  # compiles every program (eager; drained per layer)
    _drain(mesh_device)
    model.load_ids(ids)
    model.set_positions(pos)
    ttnn.synchronize_device(mesh_device)
    _drain(mesh_device)
    _signpost("PERF_DECODE")
    _signpost("PERF_EAGER")
    outs = model.decode_step(model.ids_dev, model.positions)
    ttnn.synchronize_device(mesh_device)
    _signpost("PERF_EAGER_END")
    _signpost("PERF_DECODE_END")
    _drain(mesh_device)
    for t in outs:
        if t is not None:
            ttnn.deallocate(t)
    if not PROFILING:
        # the labelled step's per-layer walls (the warm-up's first pass is excluded: it compiled) + a traced reference
        rows = [(k, 1e3 * v[-1]) for k, v in walls.items()]
        kda = [ms for k, ms in rows if k.endswith("_KDA")]
        dsa = [ms for k, ms in rows if k.endswith("_DSA")]
        for k, ms in rows:
            logger.info(
                f"LAYER-WALL {k}: {ms:.2f} ms (eager, synchronised: host dispatch dominates; use the traced step)"
            )
        logger.info(
            f"LAYER-WALL capacity {MAX_SEQ_LEN} position {POS}: KDA {sum(kda):.1f} ms over {len(kda)} layers "
            f"(mean {sum(kda) / max(1, len(kda)):.2f}), DSA {sum(dsa):.1f} ms over {len(dsa)} (mean "
            f"{sum(dsa) / max(1, len(dsa)):.2f}), eager step {sum(ms for _, ms in rows):.1f} ms"
        )
        for dec, orig in originals:  # no synchronisation / reads inside the capture
            dec.decode_forward = orig
        trace_id, touts = model.capture(ids, pos)
        for _ in range(3):
            model.replay(trace_id)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(10):
            model.replay(trace_id, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ms = (time.perf_counter() - t0) / 10 * 1e3
        logger.info(f"LAYER-WALL traced step {ms:.2f} ms = {1e3 / ms:.1f} tok/s per user at capacity {MAX_SEQ_LEN}")
    logger.info(f"FULL-PROFILE done: one labelled eager step at position {POS}, capacity {MAX_SEQ_LEN}")
