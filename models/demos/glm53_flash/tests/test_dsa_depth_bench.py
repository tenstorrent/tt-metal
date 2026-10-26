# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One DSA layer's slab prefill of a 16 K chunk at growing prefix depths, on the full mesh (tt-blaze G53F-0198):
the depth term of the long prefill (G53F-0196: ~1.2 ms per 1 K prefix tokens per chunk over 11 DSA layers) without
building the 45-layer model.

Layer ``DB_LAYER`` (3) attention only, one slot; the prefix is seeded synthetically (``TtDSA.prefill_state`` in 32 K
chunks: latent pages + pooled-key index, no attention); at every depth in ``DB_DEPTHS`` one all-sparse 16 K chunk runs
through ``prefill_forward(slab=True)`` twice (cold, warm: same rows into the same pages), then once with the DSA
``StageTimer`` (every stage synchronized: the split, not the wall). ``DB_VARIANTS`` = ``qblock:ablock[:bucket]`` run at
every depth (``GLM53_PREFILL_QBLOCK`` / ``GLM53_PREFILL_ABLOCK`` / ``GLM53_INDEXER_POOL_BUCKET``, read per call); each
variant's warm output (every mesh row's slab, column 0) is compared bitwise with the first variant's.

    GLM53_MESH_SHAPE=8,4 DB_DEPTHS=131072,524288,1015808 pytest models/demos/glm53_flash/tests/test_dsa_depth_bench.py -s
"""

from __future__ import annotations

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.test_cache_1m import _attn_keys
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.dsa import StageTimer, TtDSA, TtLatentCache
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
LAYER = int(os.environ.get("DB_LAYER", "3"))
DEPTHS = [int(v) for v in os.environ.get("DB_DEPTHS", "131072,524288,1015808").split(",")]
# qblock:ablock[:bucket] -- bucket = GLM53_INDEXER_POOL_BUCKET (G53F-0198, 0 = off); the first variant is the reference
VARIANTS = [(v.split(":") + ["0"])[:3] for v in os.environ.get("DB_VARIANTS", "512:512:0,512:512:16384").split(",")]
CHUNK = 16384
SEED_CHUNK = 32768
OUT = os.environ.get("DB_OUT", "")


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_dsa_depth_bench(mesh_device):
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot), name_filter=_attn_keys)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    mesh = GLMMesh.from_device(mesh_device)
    attn = TtDSA(
        mesh_device,
        cfg,
        {k[len("self_attn.") :]: v for k, v in raw.items() if k.startswith("self_attn.")},
        layer_idx=LAYER,
        mesh=mesh,
    )
    max_seq = max(DEPTHS) + CHUNK + 64 * mesh.rows
    cache = TtLatentCache(mesh_device, cfg, max_batch=1, max_seq_len=max_seq, mesh=mesh)
    g = torch.Generator().manual_seed(11)
    H = cfg.hidden_size
    timer = StageTimer(mesh_device)
    rows_out = []
    filled = 0
    for D in DEPTHS:
        t0 = time.perf_counter()
        while filled < D:
            T = min(SEED_CHUNK, D - filled)
            x = (torch.randn(1, 1, T, H, generator=g) * 0.5).to(torch.bfloat16)
            x_tt = mesh.upload(x, ttnn.bfloat16)
            attn.prefill_state(x_tt, cache, 0, T, start=filled)
            ttnn.deallocate(x_tt)
            filled += T
        ttnn.synchronize_device(mesh_device)
        seed_s = time.perf_counter() - t0
        xc = (torch.randn(1, 1, CHUNK, H, generator=g) * 0.5).to(torch.bfloat16)
        x_slab = mesh.upload(xc, ttnn.bfloat16, rows=2)  # row r holds slab r of the chunk
        ref = None
        for qb, ab, bucket in VARIANTS:
            os.environ["GLM53_PREFILL_QBLOCK"], os.environ["GLM53_PREFILL_ABLOCK"] = qb, ab
            os.environ["GLM53_INDEXER_POOL_BUCKET"] = bucket
            walls, outs = [], []
            for _ in range(2):  # cold (compiles), warm: the same rows into the same pages
                ttnn.synchronize_device(mesh_device)
                t1 = time.perf_counter()
                y = attn.prefill_forward(x_slab, cache, 0, CHUNK, start=D, slab=True)
                ttnn.synchronize_device(mesh_device)
                walls.append(time.perf_counter() - t1)
                outs.append(torch.cat([ttnn.to_torch(t) for t in mesh.device_tensors(y)[:: mesh.cols]], dim=-2))
                ttnn.deallocate(y)
            cold_eq_warm = torch.equal(outs[0], outs[1])  # the repeated chunk is deterministic
            ref = outs[1] if ref is None else ref
            eq_ref = torch.equal(outs[1], ref)
            timer.reset()
            attn.set_stage_timer(timer)
            y = attn.prefill_forward(x_slab, cache, 0, CHUNK, start=D, slab=True)
            ttnn.deallocate(y)
            attn.set_stage_timer(None)
            stages = {k: round(v * 1e3, 2) for k, v in timer.times.items()}
            row = dict(
                depth=D,
                qblock=int(qb),
                ablock=int(ab),
                bucket=int(bucket),
                cold_ms=round(walls[0] * 1e3, 1),
                warm_ms=round(walls[1] * 1e3, 1),
                cold_eq_warm=cold_eq_warm,
                eq_ref=eq_ref,
                seed_s=round(seed_s, 1),
                stages_ms=stages,
            )
            rows_out.append(row)
            logger.info(f"[dsa-depth] {json.dumps(row)}")
        ttnn.deallocate(x_slab)
    os.environ.pop("GLM53_INDEXER_POOL_BUCKET", None)
    if OUT:
        with open(OUT, "w") as f:
            for r in rows_out:
                f.write(json.dumps(r) + "\n")
