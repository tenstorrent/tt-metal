# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M3 prefill budget study: host wall time of ONE prefill forward vs width, history depth and capacity.

One process = one layer set on one stage sub-mesh at one forward width W. The KV cache is allocated at
BUDGET_CAPACITY tokens and filled progressively with REAL tokens (full W-token chunks, un-timed but
each one synced and reported as a `fill` row), so the timed points walk up in history h:

    for (h, n) in sorted(BUDGET_POINTS):
        fill the cache from the current depth up to h
        BUDGET_WARMUP + BUDGET_ITERS forwards of one chunk at actual_start=h, actual_end=h+n

A timed forward is `prefill_chunk` + `synchronize_device`; nothing syncs inside it. The chunk at h is
re-written by every iteration and by the next fill, so points can share one process. No zone profiler
and no device profiler: this is the timing harness, profile_prefill.py is the profiling one.

Constraint on main: h must be a multiple of W (the block-cyclic SP cache has period W).

Output: one `RESULT {json}` line per fill chunk, timed iteration and point summary, parsed by
budget_collect.py.

Env:
  BUDGET_LAYER_IDS   global layer indices, e.g. 0,1,2 (D) or 8,...,15 (S8)                [required]
  BUDGET_W           forward width in tokens (= chunk_size)                             [default 2048]
  BUDGET_POINTS      comma list of h:n (n defaults to W), e.g. 0:2048,16384:256          [default 0:W]
  BUDGET_CAPACITY    per-slot KV capacity in tokens, rounded up to a multiple of W   [default max h + W]
  BUDGET_WARMUP      warm-up forwards per point                                           [default 2]
  BUDGET_ITERS       timed forwards per point                                             [default 5]
  BUDGET_FILL        real | none. none skips the history fill (attends a zeroed cache)   [default real]
  BUDGET_STAGES      1, 2 or 4 (sub-mesh (8/S, 4)); BUDGET_STAGE picks which one      [default 2 / 0]
  BUDGET_TOKENS      metadata.json with token_ids, tiled to length                       [required]
  M3_FABRIC, M3_CCL_TOPOLOGY, EXPERT_DTYPE, HF_MODEL, TT_CACHE_PATH as for profile_prefill.py.
"""

import json
import os
import resource
import statistics
import sys
import time

from loguru import logger  # noqa: F401

import ttnn
from models.demos.minimax_m3.tt.ccl import L1_SMALL_SIZE
from models.demos.minimax_m3.utils.fabric_env import ccl_topology_from_env, fabric_config_from_env


def emit(**kw):
    print("RESULT " + json.dumps(kw), flush=True)


def parse_points(spec, W):
    pts = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        h, _, n = item.partition(":")
        h, n = int(h), int(n or W)
        assert h % W == 0, f"h={h} must be a multiple of W={W} (block-cyclic SP cache)"
        assert 0 < n <= W, f"n={n} must be in (0, {W}]"
        pts.append((h, n))
    return sorted(pts)


def build(mesh, layer_ids, W, capacity, stages, stage):
    from models.demos.minimax_m3.tt.attention import allocate_kv_caches
    from models.demos.minimax_m3.tt.model_config import ModelArgs
    from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    model_args = ModelArgs(mesh_device=mesh)
    hf_config = model_args.hf_config
    per_stage = hf_config.num_hidden_layers // stages
    first, end = stage * per_stage, (stage + 1) * per_stage
    assert all(first <= i < end for i in layer_ids), f"layers {layer_ids} outside stage [{first}, {end})"
    hf_config.num_hidden_layers = len(layer_ids)
    os.environ.setdefault("M3_WEIGHTS_FROM_CACHE", "1")
    expert_dtype = ttnn.bfloat8_b if os.getenv("EXPERT_DTYPE", "bf4") == "bf8" else ttnn.bfloat4_b
    cfg = TtPrefillRuntimeConfig(
        num_layers=len(layer_ids),
        max_seq_len=capacity,
        mesh_shape=tuple(mesh.shape),
        chunk_size=W,
        num_users=1,
        expert_weight_dtype=expert_dtype,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        first_layer_idx=first,
        layer_indices=layer_ids,
        topology=ccl_topology_from_env(),
        is_first_rank=True,  # every stage embeds its own tokens (see profile_prefill.py)
        is_last_rank=stage == stages - 1,
    )
    runtime = TtPrefillRuntime(mesh, hf_config, {}, cfg)
    kv_cache = allocate_kv_caches(
        mesh, num_layers=len(layer_ids), max_seq_len=capacity, num_users=1, head_dim=hf_config.head_dim
    )
    return runtime, kv_cache


def main():
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != hard:
        resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))

    layer_ids = [int(x) for x in os.environ["BUDGET_LAYER_IDS"].split(",") if x.strip()]
    W = int(os.getenv("BUDGET_W", "2048"))
    points = parse_points(os.getenv("BUDGET_POINTS", f"0:{W}"), W)
    need = points[-1][0] + W
    capacity = int(os.getenv("BUDGET_CAPACITY", str(need)))
    capacity = -(-max(capacity, need) // W) * W
    warmup = int(os.getenv("BUDGET_WARMUP", "2"))
    iters = int(os.getenv("BUDGET_ITERS", "5"))
    fill = os.getenv("BUDGET_FILL", "real")
    assert fill in ("real", "none"), fill
    stages = int(os.getenv("BUDGET_STAGES", "2"))
    stage = int(os.getenv("BUDGET_STAGE", "0"))

    src = json.load(open(os.environ["BUDGET_TOKENS"]))["token_ids"]
    tokens = lambda a: [src[(a + i) % len(src)] for i in range(W)]

    emit(
        kind="config",
        layers=layer_ids,
        W=W,
        points=points,
        capacity=capacity,
        warmup=warmup,
        iters=iters,
        fill=fill,
        stages=stages,
        stage=stage,
        fabric=str(fabric_config_from_env()),
    )

    ttnn.set_fabric_config(fabric_config_from_env())
    galaxy = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), l1_small_size=L1_SMALL_SIZE)
    try:
        mesh = galaxy.create_submeshes(ttnn.MeshShape(8 // stages, 4))[stage] if stages > 1 else galaxy
        t0 = time.perf_counter()
        runtime, kv_cache = build(mesh, layer_ids, W, capacity, stages, stage)
        emit(kind="built", load_s=round(time.perf_counter() - t0, 1), mesh=list(mesh.shape))

        def forward(h, n):
            inp = runtime.make_chunk_input(tokens(h))
            t = time.perf_counter()
            out = runtime.prefill_chunk(inp, kv_cache, slot_id=0, actual_start=h, actual_end=h + n)
            ttnn.synchronize_device(mesh)
            ms = (time.perf_counter() - t) * 1e3
            if out is not None:  # a non-last stage hands back its hidden state
                out.deallocate(True)
            return ms

        depth = 0
        for h, n in points:
            if fill == "real":
                for pos in range(depth, h, W):
                    emit(kind="fill", h=pos, n=W, wall_ms=round(forward(pos, W), 3))
            depth = h
            for k in range(warmup):
                emit(kind="warmup", h=h, n=n, iter=k, wall_ms=round(forward(h, n), 3))
            walls = []
            for k in range(iters):
                walls.append(forward(h, n))
                emit(kind="iter", h=h, n=n, iter=k, wall_ms=round(walls[-1], 3))
            emit(
                kind="point",
                h=h,
                n=n,
                W=W,
                iters=iters,
                wall_ms_median=round(statistics.median(walls), 3),
                wall_ms_min=round(min(walls), 3),
                wall_ms_max=round(max(walls), 3),
            )
        emit(kind="done")
    finally:
        for sub in galaxy.get_submeshes():
            ttnn.close_mesh_device(sub)
        ttnn.close_mesh_device(galaxy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
