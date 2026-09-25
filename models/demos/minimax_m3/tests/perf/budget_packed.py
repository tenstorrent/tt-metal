# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M3 prefill budget study, packed forwards: host wall time of one forward that packs B 2048-token
segments (TtPrefillRuntime.prefill_segments), for a list of compositions sharing one layer set and B.

A composition is one forward's segment list, written as in the study doc:
    548864:2048,0:2048             two slots: a deep hot segment and a cold one
    20480:2048+22528:2048          one slot, two consecutive chunks (depth-first)
    3@141312:1024                  optional "s@" picks the token stream (default: the entry's slot index)
Comma-separated entries map to slots 0..B'-1 in order; each "+"-joined run stays in its entry's slot.

Before a composition runs, every slot it uses is filled up to its first segment's cached_len with real
tokens, through packed forwards of that slot's consecutive chunks (padded with cold segments in a scratch
slot). Fill depth is remembered per slot, so compositions sharing a history do not refill it. Token
stream s at position p is token_ids[(p + 7919 * s) % len].

BUDGET_REFERENCE=1 instead runs every segment alone through the ORIGINAL path (prefill_chunk,
chunk_size = segment size, no segment_size) — the correctness reference for the packed path.

Env:
  BUDGET_LAYER_IDS   global layer indices                                                  [required]
  BUDGET_COMPOS      ';'-separated NAME=composition                                        [required]
  BUDGET_B           segments per forward (W = B * 2048); every composition has B segments   [required]
  BUDGET_CAPACITY    per-slot KV capacity in tokens                        [default: max needed + 2048]
  BUDGET_WARMUP / BUDGET_ITERS   warm-up and timed forwards per composition                [2 / 5]
  BUDGET_DUMP_KV     directory: after the last composition, save each slot's [k, v, index_k] (torch)
  BUDGET_REFERENCE   1 -> run each segment alone on the plain path (see above)            [default 0]
  BUDGET_TOPK_DUMP   file: record the MSA top-k block ids of one extra forward of the last composition
  BUDGET_STAGES / BUDGET_STAGE, BUDGET_TOKENS, M3_FABRIC, EXPERT_DTYPE, HF_MODEL, TT_CACHE_PATH as
  for budget_sweep.py.
"""

import json
import os
import resource
import statistics
import sys
import time

import ttnn
from models.demos.minimax_m3.tt.ccl import L1_SMALL_SIZE
from models.demos.minimax_m3.utils.fabric_env import ccl_topology_from_env, fabric_config_from_env

SEG = 2048
STREAM_STRIDE = 7919


def emit(**kw):
    print("RESULT " + json.dumps(kw), flush=True)


def parse_compo(spec):
    """'a:b,c:d+e:f' -> [(slot, stream, h, n), ...] in loop order."""
    segs = []
    for slot, entry in enumerate(spec.split(",")):
        stream = slot
        if "@" in entry:
            stream, entry = entry.split("@")
            stream = int(stream)
        for part in entry.split("+"):
            h, _, n = part.partition(":")
            h, n = int(h), int(n or SEG)
            assert h % SEG == 0 and 0 < n <= SEG, f"bad segment {part}"
            segs.append((slot, stream, h, n))
    return segs


def build(mesh, layer_ids, W, seg, capacity, num_users, stages, stage):
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
        segment_size=seg,
        num_users=num_users,
        expert_weight_dtype=expert_dtype,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        first_layer_idx=first,
        layer_indices=layer_ids,
        topology=ccl_topology_from_env(),
        is_first_rank=True,
        is_last_rank=stage == stages - 1,
    )
    runtime = TtPrefillRuntime(mesh, hf_config, {}, cfg)
    kv_cache = allocate_kv_caches(
        mesh, num_layers=len(layer_ids), max_seq_len=capacity, num_users=num_users, head_dim=hf_config.head_dim
    )
    return runtime, kv_cache


def main():
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != hard:
        resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))

    layer_ids = [int(x) for x in os.environ["BUDGET_LAYER_IDS"].split(",") if x.strip()]
    compos = []
    for item in os.environ["BUDGET_COMPOS"].split(";"):
        if item.strip():
            name, spec = item.split("=", 1)
            compos.append((name.strip(), spec.strip(), parse_compo(spec.strip())))
    reference = os.getenv("BUDGET_REFERENCE", "0") == "1"
    B = 1 if reference else int(os.environ["BUDGET_B"])
    for name, spec, segs in compos:
        assert reference or len(segs) == B, f"{name}: {len(segs)} segments, BUDGET_B={B}"
    n_slots = max(s for _, _, segs in compos for s, *_ in segs) + 1
    scratch = n_slots  # cold filler segments for fill forwards
    need = max(h + SEG for _, _, segs in compos for _, _, h, _ in segs)
    capacity = max(int(os.getenv("BUDGET_CAPACITY", "0")), need)
    capacity = -(-capacity // SEG) * SEG
    warmup = int(os.getenv("BUDGET_WARMUP", "2"))
    iters = int(os.getenv("BUDGET_ITERS", "5"))
    stages = int(os.getenv("BUDGET_STAGES", "2"))
    stage = int(os.getenv("BUDGET_STAGE", "0"))
    dump = os.getenv("BUDGET_DUMP_KV")

    src = json.load(open(os.environ["BUDGET_TOKENS"]))["token_ids"]
    tok = lambda stream, p: [src[(p + STREAM_STRIDE * stream + i) % len(src)] for i in range(SEG)]

    emit(
        kind="config",
        layers=layer_ids,
        B=B,
        W=B * SEG,
        compos=[(n, s) for n, s, _ in compos],
        capacity=capacity,
        slots=n_slots,
        reference=reference,
        warmup=warmup,
        iters=iters,
        stages=stages,
        stage=stage,
        fabric=str(fabric_config_from_env()),
    )

    ttnn.set_fabric_config(fabric_config_from_env())
    galaxy = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), l1_small_size=L1_SMALL_SIZE)
    try:
        mesh = galaxy.create_submeshes(ttnn.MeshShape(8 // stages, 4))[stage] if stages > 1 else galaxy
        t0 = time.perf_counter()
        runtime, kv_cache = build(
            mesh, layer_ids, B * SEG, None if reference else SEG, capacity, n_slots + 1, stages, stage
        )
        emit(kind="built", load_s=round(time.perf_counter() - t0, 1), mesh=list(mesh.shape))

        def run(segs):
            """segs: [(slot, stream, h, n)] -> wall ms of the forward(s), synced at the end only."""
            t = time.perf_counter()
            if reference:
                for slot, stream, h, n in segs:
                    out = runtime.prefill_chunk(
                        runtime.make_chunk_input(tok(stream, h)),
                        kv_cache,
                        slot_id=slot,
                        actual_start=h,
                        actual_end=h + n,
                    )
                    if out is not None:
                        out.deallocate(True)
            else:
                inp = runtime.make_segments_input([tok(stream, h) for _, stream, h, _ in segs])
                out = runtime.prefill_segments(inp, kv_cache, [(slot, h, n) for slot, _, h, n in segs])
                if out is not None:
                    out.deallocate(True)
            ttnn.synchronize_device(mesh)
            return (time.perf_counter() - t) * 1e3

        depth = {}  # slot -> (stream, filled depth)

        def fill(slot, stream, h):
            d = depth.get(slot, (stream, 0))[1] if depth.get(slot, (stream, 0))[0] == stream else 0
            todo = [(slot, stream, p, SEG) for p in range(d, h, SEG)]
            while todo:
                group, todo = todo[:B], todo[B:]
                group += [(scratch, 0, 0, SEG)] * (B - len(group))
                emit(kind="fill", slot=slot, h=group[0][2], wall_ms=round(run(group), 3))
            depth[slot] = (stream, max(d, h))

        for ci, (name, spec, segs) in enumerate(compos):
            for slot, stream, h, n in segs:
                if (slot, stream) not in [(s, st) for s, st, *_ in segs[: segs.index((slot, stream, h, n))]]:
                    fill(slot, stream, h)  # history before this slot's first segment in the composition
            for k in range(max(warmup, 5) if ci == 0 else warmup):
                emit(kind="warmup", compo=name, iter=k, wall_ms=round(run(segs), 3))
            walls = []
            for k in range(iters):
                walls.append(run(segs))
                emit(kind="iter", compo=name, iter=k, wall_ms=round(walls[-1], 3))
            emit(
                kind="point",
                compo=name,
                spec=spec,
                segments=[{"slot": s, "h": h, "n": n} for s, _, h, n in segs],
                W=B * SEG,
                iters=iters,
                wall_ms_median=round(statistics.median(walls), 3),
                wall_ms_min=round(min(walls), 3),
                wall_ms_max=round(max(walls), 3),
            )
            for slot, stream, h, n in segs:  # the composition wrote these positions: history now reaches them
                depth[slot] = (stream, max(depth.get(slot, (stream, 0))[1], h + SEG))

        topk_dump = os.getenv("BUDGET_TOPK_DUMP")
        if topk_dump:
            # One extra untimed forward of the last composition, recording every MSA call's top-k block ids
            # (order: packed = layer-major then segment; reference = segment-major then layer).
            import torch

            from models.demos.minimax_m3.tt.attention import msa

            composer = ttnn.ConcatMesh2dToTensor(mesh, dims=(2, 1), mesh_shape=mesh.shape)
            captured = []
            msa.set_block_id_sink(lambda ids: captured.append(ttnn.to_torch(ids, mesh_composer=composer)))
            run(compos[-1][2])
            msa.set_block_id_sink(None)
            segs = compos[-1][2]
            torch.save(
                {"ids": captured, "reference": reference, "segments": [(h, n) for _, _, h, n in segs]}, topk_dump
            )
            emit(kind="topk_dumped", path=topk_dump, calls=len(captured))

        if dump:
            import torch

            os.makedirs(dump, exist_ok=True)
            for slot in range(n_slots):
                torch.save(runtime.read_slot_kv(kv_cache, slot), os.path.join(dump, f"slot{slot}.pt"))
            emit(kind="dumped", dir=dump, slots=n_slots)
        emit(kind="done")
    finally:
        for sub in galaxy.get_submeshes():
            ttnn.close_mesh_device(sub)
        ttnn.close_mesh_device(galaxy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
