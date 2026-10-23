# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host wall time of one packed prefill forward, B 2048-token chunks in one forward
(TtPrefillRuntime.prefill_forward, tt/prefill_chunk.py), for a list of compositions sharing one layer set and B.

A composition is one forward's chunk list:
    548864:2048,0:2048             two slots: a deep warm chunk and a cold one
    20480:2048+22528:2048          one slot, two consecutive chunks (depth-first)
    3@141312:1024                  optional "s@" picks the token stream (default: the chunk's slot index)
Comma-separated chunks map to slots 0..N-1 in order (N chunks); each "+"-joined run stays in its chunk's slot.

Before a composition runs, every slot it uses is filled up to its first chunk's cached_len with real tokens, through
packed forwards of that slot's consecutive chunks (padded with empty chunks). Fill depth is remembered per slot, so
compositions sharing a history do not refill it; a slot a ragged chunk left mid-chunk must be restarted. Token
stream s at position p is token_ids[(p + 7919 * s) % len].

A layer subset is embedding-fed (the first listed layer sees the embedding, not layer N-1's output): shapes and
attention costs are real, MoE routing is not, the same caveat as profile_prefill.py's stage captures.

PACKED_REFERENCE=1 instead runs every chunk alone through the one-chunk path (prefill_chunk, forward_tokens_budget
= chunk_size) — the correctness reference for the packed path: dump both runs
with PACKED_DUMP_KV and compare the slots' K/V/index_k over the real positions (PCC per layer).

Env:
  PACKED_LAYER_IDS   global layer indices                                                  [required]
  PACKED_COMPOS      ';'-separated NAME=composition                                        [required]
  PACKED_B           chunks per forward (W = B * 2048); every composition has B chunks
                     [required unless PACKED_REFERENCE=1]
  PACKED_CAPACITY    per-slot KV capacity in tokens, rounded up to a multiple of 2048 [default: max h + 2048]
  PACKED_WARMUP / PACKED_ITERS   warm-up and timed forwards per composition; the first composition of a
                     process gets at least 5 warm-ups                                       [2 / 5]
  PACKED_DUMP_KV     directory: after the last composition, save each slot's [k, v, index_k] (torch)
  PACKED_FILL        real | none. none skips the history fill and attends a ZEROED cache: shapes, gathers
                     and op costs are the real ones but the block selection is degenerate (every score
                     equal), so the sparse attention runs faster than on real content. For profiling only:
                     a device-profiler capture of the real fill is tens of GB per 100k of depth  [default real]
  PACKED_REFERENCE   1 -> run each chunk alone on the plain path (see above)            [default 0]
  PACKED_PROFILE     1 -> zone-profile one extra forward of the last composition (run under
                     `python3 -m tracy -v -r -p`; root `profiled_chunk`; a packed forward re-enters the per-op
                     zones once per chunk under `attn/gather{i}` / `attn/chunk{i}` (README_profiling.md); the
                     profiler is drained every PACKED_PROFILE_READ_EVERY [16] forwards outside it)   [default 0]
  PACKED_STAGES      1, 2 or 4 (sub-mesh (8/S, 4)); PACKED_STAGE picks which one      [default 2 / 0]
  PACKED_TOKENS      metadata.json with token_ids, tiled to length                       [required]
  M3_FABRIC, M3_CCL_TOPOLOGY, EXPERT_DTYPE, HF_MODEL as for profile_prefill.py; TT_CACHE_PATH as in README.md.
"""

import json
import os
import statistics
import sys
import time

# The zone flags are read when the model modules are imported (utils/profiler_utils), so they go first.
PROFILE = os.getenv("PACKED_PROFILE", "0") == "1"
if PROFILE:
    os.environ.setdefault("M3_PROFILE_ZONES", "1")
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")

import ttnn
from models.demos.minimax_m3.tests.perf.profile_prefill import _raise_nproc_limit, build_runtime
from models.demos.minimax_m3.tt.ccl import L1_SMALL_SIZE
from models.demos.minimax_m3.tt.prefill_chunk import PrefillChunk
from models.demos.minimax_m3.utils.fabric_env import fabric_config_from_env

SEG = 2048
STREAM_STRIDE = 7919  # prime: no two streams alias for any token-file length


def emit(**kw):
    print("RESULT " + json.dumps(kw), flush=True)


def parse_compo(spec):
    """'a:b,c:d+e:f' -> [(slot, stream, h, n), ...] in loop order."""
    chunks = []
    for slot, chunk in enumerate(spec.split(",")):
        stream = slot
        if "@" in chunk:
            stream, chunk = chunk.split("@")
            stream = int(stream)
        for part in chunk.split("+"):
            h, _, n = part.partition(":")
            h, n = int(h), int(n or SEG)
            assert h % SEG == 0 and 0 < n <= SEG, f"bad chunk {part}"
            chunks.append((slot, stream, h, n))
    return chunks


def build(mesh, layer_ids, W, capacity, num_users, stages, stage):
    """profile_prefill.build_runtime sized for packed forwards: W tokens per forward over num_users cache slots."""
    runtime, kv_cache, _, _ = build_runtime(
        mesh,
        SEG,
        capacity,
        None,
        layer_ids=layer_ids,
        stages=stages,
        stage=stage,
        forward_tokens_budget=W,
        num_users=num_users,
    )
    return runtime, kv_cache


def main():
    _raise_nproc_limit()

    layer_ids = [int(x) for x in os.environ["PACKED_LAYER_IDS"].split(",") if x.strip()]
    compos = []
    for item in os.environ["PACKED_COMPOS"].split(";"):
        if item.strip():
            name, spec = item.split("=", 1)
            compos.append((name.strip(), spec.strip(), parse_compo(spec.strip())))
    reference = os.getenv("PACKED_REFERENCE", "0") == "1"
    B = 1 if reference else int(os.environ["PACKED_B"])
    for name, spec, chunks in compos:
        assert reference or len(chunks) == B, f"{name}: {len(chunks)} chunks, PACKED_B={B}"
    n_slots = max(s for _, _, chunks in compos for s, *_ in chunks) + 1
    need = max(h + SEG for _, _, chunks in compos for _, _, h, _ in chunks)
    capacity = max(int(os.getenv("PACKED_CAPACITY", "0")), need)
    capacity = -(-capacity // SEG) * SEG
    warmup = int(os.getenv("PACKED_WARMUP", "2"))
    iters = int(os.getenv("PACKED_ITERS", "5"))
    stages = int(os.getenv("PACKED_STAGES", "2"))
    stage = int(os.getenv("PACKED_STAGE", "0"))
    dump = os.getenv("PACKED_DUMP_KV")
    fill_mode = os.getenv("PACKED_FILL", "real")
    assert fill_mode in ("real", "none"), fill_mode

    src = json.load(open(os.environ["PACKED_TOKENS"]))["token_ids"]
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
        fill=fill_mode,
        warmup=warmup,
        iters=iters,
        stages=stages,
        stage=stage,
        fabric=str(fabric_config_from_env()),
    )

    ttnn.set_fabric_config(fabric_config_from_env())
    galaxy = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), l1_small_size=L1_SMALL_SIZE)
    runtime = None
    try:
        mesh = galaxy.create_submeshes(ttnn.MeshShape(8 // stages, 4))[stage] if stages > 1 else galaxy
        t0 = time.perf_counter()
        runtime, kv_cache = build(
            mesh,
            layer_ids=layer_ids,
            W=B * SEG,
            capacity=capacity,
            num_users=n_slots,
            stages=stages,
            stage=stage,
        )
        emit(kind="built", load_s=round(time.perf_counter() - t0, 1), mesh=list(mesh.shape))
        if PROFILE:
            from models.demos.minimax_m3.utils.profiler_utils import COARSE, read_profiler, zone

            # Drain the device profiler during the un-profiled forwards (fills, warm-ups, timed iterations) so its
            # buffer never overflows before the profiled forward, which must fit in it whole (profile_prefill.py,
            # TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT). A drain is a multi-second mesh read, so not per layer:
            # once per PACKED_PROFILE_READ_EVERY forwards (a packed 2-layer forward is a few hundred ops).
            state = {"in_profiled": False, "layers": 0}
            read_every = int(os.getenv("PACKED_PROFILE_READ_EVERY", "16")) * len(layer_ids)

            def on_layer_complete(layer_idx):
                if state["in_profiled"]:
                    return
                state["layers"] += 1
                if state["layers"] % read_every == 0:
                    read_profiler(mesh)

            runtime.set_on_layer_complete(on_layer_complete)

        def run(chunks):
            """chunks: [(slot, stream, h, n)] -> wall ms of the forward(s), synced at the end only."""
            t = time.perf_counter()
            if reference:
                for slot, stream, h, n in chunks:
                    out = runtime.prefill_chunk(
                        runtime.make_chunk_input(tok(stream, h), actual_start=h),
                        kv_cache,
                        slot_id=slot,
                        actual_start=h,
                        actual_end=h + n,
                    )
                    if out is not None:
                        out.deallocate(True)
            else:
                inp = runtime.make_forward_input(
                    [tok(stream, h) if n else [0] * SEG for _, stream, h, n in chunks], [h for _, _, h, _ in chunks]
                )
                out = runtime.prefill_forward(inp, kv_cache, [PrefillChunk(slot, h, h + n) for slot, _, h, n in chunks])
                if out is not None:
                    out.deallocate(True)
            ttnn.synchronize_device(mesh)
            return (time.perf_counter() - t) * 1e3

        depth = {}  # slot -> (stream, filled depth)

        def fill(slot, stream, h):
            filled_stream, filled = depth.get(slot, (stream, 0))
            d = filled if filled_stream == stream else 0
            assert h <= d or d % SEG == 0, f"slot {slot} ends mid-chunk at {d}: restart the slot after a ragged chunk"
            todo = [] if fill_mode == "none" else [(slot, stream, p, SEG) for p in range(d, h, SEG)]
            while todo:
                group, todo = todo[:B], todo[B:]
                group += [(0, 0, 0, 0)] * (B - len(group))  # empty chunks: no attention, no KV write
                emit(kind="fill", slot=slot, h=group[0][2], wall_ms=round(run(group), 3))
            depth[slot] = (stream, max(d, h))

        for ci, (name, spec, chunks) in enumerate(compos):
            seen = set()
            for slot, stream, h, n in chunks:
                if (slot, stream) not in seen:
                    seen.add((slot, stream))
                    fill(slot, stream, h)  # history before this slot's first chunk in the composition
            for k in range(max(warmup, 5) if ci == 0 else warmup):
                emit(kind="warmup", compo=name, iter=k, wall_ms=round(run(chunks), 3))
            walls = []
            for k in range(iters):
                walls.append(run(chunks))
                emit(kind="iter", compo=name, iter=k, wall_ms=round(walls[-1], 3))
            emit(
                kind="point",
                compo=name,
                spec=spec,
                chunks=[{"slot": s, "h": h, "n": n} for s, _, h, n in chunks],
                W=B * SEG,
                iters=iters,
                wall_ms_median=round(statistics.median(walls), 3),
                wall_ms_min=round(min(walls), 3),
                wall_ms_max=round(max(walls), 3),
            )
            for slot, stream, h, n in chunks:  # the composition wrote these positions: history now reaches them
                depth[slot] = (stream, max(depth.get(slot, (stream, 0))[1], h + n))

        if PROFILE:
            # One extra untimed forward of the last composition under the root zone the parser reports, into a
            # freshly drained buffer.
            read_profiler(mesh)
            state["in_profiled"] = True
            with zone("profiled_chunk", COARSE):
                run(compos[-1][2])
            state["in_profiled"] = False
            read_profiler(mesh)
            emit(kind="profiled", compo=compos[-1][0], spec=compos[-1][1])

        if dump:
            import torch

            os.makedirs(dump, exist_ok=True)
            for slot in range(n_slots):
                torch.save(runtime.read_slot_kv(kv_cache, slot), os.path.join(dump, f"slot{slot}.pt"))
            emit(kind="dumped", dir=dump, slots=n_slots)
        emit(kind="done")
    finally:
        # A sub-device manager left registered at close can segfault the teardown.
        if runtime is not None:
            runtime.release_sub_device_managers()
        for sub in galaxy.get_submeshes():
            ttnn.close_mesh_device(sub)
        ttnn.close_mesh_device(galaxy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
