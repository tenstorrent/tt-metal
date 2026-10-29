# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH (lane C): TP=1 decode profile of Qwen3.8-27B on ONE Blackhole die (mesh (1,1)).

Mirrors the served decode node's geometry without vLLM: the model is built at max_batch_size=PROF_BMAX (the node's
--max-num-seqs, sizes the per-slot GDN state + KV-update grid), the step runs at bucket width PROF_B (1 or BMAX), the
paged KV pool is bf8, the page table is PROF_MAX_SEQ/64 wide (1024 at 64k) and every step re-stages all four host inputs
(tokens, positions, rope, page table) exactly like Generator._decode_forward_trace_text does under host sampling.
No prefill: KV / GDN state are zeros and the positions are set to PROF_CTX so the paged SDPA reads a PROF_CTX-token KV
per user (timings do not depend on the values).

Phases: (1) PROF_STEPS+1 eager steps (the first compiles; each op gets a device record under tracy);
        (2) unless PROF_NO_TRACE=1: capture the decode trace, time PROF_REPS non-blocking replays (device step);
        (3) PROF_HOST_LOOP serving-like steps: host-prepare -> copy_host_to_device -> execute_trace -> .cpu() ->
            process_output_decode -> argmax, each phase timed (the model-side host glue per step).

  per-op (device profiler; no trace):  PROF_NO_TRACE=1 python -m tracy -r -v --op-support-count 30000 -o <dir> <this>
  timing (no profiler):                python <this>
Results are printed as PROF_RESULT json lines and appended to $PROF_OUT when set.
"""

import json
import os
import statistics
import time

import torch
from loguru import logger

import ttnn


def env_int(k, d):
    return int(os.environ.get(k, d))


B = env_int("PROF_B", 1)
BMAX = env_int("PROF_BMAX", 8)
CTX = env_int("PROF_CTX", 128)
STEPS = env_int("PROF_STEPS", 3)
REPS = env_int("PROF_REPS", 20)
HOST_LOOP = env_int("PROF_HOST_LOOP", 20)
LAYERS = os.environ.get("PROF_LAYERS", "all")
NO_TRACE = os.environ.get("PROF_NO_TRACE", "0") == "1"
MAX_SEQ = env_int("PROF_MAX_SEQ", 65536)
TRACE_REGION = env_int("PROF_TRACE_REGION", 536870912)
BLOCK = 64
PT_WIDTH = MAX_SEQ // BLOCK
OUT = os.environ.get("PROF_OUT", "")
TAG = os.environ.get("PROF_TAG", "")


def emit(kind, **kw):
    rec = dict(
        kind=kind, tag=TAG, B=B, BMAX=BMAX, CTX=CTX, layers=LAYERS, gdn_fused=os.environ.get("QWEN36_GDN_DECODE_FUSED")
    )
    rec.update(kw)
    line = "PROF_RESULT " + json.dumps(rec)
    print(line, flush=True)
    if OUT:
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")


def main():
    try:
        from tracy import signpost
    except Exception:  # plain run

        def signpost(*a, **k):
            pass

    from models.demos.blackhole.qwen36.tt.common import create_tt_model
    from models.tt_transformers.tt.common import copy_host_to_device

    assert 1 <= B <= BMAX
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=TRACE_REGION)
    mesh.enable_program_cache()
    try:
        run(mesh, signpost, create_tt_model, copy_host_to_device)
    finally:
        ttnn.synchronize_device(mesh)
        ttnn.close_mesh_device(mesh)


def run(mesh, signpost, create_tt_model, copy_host_to_device):
    layer_indices = None if LAYERS == "all" else [int(x) for x in LAYERS.split(",")]
    t0 = time.time()
    args, model, _ = create_tt_model(
        mesh, max_batch_size=BMAX, max_seq_len=MAX_SEQ, layer_indices=layer_indices, hf_model=os.environ.get("HF_MODEL")
    )
    logger.info(
        f"[PROF] model load {time.time() - t0:.1f}s layers={len(model.layers)} tp1={args.tp1} use_tp={model.use_tp}"
    )
    nb_user = (CTX + 1 + BLOCK - 1) // BLOCK
    num_blocks = max(
        env_int("PROF_POOL_BLOCKS", 2560), BMAX * nb_user
    )  # served pool 163840 tok = 2560 blocks (>= PT_WIDTH: paged_update_cache requires it)
    shape = [num_blocks + 1, args.n_local_kv_heads, BLOCK, args.head_dim]
    model.allocate_kv_caches(shape, ttnn.bfloat8_b, batch_size=BMAX)
    logger.info(f"[PROF] KV pool {shape} bf8; GDN slots B={BMAX}")

    page_table = torch.zeros(B, PT_WIDTH, dtype=torch.int32)
    for u in range(B):
        page_table[u, :nb_user] = torch.arange(u * nb_user, (u + 1) * nb_user, dtype=torch.int32)
    torch.manual_seed(0)
    tokens = torch.randint(1000, 20000, (B, 1), dtype=torch.int32)
    pos = torch.full((B,), CTX, dtype=torch.int32)

    dev = list(model.prepare_inputs_decode(tokens, pos, page_table=page_table))
    logger.info(
        f"[PROF] dev inputs: tokens {dev[0].shape} pos {dev[1].shape} rope {dev[2].shape} pt {dev[3].shape} ({dev[3].dtype})"
    )

    def fwd():
        return model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])[0]

    def stage(tok, p):
        host = model.prepare_decode_inputs_host(tok, p, page_table=page_table)
        copy_host_to_device(host, device_tensors=dev)

    # (1) eager steps
    eager = []
    toks = []
    for i in range(STEPS + 1):
        signpost(f"decode_eager_{i}")
        ttnn.synchronize_device(mesh)
        t1 = time.time()
        out = fwd()
        ttnn.synchronize_device(mesh)
        eager.append(time.time() - t1)
        if i == 0:
            logger.info(
                f"[PROF] logits {out.shape} padded {out.padded_shape} {out.dtype} {out.layout} {out.memory_config()}"
            )
        lg = model.process_output_decode(out, B=BMAX, S=1)
        ttnn.deallocate(out)
        nxt = torch.argmax(lg[:B, 0, :], dim=-1).to(torch.int32).reshape(B, 1)
        toks.append(nxt[:, 0].tolist())
        pos = pos + 1
        stage(nxt, pos)
    signpost("decode_eager_end")
    logger.info(f"[PROF] eager step wall ms: {['%.1f' % (t * 1e3) for t in eager]} tokens {toks}")
    emit("eager", eager_ms=[round(t * 1e3, 2) for t in eager], eager_min_ms=round(min(eager[1:]) * 1e3, 2))
    if NO_TRACE:
        return

    # (2) traced device step
    model.sync_gdn_decode_state()
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    tt_logits = fwd()
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh)
    signpost("decode_trace")
    for _ in range(3):  # warm
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t1 = time.time()
    for _ in range(REPS):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    per = (time.time() - t1) / REPS
    singles = []
    for _ in range(10):
        t2 = time.time()
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        singles.append(time.time() - t2)
    signpost("decode_trace_end")
    logger.info(
        f"[PROF] traced step {per * 1e3:.2f} ms (pipelined x{REPS}); blocking single {min(singles) * 1e3:.2f} ms"
    )
    emit(
        "trace",
        traced_step_ms=round(per * 1e3, 3),
        traced_single_min_ms=round(min(singles) * 1e3, 3),
        traced_single_med_ms=round(statistics.median(singles) * 1e3, 3),
    )

    # readback alone (device idle): padded tile readback + host untilize/convert
    ttnn.synchronize_device(mesh)
    rb, cv = [], []
    for _ in range(10):
        t2 = time.time()
        h = tt_logits.cpu()
        t3 = time.time()
        lg = model.process_output_decode(h, B=BMAX, S=1)
        t4 = time.time()
        rb.append(t3 - t2)
        cv.append(t4 - t3)
    nbytes = 1
    for d in tt_logits.padded_shape:
        nbytes *= int(d)
    nbytes *= 2
    logger.info(
        f"[PROF] readback .cpu() {min(rb) * 1e3:.2f} ms ({nbytes / 1e6:.1f} MB padded), to_torch+float {min(cv) * 1e3:.2f} ms"
    )
    emit(
        "readback",
        cpu_min_ms=round(min(rb) * 1e3, 3),
        cpu_med_ms=round(statistics.median(rb) * 1e3, 3),
        convert_min_ms=round(min(cv) * 1e3, 3),
        convert_med_ms=round(statistics.median(cv) * 1e3, 3),
        padded_bytes=nbytes,
    )

    # (3) serving-like host loop
    ph = {k: [] for k in ("prepare", "copy", "enqueue", "wait_read", "convert", "argmax", "total")}
    tok = torch.tensor(toks[-1], dtype=torch.int32).reshape(B, 1)
    for _ in range(HOST_LOOP):
        pos = pos + 1
        t0 = time.perf_counter()
        host = model.prepare_decode_inputs_host(tok, pos, page_table=page_table)
        t1 = time.perf_counter()
        copy_host_to_device(host, device_tensors=dev)
        t2 = time.perf_counter()
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        t3 = time.perf_counter()
        h = tt_logits.cpu()
        t4 = time.perf_counter()
        lg = model.process_output_decode(h, B=BMAX, S=1)
        t5 = time.perf_counter()
        tok = torch.argmax(lg[:B, 0, :], dim=-1).to(torch.int32).reshape(B, 1)
        t6 = time.perf_counter()
        for k, v in zip(ph, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t6 - t0)):
            ph[k].append(v)
    summary = {k: round(statistics.median(v) * 1e3, 3) for k, v in ph.items()}
    logger.info(f"[PROF] host loop medians ms: {summary}")
    emit("host_loop", **{f"{k}_med_ms": v for k, v in summary.items()}, total_min_ms=round(min(ph["total"]) * 1e3, 3))
    ttnn.release_trace(mesh, trace_id)


if __name__ == "__main__":
    main()
