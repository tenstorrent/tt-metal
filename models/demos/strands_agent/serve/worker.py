# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device process of the server: opens the chip, warms every shape, then answers batches."""

import signal
import time
import traceback

from models.demos.strands_agent.serve.batcher import collect_batch, run_batch


def _open_mesh(device_params):
    """Open a 1x1 mesh like the pytest mesh_device fixture does (no fabric)."""
    import ttnn
    from tests.scripts.common import get_updated_device_params

    params = get_updated_device_params(dict(device_params))
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), **params)


def _close_mesh(mesh):
    import ttnn

    for submesh in mesh.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh)


def eager_lengths(max_seq_len, traced_s):
    """Prompt lengths that hit each eager bucket (above 2048 and not traced), capped at max_seq_len."""
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    buckets = sorted({Qwen36Model._mask_bucket_for(n) for n in range(2049, max_seq_len + 1)})
    return [(b, min(b, max_seq_len)) for b in buckets if b not in traced_s]


def traced_shapes(decider, max_seq_len, traced_s, trace_batches):
    return [
        (B, S)
        for S in traced_s
        if S <= max_seq_len
        for B in trace_batches
        if B * S <= decider.max_batch_tokens and B <= decider.max_batch
    ]


def warmup(decider, cfg, log):
    from models.demos.strands_agent.tt.model import TRACE_BATCHES, TRACED_S

    max_seq_len = cfg["max_seq_len"]

    def run(B, S):
        decider.hidden_rows_batch([[1] * S] * B, [[S - 1]] * B)

    if cfg["warmup_eager"] and not cfg["traced_buckets_only"]:
        for bucket, length in eager_lengths(max_seq_len, TRACED_S):
            t = time.perf_counter()
            for B in range(1, min(decider.max_batch, decider.max_batch_tokens // bucket) + 1):
                run(B, length)
            log(f"warmup eager bucket {bucket}: {time.perf_counter() - t:.1f} s")
    shapes = traced_shapes(decider, max_seq_len, TRACED_S, TRACE_BATCHES)
    for n_pass in range(2 if cfg["trace"] else 1):
        t = time.perf_counter()
        for B, S in shapes:
            run(B, S)
        log(f"warmup traced pass {n_pass + 1}: {len(shapes)} shapes, {time.perf_counter() - t:.1f} s")


def worker_main(cfg, task_q, result_q):
    """Entry of the spawned device process; cfg keys: max_seq_len, trace, traced_buckets_only, warmup_eager,
    batch_wait_ms, max_batch_questions, verbose."""
    # The server owns shutdown (Ctrl-C reaches the whole process group).
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def log(msg):
        print(f"[worker] {msg}", flush=True)

    mesh = decider = None
    try:
        from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE
        from models.demos.strands_agent.tests.common import TRACE_REGION_SIZE
        from models.demos.strands_agent.tt.model import StrandsDecider

        t_start = time.perf_counter()
        mesh = _open_mesh({"l1_small_size": GDN_CONV1D_L1_SMALL_SIZE, "trace_region_size": TRACE_REGION_SIZE})
        decider = StrandsDecider.from_pretrained(
            mesh, max_seq_len=cfg["max_seq_len"], trace=cfg["trace"], traced_buckets_only=cfg["traced_buckets_only"]
        )
        log(f"model built in {time.perf_counter() - t_start:.1f} s")
        warmup(decider, cfg, log)
        warmup_s = time.perf_counter() - t_start
        max_length = min(decider.config.max_length, decider.max_seq_len)
        result_q.put(
            {
                "type": "ready",
                "warmup_s": round(warmup_s, 1),
                "trace_region_use": {f"{k}": v for k, v in getattr(decider, "trace_region_use", {}).items()},
                "max_length": max_length,
                "max_seq_len": decider.max_seq_len,
            }
        )
        log(f"ready after {warmup_s:.1f} s")

        def get(timeout):
            if timeout is None:
                return task_q.get()
            if timeout == 0:
                return task_q.get_nowait()
            return task_q.get(timeout=timeout)

        batch_id = 0
        while True:
            items, stop = collect_batch(get, cfg["batch_wait_ms"] / 1000, cfg["max_batch_questions"])
            if items:
                batch_id += 1
                results = run_batch(decider, items, batch_id)
                for r in results:
                    result_q.put({"type": "result", **r})
                if cfg["verbose"]:
                    ok = [r for r in results if r["ok"]]
                    timing = ok[0]["timing"] if ok else {}
                    log(f"batch requests={len(items)} ok={len(ok)} {timing}")
            if stop:
                break
    except BaseException:
        result_q.put({"type": "failed", "error": traceback.format_exc()})
    finally:
        try:
            if decider is not None:
                decider.close()
        finally:
            if mesh is not None:
                _close_mesh(mesh)
            result_q.close()
            result_q.join_thread()
