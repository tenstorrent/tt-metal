# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One prefill chunk through the first MIMO_PERF_N_LAYERS layers as one model call (no sync between layers), real
weights and real-token input: eager (host enqueue time of the call, wall to device idle) and traced replay (the whole
chunk captured once). Unlike test_layer_perf.py, the host enqueue of layer i overlaps layer i - 1 on device, so the
eager wall shows how much host time actually reaches the chunk latency.

    MIMO_PERF_N_LAYERS (6), MIMO_PERF_CHUNK (4096), MIMO_PERF_CTX (57344), MIMO_PERF_KV_ACTUAL (CTX - CHUNK),
    MIMO_PERF_ITERS (3), MIMO_PERF_TRACE (0: off, else replays), MIMO_TRACE_REGION
"""

import os
import time

import pytest
from loguru import logger

import ttnn
from models.demos.mimo_v2_d_p.reference import hf
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.reference.weights import global_state, layer_state
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS, mesh_id
from models.demos.mimo_v2_d_p.tt.model import TtMiMoModel, block_cyclic_index
from models.demos.mimo_v2_d_p.tt.options import MiMoRuntimeOptions

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None

N_LAYERS = int(os.environ.get("MIMO_PERF_N_LAYERS", "6"))
CHUNK = int(os.environ.get("MIMO_PERF_CHUNK", "4096"))
CTX = int(os.environ.get("MIMO_PERF_CTX", "57344"))
KV_ACTUAL = int(os.environ.get("MIMO_PERF_KV_ACTUAL", str(CTX - CHUNK)))
ITERS = int(os.environ.get("MIMO_PERF_ITERS", "3"))
N_TRACE = int(os.environ.get("MIMO_PERF_TRACE", "0"))


def _report(msg):
    logger.info(msg)
    if os.environ.get("MIMO_PERF_OUT"):
        with open(os.environ["MIMO_PERF_OUT"], "a") as f:
            f.write(msg + "\n")


@pytest.mark.timeout(14400)
@MESH_PARAMS
def test_model_chunk_perf(mesh_device, device_params):
    cfg = MiMoTextConfig.from_json()
    model = TtMiMoModel(
        mesh_device,
        cfg,
        lambda i: layer_state(i, cfg),
        fabric_config=device_params["fabric_config"],
        max_seq_len=CTX,
        chunk_size=CHUNK,
        layers=list(range(N_LAYERS)),
        global_state=global_state,
        options=MiMoRuntimeOptions.from_env(),
    )
    ids = hf.tokenize_prompt(CHUNK)
    tag = f"{mesh_id(mesh_device)} L{N_LAYERS} chunk {CHUNK} @ {KV_ACTUAL}"

    host, wall = [], []
    for it in range(1 + ITERS):  # the first call compiles
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"chunk_L{N_LAYERS}_start")
        t0 = time.perf_counter()
        out = model.prefill_chunk(ids, KV_ACTUAL)
        host.append((time.perf_counter() - t0) * 1e3)
        ttnn.synchronize_device(mesh_device)
        wall.append((time.perf_counter() - t0) * 1e3)
        if it:
            signpost(f"chunk_L{N_LAYERS}_end")
        out.deallocate(True)
    h, w = sorted(host[1:]), sorted(wall[1:])
    _report(
        f"EAGER {tag}: wall median {w[len(w) // 2]:.2f} ms, host enqueue median {h[len(h) // 2]:.2f} ms "
        f"(per layer {h[len(h) // 2] / N_LAYERS:.2f}); all wall {[round(v, 1) for v in wall]}"
    )

    if os.environ.get("MIMO_PERF_SPLIT"):  # where the eager wall above the traced replay goes
        idx = block_cyclic_index(KV_ACTUAL, model.sp, model.chunk_local) - KV_ACTUAL
        split = []
        for _ in range(1 + ITERS):
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            x = model.embed_device(model.tokens_to_device(ids[idx]))
            ttnn.synchronize_device(mesh_device)
            t1 = time.perf_counter()
            out = model.forward_device(x, KV_ACTUAL)
            t2 = time.perf_counter()
            ttnn.synchronize_device(mesh_device)
            t3 = time.perf_counter()
            out.deallocate(True)
            split.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t1) * 1e3))
        e, h, w = (sorted(v[k] for v in split[1:])[ITERS // 2] for k in range(3))
        _report(f"SPLIT {tag}: embed (synced) {e:.2f} ms | layers: host enqueue {h:.2f} ms, wall {w:.2f} ms")
        from models.demos.mimo_v2_d_p.tt.model import clamp_pad_tokens

        steps = {
            "H2D": lambda: model.tokens_to_device(ids[idx]),
            "H2D+clamp": lambda: clamp_pad_tokens(model.tokens_to_device(ids[idx]), model.vocab),
            "H2D+embed": lambda: model.embed_device(model.tokens_to_device(ids[idx])),
        }
        parts = []
        for name, fn in steps.items():
            ts = []
            for _ in range(1 + ITERS):
                ttnn.synchronize_device(mesh_device)
                t0 = time.perf_counter()
                o = fn()
                t1 = time.perf_counter()
                ttnn.synchronize_device(mesh_device)
                ts.append(((t1 - t0) * 1e3, (time.perf_counter() - t0) * 1e3))
                o.deallocate(True)
            hh, ww = (sorted(v[k] for v in ts[1:])[ITERS // 2] for k in range(2))
            parts.append(f"{name} host {hh:.2f} / synced {ww:.2f}")
        _report(f"SPLIT {tag}: " + " | ".join(parts) + " ms")

    if N_TRACE:
        idx = block_cyclic_index(KV_ACTUAL, model.sp, model.chunk_local) - KV_ACTUAL
        x_in = model.embed_device(model.tokens_to_device(ids[idx]))
        model.forward_device(ttnn.clone(x_in), KV_ACTUAL).deallocate(
            True
        )  # compile clone (no program builds in a trace)
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = model.forward_device(ttnn.clone(x_in), KV_ACTUAL)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        synced = []
        for _ in range(N_TRACE):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            synced.append((time.perf_counter() - t0) * 1e3)
        s = sorted(synced)
        _report(f"TRACE {tag}: replay median {s[len(s) // 2]:.2f} ms (min {s[0]:.2f}) over {N_TRACE}")
        ttnn.release_trace(mesh_device, tid)
        out.deallocate(True)
        x_in.deallocate(True)
