# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH decode-step timing of the TP=2 SDPA-decode width pin at long context (review F1 of FASTSLOT_REVIEW.md).

Full model (PIN_LAYERS, default 64), max_batch_size 32, a small paged KV pool whose blocks every row ALIASES (timing
only: 16 rows at 64k context would need 1M KV tokens, more than the served 622,592-token pool). Traced decode at width
PIN_WIDTH (default 16, every row active) is captured once with QWEN36_DECODE_SDPA_PIN_MIN_WIDTH=16 (pin on) and once
with 0 (off), then replayed alternately at each context in PIN_CTXS (default 8192,32768,65000); every replay is timed to
completion (blocking). Reports the per-step device time median / mean per (context, pin).

    pytest -svq models/demos/blackhole/qwen36/tests/test_sdpa_pin_long_ctx_tp2_scratch.py
"""

import json
import os
import statistics as st
import time

import pytest
import torch
from loguru import logger

import ttnn

BLOCK = 64


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576, "trace_region_size": 1073741824}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2")], indirect=True)
def test_sdpa_pin_long_ctx(mesh_device, reset_seeds, ensure_gc):
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model
    from models.tt_transformers.tt.common import copy_host_to_device

    n_layers = int(os.environ.get("PIN_LAYERS", "64"))
    w = int(os.environ.get("PIN_WIDTH", "16"))
    ctxs = [int(c) for c in os.environ.get("PIN_CTXS", "8192,32768,65000").split(",")]
    reps, steps = int(os.environ.get("PIN_REPS", "3")), int(os.environ.get("PIN_STEPS", "20"))
    out_path = os.environ.get("PIN_OUT", "sdpa_pin.json")
    B = 32
    n_blk = -(-(max(ctxs) + 64) // BLOCK)
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=65536 + 1024, n_layers=n_layers)
    args = model.args
    model.allocate_kv_caches((n_blk + 1, args.n_local_kv_heads, BLOCK, args.head_dim), ttnn.bfloat8_b, batch_size=B)
    pt = torch.arange(n_blk, dtype=torch.int32).reshape(1, -1).expand(w, -1).contiguous()  # every row aliases
    model.sync_gdn_decode_state()
    tokens = torch.full((w, 1), 1000, dtype=torch.int32)
    pins = {"on": "16", "off": "0"}
    for name, v in pins.items():  # compile both program sets before any trace is parked
        os.environ["QWEN36_DECODE_SDPA_PIN_MIN_WIDTH"] = v
        dev0 = model.prepare_inputs_decode(tokens, torch.full((w,), ctxs[0], dtype=torch.int32), page_table=pt)
        model.ttnn_decode_forward(dev0[0], dev0[1], rot_mat_idxs=dev0[2], page_table=dev0[3])
        ttnn.synchronize_device(mesh_device)
    traces = {}
    for name, v in pins.items():
        os.environ["QWEN36_DECODE_SDPA_PIN_MIN_WIDTH"] = v
        host = model.prepare_decode_inputs_host(tokens, torch.full((w,), ctxs[0], dtype=torch.int32), page_table=pt)
        dev = copy_host_to_device(host, mesh_device=mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        traces[name] = (tid, dev)
    os.environ.pop("QWEN36_DECODE_SDPA_PIN_MIN_WIDTH")
    res = {}
    for ctx in ctxs:
        for rep in range(reps):
            for name in pins:
                tid, dev = traces[name]
                ts = []
                for s in range(steps + 2):
                    host = model.prepare_decode_inputs_host(
                        tokens, torch.full((w,), ctx + s, dtype=torch.int32), page_table=pt
                    )
                    copy_host_to_device(host, device_tensors=dev)
                    ttnn.synchronize_device(mesh_device)
                    t0 = time.perf_counter()
                    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
                    ttnn.synchronize_device(mesh_device)
                    ts.append(1e3 * (time.perf_counter() - t0))
                res.setdefault(f"{ctx}/{name}", []).extend(ts[2:])
        for name in pins:
            v = res[f"{ctx}/{name}"]
            logger.info(
                f"[sdpa_pin] ctx {ctx} width {w} pin {name}: step ms median {st.median(v):.2f} mean {st.mean(v):.2f}"
            )
    for tid, _ in traces.values():
        ttnn.release_trace(mesh_device, tid)
    summary = {k: {"median": st.median(v), "mean": st.mean(v), "n": len(v)} for k, v in res.items()}
    with open(out_path, "w") as f:
        json.dump({"width": w, "n_layers": n_layers, "summary": summary, "raw": res}, f, indent=1)
