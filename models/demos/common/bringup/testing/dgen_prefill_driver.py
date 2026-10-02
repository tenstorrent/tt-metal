# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt-d-gen's real engine driving a prefill runner: the scheduler side of serving, for the runner tests.

    <tt-d-gen python 3.12> dgen_prefill_driver.py <plan.json> <out.json>

Runs under tt-d-gen's interpreter with only the standard library and `tt_engine` (tt-d-gen bindings/python); it
must not import ttnn or anything from tt-metal (ttnn is built for another Python). dgen_engine.py finds the build and
starts this script; the runner (tt-metal) owns the H2D stream service and the layer-ack channel, this process
connects to both as tt-d-gen's PrefillPipeline does (engine/src/pipeline/prefill_pipeline.cpp):

    te.BackendRuntime(cfg, te.device_prefill_pipeline(service_id, "/tt_prefill_layer_acks_<service_id>",
                                                      chunk_size, layers_per_chunk, connect_timeout_ms, sp_factor))

plan.json:
    service_id, chunk_size, layers_per_chunk, sp_factor, max_slots, max_seq_len, kv_block_size,
    connect_timeout_ms, run_timeout_s, stall_s, mock (true: te.mock_prefill_pipeline, host only, for a dry run),
    requests: [{"name": str, "token_ids": [int, ...], "after": null | name}]
A request with "after" is admitted when that request's PREFILL_DONE arrives (a follow-up turn); the others at once,
in list order. The engine picks slots, chunks, prefix reuse and the interleave; this script only admits and pumps
(tick + drain) until every request has PREFILL_DONE, then stop() (drains the H2D stream, releases the ack channel).

out.json: {"ok", "errors", "requests": {name: {rid, slot, resident, reused_tokens, prompt_len, done_position,
admit_s, done_s}}, "telemetry": {...}}. Exit 0 only when every request reached PREFILL_DONE at its prompt length.
Exit 2 on REJECTED / ABORTED / a short PREFILL_DONE, 3 on no progress for stall_s, 4 on run_timeout_s (a hard
faulthandler bound: it fires even while a C++ call holds the GIL), 5 on a bad plan / build.
"""

from __future__ import annotations

import faulthandler
import json
import os
import sys
import time


def log(msg: str) -> None:
    print(f"[dgen driver] {msg}", flush=True)


def main(plan_path: str, out_path: str) -> int:
    plan = json.loads(open(plan_path).read())
    rec = {"ok": False, "errors": [], "requests": {}, "telemetry": {}, "path": "tt-d-gen engine"}

    def finish(code: int) -> int:
        rec["ok"] = code == 0 and not rec["errors"]
        with open(out_path + ".tmp", "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(out_path + ".tmp", out_path)
        return code

    run_timeout = float(plan.get("run_timeout_s", 3600))
    # Hard bound: dumps every thread's stack and exits even while BackendRuntime(...) / tick() / stop() block in C++.
    faulthandler.dump_traceback_later(run_timeout, exit=True)
    try:
        import tt_engine as te
    except Exception as e:  # a build without bindings
        rec["errors"].append(f"import tt_engine: {type(e).__name__}: {e}")
        return finish(5)
    rec["tt_engine"] = te.__file__

    cfg = te.RuntimeConfig()
    cfg.role = te.RuntimeRole.PREFILL
    cfg.max_slots = int(plan["max_slots"])
    cfg.max_seq_len = int(plan["max_seq_len"])
    cfg.chunk_size = int(plan["chunk_size"])
    cfg.kv_block_size = int(plan["kv_block_size"])
    if "evict_enabled" in plan:
        cfg.evict_enabled = bool(plan["evict_enabled"])
    svc = plan["service_id"]
    ack = plan.get("ack_shm_name") or f"/tt_prefill_layer_acks_{svc}"
    lpc = int(plan["layers_per_chunk"])
    if plan.get("mock"):
        pipe = te.mock_prefill_pipeline(lpc, auto_ack=True)
        log(f"mock pipeline (host only), layers_per_chunk {lpc}")
    else:
        pipe = te.device_prefill_pipeline(
            svc, ack, cfg.chunk_size, lpc, int(plan.get("connect_timeout_ms", 600000)), int(plan["sp_factor"])
        )
        log(
            f"device pipeline: service {svc!r}, acks {ack!r}, chunk {cfg.chunk_size}, layers/chunk {lpc}, "
            f"sp {plan['sp_factor']}"
        )
    log(
        f"runtime: PREFILL, max_slots {cfg.max_slots}, max_seq_len {cfg.max_seq_len}, kv_block_size "
        f"{cfg.kv_block_size}"
    )

    reqs = plan["requests"]
    by_name = {r["name"]: r for r in reqs}
    for r in reqs:
        if r.get("after") and r["after"] not in by_name:
            rec["errors"].append(f"request {r['name']}: after {r['after']!r} is not a request")
    if rec["errors"]:
        return finish(5)

    t0 = time.monotonic()
    try:
        rt = te.BackendRuntime(cfg, pipe)  # connects (H2D service, ack channel); bounded by connect_timeout_ms
    except Exception as e:
        rec["errors"].append(f"BackendRuntime: {type(e).__name__}: {e}")
        return finish(5)
    log(f"attached in {time.monotonic() - t0:.1f} s")
    rt.event_handle()  # required before any admit
    rid_of, name_of, done = {}, {}, set()
    code = 0

    def admit(r) -> None:
        req = te.InferenceRequest()
        req.token_ids = [int(t) for t in r["token_ids"]]
        rid = rt.admit(req)
        rid_of[r["name"]], name_of[rid] = rid, r["name"]
        rec["requests"][r["name"]] = {
            "rid": rid,
            "prompt_len": len(r["token_ids"]),
            "admit_s": round(time.monotonic() - t0, 2),
        }
        log(f"admit {r['name']}: {len(r['token_ids'])} tokens -> rid {rid}")

    try:
        for r in reqs:
            if not r.get("after"):
                admit(r)
        stall = float(plan.get("stall_s", 600))
        last = time.monotonic()
        while len(done) < len(reqs):
            worked = rt.tick()
            evs = rt.drain()
            now = time.monotonic()
            if worked or evs:
                last = now
            for ev in evs:
                name = name_of.get(ev.request_id, f"rid {ev.request_id}")
                info = rec["requests"].setdefault(name, {})
                if ev.kind == te.EventKind.ADMITTED:
                    info.update(slot=ev.slot_id, resident=ev.position_id, reused_tokens=ev.reused_tokens)
                    log(f"ADMITTED {name}: slot {ev.slot_id}, resident {ev.position_id}, reused {ev.reused_tokens}")
                elif ev.kind == te.EventKind.PREFILL_DONE:
                    info.update(done_position=ev.position_id, done_s=round(now - t0, 2))
                    done.add(name)
                    log(
                        f"PREFILL_DONE {name}: slot {ev.slot_id}, KV resident to {ev.position_id} "
                        f"({now - t0:.1f} s)"
                    )
                    if ev.position_id != info.get("prompt_len"):
                        rec["errors"].append(
                            f"{name}: PREFILL_DONE at {ev.position_id}, prompt {info.get('prompt_len')}"
                        )
                        code = 2
                    for r in reqs:
                        if r.get("after") == name:
                            admit(r)
                elif ev.kind in (te.EventKind.REJECTED, te.EventKind.ABORTED):
                    rec["errors"].append(f"{name}: {ev!r}")
                    log(f"{ev!r} for {name}")
                    code = 2
            if code:
                break
            if now - last > stall:
                rec["errors"].append(f"no engine progress for {stall:.0f} s; done {sorted(done)} of {len(reqs)}")
                code = 3
                break
            if not worked and not evs:
                time.sleep(0.001)
        tel = rt.telemetry_snapshot()
        for k in (
            "admitted",
            "completed",
            "reuse_hits",
            "reused_tokens",
            "prompt_tokens",
            "prefill_chunk_tokens_total",
            "prefill_prompt_tokens_prefilled_total",
            "aborted",
        ):
            rec["telemetry"][k] = int(getattr(tel, k))
        log(f"telemetry {rec['telemetry']}")
    except Exception as e:
        rec["errors"].append(f"{type(e).__name__}: {e}")
        code = code or 2
    finally:
        log("stop: draining the H2D stream, releasing the ack channel")
        try:
            rt.stop()
        except Exception as e:
            rec["errors"].append(f"stop: {type(e).__name__}: {e}")
            code = code or 2
    log(f"{'ok' if code == 0 and not rec['errors'] else 'FAILED'} in {time.monotonic() - t0:.1f} s")
    return finish(code)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(5)
    sys.exit(main(sys.argv[1], sys.argv[2]))
