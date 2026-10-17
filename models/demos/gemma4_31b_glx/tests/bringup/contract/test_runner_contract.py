# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""serving_contract.md, "Adapter and table": the real prefill runner under tt-d-gen's engine (the hackathon's K.2).

The pytest process opens no mesh (the runner subprocess owns the device), so it gates on impl_mode() instead of the
mesh fixture: under BRINGUP_IMPL=reference / stub it skips. On the device:

  runner   `python -m <this module> <out>`: models/demos/common/prefill/runners/prefill_runner.py main() with
           PREFILL_MODEL=gemma4_31b_glx, SP 8 x TP 4, 60 layers, chunk 5120, max_seq 56320, 2 slots,
           device-to-host layer acks (PREFILL_LAYER_ACK_D2H=1), fabric 1d_ring, no trace, mock migration
           (PREFILL_ENABLE_MIGRATION=0: the table is written for an offline reader; tt-metal does not read
           PREFILL_MOCK_MIGRATION, it is set for the record). run_request_loop is wrapped to start the feeder once the
           H2D service and the ack channel are up.
  feeder   bringup_fw.testing.dgen_engine.feeder(): tt-d-gen's engine (dgen_prefill_driver.py on the runner's H2D service
           "/tt_prefill_layer_acks_<service>" channel, layers_per_chunk 60) with four golden-prefix requests: a (8017
           tokens) and b (7001) at once, a2 (56000) after a, b2 (9000) after b. The engine picks slots, prefix reuse,
           the interleave and the pulled-back last chunk (prefill_writer.cpp #430). Else (the run allows it)
           tt-metal's prefill_producer. Which one ran is printed.
  checks   the driver exits 0 (every request PREFILL_DONE at its prompt length: the acks retired every chunk); a
           follow-up reused a prefix and a last chunk was pulled back; the exported table against tt-d-gen's
           tables.py profile and the layout rules per layer type (kv_table.layout_failures); KV of each slot read
           through the table (kv_table.TableReader, UMD) scored with tt-d-gen's kv_dump_compare.py against the golden,
           PCC >= max(0.93, thresholds.state), pad rows zero; the records below a follow-up's start byte-identical
           across it (bytecmp); layer 0's rewrite of a pulled-back overlap bit-identical.
"""

import os
import sys

import pytest
from bringup_fw.testing.harness import impl_mode, spec

S = spec()
MODULE = "models.demos.gemma4_31b_glx.tests.bringup.contract.test_runner_contract"
REQUESTS = (("a", 8017, None), ("b", 7001, None), ("a2", 56000, "a"), ("b2", 9000, "b"))
SAMPLE_BELOW = 64  # records below a follow-up's start re-read across it (the boundary is where a bad write lands)


def runner_env(s, out_dir) -> dict:
    from bringup_fw.testing import contract

    from models.demos.gemma4_31b_glx.tests.bringup.contract import serving_helpers as H

    chunk, max_seq, sp = H.geometry(s)
    env = dict(os.environ)
    env.update(contract.engine_env(s))
    env.update(
        PREFILL_SP=str(sp),
        PREFILL_TP=str(s.mesh[1]),
        PREFILL_CHUNK_SIZE=str(chunk),
        PREFILL_MAX_SEQ_LEN=str(max_seq),
        PREFILL_NUM_LAYERS=str(contract.served_layers(s)[1]),
        PREFILL_NUM_USERS=str(H.slots(s)),
        PREFILL_H2D_SERVICE_ID=f"gemma31_contract_{os.getpid()}",
        PREFILL_LAYER_ACK_D2H="1",
        PREFILL_USE_TRACE="0",
        PREFILL_FABRIC_MODE="1d_ring",
        PREFILL_ENABLE_MIGRATION="0",
        PREFILL_MOCK_MIGRATION="1",
        PREFILL_MIGRATION_EXPORT_TO_FILE="0",
        PREFILL_MIGRATION_TABLE_PATH=str(out_dir / "kv_table.pb"),
        PREFILL_MIGRATION_DEVICE_MAP_PATH=str(out_dir / "device_map.json"),
        # run_safe_pytest's 5 s op timeout would kill the runner while it waits for the next chunk (known_issues.md)
        TT_METAL_OPERATION_TIMEOUT_SECONDS="600",
    )
    return env


@pytest.mark.timeout(int(S.get("box.test_timeout_s") or 3600))
def test_runner_contract(tmp_path):
    if impl_mode() != "device":
        pytest.skip("CPU proof mode: the runner needs the device")
    import json
    import subprocess

    from bringup_fw.core import metrics
    from bringup_fw.testing import dgen_engine

    from models.demos.gemma4_31b_glx.tests.bringup.contract import serving_helpers as H

    try:
        from models.demos.common.prefill.adapter import get_adapter

        get_adapter(os.environ.get("PREFILL_MODEL") or S.get("contract.adapter") or S.model)
    except (KeyError, ImportError) as e:
        H.not_built(f"no prefill adapter registered for {S.model} ({type(e).__name__}: {e})")
    kind, _, why = dgen_engine.feeder(S)
    if kind is None:
        pytest.fail(f"no feeder: {why}", pytrace=False)
    out = tmp_path
    env = runner_env(S, out)
    budget = int(S.get("box.test_timeout_s") or 3600) - 300
    with (out / "runner.log").open("w") as log:
        try:
            r = subprocess.run(
                [sys.executable, "-m", MODULE, str(out)], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=budget
            )
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = None
    tail = "".join((out / "runner.log").read_text().splitlines(keepends=True)[-40:])
    res_path = out / "result.json"
    if not res_path.is_file():
        pytest.fail(f"runner exited {rc} without a result (log tail):\n{tail}", pytrace=False)
    res = json.loads(res_path.read_text())
    print(f"feeder: {res.get('feeder')}")
    fails = list(res.get("fails", []))
    if rc != 0:
        fails.append(f"runner exited {rc}")
    if (out / "kv_table.pb").is_file():
        from bringup_fw.testing import contract

        import ttnn

        table = ttnn.experimental.disaggregation.import_from_protobuf_file(str(out / "kv_table.pb"))
        tables_mod, _ = H.dgen_checks()
        _, max_seq, _ = H.geometry(S)
        layout = H.table_failures(
            table, contract.served_layers(S)[1], H.slots(S), max_seq, tables_mod, H.DUMP_DTYPE[H.kv_dtype_name(S)]
        )
        metrics.record("runner_layout_failures", len(layout))
        fails += [f"table: {f}" for f in layout]
    else:
        fails.append("the runner exported no KV table")
    for k, v in (res.get("metrics") or {}).items():
        metrics.record(k, v)
    metrics.record("runner_checks_failed", len(fails))
    for f in fails:
        print(f"FAIL runner_contract: {f}")
    assert not fails, f"{len(fails)} runner checks failed: {fails[:4]}\n{tail if rc else ''}"


# ---------------------------------------------------------------- the runner subprocess
def _main(out_dir):
    import json
    import subprocess
    import threading
    from pathlib import Path
    from unittest.mock import patch

    from bringup_fw.testing import contract, dgen_engine, kv_table
    from bringup_fw.testing import server_rules as R
    from bringup_fw.testing.chunk_starts import kv_threshold

    import ttnn
    from models.demos.common.prefill.runners import prefill_runner
    from models.demos.gemma4_31b_glx.tests.bringup.contract import serving_helpers as H

    out = Path(out_dir)
    s = S
    chunk, max_seq, sp = H.geometry(s)
    service = os.environ["PREFILL_H2D_SERVICE_ID"]
    g = H.target_golden(s)
    tokens = g.tokens().tolist()
    kind, build, why = dgen_engine.feeder(s)
    result = {"feeder": kind if kind != dgen_engine.ENGINE else build.describe(), "fails": [], "metrics": {}}
    fails, mets = result["fails"], result["metrics"]
    print(f"[contract] feeder: {result['feeder']} {why}", flush=True)
    original_loop = prefill_runner.run_request_loop
    rewrites = []  # (what, layer, config, slot, start, end, bytes before, bytes after)

    def save():
        (out / "result.json").write_text(json.dumps(result, indent=1))

    def checked_loop(runtime, kv_caches, *args, **kwargs):
        mesh = runtime.mesh_device
        table = ttnn.experimental.disaggregation.import_from_protobuf_file(os.environ["PREFILL_MIGRATION_TABLE_PATH"])
        last_end = {}
        inner = runtime.prefill_chunk

        def grab(slot, lo, hi, layers_cfgs):
            return {
                (l_, c): {
                    p: bytes(table.read_device_chunk(layer=l_, position=p, slot=slot, config_id=c))
                    for p in range(R.align_down(lo, 32), R.ceil_to(hi, 32), 32)
                }
                for l_, c in layers_cfgs
            }

        def prefill_chunk(inp, kv, *a, slot_id, actual_start, actual_end, **kw):
            prev = last_end.get(slot_id, 0)
            watch = []
            if 0 < actual_start != prev:  # a follow-up start or a pulled-back chunk: re-read around it
                ttnn.synchronize_device(mesh)
                lo = max(0, R.align_down(actual_start, 32) - SAMPLE_BELOW * 32)
                below = [
                    (l_, c)
                    for l_ in H.sample_layers(contract.served_layers(s)[1])
                    for c in H.SAMPLE_CONFIGS["global" if H.is_global(l_) else "sliding"]
                ]
                if R.align_down(actual_start, 32) > lo:
                    watch.append(
                        (
                            "below",
                            slot_id,
                            lo,
                            R.align_down(actual_start, 32),
                            grab(slot_id, lo, R.align_down(actual_start, 32), below),
                        )
                    )
                if actual_start < prev:
                    hi = min(prev, actual_end)
                    first = [(0, 4), (0, 20)]  # layer 0's KV depends on the token and its position only
                    watch.append(("overlap", slot_id, actual_start, hi, grab(slot_id, actual_start, hi, first)))
            r = inner(inp, kv, *a, slot_id=slot_id, actual_start=actual_start, actual_end=actual_end, **kw)
            if watch:
                ttnn.synchronize_device(mesh)
                for what, slot, lo, hi, before in watch:
                    after = grab(slot, lo, hi, list(before))
                    rewrites.append((what, slot, lo, hi, before, after))
            last_end[slot_id] = max(prev, actual_end)
            return r

        runtime.prefill_chunk = prefill_chunk
        plan_path, drv_out, rc_file = out / "plan.json", out / "driver.json", out / "driver.rc"
        if kind == dgen_engine.ENGINE:
            dgen_engine.write_plan(
                plan_path,
                service_id=service,
                chunk=chunk,
                layers_per_chunk=contract.served_layers(s)[1],
                sp=sp,
                max_slots=H.slots(s),
                max_seq=max_seq,
                requests=[{"name": n, "token_ids": tokens[:k], "after": a} for n, k, a in REQUESTS],
                connect_timeout_ms=120000,
                run_timeout_s=2400,
                stall_s=900,
            )
            cmd, fenv = dgen_engine.driver_shell(build, str(plan_path), str(drv_out), str(rc_file), service), None
        else:
            cmd = [sys.executable, "-m", "models.demos.common.prefill.runners.prefill_producer"]
            fenv = dict(
                os.environ,
                PREFILL_TRACE_DIR=str(g.dir),
                PREFILL_PRODUCER_MAX_REQUESTS="4",
                PREFILL_PRODUCER_INTERLEAVE="round_robin",
                PREFILL_PRODUCER_MULTI_TURN_PROB="1",
                PREFILL_PRODUCER_MID_END_PROB="1",
                PREFILL_PRODUCER_CHECK_PCC="0",
                PREFILL_SEND_SHUTDOWN="1",
                PREFILL_H2D_CONNECT_TIMEOUT="120",
                PREFILL_PRODUCER_SEED="0",
            )
        with (out / "feeder.log").open("w") as flog:
            feeder = subprocess.Popen(cmd, env=fenv, stdout=flog, stderr=subprocess.STDOUT)

            def watchdog():  # a feeder that fails leaves the runner waiting for chunks: end its loop cleanly
                rc = feeder.wait()
                code = rc_file.read_text().strip() if rc_file.is_file() else ""
                if rc != 0 or code not in ("", "0"):
                    fails.append(f"feeder exited {rc} (driver rc {code or '-'}); see {out / 'feeder.log'}")
                    save()
                    try:  # the driver sends the sentinel only after it passed
                        dgen_engine.send_shutdown(service, timeout_s=60)
                    except Exception as e:
                        print(f"[contract] shutdown after feeder failure: {e}", flush=True)

            threading.Thread(target=watchdog, daemon=True).start()
            original_loop(runtime, kv_caches, *args, **kwargs)
            try:
                feeder.wait(timeout=120)
            except subprocess.TimeoutExpired:
                fails.append("feeder still running after the runner's loop ended")
                feeder.terminate()
        _check(runtime, table, kind, drv_out)
        save()

    def _check(runtime, table, kind, drv_out):
        mesh = runtime.mesh_device
        reader = kv_table.TableReader(table, contract.device_map(mesh))
        _, kvc = H.dgen_checks()
        thr, dtype = kv_threshold(s), H.DUMP_DTYPE[H.kv_dtype_name(s)]
        for what, slot, lo, hi, before, after in rewrites:
            for (layer, c), recs in before.items():
                w = H.config_width(c)
                a = {(layer, p): b for p, b in recs.items()}
                b = {(layer, p): x for p, x in after[(layer, c)].items()}
                res = kvc.compare(
                    a, b, method="bytecmp", layers=[layer], start=lo, end=hi, dtype=dtype, width=w, storage="tile"
                )
                if res["status"] != "passed":
                    fails.append(
                        f"{what}: slot {slot} layer {layer} {H.CONFIG_NAMES[c]} rows [{lo}, {hi}) changed "
                        f"across a later chunk: {res['findings'][:2]}"
                    )
        mets["runner_rewrites_checked"] = len(rewrites)
        if not any(r[0] == "overlap" for r in rewrites):
            fails.append("no chunk recomputed rows already written (no follow-up overlap, no pulled-back chunk)")
        if kind != dgen_engine.ENGINE:
            print("[contract] producer feeder: per-request KV read-back is checked only with the engine", flush=True)
            return
        d = json.loads(drv_out.read_text()) if drv_out.is_file() else {}
        if not d.get("ok"):
            fails.append(f"driver not ok: {d.get('errors')}")
        reqs = d.get("requests") or {}
        per_slot = {}
        for name, n, after in REQUESTS:
            q = reqs.get(name)
            if not q or q.get("done_position") != n:
                fails.append(f"request {name}: not done at {n} ({q})")
                continue
            per_slot.setdefault(q["slot"], []).append((q.get("done_s") or 0, n, q.get("resident") or 0, name))
        if not any((reqs.get(n) or {}).get("reused_tokens") for n, _, a in REQUESTS if a):
            fails.append("no follow-up turn reused a resident prefix")
        pulled = [
            n
            for n, k, a in REQUESTS
            if n in reqs and R.pulled_back(R.chunk_plan(k, reqs[n].get("resident") or 0, chunk, max_seq))
        ]
        if not pulled:
            fails.append("no request's last chunk was pulled back to max_seq - chunk")
        mets["runner_requests_done"] = sum(1 for n, k, _ in REQUESTS if (reqs.get(n) or {}).get("done_position") == k)
        layers = H.sample_layers(contract.served_layers(s)[1])
        for slot, done in per_slot.items():
            done.sort()
            longest = max(n for _, n, _, _ in done)
            f, scores = H.read_back_failures(reader, kvc, g.state, slot, longest, layers, thr, dtype, "runner")
            if done[-1][1] != longest:  # a shorter turn came last: rows past it hold the longer turn's (golden) KV
                f = [x for x in f if "pad rows" not in x]
            fails.extend(f)
            for k, v in scores.items():
                mets[f"runner_pcc_s{slot}_{k.replace(':', '_c')}"] = v

    try:
        with patch.object(prefill_runner, "run_request_loop", checked_loop):
            prefill_runner.main()
    except BaseException as e:
        fails.append(f"runner raised {type(e).__name__}: {e}")
        save()
        raise
    save()


if __name__ == "__main__":
    _main(sys.argv[1])
