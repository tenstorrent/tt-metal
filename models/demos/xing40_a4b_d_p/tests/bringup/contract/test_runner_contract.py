# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, runner test: the whole model through the adapter only, driven by tt-d-gen's real engine
(serving_contract.md, every section). Modelled on gemma4_d_p/tests/test_prefill_migration.py.

A subprocess runs tt-metal's prefill_runner.main (runner_case.py) for all 40 layers on the 4x2 mesh with FABRIC_2D,
2 slots, D2H layer acks (PREFILL_LAYER_ACK_D2H=1), and mock migration on the migration-enabled path (the table built
with first_layer_idx / num_my_layers / stage_layout, no KV Manager). The feeder, chosen by use_engine:
  engine    tt-d-gen's BackendRuntime (PREFILL role, device_prefill_pipeline on the runner's H2D service and
            /tt_prefill_layer_acks_<id>), run by testing/dgen_prefill_driver.py under tt-d-gen's Python, when
            testing/dgen_engine.find_build finds a build. It admits ENGINE_CASE (below) and the engine decides slots,
            chunks, prefix reuse and interleave; the test checks what it did against server_rules (the plan per slot,
            the remount at 2944, the interleave) and every request's PREFILL_DONE (each chunk retired on 40 acks).
  producer  fallback, only when no build is found (BRINGUP_DGEN=1 makes that a failure): producer_case.py, tt-metal's
            prefill_producer sending server_rules.interleave of
            slot 0: 3000 -> (0, 3000); 56000 from resident 2944 -> (2944, 8064) ... (49024, 54144), (51200, 56000)
            slot 1: 12345 -> (0, 5120), (5120, 10240), (10240, 12345); 20000 from 12288 -> (12288, 17408), (17408, 20000)
The output names the feeder that ran.

The KV is read through the exported table over UMD (what a KV Manager ships) and checked here on the CPU with the
server's own code (tt-d-gen tools/launch_harness/tables.py, kv_manager/tools/kv_dump_compare.py). Pass:
  - the runner's fabric: model_config.FABRIC_PAYLOAD_SIZE in [4352, 15232] B (runner_utils.py:41-53)
  - table: tables.read_table / layout accept it; layers exactly 0..39; >= 2 slots; every (slot, layer, pos < 56320)
    record present with one size; no two records on one address; every device-group replica holds the same bytes
  - acks: 40 per chunk (layers_per_chunk = 40): the engine retires every chunk (PREFILL_DONE for every request, no
    stranded acks), or the producer drains 40 x 17
  - each slot's final [0, end) vs the golden kv_latent, kv_dump_compare per-channel PCC >= max(0.93 server, 0.97 spec);
    the pad rows of each slot's last record zero
  - slot A after turn 1 (engine: right before its follow-up's first chunk; producer: at its acks) vs the golden, its
    reused prefix [0, 2944) byte-identical at the end (bytecmp), its last block [2944, 3000) PCC >= 0.99
  - the pulled-back chunk: layer-0 rows [51200, 54144) byte-identical to what the earlier chunk wrote
Fail fast (run_child): the run is killed ~3 s after the feeder exits non-zero, after 60 s without progress once it
exits 0, when nothing grows for 240 s, or after 900 s (XING_CONTRACT_AFTER_PRODUCER_S / _STALL_S / _RUN_S). The
precompile pass (UP_FRONT_COLLECT=1) skips the body.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from models.demos.common.bringup.testing.harness import device_timeout, impl_mode, spec
from models.demos.xing40_a4b_d_p.tests.bringup.contract import engine as E
from models.demos.xing40_a4b_d_p.tests.bringup.contract import producer_case as PC
from models.demos.xing40_a4b_d_p.tests.bringup.contract import server_rules as R

S = spec()
pytestmark = device_timeout(S)
NUM_USERS = 2
# Bounds (env overrides): a passing run takes ~200 s with its longest log silence ~100 s (compile).
CHILD_TIMEOUT_S = min(int(S.get("box.test_timeout_s", 3600)) - 300, int(os.environ.get("XING_CONTRACT_RUN_S", "900")))
STALL_S = float(os.environ.get("XING_CONTRACT_STALL_S", "240"))  # no growth in runner.log / producer.log
# Once the producer is done the runner only reads the KV back (~95k record files into <out>/final in 60-90 s): no
# progress (logs or final/) for this long then = stuck.
AFTER_PRODUCER_S = float(os.environ.get("XING_CONTRACT_AFTER_PRODUCER_S", "60"))


def runner_env(out: Path, golden_dir: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PREFILL_")}
    env.update(
        PREFILL_MODEL=E.MODEL,
        PREFILL_SP=str(R.SP),
        PREFILL_TP=str(R.TP),
        PREFILL_NUM_LAYERS=str(R.NUM_LAYERS),
        PREFILL_CHUNK_SIZE=str(R.CHUNK),
        PREFILL_MAX_SEQ_LEN=str(R.MAX_SEQ),
        PREFILL_NUM_USERS=str(NUM_USERS),
        PREFILL_FABRIC_MODE="2d",  # runner_utils.open_mesh_device defaults to 2d_torus_xy: never on this box
        PREFILL_LAYER_ACK_D2H="1",
        PREFILL_USE_TRACE="0",
        PREFILL_H2D_SERVICE_ID=f"xing_contract_{os.getpid()}",
        PREFILL_ENABLE_MIGRATION="1",
        PREFILL_MOCK_MIGRATION="1",
        PREFILL_MIGRATION_EXPORT_TO_FILE="0",
        PREFILL_MIGRATION_TABLE_PATH=str(out / "table.pb"),
        PREFILL_MIGRATION_DEVICE_MAP_PATH=str(out / "device_map.json"),
        PREFILL_TRACE_DIR=str(golden_dir),
        PREFILL_H2D_CONNECT_TIMEOUT="600",
        PREFILL_PRODUCER_CHECK_PCC="1",  # connects the layer-ack channel; producer_case does the counting
        PREFILL_PRODUCER_MULTI_TURN_PROB="1.0",
        PREFILL_PRODUCER_MID_END_PROB="1.0",
        PREFILL_PRODUCER_INTERLEAVE="round_robin",
        PREFILL_PRODUCER_MAX_REQUESTS=str(sum(len(t) for t in PC.TURNS.values())),
        PREFILL_SEND_SHUTDOWN="1",
        TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=env.get("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0"),
    )
    env.setdefault("OMP_NUM_THREADS", "16")
    # The runner blocks on its H2D socket between chunks (producer start-up, snapshot reads, ack waits): that wait is
    # not a hang, so the safe runner's 5 s dispatch timeout would kill it. Above the producer's ack timeout (120 s).
    env["TT_METAL_OPERATION_TIMEOUT_SECONDS"] = "180"
    return env


def feeder_files(env: dict, out: Path) -> tuple[str, Path, Path]:
    """(name, log, rc file) of the process feeding the runner: tt-d-gen's engine driver or the producer fallback."""
    if env.get("XING_CONTRACT_FEEDER_CMD"):
        return "engine driver", out / "engine.log", out / "feeder.rc"
    return "producer", out / "producer.log", out / "producer.rc"


def use_engine(out: Path, env: dict, requests: list[dict], snapshots: dict, max_slots: int) -> str:
    """Make the runner's feeder tt-d-gen's real engine when a build is found (testing/dgen_engine.find_build: spec
    serving.server_repo or /localdev/$USER/tt-d-gen with a built tt_engine), else leave the producer fallback.
    requests: [{"name", "token_ids", "after"}] (dgen_prefill_driver.py). Returns the feeder description, printed and
    recorded by the caller."""
    from models.demos.common.bringup.testing import dgen_engine as D

    build, why = D.find_build(S)
    if build is None:
        if os.environ.get("BRINGUP_DGEN") == "1":
            pytest.fail(f"BRINGUP_DGEN=1 but no tt-d-gen engine build: {why}", pytrace=False)
        return f"prefill_producer (fallback: {why})"
    plan = {
        "service_id": env["PREFILL_H2D_SERVICE_ID"],
        "ack_shm_name": f"/tt_prefill_layer_acks_{env['PREFILL_H2D_SERVICE_ID']}",  # prefill_runner.py _serve_request
        "chunk_size": R.CHUNK,
        "layers_per_chunk": R.NUM_LAYERS,  # one ack per layer per chunk (D2H, LayerAckService)
        "sp_factor": R.SP,
        "max_slots": max_slots,  # == PREFILL_NUM_USERS
        "max_seq_len": R.MAX_SEQ,
        "kv_block_size": R.KV_BLOCK,
        "connect_timeout_ms": 600000,
        "run_timeout_s": CHILD_TIMEOUT_S,
        "stall_s": STALL_S,
        "requests": requests,
    }
    (out / "engine_plan.json").write_text(json.dumps(plan))
    cmd = D.driver_shell(
        build, str(out / "engine_plan.json"), str(out / "engine.json"), str(out / "feeder.rc"), plan["service_id"]
    )
    env["XING_CONTRACT_FEEDER_CMD"] = json.dumps(cmd)
    env["XING_CONTRACT_SNAPSHOTS"] = json.dumps(snapshots)
    return build.describe()


def run_child(out: Path, env: dict, log: Path) -> tuple[int, str]:
    """Run runner_case in its own process group with bounded waits: stop at once when the feeder (engine driver or
    producer) exits non-zero (its rc file, written by runner_case's sh wrapper), after it exits 0 when the runner makes
    no progress (logs, <out>/final) for AFTER_PRODUCER_S, when nothing grows for STALL_S, or after CHILD_TIMEOUT_S.
    Stopping kills the group (runner and feeder; SIGKILL, since the runner's request loop holds the GIL and never runs
    a SIGTERM handler). Returns (exit code, reason the run was stopped or "")."""
    import signal
    import time

    who, flog, rc_file = feeder_files(env, out)
    logs = (log, flog)
    with log.open("w") as f:
        proc = subprocess.Popen(
            [sys.executable, "-m", "models.demos.xing40_a4b_d_p.tests.bringup.contract.runner_case", str(out)],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        t0 = last = time.monotonic()
        sizes, prod_done = None, None
        why = ""
        while proc.poll() is None:
            time.sleep(1.0)
            now = time.monotonic()
            fin = out / "final"
            cur = tuple(p.stat().st_size if p.exists() else -1 for p in logs) + (
                sum(1 for _ in os.scandir(fin)) if fin.is_dir() else 0,
            )
            if cur != sizes:
                sizes, last = cur, now
            if prod_done is None and rc_file.exists() and rc_file.read_text().strip():
                prod_rc, prod_done = rc_file.read_text().strip(), now
                if prod_rc != "0":
                    time.sleep(3.0)  # let the runner finish if it is already on its way out
                    if proc.poll() is not None:
                        break
                    why = f"{who} exited {prod_rc} while the runner waits for chunks"
            if why:
                pass
            elif prod_done is not None and now - max(last, prod_done) > AFTER_PRODUCER_S:
                why = f"{who} exited 0, then no runner progress (logs, final/) for {AFTER_PRODUCER_S:.0f} s"
            elif now - t0 > CHILD_TIMEOUT_S:
                why = f"runner timed out after {CHILD_TIMEOUT_S} s"
            elif now - last > STALL_S:
                why = f"no progress: {log.name} and {flog.name} unchanged for {STALL_S:.0f} s"
            if why:
                for sig, wait_s in ((signal.SIGTERM, 3), (signal.SIGKILL, 30)):
                    try:
                        os.killpg(proc.pid, sig)
                        proc.wait(timeout=wait_s)
                        break
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        continue
                break
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    return proc.returncode, why


def tail(p: Path, n: int = 40) -> str:
    return "\n".join(p.read_text(errors="replace").splitlines()[-n:]) if p.exists() else "(no log)"


# The engine case: what tt-d-gen's engine does with these admits is deterministic (checked against
# te.mock_prefill_pipeline): every prompt is a prefix of the one golden token stream, so a second request admitted
# while another slot holds a longer prefix goes cold, never remounts. Hence one follow-up turn, on slot A:
#   a   3000 cold                       -> (0, 3000)                                    mid-chunk end
#   a2  56320 after a, remount at 2944  -> (2944, 8064) ... (49024, 54144), (51200, 56320)   pulled-back last chunk
#   b   12345 after a, cold, other slot -> (0, 5120), (5120, 10240), (10240, 12345)     interleaved with a2
ENGINE_CASE = (("a", 3000, None), ("a2", R.MAX_SEQ, "a"), ("b", 12345, "a"))
# Taken by the runner right before the named chunk runs (runner_case.py), on that chunk's slot.
ENGINE_SNAPSHOTS = {
    "snap_s0_t0": {"before": [2944, 2944 + R.CHUNK], "lo": 0, "hi": 3000},
    "snap_overlap": {"before": [R.MAX_SEQ - R.CHUNK, R.MAX_SEQ], "lo": R.MAX_SEQ - R.CHUNK, "hi": 2944 + 10 * R.CHUNK},
}


def log(msg: str) -> None:
    print(f"[runner contract] {msg}", flush=True)


def engine_checks(out: Path, res: dict) -> tuple[list[str], dict]:
    """What the engine did against the server rules. Returns (failures, {name: slot})."""
    fails = []
    ej = json.loads((out / "engine.json").read_text()) if (out / "engine.json").exists() else {}
    if not ej.get("ok"):
        fails.append(f"engine driver: {ej.get('errors') or 'no engine.json'}")
    rq = ej.get("requests", {})
    slot = {n: rq.get(n, {}).get("slot") for n, _, _ in ENGINE_CASE}
    for n, length, _ in ENGINE_CASE:
        if rq.get(n, {}).get("done_position") != length:
            fails.append(f"request {n}: PREFILL_DONE at {rq.get(n, {}).get('done_position')}, prompt {length}")
    want_res = R.follow_up_resident(3000, R.MAX_SEQ)
    a2 = rq.get("a2", {})
    if a2.get("slot") != slot["a"] or a2.get("resident") != want_res:
        fails.append(
            f"follow-up a2: slot {a2.get('slot')} resident {a2.get('resident')}, expected a remount of a's slot "
            f"{slot['a']} at {want_res} (prefix_indexer.hpp reusable_prefix_cap, backend_runtime.cpp prepare_admission)"
        )
    if rq.get("b", {}).get("resident") != 0 or slot["b"] == slot["a"]:
        fails.append(f"b: slot {slot['b']} resident {rq.get('b', {}).get('resident')}, expected cold on another slot")
    # the chunks the runner was handed, per slot, are the server's plan (prefill_writer.cpp)
    chunks = res.get("chunks", [])
    want = {
        slot["a"]: R.chunk_plan(3000) + R.chunk_plan(R.MAX_SEQ, want_res),
        slot["b"]: R.chunk_plan(12345),
    }
    for sl, plan in want.items():
        got = [(c[1], c[2]) for c in chunks if c[0] == sl]
        if got != plan:
            fails.append(f"slot {sl}: runner got chunks {got}, the server plan is {plan}")
    order = [c[0] for c in chunks]
    a2_lo = next((i for i, c in enumerate(chunks) if c[0] == slot["a"] and c[1] == want_res), None)
    if a2_lo is None or slot["b"] not in order[a2_lo:] or order[-1] != slot["a"]:
        fails.append(f"slots not interleaved: runner chunk order {order}")
    log(f"engine: slots {slot}, a2 remounted at {a2.get('resident')}, runner chunk order {order}")
    elog = (out / "engine.log").read_text(errors="replace") if (out / "engine.log").exists() else ""
    for bad in ("unretired layer-ack", "stranded layer-ack"):  # prefill_pipeline.cpp: acks left unmatched
        if bad in elog:
            fails.append(f"engine.log reports {bad!r}")
    return fails, slot


def test_runner_contract(tmp_path):
    if impl_mode() != "device":
        pytest.skip("device test")
    if os.environ.get("UP_FRONT_COLLECT") == "1":
        pytest.skip("precompile pass: the runner subprocess needs the chips; runs in the real pass only")
    R.self_check()
    tables, kvc = R.harness()
    g = R.golden()
    out = tmp_path
    rlog = out / "runner.log"
    env = runner_env(out, g.dir)
    toks = [int(t) for t in g.tokens().tolist()]
    reqs = [{"name": n, "token_ids": toks[:length], "after": after} for n, length, after in ENGINE_CASE]
    feeder = use_engine(out, env, reqs, ENGINE_SNAPSHOTS, NUM_USERS)
    engine = bool(env.get("XING_CONTRACT_FEEDER_CMD"))
    log("feeder: " + feeder)
    who, flog, _ = feeder_files(env, out)
    rc, why = run_child(out, env, rlog)
    if why:
        pytest.fail(f"[{feeder}] {why}; see {rlog} and {flog}\n{tail(flog, 20)}\n{tail(rlog)}", pytrace=False)
    if (out / "not_built.txt").exists():
        pytest.fail(f"not built: {(out / 'not_built.txt').read_text()}", pytrace=False)
    res = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
    pj = json.loads((out / "producer.json").read_text()) if (out / "producer.json").exists() else {}
    if rc != 0 or res.get("errors") or pj.get("errors"):
        pytest.fail(
            f"[{feeder}] runner exit {rc}; runner errors {res.get('errors')}; {who} errors "
            f"{pj.get('errors') if not engine else tail(out / 'engine.json', 10)}\n"
            f"see {rlog} and {flog}\n{res.get('traceback', '')}\n{tail(rlog)}",
            pytrace=False,
        )

    fails = []
    if engine:
        efails, slots = engine_checks(out, res)
        fails += efails
        slot_a, snaps, follow = slots["a"], res.get("snapshots", {}), R.MAX_SEQ
        overlap = ENGINE_SNAPSHOTS["snap_overlap"]
    else:
        slot_a, snaps, follow = 0, pj.get("snapshots", {}), 56000
        overlap = PC.SNAPSHOTS["snap_overlap"]
        if pj.get("acks_drained") != pj.get("acks_expected"):
            fails.append(f"acks: {pj.get('acks_drained')} drained, {pj.get('acks_expected')} expected (40 per chunk)")
    tfails, geom = E.table_rules(str(out / "table.pb"), R.NUM_LAYERS, NUM_USERS)
    fails += [f"table: {x}" for x in tfails]
    if not geom:
        pytest.fail(f"[{feeder}] " + "\n".join(fails), pytrace=False)
    gk = {k: geom[k] for k in ("dtype", "width", "storage")}
    cbytes = geom["width"] // 32 * (2048 if geom["dtype"] == "bf16" else 1088)
    note = E.harness_geometry_note(geom)
    for where, mm in (("final", res.get("replica_mismatch")), ("snapshots", pj.get("replica_mismatch"))):
        if mm:
            fails.append(f"{where}: device-group replicas hold different bytes at (slot, layer, pos) {mm[:4]}")

    thr = R.state_threshold()
    layers = list(range(R.NUM_LAYERS))
    finals = {}
    for s, e in res["ends"].items():
        s, e = int(s), int(e)
        d = kvc.load_dump(str(out / "final"), slot=s, chunk_bytes=cbytes)
        finals[s] = d
        for layer in layers:
            got = kvc.reassemble(d, layer, 0, R.ceil_to(e, 32), **gk)
            f = R.kv_pcc_failures(kvc, got[:e], R.golden_kv(g, layer)[:e].numpy(), layer, thr, f"slot {s} [0, {e})")
            fails += [f"{x}; {note}" for x in f]
            if got[e:].size and not np.all(got[e:] == 0):
                fails.append(f"slot {s} layer {layer}: pad rows [{e}, {R.ceil_to(e, 32)}) not zero")

    for name in PC.SNAPSHOTS:
        if not snaps.get(name, {}).get("taken"):
            fails.append(f"snapshot {name} not taken ({'its chunk never ran' if engine else 'acks did not arrive'})")
    if snaps.get("snap_s0_t0", {}).get("taken"):
        a = kvc.load_dump(str(out / "snap_s0_t0"), slot=slot_a, chunk_bytes=cbytes)
        for layer in layers:
            got = kvc.reassemble(a, layer, 0, 3000, **gk)
            fails += R.kv_pcc_failures(
                kvc, got, R.golden_kv(g, layer)[:3000].numpy(), layer, thr, f"slot {slot_a} after turn 1"
            )
        prefix = R.follow_up_resident(3000, follow)
        r = kvc.compare(a, finals[slot_a], method="bytecmp", layers=layers, start=0, end=prefix, **gk)
        if r["status"] != "passed":
            fails.append(f"slot {slot_a} reused prefix [0, {prefix}) bytes changed after turn 1: {r['findings'][:3]}")
        r = kvc.compare(
            a, finals[slot_a], method="pcc", layers=layers, start=prefix, end=3000, threshold=R.LAST_BLOCK_PCC, **gk
        )
        if r["status"] != "passed":
            fails.append(
                f"slot {slot_a} last block [{prefix}, 3000) PCC < {R.LAST_BLOCK_PCC}: "
                f"{[x for x in r['findings'] if not x['passed']][:4]}"
            )
    if snaps.get("snap_overlap", {}).get("taken"):
        lo, hi = overlap["lo"], overlap["hi"]
        b = kvc.load_dump(str(out / "snap_overlap"), slot=slot_a, chunk_bytes=cbytes)
        r = kvc.compare(b, finals[slot_a], method="bytecmp", layers=[0], start=lo, end=hi, **gk)
        if r["status"] != "passed":
            fails.append(f"pulled-back chunk rewrote layer-0 [{lo}, {hi}) with other bytes: {r['findings'][:3]}")
    log(f"{len(fails)} failures; feeder {feeder}")
    assert not fails, f"[{feeder}]\n" + "\n".join(fails)
