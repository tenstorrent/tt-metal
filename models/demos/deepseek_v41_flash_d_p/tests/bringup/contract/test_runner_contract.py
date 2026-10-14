# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, runner test: the model through the adapter only, under tt-metal's real prefill runner
(serving_contract.md, every section). Modelled on gemma4_d_p/tests/test_prefill_migration.py.

A subprocess runs prefill_runner.main (runner_case.py) on the spec's mesh with its 2D fabric, for the served layers
(caches.served_prefix: all 40 in a full run, 0-3 in this partial one) at chunk 2048 / max_seq 4096, 2 slots, D2H layer
acks (PREFILL_LAYER_ACK_D2H=1) and mock migration on the migration-enabled path (kv_migration_stages ->
build_kv_chunk_table(..., stage_layouts), no KV Manager). The feeder (use_engine):
  engine    tt-d-gen's BackendRuntime (PREFILL role, device_prefill_pipeline on the runner's H2D service and
            /tt_prefill_layer_acks_<id>), run by testing/dgen_prefill_driver.py under tt-d-gen's Python when
            testing/dgen_engine.find_build finds a build. It admits ENGINE_CASE; the engine picks slots, chunks, prefix
            reuse and the interleave; the test checks what it did against server_rules
  producer  only with serving.require_engine: false (this spec: tt-d-gen does not run on the QuietBox) or BRINGUP_DGEN=0:
            producer_case.py, tt-metal's prefill_producer sending server_rules.interleave(RUNNER_TURNS):
            (0, 0, 1000), (1, 0, 1500), (0, 960, 3008), (1, 1472, 3500), (0, 2048, 4000)
The output names the feeder that ran.

The KV is read through the exported table over UMD (what a KV Manager ships) and checked on the CPU with the server's
own code (tt-d-gen tools/launch_harness/tables.py, kv_manager/tools/kv_dump_compare.py, validation.py). Pass:
  - the table (caches.table_rules): read_table accepts it, the served configs, every record present with one size,
    64-byte aligned offsets, no two records overlapping on a chip and bank; every device-group replica the same bytes
  - acks: layers_per_chunk = served layers per chunk: the engine retires every chunk (PREFILL_DONE for every request,
    no stranded acks), or the producer drains layers x chunks
  - each slot's final [0, end) of every served cache vs the golden, kv_dump_compare per-channel PCC >= max(0.93 server,
    0.97 spec); pad rows of each slot's last record zero
  - slot A after turn 1 vs the golden; its reused prefix [0, resident) byte-identical at the end; its last block
    [resident, turn 1) PCC >= 0.99 (validation.py:55-61 comparison_ranges)
  - the pulled-back chunk: rows below its start unchanged, layer 0's window_kv rewritten with the same bytes
Fail fast (run_child): the run is killed ~3 s after the feeder exits non-zero, after AFTER_FEEDER_S without progress
once it exits 0, when no log grows for STALL_S, or after CHILD_TIMEOUT_S. The precompile pass (UP_FRONT_COLLECT=1)
skips the body.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from models.demos.common.bringup.testing.harness import device_timeout, impl_mode, spec
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import server_rules as R

S = spec()
pytestmark = device_timeout(S)
NUM_USERS = 2
CHILD_TIMEOUT_S = min(int(S.get("box.test_timeout_s", 3600)) - 300, int(os.environ.get("DS41_CONTRACT_RUN_S", "2700")))
STALL_S = float(os.environ.get("DS41_CONTRACT_STALL_S", "600"))  # weight load can be quiet for minutes
AFTER_FEEDER_S = float(os.environ.get("DS41_CONTRACT_AFTER_FEEDER_S", "180"))  # final read-back over UMD

# The engine case: deterministic under tt-d-gen's engine (every prompt is a prefix of the one golden token stream;
# a request admitted while the slot holding its prefix is busy goes cold: a prefill node never copies, so there is
# one follow-up, on slot A):
#   a   1000 cold                 -> (0, 1000)                    mid-record end
#   a2  4096 after a, remount 960 -> (960, 3008), (2048, 4096)    a start off the SP period; the last chunk pulled back
#   b   1500 after a, cold        -> (0, 1500) on another slot    interleaved with a2
T1, B = 1000, 1500
ENGINE_CASE = (("a", T1, None), ("a2", R.MAX_SEQ, "a"), ("b", B, "a"))
A2_RES, A2_PLAN = R.follow_up(T1, R.MAX_SEQ)
ENGINE_SNAPSHOTS = {  # taken by the runner right before the named chunk runs, on that chunk's slot
    "snap_s0_t0": {"before": list(A2_PLAN[0]), "lo": 0, "hi": T1},
    "snap_overlap": {"before": list(A2_PLAN[-1]), "lo": A2_PLAN[-1][0], "hi": A2_PLAN[-2][1]},
}


def log(msg: str) -> None:
    print(f"[runner contract] {msg}", flush=True)


def runner_env(out: Path, trace_dir: Path, num_layers: int, num_users: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PREFILL_")}
    sid = f"ds41_contract_{os.getpid()}"
    env.update(
        PREFILL_MODEL=C.adapter_name(),
        PREFILL_SP=str(R.SP),
        PREFILL_TP=str(R.TP),
        PREFILL_NUM_LAYERS=str(num_layers),
        PREFILL_CHUNK_SIZE=str(R.CHUNK),
        PREFILL_MAX_SEQ_LEN=str(R.MAX_SEQ),
        PREFILL_NUM_USERS=str(num_users),
        PREFILL_FABRIC_MODE=C.fabric_mode(S),  # the runner defaults to 2d_torus_xy: never on a box without torus links
        PREFILL_LAYER_ACK_D2H="1",
        PREFILL_USE_TRACE="0",
        PREFILL_H2D_SERVICE_ID=sid,
        PREFILL_ENABLE_MIGRATION="1",
        PREFILL_MOCK_MIGRATION="1",
        PREFILL_MIGRATION_EXPORT_TO_FILE="0",
        PREFILL_MIGRATION_TABLE_PATH=str(out / "table.pb"),
        PREFILL_MIGRATION_DEVICE_MAP_PATH=str(out / "device_map.json"),
        PREFILL_TRACE_DIR=str(trace_dir),
        PREFILL_H2D_CONNECT_TIMEOUT="1800",
        PREFILL_PRODUCER_CHECK_PCC="1",  # connects the layer-ack channel; producer_case does the counting
        PREFILL_PRODUCER_MULTI_TURN_PROB="1.0",
        PREFILL_PRODUCER_MID_END_PROB="1.0",
        PREFILL_PRODUCER_INTERLEAVE="round_robin",
        PREFILL_PRODUCER_MAX_REQUESTS=str(sum(len(t) for t in R.RUNNER_TURNS.values())),
        PREFILL_SEND_SHUTDOWN="1",
        TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=env.get("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0"),
    )
    env.setdefault("OMP_NUM_THREADS", "16")
    # The runner blocks on its H2D socket between chunks; the safe runner's 5 s dispatch timeout would kill it (K-80).
    env["TT_METAL_OPERATION_TIMEOUT_SECONDS"] = "600"
    return env


def feeder_files(env: dict, out: Path) -> tuple[str, Path, Path]:
    if env.get("DS41_CONTRACT_FEEDER_CMD"):
        return "engine driver", out / "engine.log", out / "feeder.rc"
    return "producer", out / "producer.log", out / "producer.rc"


def use_engine(out: Path, env: dict, requests: list[dict], snapshots: dict, max_slots: int, layers: int) -> str:
    """Make the feeder tt-d-gen's real engine when a build is found; without one fail unless serving.require_engine is
    false (then the producer fallback). Before the engine, the three things both sides share are checked against the
    tt-d-gen sources at R.DGEN_SHA: the H2D service id, the ack channel name and the 12-byte header."""
    from models.demos.common.bringup.testing import dgen_engine as D

    # dgen_engine.server_repo takes spec serving.server_repo verbatim; this spec's "/localdev/$USER/tt-d-gen" needs
    # the variable expanded (server_rules.server_repo does)
    os.environ.setdefault("BRINGUP_SERVER_REPO", str(R.server_repo()))
    build, why = D.find_build(S)
    if build is None:
        if D.engine_required(S):
            pytest.fail(
                f"no tt-d-gen engine build ({why}); build it (agents/dgen-build.md) or set serving.require_engine: false",
                pytrace=False,
            )
        return f"prefill_producer (fallback: {why})"
    plan = {
        "service_id": env["PREFILL_H2D_SERVICE_ID"],
        "ack_shm_name": f"/tt_prefill_layer_acks_{env['PREFILL_H2D_SERVICE_ID']}",  # prefill_runner.py:745
        "chunk_size": R.CHUNK,
        "layers_per_chunk": layers,  # one ack per served layer per chunk (prefill_reader.cpp:155-160)
        "sp_factor": R.SP,
        "max_slots": max_slots,
        "max_seq_len": R.MAX_SEQ,
        "kv_block_size": R.KV_BLOCK,
        "connect_timeout_ms": 1800000,
        "run_timeout_s": CHILD_TIMEOUT_S,
        "stall_s": STALL_S,
        "requests": requests,
    }
    (out / "engine_plan.json").write_text(json.dumps(plan))
    cmd = D.driver_shell(
        build, str(out / "engine_plan.json"), str(out / "engine.json"), str(out / "feeder.rc"), plan["service_id"]
    )
    env["DS41_CONTRACT_FEEDER_CMD"] = json.dumps(cmd)
    env["DS41_CONTRACT_SNAPSHOTS"] = json.dumps(snapshots)
    return build.describe()


def run_child(out: Path, env: dict, log_path: Path) -> tuple[int, str]:
    """runner_case in its own process group with bounded waits. Returns (exit code, why it was stopped or "")."""
    import signal
    import time

    who, flog, rc_file = feeder_files(env, out)
    logs = (log_path, flog)
    with log_path.open("w") as f:
        proc = subprocess.Popen(
            [sys.executable, "-m", "models.demos.deepseek_v41_flash_d_p.tests.bringup.contract.runner_case", str(out)],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        t0 = last = time.monotonic()
        sizes, done_at, why = None, None, ""
        while proc.poll() is None:
            time.sleep(1.0)
            now = time.monotonic()
            fin = out / "final"
            cur = tuple(p.stat().st_size if p.exists() else -1 for p in logs) + (
                sum(len(os.listdir(d)) for d in fin.iterdir()) if fin.is_dir() else 0,
            )
            if cur != sizes:
                sizes, last = cur, now
            if done_at is None and rc_file.exists() and rc_file.read_text().strip():
                rc, done_at = rc_file.read_text().strip(), now
                if rc != "0":
                    time.sleep(3.0)
                    if proc.poll() is not None:
                        break
                    why = f"{who} exited {rc} while the runner waits for chunks"
            if why:
                pass
            elif done_at is not None and now - max(last, done_at) > AFTER_FEEDER_S:
                why = f"{who} exited 0, then no runner progress (logs, final/) for {AFTER_FEEDER_S:.0f} s"
            elif now - t0 > CHILD_TIMEOUT_S:
                why = f"runner timed out after {CHILD_TIMEOUT_S} s"
            elif now - last > STALL_S:
                why = f"no progress: {log_path.name} and {flog.name} unchanged for {STALL_S:.0f} s"
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


def child_outcome(out: Path, env: dict, rc: int, why: str, feeder: str) -> tuple[dict, dict, dict]:
    """Fail on a stopped / not-built / failed run; else (result.json, producer.json, engine.json)."""
    who, flog, _ = feeder_files(env, out)
    rlog = out / "runner.log"
    if why:
        pytest.fail(f"[{feeder}] {why}; see {rlog} and {flog}\n{tail(flog, 20)}\n{tail(rlog)}", pytrace=False)
    if (out / "not_built.txt").exists():
        pytest.fail(f"not built: {(out / 'not_built.txt').read_text()}", pytrace=False)
    res = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
    pj = json.loads((out / "producer.json").read_text()) if (out / "producer.json").exists() else {}
    ej = json.loads((out / "engine.json").read_text()) if (out / "engine.json").exists() else {}
    engine = bool(env.get("DS41_CONTRACT_FEEDER_CMD"))
    ferr = (ej.get("errors") or ([] if ej.get("ok") else ["no engine.json / not ok"])) if engine else pj.get("errors")
    if rc != 0 or res.get("errors") or ferr:
        pytest.fail(
            f"[{feeder}] runner exit {rc}; runner errors {res.get('errors')}; {who} errors {ferr}\n"
            f"see {rlog} and {flog}\n{res.get('traceback', '')}\n{tail(rlog)}",
            pytrace=False,
        )
    return res, pj, ej


def engine_checks(out: Path, res: dict, ej: dict) -> tuple[list[str], dict]:
    """What the engine did against the server rules. Returns (failures, {request name: slot})."""
    fails = []
    rq = ej.get("requests", {})
    slot = {n: rq.get(n, {}).get("slot") for n, _, _ in ENGINE_CASE}
    for n, length, _ in ENGINE_CASE:
        if rq.get(n, {}).get("done_position") != length:
            fails.append(f"request {n}: PREFILL_DONE at {rq.get(n, {}).get('done_position')}, prompt {length}")
    a2 = rq.get("a2", {})
    if a2.get("slot") != slot["a"] or a2.get("resident") != A2_RES:
        fails.append(
            f"follow-up a2: slot {a2.get('slot')} resident {a2.get('resident')}, expected a remount of a's slot "
            f"{slot['a']} at {A2_RES} (prefix_indexer.hpp:75-78, backend_runtime.cpp:527-557)"
        )
    if rq.get("b", {}).get("resident") != 0 or slot["b"] == slot["a"]:
        fails.append(f"b: slot {slot['b']} resident {rq.get('b', {}).get('resident')}, expected cold on another slot")
    chunks = res.get("chunks", [])
    for sl, plan in ((slot["a"], R.chunk_plan(T1) + A2_PLAN), (slot["b"], R.chunk_plan(B))):
        got = [(c[1], c[2]) for c in chunks if c[0] == sl]
        if got != plan:
            fails.append(f"slot {sl}: runner got chunks {got}, the server plan is {plan} (prefill_writer.cpp:51-82)")
    order = [c[0] for c in chunks]
    a2_lo = next((i for i, c in enumerate(chunks) if c[0] == slot["a"] and c[1] == A2_RES), None)
    if a2_lo is None or slot["b"] not in order[a2_lo:] or order[-1] != slot["a"]:
        fails.append(f"slots not interleaved: runner chunk order {order}")
    elog = (out / "engine.log").read_text(errors="replace") if (out / "engine.log").exists() else ""
    for bad in ("unretired layer-ack", "stranded layer-ack"):  # prefill_pipeline.cpp:77-80, 91-94
        if bad in elog:
            fails.append(f"engine.log reports {bad!r}")
    log(f"engine: slots {slot}, a2 resident {a2.get('resident')}, runner chunk order {order}")
    return fails, slot


def kv_checks(
    out: Path, res: dict, layers: list[int], slot_a: int, t1: int, follow: int, snaps: dict, overlap: dict | None
) -> list[str]:
    """Every served cache of every slot through the table vs the golden; the prefix / last-block / rewrite rules."""
    _, kvc, _ = R.harness()
    gk = C.GoldenKV()
    thr = R.state_threshold()
    fails = []
    finals = {}
    for s, e in res["ends"].items():
        s, e = int(s), int(e)
        for cfg in C.served_configs(layers):
            d = C.load_dump(out / "final", cfg, s)
            finals[(s, cfg["name"])] = d
            for layer in cfg["layers"]:
                got, pad = C.valid_and_pad(d, cfg, layer, 0, e)
                if cfg["rows"] == 32:  # the server's own reassembly agrees with ours on a 32-row record
                    ref = kvc.reassemble(d, layer, 0, e, dtype="bf16", width=cfg["width"], storage="row-major")
                    if not np.array_equal(ref, got):
                        fails.append(f"{cfg['name']} slot {s} layer {layer}: kv_dump_compare.reassemble disagrees")
                fails += C.pcc_failures(kvc, cfg, got, gk.rows(cfg, layer, 0, e), layer, thr, f"slot {s} [0, {e})")
                if pad.size and not (pad == 0).all():
                    fails.append(f"{cfg['name']} slot {s} layer {layer}: pad rows after {e} not zero")
    if snaps.get("snap_s0_t0", {}).get("taken"):
        prefix = R.follow_up_resident(t1, follow)
        for cfg in C.served_configs(layers):
            a = C.load_dump(out / "snap_s0_t0", cfg, slot_a)
            for layer in cfg["layers"]:
                got, _ = C.valid_and_pad(a, cfg, layer, 0, t1)
                fails += C.pcc_failures(kvc, cfg, got, gk.rows(cfg, layer, 0, t1), layer, thr, f"slot {slot_a} turn 1")
                x, _ = C.valid_and_pad(a, cfg, layer, prefix, t1)
                y, _ = C.valid_and_pad(finals[(slot_a, cfg["name"])], cfg, layer, prefix, t1)
                fails += C.pcc_failures(kvc, cfg, y, x, layer, R.LAST_BLOCK_PCC, f"slot {slot_a} last block")
            f = C.bytes_equal(a, finals[(slot_a, cfg["name"])], cfg, cfg["layers"], 0, prefix)
            fails += [f"slot {slot_a} reused prefix [0, {prefix}) changed after turn 1: {x}" for x in f]
            if cfg["name"] == C.CONFIG_ORDER[0]:  # the server's bytecmp on the same range
                r = kvc.compare(
                    a,
                    finals[(slot_a, C.CONFIG_ORDER[0])],
                    method="bytecmp",
                    layers=cfg["layers"],
                    start=0,
                    end=prefix,
                    dtype="bf16",
                    width=cfg["width"],
                    storage="row-major",
                )
                if r["status"] != "passed":
                    fails.append(f"kv_dump_compare bytecmp slot {slot_a} [0, {prefix}): {r['findings'][:3]}")
    else:
        fails.append("snapshot snap_s0_t0 not taken")
    if overlap is not None and snaps.get("snap_overlap", {}).get("taken"):
        lo, hi = overlap["lo"], overlap["hi"]
        cfg = C.served_configs(layers)[0]
        b = C.load_dump(out / "snap_overlap", cfg, slot_a)
        f = C.bytes_equal(b, finals[(slot_a, C.CONFIG_ORDER[0])], cfg, [0], lo, hi)
        fails += [f"the pulled-back chunk rewrote layer 0 [{lo}, {hi}) with other bytes: {x}" for x in f]
    elif overlap is not None:
        fails.append("snapshot snap_overlap not taken")
    return fails


def test_runner_contract(tmp_path):
    if impl_mode() != "device":
        pytest.skip("device test")
    if os.environ.get("UP_FRONT_COLLECT") == "1":
        pytest.skip("precompile pass: the runner subprocess needs the chips; runs in the real pass only")
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import producer_case as PC

    R.self_check()
    R.harness()
    layers = C.served_prefix()
    gk = C.GoldenKV()
    out = tmp_path
    env = runner_env(out, gk.g.dir, len(layers), NUM_USERS)
    toks = [int(t) for t in gk.tokens().tolist()]
    reqs = [{"name": n, "token_ids": toks[:length], "after": after} for n, length, after in ENGINE_CASE]
    feeder = use_engine(out, env, reqs, ENGINE_SNAPSHOTS, NUM_USERS, len(layers))
    engine = bool(env.get("DS41_CONTRACT_FEEDER_CMD"))
    log(f"feeder: {feeder}; layers {layers}; configs {[c['name'] for c in C.served_configs(layers)]}")
    rc, why = run_child(out, env, out / "runner.log")
    res, pj, ej = child_outcome(out, env, rc, why, feeder)

    fails = []
    if engine:
        efails, slots = engine_checks(out, res, ej)
        fails += efails
        slot_a, snaps, t1, follow = slots["a"], res.get("snapshots", {}), T1, R.MAX_SEQ
        overlap = ENGINE_SNAPSHOTS["snap_overlap"]
    else:
        slot_a, snaps, t1, follow = 0, pj.get("snapshots", {}), PC.TURNS[0][0], PC.TURNS[0][1]
        overlap = PC.SNAPSHOTS.get("snap_overlap")
        if pj.get("acks_drained") != pj.get("acks_expected"):
            fails.append(f"acks: {pj.get('acks_drained')} drained, {pj.get('acks_expected')} expected")
        plan = [(p["slot"], p["start"], p["end"]) for p in pj.get("plan", [])]
        if [tuple(c) for c in res.get("chunks", [])] != plan:
            fails.append(f"the runner was handed {res.get('chunks')}, the producer sent {plan}")
    fails += [f"table: {x}" for x in C.table_rules(str(out / "table.pb"), layers, NUM_USERS)]
    for where, mm in (("final", res.get("replica_mismatch")), ("snapshots", pj.get("replica_mismatch"))):
        if mm:
            fails.append(f"{where}: device-group replicas hold different bytes at {mm[:4]}")
    fails += kv_checks(out, res, layers, slot_a, t1, follow, snaps, overlap)
    log(f"{len(fails)} failures; feeder {feeder}")
    assert not fails, f"[{feeder}]\n" + "\n".join(fails[:60])
