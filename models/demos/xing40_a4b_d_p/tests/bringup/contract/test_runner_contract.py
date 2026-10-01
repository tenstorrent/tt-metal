# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, runner test: the whole model through the adapter only, driven like tt-d-gen drives it
(serving_contract.md, every section). Modelled on gemma4_d_p/tests/test_prefill_migration.py.

A subprocess runs tt-metal's prefill_runner.main (runner_case.py) for all 40 layers on the 4x2 mesh with FABRIC_2D,
2 slots, D2H layer acks (PREFILL_LAYER_ACK_D2H=1), and mock migration on the migration-enabled path (the table built
with first_layer_idx / num_my_layers / stage_layout, no KV Manager). The producer (producer_case.py) is tt-metal's
prefill_producer sending the server's chunks: 2 slots interleaved round-robin, each with a follow-up turn
(prefix reuse, kv_block_size 64), mid-chunk ends, and a 56000-token prompt near max seq 56320 whose last chunk is
pulled back over the previous one:

  slot 0: 3000 -> (0, 3000); 56000 from resident 2944 -> (2944, 8064) ... (49024, 54144), (51200, 56000)
  slot 1: 12345 -> (0, 5120), (5120, 10240), (10240, 12345); 20000 from resident 12288 -> (12288, 17408), (17408, 20000)

The KV is read through the exported table over UMD (what a KV Manager ships) and checked here on the CPU with the
server's own code (tt-d-gen tools/launch_harness/tables.py, kv_manager/tools/kv_dump_compare.py). Pass:
  - the runner's fabric: model_config.FABRIC_PAYLOAD_SIZE in [4352, 15232] B (runner_utils.py:41-53)
  - table: tables.read_table / layout accept it; layers exactly 0..39; >= 2 slots; every (slot, layer, pos < 56320)
    record present with one size; no two records on one address; every device-group replica holds the same bytes
  - acks: 40 per chunk (layers_per_chunk = 40), 17 chunks
  - each slot's final [0, end) vs the golden kv_latent, kv_dump_compare per-channel PCC >= max(0.93 server, 0.97 spec);
    the pad rows of each slot's last record zero
  - slot 0 after turn 1 (snapshot at its acks) vs the golden, and its reused prefix [0, 2944) byte-identical at the
    end (bytecmp), its last block [2944, 3000) PCC >= 0.99 (the harness's source / destination rule)
  - the pulled-back chunk: layer-0 rows [51200, 54144) byte-identical to what the previous chunk's acks shipped
Fail fast (run_child): the run is killed ~3 s after the producer exits non-zero, after 60 s without progress once it
exits 0, when nothing grows for 240 s, or after 900 s (XING_CONTRACT_AFTER_PRODUCER_S / _STALL_S / _RUN_S).
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


def run_child(out: Path, env: dict, log: Path) -> tuple[int, str]:
    """Run runner_case in its own process group with bounded waits: stop at once when the producer exits non-zero
    (<out>/producer.rc, written by runner_case's sh wrapper), after it exits 0 when the runner makes no progress (logs,
    <out>/final) for AFTER_PRODUCER_S, when nothing grows for STALL_S, or after CHILD_TIMEOUT_S. Stopping kills the group (runner and producer;
    SIGKILL, since the runner's request loop holds the GIL and never runs a SIGTERM handler). Returns (exit code,
    reason the run was stopped or "")."""
    import signal
    import time

    logs = (log, out / "producer.log")
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
        rc_file = out / "producer.rc"
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
                    why = f"producer exited {prod_rc} while the runner waits for chunks"
            if why:
                pass
            elif prod_done is not None and now - max(last, prod_done) > AFTER_PRODUCER_S:
                why = f"producer exited 0, then no runner progress (logs, final/) for {AFTER_PRODUCER_S:.0f} s"
            elif now - t0 > CHILD_TIMEOUT_S:
                why = f"runner timed out after {CHILD_TIMEOUT_S} s"
            elif now - last > STALL_S:
                why = f"no progress: runner.log and producer.log unchanged for {STALL_S:.0f} s"
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


def test_runner_contract(tmp_path):
    if impl_mode() != "device":
        pytest.skip("device test")
    R.self_check()
    tables, kvc = R.harness()
    g = R.golden()
    out = tmp_path
    log = out / "runner.log"
    rc, why = run_child(out, runner_env(out, g.dir), log)
    if why:
        pytest.fail(
            f"{why}; see {log} and {out / 'producer.log'}\n{tail(out / 'producer.log', 20)}\n{tail(log)}", pytrace=False
        )
    if (out / "not_built.txt").exists():
        pytest.fail(f"not built: {(out / 'not_built.txt').read_text()}", pytrace=False)
    res = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
    pj = json.loads((out / "producer.json").read_text()) if (out / "producer.json").exists() else {}
    if rc != 0 or res.get("errors") or pj.get("errors"):
        pytest.fail(
            f"runner exit {rc}; runner errors {res.get('errors')}; producer errors {pj.get('errors')}\n"
            f"see {log} and {out / 'producer.log'}\n{res.get('traceback', '')}\n{tail(log)}",
            pytrace=False,
        )

    fails = []
    tfails, geom = E.table_rules(str(out / "table.pb"), R.NUM_LAYERS, NUM_USERS)
    fails += [f"table: {x}" for x in tfails]
    if not geom:
        pytest.fail("\n".join(fails), pytrace=False)
    gk = {k: geom[k] for k in ("dtype", "width", "storage")}
    cbytes = geom["width"] // 32 * (2048 if geom["dtype"] == "bf16" else 1088)
    note = E.harness_geometry_note(geom)
    for where, mm in (("final", res.get("replica_mismatch")), ("snapshots", pj.get("replica_mismatch"))):
        if mm:
            fails.append(f"{where}: device-group replicas hold different bytes at (slot, layer, pos) {mm[:4]}")
    if pj.get("acks_drained") != pj.get("acks_expected"):
        fails.append(f"acks: {pj.get('acks_drained')} drained, {pj.get('acks_expected')} expected (40 per chunk)")

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

    snaps = pj.get("snapshots", {})
    for name in PC.SNAPSHOTS:
        if not snaps.get(name, {}).get("taken"):
            fails.append(f"snapshot {name} not taken (acks did not arrive)")
    if snaps.get("snap_s0_t0", {}).get("taken"):
        a = kvc.load_dump(str(out / "snap_s0_t0"), slot=0, chunk_bytes=cbytes)
        for layer in layers:
            got = kvc.reassemble(a, layer, 0, 3000, **gk)
            fails += R.kv_pcc_failures(
                kvc, got, R.golden_kv(g, layer)[:3000].numpy(), layer, thr, "slot 0 at turn-1 acks"
            )
        prefix = R.follow_up_resident(3000, 56000)
        r = kvc.compare(a, finals[0], method="bytecmp", layers=layers, start=0, end=prefix, **gk)
        if r["status"] != "passed":
            fails.append(f"slot 0 reused prefix [0, {prefix}) bytes changed after its acks: {r['findings'][:3]}")
        r = kvc.compare(
            a, finals[0], method="pcc", layers=layers, start=prefix, end=3000, threshold=R.LAST_BLOCK_PCC, **gk
        )
        if r["status"] != "passed":
            fails.append(
                f"slot 0 last block [{prefix}, 3000) PCC < {R.LAST_BLOCK_PCC}: {[x for x in r['findings'] if not x['passed']][:4]}"
            )
    if snaps.get("snap_overlap", {}).get("taken"):
        b = kvc.load_dump(str(out / "snap_overlap"), slot=0, chunk_bytes=cbytes)
        r = kvc.compare(b, finals[0], method="bytecmp", layers=[0], start=51200, end=54144, **gk)
        if r["status"] != "passed":
            fails.append(f"pulled-back chunk rewrote layer-0 [51200, 54144) with other bytes: {r['findings'][:3]}")
    assert not fails, "\n".join(fails)
