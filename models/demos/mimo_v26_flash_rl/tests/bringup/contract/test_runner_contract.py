# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, "Adapter and table" (serving_contract.md): the real prefill runner, fed like the server.

tt-metal's prefill_runner (models/demos/common/prefill/runners/prefill_runner.py) serves the model through its adapter
in a subprocess that opens the mesh itself (so this test does not open the mesh fixture; under BRINGUP_IMPL=reference
mesh_device is None and the body is skipped, as with mesh_parametrize). Runner settings: all served layers (0-5),
SP 8 x TP 4, chunk 5120, max_seq 56320, the owner's slot count, device-to-host layer acks (PREFILL_LAYER_ACK_D2H=1),
mock migration (PREFILL_ENABLE_MIGRATION=0: the table and device map are written, no KV Manager), the box's fabric.

Feeder (bringup_fw.testing.dgen_engine.feeder): tt-d-gen's engine (the driver on the runner's H2D service and ack
channel) when built, else tt-metal's prefill_producer when the run allows it; the test prints which ran. Engine plan
(every prompt a prefix of the target rung's golden tokens, so each slot's KV is checkable against the golden whatever
slot and prefix the engine picks):
    a0 8017 tokens (cold, two chunks, ends mid-record)   b0 7001 (cold, interleaved with a0)
    a1 9000 after a0 (follow-up over the 64-aligned prefix 8000)   b1 12000 after b0 (follow-up)
    a2 56000 after a1 (follow-up whose last chunk the engine pulls back to max_seq - chunk, tt-d-gen #430)
The producer feeds two slots round robin, multi-turn, mid-chunk ends (it never pulls back).

Pass (failures listed in results <task>_runner_contract.json, metric runner_contract_failures == 0):
    transport    the engine and the runner agree on the H2D service, the 12-byte header and the ack channel (the
                 runner's LayerCompletionQueue ring /tt_prefill_layer_completions_<service_id>, same wire format in
                 tt-d-gen's tt-blaze tt-metal and the workspace); the engine reads completions_per_chunk from the
                 ring header, so the driver runs through _dgen_driver_shim.py
    feeder       exit 0; engine: every request PREFILL_DONE at its prompt length (the acks retired every chunk);
                 producer: one completion per served layer per chunk drained from the runner's completion ring
    coverage     two slots interleaved, a follow-up on a resident slot, (engine) a pulled-back last chunk
    table        tt-d-gen's read_table accepts it (32-token records, owner hosts); configs key_h<n> / value_h<n> on
                 the layers that have head n; slots and capacity; kv_table.layout_failures == []
    kv           every slot's final rows [0, end) through the table (kv_table.TableReader, every replica equal),
                 decoded and scored by tt-d-gen's kv_dump_compare (tensor_pcc >= max(0.93, thresholds.state)); the
                 pad rows of the last record zero
    reuse        at every chunk that starts below what its slot already holds: the rows below its start
                 byte-identical before and after it (kv_dump_compare bytecmp), and the rewritten rows of the first
                 layer bit-identical (its KV depends on the token and position only)
Gates K.1 (adapter).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from bringup_fw.testing.harness import device_timeout, impl_mode, spec

from models.demos.mimo_v26_flash_rl.tests.bringup.contract import _serving as S

ENGINE_TURNS = {"a": [8017, 9000, 56000], "b": [7001, 12000]}  # slot-like streams of follow-up turns
MODULE = "models.demos.mimo_v26_flash_rl.tests.bringup.contract.test_runner_contract"


def runner_parametrize(fn):
    """mesh_parametrize's ids and CPU-mode None, without opening the mesh (the runner subprocess opens it)."""
    if impl_mode() != "device":
        return pytest.mark.parametrize("mesh_device", [None], ids=["nodevice"])(fn)
    s = spec()
    return pytest.mark.parametrize("mesh_device", [tuple(s.mesh)], ids=["x".join(map(str, s.mesh))])(fn)


# ---------------------------------------------------------------- the parent (pytest)
def _engine_requests(tokens) -> list[dict]:
    out = []
    for stream, turns in ENGINE_TURNS.items():
        for k, n in enumerate(turns):
            out.append({"name": f"{stream}{k}", "token_ids": tokens[:n], "after": f"{stream}{k - 1}" if k else None})
    return sorted(out, key=lambda r: (r["after"] is not None, r["name"]))


def _runner_env(s, geo, out: Path, service_id: str, golden_dir: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PREFILL_")}
    env.update(
        PREFILL_MODEL=S.adapter_name(s),
        PREFILL_SP=str(geo["sp"]),
        PREFILL_TP=str(geo["tp"]),
        PREFILL_NUM_LAYERS=str(len(geo["layers"])),
        PREFILL_CHUNK_SIZE=str(geo["chunk"]),
        PREFILL_MAX_SEQ_LEN=str(geo["max_seq"]),
        PREFILL_NUM_USERS=str(geo["slots"]),
        PREFILL_LAYER_ACK_D2H="1",
        PREFILL_H2D_SERVICE_ID=service_id,
        PREFILL_MIGRATION_TABLE_PATH=str(out / "kv_table.pb"),
        PREFILL_MIGRATION_DEVICE_MAP_PATH=str(out / "device_map.json"),
        PREFILL_ENABLE_MIGRATION="0",
        PREFILL_MIGRATION_EXPORT_TO_FILE="0",
        PREFILL_TRACE_DIR=str(golden_dir),
        PREFILL_H2D_CONNECT_TIMEOUT="600",
        TT_METAL_OPERATION_TIMEOUT_SECONDS="600",  # the runner's blocking H2D read counts as a device op
    )
    if S.fabric_mode(s):
        env["PREFILL_FABRIC_MODE"] = S.fabric_mode(s)
    return env


@runner_parametrize
@device_timeout(spec())
def test_runner_contract(mesh_device, tmp_path):
    if mesh_device is None:
        pytest.skip("no mesh (CPU proof mode): the prefill runner runs on the device")
    from bringup_fw.core import metrics
    from bringup_fw.reference.golden import Golden
    from bringup_fw.testing import chunk_starts as CS
    from bringup_fw.testing import dgen_engine as DE

    from models.demos.common.prefill.runners.layer_completion_ring_reader import scheduler_shm_name

    s = spec()
    geo = S.geometry(s)
    from models.demos.common.prefill.adapter import ADAPTER_PATHS

    if S.adapter_name(s) not in ADAPTER_PATHS:
        pytest.fail(
            f"not built: no prefill adapter {S.adapter_name(s)!r} in ADAPTER_PATHS "
            "(models/demos/common/prefill/adapter.py)",
            pytrace=False,
        )
    rung = CS.target_rung(s)
    try:
        g = Golden.for_rung(s, rung["name"])
    except (FileNotFoundError, TypeError) as e:
        pytest.fail(f"not built: the target rung's golden ({e})", pytrace=False)

    kind, build, why = DE.feeder(s)
    if kind is None:
        pytest.fail(f"no feeder: {why}", pytrace=False)
    print(f"feeder: {build.describe() if build else f'{kind} ({why})'}", flush=True)
    if kind == DE.ENGINE:
        bad = S.transport_problems(build.repo, s.repo, geo["chunk"], geo["sp"])
        if bad:
            metrics.record("runner_contract_failures", len(bad))
            for b in bad:
                print(f"FAIL transport: {b}")
            pytest.fail(f"the engine and the runner do not share a transport: {bad[0]}", pytrace=False)

    out = Path(os.environ.get("BRINGUP_SCRATCH_DIR") or tmp_path) / f"runner_contract_{os.getpid()}"
    out.mkdir(parents=True, exist_ok=True)
    service_id = f"mimo_contract_{os.getpid()}"
    case = {
        "feeder": kind,
        "service_id": service_id,
        "golden": str(g.dir),
        "geometry": geo,
        "kv_dtype": S.kv_dtype(s),
        "dgen_repo": str(build.repo if build else S.dgen_repo(s)),
    }
    if kind == DE.ENGINE:
        plan = DE.write_plan(
            out / "plan.json",
            service_id=service_id,
            chunk=geo["chunk"],
            layers_per_chunk=len(geo["layers"]),
            sp=geo["sp"],
            max_slots=geo["slots"],
            max_seq=geo["max_seq"],
            requests=_engine_requests(g.tokens().tolist()),
            kv_block_size=S.BLOCK,
            ack_shm_name=scheduler_shm_name(service_id),  # the runner's completion ring (tt-d-gen #1600)
            connect_timeout_ms=600000,
            stall_s=900,
            run_timeout_s=int(s.get("box.test_timeout_s")) - 300,
        )
        case["driver"] = S.driver_shell(
            build, str(plan), str(out / "driver_out.json"), str(out / "driver.rc"), service_id
        )
    (out / "case.json").write_text(json.dumps(case, indent=1))
    print(f"runner contract: logs and results in {out}", flush=True)
    bound = int(s.get("box.test_timeout_s")) - 240
    with open(out / "runner.log", "w") as log:
        runner = subprocess.Popen(
            [sys.executable, "-m", MODULE, str(out)],
            cwd=s.repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=_runner_env(s, geo, out, service_id, g.dir),
        )
        try:
            rc = runner.wait(timeout=bound)
        except subprocess.TimeoutExpired:
            # Never kill a runner mid-op (untrained links): end its request loop with the sentinel, then wait.
            subprocess.run(
                [sys.executable, "-m", "bringup_fw.testing.dgen_engine", "shutdown", service_id],
                env=_runner_env(s, geo, out, service_id, g.dir),
                timeout=120,
            )
            rc = runner.wait(timeout=180)
            metrics.record("runner_contract_failures", 1)
            pytest.fail(
                f"the runner made no end in {bound} s; sent it the shutdown sentinel (see {out})", pytrace=False
            )
    res_path = out / "result.json"
    if not res_path.exists():
        tail = (out / "runner.log").read_text(errors="replace").splitlines()[-25:]
        pytest.fail(
            f"the runner exited {rc} without a result (see {out / 'runner.log'}):\n" + "\n".join(tail), pytrace=False
        )
    res = json.loads(res_path.read_text())
    for k, v in res.get("metrics", {}).items():
        metrics.record(k, v)
    metrics.record("runner_contract_failures", len(res["failures"]))
    metrics.record_json("runner_contract", res)
    for line in res.get("log", []):
        print(line)
    for f in res["failures"]:
        print(f"FAIL {f}")
    assert rc == 0 and not res["failures"], (
        f"runner exit {rc}; {len(res['failures'])} contract failures, first: " f"{(res['failures'] or ['(none)'])[0]}"
    )


# ---------------------------------------------------------------- the runner process
class _Recorder:
    """Wraps runtime.prefill_chunk: records every chunk the runner was sent and, at a chunk that starts below what its
    slot already holds, snapshots the slot's records around it before and after (the reuse checks)."""

    def __init__(self, runtime, ctx):
        self.rt, self.ctx = runtime, ctx
        self.calls, self.written, self.snaps = [], {}, []
        self.inner = runtime.prefill_chunk
        runtime.prefill_chunk = self

    def __call__(self, inp, kv, *, slot_id, actual_start, actual_end, request_id, **kw):
        held = self.written.get(slot_id, 0)
        self.calls.append((int(slot_id), int(actual_start), int(actual_end)))
        snap = None
        if actual_start < held:
            lo = 0 if held <= 16384 else max(0, actual_start - 1024)  # a follow-up: its whole prefix
            snap = {
                "slot": slot_id,
                "lo": lo,
                "start": actual_start,
                "hi": held,
                "before": self.ctx.dump(slot_id, lo, held),
            }
        r = self.inner(
            inp, kv, slot_id=slot_id, actual_start=actual_start, actual_end=actual_end, request_id=request_id, **kw
        )
        if snap is not None:
            snap["after"] = self.ctx.dump(slot_id, snap["lo"], snap["hi"])
            self.snaps.append(snap)
        self.written[slot_id] = max(held, actual_end) if actual_start <= held else actual_end
        return r


class _Ctx:
    """The exported table and the device map, read as the KV Manager reads them."""

    def __init__(self, case):
        from bringup_fw.testing import kv_table

        import ttnn

        self.ttnn, self.case = ttnn, case
        from bringup_fw.reference.golden import load_spec

        self.cfg = S.hf_config(load_spec())
        self.geo = case["geometry"]
        self.layers = self.geo["layers"]
        self.mesh = None
        self.table = ttnn.experimental.disaggregation.import_from_protobuf_file(
            os.environ["PREFILL_MIGRATION_TABLE_PATH"]
        )
        dm = json.loads(Path(os.environ["PREFILL_MIGRATION_DEVICE_MAP_PATH"]).read_text())
        self.dmap = {tuple(int(x) for x in k.split(":")): int(v) for k, v in dm.items()}
        self.reader = kv_table.TableReader(self.table, self.dmap)
        names = {self.table.config_name(c): c for c in range(self.table.num_configs())}
        self.configs = {}  # config id: (state name, head, width, layers)
        for name, layers in S.expected_configs(self.cfg, self.layers).items():
            if name in names:
                kind, head = S.CONFIG_RE.fullmatch(name).groups()
                self.configs[names[name]] = (kind, int(head), S.widths(self.cfg, layers[0])[kind], layers)

    def sync(self):
        if self.mesh is not None:
            self.ttnn.synchronize_device(self.mesh)

    def dump(self, slot, lo, hi) -> dict:
        """{config id: {(layer, pos): bytes}} for the records covering [lo, hi)."""
        self.sync()
        out = {}
        for cid, (_, _, _, layers) in self.configs.items():
            out[cid] = {
                (layer, p): self.reader.record(slot, layer, p, cid)
                for layer in layers
                for p in range(lo - lo % S.RECORD, hi, S.RECORD)
            }
        return out


def _check_table(ctx, tables, case) -> list[str]:
    from bringup_fw.testing import kv_table

    fails, geo, cfg = [], ctx.geo, ctx.cfg
    try:
        geom = tables.read_table(os.environ["PREFILL_MIGRATION_TABLE_PATH"])
    except Exception as e:  # noqa: BLE001  (tt-d-gen's own check: any refusal is the finding)
        return [f"table: tt-d-gen's read_table refuses it: {type(e).__name__}: {e}"]
    want = S.expected_configs(cfg, ctx.layers)
    got = {c["name"]: c for c in geom}
    if set(got) != set(want):
        fails.append(f"table: configs {sorted(got)}, want {sorted(want)} (key_h<n> / value_h<n> per KV head)")
    tile = S.TILE_BYTES[case["kv_dtype"]]
    for name, layers in want.items():
        c = got.get(name)
        if c is None:
            continue
        width = S.widths(cfg, layers[0])[S.CONFIG_RE.fullmatch(name).group(1)]
        if c["layers"] != layers:
            fails.append(f"table: config {name} covers layers {c['layers']}, want {layers}")
        if c["slots"] < geo["slots"]:
            fails.append(f"table: config {name} has {c['slots']} slots, the server runs {geo['slots']}")
        if c["capacity"] != geo["max_seq"]:
            fails.append(f"table: config {name} max_sequence_length {c['capacity']}, want {geo['max_seq']}")
        if c["chunk_bytes"] != width // 32 * tile:
            fails.append(
                f"table: config {name} record {c['chunk_bytes']} B, want {width // 32 * tile} "
                f"({width} wide, {case['kv_dtype']} tiles)"
            )
        try:
            tables.layout("mimo", 0, c["chunk_bytes"])
        except Exception as e:  # noqa: BLE001
            print(f"note: tt-d-gen's launch harness has no layout for {name} ({e}); its KV check cannot decode it")
    recs = kv_table.records_from_table(ctx.table, ctx.layers, geo["slots"], geo["max_seq"])
    fails += [
        f"table layout: {f}"
        for f in kv_table.layout_failures(recs, ctx.layers, geo["slots"], geo["max_seq"], ctx.table.num_configs())
    ]
    return fails


def _check_kv(ctx, kvdc, golden, final: dict, thr: float, dtype: str, metrics: dict) -> list[str]:
    import numpy as np

    fails, worst = [], 1.0
    for slot, end in sorted(final.items()):
        recs = ctx.dump(slot, 0, end)
        for cid, (kind, head, width, layers) in ctx.configs.items():
            for layer in layers:
                dev = kvdc.reassemble(recs[cid], layer, 0, end, dtype=dtype, width=width, storage="tile")
                want = golden.state(layer)[kind][head, :end].float().numpy()
                r = kvdc.tensor_pcc(dev, want, layer, thr, split=0)[0]
                worst = min(worst, r["pcc"] if r["pcc"] is not None else -1.0)
                if not r["passed"]:
                    fails.append(
                        f"kv: slot {slot} layer {layer} {kind} head {head} rows [0, {end}) pcc {r['pcc']} " f"< {thr}"
                    )
                if end % S.RECORD:
                    last = kvdc.decode_chunk(
                        recs[cid][(layer, end - end % S.RECORD)], dtype=dtype, width=width, storage="tile"
                    )[end % S.RECORD :]
                    if np.any(last != 0):
                        fails.append(
                            f"kv: slot {slot} layer {layer} {kind} head {head}: pad rows [{end}, "
                            f"{end - end % S.RECORD + S.RECORD}) of the last record not zero"
                        )
    metrics["runner_kv_pcc_min"] = worst
    return fails


def _check_reuse(ctx, kvdc, snaps, dtype) -> list[str]:
    fails = []
    for sn in snaps:
        for cid, (kind, head, width, layers) in ctx.configs.items():
            geom = {"dtype": dtype, "width": width, "storage": "tile"}
            a, b = sn["before"][cid], sn["after"][cid]
            if sn["start"] > sn["lo"]:
                r = kvdc.compare(a, b, method="bytecmp", layers=layers, start=sn["lo"], end=sn["start"], **geom)
                if r["status"] != "passed":
                    fails.append(
                        f"reuse: slot {sn['slot']} {kind} head {head}: rows [{sn['lo']}, {sn['start']}) below "
                        f"the chunk at {sn['start']} changed ({r['findings'][:2]})"
                    )
            first = [layers[0]] if layers[0] == ctx.layers[0] else []
            if first:
                r = kvdc.compare(a, b, method="bytecmp", layers=first, start=sn["start"], end=sn["hi"], **geom)
                if r["status"] != "passed":
                    fails.append(
                        f"reuse: slot {sn['slot']} layer {first[0]} {kind} head {head}: the rewrite of "
                        f"[{sn['start']}, {sn['hi']}) is not bit-identical ({r['findings'][:2]})"
                    )
    return fails


def _coverage(calls, snaps, feeder, geo) -> list[str]:
    fails = []
    slots_seen = [c[0] for c in calls]
    if len(set(slots_seen)) < 2 or not any(a != b for a, b in zip(slots_seen, slots_seen[1:])):
        fails.append(f"coverage: chunks did not interleave two slots ({slots_seen})")
    pulled = [sn for sn in snaps if sn["start"] == geo["max_seq"] - geo["chunk"]]
    if not [sn for sn in snaps if sn not in pulled]:
        fails.append("coverage: no follow-up turn started over a resident prefix")
    if feeder == "tt-d-gen engine" and not pulled:
        fails.append(f"coverage: no pulled-back last chunk at {geo['max_seq'] - geo['chunk']}")
    bad = [
        c
        for c in calls
        if c[2] <= c[1] or c[2] - c[1] > geo["chunk"] or (c[1] % S.RECORD and c[1] != geo["max_seq"] - geo["chunk"])
    ]
    if bad:
        fails.append(f"coverage: chunk headers outside tt-d-gen's rules {bad[:3]}")
    return fails


def _watch(proc, rc_file, service_id, log, done):
    """When the feeder fails, send the runner its shutdown sentinel so the request loop ends (bounded)."""
    while not done.is_set():
        rc = proc.poll()
        if rc is not None:
            try:
                rc = int(Path(rc_file).read_text().strip()) if rc_file and Path(rc_file).exists() else rc
            except ValueError:
                pass
            if rc != 0:
                log.append(f"feeder exited {rc}: sending the runner its shutdown sentinel")
                subprocess.run(
                    [sys.executable, "-m", "bringup_fw.testing.dgen_engine", "shutdown", service_id], timeout=300
                )
            return
        time.sleep(1.0)


def run_case(out: Path) -> int:
    from bringup_fw.reference.golden import Golden, load_spec
    from bringup_fw.testing import chunk_starts as CS

    from models.demos.common.prefill.runners import prefill_runner

    case = json.loads((out / "case.json").read_text())
    s, geo = load_spec(), case["geometry"]
    dtype = {"bfloat8_b": "bfp8", "bfloat16": "bf16"}[case["kv_dtype"]]
    res = {"feeder": case["feeder"], "failures": [], "metrics": {}, "log": []}
    original = prefill_runner.run_request_loop

    def checked_loop(runtime, kv_caches, *args, **kwargs):
        done = threading.Event()
        drain = proc = None
        try:
            ctx = _Ctx(case)
            ctx.mesh = getattr(runtime, "mesh_device", None)
            rec = _Recorder(runtime, ctx)
            feed_log = open(out / "feeder.log", "w")
            if case["feeder"] == "tt-d-gen engine":
                proc = subprocess.Popen(case["driver"], stdout=feed_log, stderr=subprocess.STDOUT)
                rc_file = str(out / "driver.rc")
            else:
                from models.demos.common.prefill.runners.layer_completion_ring_reader import (
                    connect_layer_completion_channel,
                    start_layer_completion_drain,
                )

                drain = start_layer_completion_drain(
                    connect_layer_completion_channel(60), num_layers=len(geo["layers"]), chunk_layers=len(geo["layers"])
                )
                env = dict(
                    os.environ,
                    PREFILL_PRODUCER_MAX_REQUESTS="5",
                    PREFILL_PRODUCER_CHUNKS="1,3",
                    PREFILL_PRODUCER_INTERLEAVE="round_robin",
                    PREFILL_PRODUCER_MULTI_TURN_PROB="1",
                    PREFILL_PRODUCER_MID_END_PROB="1",
                    PREFILL_PRODUCER_CHECK_PCC="0",
                    PREFILL_PRODUCER_SEED="7",
                    PREFILL_SEND_SHUTDOWN="1",
                )
                proc = subprocess.Popen(
                    [sys.executable, "-m", "models.demos.common.prefill.runners.prefill_producer"],
                    env=env,
                    stdout=feed_log,
                    stderr=subprocess.STDOUT,
                )
                rc_file = None
            threading.Thread(
                target=_watch, args=(proc, rc_file, case["service_id"], res["log"], done), daemon=True
            ).start()
            original(runtime, kv_caches, *args, **kwargs)
            rc = proc.wait(timeout=600)
            if rc_file and Path(rc_file).exists():
                rc = int(Path(rc_file).read_text().strip() or rc)
            res["metrics"]["runner_chunks"] = len(rec.calls)
            res["log"].append(f"feeder {case['feeder']} exit {rc}; chunks (slot, start, end): {rec.calls}")
            if rc != 0:
                res["failures"].append(f"feeder: {case['feeder']} exited {rc} (see {out / 'feeder.log'})")
            final = {}
            for slot, lo, hi in rec.calls:
                final[slot] = hi
            if case["feeder"] == "tt-d-gen engine":
                drv = json.loads((out / "driver_out.json").read_text()) if (out / "driver_out.json").exists() else {}
                for name, r in drv.get("requests", {}).items():
                    res["log"].append(f"request {name}: {r}")
                    if r.get("done_position") != r.get("prompt_len"):
                        res["failures"].append(
                            f"acks: request {name} done at {r.get('done_position')}, prompt " f"{r.get('prompt_len')}"
                        )
                res["failures"] += [f"engine: {e}" for e in drv.get("errors", [])]
            else:
                want = len(geo["layers"]) * len(rec.calls)
                from models.demos.common.prefill.runners.layer_completion_ring_reader import wait_for_layer_completions

                got = wait_for_layer_completions(drain, want, timeout_s=120)
                res["metrics"]["runner_layer_completions"] = got
                if got != want:
                    res["failures"].append(
                        f"acks: {got} layer completions for {len(rec.calls)} chunks x "
                        f"{len(geo['layers'])} layers = {want}"
                    )
            tables, kvdc = S.dgen_checks(Path(case["dgen_repo"]))
            res["failures"] += _coverage(rec.calls, rec.snaps, case["feeder"], geo)
            res["failures"] += _check_table(ctx, tables, case)
            if ctx.reader.replica_mismatch:
                res["failures"].append(f"kv: replicas of a device group differ at {ctx.reader.replica_mismatch[:3]}")
            res["failures"] += _check_reuse(ctx, kvdc, rec.snaps, dtype)
            res["failures"] += _check_kv(
                ctx, kvdc, Golden(Path(case["golden"])), final, CS.kv_threshold(s), dtype, res["metrics"]
            )
        except Exception as e:  # noqa: BLE001  (recorded as the finding; the runner still tears down)
            import traceback

            res["failures"].append(f"runner test: {type(e).__name__}: {e}")
            res["log"].append(traceback.format_exc())
        finally:
            done.set()
            if drain is not None:
                drain.stop()
            if proc is not None and proc.poll() is None:
                proc.wait(timeout=600)
            (out / "result.json").write_text(json.dumps(res, indent=1, default=str))

    prefill_runner.run_request_loop = checked_loop
    prefill_runner.main()
    return 0 if (out / "result.json").exists() else 1


if __name__ == "__main__":
    sys.exit(run_case(Path(sys.argv[1])))
