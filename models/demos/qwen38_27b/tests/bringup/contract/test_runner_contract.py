# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt-metal's prefill runner, driven by tt-d-gen's engine, on the served layers (serving_contract.md, "Adapter and
table"; gates adapter, K.1).

The runner (models/demos/common/prefill/runners/prefill_runner.py) runs as a subprocess with mock migration (the table
stays on disk: PREFILL_ENABLE_MIGRATION=0 in this tt-metal), host-sink acks, every brought-up layer. The feeder is
dgen_engine.feeder(): tt-d-gen's engine (the driver subprocess on the runner's H2D service and ack channel) when built,
else tt-metal's prefill_producer when the run allows it. The engine plan, all prefixes of the s56320 golden prompt:

    a1   10000 tokens, cold           [0, 5120) [5120, 10000)            mid-record end
    b    8017 tokens, cold            [0, 5120) [5120, 8017)              interleaved with a1 on the other slot
    a2   56320 tokens after a1        resident 9984 (prefix reuse, 64-token blocks), ..., [50944, 56064), then the
                                      last chunk pulled back to [51200, 56320) (tt-d-gen #430)

The pytest process never opens the mesh (the runner owns it): the KV is read out of band through the table the model
exports (kv_table.TableReader over UMD, as the KV Manager reads it) while the runner idles, then the shutdown sentinel
is sent. The geometry and the KV are checked with tt-d-gen's own tables.py (per-layer record bytes) and
kv_dump_compare.py (record decode, PCC, byte compare), loaded from the pinned checkout.

Pass: every request PREFILL_DONE at its length (the acks retire every chunk); a2 reused a1's slot from 9984; the table
geometry accepted (8 configs k_h0..k_h3 v_h0..v_h3, 32-token 8704-byte bfp8 records on the attention layers, GDN layers
unset); every record present, no overlap, replicas equal; KV vs the golden PCC >= max(serving.min_kv_pcc,
thresholds.state) for a2 over [0, 56320) (the reused prefix and the pulled-back chunk, whose rows depend on the GDN
state restored at 51200) and b over [0, 8017); b's pad rows [8017, 8032) zero; a's and b's records of the identical
first chunk [0, 5120) byte-identical.
Metrics: runner_contract_failures, runner_kv_pcc_min, runner_records_read, runner_feeder_engine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from bringup_fw.testing.harness import device_timeout, impl_mode, spec

S = spec()
RUNG = "s56320"
KV_BLOCK = 64
HEAD_DIM = 256
N_KV = 4
RECORD_BYTES = HEAD_DIM // 32 * 1088  # bfp8_b TILE: 8 tiles of 1088 B
CONFIG_NAMES = [f"{kind}_h{h}" for kind in ("k", "v") for h in range(N_KV)]
REQUESTS = (("a1", 10000, None), ("b", 8017, None), ("a2", 56320, "a1"))
SLOTS = 2
START_TIMEOUT_S = 3600  # weights + compile of the served layers before the runner serves
STALL_S = 900  # no new runner / driver output for this long: fail


def served() -> tuple[int, int]:
    from bringup_fw.testing.contract import served_layers

    return served_layers(S)


def runner_env(service_id: str, work: Path, trace_dir: Path) -> dict:
    first, n = served()
    env = dict(os.environ)
    env.update(
        PREFILL_MODEL=str(S.get("contract.adapter") or S.model),
        PREFILL_HF_MODEL=str(S.get("paths.hf")),
        PREFILL_SP=str(S.mesh[0]),
        PREFILL_TP=str(S.mesh[1]),
        PREFILL_NUM_LAYERS=str(first + n),
        PREFILL_CHUNK_SIZE=str(S.rung(RUNG)["chunk"]),
        PREFILL_MAX_SEQ_LEN=str(S.rung(RUNG)["seq"]),
        PREFILL_NUM_USERS=str(SLOTS),
        PREFILL_H2D_SERVICE_ID=service_id,
        PREFILL_LAYER_ACK_D2H="0",
        PREFILL_USE_TRACE="0",
        PREFILL_FABRIC_MODE="1d_ring",
        PREFILL_ENABLE_MIGRATION="0",  # mock migration: the table stays on disk for an offline reader
        PREFILL_MOCK_MIGRATION="1",
        PREFILL_MIGRATION_EXPORT_TO_FILE="0",
        PREFILL_MIGRATION_TABLE_PATH=str(work / "kv_table.pb"),
        PREFILL_MIGRATION_DEVICE_MAP_PATH=str(work / "device_map.json"),
        PREFILL_LAYER_COMPLETION_RING=f"/tt_prefill_layer_completion_ring_{service_id}",
        PREFILL_TRACE_DIR=str(trace_dir),
        TT_METAL_OPERATION_TIMEOUT_SECONDS="600",  # the safe runner's 5 s kills a runner idling on its H2D socket
    )
    return env


def acks_per_chunk() -> int:
    """What the runner routes per chunk: the served layers, or the KV-owning ones when the adapter says so."""
    from models.demos.common.prefill.adapter import get_adapter

    first, n = served()
    ids = getattr(get_adapter(str(S.get("contract.adapter") or S.model)), "kv_slot_layer_ids", lambda m: None)(
        first + n
    )
    return n if ids is None else sum(1 for i in ids if first <= i < first + n)


class Proc:
    """A subprocess with its output in a log file; ``stalled`` when the log has not grown for STALL_S."""

    def __init__(self, name: str, cmd: list[str], env: dict, log: Path, cwd: str):
        self.name, self.log = name, log
        self.fh = open(log, "w")
        self.p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=self.fh, stderr=subprocess.STDOUT)
        self.size, self.t = 0, time.monotonic()

    def stalled(self) -> bool:
        size = self.log.stat().st_size
        if size != self.size:
            self.size, self.t = size, time.monotonic()
        return time.monotonic() - self.t > STALL_S

    def tail(self, n: int = 25) -> str:
        return "\n".join(self.log.read_text(errors="replace").splitlines()[-n:])

    def stop(self) -> None:
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self.fh.close()


def wait_for(feeder: Proc, runner: Proc, deadline: float) -> None:
    """Until the feeder exits; fail at once when the runner dies or either stops making progress."""
    while feeder.p.poll() is None:
        if runner.p.poll() is not None:
            pytest.fail(
                f"the runner exited {runner.p.returncode} while the {feeder.name} ran:\n{runner.tail()}", pytrace=False
            )
        if feeder.stalled() and runner.stalled():
            pytest.fail(
                f"no progress for {STALL_S} s:\n{feeder.name}: {feeder.tail(8)}\nrunner: {runner.tail(8)}",
                pytrace=False,
            )
        if time.monotonic() > deadline:
            pytest.fail(f"the {feeder.name} did not finish in time:\n{feeder.tail(8)}", pytrace=False)
        time.sleep(2)
    if feeder.p.returncode != 0:
        pytest.fail(
            f"the {feeder.name} exited {feeder.p.returncode}:\n{feeder.tail()}\nrunner:\n{runner.tail(10)}",
            pytrace=False,
        )


def table_checks(table, kv_layers: list[int], n_layers: int, max_seq: int, harness_tables) -> list[str]:
    from bringup_fw.testing import kv_table

    fails = []
    names = [table.config_name(i) for i in range(table.num_configs())]
    if names != CONFIG_NAMES:
        fails.append(f"table configs {names}, want {CONFIG_NAMES}")
        return fails
    for c in range(table.num_configs()):
        cfg = table.config(c)
        got = (int(cfg.chunk_n_tokens), int(cfg.chunk_size_bytes), int(cfg.num_slots))
        if got != (32, RECORD_BYTES, SLOTS) or int(cfg.max_sequence_length) < max_seq:
            fails.append(
                f"config {names[c]}: (chunk_n_tokens, bytes, slots) {got}, max_seq "
                f"{cfg.max_sequence_length}; want (32, {RECORD_BYTES}, {SLOTS}), >= {max_seq}"
            )
    records = kv_table.records_from_table(table, range(n_layers), SLOTS, max_seq)
    fails += [f"layout: {f}" for f in kv_table.layout_failures(records, kv_layers, SLOTS, max_seq, len(names))]
    # tt-d-gen's own rule for the bytes one chunk position moves on each layer (config_groups + layer_pattern)
    pattern = [{"slot": ["kv"] if i in kv_layers else [], "repeat": 1} for i in range(n_layers)]
    cfg = SimpleNamespace(engine=SimpleNamespace(num_layers=n_layers, chunk_size_bytes=RECORD_BYTES), epochs=())
    ep = SimpleNamespace(
        table_spec={
            "config_groups": [{"role": "kv", "count": len(names), "chunk_size_bytes": RECORD_BYTES}],
            "layer_pattern": pattern,
        }
    )
    want = harness_tables.layer_chunk_bytes(cfg, ep)
    have = [sum(records[(0, i, 0, c)][1] for c in range(len(names))) for i in range(n_layers)]
    if have != want:
        fails.append(f"bytes per layer per 32-token position {have}, tt-d-gen tables.py wants {want}")
    return fails


def dgen_harness():
    """(tables, kv_dump_compare) from the pinned checkout. tables.py imports its package siblings, so it is loaded
    as kv_manager.tests.harness.tables when server_rules.harness cannot load it standalone."""
    from bringup_fw.testing import dgen_engine as DE
    from bringup_fw.testing import server_rules as R

    repo = DE.server_repo(S)
    try:
        return R.harness(repo)
    except ImportError:
        import importlib
        import importlib.util

        sys.path.insert(0, str(repo))
        try:
            tables = importlib.import_module("kv_manager.tests.harness.tables")
            kdc_spec = importlib.util.spec_from_file_location(
                "dgen_kv_dump_compare", repo / R.HARNESS_FILES["kv_dump_compare"][0]
            )
            kdc = importlib.util.module_from_spec(kdc_spec)
            kdc_spec.loader.exec_module(kdc)
            return tables, kdc
        finally:
            sys.path.remove(str(repo))


@pytest.mark.parametrize(
    "mesh_device",
    [None] if impl_mode() != "device" else ["runner"],
    ids=["nodevice" if impl_mode() != "device" else "x".join(map(str, S.mesh))],
)
@device_timeout(S)
def test_runner_contract(mesh_device):
    if mesh_device is None:
        pytest.skip("no mesh (BRINGUP_IMPL=reference): the runner test drives the device")
    import numpy as np
    from bringup_fw.core import metrics
    from bringup_fw.reference.golden import Golden
    from bringup_fw.testing import dgen_engine as DE
    from bringup_fw.testing import kv_table
    from bringup_fw.testing import server_rules as R
    from bringup_fw.testing.chunk_starts import kv_threshold
    from bringup_fw.testing.contract import kv_layers

    try:
        from models.demos.common.prefill.adapter import ADAPTER_PATHS
    except ImportError as e:
        pytest.fail(f"not built: tt-metal's prefill adapter registry does not import ({e})", pytrace=False)
    model_name = str(S.get("contract.adapter") or S.model)
    if model_name not in ADAPTER_PATHS:
        pytest.fail(
            f"not built: no prefill adapter {model_name!r} in ADAPTER_PATHS (serving_contract.md, 'Adapter "
            "and table')",
            pytrace=False,
        )

    kind, build, why = DE.feeder(S)
    if kind is None:
        pytest.fail(f"no feeder: {why}", pytrace=False)
    print(f"feeder: {build.describe() if build else f'{DE.PRODUCER} (engine not used: {why})'}", flush=True)
    metrics.record("runner_feeder_engine", int(kind == DE.ENGINE))

    g = Golden.for_rung(S, RUNG)
    chunk, max_seq = g.chunk, g.seq
    tokens = g.tokens().tolist()
    first, n = served()
    table_kv_layers = kv_layers(S)
    thr = kv_threshold(S)
    lpc = acks_per_chunk()
    tables, kdc = dgen_harness()
    fails = []

    work = Path(tempfile.mkdtemp(prefix="qwen38_runner_", dir=os.environ.get("BRINGUP_SCRATCH_DIR")))
    service_id = f"qwen38_contract_{os.getpid()}"
    env = runner_env(service_id, work, g.dir)
    repo = str(S.repo)
    runner = Proc(
        "runner",
        [sys.executable, "-m", "models.demos.common.prefill.runners.prefill_runner"],
        env,
        work / "runner.log",
        repo,
    )
    feeder = None
    deadline = time.monotonic() + START_TIMEOUT_S + 3 * 3600
    try:
        if kind == DE.ENGINE:
            plan = DE.write_plan(
                work / "plan.json",
                service_id=service_id,
                chunk=chunk,
                layers_per_chunk=lpc,
                sp=S.mesh[0],
                max_slots=SLOTS,
                max_seq=max_seq,
                kv_block_size=KV_BLOCK,
                requests=[{"name": nm, "token_ids": tokens[:ln], "after": after} for nm, ln, after in REQUESTS],
                connect_timeout_ms=START_TIMEOUT_S * 1000,
                stall_s=STALL_S,
                run_timeout_s=4 * 3600,
            )
            feeder = Proc(
                "engine driver",
                build.argv(str(plan), str(work / "out.json")),
                build.env(env),
                work / "driver.log",
                repo,
            )
        else:
            penv = dict(
                env,
                PREFILL_PRODUCER_MAX_REQUESTS=str(SLOTS),
                PREFILL_PRODUCER_CHUNKS="2",
                PREFILL_PRODUCER_INTERLEAVE="round_robin",
                PREFILL_PRODUCER_CHECK_PCC="0",
                PREFILL_SEND_SHUTDOWN="0",
                PREFILL_H2D_CONNECT_TIMEOUT=str(START_TIMEOUT_S),
            )
            feeder = Proc(
                "prefill_producer",
                [sys.executable, "-m", "models.demos.common.prefill.runners.prefill_producer"],
                penv,
                work / "producer.log",
                repo,
            )
        wait_for(feeder, runner, deadline)

        # what was served: {name: (slot, length)}
        if kind == DE.ENGINE:
            out = json.loads((work / "out.json").read_text())
            reqs = out["requests"]
            for nm, ln, _ in REQUESTS:
                r = reqs.get(nm, {})
                if r.get("done_position") != ln:
                    fails.append(
                        f"{nm}: PREFILL_DONE at {r.get('done_position')}, prompt {ln} (acks did not retire "
                        "every chunk)"
                    )
            if reqs["a2"].get("slot") != reqs["a1"].get("slot") or reqs["a2"].get("resident") != R.follow_up_resident(
                10000, max_seq, KV_BLOCK
            ):
                fails.append(f"a2 was not served from a1's prefix: a1 {reqs['a1']}, a2 {reqs['a2']}")
            plan_a2 = R.chunk_plan(max_seq, reqs["a2"].get("resident") or 0, chunk, max_seq)
            print(
                f"a2 plan (tt-d-gen's rule): {plan_a2[:2]} ... {plan_a2[-2:]}; pulled back: "
                f"{R.pulled_back(plan_a2)}",
                flush=True,
            )
            served_kv = {"a": (reqs["a2"]["slot"], max_seq), "b": (reqs["b"]["slot"], 8017)}
        else:
            served_kv = {"a": (0, 2 * chunk), "b": (1, 2 * chunk)}

        # the table the model exported, read as the KV Manager reads it
        import ttnn

        D = ttnn.experimental.disaggregation
        table = D.import_from_protobuf_file(env["PREFILL_MIGRATION_TABLE_PATH"])
        fails += table_checks(table, table_kv_layers, n, max_seq, tables)
        dmap = {
            tuple(int(x) for x in k.split(":")): int(v)
            for k, v in json.loads(Path(env["PREFILL_MIGRATION_DEVICE_MAP_PATH"]).read_text()).items()
        }
        reader = kv_table.TableReader(table, dmap)
        pcc_min, n_read = 1.0, 0
        dumps = {}
        for tag, (slot, length) in served_kv.items():
            for c, cname in enumerate(CONFIG_NAMES):
                d = work / "dump" / tag / cname
                reader.dump(d, slot, table_kv_layers, 0, length, config=c)
                n_read += len(table_kv_layers) * (R.ceil_to(length, 32) // 32)
                dumps[(tag, c)] = kdc.load_dump(d, slot=slot, chunk_bytes=RECORD_BYTES)
                for tl in table_kv_layers:
                    want = g.state(first + tl)["key" if cname[0] == "k" else "value"][int(cname[-1])][:length]
                    got = kdc.reassemble(dumps[(tag, c)], tl, 0, length, dtype="bfp8", width=HEAD_DIM)
                    res = kdc.tensor_pcc(got, want.float().numpy(), tl, thr, split=0)[0]
                    p = res["pcc"] if res["pcc"] is not None else float("nan")
                    pcc_min = min(pcc_min, p) if p == p else 0.0
                    print(
                        f"{'ok  ' if res['passed'] else 'FAIL'} slot {slot} ({tag}) layer {first + tl} {cname} "
                        f"[0, {length}): pcc {p:.6f} (>= {thr})",
                        flush=True,
                    )
                    if not res["passed"]:
                        fails.append(f"{tag}: layer {first + tl} {cname} pcc {p:.5f} < {thr}")
                    pad_end = R.ceil_to(length, 32)
                    if pad_end > length:
                        last = kdc.decode_chunk(dumps[(tag, c)][(tl, pad_end - 32)], dtype="bfp8", width=HEAD_DIM)
                        if np.any(last[length - (pad_end - 32) :] != 0):
                            fails.append(f"{tag}: layer {first + tl} {cname} pad rows [{length}, {pad_end}) not zero")
        if reader.replica_mismatch:
            fails.append(
                f"{len(reader.replica_mismatch)} records differ between the chips of their device group, "
                f"e.g. {reader.replica_mismatch[:3]}"
            )
        for c, cname in enumerate(CONFIG_NAMES):
            res = kdc.compare(
                dumps[("a", c)],
                dumps[("b", c)],
                method="bytecmp",
                layers=table_kv_layers,
                start=0,
                end=chunk,
                dtype="bfp8",
                width=HEAD_DIM,
            )
            if res["status"] != "passed":
                fails.append(
                    f"{cname}: the identical first chunk [0, {chunk}) differs between the two slots "
                    f"({len(res['findings'])} records): the recompute is not deterministic"
                )
        metrics.record("runner_kv_pcc_min", pcc_min)
        metrics.record("runner_records_read", n_read)

        # the engine never sends the runner's shutdown sentinel; send it as the next H2D connector (dgen_engine.py)
        subprocess.run(
            [sys.executable, "-m", "bringup_fw.testing.dgen_engine", "shutdown", service_id],
            env=env,
            cwd=repo,
            timeout=300,
            check=False,
        )
        try:
            runner.p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            fails.append("the runner did not exit after the shutdown sentinel")
        if runner.p.returncode not in (0, None):
            fails.append(f"the runner exited {runner.p.returncode}:\n{runner.tail(10)}")
    finally:
        if feeder is not None:
            feeder.stop()
        runner.stop()
    metrics.record("runner_contract_failures", len(fails))
    for f in fails:
        print(f"FAIL runner contract: {f}")
    print(f"logs: {work}")
    assert not fails, f"{len(fails)} problems, first: {fails[0]}"
