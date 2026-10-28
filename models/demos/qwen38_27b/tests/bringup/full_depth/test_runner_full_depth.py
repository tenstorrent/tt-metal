# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""K.1's runner test over the whole model: tt-d-gen's engine drives tt-metal's prefill runner through all 64 layers of
Qwen3.8-27B, and the KV of every attention layer is read back through the exported table and checked against a golden.

K.1's runner test (tests/bringup/contract/test_runner_contract.py, frozen with SC.1) serves the bring-up's layers, 0-3,
and checks them against the bring-up's golden, which holds those layers only. This test runs its flow on the whole
model. The checks are that test's own helpers, imported unchanged; three inputs differ:
    layers   the bring-up spec over every layer (48 Gated DeltaNet, 16 full attention); every other value is the
             bring-up's: mesh, chunk, max seq, thresholds, the serving answers
    golden   the GPU trace isl56320 (tests/e2e/goldens.py), which holds the K / V of every attention layer at every
             position, on the tokens of the bring-up's s56320 golden, so the requests are K.1's
    feeder   tt-d-gen's engine only: no fallback to tt-metal's prefill_producer, since the engine is what this tests

    scripts/run_safe_pytest.sh --run-all models/demos/qwen38_27b/tests/bringup/full_depth/test_runner_full_depth.py

Environment: a stepwise run's env.sh (the framework, BRINGUP_SPEC), with BRINGUP_DIR naming a bringup dir whose
dgen.json pins a built tt-d-gen that reads the runner's completion ring (1cf34388 or later) and whose
serving_answers.yaml holds the contract's answers (prefill-device-test: env=BRINGUP_DIR=<run>/bringup).

Requests (K.1's): a1 and b1, 8017 tokens each, admitted together on two slots, then a2 (56320 tokens) after a1 and b2
(10240) after b1. With prefix_reuse false every turn starts at 0 and no chunk is pulled back: 17 chunks.

Pass (test_runner_contract.py): the runner and the engine agree on the header and the ack channel; every request
reaches PREFILL_DONE at its prompt length (the engine retires a chunk only after all 64 layer acks); the chunks that ran
are tt-d-gen's plan; the table matches tt-d-gen's profile for 64 layers, and every record of the 16 attention layers is
present and well placed; each slot's final KV, read through the table over UMD, has PCC >= max(serving.min_kv_pcc,
thresholds.state) against the golden for every attention layer and head.
Not covered: the conv and recurrent state of the 48 Gated DeltaNet layers, which the table leaves out (the serving
answer fixed_state_migration), so nothing reads it out of band; the e2e test checks it at full depth, and a wrong state
shows in the KV of every later attention layer. No KV Manager: mock migration, as in K.1.

Output in $DEVICE_TEST_OUT/runner_full_depth (else a temp dir): the runner's and the engine's logs, the plan, the
table, the device map, every PCC (adhoc.json, bringup_fw.core.metrics) and summary.json. The KV records are dumped to a
local temp dir, one per slot, removed after its check.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from bringup_fw.core import metrics
from bringup_fw.core.spec import Spec
from bringup_fw.testing import dgen_engine as DE
from bringup_fw.testing import kv_table
from bringup_fw.testing import server_rules as R

from models.demos.qwen38_27b.tests.bringup.contract import _common as C
from models.demos.qwen38_27b.tests.bringup.contract import test_runner_contract as K
from models.demos.qwen38_27b.tests.e2e import goldens

S = Spec(dict(K.S.data, layers="all"), K.S.path)  # the bring-up spec over every layer
GOLDEN = "isl56320"
# the runner loads 64 layers (about 5 min) and serves 17 chunks, then about 266k KV records are read back over UMD and
# compared (K.1, one attention layer: 2 min); the prefill-device-test stage allows 3 h, its device cleanup included
TIMEOUT_S = 9000


class TraceKv:
    """The GPU trace as test_runner_contract reads its golden: dir, tokens(), state(layer)["key" / "value"]
    [n_kv, positions, head_dim] over every position (the check slices [0, n)). The check asks for every attention layer
    once per config and slot, so each layer is read once and kept (16 layers x 230 MB)."""

    def __init__(self, golden: goldens.Golden):
        self.golden, self.dir, self._kv = golden, golden.dir, {}

    def tokens(self):
        return self.golden.ids(self.golden.n_tokens)

    def state(self, layer: int) -> dict:
        if layer not in self._kv:
            self._kv[layer] = self.golden.kv(layer, self.golden.n_tokens)
        return self._kv[layer]


def _runner_env(s, svc: str, out: Path, slots: int, trace_dir: Path) -> dict:
    """test_runner_contract._runner_env over every layer of ``s``. Its PREFILL_TRACE_DIR names the bring-up's golden,
    which only the prefill_producer reads (the runner logs it): here the GPU trace."""
    first, n = s.layers()[0], len(s.layers())
    fabric = str(s.get("box.device_params", {}).get("fabric_config", "FABRIC_1D_RING")).replace("FABRIC_", "").lower()
    env = {k: v for k, v in os.environ.items() if not k.startswith("PREFILL_")}
    env.update(
        PREFILL_MODEL=K.MODEL,
        PREFILL_SP=str(s.mesh[0]),
        PREFILL_TP=str(s.mesh[1]),
        PREFILL_NUM_LAYERS=str(n),
        PREFILL_CHUNK_SIZE=str(int(s.get("target.chunk"))),
        PREFILL_MAX_SEQ_LEN=str(int(s.get("target.seq"))),
        PREFILL_NUM_USERS=str(slots),
        PREFILL_FABRIC_MODE=fabric,
        PREFILL_LAYER_ACK_D2H="1" if str(C.answer(s, "ack_mode", "d2h")).lower() == "d2h" else "0",
        PREFILL_H2D_SERVICE_ID=svc,
        PREFILL_ENABLE_MIGRATION="0",
        PREFILL_MOCK_MIGRATION="1",
        PREFILL_MIGRATION_EXPORT_TO_FILE="0",
        PREFILL_MIGRATION_TABLE_PATH=str(out / "kv_table.pb"),
        PREFILL_MIGRATION_DEVICE_MAP_PATH=str(out / "device_map.json"),
        PREFILL_TRACE_DIR=str(trace_dir),
        PREFILL_HF_MODEL=str(s.hf_dir),
        # known_issues.md: "The safe runner's 5 s dispatch timeout kills a prefill runner waiting for its next chunk"
        TT_METAL_OPERATION_TIMEOUT_SECONDS="600",
    )
    assert first == 0, "the runner serves layers from 0 on a single rank"
    return env


def _kv_summary(out: Path) -> dict:
    """Per request, the number of KV checks and the worst PCC (adhoc.json: runner_kv_<request>_<kind><head>_L<layer>);
    a NaN PCC counts as the worst."""
    p = out / f"{metrics.task_id()}.json"
    recorded = json.loads(p.read_text())["metrics"] if p.is_file() else {}
    by_request = {}
    for name, m in recorded.items():
        if name.startswith("runner_kv_"):
            by_request.setdefault(name.split("_")[2], []).append((m["value"], name))
    worst = {}
    for req, vals in sorted(by_request.items()):
        pcc, name = min(vals, key=lambda v: float("-inf") if v[0] != v[0] else v[0])
        worst[req] = {"checks": len(vals), "worst_pcc": pcc, "worst": name}
    return worst


@pytest.mark.timeout(TIMEOUT_S)
def test_runner_full_depth(monkeypatch):
    s = S
    from models.demos.common.prefill.adapter import ADAPTER_PATHS

    if K.MODEL not in ADAPTER_PATHS:
        C.not_built(f"no prefill adapter {K.MODEL!r} in models/demos/common/prefill/adapter.py ADAPTER_PATHS")
    kind, build, why = DE.feeder(s)
    if kind != DE.ENGINE:
        pytest.fail(f"no tt-d-gen engine ({why}): this test runs the engine only", pytrace=False)
    print(f"feeder: {build.describe()}", flush=True)
    repo = DE.server_repo(s)
    shared = K._shared_failures(s, repo, True)
    if shared:
        metrics.record("runner_shared_failures", len(shared))
        pytest.fail("runner and engine disagree: " + "; ".join(shared), pytrace=False)

    slots = int(C.answer(s, "slots", 2))
    reuse = C.answer(s, "prefix_reuse", False) in (True, "true", "True", 1)
    chunk, max_seq, sp = int(s.get("target.chunk")), int(s.get("target.seq")), s.mesh[0]
    layers = s.layers()
    attn = C.attn_layers(s)
    tc = C.text_config(s)
    n_kv, head_dim = int(tc["num_key_value_heads"]), int(tc["head_dim"])
    assert len(layers) == int(tc["num_hidden_layers"]) and attn, f"not the whole model: {len(layers)} layers, {attn}"
    dtype = "bf16" if str(C.answer(s, "kv_dtype", "bfp8")).lower() in ("bf16", "bfloat16") else "bfp8"
    record_bytes = head_dim // 32 * (1088 if dtype == "bfp8" else 2048)
    block = R.KV_BLOCK if reuse else 0
    g = TraceKv(goldens.load(GOLDEN))
    reqs = K._requests(g.tokens())
    out = Path(os.environ.get("DEVICE_TEST_OUT") or tempfile.mkdtemp(prefix="qwen38_full_depth_")) / "runner_full_depth"
    out.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(metrics.RESULTS_ENV, str(out))  # every metric of this test in out/adhoc.json
    metrics.reset(metrics.task_id())
    svc = f"qwen38_full_depth_{os.getpid()}"
    env = _runner_env(s, svc, out, slots, g.dir)
    deadline = time.monotonic() + int(s.get("box.test_timeout_s")) - 120
    fails = []
    summary = {
        "feeder": build.describe(),
        "tt_d_gen_sha": build.sha,
        "layers": len(layers),
        "attention_layers": attn,
        "slots": slots,
        "prefix_reuse": reuse,
        "kv_dtype": dtype,
        "kv_threshold": C.kv_threshold(s),
        "golden": str(g.dir),
        "seconds": {},
    }
    t0 = time.monotonic()

    runner = K._Proc("prefill_runner", [sys.executable, "-m", K.RUNNER], env, out / "runner.log")
    feeder = None
    try:
        plan = DE.write_plan(
            out / "plan.json",
            service_id=svc,
            chunk=chunk,
            layers_per_chunk=len(layers),
            sp=sp,
            max_slots=slots,
            max_seq=max_seq,
            requests=reqs,
            kv_block_size=block,
            connect_timeout_ms=1800000,
            run_timeout_s=3000,
            stall_s=K.STALL_S,
        )
        feeder = K._Proc(
            "tt-d-gen driver", build.argv(str(plan), str(out / "driver_out.json")), build.env(), out / "driver.log"
        )
        K._wait([runner, feeder], lambda: feeder.rc() == 0, deadline, f"{feeder.name} to finish")
        summary["seconds"]["load_and_prefill"] = round(time.monotonic() - t0, 1)

        # what ran: (slot, start, end) per chunk
        ran = K._ran_chunks(runner.log)
        metrics.record("runner_chunks", len(ran))
        rec = json.loads((out / "driver_out.json").read_text())
        if not rec.get("ok"):
            fails.append(f"engine: {rec.get('errors')}")
        by_name = {r["name"]: r for r in reqs}
        served = []  # (name, slot, plan) in PREFILL_DONE order
        finals = {}  # slot -> (request, prompt length, resident)
        for name, info in sorted(rec["requests"].items(), key=lambda kv: kv[1].get("done_s", 0)):
            n = len(by_name[name]["token_ids"])
            if info.get("done_position") != n:
                fails.append(f"acks: {name} PREFILL_DONE at {info.get('done_position')}, prompt {n}")
            res = int(info.get("resident") or 0)
            want_res = 0
            if reuse and by_name[name]["after"]:
                want_res = R.follow_up_resident(len(by_name[by_name[name]["after"]]["token_ids"]), n, block)
            if res != want_res:
                fails.append(f"starts: {name} admitted at resident {res}, tt-d-gen's rule gives {want_res}")
            plan_n = R.chunk_plan(n, res, chunk, max_seq)
            served.append((name, info.get("slot"), plan_n))
            print(
                f"{name}: slot {info.get('slot')}, resident {res}, chunks {plan_n}"
                + (" (last chunk pulled back)" if R.pulled_back(plan_n) else ""),
                flush=True,
            )
            finals[int(info["slot"])] = (name, n, res)
        fails += K._start_failures(ran, served)
        pulled = any(R.pulled_back(R.chunk_plan(n, r, chunk, max_seq)) for _, n, r in finals.values())
        metrics.record("runner_pulled_back", int(pulled))
        if not pulled:
            print(
                "pulled-back chunk: not sent (prefix_reuse false: every turn starts at 0 and max_seq is a "
                "multiple of the chunk)",
                flush=True,
            )
        summary["chunks"] = len(ran)
        summary["requests"] = [{"name": nm, "slot": sl, "chunks": len(pl)} for nm, sl, pl in served]

        # the table and the KV, read out of band while the runner still holds the cache
        t1 = time.monotonic()
        import ttnn

        table = ttnn.experimental.disaggregation.import_from_protobuf_file(env["PREFILL_MIGRATION_TABLE_PATH"])
        dmap = {
            tuple(int(x) for x in k.split(":")): int(v)
            for k, v in json.loads(Path(env["PREFILL_MIGRATION_DEVICE_MAP_PATH"]).read_text()).items()
        }
        tables, kdc = R.harness(repo)
        if table.num_configs() != 2 * n_kv:
            fails.append(f"table: {table.num_configs()} configs, want {2 * n_kv} (K heads, then V heads)")
        fails += K._table_profile_failures(tables, table, len(layers), slots, max_seq, record_bytes)
        layout = kv_table.layout_failures(
            kv_table.records_from_table(table, attn, slots, max_seq), attn, slots, max_seq, table.num_configs()
        )
        metrics.record("runner_layout_failures", len(layout))
        fails += [f"table: {f}" for f in layout]
        reader = kv_table.TableReader(table, dmap)
        for slot, (name, n, _res) in sorted(finals.items()):
            # the check dumps each config's records under BRINGUP_SCRATCH_DIR (2 GB for 56320 tokens): a local temp
            # dir here, removed with its slot
            with monkeypatch.context() as m, tempfile.TemporaryDirectory(prefix=f"qwen38_kv_{name}_") as d:
                m.setenv("BRINGUP_SCRATCH_DIR", d)
                fails += K._kv_failures(s, kdc, reader, table, g, slot, n, attn, n_kv, head_dim, dtype, name)
        if reader.replica_mismatch:
            fails.append(
                f"table: {len(reader.replica_mismatch)} records differ between replicas, e.g. "
                f"{reader.replica_mismatch[:3]}"
            )
        if reuse and len(finals) >= 2:
            (sa, (_, _, ra)), (sb, (_, _, rb)) = sorted(finals.items())[:2]
            lo = min(ra, rb)
            same = all(
                reader.record(sa, layer, p, c) == reader.record(sb, layer, p, c)
                for layer in attn
                for c in range(table.num_configs())
                for p in range(0, lo, kv_table.RECORD_TOKENS)
            )
            metrics.record("runner_reused_prefix_identical", int(same))
            if not same:
                fails.append(
                    f"reuse: rows [0, {lo}) differ between slots {sa} and {sb}; a follow-up may not touch "
                    "the reused prefix"
                )
        summary["seconds"]["table_and_kv"] = round(time.monotonic() - t1, 1)

    finally:
        # end the runner with its shutdown sentinel on the same H2D service; never kill it (a run stopped mid-op
        # leaves the ethernet links untrained)
        try:
            if feeder is not None and feeder.rc() is None:
                feeder.wait(600)
            if runner.rc() is None:
                snd = subprocess.run(
                    [sys.executable, "-m", "bringup_fw.testing.dgen_engine", "shutdown", svc],
                    env=dict(env, PREFILL_SP=str(sp)),
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if snd.returncode != 0:
                    fails.append(f"shutdown sentinel failed: {snd.stderr.strip()[-300:]}")
                rc = runner.wait(600)
                if rc not in (0, None):
                    fails.append(f"prefill_runner exited {rc} after the shutdown sentinel:\n{runner.tail()}")
        finally:
            summary["seconds"]["total"] = round(time.monotonic() - t0, 1)
            summary["kv"] = _kv_summary(out)
            summary["failures"] = fails
            (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"logs: {out}; summary: {json.dumps(summary['kv'])}", flush=True)
    C.report("runner_full_depth", fails)
