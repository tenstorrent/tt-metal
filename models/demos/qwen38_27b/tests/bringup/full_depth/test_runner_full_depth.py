# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""K.1's runner test over the whole model: tt-d-gen's engine drives tt-metal's prefill runner through all 64 layers of
Qwen3.8-27B, and the KV of every attention layer is read back through the exported table and checked against a golden.

K.1's runner test (tests/bringup/contract/test_runner_contract.py, frozen with K.1) serves the bring-up's layers, 0-3,
and checks them against the bring-up's s56320 golden, which holds those layers only. This test runs its flow on the
whole model, with that test's own helpers (runner_env, Proc, wait_for, acks_per_chunk, table_checks, dgen_harness,
imported unchanged; they read their module's spec, which this test points at the whole model for its duration) and
its checks. Three inputs differ:
    layers   the bring-up spec over every layer (spec layers "all"): 48 Gated DeltaNet and 16 full attention layers,
             the table's KV layers from the spec's block types, cross-checked against the checkpoint config's
             layer_types; every other value is the bring-up's: mesh, chunk, max seq, thresholds, serving answers
    golden   the GPU trace isl56320 (tests/e2e/goldens.py), which holds the K / V of every attention layer at every
             position, on the same 56320 tokens as the bring-up's s56320 golden, so the requests are K.1's
    feeder   tt-d-gen's engine only: no fallback to tt-metal's prefill_producer, since the engine is what this tests

    scripts/run_safe_pytest.sh --run-all models/demos/qwen38_27b/tests/bringup/full_depth/test_runner_full_depth.py

Environment: a stepwise run's env.sh (the framework on PYTHONPATH, BRINGUP_SPEC), with BRINGUP_DIR naming a bringup dir
whose dgen.json pins a built tt-d-gen (prefill-device-test: env=BRINGUP_DIR=<run>/bringup). Goldens:
$QWEN38_E2E_GOLDENS (goldens.py).

Requests (K.1's REQUESTS, prefixes of the golden prompt): a1 10000 tokens and b 8017, cold, interleaved on the two
slots, then a2 (56320 tokens) after a1 on a1's slot, resident 9984 (prefix reuse, 64-token blocks), its last chunk
pulled back.

Pass (as test_runner_contract.py): every request PREFILL_DONE at its prompt length (the engine retires a chunk only
after all 64 layer acks); a2 was served from a1's prefix; the table geometry over 64 layers (8 configs k_h0..v_h3,
32-token 8704-byte bfp8 records on the 16 attention layers, the Gated DeltaNet layers unset, the bytes per layer that
tt-d-gen's tables.py wants); every record present, no overlap, replicas equal; for a2 over [0, 56320) and b over
[0, 8017), every attention layer's K and V per head has PCC >= max(serving.min_kv_pcc, thresholds.state) against the
trace (256 checks); b's pad rows [8017, 8032) zero; a's and b's records of the identical first chunk [0, 5120)
byte-identical.
Not covered: the conv and recurrent state of the 48 Gated DeltaNet layers, which the table leaves out (state.fixed),
so nothing reads it out of band; the e2e test checks it at full depth, and a wrong state shows in the KV of every
later attention layer (a2's chunks past 9984 run from the state restored at a1's end). No KV Manager: mock migration.

Output in $DEVICE_TEST_OUT/runner_full_depth (else a temp dir): the runner's and the engine's logs, the plan, the
table, the device map, the metrics (adhoc.json, bringup_fw.core.metrics) and summary.json (every KV PCC, the worst per
request and per layer). The KV records are dumped to a local temp dir, one config of one request at a time, removed
once loaded.
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
from bringup_fw.core.spec import Spec
from bringup_fw.testing.harness import impl_mode

from models.demos.qwen38_27b.tests.bringup.contract import test_runner_contract as K
from models.demos.qwen38_27b.tests.e2e import goldens

S = Spec(dict(K.S.data, layers="all"), K.S.path)  # the bring-up spec over every layer
GOLDEN = "isl56320"
# the runner loads 64 layers (about 5 min) and serves 14 chunks, then about 257k KV records are read back over UMD and
# compared (K.1, one attention layer: 2 min); the prefill-device-test stage allows 3 h, its device cleanup included
TIMEOUT_S = 9000


class TraceKv:
    """The GPU trace as test_runner_contract reads its golden: dir, chunk, seq, tokens(), state(layer)["key" /
    "value"] [n_kv, positions, head_dim] over every position (the check slices [0, n)). The check asks for every
    attention layer once per config and request, so each layer is read once and kept (16 layers x 230 MB)."""

    def __init__(self, golden: goldens.Golden, chunk: int, seq: int):
        assert golden.n_tokens >= seq, f"{golden.dir}: {golden.n_tokens} tokens, the rung needs {seq}"
        self.golden, self.dir, self.chunk, self.seq, self._kv = golden, golden.dir, chunk, seq, {}

    def tokens(self):
        return self.golden.ids(self.seq)

    def state(self, layer: int) -> dict:
        if layer not in self._kv:
            self._kv[layer] = self.golden.kv(layer, self.seq)
        return self._kv[layer]


def checkpoint_attn_layers(n: int) -> list[int]:
    """The full-attention layers among [0, n) by the checkpoint config's layer_types."""
    from models.demos.qwen38_27b.reference.qwen38_ref import Config

    cfg = Config.load(str(S.hf_dir))
    assert cfg.num_layers == n == len(cfg.layer_types), f"not the whole model: {n} layers, the checkpoint has {cfg}"
    return [i for i in range(n) if cfg.layer_types[i] == "full_attention"]


@pytest.mark.parametrize(
    "mesh_device",
    [None] if impl_mode() != "device" else ["runner"],
    ids=["nodevice" if impl_mode() != "device" else "x".join(map(str, S.mesh))],
)
@pytest.mark.timeout(TIMEOUT_S)
def test_runner_full_depth(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("no mesh (BRINGUP_IMPL=reference): the runner test drives the device")
    import numpy as np
    from bringup_fw.core import metrics
    from bringup_fw.testing import dgen_engine as DE
    from bringup_fw.testing import kv_table
    from bringup_fw.testing import server_rules as R
    from bringup_fw.testing.chunk_starts import kv_threshold
    from bringup_fw.testing.contract import kv_layers, served_layers

    monkeypatch.setattr(K, "S", S)  # K's helpers read their module's spec: the whole model while this test runs
    try:
        from models.demos.common.prefill.adapter import ADAPTER_PATHS
    except ImportError as e:
        pytest.fail(f"not built: tt-metal's prefill adapter registry does not import ({e})", pytrace=False)
    model_name = str(S.get("contract.adapter") or S.model)
    if model_name not in ADAPTER_PATHS:
        pytest.fail(f"not built: no prefill adapter {model_name!r} in ADAPTER_PATHS", pytrace=False)
    kind, build, why = DE.feeder(S)
    if kind != DE.ENGINE:
        pytest.fail(f"no tt-d-gen engine ({why}): this test runs the engine only", pytrace=False)
    print(f"feeder: {build.describe()}", flush=True)

    first, n = served_layers(S)
    table_kv_layers = kv_layers(S)
    attn = checkpoint_attn_layers(n)
    assert first == 0, "the runner serves layers from 0 on a single rank"
    if [first + tl for tl in table_kv_layers] != attn or not attn:
        pytest.fail(
            f"the spec's KV layers {table_kv_layers} are not the checkpoint's full-attention layers {attn}",
            pytrace=False,
        )
    rung = S.rung(K.RUNG)
    chunk, max_seq = int(rung["chunk"]), int(rung["seq"])
    g = TraceKv(goldens.load(GOLDEN), chunk, max_seq)
    tokens = g.tokens().tolist()
    thr = kv_threshold(S)
    lpc = K.acks_per_chunk()
    tables, kdc = K.dgen_harness()
    fails = []

    work = (
        Path(os.environ.get("DEVICE_TEST_OUT") or tempfile.mkdtemp(prefix="qwen38_full_depth_")) / "runner_full_depth"
    )
    work.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(metrics.RESULTS_ENV, str(work))  # every metric of this test in work/adhoc.json
    metrics.reset(metrics.task_id())
    metrics.record("runner_feeder_engine", 1)
    service_id = f"qwen38_full_depth_{os.getpid()}"
    env = K.runner_env(service_id, work, g.dir)
    repo = str(S.repo)
    summary = {
        "feeder": build.describe(),
        "tt_d_gen_sha": build.sha,
        "layers": n,
        "attention_layers": attn,
        "requests": [list(r) for r in K.REQUESTS],
        "slots": K.SLOTS,
        "kv_threshold": thr,
        "golden": str(g.dir),
        "seconds": {},
    }
    checks = []  # (request, slot, layer, config, pcc)
    t0 = time.monotonic()
    runner = K.Proc(
        "runner",
        [sys.executable, "-m", "models.demos.common.prefill.runners.prefill_runner"],
        env,
        work / "runner.log",
        repo,
    )
    feeder = None
    deadline = time.monotonic() + TIMEOUT_S - 900  # leaves time for the read-back's tail and the shutdown
    try:
        plan = DE.write_plan(
            work / "plan.json",
            service_id=service_id,
            chunk=chunk,
            layers_per_chunk=lpc,
            sp=S.mesh[0],
            max_slots=K.SLOTS,
            max_seq=max_seq,
            kv_block_size=K.KV_BLOCK,
            requests=[{"name": nm, "token_ids": tokens[:ln], "after": after} for nm, ln, after in K.REQUESTS],
            connect_timeout_ms=K.START_TIMEOUT_S * 1000,
            stall_s=K.STALL_S,
            run_timeout_s=TIMEOUT_S,
        )
        feeder = K.Proc(
            "engine driver", build.argv(str(plan), str(work / "out.json")), build.env(env), work / "driver.log", repo
        )
        K.wait_for(feeder, runner, deadline)
        summary["seconds"]["load_and_prefill"] = round(time.monotonic() - t0, 1)

        # what was served: {tag: (slot, length)}
        out = json.loads((work / "out.json").read_text())
        reqs = out["requests"]
        for nm, ln, _ in K.REQUESTS:
            r = reqs.get(nm, {})
            if r.get("done_position") != ln:
                fails.append(
                    f"{nm}: PREFILL_DONE at {r.get('done_position')}, prompt {ln} (acks did not retire every chunk)"
                )
        lengths = {nm: ln for nm, ln, _ in K.REQUESTS}
        if reqs["a2"].get("slot") != reqs["a1"].get("slot") or reqs["a2"].get("resident") != R.follow_up_resident(
            lengths["a1"], max_seq, K.KV_BLOCK
        ):
            fails.append(f"a2 was not served from a1's prefix: a1 {reqs['a1']}, a2 {reqs['a2']}")
        plan_a2 = R.chunk_plan(max_seq, reqs["a2"].get("resident") or 0, chunk, max_seq)
        print(
            f"a2 plan (tt-d-gen's rule): {plan_a2[:2]} ... {plan_a2[-2:]}; pulled back: {R.pulled_back(plan_a2)}",
            flush=True,
        )
        served_kv = {"a": (reqs["a2"]["slot"], lengths["a2"]), "b": (reqs["b"]["slot"], lengths["b"])}
        summary["served"] = {nm: reqs.get(nm) for nm, _, _ in K.REQUESTS}

        # the table the model exported, read as the KV Manager reads it, while the runner still holds the cache
        t1 = time.monotonic()
        import ttnn

        D = ttnn.experimental.disaggregation
        table = D.import_from_protobuf_file(env["PREFILL_MIGRATION_TABLE_PATH"])
        fails += K.table_checks(table, table_kv_layers, n, max_seq, tables)
        dmap = {
            tuple(int(x) for x in k.split(":")): int(v)
            for k, v in json.loads(Path(env["PREFILL_MIGRATION_DEVICE_MAP_PATH"]).read_text()).items()
        }
        reader = kv_table.TableReader(table, dmap)
        pcc_min, n_read = 1.0, 0
        firsts = {}  # (tag, config) -> the records of [0, chunk), for the byte compare below
        for tag, (slot, length) in served_kv.items():
            for c, cname in enumerate(K.CONFIG_NAMES):
                # one config of one request at a time (a2: 16 layers x 1760 records, 245 MB), in a local temp dir
                with tempfile.TemporaryDirectory(prefix=f"qwen38_kv_{tag}_{cname}_") as d:
                    reader.dump(d, slot, table_kv_layers, 0, length, config=c)
                    dump = kdc.load_dump(d, slot=slot, chunk_bytes=K.RECORD_BYTES)
                n_read += len(table_kv_layers) * (R.ceil_to(length, 32) // 32)
                for tl in table_kv_layers:
                    want = g.state(first + tl)["key" if cname[0] == "k" else "value"][int(cname[-1])][:length]
                    got = kdc.reassemble(dump, tl, 0, length, dtype="bfp8", width=K.HEAD_DIM)
                    res = kdc.tensor_pcc(got, want.float().numpy(), tl, thr, split=0)[0]
                    p = res["pcc"] if res["pcc"] is not None else float("nan")
                    pcc_min = min(pcc_min, p) if p == p else 0.0
                    checks.append((tag, slot, first + tl, cname, p))
                    print(
                        f"{'ok  ' if res['passed'] else 'FAIL'} slot {slot} ({tag}) layer {first + tl} {cname} "
                        f"[0, {length}): pcc {p:.6f} (>= {thr})",
                        flush=True,
                    )
                    if not res["passed"]:
                        fails.append(f"{tag}: layer {first + tl} {cname} pcc {p:.5f} < {thr}")
                    pad_end = R.ceil_to(length, 32)
                    if pad_end > length:
                        last = kdc.decode_chunk(dump[(tl, pad_end - 32)], dtype="bfp8", width=K.HEAD_DIM)
                        if np.any(last[length - (pad_end - 32) :] != 0):
                            fails.append(f"{tag}: layer {first + tl} {cname} pad rows [{length}, {pad_end}) not zero")
                firsts[(tag, c)] = {key: rec for key, rec in dump.items() if key[1] < chunk}
                del dump
        if reader.replica_mismatch:
            fails.append(
                f"{len(reader.replica_mismatch)} records differ between the chips of their device group, "
                f"e.g. {reader.replica_mismatch[:3]}"
            )
        for c, cname in enumerate(K.CONFIG_NAMES):
            res = kdc.compare(
                firsts[("a", c)],
                firsts[("b", c)],
                method="bytecmp",
                layers=table_kv_layers,
                start=0,
                end=chunk,
                dtype="bfp8",
                width=K.HEAD_DIM,
            )
            if res["status"] != "passed":
                fails.append(
                    f"{cname}: the identical first chunk [0, {chunk}) differs between the two slots "
                    f"({len(res['findings'])} records): the recompute is not deterministic"
                )
        del firsts
        metrics.record("runner_kv_pcc_min", pcc_min)
        metrics.record("runner_kv_checks", len(checks))
        metrics.record("runner_records_read", n_read)
        summary["seconds"]["table_and_kv"] = round(time.monotonic() - t1, 1)

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
        summary["seconds"]["total"] = round(time.monotonic() - t0, 1)
        summary["kv"] = kv_summary(checks)
        summary["failures"] = fails
        (work / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    metrics.record("runner_contract_failures", len(fails))
    for f in fails:
        print(f"FAIL runner full depth: {f}")
    print(f"logs: {work}; KV: {json.dumps(summary['kv']['by_request'])}", flush=True)
    assert not fails, f"{len(fails)} problems, first: {fails[0]}"


def kv_summary(checks: list) -> dict:
    """The KV checks by request and by layer: their number and the worst PCC (a NaN counts as the worst)."""

    def worst(rows):
        p, name = min(((r[4], f"{r[0]} L{r[2]} {r[3]}") for r in rows), key=lambda v: -1e9 if v[0] != v[0] else v[0])
        return {"checks": len(rows), "worst_pcc": None if p != p else p, "worst": name}

    by_request, by_layer = {}, {}
    for r in checks:
        by_request.setdefault(r[0], []).append(r)
        by_layer.setdefault(r[2], []).append(r)
    return {
        "checks": len(checks),
        "by_request": {k: worst(v) for k, v in sorted(by_request.items())},
        "by_layer": {str(k): worst(v) for k, v in sorted(by_layer.items())},
        "all": [[*r[:4], None if r[4] != r[4] else r[4]] for r in checks],
    }
