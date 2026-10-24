# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, the real prefill runner (serving_contract.md "Adapter and table", "Acks", "Deployment"; K.2).

tt-metal's prefill_runner as a subprocess (PREFILL_MODEL=qwen38_27b, the served layers, mock migration: the table is
written to disk, no KV Manager), device-to-host acks (the ack_mode answer), fed by bringup_fw.testing.dgen_engine
.feeder(): tt-d-gen's engine (the driver on the runner's H2D service and ack channel) when built, else tt-metal's
prefill_producer when the run allows it. The test prints which one ran.

Plan (prompts are prefixes of the s56320 golden's tokens, so every slot's KV [0, n) is the golden's):
  a1, b1   8017 tokens each, admitted together on two slots (cold, last chunk ends mid-record)
  a2       after a1: 56320 tokens (a follow-up turn on the holder slot; with prefix_reuse it starts at 7936 and its last
           chunk is pulled back to 51200, prefill_writer.cpp:63)
  b2       after b1: 10240 tokens
With prefix_reuse false (kv_block_size 0) every turn starts at 0 and no chunk is pulled back (printed as skipped).

Pass:
  shared   the runner and the engine agree on the 12-byte header and the ack channel (checked before any device work)
  acks     every request reaches PREFILL_DONE at its prompt length (the engine retires a chunk only on all its acks)
  starts   the chunks the runner ran are tt-d-gen's plan for each request (server_rules.chunk_plan)
  table    the geometry tt-d-gen's harness derives (kv_manager/tests/harness/tables.py table_profile) matches the
           table; every (slot, attention layer, 32-token position, config) record present, one size per config,
           64-byte aligned, inside its bank, none overlapping (kv_table.layout_failures)
  kv       each slot's final KV, read through the table over UMD (kv_table.TableReader) and compared with tt-d-gen's
           kv_dump_compare, vs the golden: PCC >= max(serving.min_kv_pcc, thresholds.state) for every layer and head
  reuse    (prefix_reuse) rows below the reused start byte-identical on both holder slots (both from the same cold
           plan; a follow-up may not touch them)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from bringup_fw.core import metrics
from bringup_fw.reference.golden import Golden
from bringup_fw.testing import dgen_engine as DE
from bringup_fw.testing import kv_table
from bringup_fw.testing import server_rules as R
from bringup_fw.testing.harness import device_timeout, impl_mode, spec

from models.demos.qwen38_27b.tests.bringup.contract import _common as C

S = spec()
MODEL = "qwen38_27b"
RUNG = "s56320"
RUNNER = "models.demos.common.prefill.runners.prefill_runner"
PRODUCER = "models.demos.common.prefill.runners.prefill_producer"
STALL_S = 900  # no new runner output and no feeder progress for this long: fail


def _scratch() -> Path:
    d = os.environ.get("BRINGUP_SCRATCH_DIR")
    return Path(tempfile.mkdtemp(prefix="qwen38_runner_", dir=d or None))


def _shared_failures(s, repo: Path, engine: bool) -> list[str]:
    """The three things the runner and the engine share, read from both sources at their pinned commits."""
    fails = []
    runner_src = (Path(s.repo) / "models/demos/common/prefill/runners/prefill_runner.py").read_text()
    m = re.search(r"^METADATA_SIZE_BYTES\s*=\s*(\d+)", runner_src, re.M)
    if not m or int(m.group(1)) != 12:
        fails.append(f"runner metadata size {m and m.group(1)} != the engine's 12-byte PrefillMetadata")
    if not engine:
        return fails
    types_src = (repo / "engine/include/engine/pipeline/pipeline_types.hpp").read_text()
    if "sizeof(PrefillMetadata) == 12" not in types_src:
        fails.append("tt-d-gen PrefillMetadata is no longer 12 bytes (pipeline_types.hpp)")
    pipe_src = (repo / "engine/src/pipeline/prefill_pipeline.cpp").read_text()
    ring_src = (Path(s.repo) / "models/demos/common/prefill/runners/layer_completion_ring_reader.py").read_text()
    engine_counter = "InterProcessCounterChannel::connect" in pipe_src
    runner_ring = "/tt_prefill_layer_completions_" in ring_src and "scheduler_shm_name(service_id)" in runner_src
    if engine_counter and runner_ring:
        fails.append(
            "ack channel: the engine attaches the counter channel /tt_prefill_layer_acks_<svc> "
            "(prefill_pipeline.cpp), this runner publishes the completion ring /tt_prefill_layer_completions_<svc> "
            "(layer_completion_ring_reader.py); tt-d-gen #1600 (1cf34388) consumes the ring: re-pin tt-d-gen "
            "(serving_contract.md, Questions: engine_pin)"
        )
    return fails


def _requests(tokens) -> list[dict]:
    t = [int(x) for x in tokens]
    return [
        {"name": "a1", "token_ids": t[:8017], "after": None},
        {"name": "b1", "token_ids": t[:8017], "after": None},
        {"name": "a2", "token_ids": t[:56320], "after": "a1"},
        {"name": "b2", "token_ids": t[:10240], "after": "b1"},
    ]


def _runner_env(s, svc: str, out: Path, slots: int) -> dict:
    first, n = s.layers()[0], len(s.layers())
    fabric = str(s.get("box.device_params", {}).get("fabric_config", "FABRIC_1D_RING")).replace("FABRIC_", "").lower()
    env = {k: v for k, v in os.environ.items() if not k.startswith("PREFILL_")}
    env.update(
        PREFILL_MODEL=MODEL,
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
        PREFILL_TRACE_DIR=str(Golden.for_rung(s, RUNG).dir),
        PREFILL_HF_MODEL=str(s.hf_dir),
        # known_issues.md: "The safe runner's 5 s dispatch timeout kills a prefill runner waiting for its next chunk"
        TT_METAL_OPERATION_TIMEOUT_SECONDS="600",
    )
    assert first == 0, "the runner serves layers from 0 on a single rank"
    return env


class _Proc:
    """A subprocess with its log; fails at once when it exits non-zero."""

    def __init__(self, name, cmd, env, log: Path):
        self.name, self.log = name, log
        self.f = open(log, "w")
        self.p = subprocess.Popen(cmd, env=env, stdout=self.f, stderr=subprocess.STDOUT, cwd=S.repo)

    def rc(self):
        return self.p.poll()

    def size(self) -> int:
        return self.log.stat().st_size if self.log.exists() else 0

    def tail(self, n=12) -> str:
        return "\n".join(self.log.read_text(errors="replace").splitlines()[-n:])

    def wait(self, timeout):
        try:
            return self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None


def _wait(watch: list, done, deadline: float, what: str):
    """Until done() is true; fail when a watched process exits non-zero, nothing makes progress, or time runs out."""
    last, sizes = time.monotonic(), [w.size() for w in watch]
    while not done():
        for w in watch:
            rc = w.rc()
            if rc not in (None, 0):
                pytest.fail(f"{w.name} exited {rc} while waiting for {what}:\n{w.tail()}", pytrace=False)
        now_sizes = [w.size() for w in watch]
        if now_sizes != sizes:
            sizes, last = now_sizes, time.monotonic()
        if time.monotonic() - last > STALL_S:
            pytest.fail(
                f"no progress for {STALL_S} s waiting for {what}:\n" + "\n".join(w.tail(6) for w in watch),
                pytrace=False,
            )
        if time.monotonic() > deadline:
            pytest.fail(f"timed out waiting for {what}", pytrace=False)
        time.sleep(2)


def _ran_chunks(log: Path) -> list[tuple[int, int, int]]:
    """(slot, start, end) of every CHUNK_START line of the runner log, in order."""
    pat = re.compile(r"CHUNK_START c=\d+ \S+ slot=(\d+) \[(\d+),(\d+)\)")
    return [tuple(int(x) for x in m.groups()) for m in pat.finditer(log.read_text(errors="replace"))]


def _start_failures(ran, served) -> list[str]:
    """The chunks each request ran vs its plan. served: (name, slot, plan) in PREFILL_DONE order.

    A slot serves its requests one after another (a follow-up is admitted only after the request before it is
    PREFILL_DONE), so a slot's CHUNK_START sequence must be its requests' plans back to back, in done order: each
    request is compared with its own segment, and nothing may be missing or extra on the slot.
    """
    fails = []
    for slot in sorted({sl for sl, _, _ in ran} | {sl for _, sl, _ in served}):
        seq = [(a, b) for sl, a, b in ran if sl == slot]
        i = 0
        for name, sl, plan in served:
            if sl != slot:
                continue
            got = seq[i : i + len(plan)]
            if got != plan:
                fails.append(f"starts: {name} on slot {slot}: ran {got}, plan {plan}")
            i += len(plan)
        if i != len(seq):
            fails.append(f"starts: slot {slot} ran {len(seq)} chunks, its requests' plans have {i}: {seq[i:]} extra")
    return fails


def _table_profile_failures(tables, table, layers: int, slots: int, max_seq: int, record_bytes: int) -> list[str]:
    """tt-d-gen's own table profile for this geometry vs what each config of the exported table declares."""
    engine = SimpleNamespace(
        num_layers=layers,
        epochs=[],
        max_seq_tokens=max_seq,
        chunk_size_bytes=record_bytes,
        chunk_tokens=kv_table.RECORD_TOKENS,
        dram_base=None,
    )
    ep = SimpleNamespace(id=0, name="prefill-0", num_slots=slots, hosts=["localhost"], table_spec={})
    prof = tables.table_profile(SimpleNamespace(engine=engine), ep)
    fails = []
    for c in range(table.num_configs()):
        cfg = table.config(c)
        got = {
            "layers": cfg.num_layers,
            "max_seq_len": cfg.max_sequence_length,
            "slots": cfg.num_slots,
            "chunk_n_tokens": cfg.chunk_n_tokens,
            "chunk_size_bytes": cfg.chunk_size_bytes,
        }
        bad = {k: (got[k], prof[k]) for k in got if int(got[k]) != int(prof[k])}
        if bad:
            fails.append(f"table config {c}: {bad} (got, tt-d-gen profile)")
    return fails


def _kv_failures(
    s, kdc, reader, table, g, slot: int, n: int, layers, n_kv: int, head_dim: int, dtype: str, tag: str
) -> list[str]:
    """One slot's KV [0, n) through the table vs the golden, with tt-d-gen's kv_dump_compare."""
    import numpy as np

    thr, fails = C.kv_threshold(s), []
    for c in range(2 * n_kv):
        kind, head = ("key", c) if c < n_kv else ("value", c - n_kv)
        d = Path(tempfile.mkdtemp(prefix=f"kv_{tag}_c{c}_", dir=os.environ.get("BRINGUP_SCRATCH_DIR") or None))
        reader.dump(d, slot, layers, 0, n, config=c)
        size = int(table.lookup(layers[0], 0, slot, c).size_bytes)
        dump = kdc.load_dump(str(d), slot=slot, chunk_bytes=size)
        for layer in layers:
            got = kdc.reassemble(dump, layer, 0, n, dtype=dtype, width=head_dim, storage="tile")
            want = g.state(layer)[kind][head, :n].float().numpy().astype(np.float32)
            res = kdc.tensor_pcc(got, want, layer, thr, split=0)[0]
            p = res["pcc"] if res["pcc"] is not None else float("nan")
            metrics.record(f"runner_kv_{tag}_{kind}{head}_L{layer:02d}", p)
            print(
                f"{'ok  ' if res['passed'] else 'FAIL'} {tag} slot {slot} layer {layer} {kind} head {head} rows "
                f"[0, {n}): pcc {p:.6f} (>= {thr})",
                flush=True,
            )
            if not res["passed"]:
                fails.append(f"{tag}: slot {slot} layer {layer} {kind} head {head} pcc {p:.5f} < {thr}")
    return fails


@device_timeout(S)
@pytest.mark.parametrize("mesh_device", [None] if impl_mode() != "device" else ["runner"], ids=["box"])
def test_runner_contract(mesh_device):
    if mesh_device is None:
        return  # CPU proof mode: the runner opens the mesh itself
    s = S
    from models.demos.common.prefill.adapter import ADAPTER_PATHS

    if MODEL not in ADAPTER_PATHS:
        C.not_built(f"no prefill adapter {MODEL!r} in models/demos/common/prefill/adapter.py ADAPTER_PATHS")
    kind, build, why = DE.feeder(s)
    if kind is None:
        pytest.fail(f"no feeder: {why}", pytrace=False)
    print(f"feeder: {build.describe() if build else DE.PRODUCER + ' (' + why + ')'}", flush=True)
    repo = DE.server_repo(s)
    shared = _shared_failures(s, repo, kind == DE.ENGINE)
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
    dtype = "bf16" if str(C.answer(s, "kv_dtype", "bfp8")).lower() in ("bf16", "bfloat16") else "bfp8"
    record_bytes = head_dim // 32 * (1088 if dtype == "bfp8" else 2048)
    block = R.KV_BLOCK if reuse else 0
    g = Golden.for_rung(s, RUNG)
    reqs = _requests(g.tokens())
    out = _scratch()
    svc = f"qwen38_contract_{os.getpid()}"
    env = _runner_env(s, svc, out, slots)
    deadline = time.monotonic() + int(s.get("box.test_timeout_s")) - 120
    fails = []

    runner = _Proc("prefill_runner", [sys.executable, "-m", RUNNER], env, out / "runner.log")
    feeder = None
    try:
        if kind == DE.ENGINE:
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
                stall_s=STALL_S,
            )
            feeder = _Proc(
                "tt-d-gen driver", build.argv(str(plan), str(out / "driver_out.json")), build.env(), out / "driver.log"
            )
        else:
            penv = dict(
                env,
                PREFILL_PRODUCER_INTERLEAVE="round_robin",
                PREFILL_PRODUCER_MAX_REQUESTS=str(2 * slots),
                PREFILL_PRODUCER_CHUNKS="2,3",
                PREFILL_PRODUCER_MID_END_PROB="1",
                PREFILL_PRODUCER_MULTI_TURN_PROB="1" if reuse else "0",
                PREFILL_PRODUCER_CHECK_PCC="0",
                PREFILL_SEND_SHUTDOWN="0",
                PREFILL_H2D_CONNECT_TIMEOUT="1800",
                PREFILL_PRODUCER_SEED="1234",
            )
            feeder = _Proc("prefill_producer", [sys.executable, "-m", PRODUCER], penv, out / "producer.log")
        _wait([runner, feeder], lambda: feeder.rc() == 0, deadline, f"{feeder.name} to finish")

        # what ran: (slot, start, end) per chunk
        ran = _ran_chunks(runner.log)
        metrics.record("runner_chunks", len(ran))
        finals = {}  # slot -> (request, prompt length, resident)
        if kind == DE.ENGINE:
            rec = json.loads((out / "driver_out.json").read_text())
            if not rec.get("ok"):
                fails.append(f"engine: {rec.get('errors')}")
            by_name = {r["name"]: r for r in reqs}
            served = []  # (name, slot, plan) in PREFILL_DONE order
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
            fails += _start_failures(ran, served)
            pulled = any(R.pulled_back(R.chunk_plan(n, r, chunk, max_seq)) for _, n, r in finals.values())
            metrics.record("runner_pulled_back", int(pulled))
            if not pulled:
                print(
                    "pulled-back chunk: not sent (prefix_reuse false: every turn starts at 0 and max_seq is a "
                    "multiple of the chunk)",
                    flush=True,
                )
        else:
            for sl, a, b in ran:  # the producer's last chunk per slot ends at that slot's resident length
                finals[sl] = (f"slot{sl}", b, 0)

        # the table and the KV, read out of band while the runner still holds the cache
        import ttnn

        table = ttnn.experimental.disaggregation.import_from_protobuf_file(env["PREFILL_MIGRATION_TABLE_PATH"])
        dmap = {
            tuple(int(x) for x in k.split(":")): int(v)
            for k, v in json.loads(Path(env["PREFILL_MIGRATION_DEVICE_MAP_PATH"]).read_text()).items()
        }
        tables, kdc = R.harness(repo)
        if table.num_configs() != 2 * n_kv:
            fails.append(f"table: {table.num_configs()} configs, want {2 * n_kv} (K heads, then V heads)")
        fails += _table_profile_failures(tables, table, len(layers), slots, max_seq, record_bytes)
        layout = kv_table.layout_failures(
            kv_table.records_from_table(table, attn, slots, max_seq), attn, slots, max_seq, table.num_configs()
        )
        metrics.record("runner_layout_failures", len(layout))
        fails += [f"table: {f}" for f in layout]
        reader = kv_table.TableReader(table, dmap)
        for slot, (name, n, _res) in sorted(finals.items()):
            fails += _kv_failures(s, kdc, reader, table, g, slot, n, attn, n_kv, head_dim, dtype, name)
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

    finally:
        # end the runner with its shutdown sentinel on the same H2D service; never kill it (a run stopped mid-op
        # leaves the ethernet links untrained)
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
    print(f"logs: {out}", flush=True)
    C.report("runner", fails)
