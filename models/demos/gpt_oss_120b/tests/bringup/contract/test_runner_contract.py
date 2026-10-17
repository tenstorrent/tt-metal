# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, runner test (serving_contract.md "## Adapter and table"; gates adapter, the hackathon's K.2).

Written by SC.1 from tt-d-gen's code; frozen with K.1. tt-metal's real prefill runner runs as a subprocess with mock
migration (the table stays on disk, no KV Manager), layer acks to the host ring, every brought-up layer, two slots
interleaved, a follow-up turn on each slot over a reused prefix, and a last chunk pulled back to max_seq - chunk
(tt-d-gen #430). The feeder is tt-d-gen's own engine (testing/dgen_engine.py: the driver on the runner's H2D service
and ack channel) when built, else tt-metal's prefill_producer wire calls on the same plan when the run allows it.

The session (server_rules.interleave at kv_block_size 64; the engine's mock pipeline reproduces it):
    turn 1, both slots   the same golden prefix of 12017 tokens: chunks (0, 5120) (5120, 10240) (10240, 12017)
    turn 2, slot a       56017 tokens over the reused prefix: resident 11968, chunks from 11968, the last pulled back
                         from (52928, 58048) to (51200, 56017)
    turn 2, slot b       52928 tokens over the reused prefix: resident 11968, last chunk (47808, 52928)

Pass (the KV read through the table the model exports, over UMD, with tt-d-gen's kv_dump_compare.py):
    the engine admitted, reused and finished every request where the server's rules say (PREFILL_DONE = every
    chunk's acks retired, prefill_reader.cpp:157-177), or the producer drained one ack per layer per chunk
    the table's geometry is what tt-d-gen's tables.py profiles for this endpoint; every (slot, layer, 32-token
    position, config) record present, one size, 64-byte aligned, inside one bank, none overlapping (kv_table)
    the KV of both slots vs the golden, per layer and config, over the reused prefix and over the follow-up, PCC >=
    max(serving.min_kv_pcc, thresholds.state); the pad rows of each prompt's last 32-token record zero
    the reused prefix [0, 11968) byte-identical across the two slots (same tokens, same chunks); the pulled-back
    rewrite of [51200, 52928) on the first layer byte-identical to slot b's single write of the same rows
"""

import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import bringup_fw
import pytest
import torch
from bringup_fw.core import metrics
from bringup_fw.reference.golden import Golden
from bringup_fw.testing import dgen_engine as DE
from bringup_fw.testing import kv_table
from bringup_fw.testing import server_rules as R
from bringup_fw.testing import serving as SV
from bringup_fw.testing.chunk_starts import kv_threshold, target_rung
from bringup_fw.testing.contract import served_layers
from bringup_fw.testing.harness import device_timeout, impl_mode, spec

# the run's own framework ($BRINGUP_FW_DIR, set by env.sh), never a copy the workspace shadows it with
assert not os.environ.get("BRINGUP_FW_DIR") or os.path.realpath(bringup_fw.__file__).startswith(
    os.path.realpath(os.environ["BRINGUP_FW_DIR"]) + os.sep
), f"bringup_fw imported from {bringup_fw.__file__}, not $BRINGUP_FW_DIR"
S = spec()
SP, TP = S.mesh
CHUNK, MAX_SEQ = int(S.get("target.chunk")), int(S.get("target.seq"))
N_KV, HEAD_DIM = int(S.get("checkpoint.config.num_key_value_heads")), 64
KV_BLOCK = R.KV_BLOCK if SV.answer(S, "prefix_reuse") in (None, True, "true") else 0
SLOTS = int(SV.answer(S, "slots") or 2)
TURN1, TURN2_A, TURN2_B = 12017, 56017, 52928
STARTUP_STALL_S, STARTUP_MAX_S = 900, 1800  # runner: no new log line for this long / startup longer than this
RUN_STALL_S, RUN_MAX_S = 600, 1200  # the feeder: no engine progress / the whole session
SHUTDOWN_S = 600


def _q(key, default):
    v = SV.answer(S, key)
    return default if v in (None, "") else str(v).lower()


KV_DTYPE = _q("kv_dtype", "bfp8").replace("bfloat8_b", "bfp8").replace("bfloat16", "bf16")
CONFIGS = _q("kv_table_configs", "per_head")  # per_head: k_h0..7, v_h0..7 (64 wide) | per_chip: k_c0..3, v_c0..3
ROPE_LAYOUT = _q("k_rope_layout", "meta")  # meta: K columns in interleaved-pair order | hf: the checkpoint's order
FABRIC = {"FABRIC_1D": "1d", "FABRIC_1D_RING": "1d_ring", "FABRIC_2D": "2d", "FABRIC_2D_TORUS_XY": "2d_torus_xy"}


def width() -> int:
    return HEAD_DIM * (N_KV // TP if CONFIGS == "per_chip" else 1)


def n_configs() -> int:
    return 2 * (TP if CONFIGS == "per_chip" else N_KV)


def record_bytes() -> int:
    return width() // 32 * (1088 if KV_DTYPE == "bfp8" else 2048)


# ---------------------------------------------------------------- tt-d-gen's own KV checks
def dgen_harness(repo: Path):
    """(tables, kv_dump_compare) from the pinned checkout. server_rules.harness copies tables.py out of its package,
    which fails since tables.py imports its siblings relatively; then load kv_manager/tests/harness as a package."""
    try:
        return R.harness(repo)
    except ImportError:
        pass
    pkg = repo / "kv_manager/tests/harness"
    name = "dgen_kvm_harness"
    if name not in sys.modules:
        sp = importlib.util.spec_from_file_location(name, pkg / "__init__.py", submodule_search_locations=[str(pkg)])
        mod = importlib.util.module_from_spec(sp)
        sys.modules[name] = mod
        sp.loader.exec_module(mod)
    tables = importlib.import_module(f"{name}.tables")
    sp = importlib.util.spec_from_file_location("dgen_kv_dump_compare", repo / "kv_manager/tools/kv_dump_compare.py")
    kdc = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(kdc)
    return tables, kdc


# ---------------------------------------------------------------- the session
def session(tokens: list[int]):
    """[{name, token_ids, after}] for the engine and the plan the server's rules give it (Push list)."""
    reqs = [
        {"name": "a1", "token_ids": tokens[:TURN1], "after": None},
        {"name": "b1", "token_ids": tokens[:TURN1], "after": None},
        {"name": "a2", "token_ids": tokens[:TURN2_A], "after": "a1"},
        {"name": "b2", "token_ids": tokens[:TURN2_B], "after": "b1"},
    ]
    plan = R.interleave({0: [TURN1, TURN2_A], 1: [TURN1, TURN2_B]}, CHUNK, MAX_SEQ, KV_BLOCK)
    return reqs, plan


def resident_of_turn2(prompt: int) -> int:
    return R.follow_up_resident(TURN1, prompt, KV_BLOCK)


# ---------------------------------------------------------------- processes
class Runner:
    """tt-metal's prefill runner as a subprocess, its log watched for progress."""

    def __init__(self, env: dict, log: Path):
        self.log = log
        self.fh = open(log, "w")
        self.p = subprocess.Popen(
            [sys.executable, "-m", "models.demos.common.prefill.runners.prefill_runner"],
            cwd=S.repo,
            env=env,
            stdout=self.fh,
            stderr=subprocess.STDOUT,
        )

    def text(self) -> str:
        return self.log.read_text(errors="replace")

    def tail(self, n: int = 25) -> str:
        return "\n".join(self.text().splitlines()[-n:])

    def wait_ready(self) -> None:
        t0 = last_t = time.monotonic()
        last_size = -1
        while "entering request loop" not in self.text():
            rc = self.p.poll()
            if rc is not None:
                pytest.fail(f"the prefill runner exited {rc} during setup:\n{self.tail()}", pytrace=False)
            size, now = self.log.stat().st_size, time.monotonic()
            if size != last_size:
                last_size, last_t = size, now
            if now - last_t > STARTUP_STALL_S or now - t0 > STARTUP_MAX_S:
                pytest.fail(
                    f"the prefill runner made no progress for {now - last_t:.0f} s ({now - t0:.0f} s in setup):\n"
                    f"{self.tail()}",
                    pytrace=False,
                )
            time.sleep(2)
        print(f"runner ready in {time.monotonic() - t0:.0f} s", flush=True)

    def stop(self, service_id: str, env: dict) -> list[str]:
        """The shutdown sentinel (the engine never sends it), then a bounded wait for the runner to exit 0."""
        fails = []
        if self.p.poll() is None:
            r = subprocess.run(
                [sys.executable, "-m", "bringup_fw.testing.dgen_engine", "shutdown", service_id],
                cwd=S.repo,
                env=env,
                capture_output=True,
                text=True,
                timeout=SHUTDOWN_S,
            )
            if r.returncode:
                fails.append(f"shutdown sentinel not sent (exit {r.returncode}): {(r.stderr or r.stdout)[-400:]}")
            try:
                self.p.wait(timeout=SHUTDOWN_S)
            except subprocess.TimeoutExpired:
                fails.append(f"the runner did not exit within {SHUTDOWN_S} s of the shutdown sentinel")
                self.p.terminate()
                self.p.wait(timeout=60)
        if self.p.returncode not in (0, None):
            fails.append(f"the runner exited {self.p.returncode}:\n{self.tail()}")
        self.fh.close()
        return fails


def run_engine(build, reqs, workdir: Path, service_id: str, layers: int, runner: Runner) -> dict:
    plan = DE.write_plan(
        workdir / "plan.json",
        service_id=service_id,
        chunk=CHUNK,
        layers_per_chunk=layers,
        sp=SP,
        max_slots=SLOTS,
        max_seq=MAX_SEQ,
        requests=reqs,
        kv_block_size=KV_BLOCK,
        connect_timeout_ms=300000,
        stall_s=RUN_STALL_S,
        run_timeout_s=RUN_MAX_S,
    )
    out = workdir / "engine_out.json"
    with open(workdir / "engine.log", "w") as fh:
        drv = subprocess.Popen(build.argv(str(plan), str(out)), stdout=fh, stderr=subprocess.STDOUT)
        while drv.poll() is None:
            if runner.p.poll() is not None:
                drv.terminate()
                drv.wait(timeout=60)
                pytest.fail(f"the runner exited {runner.p.returncode} mid-session:\n{runner.tail()}", pytrace=False)
            time.sleep(1)
    print((workdir / "engine.log").read_text(errors="replace")[-3000:], flush=True)
    rec = json.loads(out.read_text()) if out.is_file() else {}
    rec["rc"] = drv.returncode
    return rec


def run_producer(reqs, plan, service_id: str, layers: int) -> int:
    """tt-metal's prefill_producer wire calls on the server's plan: every chunk pushed as the engine would (PAD_ID
    tail, ring_sdpa_reshuffle), then one ack per layer per chunk drained. Returns the acks drained."""
    import ttnn
    from models.demos.common.prefill.runners import prefill_producer as P

    svc = ttnn.H2DStreamService.connect(service_id, timeout_ms=300000)
    acks = ttnn.InterProcessCounterChannel.connect(f"/tt_prefill_layer_acks_{service_id}", connect_timeout_ms=300000)
    by_slot = {0: [r for r in reqs if r["name"].startswith("a")], 1: [r for r in reqs if r["name"].startswith("b")]}
    nbytes = svc.payload_size_bytes()
    for push in plan:
        toks = by_slot[push.slot][push.turn]["token_ids"]
        payload = R.server_payload(toks[push.start : push.end], push.start, push.end, CHUNK, SP)
        P._push(svc, nbytes, payload, None, P._pack_metadata(push.slot, push.start, push.end))
    svc.barrier()
    want, got, last_t = len(plan) * layers, 0, time.monotonic()
    while got < want and time.monotonic() - last_t < RUN_STALL_S:
        n = acks.try_consume_all()
        if n:
            got, last_t = got + n, time.monotonic()
        time.sleep(0.01)
    del svc
    return got


# ---------------------------------------------------------------- the KV through the table
def golden_rows(g: Golden, layer: int, config: int, lo: int, hi: int):
    """The golden KV of one table config as [hi - lo, width] in the served column order."""
    st = g.state(layer)
    kind, idx = ("key", config) if config < n_configs() // 2 else ("value", config - n_configs() // 2)
    t = st[kind].float().reshape(st[kind].shape[0], N_KV, HEAD_DIM)[lo:hi]
    heads = [idx * (N_KV // TP) + j for j in range(N_KV // TP)] if CONFIGS == "per_chip" else [idx]
    rows = torch.cat([t[:, h, :] for h in heads], dim=-1)
    if kind == "key" and ROPE_LAYOUT == "meta":
        half = HEAD_DIM // 2
        perm = torch.tensor([half * (m % 2) + m // 2 for m in range(HEAD_DIM)])
        rows = torch.cat([rows[:, k * HEAD_DIM : (k + 1) * HEAD_DIM][:, perm] for k in range(len(heads))], dim=-1)
    return rows.numpy()


def check_table(table, tables_mod, layers: list[int]) -> list[str]:
    fails = []
    if table.num_configs() != n_configs():
        fails.append(f"table: {table.num_configs()} configs, want {n_configs()} ({CONFIGS})")
    from types import SimpleNamespace as N

    for c in range(table.num_configs()):
        cfg = table.config(c)
        want = {
            "num_layers": len(layers),
            "max_sequence_length": MAX_SEQ,
            "num_slots": SLOTS,
            "chunk_n_tokens": R.RECORD_TOKENS,
            "chunk_size_bytes": record_bytes(),
        }
        got = {k: int(getattr(cfg, k)) for k in want}
        if got != want:
            fails.append(f"table config {c}: geometry {got}, want {want}")
        eng = N(
            max_seq_tokens=got["max_sequence_length"],
            chunk_size_bytes=got["chunk_size_bytes"],
            chunk_tokens=R.RECORD_TOKENS,
            num_layers=got["num_layers"],
            epochs=(),
            dram_base=None,
        )
        ep = N(id=0, name="prefill-0", num_slots=got["num_slots"], hosts=["prefill-0"], table_spec={})
        prof = tables_mod.table_profile(N(engine=eng), ep)  # tt-d-gen's own profile of this endpoint
        if (prof["chunk_n_tokens"], prof["chunk_size_bytes"], prof["slots"], prof["max_seq_len"]) != (
            got["chunk_n_tokens"],
            got["chunk_size_bytes"],
            got["num_slots"],
            got["max_sequence_length"],
        ):
            fails.append(f"table config {c}: tables.py profiles {prof}, the table holds {got}")
    recs = kv_table.records_from_table(table, range(len(layers)), SLOTS, MAX_SEQ)
    layout = kv_table.layout_failures(recs, range(len(layers)), SLOTS, MAX_SEQ, table.num_configs())
    metrics.record("runner_layout_failures", len(layout))
    fails += [f"table layout: {f}" for f in layout]
    bad = sorted({v[1] for v in recs.values()} - {record_bytes()})
    if bad:
        fails.append(f"table: records of {bad} bytes, one 32-token record of width {width()} is {record_bytes()}")
    return fails


def read_slots(reader, slot_of: dict, n_layers: int) -> dict:
    """{(side, config): {(layer index, pos): bytes}}: every record of both prompts, as kv_dram_poke --dump holds them."""
    ends = {"a": R.ceil_to(TURN2_A, R.RECORD_TOKENS), "b": R.ceil_to(TURN2_B, R.RECORD_TOKENS)}
    return {
        (side, c): {
            (li, pos): reader.record(slot_of[side], li, pos, c)
            for li in range(n_layers)
            for pos in range(0, ends[side], R.RECORD_TOKENS)
        }
        for side in "ab"
        for c in range(n_configs())
    }


def check_kv(kdc, dumps: dict, g: Golden, layers: list[int], resident: dict) -> list[str]:
    """Both slots vs the golden (kv_dump_compare.py pcc per layer and config, over the reused prefix and over the
    follow-up, and on its own over the SP slab just below the follow-up's start, where a write at a wrong offset
    lands), the pad rows of each prompt's last 32-token record zero."""
    fails, thr, geo = [], kv_threshold(S), {"dtype": KV_DTYPE, "width": width(), "storage": "tile"}
    for side, end in (("a", TURN2_A), ("b", TURN2_B)):
        res = resident[side]
        for c in range(n_configs()):
            for li, layer in enumerate(layers):
                for lo, hi in ((0, res), (max(0, res - CHUNK // SP), res), (res, end)):
                    if hi <= lo:
                        continue
                    x = kdc.reassemble(dumps[(side, c)], li, lo, hi, **geo)
                    f = kdc.tensor_pcc(x, golden_rows(g, layer, c, lo, hi), li, thr, split=0)[0]
                    metrics.record(f"runner_pcc_{side}_L{layer:02d}_c{c:02d}_{lo}", f["pcc"] or 0.0)
                    if not f["passed"]:
                        fails.append(
                            f"slot {side} layer {layer} config {c} rows [{lo}, {hi}): KV pcc {f['pcc']} < {thr}"
                        )
                pad_end = R.ceil_to(end, R.RECORD_TOKENS)
                if pad_end > end:
                    z = kdc.reassemble(dumps[(side, c)], li, end, pad_end, **geo)
                    if (z != 0).any():
                        fails.append(
                            f"slot {side} layer {layer} config {c}: pad rows [{end}, {pad_end}) of the last "
                            f"record not zero (max |x| {abs(z).max():.4g}); the record ships whole"
                        )
    return fails


def check_bytes(kdc, dumps: dict, plan) -> list[str]:
    """The first layer's KV of a row depends on its token and position only, so both slots hold the same bytes over
    [0, TURN2_B): slot a's reused prefix (written by turn 1, kept through turn 2) and its pulled-back rewrite of
    [51200, 52928) against slot b's single write of the same rows."""
    fails = []
    a_plan = [(p.start, p.end) for p in plan if p.slot == 0 and p.turn == 1]
    pb = (a_plan[-1][0], a_plan[-2][1]) if R.pulled_back(a_plan) else None
    for c in range(n_configs()):
        r = kdc.compare(
            dumps[("a", c)],
            dumps[("b", c)],
            method="bytecmp",
            layers=[0],
            start=0,
            end=TURN2_B,
            dtype=KV_DTYPE,
            width=width(),
            split=0,
            storage="tile",
        )
        bad = [f["position"] for f in r["findings"]]
        prefix_bad = [p for p in bad if p < resident_of_turn2(TURN2_A)]
        pb_bad = [p for p in bad if pb and pb[0] <= p < pb[1]]
        metrics.record(f"runner_bytes_differ_c{c:02d}", len(bad))
        if prefix_bad:
            fails.append(
                f"config {c}: the reused prefix differs between the slots on the first layer at "
                f"{len(prefix_bad)} records, e.g. positions {prefix_bad[:4]}"
            )
        if pb_bad:
            fails.append(
                f"config {c}: the pulled-back rewrite of [{pb[0]}, {pb[1]}) on the first layer differs from a "
                f"single write at {len(pb_bad)} records, e.g. positions {pb_bad[:4]}"
            )
        rest = [p for p in bad if p not in prefix_bad and p not in pb_bad]
        if rest:
            fails.append(
                f"config {c}: the first layer's KV differs between the slots at {len(rest)} records, e.g. "
                f"positions {rest[:4]} (same tokens at the same positions)"
            )
    if pb is None and KV_BLOCK:
        fails.append(f"slot a's follow-up was not pulled back: {a_plan[-2:]}")
    return fails


# ---------------------------------------------------------------- the test
@device_timeout(S)
def test_runner_contract():
    if impl_mode() != "device":  # the framework's CPU proof modes: no device, no runner
        return
    from models.demos.common.prefill.adapter import ADAPTER_PATHS

    name = S.get("contract.adapter") or S.model
    if name not in ADAPTER_PATHS:
        pytest.fail(
            f"not built: no adapter {name!r} in ADAPTER_PATHS (models/demos/common/prefill/adapter.py)", pytrace=False
        )
    kind, build, why = DE.feeder(S)
    if kind is None:
        pytest.fail(f"no feeder: {why}", pytrace=False)
    print(f"feeder: {build.describe() if build else f'{DE.PRODUCER} ({why})'}", flush=True)
    first, n_layers = served_layers(S)
    layers = list(range(first, first + n_layers))
    rung = target_rung(S)
    g = Golden.for_rung(S, rung["name"])
    tokens = g.tokens().tolist()
    assert len(tokens) >= TURN2_A, f"golden {rung['name']} holds {len(tokens)} tokens, the session needs {TURN2_A}"
    assert SLOTS >= 2, f"the runner test interleaves two slots, the served slot count is {SLOTS}"
    reqs, plan = session(tokens)
    if KV_BLOCK:
        assert R.pulled_back(R.chunk_plan(TURN2_A, resident_of_turn2(TURN2_A), CHUNK, MAX_SEQ)), "no pull-back"
    tables_mod, kdc = dgen_harness(DE.server_repo(S))

    work = Path(tempfile.mkdtemp(prefix="runner_contract_", dir=os.environ.get("BRINGUP_SCRATCH_DIR") or None))
    sid = f"gpt_oss_120b_contract_{os.getpid()}"
    env = dict(os.environ)
    env.update(
        PREFILL_MODEL=name,
        PREFILL_SP=str(SP),
        PREFILL_TP=str(TP),
        PREFILL_CHUNK_SIZE=str(CHUNK),
        PREFILL_MAX_SEQ_LEN=str(MAX_SEQ),
        PREFILL_NUM_LAYERS=str(n_layers),
        PREFILL_NUM_USERS=str(SLOTS),
        PREFILL_H2D_SERVICE_ID=sid,
        PREFILL_HF_MODEL=str(S.get("paths.hf")),
        PREFILL_FABRIC_MODE=FABRIC.get(str(S.get("box.device_params.fabric_config")), "1d_ring"),
        PREFILL_LAYER_ACK_D2H="1" if _q("ack_mode", "host") == "d2h" else "0",
        PREFILL_ENABLE_MIGRATION="0",  # mock migration: the table stays on disk, no KV Manager handshake
        PREFILL_MIGRATION_EXPORT_TO_FILE="0",
        PREFILL_MIGRATION_TABLE_PATH=str(work / "kv_table.pb"),
        PREFILL_MIGRATION_DEVICE_MAP_PATH=str(work / "device_map.json"),
        PREFILL_LAYER_COMPLETION_RING=f"/tt_prefill_layer_completion_ring_{sid}",
    )
    print(f"workdir {work}; runner env " + " ".join(f"{k}={v}" for k, v in env.items() if k.startswith("PREFILL_")))
    runner = Runner(env, work / "runner.log")
    fails = []
    try:
        runner.wait_ready()
        if kind == DE.ENGINE:
            rec = run_engine(build, reqs, work, sid, n_layers, runner)
            done = rec.get("requests", {})
            if rec["rc"] != 0 or not rec.get("ok"):
                fails.append(f"the engine failed (exit {rec['rc']}): {rec.get('errors')}")
            for r in reqs:
                got = done.get(r["name"], {})
                if got.get("done_position") != len(r["token_ids"]):
                    fails.append(
                        f"{r['name']}: PREFILL_DONE at {got.get('done_position')}, prompt "
                        f"{len(r['token_ids'])}: the acks did not retire every chunk"
                    )
            slot_of = {"a": done.get("a1", {}).get("slot"), "b": done.get("b1", {}).get("slot")}
            resident = {s: done.get(f"{s}2", {}).get("resident") for s in "ab"}
            want_res = {"a": resident_of_turn2(TURN2_A), "b": resident_of_turn2(TURN2_B)}
            if resident != want_res:
                fails.append(f"follow-up turns admitted at {resident}, the server's rules give {want_res}")
            for s in "ab":
                if done.get(f"{s}2", {}).get("slot") != slot_of[s]:
                    fails.append(f"turn 2 of slot {s} was not remounted on its turn-1 slot {slot_of[s]}: {done}")
            sent = rec.get("telemetry", {}).get("prefill_chunk_tokens_total")
            if sent != sum(p.end - p.start for p in plan):
                fails.append(
                    f"the engine sent {sent} chunk tokens, the server's plan {sum(p.end - p.start for p in plan)}"
                )
        else:
            got = run_producer(reqs, plan, sid, n_layers)
            if got != len(plan) * n_layers:
                fails.append(f"{got} layer acks for {len(plan)} chunks x {n_layers} layers")
            slot_of, resident = {"a": 0, "b": 1}, {"a": resident_of_turn2(TURN2_A), "b": resident_of_turn2(TURN2_B)}
        if runner.p.poll() is not None:
            fails.append(f"the runner exited {runner.p.returncode} before the read-back:\n{runner.tail()}")
        assert not fails, "\n".join(fails)

        import ttnn

        D = ttnn.experimental.disaggregation
        table = D.import_from_protobuf_file(str(work / "kv_table.pb"))
        fails += check_table(table, tables_mod, layers)
        dmap = {
            tuple(int(x) for x in k.split(":")): int(v)
            for k, v in json.loads((work / "device_map.json").read_text()).items()
        }
        reader = kv_table.TableReader(table, dmap)
        dumps = read_slots(reader, slot_of, n_layers)
        if reader.replica_mismatch:
            fails.append(
                f"{len(reader.replica_mismatch)} records differ between the chips of their device group, "
                f"e.g. {reader.replica_mismatch[:3]}"
            )
        fails += check_kv(kdc, dumps, g, layers, resident)
        fails += check_bytes(kdc, dumps, plan)
    finally:
        fails += runner.stop(sid, env)
    metrics.record("runner_contract_failures", len(fails))
    for f in fails:
        print(f"FAIL runner contract: {f}")
    assert not fails, f"{len(fails)} runner contract failures, first: {fails[0]}"
