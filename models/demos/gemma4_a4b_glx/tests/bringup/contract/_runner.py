# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared by the runner contract tests: the real prefill runner as a subprocess (mock migration, D2H layer acks), fed
by tt-d-gen's engine (dgen_engine.feeder), the KV read back through the table the model exports.

The runner opens the mesh itself (prefill_runner.py main -> runner_utils.open_mesh_device), so these tests never hold
the mesh while it runs. Every wait is bounded and fails at once when a process exits.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from bringup_fw.testing import dgen_engine
from bringup_fw.testing.contract import contract_rung, served_layers

MODEL = "gemma4_a4b_glx"
READY = "setup complete, entering request loop"  # prefill_runner.py:1009
SETUP_TIMEOUT_S = 1800
STALL_S = 600
RUN_TIMEOUT_S = 900
BFP8_TILE_BYTES = 1088


def not_built(why: str):
    pytest.fail(f"not built: {why}", pytrace=False)


def preflight(s):
    """(feeder kind, build): fails "not built" while the feeder, the adapter or the read-back hook is missing."""
    kind, build, why = dgen_engine.feeder(s)
    print(f"feeder: {build.describe() if build else kind} ({why or 'built'})", flush=True)
    if kind is None:
        not_built(why)
    from models.demos.common.prefill.adapter import ADAPTER_PATHS

    if MODEL not in ADAPTER_PATHS:
        not_built(f"adapter {MODEL!r} not registered in models/demos/common/prefill/adapter.py ADAPTER_PATHS")
    if not hasattr(s.hooks(), "contract_read_slot_kv"):
        not_built("hooks.contract_read_slot_kv(s, table, dmap, slot, length) (the table read-back of the model's KV)")
    return kind, build


def golden(s):
    from bringup_fw.reference.golden import Golden

    try:
        return Golden.for_rung(s, contract_rung(s))
    except FileNotFoundError as e:
        not_built(f"golden for {contract_rung(s)}: {e}")


def kv_threshold(s) -> float:
    return max(float(s.get("serving.min_kv_pcc") or 0.93), s.threshold("state", 0.97))


class Runner:
    """prefill_runner.py in a subprocess; .table() and .dmap() once it is in its request loop."""

    def __init__(self, s, tmp: Path, trace_dir: str | None = None):
        rung = s.rung(contract_rung(s))
        self.s, self.tmp, self.seq, self.chunk, self.sp = s, Path(tmp), rung["seq"], rung["chunk"], s.mesh[0]
        self.first, self.n_layers = served_layers(s)
        self.service = f"bfw_{MODEL}_{uuid.uuid4().hex[:8]}"
        self.table_path = self.tmp / "kv_table.pb"
        self.dmap_path = self.tmp / "device_map.json"
        self.log = self.tmp / "runner.log"
        env = dict(os.environ)
        env.update(
            PREFILL_MODEL=MODEL,
            PREFILL_ENABLE_MIGRATION="1",
            PREFILL_MOCK_MIGRATION="1",
            PREFILL_LAYER_ACK_D2H="1",
            PREFILL_FABRIC_MODE="1d_ring",
            PREFILL_H2D_SERVICE_ID=self.service,
            PREFILL_MIGRATION_TABLE_PATH=str(self.table_path),
            PREFILL_MIGRATION_DEVICE_MAP_PATH=str(self.dmap_path),
            TT_METAL_OPERATION_TIMEOUT_SECONDS="600",
        )
        if trace_dir:
            env["PREFILL_TRACE_DIR"] = str(trace_dir)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "models.demos.common.prefill.runners.prefill_runner"],
            cwd=s.repo,
            env=env,
            stdout=open(self.log, "w"),
            stderr=subprocess.STDOUT,
        )

    def tail(self, n: int = 15) -> str:
        return "\n".join(self.log.read_text(errors="replace").splitlines()[-n:]) if self.log.exists() else ""

    def wait_ready(self) -> None:
        t0 = last_t = time.time()
        last_size = 0
        while True:
            text = self.log.read_text(errors="replace") if self.log.exists() else ""
            if READY in text:
                print(f"runner ready in {time.time() - t0:.0f} s", flush=True)
                return
            if self.proc.poll() is not None:
                pytest.fail(
                    f"runner exited {self.proc.returncode} before its request loop:\n{self.tail()}", pytrace=False
                )
            if len(text) != last_size:
                last_size, last_t = len(text), time.time()
            if time.time() - last_t > STALL_S or time.time() - t0 > SETUP_TIMEOUT_S:
                pytest.fail(f"runner made no progress to its request loop:\n{self.tail()}", pytrace=False)
            time.sleep(2)

    def alive(self) -> None:
        if self.proc.poll() is not None:
            pytest.fail(f"runner exited {self.proc.returncode}:\n{self.tail()}", pytrace=False)

    def table(self):
        import ttnn

        return ttnn.experimental.disaggregation.import_from_protobuf_file(str(self.table_path))

    def dmap(self) -> dict:
        raw = json.loads(self.dmap_path.read_text())  # as prefill_producer._read_device_map
        return {tuple(int(x) for x in k.split(":")): int(v) for k, v in raw.items()}

    def drive(self, build, requests: list[dict], tag: str) -> dict:
        """One engine driver run (no shutdown sentinel: the runner stays up for the read-back); its out.json."""
        plan = dgen_engine.write_plan(
            self.tmp / f"plan_{tag}.json",
            service_id=self.service,
            chunk=self.chunk,
            layers_per_chunk=self.n_layers,
            sp=self.sp,
            max_slots=int(self.s.get("contract.num_users")),
            max_seq=self.seq,
            requests=requests,
            connect_timeout_ms=600000,
            run_timeout_s=RUN_TIMEOUT_S,
            stall_s=STALL_S,
        )
        self.alive()
        try:
            rc, out = dgen_engine.run_driver(build, plan, self.tmp / f"out_{tag}.json", timeout_s=RUN_TIMEOUT_S + 60)
        except subprocess.TimeoutExpired:
            pytest.fail(f"engine driver ({tag}) did not finish in {RUN_TIMEOUT_S} s:\n{self.tail()}", pytrace=False)
        print(f"driver {tag}: rc {rc}, requests {json.dumps(out.get('requests', {}))}", flush=True)
        if rc != 0 or not out.get("ok"):
            pytest.fail(
                f"engine driver ({tag}) exit {rc} (2 rejected, 3 stall, 4 timeout, 5 bad plan): "
                f"{out.get('errors')}\n{self.tail()}",
                pytrace=False,
            )
        return out

    def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                dgen_engine.send_shutdown(self.service, timeout_s=60)
                self.proc.wait(timeout=300)
            except Exception as e:  # noqa: BLE001 -- teardown: kill below
                print(f"runner shutdown: {type(e).__name__}: {e}", flush=True)
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=60)


def dump_compare(s):
    """tt-d-gen's kv_dump_compare module (through server_rules.harness, else the file itself)."""
    from bringup_fw.testing import server_rules

    repo = dgen_engine.server_repo(s)
    try:
        return server_rules.harness(repo)[1]
    except Exception as e:  # noqa: BLE001 -- tables.py imports its package relatively; load the comparer alone
        print(f"NOTE server_rules.harness({repo}): {type(e).__name__}: {e}; loading kv_dump_compare.py alone")
    path = repo / "kv_manager/tools/kv_dump_compare.py"
    if not path.is_file():
        not_built(f"tt-d-gen kv_dump_compare.py not found at {path}")
    spec = importlib.util.spec_from_file_location("dgen_kv_dump_compare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows(t, lo: int, hi: int):
    """Rows [lo, hi) along the sequence dim (-2) of a KV tensor, leading 1-dims dropped."""
    while t.dim() > 3 and t.shape[0] == 1:
        t = t[0]
    return t[..., lo:hi, :].float()
