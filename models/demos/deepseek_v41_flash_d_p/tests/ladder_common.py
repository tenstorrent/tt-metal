# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MB.3 (OWNER 2026-11-21; from the batching branch's tests/packed/ladder_common.py, PB.9C 8d18418226c / PB.NR
21de88e33ad, SINGLE-REQUEST parts only: this branch has no packed path): the serving column and the host split of the
performance LADDER (tests/test_skip_positions.py) at real fixed histories.

The ladder (results/agentx_ladder_20261119.md's method): at each history p of SKIP_POS (the gate: the owner's 51200,
131072, 139264 (AgentX p50), 262144, 524288, 786432, 1024000) the steady and resumed chunk, each timed iteration after
the REAL previous chunk (untimed, synced); the clean wall (testing/profile.warm_up, WALL_RUNS timed runs with
program-cache misses refused, the median), one real-time profiled run for the device ms (slowest chip).

The serving wall (owner 2026-11-21): the same steady chunk through THE SERVING ENTRY (adapter.prefill_chunk) WITH the
per-layer device-op ack tt-d-gen uses (``D2HAcks``: a metadata-only ttnn.D2HStreamService + ttnn.LayerAckService
draining it into a layer-completion ring this module pops; outbound_socket_service_sync once a layer on the same
command queue, no host sync), ``time_kind``'s warm-up and SERVING_RUNS timed runs, the median.

``HostSplit``: the serving entry's host work item by item (wrapped callables, perf_counter ms per call).
"""

from __future__ import annotations

import os
import time

SPLIT_POS = 139264  # the host split's history (AgentX p50)
SERVING_RUNS = 5  # the serving wall: timed runs after its warm-up (median)
REST = "rest: host Python between the wrapped calls (wall - the items)"


def serving_on() -> bool:
    """LADDER_SERVING=0 skips the serving-wall column (and the host split); default on."""
    return os.environ.get("LADDER_SERVING", "1").strip() not in ("0", "off", "")


def record(name: str, value, **extra) -> None:
    """A metric under its own name and without the text tag (the gate's threshold names carry none)."""
    from models.demos.common.bringup.core import metrics

    metrics.record(name, value, **extra)
    tok = os.environ.pop("BRINGUP_POS_TOKENS", None)
    try:
        metrics.record(name, value, **extra)
    finally:
        if tok is not None:
            os.environ["BRINGUP_POS_TOKENS"] = tok


def time_kind(mesh, one_run, runs: int | None = None) -> dict:
    """testing/profile's clean wall: warm_up, then ``runs`` (WALL_RUNS) timed runs with program-cache misses refused;
    {"wall": median, "runs": [...], "spread", "warm_runs", "settled"}."""
    from models.demos.common.bringup.testing.profile import WALL_RUNS, misses_allowed, settled, warm_up

    w = warm_up(mesh, one_run)
    assert w["warm"], f"not warm after {len(w['walls'])} runs"
    walls = []
    misses_allowed(mesh, False)
    try:
        for _ in range(runs or WALL_RUNS):
            walls.append(one_run())
    finally:
        misses_allowed(mesh, True)
    kept = sorted(walls)
    return {
        "wall": kept[len(kept) // 2],
        "runs": walls,
        "spread": round(kept[-1] - kept[0], 1),
        "warm_runs": len(w["walls"]),
        "settled": int(settled(w["walls"])),
    }


# ------------------------------------------------------------------------------------------------------- D2H acks
class D2HAcks:
    """tt-d-gen's per-layer device-op ack (prefill_runner, PREFILL_LAYER_ACK_D2H=1): a metadata-only D2HStreamService
    (persistent reader on a claimed dispatch-column service core, slow-dispatched; the ack op
    outbound_socket_service_sync on worker core (0, 0) of the command queue) and a LayerAckService that drains it into
    a layer-completion ring (created here; ``drain`` pops it). One record a "chunk" (record_count 1): every ack is its
    own message, so any number of acks a chunk (21 layers x segments) decodes without a desync. ``record`` is the
    12-byte {slot, start, end} metadata tensor every ack sends."""

    RECORD_BYTES = 12

    def __init__(self, mesh, fifo_bytes: int = 4096):
        import torch
        from ttnn._experimental.layer_completion import LayerCompletionQueue

        import ttnn

        self.mesh = mesh
        self.name = f"/tt_lcq_ladder_{os.getpid()}_{int(time.time() * 1e3) % 100000}"
        self._unlink()
        self.ring = LayerCompletionQueue.create(self.name, 1)
        self.d2h = ttnn.D2HStreamService(
            mesh,
            global_spec=None,
            fifo_size_bytes=fifo_bytes,
            worker_cores=ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
            metadata_size_bytes=self.RECORD_BYTES,
            metadata_ring=True,
        )
        self.svc = ttnn.LayerAckService(self.d2h, self.name, first_record=0, record_count=1, layer_ids=[0])
        self.svc.start()
        self.record = ttnn.from_torch(
            torch.tensor([0, 0, 1], dtype=torch.int64).reshape(1, 1, 1, 3),
            device=mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.synchronize_device(mesh)
        self.sent = 0  # acks enqueued (counted at the op: ``count``)
        self.popped = 0
        self._op = None

    def _unlink(self) -> None:
        try:
            os.remove(f"/dev/shm/{self.name.lstrip('/')}")
        except FileNotFoundError:
            pass

    def __enter__(self):
        """Count the ack ops enqueued inside (the module attribute the adapter calls)."""
        import ttnn

        mod = ttnn.experimental.deepseek_prefill
        f = self._op = mod.outbound_socket_service_sync
        me = self

        def g(*a, **k):
            me.sent += 1
            return f(*a, **k)

        g.__doc__ = getattr(f, "__doc__", None)
        mod.outbound_socket_service_sync = g
        return self

    def __exit__(self, *exc):
        import ttnn

        if self._op is not None:
            ttnn.experimental.deepseek_prefill.outbound_socket_service_sync = self._op
            self._op = None
        return False

    def drain(self, timeout_s: float = 30.0) -> int:
        """Pop the ring until every ack sent so far arrived (after a device sync: the records are on their way);
        returns the number still missing (0 = every ack seen; never blocks past ``timeout_s``)."""
        deadline = time.monotonic() + timeout_s
        while self.popped < self.sent and time.monotonic() < deadline:
            m = self.ring.try_pop()
            if m is None:
                time.sleep(0.0005)
                continue
            self.popped += 1
        return self.sent - self.popped

    def close(self) -> str | None:
        """Stop the reader (raises its stored failure once: returned as text), shut the ring, free the record."""
        import ttnn

        err = None
        try:
            self.drain(5.0)
            self.svc.stop()
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:300]
        try:
            self.ring.shutdown()
        except Exception:  # noqa: BLE001
            pass
        self._unlink()
        if self.record.is_allocated():
            ttnn.deallocate(self.record)
        del self.svc
        del self.d2h
        return err


# ------------------------------------------------------------------------------------------------------ host split
class HostSplit:
    """The host ms of named callables while installed (perf_counter around each call; nested wrapped calls are
    charged to the inner item and taken out of the outer one, so the items add up). ``wrap(obj, attr, item)``: an
    instance / module attribute; ``sub(item, outer)``: report ``item`` (e.g. the readbacks) also as a part of
    ``outer``. ``begin()`` / ``end()`` per chunk; ``items()`` -> {item: [ms a chunk]}."""

    def __init__(self):
        self.undo = []
        self.cur = {}
        self.per = {}
        self.stack = []
        self.calls = {}
        self.ncur = {}
        self.active = False  # (only calls between begin() and end() count: the untimed previous chunk does not)

    def wrap(self, obj, attr: str, item: str, instance: bool = False) -> None:
        f = getattr(obj, attr)
        me = self

        def g(*a, **k):
            me.stack.append(0.0)
            t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                dt = (time.perf_counter() - t0) * 1e3
                inner = me.stack.pop()
                if me.active:
                    me.cur[item] = me.cur.get(item, 0.0) + dt - inner
                    me.ncur[item] = me.ncur.get(item, 0) + 1
                if me.stack:
                    me.stack[-1] += dt

        g.__doc__ = getattr(f, "__doc__", None)
        had = instance and attr in getattr(obj, "__dict__", {})
        setattr(obj, attr, g)
        self.undo.append((obj, attr, f, instance, had))

    def add(self, item: str, ms: float) -> None:
        self.cur[item] = self.cur.get(item, 0.0) + ms

    def begin(self) -> None:
        self.cur, self.ncur, self.active = {}, {}, True

    def end(self, wall: float | None = None) -> None:
        """Close the chunk; ``wall`` (its wall ms) adds the item REST = wall - every item (the host Python between the
        wrapped calls), per chunk."""
        self.active = False
        if wall is not None:
            self.cur[REST] = wall - sum(self.cur.values())
        for k, v in self.cur.items():
            self.per.setdefault(k, []).append(v)
        for k, v in self.ncur.items():
            self.calls[k] = v  # (calls in one timed chunk)
        self.cur, self.ncur = {}, {}

    def items(self) -> dict:
        """{item: median ms a chunk}."""
        return {k: sorted(v)[len(v) // 2] for k, v in self.per.items()}

    def restore(self) -> None:
        for obj, attr, f, instance, had in reversed(self.undo):
            if instance and not had:
                try:
                    delattr(obj, attr)
                except AttributeError:
                    setattr(obj, attr, f)
            else:
                setattr(obj, attr, f)
        self.undo.clear()


def split_table(title: str, rows: list) -> str:
    """rows: [(item, serving ms, harness ms or None, note)] -> a markdown table."""
    lines = [
        f"### {title}",
        "",
        "| item | serving entry ms | harness ms | serving - harness ms | what |",
        "|---|---|---|---|---|",
    ]
    for item, s, h, note in rows:
        hs = "-" if h is None else f"{h:.2f}"
        d = "-" if (s is None or h is None) else f"{s - h:+.2f}"
        ss = "-" if s is None else f"{s:.2f}"
        lines.append(f"| {item} | {ss} | {hs} | {d} | {note} |")
    return "\n".join(lines)


SPLIT_RUNS = 7  # the host split: chunks measured (median per item), after the walls' warm-up


def _engram_ms(model) -> dict:
    """The last chunk's engram jobs (worker threads): {"job": host ms summed over layers, "wait": the main thread's
    exposed wait summed (inside the layer calls)}."""
    t = getattr(model, "engram_times", {}) or {}
    job = sum(float(x) for v in t.values() if isinstance(v, dict) for k, x in v.items() if k != "wait")
    wait = sum(float(v.get("wait", 0.0) or 0.0) for v in t.values() if isinstance(v, dict))
    return {"job": round(job, 2), "wait": round(wait, 2), "layers": len(t)}


def print_split(path: str, h: int, sp: dict) -> None:
    for side, v in sp.items():
        items = ", ".join(f"{k} {ms:.2f}" for k, ms in sorted(v["items"].items(), key=lambda x: -x[1]))
        print(
            f"SPLIT {path} p{h} {side}: wall {v['wall']:.2f} ms (runs {v['walls']}); {items}; calls a chunk "
            f"{v['calls']}; engram {v['engram']}",
            flush=True,
        )
