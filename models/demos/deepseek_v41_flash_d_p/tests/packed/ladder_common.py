# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.9C (OWNER 2026-11-21): the performance LADDER at real fixed histories, the parts the packed walls
(tests/packed/test_packed_walls.py, test_glued_walls.py) and the single-request ladder (tests/test_skip_positions.py)
share.

The ladder (results/agentx_ladder_20261119.md's method): at each history h of LADDER_POS (default the owner's
51200, 131072, 139264 (AgentX p50), 262144, 524288, 786432, 1024000) EVERY segment of the chunk sits at h, each in its
own slot; two kinds:
    steady    every segment a mid-request chunk: write_start = start
    resumed   every segment a request's first chunk after a prefix hit: write_start = start + 128 (encoder replay)
Every timed iteration first runs the REAL previous chunk (each segment at h - its length, untimed, synced), then the
timed chunk. The clock runs from the engine's device input (made before the clock: the engine's H2D) to the sync after
the last layer; ``time_kind``: testing/profile.warm_up (until no new program and the walls settled), WALL_RUNS timed
runs with program-cache misses refused, the median; one more run under the real-time profiler gives the device ms
(slowest chip); exposed host = wall - device.

No memoised history between repeats: every timed chunk makes its engine input anew (make_input / payload_tensor) and
gathers every segment's history again (TtDS41Model.begin_chunk -> history_gather.new_chunk() drops every prefetch
marker at each chunk start, FIX.39; the untimed previous chunk in between runs at other positions); the profiled run
counts the traced history gathers of the timed chunk (ladder_<path>_gathers_p<h>).

The serving wall (owner 2026-11-21): the same chunks through THE SERVING ENTRY (adapter.prefill_chunk /
prefill_packed) WITH the per-layer device-op ack tt-d-gen uses (``D2HAcks``: a metadata-only ttnn.D2HStreamService +
ttnn.LayerAckService draining it into a layer-completion ring this module pops; outbound_socket_service_sync once a
layer (a segment) on the same command queue, no host sync), the same warm-up / median.

``HostSplit``: the serving entry's host work item by item (wrapped callables, perf_counter ms per call).
"""

from __future__ import annotations

import os
import time

LADDER_POS = (51200, 131072, 139264, 262144, 524288, 786432, 1024000)
CASES = {
    "4x2k": (2048, 2048, 2048, 2048),
    "5x1600": (1600, 1600, 1600, 1600, 1600),
    # PB.9G (chip-aligned lanes, 8x4: 256 rows a chip): four 1.6k requests (7 chips each, the last one padded) and the
    # 5th request's first 1024 rows (4 chips: the split piece; its rest continues in a later chunk)
    "4x1600s": (1600, 1600, 1600, 1600, 1024),
}
REPLAY = 128  # resumed: write_start = start + 128 (DeepSeek's Encoder SWA Bounded Replay)
SPLIT_POS = 139264  # the host split's history (AgentX p50)
ITEM_VARS = ("LADDER_POS", "LADDER_CASES", "LADDER_SERVING")
SERVING_RUNS = 5  # the serving wall: timed runs after its warm-up (median)
REST = "rest: host Python between the wrapped calls (wall - the items)"


def on() -> bool:
    return bool(os.environ.get("LADDER_POS", "").strip())


def positions() -> list:
    v = os.environ.get("LADDER_POS", "").strip()
    return [int(x) for x in v.split(",") if x.strip()] if v else list(LADDER_POS)


def cases(default: list) -> list:
    v = os.environ.get("LADDER_CASES", "").strip()
    return [c for c in v.split(",") if c.strip()] if v else list(default)


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


def alloc_failure(e: BaseException) -> str | None:
    """tests/test_skip_positions._alloc_failure (FIX.36): the reason line of a device allocation failure, else None."""
    from models.demos.deepseek_v41_flash_d_p.tests.test_skip_positions import _alloc_failure

    return _alloc_failure(e)


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


# --------------------------------------------------------------------------------------------- the packed ladder
def _device(prof: dict) -> float:
    """The slowest chip's device ms of one real-time profiled run."""
    from collections import defaultdict

    per_chip = defaultdict(float)
    for d in prof["kernel_ns_dev"].values():
        for c, ns in d.items():
            per_chip[c] += ns
    return max(per_chip.values()) / 1e6 if per_chip else 0.0


def _phases(prof: dict) -> dict:
    """{phase: ms} of one real-time profiled run (testing/profiler.phase_of over its sections, the slowest chip a
    section), as test_skip_positions' device_ms_<phase>."""
    import re
    from collections import defaultdict

    from models.demos.common.bringup.testing import profiler

    out = defaultdict(float)
    for sec, ns in prof["kernel_ns"].items():
        out[re.sub(r"[^A-Za-z0-9]+", "_", profiler.phase_of(sec))] += ns / 1e6
    return dict(out)


def packed_ladder(runtime, kv, mesh, layers, path: str, lens, make_desc, text, shift: int, cap: int, split=False):
    """One packed path's ladder (module docstring). ``lens``: the segments' new tokens; ``make_desc(segs)``: the chunk
    of those Segments in chunk order (ChunkDescriptor.packed for the fixed lanes, GluedChunk for the glued path);
    segment k reads ``text`` from k x ``shift`` (different stretches of the session) and sits in serving slot k of
    ``kv`` (runtime.state: the same slot states for the harness and the serving entry); the segments' order in the
    chunk rotates by one every iteration (no lane / row keeps a slot). ``split``: the serving entry's host split at
    SPLIT_POS. Returns the table rows [(h, row dict or None, reason)]."""
    import ttnn
    from models.demos.common.bringup.testing import profiler
    from models.demos.deepseek_v41_flash_d_p.tt import history_gather
    from models.demos.deepseek_v41_flash_d_p.tt import packed_device as PD
    from models.demos.deepseek_v41_flash_d_p.tt.packed import Segment

    model = runtime.model
    n = len(lens)
    states = [runtime.state(kv, k) for k in range(n)]
    top = min(cap, model.max_seq)
    acks = None
    if serving_on():
        try:
            acks = D2HAcks(mesh)
        except Exception as e:  # noqa: BLE001 (recorded: the serving column then says why it is missing)
            print(f"LADDER {path}: D2H ack service not available: {type(e).__name__}: {e}", flush=True)
            record(f"ladder_{path}_serving_error", f"{type(e).__name__}: {e}"[:300])
    rows = []
    it = [0]

    def chunk(h: int, prev: bool, ws: bool, back=None):
        order = [(k + it[0]) % n for k in range(n)]  # chunk position j holds segment order[j]
        segs, ids, sts = [], [], []
        for k in order:
            s = h - (0 if back is None else back[k]) - (lens[k] if prev else 0)
            w = s + REPLAY if (ws and not prev) else None
            segs.append(Segment(k, s, s + lens[k], w))
            ids.append(text[k * shift + s : k * shift + s + lens[k]].long())
            sts.append(states[k])
        return make_desc(segs), ids, sts

    def harness(h: int, prev: bool, ws: bool, prof: bool = False, hs: HostSplit | None = None, back=None):
        desc, ids, sts = chunk(h, prev, ws, back)
        inp = PD.make_input(mesh, desc, ids)  # (the engine's H2D: before the clock)
        model.sync()
        if hs is not None:
            hs.begin()
        if prof:
            profiler.enable(mesh, rt=True)
            history_gather.TRACE = []
            profiler.set_layer(None)
        t0 = time.time()
        res = None
        try:
            nat = PD.ids_from_input(mesh, inp, desc)  # (module attributes: HostSplit wraps them)
            PD.run_chunk(model, desc, nat, sts, layers, last=False, inp=inp)
            t1 = time.time()
            model.sync()
            wall = (time.time() - t0) * 1e3
            if hs is not None:
                hs.add("device tail (sync after the last enqueue)", (time.time() - t1) * 1e3)
                hs.end(wall)
            if prof:
                profiler.signpost("end")
                profiler.finish_rt()
                gathers = sum(1 for k, *_ in (history_gather.TRACE or []) if k in ("gather", "tail"))
                pr = profiler.result()
                res = (_device(pr), gathers, _phases(pr))
        finally:
            if prof:
                history_gather.TRACE = None
                profiler.disable()
            ttnn.deallocate(inp)
        return wall, res

    def serving(h: int, prev: bool, ws: bool, hs: HostSplit | None = None, ack: bool = True, back=None):
        desc, ids, _ = chunk(h, prev, ws, back)
        inp = runtime.make_packed_input(desc, ids)  # (the engine's H2D: before the clock)
        model.sync()
        if hs is not None:
            hs.begin()
        t0 = time.time()
        try:
            if ack:
                with acks:
                    runtime.prefill_packed(inp, kv, desc, d2h_service=acks.d2h, metadata_msg=acks.record)
            else:  # (the serving entry without any ack: the walls' "no ack at all")
                runtime.prefill_packed(inp, kv, desc)
            t1 = time.time()
            model.sync()
            wall = (time.time() - t0) * 1e3
            if hs is not None:
                hs.add("device tail (sync after the last enqueue)", (time.time() - t1) * 1e3)
                hs.end(wall)
        finally:
            ttnn.deallocate(inp)
        missing = acks.drain()
        assert missing == 0, f"{missing} D2H acks not seen after the chunk ({acks.sent} sent)"
        return wall

    try:
        for h in positions():
            why = None
            if h % 32 or h < max(lens):
                why = f"history {h}: needs a 32-aligned history >= a segment ({max(lens)}) for the previous chunk"
            elif h + max(lens) > top:
                why = f"history {h} + {max(lens)} > the slots' capacity {top}"
            if why is not None:
                record(f"fits_{path}_p{h}", 0, reason=why)
                print(f"LADDER {path} p{h} NOT RUN: {why}", flush=True)
                rows.append((h, None, why))
                continue
            try:
                row = {}
                for kind, ws in (("steady", False), ("resumed", True)):

                    def one_run():
                        harness(h, True, ws)  # the real previous chunk, untimed
                        w, _ = harness(h, False, ws)
                        it[0] += 1
                        return round(w, 1)

                    t = time_kind(mesh, one_run)
                    harness(h, True, ws)
                    _, (dev, gathers, phases) = harness(h, False, ws, prof=True)
                    it[0] += 1
                    new = sum(lens) - (n * REPLAY if ws else 0)
                    row[kind] = {**t, "device": dev, "gathers": gathers, "new": new, "phases": phases}
                    pre = f"ladder_{path}_{kind}"
                    record(f"{pre}_wall_ms_p{h}", t["wall"])
                    record(f"{pre}_wall_ms_runs_p{h}", t["runs"])
                    record(f"{pre}_wall_ms_spread_p{h}", t["spread"])
                    record(f"{pre}_wall_settled_p{h}", t["settled"])
                    record(f"{pre}_device_ms_p{h}", round(dev, 2))
                    record(f"{pre}_tok_s_p{h}", round(new / (t["wall"] / 1e3), 1))
                    for ph, ms in phases.items():
                        record(f"{pre}_device_ms_{ph}_p{h}", round(ms, 2))
                    print(
                        f"LADDER {path} p{h} {kind}: wall {t['wall']} ms (runs {t['runs']}, spread {t['spread']}, "
                        f"warm-up {t['warm_runs']}), device {dev:.2f} ms, {new / (t['wall'] / 1e3):.0f} new tok/s, "
                        f"{gathers} history gathers traced in the timed chunk; phases (sections' slowest chip) "
                        + ", ".join(f"{k} {v:.2f}" for k, v in sorted(phases.items(), key=lambda x: -x[1])),
                        flush=True,
                    )
                st = row["steady"]
                record(f"ladder_{path}_tok_s_p{h}", round(st["new"] / (st["wall"] / 1e3), 1))
                record(f"ladder_{path}_device_ms_p{h}", round(st["device"], 2))
                record(f"ladder_{path}_host_exposed_ms_p{h}", round(st["wall"] - st["device"], 1))
                record(f"ladder_{path}_gathers_p{h}", st["gathers"])
                if acks is not None:

                    def one_serving():
                        serving(h, True, False)
                        w = serving(h, False, False)
                        it[0] += 1
                        return round(w, 1)

                    sv = time_kind(mesh, one_serving, runs=SERVING_RUNS)
                    row["serving"] = sv
                    record(f"ladder_{path}_serving_wall_ms_p{h}", sv["wall"])
                    record(f"ladder_{path}_serving_wall_ms_runs_p{h}", sv["runs"])
                    record(f"ladder_{path}_serving_extra_ms_p{h}", round(sv["wall"] - st["wall"], 1))
                    record(f"ladder_{path}_serving_acks_per_chunk", len(layers) * n)
                    print(
                        f"LADDER {path} p{h} serving entry (prefill_packed + D2H ack a layer and segment): wall "
                        f"{sv['wall']} ms (runs {sv['runs']}), {sv['wall'] - st['wall']:+.1f} ms over the harness",
                        flush=True,
                    )
                    if split and h == SPLIT_POS:
                        row["split"] = packed_split(runtime, kv, mesh, layers, harness, serving, h, it)
                        record(f"ladder_{path}_host_split_p{h}", row["split"])
                        print_split(path, h, row["split"])
                record(f"fits_{path}_p{h}", 1)
                rows.append((h, row, None))
            except RuntimeError as e:
                why = alloc_failure(e)
                if why is None:  # (a hang or any other error fails the test, FIX.36)
                    raise
                record(f"fits_{path}_p{h}", 0, reason=why[:300])
                print(f"LADDER {path} p{h} DOES NOT FIT: {why[:300]}", flush=True)
                rows.append((h, None, why[:300]))
    finally:
        if acks is not None:
            err = acks.close()
            if err:
                record(f"ladder_{path}_serving_error", err)
                print(f"LADDER {path}: D2H ack service: {err}", flush=True)
    return rows


SPLIT_RUNS = 7  # the host split: chunks measured (median per item), after the walls' warm-up


def _packed_split_wraps(hs: HostSplit, runtime, model, harness_side: bool) -> None:
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt import packed_device

    hs.wrap(ttnn, "from_device", "ids readback (from_device / to_torch, mesh column 0's SP rows)")
    hs.wrap(ttnn, "to_torch", "ids readback (from_device / to_torch, mesh column 0's SP rows)")
    if harness_side:
        hs.wrap(packed_device, "ids_from_input", "ids un-shuffle (engine rows -> model rows, host)")
    else:
        hs.wrap(runtime, "_host_ids_packed", "ids un-shuffle (engine rows -> model rows, host)", instance=True)
    hs.wrap(model, "begin_packed", "begin_packed (per segment note_tokens + begin_chunk fields, engram jobs)", True)
    hs.wrap(packed_device, "glued_dev_ids", "glued_dev_ids (embedding ids reordered on the device: enqueue)")
    hs.wrap(model, "embed", "embed (enqueue)", True)
    hs.wrap(model, "layer", "layers 0-20 (host enqueue, engram waits included)", True)
    hs.wrap(model, "end_packed", "end_packed", True)
    if not harness_side:
        hs.wrap(ttnn.experimental.deepseek_prefill, "outbound_socket_service_sync", "D2H ack ops (enqueue)")


def packed_split(runtime, kv, mesh, layers, harness, serving, h: int, it) -> dict:
    """The serving entry's host work item by item for one packed path at history ``h`` vs the harness's same chunk:
    SPLIT_RUNS chunks a side (each after its real previous chunk), the median of every item. The device tail (the
    sync after the last enqueue) is measured; the rest of the wall (host Python between the wrapped calls: the
    adapter's asserts, descriptor routing, slot-state lookups, ack bookkeeping) is REST. Sides: harness, serving (D2H
    ack), serving_noack, and harness_freshpos / serving_freshpos (every chunk at histories not run before: the
    fresh-traffic host cost, the same slots)."""
    model = runtime.model
    out = {}
    for side in ("harness", "serving", "serving_noack", "harness_freshpos", "serving_freshpos"):
        hs = HostSplit()
        walls = []
        try:
            _packed_split_wraps(hs, runtime, model, side.startswith("harness"))
            for j in range(SPLIT_RUNS):
                # (_freshpos: every chunk at new histories, h - 64 (j n + k + 1) for segment k: no position repeats)
                sh = None if not side.endswith("freshpos") else [64 * (j * 8 + k + 1) for k in range(8)]
                if side.startswith("harness"):
                    harness(h, True, False, back=sh)
                    walls.append(harness(h, False, False, hs=hs, back=sh)[0])
                else:
                    ack = side != "serving_noack"
                    serving(h, True, False, ack=ack, back=sh)
                    walls.append(serving(h, False, False, hs, ack=ack, back=sh))
                it[0] += 1
        finally:
            hs.restore()
        out[side] = {
            "wall": sorted(walls)[len(walls) // 2],
            "walls": [round(w, 2) for w in walls],
            "items": hs.items(),
            "calls": dict(hs.calls),
            "engram": _engram_ms(model),
        }
    return out


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


def ladder_table(path: str, title: str, rows: list) -> str:
    """The path's ladder table (markdown): the metric ladder_<path>_table (and printed)."""
    head = [
        "history",
        "steady wall ms (spread)",
        "new tok/s",
        "resumed wall ms",
        "device ms",
        "exposed host ms",
        "serving wall ms (prefill entry + D2H ack)",
        "gathers traced",
    ]
    lines = [f"# Ladder {path}: {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for h, r, why in rows:
        if r is None:
            lines.append(f"| {h} | does not fit / not run: {why} |" + " |" * (len(head) - 2))
            continue
        st, rs, sv = r["steady"], r["resumed"], r.get("serving")
        sw = "-" if sv is None else f"{sv['wall']:.1f}"
        lines.append(
            f"| {h} | {st['wall']:.1f} ({st['spread']:.1f}) | {st['new'] / (st['wall'] / 1e3):.0f} | {rs['wall']:.1f} | "
            f"{st['device']:.1f} | {st['wall'] - st['device']:.1f} | {sw} | {st['gathers']} |"
        )
    table = "\n".join(lines)
    record(f"ladder_{path}_table", table)  # (the metric only: the owner's report is written from the metrics)
    print("\n" + table, flush=True)
    return table
