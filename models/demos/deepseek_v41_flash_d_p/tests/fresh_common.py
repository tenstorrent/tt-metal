# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MB.1 (OWNER 2026-11-20, from PB.9F a72c8c6918d tests/packed/fresh_common.py, single-request parts only): the FRESH
walls of the shipped single-request path (WALLS_FRESH=1 in tests/test_skip_positions.py).

Every timed chunk runs at NEW positions and in a NEW serving slot, as serving sees them:
- positions: each request's history drawn from the AgentX per-turn distribution (history p50 139k / p90 547k; new
  tokens p25 640 / p50 1.6k / p75 2.9k / p90 5.9k), 64-aligned, every draw its own seed (logged);
- slots: ``Slots`` = serving's slots: the adapter's engine-owned KV cache (DS41PrefillAdapter.allocate_kv_cache: a
  DS41ContractKV of WALLS_FRESH_SLOTS slots at the walls' capacity over the served layers) and the runtime's
  ``state(kv, slot)`` (bound to the slot's slabs, keep_history=False), every slot's state built before timing (a
  serving process holds them after a slot's first request). Warm-up draws take slots of the warm pool only, timed
  draws slots of the timed pool only; a slot index enters the served slab's batch row (slot x layers + row), so a
  slot-keyed program shows;
- program-cache misses ALLOWED and COUNTED (never refused): program_builds_after_warmup_<case> = the cache entries the
  timed fresh iterations added (the untimed real previous chunk included: in serving it is a fresh chunk too);
  ``PrepCount``: ttnn.bringup.prepare_program calls in them (a prepared program with value slots keeps one cache hash,
  so a new prepare adds no cache entry);
- the ops that built: ``BuildProbe``: extra fresh draws (untimed) with every ttnn / ttnn.bringup / ttnn.experimental
  callable wrapped; a call that added a cache entry (not one of a nested wrapped call) is charged to (op, the model's
  call site) with the call's arguments diffed against the previous call at that site: the attribute that changed
  (e.g. ttnn.slice's slice_start);
- next to each fresh number the warm-repeated one of the same chunks (each timed draw run again right after its fresh
  run: same positions, slot and layout), so the gap is explicit; the device ms from a profiled repeat (the real-time
  profiler) of the first WALLS_FRESH_PROF draws; host ms a chunk = wall - device of the same chunks.

Metrics: <case>_fresh_tok_s (> 0), program_builds_after_warmup_<case> (>= 0); recorded: repeat_tok_s_<case>,
fresh_wall_ms_chunk_<case>, repeat_wall_ms_chunk_<case>, fresh_host_ms_chunk_<case>, repeat_host_ms_chunk_<case>,
device_ms_chunk_<case>, probe_builds_<case>, prepared_builds_<case>, fresh_row_<case> (the table row), and the table
<results>/<task>_fresh_walls.md (every fresh_row_* of the task).
"""

from __future__ import annotations

import collections
import functools
import os
import random
import sys
import time

from models.demos.common.bringup.core import metrics

ENV = "WALLS_FRESH"
ITEM_VARS = (
    "WALLS_FRESH",
    "WALLS_FRESH_SLOTS",
    "WALLS_FRESH_WARM",
    "WALLS_FRESH_RUNS",
    "WALLS_FRESH_PROBE",
    "WALLS_FRESH_PROF",
    "WALLS_FRESH_CASES",
)
AGENTX_NEW = [(0.0, 64), (0.25, 640), (0.5, 1600), (0.75, 2900), (0.9, 5900), (1.0, 12000)]
AGENTX_HIST = [(0.0, 4096), (0.25, 61440), (0.5, 139264), (0.75, 307200), (0.9, 546816), (1.0, 1000000)]
HANG_MARKS = ("timeout", "potential hang", "timed out")


def on() -> bool:
    return os.environ.get(ENV, "").strip() not in ("", "0", "off")


def knob(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else int(default)


def cases(default: list) -> list:
    """WALLS_FRESH_CASES (comma-separated) restricts a test's fresh cases."""
    v = os.environ.get("WALLS_FRESH_CASES", "").strip()
    return [c for c in default if c in set(v.split(","))] if v else list(default)


def inv(q: list, u: float) -> int:
    """Piecewise-linear inverse CDF through the quantile points ``q`` [(p, value)]."""
    for (p0, v0), (p1, v1) in zip(q, q[1:]):
        if u <= p1:
            return int(v0 + (v1 - v0) * (u - p0) / max(p1 - p0, 1e-9))
    return int(q[-1][1])


class Draw:
    """One seeded draw of an AgentX request (its position, new tokens and slot)."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.rnd = random.Random(self.seed)

    def history(self, lo: int = 0, hi: int | None = None, align: int = 64) -> int:
        """A history length from AGENTX_HIST, ``align``-aligned, in [lo, hi] (redrawn outside: the few AgentX
        histories shorter than a previous chunk)."""
        for _ in range(1000):
            h = inv(AGENTX_HIST, self.rnd.random()) // align * align
            if h >= lo and (hi is None or h <= hi):
                return h
        raise AssertionError(f"no AgentX history in [{lo}, {hi}]")

    def new_tokens(self) -> int:
        return max(1, inv(AGENTX_NEW, self.rnd.random()))


def is_hang(e: BaseException) -> bool:
    low = str(e).lower()
    return any(m in low for m in HANG_MARKS)


# ------------------------------------------------------------------------------------------------------------ slots
class Slots:
    """Serving's slots: the adapter's engine-owned KV cache (``kv``: DS41KvCaches of ``kv.contract.num_users`` slots)
    and the runtime's per-slot state (``runtime.state(kv, slot)``: bound to the slot's slabs, keep_history=False),
    every slot's state built here, untimed. Slots [0, warm) are the warm-up pool, the rest the timed pool; ``pick``
    hands out the least used slots of a pool (random among equals), distinct within a call."""

    def __init__(self, runtime, kv, warm: int | None = None):
        self.runtime, self.kv = runtime, kv
        self.n = int(kv.contract.num_users)
        self.warm = warm or max(1, self.n // 2)
        self.used = collections.Counter()
        t0 = time.time()
        for slot in range(self.n):  # (a serving process holds every slot's state after its first request)
            runtime.state(kv, slot)
        print(
            f"FRESH slots: {self.n} serving slots (adapter KV cache, layers {kv.layer_ids[0]}..{kv.layer_ids[-1]}), "
            f"states built in {time.time() - t0:.1f} s (untimed); warm-up pool 0..{self.warm - 1}, timed pool "
            f"{self.warm}..{self.n - 1}",
            flush=True,
        )

    def pool(self, phase: str) -> list:
        return list(range(self.warm)) if phase == "warm" else list(range(self.warm, self.n))

    def pick(self, k: int, phase: str, rnd: random.Random, exclude=()) -> list:
        cand = [s for s in self.pool(phase) if s not in set(exclude)]
        assert len(cand) >= k, f"{k} slots asked of the {phase} pool ({len(cand)} free): raise WALLS_FRESH_SLOTS"
        rnd.shuffle(cand)
        cand.sort(key=lambda s: self.used[s])
        out = cand[:k]
        self.used.update(out)
        return out

    def free(self) -> None:
        import ttnn

        for key in [k for k in self.runtime._states if k[0] == id(self.kv)]:
            st = self.runtime._states.pop(key)
            try:
                st.free()
            except Exception as e:  # noqa: BLE001 (a bound state frees its own buffers only)
                print(f"FRESH slots: state free: {e}")
        for t in self.kv.contract.by_config.values():
            if t.is_allocated():
                ttnn.deallocate(t)


# ------------------------------------------------------------------------------------------------------- the probe
ARG_NAMES = {
    "slice": ("input", "slice_start", "slice_end", "slice_step"),
    "concat": ("tensors", "dim"),
    "reshape": ("input", "shape"),
    "pad": ("input", "padding", "value"),
    "to_memory_config": ("input", "memory_config"),
}
SKIP = {"num_program_cache_entries", "deallocate", "is_tensor_storage_on_device", "get_device_tensors", "to_torch"}


def _sig(v, depth: int = 0):
    """A compact, comparable summary of an op argument (tensors by shape / dtype / layout / memory config)."""
    import ttnn

    if isinstance(v, ttnn.Tensor):
        try:
            mc = str(v.memory_config()) if v.is_allocated() and ttnn.is_tensor_storage_on_device(v) else "host"
        except Exception:  # noqa: BLE001
            mc = "?"
        return ("T", tuple(int(x) for x in v.shape), str(v.dtype), str(v.layout), hash(mc))
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        if depth > 2:
            return ("seq", len(v))
        return tuple(_sig(x, depth + 1) for x in v[:64]) + ((("+", len(v)),) if len(v) > 64 else ())
    if isinstance(v, dict):
        return tuple(sorted((str(k), _sig(x, depth + 1)) for k, x in list(v.items())[:32]))
    r = repr(v)
    return (type(v).__name__, hash(r), r[:60])


def _short(v) -> str:
    s = str(v)
    return s if len(s) <= 48 else s[:45] + "..."


def _diff(a, b) -> str:
    """Where two argument signatures differ: 'name[i] x -> y' (first few elements)."""
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b) and a[:1] != ("T",):
        idx = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if idx and len(idx) <= 4 and all(not isinstance(a[i], tuple) for i in idx):
            return ", ".join(f"[{i}] {_short(a[i])} -> {_short(b[i])}" for i in idx)
    if isinstance(a, tuple) and a[:1] == ("T",) and isinstance(b, tuple) and b[:1] == ("T",):
        parts = [n for n, x, y in zip(("", "shape", "dtype", "layout", "memory_config"), a, b) if x != y]
        return "tensor " + "/".join(parts) + (f" {a[1]} -> {b[1]}" if "shape" in parts else "")
    return f"{_short(a)} -> {_short(b)}"


class PrepCount:
    """Counts ttnn.bringup.prepare_program calls (a prepared program built: a Python descriptor + its prepare) and the
    host ms inside them while installed. A prepared program whose per-run values are value slots keeps one program-cache
    hash for every layout, so a NEW prepare for a new layout adds no cache entry: counted here instead. Cheap (one
    counter around one op), so it is on in the timed runs."""

    def __init__(self):
        self.n, self.ms, self.f = 0, 0.0, None

    def __enter__(self):
        import ttnn

        f = self.f = ttnn.bringup.prepare_program
        me = self

        def g(*a, **k):
            t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                me.n += 1
                me.ms += (time.perf_counter() - t0) * 1e3

        try:
            functools.update_wrapper(g, f)
        except Exception:  # noqa: BLE001
            g.__doc__ = getattr(f, "__doc__", None)
        ttnn.bringup.prepare_program = g
        return self

    def __exit__(self, *exc):
        import ttnn

        ttnn.bringup.prepare_program = self.f
        return False


WAITS = ("synchronize", "from_device", "to_torch", "wait", "read", "event")  # (device waits: not host work)
SLOW_S = 0.005  # a host gap / an op call longer than this is attributed


class BuildProbe:
    """Charge every program-cache entry added inside the ``with`` block to the innermost wrapped ttnn call that added it
    (op, the model's call site: up to 3 tt/ frames), with the arguments that differ from the previous call at that
    site. ``rows()``: [(builds, op, site, {attribute: count}, an example change)]."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.hits = {}
        self.last = {}
        self.stack = []
        self.undo = []
        self.total = 0
        self.e0 = None
        self.preps = collections.Counter()  # (op site) -> prepare_program calls
        self.prep_ms = collections.Counter()
        self.gaps = collections.Counter()  # site -> host ms between the previous op's return and this call (> SLOW_S)
        self.gap_n = collections.Counter()
        self.slow = collections.Counter()  # (op, site) -> host ms inside top-level non-wait calls longer than SLOW_S
        self.slow_n = collections.Counter()
        self.t_last = None

    def _key(self, mod, name):
        fr, where = sys._getframe(2), []
        while fr is not None and len(where) < 3:
            fn = fr.f_code.co_filename
            if "deepseek_v41_flash_d_p/tt/" in fn:
                where.append(f"{fn.split('/tt/')[-1]}:{fr.f_lineno}:{fr.f_code.co_name}")
            fr = fr.f_back
        return (f"{mod.__name__}.{name}", " < ".join(where) or "?")

    def _wrap(self, mod, name, f):
        probe = self
        names = ARG_NAMES.get(name, ())

        wait = any(w in name for w in WAITS)

        def g(*a, **k):
            t_in = time.perf_counter()
            key = probe._key(mod, name)
            top = not probe.stack
            if top and probe.t_last is not None and t_in - probe.t_last > SLOW_S and key[1] != "?":
                probe.gaps[key[1].split(" < ")[0]] += (t_in - probe.t_last) * 1e3
                probe.gap_n[key[1].split(" < ")[0]] += 1
            sig = {**{(names[i] if i < len(names) else f"arg{i}"): _sig(x) for i, x in enumerate(a)}}
            sig.update({kk: _sig(x) for kk, x in k.items()})
            probe.stack.append(0)
            e0 = probe.mesh.num_program_cache_entries()
            t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                t1 = time.perf_counter()
                if name == "prepare_program":
                    probe.preps[key[1]] += 1
                    probe.prep_ms[key[1]] += (t1 - t0) * 1e3
                if top and not wait and t1 - t0 > SLOW_S:
                    probe.slow[key] += (t1 - t0) * 1e3
                    probe.slow_n[key] += 1
                e1 = probe.mesh.num_program_cache_entries()
                inner = probe.stack.pop()
                own = (e1 - e0) - inner
                if probe.stack:
                    probe.stack[-1] += e1 - e0
                if own > 0:
                    prev = probe.last.get(key)
                    h = probe.hits.setdefault(key, {"n": 0, "attrs": collections.Counter(), "ex": ""})
                    h["n"] += own
                    if prev is None:
                        h["attrs"]["(first call at this site)"] += own
                    else:
                        changed = [n for n in sorted(set(prev) | set(sig)) if prev.get(n) != sig.get(n)]
                        for n in changed or ["(no argument changed)"]:
                            h["attrs"][n] += own
                        if changed and not h["ex"]:
                            n = changed[0]
                            h["ex"] = f"{n}: {_diff(prev.get(n), sig.get(n))}"
                probe.last[key] = sig
                if not probe.stack:
                    probe.t_last = time.perf_counter()

        try:
            functools.update_wrapper(g, f)
        except Exception:  # noqa: BLE001
            g.__doc__ = getattr(f, "__doc__", None)
        return g

    def __enter__(self):
        import ttnn

        self.e0 = self.mesh.num_program_cache_entries()
        for mod in (ttnn, ttnn.bringup, ttnn.experimental):
            for name in dir(mod):
                if name.startswith("_") or name in SKIP:
                    continue
                f = getattr(mod, name, None)
                if f is None or isinstance(f, type) or not callable(f) or type(f).__name__ == "module":
                    continue
                try:
                    setattr(mod, name, self._wrap(mod, name, f))
                    self.undo.append((mod, name, f))
                except (AttributeError, TypeError):
                    pass
        return self

    def __exit__(self, *exc):
        for mod, name, f in reversed(self.undo):
            setattr(mod, name, f)
        self.undo.clear()
        self.total = self.mesh.num_program_cache_entries() - self.e0
        return False

    def rows(self) -> list:
        out = []
        for (op, site), h in sorted(self.hits.items(), key=lambda x: -x[1]["n"]):
            out.append((h["n"], op, site, dict(h["attrs"].most_common(4)), h["ex"]))
        return out

    def unattributed(self) -> int:
        return self.total - sum(h["n"] for h in self.hits.values())

    def summary(self, top: int = 8) -> list:
        """Short strings for the table: 'op xN (attr: example) @ site'."""
        out = []
        for n, op, site, attrs, ex in self.rows()[:top]:
            a = ", ".join(attrs)
            out.append(f"{op.replace('ttnn.', '')} x{n} [{a}{'; ' + ex if ex else ''}] @ {site.split(' < ')[0]}")
        if self.unattributed():
            out.append(f"(unattributed x{self.unattributed()}: built outside a wrapped call)")
        for site, n in self.preps.most_common(4):
            out.append(f"prepare_program x{n} ({self.prep_ms[site]:.0f} ms) @ {site.split(' < ')[0]}")
        return out

    def host_spots(self, top: int = 5) -> list:
        """The probe chunks' largest host costs outside device waits: gaps before a call (Python work since the previous
        op returned, e.g. a descriptor build) by site, and slow op calls by (op, site), ms summed."""
        out = [(ms, f"gap {ms:.0f} ms ({self.gap_n[s]}x) before @ {s}") for s, ms in self.gaps.items()]
        out += [
            (ms, f"{op.replace('ttnn.', '')} {ms:.0f} ms ({self.slow_n[(op, site)]}x) @ {site.split(' < ')[0]}")
            for (op, site), ms in self.slow.items()
        ]
        return [t for _, t in sorted(out, reverse=True)[:top]]

    def print(self, label: str) -> None:
        print(f"PROBE {label}: {self.total} program-cache entries added, {self.unattributed()} unattributed")
        for n, op, site, attrs, ex in self.rows():
            print(f"PROBE MISS {n:4d}  {op}  [{attrs}]  {ex}  <-  {site}", flush=True)
        for site, n in self.preps.most_common():
            print(f"PROBE PREPARE {n:4d}  {self.prep_ms[site]:.1f} ms  <-  {site}", flush=True)
        for t in self.host_spots(20):
            print(f"PROBE HOST {t}", flush=True)


# ------------------------------------------------------------------------------------------------- the results
def _record_gate_name(name: str, value) -> None:
    """The gate's threshold names carry no text tag: the metric under both names (test_packed_walls' helper)."""
    metrics.record(name, value)
    tok = os.environ.pop("BRINGUP_POS_TOKENS", None)
    try:
        metrics.record(name, value)
    finally:
        if tok is not None:
            os.environ["BRINGUP_POS_TOKENS"] = tok


def record(case: str, row: dict) -> None:
    """The gated metrics of a fresh case and its table row (fresh_row_<case>)."""
    if row.get("error"):
        metrics.record(f"fresh_failed_{case}", row["error"][:300])
    else:
        _record_gate_name(f"{case}_fresh_tok_s", round(row["fresh_tok_s"], 1))
        _record_gate_name(f"program_builds_after_warmup_{case}", int(row["builds"]))
        for k in (
            "repeat_tok_s",
            "fresh_wall_ms_chunk",
            "repeat_wall_ms_chunk",
            "fresh_host_ms_chunk",
            "repeat_host_ms_chunk",
            "device_ms_chunk",
            "probe_builds",
            "prepared_builds",
            "prepared_ms",
        ):
            if row.get(k) is not None:
                metrics.record(f"{k}_{case}", round(row[k], 2) if isinstance(row[k], float) else row[k])
    metrics.record(f"fresh_row_{case}", row)
    write_table()


def summarize(
    case: str, path: str, fresh: list, repeat: list, prof: list, builds: int, probe, extra: dict, preps=None
) -> dict:
    """``fresh`` / ``repeat``: [(wall ms, new tokens)] a timed chunk (the same chunks, in order); ``prof``: [(fresh
    wall, repeat wall, device ms)] of the profiled chunks; ``probe``: a BuildProbe (or None)."""
    wf, wr = sum(w for w, _ in fresh), sum(w for w, _ in repeat)
    new = sum(n for _, n in fresh)
    row = {
        "case": case,
        "path": path,
        "chunks": len(fresh),
        "new_tokens": new,
        "fresh_tok_s": new / (wf / 1e3) if wf else 0.0,
        "repeat_tok_s": new / (wr / 1e3) if wr else 0.0,
        "fresh_wall_ms_chunk": wf / max(1, len(fresh)),
        "repeat_wall_ms_chunk": wr / max(1, len(repeat)),
        "builds": int(builds),
        "prepared_builds": None if preps is None else int(preps[0]),
        "prepared_ms": None if preps is None else round(preps[1], 1),
        "fresh_walls": [round(w, 1) for w, _ in fresh],
        "repeat_walls": [round(w, 1) for w, _ in repeat],
    }
    if prof:
        d = sum(x[2] for x in prof) / len(prof)
        row["device_ms_chunk"] = d
        row["fresh_host_ms_chunk"] = sum(x[0] for x in prof) / len(prof) - d
        row["repeat_host_ms_chunk"] = sum(x[1] for x in prof) / len(prof) - d
        row["profiled_chunks"] = len(prof)
    if probe is not None:
        row["probe_builds"] = int(probe.total)
        row["probe_ops"] = probe.summary()
        row["probe_detail"] = [list(r) for r in probe.rows()[:30]]
        row["probe_host"] = probe.host_spots()
        row["probe_prepares"] = int(sum(probe.preps.values()))
    row.update(extra)
    print(
        f"FRESH {case} [{path}]: {len(fresh)} timed chunks, {new} new tokens: fresh {row['fresh_tok_s']:.0f} tok/s "
        f"({row['fresh_wall_ms_chunk']:.1f} ms a chunk), warm-repeated {row['repeat_tok_s']:.0f} tok/s "
        f"({row['repeat_wall_ms_chunk']:.1f} ms), builds after warm-up {builds} (+ prepared {preps}), device "
        f"{row.get('device_ms_chunk', float('nan')):.1f} ms, host fresh {row.get('fresh_host_ms_chunk', float('nan')):.1f} / "
        f"repeat {row.get('repeat_host_ms_chunk', float('nan')):.1f} ms a chunk; probe {row.get('probe_ops')}",
        flush=True,
    )
    return row


def failed(case: str, path: str, e: BaseException, extra: dict | None = None) -> dict:
    msg = f"{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ''}"[:400]
    print(f"FRESH {case} [{path}] FAILED: {msg}", flush=True)
    return {"case": case, "path": path, "error": msg, **(extra or {})}


def _f(row, k, fmt="{:.0f}"):
    v = row.get(k)
    return "-" if v is None else fmt.format(v)


def write_table() -> None:
    task = metrics.record_task()
    rows = [v["value"] for k, v in sorted(metrics.load(task).items()) if k.startswith("fresh_row_")]
    rows = [r for r in rows if isinstance(r, dict)]
    kinds = ["steady", "resumed", "partial"]
    rows.sort(
        key=lambda r: (r["case"].split("_")[0], next((i for i, k in enumerate(kinds) if r["case"].endswith(k)), 9))
    )
    head = [
        "case",
        "path",
        "warm-repeated tok/s",
        "fresh tok/s",
        "gap",
        "builds after warm-up: program cache + prepared (timed chunks)",
        "ops that built (probe: op xN [attribute changed] @ call site)",
        "host spots (probe: gaps before a call / slow calls)",
        "host ms / chunk fresh (repeated)",
        "device ms / chunk",
    ]
    lines = [
        f"# Fresh walls ({task}: WALLS_FRESH=1, every timed chunk at new positions and slots)",
        "",
        "Fresh = every timed chunk at NEW positions (AgentX history draws, seeds logged) in a NEW serving slot (the "
        "adapter's engine-owned KV cache; warm-up and timed draws use disjoint slot pools), program-cache misses allowed "
        "and counted. Warm-repeated = the same chunks run again right after their fresh run (positions and slot seen "
        "before: a best case). Wall = runtime.prefill_chunk (the engine's device input already on the device -> ids "
        "readback -> note_tokens / begin_chunk -> every served layer, layers 0-20, DECODER_SKIP=decode) -> sync. Host = "
        "wall - device (the slowest chip, real-time profiler, profiled repeats). Ops: extra fresh draws (untimed) with "
        "every ttnn call wrapped; a build is charged to the innermost call that made it and its arguments diffed "
        "against the previous call at that site.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r['case']} | {r['path']} | - | FAILED: {r['error']} | - | - | - | - | - | - |")
            continue
        gap = r["fresh_tok_s"] / r["repeat_tok_s"] - 1 if r.get("repeat_tok_s") else 0.0
        ops = "<br>".join(r.get("probe_ops") or []) or "none"
        pb = (
            f" (probe {r['probe_builds']} + {r.get('probe_prepares', '?')} in {r.get('probe_chunks', '?')} chunks)"
            if "probe_builds" in r
            else ""
        )
        spots = "<br>".join(r.get("probe_host") or []) or "-"
        prep = "?" if r.get("prepared_builds") is None else f"{r['prepared_builds']} ({r.get('prepared_ms', 0):.0f} ms)"
        lines.append(
            f"| {r['case']} | {r['path']} | {r['repeat_tok_s']:.0f} | {r['fresh_tok_s']:.0f} | {gap:+.1%} | "
            f"{r['builds']} + {prep} ({r['chunks']}){pb} | {ops} | {spots} | {_f(r, 'fresh_host_ms_chunk', '{:.1f}')} "
            f"({_f(r, 'repeat_host_ms_chunk', '{:.1f}')}) | {_f(r, 'device_ms_chunk', '{:.1f}')} |"
        )
    lines += ["", "## Draws (seeds, positions, slots)", ""]
    for r in rows:
        d = {k: r[k] for k in ("seeds", "draws", "note") if k in r}
        lines.append(f"- {r['case']}: {d}")
    out = metrics.results_dir() / f"{task}_fresh_walls.md"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        print(f"fresh walls table -> {out}")
    except OSError as e:
        print(f"fresh walls table not written ({out}): {e}")
