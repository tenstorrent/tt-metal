# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Startup progress for the demo: one line when each phase starts, then at most one line every ~10 s with
what the phase has done so far, and an end line with each phase's time.

    st = Startup(log, key="<cache>|<max_ctx>|<spec>")
    st.phase("open device")
    ...
    st.phase("weights", tensors=661, gb=17.0)   # totals when known
    ...
    st.done()                                   # "ready in 44 s (open device 3 s, weights 24 s, ...)"

Counters, all cheap and read only while starting:
- weights: `ttnn.load_tensor` calls and their file sizes (wrapped while starting, the original put back by done()).
- traces: `ttnn.end_trace_capture` calls (wrapped the same way).
- kernels: new kernel variant dirs in the JIT cache (`<cache>/<build key>/kernels/<kernel>/<hash>/`, one per
  compiled kernel); a tick stats only the kernel name dirs and lists the ones changed since the start.
  shortcut: another process compiling into the same JIT cache at the same time is counted too; a per-process
  count needs a tt-metal hook (the JIT build telemetry is not exposed to Python).

ETAs come from the last start with the same key (~/.cache/inkling2-mini/startup.json: each phase's time and
final counts), so the second start knows how many tensors, traces and kernels to expect; the first one gives
running counts only (weights: from the manifests' sizes)."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

LOAD_PHASES = ("weights", "drafter")  # phases that count tensors; the others count traces
RECORD = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "inkling2-mini" / "startup.json"


def jit_cache_root():
    return Path(os.environ.get("TT_METAL_CACHE") or Path.home() / ".cache" / "tt-metal-cache")


class KernelCount:
    """New kernel variant dirs in the JIT cache since t0."""

    def __init__(self, root=None, t0=None):
        self.root, self.t0 = Path(root or jit_cache_root()), (t0 or time.time()) - 1.0

    def __call__(self):
        n, keys = 0, []
        # $TT_METAL_CACHE=D puts the build keys in D/tt-metal-cache/ or as D/tt-metal-cache<key> (no slash)
        for r in (self.root, self.root / "tt-metal-cache"):
            try:
                keys += [e.path for e in os.scandir(r) if e.is_dir()]
            except OSError:
                pass
        for k in keys:
            try:
                names = list(os.scandir(os.path.join(k, "kernels")))
            except OSError:
                continue
            for d in names:
                try:
                    if d.stat().st_mtime < self.t0:
                        continue  # nothing added under this kernel since the start
                    n += sum(1 for h in os.scandir(d.path) if h.stat().st_mtime >= self.t0)
                except OSError:
                    continue
        return n


def _fmt_s(s):
    return f"{s:.0f} s" if s < 120 else f"{s / 60:.1f} min"


class Startup:
    def __init__(self, log=print, key="", every=10.0, record=RECORD, kernels=None):
        self.log, self.key, self.every, self.record = log, key, every, Path(record)
        self.t0 = time.time()
        self.kernels = kernels or KernelCount(t0=self.t0)
        try:
            self.last = json.loads(self.record.read_text()).get(key) or {}
        except (OSError, ValueError):
            self.last = {}
        self.times = {}  # phase -> seconds
        self.name, self.t_phase, self.totals = None, self.t0, {}
        self.tensors = self.bytes = self.traces = 0
        self.lock = threading.Lock()
        self.stop_ev = threading.Event()
        self._patched = None  # ttnn is wrapped by the first phase() after it is imported
        self.thread = threading.Thread(target=self._tick, name="inkling-startup", daemon=True)
        self.thread.start()

    # ------------------------------------------------------------ counters
    def _patch(self):
        import sys

        ttnn = sys.modules.get("ttnn")
        if ttnn is None or self._patched is not None:
            return
        load, end = ttnn.load_tensor, ttnn.end_trace_capture

        def load_tensor(path, *a, **kw):
            t = load(path, *a, **kw)
            try:
                size = os.path.getsize(path)
            except (OSError, TypeError):
                size = 0
            with self.lock:
                self.tensors += 1
                self.bytes += size
            return t

        def end_trace_capture(*a, **kw):
            r = end(*a, **kw)
            with self.lock:
                self.traces += 1
            return r

        ttnn.load_tensor, ttnn.end_trace_capture = load_tensor, end_trace_capture
        self._patched = [(ttnn, "load_tensor", load), (ttnn, "end_trace_capture", end)]

    def _unpatch(self):
        for mod, attr, orig in self._patched or []:
            setattr(mod, attr, orig)
        self._patched = []  # not again

    # ------------------------------------------------------------ phases
    def phase(self, name, **totals):
        """Ends the phase under way and starts `name`. totals: tensors=, gb=, traces=, kernels= estimates for when
        there is no last start (its counts are exact: the estimates miss tensors loaded twice)."""
        self._patch()
        now = time.time()
        with self.lock:
            self._end(now)
            self.name, self.t_phase, self.totals = name, now, totals
        last = self.last.get("phases", {}).get(name)
        self.log(f"startup: {name} ...{f' (last start: {_fmt_s(last)})' if last is not None else ''}")

    def _end(self, now):
        if self.name is not None:
            self.times[self.name] = self.times.get(self.name, 0.0) + now - self.t_phase

    def line(self):
        """'<phase>: <counts> , <elapsed> (eta ...)'; None before the first phase."""
        with self.lock:
            if self.name is None:
                return None
            name, t, tot = self.name, time.time() - self.t_phase, self.totals
            tensors, nbytes, traces = self.tensors, self.bytes, self.traces
        k = self.kernels()
        last = self.last
        parts, frac = [], None
        loading = name in LOAD_PHASES
        if tensors and loading:
            n_t, gb = last.get("tensors") or tot.get("tensors"), last.get("gb") or tot.get("gb")
            parts.append(f"tensors {tensors}{f'/{n_t}' if n_t else ''}")
            parts.append(f"{nbytes / 1e9:.1f}{f'/{gb:.1f}' if gb else ''} GB")
            if gb:  # counted from the first tensor on
                frac = nbytes / 1e9 / gb
        if k:
            n_k = last.get("kernels_cold") or tot.get("kernels")
            parts.append(f"kernels compiled {k}{f' (a cold start builds ~{n_k})' if n_k else ''}")
        if traces and not loading:
            n_tr = last.get("traces") or tot.get("traces")
            parts.append(f"traces {traces}{f'/{n_tr}' if n_tr else ''}")
            if n_tr and name in ("traces", "logits tails"):
                frac = traces / n_tr
        eta = None
        if frac and 0.02 < frac < 1:
            eta = t * (1 - frac) / frac
        elif last.get("phases", {}).get(name) is not None:
            eta = max(0.0, last["phases"][name] - t)
        e = f", eta {_fmt_s(eta)}" if eta is not None else ""
        total = _fmt_s(time.time() - self.t0)
        return f"startup: {name}: {', '.join(parts) or 'working'}; {_fmt_s(t)} in phase, {total} total{e}"

    def _tick(self):
        while not self.stop_ev.wait(self.every):
            if time.time() - self.t_phase < self.every / 2:
                continue  # the phase's start line was just logged
            s = self.line()
            if s:
                self.log(s)

    def done(self, what="model ready"):
        """Ends the last phase, puts ttnn back, logs the phase times and saves them for the next start's ETAs."""
        now = time.time()
        self.stop_ev.set()
        with self.lock:
            self._end(now)
            self.name = None
        self._unpatch()
        k = self.kernels()
        total = now - self.t0
        times = ", ".join(f"{n} {v:.1f} s" for n, v in self.times.items())
        self.log(
            f"startup: {what} in {total:.1f} s ({times}; {self.tensors} tensors, {self.bytes / 1e9:.1f} GB, "
            f"{k} kernels compiled, {self.traces} traces)"
        )
        rec = {
            "phases": {n: round(v, 1) for n, v in self.times.items()},
            "total": round(total, 1),
            "tensors": self.tensors,
            "gb": round(self.bytes / 1e9, 2),
            "traces": self.traces,
            "kernels": k,
            # the most kernels one start of this key compiled (a cold start); a warm start compiles none
            "kernels_cold": max(k, self.last.get("kernels_cold", 0)),
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            self.record.parent.mkdir(parents=True, exist_ok=True)
            try:
                allrec = json.loads(self.record.read_text())
            except (OSError, ValueError):
                allrec = {}
            allrec[self.key] = rec
            tmp = self.record.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(allrec, indent=1))
            tmp.replace(self.record)
        except OSError as ex:
            self.log(f"startup: could not save {self.record}: {ex}")
        return rec

    def fail(self):
        """Stop the ticker and put ttnn back (load failed); nothing is saved."""
        self.stop_ev.set()
        self._unpatch()
