# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.36: loop reproduction of the intermittent test_skip_positions hang (DECODER_SKIP=decode, shipped tree).

The same per-point work as tests/test_skip_positions.py (its _point loop: a fresh state sized p + chunk, then per kind
steady / resumed: the real previous chunk at p - chunk, then the chunk at p, last=False, resumed with write_start =
p + 128), repeated over HANGLOOP_POS for HANGLOOP_MINUTES (or HANGLOOP_ROUNDS rounds), one model load. Every chunk run
prints one progress line (round, point, kind, run, side) with flush first, so a hang names the chunk it stopped in;
the dispatch timeout's triage (scripts/run_safe_pytest.sh) names the op. Outputs: the last layer's hidden of the
steady chunk at the first point (the last full layer, model.decoder[0] - 1) is read back once per round and compared with round 0's (determinism, bytes).

    HANGLOOP_POS       points (default 256000,1024000)
    HANGLOOP_RUNS      one_run calls a kind a point a round (default 23 = test_skip_positions' ~12 warm + 10 timed + 1)
    HANGLOOP_PROFILE   1: the rt profiler around the last run of each kind, as the test (default 1)
    HANGLOOP_MINUTES   stop after this many minutes (default 20); HANGLOOP_ROUNDS caps the rounds (default 1000)
    HANGLOOP_WATCH     1 (default): a real-time profiler callback for the whole loop keeps, per kernel set, the
                       most cores any program ran on (rule 18: anything over tt/cores.max_worker_cores is printed as
                       OVER at the end of every round), and per chip the last programs that finished; on an error
                       (a hang) it prints those per chip before the teardown: the program after them is the one in
                       flight (FIX.36)
    HANGLOOP_GATHER_ROWS  P.73: "alt" alternates the STATIC_TABLES=off history gather per round (even rounds as
                       loaded, odd rounds the sparse attentions' ``gather_rows`` flipped: HISTORY_GATHER_OVERLAP on /
                       off in one load); each side keeps its own determinism reference, and the other side's output
                       is compared with it too (on vs off bit-identity, recorded, asserted)
    HANGLOOP_STOP_OVER  1: stop after the first round that ran a program over the core cap (an audit run)

    PYTHONPATH=$PWD DS41_DECODER_SKIP=decode scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_skip_hang_loop.py -s
"""

from __future__ import annotations

import collections
import os
import threading
import time

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden
from models.demos.deepseek_v41_flash_d_p.tt.cores import max_worker_cores
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True
REPLAY = 128


class _Watch:
    """FIX.36: the real-time profiler's records for the whole loop (see HANGLOOP_WATCH)."""

    def __init__(self, cap: int):
        import ttnn

        self.cap = cap
        self.lock = threading.Lock()
        self.cores = {}  # kernel set -> (most cores, programs seen)
        self.last = collections.defaultdict(lambda: collections.deque(maxlen=6))  # chip -> recent records
        self.n = 0
        self.dropped = 0
        self.ttnn = ttnn
        self.h = ttnn.device.RegisterProgramRealtimeProfilerCallback(self.on_batch)

    def on_batch(self, batch):
        with self.lock:
            self.dropped += int(getattr(batch, "dropped", 0) or 0)
            for r in batch.records:
                self.n += 1
                c = int(r.core_count)
                key = tuple(x.rsplit("/", 1)[-1] for x in r.kernel_sources)
                m, k = self.cores.get(key, (0, 0))
                self.cores[key] = (max(m, c), k + 1)
                f = float(r.frequency) or 1.0
                self.last[int(r.chip_id)].append(
                    (int(r.runtime_id), c, (r.end_timestamp - r.start_timestamp) / f / 1e3, key, r.end_timestamp / f)
                )

    def over(self) -> list:
        with self.lock:
            return sorted(((m, k, key) for key, (m, k) in self.cores.items() if m > self.cap), reverse=True)

    def report(self, tag: str) -> None:
        with self.lock:
            print(
                f"HANGWATCH {tag}: {self.n} records, {self.dropped} dropped, {len(self.cores)} kernel sets", flush=True
            )
            top = sorted(((m, k, key) for key, (m, k) in self.cores.items()), reverse=True)[:8]
        for m, k, key in top:
            print(f"HANGWATCH cores {m}{' OVER' if m > self.cap else ''} x{k}: {' '.join(key)}", flush=True)

    def dump_last(self) -> None:
        with self.lock:
            for chip in sorted(self.last):
                t_end = max(x[4] for x in self.last[chip])
                for rid, c, us, key, end in self.last[chip]:
                    print(
                        f"HANGWATCH chip {chip} last rid {rid} cores {c} {us:.1f} us end {end - t_end:+.0f} ns: "
                        f"{' '.join(key)}",
                        flush=True,
                    )

    def close(self) -> None:
        self.ttnn.device.UnregisterProgramRealtimeProfilerCallback(self.h)


def _env_list(name: str, default: str) -> list[int]:
    return [int(x) for x in os.environ.get(name, default).split(",") if x.strip()]


@mesh_parametrize(timeout_s=4 * 3600)
def test_skip_hang_loop(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    assert settings.get("DECODER_SKIP") == "decode", "run with DS41_DECODER_SKIP=decode"
    if os.environ.get(settings.env_name("ENGRAM_TABLE")) is None:
        monkeypatch.setenv(settings.env_name("ENGRAM_TABLE"), str(settings.table["ENGRAM_TABLE"].default))
    chunk = int(S.get("target.chunk"))
    points = _env_list("HANGLOOP_POS", "256000,1024000")
    runs = int(os.environ.get("HANGLOOP_RUNS", "23"))
    prof = os.environ.get("HANGLOOP_PROFILE", "1") == "1"
    minutes = float(os.environ.get("HANGLOOP_MINUTES", "20"))
    max_rounds = int(os.environ.get("HANGLOOP_ROUNDS", "1000"))
    layers = list(S.layers())
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    want = max([p + chunk for p in points if p + chunk <= deploy] + [int(S.data["target"]["seq"])])
    monkeypatch.setitem(S.data["target"], "seq", want)
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    keep_layer = model.decoder[0] - 1  # (model.decoder: the skipped layers 20..39; 19 is the last full one)
    watch = _Watch(max_worker_cores(mesh_device)) if os.environ.get("HANGLOOP_WATCH", "1") == "1" else None
    print(
        f"HANGLOOP points {points} runs {runs} profile {prof} minutes {minutes} watch {watch is not None}", flush=True
    )

    ids = {p: PositionGolden(S, p, chunk).tokens() for p in points}
    alt = os.environ.get("HANGLOOP_GATHER_ROWS", "") == "alt"
    stop_over = os.environ.get("HANGLOOP_STOP_OVER", "0") == "1"
    sparse = [b.attention for b in model.blocks.values() if hasattr(b.attention, "gather_rows")]
    loaded_rows = {id(b): b.gather_rows for b in sparse}
    if alt:
        assert (
            settings.get("STATIC_TABLES") == "off" and sparse
        ), "HANGLOOP_GATHER_ROWS=alt: run with DS41_STATIC_TABLES=off"
    refs = {}  # gather side -> the steady layer output of its first round
    rounds_of = collections.Counter()
    ref = None
    chunks = 0
    t_start = time.time()
    rnd = 0
    try:
        while rnd < max_rounds and time.time() - t_start < minutes * 60:
            side = "as_loaded"
            if alt:
                flip = rnd % 2 == 1
                for b in sparse:
                    b.gather_rows = loaded_rows[id(b)] != flip
                side = "rows_on" if sparse[0].gather_rows else "rows_off"
                print(f"HANGLOOP r{rnd} side {side}", flush=True)
            for p in points:
                prev_t, cur_t = ids[p][p - chunk : p], ids[p][p : p + chunk]
                state = model.new_state(p + chunk)
                try:

                    def run(tokens, start, write_start=None, keep=False):
                        out = None
                        model.begin_chunk(tokens, start, last=False, write_start=write_start)
                        h = model.embed(tokens)
                        for i in layers:
                            h2 = model.layer(i, h, start, state)
                            model.free(h)
                            h = h2
                            if keep and i == keep_layer:  # the last full layer's output (decode: later ones Skipped)
                                out = model.to_host(h)
                        model.free(h)
                        return out

                    for kind, ws in (("steady", None), ("resumed", p + REPLAY)):
                        for r in range(runs):
                            last = prof and r == runs - 1
                            print(f"HANGLOOP r{rnd} p{p} {kind} run{r} prev t={time.time() - t_start:.1f}", flush=True)
                            run(prev_t, p - chunk)
                            model.sync()
                            print(f"HANGLOOP r{rnd} p{p} {kind} run{r} cur{' prof' if last else ''}", flush=True)
                            if last:
                                profiler.enable(mesh_device, rt=True)
                            try:
                                keep = kind == "steady" and p == points[0] and r == 0
                                out = run(cur_t, p, ws, keep=keep)
                                model.sync()
                                if last:
                                    profiler.finish_rt()
                            finally:
                                if last:
                                    profiler.disable()
                            if out is not None and alt:
                                for k, v in refs.items():
                                    if k != side:
                                        x = (out.float() - v.float()).abs().max().item()
                                        print(
                                            f"HANGLOOP r{rnd} {side} vs {k} L{keep_layer} max_abs_diff {x}", flush=True
                                        )
                                        metrics.record("hangloop_gather_rows_on_off_diff", x)
                                        assert x == 0, f"round {rnd}: {side} differs from {k} by {x}"
                                ref = refs.get(side)
                                if ref is None:
                                    refs[side] = out.clone()
                                    out = None
                            if out is not None:
                                if ref is None:
                                    ref = out.clone()
                                else:
                                    diff = (out.float() - ref.float()).abs().max().item()
                                    print(f"HANGLOOP r{rnd} determinism L{keep_layer} max_abs_diff {diff}", flush=True)
                                    metrics.record("hangloop_max_abs_diff", diff)
                                    assert diff == 0, f"round {rnd}: output differs from round 0 by {diff}"
                            chunks += 2
                finally:
                    state.free()
            rnd += 1
            rounds_of[side] += 1
            print(
                f"HANGLOOP round {rnd} done ({side}): {chunks} chunks, {time.time() - t_start:.0f} s, "
                f"rounds a side {dict(rounds_of)}",
                flush=True,
            )
            if watch is not None:
                watch.report(f"round {rnd}")
                if stop_over and watch.over():
                    print("HANGLOOP STOP: a program over the cap (HANGLOOP_STOP_OVER=1)", flush=True)
                    break
    except BaseException:
        if watch is not None:  # (a hang: what each chip finished last, before the teardown)
            time.sleep(1.0)
            watch.report("at the error")
            watch.dump_last()
        raise
    finally:
        if watch is not None:
            watch.close()
        for b in sparse:
            b.gather_rows = loaded_rows[id(b)]
        if loaded != "decode":
            model.use_decoder_skip(loaded)
    metrics.record("hangloop_rounds", rnd)
    for k, v in rounds_of.items():
        metrics.record(f"hangloop_rounds_{k}", v)
    metrics.record("hangloop_chunks", chunks)
    if watch is not None:
        over = watch.over()
        metrics.record("hangloop_programs_over_cap", len(over))
        assert not over, f"programs over {watch.cap} cores (rule 18): {over[:5]}"
    print(f"HANGLOOP CLEAN: {rnd} rounds {dict(rounds_of)}, {chunks} chunks, {time.time() - t_start:.0f} s", flush=True)
