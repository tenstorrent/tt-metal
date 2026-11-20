# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.32 (overseer 2026-11-14, DS41-276): test_decoder_skip_decode many times in ONE process, per side of the two
P.64 / P.65 switches, to prove the served-snapshot failure gone and bisect it.

The frozen test (tests/test_decoder_skip_decode.py) runs unchanged, called RUNS times a side on the one loaded model;
its serving runtime and KV cache are built once (C.build is the same adapter -> runtime -> engine cache -> compile;
the later calls get the first build), so every run differs only in what it sends. A side sets ATTN_FUSE and
CHEAP_CCL in the loaded model (settings.refresh: both are read per call, tt/attention.py). Every run is printed
(side, run, served_bytes_differ, the first failure line) and the counts are recorded.

    SNAPRACE_SIDES   comma list of legacy | fuse | ccl | both (fuse = ATTN_FUSE on only, ccl = CHEAP_CCL on only);
                     default legacy (the shipped defaults since FIX.32)
    SNAPRACE_RUNS    runs a side (default 20)
    SNAPRACE_DIAG    "1": after every whole chunk (and the test's own sync), layers SNAPRACE_DIAG_LAYERS (default 0)
                     of the slot: every chip's served snapshot vs the owner chip's ring rows (host reads, diagnostic:
                     which chips / blocks are wrong)

Found with it (DS41-277): not a race; SnapshotSend's program of the steady pattern, built by test_determinism for a
ladder cache's 1-batch slab, wrote the engine slab's snapshots into batch 0 (fixed: the program cache keyed by the io
geometry too). The P.67 batch where it showed ran the determinism test first on the same model; to repeat that
order, run both in one process (the model loaded once, at the switches given on the command line):

    DS41_ATTN_FUSE=on DS41_CHEAP_CCL=on SNAPRACE_SIDES=both PYTHONPATH=$PWD scripts/run_safe_pytest.sh --one-process \\
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_determinism.py \\
        models/demos/deepseek_v41_flash_d_p/tests/test_snapshot_race_loop.py

    SNAPRACE_SIDES=legacy SNAPRACE_RUNS=20 PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_snapshot_race_loop.py -s
"""

from __future__ import annotations

import os
import time

from loguru import logger

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.harness import mesh_parametrize
from models.demos.deepseek_v41_flash_d_p.tests import test_decoder_skip_decode as T
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

BATCH_SHARED = True
SIDES = {
    "legacy": dict(ATTN_FUSE="legacy", CHEAP_CCL="legacy"),
    "fuse": dict(ATTN_FUSE="on", CHEAP_CCL="legacy"),
    "ccl": dict(ATTN_FUSE="legacy", CHEAP_CCL="on"),
    "both": dict(ATTN_FUSE="on", CHEAP_CCL="on"),
}


def _set(monkeypatch, **kw):
    for k, v in kw.items():
        monkeypatch.setenv(settings.env_name(k), str(v))
    settings.refresh()


@mesh_parametrize(timeout_s=0)
def test_snapshot_race_loop(mesh_device, tmp_path, monkeypatch):
    sides = [s.strip() for s in os.environ.get("SNAPRACE_SIDES", "legacy").split(",") if s.strip()]
    runs = int(os.environ.get("SNAPRACE_RUNS", "20"))
    assert all(s in SIDES for s in sides), sides
    built = {}
    real_build = C.build

    def build_once(mesh, layers, users):
        key = (tuple(layers), users)
        if key not in built:
            built[key] = real_build(mesh, layers, users)
        return built[key]

    monkeypatch.setattr(C, "build", build_once)
    diag = os.environ.get("SNAPRACE_DIAG") == "1"
    diag_layers = [int(x) for x in os.environ.get("SNAPRACE_DIAG_LAYERS", "0").split(",")]
    diag_log = []
    got = {}
    real_record = metrics.record

    def record(name, value, **kw):
        got[name] = value
        return real_record(name, value, **kw)

    monkeypatch.setattr(metrics, "record", record)
    # (built first, at the process's switches: the batch's shared model is keyed by the switches read at load)
    _, runtime, kv = C.build(mesh_device, C.served_prefix(), 3)
    if diag:
        real_chunk = runtime.prefill_chunk

        def chunk_diag(inp, kvc, *, slot_id, actual_start, actual_end, **kw):
            out = real_chunk(inp, kvc, slot_id=slot_id, actual_start=actual_start, actual_end=actual_end, **kw)
            if actual_end - actual_start == runtime.model.chunk:
                import ttnn

                ttnn.synchronize_device(mesh_device)
                diag_log.append(_diag(runtime, kvc, slot_id, actual_end, diag_layers))
            return out

        monkeypatch.setattr(runtime, "prefill_chunk", chunk_diag)
    table = {}
    for side in sides:
        _set(monkeypatch, **SIDES[side])
        rows = []
        for k in range(runs):
            got.clear()
            t0 = time.perf_counter()
            msg = ""
            run_dir = tmp_path / f"{side}_{k}"
            run_dir.mkdir()
            try:
                T.test_decoder_skip_decode(mesh_device, run_dir)
                ok = True
            except AssertionError as e:
                ok, msg = False, str(e).splitlines()[0][:300]
            d = got.get("served_bytes_differ")
            cases = [(c["prompt"], c["stop"], c["differ"]) for c in got.get("decode_cases", []) if c["differ"]]
            rows.append((ok, d))
            line = (
                f"[snaprace] side {side} ({SIDES[side]}) run {k + 1}/{runs}: {'PASS' if ok else 'FAIL'} "
                f"served_bytes_differ {d} differing cases {cases} ({time.perf_counter() - t0:.0f} s) {msg}"
            )
            print(line)
            logger.info(line)
            for entry in diag_log:
                if entry["bad"]:
                    print(f"[snaprace] DIAG {side} run {k + 1}: {entry}")
            diag_log.clear()
        fails = sum(1 for ok, _ in rows if not ok)
        table[side] = (fails, runs, [d for _, d in rows])
        print(f"[snaprace] side {side}: {fails} of {runs} runs failed; served_bytes_differ per run {table[side][2]}")
    monkeypatch.setattr(metrics, "record", real_record)
    for side, (fails, n, ds) in table.items():
        metrics.record(f"snaprace_{side}_fails", fails, runs=n, differ=ds)
    for side, (fails, n, _) in table.items():
        print(f"[snaprace] SUMMARY {side}: {fails}/{n} failed")


def _diag(runtime, kv, slot: int, end: int, layers: list[int]) -> dict:
    """Every chip's served snapshot of ``layers`` vs the rows it must hold: position p (p in [cap - 128, cap)) at
    snapshot row p % 128, read from the ring of the block's owner (every chip's own ring when no chip owns it)."""
    import torch

    from models.demos.deepseek_v41_flash_d_p.tt.attention import RING, SNAP, snapshot_cap

    state = runtime.state(kv, slot)
    cap = snapshot_cap(end)
    bad = []
    owners = None
    for layer in layers:
        c = state.window[layer]
        if c.snap is None or cap < SNAP:
            continue
        ring = c.ring
        rings = ring.host_rows()
        snaps = c.snapshot_host()
        owners = [ring.own[(p // 32) % ring.BLOCKS] for p in range(cap - SNAP, cap, 32)]
        for chip, got in enumerate(snaps):
            for b in range(SNAP // 32):
                p0 = cap - SNAP + b * 32
                o = owners[b] if owners[b] is not None else chip
                want = rings[o][[(p % RING) for p in range(p0, p0 + 32)]]
                rows = got[[(p % SNAP) for p in range(p0, p0 + 32)]]
                if not torch.equal(rows, want):
                    n = int((rows != want).any(dim=1).sum())
                    zero = bool((rows == 0).all())
                    bad.append(dict(layer=layer, chip=chip, block=b, rows_wrong=n, all_zero=zero))
    return dict(slot=slot, end=end, cap=cap, owners=owners, bad=bad[:64], n_bad=len(bad))
