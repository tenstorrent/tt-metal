# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.6 ENGRAM_ASYNC: the engram rows filled by the worker thread are the off path's rows, and what the overlap hides.

    # the rows, byte for byte (no model; any ENGRAM_TABLE, both ENGRAM_ROWS formats)
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \
        "models/demos/deepseek_v41_flash_d_p/tt/perf_engram_async.py::test_async_rows_bit_identical"
    # the profile rung's last chunk, every layer, ENGRAM_ASYNC on and off in one process (alternating warm runs):
    # per engram layer the worker's ms (hash, lookup + pack, wrap, upload) and the main thread's wait; walls; on == off
    DS41_ENGRAM_TABLE=resident PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \
        "models/demos/deepseek_v41_flash_d_p/tt/perf_engram_async.py::test_engram_async_chunk"

Not a gate: the A/B the owner decides on is the orchestrator's (frozen tests, ladder, profile with the switch on / off).
"""

from __future__ import annotations

import statistics
import threading
import time

import pytest
import torch

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden, hf_path
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config
from models.demos.deepseek_v41_flash_d_p.reference.ngram import DEAD
from models.demos.deepseek_v41_flash_d_p.reference.weights import Checkpoint, EngramTable
from models.demos.deepseek_v41_flash_d_p.tt.engram import TtEngramRows, engram_hasher

RUNG = "last"
WARM = 8  # warm-up runs per side (program cache, host caches)
RUNS = 6  # timed runs per side, alternating on / off


def _same_on_device(a, b) -> bool:
    return all(
        torch.equal(ttnn.to_torch(x), ttnn.to_torch(y))
        for x, y in zip(ttnn.get_device_tensors(a), ttnn.get_device_tensors(b))
    )


@mesh_parametrize
def test_async_rows_bit_identical(mesh_device):
    """fill() on a worker thread into allocate()'s tensor == the off path's upload, on every chip; next_tail() ==
    the hashed tail, for a full chunk, a padded one, and chunks with fewer than 3 real ids."""
    path = str(hf_path(spec()))
    cfg = Config.from_dir(path)
    hasher = engram_hasher(path, cfg)
    ck = Checkpoint(path)
    s = 256 * tuple(mesh_device.shape)[0]
    g = torch.Generator().manual_seed(0)
    for layer in (1, 14):
        for fmt in ("bfp8", "bf16"):
            r = TtEngramRows(mesh_device, hasher, EngramTable(ck, layer), layer, rows_format=fmt)
            for valid in (s, s - 9, 2, 0):
                ids = torch.randint(0, cfg.vocab_size, (s,), generator=g)
                ids[valid:] = -1  # engine pad past actual_end
                tail = torch.randint(0, 1000, (hasher.max_ngram_size - 1,), generator=g)
                tail[0] = DEAD
                ref, ref_tail = r(ids, tail)
                want = r.next_tail(ids, tail)
                assert torch.equal(want, ref_tail.long()), (layer, fmt, valid, want, ref_tail)
                dev = r.allocate(s)
                out = {}
                th = threading.Thread(target=lambda: out.update(times=r.fill(dev, ids, tail, None, want)))
                th.start()
                th.join()
                assert "times" in out, "fill raised on the worker"
                assert dev.dtype == ref.dtype and tuple(dev.shape) == tuple(ref.shape), (dev, ref)
                assert _same_on_device(dev, ref), f"layer {layer} {fmt} valid {valid}: a chip's rows differ"
                print(f"ENGRAM_ASYNC rows layer {layer} {fmt} valid {valid}: equal; worker ms {out['times']}")
                ttnn.deallocate(dev)
                ttnn.deallocate(ref)


@pytest.mark.timeout(3600)
@mesh_parametrize
def test_engram_async_chunk(mesh_device):
    s = spec()
    rung = s.rung(RUNG)
    gold = Golden.for_rung(s, RUNG)
    layers = [i for i in s.layers() if i in gold.layers]
    chunk = rung["chunk"]
    start = rung["seq"] - chunk
    model = s.hooks().device_model(mesh_device, s, layers, lm_head=False)
    pool = model._engram_pool
    assert pool is not None and model.engram, "run with ENGRAM_ASYNC=on (default) and a stack with engram layers"
    state = model.new_state(rung["seq"])
    for i in layers:
        state.load_prefix(i, gold.state(i, at=start), start)
    tails = {i: t.clone() for i, t in state.tail.items()}
    tokens = gold.tokens()[start:]
    starts = {i for k, i in enumerate(layers) if k == 0 or layers[k - 1] != i - 1}
    gaps = {i: gold.layer(gold.n_chunks - 1, i)["in"].float() for i in starts if i != 0}
    want = gold.layer(gold.n_chunks - 1, layers[-1])["out"].float()

    def run(on: bool):
        model._engram_pool = pool if on else None
        state.tail = {i: t.clone() for i, t in tails.items()}  # the same chunk every run
        model.state = state
        if state.memo is not None:
            state.memo.clear()
        gap_ms = 0.0
        t0 = time.perf_counter()
        model.begin_chunk(tokens, start)
        t_bc = time.perf_counter()
        h = None
        for i in layers:
            if i in starts:
                if h is not None:
                    model.free(h)
                if i == 0:
                    h = model.embed(tokens, start)
                else:  # (golden input at a layer gap: timed apart, as the profile does, F119)
                    tg = time.perf_counter()
                    h = model.from_host(gaps[i], start)
                    gap_ms += (time.perf_counter() - tg) * 1e3
            h2 = model.layer(i, h, start, state)
            model.free(h)
            h = h2
        ttnn.synchronize_device(mesh_device)
        wall = (time.perf_counter() - t0) * 1e3 - gap_ms
        return h, wall, (t_bc - t0) * 1e3

    walls = {True: [], False: []}
    bcs = {True: [], False: []}
    times = []
    outs = {}
    for k in range(2 * (WARM + RUNS)):
        on = k % 2 == 0
        h, wall, bc = run(on)
        if k >= 2 * WARM:
            walls[on].append(wall)
            bcs[on].append(bc)
            if on:
                times.append({i: dict(t) for i, t in model.engram_times.items()})
        if k >= 2 * (WARM + RUNS) - 2:
            outs[on] = model.to_host(h).float()
        model.free(h)
    model._engram_pool = pool
    same = torch.equal(outs[True], outs[False])
    for on in (False, True):
        print(
            f"ENGRAM_ASYNC={'on' if on else 'off'}: chunk wall median {statistics.median(walls[on]):.1f} ms of "
            f"{[round(x, 1) for x in walls[on]]}; begin_chunk median {statistics.median(bcs[on]):.2f} ms; "
            f"last layer vs golden pcc {metrics.pcc(outs[on], want):.6f}"
        )
    for i in sorted(model.engram):
        keys = sorted({k for t in times for k in t[i]})
        med = {k: round(statistics.median(t[i].get(k, 0.0) for t in times), 2) for k in keys}
        work = sum(v for k, v in med.items() if k not in ("wait", "build_wait"))
        print(
            f"ENGRAM_ASYNC layer {i}: worker ms (median of {len(times)}) {med}; work {work:.2f}, waited {med['wait']}"
        )
    print(f"ENGRAM_ASYNC on vs off: last layer output bit-identical: {same}")
    assert same, "ENGRAM_ASYNC on and off differ"
