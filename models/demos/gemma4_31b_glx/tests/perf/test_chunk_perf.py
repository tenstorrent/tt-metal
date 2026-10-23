# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Warm time of one 5120-token chunk of the whole 60-layer model at given starts (manual perf pass, not a frozen test).

Timing only: the KV prefix is the zeroed cache and the tokens are random ids (the work does not depend on the values).
Per start: one run to compile, then BENCH_REPEATS timed runs (embed -> 60 layers -> final norm, one sync), the median
is reported as the wall time. With TT_METAL_DEVICE_PROFILER=1 a further run syncs after every layer and sums the
device kernel durations per chip (device ms = the slowest chip), and BENCH_OP_LAYERS (e.g. "0,5") breaks those layers
down per ttnn op with the bring-up framework's profiler.

    BENCH_STARTS=0,51200 BENCH_TRACE=1 pytest models/demos/gemma4_31b_glx/tests/perf/test_chunk_perf.py -s

BENCH_TRACE=1 captures the chunk forward as a ttnn trace (Model.capture_chunk) and times its replay.
Results go to stdout as "BENCH ..." lines and, with BENCH_OUT, to a JSON file.
"""

import json
import os
import statistics
import time
from types import SimpleNamespace

import pytest
import torch

import ttnn

HF = "/mnt/models/huggingface/hub/models--google--gemma-4-31B-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"
CHUNK, MAX_SEQ = 5120, int(os.environ.get("BENCH_MAX_SEQ", "56320"))
TRACE_REGION = int(os.environ.get("BENCH_TRACE_REGION", 200 * 1024 * 1024))


def _starts():
    return [int(x) for x in os.environ.get("BENCH_STARTS", "0,51200").split(",") if x.strip()]


def _layers():
    n = int(os.environ.get("BENCH_LAYERS", "60"))
    return list(range(n))


def _device_ms(mesh) -> float:
    """Sum of device kernel durations per chip since the last read; the slowest chip (ms)."""
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    data = ttnn.get_latest_programs_perf_data() or {}
    per = {}
    for dev, programs in data.items():
        for prog in programs:
            e = (getattr(prog, "program_analyses_results", None) or {}).get("DEVICE KERNEL DURATION [ns]")
            if e is not None:
                per[dev] = per.get(dev, 0.0) + float(e.duration)
    return per


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": TRACE_REGION}],
    indirect=True,
    ids=["ring"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["8x4"])
def test_chunk_perf(mesh_device):
    """BENCH_SWEEP="name:K=V,K=V;name2:..." runs each settings variant (GEMMA4_31B_GLX_<K>=V) in turn on a fresh model
    in this process (short stacks: BENCH_LAYERS=6 is 5 sliding layers and 1 global one)."""
    import gc

    sweep = os.environ.get("BENCH_SWEEP")
    if not sweep:
        return _bench(mesh_device, "default")
    for item in sweep.split(";"):
        name, _, kv = item.partition(":")
        env = dict(x.split("=", 1) for x in kv.split(",") if x.strip())
        old = {k: os.environ.get(f"GEMMA4_31B_GLX_{k}") for k in env}
        os.environ.update({f"GEMMA4_31B_GLX_{k}": v for k, v in env.items()})
        try:
            _bench(mesh_device, name.strip())
        except Exception as e:  # noqa: BLE001  (one failing variant does not stop the sweep)
            print(f"BENCH variant {name} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(f"GEMMA4_31B_GLX_{k}", None)
                else:
                    os.environ[f"GEMMA4_31B_GLX_{k}"] = v
            gc.collect()
            ttnn.synchronize_device(mesh_device)


def _bench(mesh_device, variant):
    from models.demos.gemma4_31b_glx.tt.model import Model
    from models.demos.gemma4_31b_glx.tt.settings import settings

    mesh = mesh_device
    layers = _layers()
    full = len(layers) == 60
    spec = {"paths.hf": HF, "target.chunk": CHUNK, "target.seq": MAX_SEQ}
    t0 = time.time()
    model = Model(mesh, SimpleNamespace(get=spec.__getitem__), layers, lm_head=False)
    print(f"BENCH load {time.time() - t0:.1f}s layers {len(layers)} settings {settings.all()}", flush=True)
    profiling = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    traced = os.environ.get("BENCH_TRACE") == "1"
    repeats = int(os.environ.get("BENCH_REPEATS", "3"))
    op_layers = bool(os.environ.get("BENCH_OP_LAYERS"))  # any value: the per-section breakdown
    out = {"layers": len(layers), "traced": traced, "starts": {}}
    gen = torch.Generator().manual_seed(0)

    for start in _starts():
        model.new_state(MAX_SEQ)
        tokens = torch.randint(0, model.vocab, (CHUNK,), generator=gen)

        def rows(t):
            composer = ttnn.create_mesh_composer(
                mesh, ttnn.MeshComposerConfig(dims=[0, 1], mesh_shape_override=ttnn.MeshShape(8, 4))
            )
            return ttnn.to_torch(t, mesh_composer=composer).float()

        def eager(sync_each=None, keep=None):
            h = model.embed(tokens, start=start)
            for i in layers:
                h2 = model.layer(i, h, start, None)
                model.free(h)
                h = h2
                if sync_each is not None:
                    sync_each(i)
            if full:
                h2 = model.final_norm(h)
                model.free(h)
                h = h2
            model.sync()
            if keep is not None:
                keep.append(rows(h))
            model.free(h)

        row = {}
        if traced:
            ref = []
            eager(keep=ref)
            t1 = time.time()
            model.capture_chunk(tokens, start)  # compiles, captures; replay_chunk restages the inputs
            row["capture_s"] = round(time.time() - t1, 1)
            got = rows(model.replay_chunk(tokens, start))
            row["trace_vs_eager_maxabs"] = float((got - ref[0]).abs().max())

            def run():
                model.replay_chunk(tokens, start)
                model.sync()

        else:
            run = eager
            if profiling:  # drain the profiler after every layer, or its DRAM buffers overflow (and reads stall)

                def run():
                    eager(lambda i: _device_ms(mesh))

        run()  # compile / warm
        ms = []
        for _ in range(repeats):
            t1 = time.perf_counter()
            run()
            ms.append((time.perf_counter() - t1) * 1e3)
        row["wall_ms"] = round(statistics.median(ms), 1)
        row["wall_ms_all"] = [round(x, 1) for x in ms]
        if profiling:
            _device_ms(mesh)  # drop what ran so far
            if traced:
                run()
                per = _device_ms(mesh)
            else:
                per = {}

                def drain(i):
                    for d, ns in _device_ms(mesh).items():
                        per[d] = per.get(d, 0.0) + ns

                eager(drain)
                for d, ns in _device_ms(mesh).items():
                    per[d] = per.get(d, 0.0) + ns
            row["device_ms"] = round(max(per.values()) / 1e6, 1) if per else None
            row["device_ms_min_chip"] = round(min(per.values()) / 1e6, 1) if per else None
            if op_layers and not traced:
                row["ops"] = _op_breakdown(mesh, model, tokens, start, op_layers)
        print(f"BENCH start {start} [{variant}]: {json.dumps(row)}", flush=True)
        out["starts"][start] = row
        if traced:
            model.release_chunk_traces()

    model.release_traces()
    del model
    if os.environ.get("BENCH_OUT"):
        with open(os.environ["BENCH_OUT"] + ("" if variant == "default" else f".{variant}"), "w") as f:
            json.dump(out, f, indent=1)


def _op_breakdown(mesh, model, tokens, start, op_layers):
    """Per-section device ms (slowest chip, summed over the given layers) with the framework's section profiler: one
    section per block step (run_block's signposts) and any finer signpost the modules mark."""
    from bringup_fw.testing import profiler

    h = model.embed(tokens, start=start)
    model.sync()
    profiler.enable(mesh)
    for i in model.layers:
        profiler.set_layer(i)
        h2 = model.layer(i, h, start, None)
        model.free(h)
        h = h2
    profiler.signpost("end")
    res = profiler.result()
    profiler.disable()
    model.sync()
    model.free(h)
    rows = {k: round(v / 1e6, 3) for k, v in sorted(res["kernel_ns"].items(), key=lambda x: -x[1])}
    for k, v in rows.items():
        print(f"BENCH sec {start} {k:40s} {v:9.3f} ms ({res['programs'].get(k, 0)} programs)", flush=True)
    return rows
