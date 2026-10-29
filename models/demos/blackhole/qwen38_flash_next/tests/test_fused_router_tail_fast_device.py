# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fast-tail sort-core router top-k (QWEN38_ROUTER_TAIL_FAST) versus the current default (the sort-core form) on the TP=8 mesh: bit-for-bit scores (bf16 bits)
and indices (all ten slots in order) on many random bf16-exact logit rows, tie-heavy ones included, eager and from a
replayed trace; and the traced per-call time of both (slope of a 12-call and a 4-call trace).

Run: QWEN38_FUSED_DEVICE_TEST=1 python -m pytest <this file> -s -p conftest -p no:cacheprovider -o addopts=
(the MESH_DEVICE / TT_VISIBLE_DEVICES environment of the model's TP=8 runs)."""

import contextlib
import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo import text_demo as demo
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail_sc

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="needs the TP=8 mesh")
ROWS_PER_KIND = int(os.environ.get("X42_ROWS", "100"))
REPLAYS = int(os.environ.get("X36_REPLAYS", "200"))
OUT = os.environ.get("X42_OUT")
VARIANTS = [v for v in os.environ.get("X42_VARIANTS", "row0,nostart,writer,incr").split(";")]


def _device0(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def _all_equal(tensor):
    replicas = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(tensor)]
    return all(torch.equal(replicas[0], other) for other in replicas[1:])


def _rows(kind: str, count: int, generator) -> list[torch.Tensor]:
    out = []
    for _ in range(count):
        if kind == "normal2":
            v = torch.randn(512, generator=generator) * 2.0
        elif kind == "normal4":
            v = torch.randn(512, generator=generator) * 4.0
        elif kind == "levels":  # few distinct values: ties everywhere, inside the top 10 too
            v = torch.randint(-3, 4, (512,), generator=generator).float() * 0.5
        elif kind == "top_ties":  # 12..60 columns share the maximum, the rest random
            v = torch.randn(512, generator=generator)
            n = int(torch.randint(12, 61, (1,), generator=generator))
            v[torch.randperm(512, generator=generator)[:n]] = 5.0
        elif kind == "few_ties":  # 2..9 columns share the maximum (ties inside the top 10 and at its edge)
            v = torch.randn(512, generator=generator)
            n = int(torch.randint(2, 10, (1,), generator=generator))
            v[torch.randperm(512, generator=generator)[:n]] = 4.0
        elif kind == "pairs":  # every value appears twice at random places
            base = torch.randn(256, generator=generator) * 2.0
            v = torch.cat([base, base])[torch.randperm(512, generator=generator)]
        elif kind == "constant":
            v = torch.full((512,), float(torch.randint(-4, 5, (1,), generator=generator)))
        elif kind == "neg_levels":  # ties among NEGATIVE values at the top (the raw-logit sort failed these)
            v = -1.0 - torch.randint(0, 6, (512,), generator=generator).float() * 0.5
        elif kind == "neg_top_ties":  # a negative-valued top: 3..40 shared maxima, a bf16 tail near -20 (more ties)
            v = torch.randn(512, generator=generator) - 20.0
            n = int(torch.randint(3, 41, (1,), generator=generator))
            v[torch.randperm(512, generator=generator)[:n]] = (
                -float(torch.randint(1, 8, (1,), generator=generator)) * 0.5
            )
        elif kind == "tiny":  # |logit| < 2^-9: outside the sort-key shift's exact range (not asserted, reported)
            v = torch.randn(512, generator=generator) * 2.0**-10
        else:
            raise AssertionError(kind)
        out.append(v.to(torch.bfloat16).float().reshape(1, 1, 1, 512))
    return out


KINDS = {
    "normal2": ROWS_PER_KIND,
    "normal4": ROWS_PER_KIND,
    "levels": ROWS_PER_KIND,
    "top_ties": ROWS_PER_KIND,
    "few_ties": ROWS_PER_KIND,
    "pairs": ROWS_PER_KIND,
    "constant": 12,
    "neg_levels": ROWS_PER_KIND,
    "neg_top_ties": ROWS_PER_KIND,
    "tiny": ROWS_PER_KIND,
}
INFO_KINDS = ("tiny",)  # reported, not asserted
BASE = "SORTCORE"  # the current default: the sort-core form, fast tail off


def _upload(mesh, values):
    return ttnn.from_torch(
        values,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _outputs(mesh):
    return (
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 1, 10]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        ),
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 1, 10]), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        ),
    )


def _traced_us(mesh, call):
    short_calls, long_calls = 4, 12

    def capture(n):
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(n):
            call()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        return trace

    def wall(trace):
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        best = 1e30
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(REPLAYS):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            best = min(best, time.perf_counter() - t0)
        return best / REPLAYS * 1e6

    short, long_ = capture(short_calls), capture(long_calls)
    try:
        ts, tl = wall(short), wall(long_)
    finally:
        ttnn.release_trace(mesh, short)
        ttnn.release_trace(mesh, long_)
    return (tl - ts) / (long_calls - short_calls), tl / long_calls


@contextlib.contextmanager
def _variant(opts):
    """Environment of one variant: opts == BASE -> fast tail off; else FAST=1 with the option list."""

    saved = {k: os.environ.get(k) for k in (router_tail_sc.FAST_ENV, router_tail_sc.FAST_OPTS_ENV)}
    try:
        if opts == BASE:
            os.environ.pop(router_tail_sc.FAST_ENV, None)
            os.environ.pop(router_tail_sc.FAST_OPTS_ENV, None)
        else:
            os.environ[router_tail_sc.FAST_ENV] = "1"
            os.environ[router_tail_sc.FAST_OPTS_ENV] = opts
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{**demo.DEVICE_PARAMS, "trace_region_size": 8 * 1024 * 1024}], indirect=True)
def test_fast_tail_is_bitwise_the_sort_core_form(mesh_device):
    mesh = mesh_device
    router_tail_sc.prepare(mesh)
    results = {}
    for opts in [BASE, *VARIANTS]:
        generator = torch.Generator().manual_seed(42)
        table = {}
        mismatches = []
        compare = opts != BASE and not ("direct" in opts and "inline" not in opts)
        for kind, count in KINDS.items():
            stats = {"rows": 0, "both_equal": 0}
            for n, values in enumerate(_rows(kind, count if compare else 0, generator)):
                logits = _upload(mesh, values)
                a, b = _outputs(mesh), _outputs(mesh)
                with _variant(BASE):
                    router_tail_sc.router_tail_sc_into(logits, *a)
                with _variant(opts):
                    router_tail_sc.router_tail_sc_into(logits, *b)
                eq = torch.equal(_device0(a[0]).view(torch.int16), _device0(b[0]).view(torch.int16)) and torch.equal(
                    _device0(a[1]), _device0(b[1])
                )
                stats["rows"] += 1
                stats["both_equal"] += int(eq)
                if not eq and len(mismatches) < 5:
                    mismatches.append({"kind": kind, "row": n})
                for t in (logits, *a, *b):
                    ttnn.deallocate(t)
            table[kind] = stats
        logits = _upload(mesh, _rows("normal2", 1, generator)[0])
        a, b = _outputs(mesh), _outputs(mesh)
        with _variant(BASE):
            router_tail_sc.router_tail_sc_into(logits, *a)
        with _variant(opts):
            router_tail_sc.router_tail_sc_into(logits, *b)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            router_tail_sc.router_tail_sc_into(logits, *b)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
        trace_equal = trace_total = 0
        try:
            if compare:
                for values in _rows("top_ties", 20, generator) + _rows("normal2", 20, generator):
                    host = ttnn.from_torch(
                        values,
                        dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                    )
                    ttnn.copy_host_to_device_tensor(host, logits)
                    with _variant(BASE):
                        router_tail_sc.router_tail_sc_into(logits, *a)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    trace_total += 1
                    trace_equal += int(
                        torch.equal(_device0(a[0]).view(torch.int16), _device0(b[0]).view(torch.int16))
                        and torch.equal(_device0(a[1]), _device0(b[1]))
                    )
        finally:
            ttnn.release_trace(mesh, trace)
        with _variant(opts):
            us, us12 = _traced_us(mesh, lambda: router_tail_sc.router_tail_sc_into(logits, *b))
        results[opts] = {
            "table": table,
            "trace_replay_equal": [trace_equal, trace_total],
            "us_slope": round(us, 2),
            "us_t12": round(us12, 2),
            "mismatches": mismatches,
            "load": os.getloadavg()[0],
        }
        print("X42VARIANT", opts, json.dumps({k: v for k, v in results[opts].items() if k != "table"}), flush=True)
        if compare:
            print("X42TABLE", opts, json.dumps(table), flush=True)
    if OUT:
        with open(OUT, "w") as f:
            json.dump(results, f, indent=1)
    for opts, r in results.items():
        if opts == BASE or ("direct" in opts and "inline" not in opts):
            continue
        for kind, stats in r["table"].items():
            if kind not in INFO_KINDS:
                assert stats["both_equal"] == stats["rows"], (opts, kind, stats, r["mismatches"])
        assert r["trace_replay_equal"][0] == r["trace_replay_equal"][1], opts
