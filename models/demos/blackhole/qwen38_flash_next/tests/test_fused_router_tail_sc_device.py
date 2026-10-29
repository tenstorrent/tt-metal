# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sort-core router top-k (router_tail_sc) versus today's router_tail on the TP=8 mesh: bit-for-bit scores (bf16 bits)
and indices (all ten slots in order) on many random bf16-exact logit rows, tie-heavy ones included, eager and from a
replayed trace; and the traced per-call time of both (slope of a 12-call and a 4-call trace).

Run: QWEN38_FUSED_DEVICE_TEST=1 python -m pytest <this file> -s -p conftest -p no:cacheprovider -o addopts=
(the MESH_DEVICE / TT_VISIBLE_DEVICES environment of the model's TP=8 runs)."""

import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo import text_demo as demo
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail, router_tail_sc

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="needs the TP=8 mesh")
ROWS_PER_KIND = int(os.environ.get("X36_ROWS", "100"))
REPLAYS = int(os.environ.get("X36_REPLAYS", "200"))
OUT = os.environ.get("X36_OUT")


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


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{**demo.DEVICE_PARAMS, "trace_region_size": 8 * 1024 * 1024}], indirect=True)
def test_sort_core_is_bitwise_todays_router_tail(mesh_device):
    mesh = mesh_device
    generator = torch.Generator().manual_seed(36)
    router_tail_sc.prepare(mesh)
    table = {}
    mismatches = []
    for kind, count in KINDS.items():
        stats = {"rows": 0, "scores_equal": 0, "indices_equal": 0, "both_equal": 0, "replicas_equal": 0, "replicas": 0}
        for n, values in enumerate(_rows(kind, count, generator)):
            logits = _upload(mesh, values)
            today = _outputs(mesh)
            sort_core = _outputs(mesh)
            router_tail.router_tail_into(logits, *today)
            router_tail_sc.router_tail_sc_into(logits, *sort_core)
            ts, ti = (_device0(t) for t in today)
            ss, si = (_device0(t) for t in sort_core)
            s_eq = torch.equal(ts.view(torch.int16), ss.view(torch.int16))
            i_eq = torch.equal(ti, si)
            stats["rows"] += 1
            stats["scores_equal"] += int(s_eq)
            stats["indices_equal"] += int(i_eq)
            stats["both_equal"] += int(s_eq and i_eq)
            if not (s_eq and i_eq) and len(mismatches) < 10:
                mismatches.append(
                    {"kind": kind, "row": n, "today": [ts.tolist(), ti.tolist()], "sc": [ss.tolist(), si.tolist()]}
                )
            if n % 10 == 0:  # every device of the mesh
                stats["replicas"] += 1
                stats["replicas_equal"] += int(all(_all_equal(t) for t in sort_core))
            for t in (logits, *today, *sort_core):
                ttnn.deallocate(t)
        table[kind] = stats
        print("X36ROW", kind, json.dumps(stats), flush=True)

    # a replayed trace of the sort-core program: fresh inputs through the same buffers
    logits = _upload(mesh, _rows("normal2", 1, generator)[0])
    today = _outputs(mesh)
    sort_core = _outputs(mesh)
    router_tail.router_tail_into(logits, *today)
    router_tail_sc.router_tail_sc_into(logits, *sort_core)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    router_tail_sc.router_tail_sc_into(logits, *sort_core)
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    trace_equal = 0
    trace_total = 0
    try:
        for values in _rows("top_ties", 20, generator) + _rows("normal2", 20, generator):
            host = ttnn.from_torch(
                values, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
            )
            ttnn.copy_host_to_device_tensor(host, logits)
            router_tail.router_tail_into(logits, *today)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            trace_total += 1
            trace_equal += int(
                torch.equal(_device0(today[0]).view(torch.int16), _device0(sort_core[0]).view(torch.int16))
                and torch.equal(_device0(today[1]), _device0(sort_core[1]))
            )
    finally:
        ttnn.release_trace(mesh, trace)

    us_today, us_today_12 = _traced_us(mesh, lambda: router_tail.router_tail_into(logits, *today))
    us_sc, us_sc_12 = _traced_us(mesh, lambda: router_tail_sc.router_tail_sc_into(logits, *sort_core))
    result = {
        "table": table,
        "trace_replay_equal": [trace_equal, trace_total],
        "us_slope_today": round(us_today, 2),
        "us_slope_sort_core": round(us_sc, 2),
        "us_t12_today": round(us_today_12, 2),
        "us_t12_sort_core": round(us_sc_12, 2),
        "load": os.getloadavg(),
        "mismatches": mismatches,
    }
    print("X36RESULT", json.dumps(result), flush=True)
    if OUT:
        with open(OUT, "w") as f:
            json.dump(result, f, indent=1)
    for kind, stats in table.items():
        if kind in INFO_KINDS:
            continue
        assert stats["both_equal"] == stats["rows"], (kind, stats, mismatches[:2])
        assert stats["replicas_equal"] == stats["replicas"], (kind, stats)
    assert trace_equal == trace_total
