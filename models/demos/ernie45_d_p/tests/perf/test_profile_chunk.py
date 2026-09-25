# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P3.3: device-time breakdown of one 5120-token chunk at 51200 -> 56320 (golden 50k KV prefix loaded).

Tracy-free: tt/common.signpost() syncs + flushes the device profiler at every section boundary and charges
the programs that ran to the section (max over the 4 devices). Two measured runs:
  1. unsynced chunk -> real end-to-end chunk wall time
  2. section-profiled chunk -> per-section DEVICE KERNEL DURATION (the syncs inflate only this run's wall time)
Run with TT_METAL_DEVICE_PROFILER=1 (see the P3.3 gate command).
"""

import json
import re
import time
from collections import defaultdict

import ttnn
from models.demos.ernie45_d_p.bringup import metrics
from models.demos.ernie45_d_p.tests.conftest import mesh_1x4
from models.demos.ernie45_d_p.tt.common import REPO, Golden, enable_section_profiling, section_profile, signpost
from models.demos.ernie45_d_p.tt.model import TtErnieModel

TASK = "P3.3"


def category(section: str) -> str:
    if section in ("attn.all_reduce", "moe.all_reduce"):
        return "ccl: all_reduce (attn + moe)"
    if section == "attn.sdpa":
        return "attention: sdpa (chunked, 5k q x 56k kv)"
    if section.startswith("attn."):
        return "attention: qkv / rope / kv write / o_proj"
    if section == "moe.experts":
        return "moe: routed experts (fused FFN)"
    if section in ("moe.router", "moe.dispatch", "moe.combine_reduce"):
        return "moe: router + dispatch + combine/reduce"
    if section == "moe.shared":
        return "moe: shared expert"
    if section == "dense_mlp":
        return "dense MLP (layer 0)"
    return "other: norms, residual adds, embed"


@mesh_1x4
def test_profile_last_chunk(mesh_device, cfg, loader, record):
    G = Golden(56320, 5120)
    start = G.seq - G.chunk
    model = TtErnieModel(mesh_device, loader, cfg, lm_head=False)
    cache = model.new_cache(G.seq)
    for i in range(cfg.num_hidden_layers):
        k, v = G.kv(i)
        cache.load_prefix(i, k[:, :start], v[:, :start])
    tokens = G.tokens()[start:]

    def run():
        h = model.prefill_chunk(tokens, start, cache)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(h)

    run()  # warm-up: compile + program cache
    t0 = time.time()
    run()
    wall = time.time() - t0

    enable_section_profiling(mesh_device)
    run()
    signpost("end")  # charge the final section
    prof = section_profile()

    cats = defaultdict(lambda: [0.0, 0])
    for sec, ns in prof["kernel_ns"].items():
        c = category(sec)
        cats[c][0] += ns
        cats[c][1] += prof["programs"][sec]
    total = sum(v[0] for v in cats.values())
    assert total > 0, "device profiler returned no program durations (check TT_METAL_PROFILER_* env)"
    print(
        f"\nchunk 51200->56320: end-to-end wall {wall * 1e3:.0f} ms; device kernel time (critical device) {total / 1e6:.0f} ms"
    )
    print(f"{'category':44s} {'device ms':>10s} {'share':>7s} {'programs':>9s}")
    out = {}
    for c, (ns, n) in sorted(cats.items(), key=lambda x: -x[1][0]):
        print(f"{c:44s} {ns / 1e6:10.1f} {100 * ns / total:6.1f}% {n:9d}")
        slug = re.sub(r"[^a-z0-9]+", "_", c.split(" (")[0].lower()).strip("_")
        metrics.record(record.task, f"device_ms_{slug}", round(ns / 1e6, 2))
        out[c] = round(ns / 1e6, 2)
    print("\nper chip (device ms, sum of sections):")
    per_chip = {}
    for d in prof["kernel_ns_dev"].values():
        for c, ns in d.items():
            per_chip[c] = per_chip.get(c, 0.0) + ns
    for c in sorted(per_chip):
        print(f"  chip {c}: {per_chip[c] / 1e6:.1f} ms")
        metrics.record(record.task, f"device_ms_chip{c}", round(per_chip[c] / 1e6, 2))
    host_ms = wall * 1e3 - total / 1e6
    print(f"{'host/dispatch overhead (wall - device)':44s} {host_ms:10.1f}")
    metrics.record(record.task, "chunk_wall_ms", round(wall * 1e3, 1))
    metrics.record(record.task, "device_ms_total", round(total / 1e6, 2))
    metrics.record(record.task, "host_overhead_ms", round(host_ms, 1))
    metrics.record(record.task, "profiled_programs", sum(v[1] for v in cats.values()))
    d = REPO / "models/demos/ernie45_d_p/bringup/results"  # committed with the gate: feeds the dashboard
    d.mkdir(parents=True, exist_ok=True)
    (d / "P3.3_profile.json").write_text(
        json.dumps(
            {
                "wall_ms": wall * 1e3,
                "device_ms": out,
                "sections_ms": {k: v / 1e6 for k, v in prof["kernel_ns"].items()},
                "sections_ms_per_chip": {
                    k: {str(c): v / 1e6 for c, v in d.items()} for k, d in prof["kernel_ns_dev"].items()
                },
                "programs": prof["programs"],
            },
            indent=1,
        )
    )
