# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Real MoE routing and flat_routed_expert utilization at the final chunk of the target prefill (all 45 layers).

Runs the whole prefill (routing depends on the context); during the last chunk every MoE layer logs its route-plan
token counts per global expert (tt/experts_ag.ROUTE_LOG) and the flat op is fenced by profiler signposts
(experts_ag.TIME_FLAT), so its device time per chip is exact. Per (layer, chip), from the counts and the chip's local
experts (an expert with no token is skipped by the op, weights included - flat_routed_expert se_dyn.hpp):
  active experts, routed rows (and rows padded to tiles per expert), weight bytes actually read
  (active x 3 x H x I x bfp4), activation bytes (rows x H x bf16 in and out), FLOPs (padded rows x 6 H I), time,
  DRAM utilization (bytes / time / 512 GB/s) and math utilization (FLOPs / time / LoFi peak).
Writes <repo>/generated/<model>/moe_routing.json and prints per-layer and summary tables.

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \\
    TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 BRINGUP_SPEC=<lb spec> scripts/run_safe_pytest.sh --run-all \\
    --no-precompile models/demos/glm53_flash_d_p_lb/tests/test_moe_routing.py -s
"""

import json
import math
import statistics

import torch

from models.demos.common.bringup.testing import profiler
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)
BFP4 = 576 / 1024
DRAM_GB_S = 512.0


@mesh_parametrize
def test_moe_routing(mesh_device):
    import os

    from models.demos.common.bringup.reference.prompt import tokens as prompt_tokens
    from models.demos.glm53_flash_d_p.bringup import hooks
    from models.demos.glm53_flash_d_p.tt import experts_ag

    for k, v in profiler.PROFILER_ENV.items():
        assert os.environ.get(k) == v, f"set {k}={v}"
    _, cfg = hooks._loader_cfg(S)
    H, I = cfg.hidden_size, cfg.moe_intermediate_size
    seq, chunk = int(S.get("target.seq")), int(S.get("target.chunk"))
    layers = S.layers()
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)
    toks = prompt_tokens(S, seq).to(torch.long)
    grid = mesh_device.compute_with_storage_grid_size()
    lofi = grid.x * grid.y * 4096 * 1.35e9

    def run_chunk(start):
        h = model.embed(toks[start : start + chunk])
        for i in layers:
            profiler.set_layer(i)
            h2 = model.layer(i, h, start, None)
            model.free(h)
            h = h2
        profiler.set_layer(None)
        model.free(h)

    last = seq - chunk
    for start in range(0, last, chunk):  # the context: every chunk before the last, unprofiled
        run_chunk(start)
    model.sync()
    experts_ag.ROUTE_LOG, experts_ag.TIME_FLAT = [], True
    profiler.enable(mesh_device)
    try:
        run_chunk(last)
        profiler.signpost("end")
        res = profiler.result()
    finally:
        profiler.disable()
        log = experts_ag.ROUTE_LOG
        experts_ag.ROUTE_LOG, experts_ag.TIME_FLAT = None, False

    recs = []
    wbytes = 3 * H * I * BFP4
    for e in log:
        L = e["layer"]
        times = res["kernel_ns_dev"].get(f"flat{L}", {})
        counts = e["counts"]
        for chip, gids in enumerate(e["gids"]):
            cnt = [int(counts[chip][g]) for g in gids]
            active = sum(1 for c in cnt if c > 0)
            rows = sum(cnt)
            padded = sum(math.ceil(c / 32) * 32 for c in cnt)
            nbytes = active * wbytes + rows * H * 2 * 2
            flops = padded * 6.0 * H * I
            t = times.get(chip, 0.0) / 1e9
            recs.append(
                {
                    "layer": L,
                    "chip": chip,
                    "experts": len(gids),
                    "active": active,
                    "rows": rows,
                    "padded_rows": padded,
                    "max_expert_rows": max(cnt) if cnt else 0,
                    "weight_mb": active * wbytes / 1e6,
                    "bytes_mb": nbytes / 1e6,
                    "gflop": flops / 1e9,
                    "ms": t * 1e3,
                    "dram_util": nbytes / t / (DRAM_GB_S * 1e9) if t else None,
                    "math_util": flops / t / lofi if t else None,
                    "counts_identical_across_chips": all(c == counts[0] for c in counts),
                }
            )
    out = S.repo / "generated" / S.model / "moe_routing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"H": H, "I": I, "chunk_start": last, "records": recs}, indent=1))

    print(
        f"[moe] final chunk [{last}, {seq}), {len(log)} MoE layers, H {H} I {I}, bfp4 expert {wbytes / 1e6:.1f} MB",
        flush=True,
    )
    print("[moe] per layer (over chips: min / median / max)", flush=True)
    print(
        f"[moe] {'L':>3} {'active experts':>18} {'rows':>20} {'flat ms':>20} {'slowest':>7} {'DRAM% slowest':>13} {'math% slowest':>13}"
    )
    by_layer = {}
    for r in recs:
        by_layer.setdefault(r["layer"], []).append(r)
    for L, rs in sorted(by_layer.items()):
        a = sorted(r["active"] for r in rs)
        n = sorted(r["rows"] for r in rs)
        t = sorted(r["ms"] for r in rs)
        w = max(rs, key=lambda r: r["ms"])
        print(
            f"[moe] {L:>3} {a[0]:>5} /{statistics.median(a):>5} /{a[-1]:>5} {n[0]:>6} /{statistics.median(n):>6} /{n[-1]:>6} "
            f"{t[0]:>6.2f} /{statistics.median(t):>6.2f} /{t[-1]:>6.2f} {w['chip']:>7} {100 * (w['dram_util'] or 0):>12.1f}% "
            f"{100 * (w['math_util'] or 0):>12.1f}%",
            flush=True,
        )
    tot = sum(max(r["ms"] for r in rs) for rs in by_layer.values())
    med = sum(statistics.median([r["ms"] for r in rs]) for rs in by_layer.values())
    slow = {}
    for rs in by_layer.values():
        c = max(rs, key=lambda r: r["ms"])["chip"]
        slow[c] = slow.get(c, 0) + 1
    xs = [r["rows"] for r in recs]
    ys = [r["ms"] for r in recs]
    zs = [r["active"] for r in recs]
    corr = lambda a, b: float(torch.corrcoef(torch.tensor([a, b], dtype=torch.float64))[0, 1])  # noqa: E731
    print(
        f"[moe] summary: flat op, slowest chip per layer summed {tot:.1f} ms, median chip summed {med:.1f} ms; slowest chip "
        f"counts {dict(sorted(slow.items()))}; time vs rows corr {corr(xs, ys):.3f}, vs active experts {corr(zs, ys):.3f}; "
        f"counts identical across chips: {all(r['counts_identical_across_chips'] for r in recs)}",
        flush=True,
    )
    agg = lambda key: sum(r[key] for r in recs)  # noqa: E731
    t_all = sum(r["ms"] for r in recs) / 1e3
    print(
        f"[moe] all (layer, chip): DRAM {agg('bytes_mb') / 1e3 / t_all / DRAM_GB_S * 100:.1f}% of peak, math "
        f"{agg('gflop') / t_all / (lofi / 1e9) * 100:.1f}% of LoFi peak (time-weighted); wrote {out}",
        flush=True,
    )
