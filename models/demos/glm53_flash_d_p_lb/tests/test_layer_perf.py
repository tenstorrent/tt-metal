# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Per-layer device perf of GLM-5.3-Flash, one layer of each block type, fake weights by default (no checkpoint).

Builds TtGlmBlock for each layer in GLM_LP_LAYERS (default: the spec's block_types representatives, 0 kda_dense,
3 dsa_moe, 4 kda_moe) from reference/fake_weights.py:FakeLoader (random weights in the checkpoint's shapes; the routed
experts uninitialised in the flat layout, so there is no expert packing or upload), feeds each one the same random
residual chunk at position GLM_LP_START (default target.seq - target.chunk: the longest DSA context) and reports:
  wall    median eager wall time per call with a sync after it (host dispatch included), GLM_LP_ITERS calls
  device  with the program real-time profiler (on by default where the host has an IOMMU): per chip, the sum of the
          layer's program durations; reported as the busiest chip, plus per step (run_block section) and the top ops
          (no syncs inside the call; programs that wait on a CCL count their wait)
Builds take seconds per layer (KDA / MLA / dense weights are random host tensors converted to the device).

Numerics are meaningless (random weights). Routing is near-uniform; GLM_FAKE_HOT=n makes experts 0 .. n-1 hot.
GLM_LP_REAL=1 loads the real checkpoint instead (spec paths.hf / BRINGUP_HF; the flat expert cache is used).

Knobs: GLM_LP_TRACE=1 (+ BRINGUP_TRACE_REGION_BYTES) times a traced replay of the block, GLM_LP_SAVE=<prefix> (save each block's output: A/B accuracy), GLM_LP_CALLS=<prefix> (dump every call per layer), GLM_LP_STEP_OPS (steps whose ops are listed, default experts), GLM_LP_LAYERS (comma list), GLM_LP_CHUNK, GLM_LP_START, GLM_LP_ITERS (default 3), GLM_LP_TOP (ops listed per
layer, default 12), GLM_LP_JSON (write the rows there). The spec's device settings (experts dtype / fidelity, links)
apply as in the model. Mesh and fabric come from the spec (BRINGUP_SPEC).

  TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 PYTHONPATH=$PWD BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --no-precompile models/demos/glm53_flash_d_p_lb/tests/test_layer_perf.py -s
"""

import json
import os
import statistics
import time
from collections import defaultdict

import torch

import ttnn
from models.demos.common.bringup.testing import profiler
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)


def _layers():
    if os.environ.get("GLM_LP_LAYERS"):
        return [int(v) for v in os.environ["GLM_LP_LAYERS"].split(",")]
    return sorted(int(b["representative"]) for b in S.get("block_types").values())


@mesh_parametrize
def test_layer_perf(mesh_device):
    from models.demos.glm53_flash_d_p.tt.common import replicate, residual_layout, split_from_host
    from models.demos.glm53_flash_d_p.tt.model import TtGlmBlock

    hooks = S.hooks()
    hooks.apply_device_settings(S)
    real = os.environ.get("GLM_LP_REAL") == "1"
    if real:
        loader, cfg = hooks._loader_cfg(S)
    else:
        from models.demos.glm53_flash_d_p.reference.fake_weights import FakeLoader

        loader = FakeLoader()
        cfg = loader.cfg
    seq = int(S.get("target.seq"))
    chunk = int(os.environ.get("GLM_LP_CHUNK", S.get("target.chunk")))
    start = int(os.environ.get("GLM_LP_START", seq - chunk))
    assert start % chunk == 0, f"start {start} must be a multiple of the chunk {chunk}"
    iters = int(os.environ.get("GLM_LP_ITERS", "3"))
    layout = residual_layout()
    layers = _layers()
    mesh_shape = tuple(mesh_device.shape)
    print(
        f"[lp] mesh {mesh_shape} chunk {chunk} at {start} ({'real' if real else 'fake'} weights, residual {layout}) "
        f"layers {layers}",
        flush=True,
    )

    torch.manual_seed(0)
    x_host = torch.randn(1, 1, chunk, cfg.hc_mult * cfg.hidden_size) * 0.5
    x = split_from_host(mesh_device, x_host) if layout == "split" else replicate(mesh_device, x_host)
    rt = ttnn.device.IsProgramRealtimeProfilerActive()
    rows = []
    for i in layers:
        kind = "kda" if cfg.is_kda(i) else "dsa"
        kind += "_moe" if cfg.is_moe(i) else "_dense"
        t0 = time.time()
        blk = TtGlmBlock(
            mesh_device, cfg, loader, i, start + chunk, [chunk], layout=layout, experts_dtype=hooks.experts_dtype(S)
        )
        build_s = time.time() - t0
        t0 = time.time()
        y = blk(x, start)  # compile
        ttnn.synchronize_device(mesh_device)
        if os.environ.get("GLM_LP_SAVE"):  # the block output (per-chip rows in row-major chip order) for A/B checks
            torch.save(
                torch.cat([ttnn.to_torch(t).reshape(-1, t.shape[-1]) for t in ttnn.get_device_tensors(y)]),
                f"{os.environ['GLM_LP_SAVE']}.L{i}.pt",
            )
        ttnn.deallocate(y)
        compile_s = time.time() - t0
        walls = []
        for _ in range(iters):
            t0 = time.perf_counter()
            ttnn.deallocate(blk(x, start))
            ttnn.synchronize_device(mesh_device)
            walls.append((time.perf_counter() - t0) * 1e3)
        row = {"layer": i, "kind": kind, "build_s": round(build_s, 1), "compile_s": round(compile_s, 1)}
        row["wall_ms"] = round(statistics.median(walls), 3)
        if os.environ.get("GLM_LP_TRACE") == "1":  # the block as one trace: replay wall per call (no host dispatch)
            tr = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            yt = blk(x, start)
            ttnn.end_trace_capture(mesh_device, tr, cq_id=0)
            ttnn.execute_trace(mesh_device, tr, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(iters):
                ttnn.execute_trace(mesh_device, tr, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            row["trace_ms"] = round((time.perf_counter() - t0) / iters * 1e3, 3)
            ttnn.release_trace(mesh_device, tr)
            ttnn.deallocate(yt)
            print(f"[lp] L{i} traced replay {row['trace_ms']:.2f} ms per call (eager wall {row['wall_ms']:.2f})", flush=True)
        if rt:
            profiler.enable(mesh_device, ops=True, calls=True, rt=True)
            try:
                profiler.set_layer(i)
                ttnn.deallocate(blk(x, start))
                profiler.set_layer(None)
                calls = profiler.finish_rt()
            finally:
                profiler.disable()
            if os.environ.get("GLM_LP_CALLS"):  # every call in order: step, op, per-chip ms, tensor shapes
                with open(f"{os.environ['GLM_LP_CALLS']}.L{i}.json", "w") as f:
                    json.dump(calls, f, default=str)
            chip_ns, step_ns, op_ns = defaultdict(float), defaultdict(lambda: defaultdict(float)), defaultdict(float)
            step_op = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))  # step -> op -> [ms on the busiest chip, calls]
            for c in calls:
                step = c["key"].split(".", 1)[-1]
                for chip, ns in c["ns_dev"].items():
                    chip_ns[chip] += ns
                    step_ns[step][chip] += ns
                op_ns[c["op"]] += max(c["ns_dev"].values())
                step_op[step][c["op"]][0] += max(c["ns_dev"].values()) / 1e6
                step_op[step][c["op"]][1] += 1
            busiest = max(chip_ns, key=chip_ns.get)
            row["device_ms"] = round(chip_ns[busiest] / 1e6, 3)
            row["device_ms_min_chip"] = round(min(chip_ns.values()) / 1e6, 3)
            row["steps_ms"] = {k: round(max(v.values()) / 1e6, 3) for k, v in step_ns.items()}
            row["step_ops_ms"] = {
                st: {op: [round(v[0], 3), v[1]] for op, v in sorted(ops.items(), key=lambda kv: -kv[1][0])}
                for st, ops in step_op.items()
            }
            row["top_ops_ms"] = {
                k: round(v / 1e6, 3)
                for k, v in sorted(op_ns.items(), key=lambda kv: -kv[1])[: int(os.environ.get("GLM_LP_TOP", "12"))]
            }
        rows.append(row)
        dev = f"device {row['device_ms']:.2f} ms (min chip {row['device_ms_min_chip']:.2f})" if rt else "device -"
        print(
            f"[lp] L{i} {kind}: wall {row['wall_ms']:.2f} ms, {dev}; build {build_s:.1f} s, compile {compile_s:.1f} s",
            flush=True,
        )
        if rt:
            for k, v in row["steps_ms"].items():
                print(f"[lp]     step {k:<16} {v:8.3f} ms", flush=True)
            for st in os.environ.get("GLM_LP_STEP_OPS", "experts").split(","):
                for op, (ms, n) in row["step_ops_ms"].get(st, {}).items():
                    print(f"[lp]     {st:<8} {op:<38} {ms:8.3f} ms ({n} calls)", flush=True)
            for k, v in row["top_ops_ms"].items():
                print(f"[lp]     op   {k:<40} {v:8.3f} ms", flush=True)
        del blk
    ttnn.deallocate(x)
    print("[lp] | layer | type | wall ms | device ms (busiest chip) |")
    for r in rows:
        print(f"[lp] | {r['layer']} | {r['kind']} | {r['wall_ms']:.2f} | {r.get('device_ms', float('nan')):.2f} |")
    from models.demos.common.bringup.core.spec import parse_layers

    total = 0.0
    for b, info in S.get("block_types").items():
        n = len(parse_layers(info["layers"], S.num_layers))
        r = next((r for r in rows if r["kind"] == b), None)
        if r is not None:
            total += n * r.get("device_ms", r["wall_ms"])
    if total:
        print(f"[lp] whole model estimate (count x representative, device time where measured): {total:.1f} ms / chunk")
    if os.environ.get("GLM_LP_JSON"):
        with open(os.environ["GLM_LP_JSON"], "w") as f:
            json.dump({"mesh": list(mesh_shape), "chunk": chunk, "start": start, "rows": rows}, f, indent=1)


@mesh_parametrize
def test_chunk_perf(mesh_device):
    """One chunk through a chain of layers (GLM_LP_CHAIN, default all of the spec's), fake weights by default: eager
    wall with one sync per chunk (host dispatch overlapping the device, as the model runs) vs a traced replay of the
    whole chain (needs BRINGUP_TRACE_REGION_BYTES). Checks the traced output equals the eager one."""
    from models.demos.common.bringup.core.spec import parse_layers
    from models.demos.glm53_flash_d_p.tt.common import replicate, residual_layout, split_from_host
    from models.demos.glm53_flash_d_p.tt.model import TtGlmBlock

    hooks = S.hooks()
    hooks.apply_device_settings(S)
    if os.environ.get("GLM_LP_REAL") == "1":
        loader, cfg = hooks._loader_cfg(S)
    else:
        from models.demos.glm53_flash_d_p.reference.fake_weights import FakeLoader

        loader = FakeLoader()
        cfg = loader.cfg
    layers = parse_layers(os.environ.get("GLM_LP_CHAIN", "all"), S.num_layers)
    seq = int(S.get("target.seq"))
    chunk = int(os.environ.get("GLM_LP_CHUNK", S.get("target.chunk")))
    start = int(os.environ.get("GLM_LP_START", seq - chunk))
    iters = int(os.environ.get("GLM_LP_ITERS", "3"))
    layout = residual_layout()
    t0 = time.time()
    blocks = [
        TtGlmBlock(
            mesh_device, cfg, loader, i, start + chunk, [chunk], layout=layout, experts_dtype=hooks.experts_dtype(S)
        )
        for i in layers
    ]
    print(f"[cp] {len(blocks)} layers built in {time.time() - t0:.0f} s", flush=True)
    torch.manual_seed(0)
    x_host = torch.randn(1, 1, chunk, cfg.hc_mult * cfg.hidden_size) * 0.5
    x = split_from_host(mesh_device, x_host) if layout == "split" else replicate(mesh_device, x_host)

    def run():
        h = x
        for b in blocks:
            h2 = b(h, start)
            if h is not x:
                ttnn.deallocate(h)
            h = h2
        return h

    host = lambda t: torch.cat([ttnn.to_torch(p).reshape(-1, p.shape[-1]) for p in ttnn.get_device_tensors(t)])  # noqa
    ttnn.deallocate(run())  # compile
    ttnn.synchronize_device(mesh_device)
    walls = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = run()
        ttnn.synchronize_device(mesh_device)
        walls.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(y)
    y_eager = run()
    ttnn.synchronize_device(mesh_device)
    ref = host(y_eager)
    ttnn.deallocate(y_eager)
    eager = statistics.median(walls)
    print(f"[cp] eager chunk wall {eager:.1f} ms ({len(blocks)} layers, chunk {chunk} at {start})", flush=True)
    if ttnn.device.IsProgramRealtimeProfilerActive():  # device time per layer and step in the chain
        profiler.enable(mesh_device, ops=True, calls=True, rt=True)
        try:
            h = x
            for b in blocks:
                profiler.set_layer(b.i)
                h2 = b(h, start)
                if h is not x:
                    ttnn.deallocate(h)
                h = h2
            profiler.set_layer(None)
            ttnn.deallocate(h)
            calls = profiler.finish_rt()
        finally:
            profiler.disable()
        per = defaultdict(lambda: defaultdict(float))  # (layer, step) -> chip -> ns
        for c in calls:
            for chip, ns in c["ns_dev"].items():
                per[(c["layer"], c["key"].split(".", 1)[-1])][chip] += ns
        lay = defaultdict(lambda: defaultdict(float))
        for (li, st), v in per.items():
            for chip, ns in v.items():
                lay[li][chip] += ns
        tot = sum(max(v.values()) for v in lay.values()) / 1e6
        print(f"[cp] device sum of per-layer busiest chips {tot:.1f} ms", flush=True)
        for li in sorted(lay):
            steps = {st: max(v.values()) / 1e6 for (l2, st), v in per.items() if l2 == li}
            top = sorted(steps.items(), key=lambda kv: -kv[1])[:4]
            print(
                f"[cp]   L{li}: {max(lay[li].values()) / 1e6:6.2f} ms  "
                + "  ".join(f"{k} {v:.2f}" for k, v in top),
                flush=True,
            )
    if os.environ.get("BRINGUP_TRACE_REGION_BYTES"):
        tr = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        yt = run()
        ttnn.end_trace_capture(mesh_device, tr, cq_id=0)
        ttnn.execute_trace(mesh_device, tr, cq_id=0, blocking=True)
        same = torch.equal(host(yt), ref)
        t0 = time.perf_counter()
        for _ in range(iters):
            ttnn.execute_trace(mesh_device, tr, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        traced = (time.perf_counter() - t0) / iters * 1e3
        ttnn.release_trace(mesh_device, tr)
        ttnn.deallocate(yt)
        print(
            f"[cp] traced chunk replay {traced:.1f} ms (eager {eager:.1f}, -{eager - traced:.1f} ms); "
            f"traced output {'identical to' if same else 'DIFFERS from'} eager",
            flush=True,
        )
