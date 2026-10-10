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

Knobs: GLM_LP_LAYERS (comma list), GLM_LP_CHUNK, GLM_LP_START, GLM_LP_ITERS (default 3), GLM_LP_TOP (ops listed per
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
        ttnn.deallocate(blk(x, start))  # compile
        ttnn.synchronize_device(mesh_device)
        compile_s = time.time() - t0
        walls = []
        for _ in range(iters):
            t0 = time.perf_counter()
            ttnn.deallocate(blk(x, start))
            ttnn.synchronize_device(mesh_device)
            walls.append((time.perf_counter() - t0) * 1e3)
        row = {"layer": i, "kind": kind, "build_s": round(build_s, 1), "compile_s": round(compile_s, 1)}
        row["wall_ms"] = round(statistics.median(walls), 3)
        if rt:
            profiler.enable(mesh_device, ops=True, calls=True, rt=True)
            try:
                profiler.set_layer(i)
                ttnn.deallocate(blk(x, start))
                profiler.set_layer(None)
                calls = profiler.finish_rt()
            finally:
                profiler.disable()
            chip_ns, step_ns, op_ns = defaultdict(float), defaultdict(lambda: defaultdict(float)), defaultdict(float)
            for c in calls:
                step = c["key"].split(".", 1)[-1]
                for chip, ns in c["ns_dev"].items():
                    chip_ns[chip] += ns
                    step_ns[step][chip] += ns
                op_ns[c["op"]] += max(c["ns_dev"].values())
            busiest = max(chip_ns, key=chip_ns.get)
            row["device_ms"] = round(chip_ns[busiest] / 1e6, 3)
            row["device_ms_min_chip"] = round(min(chip_ns.values()) / 1e6, 3)
            row["steps_ms"] = {k: round(max(v.values()) / 1e6, 3) for k, v in step_ns.items()}
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
