# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""moe_decode (tt/moe_decode) batch-1 expert-weight read rate per chip on the QB2 1x4 mesh (WS4 proof).

  python models/demos/gemma4_26b_a4b_qb2/perf/tools/moe_decode_bench.py [gemma|gpt_oss] [bfp8|bfp4] [experts]

Random weights in the op's layouts, k distinct routed experts; times gate|up + down (20 calls captured per trace, so
the per-trace launch is amortized) and the two programs separately. Bytes are the routed experts' weight tiles read
per chip (padding included). Prints MOE_TIME / MOE_PART lines.
"""

import statistics
import sys
import time

import torch

import ttnn
from models.demos.gemma4_26b_a4b_qb2.tt.generator_vllm import Gemma4ForCausalLM
from models.demos.gemma4_26b_a4b_qb2.tt.moe_decode import (
    MoeDecodeB1,
    down_layout,
    gate_up_local_layout,
    upload_per_chip,
)

SHAPES = {
    "gemma": dict(H=2816, I=192, k=8, activation="geglu"),
    "gpt_oss": dict(H=2880, I=768, k=4, activation="swiglu"),
}
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
INNER, REPEATS = 20, 10


def timed(mesh, fn):
    fn()
    ttnn.synchronize_device(mesh)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(INNER):
        fn()
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        samples.append((time.perf_counter() - t0) * 1e6 / (REPEATS * INNER))
    ttnn.release_trace(mesh, trace)
    return statistics.median(samples)


def main():
    shape = sys.argv[1] if len(sys.argv) > 1 else "gemma"
    dtype = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[sys.argv[2] if len(sys.argv) > 2 else "bfp8"]
    E = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    cfg = SHAPES[shape]
    H, inter, k = cfg["H"], cfg["I"], cfg["k"]
    torch.manual_seed(0)
    ttnn.set_fabric_config(**Gemma4ForCausalLM.model_capabilities["fabric_config"])
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=16384, trace_region_size=50_000_000)
    try:
        banks = mesh.dram_grid_size().x
        g = [torch.randn(E, H, inter).bfloat16() for _ in range(4)]
        u = [torch.randn(E, H, inter).bfloat16() for _ in range(4)]
        d = [torch.randn(E, inter, H).bfloat16() for _ in range(4)]
        w_gu = upload_per_chip(mesh, [gate_up_local_layout(a, b, banks, 3) for a, b in zip(g, u)], dtype)
        w_dn = upload_per_chip(mesh, [down_layout(a, banks) for a in d], dtype)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        x = ttnn.from_torch(
            torch.randn(1, 1, 1, H),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        row = torch.zeros(1, 1, 32, 2 * k)
        row[0, 0, 0, :k], row[0, 0, 0, k:] = 1.0 / k, torch.randperm(E)[:k].float()
        route = ttnn.from_torch(row, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        op = MoeDecodeB1(mesh, hidden=H, inter_chip=inter, k=k, weight_dtype=dtype, activation=cfg["activation"])
        tb = TILE_BYTES[dtype]
        gu_bytes, dn_bytes = k * 2 * (H // 32) * (inter // 32) * tb, k * (inter // 32) * (H // 32) * tb
        us = timed(mesh, lambda: op(x, route, w_gu, w_dn))
        print(
            f"MOE_TIME {shape} {dtype.name} us={us:.1f} MB_per_chip={(gu_bytes + dn_bytes) / 1e6:.2f} "
            f"GBps={(gu_bytes + dn_bytes) / us / 1e3:.0f}",
            flush=True,
        )
        h = op.gate_up_local(x, route, w_gu, readers=op.local_readers)
        for name, fn, nb in (
            ("gate_up", lambda: op.gate_up_local(x, route, w_gu, readers=op.local_readers), gu_bytes),
            ("down", lambda: op.down(h, route, w_dn), dn_bytes),
        ):
            t = timed(mesh, fn)
            print(f"MOE_PART {name} us={t:.1f} MB={nb / 1e6:.2f} GBps={nb / t / 1e3:.0f}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
