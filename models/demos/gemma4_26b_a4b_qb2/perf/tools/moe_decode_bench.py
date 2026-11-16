"""moe_decode batch-1 prototype: PCC vs torch per chip and expert-weight GB/s per chip (trace replay)."""
import os
import sys
import time

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4_26b_a4b_qb2.tt.generator_vllm import Gemma4ForCausalLM
from models.demos.gemma4_26b_a4b_qb2.tt.moe_decode import (
    MoeDecodeB1,
    MoeDecodeB1Fused,
    down_layout,
    gate_up_layout,
    upload_per_chip,
)

H, I_CHIP, E, K = 2816, 192, int(sys.argv[1]) if len(sys.argv) > 1 else 16, 8
DTYPE = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[sys.argv[2] if len(sys.argv) > 2 else "bfp8"]


def main():
    torch.manual_seed(0)
    ttnn.set_fabric_config(**Gemma4ForCausalLM.model_capabilities["fabric_config"])
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=16384, trace_region_size=50_000_000)
    try:
        gates = [torch.randn(E, H, I_CHIP) / H**0.5 for _ in range(4)]
        ups = [torch.randn(E, H, I_CHIP) / H**0.5 for _ in range(4)]
        downs = [torch.randn(E, I_CHIP, H) / I_CHIP**0.5 for _ in range(4)]
        w_gu = upload_per_chip(mesh, [gate_up_layout(g, u, mesh.dram_grid_size().x) for g, u in zip(gates, ups)], DTYPE)
        w_down = upload_per_chip(mesh, [down_layout(d, mesh.dram_grid_size().x) for d in downs], DTYPE)
        x = torch.randn(1, 1, 1, H)
        ids = torch.randperm(E)[:K]
        w = torch.rand(K) + 0.1
        w = w / w.sum()
        row = torch.zeros(1, 1, 32, 2 * K)
        row[0, 0, 0, :K], row[0, 0, 0, K:] = w, ids.float()
        tt_x = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        tt_route = ttnn.from_torch(
            row,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        op = (MoeDecodeB1Fused if os.environ.get("MOE_FUSED") == "1" else MoeDecodeB1)(
            mesh, hidden=H, inter_chip=I_CHIP, k=K, weight_dtype=DTYPE
        )
        out = op(tt_x, tt_route, w_gu, w_down)
        ttnn.synchronize_device(mesh)
        xb = x.bfloat16().float().reshape(-1)
        wb = row[0, 0, 0, :K].bfloat16().float()
        for chip, dev in enumerate(ttnn.get_device_tensors(out)):
            got = ttnn.to_torch(dev).float().reshape(-1)[:H]
            ref = torch.zeros(H)
            for s in range(K):
                e = int(ids[s])
                hcur = torch.nn.functional.gelu(xb @ gates[chip][e]) * (xb @ ups[chip][e]) * wb[s]
                ref += hcur @ downs[chip][e]
            ok, pcc = comp_pcc(ref, got, 0.99)
            print(
                f"MOE chip={chip} pcc={pcc:.5f} ok={ok} ref_norm={ref.norm():.3f} got_norm={got.norm():.3f}", flush=True
            )
        # timing: trace of gate_up + down
        INNER = 20  # calls per trace: a single-call trace mostly measures the trace launch
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(INNER):
            op(tt_x, tt_route, w_gu, w_down)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        for _ in range(5):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        us = (time.perf_counter() - t0) * 1e6 / reps / INNER
        tb = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}[DTYPE]
        nbytes = K * (2 * (H // 32) * (I_CHIP // 32) + (I_CHIP // 32) * (H // 32)) * tb
        print(f"MOE_TIME us={us:.1f} bytes_per_chip={nbytes/1e6:.2f}MB GBps={nbytes/us/1e3:.0f}", flush=True)
        ttnn.release_trace(mesh, trace)
        h = op.gate_up(tt_x, tt_route, w_gu)
        for name, fn, nb in (
            ("gate_up", lambda: op.gate_up(tt_x, tt_route, w_gu), K * 2 * (H // 32) * (I_CHIP // 32) * tb),
            ("down", lambda: op.down(h, tt_route, w_down), K * (I_CHIP // 32) * (H // 32) * tb),
        ):
            fn()
            ttnn.synchronize_device(mesh)
            tr = ttnn.begin_trace_capture(mesh, cq_id=0)
            for _ in range(INNER):
                fn()
            ttnn.end_trace_capture(mesh, tr, cq_id=0)
            for _ in range(5):
                ttnn.execute_trace(mesh, tr, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(reps):
                ttnn.execute_trace(mesh, tr, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            us = (time.perf_counter() - t0) * 1e6 / reps / INNER
            print(f"MOE_PART {name} us={us:.1f} MB={nb/1e6:.2f} GBps={nb/us/1e3:.0f}", flush=True)
            ttnn.release_trace(mesh, tr)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
