# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PcieCCL on the Qwen3.6-27B TP2 shapes: interleaved decode activations, 5120-wide prefill pieces, the
hidden-fractured embedding and the vocab-sharded logits gather.

    TT_VISIBLE_DEVICES=<bdf>,<bdf> TT_PCIE_P2P_CHIPS=0,1 python tests/ttnn/unit_tests/operations/pcie_exchange/test_pcie_ccl_shapes.py
"""
import os
import time

import torch
import ttnn

from models.tt_transformers.tt.pcie_ccl import PcieCCL


def main():
    p2p = ttnn._ttnn.pcie_p2p
    chips = [int(c) for c in os.environ.get("TT_PCIE_P2P_CHIPS", "4,5").split(",")]
    p2p.use_isolated_mesh(chips)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), trace_region_size=32 * 1024 * 1024)
    C = ttnn.ConcatMeshToTensor(mesh, dim=0)
    try:
        ccl = PcieCCL(mesh)
        # --- all_reduce on interleaved decode activations (L1 and DRAM), 5120 wide
        for mc, nm in ((ttnn.L1_MEMORY_CONFIG, "L1"), (ttnn.DRAM_MEMORY_CONFIG, "DRAM")):
            x = torch.randn(2, 1, 32, 5120, dtype=torch.bfloat16)
            t = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=mc,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )
            out = ccl.all_reduce(t)
            assert out.memory_config() == t.memory_config(), (out.memory_config(), t.memory_config())
            got = ttnn.to_torch(out, mesh_composer=C).float()
            exp = x[0].float() + x[1].float()
            err = max((got[d] - exp).abs().max().item() for d in range(2))
            assert (
                torch.equal(got[0], got[1]) and err <= 0.0625 * exp.abs().max().item() + 1e-3
            ), f"{nm} all_reduce err {err}"
            print(f"all_reduce [32,5120] {nm} interleaved: ok (max err {err:.4f}); fallbacks={ccl.host_fallbacks}")
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            outs = [ccl.all_reduce(t) for _ in range(20)]
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(50):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            print(f"  traced: {(time.perf_counter() - t0) / 1000 * 1e6:.2f} us per all_reduce incl. reshards")
            ttnn.release_trace(mesh, tid)
            for o in outs:
                ttnn.deallocate(o)
            ttnn.deallocate(out)
            ttnn.deallocate(t)
        # --- prefill all_reduce [1,1,128,5120] bf16 and bfp8 (32-core pieces)
        for dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
            x = torch.randn(2, 1, 128, 5120, dtype=torch.bfloat16)
            t = ttnn.from_torch(
                x,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )
            out = ccl.all_reduce(t)
            got = ttnn.to_torch(out, mesh_composer=C).float()
            ref = ttnn.to_torch(t, mesh_composer=C).float()
            exp = ref[0] + ref[1]
            err = max((got[d] - exp).abs().max().item() for d in range(2))
            tol = 0.0625 * exp.abs().max().item() * (4 if dtype == ttnn.bfloat8_b else 1)
            assert err <= tol, f"prefill all_reduce {dtype} err {err}"
            print(f"all_reduce [128,5120] {dtype} DRAM: ok (max err {err:.4f}); fallbacks={ccl.host_fallbacks}")
            ttnn.deallocate(out)
            ttnn.deallocate(t)
        # --- fp32 GDN out-proj partials in L1 interleaved (prefill) and 1-row logits
        x = torch.randn(2, 1, 128, 5120, dtype=torch.float32)
        t = ttnn.from_torch(
            x,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        out = ccl.all_reduce(t)
        got = ttnn.to_torch(out, mesh_composer=C).float()
        exp = x[0] + x[1]
        err = max((got[d] - exp).abs().max().item() for d in range(2))
        # The FPU adds fp32 operands at tf32 precision (same as the fabric reduce-scatter on these partials).
        assert out.dtype == ttnn.float32 and err <= 2e-3 * exp.abs().max().item() + 1e-5, f"fp32 all_reduce err {err}"
        print(f"all_reduce [128,5120] fp32 L1 interleaved: ok (max err {err:.2e}); fallbacks={ccl.host_fallbacks}")
        ttnn.deallocate(out)
        ttnn.deallocate(t)
        x = torch.randn(2, 1, 1, 124160, dtype=torch.bfloat16)
        t = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        out = ccl.all_gather(t, -1)
        got = ttnn.to_torch(out, mesh_composer=C)
        assert tuple(out.shape) == (1, 1, 1, 248320), out.shape
        for d in range(2):
            assert torch.equal(got[d], torch.cat([x[0], x[1]], dim=-1)), f"1-row logits gather dev {d}"
        print(f"all_gather logits [1,124160] -> [1,248320]: ok; fallbacks={ccl.host_fallbacks}")
        ttnn.deallocate(out)
        ttnn.deallocate(t)
        # --- embedding gathers: prefill [1,128,2560] and decode [1,1,2560] (hidden-fractured -> replicated)
        for shape in ((1, 128, 2560), (1, 1, 2560)):
            x = torch.randn(2, *shape[1:], dtype=torch.bfloat16)
            t = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )
            out = ccl.all_gather(t, -1)
            got = ttnn.to_torch(out, mesh_composer=C)
            exp = torch.cat([x[0], x[1]], dim=-1)
            assert tuple(out.shape) == (1, *shape[1:-1], 5120), out.shape
            for d in range(2):
                assert torch.equal(got[d], exp), f"embedding gather {shape} dev {d}"
            print(f"all_gather {shape} -> {tuple(out.shape)}: ok; fallbacks={ccl.host_fallbacks}")
            ttnn.deallocate(out)
            ttnn.deallocate(t)
        # --- vocab-sharded logits gather [1,1,32,124160] -> [1,1,32,248320] (wide row-major path)
        x = torch.randn(2, 1, 32, 124160, dtype=torch.bfloat16)
        t = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        t0 = time.perf_counter()
        out = ccl.all_gather(t, -1)
        ttnn.synchronize_device(mesh)
        dt = time.perf_counter() - t0
        got = ttnn.to_torch(out, mesh_composer=C)
        exp = torch.cat([x[0], x[1]], dim=-1)
        assert tuple(out.shape) == (1, 1, 32, 248320), out.shape
        for d in range(2):
            assert torch.equal(got[d], exp), f"logits gather dev {d}"
        print(
            f"all_gather logits [32,124160] -> [32,248320]: ok, {dt * 1e3:.1f} ms eager first call; fallbacks={ccl.host_fallbacks}"
        )
        assert ccl.host_fallbacks == 0, "a host fallback was taken"
    finally:
        ttnn._ttnn.pcie_p2p.teardown()
        ttnn.close_mesh_device(mesh)
    print("ALL OK")


if __name__ == "__main__":
    main()
