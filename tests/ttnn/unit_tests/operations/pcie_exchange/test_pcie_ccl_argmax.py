# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PcieCCL.argmax_exchange: greedy token over a column-parallel lm_head on the PCIe P2P mesh.

    source env_vars_setup.sh
    python tests/ttnn/unit_tests/operations/pcie_exchange/test_pcie_ccl_argmax.py
"""
import os
import time

import torch
import ttnn

from models.tt_transformers.tt.pcie_ccl import PcieCCL

V_LOCAL = 64128  # Llama-3.1 padded vocab 128256 / 2 devices


def main():
    p2p = ttnn._ttnn.pcie_p2p
    chips = [int(c) for c in os.environ.get("TT_PCIE_P2P_CHIPS", "4,5").split(",")]
    p2p.use_isolated_mesh(chips)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), trace_region_size=32 * 1024 * 1024)
    try:
        ccl = PcieCCL(mesh)
        for it in range(5):
            logits = torch.randn(2, 1, 32, V_LOCAL, dtype=torch.bfloat16)
            # Make the winner unambiguous and spread across both halves.
            win = torch.randint(0, 2 * V_LOCAL, (32,))
            for r in range(32):
                d, c = divmod(int(win[r]), V_LOCAL)
                logits[d, 0, r, c] = 40.0
            t = ttnn.from_torch(
                logits,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )
            tok = ccl.argmax_exchange(t)
            got = ttnn.to_torch(tok, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))  # [2,1,32]
            exp = torch.cat([logits[0], logits[1]], dim=-1).float().argmax(-1).reshape(1, 32)
            for d in range(2):
                assert torch.equal(
                    got[d].reshape(1, 32).to(torch.int64), exp
                ), f"iter {it} dev {d}: {got[d].flatten()[:8]} vs {exp.flatten()[:8]}"
            ttnn.deallocate(t)
            ttnn.deallocate(tok)
        print("argmax_exchange correct over 5 iterations, host fallbacks:", ccl.host_fallbacks)

        # Decode residual all-reduce with the bfp8 payload: result must equal bfp8(a) + bfp8(b) on both devices.
        x = torch.randn(2, 1, 32, 4096, dtype=torch.bfloat16)
        sh = ccl.width_sharded(64, 64)
        t = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=sh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        xb = ccl._exchange_for(t)
        for rep in range(3):
            out = ttnn.experimental.pcie_exchange(t, xb, 0, bfp8_payload=True)
            got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
            q = ttnn.to_torch(
                ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
            ).float()  # bfp8-rounded inputs
            exp = q[0] + q[1]
            exact = x[0].float() + x[1].float()
            err_vs_bfp8 = max((got[d] - exp).abs().max().item() for d in range(2))
            err_vs_exact = max((got[d] - exact).abs().max().item() for d in range(2))
            assert torch.equal(got[0], got[1]), "bfp8 payload: replicas differ"
            assert err_vs_bfp8 <= 0.0625 * exp.abs().max().item() + 1e-3, f"bfp8 payload rep {rep}: err {err_vs_bfp8}"
            ttnn.deallocate(out)
        print(
            f"bfp8 payload all_reduce [32,4096] x 64 cores: ok, max err vs bfp8-rounded sum {err_vs_bfp8:.4f}, vs exact bf16 sum {err_vs_exact:.4f}"
        )
        # Timing: bf16 vs bfp8 payload, traced.
        for bf in (False, True):
            ttnn.deallocate(ttnn.experimental.pcie_exchange(t, xb, 0, bfp8_payload=bf))  # compile before capture
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            outs = [ttnn.experimental.pcie_exchange(t, xb, 0, bfp8_payload=bf) for _ in range(20)]
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(50):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            dt = (time.perf_counter() - t0) / (50 * 20)
            print(f"traced all_reduce [32,4096] x 64 cores, bfp8_payload={bf}: {dt * 1e6:.2f} us per op")
            ttnn.release_trace(mesh, tid)
            for o in outs:
                ttnn.deallocate(o)
        ttnn.deallocate(t)

        # Prefill-shaped DRAM collectives (what the model's prefill calls).
        for shape, dtype in (
            ((1, 1, 128, 4096), ttnn.bfloat16),
            ((1, 1, 128, 4096), ttnn.bfloat8_b),
            ((1, 2, 512, 4096), ttnn.bfloat16),
        ):
            x = torch.randn(2, *shape[1:], dtype=torch.bfloat16)
            t = ttnn.from_torch(
                x,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )
            t0 = time.perf_counter()
            out = ccl.all_reduce(t)
            ttnn.synchronize_device(mesh)
            dt = time.perf_counter() - t0
            assert tuple(out.shape) == shape and out.dtype == dtype and not out.is_sharded(), (
                out.shape,
                out.dtype,
                out.memory_config(),
            )
            got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
            ref = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()  # bfp8-rounded inputs
            exp = ref[0] + ref[1]
            tol = 0.0625 * exp.abs().max().item() * (4 if dtype == ttnn.bfloat8_b else 1)
            for d in range(2):
                err = (got[d] - exp).abs().max().item()
                assert err <= tol, f"dram all_reduce {shape} {dtype} dev {d}: max err {err} > {tol}"
            print(f"dram all_reduce {shape} {dtype}: ok, {dt * 1e3:.2f} ms eager (incl. host sync)")
            ttnn.deallocate(out)
            ttnn.deallocate(t)
        x = torch.randn(2, 128, 2048, dtype=torch.bfloat16)
        t = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        out = ccl.all_gather(t, -1)
        assert tuple(out.shape) == (1, 128, 4096), out.shape
        got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        exp = torch.cat([x[0], x[1]], dim=-1)
        for d in range(2):
            assert torch.equal(got[d], exp), f"dram all_gather dev {d}"
        print("dram all_gather [1,128,2048] -> [1,128,4096]: ok; host fallbacks so far:", ccl.host_fallbacks)
        ccl.release_prefill_buffers()

        # Timing under trace, as the model would run it (output buffer given), for all 32 rows and for 1 valid row.
        logits = torch.randn(2, 1, 32, V_LOCAL, dtype=torch.bfloat16)
        t = ttnn.from_torch(
            logits,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        exp = torch.cat([logits[0], logits[1]], dim=-1).float().argmax(-1).flatten()
        for valid_rows in (None, 1):
            out_tok = ttnn.from_torch(
                torch.zeros(1, 1, 32, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ccl.argmax_exchange(t, out_tok, valid_rows=valid_rows)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            ccl.argmax_exchange(t, out_tok, valid_rows=valid_rows)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            reps = 50
            t0 = time.perf_counter()
            for _ in range(reps):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            dt = (time.perf_counter() - t0) / reps
            got = ttnn.to_torch(out_tok, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            n = valid_rows or 32
            for d in range(2):
                assert torch.equal(got[d].flatten()[:n].to(torch.int64), exp[:n]), f"valid_rows={valid_rows} dev {d}"
                if valid_rows:
                    assert int(got[d].flatten()[n:].to(torch.int64).abs().sum()) == 0, "padded rows must be 0"
            print(
                f"traced argmax_exchange over [32, 2x{V_LOCAL}] valid_rows={valid_rows}: {dt * 1e6:.1f} us per token step"
            )
            ttnn.release_trace(mesh, tid)
    finally:
        ttnn._ttnn.pcie_p2p.teardown()
        ttnn.close_mesh_device(mesh)
    print("ALL OK")


if __name__ == "__main__":
    main()
