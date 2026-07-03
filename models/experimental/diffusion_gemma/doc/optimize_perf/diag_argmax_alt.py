# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare argmax-over-vocab alternatives at real DiffusionGemma logits shape."""
from __future__ import annotations
import time
import torch
import ttnn

VOCAB, CANVAS = 262144, 256


def log(m):
    print(m, flush=True)


def repl(mesh):
    return ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None


def timed(name, fn, mesh, reps=3):
    o = fn()
    ttnn.synchronize_device(mesh)
    outs = o if isinstance(o, (tuple, list)) else [o]
    for t in outs:
        if hasattr(t, "deallocate"):
            t.deallocate(True)
    t0 = time.perf_counter()
    for _ in range(reps):
        o = fn()
        ttnn.synchronize_device(mesh)
        outs = o if isinstance(o, (tuple, list)) else [o]
        for t in outs:
            if hasattr(t, "deallocate"):
                t.deallocate(True)
    dt = (time.perf_counter() - t0) * 1e3 / reps
    log(f"OP {name}: {dt:.3f} ms")
    return dt


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        torch.manual_seed(0)
        lt = (torch.randn(1, 1, CANVAS, VOCAB) * 4).float()
        logits = ttnn.from_torch(lt, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=repl(mesh))
        ref = lt.argmax(dim=-1)  # torch reference indices

        # Baseline: TILE argmax (single-core)
        timed("argmax_TILE(baseline)", lambda: ttnn.argmax(logits, dim=-1, keepdim=True), mesh)

        # Candidate A: convert to ROW_MAJOR, then argmax (multi-core)
        def rm_argmax():
            rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
            am = ttnn.argmax(rm, dim=-1, keepdim=True)
            rm.deallocate(True)
            return am

        timed("argmax_via_ROWMAJOR", rm_argmax, mesh)

        # Candidate B: topk k=1 (TILE) — vocab is 2^18
        try:
            timed("topk_k1_TILE", lambda: ttnn.topk(logits, k=1, dim=-1), mesh)
        except Exception as e:
            log(f"topk_k1_TILE FAILED: {type(e).__name__}: {str(e)[:200]}")
        # Candidate C: topk k=32 (TILE)
        try:
            timed("topk_k32_TILE", lambda: ttnn.topk(logits, k=32, dim=-1), mesh)
        except Exception as e:
            log(f"topk_k32_TILE FAILED: {type(e).__name__}: {str(e)[:200]}")

        # correctness: ROW_MAJOR argmax vs torch
        rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        am = ttnn.argmax(rm, dim=-1, keepdim=True)
        am_t = ttnn.to_torch(ttnn.get_device_tensors(am)[0]).squeeze().to(torch.long)
        match = (am_t == ref.squeeze()).float().mean().item()
        log(f"ROWMAJOR argmax match vs torch: {match*100:.2f}%")
        rm.deallocate(True)
        am.deallocate(True)

        # topk k=1 correctness if it worked
        try:
            tv, ti = ttnn.topk(logits, k=1, dim=-1)
            ti_t = ttnn.to_torch(ttnn.get_device_tensors(ti)[0]).squeeze().to(torch.long)
            m2 = (ti_t == ref.squeeze()).float().mean().item()
            log(f"topk_k1 index match vs torch: {m2*100:.2f}%")
            tv.deallocate(True)
            ti.deallocate(True)
        except Exception as e:
            log(f"topk_k1 correctness skipped: {type(e).__name__}")
        log("DIAG_DONE")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
