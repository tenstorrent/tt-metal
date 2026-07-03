# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Eager per-op timing of the DiffusionGemma terminal sampling chain at real shapes."""
from __future__ import annotations
import time
import torch
import ttnn
from models.experimental.diffusion_gemma.tt import sampling as TS
from models.experimental.diffusion_gemma.tt import denoise_loop as DL

VOCAB, CANVAS, TEMP, BUDGET = 262144, 256, 0.6, 0.1


def log(m):
    print(m, flush=True)


def timed(name, fn, mesh, reps=3):
    # warm
    o = fn()
    ttnn.synchronize_device(mesh)
    if hasattr(o, "deallocate"):
        o.deallocate(True)
    elif isinstance(o, (tuple, list)):
        for t in o:
            if hasattr(t, "deallocate"):
                t.deallocate(True)
    t0 = time.perf_counter()
    for _ in range(reps):
        o = fn()
        ttnn.synchronize_device(mesh)
        if hasattr(o, "deallocate"):
            o.deallocate(True)
        elif isinstance(o, (tuple, list)):
            for t in o:
                if hasattr(t, "deallocate"):
                    t.deallocate(True)
    dt = (time.perf_counter() - t0) * 1e3 / reps
    log(f"OP {name}: {dt:.3f} ms")
    return dt


def repl(mesh):
    return ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        torch.manual_seed(0)
        logits = ttnn.from_torch(
            (torch.randn(1, 1, CANVAS, VOCAB) * 4).float(),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=repl(mesh),
        )
        log(f"logits shape {list(logits.shape)} dtype {logits.get_dtype()}")

        timed("argmax_vocab", lambda: ttnn.argmax(logits, dim=-1, keepdim=True), mesh)
        timed("max_vocab", lambda: ttnn.max(logits, dim=-1, keepdim=True), mesh)
        timed("temp_scale(mul)", lambda: ttnn.multiply(logits, 1.0 / TEMP), mesh)
        timed("token_entropy_full", lambda: TS.token_entropy(logits, temperature=TEMP), mesh)

        # entropy-accept over the 256 axis
        ent = TS.token_entropy(logits, temperature=TEMP)  # [1,1,256,1]
        ent_flat = ttnn.reshape(ent, (ent.shape[0] * ent.shape[1], ent.shape[2]))
        log(f"entropy_flat shape {list(ent_flat.shape)}")
        timed("sort_256", lambda: ttnn.sort(ent_flat, dim=-1), mesh)
        sv, si = ttnn.sort(ent_flat, dim=-1)
        timed("cumsum_256", lambda: ttnn.cumsum(sv, dim=-1), mesh)
        timed("entropy_budget_accept_256", lambda: DL.entropy_budget_accept(ent_flat, BUDGET), mesh)
        sv.deallocate(True)
        si.deallocate(True)
        ent.deallocate(True)
        ent_flat.deallocate(True)
        log("DIAG_DONE")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
