# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify the fixed-step device denoise loop is trace-safe with device canvas feedback.

Uses a synthetic canvas-dependent logits_fn (embed canvas -> project to a small vocab)
so that step N+1 truly consumes step N's accepted canvas. Runs the fixed-step loop
eager, then captures + replays it as a Metal trace, and asserts the committed argmax
is identical — proving the accepted canvas feeds the next step entirely on device with
no host readback of the cutoff (RUN-first argmax path, deterministic).
"""
from __future__ import annotations

import torch
import ttnn
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.tt import denoise_loop as DL

VOCAB, CANVAS, HIDDEN, STEPS = 256, 64, 128, 4


def log(m):
    print(m, flush=True)


def repl(mesh):
    return ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    try:
        torch.manual_seed(0)
        emb_t = torch.randn(VOCAB, HIDDEN) * 0.1
        proj_t = torch.randn(HIDDEN, VOCAB) * 0.1
        emb = ttnn.from_torch(emb_t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=repl(mesh))
        proj = ttnn.from_torch(
            proj_t.unsqueeze(0).unsqueeze(0),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=repl(mesh),
        )

        def logits_fn(canvas, step):
            # canvas: [1,1,C,1] uint32 TILE. embed -> [1,1,C,H] -> project -> [1,1,C,V]
            ids = ttnn.reshape(canvas, (1, CANVAS))
            ids = ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT)
            h = ttnn.embedding(ids, emb, layout=ttnn.TILE_LAYOUT)  # [1,C,H]
            ids.deallocate(True)
            h = ttnn.reshape(h, (1, 1, CANVAS, HIDDEN))
            lg = ttnn.matmul(h, proj)  # [1,1,C,V]
            h.deallocate(True)
            return lg

        cfg = DiffusionConfig(canvas_length=CANVAS, max_denoise_steps=STEPS)

        # fixed per-step renoise tokens (device-resident, deterministic)
        noise_list = []
        for s in range(STEPS):
            nt = torch.randint(0, VOCAB, (1, 1, CANVAS, 1), generator=torch.Generator().manual_seed(100 + s)).int()
            noise_list.append(
                ttnn.from_torch(nt, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32, mesh_mapper=repl(mesh))
            )

        def noise_tokens_fn(step):
            return ttnn.clone(noise_list[step])

        def make_init():
            init_t = torch.randint(0, VOCAB, (1, 1, CANVAS, 1), generator=torch.Generator().manual_seed(7)).int()
            return ttnn.from_torch(
                init_t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32, mesh_mapper=repl(mesh)
            )

        # preallocate trace-safe accept/renoise constants OUTSIDE the trace
        consts = DL.make_denoise_constants(mesh, batch=1, canvas_len=CANVAS, budget=cfg.entropy_budget)

        # eager
        committed_eager = DL.run_fixed_denoise_steps(
            logits_fn, make_init(), cfg, gumbel_noise_fn=None, noise_tokens_fn=noise_tokens_fn, constants=consts
        )
        ttnn.synchronize_device(mesh)
        eager_ids = ttnn.to_torch(ttnn.get_device_tensors(committed_eager)[0]).squeeze().long()
        committed_eager.deallocate(True)
        log(f"eager committed ids[:8]={eager_ids[:8].tolist()}")

        # traced
        # warm program cache first
        warm = DL.run_fixed_denoise_steps(
            logits_fn, make_init(), cfg, gumbel_noise_fn=None, noise_tokens_fn=noise_tokens_fn, constants=consts
        )
        ttnn.synchronize_device(mesh)
        warm.deallocate(True)

        init_dev = make_init()
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        committed_traced = DL.run_fixed_denoise_steps(
            logits_fn, init_dev, cfg, gumbel_noise_fn=None, noise_tokens_fn=noise_tokens_fn, constants=consts
        )
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        ttnn.execute_trace(mesh, tid, blocking=False)
        ttnn.synchronize_device(mesh)
        traced_ids = ttnn.to_torch(ttnn.get_device_tensors(committed_traced)[0]).squeeze().long()
        ttnn.release_trace(mesh, tid)
        log(f"traced committed ids[:8]={traced_ids[:8].tolist()}")

        match = (eager_ids == traced_ids).float().mean().item()
        log(f"TRACE_SAFE_MATCH {match*100:.2f}%")
        log("TRACE_SAFE_OK" if match > 0.999 else "TRACE_SAFE_FAIL")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
