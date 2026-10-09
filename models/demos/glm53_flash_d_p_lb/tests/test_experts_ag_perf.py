# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-stage time of the routed experts (split layout, the model's default) at the model's chunk, all-gather path
(GLM_EXPERTS_MODE=ag, tt/experts_ag.py) vs the whole unified call. Layer 4's real weights; x and the dense routing
are the component golden's (2048 rows) tiled to the chunk, so the routing has the real per-expert distribution.

Each ag stage runs between two device syncs, ITERS times after a warm-up; wall ms per stage (host dispatch included,
the sync adds ~0.05 ms per stage). The whole call is also timed without the inner syncs.
GLM_AGP_CHUNK (default spec target.chunk), GLM_AGP_ITERS (default 10)."""

import os
import time

import torch

from models.demos.common.bringup.testing.component import _step
from models.demos.common.bringup.testing.harness import component_golden, device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)
LAYER = 4
ITERS = int(os.environ.get("GLM_AGP_ITERS", "10"))


@mesh_parametrize
def test_experts_ag_perf(mesh_device):
    import ttnn
    from models.demos.glm53_flash_d_p.bringup import hooks
    from models.demos.glm53_flash_d_p.tt.experts import build_experts
    from models.demos.glm53_flash_d_p.tt.experts_ag import reduce_scatter_rows

    chunk = int(os.environ.get("GLM_AGP_CHUNK", S.get("target.chunk")))
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, "experts")
    gl = g.layer(c, LAYER)
    x, r = (gl[i].float() for i in st.inputs)
    x, r = x.reshape(-1, x.shape[-1]), r.reshape(x.shape[0] if x.dim() == 2 else -1, r.shape[-1])
    rep = -(-chunk // x.shape[0])
    x, r = x.repeat(rep, 1)[:chunk], r.repeat(rep, 1)[:chunk]
    H = x.shape[-1]
    rows, cols = tuple(mesh_device.shape)
    hooks.apply_device_settings(S)
    loader, cfg = hooks._loader_cfg(S)
    half = lambda t: ttnn.from_torch(  # noqa: E731  mesh row i holds rows [i S/2, (i+1) S/2) on all its chips
        t.reshape(1, 1, chunk, -1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(2, None)),
    )
    xd, rd = half(x), half(r)
    sync = lambda: ttnn.synchronize_device(mesh_device)  # noqa: E731
    results = {}

    def timed(fn, n=ITERS):
        fn()
        sync()
        t0 = time.time()
        for _ in range(n):
            fn()
        sync()
        return (time.time() - t0) / n * 1e3

    for mode in ("ag", "unified"):
        os.environ["GLM_EXPERTS_MODE"] = mode
        mod = build_experts(
            mesh_device, loader, cfg, LAYER, max(hooks._chunks(S)), weights_dtype=hooks.experts_dtype(S)
        )
        results[f"{mode} whole call"] = timed(lambda: ttnn.deallocate(mod(xd, dense=rd, split=True)))
        if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":  # per-op device time of one call (bring-up profiler)
            from models.demos.common.bringup.testing import profiler
            from models.demos.common.bringup.testing.profile import op_profile

            profiler.set_layer(LAYER)

            def one():
                profiler.signpost("experts")
                ttnn.deallocate(mod(xd, dense=rd, split=True))

            ops, _ = op_profile(mesh_device, one)
            tot = 0.0
            for sec, rows_ in ops.items():
                for row in rows_:
                    tot += row["ms"]
                    print(
                        f"[agp-dev] {mode:8s} {row['op']:42s} calls {row['calls']:2d}  {row['ms']:7.3f} ms  ({row['shape'][:70]})"
                    )
            print(f"[agp-dev] {mode:8s} device total {tot:.3f} ms", flush=True)
        if mode != "ag":
            continue
        s = chunk // 2
        blk = mod._block(s)
        stages = {}

        def stage(name, fn):
            sync()
            t0 = time.time()
            out = fn()
            sync()
            stages.setdefault(name, []).append((time.time() - t0) * 1e3)
            return out

        for it in range(ITERS + 1):
            idx, wts = stage("topk (dense -> idx, w)", lambda: mod.topk_from_dense(rd))
            stage("gather x tiles (fabric_all_gather axis 0)", lambda: blk_ag(xd, blk.gx))
            x_rm = stage("gathered x tile -> row major", lambda: ttnn.to_layout(blk.gx, ttnn.ROW_MAJOR_LAYOUT))
            stage("gather idx + w", lambda: (blk_ag(idx, blk.gidx_t), blk_ag(wts, blk.gw_t)))
            gi, gw = stage(
                "idx / w tile -> row major",
                lambda: (
                    ttnn.to_layout(blk.gidx_t, ttnn.ROW_MAJOR_LAYOUT),
                    ttnn.to_layout(blk.gw_t, ttnn.ROW_MAJOR_LAYOUT),
                ),
            )
            stage("route plan", lambda: blk.plan(gi, blk.lmap))
            y = stage(
                "flat expert",
                lambda: mod.flat(
                    ttnn.reshape(x_rm, (blk.T, H)),
                    blk.counts,
                    blk.regions,
                    token_index=blk.token_index,
                    y_row_major=True,
                    down_fp32=mod.down_fp32,
                    pack_stochastic_rounding=mod.pack_srnd,
                ),
            )
            stage(
                "local reduce phase 1",
                lambda: ttnn.bringup.moe_ag_local_reduce(
                    y, blk.y_slot, gw, blk.info, blk.S, phase=1, outputs=[blk.other]
                ),
            )
            stage("send-back gather (axis 0)", lambda: blk_ag(blk.other, blk.g_sp))
            stage(
                "local reduce phase 2",
                lambda: ttnn.bringup.moe_ag_local_reduce(
                    y, blk.y_slot, gw, blk.info, blk.S, phase=2, peer=blk.g_sp, tiled=True, outputs=[blk.own]
                ),
            )
            out = stage("reduce-scatter axis 1", lambda: reduce_scatter_rows(blk.own, 1, mod.links))
            if it == 0:
                stages.clear()
            for t in (idx, wts, x_rm, gi, gw, y, out):
                ttnn.deallocate(t)
        for name, ts in stages.items():
            results[f"ag  {name}"] = sum(ts) / len(ts)
        del mod

    print(f"[agp] chunk {chunk} (S/2 = {chunk // 2} per mesh row), layer {LAYER}, {ITERS} iters", flush=True)
    for k, ms in results.items():
        print(f"[agp] {k:45s} {ms:8.3f} ms", flush=True)
    print(f"[agp] ag stages sum {sum(v for k, v in results.items() if k.startswith('ag  ')):.3f} ms", flush=True)


def blk_ag(x, out):
    from models.demos.glm53_flash_d_p.tt.experts_ag import all_gather_rows

    return all_gather_rows(x, out, 0, int(os.environ.get("GLM_MOE_LINKS", "1")))
