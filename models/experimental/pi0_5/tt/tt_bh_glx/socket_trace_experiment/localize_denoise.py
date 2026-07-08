# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Localize the real-denoise socket-trace divergence (no p2p).

The socket+trace MECHANISM is proven correct (socket_relay_cycle_minitest.py:
K=6 denoise-shape model = PCC 1.0). The REAL denoise loop degrades on replay
(0.75->0.65->0.44). This script localizes WHERE, in two axes, from ONE pipeline
build:

  AXIS 1 (step): sweep num_denoising_steps n=1..5. For each n: eager denoise ->
    reference; capture per-submesh traces; replay 3x; PCC vs eager. Tells us the
    first n that breaks (N=1 known ~0.99).

  AXIS 2 (chip): at the smallest breaking n, after the eager run and after a
    replay (both from identical refreshed noise), snapshot EVERY persistent socket
    recv buffer (pipe.transport._recv_bufs) + x_t and diff per chip. The first chip
    (in snake order) whose recv buffer diverges localizes the op that breaks.

Run:
  source _bench_runs/pi05_production.env
  export PI05_CHECKPOINT_DIR=/path/to/pi05_libero_upstream
  tt-smi -glx_reset
  python_env/bin/python models/experimental/pi0_5/tt/tt_bh_glx/socket_trace_experiment/localize_denoise.py
"""

import os
import sys

import torch
import ttnn

from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.tt.tt_bh_glx.kv_migration import migrate_layer_paired
from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_galaxy_mesh
from models.experimental.pi0_5.tt.tt_bh_glx.pipeline import Pi0_5GLXPipeline, use_upstream_masks

CKPT = os.environ["PI05_CHECKPOINT_DIR"]
N_CAMS = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
SEED = 42
MAXN = int(os.environ.get("MAXN", "5"))


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    def log(m):
        print(f"[localize] {m}", flush=True)

    cfg = Pi0_5ModelConfig(action_horizon=action_horizon_from_checkpoint(CKPT), num_denoising_steps=MAXN)
    loader = Pi0_5WeightLoader(CKPT)
    img_h = img_w = cfg.siglip_config.image_size

    torch.manual_seed(SEED)
    images = [torch.randn(1, 3, img_h, img_w) for _ in range(N_CAMS)]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(N_CAMS)]
    lang_tokens = torch.randint(0, 256000, (1, 256), dtype=torch.int64)
    lang_masks = torch.ones(1, 256, dtype=torch.bool)

    with open_galaxy_mesh(l1_small_size=24576) as h:
        pipe = Pi0_5GLXPipeline(cfg, loader.categorized_weights, h)
        log("pipeline built (eager socket path)")

        pipe._ensure_persistent_input_buffers(images, lang_tokens)

        # CRITICAL: the real _refresh_noise_buffer() draws FRESH torch.randn noise on
        # every call (no seed) -> eager reference and trace replay would see DIFFERENT
        # initial noise, making "PCC vs eager" meaningless. Pin it to ONE fixed noise so
        # eager and replay compare apples-to-apples (the denoise is deterministic in x_t).
        if os.environ.get("FIXED_NOISE", "1").lower() in ("1", "true", "yes"):
            ah, ahp = pipe.action_horizon, pipe._action_horizon_padded
            _np = torch.zeros(1, ahp, pipe.action_dim, dtype=torch.float32)
            _np[:, :ah, :] = torch.randn(1, ah, pipe.action_dim)
            _fixed = ttnn.from_torch(_np, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

            def _fixed_refresh():
                ttnn.copy_host_to_device_tensor(_fixed, pipe.x_t_fp32)

            pipe._refresh_noise_buffer = _fixed_refresh
            log("FIXED_NOISE: _refresh_noise_buffer pinned to one noise sample")

        pipe._refresh_noise_buffer()
        upstream = None
        if use_upstream_masks():
            npatch = cfg.siglip_config.num_patches
            prefix_len = npatch * len(images) + int(lang_tokens.shape[-1])
            pipe._build_and_upload_upstream_artifacts(img_masks, lang_masks, prefix_len)
            upstream = pipe._upstream_per_chip

        # vision + prefill + KV migration EAGER once -> prefix_kv (read-only in denoise)
        vis = pipe.stage_vision.run(pipe.pixel_values_buf)
        vis_p0 = pipe.transport.send(vis, h.prefill_per_chip[0], tag="v2p")
        prefix_embs = pipe._build_prefix(vis_p0, pipe.lang_tokens_buf)
        _, per_layer_kv = pipe.stage_prefill.run(
            prefix_embs,
            attention_mask=None,
            position_ids=None,
            per_chip_attn_mask=(upstream["prefix_attn_mask"] if upstream else None),
            per_chip_cos=(upstream["prefix_cos"] if upstream else None),
            per_chip_sin=(upstream["prefix_sin"] if upstream else None),
        )
        prefix_kv = migrate_layer_paired(
            per_layer_kv, h.denoise_per_chip, transport=pipe.transport, to_l1=pipe._denoise_l1
        )
        ttnn.synchronize_device(h.denoise_per_chip[0])
        log("vision+prefill+KV done (eager, fixed)")

        subs = list(h.denoise_per_chip)

        def eager_x(n):
            pipe.num_denoising_steps = n
            pipe._refresh_noise_buffer()
            pipe._run_denoise_loop_device(prefix_kv, upstream)
            return ttnn.to_torch(pipe.x_t_fp32, mesh_composer=ttnn.ConcatMeshToTensor(subs[0], dim=0))[0].clone()

        def snapshot_recv():
            """torch snapshot of every cached socket recv buffer, keyed for readability."""
            out = {}
            for k, t in pipe.transport._recv_bufs.items():
                try:
                    out[k] = ttnn.to_torch(t).float().clone()
                except Exception as e:
                    out[k] = f"ERR {e}"
            return out

        # ---------- AXIS 1: step sweep ----------
        log("==== AXIS 1: num_denoising_steps sweep ====")
        first_break = None
        for n in range(1, MAXN + 1):
            pipe.num_denoising_steps = n
            ref = eager_x(n)
            pipe.num_denoising_steps = n
            pipe._refresh_noise_buffer()
            tids = [ttnn.begin_trace_capture(sm, cq_id=0) for sm in subs]
            pipe._run_denoise_loop_device(prefix_kv, upstream)
            for sm, tid in zip(subs, tids):
                ttnn.end_trace_capture(sm, tid, cq_id=0)
            pccs = []
            for rnd in range(3):
                pipe._refresh_noise_buffer()
                for sm, tid in zip(subs, tids):
                    ttnn.execute_trace(sm, tid, cq_id=0, blocking=False)
                for sm in subs:
                    ttnn.synchronize_device(sm)
                tr = ttnn.to_torch(pipe.x_t_fp32, mesh_composer=ttnn.ConcatMeshToTensor(subs[0], dim=0))[0]
                pccs.append(_pcc(tr, ref))
            for sm, tid in zip(subs, tids):
                ttnn.release_trace(sm, tid)
            log(f"  N={n}: replay PCC = {[round(p, 4) for p in pccs]}")
            if first_break is None and min(pccs) < 0.99:
                first_break = n

        # ---------- AXIS 2: per-chip recv-buffer diff at the breaking N ----------
        nb = first_break if first_break is not None else MAXN
        log(f"==== AXIS 2: per-chip recv-buffer diff at N={nb} ====")
        pipe.num_denoising_steps = nb
        pipe._refresh_noise_buffer()
        pipe._run_denoise_loop_device(prefix_kv, upstream)
        ttnn.synchronize_device(subs[0])
        eager_recv = snapshot_recv()
        eager_xt = ttnn.to_torch(pipe.x_t_fp32, mesh_composer=ttnn.ConcatMeshToTensor(subs[0], dim=0))[0].clone()

        pipe._refresh_noise_buffer()
        tids = [ttnn.begin_trace_capture(sm, cq_id=0) for sm in subs]
        pipe._run_denoise_loop_device(prefix_kv, upstream)
        for sm, tid in zip(subs, tids):
            ttnn.end_trace_capture(sm, tid, cq_id=0)
        pipe._refresh_noise_buffer()
        for sm, tid in zip(subs, tids):
            ttnn.execute_trace(sm, tid, cq_id=0, blocking=False)
        for sm in subs:
            ttnn.synchronize_device(sm)
        traced_recv = snapshot_recv()
        traced_xt = ttnn.to_torch(pipe.x_t_fp32, mesh_composer=ttnn.ConcatMeshToTensor(subs[0], dim=0))[0]
        for sm, tid in zip(subs, tids):
            ttnn.release_trace(sm, tid)

        log(f"  x_t   eager-vs-traced PCC = {_pcc(traced_xt, eager_xt):.6f}")
        for k in eager_recv:
            ev, tv = eager_recv[k], traced_recv.get(k)
            if isinstance(ev, str) or isinstance(tv, str) or tv is None:
                log(f"  recv {k}: ERR/missing")
                continue
            log(
                f"  recv {k}: PCC={_pcc(tv, ev):.6f}  eager|mean|={ev.abs().mean():.4f} traced|mean|={tv.abs().mean():.4f}"
            )
        log("DONE")


if __name__ == "__main__":
    main()
    sys.exit(0)
