# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED experiment — does NOT modify the production pipeline.

Take the real eager socket pipeline (Pi0_5GLXPipeline: vision 4 + prefill 18 +
denoise 6 on per-chip 1x1 submeshes, FABRIC_2D sockets) and try to TRACE it by
capturing a trace on EACH per-chip submesh (not the parent). Reuses the pipeline's
own pure-device body (_sample_actions_device) unchanged; only wraps it with
begin/end_trace_capture per submesh.

Run:
  source _bench_runs/pi05_production.env
  export PI05_CHECKPOINT_DIR=/path/to/pi05_libero_upstream
  tt-smi -glx_reset
  python_env/bin/python models/experimental/pi0_5/tt/tt_bh_glx/socket_trace_experiment/run_socket_traced.py
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
N_STEPS = int(os.environ.get("PI05_NUM_DENOISE_STEPS", "5"))
SCOPE = os.environ.get("TRACE_SCOPE", "full")  # full | denoise
SEED = 42


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    def log(m):
        print(f"[sock-traced] {m}", flush=True)

    cfg = Pi0_5ModelConfig(action_horizon=action_horizon_from_checkpoint(CKPT), num_denoising_steps=N_STEPS)
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

        # --- replicate capture_trace setup steps 1-3 (persistent buffers + upstream) ---
        pipe._ensure_persistent_input_buffers(images, lang_tokens)
        pipe._refresh_noise_buffer()
        upstream = None
        if use_upstream_masks():
            npatch = cfg.siglip_config.num_patches
            prefix_len = npatch * len(images) + int(lang_tokens.shape[-1])
            pipe._build_and_upload_upstream_artifacts(img_masks, lang_masks, prefix_len)
            upstream = pipe._upstream_per_chip
        log(f"buffers staged; upstream_masks={upstream is not None}; scope={SCOPE}")

        # FIX UNDER TEST: the denoise loop reuses the SAME cached socket recv buffer
        # every Euler step -> under trace replay the N steps race on it. Give each
        # send call a UNIQUE tag so it gets its own recv buffer (no reuse race). The
        # trace then captures distinct buffers per step. Keeps sockets (no p2p).
        if os.environ.get("USE_UNIQUE_TAGS", "").lower() in ("1", "true", "yes"):
            _orig_send = pipe.transport.send
            _ctr = [0]

            def _tagged_send(src, dst, *, out_buf=None, tag=None):
                _ctr[0] += 1
                return _orig_send(src, dst, out_buf=out_buf, tag=f"{tag}:{_ctr[0]}")

            pipe.transport.send = _tagged_send
            log("transport.send monkey-patched to unique-per-call tags (distinct recv buffers)")

        if SCOPE == "denoise":
            # Localize: run vision+prefill+KV EAGER (populate cached KV), then trace
            # ONLY the denoise loop on denoise_per_chip (persistent x_t + cached KV +
            # transient per-step intermediates + socket velocity-wrap).
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
            pipe._refresh_noise_buffer()
            eager = ttnn.to_torch(pipe.x_t_fp32, mesh_composer=ttnn.ConcatMeshToTensor(h.denoise_per_chip[0], dim=0))
            pipe._run_denoise_loop_device(prefix_kv, upstream)
            eager = ttnn.to_torch(pipe.x_t_fp32, mesh_composer=ttnn.ConcatMeshToTensor(h.denoise_per_chip[0], dim=0))[0]
            log(f"eager denoise done: actions {tuple(eager.shape)}")
            subs = list(h.denoise_per_chip)
            pipe._refresh_noise_buffer()
            tids = [ttnn.begin_trace_capture(sm, cq_id=0) for sm in subs]
            pipe._run_denoise_loop_device(prefix_kv, upstream)
            for sm, tid in zip(subs, tids):
                ttnn.end_trace_capture(sm, tid, cq_id=0)
            log("captured denoise-only per-submesh traces")
            for rnd in (1, 2, 3):
                pipe._refresh_noise_buffer()
                for sm, tid in zip(subs, tids):
                    ttnn.execute_trace(sm, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(h.denoise_per_chip[0])
                traced = ttnn.to_torch(
                    pipe.x_t_fp32, mesh_composer=ttnn.ConcatMeshToTensor(h.denoise_per_chip[0], dim=0)
                )[0]
                log(f"[denoise-only] replay {rnd}: PCC vs eager = {_pcc(traced, eager):.6f}")
            log("DONE (denoise-only scope)")
            return

        # --- eager warmup (also the reference actions) ---
        eager = ttnn.to_torch(
            pipe._sample_actions_device(upstream), mesh_composer=ttnn.ConcatMeshToTensor(h.denoise_per_chip[0], dim=0)
        )[0]
        pipe._refresh_noise_buffer()
        log(f"eager warmup done: actions {tuple(eager.shape)} finite={torch.isfinite(eager).all().item()}")

        # --- PER-SUBMESH trace capture (begin on ALL per-chip submeshes, run, end on ALL) ---
        all_subs = list(h.vision_per_chip) + list(h.prefill_per_chip) + list(h.denoise_per_chip)
        log(f"begin_trace_capture on {len(all_subs)} per-chip submeshes ...")
        tids = []
        for i, sm in enumerate(all_subs):
            tids.append(ttnn.begin_trace_capture(sm, cq_id=0))
        log("all submeshes in capture mode; running _sample_actions_device ...")
        pipe._sample_actions_device(upstream)
        log("calling end_trace_capture on all submeshes ...")
        for sm, tid in zip(all_subs, tids):
            ttnn.end_trace_capture(sm, tid, cq_id=0)
        log("captured per-submesh traces")

        # --- replay (execute in stage dependency order: vision -> prefill -> denoise) ---
        for rnd in (1, 2, 3):
            pipe._refresh_noise_buffer()
            ordered = list(zip(all_subs, tids))
            for sm, tid in ordered:
                ttnn.execute_trace(sm, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(h.denoise_per_chip[0])
            traced = ttnn.to_torch(pipe.x_t_fp32, mesh_composer=ttnn.ConcatMeshToTensor(h.denoise_per_chip[0], dim=0))[
                0
            ]
            log(f"replay {rnd}: actions {tuple(traced.shape)} PCC vs eager = {_pcc(traced, eager):.6f}")
        log("DONE — per-submesh socket-pipeline trace replayed")


if __name__ == "__main__":
    main()
    sys.exit(0)
