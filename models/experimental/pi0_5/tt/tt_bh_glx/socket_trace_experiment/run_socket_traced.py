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

        # CRITICAL: the real _refresh_noise_buffer() draws FRESH torch.randn noise on
        # every call (no seed). The eager reference and the trace replay would then see
        # DIFFERENT initial noise, so "PCC vs eager" compares two different inferences and
        # looks degraded even when the trace is correct. Pin it to ONE fixed noise so the
        # replay reproduces the eager reference exactly (denoise is deterministic in x_t).
        if os.environ.get("FIXED_NOISE", "1").lower() in ("1", "true", "yes"):
            _ah, _ahp = pipe.action_horizon, pipe._action_horizon_padded
            # Match the torch reference noise contract EXACTLY: seed SEED+1, then the
            # first randn(1, ah, adim) IS the denoise noise (same as _trace_e2e_full's
            # x_t and TorchPi0_5Model.sample_actions' internal first draw). This makes the
            # PI05_E2E_PCC torch comparison valid AND pins eager==replay for trace fidelity.
            torch.manual_seed(SEED + 1)
            _np = torch.zeros(1, _ahp, pipe.action_dim, dtype=torch.float32)
            _np[:, :_ah, :] = torch.randn(1, _ah, pipe.action_dim)
            _fixed_noise = ttnn.from_torch(_np, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            pipe._refresh_noise_buffer = lambda: ttnn.copy_host_to_device_tensor(_fixed_noise, pipe.x_t_fp32)
            log("FIXED_NOISE: _refresh_noise_buffer pinned to torch-matched seed-(SEED+1) noise")

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

        # FIX UNDER TEST #2: send_direct_async is ASYNC and reads the SEND SOURCE
        # buffer; in the multi-step loop the next step's compute overwrites that
        # (reused transient) source before the in-flight fabric read completes ->
        # corruption under trace replay (no per-step host sync). Copy each send
        # source into a PERSISTENT, HELD, distinct-per-call buffer so it is never
        # overwritten mid-transfer. Keeps sockets.
        if os.environ.get("USE_PERSIST_SRC", "").lower() in ("1", "true", "yes"):
            _orig_send2 = pipe.transport.send
            _held = []

            def _persist_send(src, dst, *, out_buf=None, tag=None):
                psrc = ttnn.clone(src)  # fresh buffer; held below so it is not freed/reused
                _held.append(psrc)
                return _orig_send2(psrc, dst, out_buf=out_buf, tag=f"{tag}:p{len(_held)}")

            pipe.transport.send = _persist_send
            log("transport.send monkey-patched to persistent+distinct send-source buffers")

        # FIX UNDER TEST #3: the transport caches ONE socket pair per (src,dst), so
        # all N Euler steps share the same socket FIFO/credit state. p2p works at
        # N=5 but sockets don't, and buffer-only patches were byte-identical -> the
        # shared SOCKET (not the buffer) is the suspect. Give each send call its OWN
        # fresh socket pair so no step shares FIFO/credit state with another.
        if os.environ.get("USE_PERSTEP_SOCKET", "").lower() in ("1", "true", "yes"):
            _held_socks = []

            def _fresh_pair(src_mesh, dst_mesh):
                conn = ttnn.SocketConnection(
                    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
                    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 1)),
                )
                mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 4096 * 4)
                p = ttnn.create_socket_pair(src_mesh, dst_mesh, ttnn.SocketConfig([conn], mem))
                _held_socks.append(p)
                return p

            pipe.transport._pair = _fresh_pair
            # also force a unique recv buffer per call so nothing is shared across steps
            _orig_send3 = pipe.transport.send
            _ctr3 = [0]

            def _send3(src, dst, *, out_buf=None, tag=None):
                _ctr3[0] += 1
                return _orig_send3(src, dst, out_buf=out_buf, tag=f"{tag}:ps{_ctr3[0]}")

            pipe.transport.send = _send3
            log("transport patched: FRESH socket pair + distinct recv buffer per send call")

        if SCOPE == "vp":
            # SigLIP(vision) + prefill ONLY (single-pass, no denoise loop). Trace
            # vision_per_chip + prefill_per_chip via sockets; compare the prefill
            # per-layer KV (layer 0's K) eager vs traced across replays.
            def _vp_run():
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
                return per_layer_kv

            kv = _vp_run()
            ttnn.synchronize_device(h.prefill_per_chip[-1])
            eager = ttnn.to_torch(kv[0][0], mesh_composer=ttnn.ConcatMeshToTensor(h.prefill_per_chip[0], dim=0))
            log(f"eager vision+prefill done: K0 {tuple(eager.shape)}")
            subs = list(h.vision_per_chip) + list(h.prefill_per_chip)
            tids = [ttnn.begin_trace_capture(sm, cq_id=0) for sm in subs]
            kv_t = _vp_run()
            for sm, tid in zip(subs, tids):
                ttnn.end_trace_capture(sm, tid, cq_id=0)
            log("captured vision+prefill per-submesh traces")
            for rnd in (1, 2, 3):
                for sm, tid in zip(subs, tids):
                    ttnn.execute_trace(sm, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(h.prefill_per_chip[-1])
                traced = ttnn.to_torch(kv_t[0][0], mesh_composer=ttnn.ConcatMeshToTensor(h.prefill_per_chip[0], dim=0))
                log(f"[vision+prefill] replay {rnd}: K0 PCC vs eager = {_pcc(traced, eager):.6f}")
            log("DONE (vision+prefill scope)")
            return

        if os.environ.get("NO_DEALLOC", "").lower() in ("1", "true", "yes"):
            ttnn.deallocate = lambda *a, **k: None
            log("ttnn.deallocate patched to NO-OP (transient addresses frozen)")

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
            _sync_all = os.environ.get("SYNC_ALL", "").lower() in ("1", "true", "yes")
            for rnd in (1, 2, 3):
                pipe._refresh_noise_buffer()
                for sm, tid in zip(subs, tids):
                    ttnn.execute_trace(sm, tid, cq_id=0, blocking=False)
                if _sync_all:
                    for sm in subs:
                        ttnn.synchronize_device(sm)
                else:
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
        import time

        nperf = int(os.environ.get("PERF_ITERS", "20"))
        ordered = list(zip(all_subs, tids))
        for rnd in (1, 2, 3):
            pipe._refresh_noise_buffer()
            for sm, tid in ordered:
                ttnn.execute_trace(sm, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(h.denoise_per_chip[0])
            traced = ttnn.to_torch(pipe.x_t_fp32, mesh_composer=ttnn.ConcatMeshToTensor(h.denoise_per_chip[0], dim=0))[
                0
            ]
            log(f"replay {rnd}: actions {tuple(traced.shape)} PCC vs eager = {_pcc(traced, eager):.6f}")

        # --- perf: time the traced replay (execute all submesh traces + drain) ---
        ttnn.synchronize_device(h.denoise_per_chip[0])
        t0 = time.perf_counter()
        for _ in range(nperf):
            for sm, tid in ordered:
                ttnn.execute_trace(sm, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(h.denoise_per_chip[0])
        dt_ms = (time.perf_counter() - t0) * 1000.0 / nperf
        log(
            f"PERF: traced all-socket e2e replay = {dt_ms:.2f} ms/inference (avg of {nperf}); {1000.0/dt_ms:.1f} infer/s"
        )

        # ===================== numerical validation vs torch (opt-in) =====================
        # PI05_E2E_PCC=1 compares the all-socket eager actions against the torch
        # Pi0_5Model.sample_actions reference. Same input + noise contract as
        # _trace_e2e_full.py: inputs drawn at SEED, noise drawn at SEED+1 (FIXED_NOISE
        # above already pins x_t to that exact seed-(SEED+1) draw), torch reseeds SEED+1
        # internally so its first randn matches. Compare on the logical action_horizon.
        if os.environ.get("PI05_E2E_PCC", "").lower() in ("1", "true", "yes", "on"):
            from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model as TorchPi0_5Model

            ah = pipe.action_horizon
            t_images = [images[i] for i in range(N_CAMS)]  # each already (1,3,S,S)
            t_img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(N_CAMS)]
            t_lang_masks = torch.ones(1, lang_tokens.shape[-1], dtype=torch.bool)
            torch.manual_seed(SEED)
            ref = TorchPi0_5Model(cfg, loader)
            with torch.no_grad():
                torch.manual_seed(SEED + 1)
                ref_actions = ref.sample_actions(
                    images=t_images,
                    img_masks=t_img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=t_lang_masks,
                    state=None,
                )[0].float()
            sock_actions = eager[:ah].float()  # eager is (ah_pad, adim); slice logical ah
            pcc = _pcc(sock_actions, ref_actions)
            mae = (sock_actions - ref_actions).abs().mean().item()
            log(
                f"PCC vs torch: {pcc:.6f}  MAE={mae:.5f}  ref(mean={ref_actions.mean():.4f} std={ref_actions.std():.4f})"
            )
            log(f"PCC {'>=' if pcc >= 0.95 else '<'} 0.95 target")

        log("DONE — per-submesh socket-pipeline trace replayed")


if __name__ == "__main__":
    main()
    sys.exit(0)
