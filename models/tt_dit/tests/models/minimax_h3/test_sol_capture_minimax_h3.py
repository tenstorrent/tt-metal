# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Capture real MiniMax-H3 attention activations and Sol-Attn routing statistics from a 50-step
t2va generation (Sol-Attn port, DESIGN.md step 2/3). Dense output is kept as the quality reference.

Run (4x8 Blackhole Galaxy):

    MINIMAX_H3_MODEL_PATH=/data/nwoodall/MiniMax-H3-diffusers TT_DIT_CACHE_DIR=/data/losullivan/tt_dit_cache \
    SOL_CAPTURE_DIR=/data/losullivan/sol_attn_captures \
    pytest models/tt_dit/tests/models/minimax_h3/test_sol_capture_minimax_h3.py -k "15s" -s

Per (step, layer) in STAT_STEPS x all layers: pooled Q/K (on device, 1/64 readback) -> host routing
statistics for a tau grid (block density, chunk density at candidate chunk sizes, chain-slot union
density and imbalance). Pooled tensors are saved for POOLED_STEPS, full Q/K/V for FULL_PAIRS.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3 import transformer_minimax_h3 as T
from ....pipelines.minimax_h3 import pipeline_minimax_h3 as P
from ....pipelines.minimax_h3.packing import resolve_canvas_size
from ....pipelines.minimax_h3.policy import get_num_frames
from .common import MESH_4X8_RING
from .common_av import CALIBRATED_FOX_PROMPT, to_uint8_frames, weights_dir, write_artifacts

sys.path.insert(0, os.environ.get("SOL_ATTN_WORK", "/home/losullivan/sol_attn_work"))  # /data copy on Slurm nodes
import sol_ref as R  # noqa: E402

NUM_INFERENCE_STEPS = 50
SEED = 0
TAUS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
CHUNK_SIZES = ((256, 512), (256, 256), (128, 512), (128, 256), (64, 64))
# The scheduler runs num_inference_steps - 1 model evaluations (grid points incl. terminal 0): steps 0..48.
STAT_STEPS = ()  # inline host stats are off: offline_stats.py computes them from the saved pooled tensors
POOLED_STEPS = (0, 3, 6, 9, 10, 12, 15, 20, 25, 30, 35, 40, 44, 48)
FULL_PAIRS = {(s, l) for s in (10, 25, 48) for l in (0, 2, 12, 25, 40, 49)}
# bucket rung per working point with the 39-token calibrated prompt (matches ~/h3_attn_timing)
RUNG = {5: 37888, 10: 73472, 15: 109312}


class Capture:
    def __init__(self, mesh_device, out_dir: Path, num_layers: int):
        self.mesh_device = mesh_device
        self.out = out_dir
        self.num_layers = num_layers
        self.enabled = False
        self.step = -1
        self.layer = 0
        self.layout = None
        self.rung = None
        self.pool_t = None  # device [1,1,rows,224] per shard: pooling matrix transposed
        self.pool_mode = "device"
        self.stats_path = out_dir / "stats.jsonl"
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.futures = []
        self.lock = threading.Lock()
        self.t_capture = 0.0
        (out_dir / "pooled").mkdir(parents=True, exist_ok=True)
        (out_dir / "full").mkdir(parents=True, exist_ok=True)

    # -- geometry ---------------------------------------------------------------------------
    def geometry(self) -> R.Geometry:
        return R.Geometry(shards=8, rows=self.rung // 8, logical_n=self.layout.sequence_length)

    def build_pool_matrix(self):
        geo = self.geometry()
        nb = geo.blocks_per_shard
        nb_pad = -(-nb // 32) * 32
        mats = torch.zeros(8, 1, geo.rows, nb_pad, dtype=torch.float32)
        lengths = geo.block_lengths().view(8, nb)
        for s in range(8):
            for b in range(nb):
                length = int(lengths[s, b])
                if length == 0:
                    continue
                r0 = b * geo.block
                mats[s, 0, r0 : r0 + length, b] = 1.0 / length
        self.pool_t = ttnn.from_torch(
            mats,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=[None, 0], mesh_shape=tuple(self.mesh_device.shape)
            ),
        )
        self.nb, self.nb_pad = nb, nb_pad
        self.ckc = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    # -- readback helpers ---------------------------------------------------------------------
    def _composer(self, dims):
        return ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=dims, mesh_shape=tuple(self.mesh_device.shape))

    def pooled(self, x: ttnn.Tensor) -> torch.Tensor:
        """[1,14,rows,128] per device -> host [56, 8*nb, 128] fp32 block means."""
        if self.pool_mode == "device":
            try:
                xt = ttnn.transpose(x, -2, -1)  # [1,14,128,rows]
                pt = ttnn.matmul(xt, self.pool_t, compute_kernel_config=self.ckc)  # [1,14,128,nb_pad]
                ttnn.deallocate(xt)
                host = ttnn.to_torch(pt, mesh_composer=self._composer([1, 3]))  # [1,56,128,8*nb_pad]
                ttnn.deallocate(pt)
                host = host[0].float().transpose(1, 2)  # [56, 8*nb_pad, 128]
                host = host.reshape(56, 8, self.nb_pad, 128)[:, :, : self.nb].reshape(56, 8 * self.nb, 128)
                return host
            except Exception as exc:  # fall back to host pooling for the rest of the run
                logger.warning(f"device pooling failed ({exc}); pooling on host from now on")
                self.pool_mode = "host"
        full = ttnn.to_torch(x, mesh_composer=self._composer([1, 2]))[0]  # [56, N, 128]
        return R.pool_blocks(full, self.geometry())

    def full(self, x: ttnn.Tensor) -> torch.Tensor:
        return ttnn.to_torch(x, mesh_composer=self._composer([1, 2]))[0].contiguous()  # [56, N, 128] bf16

    # -- hooks --------------------------------------------------------------------------------
    def on_forward(self):
        if self.enabled:
            self.step += 1
            self.layer = 0

    def on_ring_sdpa(self, q, k, v):
        if not self.enabled:
            return
        step, layer = self.step, self.layer
        self.layer += 1
        want_stats = step in STAT_STEPS
        want_pooled = step in POOLED_STEPS
        want_full = (step, layer) in FULL_PAIRS
        if not (want_stats or want_pooled or want_full):
            return
        t0 = time.time()
        if self.pool_t is None:
            self.build_pool_matrix()
        q_bar = self.pooled(q)
        kc = self.pooled(k)
        vc = self.pooled(v) if want_pooled else None
        if want_full:
            qf, kf, vf = self.full(q), self.full(k), self.full(v)
            torch.save(
                {"q": qf, "k": kf, "v": vf, "step": step, "layer": layer},
                self.out / "full" / f"s{step:02d}_l{layer:02d}.pt",
            )
            del qf, kf, vf
        if want_pooled:
            torch.save(
                {"q_bar": q_bar.bfloat16(), "kc": kc.bfloat16(), "vc": vc.bfloat16(), "step": step, "layer": layer},
                self.out / "pooled" / f"s{step:02d}_l{layer:02d}.pt",
            )
        if want_stats:
            self.futures.append(self.pool.submit(self._stats, step, layer, q_bar, kc))
        self.t_capture += time.time() - t0

    # -- host statistics ------------------------------------------------------------------------
    def _stats(self, step: int, layer: int, q_bar: torch.Tensor, kc: torch.Tensor):
        try:
            torch.set_num_threads(16)
            geo = self.geometry()
            lengths = geo.block_lengths()
            valid = lengths > 0
            scale = 128**-0.5
            video_start = int(self.layout.video_indices[self.layout.num_condition_video_rows].item())
            scores = R.proxy_scores(q_bar, kc, scale=scale)
            rec = {
                "step": step,
                "layer": layer,
                "logical_n": geo.logical_n,
                "rung": self.rung,
                "video_start": video_start,
                "taus": {},
            }
            cores_per_head = 8
            for tau in TAUS:
                thr = R.routing_threshold(q_bar, kc, lengths, tau=tau, scale=scale, thresh_type="diag")
                thr_only = (scores > thr.unsqueeze(-1))[:, valid][:, :, valid].float().mean().item()
                exact = R.route_blocks(scores, thr, lengths, band=1, sink_tokens=video_start, geo=geo)
                eff = exact[:, valid][:, :, valid].float().mean().item()
                entry = {"block_density_threshold_only": thr_only, "block_density_effective": eff, "chunk": {}}
                for qc, kcs in CHUNK_SIZES:
                    dens = R.chunk_density(exact, geo, qc, kcs)
                    entry["chunk"][f"{qc}x{kcs}"] = {
                        "mean": float(dens.mean()),
                        "per_head": [round(float(x), 4) for x in dens],
                    }
                cm = R.chunk_mask_from_blocks(exact, geo)
                nqc = geo.q_chunks_per_shard
                slot = {"strided": [], "adjacent": []}
                for s in range(8):
                    sub = cm[:, s * nqc : (s + 1) * nqc]
                    for mapping in slot:
                        slot[mapping].append(R.slot_union_stats(sub, geo, cores_per_head, mapping))
                entry["slot_256x512"] = {
                    m: {
                        "union_density": float(np.mean([x["union_density"] for x in slot[m]])),
                        "slot_max_over_mean": float(np.mean([x["slot_max_over_mean"] for x in slot[m]])),
                    }
                    for m in slot
                }
                # per-shard-row density (SP rank 0 holds the prefix) at production chunking
                entry["per_shard_q_density_256x512"] = [
                    float(cm[:, s * nqc : (s + 1) * nqc].float().mean()) for s in range(8)
                ]
                rec["taus"][str(tau)] = entry
            with self.lock:
                with open(self.stats_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        except Exception as exc:  # never kill the generation
            logger.error(f"stats failed for step {step} layer {layer}: {exc!r}")

    def finish(self):
        for fut in self.futures:
            fut.result()
        self.pool.shutdown()


@pytest.mark.parametrize("duration_s", [5, 10, 15], ids=lambda d: f"{d}s")
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_sol_capture(mesh_device, reset_seeds, duration_s):
    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")
    out = Path(os.environ.get("SOL_CAPTURE_DIR", str(Path.home() / "sol_attn_captures"))) / f"{duration_s}s"
    out.mkdir(parents=True, exist_ok=True)
    height, width = resolve_canvas_size(16, 9)
    num_frames = get_num_frames(duration_s)
    rung = RUNG[duration_s]
    logger.info(f"capture {duration_s}s: {width}x{height}, {num_frames} frames, rung {rung}, out {out}")

    pipeline = P.MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights,
        vae_output_type="float",
        bucket_ladder=(rung, 120320) if rung < 120320 else (rung,),
    )
    num_layers = len(pipeline._transformer.transformer_blocks)
    cap = Capture(mesh_device, out, num_layers)

    # hooks: step counter on the transformer forward, layout from _denoise, Q/K/V from the ring op
    orig_forward = T.MiniMaxH3Transformer3DModel.forward
    orig_ring = ttnn.transformer.ring_joint_scaled_dot_product_attention
    orig_denoise = P.MiniMaxH3Pipeline._denoise

    def forward(self, *args, **kwargs):
        cap.on_forward()
        return orig_forward(self, *args, **kwargs)

    def ring(q, k, v, *args, **kwargs):
        cap.on_ring_sdpa(q, k, v)
        return orig_ring(q, k, v, *args, **kwargs)

    def denoise(self, transformer, layout, *args, **kwargs):
        cap.layout = layout
        cap.rung = self._select_bucket(layout.sequence_length) if self.bucket_denoise else rung
        return orig_denoise(self, transformer, layout, *args, **kwargs)

    T.MiniMaxH3Transformer3DModel.forward = forward
    ttnn.transformer.ring_joint_scaled_dot_product_attention = ring
    P.MiniMaxH3Pipeline._denoise = denoise
    try:
        gen_kwargs = dict(num_frames=num_frames, height=height, width=width, seed=SEED)
        with pipeline.quiet():
            pipeline(CALIBRATED_FOX_PROMPT, num_inference_steps=3, **gen_kwargs)
        assert cap.rung == rung, f"request landed on rung {cap.rung}, expected {rung}"
        logger.info(
            f"layout: seq {cap.layout.sequence_length}, text {cap.layout.text_indices.numel()}, "
            f"video_start {int(cap.layout.video_indices[cap.layout.num_condition_video_rows])}, rung {cap.rung}"
        )
        cap.enabled = True
        t0 = time.time()
        output = pipeline(CALIBRATED_FOX_PROMPT, num_inference_steps=NUM_INFERENCE_STEPS, **gen_kwargs)
        ttnn.synchronize_device(mesh_device)
        t_gen = time.time() - t0
        cap.enabled = False
    finally:
        T.MiniMaxH3Transformer3DModel.forward = orig_forward
        ttnn.transformer.ring_joint_scaled_dot_product_attention = orig_ring
        P.MiniMaxH3Pipeline._denoise = orig_denoise

    logger.info(f"generation {t_gen:.1f}s of which capture readback {cap.t_capture:.1f}s; steps seen {cap.step + 1}")
    frames = to_uint8_frames(output)
    np.save(out / "dense_frames.npy", frames)
    write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, out, stem="dense")
    meta = {
        "duration_s": duration_s,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "steps": NUM_INFERENCE_STEPS,
        "seed": SEED,
        "prompt": CALIBRATED_FOX_PROMPT,
        "sequence_length": cap.layout.sequence_length,
        "rung": cap.rung,
        "text_tokens": int(cap.layout.text_indices.numel()),
        "video_start": int(cap.layout.video_indices[cap.layout.num_condition_video_rows]),
        "audio_rows": int(cap.layout.audio_indices.numel()),
        "num_layers": num_layers,
        "generation_seconds": t_gen,
        "capture_readback_seconds": cap.t_capture,
        "pool_mode": cap.pool_mode,
    }
    meta["steps_seen"] = cap.step + 1
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"meta: {meta}")
    cap.finish()
    assert cap.step + 1 == NUM_INFERENCE_STEPS - 1, cap.step
    assert cap.layer == num_layers, cap.layer
