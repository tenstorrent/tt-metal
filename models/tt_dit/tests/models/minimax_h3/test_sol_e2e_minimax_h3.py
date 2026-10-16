# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end dense vs Sol-Attn generation with the paper's step/layer gating, same seed and prompt.

Configs come from SOL_E2E_CONFIGS, ';'-separated: `dense` or `sol:<tau>:<dense_steps>:<dense_layers>[:<end_step>[:dp0[:chunk]]]`
(`dp0` = prefix query rows sparse like kijai's default instead of dense like NVlabs).
`dense_steps` = number of leading denoising forwards run dense (paper: first 20 % of steps -> 10 of 49),
`dense_layers` = number of leading transformer layers kept dense at every step (NVlabs H3 policy: 2),
`end_step` = first forward after which attention is dense again (kijai end_percent; default: never).
Frames are saved as .npy next to the mp4 so later runs can be compared without regenerating; every sol
config reports PSNR against the dense frames of this run (or SOL_E2E_DENSE_NPY).

Run (4x8 Blackhole Galaxy):
    MINIMAX_H3_MODEL_PATH=... TT_DIT_CACHE_DIR=~/tt_dit_cache SOL_E2E_DURATION=10 \
    SOL_E2E_CONFIGS="dense;sol:1.0:10:2;sol:2.0:10:2" \
    pytest models/tt_dit/tests/models/minimax_h3/test_sol_e2e_minimax_h3.py -s
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3 import attention_minimax_h3 as A
from ....models.transformers.minimax_h3 import transformer_minimax_h3 as T
from ....pipelines.minimax_h3 import pipeline_minimax_h3 as P
from ....pipelines.minimax_h3.packing import resolve_canvas_size
from ....pipelines.minimax_h3.policy import get_num_frames
from .common import MESH_4X8_RING
from .common_av import CALIBRATED_FOX_PROMPT, to_uint8_frames, weights_dir, write_artifacts

NUM_INFERENCE_STEPS = 50
SEED = 0


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 10 * np.log10(255.0**2 / max(mse, 1e-9))


def _parse(cfg: str) -> dict:
    parts = cfg.split(":")
    if parts[0] == "dense":
        return {"label": "dense", "sol": False}
    assert parts[0] == "sol", cfg
    tau, dense_steps, dense_layers = float(parts[1]), int(parts[2]), int(parts[3])
    end_step = int(parts[4]) if len(parts) > 4 and parts[4] else 10**9
    dense_prefix = parts[5] != "dp0" if len(parts) > 5 and parts[5] else True
    mode = parts[6] if len(parts) > 6 and parts[6] else "gather"
    fused = len(parts) > 7 and parts[7] == "fused"  # sol-video's fused scoring step instead of the ttnn pre-pass
    label = (
        f"sol_tau{tau:g}_ds{dense_steps}_dl{dense_layers}"
        + (f"_end{end_step}" if end_step < 10**9 else "")
        + ("" if dense_prefix else "_dp0")
        + ("" if mode == "gather" else f"_{mode}")
        + ("_fused" if fused else "")
    )
    return {
        "label": label,
        "sol": True,
        "tau": tau,
        "dense_steps": dense_steps,
        "dense_layers": dense_layers,
        "end_step": end_step,
        "dense_prefix": dense_prefix,
        "mode": mode,
        "fused": fused,
    }


@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_sol_e2e(mesh_device, reset_seeds):
    duration_s = int(os.environ.get("SOL_E2E_DURATION", "10"))
    configs = [_parse(c) for c in os.environ.get("SOL_E2E_CONFIGS", "dense;sol:1.0:10:2").split(";")]
    artifacts = Path(os.environ.get("SOL_E2E_DIR", str(Path.home() / "h3_t2va_artifacts" / "gated")))
    artifacts.mkdir(parents=True, exist_ok=True)
    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")
    height, width = resolve_canvas_size(16, 9)
    num_frames = get_num_frames(duration_s)
    rung = {5: 37888, 10: 73472, 15: 109312}[duration_s]
    pipeline = P.MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights,
        vae_output_type="float",
        bucket_ladder=(rung, 120320) if rung < 120320 else (rung,),
    )
    attns = [block.attn for block in pipeline._transformer.transformer_blocks]
    state = {"step": -1, "cfg": None, "layout": {}, "warm": False, "time_steps": False, "step_times": [], "gen": 0}

    orig_forward = T.MiniMaxH3Transformer3DModel.forward
    orig_denoise = P.MiniMaxH3Pipeline._denoise

    def forward(self, *args, **kwargs):
        state["step"] += 1
        cfg = state["cfg"]
        sparse = False
        if cfg is not None and cfg["sol"]:
            # Warm-up pass: force sparse forwards so kernels compile, pre-pass constants upload and the
            # program cache fills before the timed run (with dense_steps=10 a 3-step warm-up never got there).
            step_on = state["warm"] or (cfg["dense_steps"] <= state["step"] < cfg["end_step"])
            for i, attn in enumerate(attns):
                attn.sol_active = bool(step_on and i >= cfg["dense_layers"])
            sparse = bool(step_on)
        density_step = state["step"] in density_steps and sparse
        if density_step:
            for attn in attns:
                attn.sol_report_density = True
        if state["time_steps"]:
            ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = orig_forward(self, *args, **kwargs)
        if state["time_steps"]:
            ttnn.synchronize_device(mesh_device)
            state["step_times"].append((state["step"], sparse, time.perf_counter() - t0))
        if state.get("sem_log"):
            logger.info(
                f"SEMLOG gen={state['gen']} step={state['step']} calls={len(state['sem_log'])} "
                + " ".join(state["sem_log"])
            )
            state["sem_log"] = []
        if density_step:
            dens = [
                (i, attn.sol_last_density)
                for i, attn in enumerate(attns)
                if attn.sol_active and attn.sol_last_density is not None
            ]
            for attn in attns:
                attn.sol_report_density = False
            if dens:
                vals = [d for _, d in dens]
                logger.info(
                    f"DENSITY gen={state['gen']} step={state['step']} layers={len(vals)} mean={sum(vals)/len(vals):.4f} min={min(vals):.4f} max={max(vals):.4f} per_layer="
                    + " ".join(f"{i}:{d:.3f}" for i, d in dens)
                )
        return out

    def denoise(self, transformer, layout, *args, **kwargs):
        video_start = int(layout.video_indices[layout.num_condition_video_rows])
        state["layout"] = dict(sequence_length=int(layout.sequence_length), video_start=video_start)
        for attn in attns:
            attn.sol_logical_n = int(layout.sequence_length)
            attn.sol_sink_tokens = video_start
        state["step"] = -1
        return orig_denoise(self, transformer, layout, *args, **kwargs)

    T.MiniMaxH3Transformer3DModel.forward = forward
    P.MiniMaxH3Pipeline._denoise = denoise

    # Diagnostic: at forward SOL_E2E_DUMP_STEP, read back the pre-pass outputs of layers SOL_E2E_DUMP_LAYERS
    # (before the ring op consumes them) and log checksums/stats, so two generations can be compared.
    dump_step = int(os.environ.get("SOL_E2E_DUMP_STEP", "-1"))
    density_steps = {int(x) for x in os.environ.get("SOL_E2E_DENSITY_STEPS", "").split(",") if x}
    dump_layers = [int(x) for x in os.environ.get("SOL_E2E_DUMP_LAYERS", "2,25").split(",") if x]
    orig_sol_tensors = A.MiniMaxH3Attention._sol_tensors

    drop_held = os.environ.get("SOL_E2E_DROP_HELD_POOLED", "0") == "1"
    post_check = os.environ.get("SOL_E2E_POSTOP_CHECK", "0") == "1"

    def _mask_sum(t):
        import torch

        h = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).to(torch.int64)
        return int(h.sum()), int(
            (h.flatten()[::31] * torch.arange(1, h.flatten()[::31].numel() + 1)).sum() & 0xFFFFFFFF
        )

    def sol_tensors(self, q, k, v):
        # E2 (sol-vid2): free the pooled K/V held from the previous call before the pre-pass reallocates them,
        # restoring the generation-1 DRAM placement without recreating the instance.
        if drop_held and self._sol_prepass is not None:
            self._sol_prepass._pooled_k = None
            self._sol_prepass._pooled_v = None
        out = orig_sol_tensors(self, q, k, v)
        layer = attns.index(self) if self in attns else -1
        if post_check and state["step"] in (dump_step, dump_step + 1) and layer in dump_layers:
            ttnn.synchronize_device(mesh_device)
            state["post_check"] = (layer, out["sol_block_mask"], _mask_sum(out["sol_block_mask"]))
        if state["step"] == dump_step and layer in dump_layers:
            import torch

            stats = {}
            for name in ("sol_block_mask", "sol_pooled_k", "sol_pooled_v", "sol_bias"):
                t = ttnn.to_torch(out[name], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
                tf = t.float()
                stats[name] = dict(
                    shape=tuple(t.shape),
                    sum=float(tf.sum()),
                    absum=float(tf.abs().sum()),
                    nan=int(torch.isnan(tf).sum()),
                    hash=int(torch.sum(tf.flatten()[::97] * torch.arange(1, tf.flatten()[::97].numel() + 1)).item())
                    & 0xFFFFFFFF,
                )
                if name == "sol_block_mask":
                    stats[name]["kept_bits"] = int(
                        sum(bin(int(x) & 0xFFFFFFFF).count("1") for x in t.flatten().tolist())
                    )
            logger.info(f"SOLDUMP gen={state['gen']} step={state['step']} layer={layer} {json.dumps(stats)}")
        return out

    A.MiniMaxH3Attention._sol_tensors = sol_tensors

    # E1 (sol-vid2): checksum the mask again after the ring op completes; a difference means its DRAM was
    # overwritten while the op was still reading it.
    orig_ring = ttnn.transformer.ring_joint_scaled_dot_product_attention

    sem_log_steps = {int(x) for x in os.environ.get("SOL_E2E_SEM_LOG_STEPS", "").split(",") if x}

    def ring_wrapped(*a, **kw):
        if state["step"] in sem_log_steps:
            sems = kw.get("multi_device_global_semaphore")
            addr = ttnn.get_global_semaphore_address(sems[0]) if sems else -1
            state.setdefault("sem_log", []).append(("s" if "sol_block_mask" in kw else "d") + f"{addr:x}")
            if state["gen"] >= 1:  # this forward may hang before the batched line prints: log each call immediately
                logger.info(
                    f"SEMCALL gen={state['gen']} step={state['step']} call={len(state['sem_log']) - 1} {state['sem_log'][-1]}"
                )
        res = orig_ring(*a, **kw)
        pc = state.pop("post_check", None)
        if pc is not None:
            ttnn.synchronize_device(mesh_device)
            layer, mask_t, before = pc
            after = _mask_sum(mask_t)
            logger.info(
                f"POSTOP gen={state['gen']} step={state['step']} layer={layer} mask before={before} after={after} same={before == after}"
            )
        return res

    if post_check or sem_log_steps:
        ttnn.transformer.ring_joint_scaled_dot_product_attention = ring_wrapped
    gen_kwargs = dict(num_frames=num_frames, height=height, width=width, seed=SEED)
    results = []
    dense_npy = os.environ.get("SOL_E2E_DENSE_NPY")
    dense_frames = np.load(dense_npy) if dense_npy else None
    try:
        for cfg in configs:
            for attn in attns:
                attn.sol_attn = cfg["sol"]
                attn.sol_active = False
                if cfg["sol"]:
                    attn.sol_tau = cfg["tau"]
                    attn.sol_thresh_type = "diag"
                    attn.sol_band = 1
                    attn.sol_dense_prefix = cfg["dense_prefix"]
                    attn.sol_mode = cfg["mode"]
                    attn._sol_prepass = None  # rebuild with this tau
                    attn.sol_fused = cfg.get("fused", False)
                    attn.sol_report_density = False
            state["cfg"] = cfg
            state["warm"] = os.environ.get("SOL_E2E_WARM_SPARSE", "1") == "1"  # 0 = old behaviour (warm-up stays dense)
            with pipeline.quiet():
                pipeline(
                    CALIBRATED_FOX_PROMPT, num_inference_steps=3, **gen_kwargs
                )  # warm/compile this config (sparse forwards forced)
            state["warm"] = False
            ttnn.synchronize_device(mesh_device)
            state["time_steps"] = os.environ.get("SOL_E2E_STEP_TIMES", "1") == "1"
            n_gens = int(
                os.environ.get("SOL_E2E_GENS", "1")
            )  # >1: repeat the timed generation (sparse program reuse across generations)
            for gen in range(n_gens):
                state["gen"] = gen
                state["step_times"] = []
                t0 = time.perf_counter()
                output = pipeline(CALIBRATED_FOX_PROMPT, num_inference_steps=NUM_INFERENCE_STEPS, **gen_kwargs)
                ttnn.synchronize_device(mesh_device)
                t_total = time.perf_counter() - t0
                if gen + 1 < n_gens:
                    logger.info(f"GEN {gen} of {cfg['label']} done in {t_total:.1f} s; steps {state['step'] + 1}")
                    # Bisect knobs for the cross-generation hang: drop the pre-pass instances (fresh constants) and/or
                    # clear the program cache (fresh ring program + workload-scoped resources) between generations.
                    if os.environ.get("SOL_E2E_GEN_RESET_PREPASS", "0") == "1":
                        for attn in attns:
                            attn._sol_prepass = None
                        logger.info("GEN knob: pre-pass instances dropped")
                    if os.environ.get("SOL_E2E_GEN_CLEAR_CACHE", "0") == "1":
                        ttnn.synchronize_device(mesh_device)
                        mesh_device.disable_and_clear_program_cache()
                        mesh_device.enable_program_cache()
                        logger.info("GEN knob: program cache cleared")
            state["time_steps"] = False
            steps_seen = state["step"] + 1
            st = state["step_times"]
            dense_fw = [t for _, sp, t in st if not sp]
            sparse_fw = [t for _, sp, t in st if sp]
            step_summary = {
                "n_dense_forwards": len(dense_fw),
                "n_sparse_forwards": len(sparse_fw),
                "dense_forward_ms_mean": 1e3 * float(np.mean(dense_fw)) if dense_fw else None,
                "sparse_forward_ms_mean": 1e3 * float(np.mean(sparse_fw)) if sparse_fw else None,
                "sparse_forward_ms_first": 1e3 * sparse_fw[0] if sparse_fw else None,
                "sparse_forward_ms_steady_mean": 1e3 * float(np.mean(sparse_fw[1:])) if len(sparse_fw) > 1 else None,
                "sparse_forward_ms_min": 1e3 * min(sparse_fw) if sparse_fw else None,
                "sparse_forward_ms_max": 1e3 * max(sparse_fw) if sparse_fw else None,
            }
            logger.info(
                f"STEP_TIMES {cfg['label']}: " + " ".join(f"{i}{'s' if sp else 'd'}={1e3 * t:.0f}" for i, sp, t in st)
            )
            frames = to_uint8_frames(output)
            stem = f"{cfg['label']}_{width}x{height}_{duration_s}s"
            np.save(artifacts / f"{stem}.npy", frames)
            paths = write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, artifacts, stem=stem)
            rec = dict(
                label=cfg["label"],
                config=cfg,
                wall_total_s=t_total,
                steps_seen=steps_seen,
                files={k: str(v) for k, v in paths.items()},
                step_times=step_summary,
                **state["layout"],
            )
            if not cfg["sol"]:
                dense_frames = frames
            elif dense_frames is not None:
                n = min(len(frames), len(dense_frames))
                per_frame = [_psnr(dense_frames[i], frames[i]) for i in range(n)]
                rec["psnr_vs_dense_mean"] = float(np.mean(per_frame))
                rec["psnr_vs_dense_min"] = float(np.min(per_frame))
            results.append(rec)
            logger.info(f"RESULT {json.dumps(rec)}")
    finally:
        T.MiniMaxH3Transformer3DModel.forward = orig_forward
        P.MiniMaxH3Pipeline._denoise = orig_denoise
        A.MiniMaxH3Attention._sol_tensors = orig_sol_tensors
        ttnn.transformer.ring_joint_scaled_dot_product_attention = orig_ring
        for attn in attns:
            attn.sol_attn = False
            attn.sol_active = False
    (artifacts / f"results_{duration_s}s_{int(time.time())}.json").write_text(json.dumps(results, indent=1))
    for r in results:
        logger.info(
            f"{r['label']:>28}: total {r['wall_total_s']:.1f} s, steps {r['steps_seen']}, "
            f"psnr vs dense {r.get('psnr_vs_dense_mean', float('nan')):.2f} dB (min {r.get('psnr_vs_dense_min', float('nan')):.2f})"
        )
