# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full DiT denoise-loop PCC: PyTorch reference vs TTNN (15s vs 30s latent length).

Uses the same timestep schedule and latent init as ``run_prompt_to_wav`` (production).
``do_cfg=False`` so random Gaussian conditioning stays finite; ADG+CFG with synthetic
cond routinely diverges to NaN on both torch and TTNN.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import print_pcc_result


def _pearson_pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite values in PCC inputs")
    if float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        raise ValueError("zero-variance tensor in PCC inputs")
    return float(np.corrcoef(a, b)[0, 1])


def _checkpoint_paths() -> Path | None:
    root = os.environ.get("ACE_STEP_CHECKPOINT_DIR", "~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints")
    safetensors = Path(root).expanduser() / "acestep-v15-base" / "model.safetensors"
    if not safetensors.is_file():
        return None
    return safetensors


def _build_production_t_schedule(infer_steps: int) -> list[float]:
    """Match ``run_prompt_to_wav._build_t_schedule`` for base (shift=1)."""
    return [1.0 - (i / float(infer_steps)) for i in range(int(infer_steps))]


def _run_torch_denoise_production(
    pipe,
    *,
    t_schedule: list[float],
    frames: int,
    enc_hs: torch.Tensor,
    ctx_lat: torch.Tensor,
    xt0: torch.Tensor,
) -> torch.Tensor:
    """Euler loop matching production ``run_prompt_to_wav`` host sampler semantics."""
    num_steps = len(t_schedule)
    xt = xt0.clone().to(dtype=torch.bfloat16)
    ctx_bf = ctx_lat.to(torch.bfloat16)
    enc_bf = enc_hs.to(torch.bfloat16)
    with torch.inference_mode():
        for step_idx in range(num_steps - 1):
            t_curr = float(t_schedule[step_idx])
            t_next = float(t_schedule[step_idx + 1])
            dt = t_curr - t_next
            acoustic = pipe.forward(
                xt_bt64=xt,
                context_latents_bt128=ctx_bf,
                timestep_index=step_idx,
                encoder_hidden_states_btd=enc_bf,
            )
            xt = xt.float() - acoustic.float() * dt
            xt = xt.to(torch.bfloat16)
        step_idx = num_steps - 1
        t_curr = float(t_schedule[-1])
        acoustic = pipe.forward(
            xt_bt64=xt,
            context_latents_bt128=ctx_bf,
            timestep_index=step_idx,
            encoder_hidden_states_btd=enc_bf,
        )
        xt = xt.float() - acoustic.float() * t_curr
    return xt.float()


@pytest.mark.parametrize("frames,label", [(375, "15s"), (750, "30s"), (1500, "60s")])
def test_dit_denoise_loop_pcc_vs_torch(frames: int, label: str):
    """Compare full Euler denoise final latents: TTNN vs PyTorch (no CFG)."""
    safetensors = _checkpoint_paths()
    if safetensors is None:
        pytest.skip("ACE-Step v1.5 checkpoints not found; set ACE_STEP_CHECKPOINT_DIR.")

    import ttnn
    from models.experimental.ace_step_v1_5.torch_ref.full_pipeline import AceStepV15TorchPipeline
    from models.experimental.ace_step_v1_5.ttnn_impl.dit_sampling_ttnn import (
        dit_init_latents_host_f32,
        precompute_dit_temb_steps,
        stage_host_temb_steps_to_device,
    )
    from models.experimental.ace_step_v1_5.ttnn_impl.e2e_model_tt import run_ttnn_denoise_loop
    from models.experimental.ace_step_v1_5.ttnn_impl.full_pipeline import AceStepV15TTNNPipeline
    from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_configure_dit_long_clip_quality
    from models.experimental.ace_step_v1_5.utils.tt_device import close_ace_step_device, open_dit_device

    infer_steps = int(os.environ.get("ACE_STEP_DIT_DENOISE_PCC_STEPS", "20"))
    seed = 42
    t_schedule = _build_production_t_schedule(infer_steps)
    timesteps_host = np.asarray(t_schedule + [0.0], dtype=np.float32)
    num_steps = len(t_schedule)
    duration_sec = float(frames) / 25.0

    if int(frames) >= 750:
        from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import (
            ace_step_configure_dit_long_clip_quality,
            ace_step_configure_dit_ultra_long_clip_quality,
        )

        ace_step_configure_dit_long_clip_quality(
            latent_frames=int(frames),
            duration_sec=duration_sec,
            mesh_sku="BH_QB",
        )
        if int(frames) >= 1125:
            ace_step_configure_dit_ultra_long_clip_quality(
                latent_frames=int(frames),
                duration_sec=duration_sec,
                mesh_sku="BH_QB",
            )

    torch_pipe = AceStepV15TorchPipeline(
        checkpoint_safetensors_path=str(safetensors),
        timesteps_host=timesteps_host,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    device = open_dit_device(ttnn, mesh_sku="BH_QB")
    mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    try:
        tt_pipe = AceStepV15TTNNPipeline(
            device=device,
            checkpoint_safetensors_path=str(safetensors),
            timesteps_host=timesteps_host,
            expected_input_length=int(frames),
        )

        cond_dim = int(tt_pipe.cond_dim)
        enc_seq = 64
        rng = np.random.default_rng(seed)
        enc_hs = torch.from_numpy(rng.standard_normal((1, enc_seq, cond_dim), dtype=np.float32))
        ctx_lat = torch.from_numpy(rng.standard_normal((1, frames, 128), dtype=np.float32))
        null_emb = torch.zeros((1, 1, cond_dim), dtype=torch.float32)
        enc_mask = torch.ones(1, enc_seq, dtype=torch.float32)
        xt0 = dit_init_latents_host_f32(batch=1, frames=int(frames), channels=64, seed=seed)

        ref = _run_torch_denoise_production(
            torch_pipe,
            t_schedule=t_schedule,
            frames=int(frames),
            enc_hs=enc_hs,
            ctx_lat=ctx_lat,
            xt0=xt0,
        )

        temb_per_step, tp_per_step, temb_on_host = precompute_dit_temb_steps(
            tt_pipe,
            num_steps=num_steps,
            target_batch=1,
            device=device,
            checkpoint_safetensors_path=str(safetensors),
            timesteps_host=timesteps_host,
        )
        if temb_on_host:
            temb_per_step, tp_per_step = stage_host_temb_steps_to_device(
                temb_per_step, tp_per_step, device=device, dram=mem
            )

        tt_out = run_ttnn_denoise_loop(
            tt_pipe,
            device=device,
            act_dtype=ttnn.bfloat16,
            mem=mem,
            t_schedule=t_schedule,
            frames=int(frames),
            enc_hs=enc_hs,
            enc_mask=enc_mask,
            ctx_lat=ctx_lat,
            null_emb=null_emb,
            do_cfg=False,
            seed=seed,
            trace_state=None,
            temb_per_step=temb_per_step,
            tp_per_step=tp_per_step,
        )
        tt_np = tt_out.detach().cpu().numpy()

        pcc = _pearson_pcc(ref.numpy(), tt_np)
        mad = float(np.max(np.abs(ref.numpy() - tt_np)))
        min_pcc = 0.88 if int(frames) >= 1500 else (0.90 if int(frames) >= 750 else 0.92)
        print_pcc_result(f"dit_denoise_loop_{label}_no_cfg", pcc, threshold=min_pcc)
        print(
            f"[ace_step_v1_5][PCC] dit_denoise_loop_{label}_no_cfg_detail: "
            f"frames={frames} steps={infer_steps} max_abs={mad:.4f}",
            flush=True,
        )
        assert pcc >= min_pcc, f"{label} denoise loop PCC {pcc:.4f} < {min_pcc} (max_abs={mad:.4f})"
    finally:
        close_ace_step_device(ttnn, device)
