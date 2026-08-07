# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Stage 4 -- generate a video from the trained LoRA(s) via tt_dit's runtime-LoRA pipeline."""

from __future__ import annotations

import time
from pathlib import Path

from pipeline_config import Config
from utils.tt_encoders import close_mesh, open_mesh


def infer(cfg: Config) -> None:
    from models.tt_dit.experimental.pipelines.pipeline_wan_runtime_lora import WanPipelineRuntimeLoRA

    f = cfg.INFER_FRAMES
    if f < 1 or (f - 1) % 4 != 0:
        raise ValueError(
            f"INFER_FRAMES={f} must be 1 or 4k+1 (1, 5, 9, ... 49); other values decode to "
            f"black video (VAE temporal stride 4)."
        )
    if f <= 5:
        print(
            f"[infer] warning: {f} frames = {(f - 1) // 4 + 1} latent frame(s); very short clips "
            f"can decode to black. Use >= 13 for reliable video."
        )

    print(f"[infer] opening mesh device {tuple(cfg.MESH_SHAPE)} ...")
    mesh_device = open_mesh(cfg.MESH_SHAPE)
    try:
        pipe = WanPipelineRuntimeLoRA.create_pipeline(
            mesh_device=mesh_device,
            checkpoint_name=cfg.MODEL_ID,
            height=cfg.INFER_H,
            width=cfg.INFER_W,
            num_frames=cfg.INFER_FRAMES,
            max_sequence_length=cfg.MAX_SEQ,
        )

        if cfg.INFER_NO_LORA:
            print("[infer] INFER_NO_LORA: running the BASE model (no adapter bound)")
        else:
            high_p = cfg.INFER_HIGH_LORA or cfg.expert_path("high")
            low_p = cfg.INFER_LOW_LORA or cfg.expert_path("low")
            high_p = high_p if Path(high_p).exists() else None
            low_p = low_p if Path(low_p).exists() else None
            if not high_p and not low_p:
                raise FileNotFoundError(
                    f"no LoRA files found ({cfg.expert_path('high')!r}, {cfg.expert_path('low')!r}) — "
                    f"train first, or set INFER_NO_LORA=true to run the base model"
                )
            print(f"[infer] registering LoRA (high={high_p}, low={low_p}, scale={cfg.LORA_SCALE})")
            handle = pipe.register_lora("style", high_path=high_p, low_path=low_p, scale=cfg.LORA_SCALE)
            pipe.set_active_lora(handle)

        print(f"[infer] generating {cfg.INFER_FRAMES}f @ {cfg.INFER_H}x{cfg.INFER_W}, {cfg.INFER_STEPS} steps ...")
        t0 = time.time()
        frames = pipe(
            prompts=[cfg.TRIGGER + cfg.VAL_PROMPT],
            negative_prompts=[cfg.NEG_PROMPT] if cfg.NEG_PROMPT else None,
            num_inference_steps=cfg.INFER_STEPS,
            guidance_scale=cfg.INFER_GUIDANCE,
            guidance_scale_2=cfg.INFER_GUIDANCE_2,
            flow_shift=cfg.INFER_FLOW_SHIFT,
            boundary_ratio=cfg.BOUNDARY_RATIO,
            seed=cfg.SEED,
        )
        print(f"[infer] done in {(time.time() - t0) / 60:.1f} min")
    finally:
        close_mesh(mesh_device)

    _write_output(frames, cfg.INFER_OUTPUT, cfg.INFER_FPS)


def _write_output(frames, out_path: str, fps: int) -> None:
    from PIL import Image

    frames_dir = Path(out_path).with_suffix("")
    frames_dir.mkdir(parents=True, exist_ok=True)
    pil_frames = [fr if isinstance(fr, Image.Image) else Image.fromarray(fr) for fr in frames]
    for i, fr in enumerate(pil_frames):
        fr.save(frames_dir / f"frame_{i:03d}.png")
    print(f"[infer] saved {len(pil_frames)} PNG frames -> {frames_dir}/")

    try:
        from diffusers.utils import export_to_video

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        export_to_video(pil_frames, out_path, fps=fps)
        print(f"[infer] saved video -> {out_path}")
    except Exception as e:
        print(
            f"[infer] mp4 export skipped ({type(e).__name__}: {e}); PNG frames are in {frames_dir}/. "
            f"Install imageio + imageio-ffmpeg and re-run for the mp4."
        )
