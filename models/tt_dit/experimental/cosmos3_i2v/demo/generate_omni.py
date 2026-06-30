# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone CLI for the Cosmos3 omni native pipeline (I2V / T2V, ± audio).

Examples:

    # I2V (silent)
    python -m models.tt_dit.experimental.cosmos3_i2v.demo.generate_omni \\
        --image ~/cat.jpg --prompt "a cat walks" --out /tmp/cat.mp4

    # T2V + audio
    python -m models.tt_dit.experimental.cosmos3_i2v.demo.generate_omni \\
        --prompt "a thunderstorm rolls in" --enable-sound --out /tmp/storm.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from models.tt_dit.experimental.cosmos3_i2v.demo.generate import (  # noqa: E402
    close_mesh,
    open_mesh,
    resolve_mesh_shape,
    resolve_weight_dtype,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate a Cosmos3 omni clip (I2V/T2V, optional audio) on Tenstorrent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", type=Path, default=None, help="Reference image; omit for T2V.")
    p.add_argument("--prompt", required=True)
    p.add_argument("--enable-sound", action="store_true", help="Jointly generate synchronized audio.")
    p.add_argument(
        "--model-id",
        default="nvidia/Cosmos3-Super",
        help="HF repo. Must be sound_gen=True if --enable-sound (Cosmos3-Super).",
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--frames", type=int, default=17, help="Must be 4k+1 (VAE constraint).")
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path("cosmos3_omni.mp4"))
    p.add_argument("--mesh-shape", default="auto")
    p.add_argument("--weight-dtype", default="auto", choices=["auto", "bfloat16", "bfloat8_b"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--pipeline",
        default="native-cfg",
        choices=["native", "native-cfg"],
        help="native-cfg splits 4x8 mesh into dual 2x8 submeshes (sp_factor halves).",
    )
    p.add_argument("--cfg-serial-dispatch", action="store_true")
    p.add_argument("--no-tt-vae", action="store_true")
    p.add_argument(
        "--flow-shift",
        type=float,
        default=10.0,
        help="Cosmos3 audio-visual omni preset = 10. Also improves 720p sharpness silent.",
    )
    p.add_argument("--num-links", type=int, default=None)
    p.add_argument("--vae-decoder-t-chunk-size", type=int, default=None)
    p.add_argument("--vae-encoder-t-chunk-size", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    return p.parse_args(argv)


def _write_wav(waveform, path: Path, sampling_rate: int) -> None:
    import torch

    pcm = (waveform.float().clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16).cpu()
    if pcm.ndim == 1:
        pcm = pcm.unsqueeze(0)
    interleaved = pcm.T.contiguous().numpy()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(pcm.shape[0])
        wf.setsampwidth(2)
        wf.setframerate(sampling_rate)
        wf.writeframes(interleaved.tobytes())


def _find_ffmpeg() -> str | None:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _mux_audio(video_path: Path, wav_path: Path, out_path: Path) -> bool:
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        print("[generate_omni] ffmpeg not found (PATH or imageio_ffmpeg); keeping silent MP4 + WAV.", flush=True)
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(wav_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[generate_omni] ffmpeg mux failed (rc={r.returncode}):\n{r.stderr}", flush=True)
        return False
    return True


def main(argv=None) -> int:
    args = parse_args(argv)

    if (args.frames - 1) % 4 != 0:
        raise SystemExit(
            f"--frames must be 4k+1 (got {args.frames}). "
            f"Try {((args.frames - 1) // 4) * 4 + 1} or {((args.frames - 1) // 4 + 1) * 4 + 1}."
        )
    if args.image is not None and not args.image.exists():
        raise SystemExit(f"Reference image not found: {args.image}")

    import torch
    from diffusers.utils import export_to_video
    from PIL import Image

    import ttnn

    available = ttnn.get_num_devices()
    mesh_shape = resolve_mesh_shape(args.mesh_shape, available)
    weight_dtype = resolve_weight_dtype(args.weight_dtype, mesh_shape)

    mode = ("T2V" if args.image is None else "I2V") + ("+audio" if args.enable_sound else "")
    print(
        f"[generate_omni] mode={mode} model={args.model_id} prompt={args.prompt!r}\n"
        f"[generate_omni] mesh={mesh_shape} (of {available}), weight_dtype={weight_dtype}\n"
        f"[generate_omni] steps={args.steps} frames={args.frames} size={args.width}x{args.height} "
        f"fps={args.fps} flow_shift={args.flow_shift}\n"
        f"[generate_omni] out={args.out}",
        flush=True,
    )

    ref_image = None
    if args.image is not None:
        ref_image = Image.open(args.image).convert("RGB").resize((args.width, args.height))

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

    mesh = open_mesh(mesh_shape)
    try:
        t0 = time.time()
        from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_omni_native import (
            build_cosmos3_omni_native_pipeline,
            run_cosmos3_omni,
        )

        if args.pipeline == "native-cfg":
            from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_i2v_native_cfg import (
                build_cosmos3_i2v_native_cfg_pipeline,
            )

            pipe = build_cosmos3_i2v_native_cfg_pipeline(
                mesh,
                dtype=torch.bfloat16,
                hf_repo=args.model_id,
                use_tt_vae=not args.no_tt_vae,
                num_links=args.num_links,
                flow_shift=args.flow_shift,
                trunk_weight_dtype=weight_dtype,
                vae_decoder_t_chunk_size=args.vae_decoder_t_chunk_size,
                vae_encoder_t_chunk_size=args.vae_encoder_t_chunk_size,
                serial_dispatch=args.cfg_serial_dispatch,
                cache_namespace="cosmos3-omni",
            )
        else:
            pipe = build_cosmos3_omni_native_pipeline(
                mesh,
                dtype=torch.bfloat16,
                hf_repo=args.model_id,
                use_tt_vae=not args.no_tt_vae,
                num_links=args.num_links,
                flow_shift=args.flow_shift,
                trunk_weight_dtype=weight_dtype,
                vae_decoder_t_chunk_size=args.vae_decoder_t_chunk_size,
                vae_encoder_t_chunk_size=args.vae_encoder_t_chunk_size,
            )
        print(f"[generate_omni] pipeline built in {time.time() - t0:.1f}s", flush=True)

        t1 = time.time()
        result = run_cosmos3_omni(
            pipe,
            prompt=args.prompt,
            image=ref_image,
            enable_sound=args.enable_sound,
            num_frames=args.frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            fps=float(args.fps),
            generator=generator,
            output_type="pil",
        )
        print(f"[generate_omni] denoise + decode in {time.time() - t1:.1f}s", flush=True)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        silent_path = args.out.with_suffix(".silent.mp4") if args.enable_sound else args.out
        export_to_video(result.video, str(silent_path), fps=args.fps)
        print(f"[generate_omni] wrote {silent_path} ({silent_path.stat().st_size / 1024:.1f} KB)", flush=True)

        if args.enable_sound:
            sound = getattr(result, "sound", None)
            if sound is None:
                print("[generate_omni] WARNING: enable_sound=True but result.sound is None.", flush=True)
                shutil.move(str(silent_path), str(args.out))
                return 0
            sr = int(pipe.sound_tokenizer.config.sampling_rate)
            wav_path = args.out.with_suffix(".wav")
            _write_wav(sound, wav_path, sr)
            print(
                f"[generate_omni] wrote {wav_path} ({wav_path.stat().st_size / 1024:.1f} KB, "
                f"sr={sr}, shape={tuple(sound.shape)})",
                flush=True,
            )
            if _mux_audio(silent_path, wav_path, args.out):
                print(f"[generate_omni] muxed {args.out} ({args.out.stat().st_size / 1024:.1f} KB)", flush=True)
                silent_path.unlink(missing_ok=True)
    finally:
        close_mesh(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
