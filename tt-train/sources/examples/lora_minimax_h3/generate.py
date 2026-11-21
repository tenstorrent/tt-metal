# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Text-to-video(+audio) with the in-tree ttnn MiniMax-H3 pipeline and a LoRA adapter, on the Loud Box.

    PYTHONPATH=$TT_METAL_HOME python generate.py --prompt "..." --lora lora/pixelart-300 --out out/clip

The adapter is applied at weight-load time: `W + scale * B @ A` on the 300 adapted matrices of the torch state
dict the pipeline reads, before upload. No merged checkpoint on disk. tt_dit's own cache key carries no weight
hash, so each (adapter file content, scale) gets its own folder under --cache-dir: retraining into the same
adapter folder never reuses stale merged weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger("minimax_h3.generate")

MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
DEFAULT_WEIGHTS = Path.home() / ".cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots" / MODEL_REVISION
LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
FPS = 24


MIN_FRAMES, MAX_FRAMES = 107, 362  # served range, 17n + 5


def frames_for_duration(seconds: float) -> int:
    """Nearest served frame count (17n + 5 at 24 fps) for a duration in seconds."""
    n = round((seconds * FPS - 5) / 17)
    return int(min(MAX_FRAMES, max(MIN_FRAMES, 17 * n + 5)))


def apply_trigger(prompt: str, trigger: str | None) -> str:
    if not trigger:
        return prompt
    marker = "[Shot 1] "
    if marker in prompt:
        return prompt.replace(marker, f"{marker}{trigger}, ", 1)
    return f"{trigger}, {prompt}"


def load_adapter(folder: Path):
    """(base_key -> (A, B)) float32 torch tensors, plus the trigger from training_args.json."""
    import torch
    from safetensors.torch import load_file

    state = load_file(str(folder / LORA_WEIGHT_NAME))
    pairs = {}
    for key, value in state.items():
        base, slot = key.rsplit(".lora_", 1)  # "...to_q" , "A.weight"
        pairs.setdefault(base + ".weight", {})[slot[0]] = value.float()
    trigger = None
    args_path = folder / "training_args.json"
    if args_path.exists():
        trigger = json.loads(args_path.read_text()).get("trigger")
    return {k: (v["A"], v["B"]) for k, v in pairs.items()}, trigger


def adapter_digest(folder: Path, length: int = 12) -> str:
    h = hashlib.sha256()
    with open(folder / LORA_WEIGHT_NAME, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def make_pipeline_class(adapter: dict, scale: float):
    """Pipeline subclass whose transformer state dict carries the LoRA deltas."""
    from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    class LoraPipeline(MiniMaxH3Pipeline):
        def _read_safetensors(self, subfolder: str):
            state = super()._read_safetensors(subfolder)
            if subfolder != self.transformer_subfolder:
                return state
            import torch

            applied = 0
            t0 = time.perf_counter()
            for key, (a, b) in adapter.items():
                if key not in state:
                    raise KeyError(f"adapter targets {key}, not in {subfolder}")
                w = state[key]
                if (b.shape[0], a.shape[1]) != tuple(w.shape):
                    raise ValueError(f"{key}: weight {tuple(w.shape)} vs B@A {(b.shape[0], a.shape[1])}")
                state[key] = (w.float() + scale * (b @ a)).to(w.dtype)
                applied += 1
            log.info("LoRA applied to %d matrices (scale %.3g) in %.0f s", applied, scale, time.perf_counter() - t0)
            return state

    return LoraPipeline


def write_outputs(output, out: Path, stem: str) -> None:
    """mp4 (H.264 + AAC) via PyAV, plus the wav and the first frame as png."""
    import torch

    out.mkdir(parents=True, exist_ok=True)
    video = output.video
    frames = video[0].permute(1, 2, 3, 0).clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()  # (F, H, W, 3)
    audio = output.audio[0].float().cpu().numpy()  # (2, samples)
    sr = int(output.sampling_rate)

    pcm = np.clip(audio.T, -1.0, 1.0)
    with wave.open(str(out / f"{stem}.wav"), "wb") as handle:
        handle.setnchannels(pcm.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes((pcm * 32767.0).astype("<i2").tobytes())
    from PIL import Image

    Image.fromarray(frames[0]).save(out / f"{stem}_frame0.png")

    try:
        import av
    except ImportError:
        np.save(out / f"{stem}_frames.npy", frames)
        log.warning("PyAV missing: frames saved as npy, no mp4")
        return
    with av.open(str(out / f"{stem}.mp4"), mode="w") as container:
        vs = container.add_stream("libx264", rate=FPS)
        vs.width, vs.height, vs.pix_fmt = int(frames.shape[2]), int(frames.shape[1]), "yuv420p"
        vs.options = {"crf": "16"}
        aus = container.add_stream("aac", rate=sr)
        aus.layout = "stereo"
        for frame in frames:
            for packet in vs.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
                container.mux(packet)
        for packet in vs.encode():
            container.mux(packet)
        samples = np.ascontiguousarray(pcm.T.astype(np.float32))  # (2, n) planar
        aframe = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
        aframe.sample_rate = sr
        for packet in aus.encode(aframe):
            container.mux(packet)
        for packet in aus.encode():
            container.mux(packet)
    log.info(
        "wrote %s (%d frames, %dx%d, %d Hz audio)",
        out / f"{stem}.mp4",
        frames.shape[0],
        frames.shape[2],
        frames.shape[1],
        sr,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", default=None, help="one prompt; or use --prompts-dir")
    p.add_argument("--prompts-dir", default=None, help="folder of N.txt briefs")
    p.add_argument("--only", default=None, help="prompt indices from --prompts-dir, e.g. 1 or 0,2 (default: all)")
    p.add_argument("--lora", default=None, help="adapter folder; omit for the base model")
    p.add_argument("--lora-scale", type=float, default=1.0)
    p.add_argument("--trigger", default=None, help="override the trigger from training_args.json; '' disables")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="HF snapshot folder")
    p.add_argument("--mesh", default="1,8")
    p.add_argument("--duration", type=float, default=None, help="seconds, snapped to 17n+5 frames; served 4.5 to 15 s")
    p.add_argument("--frames", type=int, default=None, help="explicit frame count (17n+5); overrides --duration")
    p.add_argument("--aspect", default="1:1", help="w:h, e.g. 1:1 or 16:9; canvas short edge is 768")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", "--out", dest="out", default="out")
    p.add_argument("--stem", default=None, help="output file stem (default: prompt index, or lora name + seed)")
    p.add_argument(
        "--cache-dir", default=str(Path.home() / "tt_dit_cache"), help="tt_dit weight cache root; per adapter"
    )
    p.add_argument("--no-coresident", action="store_true", help="evict partitions between stages if DRAM is tight")
    p.add_argument("--links", type=int, default=1)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    root = os.environ.get("TT_METAL_HOME")
    if root and root not in sys.path:
        sys.path.insert(0, root)

    if (args.prompt is None) == (args.prompts_dir is None):
        raise SystemExit("give exactly one of --prompt or --prompts-dir")
    if args.prompts_dir:
        folder = Path(args.prompts_dir)
        indices = (
            [s.strip() for s in args.only.split(",")]
            if args.only
            else sorted((q.stem for q in folder.glob("*.txt")), key=lambda s: int(s) if s.isdigit() else s)
        )
        prompts = [(idx, (folder / f"{idx}.txt").read_text(encoding="utf-8").strip()) for idx in indices]
    else:
        prompts = [(None, args.prompt)]
    frames = args.frames if args.frames is not None else frames_for_duration(args.duration if args.duration else 4.5)
    if frames < MIN_FRAMES or frames > MAX_FRAMES or (frames - 5) % 17:
        raise SystemExit(f"frames {frames} must be 17n+5 in [{MIN_FRAMES}, {MAX_FRAMES}]")
    log.info(
        "%d prompt(s), %d frames (%.1f s), %d steps, seed %d", len(prompts), frames, frames / FPS, args.steps, args.seed
    )

    adapter, trigger = ({}, None)
    lora_name = "base"
    cache_key = f"base-s{args.lora_scale:g}"
    if args.lora:
        adapter, trigger = load_adapter(Path(args.lora))
        lora_name = (
            Path(args.lora).name if Path(args.lora).name not in ("raw",) else Path(args.lora).parent.name + "-raw"
        )
        digest = adapter_digest(Path(args.lora))
        cache_key = f"{lora_name}-{digest}-s{args.lora_scale:g}"
        log.info("adapter %s (sha256 %s): %d matrices, trigger %r", args.lora, digest, len(adapter), trigger)
    if args.trigger is not None:
        trigger = args.trigger or None
    os.environ["TT_DIT_CACHE_DIR"] = str(Path(args.cache_dir) / cache_key)
    log.info("tt_dit cache %s", os.environ["TT_DIT_CACHE_DIR"])

    import ttnn
    from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3 import MINIMAX_H3_BUCKET_LADDER_4X8, MiniMaxH3Pipeline

    rows, cols = (int(v) for v in args.mesh.split(","))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols), l1_small_size=65536)
    try:
        cls = make_pipeline_class(adapter, args.lora_scale) if adapter else MiniMaxH3Pipeline
        t0 = time.perf_counter()
        pipe = cls.create_pipeline(
            mesh_device=mesh,
            weights_dir=args.weights,
            tp_axis=1,
            sp_axis=0,
            num_links=args.links,
            topology=ttnn.Topology.Linear,
            bucket_ladder=MINIMAX_H3_BUCKET_LADDER_4X8,
            vae_output_type="float",
            warmup=False,
            coresident=False if args.no_coresident else None,
        )
        log.info("pipeline ready in %.0f s", time.perf_counter() - t0)
        w, h = (int(v) for v in args.aspect.split(":"))
        for idx, raw in prompts:
            prompt = apply_trigger(raw, trigger)
            log.info("prompt %s: %s", idx if idx is not None else "-", prompt[:160])
            t0 = time.perf_counter()
            output = pipe(
                prompt, num_frames=frames, aspect_ratio=(w, h), num_inference_steps=args.steps, seed=args.seed
            )
            log.info("generated %d frames in %.0f s", output.num_frames, time.perf_counter() - t0)
            stem = args.stem or (str(idx) if idx is not None else f"{lora_name}-seed{args.seed}")
            write_outputs(output, Path(args.out), stem)
            (Path(args.out) / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "prompt": prompt,
                        "lora": args.lora,
                        "lora_scale": args.lora_scale,
                        "seed": args.seed,
                        "frames": frames,
                        "steps": args.steps,
                        "aspect": args.aspect,
                    },
                    indent=1,
                )
            )
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
