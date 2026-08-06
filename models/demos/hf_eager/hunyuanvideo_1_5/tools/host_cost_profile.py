# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only profile of the host stages of a HunyuanVideo-1.5 generation.

Every stage here runs on the host in the measured production configuration
(`HY_TT_QWEN=0 HY_TT_SIGLIP=0 HY_TT_VAE=0`), so this script never opens a
Tenstorrent device and never imports ``ttnn``.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python models/demos/hf_eager/hunyuanvideo_1_5/tools/host_cost_profile.py all

Sub-commands can also be run one at a time so each measurement gets a fresh
process (relevant for page-cache and RSS effects):

    ... host_cost_profile.py checkpoint     # pipeline from_pretrained
    ... host_cost_profile.py text           # Qwen + byT5 conditioning
    ... host_cost_profile.py siglip         # i2v image encoder
    ... host_cost_profile.py vae            # first-frame encode + 13f decode
    ... host_cost_profile.py writeout       # PNG + gif + mp4
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import time

DEFAULT_REPO = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"
DEFAULT_PROMPT = "A cat walks on the grass, realistic"
GLYPH_PROMPT = 'A neon sign that reads "OPEN" above a rainy street, cinematic'
# 13 output frames at 480x848 with 16x spatial / 4x temporal VAE compression.
LATENT_SHAPE = (1, 32, 4, 30, 53)
FRAME_COUNT = 13
FRAME_HW = (480, 848)


def _snapshot(repo: str) -> str:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    return snapshot_download(repo)


@contextlib.contextmanager
def _timed(results: dict, name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        results[name] = round(time.perf_counter() - start, 4)


def _rss_gib() -> float:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / (1024 * 1024), 3)
    return -1.0


def _page_cache_gib() -> float:
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("Cached:"):
                return round(int(line.split()[1]) / (1024 * 1024), 3)
    return -1.0


# --------------------------------------------------------------------------- checkpoint


def measure_checkpoint(args, results: dict) -> None:
    import torch

    path = _snapshot(args.repo)
    results["page_cache_gib_before"] = _page_cache_gib()

    from diffusers import HunyuanVideo15ImageToVideoPipeline

    with _timed(results, "pipeline_from_pretrained_s"):
        pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(path, torch_dtype=torch.bfloat16)
    results["rss_after_gib"] = _rss_gib()
    results["page_cache_gib_after"] = _page_cache_gib()

    # Per-component attribution: reload each one on its own so the shard/tensor
    # counts in the generation log can be matched to seconds.
    del pipe
    import gc

    gc.collect()

    from diffusers import AutoencoderKLHunyuanVideo15, HunyuanVideo15Transformer3DModel
    from transformers import Qwen2_5_VLTextModel, SiglipVisionModel, T5EncoderModel

    per_component = {}
    for name, loader in (
        (
            "transformer_33GB_4shards",
            lambda: HunyuanVideo15Transformer3DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=torch.bfloat16
            ),
        ),
        (
            "text_encoder_qwen_14GB_3shards",
            lambda: Qwen2_5_VLTextModel.from_pretrained(path, subfolder="text_encoder", torch_dtype=torch.bfloat16),
        ),
        (
            "vae_5GB",
            lambda: AutoencoderKLHunyuanVideo15.from_pretrained(path, subfolder="vae", torch_dtype=torch.bfloat16),
        ),
        (
            "image_encoder_siglip_0.9GB",
            lambda: SiglipVisionModel.from_pretrained(path, subfolder="image_encoder", torch_dtype=torch.bfloat16),
        ),
        (
            "text_encoder_2_byt5_0.9GB",
            lambda: T5EncoderModel.from_pretrained(path, subfolder="text_encoder_2", torch_dtype=torch.bfloat16),
        ),
    ):
        start = time.perf_counter()
        module = loader()
        per_component[name] = round(time.perf_counter() - start, 4)
        del module
        gc.collect()
    results["per_component_s"] = per_component


# --------------------------------------------------------------------------- text


def _text_pipeline(path):
    """A pipeline carrying only what ``encode_prompt`` touches."""
    import torch
    from diffusers import HunyuanVideo15ImageToVideoPipeline

    return HunyuanVideo15ImageToVideoPipeline.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        transformer=None,
        image_encoder=None,
    )


def measure_text(args, results: dict) -> None:
    import torch

    path = _snapshot(args.repo)
    pipe = _text_pipeline(path)
    dtype = torch.bfloat16

    def qwen(prompt):
        return pipe._get_mllm_prompt_embeds(
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            prompt=prompt,
            device=torch.device("cpu"),
            tokenizer_max_length=pipe.tokenizer_max_length,
            system_message=pipe.system_message,
            crop_start=pipe.prompt_template_encode_start_idx,
        )

    def byt5(prompt):
        return pipe._get_byt5_prompt_embeds(
            tokenizer=pipe.tokenizer_2,
            text_encoder=pipe.text_encoder_2,
            prompt=prompt,
            device=torch.device("cpu"),
            tokenizer_max_length=pipe.tokenizer_2_max_length,
        )

    repeats = args.repeats
    with torch.no_grad():
        for label, prompt in (("positive", args.prompt), ("negative", args.negative or "")):
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                embeds, _ = qwen(prompt)
                samples.append(round(time.perf_counter() - start, 4))
            results[f"qwen_{label}_s"] = samples
            results[f"qwen_{label}_shape"] = list(embeds.shape)

            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                embeds2, _ = byt5(prompt)
                samples.append(round(time.perf_counter() - start, 4))
            results[f"byt5_{label}_s"] = samples
            results[f"byt5_{label}_shape"] = list(embeds2.shape)

        # byT5 only runs its encoder when the prompt carries quoted glyph text;
        # without it the pipeline emits a zero tensor and never calls the model.
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            byt5(GLYPH_PROMPT)
            samples.append(round(time.perf_counter() - start, 4))
        results["byt5_glyph_prompt_s"] = samples

        # What the production path actually pays, end to end.
        from models.demos.hf_eager.hunyuanvideo_1_5.tt.text_conditioning import encode_prompt_pair

        pipe.transformer = type("_D", (), {"dtype": dtype})()
        start = time.perf_counter()
        values, hit = encode_prompt_pair(pipe, args.prompt, args.negative, use_cache=False)
        results["encode_prompt_pair_cold_s"] = round(time.perf_counter() - start, 4)
        results["encode_prompt_pair_cold_hit"] = hit
        results["conditioning_tensor_bytes"] = int(sum(v.numel() * v.element_size() for v in values.values()))

        cache_dir = args.cache_dir or "/tmp/hy_host_profile_cache"
        shutil.rmtree(cache_dir, ignore_errors=True)
        start = time.perf_counter()
        encode_prompt_pair(pipe, args.prompt, args.negative, use_cache=True, cache_dir=cache_dir)
        results["encode_prompt_pair_cache_write_s"] = round(time.perf_counter() - start, 4)

        # A warm hit inside the same process comes from the module-level dict; a
        # served run in a fresh process pays the torch.load instead.
        start = time.perf_counter()
        _, hit = encode_prompt_pair(pipe, args.prompt, args.negative, use_cache=True, cache_dir=cache_dir)
        results["encode_prompt_pair_memory_hit_s"] = round(time.perf_counter() - start, 4)
        results["encode_prompt_pair_memory_hit"] = hit

        from models.demos.hf_eager.hunyuanvideo_1_5.tt import text_conditioning as _tc

        _tc._MEMORY_CACHE.clear()
        start = time.perf_counter()
        _, hit = encode_prompt_pair(pipe, args.prompt, args.negative, use_cache=True, cache_dir=cache_dir)
        results["encode_prompt_pair_disk_hit_s"] = round(time.perf_counter() - start, 4)
        results["encode_prompt_pair_disk_hit"] = hit

    results["rss_gib"] = _rss_gib()


# --------------------------------------------------------------------------- siglip


def measure_siglip(args, results: dict) -> None:
    import torch
    from PIL import Image
    from transformers import SiglipImageProcessor, SiglipVisionModel

    path = _snapshot(args.repo)
    with _timed(results, "load_s"):
        encoder = SiglipVisionModel.from_pretrained(path, subfolder="image_encoder", torch_dtype=torch.bfloat16)
        extractor = SiglipImageProcessor.from_pretrained(path, subfolder="feature_extractor")

    image = (
        Image.open(args.image).convert("RGB")
        if args.image and os.path.exists(args.image)
        else Image.new("RGB", (FRAME_HW[1], FRAME_HW[0]), (90, 140, 60))
    )

    samples, preprocess = [], []
    with torch.no_grad():
        for _ in range(args.repeats):
            start = time.perf_counter()
            inputs = extractor.preprocess(images=image, do_resize=True, return_tensors="pt", do_convert_rgb=True)
            preprocess.append(round(time.perf_counter() - start, 4))
            inputs = inputs.to(dtype=torch.bfloat16)
            start = time.perf_counter()
            out = encoder(**inputs).last_hidden_state
            samples.append(round(time.perf_counter() - start, 4))
    results["preprocess_s"] = preprocess
    results["encode_s"] = samples
    results["output_shape"] = list(out.shape)
    results["rss_gib"] = _rss_gib()


# --------------------------------------------------------------------------- vae


def measure_vae(args, results: dict) -> None:
    import torch
    from diffusers import AutoencoderKLHunyuanVideo15

    path = _snapshot(args.repo)
    with _timed(results, "load_s"):
        vae = AutoencoderKLHunyuanVideo15.from_pretrained(path, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.eval()

    torch.manual_seed(0)
    latents = torch.randn(LATENT_SHAPE, dtype=torch.bfloat16)
    image = torch.randn(1, 3, 1, *FRAME_HW, dtype=torch.bfloat16)

    with torch.no_grad():
        with _timed(results, "encode_first_frame_s"):
            encoded = vae.encode(image)
            _ = encoded.latent_dist.mode() if hasattr(encoded, "latent_dist") else encoded[0]

        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            video = vae.decode(latents, return_dict=False)[0]
            samples.append(round(time.perf_counter() - start, 4))
    results["decode_13f_s"] = samples
    results["decode_output_shape"] = list(video.shape)

    from diffusers.video_processor import VideoProcessor

    processor = VideoProcessor(vae_scale_factor=16, do_resize=False, do_convert_rgb=True)
    with torch.no_grad():
        with _timed(results, "postprocess_video_pil_s"):
            frames = processor.postprocess_video(video.float(), output_type="pil")
    results["postprocess_frames"] = len(frames[0])
    results["rss_gib"] = _rss_gib()


# --------------------------------------------------------------------------- writeout


def _reference_frames(source: str | None, count: int):
    """Real generated frames when available: PNG cost is content dependent."""
    import numpy as np
    from PIL import Image

    if source and os.path.isdir(source):
        names = sorted(n for n in os.listdir(source) if n.startswith("frame_") and n.endswith(".png"))
        if names:
            loaded = [Image.open(os.path.join(source, n)).convert("RGB") for n in names]
            return [loaded[i % len(loaded)] for i in range(count)]
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(*FRAME_HW, 3), dtype=np.uint8)
    return [Image.fromarray(base) for _ in range(count)]


def measure_writeout(args, results: dict) -> None:
    from concurrent.futures import ThreadPoolExecutor

    frames = _reference_frames(args.frames_dir, args.frames)
    results["frames"] = len(frames)
    results["frame_size"] = list(frames[0].size)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    def clean():
        for name in os.listdir(outdir):
            os.remove(os.path.join(outdir, name))

    for level in (6, 1, 0):
        clean()
        start = time.perf_counter()
        for index, frame in enumerate(frames):
            frame.save(f"{outdir}/frame_{index:03d}.png", compress_level=level)
        elapsed = round(time.perf_counter() - start, 4)
        size = sum(os.path.getsize(os.path.join(outdir, n)) for n in os.listdir(outdir))
        results[f"png_serial_level{level}_s"] = elapsed
        results[f"png_serial_level{level}_bytes"] = size

    for workers, level in ((4, 6), (16, 6), (16, 1)):
        clean()
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(
                pool.map(
                    lambda pair: pair[1].save(f"{outdir}/frame_{pair[0]:03d}.png", compress_level=level),
                    enumerate(frames),
                )
            )
        results[f"png_threads{workers}_level{level}_s"] = round(time.perf_counter() - start, 4)

    start = time.perf_counter()
    frames[0].save(f"{outdir}/tt_blackhole.gif", save_all=True, append_images=frames[1:], duration=125, loop=0)
    results["gif_pillow_s"] = round(time.perf_counter() - start, 4)
    results["gif_pillow_bytes"] = os.path.getsize(f"{outdir}/tt_blackhole.gif")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        start = time.perf_counter()
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                "24",
                "-i",
                f"{outdir}/frame_%03d.png",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                f"{outdir}/tt_blackhole.mp4",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        results["mp4_s"] = round(time.perf_counter() - start, 4)

        start = time.perf_counter()
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                "8",
                "-i",
                f"{outdir}/frame_%03d.png",
                "-vf",
                "split[a][b];[a]palettegen[p];[b][p]paletteuse",
                f"{outdir}/ffmpeg.gif",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        results["gif_ffmpeg_s"] = round(time.perf_counter() - start, 4)
        results["gif_ffmpeg_bytes"] = os.path.getsize(f"{outdir}/ffmpeg.gif")


# --------------------------------------------------------------------------- driver


_COMMANDS = {
    "checkpoint": measure_checkpoint,
    "text": measure_text,
    "siglip": measure_siglip,
    "vae": measure_vae,
    "writeout": measure_writeout,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[*_COMMANDS, "all"])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--image", default=os.environ.get("HY_IMAGE"))
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--frames", type=int, default=FRAME_COUNT)
    parser.add_argument("--outdir", default="/tmp/hy_host_profile_out")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    names = list(_COMMANDS) if args.command == "all" else [args.command]
    report = {"host": os.uname().nodename, "repo": args.repo, "prompt": args.prompt}
    for name in names:
        results: dict = {}
        # Recorded because this host is shared: a concurrent job changes every
        # number below, so a measurement is only comparable at a similar load.
        results["_loadavg_before"] = os.getloadavg()[0]
        start = time.perf_counter()
        _COMMANDS[name](args, results)
        results["_total_s"] = round(time.perf_counter() - start, 4)
        results["_loadavg_after"] = os.getloadavg()[0]
        report[name] = results
        print(f"[{name}] " + json.dumps(results, sort_keys=True), flush=True)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
