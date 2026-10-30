# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Ideogram 4 prompt-quality batch: every caption x every preset from ONE shared initial latent.
#
# Holds everything fixed except the caption, so differences between images come from the prompt:
#   - initial noise is loaded from a .pt drawn once on the CPU (IDEOGRAM_BATCH_LATENTS) and passed
#     to every call, instead of being drawn per call from a seed
#   - one resolution (taken from the latents file), one tt-metal tree, one node
#   - each caption file is sent to the model byte for byte (files must already be model-ready:
#     compact, key-ordered, UTF-8, no trailing newline)
#
# Captions are laid out as <IDEOGRAM_BATCH_CAPTIONS>/<backend>/<prompt id>.json. When the parent
# folder holds a SHA256SUMS, every caption and the latents file are checked against it first.
#
# Outputs under IDEOGRAM_BATCH_OUT:
#   images/<backend>/<prompt id>__<preset>.png
#   manifest.jsonl   one row per image: caption path + sha256 + token count, preset, timings, status
#   run.json         node, power limit, tree revision, latents sha256, load and warmup times
#
# Knobs (env):
#   IDEOGRAM4_WEIGHTS        fp8 checkpoint dir (required)
#   IDEOGRAM_BATCH_CAPTIONS  caption root (required)
#   IDEOGRAM_BATCH_LATENTS   .pt holding {"latents": [1, tokens, 128] float32, "height", "width", ...} (required)
#   IDEOGRAM_BATCH_PRESETS   comma list                    (default V4_QUALITY_48,V4_DEFAULT_20)
#   IDEOGRAM_BATCH_TRACED    1 = traced denoise step       (default 1; untraced hangs at 2048 today)
#   IDEOGRAM_BATCH_LIMIT     run only the first N prompt ids per backend (smoke runs; default all)
#   IDEOGRAM_BATCH_OUT       output dir                    (default ./ideogram_batch_out)
#   IDEOGRAM_BATCH_TDP       power-limit string the job script read from tt-smi (provenance only)
#
# Timing is the GPU-bench test's: wall_s around the whole call with the submesh synchronized on
# both sides, denoising_s from denoising start to VAE start, encoder and VAE from the pipeline's
# section events.
# =============================================================================

import hashlib
import json
import os
import socket
import time
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

from ....pipelines.ideogram4.pipeline_ideogram4 import MAX_TEXT_TOKENS, Ideogram4Pipeline
from .test_perf_ideogram4_gpu_bench import _RING_PIPE, SectionTimer, _git_rev

_REQUIRED = ("IDEOGRAM4_WEIGHTS", "IDEOGRAM_BATCH_CAPTIONS", "IDEOGRAM_BATCH_LATENTS")
_NEEDS_INPUTS = pytest.mark.skipif(
    not all(os.environ.get(k) for k in _REQUIRED), reason=f"needs env {', '.join(_REQUIRED)}"
)

WARMUP_PRESET = "V4_TURBO_12"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_sums(root: Path, files: list[Path]) -> None:
    """Fail fast if a frozen input changed since SHA256SUMS was written next to it."""
    sums_file = root / "SHA256SUMS"
    if not sums_file.exists():
        logger.warning(f"no {sums_file}; inputs not checksum-verified")
        return
    expected = {}
    for line in sums_file.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[(root / name.strip()).resolve()] = digest
    for f in files:
        want = expected.get(f.resolve())
        assert want is not None, f"{f} is not listed in {sums_file}"
        assert _sha256(f) == want, f"{f} changed since {sums_file} was written"


def _done(manifest: Path) -> set[tuple[str, str, str]]:
    if not manifest.exists():
        return set()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    return {(r["backend"], r["prompt_id"], r["preset"]) for r in rows if r["status"] == "ok"}


@_NEEDS_INPUTS
@pytest.mark.parametrize(
    ("mesh_device", "device_params"), [((4, 8), _RING_PIPE)], indirect=True, ids=["bh_galaxy_sp8tp4"]
)
def test_prompt_batch(*, mesh_device) -> None:
    caption_root = Path(os.environ["IDEOGRAM_BATCH_CAPTIONS"])
    latents_path = Path(os.environ["IDEOGRAM_BATCH_LATENTS"])
    presets = os.environ.get("IDEOGRAM_BATCH_PRESETS", "V4_QUALITY_48,V4_DEFAULT_20").split(",")
    traced = os.environ.get("IDEOGRAM_BATCH_TRACED", "1") == "1"
    limit = int(os.environ["IDEOGRAM_BATCH_LIMIT"]) if os.environ.get("IDEOGRAM_BATCH_LIMIT") else None
    out_dir = Path(os.environ.get("IDEOGRAM_BATCH_OUT", "ideogram_batch_out"))
    manifest = out_dir / "manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inputs: <backend>/<prompt id>.json, first `limit` prompt ids per backend.
    captions = []
    for backend_dir in sorted(p for p in caption_root.iterdir() if p.is_dir()):
        files = sorted(backend_dir.glob("*.json"))[:limit]
        captions += [(backend_dir.name, f.stem, f) for f in files]
    assert captions, f"no captions under {caption_root}"
    _check_sums(caption_root.parent, [f for _, _, f in captions] + [latents_path])

    blob = torch.load(latents_path, map_location="cpu", weights_only=True)
    latents = blob["latents"]
    height, width = int(blob["height"]), int(blob["width"])
    latents_sha = _sha256(latents_path)
    logger.info(f"{len(captions)} captions x {presets} at {height}x{width}; latents {list(latents.shape)} {latents_sha[:12]}")

    t0 = time.perf_counter()
    pipe = Ideogram4Pipeline.create_pipeline(mesh_device=mesh_device, height=height, width=width)
    load_s = time.perf_counter() - t0
    dev = pipe.mesh_device  # sync the pipeline's submesh, never the parent mesh (see the GPU-bench test)

    def generate(text: str, preset: str):
        timer = SectionTimer(dev, sync_steps=False)
        ttnn.synchronize_device(dev)
        start = time.perf_counter()
        image = pipe(
            prompts=[text], preset=preset, latents=latents, height=height, width=width, traced=traced, on_event=timer
        )[0]
        ttnn.synchronize_device(dev)
        wall_s = time.perf_counter() - start
        steps = len(timer.step_ms())
        denoising_s = timer.starts["vae"] - timer.starts["denoising"]
        return image, {
            "wall_s": round(wall_s, 3),
            "text_encoder_ms": round(timer.durations["encoder"] * 1000, 1),
            "denoising_s": round(denoising_s, 3),
            "steps": steps,
            "mean_step_ms": round(denoising_s * 1000 / steps, 1),
            "decoder_ms": round(timer.durations["vae"] * 1000, 1),
        }

    # Warmup compiles every op for this resolution and captures the step trace.
    warm_text = captions[0][2].read_text(encoding="utf-8")
    _, warm = generate(warm_text, WARMUP_PRESET)
    logger.info(f"load {load_s:.1f}s | warmup ({WARMUP_PRESET}) {warm['wall_s']:.1f}s")

    run_info = {
        "node": socket.gethostname(),
        "tdp": os.environ.get("IDEOGRAM_BATCH_TDP"),
        "tt_metal_rev": _git_rev(),
        "weights": os.environ["IDEOGRAM4_WEIGHTS"],
        "caption_root": str(caption_root),
        "latents": str(latents_path),
        "latents_sha256": latents_sha,
        "height": height,
        "width": width,
        "presets": presets,
        "mode": "traced" if traced else "untraced",
        "load_s": round(load_s, 1),
        "warmup": warm,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=2))

    done = _done(manifest)
    failures = []
    for backend, prompt_id, path in captions:
        text = path.read_text(encoding="utf-8")
        tokens = pipe.count_text_tokens(text)
        for preset in presets:
            if (backend, prompt_id, preset) in done:
                logger.info(f"skip {backend}/{prompt_id} {preset}: already done")
                continue
            image_path = out_dir / "images" / backend / f"{prompt_id}__{preset}.png"
            row = {
                "backend": backend,
                "prompt_id": prompt_id,
                "preset": preset,
                "caption_path": str(path),
                "caption_sha256": _sha256(path),
                "caption_chars": len(text),
                "text_tokens": tokens,
                "height": height,
                "width": width,
                "latents_sha256": latents_sha,
                "image_path": str(image_path),
                "node": run_info["node"],
                "tdp": run_info["tdp"],
                "tt_metal_rev": run_info["tt_metal_rev"],
            }
            try:
                if tokens > MAX_TEXT_TOKENS:
                    raise ValueError(f"{tokens} tokens > {MAX_TEXT_TOKENS}")
                image, timings = generate(text, preset)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(image_path)
                row.update(timings, status="ok")
                logger.info(f"{backend}/{prompt_id} {preset}: {timings['wall_s']:.1f}s ({tokens} tokens)")
            except Exception as error:  # noqa: BLE001 - one bad caption must not drop the rest
                logger.exception(f"{backend}/{prompt_id} {preset} failed")
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
                failures.append(f"{backend}/{prompt_id}/{preset}")
            with manifest.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    assert not failures, f"failed: {failures}"
