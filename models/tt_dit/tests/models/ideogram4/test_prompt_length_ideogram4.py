# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Ideogram 4: ring-SDPA ``logical_n`` as a device tensor vs an int, across prompt lengths.

The denoiser sequence is [image | real text | pad] with a constant total length per resolution;
only the real length (``logical_n``) varies with the prompt. As an int it is a compile-time
kernel arg (one SDPA compile per distinct prompt length); as a device tensor the kernels compile
once per resolution and read the live length on device.

For each shape (eager, turbo, seed 123), each prompt runs cold then warm, first with the tensor
form, then with the int form. Per call it records wall / denoise / step time and the number of
new program-cache entries; it saves the warm image and compares tensor vs int images per prompt.

Pass criteria (asserted):
  * tensor form: no new program-cache entries DURING THE DENOISE LOOP for any prompt after the
    first at a shape (the encoder's n_text slice is length-keyed by design and excluded)
  * tensor vs int image for the same prompt: PSNR >= IDEOGRAM_PLEN_MIN_PSNR (default 40 dB)
  * an over-length prompt (> MAX_TEXT_TOKENS) raises ValueError
Image quality is gated separately by ideogram_image_check.py in the job script.

Env: IDEOGRAM4_WEIGHTS, IDEOGRAM_PLEN_CAPTIONS (dir with sonnet/ and opus/ caption JSONs),
IDEOGRAM_PLEN_OUT (default ./prompt_length), IDEOGRAM_PLEN_SIZES (default the 4 2K shapes),
IDEOGRAM_PLEN_PRESET (default V4_TURBO_12).
"""

import json
import math
import os
import socket
import time
from pathlib import Path

import numpy as np
import pytest
import ttnn
from loguru import logger
from PIL import Image

from ....pipelines.events import SectionStart
from ....pipelines.ideogram4 import pipeline_ideogram4 as pipe_mod
from ....pipelines.ideogram4.pipeline_ideogram4 import MAX_TEXT_TOKENS, Ideogram4Pipeline
from .test_perf_ideogram4_gpu_bench import _RING_PIPE, SectionTimer, _git_rev

_REQUIRED = ("IDEOGRAM4_WEIGHTS", "IDEOGRAM_PLEN_CAPTIONS")
_NEEDS_INPUTS = pytest.mark.skipif(
    not all(os.environ.get(k) for k in _REQUIRED), reason=f"needs env {', '.join(_REQUIRED)}"
)
SEED = 123
DEFAULT_SIZES = "2048x2048,2048x1152,1152x2048,1360x2048"


def _build_prompts(pipe: Ideogram4Pipeline, cap_dir: Path) -> dict[str, str]:
    """tiny -> shortest real caption -> mid -> longest real caption -> just under the max."""
    tiny = json.dumps({"high_level_description": "A red apple on a wooden table."})
    short = (cap_dir / "opus" / "Q07.json").read_text(encoding="utf-8")
    mid = (cap_dir / "sonnet" / "Q05.json").read_text(encoding="utf-8")
    base = (cap_dir / "sonnet" / "Q03.json").read_text(encoding="utf-8")

    # Near-max: Q03 plus an "additional_details" field filled from the other captions'
    # descriptions, trimmed (binary search on characters) to the largest length <= MAX_TEXT_TOKENS.
    filler_parts = []
    for p in sorted((cap_dir / "sonnet").glob("*.json")) + sorted((cap_dir / "opus").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        filler_parts.append(json.dumps(d.get("compositional_deconstruction", d), ensure_ascii=False))
    filler = " ".join(filler_parts)
    base_obj = json.loads(base)

    def with_filler(n_chars: int) -> str:
        return json.dumps({**base_obj, "additional_details": filler[:n_chars]}, ensure_ascii=False)

    lo, hi = 0, len(filler)
    assert pipe.count_text_tokens(with_filler(hi)) > MAX_TEXT_TOKENS, "filler too short to reach the max"
    while lo < hi:
        mid_c = (lo + hi + 1) // 2
        if pipe.count_text_tokens(with_filler(mid_c)) <= MAX_TEXT_TOKENS:
            lo = mid_c
        else:
            hi = mid_c - 1
    near_max = with_filler(lo)
    over_max = with_filler(min(len(filler), lo + 64))
    return {"tiny": tiny, "short": short, "mid": mid, "base_q03": base, "near_max": near_max, "_over": over_max}


def _psnr(a: Image.Image, b: Image.Image) -> tuple[float, int]:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mse = float(((x - y) ** 2).mean())
    max_diff = int(np.abs(x - y).max())
    return (math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)), max_diff


@_NEEDS_INPUTS
@pytest.mark.parametrize(
    ("mesh_device", "device_params"), [((4, 8), _RING_PIPE)], indirect=True, ids=["bh_galaxy_sp8tp4"]
)
def test_prompt_length_logical_n(*, mesh_device) -> None:
    cap_dir = Path(os.environ["IDEOGRAM_PLEN_CAPTIONS"])
    out = Path(os.environ.get("IDEOGRAM_PLEN_OUT", "prompt_length"))
    (out / "images").mkdir(parents=True, exist_ok=True)
    preset = os.environ.get("IDEOGRAM_PLEN_PRESET", "V4_TURBO_12")
    min_psnr = float(os.environ.get("IDEOGRAM_PLEN_MIN_PSNR", "40"))
    sizes = [
        tuple(int(v) for v in s.split("x")) for s in os.environ.get("IDEOGRAM_PLEN_SIZES", DEFAULT_SIZES).split(",")
    ]

    t0 = time.perf_counter()
    pipe = Ideogram4Pipeline.create_pipeline(mesh_device=mesh_device, height=sizes[0][1], width=sizes[0][0])
    load_s = time.perf_counter() - t0
    dev = pipe.mesh_device
    assert pipe.sp_factor > 1, "test needs the ring-SDPA (SP>1) path"

    prompts = _build_prompts(pipe, cap_dir)
    over = prompts.pop("_over")
    tokens = {k: pipe.count_text_tokens(v) for k, v in prompts.items()}
    info = {
        "node": socket.gethostname(),
        "tt_metal_rev": _git_rev(),
        "preset": preset,
        "seed": SEED,
        "sizes": [f"{w}x{h}" for w, h in sizes],
        "text_tokens": tokens,
        "load_s": round(load_s, 1),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "run.json").write_text(json.dumps(info, indent=2))
    for k, v in prompts.items():
        (out / f"prompt_{k}.txt").write_text(v, encoding="utf-8")
    logger.info(f"PLEN load {load_s:.1f}s tokens={tokens}")

    # Over-length guard: host-only, raises before any device work.
    n_over = pipe.count_text_tokens(over)
    assert n_over > MAX_TEXT_TOKENS, n_over
    with pytest.raises(ValueError):
        pipe(prompts=[over], preset=preset, seed=SEED, height=sizes[0][1], width=sizes[0][0], traced=False)
    logger.info(f"PLEN over-length prompt ({n_over} tokens) raised ValueError as expected")

    def generate(text: str, width: int, height: int):
        timer = SectionTimer(dev, sync_steps=False)
        # Program-cache count at the start of the denoise loop and at the start of decode, so the
        # assert covers only programs the denoiser (where logical_n lives) creates. The text
        # encoder legitimately adds one prompt-length-keyed program (its on-device slice of the
        # taps to n_text) that is resolution-independent and costs no visible time.
        marks: dict[str, int] = {}

        def on_event(event) -> None:
            timer(event)
            if isinstance(event, SectionStart) and event.name in ("denoising", "vae"):
                marks[event.name] = dev.num_program_cache_entries()

        pc0 = dev.num_program_cache_entries()
        ttnn.synchronize_device(dev)
        start = time.perf_counter()
        image = pipe(
            prompts=[text], preset=preset, seed=SEED, height=height, width=width, traced=False, on_event=on_event
        )[0]
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - start
        steps = len(timer.step_ms())
        denoise = timer.starts["vae"] - timer.starts["denoising"]
        return image, {
            "wall_s": round(wall, 3),
            "encoder_ms": round(timer.durations["encoder"] * 1000, 1),
            "denoise_s": round(denoise, 3),
            "step_ms": round(denoise * 1000 / steps, 1),
            "decoder_ms": round(timer.durations["vae"] * 1000, 1),
            "new_programs": dev.num_program_cache_entries() - pc0,
            "new_denoise_programs": marks["vae"] - marks["denoising"],
        }

    failures = []
    images: dict[tuple, Image.Image] = {}
    for width, height in sizes:
        shape = f"{width}x{height}"
        for mode in ("tensor", "int"):
            pipe_mod._LOGICAL_N_TENSOR = mode == "tensor"
            for idx, (pname, text) in enumerate(prompts.items()):
                for phase in ("cold", "warm"):
                    row = {"shape": shape, "mode": mode, "prompt": pname, "tokens": tokens[pname], "phase": phase}
                    try:
                        image, stats = generate(text, width, height)
                    except Exception as e:  # noqa: BLE001 - keep going, report at the end
                        logger.exception(f"PLEN {shape} {mode} {pname} {phase} failed")
                        failures.append(f"{shape} {mode} {pname} {phase}: {type(e).__name__}: {e}")
                        row["error"] = f"{type(e).__name__}: {e}"
                        stats = None
                    if stats:
                        row.update(stats)
                        if phase == "warm":
                            images[(shape, mode, pname)] = image
                            image.save(out / "images" / f"{shape}__{pname}__{mode}.png")
                        # Tensor form: only the very first call at a shape may compile in the denoiser.
                        if mode == "tensor" and not (idx == 0 and phase == "cold") and stats["new_denoise_programs"]:
                            failures.append(
                                f"{shape} tensor {pname} {phase}: {stats['new_denoise_programs']} new denoise programs"
                            )
                    logger.info(f"PLEN {json.dumps(row)}")
                    with (out / "runs.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")

        for pname in prompts:
            a, b = images.get((shape, "tensor", pname)), images.get((shape, "int", pname))
            if a is None or b is None:
                continue
            psnr, max_diff = _psnr(a, b)
            rec = {
                "shape": shape,
                "prompt": pname,
                "tokens": tokens[pname],
                "psnr_tensor_vs_int": psnr if math.isfinite(psnr) else "inf",
                "max_abs_diff": max_diff,
            }
            logger.info(f"PLEN-CMP {json.dumps(rec)}")
            with (out / "compare.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if psnr < min_psnr:
                failures.append(f"{shape} {pname}: tensor vs int PSNR {psnr:.1f} dB < {min_psnr}")

    pipe_mod._LOGICAL_N_TENSOR = True
    assert not failures, "\n".join(failures)
