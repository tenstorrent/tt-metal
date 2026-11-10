# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3-Image end to end on a Blackhole Galaxy: the release guide's two examples, and the latency matrix.

`test_guide_examples` runs the two requests from the release's usage guide exactly as specified there (prompt, size,
seed, generator_device=cpu) and saves the PNGs for comparison with the guide's published outputs:
  t2i : examples/request.t2i.json            -> 4736x3552, seed 42
  edit: examples/reference1.png + prompt.edit.txt -> 4096x4096, seed 1234
`test_latency` warms each (task, size) once, then times a second request: 2K (2048^2) and 4K (4096^2), t2i and edit.

Env: MINIMAX_H3_IMAGE_MODEL_PATH (full converted snapshot, with text_encoder/), H3_IMAGE_OUT (output dir).
"""

import json
import os
import time

import pytest
from loguru import logger
from PIL import Image

from ....pipelines.minimax_h3.pipeline_minimax_h3_image import MiniMaxH3ImagePipeline
from .common import MESH_4X8_RING


def _weights_dir() -> str:
    path = os.environ.get("MINIMAX_H3_IMAGE_MODEL_PATH")
    if not path:
        pytest.skip("set MINIMAX_H3_IMAGE_MODEL_PATH to the converted H3-Image snapshot")
    return path


def _out_dir() -> str:
    path = os.environ.get("H3_IMAGE_OUT", os.path.expanduser("~/h3_image_out"))
    os.makedirs(path, exist_ok=True)
    return path


def _log_result(tag: str, result) -> None:
    t = result.timings
    steps = ", ".join(f"{s:.2f}" for s in t.denoise_steps)
    logger.info(
        f"RESULT {tag}: canvas {result.canvas[0]}x{result.canvas[1]}, packed {result.packed_length} -> rung "
        f"{result.rung} | TIMING total {t.total:.2f}s = prompt {t.encode_prompt:.2f} + refs {t.encode_references:.2f} "
        f"+ denoise {t.denoise:.2f} [{steps}] + decode {t.decode:.2f}"
    )


@pytest.fixture(scope="module")
def guide_examples():
    examples = os.path.join(_weights_dir(), "examples")
    t2i = json.load(open(os.path.join(examples, "request.t2i.json")))
    edit_prompt = open(os.path.join(examples, "prompt.edit.txt")).read()  # verbatim (the file has no trailing newline)
    # The public release names the reference reference1.png; the pre-release snapshot called it picture1.png.
    name = next(n for n in ("reference1.png", "picture1.png") if os.path.exists(os.path.join(examples, n)))
    return t2i, edit_prompt, Image.open(os.path.join(examples, name))


@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_guide_examples(mesh_device, guide_examples):
    t2i, edit_prompt, reference = guide_examples
    pipe = MiniMaxH3ImagePipeline(mesh_device=mesh_device, weights_dir=_weights_dir())
    out = _out_dir()

    width, height = (int(v) for v in t2i["size"].split("x"))
    result = pipe(t2i["prompt"], width=width, height=height, seed=int(t2i["seed"]))
    result.image.save(os.path.join(out, "t2i.png"))
    _log_result("guide_t2i", result)
    logger.info(f"SAVED {os.path.join(out, 't2i.png')}")

    result = pipe(edit_prompt, width=4096, height=4096, seed=1234, images=[reference])
    result.image.save(os.path.join(out, "edit.png"))
    _log_result("guide_edit", result)
    logger.info(f"SAVED {os.path.join(out, 'edit.png')}")


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_latency_2k_edit_claim(mesh_device, guide_examples):
    """The case a "2K edit" claim must hold for: a 2048x2048 reference in, 2048x2048 out (the VAE keeps up to a 2048
    short edge, so this is the largest single-reference anchor block: 4096 rows), with the guide's edit prompt and with
    a ~4.5k-character prompt (the guide's evaluation prompts run 1100-4600 characters). Warm, then 3 timed requests."""
    _, edit_prompt, reference = guide_examples
    pipe = MiniMaxH3ImagePipeline(mesh_device=mesh_device, weights_dir=_weights_dir())
    ref_2k = reference.convert("RGB").resize((2048, 2048), Image.Resampling.LANCZOS)  # content is irrelevant to timing
    long_prompt = (edit_prompt + " ") * 3
    long_prompt = long_prompt[:4500]
    for tag, prompt in (("guide_prompt", edit_prompt), ("long_prompt_4500ch", long_prompt)):
        mark = time.perf_counter()
        pipe(prompt, width=2048, height=2048, seed=7, images=[ref_2k])
        logger.info(f"WARM edit_2K_ref2K_{tag}: {time.perf_counter() - mark:.1f}s")
        totals = []
        for seed in (8, 9, 10):
            result = pipe(prompt, width=2048, height=2048, seed=seed, images=[ref_2k])
            _log_result(f"latency_edit_2K_ref2K_{tag}", result)
            totals.append(result.timings.total)
        logger.info(
            f"RESULT edit_2K_ref2K_{tag}: mean {sum(totals) / 3:.2f}s over 3 (min {min(totals):.2f} max {max(totals):.2f})"
        )


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_zero_jit_across_lengths(mesh_device, guide_examples):
    """No JIT on the serving path: warm ONE size (2048^2 t2i), then requests whose padded packed lengths all differ
    from it (and from each other), inside the same length class. Every one must add zero programs."""
    t2i, _, _ = guide_examples
    pipe = MiniMaxH3ImagePipeline(mesh_device=mesh_device, weights_dir=_weights_dir())
    warm = pipe(t2i["prompt"], width=2048, height=2048, seed=1)
    seen = {warm.rung}
    logger.info(f"RESULT zero_jit warm 2048x2048: padded {warm.rung} (dynamic_length={pipe.dynamic_length})")
    for width, height in ((2560, 2048), (3072, 2048), (2048, 2560), (2304, 2304)):
        before = mesh_device.num_program_cache_entries()
        result = pipe(t2i["prompt"], width=width, height=height, seed=2)
        misses = mesh_device.num_program_cache_entries() - before
        new_length = result.rung not in seen
        seen.add(result.rung)
        logger.info(
            f"RESULT zero_jit {width}x{height}: packed {result.packed_length} -> padded {result.rung} "
            f"(new length: {new_length}) | misses {misses} by stage {result.timings.program_misses} | "
            f"total {result.timings.total:.2f}s"
        )


@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_latency(mesh_device, guide_examples):
    t2i, edit_prompt, reference = guide_examples
    pipe = MiniMaxH3ImagePipeline(mesh_device=mesh_device, weights_dir=_weights_dir())
    cases = [
        ("t2i_2K", t2i["prompt"], 2048, ()),
        ("t2i_4K", t2i["prompt"], 4096, ()),
        ("edit_2K", edit_prompt, 2048, (reference,)),
        ("edit_4K", edit_prompt, 4096, (reference,)),
    ]
    for tag, prompt, size, images in cases:
        mark = time.perf_counter()
        pipe(prompt, width=size, height=size, seed=7, images=images)  # warm: compiles this shape's programs
        logger.info(f"WARM {tag}: {time.perf_counter() - mark:.1f}s")
        before = mesh_device.num_program_cache_entries()
        result = pipe(prompt, width=size, height=size, seed=8, images=images)
        _log_result(f"latency_{tag}", result)
        logger.info(
            f"RESULT latency_{tag}: program cache misses on the measured request = "
            f"{mesh_device.num_program_cache_entries() - before}"
        )

    # No JIT on the serving path: after the 2K warm-up, a different 2K-class size (packed length differs, same length
    # class on the dynamic path; a different rung on the bucketed one) must compile nothing new.
    before = mesh_device.num_program_cache_entries()
    result = pipe(t2i["prompt"], width=2304, height=1792, seed=9)
    _log_result("latency_t2i_2304x1792_unwarmed_size", result)
    misses = mesh_device.num_program_cache_entries() - before
    logger.info(f"RESULT unwarmed-size program cache misses = {misses} (dynamic_length={pipe.dynamic_length})")


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("precompute_adaln", [True, False], ids=["adaln_precomputed", "adaln_per_step"])
def test_step_breakdown_4k(mesh_device, guide_examples, precompute_adaln):
    """Where a warm 4K t2i denoise step goes: host enqueue vs device wait vs readback per step (STEP_TIMING lines).

    Under `scripts/run_safe_pytest.sh --profile` with H3_IMAGE_PROFILE_STEP=k, step k of the timed request is
    bracketed by Tracy signposts, so the per-op CSV covers exactly one warm 4K forward.
    """
    t2i, _, _ = guide_examples
    pipe = MiniMaxH3ImagePipeline(
        mesh_device=mesh_device, weights_dir=_weights_dir(), precompute_adaln=precompute_adaln
    )
    pipe(t2i["prompt"], width=4096, height=4096, seed=0)  # warm: compiles, and precomputes the adaLN tables
    profile_step = os.environ.pop("H3_IMAGE_PROFILE_STEP", None)  # only the timed request is profiled
    pipe(t2i["prompt"], width=4096, height=4096, seed=1)
    os.environ["H3_IMAGE_STEP_TIMING"] = "1"
    if profile_step is not None:
        os.environ["H3_IMAGE_PROFILE_STEP"] = profile_step
    try:
        result = pipe(t2i["prompt"], width=4096, height=4096, seed=2)
    finally:
        os.environ.pop("H3_IMAGE_STEP_TIMING", None)
    _log_result(f"step_breakdown_4k_{'precomputed' if precompute_adaln else 'per_step'}", result)
