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

from ....pipelines.minimax_h3 import image_host_minimax_h3 as host
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
    compiled = {}
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
        if misses:
            compiled[(width, height)] = result.timings.program_misses
    # Asserted after the loop, so every size is logged before a failure.
    assert not compiled, f"JIT compiles on the serving path (size -> misses by stage): {compiled}"


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


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_device_latents_ab(mesh_device, guide_examples):
    """Host Euler update (reference) vs latents kept on device: warm latency of each, and how far the images differ."""
    import numpy as np

    t2i, _, _ = guide_examples
    prompt = t2i["prompt"]
    pipe = MiniMaxH3ImagePipeline(mesh_device=mesh_device, weights_dir=_weights_dir())
    for size in (4096, 2048):
        runs = {}
        for mode in ("0", "1"):
            pipe.device_latents = mode == "1"
            pipe(prompt, width=size, height=size, seed=0)  # warm this mode at this size
            runs[mode] = [pipe(prompt, width=size, height=size, seed=s) for s in (1, 2, 3)]
            totals = [r.timings.total for r in runs[mode]]
            _log_result(f"device_latents={mode}_{size}", runs[mode][-1])
            logger.info(f"RESULT device_latents={mode} {size}: mean {sum(totals) / 3:.2f}s (min {min(totals):.2f})")
        for a, b in zip(runs["0"], runs["1"]):
            x, y = np.asarray(a.image, dtype=np.float64), np.asarray(b.image, dtype=np.float64)
            mse = ((x - y) ** 2).mean()
            psnr = float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)
            logger.info(f"RESULT {size} host vs device latents: max abs {np.abs(x - y).max():.0f}, PSNR {psnr:.1f} dB")
            assert np.array_equal(x, y), "device latents must reproduce the host Euler update exactly"


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_zero_jit_across_prompts(mesh_device, guide_examples):
    """After `warmup()` (what the server runs), vary the prompt length: no stage may compile a program for a prompt or
    presentation length it has not seen."""
    t2i, edit_prompt, reference = guide_examples
    pipe = MiniMaxH3ImagePipeline(mesh_device=mesh_device, weights_dir=_weights_dir())
    logger.info(f"RESULT warmup: {pipe.warmup():.1f}s, presentation ladder {pipe.presentation_ladder}")
    total_misses = 0
    prompts = {
        "short": "A red apple on a white table.",
        "medium": t2i["prompt"],
        "long": edit_prompt,
        "very_long": edit_prompt + " " + t2i["prompt"] + " " + edit_prompt,
    }
    for name, prompt in prompts.items():
        for task, images in (("t2i", ()), ("edit", (reference,))):
            result = pipe(prompt, width=2048, height=2048, seed=1, images=images)
            misses = result.timings.program_misses
            logger.info(
                f"RESULT zero_jit_prompt {task} {name} ({len(prompt)} chars): misses {sum(misses.values())} {misses} "
                f"| prompt stage {result.timings.encode_prompt:.2f}s"
            )
            total_misses += sum(misses.values())
    assert total_misses == 0, f"{total_misses} programs compiled on the serving path after warmup()"


# The served envelope (bring-up record, Stage 2): five ratios and their portrait flips at ~16 MP and ~4 MP (2K), and
# 1 to 16 edit references, in order <Picture 1>..<Picture N>.
SERVED_SIZES_16MP = ((4096, 4096), (4736, 3552), (5024, 3360), (5472, 3072), (6272, 2688))
SERVED_SIZES_2K = ((2048, 2048), (2368, 1760), (2496, 1664), (2720, 1536), (3136, 1344))
SERVED_SIZES = tuple(
    dict.fromkeys(s for w, h in SERVED_SIZES_16MP + SERVED_SIZES_2K for s in ((w, h), (h, w)))
)  # 18 sizes: 1:1 has no flip


@pytest.mark.timeout(4 * 3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_served_envelope(mesh_device, guide_examples):
    """Every served size (t2i) and 1..16 edit references at two reference sizes, after `warmup()`: pass or the error,
    latency, and programs compiled. Runs every case even when one fails, then fails if any did."""
    t2i, _, _ = guide_examples
    pipe = MiniMaxH3ImagePipeline(mesh_device=mesh_device, weights_dir=_weights_dir())
    logger.info(f"RESULT warmup {pipe.warmup():.1f}s")
    failures = []

    def run(tag, **kwargs):
        images = kwargs.get("images") or []
        over = bool(images) and host.effective_reference_area([im.size for im in images]) > pipe.reference_area_limit
        try:
            result = pipe(**kwargs)
        except host.ReferenceLimitError as err:
            if not over:
                failures.append(tag)
            logger.info(f"RESULT envelope {tag}: {'rejected as expected' if over else 'FAIL rejected'}: {err}")
            return
        except Exception as err:  # keep going: the matrix is the deliverable
            failures.append(tag)
            logger.info(f"RESULT envelope {tag}: FAIL {type(err).__name__}: {str(err)[:240]}")
            return
        if over:
            failures.append(tag)
            logger.info(f"RESULT envelope {tag}: FAIL admitted over the reference limit")
        t = result.timings
        logger.info(
            f"RESULT envelope {tag}: ok {result.image.size[0]}x{result.image.size[1]} packed {result.packed_length} | "
            f"{t.total:.2f}s (prompt {t.encode_prompt:.2f}, refs {t.encode_references:.2f}, denoise {t.denoise:.2f}, "
            f"decode {t.decode:.2f}) | misses {sum(t.program_misses.values())} "
            f"{ {k: v for k, v in t.program_misses.items() if v} }"
        )

    for w, h in SERVED_SIZES:
        run(f"t2i {w}x{h}", prompt=t2i["prompt"], width=w, height=h, seed=1)
    for ref_w, ref_h, counts in (
        (1024, 1024, (1, 2, 4, 8, 16)),
        (3000, 2000, (1, 2, 4, 8, 16)),
        (2048, 2048, (8,)),
        (8192, 2048, (2, 16)),  # MiniMax's 8192-px side at 4:1: 16.8 MP effective each; sixteen are rejected
    ):
        for count in counts:
            refs = [Image.new("RGB", (ref_w, ref_h), (40 + 13 * i, 90, 160 - 7 * i)) for i in range(count)]
            labels = " ".join(f"<Picture {i + 1}>" for i in range(count))
            run(
                f"edit {count} refs {ref_w}x{ref_h} -> 4096x4096",
                prompt=f"Combine the subjects of {labels} into one photorealistic scene.",
                width=4096,
                height=4096,
                seed=1,
                images=refs,
            )
    refs = [Image.new("RGB", (1600, 1200), (90, 120, 60)), Image.new("RGB", (1200, 1600), (60, 90, 120))]
    run("edit 2 refs -> 2720x1536", prompt="Put <Picture 1> beside <Picture 2>.", width=2720, height=1536, seed=1,
        images=refs)
    assert not failures, f"{len(failures)} envelope cases failed: {failures}"


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("capacity", [32768, 49152, 82688], ids=["cap32k", "cap48k", "cap81k"])
def test_capacity_cost(mesh_device, guide_examples, capacity):
    """Warm t2i step time against the DiT's max_sequence_length (it sizes the norms' worker split and the K/V
    gather): what a larger multi-reference capacity costs requests that do not need it."""
    t2i, _, _ = guide_examples
    pipe = MiniMaxH3ImagePipeline(mesh_device=mesh_device, weights_dir=_weights_dir(), max_sequence_length=capacity)
    for size in (4096, 2048):
        pipe(t2i["prompt"], width=size, height=size, seed=0)
        results = [pipe(t2i["prompt"], width=size, height=size, seed=s) for s in (1, 2)]
        steps = [s for r in results for s in r.timings.denoise_steps[2:]]
        logger.info(
            f"RESULT capacity {capacity} {size}: total {sum(r.timings.total for r in results) / 2:.2f}s, "
            f"steady step {1e3 * sum(steps) / len(steps):.1f} ms"
        )


def _tall_mix(limit: int) -> list[tuple[int, int]]:
    """The densest request under `limit`: 4:1 references (each carries the most vision tokens, 4096, and one VAE row
    per 1024 px). Up to sixteen 1024x4096; past sixteen, the area left grows references toward 2048x8192 (more rows,
    the same vision tokens); under sixteen, one 1024-wide reference takes what is left."""
    tall = 1024 * 4096
    sizes = [(1024, 4096)] * min(host.MAX_REFERENCES, limit // tall)
    rest = limit - tall * len(sizes)
    if len(sizes) < host.MAX_REFERENCES:
        width = rest // 1024 // 32 * 32
        return sizes + [(1024, width)] if width >= 1024 else sizes
    for i in range(len(sizes)):
        side = min(2048, int(((tall + rest) / 4) ** 0.5) // 32 * 32)
        rest -= side * 4 * side - tall
        sizes[i] = (side, 4 * side)
        if side < 2048:
            break
    return sizes


@pytest.mark.timeout(4 * 3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("limit_mp", [32, 48, 64, 96], ids=["32mp", "48mp", "64mp", "96mp"])
def test_reference_area_limit(mesh_device, guide_examples, limit_mp):
    """One candidate per-request reference limit: build the pipeline sized for it, warm it, then run the worst
    requests it admits at 4096^2 (densest mix, square 2048^2 copies, 16 references, a 4096-token prompt) and one just
    over it. Reports warmup, t2i latency (does the capacity cost requests that do not need it?), each edit's latency
    and program misses; fails on any miss, any admitted request that fails, or an over-limit request admitted."""
    t2i, _, _ = guide_examples
    limit = limit_mp * 1_000_000
    pipe = MiniMaxH3ImagePipeline(mesh_device=mesh_device, weights_dir=_weights_dir(), reference_area_limit=limit)
    c = pipe.arena_caps
    logger.info(
        f"RESULT limit {limit_mp}MP caps: prompt {c.prompt}, cond rows {c.condition_video_rows}, vision patches "
        f"{pipe.vision_patch_cap}, max_sequence_length {pipe.max_sequence_length}"
    )
    logger.info(f"RESULT limit {limit_mp}MP warmup {pipe.warmup():.1f}s")
    failures = []
    filler = " ".join(["Keep every subject's identity, pose, lighting and colors exactly as in its reference."] * 400)

    def run(tag, sizes, prompt_text, width=4096, height=4096):
        refs = [Image.new("RGB", size, (40 + 9 * i, 90, 200 - 9 * i)) for i, size in enumerate(sizes)]
        labels = " ".join(f"<Picture {i + 1}>" for i in range(len(refs)))
        area = host.effective_reference_area(sizes) if sizes else 0
        try:
            result = pipe(f"Combine {labels} into one scene. {prompt_text}".strip(), width=width, height=height,
                          seed=3, images=refs or None)
        except host.ReferenceLimitError as err:
            ok = area > limit
            if not ok:
                failures.append(tag)
            logger.info(f"RESULT limit {limit_mp}MP {tag}: {'rejected as expected' if ok else 'FAIL rejected'}: {err}")
            return
        except Exception as err:
            failures.append(tag)
            logger.info(f"RESULT limit {limit_mp}MP {tag}: FAIL {type(err).__name__}: {str(err)[:300]}")
            return
        t = result.timings
        misses = sum(t.program_misses.values())
        if misses or area > limit:
            failures.append(tag)
        logger.info(
            f"RESULT limit {limit_mp}MP {tag}: {len(sizes)} refs {area / 1e6:.1f} MP -> packed {result.packed_length} | "
            f"{t.total:.2f}s (prompt {t.encode_prompt:.2f}, refs {t.encode_references:.2f}, denoise {t.denoise:.2f}, "
            f"decode {t.decode:.2f}) | misses {misses} { {k: v for k, v in t.program_misses.items() if v} }"
        )

    run("t2i 4096^2", [], "")
    run("t2i 4096^2 again", [], "")
    tall = _tall_mix(limit)
    run("densest mix", tall, filler)
    run("densest mix, short prompt", tall, "")
    run("2048^2 copies", [(2048, 2048)] * min(16, limit // 2048**2), filler)
    run("16 x 1024^2", [(1024, 1024)] * 16, filler)
    run("phone photos 4032x3024", [(4032, 3024)] * min(16, limit // (2731 * 2048)), "")
    run("densest mix to 2K", tall, "", width=2048, height=2048)
    over = tall + [(1024, 1024)] if len(tall) < 16 else tall[:-1] + [(2048, 8192)]
    run("just over the limit", over, "")
    assert not failures, f"{len(failures)} cases failed: {failures}"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_host_time_4k(mesh_device, guide_examples):
    """Where a warm 4K request spends host-only time (the device idle while it runs): the work a pipelined server
    could overlap with the next request's device work. Wraps the host steps and the VAE's device decode with timers;
    reports per request, never asserts."""
    import time as _time
    from collections import defaultdict

    from ....pipelines.minimax_h3 import pipeline_minimax_h3_image as pmod

    t2i, edit_prompt, ref = guide_examples
    pipe = MiniMaxH3ImagePipeline(mesh_device=mesh_device, weights_dir=_weights_dir())
    spent = defaultdict(float)

    def timed(owner, name, label, materialize=False):
        fn = getattr(owner, name)

        def wrapper(*args, **kwargs):
            mark = _time.perf_counter()
            out = fn(*args, **kwargs)
            if materialize:
                out = list(out)
            spent[label] += _time.perf_counter() - mark
            return out

        setattr(owner, name, wrapper)

    for name in ("load_reference_image", "bound_short_edge", "draw_request_noise", "vae_reference_pixels",
                 "reference_rows", "anchor_rows", "build_image_packed_sequence", "patchify", "unpatchify",
                 "finish_image"):
        timed(pmod.host, name, f"host {name}")
    timed(pipe, "_build_presentation", "host tokenize + image processor")
    pipe("warm", width=4096, height=4096, seed=0)
    pipe("warm", width=4096, height=4096, seed=0, images=[ref])
    timed(pipe._vae, "_stream_decoder_units", "vae device decode + readback", materialize=True)
    timed(pipe._vae, "_stitch_decoded", "host vae tile stitch")
    for tag, kwargs in (("t2i 4K", dict(prompt=t2i["prompt"], width=4096, height=4096, seed=1)),
                        ("edit 4K 1 ref", dict(prompt=edit_prompt, width=4096, height=4096, seed=1, images=[ref]))):
        spent.clear()
        result = pipe(**kwargs)
        t = result.timings
        post = t.decode - spent["vae device decode + readback"] - spent["host vae tile stitch"] - spent[
            "host finish_image"] - spent["host unpatchify"]
        host_total = sum(v for k, v in spent.items() if k.startswith("host")) + post
        lines = [f"{k}: {v * 1e3:.0f} ms" for k, v in sorted(spent.items(), key=lambda kv: -kv[1])]
        lines.append(f"host un-normalize (mul/add/clamp): {post * 1e3:.0f} ms")
        logger.info(f"RESULT host_time {tag}: total {t.total:.2f}s, host-only {host_total * 1e3:.0f} ms | " + "; ".join(lines))


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_defer_finish_overlap(mesh_device, guide_examples):
    """`defer_finish` as a server uses it: the host tail (stitch, un-normalize, uint8) of request N runs on a thread
    while request N+1 runs on the device. Asserts the deferred image is byte-identical to the normal path; reports
    back-to-back 4K throughput serial vs overlapped, and whether the overlapped tail slows the next request."""
    import threading

    t2i, edit_prompt, ref = guide_examples
    pipe = MiniMaxH3ImagePipeline(mesh_device=mesh_device, weights_dir=_weights_dir())
    pipe.warmup()
    requests = [dict(prompt=t2i["prompt"], width=4096, height=4096, seed=s) for s in range(4)]
    requests += [dict(prompt=edit_prompt, width=4096, height=4096, seed=s, images=[ref]) for s in range(2)]

    expected = pipe(**requests[0]).image
    deferred = pipe(**requests[0], defer_finish=True)
    assert deferred.image is None and deferred.finish is not None
    assert deferred.finish().tobytes() == expected.tobytes(), "deferred finish changed the image"

    started = time.perf_counter()
    serial = [pipe(**r) for r in requests]
    serial_s = time.perf_counter() - started

    started = time.perf_counter()
    totals, finishes, pending = [], [], None
    for r in requests:
        result = pipe(**r, defer_finish=True)
        totals.append(result.timings.total)
        if pending is not None:
            pending.join()
        pending = threading.Thread(target=lambda res=result: finishes.append((res.finish(), res.timings.finish)))
        pending.start()
    pending.join()
    overlapped_s = time.perf_counter() - started

    for a, (b, _) in zip(serial, finishes):
        assert a.image.tobytes() == b.tobytes()
    serial_totals = ", ".join(f"{r.timings.total:.2f}" for r in serial)
    logger.info(
        f"RESULT defer_finish: {len(requests)} requests (4 t2i + 2 edit, 4K) serial {serial_s:.2f}s vs overlapped "
        f"{overlapped_s:.2f}s ({100 * (1 - overlapped_s / serial_s):.1f}% less) | device-path totals serial "
        f"[{serial_totals}] vs deferred [{', '.join(f'{t:.2f}' for t in totals)}] | finish each "
        f"[{', '.join(f'{f:.2f}' for _, f in finishes)}]"
    )


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_prompt_encode_scaling(mesh_device, guide_examples):
    """Warm prompt-encode time at the presentation rungs (text only): flat across rungs means per-op dispatch, not
    compute or weight bandwidth, dominates -- the case for tracing the encoder."""
    import ttnn as _ttnn

    t2i, _, _ = guide_examples
    pipe = MiniMaxH3ImagePipeline(mesh_device=mesh_device, weights_dir=_weights_dir())
    for rung in (1024, 2048, 4096, 8192):
        pipe._force_prompt_pad = rung
        try:
            times = []
            for _ in range(4):
                mark = time.perf_counter()
                embeds, _ = pipe.encode_prompt(t2i["prompt"])
                _ttnn.synchronize_device(mesh_device)
                times.append(time.perf_counter() - mark)
                _ttnn.deallocate(embeds)
        finally:
            pipe._force_prompt_pad = None
        warm = sorted(times[1:])
        logger.info(f"RESULT prompt_encode rung {rung}: warm min {warm[0] * 1e3:.0f} ms, median {warm[1] * 1e3:.0f} ms")
