# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The FLUX.2 Klein pipeline (``pipelines/flux2_klein/pipeline_flux2_klein.py``) end to end on the Galaxy 4x8 ring:
device text encoder -> traced denoise -> device VAE decode -> PIL, vs the CPU diffusers ``Flux2KleinPipeline`` render of
the same seed (``tests/models/flux3_klein/test_pipeline_flux3_klein.py`` pattern).

Goldens: ``$FLUX2_KLEIN_GOLDENS`` cases written by ``generate_goldens_flux2_klein.py`` (this directory; the layout is in
``common.py``): the diffusers pipeline on a node CPU in bf16 with the per-step latents, the velocities, the recorded
text encodings and ``golden.png``. The recorded text makes a device trajectory bit-comparable to the golden's inputs;
the device-text rows are logged (the text differs by the encoder's own bf16 deviation, so ``golden.png`` is not their
proper reference).

Rows (``layout`` sp8tp4 = SP8 axis 1 x TP4 axis 0, the default, and sp4tp8; ``size`` 4b / 9b; PNGs under
``$FLUX2_KLEIN_OUT``, never inside the tree):
  P0 test_request_resolve (host)   ``Flux2KleinRequest.resolve`` defaults (distilled 4 / None, base 50 / 4.0), the
                          distilled-guidance error, ``policy.admit`` rejections (side, references, stream_rows),
                          ``make_noise`` == diffusers ``randn_tensor`` packed (bit equality) and == every golden's
                          ``noise`` / ``noise_sha256``; ``Flux2KleinScheduler`` sigmas / timesteps == every golden's
                          recorded ``schedule`` / ``t`` (fp32, bit equality).
  P1 test_distilled_render[size, layout]   the distilled 4-step 1024x1024 golden; one resident distilled pipeline:
                          (a) recorded text (``text_override``): one traced ``generate``; PSNR of the device render vs
                          ``golden.png`` >= PSNR_DISTILLED; final-state ``(pcc, rel_rmse)`` inside FINAL_BAND; the
                          pipeline's ``final_tokens`` ``torch.equal`` the denoiser's own
                          ``denoise(request_from_golden)``;
                          the teacher-forced per-step velocity floor TEACHER_FORCED_PCC (golden ``x_t(i)`` written into
                          the state slot, one eager step, velocity vs the golden's; the device state equals the host
                          ``Flux2KleinScheduler.update`` of the device velocity bit for bit: the Euler chain is exact).
                          (b) device text: ``generate`` twice (determinism across replays and encodes), device text vs
                          the recorded text pcc, PSNR vs ``golden.png`` and final pcc LOGGED; (b2) the device
                          handoff (the rows ``ttnn.mesh_partition`` ed to the DiT, never read) vs the host upload of
                          the SAME rows (``text_override`` of its own ``result.text``): ``final_tokens``
                          ``torch.equal``; a ``return_text=False`` request renders the same and carries ``text``
                          None.
                          (c) admission: a 4112-wide request raises ``Flux2KleinAdmissionError`` (budget ``side``)
                          before any device work and leaves the captured trace keys untouched.
  P2 test_base_cfg_smoke[size, layout]   the base 4-step CFG 4.0 1024x1024 golden (a SMOKE schedule: the base model
                          ships 50 steps; 4 keeps the CPU golden at minutes); recorded text and empty-prompt text; one
                          traced ``generate`` with ``steps=4, guidance=4.0``: two samples per step, final band
                          FINAL_BAND["base"], teacher-forced floor, PSNR vs ``golden.png`` logged.

Bands: the FLUX 3 Klein bands at the same stage, applied unmeasured: distilled render vs the CPU golden >= 30.0 dB
(FLUX 3 ``PSNR_INTERIM`` distilled under the approximate-exp ring SDPA) and final state 0.99 / 0.15; base 0.97 / 0.25
with the PSNR logged; teacher-forced per-step velocity pcc 0.995 (the DiT gate). Frozen from the first measured ring
run (rounded down, with the FLUX 3 margins) and never loosened without a measurement.

Environment: ``FLUX2_KLEIN_MODEL_PATH`` / the HuggingFace cache, ``FLUX2_KLEIN_GOLDENS``, ``FLUX2_KLEIN_OUT``,
``TT_DIT_CACHE_DIR``.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn

from ....pipelines.flux2_klein import packing
from ....pipelines.flux2_klein.denoise_flux2_klein import (
    Flux2KleinDenoiser,
    Flux2KleinDenoiseRequest,
    Flux2KleinPreparedRequest,
)
from ....pipelines.flux2_klein.pipeline_flux2_klein import (
    MODELS,
    Flux2KleinAdmissionError,
    Flux2KleinPipeline,
    Flux2KleinRequest,
    Flux2KleinResult,
    make_noise,
)
from ....pipelines.flux2_klein.policy import REFERENCES_REASON, admit
from ....pipelines.flux2_klein.references import (
    ReferenceCache,
    prepare_reference,
    prepare_references,
    reference_key,
    to_pixels,
)
from ....pipelines.flux2_klein.scheduler import Flux2KleinScheduler
from .common import (
    BF16,
    DEFAULT_LAYOUT,
    GOLDENS_ENV,
    LAYOUTS,
    MESH_SHAPE,
    SIZES,
    GoldenCase,
    golden_cases,
    golden_for,
    golden_png,
    load_golden_case,
    measure,
    pcc,
    psnr_uint8,
    renders_dir,
    require_checkpoint_dir,
    sha256_of,
)
from .device_params import ring_params_8k_flux2_klein_transformer

CANVAS = (1024, 1024)
SMOKE_STEPS = 4
# Bands (module docstring; never loosened without a measurement).
PSNR_DISTILLED = 30.0
FINAL_BAND = {"distilled": (0.99, 0.15), "base": (0.97, 0.25)}
TEACHER_FORCED_PCC = 0.995

mesh_rows = pytest.mark.parametrize(
    "mesh_device, device_params",
    [[MESH_SHAPE, ring_params_8k_flux2_klein_transformer]],
    ids=["bh_4x8_ring"],
    indirect=["mesh_device", "device_params"],
)
layout_rows = pytest.mark.parametrize("layout", [DEFAULT_LAYOUT, *(n for n in LAYOUTS if n != DEFAULT_LAYOUT)])
size_rows = pytest.mark.parametrize("size", list(SIZES))


# --- requests from goldens --------------------------------------------------------------------------------------------


def request_from_golden(golden: GoldenCase) -> Flux2KleinDenoiseRequest:
    return Flux2KleinDenoiseRequest(
        noise=golden.inputs["noise"].clone(),
        image_ids=golden.inputs["image_ids"],
        text=golden.text(),
        num_steps=golden.num_steps,
        guidance=golden.guidance,
        text_null=golden.text_null() if golden.model == "base" else None,
    )


def request_for(golden: GoldenCase, *, text_override: bool) -> Flux2KleinRequest:
    width, height = golden.canvas
    override = None
    if text_override:
        override = (golden.text(), golden.text_null() if golden.model == "base" else None)
    return Flux2KleinRequest(
        prompt=golden.prompt,
        model=golden.model,
        size=golden.size,
        width=width,
        height=height,
        seed=golden.seed,
        steps=golden.num_steps,
        guidance=golden.guidance,
        text_override=override,
    )


def build_pipeline(mesh_device, layout: str, size: str, variant: str) -> Flux2KleinPipeline:
    snapshot = require_checkpoint_dir(variant, size)
    sp_axis, tp_axis = LAYOUTS[layout]
    t0 = time.time()
    pipeline = Flux2KleinPipeline(
        mesh_device, snapshots={variant: snapshot}, size=size, resident=(variant,), sp_axis=sp_axis, tp_axis=tp_axis
    )
    logger.info(f"RECORD pipeline init [{variant}/{size}] {layout} {time.time() - t0:.1f}s: {pipeline.timings_init}")
    return pipeline


def mesh_tag(mesh_device, layout: str) -> str:
    shape = tuple(mesh_device.shape)
    return f"{shape[0]}x{shape[1]}_{layout}_ring"


def save_render(result: Flux2KleinResult, tag: str, label: str) -> None:
    out_dir = renders_dir()
    if out_dir is None:
        return
    width, height = result.image.size
    path = out_dir / f"flux2_klein_{tag}_{result.model}_{result.size}_{width}x{height}_seed{result.seed}_{label}.png"
    result.image.save(path)
    logger.info(f"render saved: {path}")


def report(name: str, result: Flux2KleinResult, golden: GoldenCase, png) -> dict[str, float]:
    """PSNR vs golden.png, (pcc, rel_rmse) vs the golden final_tokens, the digest; returns the numbers (the caller
    gates)."""
    psnr = psnr_uint8(result.image, png)
    p, r, m = measure(golden.steps["final_tokens"], result.final_tokens)
    digest = sha256_of(result.final_tokens)
    t = result.timings
    logger.info(
        f"RESULT {name}: PSNR vs golden.png {psnr:.2f} dB; final_tokens vs golden pcc {p:.6f} rel_rmse {r:.4f} max_abs "
        f"{m:.3g}; sha256 {digest[:16]}; image {result.image.size} finite "
        f"{bool(torch.isfinite(result.final_tokens).all())}"
    )
    logger.info(
        f"RECORD {name} walls: encode {t['encode_s']:.3f}s prepare {t['prepare_s']:.3f}s denoise {t['denoise_s']:.3f}s "
        f"({t['denoise_s'] / result.steps:.3f} s/step incl. capture {t['trace_captured']}) decode {t['decode_s']:.3f}s "
        f"(vae capture {t['vae_trace_captured']}) to_host {t['to_host_s']:.3f}s total {t['total_s']:.3f}s"
    )
    return {"psnr": psnr, "pcc": p, "rel_rmse": r, "digest": digest}


def teacher_forced_table(
    name: str, denoiser: Flux2KleinDenoiser, prepared: Flux2KleinPreparedRequest, golden: GoldenCase
) -> float:
    """Per step: the golden ``x_t(i)`` written into the state slot, one EAGER device step, the device velocity vs the
    golden's (pcc / rel_rmse logged); the device state after the step must equal the host ``Flux2KleinScheduler.update``
    of the device velocity bit for bit. Returns the worst per-step velocity pcc."""
    worst = 1.0
    for i in range(golden.num_steps):
        step = prepared.scheduler.step(i)
        x_in = golden.x_t(i)
        denoiser.write_state(prepared, x_in)
        velocity = denoiser.step(prepared, i, traced=False)
        v = denoiser.read_velocity(prepared, velocity)
        ttnn.deallocate(velocity)
        x_out = denoiser.read_state(prepared)
        expected = Flux2KleinScheduler.update(x_in, v, step).float()
        assert torch.equal(x_out, expected), f"step {i}: device state != host update(x, v_device)"
        p, r, _ = measure(golden.velocity(i).float(), v.float())
        px, _, _ = measure(golden.x_t(i + 1), x_out)
        logger.info(
            f"RESULT {name} teacher-forced step {i} (t={step.timestep:.3f}): velocity pcc {p:.6f} rel_rmse {r:.4f}; "
            f"state pcc vs golden x_t({i + 1}) {px:.6f}"
        )
        worst = min(worst, p)
    return worst


# --- P0: host ---------------------------------------------------------------------------------------------------------


def test_request_resolve(expect_error, tmp_path) -> None:
    distilled = Flux2KleinRequest(prompt="x").resolve()
    assert (distilled.model, distilled.steps, distilled.guidance, distilled.cfg) == ("distilled", 4, None, False)
    base = Flux2KleinRequest(prompt="x", model="base").resolve()
    assert (base.steps, base.guidance, base.cfg) == (50, 4.0, True)
    assert Flux2KleinRequest(prompt="x", guidance=1.0).resolve().guidance is None
    assert Flux2KleinRequest(prompt="x", model="base", steps=4, guidance=2.5).resolve().guidance == 2.5
    assert MODELS["base"]["steps"] == 50 and MODELS["distilled"]["guidance"] is None
    with expect_error(ValueError, "distilled"):
        Flux2KleinRequest(prompt="x", guidance=2.0).resolve()
    with expect_error(ValueError, "unknown model"):
        Flux2KleinRequest(prompt="x", model="turbo").resolve()
    with expect_error(ValueError, "unknown size"):
        Flux2KleinRequest(prompt="x", size="2b").resolve()
    with expect_error(ValueError, "output_format"):
        Flux2KleinRequest(prompt="x", output_format="webp").resolve()
    with expect_error(ValueError, "text_override"):
        Flux2KleinRequest(
            prompt="x", text_override=((torch.zeros(512, 7680), packing.text_ids(512)), None), model="base"
        ).resolve()
    zero = Flux2KleinRequest(prompt="x", width=0, height=0).resolve()
    assert (zero.width, zero.height) == CANVAS
    with expect_error(ValueError, "negative"):
        Flux2KleinRequest(prompt="x", width=-16).resolve()
    from PIL import Image

    with expect_error(ValueError, REFERENCES_REASON):
        Flux2KleinRequest(prompt="x", images=(Image.new("RGB", (64, 64)),) * 11).resolve()

    # the prepared references carry their fp32 pixels (what the VAE encoder uploads) and a PATH source is served from
    # the pipeline's ReferenceCache on the second request, bit for bit (the PIL image and the pixels are the cached
    # objects); a PIL source is prepared per request and never cached
    rng = np.random.default_rng(3)
    source = Image.fromarray(rng.integers(0, 256, (80, 96, 3), dtype=np.uint8))
    path = tmp_path / "reference.png"
    source.save(path)
    cache = ReferenceCache(maxsize=2)
    once = Flux2KleinRequest(prompt="x", images=(str(path), source)).resolve(reference_cache=cache)
    assert once.prepared.sizes == ((96, 80), (96, 80)) and (once.width, once.height) == (96, 80)
    assert tuple(once.prepared.pixels[0].shape) == (1, 80, 96, 3) and once.prepared.pixels[0].dtype == torch.float32
    assert torch.equal(once.prepared.pixels[0], to_pixels(prepare_reference(source)))
    assert torch.equal(once.prepared.pixels[1], once.prepared.pixel_tensor(1))
    assert (cache.hits, cache.misses, len(cache)) == (0, 1, 1) and str(path) in cache and source not in cache
    twice = Flux2KleinRequest(prompt="x", images=(str(path), source)).resolve(reference_cache=cache)
    assert (cache.hits, cache.misses, len(cache)) == (1, 1, 1)
    assert twice.prepared.images[0] is once.prepared.images[0] and twice.prepared.pixels[0] is once.prepared.pixels[0]
    assert twice.prepared.images[1] is not once.prepared.images[1]
    assert twice.prepared.images[1].tobytes() == once.prepared.images[1].tobytes()
    plain = Flux2KleinRequest(prompt="x", images=(str(path),)).resolve()
    assert plain.prepared.images[0].tobytes() == once.prepared.images[0].tobytes()
    assert torch.equal(plain.prepared.pixels[0], once.prepared.pixels[0])
    assert reference_key(str(path)) == reference_key(path) and reference_key(source) is None
    assert prepare_references((), width=None, height=None).pixels == ()

    side = admit(Flux2KleinRequest(prompt="x", width=1000).resolve(), MESH_SHAPE)
    assert not side.ok and side.budget == "side", side
    big = admit(Flux2KleinRequest(prompt="x", height=4112).resolve(), MESH_SHAPE)
    assert not big.ok and big.budget == "side", big
    for layout, (sp_axis, _) in LAYOUTS.items():
        sp = MESH_SHAPE[sp_axis]
        ok = admit(Flux2KleinRequest(prompt="x").resolve(), MESH_SHAPE, sp_factor=sp)
        assert ok.ok and ok.measured["image_tokens"] == 4096 and ok.measured["text_length"] == 512, (layout, ok)
        # 320 x 320 = 400 spatial rows pad to 512 on SP 4 and SP 8: the prompt stream's per-device row count.
        alias = admit(Flux2KleinRequest(prompt="x", width=320, height=320).resolve(), MESH_SHAPE, sp_factor=sp)
        assert not alias.ok and alias.budget == "stream_rows", (layout, alias)
    cfg = admit(Flux2KleinRequest(prompt="x", model="base").resolve(), MESH_SHAPE, sp_factor=8)
    assert cfg.ok and cfg.measured["samples"] == 2 and cfg.measured["text_null"] == 512, cfg

    # The noise: packing.make_noise is prepare_latents (P:478-510) with randn_tensor on a CPU generator in bf16.
    h, w = packing.latent_shape(*reversed(CANVAS))
    noise = make_noise(7, h, w)
    assert noise.dtype == torch.float32 and torch.equal(noise, noise.to(BF16).float())
    try:
        from diffusers.utils.torch_utils import randn_tensor
    except ImportError:
        logger.warning("diffusers not importable: make_noise vs randn_tensor not checked")
    else:
        generator = torch.Generator("cpu").manual_seed(7)
        drawn = randn_tensor(
            (1, packing.FLUX2_KLEIN_IN_CHANNELS, h, w), generator=generator, device=torch.device("cpu"), dtype=BF16
        )
        packed = drawn.reshape(1, packing.FLUX2_KLEIN_IN_CHANNELS, h * w).permute(0, 2, 1)[0].float()
        assert torch.equal(noise, packed), "make_noise differs from randn_tensor + _pack_latents"
        logger.info("RESULT make_noise(7, 64, 64) == diffusers randn_tensor packed")

    checked = 0
    for case_dir in golden_cases():
        golden = load_golden_case(case_dir)
        gh, gw = (int(v) for v in golden.replay["latent_hw"])
        own = make_noise(golden.seed, gh, gw)
        assert sha256_of(own) == golden.replay["noise_sha256"], golden.name
        assert torch.equal(golden.inputs["noise"], own) and torch.equal(golden.x_t(0), own), golden.name
        assert torch.equal(golden.inputs["image_ids"], packing.latent_ids(gh, gw)), golden.name
        scheduler = Flux2KleinScheduler(golden.num_steps, gh * gw)
        assert torch.equal(scheduler.sigmas, golden.inputs["schedule"]), f"{golden.name}: schedule"
        assert torch.equal(scheduler.timesteps, golden.steps["t"]), f"{golden.name}: timesteps"
        assert torch.equal(scheduler.sigmas[1:] * 1000, golden.steps["t_next"]), f"{golden.name}: t_next"
        checked += 1
    logger.info(f"RESULT noise / schedule match {checked} golden record(s)")
    if not checked:
        pytest.skip(f"no golden under {GOLDENS_ENV} to compare the noise and the schedule against")


# --- P1: distilled 4-step render --------------------------------------------------------------------------------------


@mesh_rows
@layout_rows
@size_rows
def test_distilled_render(mesh_device, layout: str, size: str) -> None:
    golden = golden_for("distilled", size, canvas=CANVAS, steps=MODELS["distilled"]["steps"])
    png = golden_png(golden)
    pipeline = build_pipeline(mesh_device, layout, size, "distilled")
    tag = mesh_tag(mesh_device, layout)
    name = f"distilled_render[{golden.name}] {size} {tag}"
    try:
        # (a) the recorded text: bit-comparable to the golden's inputs.
        result = pipeline.generate(request_for(golden, text_override=True), traced=True)
        save_render(result, tag, "golden_text")
        numbers = report(f"{name} golden_text", result, golden, png)
        assert result.image.size == golden.canvas == png.size
        assert result.layout == layout and result.admission.measured["samples"] == 1
        assert torch.isfinite(result.final_tokens).all()
        denoiser = pipeline.denoisers["distilled"]
        prepared = denoiser.prepare(request_from_golden(golden))
        own = denoiser.denoise(prepared, traced=True)
        assert torch.equal(
            own, result.final_tokens
        ), "the pipeline's denoise differs from the denoiser's own traced run"
        logger.info(f"RESULT {name}: pipeline final_tokens == denoiser.denoise(request_from_golden)")
        pipeline.release_traces()
        worst = teacher_forced_table(name, denoiser, prepared, golden)
        gate_pcc, gate_rmse = FINAL_BAND["distilled"]
        logger.info(
            f"RESULT {name}: teacher-forced worst velocity pcc {worst:.6f} (floor {TEACHER_FORCED_PCC}); final pcc "
            f"{numbers['pcc']:.6f} rel_rmse {numbers['rel_rmse']:.4f} (band {gate_pcc} / {gate_rmse}); PSNR "
            f"{numbers['psnr']:.2f} dB (gate {PSNR_DISTILLED})"
        )
        # (b) the device text: determinism; the numbers vs the golden are logged (its text is the recorded one).
        request = request_for(golden, text_override=False)
        first = pipeline.generate(request, traced=True)
        save_render(first, tag, "device_text")
        report(f"{name} device_text first", first, golden, png)
        second = pipeline.generate(request, traced=True)
        report(f"{name} device_text second", second, golden, png)
        assert first.text is not None and tuple(first.text[0].shape) == tuple(golden.inputs["text_tokens"].shape)
        assert torch.equal(second.text[0], first.text[0]), "second encode differs from the first"
        assert torch.equal(second.final_tokens, first.final_tokens), "second generate differs from the first"
        assert not second.timings["trace_captured"] and not second.timings["vae_trace_captured"]
        text_pcc = pcc(golden.inputs["text_tokens"].float(), first.text[0].float())
        logger.info(f"RESULT {name}: device text vs the recorded text pcc {text_pcc:.6f}")
        # (b2) the device handoff vs the host upload of the same rows: the SP partition is a per-device slice, so the
        # two inputs to embed_prompt are bit-identical and so are the renders; a request that does not read the text
        # back renders the same and carries text None.
        uploaded = pipeline.generate(replace(request, text_override=(first.text, None)), traced=True)
        assert torch.equal(
            uploaded.final_tokens, first.final_tokens
        ), "the host upload of the device text renders differently from the device handoff"
        unread = pipeline.generate(replace(request, return_text=False), traced=True)
        assert unread.text is None and torch.equal(unread.final_tokens, first.final_tokens)
        logger.info(
            f"RESULT {name}: device handoff == host upload of its own rows (final_tokens equal); return_text=False -> "
            f"text None, same render; encode walls handoff {first.timings['encode_s']:.3f}s / upload "
            f"{uploaded.timings['encode_s']:.3f}s / unread {unread.timings['encode_s']:.3f}s, prepare "
            f"{first.timings['prepare_s']:.3f}s / {uploaded.timings['prepare_s']:.3f}s / {unread.timings['prepare_s']:.3f}s"
        )
        # (c) a refused canvas never reaches the device.
        captured = {v: d.captured_keys for v, d in pipeline.denoisers.items()}
        with pytest.raises(Flux2KleinAdmissionError) as info:
            pipeline.generate(Flux2KleinRequest(prompt=golden.prompt, size=size, width=4112, height=1024, seed=1))
        assert info.value.admission.budget == "side", info.value.admission
        assert {v: d.captured_keys for v, d in pipeline.denoisers.items()} == captured
        logger.info(f"RESULT {name}: 4112-wide request refused on the host ({info.value.admission.reason})")
        # The gates, after every number is on record.
        assert worst >= TEACHER_FORCED_PCC, f"teacher-forced velocity floor {worst:.6f} < {TEACHER_FORCED_PCC}"
        assert numbers["pcc"] >= gate_pcc and numbers["rel_rmse"] <= gate_rmse, numbers
        assert numbers["psnr"] >= PSNR_DISTILLED, f"PSNR vs golden.png {numbers['psnr']:.2f} dB < {PSNR_DISTILLED}"
    finally:
        pipeline.close()
        ttnn.synchronize_device(mesh_device)


# --- P2: base CFG smoke -----------------------------------------------------------------------------------------------


@mesh_rows
@layout_rows
@size_rows
def test_base_cfg_smoke(mesh_device, layout: str, size: str) -> None:
    golden = golden_for("base", size, canvas=CANVAS, steps=SMOKE_STEPS)
    assert golden.guidance == MODELS["base"]["guidance"], golden.replay
    png = golden_png(golden)
    pipeline = build_pipeline(mesh_device, layout, size, "base")
    tag = mesh_tag(mesh_device, layout)
    name = f"base_cfg_smoke[{golden.name}] {size} {tag}"
    try:
        result = pipeline.generate(request_for(golden, text_override=True), traced=True)
        save_render(result, tag, "golden_text")
        numbers = report(f"{name} golden_text", result, golden, png)
        assert result.image.size == golden.canvas == png.size
        assert (result.steps, result.guidance) == (SMOKE_STEPS, golden.guidance)
        assert result.admission.measured["samples"] == 2 and result.layout_key[-1] is True
        assert torch.isfinite(result.final_tokens).all()
        denoiser = pipeline.denoisers["base"]
        prepared = denoiser.prepare(request_from_golden(golden))
        own = denoiser.denoise(prepared, traced=True)
        assert torch.equal(
            own, result.final_tokens
        ), "the pipeline's denoise differs from the denoiser's own traced run"
        pipeline.release_traces()
        worst = teacher_forced_table(name, denoiser, prepared, golden)
        gate_pcc, gate_rmse = FINAL_BAND["base"]
        logger.info(
            f"RESULT {name}: teacher-forced worst velocity pcc {worst:.6f} (floor {TEACHER_FORCED_PCC}); final pcc "
            f"{numbers['pcc']:.6f} rel_rmse {numbers['rel_rmse']:.4f} (band {gate_pcc} / {gate_rmse}); PSNR "
            f"{numbers['psnr']:.2f} dB (logged)"
        )
        assert worst >= TEACHER_FORCED_PCC, f"teacher-forced velocity floor {worst:.6f} < {TEACHER_FORCED_PCC}"
        assert numbers["pcc"] >= gate_pcc and numbers["rel_rmse"] <= gate_rmse, numbers
    finally:
        pipeline.close()
        ttnn.synchronize_device(mesh_device)
