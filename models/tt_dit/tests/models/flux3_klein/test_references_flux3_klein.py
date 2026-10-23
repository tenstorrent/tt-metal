# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""M8: the FLUX 3 Image Klein reference sizing (``pipelines/flux3_klein/references.py``) bit-exact against the vendored
reference (``reference/flux3_klein/pipeline.py:147-206``, ``generate.py:65-77``), the admission policy with references
and the golden records' reference rows. Host only, no device. Design record ``docs/M8_DESIGN.md`` sections 3, 6, 9.1.

Rows:
  R1  test_sizing_bit_exact[input]      the three example inputs and the synthetic sizes of design 2.1 (+ (10, 10) and
                                        (80, 80)): ``reference_canvas``, ``prepare_reference`` (pixels + crop box),
                                        ``prepare_multi_reference``, ``default_size``, ``to_tensor`` vs the vendored
                                        functions: ``np.array_equal`` / tuple equality / ``torch.equal``.
  R1b test_prepare_references_glue      1 / 2 / 3 images with and without a size: the crop box only for one image and no
                                        size (generate.py:70); 11 images -> ValueError via ``expect_error``.
  R1c test_crop_back                    a 1016x1016 synthetic reference -> canvas 1024x1024, crop (4, 4, 1020, 1020); the
                                        cropped 1024 render is 1016x1016 and its pixel (0, 0) is the render's (4, 4).
                                        (The design's 1000x1000 example gives a 1008x1008 canvas whose /8 VAE level has
                                        15876 rows, not tile-aligned: the policy refuses it, see R3.)
  R2  test_golden_cond_rows[case]       (skips until the golden exists) the golden's reference images through
                                        ``prepare_references``: sizes == ``replay.prepared_reference_sizes``, crop ==
                                        ``replay.crop``, default size == ``replay.default_size``, ``reference_ids_for``
                                        ``torch.equal`` ``inputs["cond_ids"]``, row count == ``replay.tokens.image_cond``;
                                        the host oracle ``AutoEncoder.encode(to_tensor(prepared_i))`` cast to bf16 vs the
                                        golden ``cond_tokens`` rows: pcc >= GOLDEN_COND_PCC and max |d| <= one bf16 ulp
                                        of the row max (fp32 reduction order of the CPU convolutions differs with the
                                        thread count / BLAS blocking, reproducibly on the producing node itself;
                                        bit-equality is logged, not asserted).
  R3  test_policy_references            ``admit`` rows: 1024 + 1 ref (8512) admitted; the 4096x4096 + 10 refs + 4096
                                        text structural worst case ADMITTED with ``measured["total_tokens"] ==
                                        110592`` (no token budget since 2026-10-19); a 4112 side -> ``side``;
                                        11 refs -> ValueError at resolve; an
                                        80x80 reference (level rows 6400 / 1600 / 400 / 100) -> ADMITTED (2026-10-21);
                                        a 1000x1000 reference (canvas 1008, level 15876 rows) -> ADMITTED;
                                        the measured dict carries every field of design section 6.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

from ....pipelines.flux3_klein import references as R
from ....pipelines.flux3_klein.packing import FLUX3_KLEIN_MAX_REFERENCES, latent_ids, reference_ids
from ....pipelines.flux3_klein.pipeline_flux3_klein import Flux3KleinRequest
from ....pipelines.flux3_klein.policy import CAPACITY, admit, vae_level_rows, vae_levels_tile_aligned
from ....reference.flux3_klein import pipeline as vend
from .common import (
    GOLDENS_ENV,
    cached_reference_encode,
    checkpoint_dir,
    golden_reference_images,
    load_golden_case,
    pcc,
    reference_golden_cases,
    require_example_input,
    sha256_of,
)

# R2: the golden cond_tokens are the fp32 oracle of the SAME prepared image cast to bf16 on the producing node; the fp32
# reduction order of the CPU convolutions (thread count / BLAS blocking) moves a few elements across a bf16 rounding
# boundary, on the producing node itself as much as on the login box (the M8 numerics verifier re-encoded on b05u08
# with a different thread count and saw the same 160-element difference: it is not a cross-node effect). Measured
# 2026-10-17 (login box logs/m8_builder_host_references.log:128; node b05u08
# logs/m8_verifier_numerics_golden_integrity_encode_b05u08.log:20,31,43): 3_bandana pcc 1.00000000, max|d| 7.8125e-03,
# 160 / 524288 elements differ (0.03 %); 2_3_mugprint max|d| 1.5625e-02 (one bf16 ulp at |g|max 6.375), 319 / 1048576;
# 1_figurine bit-equal (0 / 501760). Gate: pcc 0.99999 (five nines; measured 1.00000000 at eight digits) and one ulp.
GOLDEN_COND_PCC = 0.99999
EXAMPLE_INPUTS = ("1.png", "2.png", "3.png")
# design 2.1 synthetic sizes (w, h) + (10, 10) ("symmetric" pad) + (80, 80) (a non-tile-aligned VAE level)
SYNTHETIC_SIZES = (
    (3000, 2000),
    (4000, 4000),
    (5000, 1000),
    (100, 100),
    (50, 50),
    (1000, 1000),
    (1500, 1500),
    (1920, 1080),
    (10, 10),
    (80, 80),
)
REFERENCE_CASES = reference_golden_cases()


def synthetic_image(width: int, height: int, seed: int | None = None):
    from PIL import Image

    rng = np.random.default_rng(width * 7 + height if seed is None else seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), np.uint8))


def example_image(name: str):
    from PIL import Image

    return Image.open(require_example_input(name))


@pytest.mark.parametrize("source", [*EXAMPLE_INPUTS, *(f"{w}x{h}" for w, h in SYNTHETIC_SIZES)])
def test_sizing_bit_exact(source: str) -> None:
    if source.endswith(".png"):
        image = example_image(source)
    else:
        w, h = (int(v) for v in source.split("x"))
        image = synthetic_image(w, h)
    width, height = image.size
    canvas_ours, canvas_ref = R.reference_canvas(width, height), vend.reference_canvas(width, height)
    assert canvas_ours == canvas_ref, (canvas_ours, canvas_ref)
    ours, crop_ours = R.prepare_reference(image)
    ref, crop_ref = vend.prepare_reference(image)
    assert crop_ours == crop_ref, (crop_ours, crop_ref)
    assert ours.size == ref.size and ours.mode == ref.mode == "RGB"
    assert np.array_equal(np.asarray(ours), np.asarray(ref)), "prepare_reference pixels differ"
    multi_ours, multi_ref = R.prepare_multi_reference(image), vend.prepare_multi_reference(image)
    assert multi_ours.size == multi_ref.size
    assert np.array_equal(np.asarray(multi_ours), np.asarray(multi_ref)), "prepare_multi_reference pixels differ"
    assert R.default_size([ours]) == vend.default_size([ref]) == ours.size
    assert R.default_size([]) == vend.default_size([]) == (1024, 1024)
    assert torch.equal(R.to_tensor(ours), vend.to_tensor(ref))
    if multi_ours.size[0] and multi_ours.size[1]:
        assert torch.equal(R.to_tensor(multi_ours), vend.to_tensor(multi_ref))
    assert ours.size[0] % 16 == 0 and ours.size[1] % 16 == 0
    logger.info(
        f"RESULT sizing[{source}] {width}x{height}: canvas {canvas_ours} crop {crop_ours} prepared {ours.size} "
        f"multi {multi_ours.size} VAE levels {vae_level_rows(*ours.size)} tile-aligned {vae_levels_tile_aligned(*ours.size)}: "
        f"bit-exact vs the vendored reference"
    )


def test_prepare_references_glue(expect_error) -> None:
    big = synthetic_image(2000, 1200, seed=1)  # resized (over 1 MiP as a multi reference; fits 4 MiP single)
    small = synthetic_image(1000, 1000, seed=2)  # padded to 1008 as a single reference
    one = R.prepare_references([small], width=None, height=None)
    assert one.sizes == ((1008, 1008),) and one.crop == (4, 4, 1004, 1004) and one.default_size == (1008, 1008)
    assert one.latent_grids() == ((63, 63),) and one.cond_tokens() == 63 * 63
    sized = R.prepare_references([small], width=1024, height=None)
    assert sized.sizes == one.sizes and sized.crop is None, "a given side disables the crop-back (generate.py:70)"
    sized_h = R.prepare_references([small], width=None, height=512)
    assert sized_h.crop is None
    two = R.prepare_references([big, small], width=None, height=None)
    assert two.crop is None and two.sizes == (
        tuple(vend.prepare_multi_reference(big).size),
        tuple(vend.prepare_multi_reference(small).size),
    )
    assert two.default_size == two.sizes[0]
    three = R.prepare_references([small, big, example_image("3.png")], width=None, height=None)
    assert len(three.images) == 3 and three.crop is None and three.sizes[2] == (1024, 1024)
    assert three.cond_tokens() == sum(h * w for h, w in three.latent_grids())
    ids = R.reference_ids_for(three)
    assert ids.shape == (three.cond_tokens(), 4) and ids.dtype == torch.float32
    expected = torch.cat([reference_ids(h, w, i) for i, (h, w) in enumerate(three.latent_grids())])
    assert torch.equal(ids, expected)
    assert torch.equal(ids[:, 3].unique(sorted=True), torch.tensor([-3.0, -2.0, -1.0]))
    none = R.prepare_references([], width=None, height=None)
    assert none.images == () and none.default_size == (1024, 1024) and none.cond_tokens() == 0
    assert R.reference_ids_for(none).shape == (0, 4)
    path_opened = R.prepare_references([require_example_input("2.png")], width=None, height=None)
    assert path_opened.sizes == ((1024, 1024),)
    with expect_error(ValueError, "at most 10 references"):
        R.prepare_references([small] * (FLUX3_KLEIN_MAX_REFERENCES + 1), width=None, height=None)
    logger.info(
        f"RESULT prepare_references glue: single {one.sizes} crop {one.crop}; two {two.sizes}; three {three.sizes} "
        f"cond rows {three.cond_tokens()}; 11 images rejected"
    )


def test_crop_back() -> None:
    from PIL import Image

    reference = synthetic_image(1016, 1016, seed=3)
    prepared = R.prepare_references([reference], width=None, height=None)
    assert prepared.sizes == ((1024, 1024),) and prepared.crop == (4, 4, 1020, 1020)
    assert prepared.default_size == (1024, 1024) and vae_levels_tile_aligned(1024, 1024)
    # the padded canvas holds the reference centred: pixel (4, 4) of the canvas is pixel (0, 0) of the reference
    canvas = np.asarray(prepared.images[0])
    assert np.array_equal(canvas[4:1020, 4:1020], np.asarray(reference))
    assert np.array_equal(canvas[3, 4:1020], np.asarray(reference)[1]), "reflect pad: row 3 mirrors row 1"
    render = synthetic_image(1024, 1024, seed=4)
    cropped = render.crop(prepared.crop)
    assert cropped.size == (1016, 1016)
    assert np.array_equal(np.asarray(cropped)[0, 0], np.asarray(render)[4, 4])
    assert np.array_equal(np.asarray(cropped), np.asarray(render)[4:1020, 4:1020])
    # the design's 1000x1000 example: canvas 1008x1008, crop (4, 4, 1004, 1004), /8 level 15876 rows: NOT tile-aligned
    p1000 = R.prepare_references([synthetic_image(1000, 1000, seed=5)], width=None, height=None)
    assert p1000.sizes == ((1008, 1008),) and p1000.crop == (4, 4, 1004, 1004)
    assert vae_level_rows(1008, 1008) == (1016064, 254016, 63504, 15876) and not vae_levels_tile_aligned(1008, 1008)
    assert isinstance(cropped, Image.Image)
    logger.info(
        "RESULT crop_back: 1016x1016 -> canvas (1024, 1024) crop (4, 4, 1020, 1020), cropped render 1016x1016 exact; "
        "1000x1000 -> canvas 1008 (level rows 15876, not tile-aligned)"
    )


@pytest.mark.parametrize(
    "case_dir", REFERENCE_CASES or [None], ids=[f"{p.parent.name}-{p.name}" for p in REFERENCE_CASES] or ["no_goldens"]
)
def test_golden_cond_rows(case_dir: Path | None) -> None:
    if case_dir is None:
        pytest.skip(f"{GOLDENS_ENV} not set or no reference case under it")
    golden = load_golden_case(case_dir)
    replay = golden.replay
    images = golden_reference_images(golden)
    prepared = R.prepare_references(images, width=None, height=None)
    assert [list(s) for s in prepared.sizes] == replay["prepared_reference_sizes"], (prepared.sizes, replay)
    assert (list(prepared.crop) if prepared.crop else None) == replay["crop"]
    assert list(prepared.default_size) == replay["default_size"]
    assert list(prepared.default_size) == replay["size"], "the goldens render at default_size (no --size override)"
    assert prepared.cond_tokens() == replay["tokens"]["image_cond"]
    cond_ids = golden.inputs["cond_ids"]
    assert torch.equal(R.reference_ids_for(prepared), cond_ids), "reference ids differ from the golden cond_ids"
    cond_tokens = golden.inputs["cond_tokens"]
    assert cond_tokens.dtype == torch.bfloat16 and cond_tokens.shape == (prepared.cond_tokens(), 128)
    snapshot = checkpoint_dir("distilled", required=("ae.safetensors",))
    if snapshot is None:
        pytest.skip("FLUX 3 Klein snapshot with ae.safetensors not found: the token oracle needs it")
    start = 0
    for index, (image, (h, w)) in enumerate(zip(prepared.images, prepared.latent_grids(), strict=True)):
        tokens, ids = cached_reference_encode(snapshot, image, f"{golden.name} reference {index}")
        assert torch.equal(ids, latent_ids(h, w))
        rows = cond_tokens[start : start + h * w]
        start += h * w
        oracle_bf16 = tokens.to(torch.bfloat16)
        p = pcc(oracle_bf16.float(), rows.float())
        diff = (oracle_bf16.float() - rows.float()).abs()
        ulp = (rows.float().abs().max() * 2.0**-7).item()  # one bf16 ulp at the row max (8 bit mantissa)
        equal = torch.equal(oracle_bf16, rows)
        logger.info(
            f"RESULT golden_cond[{golden.name}] reference {index} {image.size}: host oracle encode (bf16) vs golden "
            f"cond_tokens pcc {p:.8f} max|d| {diff.max().item():.4e} (one bf16 ulp at the row max {ulp:.4e}) "
            f"mismatching elements {int((diff > 0).sum())} of {rows.numel()} bit-equal {equal}; sha256 {sha256_of(rows)[:16]}"
        )
        assert p >= GOLDEN_COND_PCC, p
        assert diff.max().item() <= ulp, diff.max().item()
    assert start == cond_tokens.shape[0]


def test_policy_references(expect_error) -> None:
    from PIL import Image

    cap = CAPACITY[(4, 8)]
    assert cap.max_references == 10 and not hasattr(cap, "max_total_tokens")
    ref1024 = Image.new("RGB", (1024, 1024))
    one = replace(Flux3KleinRequest(prompt="x", model="distilled", images=(ref1024,)).resolve(), text_length=320)
    assert (one.width, one.height, one.crop) == (1024, 1024, None)
    a = admit(one, (4, 8))
    m = a.measured
    assert set(m) >= {
        "image_tokens",
        "cond_tokens",
        "references",
        "reference_sizes",
        "text_length",
        "text_null",
        "total_tokens",
        "samples",
        "width",
        "height",
    }
    assert "max_total_tokens" not in m
    assert (m["image_tokens"], m["cond_tokens"], m["references"], m["total_tokens"], m["samples"]) == (
        4096,
        4096,
        1,
        4096 + 4096 + 320,
        1,
    )
    assert a.ok, a
    base = replace(Flux3KleinRequest(prompt="x", model="base", images=(ref1024,)).resolve(), text_length=320)
    assert admit(base, (4, 8)).measured["text_null"] == 80 and admit(base, (4, 8)).measured["samples"] == 2
    # the structural worst case: 4096 x 4096 + 10 references of 1 MP + 4096 text tokens is ADMITTED (no token budget)
    worst = Flux3KleinRequest(prompt="x", model="distilled", width=4096, height=4096, images=(ref1024,) * 10).resolve()
    worst = replace(worst, text_length=4096)
    w = admit(worst, (4, 8))
    assert w.ok and w.budget is None and w.measured["total_tokens"] == 110592, w
    assert w.measured["references"] == 10 and w.measured["cond_tokens"] == 40960
    # a side over MAX_SIDE is the structural refusal
    wide = replace(Flux3KleinRequest(prompt="x", model="distilled", width=4112, height=4096).resolve(), text_length=80)
    wd = admit(wide, (4, 8))
    assert not wd.ok and wd.budget == "side" and "outside" in wd.reason, wd
    # eleven references never reach admit
    with expect_error(ValueError, "at most 10 references"):
        Flux3KleinRequest(prompt="x", images=(ref1024,) * 11).resolve()
    # an 80x80 reference: VAE level rows 6400 / 1600 / 400 / 100 -> the /4 and /8 levels are not tile-aligned
    small = replace(
        Flux3KleinRequest(prompt="x", model="distilled", images=(Image.new("RGB", (80, 80)),)).resolve(), text_length=80
    )
    s = admit(small, (4, 8))
    # The REFERENCE is admitted since 2026-10-21 (the encoder's GroupNorm has an exact path for row counts that are
    # not tile multiples: layers/normalization.py _forward_unaligned, test_encode_unaligned_reference; the former
    # ``reference_size`` refusal rejected most real photos at BFL's own prepared sizes). With no explicit size the
    # CANVAS defaults to the reference's 80x80, which the decoder's canvas rule still refuses.
    assert not s.ok and s.budget == "canvas_size", s
    assert vae_level_rows(80, 80) == (6400, 1600, 400, 100) and not vae_levels_tile_aligned(80, 80)
    small_on_1024 = replace(
        Flux3KleinRequest(prompt="x", model="distilled", width=1024, height=1024, images=(Image.new("RGB", (80, 80)),)).resolve(),
        text_length=80,
    )
    assert admit(small_on_1024, (4, 8)).ok, admit(small_on_1024, (4, 8))
    # a 1000x1000 reference pads to 1008x1008 whose /8 level has 15876 rows
    padded = replace(
        Flux3KleinRequest(prompt="x", model="distilled", images=(Image.new("RGB", (1000, 1000)),)).resolve(),
        text_length=80,
    )
    pd = admit(padded, (4, 8))
    # the reference itself is admitted; the DEFAULT canvas (1008x1008, from the reference) is what the canvas rule refuses
    assert not pd.ok and pd.budget == "canvas_size" and pd.measured["reference_sizes"] == [[1008, 1008]], pd
    padded_on_1024 = replace(
        Flux3KleinRequest(prompt="x", model="distilled", width=1024, height=1024, images=(Image.new("RGB", (1000, 1000)),)).resolve(),
        text_length=80,
    )
    assert admit(padded_on_1024, (4, 8)).ok, admit(padded_on_1024, (4, 8))
    # an explicit canvas with a non-tile-aligned level, no references
    canvas = replace(
        Flux3KleinRequest(prompt="x", model="distilled", width=1008, height=1008).resolve(), text_length=80
    )
    c = admit(canvas, (4, 8))
    assert not c.ok and c.budget == "canvas_size", c
    # T2I 1024 stays admitted at the shipped capacity (the M7 row)
    t2i = replace(Flux3KleinRequest(prompt="x", model="base").resolve(), text_length=320)
    assert admit(t2i, (4, 8)).ok and admit(t2i, (4, 8)).measured["total_tokens"] == 4416
    # a given size wins over default_size and disables the crop-back
    sized = Flux3KleinRequest(prompt="x", model="distilled", width=512, height=768, images=(ref1024,)).resolve()
    assert (sized.width, sized.height, sized.crop) == (512, 768, None)
    logger.info(
        f"RESULT policy references: 1 ref 1024 total {m['total_tokens']} admitted; worst case "
        f"{w.measured['total_tokens']} admitted (no token budget); 4112 side -> {wd.budget}; 80x80 -> "
        f"{s.budget}; 1000x1000 (1008 canvas) -> {pd.budget}; 1008 canvas -> {c.budget}; 11 refs -> ValueError"
    )
