# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only parity gate for the FLUX 3 Image Klein packed sequence: exact assertions against the vendored BFL
reference (drift is silent). The pipeline's ``packing.py`` never imports the reference; this test does, and
compares layout, bounds, ids and dtypes bit-exactly with ``pipeline.pack()`` / ``autoencoder.latent_ids`` /
``pipeline.scaled_ids`` and with what the vendored DiT itself derives inside ``forward`` (checked through hooks
on a depth-1 random DiT)."""

import pytest
import torch

from ....pipelines.flux3_klein import packing as p
from ....reference.flux3_klein import autoencoder as ref_ae
from ....reference.flux3_klein import dit as ref_dit
from ....reference.flux3_klein import pipeline as ref_pipeline
from .common import patch_reference_depth

# (label, text length, reference sizes (w, h), canvas (w, h)); the sizes PLAN.md section 4 calls canonical.
T2I_1024 = ("t2i_1024", 320, (), (1024, 1024))
ONE_REF_1024 = ("one_ref_1024", 240, ((1024, 1024),), (1024, 1024))
TWO_REFS_MIXED = ("two_refs_mixed", 320, ((1024, 1024), (896, 1120)), (1024, 1024))
# Small enough to push through the vendored DiT on CPU in the hook tests.
SMALL_T2I = ("small_t2i", 80, (), (256, 256))
SMALL_ONE_REF = ("small_one_ref", 80, ((128, 192),), (256, 256))

CASES = [T2I_1024, ONE_REF_1024, TWO_REFS_MIXED]
HOOK_CASES = [SMALL_T2I, SMALL_ONE_REF]

SEED = 1234


def _tokens(rows: int, width: int, dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(rows, width, generator=generator).to(dtype)


def _reference_text_ids(length: int) -> torch.Tensor:
    # text_encoder.py:48-49, verbatim (the encoder itself is not loaded here).
    positions = torch.arange(length)
    return torch.stack([torch.zeros_like(positions)] * 3 + [positions], dim=-1)


def _reference_cond(sizes, generator) -> tuple[torch.Tensor, torch.Tensor] | None:
    # pipeline.py:102-112 with the autoencoder's (tokens, ids) replaced by random tokens of the right shape.
    if not sizes:
        return None
    tokens, ids = [], []
    for index, (width, height) in enumerate(sizes):
        h, w = ref_ae.latent_shape(height, width)
        t = _tokens(h * w, ref_ae.LATENT_CHANNELS, torch.float32, generator)
        i = ref_pipeline.scaled_ids(ref_ae.latent_ids(h, w))
        i[:, 3] = -(1 + index)
        tokens.append(t.to(torch.bfloat16))
        ids.append(i)
    return torch.cat(tokens), torch.cat(ids)


def _reference_triple(case, generator):
    """``[text, image_cond, image]`` built from the reference's own helpers (pipeline.py:86-98)."""
    _, text_len, sizes, (width, height) = case
    text = (_tokens(text_len, ref_dit.IN_CHANNELS["text"], torch.bfloat16, generator), _reference_text_ids(text_len))
    cond = _reference_cond(sizes, generator)
    h, w = ref_ae.latent_shape(height, width)
    noise = torch.randn((h * w, ref_dit.IN_CHANNELS["image"]), generator=generator).to(torch.bfloat16).float()
    image = (noise, ref_pipeline.scaled_ids(ref_ae.latent_ids(h, w)))
    return [text, cond, image]


def _our_sample(case, triple) -> p.Flux3KleinSample:
    """The same tokens with ids rebuilt by our helpers, so ids parity is part of every comparison."""
    _, text_len, sizes, (width, height) = case
    text_tokens, _ = triple[0]
    image_tokens, _ = triple[2]
    cond = None
    if sizes:
        cond_tokens, _ = triple[1]
        ids = [p.reference_ids(*p.latent_shape(h_, w_), index) for index, (w_, h_) in enumerate(sizes)]
        cond = (cond_tokens, torch.cat(ids))
    h, w = p.latent_shape(height, width)
    return p.Flux3KleinSample(
        text=(text_tokens, p.text_ids(text_len)), image=(image_tokens, p.image_ids(h, w)), image_cond=cond
    )


def _reference_single_stage(packed):
    """dit.py:216-228 verbatim, minus the tensor concatenations: the layout the single blocks see."""
    bounds, placement, offset = [], [], 0
    for s in range(packed.num_samples):
        start = offset
        for m in ref_dit.MODES:
            a, b = packed.mode_indices[m][s]
            if b > a:
                placement.append((m, a, b, offset))
                offset += b - a
        bounds.append((start, offset))
    return bounds, placement


def _reference_phantom(packed):
    """dit.py:191-198 verbatim."""
    data, ids, timesteps = dict(packed.data), dict(packed.ids), dict(packed.timesteps)
    for m in ref_dit.MODES:
        if data[m].shape[0] == 0:
            data[m], ids[m], timesteps[m] = (
                data[m].new_zeros(1, data[m].shape[1]),
                ids[m].new_zeros(1, 4),
                timesteps[m].new_zeros(1),
            )
    return data, ids, timesteps


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


@pytest.fixture(params=CASES + HOOK_CASES, ids=lambda c: c[0])
def case(request):
    return request.param


@pytest.fixture
def packs(case):
    generator = torch.Generator().manual_seed(SEED)
    triple = _reference_triple(case, generator)
    theirs = ref_pipeline.pack([triple])
    ours = p.pack([_our_sample(case, triple)])
    return case, triple, theirs, ours


# --- geometry and ids ---------------------------------------------------------------------------------------------


def test_constants_match_reference():
    assert p.FLUX3_KLEIN_MODES == ref_dit.MODES == ("image", "image_cond", "text")
    assert p.FLUX3_KLEIN_IN_CHANNELS == ref_dit.IN_CHANNELS
    assert p.FLUX3_KLEIN_IMAGE_ROPE_SCALE == ref_pipeline.IMAGE_ROPE_SCALE
    assert p.FLUX3_KLEIN_DOWNSAMPLING_RATIO == ref_ae.DOWNSAMPLING_RATIO
    assert p.FLUX3_KLEIN_LATENT_CHANNELS == ref_ae.LATENT_CHANNELS
    assert p.FLUX3_KLEIN_VAE_CHANNELS == ref_ae.EMBED_DIM
    assert p.FLUX3_KLEIN_MAX_REFERENCES == ref_pipeline.MAX_REFERENCES


@pytest.mark.parametrize("size,grid", [((1024, 1024), (64, 64)), ((896, 1120), (70, 56)), ((4096, 2304), (144, 256))])
def test_latent_shape(size, grid):
    width, height = size
    assert p.latent_shape(height, width) == ref_ae.latent_shape(height, width) == grid


def test_latent_shape_rejects_unaligned(expect_error):
    with expect_error(ValueError, "multiples of 16"):
        p.latent_shape(1000, 1024)


@pytest.mark.parametrize("grid", [(64, 64), (70, 56), (1, 1), (3, 5)])
def test_latent_and_image_ids_match_reference(grid):
    h, w = grid
    ours, theirs = p.latent_ids(h, w), ref_ae.latent_ids(h, w)
    assert _same(ours, theirs) and ours.dtype == torch.int64
    assert _same(p.image_ids(h, w), ref_pipeline.scaled_ids(theirs))
    assert p.image_ids(h, w).dtype == torch.float32
    assert p.latent_grid_from_ids(ours) == grid


@pytest.mark.parametrize("index", [0, 1, 9])
def test_reference_ids_match_reference(index):
    h, w = 56, 70
    theirs = ref_pipeline.scaled_ids(ref_ae.latent_ids(h, w))
    theirs[:, 3] = -(1 + index)  # pipeline.py:109
    assert _same(p.reference_ids(h, w, index), theirs)
    assert (p.reference_ids(h, w, index)[:, 3] == -(1 + index)).all()


def test_scaled_ids_does_not_alias_its_input():
    ids = p.latent_ids(4, 4).float()
    scaled = p.scaled_ids(ids)
    assert torch.equal(ids, p.latent_ids(4, 4).float())
    assert torch.equal(scaled, ref_pipeline.scaled_ids(p.latent_ids(4, 4)))


@pytest.mark.parametrize("length", [80, 240, 320, 560, 4096])
def test_text_ids_match_reference(length):
    ours = p.text_ids(length)
    assert _same(ours, _reference_text_ids(length)) and ours.dtype == torch.int64
    assert (ours[:, :3] == 0).all() and torch.equal(ours[:, 3], torch.arange(length))


@pytest.mark.parametrize(
    "real,padded", [(1, 80), (80, 80), (81, 160), (239, 240), (320, 320), (4095, 4096), (5000, 4096)]
)
def test_padded_text_length(real, padded):
    assert p.padded_text_length(real) == padded


# --- 2x2 patchify / unpatchify ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("grid", [(64, 64), (70, 56), (2, 3)])
def test_patchify_matches_reference(grid):
    h, w = grid
    generator = torch.Generator().manual_seed(SEED)
    mean = torch.randn(1, ref_ae.EMBED_DIM, 2 * h, 2 * w, generator=generator)
    # autoencoder.py:167-170 with the (per-channel) BatchNorm left out.
    z = mean.reshape(1, ref_ae.EMBED_DIM, h, 2, w, 2).permute(0, 1, 3, 5, 2, 4)
    z = z.reshape(1, ref_ae.LATENT_CHANNELS, h, w)
    theirs = z[0].permute(1, 2, 0).reshape(h * w, ref_ae.LATENT_CHANNELS)
    ours = p.patchify_latents(mean[0])
    assert _same(ours, theirs)
    # autoencoder.py:176-178
    back = theirs.reshape(h, w, ref_ae.LATENT_CHANNELS).permute(2, 0, 1).unsqueeze(0)
    back = (
        back.reshape(1, ref_ae.EMBED_DIM, 2, 2, h, w)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(1, ref_ae.EMBED_DIM, 2 * h, 2 * w)
    )
    assert _same(p.unpatchify_tokens(ours, h, w), back[0])
    assert torch.equal(p.unpatchify_tokens(ours, h, w), mean[0])


def test_patchify_rejects_bad_shapes(expect_error):
    with expect_error(ValueError, "are not \\(32, 2h, 2w\\)"):
        p.patchify_latents(torch.zeros(32, 5, 4))
    with expect_error(ValueError, "do not fill a 3x3 grid"):
        p.unpatchify_tokens(torch.zeros(12, 128), 3, 3)


# --- pack ----------------------------------------------------------------------------------------------------------


def test_pack_matches_reference(packs):
    case, triple, theirs, ours = packs
    for m in ref_dit.MODES:
        assert _same(ours.data[m], theirs.data[m]), m
        assert _same(ours.ids[m], theirs.ids[m]), m
        assert _same(ours.timesteps[m], theirs.timesteps[m]), m
        assert ours.mode_indices[m] == theirs.mode_indices[m], m
    assert ours.num_samples == theirs.num_samples == 1
    # The reference-triple constructor yields the same pack.
    again = p.pack([p.Flux3KleinSample.from_reference_triple(triple)])
    for m in ref_dit.MODES:
        assert _same(again.data[m], theirs.data[m]) and _same(again.ids[m], theirs.ids[m])


def test_pack_dtypes_are_the_reference_contract(packs):
    case, _, _, ours = packs
    _, text_len, sizes, (width, height) = case
    h, w = p.latent_shape(height, width)
    assert ours.data["text"].dtype == torch.bfloat16 and ours.ids["text"].dtype == torch.int64
    assert ours.data["image"].dtype == torch.float32 and ours.ids["image"].dtype == torch.float32
    assert ours.data["image_cond"].dtype == torch.bfloat16 and ours.ids["image_cond"].dtype == torch.float32
    assert all(ours.timesteps[m].dtype == torch.bfloat16 for m in ref_dit.MODES)
    assert ours.mode_rows("text") == text_len and ours.mode_rows("image") == h * w
    assert ours.mode_rows("image_cond") == sum(
        p.latent_shape(h_, w_)[0] * p.latent_shape(h_, w_)[1] for w_, h_ in sizes
    )
    assert ours.sequence_ids().dtype == torch.float32  # torch.cat promotes the int64 text ids, as dit.py:230 does


def test_single_stage_layout_matches_reference_loop(packs):
    case, _, theirs, ours = packs
    bounds, placement = _reference_single_stage(theirs)
    assert ours.bounds == bounds
    assert [(s.mode, s.mode_start, s.mode_end, s.start) for s in ours.segments] == placement
    assert ours.sequence_length == bounds[-1][1] == sum(ours.mode_rows(m) for m in ref_dit.MODES)
    # [image | image_cond | text]: image first, text last, image_cond (when present) in between.
    modes = [s.mode for s in ours.segments]
    expected = ["image"] + (["image_cond"] if ours.mode_rows("image_cond") else []) + ["text"]
    assert modes == expected
    seq_ids = ours.sequence_ids()
    assert torch.equal(seq_ids[: ours.mode_rows("image")], ours.ids["image"])
    assert torch.equal(seq_ids[-ours.mode_rows("text") :], ours.ids["text"].float())


def test_two_sample_cfg_pack_matches_reference():
    """The base model packs [unconditional, conditional] sharing cond and image (pipeline.py:97-98)."""
    generator = torch.Generator().manual_seed(SEED)
    triple = _reference_triple(ONE_REF_1024, generator)
    empty = (_tokens(80, ref_dit.IN_CHANNELS["text"], torch.bfloat16, generator), _reference_text_ids(80))
    theirs = ref_pipeline.pack([[empty, triple[1], triple[2]], triple])
    ours = p.pack(
        [
            p.Flux3KleinSample(text=empty, image=triple[2], image_cond=triple[1]),
            p.Flux3KleinSample.from_reference_triple(triple),
        ]
    )
    for m in ref_dit.MODES:
        assert _same(ours.data[m], theirs.data[m]) and _same(ours.ids[m], theirs.ids[m])
        assert ours.mode_indices[m] == theirs.mode_indices[m]
    bounds, placement = _reference_single_stage(theirs)
    assert ours.bounds == bounds and len(bounds) == 2
    assert [(s.mode, s.mode_start, s.mode_end, s.start) for s in ours.segments] == placement
    assert ours.mode_indices["text"] == [(0, 80), (80, 80 + 240)]
    assert ours.bounds[0] == (0, 4096 + 4096 + 80) and ours.bounds[1] == (4096 + 4096 + 80, 2 * 8192 + 80 + 240)


def test_pack_rejects_mismatched_inputs(expect_error):
    sample = p.Flux3KleinSample(
        text=(torch.zeros(80, 20480, dtype=torch.bfloat16), p.text_ids(79)),
        image=(torch.zeros(4, 128), p.image_ids(2, 2)),
    )
    with expect_error(ValueError, "disagree"):
        p.pack([sample])
    with expect_error(ValueError, "expected 20480 channels"):
        p.pack(
            [
                p.Flux3KleinSample(
                    text=(torch.zeros(80, 64, dtype=torch.bfloat16), p.text_ids(80)),
                    image=(torch.zeros(4, 128), p.image_ids(2, 2)),
                )
            ]
        )
    with expect_error(ValueError, "at least one sample"):
        p.pack([])


# --- phantom tokens ------------------------------------------------------------------------------------------------


def test_phantom_rule_matches_reference():
    generator = torch.Generator().manual_seed(SEED)
    triple = _reference_triple(T2I_1024, generator)
    theirs = ref_pipeline.pack([triple])
    ours = p.pack([_our_sample(T2I_1024, triple)])
    assert ours.phantom_modes == ("image_cond",)
    assert ours.mode_indices["image_cond"] == theirs.mode_indices["image_cond"] == [(0, 0)]
    assert ours.data["image_cond"].shape == (0, 128) and ours.data["image_cond"].dtype == torch.bfloat16
    assert ours.ids["image_cond"].shape == (0, 4) and ours.ids["image_cond"].dtype == torch.float32
    assert ours.projection_rows("image_cond") == 1 and ours.projection_rows("image") == 4096
    assert ours.segment(0, "image_cond") is None
    assert [s.mode for s in ours.segments] == ["image", "text"]

    data, ids, timesteps = p.with_phantom_tokens(ours)
    ref_data, ref_ids, ref_timesteps = _reference_phantom(theirs)
    for m in ref_dit.MODES:
        assert _same(data[m], ref_data[m]) and _same(ids[m], ref_ids[m]) and _same(timesteps[m], ref_timesteps[m])
    assert (
        data["image_cond"].shape == (1, 128)
        and ids["image_cond"].shape == (1, 4)
        and timesteps["image_cond"].shape == (1,)
    )
    # The phantom is never counted: mode_indices and the single-stage layout do not change.
    assert ours.sequence_length == 4096 + 320


# --- the vendored DiT consumes exactly this layout -------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_dit():
    """A random depth-(1, 1) flow-map DiT: real widths, so the hooks see the real tensors the single stage gets."""
    mp = pytest.MonkeyPatch()
    patch_reference_depth(mp, ref_dit, (1, 1))
    with torch.device("meta"):
        dit = ref_dit.DiT(flow_map=True)
    dit = dit.to_empty(device="cpu").to(torch.bfloat16)
    generator = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, parameter in dit.named_parameters():
            if name.endswith("scale"):
                parameter.fill_(1.0)
            else:
                parameter.copy_(torch.randn(parameter.shape, generator=generator).to(torch.bfloat16) * 0.02)
    dit.eval()
    yield dit
    mp.undo()


def _run_with_hooks(dit, packed):
    captured = {}
    handles = [
        dit.single_blocks[0].register_forward_pre_hook(lambda _m, args: captured.__setitem__("single", args)),
        dit.content_mode_blocks["image"][0].register_forward_hook(
            lambda _m, args, out: captured.__setitem__("image", (args, out))
        ),
        dit.content_mode_blocks["image_cond"][0].register_forward_hook(
            lambda _m, args, out: captured.__setitem__("image_cond", (args, out))
        ),
        dit.txt_mode_blocks[0].register_forward_hook(lambda _m, args, out: captured.__setitem__("text", (args, out))),
    ]
    try:
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            velocity = dit(packed, timesteps_ref=torch.tensor([0.5]))
    finally:
        for handle in handles:
            handle.remove()
    return velocity, captured


@pytest.mark.parametrize("hook_case", HOOK_CASES, ids=lambda c: c[0])
def test_dit_single_stage_consumes_our_layout(small_dit, hook_case):
    generator = torch.Generator().manual_seed(SEED)
    triple = _reference_triple(hook_case, generator)
    theirs = ref_pipeline.pack([triple])
    ours = p.pack([_our_sample(hook_case, triple)])

    velocity, captured = _run_with_hooks(small_dit, theirs)
    assert velocity.shape == (ours.mode_rows("image"), 128) and torch.isfinite(velocity.float()).all()

    h, freqs_sm, bounds, mods = captured["single"]
    assert list(bounds) == ours.bounds
    assert h.shape[0] == ours.sequence_length
    # The rotary tables of the single stage are built from the ids in our sequence order, bit for bit.
    assert torch.equal(freqs_sm, ref_dit.rope_embed(ours.sequence_ids()[None]))
    # And the rows themselves are the per-mode block outputs placed at our segment offsets.
    for segment in ours.segments:
        (_args, mode_out) = captured[segment.mode]
        assert torch.equal(h[segment.start : segment.end], mode_out[segment.mode_start : segment.mode_end]), segment
    # Per-token modulation of the single stage is constant inside a segment and differs across modes.
    for tensor in mods:
        for segment in ours.segments:
            rows = tensor[segment.start : segment.end]
            assert torch.equal(rows, rows[:1].expand_as(rows)), segment
    for segment, other in zip(ours.segments, ours.segments[1:]):
        assert not torch.equal(mods[0][segment.start], mods[0][other.start])


def test_dit_runs_the_phantom_outside_the_sequence(small_dit):
    generator = torch.Generator().manual_seed(SEED)
    triple = _reference_triple(SMALL_T2I, generator)
    theirs = ref_pipeline.pack([triple])
    ours = p.pack([_our_sample(SMALL_T2I, triple)])
    assert ours.phantom_modes == ("image_cond",)

    _velocity, captured = _run_with_hooks(small_dit, theirs)
    (args, out) = captured["image_cond"]
    x, _freqs, bounds, _mod = args
    assert x.shape == (ours.projection_rows("image_cond"), ref_dit.HIDDEN) == (1, 3072)
    assert list(bounds) == ours.mode_indices["image_cond"] == [(0, 0)]
    assert out.shape == (1, ref_dit.HIDDEN)
    h, _freqs_sm, single_bounds, _mods = captured["single"]
    assert list(single_bounds) == ours.bounds == [(0, ours.sequence_length)]
    assert h.shape[0] == ours.sequence_length == ours.mode_rows("image") + ours.mode_rows("text")


# --- tile padding ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "length,sp,padded",
    [
        (320, 1, 320),
        (320, 4, 384),
        (240, 1, 256),
        (240, 4, 256),
        (4096, 4, 4096),
        (3920, 4, 3968),
        (0, 4, 0),
        (1, 1, 32),
    ],
)
def test_padded_length(length, sp, padded):
    assert p.padded_length(length, sp) == padded


def test_padded_length_rejects_bad_factors(expect_error):
    with expect_error(ValueError, "must be positive"):
        p.padded_length(10, 0)


def test_pad_segment_keeps_encoder_padded_text_rows_real(expect_error):
    """The encoder pads text to a multiple of 80 (e.g. 240 for 2_coffee); those rows are real to the DiT, so the
    mask marks only the rows appended here for tile alignment."""
    text = torch.randn(240, 8).to(torch.bfloat16)
    padded, mask = p.pad_segment(text, sp_factor=4)
    assert padded.shape == (256, 8) and padded.dtype == torch.bfloat16
    assert mask.dtype == torch.bool and mask.shape == (256,)
    assert mask[:240].all() and not mask[240:].any()
    assert torch.equal(padded[:240], text) and (padded[240:] == 0).all()

    padded_1, mask_1 = p.pad_segment(text, sp_factor=1)
    assert padded_1.shape == (256, 8) and mask_1[:240].all() and not mask_1[240:].any()  # 240 = 7.5 tiles
    aligned = torch.randn(320, 8).to(torch.bfloat16)
    same, full = p.pad_segment(aligned, sp_factor=1)
    assert same is aligned and full.all()  # 320 is already a multiple of 32: no copy, every row real
    with expect_error(ValueError, "cannot pad 240 rows down to 200"):
        p.pad_rows(text, 200)
