# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of ``layout_flux3_klein.Flux3KleinSequenceLayout`` (no ``ttnn``): the index maps the device
assembly uses are checked against torch concatenation on a host model of the all-gather buffer, built exactly
as the device builds it (per mode split the zero-padded tensor into SP equal shards, concatenate shard ``d`` of
every mode, concatenate over ``d``)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ....models.transformers.flux3_klein.layout_flux3_klein import (
    FLUX3_KLEIN_PAD_MODE_ID,
    Flux3KleinSequenceLayout,
    mode_id,
)
from ....pipelines.flux3_klein import packing
from ....pipelines.flux3_klein.packing import FLUX3_KLEIN_MODES, padded_length

# (image, image_cond, text) lengths: T2I 1024 (320-token prompt), the unconditional prompt with a small reference,
# a 896x1120 canvas, one 1024 reference, ten 1024 references with the longest text, the M3 fast rows, and the
# smallest sample (image 80 = 8x10, empty cond, text 80).
LENGTH_CASES = [
    (4096, 0, 320),
    (4096, 80, 80),
    (3920, 80, 320),
    (4096, 4096, 320),
    (4096, 40960, 4096),
    (1024, 80, 80),
    (80, 0, 80),
]
SP_FACTORS = [1, 2, 4]


def lengths_of(case: tuple[int, int, int]) -> dict[str, int]:
    return dict(zip(FLUX3_KLEIN_MODES, case))


def mode_tensors(layout: Flux3KleinSequenceLayout, seed: int) -> dict[str, torch.Tensor]:
    """Per non-empty mode a ``[padded[m], 2]`` tensor of unique values on the real rows and zeros on the tail."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    base = 1
    for m in layout.modes:
        n = layout.lengths[m]
        real = torch.stack([torch.arange(base, base + n, dtype=torch.float32), torch.randn(n, generator=g)], dim=-1)
        base += n
        out[m] = F.pad(real, (0, 0, 0, layout.padded[m] - n))
    return out


def gathered_buffer(layout: Flux3KleinSequenceLayout, parts: dict[str, torch.Tensor]) -> torch.Tensor:
    """The device's ``P``: shard ``d`` of every mode concatenated per device, devices concatenated in order."""
    shards = {m: parts[m].chunk(layout.sp_factor, dim=0) for m in layout.modes}
    for m in layout.modes:
        assert all(s.shape[0] == layout.rows[m] for s in shards[m])
    return torch.cat([torch.cat([shards[m][d] for m in layout.modes]) for d in range(layout.sp_factor)])


def real_rows(layout: Flux3KleinSequenceLayout, parts: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([parts[m][: layout.lengths[m]] for m in layout.modes])


@pytest.mark.parametrize("sp", SP_FACTORS)
@pytest.mark.parametrize("case", LENGTH_CASES, ids=["_".join(map(str, c)) for c in LENGTH_CASES])
def test_derived_fields(case, sp) -> None:
    layout = Flux3KleinSequenceLayout.from_lengths(lengths_of(case), sp)
    for m, n in zip(FLUX3_KLEIN_MODES, case):
        assert layout.lengths[m] == n
        assert layout.padded[m] == padded_length(n, sp)
        assert layout.padded[m] % (32 * sp) == 0
        assert layout.rows[m] * sp == layout.padded[m]
    assert layout.rows_total == sum(layout.rows.values())
    assert layout.total_padded == layout.rows_total * sp == sum(layout.padded.values())
    assert layout.total == sum(case)
    assert layout.modes == tuple(m for m, n in zip(FLUX3_KLEIN_MODES, case) if n > 0)
    assert layout.rows_total % 32 == 0
    # The pad-source rule: without any alignment row there is no pad source and none is needed.
    all_aligned = all(padded_length(n, sp) == n for n in case)
    assert (layout.total_padded == layout.total) == all_aligned
    if all_aligned:
        assert layout.assembly_indices().shape[0] == layout.total


@pytest.mark.parametrize("sp", SP_FACTORS)
@pytest.mark.parametrize("case", LENGTH_CASES, ids=["_".join(map(str, c)) for c in LENGTH_CASES])
def test_assembly_indices_reproduce_concat(case, sp) -> None:
    layout = Flux3KleinSequenceLayout.from_lengths(lengths_of(case), sp)
    parts = mode_tensors(layout, seed=sum(case) + sp)
    buffer = gathered_buffer(layout, parts)
    assert buffer.shape[0] == layout.total_padded
    indices = layout.assembly_indices()
    assert indices.dtype == torch.int64 and indices.shape == (layout.total_padded,)
    assert int(indices.min()) >= 0 and int(indices.max()) < layout.total_padded
    expected = F.pad(real_rows(layout, parts), (0, 0, 0, layout.total_padded - layout.total))
    assert torch.equal(buffer[indices], expected)
    # Every pad row of G copies the first alignment row of the first padded mode (a zero row by contract).
    if layout.total_padded > layout.total:
        first = next(m for m in layout.modes if layout.padded[m] > layout.lengths[m])
        pad_source = layout.source_row(first, layout.lengths[first])
        assert torch.equal(indices[layout.total :], torch.full((layout.total_padded - layout.total,), pad_source))
        assert torch.equal(buffer[pad_source], torch.zeros(2))
    # The real rows are a permutation of distinct P rows.
    assert indices[: layout.total].unique().numel() == layout.total
    # Each device's slice of the index vector is a tile multiple.
    assert layout.total_padded % (32 * sp) == 0


@pytest.mark.parametrize("sp", SP_FACTORS)
@pytest.mark.parametrize("case", LENGTH_CASES, ids=["_".join(map(str, c)) for c in LENGTH_CASES])
def test_image_out_indices(case, sp) -> None:
    layout = Flux3KleinSequenceLayout.from_lengths(lengths_of(case), sp)
    parts = mode_tensors(layout, seed=7)
    sequence = F.pad(real_rows(layout, parts), (0, 0, 0, layout.total_padded - layout.total))
    indices = layout.image_out_indices()
    length, padded = layout.lengths["image"], layout.padded["image"]
    assert indices.shape == (padded,) and indices.dtype == torch.int64
    assert padded % (32 * sp) == 0
    image = sequence[indices]
    assert torch.equal(image[:length], parts["image"][:length])
    assert torch.equal(indices[length:], torch.zeros(padded - length, dtype=torch.int64))


@pytest.mark.parametrize("sp", SP_FACTORS)
@pytest.mark.parametrize("case", LENGTH_CASES, ids=["_".join(map(str, c)) for c in LENGTH_CASES])
def test_mode_ids(case, sp) -> None:
    layout = Flux3KleinSequenceLayout.from_lengths(lengths_of(case), sp)
    seq = layout.sequence_mode_ids()
    assert seq.shape == (layout.total_padded,)
    expected = torch.cat(
        [torch.full((layout.lengths[m],), mode_id(m)) for m in layout.modes]
        + [torch.full((layout.total_padded - layout.total,), FLUX3_KLEIN_PAD_MODE_ID)]
    )
    assert torch.equal(seq, expected)
    for m in FLUX3_KLEIN_MODES:
        tail = layout.mode_tail_ids(m)
        assert tail.shape == (layout.padded[m],)
        assert int((tail == mode_id(m)).sum()) == layout.lengths[m]
        assert int((tail == FLUX3_KLEIN_PAD_MODE_ID).sum()) == layout.padded[m] - layout.lengths[m]
        assert torch.equal(tail[: layout.lengths[m]], torch.full((layout.lengths[m],), mode_id(m)))
    assert FLUX3_KLEIN_PAD_MODE_ID == 3 and [mode_id(m) for m in FLUX3_KLEIN_MODES] == [0, 1, 2]


def _sample(
    text_len: int, cond_sizes: list[tuple[int, int]], image_size: tuple[int, int], g
) -> packing.Flux3KleinSample:
    text = (torch.randn(text_len, 20480, generator=g).to(torch.bfloat16), packing.text_ids(text_len))
    h, w = image_size
    image = (torch.randn(h * w, 128, generator=g), packing.image_ids(h, w))
    cond = None
    if cond_sizes:
        tokens = torch.cat([torch.randn(ch * cw, 128, generator=g).to(torch.bfloat16) for ch, cw in cond_sizes])
        ids = torch.cat([packing.reference_ids(ch, cw, k) for k, (ch, cw) in enumerate(cond_sizes)])
        cond = (tokens, ids)
    return packing.Flux3KleinSample(text=text, image=image, image_cond=cond)


@pytest.mark.parametrize("sp", SP_FACTORS)
def test_from_packed_and_sequence_ids(sp) -> None:
    g = torch.Generator().manual_seed(3)
    # Sample 0: image 32x32 + one 8x10 reference + text 80; sample 1: image 16x20 + text 160, no reference.
    samples = [_sample(80, [(8, 10)], (32, 32), g), _sample(160, [], (16, 20), g)]
    packed = packing.pack(samples)
    for s, (image, cond, text) in enumerate(((1024, 80, 80), (320, 0, 160))):
        layout = Flux3KleinSequenceLayout.from_packed(packed, s, sp)
        assert layout == Flux3KleinSequenceLayout.from_lengths({"image": image, "image_cond": cond, "text": text}, sp)
        assert layout.total == packed.bounds[s][1] - packed.bounds[s][0]
        # Per-sample ids in the mode-major pack: the sample's own rows of every mode.
        ids = {m: packed.ids[m][packed.mode_indices[m][s][0] : packed.mode_indices[m][s][1]] for m in FLUX3_KLEIN_MODES}
        seq_ids = layout.sequence_ids(ids)
        assert seq_ids.dtype == torch.float32 and seq_ids.shape == (layout.total, 4)
        assert torch.equal(seq_ids, packed.sequence_ids()[packed.bounds[s][0] : packed.bounds[s][1]])
        assert [
            mode_id(m) for m in packed.sequence_modes()[packed.bounds[s][0] : packed.bounds[s][1]]
        ] == layout.sequence_mode_ids()[: layout.total].tolist()
    # One-sample pack: sequence_ids of the raw mode-major ids equals the pack's own.
    single = packing.pack(samples[:1])
    layout = Flux3KleinSequenceLayout.from_packed(single, 0, sp)
    assert torch.equal(layout.sequence_ids(single.ids), single.sequence_ids())
    assert torch.equal(
        layout.sequence_ids(single.ids),
        torch.cat([packing.image_ids(32, 32), packing.reference_ids(8, 10, 0), packing.text_ids(80).float()]),
    )


def test_tail_confined_truth_table() -> None:
    confined = lambda image, cond, text, sp: Flux3KleinSequenceLayout.from_lengths(
        {"image": image, "image_cond": cond, "text": text}, sp
    ).tail_confined()
    # SP4: text 80 pads to 128 (48-row tail over 32-row shards), 160 pads to 256 (96 over 64): the tail spills.
    assert confined(4096, 0, 80, 4) is False
    assert confined(4096, 0, 160, 4) is False
    for text in (240, 320, 400, 480, 560, 640, 4096):
        assert confined(4096, 0, text, 4) is True, text
    # SP2: every text multiple of 80 keeps its tail (< 64 rows) inside the last 64-row-or-larger shard.
    for text in (80, 160, 240, 320, 560, 4096):
        assert confined(4096, 0, text, 2) is True, text
        assert confined(4096, 80, text, 2) is True, text
    # SP1: trivially (one shard holds everything).
    assert confined(4096, 80, 80, 1) is True
    # A small reference at SP4: 80 rows pad to 128, 48-row tail over 32-row shards.
    assert confined(4096, 80, 320, 4) is False


def test_validation(expect_error) -> None:
    with expect_error(ValueError, "sp_factor"):
        Flux3KleinSequenceLayout.from_lengths({"image": 1, "image_cond": 0, "text": 1}, 0)
    with expect_error(ValueError, "exactly"):
        Flux3KleinSequenceLayout.from_lengths({"image": 1, "text": 1}, 1)
    with expect_error(ValueError, "negative"):
        Flux3KleinSequenceLayout.from_lengths({"image": 1, "image_cond": -1, "text": 1}, 1)
    layout = Flux3KleinSequenceLayout.from_lengths({"image": 64, "image_cond": 0, "text": 80}, 2)
    with expect_error(IndexError, "outside"):
        layout.source_row("text", 128)
    with expect_error(ValueError, "image rows"):
        Flux3KleinSequenceLayout.from_lengths({"image": 0, "image_cond": 0, "text": 80}, 2).image_out_indices()


def test_source_row_formula() -> None:
    """The documented formula, spelled out on the (4096, 80, 80) SP4 case where text and cond pads span devices 2-3."""
    layout = Flux3KleinSequenceLayout.from_lengths({"image": 4096, "image_cond": 80, "text": 80}, 4)
    assert (layout.rows, layout.rows_total, layout.total_padded) == (
        {"image": 1024, "image_cond": 32, "text": 32},
        1088,
        4352,
    )
    assert layout.source_row("image", 0) == 0
    assert layout.source_row("image", 1024) == 1088  # device 1's first image row
    assert layout.source_row("image_cond", 0) == 1024  # after device 0's image rows
    assert layout.source_row("image_cond", 32) == 1088 + 1024  # device 1's cond shard
    assert layout.source_row("text", 79) == 2 * 1088 + 1024 + 32 + 15  # device 2's text shard, row 15
    assert layout.source_row("text", 80) == 2 * 1088 + 1024 + 32 + 16  # the first text pad row (device 2)
    # The first padded mode is image_cond (80 -> 128): its first pad row is cond row 80 on device 2.
    assert torch.equal(layout.assembly_indices()[layout.total :], torch.full((96,), 2 * 1088 + 1024 + 16))
