# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
generalized_moe_gate LLK test.

DEST holds four 32x32 tiles and the op only touches face 0 of each: tile 0 the payload score it
emits, tile 1 a 16-bit id, tile 2 the score+bias sort key, tile 3 scratch. Every DEST tile is
packed back raw as uint16.

An SFPU column offset k names face rows 4*(k//4)..+3 at column parity (k>>1)&1, so a run stored
at the pair {lo, hi} covers four rows across all 16 columns. A grouped-gate group is a column
pair {2g, 2g+1}. Gate output lands at row 0, rank r in column r; columns 8-15 of that row hold
SrcB residue from the last copy4rows and are not part of the contract.
"""

import struct

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    EltwiseBinaryGolden,
    GeneralizedMoeGateGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    DestSync,
    MathFidelity,
    MathOperation,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    ACC_TO_DEST,
    APPROX_MODE,
    DEST_SYNC,
    GENERALIZED_MOE_GATE,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    TILE_COUNT,
)
from helpers.utils import passed_test

FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.UInt16)

MODE_GATE, MODE_BINARY, MODE_MOVE, MODE_RUN = 0, 1, 2, 3
MOVE_STEP1_HI, MOVE_COPY4ROWS = 2, 4
RUN_MERGE4_TOP8, RUN_COPY_TOPK_RUN, RUN_PLACE_FIELD, RUN_MERGE16 = 0, 1, 2, 3

SCORES, IDS, KEYS, INTERM = 0, 1, 2, 3

EPS = 0.5
SCALE = 2.5


def _bits(value):
    """The fp32 bit pattern Converter::as_float expects."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _zeros():
    return torch.zeros(16, 16, dtype=torch.bfloat16)


def _offset_cells(offset):
    """The 32 face cells an SFPU column offset names."""
    row_base = 4 * (offset // 4)
    return [
        (r, c)
        for r in range(row_base, row_base + 4)
        for c in range((offset >> 1) & 1, 16, 2)
    ]


def _run_cells(lo, hi):
    return _offset_cells(lo) + _offset_cells(hi)


def _run_slots(lo, hi):
    """The eight cells a stored run's ranks occupy, rank 0 first.

    The sort leaves ranks 0-3 on lanes 0, 8, 16 and 24 of the low half and ranks 4-7 on the same
    lanes of the high half, which is one column of the four rows each offset names.
    """
    return [(4 * (lo // 4) + r, (lo >> 1) & 1) for r in range(4)] + [
        (4 * (hi // 4) + r, (hi >> 1) & 1) for r in range(4)
    ]


def _tile(face):
    """A 32x32 tile in tiled order carrying `face` in face 0; the other three faces are zero."""
    tile = torch.zeros(1024, dtype=torch.bfloat16)
    tile[:256] = face.reshape(-1)
    return tile


def _faces(result):
    """Face 0 of each packed DEST tile, as [4, 16, 16] uint16 words."""
    words = torch.tensor(result, dtype=torch.int64) & 0xFFFF
    return words.reshape(4, 1024)[:, :256].reshape(4, 16, 16).to(torch.int32)


def _from_dst(words):
    """DEST 16-bit float words back to values. DEST holds them as {sign, mantissa(7), exponent(8)},
    the order device_print.py:_make_float decodes, and packing raw hands that layout through.
    """
    words = words.to(torch.int32) & 0xFFFF
    ieee = (words & 0x8000) | ((words & 0xFF) << 7) | ((words >> 8) & 0x7F)
    return ieee.to(torch.uint16).view(torch.bfloat16).to(torch.float32)


def _to_dst(values):
    """Values to DEST 16-bit float words, for a region the test loads through the integer path."""
    words = values.to(torch.bfloat16).view(torch.uint16).to(torch.int32)
    return (words & 0x8000) | ((words & 0x7F) << 8) | ((words >> 7) & 0xFF)


def _word_tile(words):
    return words.to(torch.int32).to(torch.uint16).view(torch.bfloat16).reshape(16, 16)


def _ids_face():
    return torch.arange(256, dtype=torch.int32).reshape(16, 16)


def _id_tile(ids):
    return ids.to(torch.uint16).view(torch.bfloat16)


def _config(
    gmg, tiles, src_b=None, math_op=MathOperation.Elwadd, fidelity=MathFidelity.HiFi4
):
    return TestConfig(
        "sources/generalized_moe_gate_test.cpp",
        FORMATS,
        templates=[
            gmg,
            MATH_OP(mathop=math_op),
            MATH_FIDELITY(fidelity),
            APPROX_MODE(ApproximationMode.No),
            ACC_TO_DEST(False),
            DEST_SYNC(DestSync.Half),
        ],
        runtimes=[TILE_COUNT(4), NUM_FACES()],
        variant_stimuli=StimuliConfig(
            torch.cat([_tile(t) for t in tiles]),
            DataFormat.Float16_b,
            _tile(_zeros() if src_b is None else src_b),
            DataFormat.Float16_b,
            DataFormat.UInt16,
            tile_count_A=4,
            tile_count_B=1,
            tile_count_res=4,
        ),
        dest_acc=DestAccumulation.No,
    )


def _run(*configurations):
    """Build every variant before running any of them.

    run() raises the compile-producer skip, so a test that builds inside the run loop never reaches
    its later variants and the consumer pass then looks for an ELF nobody produced.
    """
    for configuration in configurations:
        configuration.prepare()
    results = [_faces(configuration.run().result) for configuration in configurations]
    return results[0] if len(results) == 1 else results


def _gate_stimuli(seed):
    """A payload/bias pair whose bf16 sum hits 256 distinct keys.

    Everything is a multiple of 1/16 below magnitude 12, which bf16 holds exactly, so the FPU's
    payload+bias lands on the intended key and no two experts tie at the rank-8 cut. A tie there
    leaves the selection genuinely ambiguous and makes an exact comparison meaningless. The payload
    stays positive so the normalization denominator cannot approach zero and blow the weights up.
    """
    generator = torch.Generator().manual_seed(seed)
    key = (torch.randperm(256, generator=generator) - 128).to(torch.float32) / 16.0
    payload = torch.randint(1, 65, (256,), generator=generator).to(torch.float32) / 16.0
    return (
        payload.reshape(16, 16).to(torch.bfloat16),
        (key - payload).reshape(16, 16).to(torch.bfloat16),
        key.reshape(16, 16).to(torch.bfloat16),
    )


def _tagged_words(base):
    """A distinct DEST word per cell, so a moved datum can be traced back to where it came from.

    Built in the DEST layout directly: the low byte is the exponent, kept clear of 0 and 255 so a
    float load/store round trip neither flushes nor saturates it.
    """
    tags = base + torch.arange(256)
    return ((((tags // 200) % 128) << 8) | (tags % 200 + 20)).reshape(16, 16)


def _gate_output(faces):
    """The gate's answer: rank r at row 0 column r of the score and id regions."""
    return _from_dst(faces[SCORES])[0, :8], faces[IDS][0, :8]


@parametrize(topk=[4, 6, 8], softmax=[False, True])
def test_generalized_moe_gate(topk, softmax):
    payload, bias, keys = _gate_stimuli(seed=topk * 2 + int(softmax))
    ids = _ids_face()

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_GATE,
                topk=topk,
                softmax=softmax,
                eps=_bits(EPS),
                scale=_bits(SCALE),
            ),
            [payload, _id_tile(ids), _zeros(), _zeros()],
            src_b=bias,
        )
    )

    golden = get_golden_generator(GeneralizedMoeGateGolden)(
        keys, payload, ids, topk=topk, output_softmax=softmax, eps=EPS, scale=SCALE
    )
    got_weights, got_ids = _gate_output(faces)

    assert got_ids[:topk].tolist() == [
        int(i) for i in golden[1][:topk]
    ], f"wrong experts: {got_ids.tolist()} vs {golden[1].tolist()}"
    assert torch.allclose(
        got_weights[:topk], golden[0][:topk], rtol=2e-2, atol=1e-3
    ), f"weights differ:\n got {got_weights.tolist()}\n want {golden[0].tolist()}"

    # Ranks past topk are masked before the sum, so they must read back as an empty slot rather
    # than as whatever the sort left there.
    assert got_weights[topk:8].tolist() == [0.0] * (
        8 - topk
    ), "dropped ranks kept a weight"
    assert got_ids[topk:8].tolist() == [0] * (8 - topk), "dropped ranks kept an id"


def test_generalized_moe_gate_grouped():
    # Seed 128 leaves the eight group top-2 sums distinct with a wide margin at the top-4 cut.
    payload, bias, keys = _gate_stimuli(seed=128)
    ids = _ids_face()

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_GATE, grouped=True, eps=_bits(EPS), scale=_bits(SCALE)
            ),
            [payload, _id_tile(ids), _zeros(), _zeros()],
            src_b=bias,
        )
    )

    golden = get_golden_generator(GeneralizedMoeGateGolden)(
        keys, payload, ids, eps=EPS, scale=SCALE, grouped=True
    )

    # The grouped merge stops at "step 4 only", so its top-8 comes out in bitonic order rather
    # than sorted. Pair each weight with its own id and compare the two sets.
    got_weights, got_ids = _gate_output(faces)
    got = sorted(zip(got_ids.tolist(), got_weights.tolist()))
    want = sorted(zip([int(i) for i in golden[1]], golden[0].tolist()))
    assert [p[0] for p in got] == [
        p[0] for p in want
    ], f"wrong experts: {got} vs {want}"
    assert torch.allclose(
        torch.tensor([p[1] for p in got]),
        torch.tensor([p[1] for p in want]),
        rtol=2e-2,
        atol=1e-3,
    ), f"weights differ: {got} vs {want}"


def test_generalized_moe_gate_ties():
    generator = torch.Generator().manual_seed(5)
    key = torch.randint(0, 22, (256,), generator=generator).to(torch.float32) / 4.0
    payload = torch.randint(1, 9, (256,), generator=generator).to(torch.float32) / 4.0
    ids = _ids_face()

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(mode=MODE_GATE, eps=_bits(EPS), scale=_bits(SCALE)),
            [
                payload.reshape(16, 16).to(torch.bfloat16),
                _id_tile(ids),
                _zeros(),
                _zeros(),
            ],
            src_b=(key - payload).reshape(16, 16).to(torch.bfloat16),
        )
    )
    got_weights, got_ids = _gate_output(faces)

    # Which of the tied experts wins is not in the contract; the eight largest keys, distinct ids,
    # and a sum that still hits scale are.
    assert len(set(got_ids.tolist())) == 8, f"repeated expert: {got_ids.tolist()}"
    assert got_ids.max() < 256, f"id out of range: {got_ids.tolist()}"

    selected = key[got_ids.to(torch.int64)].sort(descending=True).values
    largest = key.sort(descending=True).values[:8]
    assert torch.equal(
        selected, largest
    ), f"selected keys are not the eight largest: {selected.tolist()} vs {largest.tolist()}"
    # Pair each weight with the payload at the id the device returned. A sort whose index tracking
    # desynced from the values returns the right keys with the wrong weights, and only this sees it.
    selected_payload = payload[got_ids.to(torch.int64)]
    want = selected_payload * (SCALE / (selected_payload.sum() + EPS))
    assert torch.allclose(
        got_weights, want, rtol=2e-2, atol=1e-3
    ), f"weights {got_weights.tolist()} do not normalize {selected_payload.tolist()}"


# The multi-block path stops at merge16_to_run, leaving a re-mergeable run and no normalize.
# idx_offset is the block's expert-id base. The metal path leaves it 0 and carries global ids in the
# id tile instead, so this is the only cover it has.
@parametrize(store=[(0, 2), (4, 6)], idx_offset=[0, 256])
def test_generalized_moe_gate_produce_run(store, idx_offset):
    store_lo, store_hi = store
    payload, bias, keys = _gate_stimuli(seed=23)
    ids = _ids_face()

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_GATE,
                produce_run=True,
                to_lo=store_lo,
                to_hi=store_hi,
                idx_offset=idx_offset,
                eps=_bits(EPS),
                scale=_bits(SCALE),
            ),
            [payload, _id_tile(ids), _zeros(), _zeros()],
            src_b=bias,
        )
    )

    winners = keys.reshape(-1).to(torch.float32).argsort(descending=True)[:8]
    slots = _run_slots(store_lo, store_hi)

    got_ids = [int(faces[IDS][r, c]) for r, c in slots]
    assert got_ids == [
        int(i) + idx_offset for i in winners
    ], f"run holds ids {got_ids}, want {[int(i) + idx_offset for i in winners]}"

    # A run is un-normalized: the scores it carries are still the raw payload of the winners.
    got_scores = [float(_from_dst(faces[SCORES])[r, c]) for r, c in slots]
    want_scores = payload.reshape(-1).to(torch.float32)[winners].tolist()
    assert got_scores == pytest.approx(
        want_scores, abs=2e-2
    ), f"run holds scores {got_scores}"


# Fidelity is not swept: the LLK reads it only to reject high fidelity for ELWMUL, and the mop it
# programs is the same either way.
@parametrize(
    math_op=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    reload=[False, True],
)
def test_generalized_moe_gate_binary(math_op, reload):
    generator = torch.Generator().manual_seed(31)

    def values():
        raw = (
            torch.randint(-64, 64, (256,), generator=generator).to(torch.float32) / 16.0
        )
        return raw.reshape(16, 16).to(torch.bfloat16)

    # COPY takes SrcA from the unpacker and drops a copy in the score region on the way; RELOAD
    # reads it back out. Seeding that region with something other than the unpacked operand is what
    # makes the two distinguishable.
    src_a, src_b, seed = values(), values(), values()
    fidelity = (
        MathFidelity.LoFi if math_op == MathOperation.Elwmul else MathFidelity.HiFi4
    )

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(mode=MODE_BINARY, reload=reload),
            [src_a, _word_tile(_to_dst(seed)), _zeros(), _zeros()],
            src_b=src_b,
            math_op=math_op,
            fidelity=fidelity,
        )
    )

    operand_a = seed if reload else src_a
    want = get_golden_generator(EltwiseBinaryGolden)(
        math_op,
        operand_a.reshape(-1),
        src_b.reshape(-1),
        DataFormat.Float16_b,
        fidelity,
        input_format=DataFormat.Float16_b,
    )

    assert torch.equal(
        _from_dst(faces[SCORES]).to(torch.bfloat16), operand_a
    ), "the score region does not hold the operand this mode sources SrcA from"
    got = _from_dst(faces[KEYS]).reshape(-1).to(torch.bfloat16)
    if math_op == MathOperation.Elwmul:
        # LoFi truncates both mantissas, so the product lands within a bf16 ulp of the golden.
        assert passed_test(
            want, got, DataFormat.Float16_b, custom_atol=0.07, custom_rtol=0.0
        ), "the product did not reach the key region"
    else:
        assert torch.equal(got, want), "the sum did not reach the key region"


@parametrize(src_dst=[(0, 4), (4, 8), (8, 12), (12, 0)])
def test_generalized_moe_gate_copy4rows(src_dst):
    # A single parametrize argname arrives wrapped; pytest force-tuples the value.
    (src, dst) = src_dst[0]
    tags = [_tagged_words(base) for base in (0, 1000, 2000, 3000)]
    tiles = [_word_tile(t) for t in tags]

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_MOVE, sub_op=MOVE_COPY4ROWS, row_src=src, row_dst=dst
            ),
            tiles,
        )
    )

    for region in (SCORES, IDS, KEYS):
        want = tags[region].clone()
        want[dst : dst + 4] = tags[region][src : src + 4]
        assert torch.equal(
            faces[region], want
        ), f"region {region} rows {dst}..{dst + 3} are wrong"
    assert torch.equal(
        faces[INTERM], tags[INTERM]
    ), "copy4rows touched the scratch region"


def test_generalized_moe_gate_copy4rows_back_to_back():
    tags = [_tagged_words(base) for base in (0, 1000, 2000, 3000)]
    tiles = [_word_tile(t) for t in tags]

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_MOVE,
                sub_op=MOVE_COPY4ROWS,
                row_src=4,
                row_dst=8,
                srcb=16,
                second_copy=True,
                row_src_2=0,
                row_dst_2=12,
                srcb_2=20,
            ),
            tiles,
        )
    )

    for region in (SCORES, IDS, KEYS):
        want = tags[region].clone()
        want[8:12] = tags[region][4:8]
        want[12:16] = tags[region][0:4]
        assert torch.equal(faces[region], want), f"region {region} is wrong"
    assert torch.equal(
        faces[INTERM], tags[INTERM]
    ), "copy4rows touched the scratch region"


@parametrize(knobs=[(0, 8), (4, 0), (4, 8)])
def test_generalized_moe_gate_step1_hi(knobs):
    """step1_hi's two knobs pick which four rows it reads and where it writes the result.

    The layout it produces is not specified anywhere outside the instruction sequence, so rather
    than pin that, this compares against the same op with both knobs at zero over an image
    shifted to match.
    """
    d2b_dst, b2d_base = knobs[0]
    tags = [_tagged_words(base) for base in (0, 1000, 2000, 3000)]
    tiles = [_word_tile(t) for t in tags]
    shifted = [_word_tile(torch.roll(t, shifts=-d2b_dst, dims=0)) for t in tags]

    faces, reference = _run(
        _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_MOVE,
                sub_op=MOVE_STEP1_HI,
                d2b_dst=d2b_dst,
                b2d_base=b2d_base,
            ),
            tiles,
        ),
        _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_MOVE, sub_op=MOVE_STEP1_HI, d2b_dst=0, b2d_base=0
            ),
            shifted,
        ),
    )

    for region in (SCORES, IDS, KEYS):
        got = faces[region][b2d_base : b2d_base + 8]
        assert torch.equal(
            got, reference[region][0:8]
        ), f"region {region} does not match the shifted reference"

        # Columns 4-11 of the run are left undefined by the MOP, so only the defined ones carry a
        # claim: between them they hold exactly the even-column datums of the four selected rows.
        defined = got[:, list(range(4)) + list(range(12, 16))]
        source = tags[region][d2b_dst : d2b_dst + 4][:, 0::2]
        assert set(defined.flatten().tolist()) == set(
            source.flatten().tolist()
        ), f"region {region} run was not built from rows {d2b_dst}..{d2b_dst + 3}"

        # The gate parks groups 4-7 outside the run's window and expects them back, so a MOP that
        # wrote more than eight rows would break it without any of the above noticing.
        outside = [r for r in range(16) if not b2d_base <= r < b2d_base + 8]
        assert torch.equal(
            faces[region][outside], tags[region][outside]
        ), f"region {region} was written outside rows {b2d_base}..{b2d_base + 7}"
    assert torch.equal(
        faces[INTERM], tags[INTERM]
    ), "step1_hi touched the scratch region"


@parametrize(
    placement=[(0, 2, 4, 6), (0, 2, 8, 10), (4, 6, 0, 2), (0, 4, 8, 12)],
    after_mop=lambda placement: [False, True] if placement == (0, 2, 4, 6) else [False],
)
def test_generalized_moe_gate_relocate_run(placement, after_mop):
    from_lo, from_hi, to_lo, to_hi = placement
    tags = [_tagged_words(base) for base in (0, 1000, 2000, 3000)]
    tiles = [_word_tile(t) for t in tags]

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_RUN,
                sub_op=RUN_COPY_TOPK_RUN,
                from_lo=from_lo,
                from_hi=from_hi,
                to_lo=to_lo,
                to_hi=to_hi,
                pre_copy4rows=after_mop,
                row_src=0,
                row_dst=12,
            ),
            tiles,
        )
    )

    for region in (SCORES, IDS, KEYS):
        want = tags[region].clone()
        if after_mop:
            want[12:16] = tags[region][0:4]
        for (sr, sc), (dr, dc) in zip(
            _run_cells(from_lo, from_hi), _run_cells(to_lo, to_hi)
        ):
            want[dr, dc] = want[sr, sc] if after_mop else tags[region][sr, sc]
        assert torch.equal(faces[region], want), f"region {region} is wrong"
    assert torch.equal(
        faces[INTERM], tags[INTERM]
    ), "copy_topk_run touched the scratch region"


@parametrize(
    field=[0, 1, 2],
    src=[(0, 4), (8, 12)],
    dst=[(0, 2), (4, 6)],
    after_mop=lambda field, src, dst: (
        [False, True] if (field, src, dst) == (0, (0, 4), (0, 2)) else [False]
    ),
)
def test_generalized_moe_gate_place_field(field, src, dst, after_mop):
    src_lo, src_hi = src
    dst_lo, dst_hi = dst
    tags = [_tagged_words(base) for base in (0, 1000, 2000, 3000)]
    tiles = [_word_tile(t) for t in tags]
    home = {0: KEYS, 1: IDS, 2: SCORES}[field]

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_RUN,
                sub_op=RUN_PLACE_FIELD,
                field=field,
                from_lo=src_lo,
                from_hi=src_hi,
                to_lo=dst_lo,
                to_hi=dst_hi,
                pre_copy4rows=after_mop,
                row_src=0,
                row_dst=12,
            ),
            tiles,
        )
    )

    want = tags[home].clone()
    if after_mop:
        want[12:16] = tags[home][0:4]
    for (sr, sc), (dr, dc) in zip(
        _run_cells(src_lo, src_hi), _run_cells(dst_lo, dst_hi)
    ):
        want[dr, dc] = tags[INTERM][sr, sc]
    assert torch.equal(faces[home], want), f"field {field} landed wrong"

    for region in (SCORES, IDS, KEYS, INTERM):
        if region == home:
            continue
        untouched = tags[region].clone()
        if after_mop and region != INTERM:
            untouched[12:16] = tags[region][0:4]
        assert torch.equal(
            faces[region], untouched
        ), f"field {field} also wrote region {region}"


# The gate-level cover above cannot run under ttsim, and this can, so it is the one that will catch
# a regression in the offset add before silicon does. Comparing two runs of the same stimulus needs
# no golden and no knowledge of where in the run a rank lands.
def test_generalized_moe_gate_idx_offset():
    tags = [_tagged_words(base) for base in (0, 1000, 2000, 3000)]
    tiles = [_word_tile(t) for t in tags]

    def config(idx_offset):
        return _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_RUN,
                sub_op=RUN_MERGE16,
                to_lo=0,
                to_hi=2,
                idx_offset=idx_offset,
            ),
            tiles,
        )

    base, shifted = _run(config(0), config(256))
    slots = _run_slots(0, 2)

    base_ids = [int(base[IDS][r, c]) for r, c in slots]
    shifted_ids = [int(shifted[IDS][r, c]) for r, c in slots]
    assert shifted_ids == [
        i + 256 for i in base_ids
    ], f"idx_offset did not shift the ids: {base_ids} -> {shifted_ids}"

    for region in (SCORES, KEYS, INTERM):
        assert torch.equal(
            base[region], shifted[region]
        ), f"idx_offset also wrote region {region}"
