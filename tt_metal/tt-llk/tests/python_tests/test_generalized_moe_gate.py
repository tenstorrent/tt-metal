# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
generalized_moe_gate LLK test.

DEST holds four 32x32 tiles and the op only touches face 0 of each: tile 0 the payload score it
emits, tile 1 a 16-bit id, tile 2 the score+bias sort key, tile 3 scratch. Every DEST tile is
packed back raw as uint16.

Both of the op's front-ends transpose the payload and neither transposes the bias or the ids, so
expert i's score sits at the transposed position while its id does not. Gate stimuli here are
written in the gate's own frame -- expert i at flat position i -- and _payload_tile puts the payload
back into the frame L1 has to hold. Getting this wrong still selects the right experts and reports
the wrong scores for them, which is why the weights are compared and not just the ids.

An SFPU column offset k names face rows 4*(k//4)..+3 at column parity (k>>1)&1, so a run stored
at the pair {lo, hi} covers four rows across all 16 columns. A grouped-gate group is a column
pair {2g, 2g+1}. Gate output lands at row 0, rank r in column r; columns 8-15 of that row hold
SrcB residue from the last copy4rows and are not part of the contract.

A merge is eight independent 16-element sorts running side by side, one per column pair: instance
c reads DEST rows 0-7 of columns {2c, 2c+1} and writes its ranks back to those same two columns.
The gate reads instance 0, which is why a run's other 56 cells look like residue; the combine
tests drive all eight, since they are the same network over different columns.

Where a MOP's DEST layout is derivable from its instruction sequence, the test pins it outright
(step0, step2). step1_hi is the exception and stays differential: its run occupies eight rows of
which only columns 0-3 and 12-15 are defined, so there is no full layout to pin.

The transpose MOP runners take no dst_index, so what tile they address depends on the DEST offset
left by whatever ran before them; the after_mop axis on relocate_run and place_field sequences an
SFPU call ahead of one to hold that. Their MOVB2D also has to keep datums with a zero low byte,
which is what carries a small expert id through copy4rows intact.
"""

import struct

import pytest
import torch
from conftest import skip_for_quasar
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
from helpers.test_config import StimuliMode, TestConfig
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

# There is no Quasar implementation of this op, so the includes the kernel needs do not exist there.
pytestmark = skip_for_quasar

FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.UInt16)

MODE_GATE, MODE_BINARY, MODE_MOVE, MODE_RUN = 0, 1, 2, 3
MOVE_STEP0, MOVE_STEP1, MOVE_STEP1_HI, MOVE_STEP2, MOVE_COPY4ROWS = 0, 1, 2, 3, 4
RUN_MERGE4_TOP8, RUN_COPY_TOPK_RUN, RUN_PLACE_FIELD, RUN_MERGE16 = 0, 1, 2, 3
RUN_COMBINE, RUN_COMBINE_RELOCATED, RUN_COMBINE_FINALIZE = 4, 5, 6

SCORES, IDS, KEYS, INTERMEDIATE = 0, 1, 2, 3

EPS = 0.5
SCALE = 2.5


def _bits(value):
    """The fp32 bit pattern Converter::as_float expects."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _one(argument):
    """Unwrap a single-argname parametrize value.

    `parametrize` always hands pytest a list of tuples, and pytest passes a one-element tuple
    through as the value itself when there is only one argname. Every single-axis test in the repo
    unwraps it by hand; this just names the reason once.
    """
    return argument[0]


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


def _run_slots(lo, hi, instance=0):
    """The eight cells a stored run's ranks occupy, rank 0 first.

    A merge runs eight independent 16-element sorts side by side. Instance c takes its candidates
    from DEST columns {2c, 2c+1} and writes its ranks back to the same two columns, so a run stored
    at {lo, hi} holds eight separate answers and instance 0 is the one the gate's output comes from.
    Rank j lands at row (lo & ~3) + j in the low half, rank 4+j at (hi & ~3) + j in the high.
    """
    return [((lo & ~3) + j, 2 * instance + ((lo >> 1) & 1)) for j in range(4)] + [
        ((hi & ~3) + j, 2 * instance + ((hi >> 1) & 1)) for j in range(4)
    ]


def _tile(face):
    """A 32x32 tile in tiled order: a 16x16 face lands in face 0 with the rest zero, and a full
    1024-element tile passes through for the tests that populate all four faces."""
    face = face.reshape(-1)
    if face.numel() == 1024:
        return face.to(torch.bfloat16)
    tile = torch.zeros(1024, dtype=torch.bfloat16)
    tile[:256] = face
    return tile


def _all_faces(result, tiles=4):
    """Every packed face, as [tile, face, 16, 16] uint16 words.

    Packing writes num_faces faces per tile and leaves the rest of the tile stride in L1 alone, so
    the harness hands back num_faces faces and the face axis follows the result length.
    """
    words = torch.tensor(result, dtype=torch.int64) & 0xFFFF
    return words.reshape(tiles, -1, 16, 16).to(torch.int32)


def _sections(result):
    """Face 0 of each DEST tile, per section: [section, 4, 16, 16].

    A two-section run packs the lower DEST half to tiles 0-3 and the upper to 4-7.
    """
    return _all_faces(result, tiles=8)[:, 0].reshape(2, 4, 16, 16)


def _faces(result):
    """Face 0 of each packed DEST tile, as [4, 16, 16] uint16 words.

    The gate only ever touches face 0, so everything but the binary reads back through here.
    """
    return _all_faces(result)[:, 0]


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
    """A DEST word image as a stimulus, carried by bit pattern rather than by value."""
    tile = words.to(torch.int32).to(torch.uint16).view(torch.bfloat16)
    # The harness packs stimuli through float32, which does not preserve a NaN payload, so a word
    # that reads as NaN reaches the device as some other word. _to_dst is the way in: it rotates
    # the exponent into the low byte, and any value >= 2 whose bf16 mantissa is all ones comes out
    # with an exponent of 0xFF.
    #
    # Nothing here trips it today -- _tagged_words tops out at a high byte of 16, and every _to_dst
    # stimulus is either below 2.0 or a multiple of 1/8, which cannot reach mantissa 0x7F. This is
    # here so that a future stimulus fails loudly instead of silently reaching the device altered.
    assert torch.isfinite(
        tile.to(torch.float32)
    ).all(), "DEST word image holds a bit pattern that reads as NaN or Inf; it will not survive packing"
    return tile


def _ids_face():
    return torch.arange(256, dtype=torch.int32).reshape(16, 16)


def _id_tile(ids):
    return ids.to(torch.uint16).view(torch.bfloat16)


def _config(
    gmg,
    tiles,
    src_b=None,
    math_op=MathOperation.Elwadd,
    fidelity=MathFidelity.HiFi4,
    approx=ApproximationMode.No,
    dest_sync=DestSync.Half,
    acc_to_dest=False,
    num_faces=4,
):
    return TestConfig(
        "sources/generalized_moe_gate_test.cpp",
        FORMATS,
        templates=[
            gmg,
            MATH_OP(mathop=math_op),
            MATH_FIDELITY(fidelity),
            APPROX_MODE(approx),
            ACC_TO_DEST(acc_to_dest),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[TILE_COUNT(4), NUM_FACES(num_faces, num_faces, num_faces)],
        variant_stimuli=StimuliConfig(
            torch.cat([_tile(t) for t in tiles]),
            DataFormat.Float16_b,
            _tile(_zeros() if src_b is None else src_b),
            DataFormat.Float16_b,
            DataFormat.UInt16,
            tile_count_A=4,
            tile_count_B=1,
            tile_count_res=4 * gmg.sections,
            num_faces=num_faces,
        ),
        dest_acc=DestAccumulation.No,
    )


def _run(*configurations, view=_faces):
    """Build every variant before running any of them.

    run() raises the compile-producer skip, so a test that builds inside the run loop never reaches
    its later variants and the consumer pass then looks for an ELF nobody produced.
    """
    if len(configurations) > 1 and TestConfig.STIMULI_MODE != StimuliMode.INLINE:
        # StimuliConfig.save_to_cache and GeneratorProxy both key on sha256(PYTEST_CURRENT_TEST),
        # i.e. one cache slot per test node, so a test with several configurations would have them
        # all collapse onto whichever ran first. Under --stimuli-only it is worse: the first run()
        # raises the cache-and-skip, so the later configurations never get written at all. The
        # differential tests here are exactly the ones that would then pass vacuously, comparing a
        # stimulus against itself, so refuse rather than report a green.
        pytest.skip(
            "multi-configuration test cannot use the per-test-node stimuli cache "
            "(--stimuli-only / --use-stimuli); see tt-llk stimuli cache keying"
        )
    for configuration in configurations:
        configuration.prepare()
    results = [view(configuration.run().result) for configuration in configurations]
    return results[0] if len(results) == 1 else results


def _payload_tile(payload):
    """The L1 tile whose unpacked form is `payload`.

    The gate's non-sigmoid front-end unpacks SrcA under ckernel::Transpose::Both, so the datum it
    scores as expert i is the transpose of what L1 holds. Every gate stimulus below is written in
    the frame the gate works in -- expert i at flat position i, matching the id tile, which is
    unpacked without a transpose -- and this puts it back into the frame L1 has to hold.
    """
    return payload.t().contiguous()


def _gate_tiles(payload, ids):
    """The four DEST tiles a gate run starts from: the payload in the frame L1 has to hold, the ids
    untransposed alongside it, and the key and scratch regions the op fills itself."""
    return [_payload_tile(payload), _id_tile(ids), _zeros(), _zeros()]


def _gate_stimuli(seed):
    """A payload/bias pair whose bf16 sum hits 256 distinct keys.

    Everything is a multiple of 1/16 below magnitude 12, which bf16 holds exactly, so the FPU's
    payload+bias lands on the intended key and no two experts tie at the rank-8 cut. A tie there
    leaves the selection genuinely ambiguous and makes an exact comparison meaningless. The payload
    stays positive so the normalization denominator cannot approach zero and blow the weights up.

    Returned in the gate's frame: `payload` is the score the gate emits for expert i, not the tile
    to upload -- see _payload_tile. Transposing the payload only rearranges that value set, so the
    exactness argument above is unaffected. The bias is not transposed on its way in, so it is built
    against the already-transposed payload and needs no correction of its own.
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


def _tag_tiles():
    """The four tagged DEST regions, as the word images to check against and the tiles to upload.

    The bases are 1000 apart so a datum that crosses regions is traceable to the one it came from.
    """
    tags = [_tagged_words(base) for base in (0, 1000, 2000, 3000)]
    return tags, [_word_tile(t) for t in tags]


def _gate_output(faces):
    """The gate's answer: rank r at row 0 column r of the score and id regions."""
    return _from_dst(faces[SCORES])[0, :8], faces[IDS][0, :8]


# Approximation mode reaches only the normalization tail — sfpu_reciprocal for the sum, and the
# bf16 exp when output_softmax is set. Selection is a bitonic compare network either way, so the
# expert ids are exact in both modes and only the weights get the looser bound.
def _weight_tolerance(approx):
    return (
        dict(rtol=5e-2, atol=1e-2)
        if approx == ApproximationMode.Yes
        else dict(rtol=2e-2, atol=1e-3)
    )


def _assert_gate_output(faces, golden, topk, ordered=True, **tolerance):
    """Compare the gate's row-0 output against a golden [2, 8] of weights and ids.

    ordered=False for the grouped path, whose merge stops at "step 4 only" and so emits its top-8 in
    bitonic rather than sorted order: pair each weight with its own id and compare the two sets.
    """
    got_weights, got_ids = _gate_output(faces)
    got = list(zip(got_ids[:topk].tolist(), got_weights[:topk].tolist()))
    want = list(zip([int(i) for i in golden[1][:topk]], golden[0][:topk].tolist()))
    if not ordered:
        got, want = sorted(got), sorted(want)

    assert [p[0] for p in got] == [
        p[0] for p in want
    ], f"wrong experts: {got} vs {want}"
    assert torch.allclose(
        torch.tensor([p[1] for p in got]),
        torch.tensor([p[1] for p in want]),
        **tolerance,
    ), f"weights differ: {got} vs {want}"

    # Ranks past topk are masked before the sum, so they must read back as an empty slot rather
    # than as whatever the sort left there.
    assert got_weights[topk:8].tolist() == [0.0] * (
        8 - topk
    ), "dropped ranks kept a weight"
    assert got_ids[topk:8].tolist() == [0] * (8 - topk), "dropped ranks kept an id"


# topk is {4, 6, 8} and not a free parameter: finalize_ungrouped static_asserts it, because its
# rank-mask is only correct for those three. 1-3 would fall into the `topk <= 4` branch and silently
# keep four ranks, and 5/7 take the masked branch untested.
#
# Approximation mode reaches only the normalization tail, so it is crossed with softmax (whose exp it
# changes) rather than with every topk.
@parametrize(
    topk=[4, 6, 8],
    softmax=[False, True],
    approx=lambda topk: (
        [ApproximationMode.No, ApproximationMode.Yes]
        if topk == 8
        else [ApproximationMode.No]
    ),
)
def test_generalized_moe_gate(topk, softmax, approx):
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
            _gate_tiles(payload, ids),
            src_b=bias,
            approx=approx,
        )
    )

    golden = get_golden_generator(GeneralizedMoeGateGolden)(
        keys, payload, ids, topk=topk, output_softmax=softmax, eps=EPS, scale=SCALE
    )
    _assert_gate_output(faces, golden, topk, **_weight_tolerance(approx))


def _sigmoid_stimuli(seed, grouped):
    """A payload/bias pair for the sigmoid front-end, with eight winners the bias alone decides.

    The gate transposes the payload and activates it but takes the bias straight from SrcB, so the
    sort key is sigmoid(payload.T) + bias. Winners are spaced two apart in the bias and everything
    else sits below -1, which is more than the at-most-1 an activation can contribute: the ranking is
    then unambiguous however coarse the sigmoid turns out to be, and the activation only has to be
    right in the weights.

    Grouped puts two winners in each of four groups, so those four hold by far the largest top-2 key
    sums and the global top-8 is exactly the eight marked cells.
    """
    generator = torch.Generator().manual_seed(seed)
    payload = (
        torch.randint(-32, 33, (256,), generator=generator).to(torch.float32) / 8.0
    ).reshape(16, 16)
    bias = (
        torch.randint(-64, -8, (256,), generator=generator).to(torch.float32) / 8.0
    ).reshape(16, 16)

    if grouped:
        groups = torch.randperm(8, generator=generator)[:4]
        rows = torch.randperm(16, generator=generator)[:8]
        cells = [
            (int(rows[2 * n + i]), int(2 * groups[n] + i))
            for n in range(4)
            for i in range(2)
        ]
    else:
        cells = [
            (int(flat) // 16, int(flat) % 16)
            for flat in torch.randperm(256, generator=generator)[:8]
        ]
    for rank, (row, column) in enumerate(cells):
        bias[row, column] = 40.0 - 2.0 * rank
    return payload.to(torch.bfloat16), bias.to(torch.bfloat16)


# The op's enable_sigmoid front-end, which nothing else reaches: transpose_wh_tile, then sigmoid_tile,
# then a RELOAD binary that takes SrcA back out of DEST through MOVD2A while the unpacker feeds only
# SrcB under DEST_TO_SRCA reuse. The plain path never instantiates that unpack configuration, so this
# is also the only cover of RELOAD as production actually drives it.
#
# The bias picks the winners and is not transposed; the payload is, so a transpose that did not
# happen still selects the right experts and returns the wrong weights for them.
#
# grouped is crossed in because sigmoid + grouped is the DeepSeek gate, i.e. the configuration the
# grouped path exists for, and the two front-ends had never met. Grouped pins topk to 8 itself, so
# there is nothing to sweep there.
@parametrize(grouped=[False, True], topk=lambda grouped: [8] if grouped else [4, 8])
def test_generalized_moe_gate_sigmoid(grouped, topk):
    payload, bias = _sigmoid_stimuli(seed=77, grouped=grouped)
    ids = _ids_face()

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_GATE,
                sigmoid=True,
                grouped=grouped,
                topk=topk,
                eps=_bits(EPS),
                scale=_bits(SCALE),
            ),
            # Not _payload_tile: the sigmoid front-end does its transpose in transpose_wh_tile on
            # the raw L1 tile, so this path uploads the payload as-is and the golden transposes.
            [payload, _id_tile(ids), _zeros(), _zeros()],
            src_b=bias,
        )
    )

    # The emitted score is the activation, not the raw payload, and it is read at the transposed
    # position: both claims land in the weight comparison. The key is the activation plus the
    # untransposed bias, which is what makes the winners' ids come back untransposed.
    activated = torch.sigmoid(payload.to(torch.float32).t()).to(torch.bfloat16)
    golden = get_golden_generator(GeneralizedMoeGateGolden)(
        activated.to(torch.float32) + bias.to(torch.float32),
        activated,
        ids,
        topk=topk,
        eps=EPS,
        scale=SCALE,
        grouped=grouped,
    )
    # Looser than _weight_tolerance's approx bound: this compares against torch.sigmoid, so the
    # activation's own error is in the budget on top of the normalization's.
    _assert_gate_output(faces, golden, topk, ordered=not grouped, rtol=5e-2, atol=1e-2)


# The grouped answer is well defined only when the eight group top-2 sums are pairwise distinct, so
# that which four groups survive is unambiguous. Most seeds tie somewhere; 128 does not.
@parametrize(seed=[128], approx=[ApproximationMode.No, ApproximationMode.Yes])
def test_generalized_moe_gate_grouped(seed, approx):
    payload, bias, keys = _gate_stimuli(seed=seed)
    ids = _ids_face()

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_GATE, grouped=True, eps=_bits(EPS), scale=_bits(SCALE)
            ),
            _gate_tiles(payload, ids),
            src_b=bias,
            approx=approx,
        )
    )

    golden = get_golden_generator(GeneralizedMoeGateGolden)(
        keys, payload, ids, eps=EPS, scale=SCALE, grouped=True
    )
    _assert_gate_output(faces, golden, 8, ordered=False, **_weight_tolerance(approx))


# The op ships at DstSync::SyncFull over a single face; every test here otherwise runs SyncHalf over
# four. Neither reaches the gate's arithmetic, so the answer has to be the one the golden already
# pins. The SyncHalf and four-face corners are controls: they keep a failure attributable to the
# axis that moved rather than to the seed.
@parametrize(dest_sync=[DestSync.Half, DestSync.Full], num_faces=[1, 4])
def test_generalized_moe_gate_shipping_config(dest_sync, num_faces):
    payload, bias, keys = _gate_stimuli(seed=61)
    ids = _ids_face()

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(mode=MODE_GATE, eps=_bits(EPS), scale=_bits(SCALE)),
            _gate_tiles(payload, ids),
            src_b=bias,
            dest_sync=dest_sync,
            num_faces=num_faces,
        )
    )

    golden = get_golden_generator(GeneralizedMoeGateGolden)(
        keys, payload, ids, eps=EPS, scale=SCALE
    )
    _assert_gate_output(faces, golden, 8, **_weight_tolerance(ApproximationMode.No))


# eps and scale are the normalization's two knobs and every other gate test pins them at 0.5/2.5.
# (0, 1) is both the default the API declares and what a caller wanting a plain softmax passes: it
# is the one setting where the weights have to sum to exactly 1, and the only one where the
# reciprocal sees the sum unpadded.
@parametrize(norm=[(0.0, 1.0), (EPS, SCALE)], softmax=[False, True])
def test_generalized_moe_gate_normalization(norm, softmax):
    eps, scale = norm
    payload, bias, keys = _gate_stimuli(seed=19)
    ids = _ids_face()

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_GATE,
                softmax=softmax,
                eps=_bits(eps),
                scale=_bits(scale),
            ),
            _gate_tiles(payload, ids),
            src_b=bias,
        )
    )

    golden = get_golden_generator(GeneralizedMoeGateGolden)(
        keys, payload, ids, output_softmax=softmax, eps=eps, scale=scale
    )
    _assert_gate_output(faces, golden, 8, **_weight_tolerance(ApproximationMode.No))

    # With no eps padding the denominator is the sum itself, so the weights are a distribution
    # scaled by `scale`. That is the claim callers actually rely on, and it holds without a golden.
    got_weights, _ = _gate_output(faces)
    if eps == 0.0:
        assert float(got_weights.sum()) == pytest.approx(
            scale, rel=2e-2
        ), f"weights sum to {float(got_weights.sum())}, not {scale}"


# The softmax path subtracts rank 0's score before the exp, and finalize_ungrouped's own comment says
# why that is required rather than cosmetic: score_func="softmax" feeds the op raw router logits,
# which are unbounded, and exp of a large logit saturates bf16 to inf -> reciprocal 0 -> NaN weights.
#
# That reasoning holds only when rank 0 carries the largest score, i.e. when the sort key IS the
# emitted score -- so only when the bias is zero. Every other softmax test here runs a non-zero bias
# (the key is payload+bias, so rank 0's payload is not the maximum) with scores small enough that
# nothing could overflow either way. This is the regime the max-subtraction was written for: bias
# zero, logits large enough that removing the subtraction saturates.
def test_generalized_moe_gate_softmax_logits():
    generator = torch.Generator().manual_seed(13)
    # Eight winners near the top of what exp can take, spaced 0.5 apart: bf16's step at that
    # magnitude is exactly 0.5, so the logits are exact and the ranking is unambiguous. exp(95.5)
    # is inf in bf16, so a kernel that skipped the subtraction returns NaN here rather than weights.
    logits = torch.randint(0, 65, (256,), generator=generator).to(torch.float32) / 2.0
    winners = torch.randperm(256, generator=generator)[:8]
    logits[winners] = torch.tensor([95.5 - 0.5 * rank for rank in range(8)])
    payload = logits.reshape(16, 16).to(torch.bfloat16)
    ids = _ids_face()

    faces = _run(
        _config(
            # eps 0 and scale 1, so the answer is a plain softmax over the selected logits.
            GENERALIZED_MOE_GATE(
                mode=MODE_GATE, softmax=True, eps=_bits(0.0), scale=_bits(1.0)
            ),
            # src_b defaults to zero: with no bias the sort key is the logit itself.
            _gate_tiles(payload, ids),
        )
    )

    golden = get_golden_generator(GeneralizedMoeGateGolden)(
        payload, payload, ids, output_softmax=True, eps=0.0, scale=1.0
    )
    _assert_gate_output(faces, golden, 8, **_weight_tolerance(ApproximationMode.No))

    # Independent of the golden: the eight winners are known by construction, so this is softmax of
    # eight known logits and nothing about the kernel's shift is assumed.
    got_weights, got_ids = _gate_output(faces)
    assert got_ids.tolist() == winners.tolist(), f"wrong experts: {got_ids.tolist()}"
    assert torch.allclose(
        got_weights,
        torch.softmax(logits[winners].to(torch.bfloat16).to(torch.float32), dim=0),
        rtol=2e-2,
        atol=1e-3,
    ), f"weights are not the softmax of the selected logits: {got_weights.tolist()}"


def test_generalized_moe_gate_ties():
    generator = torch.Generator().manual_seed(5)
    key = torch.randint(0, 22, (256,), generator=generator).to(torch.float32) / 4.0
    payload = torch.randint(1, 9, (256,), generator=generator).to(torch.float32) / 4.0
    ids = _ids_face()

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(mode=MODE_GATE, eps=_bits(EPS), scale=_bits(SCALE)),
            _gate_tiles(payload.reshape(16, 16).to(torch.bfloat16), ids),
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
# id tile instead, so this is the only cover it has. 1792 is the largest legal value: the offset goes
# through SFPIADD's sign-extended 12-bit immediate, which the LLK static_asserts below 2048, and 1792
# is the last 256-aligned block base under it (block 7 of the 8-block ceiling).
@parametrize(store=[(0, 2), (4, 6)], idx_offset=[0, 256, 1792])
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
            _gate_tiles(payload, ids),
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


def _binary_stimuli(seed):
    """SrcA, SrcB, and the two DEST regions the kernel seeds, as whole tiles.

    Whole tiles rather than face 0 alone: the mop's outer loop runs once per face, so checking only
    face 0 would leave three quarters of a four-face op unexercised.

    Non-zero quarters of magnitude at most 2, which is deliberate. LoFi truncates SrcA to five
    significant bits and SrcB to seven and these need three, so the multiply is exact, and its
    product plus any accumulate onto it stays under 8 and lands on a bf16 value. That buys equality
    rather than a tolerance for all three ops, and keeps the checks off the question of whether the
    simulator models LoFi at all. Zero is excluded because EltwiseBinaryGolden returns -2^-126 for
    a product that should be signed zero, which only an exact comparison would ever notice.
    """
    generator = torch.Generator().manual_seed(seed)

    def values():
        magnitude = torch.randint(1, 9, (1024,), generator=generator).to(torch.float32)
        sign = (
            torch.randint(0, 2, (1024,), generator=generator).to(torch.float32) * 2 - 1
        )
        return (sign * magnitude / 4.0).to(torch.bfloat16)

    return values(), values(), values(), values()


def _binary_want(math_op, operand_a, src_b, fidelity):
    return get_golden_generator(EltwiseBinaryGolden)(
        math_op,
        operand_a.reshape(-1),
        src_b.reshape(-1),
        DataFormat.Float16_b,
        fidelity,
        input_format=DataFormat.Float16_b,
    )


def _binary_fidelity(math_op):
    # The LLK reads fidelity only to reject high fidelity for ELWMUL; the mop it programs is the
    # same either way, so this picks the one legal value rather than sweeping.
    return MathFidelity.LoFi if math_op == MathOperation.Elwmul else MathFidelity.HiFi4


def _binary_key_region(math_op, operand_a, src_b, fidelity, acc_base, acc_to_dest):
    """What the key region should hold once the binary has run.

    ELWMUL accumulates onto whatever the region already held, whatever acc_to_dest says. Bit 21 of
    the instruction word carries AddDst for ELWADD and ELWSUB but is unallocated for ELWMUL, so the
    value the LLK passes has nowhere to land, and the ISA defines the instruction as Dst += SrcA *
    SrcB outright. Callers wanting a plain product must zero the region first; the standard
    eltwise_binary leans on the packer's section-end ZEROACC for that, and so does this op. The gate
    is unaffected either way, since it only ever instantiates ELWADD.
    """
    want = _binary_want(math_op, operand_a, src_b, fidelity)
    if acc_to_dest or math_op == MathOperation.Elwmul:
        want = (want.to(torch.float32) + acc_base.to(torch.float32)).to(torch.bfloat16)
    return want


@parametrize(
    math_op=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    reload=[False, True],
)
def test_generalized_moe_gate_binary(math_op, reload):
    # COPY takes SrcA from the unpacker and drops a copy in the score region on the way; RELOAD
    # reads it back out. Seeding that region with something other than the unpacked operand is what
    # makes the two distinguishable.
    src_a, src_b, seed, acc_base = _binary_stimuli(seed=31)
    fidelity = _binary_fidelity(math_op)

    tiles = _run(
        _config(
            GENERALIZED_MOE_GATE(mode=MODE_BINARY, reload=reload),
            [src_a, _word_tile(_to_dst(seed)), _word_tile(_to_dst(acc_base)), _zeros()],
            src_b=src_b,
            math_op=math_op,
            fidelity=fidelity,
        ),
        view=_all_faces,
    )

    operand_a = seed if reload else src_a
    assert torch.equal(
        _from_dst(tiles[SCORES]).reshape(-1).to(torch.bfloat16), operand_a
    ), "the score region does not hold the operand this mode sources SrcA from"
    # The key region starts holding acc_base, so this is also where the ELWADD/ELWSUB overwrite and
    # the ELWMUL accumulate part company: same acc_to_dest=false, different result.
    assert torch.equal(
        _from_dst(tiles[KEYS]).reshape(-1).to(torch.bfloat16),
        _binary_key_region(
            math_op, operand_a, src_b, fidelity, acc_base, acc_to_dest=False
        ),
    ), "the result did not reach the key region"


# acc_to_dest makes the FPU add its result onto whatever the key region already holds, so with it
# set all three ops accumulate. Read this together with the test above, where the same three run
# with it clear: that pair is what pins ELWMUL accumulating regardless.
@parametrize(math_op=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul])
def test_generalized_moe_gate_binary_acc_to_dest(math_op):
    math_op = _one(math_op)
    src_a, src_b, seed, acc_base = _binary_stimuli(seed=57)
    fidelity = _binary_fidelity(math_op)

    tiles = _run(
        _config(
            GENERALIZED_MOE_GATE(mode=MODE_BINARY),
            [src_a, _word_tile(_to_dst(seed)), _word_tile(_to_dst(acc_base)), _zeros()],
            src_b=src_b,
            math_op=math_op,
            fidelity=fidelity,
            acc_to_dest=True,
        ),
        view=_all_faces,
    )

    assert torch.equal(
        _from_dst(tiles[KEYS]).reshape(-1).to(torch.bfloat16),
        _binary_key_region(math_op, src_a, src_b, fidelity, acc_base, acc_to_dest=True),
    ), "acc_to_dest did not accumulate onto the key region"


# num_faces is the mop's outer loop count, so it decides how much of the tile the binary touches.
# The tests above run it at 4; the op ships at 1.
@parametrize(num_faces=[1, 2])
def test_generalized_moe_gate_binary_num_faces(num_faces):
    num_faces = _one(num_faces)
    src_a, src_b, seed, acc_base = _binary_stimuli(seed=88)

    tiles = _run(
        _config(
            GENERALIZED_MOE_GATE(mode=MODE_BINARY),
            [src_a, _word_tile(_to_dst(seed)), _word_tile(_to_dst(acc_base)), _zeros()],
            src_b=src_b,
            num_faces=num_faces,
        ),
        view=_all_faces,
    )

    assert (
        tiles.shape[1] == num_faces
    ), f"packed {tiles.shape[1]} faces per tile, expected {num_faces}"
    want = _binary_want(MathOperation.Elwadd, src_a, src_b, MathFidelity.HiFi4)[
        : num_faces * 256
    ]
    assert torch.equal(
        _from_dst(tiles[KEYS]).reshape(-1).to(torch.bfloat16), want
    ), f"the sum did not reach the key region across {num_faces} faces"


# srcb names the 4-row SrcB scratch window the copy stages through. The gate uses all four windows
# (16/20/24/28) so that back-to-back copies cannot read a previous copy's leftover; sweeping it here
# covers the two the fused path only reaches indirectly, and each window must move the rows the same
# way.
@parametrize(src_dst=[(0, 4), (4, 8), (8, 12), (12, 0)], srcb=[16, 20, 24, 28])
def test_generalized_moe_gate_copy4rows(src_dst, srcb):
    # Two argnames, so pytest passes the tuple through as-is -- no _one() unwrap.
    src, dst = src_dst
    tags, tiles = _tag_tiles()

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_MOVE,
                sub_op=MOVE_COPY4ROWS,
                row_src=src,
                row_dst=dst,
                srcb=srcb,
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
        faces[INTERMEDIATE], tags[INTERMEDIATE]
    ), "copy4rows touched the scratch region"


# Every set_dst_write_addr in the op resolves to tile*64 + get_dest_buffer_base(), and that base is
# zero for the whole of a single-section kernel. Only a second section reaches the upper DEST half,
# which is where every other block of the multi-block path runs. Comparing the two sections needs no
# golden of its own: the same stimulus through the same code has to come back the same either side.
#
# Both halves being equally wrong would satisfy that, so every arm also pins its lower half against
# the golden its single-section test uses.
#
# sections=2 only means anything under DstSync::Half. Under SyncFull get_dest_buffer_base() stays 0
# for both sections and the comparison is vacuous, which is why dest_sync is not an axis here.
@parametrize(
    what=[
        (MODE_MOVE, MOVE_COPY4ROWS),  # a MOP runner: exercises the DEST-offset setup
        (
            MODE_RUN,
            RUN_PLACE_FIELD,
        ),  # SFPU only, so a difference here is not MOP-specific
        (
            MODE_GATE,
            0,
        ),  # the whole pipeline: MOPs, merges and the normalize interleaved
    ]
)
def test_generalized_moe_gate_dest_sections(what):
    mode, sub_op = _one(what)
    tags, tiles = _tag_tiles()
    payload, bias, keys = _gate_stimuli(seed=61)
    ids = _ids_face()

    gate = mode == MODE_GATE
    lower, upper = _run(
        _config(
            GENERALIZED_MOE_GATE(
                mode=mode,
                sub_op=sub_op,
                row_src=0,
                row_dst=4,
                sections=2,
                eps=_bits(EPS),
                scale=_bits(SCALE),
            ),
            (_gate_tiles(payload, ids) if gate else tiles),
            src_b=bias if gate else None,
        ),
        view=_sections,
    )

    if gate:
        # The gate leaves SrcB residue in columns 8-15 of its output row, which is not part of the
        # contract and need not match, so the halves are compared through the gate's answer rather
        # than cell by cell.
        golden = get_golden_generator(GeneralizedMoeGateGolden)(
            keys, payload, ids, eps=EPS, scale=SCALE
        )
        for half in (lower, upper):
            _assert_gate_output(
                half, golden, 8, **_weight_tolerance(ApproximationMode.No)
            )
        return

    for region in (SCORES, IDS, KEYS, INTERMEDIATE):
        assert torch.equal(
            upper[region], lower[region]
        ), f"region {region} differs between the two DEST halves"

    if mode == MODE_MOVE:
        for region in (SCORES, IDS, KEYS):
            want = tags[region].clone()
            want[4:8] = tags[region][0:4]
            assert torch.equal(
                lower[region], want
            ), f"region {region} is wrong in the first section"
    else:
        # place_field at its defaults: field 0 takes the run at {0,2} out of the intermediate region
        # and lands it in the key region, leaving everything else alone.
        want = tags[KEYS].clone()
        for row, column in _run_cells(0, 2):
            want[row, column] = tags[INTERMEDIATE][row, column]
        assert torch.equal(
            lower[KEYS], want
        ), "the placed field is wrong in the first section"
        for region in (SCORES, IDS, INTERMEDIATE):
            assert torch.equal(
                lower[region], tags[region]
            ), f"place_field also wrote region {region}"


def test_generalized_moe_gate_copy4rows_back_to_back():
    tags, tiles = _tag_tiles()

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
        faces[INTERMEDIATE], tags[INTERMEDIATE]
    ), "copy4rows touched the scratch region"


@parametrize(knobs=[(0, 8), (4, 0), (4, 8)])
def test_generalized_moe_gate_step1_hi(knobs):
    """step1_hi's two knobs pick which four rows it reads and where it writes the result.

    The layout it produces is not specified anywhere outside the instruction sequence, so rather
    than pin that, this compares against the same op with both knobs at zero over an image
    shifted to match.
    """
    d2b_dst, b2d_base = _one(knobs)
    tags, tiles = _tag_tiles()
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
        faces[INTERMEDIATE], tags[INTERMEDIATE]
    ), "step1_hi touched the scratch region"


# step1 and step1_hi<d2b_dst=0, b2d_base=0> program byte-identical MOPs today: the same two
# MOV_4_ROWS from DEST row 0, the same TRNSPSRCB, the same eight MOV_1_ROW back to rows 0-7, both at
# num_tiles=3. The grouped path calls step1 and the ungrouped path calls step1_hi, so nothing else
# would notice if one were edited and the other left behind. That equivalence is the claim here, and
# it is also what lets the step1_hi test above stand as step1's layout cover.
def test_generalized_moe_gate_step1_matches_step1_hi():
    tags, tiles = _tag_tiles()

    step1, step1_hi = _run(
        _config(GENERALIZED_MOE_GATE(mode=MODE_MOVE, sub_op=MOVE_STEP1), tiles),
        _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_MOVE, sub_op=MOVE_STEP1_HI, d2b_dst=0, b2d_base=0
            ),
            tiles,
        ),
    )

    for region in (SCORES, IDS, KEYS, INTERMEDIATE):
        assert torch.equal(
            step1[region], step1_hi[region]
        ), f"region {region} differs between step1 and step1_hi at its zero knobs"

    # Anchor the pair to something absolute, so that two MOPs broken the same way still fail: step1
    # writes rows 0-7 of the three data regions and nothing else.
    for region in (SCORES, IDS, KEYS):
        assert torch.equal(
            step1[region][8:16], tags[region][8:16]
        ), f"region {region} was written below row 7"
    assert torch.equal(
        step1[INTERMEDIATE], tags[INTERMEDIATE]
    ), "step1 touched the scratch region"


def test_generalized_moe_gate_step0():
    """step0 puts group g on DEST row g, in all four regions.

    It reads DEST rows 0-7 into every other row of the SrcB transpose window and reads the same
    rows back afterwards, which makes out[i][2k] = in[k][2i]: the even columns of output row i are
    input column 2i, and a group is a column pair. Odd columns come back holding whatever the
    transpose found in the window rows nothing wrote, so they carry no claim -- that is where the
    gate's residue in columns 8-15 comes from.
    """
    tags, tiles = _tag_tiles()

    faces = _run(
        _config(GENERALIZED_MOE_GATE(mode=MODE_MOVE, sub_op=MOVE_STEP0), tiles)
    )

    # num_tiles=4, so unlike every other MOP here this one does reach the scratch region.
    for region in (SCORES, IDS, KEYS, INTERMEDIATE):
        assert torch.equal(
            faces[region][0:8, 0::2], tags[region][0:8, 0::2].T
        ), f"region {region} did not transpose rows 0-7 onto the even columns"
        assert torch.equal(
            faces[region][8:16], tags[region][8:16]
        ), f"region {region} rows 8-15 were written"


def test_generalized_moe_gate_step2():
    """step2 turns the merged run into the output layout: rank r moves to row 0, column r.

    The run arrives down column 0 of rows 0-7, so this is the same window transpose read back one
    row instead of eight -- out[0][j] = in[j][0]. Columns 8-15 of that row are SrcB residue.
    num_tiles=3, so the scratch region is untouched; the gate needs the bias transposed too, which
    is what makes it 3 rather than 2.
    """
    tags, tiles = _tag_tiles()

    faces = _run(
        _config(GENERALIZED_MOE_GATE(mode=MODE_MOVE, sub_op=MOVE_STEP2), tiles)
    )

    for region in (SCORES, IDS, KEYS):
        assert torch.equal(
            faces[region][0, 0:8], tags[region][0:8, 0]
        ), f"region {region} did not land the run on row 0"
        assert torch.equal(
            faces[region][1:16], tags[region][1:16]
        ), f"region {region} was written below row 0"
    assert torch.equal(
        faces[INTERMEDIATE], tags[INTERMEDIATE]
    ), "step2 touched the scratch region"


def _offset_block(offset):
    """The 4 x 8 block of face cells an SFPU column offset names, as broadcast row/column indices."""
    rows = torch.arange(4 * (offset // 4), 4 * (offset // 4) + 4)
    cols = torch.arange((offset >> 1) & 1, 16, 2)
    return rows[:, None], cols[None, :]


def _merge4_stimuli(seed):
    """Key, id and score for the four sorted quarters merge4_top8 reads, at offsets {0,2,4,6}.

    Each offset arrives descending by key, since the op merges already-sorted sequences rather
    than sorting from scratch. Keys, ids and scores are independent permutations.
    """
    generator = torch.Generator().manual_seed(seed)

    def draw():
        # Below 2.0, so no value can rotate into a NaN word: see _word_tile. Ordering is all this
        # stimulus needs, and k/256 is exact in bf16 for every k here.
        return (torch.randperm(256, generator=generator) + 1).to(torch.float32) / 256.0

    key_pool, score_pool = draw(), draw()
    id_pool = torch.randperm(256, generator=generator).to(torch.int32)

    keys = torch.zeros(16, 16)
    ids = torch.zeros(16, 16, dtype=torch.int32)
    scores = torch.zeros(16, 16)
    for n, offset in enumerate((0, 2, 4, 6)):
        rows, cols = _offset_block(offset)
        quarter = slice(32 * n, 32 * (n + 1))
        order = key_pool[quarter].reshape(4, 8).argsort(dim=0, descending=True)
        keys[rows, cols] = torch.gather(key_pool[quarter].reshape(4, 8), 0, order)
        ids[rows, cols] = torch.gather(id_pool[quarter].reshape(4, 8), 0, order)
        scores[rows, cols] = torch.gather(score_pool[quarter].reshape(4, 8), 0, order)
    return keys, ids, scores


def _merge4_config(read_base, store, keys, ids, scores):
    store_lo, store_hi = store
    return _config(
        GENERALIZED_MOE_GATE(
            mode=MODE_RUN,
            sub_op=RUN_MERGE4_TOP8,
            read_base=read_base,
            to_lo=store_lo,
            to_hi=store_hi,
        ),
        [
            _word_tile(_to_dst(scores)),
            _word_tile(ids),
            _word_tile(_to_dst(keys)),
            _word_tile(torch.zeros(16, 16, dtype=torch.int32)),
        ],
    )


def _stored_run(faces, store_lo, store_hi):
    """Every instance's stored run, as (id, score) in slot order."""
    got_scores = _from_dst(faces[SCORES])
    return [
        (int(faces[IDS][r, c]), float(got_scores[r, c]))
        for instance in range(8)
        for r, c in _run_slots(store_lo, store_hi, instance)
    ]


# merge4_top8 is the gate's second merge stage, and the gate calls it only at read_base 0 storing
# to {0,2}. What it selects is covered there, against exact goldens; what has no cover at all is
# that its two parameters are honest, and that the idx|score concat survives the round trip. Those
# are the claims here, and not which eight of the sixteen it keeps: the direction convention a
# bitonic merge wants for its four input quarters is not written down outside the instruction
# sequence, so a selection golden would enshrine whatever the op does today.
def test_generalized_moe_gate_merge4_top8():
    keys, ids, scores = _merge4_stimuli(seed=57)
    shifted = [torch.roll(t, shifts=8, dims=0) for t in (keys, ids, scores)]

    at_zero, at_eight, stored_elsewhere = _run(
        _merge4_config(0, (8, 10), keys, ids, scores),
        _merge4_config(8, (8, 10), *shifted),
        _merge4_config(0, (12, 14), keys, ids, scores),
    )

    run = _stored_run(at_zero, 8, 10)
    # read_base names the first of the four offsets it reads, so base 8 reads {8,10,12,14}: rows
    # 8-15 rather than rows 0-7, which is the eight-row roll the shifted stimulus applies.
    assert run == _stored_run(
        at_eight, 8, 10
    ), "read_base did not move the read window by the eight rows it names"
    assert run == _stored_run(
        stored_elsewhere, 12, 14
    ), "the store pair did not move the run"

    # A merge may drop a candidate but never invent one, and a score has to come back still
    # attached to the id it arrived with. The concat is where that can break: erratum TEN-2932
    # corrupts SFPU results written to LREG4-7 while index tracking is enabled, and the concat
    # lives there.
    #
    # Only rows 0-7 are in read_base 0's window, so restricting the source to those makes this a
    # claim about the window too: a candidate from outside it is as invented as one from nowhere.
    source = {int(ids[r, c]): float(scores[r, c]) for r in range(8) for c in range(16)}
    for stored_id, stored_score in run:
        assert (
            stored_id in source
        ), f"the run holds id {stored_id}, which was not in the read window"
        assert (
            stored_score == source[stored_id]
        ), f"id {stored_id} came back carrying {stored_score}, not {source[stored_id]}"


@parametrize(
    placement=[(0, 2, 4, 6), (0, 2, 8, 10), (4, 6, 0, 2), (0, 4, 8, 12)],
    after_mop=lambda placement: [False, True] if placement == (0, 2, 4, 6) else [False],
)
def test_generalized_moe_gate_relocate_run(placement, after_mop):
    from_lo, from_hi, to_lo, to_hi = placement
    tags, tiles = _tag_tiles()

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
        faces[INTERMEDIATE], tags[INTERMEDIATE]
    ), "copy_topk_run touched the scratch region"


# field selects the home region and src/dst select offsets; they are independent in the LLK, so the
# offsets are swept at one field rather than at all three.
@parametrize(
    field=[0, 1, 2],
    src=lambda field: [(0, 4)] if field else [(0, 4), (8, 12)],
    dst=lambda field: [(0, 2)] if field else [(0, 2), (4, 6)],
    after_mop=lambda field, src, dst: (
        [False, True] if (field, src, dst) == (0, (0, 4), (0, 2)) else [False]
    ),
)
def test_generalized_moe_gate_place_field(field, src, dst, after_mop):
    src_lo, src_hi = src
    dst_lo, dst_hi = dst
    tags, tiles = _tag_tiles()
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
        want[dr, dc] = tags[INTERMEDIATE][sr, sc]
    assert torch.equal(faces[home], want), f"field {field} landed wrong"

    for region in (SCORES, IDS, KEYS, INTERMEDIATE):
        if region == home:
            continue
        untouched = tags[region].clone()
        if after_mop and region != INTERMEDIATE:
            untouched[12:16] = tags[region][0:4]
        assert torch.equal(
            faces[region], untouched
        ), f"field {field} also wrote region {region}"


def _combine_stimuli(seed, score_divisor=8.0):
    """The merge input as it stands once both runs are in place: key, id and score for DEST rows
    0-7 of every column.

    Rows 0-3 are the run already resident at {0,2}; rows 4-7 are the run that arrives field by field
    through the intermediate region. Keys are distinct so the top-8 is unambiguous, and the scores
    are unrelated to the keys so a sort whose payload desynced from its key gets caught.

    score_divisor holds the scores inside one octave for finalize, which exponentiates their
    differences on the softmax path and needs a spread exp and a bf16 sum can both hold.
    """
    generator = torch.Generator().manual_seed(seed)
    keys = (torch.randperm(128, generator=generator) + 1).to(torch.float32) / 8.0
    ids = torch.randperm(128, generator=generator).to(torch.int32)
    scores = (torch.randperm(128, generator=generator) + 1).to(
        torch.float32
    ) / score_divisor
    return keys.reshape(8, 16), ids.reshape(8, 16), scores.reshape(8, 16)


def _assert_combined(faces, keys, ids, scores, store_lo, store_hi, idx_offset):
    """Every instance's stored run holds the top 8 of that instance's own 16 candidates.

    All eight are checked, not just the one the gate reads: they run the same network over
    different columns, so a lane-addressing error that spares instance 0 still shows up here.
    """
    got_scores = _from_dst(faces[SCORES])
    for instance in range(8):
        columns = [2 * instance, 2 * instance + 1]
        candidates = keys[:, columns].reshape(-1).argsort(descending=True)[:8]
        want_ids = ids[:, columns].reshape(-1)[candidates]
        want_scores = scores[:, columns].reshape(-1)[candidates]

        slots = _run_slots(store_lo, store_hi, instance)
        assert [int(faces[IDS][r, c]) for r, c in slots] == [
            int(i) + idx_offset for i in want_ids
        ], f"instance {instance} merged to the wrong experts"
        # The score rides through the merge as the high half of the concat and is never computed
        # on, so it comes back bit-exact.
        assert [
            float(got_scores[r, c]) for r, c in slots
        ] == want_scores.tolist(), f"instance {instance} carried the wrong scores"


IDX_OFFSET_NONE = 0


def _combine_tiles(keys, ids, scores):
    """The four DEST tiles a combine starts from.

    Rows 0-3 of each data region hold the resident run. Rows 4-7 are the merge slot the arriving run
    has to land in, seeded with keys far above anything real so a placement that silently does not
    land leaves poison the merge would rank first. The arriving run itself waits in the intermediate
    region, one field per four-row band, which is where place_field reads it from.
    """
    resident = slice(0, 4)
    arriving = slice(4, 8)
    poison = _to_dst(torch.full((4, 16), 4096.0))

    def region(rows_0_3, poisoned):
        face = torch.zeros(16, 16, dtype=torch.int32)
        face[resident] = rows_0_3
        face[arriving] = poisoned
        return face

    intermediate = torch.zeros(16, 16, dtype=torch.int32)
    intermediate[0:4] = _to_dst(keys[arriving])
    intermediate[4:8] = ids[arriving]
    intermediate[8:12] = _to_dst(scores[arriving])

    return [
        _word_tile(region(_to_dst(scores[resident]), poison)),
        _word_tile(region(ids[resident], torch.full((4, 16), 900))),
        _word_tile(region(_to_dst(keys[resident]), poison)),
        _word_tile(intermediate),
    ]


# The multi-block combine tail: a block's run reaches its home regions through place_field, and the
# merge then has to accept it as an equal of the run already sitting there. That the placed cells
# form a run a merge can consume is the run format's whole contract, and nothing else checks it.
@parametrize(store=[(8, 10), (12, 14)])
def test_generalized_moe_gate_combine(store):
    store_lo, store_hi = _one(store)
    keys, ids, scores = _combine_stimuli(seed=404)

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_RUN,
                sub_op=RUN_COMBINE,
                to_lo=store_lo,
                to_hi=store_hi,
                idx_offset=IDX_OFFSET_NONE,
            ),
            _combine_tiles(keys, ids, scores),
        )
    )

    _assert_combined(faces, keys, ids, scores, store_lo, store_hi, IDX_OFFSET_NONE)


# Same combine, except the arriving run is already sitting in DEST at {8,10} and reaches the merge
# slot by relocation. copy_topk_run's own test only checks that the cells moved; that what lands is
# still a run a merge will accept is a separate claim, and this is what makes it.
def test_generalized_moe_gate_combine_relocated():
    keys, ids, scores = _combine_stimuli(seed=404)
    poison = torch.full((4, 16), 4096.0)
    store_lo, store_hi = 12, 14

    def region(resident, poisoned, arriving):
        """Resident run at rows 0-3, poison in the merge slot, arriving run parked at rows 8-11."""
        face = torch.zeros(16, 16, dtype=torch.int32)
        face[0:4] = resident
        face[4:8] = poisoned
        face[8:12] = arriving
        return face

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_RUN,
                sub_op=RUN_COMBINE_RELOCATED,
                to_lo=store_lo,
                to_hi=store_hi,
                idx_offset=IDX_OFFSET_NONE,
            ),
            [
                _word_tile(
                    region(_to_dst(scores[0:4]), _to_dst(poison), _to_dst(scores[4:8]))
                ),
                _word_tile(region(ids[0:4], torch.full((4, 16), 900), ids[4:8])),
                _word_tile(
                    region(_to_dst(keys[0:4]), _to_dst(poison), _to_dst(keys[4:8]))
                ),
                _zeros(),
            ],
        )
    )

    _assert_combined(faces, keys, ids, scores, store_lo, store_hi, IDX_OFFSET_NONE)


# generalized_moe_gate_combine_finalize: the >256 path's actual output. The arriving run is placed
# at {4,6}, then finalize sorts the pair at {0,2}+{4,6}, normalizes and step2 transposes to the
# output layout. finalize runs its own merge, so unlike the RUN_COMBINE tail no merge16 precedes it.
# Nothing else composes a placed run with the normalize, which is where a run that merges correctly
# but carries the wrong payload would still show up.
@parametrize(topk=[4, 6, 8], softmax=[False, True])
def test_generalized_moe_gate_combine_finalize(topk, softmax):
    keys, ids, scores = _combine_stimuli(seed=404, score_divisor=128.0)

    faces = _run(
        _config(
            GENERALIZED_MOE_GATE(
                mode=MODE_RUN,
                sub_op=RUN_COMBINE_FINALIZE,
                topk=topk,
                softmax=softmax,
                eps=_bits(EPS),
                scale=_bits(SCALE),
            ),
            _combine_tiles(keys, ids, scores),
        )
    )

    # The output is instance 0's answer, so its candidates are DEST columns 0 and 1 of rows 0-7.
    golden = get_golden_generator(GeneralizedMoeGateGolden)(
        keys[:, :2],
        scores[:, :2],
        ids[:, :2],
        topk=topk,
        output_softmax=softmax,
        eps=EPS,
        scale=SCALE,
    )
    _assert_gate_output(faces, golden, topk, **_weight_tolerance(ApproximationMode.No))


# The gate-level cover above cannot run under ttsim, and this can, so it is the one that will catch
# a regression in the offset add before silicon does. Comparing two runs of the same stimulus needs
# no golden and no knowledge of where in the run a rank lands.
def test_generalized_moe_gate_idx_offset():
    tags, tiles = _tag_tiles()

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

    # A merge is eight independent sorts side by side and the offset add is applied to all of them,
    # so check every instance: an add that reaches only instance 0 is a live failure mode, the ids
    # and the scores coming from different registers.
    for instance in range(8):
        slots = _run_slots(0, 2, instance)
        base_ids = [int(base[IDS][r, c]) for r, c in slots]
        shifted_ids = [int(shifted[IDS][r, c]) for r, c in slots]
        assert shifted_ids == [
            i + 256 for i in base_ids
        ], f"idx_offset did not shift instance {instance}'s ids: {base_ids} -> {shifted_ids}"

    # The ids the offset does not own must be untouched, which the region sweep below cannot see:
    # the stores land in IDS either way, so only the cells outside the run tell a stray one apart.
    run_rows = {r for r, _ in _run_slots(0, 2)}
    rest = [r for r in range(16) if r not in run_rows]
    assert torch.equal(
        base[IDS][rest], shifted[IDS][rest]
    ), "idx_offset wrote ids outside the run"

    for region in (SCORES, KEYS, INTERMEDIATE):
        assert torch.equal(
            base[region], shifted[region]
        ), f"idx_offset also wrote region {region}"
