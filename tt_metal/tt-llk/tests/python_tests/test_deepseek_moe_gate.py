# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
deepseek_moe_gate LLK test.

The grouped 256-expert MoE gate the ttnn deepseek_moe_gate op ships, driven through the three LLKs
its Compute API calls: the eltwise binary FPU frontend, the step0/step1/step2 Dest transposes, and
the top-k SFPU. The sequence is sum_top2 -> step0 -> sort_top4_groups -> step1 -> top8 -> step2.

The golden is shared with GeneralizedMoeGate.

DEST holds four tiles and the op only touches face 0 of each: tile 0 the payload score it emits,
tile 1 a 16-bit id, tile 2 the payload+bias sort key, tile 3 scratch.
"""

import struct

import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import GeneralizedMoeGateGolden, get_golden_generator
from helpers.llk_params import (
    ApproximationMode,
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
    DEEPSEEK_MOE_GATE,
    DEST_SYNC,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    TILE_COUNT,
)

pytestmark = skip_for_quasar

FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.UInt16)

MODE_GATE, MODE_BINARY, MODE_MOVE = 0, 1, 2
MOVE_STEP0, MOVE_STEP1, MOVE_STEP2 = 0, 1, 2

# Dest tile per gate region, in the order the SFPU's region offsets walk them.
SCORES, IDS, KEYS, INTERMEDIATE = 0, 1, 2, 3

EPS = 0.5
SCALE = 2.5


def _fp32_bits(value):
    """The fp32 bit pattern Converter::as_float expects."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _zeros():
    return torch.zeros(16, 16, dtype=torch.bfloat16)


def _tile(face):
    """A 32x32 tile in tiled order. A 16x16 face lands in face 0 with the rest zero."""
    face = face.reshape(-1)
    if face.numel() == 1024:
        return face.to(torch.bfloat16)
    padded = torch.zeros(1024, dtype=torch.bfloat16)
    padded[:256] = face
    return padded


def _faces(result):
    """Every packed face of the four DEST tiles, as [tile, face, 16, 16] uint16 words."""
    words = torch.tensor(result, dtype=torch.int64) & 0xFFFF
    return words.reshape(4, -1, 16, 16).to(torch.int32)


def _from_dest(words):
    """Dest 16-bit float words back to values."""
    words = words.to(torch.int32) & 0xFFFF
    ieee = (words & 0x8000) | ((words & 0xFF) << 7) | ((words >> 8) & 0x7F)
    return ieee.to(torch.uint16).view(torch.bfloat16).to(torch.float32)


def _to_dest(values):
    """Values to Dest 16-bit float words, for a region a test loads through the integer path."""
    words = values.to(torch.bfloat16).view(torch.uint16).to(torch.int32)
    return (words & 0x8000) | ((words & 0x7F) << 8) | ((words >> 7) & 0xFF)


def _word_tile(words):
    """A Dest word image as a stimulus, carried by bit pattern rather than by value."""
    image = words.to(torch.int32).to(torch.uint16).view(torch.bfloat16)
    # Stimuli are packed through float32, which does not preserve a NaN payload, so a word that reads
    # as NaN would reach the device as some other word. Nothing here trips it; this is so a future
    # stimulus fails loudly instead of arriving altered.
    assert torch.isfinite(
        image.to(torch.float32)
    ).all(), "Dest word image holds a bit pattern that reads as NaN or Inf; it will not survive packing"
    return image


def _tag_tiles():
    """The four tagged Dest regions: the word images to check against, and the tiles to upload.

    A distinct word per cell, so a moved datum can be traced back to where it came from. Built in the
    Dest layout directly, with the low byte the exponent, kept clear of 0 and 255 so a float load/store
    round trip neither flushes nor saturates it. The bases are 1000 apart so a datum that crosses
    regions is traceable to the one it came from.
    """
    tags = []
    for base in (0, 1000, 2000, 3000):
        cells = base + torch.arange(256)
        tags.append(
            ((((cells // 200) % 128) << 8) | (cells % 200 + 20)).reshape(16, 16)
        )
    return tags, [_word_tile(t) for t in tags]


def _expert_ids():
    """Expert i at flat position i, which is the frame the id tile is unpacked in."""
    return torch.arange(256, dtype=torch.int32).reshape(16, 16)


def _gate_tiles(payload):
    """The four Dest tiles a gate run starts from.

    The payload is transposed on the way in: the non-sigmoid front-end unpacks SrcA under
    ckernel::Transpose::Both, so the datum it scores as expert i is the transpose of what L1 holds, and
    stimuli are written in the gate's frame. The ids are not transposed, and the key and scratch
    regions the op fills itself.
    """
    return [
        payload.t().contiguous(),
        _expert_ids().to(torch.uint16).view(torch.bfloat16),
        _zeros(),
        _zeros(),
    ]


def _assert_unambiguous(keys):
    """The grouped answer is well defined only when the eight per-group top-2 key sums are pairwise
    distinct, so that which four groups survive is unambiguous. Most seeds tie somewhere.

    A group is a column pair {2g, 2g+1} of the gate's frame, which flattens to 32 experts.
    """
    groups = torch.arange(256).reshape(16, 16).t().reshape(8, 32)
    top2 = keys.reshape(-1)[groups].sort(dim=-1, descending=True).values[:, :2]
    sums = top2.sum(dim=-1)
    assert len(set(sums.tolist())) == 8, (
        f"stimuli tie the per-group top-2 key sums {sorted(sums.tolist())}, so which four groups "
        "survive is ambiguous; pick another seed"
    )


def _gate_stimuli(seed):
    """A payload/bias pair whose bf16 sum hits 256 distinct keys.

    Everything is a multiple of 1/16 below magnitude 12, which bf16 holds exactly, so the FPU's
    payload+bias lands on the intended key and no two experts tie at the rank-8 cut. The payload stays
    positive so the normalization denominator cannot approach zero and blow the weights up.
    """
    generator = torch.Generator().manual_seed(seed)
    key = (torch.randperm(256, generator=generator) - 128).to(torch.float32) / 16.0
    payload = torch.randint(1, 65, (256,), generator=generator).to(torch.float32) / 16.0
    keys = key.reshape(16, 16).to(torch.bfloat16)
    _assert_unambiguous(keys)
    return (
        payload.reshape(16, 16).to(torch.bfloat16),
        (key - payload).reshape(16, 16).to(torch.bfloat16),
        keys,
    )


def _sigmoid_stimuli(seed):
    """A payload/bias pair for the sigmoid front-end, with eight winners the bias alone decides.

    The gate transposes the payload and activates it but takes the bias straight from SrcB, so the sort
    key is sigmoid(payload.T) + bias. Winners are spaced two apart in the bias and everything else sits
    below -1, which is more than the at-most-1 an activation can contribute: the ranking is unambiguous
    however coarse the sigmoid turns out to be, and the activation only has to be right in the weights.
    """
    generator = torch.Generator().manual_seed(seed)
    payload = (
        torch.randint(-32, 33, (256,), generator=generator).to(torch.float32) / 8.0
    ).reshape(16, 16)
    bias = (
        torch.randint(-64, -8, (256,), generator=generator).to(torch.float32) / 8.0
    ).reshape(16, 16)

    groups = torch.randperm(8, generator=generator)[:4]
    rows = torch.randperm(16, generator=generator)[:8]
    cells = [
        (int(rows[2 * n + i]), int(2 * groups[n] + i))
        for n in range(4)
        for i in range(2)
    ]
    for rank, (row, column) in enumerate(cells):
        bias[row, column] = 40.0 - 2.0 * rank
    return payload.to(torch.bfloat16), bias.to(torch.bfloat16)


# Approximation mode reaches only sfpu_reciprocal for the sum. Selection is a bitonic
# compare network either way, so the expert ids are exact in both modes and only the weights
# get the looser bound.
def _weight_tolerance(approx):
    return (
        dict(rtol=5e-2, atol=1e-2)
        if approx == ApproximationMode.Yes
        else dict(rtol=2e-2, atol=1e-3)
    )


def _assert_gate_output(regions, golden, **tolerance):
    """Compare the gate's row-0 output against a golden [2, 8] of weights and ids.

    The grouped merge stops at "step 4 only", so it emits its top-8 in bitonic rather than sorted
    order: pair each weight with its own id and compare the two sets.
    """
    got_weights, got_ids = _from_dest(regions[SCORES])[0, :8], regions[IDS][0, :8]
    got = sorted(zip(got_ids.tolist(), got_weights.tolist()))
    want = sorted(zip([int(i) for i in golden[1]], golden[0].tolist()))

    assert [p[0] for p in got] == [
        p[0] for p in want
    ], f"wrong experts: {got} vs {want}"
    assert torch.allclose(
        torch.tensor([p[1] for p in got]),
        torch.tensor([p[1] for p in want]),
        **tolerance,
    ), f"weights differ: {got} vs {want}"


def _golden(keys, payload, eps=EPS, scale=SCALE):
    """The grouped gate's answer."""
    return get_golden_generator(GeneralizedMoeGateGolden)(
        keys, payload, _expert_ids(), eps=eps, scale=scale, grouped=True
    )


def _config(
    gate,
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
        "sources/deepseek_moe_gate_test.cpp",
        FORMATS,
        templates=[
            gate,
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
            tile_count_res=4,
            num_faces=num_faces,
        ),
    )


def _run(configuration):
    return _faces(configuration.run().result)


def _regions(configuration):
    """Face 0 of every Dest tile."""
    return _run(configuration)[:, 0]


# 256 experts, eight groups, top-4 groups then top-8, linear renorm. Not every seed leaves the grouped
# answer unambiguous; 19 and 128 are shared with generalized_moe_gate's grouped test.
@parametrize(
    seed=[19, 128, 4],
    approx=[ApproximationMode.No, ApproximationMode.Yes],
)
def test_deepseek_moe_gate(seed, approx):
    payload, bias, keys = _gate_stimuli(seed)

    regions = _regions(
        _config(
            DEEPSEEK_MOE_GATE(
                dmg_mode=MODE_GATE, dmg_eps=_fp32_bits(EPS), dmg_scale=_fp32_bits(SCALE)
            ),
            _gate_tiles(payload),
            src_b=bias,
            approx=approx,
        )
    )

    _assert_gate_output(regions, _golden(keys, payload), **_weight_tolerance(approx))


# The op's enable_sigmoid front-end, which the plain path never reaches: transpose_wh_tile, then
# sigmoid_tile, then a RELOAD binary that takes SrcA back out of DEST through MOVD2A while the unpacker
# feeds only SrcB under DEST_TO_SRCA reuse. This is also the only cover of RELOAD as production drives
# it.
@parametrize(seed=[77, 78])
def test_deepseek_moe_gate_sigmoid(seed):
    (seed,) = seed
    payload, bias = _sigmoid_stimuli(seed)

    regions = _regions(
        _config(
            DEEPSEEK_MOE_GATE(
                dmg_mode=MODE_GATE,
                dmg_sigmoid=True,
                dmg_eps=_fp32_bits(EPS),
                dmg_scale=_fp32_bits(SCALE),
            ),
            # Not _gate_tiles: this front-end does its transpose in transpose_wh_tile on the raw L1
            # tile, so the payload goes up as-is and the golden transposes.
            [
                payload,
                _expert_ids().to(torch.uint16).view(torch.bfloat16),
                _zeros(),
                _zeros(),
            ],
            src_b=bias,
        )
    )

    # The emitted score is the activation, not the raw payload, and it is read at the transposed
    # position: both claims land in the weight comparison. The key is the activation plus the
    # untransposed bias, which is what makes the winners' ids come back untransposed.
    activated = torch.sigmoid(payload.to(torch.float32).t()).to(torch.bfloat16)
    keys = (activated.to(torch.float32) + bias.to(torch.float32)).to(torch.bfloat16)
    _assert_unambiguous(keys)

    # Looser than _weight_tolerance's approx bound: this compares against torch.sigmoid, so the
    # activation's own error is in the budget on top of the normalization's.
    _assert_gate_output(regions, _golden(keys, activated), rtol=5e-2, atol=1e-2)


# eps and scale are the normalization's two knobs and every other gate test pins them at 0.5/2.5.
# (0, 1) is both the API default and what a caller wanting a plain renorm passes: it is the one
# setting where the weights have to sum to exactly 1, and the only one where the reciprocal sees the
# sum unpadded.
@parametrize(norm=[(0.0, 1.0), (EPS, SCALE), (2.0, 0.5)])
def test_deepseek_moe_gate_normalization(norm):
    (norm,) = norm
    eps, scale = norm
    payload, bias, keys = _gate_stimuli(seed=19)

    regions = _regions(
        _config(
            DEEPSEEK_MOE_GATE(
                dmg_mode=MODE_GATE, dmg_eps=_fp32_bits(eps), dmg_scale=_fp32_bits(scale)
            ),
            _gate_tiles(payload),
            src_b=bias,
        )
    )

    _assert_gate_output(
        regions,
        _golden(keys, payload, eps=eps, scale=scale),
        **_weight_tolerance(ApproximationMode.No),
    )


# The configuration axes the op is built under.
@parametrize(dest_sync=[DestSync.Half, DestSync.Full], num_faces=[1, 4])
def test_deepseek_moe_gate_shipping_config(dest_sync, num_faces):
    payload, bias, keys = _gate_stimuli(seed=128)

    regions = _regions(
        _config(
            DEEPSEEK_MOE_GATE(
                dmg_mode=MODE_GATE, dmg_eps=_fp32_bits(EPS), dmg_scale=_fp32_bits(SCALE)
            ),
            _gate_tiles(payload),
            src_b=bias,
            dest_sync=dest_sync,
            num_faces=num_faces,
        )
    )

    _assert_gate_output(
        regions,
        _golden(keys, payload),
        **_weight_tolerance(ApproximationMode.No),
    )


_BINARY_GOLDEN = {
    MathOperation.Elwadd: lambda a, b: a + b,
    MathOperation.Elwsub: lambda a, b: a - b,
    MathOperation.Elwmul: lambda a, b: a * b,
}


def _binary_stimuli(seed):
    """Two four-face operands of integers in [-8, 8]."""
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randint(-8, 9, (4, 16, 16), generator=generator).to(torch.bfloat16)
        for _ in range(2)
    ]


def _binary_tiles(scores, acc_base=None):
    """The four buffer_A tiles a BINARY run needs."""
    return [
        scores,
        _word_tile(_to_dest(scores)),
        _zeros() if acc_base is None else _word_tile(_to_dest(acc_base)),
        _zeros(),
    ]


def _assert_binary(faces, scores, bias, math_op, acc_base=None):
    """tile 0 holds SrcA verbatim, tile 2 holds srcA (op) srcB, over every face packed back.

    The MOP is a MOVA2D of SrcA into Dest tile 0 followed by the binary at dst_math_offset = 2 * 64
    rows, innerloop 2 and outerloop num_faces, so the face count is exactly how much of Dest it
    covers, and a wrong outer loop leaves a later face unwritten.
    """
    num_faces = faces.shape[1]
    scores, bias = scores[:num_faces], bias[:num_faces]

    expected = _BINARY_GOLDEN[math_op](scores.to(torch.float32), bias.to(torch.float32))
    if acc_base is not None:
        expected = expected + acc_base[:num_faces].to(torch.float32)

    assert torch.equal(
        _from_dest(faces[SCORES]), scores.to(torch.float32)
    ), "Dest tile 0 is not SrcA verbatim"
    assert torch.equal(
        _from_dest(faces[KEYS]), expected
    ), f"Dest tile 2 is not SrcA {math_op.name} srcB"


# ELWADD is what the Compute API instantiates; ELWSUB and ELWMUL are reachable through the same
# template and are here because the MOP selects a different instruction for each. ELWMUL takes LoFi
# only.
@parametrize(
    math_op=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    fidelity=lambda math_op: (
        [MathFidelity.LoFi]
        if math_op == MathOperation.Elwmul
        else [MathFidelity.LoFi, MathFidelity.HiFi4]
    ),
)
def test_deepseek_moe_gate_binary(math_op, fidelity):
    scores, bias = _binary_stimuli(seed=11)

    faces = _run(
        _config(
            DEEPSEEK_MOE_GATE(dmg_mode=MODE_BINARY),
            _binary_tiles(scores),
            src_b=bias,
            math_op=math_op,
            fidelity=fidelity,
        )
    )

    _assert_binary(faces, scores, bias, math_op)


# RELOAD swaps the MOP's MOVA2D for a replayed MOVD2A, so SrcA comes back out of Dest tile 0.
def test_deepseek_moe_gate_binary_reload():
    scores, bias = _binary_stimuli(seed=12)

    faces = _run(
        _config(
            DEEPSEEK_MOE_GATE(dmg_mode=MODE_BINARY, dmg_reload=True),
            _binary_tiles(scores),
            src_b=bias,
        )
    )

    _assert_binary(faces, scores, bias, MathOperation.Elwadd)


# acc_to_dest turns the binary's write into an accumulate onto whatever Dest tile 2 already held, so
# this seeds that region with integers of its own.
def test_deepseek_moe_gate_binary_acc_to_dest():
    scores, bias = _binary_stimuli(seed=13)
    base, _ = _binary_stimuli(seed=15)

    faces = _run(
        _config(
            DEEPSEEK_MOE_GATE(dmg_mode=MODE_BINARY),
            _binary_tiles(scores, acc_base=base),
            src_b=bias,
            acc_to_dest=True,
        )
    )

    _assert_binary(faces, scores, bias, MathOperation.Elwadd, acc_base=base)


# num_faces is the MOP's outer loop. The tests above all run it at 4, so here we assert partial works.
@parametrize(num_faces=[1, 2])
def test_deepseek_moe_gate_binary_num_faces(num_faces):
    (num_faces,) = num_faces
    scores, bias = _binary_stimuli(seed=14)

    faces = _run(
        _config(
            DEEPSEEK_MOE_GATE(dmg_mode=MODE_BINARY),
            _binary_tiles(scores),
            src_b=bias,
            num_faces=num_faces,
        )
    )

    assert faces.shape[1] == num_faces
    _assert_binary(faces, scores, bias, MathOperation.Elwadd)


def test_deepseek_moe_gate_step0():
    """step0 puts group g on Dest row g, in all four regions.

    Rows 0-7 go into every other window row and the same rows come back, making
    out[i][2k] = in[k][2i]: the even columns of output row i are input column 2i, and a group is a
    column pair. Odd columns are residue.
    """
    tags, tiles = _tag_tiles()

    regions = _regions(
        _config(DEEPSEEK_MOE_GATE(dmg_mode=MODE_MOVE, dmg_sub_op=MOVE_STEP0), tiles)
    )

    for region in (SCORES, IDS, KEYS, INTERMEDIATE):
        assert torch.equal(
            regions[region][0:8, 0::2], tags[region][0:8, 0::2].T
        ), f"region {region} did not transpose rows 0-7 onto the even columns"
        assert torch.equal(
            regions[region][8:16], tags[region][8:16]
        ), f"region {region} rows 8-15 were written"


def test_deepseek_moe_gate_step1():
    """step1 reads Dest rows 0-3 and writes Dest rows 0-7.

    Two MOVD2Bs put rows 0-3 into the window twice, at window rows 0-3 and 12-15, then eight MOVB2Ds
    read window columns 0, 2, ... 14 back. So output row i is input column 2i, appearing once in
    columns 0-3 and again in columns 12-15; columns 4-11 are residue. Its num_tiles is 3, so it covers
    the score, id and key regions and not the scratch one.
    """
    tags, tiles = _tag_tiles()

    regions = _regions(
        _config(DEEPSEEK_MOE_GATE(dmg_mode=MODE_MOVE, dmg_sub_op=MOVE_STEP1), tiles)
    )

    for region in (SCORES, IDS, KEYS):
        for half in (slice(0, 4), slice(12, 16)):
            assert torch.equal(
                regions[region][0:8, half], tags[region][0:4, 0::2].T
            ), f"region {region} did not transpose rows 0-3 into columns {half.start}-{half.stop - 1}"
        assert torch.equal(
            regions[region][8:16], tags[region][8:16]
        ), f"region {region} rows 8-15 were written"
    assert torch.equal(
        regions[INTERMEDIATE], tags[INTERMEDIATE]
    ), "step1 touched the scratch region, which its num_tiles=3 mop does not reach"


def test_deepseek_moe_gate_step2():
    """step2 turns the merged run into the output layout: rank r moves to row 0, column r.

    The run arrives down column 0 of rows 0-7. Columns 8-15 are residue. Its num_tiles is 2, so
    it only covers the score and id regions: by then the key that ranked each winner has done
    its job and the gate emits weights and ids.
    """
    tags, tiles = _tag_tiles()

    regions = _regions(
        _config(DEEPSEEK_MOE_GATE(dmg_mode=MODE_MOVE, dmg_sub_op=MOVE_STEP2), tiles)
    )

    for region in (SCORES, IDS):
        assert torch.equal(
            regions[region][0, 0:8], tags[region][0:8, 0]
        ), f"region {region} did not land the run on row 0"
        assert torch.equal(
            regions[region][1:16], tags[region][1:16]
        ), f"region {region} was written below row 0"
    for region in (KEYS, INTERMEDIATE):
        assert torch.equal(
            regions[region], tags[region]
        ), f"step2 touched region {region}, which its num_tiles=2 mop does not reach"
