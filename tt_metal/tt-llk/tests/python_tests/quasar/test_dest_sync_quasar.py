# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import truncate_to_bfloat16
from helpers.llk_params import DestAccumulation, DestSync, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    DEST_SYNC_SCHEDULE,
    DEST_SYNC_TEST_CONFIG,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.tile_constants import MAX_NUM_FACES
from helpers.utils import passed_test

# Client bits, 1 << DestClient in the kernel.
U = 1  # unpack to dest
F = 2  # FPU
S = 4  # SFPU
P = 8  # PACK

# Short step names for the schedules below.
u, f, sq, p = (U,), (F,), (S,), (P,)

# name -> the order of each section, the tuple of client steps.
SCHEDULES = {
    "fpu_pack": [f + p] * 6,
    "unpack_pack": [u + p] * 6,
    "fpu_sfpu_pack": [f + sq + p] * 6,
    "unpack_sfpu_pack": [u + sq + p] * 6,
    "sfpu_last": [
        f + sq,
        f + sq + p,
        u + sq,
        u + sq + p,
        f + sq + p,
        f + p,
    ],
    "head_alone": [
        f,
        f + p,
        u,
        u + p,
        f + p,
        f + p,
    ],
    "members_idle": [
        f + p,
        u + p,
        f + sq + p,
        u + sq + p,
        f + p,
        f + sq + p,
    ],
    "sfpu_join_odd": [f + p] + [f + sq + p] * 5,
    "sfpu_join_even": [f + p] * 2 + [f + sq + p] * 4,
    "sfpu_leave_rejoin": [f + sq + p] * 2 + [f + p, f + sq + p, f + p, f + p],
    "unpack_fpu_alternate": [
        u + p,
        f + p,
        u + p,
        f + p,
        u + sq + p,
        f + sq + p,
    ],
    "unpack_mop_switch_revisit": [
        u + f + u + p,
        f + u + f + p,
    ]
    * 3,
    "join_leave_all": [
        f + p,
        u + p,
        f + sq + p,
        u + sq + p,
        f + p,
        u + sq + p,
    ],
    "sfpu_fill_head": [
        sq + p,
        f + sq + p,
        sq,
        u + p,
        sq + p,
        f + p,
    ],
    # sfpu_fill_head loses the square of (f + sq + p) right after a fill, where the datacopy is also inited for the
    # first time mid-kernel. These two split the causes: a fill before the square with the datacopy inited first,
    # and a datacopy first inited mid-kernel with no fill.
    "fill_then_square": [f + p, sq + p, f + sq + p] * 2,
    "late_datacopy_init": [u + sq + p, f + sq + p] * 3,
    # Both pass, so the fill being the first dest write of the kernel matters. Narrow that down: a fill first and the
    # square right after the first datacopy (the failing pair alone), one section later, and after unpack to dest.
    "fill_first": [sq + p, f + sq + p] * 3,
    "fill_first_square_later": [
        sq + p,
        f + p,
        f + sq + p,
    ]
    * 2,
    "fill_first_unpack_square": [sq + p, u + sq + p] * 3,
    # revisits: the head comes back, a middle unit comes back, PACK reads twice
    "head_revisit": [f + sq + f + p] * 6,
    "unpack_head_revisit": [u + sq + u + p] * 6,
    "sfpu_head_revisit": [sq + f + sq + p] * 6,
    "pack_twice": [f + sq + p + sq + p] * 6,
    "revisit_ends_section": [
        f + sq + f,
        f + sq + f + p,
        sq + f + sq,
        f + p,
        u + sq + u,
        u + p,
    ],
    "take_twice": [
        f + f + sq + p,
        sq + sq + p,
        f + sq + sq + p,
        f + sq + p + p,
        u + u + p,
        f + sq + f + f + p,
    ],
    "revisit_mixed": [
        f + sq + f + p,
        u + p,
        sq + f + sq + p,
        f + sq + p + sq + p,
        u + sq + u + p,
        f + p,
    ],
    "changed_head_after_revisit": [
        f + sq + f + p,
        sq + p,
        f + sq + f + p,
        sq + p,
        u + sq + u + p,
        sq + p,
    ],
    "future_head_runs_ahead": [
        f + p,
        sq + p,
        u + p,
        f + p,
        sq + p,
        u + p,
    ],
}

# Every chain shape a section can have: a producer or the SFPU fill first, optional SFPU, optional PACK.
SHAPES = [u, f, sq, u + sq, f + sq, u + p, f + p, sq + p, u + sq + p, f + sq + p]
# Chains with a revisit, walked together with the plain ones they are built from.
REVISIT_SHAPES = [
    f + p,
    u + p,
    f + sq + f + p,
    u + sq + u + p,
    sq + f + sq + p,
    f + sq + p + sq + p,
    sq + sq + p,
]
SECTIONS_PER_KERNEL = (
    8  # 16 bits of order per section, two sections in each of four words
)
STEPS_PER_SECTION = 5  # three bits per step in 16 bits of order per section


def euler_circuit(nodes):
    unused = {a: [b for b in nodes] for a in nodes}
    stack, circuit = [nodes[0]], []
    while stack:
        a = stack[-1]
        if unused[a]:
            stack.append(unused[a].pop())
        else:
            circuit.append(stack.pop())
    return circuit[::-1]


def all_transition_schedules(shapes, prefix):
    circuit = euler_circuit(shapes)
    pads = 1 if len(circuit) % 2 == 1 else 2
    walk = circuit + [circuit[0]] * pads + circuit[1:]
    schedules = {}
    step = SECTIONS_PER_KERNEL - 2
    for k, start in enumerate(range(0, len(walk) - 1, step)):
        window = walk[start : start + SECTIONS_PER_KERNEL]
        schedules[f"{prefix}_{k}"] = window
    return schedules


SCHEDULES.update(all_transition_schedules(SHAPES, "walk"))
SCHEDULES.update(all_transition_schedules(REVISIT_SHAPES, "revisit"))

TILE_ELEMENTS = 1024
SPLIT_PRODUCERS = ["unpack", "srcs_copy", "srcs_square", "fpu"]


def pack_orders(schedule):
    words = [0, 0, 0, 0]
    for section, order in enumerate(schedule):
        half = 0
        for k, unit in enumerate(order):
            half |= (unit.bit_length()) << (3 * k)
        words[section // 2] |= half << (16 * (section % 2))
    return words


def check_schedule(schedule):
    assert len(schedule) <= SECTIONS_PER_KERNEL
    for order in schedule:
        assert 0 < len(order) <= STEPS_PER_SECTION
        assert order[0] != P, "PACK only reads dest, so it cannot start a section"


def golden(src_a, schedule):
    out = src_a.clone()
    packed = []
    for section, order in enumerate(schedule):
        lo, hi = section * TILE_ELEMENTS, (section + 1) * TILE_ELEMENTS
        dest = None
        wrote = False
        packed_last = False
        for unit in order:
            if packed_last and unit != P:
                dest = None  # a take after PACK starts a new section on an empty bank
            packed_last = unit == P
            if unit in (U, F):
                dest = src_a[lo:hi]
            elif unit == S:
                dest = (
                    dest * dest
                    if dest is not None
                    else torch.full_like(src_a[lo:hi], (section + 1) / 8)
                )
            else:
                out[lo:hi] = dest
                wrote = True
        packed.append(wrote)
    return out, packed


def packed_tiles(tensor, packed):
    parts = [
        tensor[i * TILE_ELEMENTS : (i + 1) * TILE_ELEMENTS]
        for i, p in enumerate(packed)
        if p
    ]
    return torch.cat(parts)


@pytest.mark.quasar
@parametrize(
    schedule=list(SCHEDULES.keys()),
    dest_sync=[DestSync.Half, DestSync.Full],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_dest_sync_quasar(schedule, dest_sync, dest_acc):
    run_dest_sync(schedule, dest_sync, dest_acc)


@pytest.mark.quasar
@pytest.mark.parametrize("dest_acc", [DestAccumulation.No, DestAccumulation.Yes])
def test_dest_sync_persistent_tensors_quasar(dest_acc):
    run_dest_sync("fpu_pack", DestSync.Full, dest_acc, persistent_tensors=True)


@pytest.mark.quasar
@parametrize(
    srcs_mode=SPLIT_PRODUCERS,
    swap_halves=[False, True],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    chunks=[1, 49],
)
def test_dest_sync_split_pipeline_quasar(srcs_mode, swap_halves, dest_acc, chunks):
    """Exercise SDPA's lifetime/dependency pattern using exact, elementary arithmetic.

    Scores and history occupy one half; the two-tile output survives all chunks
    in the other half. SrcS has its own unpack/SFPU/PACK1 stream, plus transfers
    into scores and out of either Dest half. This is a sync test, not SDPA math.
    """
    fp32 = dest_acc == DestAccumulation.Yes
    input_format = DataFormat.Float32 if fp32 else DataFormat.Float16_b
    dtype = format_dict[input_format]
    output_format = DataFormat.Float16_b
    out_dtype = format_dict[output_format]
    torch.manual_seed(91)
    a = (torch.randint(-8, 9, (chunks, TILE_ELEMENTS)).float() / 32).to(dtype)
    b = (torch.randint(-4, 5, (chunks, TILE_ELEMENTS)).float() / 64).to(dtype)
    s = (torch.randint(-4, 13, (chunks, TILE_ELEMENTS)).float() / 32).to(dtype)

    def sfpu_store(values):
        return (values if fp32 else truncate_to_bfloat16(values)).to(dtype)

    # FPU writes round to nearest-even; plain SFPU BF16 stores truncate.
    history = torch.zeros(TILE_ELEMENTS, dtype=dtype)
    output = torch.zeros((2, TILE_ELEMENTS), dtype=dtype)
    dest_expected, srcs_expected = [], []
    for chunk in range(chunks):
        score_input = a[chunk] if srcs_mode in ("unpack", "fpu") else s[chunk]
        scores = sfpu_store(score_input.float() * score_input.float())
        if chunk:
            output = sfpu_store(output.float() + history.float() * 0.5)
        inputs = [chunk, chunks - 1 - chunk]
        output = (output.float() + a[inputs].float() + b[inputs].float()).to(dtype)
        history = sfpu_store(history.float() + scores.float())
        dest_expected.extend((scores, history))
        srcs_expected.extend(
            (
                sfpu_store(s[chunk].float() * s[chunk].float()),
                scores,
                output[0],
                output[1],
            )
        )
    dest_expected.extend(
        (sfpu_store(output[0].float() * 0.5), sfpu_store(output[1].float() * 0.5))
    )
    dest_expected.append(torch.full((TILE_ELEMENTS,), -0.75, dtype=dtype))
    golden_dest = torch.stack(dest_expected).to(out_dtype)
    golden_srcs = torch.stack(srcs_expected).to(out_dtype)

    stimuli = StimuliConfig(
        a.flatten(),
        input_format,
        b.flatten(),
        input_format,
        output_format,
        tile_count_A=chunks,
        tile_count_B=chunks,
        tile_count_res=len(golden_dest),
        buffer_S=s.flatten(),
        stimuli_S_format=input_format,
        tile_count_S=chunks,
        buffer_C=torch.full((4 * chunks * TILE_ELEMENTS,), -99.0, dtype=out_dtype),
        stimuli_C_format=output_format,
        tile_count_C=4 * chunks,
        srcs_layout_operands=frozenset({"S", "C"}),
        num_faces=MAX_NUM_FACES,
    )
    configuration = TestConfig(
        "sources/quasar/dest_sync_quasar_test.cpp",
        InputOutputFormat(input_format, output_format),
        templates=[
            DEST_SYNC(DestSync.Full),
            DEST_SYNC_TEST_CONFIG(
                split_pipeline=True,
                srcs_mode=SPLIT_PRODUCERS.index(srcs_mode),
                swap_halves=swap_halves,
            ),
        ],
        runtimes=[
            TILE_COUNT(chunks),
            NUM_FACES(MAX_NUM_FACES),
            TEST_FACE_DIMS(),
            DEST_SYNC_SCHEDULE(0, 0, 0, 0),
        ],
        variant_stimuli=stimuli,
        dest_acc=dest_acc,
        unpack_to_srcs=True,
    )
    outcome = configuration.run()
    actual_dest = torch.tensor(outcome.result, dtype=out_dtype).reshape(
        -1, TILE_ELEMENTS
    )
    actual_srcs = torch.tensor(
        stimuli.collect_buffer_c_results(TestConfig.TENSIX_LOCATION), dtype=out_dtype
    ).reshape(-1, TILE_ELEMENTS)
    assert actual_dest.shape == golden_dest.shape
    assert actual_srcs.shape == golden_srcs.shape
    for name, golden_tiles, actual_tiles in (
        ("Dest/PACK0", golden_dest, actual_dest),
        ("SrcS/PACK1", golden_srcs, actual_srcs),
    ):
        for tile, (golden_tile, actual_tile) in enumerate(
            zip(golden_tiles, actual_tiles)
        ):
            assert passed_test(
                golden_tile, actual_tile, output_format
            ), f"{name} tile {tile}"
    assert torch.equal(
        actual_dest[-1], golden_dest[-1]
    ), "Unrelated persistent guard tensor was corrupted"


def run_dest_sync(schedule, dest_sync, dest_acc, persistent_tensors=False):
    sections = [f + p] * 24 if persistent_tensors else SCHEDULES[schedule]
    if not persistent_tensors:
        check_schedule(sections)
    tile_cnt = len(sections)
    # Every client must address dest in the same width, and the unpacker writes the L1 format into dest: with a 32-bit
    # dest a 16-bit unpack to dest would leave dest in a different format than the FPU and SFPU write.
    input_format = (
        DataFormat.Float32 if dest_acc == DestAccumulation.Yes else DataFormat.Float16_b
    )
    formats = InputOutputFormat(input_format, DataFormat.Float16_b)

    torch.manual_seed(42)
    # bfloat16 values, held in the L1 format's dtype: exact in Float32 too.
    stimuli_dtype = format_dict[input_format]
    src_a = (torch.rand(tile_cnt * TILE_ELEMENTS) * 2 - 1).to(torch.bfloat16)
    src_a = src_a.to(stimuli_dtype)
    src_b = torch.zeros(TILE_ELEMENTS, dtype=stimuli_dtype)
    golden_src = src_a.float() if dest_acc == DestAccumulation.Yes else src_a
    if persistent_tensors:
        tiles = golden_src.reshape(tile_cnt, TILE_ELEMENTS)
        other_squared = tiles.flip(0) * tiles.flip(0)
        other_fourth = other_squared * other_squared
        golden_tensor = torch.stack(
            (tiles, other_fourth, tiles * tiles, other_fourth), dim=1
        ).flatten()
        packed = [True] * (tile_cnt * 4)
    else:
        golden_tensor, packed = golden(golden_src, sections)
    golden_tensor = golden_tensor.to(format_dict[formats.output_format])

    configuration = TestConfig(
        "sources/quasar/dest_sync_quasar_test.cpp",
        formats,
        templates=[
            DEST_SYNC(dest_sync),
            DEST_SYNC_TEST_CONFIG(persistent_tensors),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            NUM_FACES(MAX_NUM_FACES),
            TEST_FACE_DIMS(),
            DEST_SYNC_SCHEDULE(
                *([0] * 4 if persistent_tensors else pack_orders(sections))
            ),
        ],
        variant_stimuli=StimuliConfig(
            src_a,
            formats.input_format,
            src_b,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=1,
            tile_count_res=len(packed),
            num_faces=MAX_NUM_FACES,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(golden_tensor)
    if not any(packed):
        return  # no section packed, finishing without a hang is the whole check
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(
        packed_tiles(golden_tensor, packed),
        packed_tiles(res_tensor, packed),
        formats.output_format,
    )
