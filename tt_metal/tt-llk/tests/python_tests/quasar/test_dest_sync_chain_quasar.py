# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_CHAIN_SCHEDULE,
    DEST_SYNC,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.tile_constants import MAX_NUM_FACES
from helpers.utils import passed_test

# Unit bits, as p_cleardvalid in the kernel.
U = 1  # unpack to dest
F = 2  # FPU
S = 4  # SFPU
P = 8  # PACK
ALL = U | F | S | P

# Short step names for the schedules below.
u, f, sq, p = (U,), (F,), (S,), (P,)

# name -> list of (order, members) per section, order is the tuple of unit steps, members are the
# ops still inited
SCHEDULES = {
    "fpu_pack": [(f + p, F | P)] * 6,
    "unpack_pack": [(u + p, U | P)] * 6,
    "fpu_sfpu_pack": [(f + sq + p, F | S | P)] * 6,
    "unpack_sfpu_pack": [(u + sq + p, U | S | P)] * 6,
    "sfpu_last": [
        (f + sq, ALL),
        (f + sq + p, ALL),
        (u + sq, ALL),
        (u + sq + p, ALL),
        (f + sq + p, ALL),
        (f + p, ALL),
    ],
    "head_alone": [
        (f, F | P),
        (f + p, F | P),
        (u, U | P),
        (u + p, U | P),
        (f + p, F | P),
        (f + p, F | P),
    ],
    "members_idle": [
        (f + p, ALL),
        (u + p, ALL),
        (f + sq + p, ALL),
        (u + sq + p, ALL),
        (f + p, ALL),
        (f + sq + p, ALL),
    ],
    "sfpu_join_odd": [(f + p, F | P)] + [(f + sq + p, F | S | P)] * 5,
    "sfpu_join_even": [(f + p, F | P)] * 2 + [(f + sq + p, F | S | P)] * 4,
    "sfpu_leave_rejoin": [(f + sq + p, F | S | P)] * 2
    + [(f + p, F | P), (f + sq + p, F | S | P), (f + p, F | P), (f + p, F | P)],
    "unpack_fpu_alternate": [
        (u + p, U | P),
        (f + p, F | P),
        (u + p, U | P),
        (f + p, F | P),
        (u + sq + p, U | S | P),
        (f + sq + p, F | S | P),
    ],
    "unpack_mop_switch_revisit": [
        (u + f + u + p, U | F | P),
        (f + u + f + p, U | F | P),
    ]
    * 3,
    "join_leave_all": [
        (f + p, F | P),
        (u + p, U | P),
        (f + sq + p, F | S | P),
        (u + sq + p, U | S | P),
        (f + p, F | P),
        (u + sq + p, U | S | P),
    ],
    "sfpu_fill_head": [
        (sq + p, S | P),
        (f + sq + p, F | S | P),
        (sq, S | P),
        (u + p, U | P),
        (sq + p, ALL),
        (f + p, ALL),
    ],
    # revisits: the head comes back, a middle unit comes back, PACK reads twice
    "head_revisit": [(f + sq + f + p, F | S | P)] * 6,
    "unpack_head_revisit": [(u + sq + u + p, U | S | P)] * 6,
    "sfpu_head_revisit": [(sq + f + sq + p, F | S | P)] * 6,
    "pack_twice": [(f + sq + p + sq + p, F | S | P)] * 6,
    "revisit_ends_section": [
        (f + sq + f, F | S | P),
        (f + sq + f + p, F | S | P),
        (sq + f + sq, F | S | P),
        (f + p, F | S | P),
        (u + sq + u, ALL),
        (u + p, ALL),
    ],
    "take_twice": [
        (f + f + sq + p, F | S | P),
        (sq + sq + p, F | S | P),
        (f + sq + sq + p, F | S | P),
        (f + sq + p + p, F | S | P),
        (u + u + p, ALL),
        (f + sq + f + f + p, ALL),
    ],
    "revisit_mixed": [
        (f + sq + f + p, ALL),
        (u + p, ALL),
        (sq + f + sq + p, ALL),
        (f + sq + p + sq + p, ALL),
        (u + sq + u + p, ALL),
        (f + p, ALL),
    ],
    "changed_head_after_revisit": [
        (f + sq + f + p, ALL),
        (sq + p, ALL),
        (f + sq + f + p, ALL),
        (sq + p, ALL),
        (u + sq + u + p, ALL),
        (sq + p, ALL),
    ],
    "future_head_runs_ahead": [
        (f + p, ALL),
        (sq + p, ALL),
        (u + p, ALL),
        (f + p, ALL),
        (sq + p, ALL),
        (u + p, ALL),
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
SECTIONS_PER_KERNEL = 8  # four bits of members per section in a 32 bit word
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


def all_transition_schedules(shapes, prefix, idle_members):
    circuit = euler_circuit(shapes)
    pads = 1 if len(circuit) % 2 == 1 else 2
    walk = circuit + [circuit[0]] * pads + circuit[1:]
    schedules = {}
    step = SECTIONS_PER_KERNEL - 2
    for k, start in enumerate(range(0, len(walk) - 1, step)):
        window = walk[start : start + SECTIONS_PER_KERNEL]
        members = [ALL if idle_members else units_of(order) for order in window]
        schedules[f"{prefix}_{k}"] = list(zip(window, members))
    return schedules


def units_of(order):
    bits = 0
    for unit in order:
        bits |= unit
    return bits


SCHEDULES.update(all_transition_schedules(SHAPES, "walk", idle_members=False))
SCHEDULES.update(all_transition_schedules(SHAPES, "idle", idle_members=True))
SCHEDULES.update(
    all_transition_schedules(REVISIT_SHAPES, "revisit", idle_members=False)
)

TILE_ELEMENTS = 1024


def pack_nibbles(values):
    packed = 0
    for section, value in enumerate(values):
        packed |= value << (4 * section)
    return packed


def pack_orders(schedule):
    words = [0, 0, 0, 0]
    for section, (order, _) in enumerate(schedule):
        half = 0
        for k, unit in enumerate(order):
            half |= (unit.bit_length()) << (3 * k)
        words[section // 2] |= half << (16 * (section % 2))
    return words


def check_schedule(schedule):
    assert len(schedule) <= SECTIONS_PER_KERNEL
    for order, members in schedule:
        assert 0 < len(order) <= STEPS_PER_SECTION
        assert (
            units_of(order) & ~members == 0
        ), "a unit takes dest only while in the chain"
        assert order[0] != P, "PACK only reads dest, so it cannot start a section"


def golden(src_a, schedule):
    out = src_a.clone()
    packed = []
    for section, (order, _) in enumerate(schedule):
        lo, hi = section * TILE_ELEMENTS, (section + 1) * TILE_ELEMENTS
        dest = None
        wrote = False
        for unit in order:
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
)
def test_dest_sync_chain_quasar(schedule, dest_sync):
    sections = SCHEDULES[schedule]
    check_schedule(sections)
    tile_cnt = len(sections)
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)

    torch.manual_seed(42)
    src_a = (torch.rand(tile_cnt * TILE_ELEMENTS) * 2 - 1).to(torch.bfloat16)
    src_b = torch.zeros(TILE_ELEMENTS, dtype=torch.bfloat16)
    golden_tensor, packed = golden(src_a, sections)

    configuration = TestConfig(
        "sources/quasar/dest_sync_chain_quasar_test.cpp",
        formats,
        templates=[
            DEST_SYNC(dest_sync),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            NUM_FACES(MAX_NUM_FACES),
            TEST_FACE_DIMS(),
            DEST_CHAIN_SCHEDULE(
                *pack_orders(sections),
                pack_nibbles([members for _, members in sections]),
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
            tile_count_res=tile_cnt,
            num_faces=MAX_NUM_FACES,
        ),
        dest_acc=DestAccumulation.No,
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
