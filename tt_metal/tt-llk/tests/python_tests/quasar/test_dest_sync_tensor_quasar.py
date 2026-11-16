# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    RuntimeParameter,
)
from helpers.tile_constants import MAX_NUM_FACES
from helpers.utils import passed_test

U = 1
F = 2
S = 4
P = 8

T0, T1, T2, T3 = 0, 1, 2, 3
TENSOR_CNT = 4
SLOTS = 2

SCHEDULES = {
    "single_chain": [[(F, T0), (P, T0)]] * 4,
    "four_parallel": [
        [(F, T0), (P, T0), (F, T1)],
        [(P, T1), (F, T2), (P, T2)],
        [(F, T3), (P, T3), (F, T0)],
        [(P, T0), (S, T1), (P, T1)],
    ]
    * 2,
    "all_tensors": [[(F, t), (P, t)] for t in (T0, T1, T2, T3)] * 2,
    "deep_pipeline": [
        [(F, T0), (S, T0), (P, T0)],
        [(U, T1), (S, T1), (P, T1)],
        [(F, T2), (P, T2), (F, T3)],
        [(P, T3), (S, T0), (P, T0)],
    ]
    * 2,
    "unpack_to_dest": [[(U, T0), (P, T0)]] * 4,
    "unpack_sfpu_pack": [[(U, T1), (S, T1), (P, T1)]] * 2,
    "sfpu_last": [[(F, T2), (S, T2)]] * 4,
    "revisits": [[(F, T3), (S, T3), (F, T3), (P, T3)]] * 2,
    "fill_then_pack": [[(S, T1), (P, T1), (F, T1), (S, T1), (P, T1)]] * 2,
}

SECTIONS_PER_KERNEL = 8
STEPS_PER_SECTION = 3
TILE_ELEMENTS = 1024


@dataclass
class SCHEDULE_WORDS(RuntimeParameter):
    order0: int = 0
    order1: int = 0
    order2: int = 0
    order3: int = 0

    def convert_to_cpp(self) -> str:
        return ""

    def convert_to_struct_fields(self) -> tuple[str, str]:
        fields = "\n".join(f"std::uint32_t ORDER{k};" for k in range(4))
        return fields, "IIII"


def pack_orders(sections):
    words = [0, 0, 0, 0]
    for section, steps in enumerate(sections):
        half = 0
        for k, (client, tensor) in enumerate(steps):
            half |= (client.bit_length() + (tensor << 3)) << (5 * k)
        words[section // 2] |= half << (16 * (section % 2))
    return words


def check_schedule(sections):
    assert len(sections) <= SECTIONS_PER_KERNEL
    packed = set()
    for steps in sections:
        assert 0 < len(steps) <= STEPS_PER_SECTION
        assert steps[0][0] != P, "PACK only reads dest, so it cannot start a section"
        assert steps[-1][0] in (
            S,
            P,
        ), "the last client of a section must be a releaser, PACK or SFPU"
        for client, tensor in steps:
            assert (
                tensor not in packed
            ), "nothing follows PACK on a tensor within a section"
            if client == P:
                packed.add(tensor)
        packed.clear()


def golden(src_a, sections):
    values = {}
    out_tiles = []
    uses = [0] * TENSOR_CNT
    for i, steps in enumerate(sections):
        produced = [False] * TENSOR_CNT
        for client, tensor in steps:
            slot = uses[tensor] % SLOTS
            key = (tensor, slot)
            if client in (U, F):
                values[key] = src_a[i * TILE_ELEMENTS : (i + 1) * TILE_ELEMENTS]
                produced[tensor] = True
            elif client == S:
                values[key] = (
                    values[key] * values[key]
                    if produced[tensor]
                    else torch.full_like(src_a[0:TILE_ELEMENTS], (i + 1) / 8)
                )
                produced[tensor] = True
            else:
                if produced[tensor]:
                    out_tiles.append(values[key])
                produced[tensor] = False
        for t in range(TENSOR_CNT):
            if any(tt == t for _, tt in steps):
                uses[t] += 1
    if not out_tiles:
        return None
    return torch.cat(out_tiles)


@pytest.mark.quasar
@parametrize(
    schedule=list(SCHEDULES.keys()),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_dest_sync_tensor_quasar(schedule, dest_acc):
    sections = SCHEDULES[schedule]
    check_schedule(sections)
    tile_cnt = len(sections)
    input_format = (
        DataFormat.Float32 if dest_acc == DestAccumulation.Yes else DataFormat.Float16_b
    )
    formats = InputOutputFormat(input_format, DataFormat.Float16_b)

    torch.manual_seed(42)
    stimuli_dtype = format_dict[input_format]
    src_a = (torch.rand(tile_cnt * TILE_ELEMENTS) * 2 - 1).to(torch.bfloat16)
    src_a = src_a.to(stimuli_dtype)
    src_b = torch.zeros(TILE_ELEMENTS, dtype=stimuli_dtype)
    golden_out = golden(
        src_a.float() if dest_acc == DestAccumulation.Yes else src_a, sections
    )
    if golden_out is not None:
        golden_out = golden_out.to(format_dict[formats.output_format])

    words = pack_orders(sections)
    configuration = TestConfig(
        "sources/quasar/dest_sync_tensor_quasar_test.cpp",
        formats,
        templates=[DEST_SYNC(DestSync.Half)],
        runtimes=[
            TILE_COUNT(tile_cnt),
            NUM_FACES(MAX_NUM_FACES),
            TEST_FACE_DIMS(),
            SCHEDULE_WORDS(*words),
        ],
        variant_stimuli=StimuliConfig(
            src_a,
            formats.input_format,
            src_b,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=1,
            tile_count_res=2 * tile_cnt,
            num_faces=MAX_NUM_FACES,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    if golden_out is None:
        return
    res_tensor = torch.tensor(
        res_from_L1[: len(golden_out)], dtype=format_dict[formats.output_format]
    )
    assert passed_test(golden_out, res_tensor, formats.output_format)
