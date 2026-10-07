# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import FACE_DIM, TILE_DIM
from helpers.llk_params import (
    DestAccumulation,
    ImpliedMathFormat,
    UnpackerEngine,
    format_dict,
)
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TWO_PASS_STATS,
    UNPACKER_ENGINE_SEL,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

FORMATS = [
    InputOutputFormat(fmt, fmt)
    for fmt in (DataFormat.Float16_b, DataFormat.Float16, DataFormat.Float32)
]

# (atol, rtol): two output steps of relative slack plus a near-zero floor.
TOLERANCE = {
    DataFormat.Float16_b: (0.02, 2.0**-6),
    DataFormat.Float16: (0.005, 2.0**-9),
    DataFormat.Float32: (1e-4, 1e-4),
}

STREAM, COMBINE, SWITCH = 0, 1, 2
ROW, RAW, SPLIT, COMBINED, ROW_VAR_ONLY = 0, 1, 2, 3, 4

# After the row-quad transpose, statistics lane (r, c) tracks tile column QUAD_COLUMN[r](c).
GROUP_UNITS = 4
LANE_COLUMNS = 8
QUAD_COLUMN = (
    lambda c: 2 * c,
    lambda c: 2 * c + 1,
    lambda c: FACE_DIM + 2 * c,
    lambda c: FACE_DIM + 1 + 2 * c,
)


@dataclass(frozen=True)
class Scenario:
    tile_count: int
    mode: int = STREAM
    dual: bool = True
    finalize: int = ROW
    retain_anchor: bool = False
    average_variance: bool = True
    group_a: int = 0
    group_b: int = 0
    final_dst: int = 0
    state_dst: int = 0
    # STREAM: tiles per Dest section (0 = one section); COMBINE/SWITCH: tiles per statistics block.
    tiles_per_block: int = 0
    block_tiles: int = 1
    partial_last_tile: bool = False
    start_row: int = 0
    num_rows: int = TILE_DIM

    @property
    def block(self):
        if self.mode == STREAM:
            return self.tiles_per_block or self.tile_count
        return self.tile_count

    def validate(self):
        assert (
            self.tile_count % self.block == 0
        ), "tile_count must be whole Dest sections"
        footprint = 3 if self.finalize == COMBINED else 2
        assert (
            self.final_dst + footprint <= self.block
        ), "finaliser tiles must fit in one Dest section"
        assert (
            self.start_row + self.num_rows <= TILE_DIM
        ), "partial row window [start_row, start_row + num_rows) must fit in one tile"
        if self.retain_anchor:
            assert self.dual, "anchor retention needs the dual accumulator"
        if self.mode != STREAM:
            assert (
                self.tile_count % self.block_tiles == 0
            ), "tile_count must be whole blocks"
            # The state tiles are written after block 0 is done with them.
            assert self.state_dst + 1 < self.block_tiles or self.mode == COMBINE
        if self.mode == SWITCH:
            assert not self.dual, "group switching is single-accumulator only"
            assert self.tile_count == 2 * self.block_tiles

    def final_tile(self):
        """Result tile of the finaliser's first output."""
        return self.tile_count - self.block + self.final_dst


SCENARIOS = {
    "stream_row_single_2t": Scenario(tile_count=2, dual=False),
    # Three Dest sections per pass, so the state crosses two section handoffs in each pass.
    "stream_row_dual_3x4_partial": Scenario(
        tile_count=12,
        tiles_per_block=4,
        partial_last_tile=True,
        start_row=7,
        num_rows=20,
    ),
    "stream_var_only_dual_2t": Scenario(tile_count=2, finalize=ROW_VAR_ONLY),
    "stream_raw_g5_dual_2x2_partial": Scenario(
        tile_count=4,
        tiles_per_block=2,
        finalize=RAW,
        group_a=5,
        partial_last_tile=True,
        start_row=0,
        num_rows=9,
    ),
    "stream_raw_g3_single_2t": Scenario(
        tile_count=2, dual=False, finalize=RAW, group_a=3
    ),
    "stream_split_anchor_roundtrip_2x2": Scenario(
        tile_count=4, tiles_per_block=2, finalize=SPLIT, retain_anchor=True
    ),
    "stream_combined_avg_g2_dual_3t": Scenario(
        tile_count=3, finalize=COMBINED, group_a=2
    ),
    "stream_combined_sum_g7_single_partial_4t": Scenario(
        tile_count=4,
        dual=False,
        finalize=COMBINED,
        average_variance=False,
        group_a=7,
        final_dst=1,
        partial_last_tile=True,
        start_row=3,
        num_rows=26,
    ),
    "combine_2x2_dual": Scenario(
        tile_count=4, mode=COMBINE, block_tiles=2, final_dst=2
    ),
    "combine_2x2_single_partial": Scenario(
        tile_count=4,
        mode=COMBINE,
        dual=False,
        block_tiles=2,
        final_dst=2,
        partial_last_tile=True,
        start_row=5,
        num_rows=17,
    ),
    "switch_groups_1_6_single": Scenario(
        tile_count=4, mode=SWITCH, dual=False, block_tiles=2, group_a=1, group_b=6
    ),
}


def make_stimuli(tile_count, torch_format):
    """Row-major [32, 32 * tile_count]; per-column offset and spread expose lane or row mix-ups."""
    torch.manual_seed(0)
    column = torch.arange(TILE_DIM, dtype=torch.float32)
    offset = (column - 15.5) * 0.25
    spread = 0.5 + column / TILE_DIM
    noise = torch.empty(TILE_DIM, tile_count * TILE_DIM).uniform_(-1.0, 1.0)
    block = offset.repeat(tile_count) + spread.repeat(tile_count) * noise
    return block.to(torch_format)


def tile_view(block, tile):
    return block[:, tile * TILE_DIM : (tile + 1) * TILE_DIM]


def selected_rows(block, scenario, tiles):
    """The rows of `tiles` the kernel folds, in order, as float64."""
    rows = []
    for t in tiles:
        view = tile_view(block, t).to(torch.float64)
        if scenario.partial_last_tile and t == scenario.tile_count - 1:
            view = view[scenario.start_row : scenario.start_row + scenario.num_rows]
        rows.append(view)
    return torch.cat(rows)


def column_stats(rows):
    mean = rows.mean(dim=0)
    m2 = ((rows - mean) ** 2).sum(dim=0)
    return mean, m2 / rows.shape[0], m2


def slot_positions(group_id, rows=range(GROUP_UNITS), parities=(0,)):
    """(tile_row, tile_col, stat_column) for the lanes of a raw group slot."""
    positions = []
    for r in rows:
        unit = GROUP_UNITS * group_id + r
        face, face_row = divmod(unit, FACE_DIM)
        tile_row = (face // 2) * FACE_DIM + face_row
        for c in range(LANE_COLUMNS):
            for parity in parities:
                tile_col = (face % 2) * FACE_DIM + 2 * c + parity
                positions.append((tile_row, tile_col, QUAD_COLUMN[r](c)))
    return positions


def raw_slot(tile, stat, group_id):
    """(golden, device) lanes of a per-column statistic in a raw group slot."""
    positions = slot_positions(group_id)
    device = torch.stack([tile[r, c] for r, c, _ in positions])
    golden = torch.stack([stat[s] for _, _, s in positions])
    return golden, device


def scalar_slot(tile, value, group_id, rows):
    """(golden, device) lanes of a broadcast scalar in a raw group slot, both column parities."""
    positions = slot_positions(group_id, rows=rows, parities=(0, 1))
    device = torch.stack([tile[r, c] for r, c, _ in positions])
    golden = torch.full_like(device, float(value), dtype=torch.float64)
    return golden, device


def row_slot(tile, stat, row=0):
    """(golden, device) of tile row `row` = stat with the next three rows zero."""
    golden = torch.zeros(4, TILE_DIM, dtype=torch.float64)
    golden[0] = stat
    return golden.flatten(), tile[row : row + 4].flatten()


@pytest.mark.quasar
@pytest.mark.parametrize("formats", FORMATS, ids=lambda f: f.input_format.name)
@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
def test_sfpu_welfords_two_pass_quasar(formats, scenario_name):
    """Shifted two-pass per-column mean / population variance (the _two_pass_* helpers)."""
    scenario = SCENARIOS[scenario_name]
    scenario.validate()
    dest_acc = (
        DestAccumulation.Yes
        if formats.input_format.is_32_bit()
        else DestAccumulation.No
    )
    torch_format = format_dict[formats.input_format]
    tile_count = scenario.tile_count
    dims = [TILE_DIM, tile_count * TILE_DIM]

    block = make_stimuli(tile_count, torch_format)
    src_A = tilize_block(
        block.flatten(), dims, stimuli_format=formats.input_format
    ).flatten()
    src_B = torch.zeros_like(src_A)

    configuration = TestConfig(
        "sources/quasar/sfpu_welfords_two_pass_quasar_test.cpp",
        formats,
        templates=[
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
            DEST_SYNC(),
            TWO_PASS_STATS(
                two_pass_mode=scenario.mode,
                two_pass_dual=scenario.dual,
                two_pass_finalize=scenario.finalize,
                two_pass_retain_anchor=scenario.retain_anchor,
                two_pass_average_variance=scenario.average_variance,
                two_pass_group_a=scenario.group_a,
                two_pass_group_b=scenario.group_b,
                two_pass_final_dst=scenario.final_dst,
                two_pass_state_dst=scenario.state_dst,
                two_pass_tiles_per_block=scenario.tiles_per_block,
                two_pass_block_tiles=scenario.block_tiles,
                two_pass_partial_last_tile=scenario.partial_last_tile,
                two_pass_start_row=scenario.start_row,
                two_pass_num_rows=scenario.num_rows,
            ),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
            NUM_FACES(4),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
            num_faces=4,
        ),
        unpack_to_dest=True,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    res = untilize_block(
        torch.tensor(res_from_L1, dtype=format_dict[formats.output_format]),
        formats.output_format,
        dims,
    ).to(torch.float64)

    all_rows = selected_rows(block, scenario, range(tile_count))
    mean, var, m2 = column_stats(all_rows)
    checks = []
    if scenario.mode == STREAM:
        first = tile_view(res, scenario.final_tile())
        second = tile_view(res, scenario.final_tile() + 1)
        if scenario.finalize == ROW:
            checks.append(("mean", *row_slot(first, mean)))
            checks.append(("var", *row_slot(second, var)))
        elif scenario.finalize == ROW_VAR_ONLY:
            untouched = tile_view(block, scenario.final_tile()).to(torch.float64)
            checks.append(("mean_tile_untouched", untouched.flatten(), first.flatten()))
            checks.append(("var", *row_slot(second, var)))
        elif scenario.finalize == RAW:
            checks.append(("mean", *raw_slot(first, mean, scenario.group_a)))
            checks.append(("var", *raw_slot(second, var, scenario.group_a)))
        elif scenario.finalize == SPLIT:
            anchor = all_rows[0]
            checks.append(("anchor", *row_slot(first, anchor)))
            checks.append(
                ("anchor_minus_mean", *row_slot(first, anchor - mean, FACE_DIM))
            )
            checks.append(("var", *row_slot(second, var)))
        else:
            total_mean = all_rows.mean()
            total_var = ((all_rows - total_mean) ** 2).mean()
            if not scenario.average_variance:
                total_var = total_var * TILE_DIM
            group = scenario.group_a
            checks.append(
                ("mean", *scalar_slot(first, total_mean, group, range(GROUP_UNITS)))
            )
            checks.append(("var", *scalar_slot(second, total_var, group, range(1))))
    elif scenario.mode == COMBINE:
        final = tile_view(res, scenario.final_dst)
        final_var = tile_view(res, scenario.final_dst + 1)
        checks.append(("mean", *row_slot(final, mean)))
        checks.append(("var", *row_slot(final_var, var)))
        state = tile_view(res, scenario.state_dst)
        state_m2 = tile_view(res, scenario.state_dst + 1)
        checks.append(("state_mean", *raw_slot(state, mean, 0)))
        checks.append(("state_m2", *raw_slot(state_m2, m2, 0)))
    else:
        rows_a = selected_rows(block, scenario, range(scenario.block_tiles))
        rows_b = selected_rows(block, scenario, range(scenario.block_tiles, tile_count))
        mean_a, var_a, _ = column_stats(rows_a)
        mean_b, _, m2_b = column_stats(rows_b)
        state = tile_view(res, scenario.state_dst)
        state_2 = tile_view(res, scenario.state_dst + 1)
        checks.append(("mean_a", *raw_slot(state, mean_a, scenario.group_a)))
        checks.append(("var_a", *raw_slot(state_2, var_a, scenario.group_a)))
        checks.append(("mean_b", *raw_slot(state, mean_b, scenario.group_b)))
        checks.append(("m2_b", *raw_slot(state_2, m2_b, scenario.group_b)))

    atol, rtol = TOLERANCE[formats.output_format]
    failed = []
    for name, golden_lanes, device_lanes in checks:
        if not passed_test(
            golden_lanes.to(torch.float32),
            device_lanes.to(torch.float32),
            DataFormat.Float32,
            custom_atol=atol,
            custom_rtol=rtol,
        ):
            failed.append(name)
    assert not failed, f"two-pass {scenario_name}: {failed} mismatch against golden"
