# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    FACE_DIM,
    TILE_DIM,
    WelfordsGolden,
    get_golden_generator,
)
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
    UNPACKER_ENGINE_SEL,
    WELFORDS,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

WELFORDS_FORMATS = [
    InputOutputFormat(fmt, fmt)
    for fmt in (DataFormat.Float16_b, DataFormat.Float16, DataFormat.Float32)
]

# (atol, rtol) per format: two output steps of relative slack plus a near-zero floor.
WELFORDS_TOLERANCE = {
    DataFormat.Float16_b: (0.02, 2.0**-6),
    DataFormat.Float16: (0.005, 2.0**-9),
    DataFormat.Float32: (1e-4, 1e-4),
}

# Dest units a Face-layout group slot covers, and the tile column each lane of the state
# registers tracks after the row-quad transpose: lane (r, c) -> column QUAD_COLUMN[r](c).
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
    use_lut: bool = True
    partial_last_tile: bool = False
    start_row: int = 0
    num_rows: int = TILE_DIM
    face_layout: bool = False
    final_grouped: bool = False
    final_group_id: int = 0
    final_dst: int = 0
    save_restore: bool = False
    state_grouped: bool = False
    state_group_id: int = 0
    state_dst: int = 0
    save_after_tiles: int = 0
    # Tiles per Dest section; 0 puts every tile in one section. A smaller value runs the
    # stream as tile_count / tiles_per_block sections, so the LREG4/LREG5 state has to
    # survive the section handoffs and bank flips, as it does for a Metal consumer.
    tiles_per_block: int = 0

    @property
    def block(self):
        return self.tiles_per_block or self.tile_count

    def validate(self):
        assert self.tile_count % self.block == 0, "tile_count must be whole blocks"
        assert self.final_dst + 1 < self.block, "finalize tiles must fit in one block"
        assert (
            self.start_row + self.num_rows <= TILE_DIM
        ), "partial row window [start_row, start_row + num_rows) must fit in one tile"
        if self.save_restore:
            assert (
                0 < self.save_after_tiles < self.tile_count
            ), "save point must fall strictly inside the tile stream"
            # The save lands in the current block, so both state tiles must already be folded.
            assert (
                self.state_dst + 1 < self.save_after_tiles % self.block
            ), "both saved-state tiles must already be folded in the current Dest block"

    def final_tile(self):
        """Result tile holding the finalize mean (the variance is the next one)."""
        return self.tile_count - self.block + self.final_dst

    def state_tile(self):
        """Result tile holding the saved mean (M2 is the next one)."""
        return (self.save_after_tiles // self.block) * self.block + self.state_dst


SCENARIOS = {
    "row_2tiles_nolut": Scenario(tile_count=2, use_lut=False),
    "row_4tiles_lut": Scenario(tile_count=4),
    "partial_s5_n17_lut": Scenario(
        tile_count=2, partial_last_tile=True, start_row=5, num_rows=17
    ),
    "partial_s0_n9_nolut": Scenario(
        tile_count=2, use_lut=False, partial_last_tile=True, start_row=0, num_rows=9
    ),
    "partial_s18_n11_lut": Scenario(
        tile_count=2, partial_last_tile=True, start_row=18, num_rows=11
    ),
    "face_2tiles_lut": Scenario(tile_count=2, face_layout=True),
    "roundtrip_ungrouped_face": Scenario(
        tile_count=4,
        face_layout=True,
        final_dst=2,
        save_restore=True,
        save_after_tiles=2,
    ),
    "roundtrip_g1_face_g3": Scenario(
        tile_count=4,
        face_layout=True,
        final_grouped=True,
        final_group_id=3,
        final_dst=2,
        save_restore=True,
        state_grouped=True,
        state_group_id=1,
        save_after_tiles=2,
    ),
    "roundtrip_g5_row_partial_nolut": Scenario(
        tile_count=4,
        use_lut=False,
        partial_last_tile=True,
        start_row=3,
        num_rows=26,
        final_dst=2,
        save_restore=True,
        state_grouped=True,
        state_group_id=5,
        save_after_tiles=2,
    ),
    # Three Dest sections of four tiles each (3x a 32-bit half-sync section): the state is
    # carried in LREG4/LREG5 across two section handoffs.
    "blocked_3x4_row_nolut": Scenario(tile_count=12, use_lut=False, tiles_per_block=4),
    # Blocked, with the save/restore in the middle section at a non-zero state_dst, a
    # partial last tile and the LUT covering all 384 rows.
    "blocked_3x4_dst1_g2_face_g3_partial_lut": Scenario(
        tile_count=12,
        tiles_per_block=4,
        partial_last_tile=True,
        start_row=7,
        num_rows=20,
        face_layout=True,
        final_grouped=True,
        final_group_id=3,
        final_dst=2,
        save_restore=True,
        state_grouped=True,
        state_group_id=2,
        state_dst=1,
        save_after_tiles=7,
    ),
}


def make_stimuli(tile_count, torch_format):
    """Row-major [32, 32 * tile_count] block; tile t is columns [32t, 32t + 32).

    Every tile column has its own offset and spread, so a lane permutation or a dropped or
    duplicated row shows up as a wrong mean or variance.
    """
    torch.manual_seed(0)
    column = torch.arange(TILE_DIM, dtype=torch.float32)
    offset = (column - 15.5) * 0.25
    spread = 0.5 + column / TILE_DIM
    noise = torch.empty(TILE_DIM, tile_count * TILE_DIM).uniform_(-1.0, 1.0)
    block = offset.repeat(tile_count) + spread.repeat(tile_count) * noise
    return block.to(torch_format)


def tile_view(block, tile):
    return block[:, tile * TILE_DIM : (tile + 1) * TILE_DIM]


def face_slot_positions(group_id):
    """(tile_row, tile_col, stat_column) for every lane a Face-layout store writes."""
    positions = []
    for r in range(GROUP_UNITS):
        unit = GROUP_UNITS * group_id + r
        face, face_row = divmod(unit, FACE_DIM)
        tile_row = (face // 2) * FACE_DIM + face_row
        for c in range(LANE_COLUMNS):
            tile_col = (face % 2) * FACE_DIM + 2 * c
            positions.append((tile_row, tile_col, QUAD_COLUMN[r](c)))
    return positions


def face_slot(tile, stat, group_id):
    """(golden, device) lanes of a Face-layout slot."""
    positions = face_slot_positions(group_id)
    device = torch.stack([tile[r, c] for r, c, _ in positions])
    golden = torch.stack([stat[s] for _, _, s in positions])
    return golden, device


def row_slot(tile, stat):
    """(golden, device) of a Row-layout slot: tile row 0 = stat, rows 1-3 = 0."""
    golden = torch.zeros(4, TILE_DIM, dtype=torch.float64)
    golden[0] = stat
    return golden.flatten(), tile[:4].flatten()


@pytest.mark.quasar
@pytest.mark.parametrize("formats", WELFORDS_FORMATS, ids=lambda f: f.input_format.name)
@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
def test_sfpu_welfords_quasar(formats, scenario_name):
    """Welford running per-column mean / population variance over a stream of Dest tiles.

    The state lives in SFPU registers across calls, so one test folds several tiles and
    checks the finalize output (mean at final_dst, variance at final_dst + 1). Scenarios
    cover the LUT and RISC-V reciprocal paths, partial last tiles, Row and Face layouts,
    and a save -> clear -> restore round trip, whose saved state is checked too. The blocked
    scenarios stream the tiles through several Dest sections.
    """
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

    row_blocks = [tile_view(block, t) for t in range(tile_count)]
    if scenario.partial_last_tile:
        end_row = scenario.start_row + scenario.num_rows
        row_blocks[-1] = row_blocks[-1][scenario.start_row : end_row]
    golden = get_golden_generator(WelfordsGolden)(
        row_blocks,
        save_before_block=(
            scenario.save_after_tiles if scenario.save_restore else None
        ),
        state_format=formats.input_format,
    )

    src_A = tilize_block(
        block.flatten(), dims, stimuli_format=formats.input_format
    ).flatten()
    src_B = torch.zeros_like(src_A)

    reciprocal_size = tile_count * TILE_DIM if scenario.use_lut else 0
    configuration = TestConfig(
        "sources/quasar/sfpu_welfords_quasar_test.cpp",
        formats,
        templates=[
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
            DEST_SYNC(),
            WELFORDS(
                reciprocal_size=reciprocal_size,
                partial_last_tile=scenario.partial_last_tile,
                start_row=scenario.start_row,
                num_rows=scenario.num_rows,
                face_layout=scenario.face_layout,
                final_grouped=scenario.final_grouped,
                final_group_id=scenario.final_group_id,
                final_dst=scenario.final_dst,
                save_restore=scenario.save_restore,
                state_grouped=scenario.state_grouped,
                state_group_id=scenario.state_group_id,
                state_dst=scenario.state_dst,
                save_after_tiles=scenario.save_after_tiles,
                tiles_per_block=scenario.tiles_per_block,
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

    checks = []
    final_mean_tile = tile_view(res, scenario.final_tile())
    final_var_tile = tile_view(res, scenario.final_tile() + 1)
    if scenario.face_layout:
        group = scenario.final_group_id if scenario.final_grouped else 0
        checks.append(("mean", *face_slot(final_mean_tile, golden["mean"], group)))
        checks.append(("var", *face_slot(final_var_tile, golden["var"], group)))
    else:
        checks.append(("mean", *row_slot(final_mean_tile, golden["mean"])))
        checks.append(("var", *row_slot(final_var_tile, golden["var"])))

    if scenario.save_restore:
        group = scenario.state_group_id if scenario.state_grouped else 0
        saved_mean_tile = tile_view(res, scenario.state_tile())
        saved_m2_tile = tile_view(res, scenario.state_tile() + 1)
        checks.append(
            ("saved_mean", *face_slot(saved_mean_tile, golden["saved_mean"], group))
        )
        checks.append(
            ("saved_m2", *face_slot(saved_m2_tile, golden["saved_m2"], group))
        )

    atol, rtol = WELFORDS_TOLERANCE[formats.output_format]
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
    assert not failed, f"Welford {scenario_name}: {failed} mismatch against golden"
