"""Crossover sweep (ad hoc): the closed block against the per-tile pack for every block float output, tiles 2 to 8, SyncHalf and SyncFull, alternating outputs and a single output."""

from dataclasses import dataclass

import pytest
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import FACES_PER_TILE
from helpers.llk_params import DestAccumulation, DestSync
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES, create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PACK_SRC_FROM_OUTPUT,
    RELU_CONFIG,
    TILE_COUNT,
    TILIZE,
    TemplateParameter,
    generate_input_dim,
)

F16B = DataFormat.Float16_b
BFP = [
    DataFormat.Bfp8_b,
    DataFormat.Bfp4_b,
    DataFormat.Bfp2_b,
    DataFormat.Bfp8,
    DataFormat.Bfp4,
    DataFormat.Bfp2,
]
BLOCKS = 6


@dataclass
class PACK_BLOCK_SWEEP(TemplateParameter):
    """<mode: alt|single>/<form: h|m>: h the closed block of the block float output, m its per-tile packs"""

    block_sweep_mode: str

    def convert_to_cpp(self) -> str:
        mode, form = self.block_sweep_mode.split("/")
        return (
            "constexpr DataFormat alt_pack_dst = DataFormat::Float16_b;\n"
            "constexpr bool alt_l1_acc = false;\n"
            f"constexpr bool alt_head_form = {str(form == 'h').lower()};\n"
            f"constexpr bool sweep_single = {str(mode == 'single').lower()};"
        )


def _cases():
    out = []
    for fmt in BFP:
        for mode in ("alt", "single"):
            for acc in (DestAccumulation.Yes, DestAccumulation.No):
                for sync in (DestSync.Half, DestSync.Full):
                    for tiles in range(2, 9):
                        wide = acc == DestAccumulation.Yes
                        if sync == DestSync.Half and tiles > (4 if wide else 8):
                            continue
                        out.append((fmt, mode, acc, sync, tiles))
    return out


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    case=_cases(),
    form=["h", "m"],
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_pack_block_sweep(
    perf_report, case, form, run_types, loop_factor, is_perf
):
    fmt_a, mode, dest_acc, dest_sync, block_tiles = case
    formats = InputOutputFormat(F16B, fmt_a)
    tile_cnt = BLOCKS * block_tiles
    dims = [32, 32 * tile_cnt]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=F16B,
        input_dimensions_A=dims,
        stimuli_format_B=F16B,
        input_dimensions_B=dims,
    )
    kwargs = {
        "test_name": "sources/pack_block_sweep_test.cpp",
        "formats": formats,
        "templates": [
            generate_input_dim(dims, dims),
            TILIZE(),
            DEST_SYNC(dest_sync),
            PACK_BLOCK_SWEEP(f"{mode}/{form}"),
            PACK_SRC_FROM_OUTPUT(),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt_A),
            DEST_INDEX(0),
            RELU_CONFIG(0),
            NUM_FACES(num_faces=FACES_PER_TILE),
            NUM_BLOCKS(BLOCKS),
            NUM_TILES_IN_BLOCK(block_tiles),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            F16B,
            src_B,
            F16B,
            fmt_a,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            operand_res_tile_size=4096,
        ),
        "dest_acc": dest_acc,
        "unpack_to_dest": False,
    }
    create_test_or_perf_config(
        is_perf=True, run_types=run_types, test_config_kwargs=kwargs
    ).run(perf_report)
