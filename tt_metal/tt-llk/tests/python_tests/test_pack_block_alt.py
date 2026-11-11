# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""pack_block on output A alternating with pack_block on output B, the two outputs of different format and page size.

The gathered bmm kernels pack the output and the partials this way, so every block is the first block of its output (llk_pack_block takes the per-tile pack, the plain block or
the closed block, never the W offset). The head form is llk_pack_block's choice for that case; the main form packs every tile one by one. Both are checked against the golden
(A: output of the even blocks, B: output of the odd blocks), with packer L1 accumulate off and on for B.
"""

import pytest
import torch
from conftest import skip_for_wormhole
from helpers.bfp_format_utils import QUANTIZE_TO_FORMAT
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    FACES_PER_TILE,
    PackGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, DestSync, PerfRunType, format_dict
from helpers.param_config import parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PACK_BLOCK_ALT,
    PACK_SRC_FROM_OUTPUT,
    RELU_CONFIG,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)
from helpers.unpack import unpack_res_tiles
from helpers.utils import passed_test, result_tensor
from ttexalens.tt_exalens_lib import read_from_device, write_to_device

F16B, F32, B8 = DataFormat.Float16_b, DataFormat.Float32, DataFormat.Bfp8_b
BLOCKS = 6
PAGE = {F32: 4096, B8: 1088, F16B: 2048}
# (output A, output B): the pairs of the gathered bmm kernels, A the output and B the partials
PAIRS = [(F16B, F32), (B8, F32), (B8, F16B)]
# the tile counts of the gathered bmm kernels (out_subblock_w); a 32 bit DEST holds 4 tiles per half section
TILES = [1, 2, 3, 4, 5, 7, 8]


def _golden(src, fmt, first_tile, tiles):
    gen = get_golden_generator(PackGolden)
    out = []
    for t in range(first_tile, first_tile + tiles):
        g = gen(src[t * 1024 : (t + 1) * 1024], fmt, input_dimensions=[32, 32])
        if fmt in QUANTIZE_TO_FORMAT:
            g = QUANTIZE_TO_FORMAT[fmt](g).to(format_dict[fmt])
        out.append(g)
    return torch.cat(out)


@skip_for_wormhole
@parametrize(
    pair=PAIRS,
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    block_tiles=TILES,
    l1_acc=[False, True],
    head_form=[True, False],
)
def test_pack_block_alt(
    pair,
    dest_acc,
    block_tiles,
    l1_acc,
    head_form,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    fmt_a, fmt_b = pair
    wide = dest_acc == DestAccumulation.Yes
    if F32 in pair and not wide:
        pytest.skip(
            "A Float32 output from a 16 bit DEST keeps main's destination write; the gathered kernels pack it from a 32 bit DEST"
        )
    if l1_acc and fmt_b == B8:
        pytest.skip("packer L1 accumulate on a block float output is not used")
    dest_sync = DestSync.Half if block_tiles <= (4 if wide else 8) else DestSync.Full
    if wide and block_tiles > 8:
        pytest.skip("block exceeds the 32 bit DEST section")
    formats = InputOutputFormat(F16B, fmt_a)
    tile_cnt = BLOCKS * block_tiles
    dims = [32, 32 * tile_cnt]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=F16B,
        input_dimensions_A=dims,
        stimuli_format_B=F16B,
        input_dimensions_B=dims,
    )
    # source block k goes to output A (k even) or B (k odd), tile (k // 2) * block_tiles + i of it
    gold_a, gold_b = [], []
    for k in range(BLOCKS):
        (gold_a if k % 2 == 0 else gold_b).append(
            _golden(src_A, fmt_a if k % 2 == 0 else fmt_b, k * block_tiles, block_tiles)
        )
    gold_a, gold_b = torch.cat(gold_a), torch.cat(gold_b)
    tiles_a = tiles_b = (BLOCKS // 2) * block_tiles
    off_b = (tiles_a * PAGE[fmt_a] + 1023) & ~1023
    res_page = 4096

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")
    if run_types is None:
        run_types = [PerfRunType.L1_TO_L1]
    kwargs = {
        "test_name": "sources/pack_block_alt_test.cpp",
        "formats": formats,
        "templates": [
            generate_input_dim(dims, dims),
            TILIZE(),
            DEST_SYNC(dest_sync),
            PACK_BLOCK_ALT(
                f"{fmt_b.name}/{'a' if l1_acc else 'n'}/{'h' if head_form else 'm'}"
            ),
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
            operand_res_tile_size=res_page,
        ),
        "dest_acc": dest_acc,
        "unpack_to_dest": False,
    }
    configuration = create_test_or_perf_config(
        is_perf=is_perf, run_types=run_types, test_config_kwargs=kwargs
    )
    if is_perf:
        configuration.run(perf_report)
        return
    if TestConfig.BUILD_MODE != BuildMode.PRODUCE:
        # the accumulating output starts from zero; a misplaced tile elsewhere cannot match by luck
        write_to_device(
            TestConfig.TENSIX_LOCATION,
            configuration.variant_stimuli.buf_res_addr,
            [0] * (tile_cnt_A * res_page),
        )
    configuration.run()
    raw = read_from_device(
        TestConfig.TENSIX_LOCATION,
        configuration.variant_stimuli.buf_res_addr,
        num_bytes=off_b + tiles_b * PAGE[fmt_b],
    )
    res_a = unpack_res_tiles(
        raw[: tiles_a * PAGE[fmt_a]],
        fmt_a,
        tiles_a,
        False,
        FACES_PER_TILE,
        16,
        tile_stride_bytes=PAGE[fmt_a],
    )
    res_b = unpack_res_tiles(
        raw[off_b : off_b + tiles_b * PAGE[fmt_b]],
        fmt_b,
        tiles_b,
        False,
        FACES_PER_TILE,
        16,
        tile_stride_bytes=PAGE[fmt_b],
    )
    ok_a = passed_test(
        gold_a, result_tensor(res_a, fmt_a, gold_a), fmt_a, print_errors=False
    )
    ok_b = passed_test(
        gold_b, result_tensor(res_b, fmt_b, gold_b), fmt_b, print_errors=False
    )
    assert (
        ok_a and ok_b
    ), f"output A {'ok' if ok_a else 'wrong'}, output B {'ok' if ok_b else 'wrong'}"
