# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Regression test: rand stores the whole tile even when another op's init last programmed the
dest address-mode slot it stores under.

The kernel seeds the PRNG once (as ttnn.manual_seed does), fills Dest with a sentinel, runs the
topk init (slot 6 = dest incr 32) and then rand with no further rand init. Every element of the
packed tile that can hold a draw must be one in [0, 1]; a rand that depends on its own init for
the address mode stores only the first row group of each face and leaves the sentinel elsewhere.

rand stores FP32 datums. With a 32-bit Dest every element of the tile is a draw. With a 16-bit
Dest a 32-bit store at row r lands its high half at 16-bit row 2 * (r & ~7) + (r & 7) and its low
mantissa half eight rows below (the ISA's Dst32b-over-Dst16b mapping), so only rows 0-7 of each
face of the packed tile carry draws and rows 8-15 hold opaque low halves; the checks cover the
rows that carry draws. Wormhole and Blackhole both carry the metal rand kernel.
"""

import pytest
import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig

pytestmark = [skip_for_quasar]

ELEMENTS_PER_TILE = 1024
SENTINEL = 2.0  # exact in every format; outside the draw range [0, 1]
# At least 512 draws from a 31-bit grid; even in bfloat16 far more than a quarter are distinct.
MIN_DISTINCT = 128

FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True)


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_rand_stale_addrmod(formats, dest_acc):
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip(
            "16-bit Dest cannot hold Float32 draws exactly; the Float16_b rows cover 16-bit Dest"
        )

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[32, 32],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[32, 32],
        spec_A=StimuliSpec.constant(SENTINEL),
        spec_B=StimuliSpec.constant(SENTINEL),
    )

    configuration = TestConfig(
        "sources/sfpu_rand_stale_addrmod_test.cpp",
        formats,
        templates=[],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result[:ELEMENTS_PER_TILE]
    tile = (
        torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
        .flatten()
        .to(torch.float32)
    )

    # Elements that can hold a draw: all of them with a 32-bit Dest; rows 0-7 of each 16x16 face
    # with a 16-bit Dest (face-major tile layout: element = face * 256 + row * 16 + column).
    rows = torch.arange(ELEMENTS_PER_TILE) % 256 // 16
    draw_rows = (
        torch.ones(ELEMENTS_PER_TILE, dtype=torch.bool)
        if dest_acc == DestAccumulation.Yes
        else rows < 8
    )
    draw_elements = draw_rows.nonzero().flatten()
    draws = tile[draw_rows]

    sentinel_count = int((draws == SENTINEL).sum().item())
    survivors = draw_elements[(draws == SENTINEL).nonzero().flatten()[:8]].tolist()
    assert sentinel_count == 0, (
        f"{sentinel_count} of {draws.numel()} draw elements still hold the sentinel after rand: rand "
        f"stored under the address mode the topk init left (first survivors at elements {survivors})"
    )
    out_of_range = draws[(draws < 0.0) | (draws > 1.0)]
    assert (
        out_of_range.numel() == 0
    ), f"draws outside [0, 1]: {out_of_range[:8].tolist()}"
    distinct = int(torch.unique(draws).numel())
    assert (
        distinct >= MIN_DISTINCT
    ), f"only {distinct} distinct values among the draws; not a rand tile"
