# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Coexistence test for the two experimental sort SFPU headers (Blackhole only).

tt-metal #52713 extracts ``set_dst_write_addr_offset`` out of
``ckernel_sfpu_topk_xl.h`` and ``ckernel_sfpu_deepseek_top32_rm.h`` into a shared
``sfpu/experimental/ckernel_sfpu_set_dst_write_addr_offset.h``, because both previously
defined the identical helper themselves and a math TU including both would fail with a
redefinition error. (tt-blaze papers over the same collision with ``#ifndef`` guards.)

Nothing else in the tree compiles both headers into one translation unit, so the error
that extraction fixes is unreachable from every other test. This test makes it reachable.

It is primarily a **compile-time** assertion -- the value is in the build succeeding --
which is why the runtime assertion is deliberately modest: the kernel calls the shared
helper to rebase the Dst write pointer, sets it back to 0, and then a plain datacopy must
still land correctly. That catches a helper that leaves the offset dirty, without
duplicating the real sort coverage in test_topk_xl.py.

The offset swept is in Dst rows. 0 is the no-op baseline; 2 is what topk_xl itself uses
(``set_dst_write_addr_offset(tile_offset + (col ? 0 : 2))``); 64 is a whole-tile rebase,
the granularity deepseek_top32_rm uses for its multi-tile Dst region.
"""

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.golden_generators import DataCopyGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import SORT_DST_WRITE_OFFSET
from helpers.utils import passed_test

pytestmark = [skip_for_wormhole, skip_for_quasar]

ELEMENTS_PER_TILE = 1024

# Float16_b only: this test is about header coexistence and one CFG write, not about
# format conversion. A second format would double the build for no extra signal.
FORMATS = input_output_formats([DataFormat.Float16_b], same=True)


@parametrize(
    formats=FORMATS,
    dst_write_offset=[0, 2, 64],
)
def test_sort_headers_coexist(formats, dst_write_offset):
    torch.manual_seed(0)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[32, 32],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[32, 32],
    )

    configuration = TestConfig(
        "sources/sort_headers_coexist_test.cpp",
        formats,
        templates=[SORT_DST_WRITE_OFFSET(dst_write_offset)],
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
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=DestAccumulation.No,
    )

    res_from_L1 = configuration.run().result[:ELEMENTS_PER_TILE]
    torch_format = format_dict[formats.output_format]

    generate_golden = get_golden_generator(DataCopyGolden)
    golden = generate_golden(
        src_A.flatten(), formats.output_format, input_format=formats.input_format
    )

    device = torch.tensor(res_from_L1, dtype=torch_format).flatten()
    golden = torch.tensor(golden[:ELEMENTS_PER_TILE], dtype=torch_format).flatten()

    assert passed_test(golden, device, formats.output_format), (
        "the datacopy after set_dst_write_addr_offset("
        f"{dst_write_offset}) -> set_dst_write_addr_offset(0) was corrupted: the shared "
        "helper appears to leave the Dst write offset dirty"
    )
