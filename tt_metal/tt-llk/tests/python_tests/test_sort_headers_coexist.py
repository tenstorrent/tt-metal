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

This is a **compile-time** assertion -- the value is in the build succeeding. The kernel
also runs, which shows the combined translation unit executes rather than merely building,
but the run deliberately claims nothing about the helper's offset: the datacopy used to
read DEST back reprograms ``DEST_TARGET_REG_CFG_MATH_Offset_ADDR32`` itself before it
touches DEST, so any offset the helper left is discarded. See the note in
``sources/sort_headers_coexist_test.cpp``. The helper is covered in its real context by the
topk_xl and deepseek_top32_rm kernels.

``dest_acc`` is swept because it is the one axis that changes what the combined TU
actually builds -- it flips ``is_fp32_dest_acc_en`` through both headers' DEST addressing
and the datacopy -- unlike the Dst-row offset that used to be swept here, which no
downstream check could observe.
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
from helpers.utils import passed_test

pytestmark = [skip_for_wormhole, skip_for_quasar]

ELEMENTS_PER_TILE = 1024

# Float16_b only: this test is about header coexistence and one CFG write, not about
# format conversion. A second format would double the build for no extra signal.
FORMATS = input_output_formats([DataFormat.Float16_b], same=True)


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sort_headers_coexist(formats, dest_acc):
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
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
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
        "the datacopy in the translation unit that includes both experimental sort SFPU "
        "headers was corrupted -- the two headers no longer coexist cleanly at runtime. "
        "This does not implicate set_dst_write_addr_offset's value; see the module "
        "docstring for why the offset itself is not observable here."
    )
