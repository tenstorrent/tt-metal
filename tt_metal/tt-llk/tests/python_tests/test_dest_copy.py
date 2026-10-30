# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16_b,
            DataFormat.Int32,
            DataFormat.Int8,
            DataFormat.UInt32,
            DataFormat.UInt16,
            DataFormat.UInt8,
        ],
        same=True,
    ),
)
def test_dump_dest(formats):

    formats = formats[0]

    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("dbg_copy_dest_tile is only available on Blackhole.")

    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    configuration = TestConfig(
        "sources/debug_dest_copy.cpp",
        formats,
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
    )

    res_from_L1 = configuration.run().result

    assert (
        len(res_from_L1) == src_A.numel()
    ), f"Result tensor length {len(res_from_L1)} does not match source length {src_A.numel()}"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # The kernel performs a bit-exact copy through DEST, so the result must
    # match the source exactly (no tolerance needed).
    assert torch.equal(
        src_A.to(torch_format), res_tensor
    ), "L1 -> DEST -> L1 round-trip did not preserve the tile bit-exactly."


# Bit patterns with bit 31 set, which a converting (two's complement <->
# sign-magnitude) view of DEST would change; the second half of each list has
# bit 31 clear as controls. The harness stores Int32 in L1 as sign-magnitude,
# so -1 is the word 0x80000001 and -2147483647 is 0xFFFFFFFF; it cannot store
# 0x80000000 (INT32_MIN is clipped), so that pattern is covered only by the
# UInt32 cells, which are stored as raw bits.
_EDGE_VALUES = {
    DataFormat.Int32: [-1, -2, -2147483647, -65536, 2147483647, 65536, 1, 0],
    DataFormat.UInt32: [
        0x80000000,
        0x80000001,
        0xFFFFFFFE,
        0xFFFFFFFF,
        0x7FFFFFFF,
        0x00010000,
        1,
        0,
    ],
}


def _edge_stimuli(fmt):
    """Random 32x32 tile of ``fmt`` with the edge values planted in two faces."""
    src_A, tile_cnt_A, _, _ = generate_stimuli(
        stimuli_format_A=fmt,
        input_dimensions_A=[32, 32],
        stimuli_format_B=fmt,
        input_dimensions_B=[32, 32],
    )
    edge = torch.tensor(_EDGE_VALUES[fmt], dtype=format_dict[fmt])
    src_A = src_A.flatten().clone()
    src_A[: len(edge)] = edge  # face 0
    src_A[512 : 512 + len(edge)] = edge  # face 2
    return src_A, tile_cnt_A


def _assert_raw_words_equal(src_A, configuration, fmt, what):
    """Compare the result buffer with the stimuli as raw 32-bit L1 words.

    The decoded result cannot do this for Int32: the harness decodes it from
    sign-magnitude, which maps the words 0x80000000 and 0x00000000 to the same
    value and hides which word was actually stored.
    """
    expected = np.frombuffer(StimuliConfig.get_packer(fmt)(src_A), dtype=np.uint32)
    got = np.frombuffer(
        configuration.variant_stimuli.collect_raw_result_bytes(
            TestConfig.TENSIX_LOCATION
        ),
        dtype=np.uint32,
    )
    assert len(got) == len(
        expected
    ), f"Result buffer holds {len(got)} words, source has {len(expected)}"
    mismatch = np.nonzero(got != expected)[0].tolist()
    detail = ", ".join(
        f"[{i}] expected 0x{expected[i]:08X} got 0x{got[i]:08X}" for i in mismatch[:8]
    )
    assert not mismatch, (
        f"{what} is not bit-exact; {len(mismatch)} of {len(expected)} "
        f"L1 words differ: {detail}"
    )


@parametrize(
    formats=input_output_formats([DataFormat.Int32, DataFormat.UInt32], same=True),
)
def test_dest_write_then_pack(formats):
    """RISC-V debug write into DEST, consumed by the packer.

    The packer moves 32-bit integers out of DEST unchanged, so the tile it
    emits must equal the one the debug window was asked to write.
    """
    formats = formats[0]

    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("dbg_copy_dest_tile is only available on Blackhole.")

    src_A, tile_cnt_A = _edge_stimuli(formats.input_format)

    configuration = TestConfig(
        "sources/debug_dest_write_pack.cpp",
        formats,
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_A,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=DestAccumulation.Yes,
    )

    configuration.run()
    _assert_raw_words_equal(
        src_A, configuration, formats.output_format, "RISC-V DEST write -> pack"
    )


@parametrize(
    formats=input_output_formats([DataFormat.Int32, DataFormat.UInt32], same=True),
)
def test_unpack_then_dest_read(formats):
    """Unpacker fills DEST, the RISC-V debug window reads it back.

    The unpacker moves 32-bit integers into DEST unchanged, so the debug
    read must return the tile that was unpacked.
    """
    formats = formats[0]

    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("dbg_copy_dest_tile is only available on Blackhole.")

    src_A, tile_cnt_A = _edge_stimuli(formats.input_format)

    configuration = TestConfig(
        "sources/debug_dest_unpack_read.cpp",
        formats,
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_A,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
    )

    configuration.run()
    _assert_raw_words_equal(
        src_A, configuration, formats.output_format, "unpack -> RISC-V DEST read"
    )
