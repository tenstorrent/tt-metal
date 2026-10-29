# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The pack LLK conversion matrix of Blackhole (cpack_common.h is_packer_to_L1_early_conversion_supported and is_packer_to_L1_late_conversion_supported, asserted
by configure_pack) as the pack_src and output pairs the LLK accepts, and the format pairs the pack sweeps of perf_pack_llkx and test_pack_llkx run."""

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.data_format_inference import infer_data_formats
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation

F = DataFormat
IO = InputOutputFormat

# early conversion stage: DEST register format -> intermediate format (Lf8 is not a python DataFormat; Fp8_e4m3 takes its row)
_EARLY = {
    F.Float32: {F.Float32, F.Tf32, F.Float16_b, F.Bfp8_b, F.Int32, F.Int8, F.UInt8},
    F.Float16_b: {F.Tf32, F.Float16_b, F.Bfp8_b, F.Int8},
    F.Float16: {F.Float16, F.Bfp8, F.Fp8_e4m3, F.Int8},
    F.Int32: {F.Float32, F.Tf32, F.Float16_b, F.Int32, F.Int8, F.UInt8},
    F.UInt16: {F.UInt16},
}
_FP32_LATE = {F.Float32, F.Float16_b, F.Float16, F.Fp8_e4m3, F.Bfp8, F.Bfp8_b, F.Bfp4_b, F.Bfp2_b}
_COMBINED_SRC = {F.Tf32, F.Float16_b, F.Bfp8_b, F.Bfp4_b, F.Bfp2_b, F.Float16, F.Bfp8}


def llk_accepts(pack_src: DataFormat, out: DataFormat) -> bool:
    """is_packer_to_L1_conversion_supported(pack_src, out): early or late stage."""
    if out in _EARLY.get(pack_src, ()):
        return True
    if pack_src == F.Float32:
        return out in _FP32_LATE
    if pack_src in _COMBINED_SRC:
        return out == F.Tf32 or out in _FP32_LATE
    if pack_src == F.Int32:
        return out == F.Int32
    if pack_src == F.UInt16:
        return out == F.UInt16
    if pack_src in (F.Int8, F.UInt8):
        return out in (F.Int8, F.UInt8)
    if pack_src == F.UInt32:
        return out == F.UInt32
    return False


def pack_src_of(formats, dest_acc) -> DataFormat:
    """The packer input format the test configures (test_pack: Float32 for a 32-bit DEST of a float input on Blackhole)."""
    fi, fo = formats.input_format, formats.output_format
    d = infer_data_formats(
        input_format=fi, output_format=fo, is_fp32_dest_acc_en=dest_acc, unpacking_to_dest=fi.is_32_bit() and dest_acc == DestAccumulation.Yes
    )
    src = d.pack_src
    if dest_acc == DestAccumulation.Yes and get_chip_architecture() == ChipArchitecture.BLACKHOLE and not fi.is_integer():
        src = F.Float32
    return src


# Bfp8 (A exponent) is an output only: the stimuli generator has no Bfp8 input
# Tf32 is not an input either: the unpacker configuration rejects srcA Tf32 (helpers/test_config.py)
LLKX_INPUTS = [F.Float32, F.Float16_b, F.Float16, F.Int32, F.UInt16, F.Int8, F.UInt8, F.UInt32, F.Bfp8_b]
LLKX_OUTPUTS = [F.Float32, F.Tf32, F.Float16_b, F.Float16, F.Bfp8_b, F.Bfp8, F.Bfp4_b, F.Bfp2_b, F.Int32, F.Int8, F.UInt8, F.UInt16, F.UInt32, F.Fp8_e4m3]

# pairs the cov2, cov3 and grid modules already run (all the pairs of the six cov3 formats, the grid pairs)
_SIX = [F.Float16_b, F.Float16, F.Float32, F.Int32, F.Bfp8_b, F.Bfp4_b]
COVERED = (
    {(i, o) for i in _SIX for o in _SIX}
    | {(F.Float16_b, F.Bfp8_b), (F.Float32, F.Bfp8_b), (F.Float16_b, F.Bfp4_b), (F.Float32, F.Bfp4_b), (F.Float16_b, F.Float32), (F.Float32, F.Float16_b)}
    | {(n, n) for n in (F.UInt16, F.UInt32, F.UInt8, F.Int8)}
)


def llkx_formats():
    return [IO(i, o) for i in LLKX_INPUTS for o in LLKX_OUTPUTS if (i, o) not in COVERED]


def dest_format_of(input_format, dest_acc) -> DataFormat:
    """The format the unpacker leaves in DEST: integers stay as they are, floats are Float32 with a 32-bit DEST and else a 16-bit float of the input exponent family."""
    if input_format.is_integer():
        return input_format
    if dest_acc == DestAccumulation.Yes:
        return F.Float32
    return F.Float16 if input_format.is_exponent_A() else F.Float16_b


def llkx_accs(formats):
    """DEST accumulation modes of the pair for which the LLK accepts the packer conversion from the format DEST holds, and the harness configures the packer for that format
    (the format inference takes pack_src from the output for integers, which would misread a float DEST)."""
    out = []
    for a in get_valid_dest_accumulation_modes(formats):
        src = pack_src_of(formats, a)
        if llk_accepts(src, formats.output_format) and src == dest_format_of(formats.input_format, a):
            out.append(a)
    return out
