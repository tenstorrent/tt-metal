# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
import os
import struct
from abc import ABC, abstractmethod
from ctypes import c_uint32
from dataclasses import dataclass

from .chip_architecture import ChipArchitecture, get_chip_architecture
from .format_config import DataFormat, is_dest_acc_needed
from .golden_generators import TILE_DIMENSIONS
from .llk_params import (
    FPU_BINARY_OPERATIONS,
    REDUCE_OPERATIONS,
    SFPU_BINARY_OPERATIONS,
    SFPU_UNARY_OPERATIONS,
    ApproximationMode,
    BroadcastType,
    CustomMmDestLayout,
    DataCopyType,
    DestAccumulation,
    DestSync,
    DstRoundingMode,
    EltwiseBinaryReuseDestType,
    FastMode,
    FusedSort,
    GatedReduceScale,
    L1Accumulation,
    MathFidelity,
    MathOperation,
    NarrowTile,
    PerfRunType,
    ReduceOrder,
    ReducePool,
    SdpaFwOp,
    SdpaOp,
    SfpuSrcsImpl,
    StableSort,
    StochasticRounding,
    Tilize,
    TopKSortDirection,
    TopKXLChunkBaseMode,
    TopKXLIndexOp,
    TopKXLSortMode,
    Transpose,
    UnpackerEngine,
    VectorMode,
)
from .matmul_sweep import validate_tile_dimensions
from .sfpu_dispatch_constants import (
    DROPOUT_PROBABILITY_MAX,
    DROPOUT_SCALE,
    DROPOUT_SEED,
    RELU_MAX_THRESHOLD,
)


def _fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


# Base parameter classes


@dataclass
class TemplateParameter(ABC):
    @abstractmethod
    def convert_to_cpp(self) -> str:
        pass


@dataclass
class RuntimeParameter(ABC):

    @abstractmethod
    def convert_to_cpp(self) -> str:
        pass

    @abstractmethod
    def convert_to_struct_fields(self) -> tuple[str, str]:
        pass


# === TEMPLATE PARAMETER IMPLEMENTATIONS ===


@dataclass
class THROTTLE_LEVEL(TemplateParameter):
    throttle_level: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int THROTTLE_LEVEL = {self.throttle_level};"


def matmul_reuse_dir_for(formats, math_fidelity, ct_dim: int, rt_dim: int, dest_sync=None) -> int:
    """The reuse direction llk_matmul_forms.h matmul_reuse_dir picks (0 default, 1 hold in0, 2 hold in1): at LoFi a block of 4 or more
    tiles with one 32-bit and one 16-bit operand holds the 32-bit one when it is reused (input_format feeds SrcA = in1, input_format_B SrcB = in0)."""
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE or math_fidelity != MathFidelity.LoFi or ct_dim * rt_dim < 4:
        return 0
    # half sync with a Float32 output is pack bound and measured up to 1.7 percent slower
    if dest_sync == DestSync.Half and formats.output_format == DataFormat.Float32:
        return 0
    wide = (DataFormat.Float32, DataFormat.Tf32)
    half = (DataFormat.Float16, DataFormat.Float16_b)
    in0, in1 = formats.input_format_B, formats.input_format
    if in1 in wide and in0 in half and ct_dim >= rt_dim and rt_dim >= 2:
        return 2
    if in0 in wide and in1 in half and ct_dim < rt_dim and ct_dim >= 2:
        return 1
    return 0


@dataclass(init=False, repr=False, eq=False)
class MATMUL_REUSE_DIR(TemplateParameter):
    """Blackhole matmul that holds the 32-bit operand and streams the 16-bit one (reuse_dir of the LLK matmul); runs main's tile form.

    The direction is kept out of the dataclass fields (a field would add a report column and split the CSV schema); the repr still
    holds it, as the variant hash reads it."""

    def __init__(self, reuse_dir: int = 0):
        self.reuse_dir = reuse_dir

    def __repr__(self) -> str:
        return f"MATMUL_REUSE_DIR({self.reuse_dir})"

    def convert_to_cpp(self) -> str:
        return f"#define MATMUL_REUSE_DIR {self.reuse_dir}"

    @staticmethod
    def for_block(formats, math_fidelity, ct_dim: int, rt_dim: int, dest_sync=None) -> list:
        """[MATMUL_REUSE_DIR(n)] where matmul_reuse_dir is not 0, else []."""
        d = matmul_reuse_dir_for(formats, math_fidelity, ct_dim, rt_dim, dest_sync)
        return [MATMUL_REUSE_DIR(d)] if d else []


@dataclass
class MATMUL_ROW_MOP(TemplateParameter):
    """Blackhole matmul math with one MOP per reuse row."""

    def convert_to_cpp(self) -> str:
        return "#define MATMUL_ROW_MOP"

    @staticmethod
    def for_block(formats, dest_acc, math_fidelity, dest_sync, ct_dim: int, rt_dim: int, kt_dim: int, sections: int = 1) -> list:
        """[MATMUL_ROW_MOP()] where the ttnn matmul kernels run the opt-in row MOP for a half-sync block of the sweep (one inner block, so
        the section packs to the output format): helpers/matmul_prod_keys.py m2_row_mop, less the cells that keep main's form."""
        from helpers.matmul_prod_keys import forced_form, main_form_cell, m2_row_mop

        if get_chip_architecture() != ChipArchitecture.BLACKHOLE or (dest_sync != DestSync.Half and math_fidelity != MathFidelity.LoFi):
            return []
        if matmul_reuse_dir_for(formats, math_fidelity, ct_dim, rt_dim, dest_sync):
            return []
        fp32_dest = dest_acc == DestAccumulation.Yes or is_dest_acc_needed(formats)
        out_format = formats.output_format
        # Float16_b against Bfp8_b at LoFi, full sync and a 32-bit DEST: the row MOP reads slower than the tile MOP on the 7 tile row at K 32 (+188 cycles)
        if (
            math_fidelity == MathFidelity.LoFi
            and dest_sync == DestSync.Full
            and fp32_dest
            and {formats.input_format, formats.input_format_B} == {DataFormat.Float16_b, DataFormat.Bfp8_b}
        ):
            return []
        if main_form_cell(math_fidelity, kt_dim, fp32_dest, formats.input_format, out_format, out_format, False, max(ct_dim, rt_dim), dest_sync == DestSync.Full, min(ct_dim, rt_dim), in_format_b=formats.input_format_B, ct=ct_dim, rt=rt_dim):
            return []
        if dest_sync == DestSync.Half and forced_form(math_fidelity, kt_dim, fp32_dest, formats.input_format, out_format, out_format, False, rt_dim, ct_dim, False, sections, in_format_b=formats.input_format_B):
            return []
        return [MATMUL_ROW_MOP()] if m2_row_mop(math_fidelity, rt_dim, ct_dim, kt_dim, fp32_dest, out_format, out_format) else []


# Operand formats of 8 bits per datum or less (_llk_unpack_AB_matmul_narrow_format_).
MATMUL_NARROW_STREAM_FORMATS = (
    DataFormat.Bfp8,
    DataFormat.Bfp8_b,
    DataFormat.Bfp4_b,
    DataFormat.Bfp2_b,
    DataFormat.Fp8_e4m3,
    DataFormat.Int8,
    DataFormat.UInt8,
)


@dataclass
class MATMUL_STREAM_NARROW(TemplateParameter):
    """Blackhole matmul unpack that streams an operand of 8 bits per datum or less into a 16-bit DEST."""

    def convert_to_cpp(self) -> str:
        return "#define MATMUL_STREAM_NARROW"

    @staticmethod
    def for_block(formats, dest_acc, dest_sync, ct_dim: int, rt_dim: int, in0_r_dim: int = 32, math_fidelity=None) -> list:
        """[MATMUL_STREAM_NARROW()] where _llk_unpack_AB_matmul_stream_narrow_ is true (full sync, 16-bit DEST, not an 8-row in0 tile, nor a 4-row or shorter one with ct_dim below 8, nor a 1x1 block at HiFi4, nor a column of tiles at LoFi), else []."""
        if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
            return []
        if dest_sync != DestSync.Full:
            return []
        if dest_acc == DestAccumulation.Yes or is_dest_acc_needed(formats):
            return []
        if in0_r_dim == 8 or (in0_r_dim <= 4 and ct_dim < 8):
            return []
        if math_fidelity == MathFidelity.HiFi4 and ct_dim == 1 and rt_dim == 1:
            return []
        if math_fidelity == MathFidelity.LoFi and ct_dim == 1 and rt_dim > 1:
            return []
        streamed = formats.input_format if ct_dim >= rt_dim else formats.input_format_B
        return (
            [MATMUL_STREAM_NARROW()] if streamed in MATMUL_NARROW_STREAM_FORMATS else []
        )


# Operand formats whose tile is its faces back to back in L1 (the Z counter steps a tile at a time).
MATMUL_DENSE_STREAM_FORMATS = (DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32)


# (dest sync, rt, ct, kt, static throttle, input format, output format) of blocks where the Z step read slower than main or more than 0.3 cycles
# slower than the config write and read (tt-llk sweep, A B B A on one card); the rest of the keyed blocks gain
MATMUL_STREAM_Z_LOSES = frozenset(
    {
        ("Full", 1, 6, 1, 0, "Float32", "Float16_b"),
        ("Full", 1, 7, 1, 0, "Float32", "Bfp8_b"),
        ("Full", 1, 7, 4, 0, "Float16_b", "Bfp8_b"),
        ("Full", 1, 7, 4, 0, "Float16_b", "Float16_b"),
        ("Full", 1, 7, 4, 0, "Float16_b", "Float32"),
        ("Full", 1, 7, 4, 0, "Float32", "Float32"),
        ("Full", 1, 7, 32, 0, "Float16_b", "Bfp8_b"),
        ("Full", 1, 7, 32, 0, "Float16_b", "Float32"),
        ("Half", 1, 4, 1, 0, "Float16_b", "Bfp8_b"),
        ("Half", 1, 4, 1, 0, "Float16_b", "Float16_b"),
        ("Half", 1, 4, 32, 0, "Float16_b", "Bfp8_b"),
        ("Half", 1, 5, 1, 0, "Float16_b", "Float32"),
        ("Half", 1, 6, 1, 0, "Float16_b", "Bfp8_b"),
        ("Half", 1, 7, 32, 0, "Float16_b", "Bfp8_b"),
        ("Half", 1, 7, 32, 0, "Float16_b", "Float16_b"),
        ("Half", 1, 7, 32, 0, "Float16_b", "Float32"),
        ("Half", 1, 8, 32, 0, "Float16_b", "Bfp8_b"),
        ("Half", 2, 2, 4, 5, "Float16_b", "Bfp8_b"),
        ("Half", 2, 2, 4, 5, "Float16_b", "Float32"),
        ("Half", 2, 3, 1, 0, "Float16_b", "Bfp8_b"),
        ("Half", 2, 3, 4, 0, "Float16_b", "Bfp8_b"),
        ("Half", 2, 3, 4, 0, "Float16_b", "Float16_b"),
        ("Half", 2, 4, 4, 0, "Float16_b", "Bfp8_b"),
    }
)


@dataclass
class MATMUL_STREAM_Z(TemplateParameter):
    """Blackhole matmul unpack that steps the streamed in1 tile by the unpacker Z counter instead of a config read and write."""

    def convert_to_cpp(self) -> str:
        return "#define MATMUL_STREAM_Z"

    @staticmethod
    def for_block(formats, dest_acc, dest_sync, ct_dim: int, rt_dim: int, in0_r_dim: int = 32, math_fidelity=None, in1_faces: int = 4, in1_partial: bool = False, kt_dim: int = 0, throttle: int = 0) -> list:
        """[MATMUL_STREAM_Z()] at LoFi where in1 streams (ct_dim >= rt_dim, 4 or more output tiles), is a dense 4-face format and the narrow stream is off, else []."""
        if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
            return []
        if ct_dim < rt_dim or in1_faces != 4 or in1_partial:
            return []
        if matmul_reuse_dir_for(formats, math_fidelity, ct_dim, rt_dim, dest_sync):
            return []
        # measured: only LoFi rows of 4 or more output tiles never lose; the fidelities above it, 1x1 and 1x2 are pack or hand-off bound
        if math_fidelity != MathFidelity.LoFi or ct_dim * rt_dim < 4:
            return []
        # 32-bit input at half sync loses 1 percent on 1x4 with 4 blocks
        if formats.input_format == DataFormat.Float32 and dest_sync == DestSync.Half:
            return []
        # at half sync the Z step loses where the unpack is not the bound: k 4 on a single row, a 2 wide block at k 1, and k 1 under a static throttle
        if dest_sync == DestSync.Half and (
            (kt_dim == 4 and rt_dim == 1) or (kt_dim == 1 and ct_dim == 2) or (kt_dim == 1 and throttle == 5)
        ):
            return []
        if MATMUL_STREAM_NARROW.for_block(formats, dest_acc, dest_sync, ct_dim, rt_dim, in0_r_dim, math_fidelity):
            return []
        if (dest_sync.name, rt_dim, ct_dim, kt_dim, throttle, formats.input_format.name, formats.output_format.name) in MATMUL_STREAM_Z_LOSES:
            return []
        return [MATMUL_STREAM_Z()] if formats.input_format in MATMUL_DENSE_STREAM_FORMATS else []


def _matmul_unfused_post_cell(formats, math_fidelity, dest_sync, dest_acc, ct_dim: int, rt_dim: int, kt_dim: int, in0_r_dim: int = 32, cov_only: bool = False) -> bool:
    """Half sync cells where the fused post and DEST flip of the harness reads slower than main's separate stalls (ABBA with pure-shift arms, 0.1 to 0.25 cycles):
    Bfp8_b to Float16_b at HiFi2 on a row of 3 or 6 tiles, Float32 to Float32 at HiFi2 with a 16-bit DEST on 2x4 (and with a 32-bit one on 1x4 of an 8 row in0 tile),
    and at LoFi with a 32-bit DEST on 4x1 over four k steps."""
    if dest_sync != DestSync.Half:
        return False
    in_f, out_f = formats.input_format, formats.output_format
    if math_fidelity == MathFidelity.HiFi2 and kt_dim == 1:
        if in_f == out_f == DataFormat.Float32 and dest_acc == DestAccumulation.Yes and (ct_dim, rt_dim) == (4, 1) and in0_r_dim == 8:
            return True
        if cov_only:
            return False
        if in_f == DataFormat.Bfp8_b and out_f == DataFormat.Float16_b and dest_acc == DestAccumulation.No and rt_dim == 1 and ct_dim in (3, 6):
            return True
        if in_f == out_f == DataFormat.Float32 and dest_acc == DestAccumulation.No and (ct_dim, rt_dim) == (2, 4):
            return True
    return (
        not cov_only
        and math_fidelity == MathFidelity.LoFi
        and kt_dim == 4
        and in_f == out_f == DataFormat.Float32
        and dest_acc == DestAccumulation.Yes
        and (ct_dim, rt_dim) == (4, 1)
    )


def _matmul_throttled_h2_cell(formats, math_fidelity, dest_sync, ct_dim: int, rt_dim: int, kt_dim: int, throttle: int) -> bool:
    """One Float16_b tile row of four k steps at HiFi2, full sync and a static throttle of 5: the k loop and the keyed forms read a cycle slower than main's, so it keeps main's."""
    return (
        throttle >= 5
        and math_fidelity == MathFidelity.HiFi2
        and dest_sync == DestSync.Full
        and formats.input_format == formats.output_format == DataFormat.Float16_b
        and (ct_dim, rt_dim, kt_dim) == (1, 1, 4)
    )


@dataclass
class MATMUL_L1_ACC(TemplateParameter):
    """Blackhole matmul pack thread with the packer L1 accumulate switch on, as the ttnn bmm kernels run the later K blocks of a spilled output
    (llk_pack_reconfig_l1_acc(1) after the first pack). The perf loop accumulates into the same L1 tiles; the values do not matter."""

    matmul_l1_acc: L1Accumulation = L1Accumulation.No

    def convert_to_cpp(self) -> str:
        return f"#define MATMUL_L1_ACC {1 if self.matmul_l1_acc == L1Accumulation.Yes else 0}"


@dataclass
class MATMUL_ROW_END_FOLD(TemplateParameter):
    """Blackhole math: the default row form of a block that walks down the held operand (rt_dim > ct_dim) ends each row on a MVMUL that frees both sources and clears the counters, no SETRWC per row."""

    def convert_to_cpp(self) -> str:
        return "#define MATMUL_ROW_END_FOLD"

    @staticmethod
    def for_block(ct_dim: int, rt_dim: int) -> list:
        """[MATMUL_ROW_END_FOLD()] for blocks with rt_dim > ct_dim (the harness shapes never run SFPU ops between init and call), else []."""
        if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
            return []
        return [MATMUL_ROW_END_FOLD()] if rt_dim > ct_dim else []


@dataclass
class MATMUL_K_LOOP(TemplateParameter):
    """Blackhole matmul unpack of the whole inner dimension in one _llk_unpack_AB_matmul_k_ call."""

    def convert_to_cpp(self) -> str:
        return "#define MATMUL_K_LOOP"

    @staticmethod
    def for_block(
        formats,
        dest_acc,
        math_fidelity,
        ct_dim: int,
        rt_dim: int,
        kt_dim: int,
        in0_r_dim: int = 32,
        in0_c_dim: int = 32,
        in1_r_dim: int = 32,
        in1_c_dim: int = 32,
        dest_sync=None,
        throttle: int = 0,
    ) -> list:
        """[MATMUL_K_LOOP()] for 32x32 tiles of one 16-bit, bfp8 or float32 format, where the unpack call per k step bounds the block: 1x1 at LoFi and HiFi2 from kt 4, 2x1 at LoFi from kt 8, 1x2 and 2x2 at LoFi from kt 16, else []."""
        if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
            return []
        if _matmul_throttled_h2_cell(formats, math_fidelity, dest_sync, ct_dim, rt_dim, kt_dim, throttle):
            return []
        if (in0_r_dim, in0_c_dim, in1_r_dim, in1_c_dim) != (32, 32, 32, 32):
            return []
        fmt = formats.input_format
        if not (fmt == formats.input_format_B == formats.output_format):
            return []
        if (fmt, dest_acc) not in (
            (DataFormat.Float16_b, DestAccumulation.No),
            (DataFormat.Float16_b, DestAccumulation.Yes),
            (DataFormat.Bfp8_b, DestAccumulation.No),
            (DataFormat.Float32, DestAccumulation.Yes),
        ):
            return []
        key = (rt_dim, ct_dim) == (1, 1) and kt_dim >= 4 and math_fidelity in (MathFidelity.LoFi, MathFidelity.HiFi2)
        key = key or (
            math_fidelity == MathFidelity.LoFi
            and (((rt_dim, ct_dim) == (2, 1) and kt_dim >= 8) or ((rt_dim, ct_dim) in ((1, 2), (2, 2)) and kt_dim >= 16))
        )
        # a Float32 1x2 block reads 0.3 cycles slower in the k loop than in main's per-step unpack
        key = key and not (fmt == DataFormat.Float32 and (rt_dim, ct_dim) == (1, 2))
        return [MATMUL_K_LOOP()] if key else []


class MATMUL_UNPACK_TTSYNC_EXCL(TemplateParameter):
    """Blackhole matmul ttsync unpack of a kernel whose only unpack calls are ttsync matmul rows: no context-count post and get."""

    def convert_to_cpp(self) -> str:
        return "#define MATMUL_UNPACK_TTSYNC_EXCL"


@dataclass
class MATMUL_IN1_ALT(TemplateParameter):
    """Blackhole matmul with in1 32x16 tiles whose body alternates the DEST row groups (in1_order 1)."""

    def convert_to_cpp(self) -> str:
        return "#define MATMUL_IN1_ALT"

    @staticmethod
    def for_tiles(
        formats,
        dest_acc,
        math_fidelity,
        in0_r_dim: int,
        in0_c_dim: int,
        in1_r_dim: int,
        in1_c_dim: int,
    ) -> list:
        """[MATMUL_IN1_ALT()] for in1 32x16 with in0 other than 16x32, except LoFi with 32-bit DEST, else []."""
        if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
            return []
        if not (in1_r_dim > 16 and in1_c_dim <= 16):
            return []
        if in0_r_dim <= 16 and in0_c_dim > 16:
            return []
        fp32_dest = dest_acc == DestAccumulation.Yes or is_dest_acc_needed(formats)
        if math_fidelity == MathFidelity.LoFi and fp32_dest:
            return []
        return [MATMUL_IN1_ALT()]


@dataclass(init=False, repr=False, eq=False)
class MATMUL_FORM_KEYS(TemplateParameter):
    """Blackhole matmul: the k steps per DEST section and output format matmul_default_form picks the forms by.

    The report already carries both (k_dimm, formats.output), so they are kept out of the dataclass fields; the repr
    still holds them, as the variant hash reads it."""

    def __init__(
        self, kt_per_section: int, output_format: DataFormat, row_le_8: bool = False, unfused_post: bool = False, force_form: str = ""
    ):
        self.kt_per_section = kt_per_section
        self.output_format = output_format
        self.row_le_8 = row_le_8
        self.unfused_post = unfused_post
        self.force_form = force_form
        self.keep_flip = False
        self.in0 = None
        self.in1 = None
        self.pack_src_from_output = True

    def __repr__(self) -> str:
        return f"MATMUL_FORM_KEYS({self.kt_per_section}, {self.output_format.name}, {self.row_le_8}{', unfused_post' if self.unfused_post else ''}{', keep_flip' if self.keep_flip else ''}{', in ' + self.in0.name + ' ' + self.in1.name if self.in0 is not None else ''}{', force ' + self.force_form if self.force_form else ''})"

    def convert_to_cpp(self) -> str:
        return (
            "#define MATMUL_FORM_KEYS\n"
            f"constexpr std::uint32_t MATMUL_KT_PER_SECTION = {self.kt_per_section};\n"
            f"constexpr DataFormat MATMUL_OUT_FORMAT = DataFormat::{self.output_format.name};"
            + ("\n#define MATMUL_ROW_LE_8" if self.row_le_8 else "")
            + ("\n#define MATMUL_UNFUSED_POST" if self.unfused_post else "")
            + ("\n#define MATMUL_KEEP_FLIP_WRITE" if self.keep_flip else "")
            + (f"\n#define MATMUL_KEY_IN0 DataFormat::{self.in0.name}\n#define MATMUL_KEY_IN1 DataFormat::{self.in1.name}" if self.in0 is not None else "")
            + (f"\n#define MATMUL_FORCE_FORM_CHAR '{self.force_form}'" if self.force_form else "")
        )

    @staticmethod
    def for_block(formats, kt_dim: int, ct_dim: int = 0, rt_dim: int = 0, math_fidelity=None, dest_sync=None, dest_acc=None, unfuse_one_tile: bool = False, throttle: int = 0, sections: int = 1, in0_r_dim: int = 32, unfuse_cells: bool = False) -> list:
        keys = MATMUL_FORM_KEYS._for_block(formats, kt_dim, ct_dim, rt_dim, math_fidelity, dest_sync, dest_acc, unfuse_one_tile, throttle, sections, in0_r_dim, unfuse_cells)
        # a Bfp8_b pack below HiFi3 (the unfused post cells) reads slower without the half-sync flip's base write (the bound phase moves): the kernels keep it
        keep = formats.output_format == DataFormat.Bfp8_b and math_fidelity in (MathFidelity.LoFi, MathFidelity.HiFi2)
        for k in keys:
            k.keep_flip = keep
            k.in0, k.in1 = formats.input_format_B, formats.input_format
        return keys

    @staticmethod
    def _for_block(formats, kt_dim: int, ct_dim: int = 0, rt_dim: int = 0, math_fidelity=None, dest_sync=None, dest_acc=None, unfuse_one_tile: bool = False, throttle: int = 0, sections: int = 1, in0_r_dim: int = 32, unfuse_cells: bool = False) -> list:
        """[MATMUL_FORM_KEYS(...)] on Blackhole, else []; row_le_8 when the block's row (the larger of ct_dim and rt_dim) has 8 tiles or fewer."""
        if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
            return []
        unfused = (
            # Bfp8_b input to a Float32 output at half sync reads slower with the fused post and DEST flip than main's separate stalls:
            # at HiFi2 on one k step blocks with a 16-bit DEST (and the one tile block with a 32-bit one), at HiFi3 on the
            # one k step blocks of up to five tiles that are not four rows tall
            unfuse_one_tile
            and dest_sync == DestSync.Half
            and formats.input_format == DataFormat.Bfp8_b
            and formats.output_format == DataFormat.Float32
            and kt_dim == 1
            and (
                (math_fidelity == MathFidelity.HiFi2 and (dest_acc == DestAccumulation.No or (ct_dim, rt_dim) == (1, 1)))
                or (
                    math_fidelity == MathFidelity.HiFi3
                    and dest_acc == DestAccumulation.No
                    and (ct_dim * rt_dim <= 5 or (ct_dim, rt_dim) == (3, 2))
                    and rt_dim < 4
                )
                or (math_fidelity == MathFidelity.HiFi4 and dest_acc == DestAccumulation.No and (ct_dim, rt_dim) == (1, 4))
            )
            or (unfuse_one_tile or unfuse_cells)
            and _matmul_unfused_post_cell(formats, math_fidelity, dest_sync, dest_acc, ct_dim, rt_dim, kt_dim, in0_r_dim, not unfuse_one_tile)
        )
        # a single Float32 tile at HiFi3 and half sync with a 32-bit DEST reads slower with the row MOP: the unknown-k key, which is the tile form
        if (
            (kt_dim, ct_dim, rt_dim) == (1, 1, 1)
            and math_fidelity == MathFidelity.HiFi3
            and dest_sync == DestSync.Half
            and dest_acc == DestAccumulation.Yes
            and formats.output_format == DataFormat.Float32
        ):
            return [MATMUL_FORM_KEYS(0, formats.output_format, True)]
        if _matmul_throttled_h2_cell(formats, math_fidelity, dest_sync, ct_dim, rt_dim, kt_dim, throttle):
            return [MATMUL_FORM_KEYS(0, formats.output_format, True)]
        # a block that changes the reuse direction runs main's tile form and per-row unpack: the unknown-k key
        if matmul_reuse_dir_for(formats, math_fidelity, ct_dim, rt_dim, dest_sync):
            return [MATMUL_FORM_KEYS(0, formats.output_format, True)]
        # 32-bit input at LoFi and full sync with four k steps: the 1x2 block with a 32-bit DEST and 16-bit or bfp8 output, and the 2x1 block
        # with a Float16_b output and a 16-bit DEST, read slower with the row forms: the unknown-k key, which is the tile form
        if (
            kt_dim == 4
            and math_fidelity == MathFidelity.LoFi
            and dest_sync == DestSync.Full
            and formats.input_format == DataFormat.Float32
            and (
                ((ct_dim, rt_dim) == (1, 2) and dest_acc == DestAccumulation.Yes and formats.output_format in (DataFormat.Float16_b, DataFormat.Bfp8_b))
                or ((ct_dim, rt_dim) == (2, 1) and dest_acc == DestAccumulation.No and formats.output_format == DataFormat.Float16_b)
            )
        ):
            return [MATMUL_FORM_KEYS(0, formats.output_format, True)]
        # LoFi, full sync, four k steps, a 3 tile row, 16-bit DEST and Float16_b output: the hoisted unpack of the Float16_b column reads 0.4 cycles
        # slower than main and main's own form does not recover it; the per-row unpack with the row MOP (the Float32 column) does
        if (
            (kt_dim, ct_dim, rt_dim) == (4, 3, 1)
            and math_fidelity == MathFidelity.LoFi
            and dest_sync == DestSync.Full
            and dest_acc == DestAccumulation.No
            and formats.output_format == DataFormat.Float16_b
            and DataFormat.Float32 not in (formats.input_format, formats.input_format_B)
        ):
            return [MATMUL_FORM_KEYS(2, DataFormat.Float32, True)]
        # cells where the keyed forms read slower than main's keep main's form (the unknown key), as the ttnn kernels do
        if math_fidelity is not None and dest_sync in (DestSync.Half, DestSync.Full) and dest_acc is not None:
            from helpers.matmul_prod_keys import forced_form, main_form_cell

            fp32_dest = dest_acc == DestAccumulation.Yes or is_dest_acc_needed(formats)
            forced = (
                forced_form(math_fidelity, kt_dim, fp32_dest, formats.input_format, formats.output_format, formats.output_format, dest_sync == DestSync.Full, rt_dim, ct_dim, False, sections, in_format_b=formats.input_format_B)
            )
            if forced:
                return [MATMUL_FORM_KEYS(1 if kt_dim == 1 else 2, formats.output_format, 0 < max(ct_dim, rt_dim) <= 8, False, forced)]
            if main_form_cell(math_fidelity, kt_dim, fp32_dest, formats.input_format, formats.output_format, formats.output_format, False, max(ct_dim, rt_dim), dest_sync == DestSync.Full, min(ct_dim, rt_dim), in_format_b=formats.input_format_B, ct=ct_dim, rt=rt_dim):
                return [MATMUL_FORM_KEYS(0, formats.output_format, 0 < max(ct_dim, rt_dim) <= 8, unfused)]
        # the forms depend on k only as 1, 2 to 4 or 5 and more; one representative keeps the variant count down
        return [
            MATMUL_FORM_KEYS(
                1 if kt_dim == 1 else (2 if kt_dim <= 4 else 5),
                formats.output_format,
                0 < max(ct_dim, rt_dim) <= 8,
                unfused,
            )
        ]


class MATMUL_PHASE_ELIDE(TemplateParameter):
    """Blackhole matmul that runs the fidelity phases its operand formats can use (matmul_phase_fidelity in llk_matmul_forms.h)."""

    def __init__(self, in0: DataFormat, in1: DataFormat):
        self.in0 = in0
        self.in1 = in1

    def __repr__(self) -> str:
        return f"MATMUL_PHASE_ELIDE({self.in0.name}, {self.in1.name})"

    def convert_to_cpp(self) -> str:
        return (
            "#define MATMUL_PHASE_ELIDE\n"
            f"constexpr DataFormat MATMUL_IN0_FORMAT = DataFormat::{self.in0.name};\n"
            f"constexpr DataFormat MATMUL_IN1_FORMAT = DataFormat::{self.in1.name};"
        )

    @staticmethod
    def fidelity(formats, math_fidelity, dest_acc=None, throttle=0, transpose=None, keep=False):
        """The fidelity the elided matmul runs, the same rule as matmul_phase_fidelity."""
        if get_chip_architecture() != ChipArchitecture.BLACKHOLE or keep:
            return math_fidelity
        if formats.input_format != formats.input_format_B or transpose == Transpose.Yes:
            return math_fidelity
        # Bfp8_b x Bfp8_b HiFi3/HiFi4 with a 32-bit output or DEST runs faster on main's own forms (measured).
        if (
            formats.input_format == formats.input_format_B == DataFormat.Bfp8_b
            and math_fidelity in (MathFidelity.HiFi3, MathFidelity.HiFi4)
            and throttle == 0
            and (formats.output_format == DataFormat.Float32 or dest_acc == DestAccumulation.Yes)
        ):
            return math_fidelity
        lo_a_zero = formats.input_format_B == DataFormat.Bfp4_b
        lo_b_zero = formats.input_format in (DataFormat.Bfp8_b, DataFormat.Bfp4_b)
        if lo_a_zero and lo_b_zero:
            return MathFidelity.LoFi
        if lo_b_zero:
            return MathFidelity.LoFi if math_fidelity == MathFidelity.LoFi else MathFidelity.HiFi2
        if lo_a_zero and math_fidelity == MathFidelity.HiFi2:
            return MathFidelity.LoFi
        return math_fidelity

    @staticmethod
    def keeps_requested(tile_dims, dest_sync, math_fidelity) -> bool:
        """Half sync 1x4 one k step at HiFi2 or HiFi3 on a tile with a partial K face (in0 tile width under 16): the elided form lands at 130.2 where the
        requested one reads 128.7 to 129.3, so it keeps the requested fidelity. ttnn tiles are 16 or 32 wide, so no ttnn kernel reaches the cell."""
        return (
            tile_dims.in0_tile_c_dim < 16
            and dest_sync == DestSync.Half
            and (tile_dims.rt_dim, tile_dims.ct_dim, tile_dims.kt_dim) == (1, 4, 1)
            and math_fidelity in (MathFidelity.HiFi2, MathFidelity.HiFi3)
        )

    @staticmethod
    def for_block(formats, math_fidelity, dest_acc=None, throttle=0, transpose=None, keep=False) -> list:
        """[MATMUL_PHASE_ELIDE(in0, in1)] where the elided fidelity is lower than the requested one, else []."""
        if MATMUL_PHASE_ELIDE.fidelity(formats, math_fidelity, dest_acc, throttle, transpose, keep) == math_fidelity:
            return []
        return [MATMUL_PHASE_ELIDE(formats.input_format, formats.input_format_B)]


@dataclass
class MATMUL_PROD_KEYS(TemplateParameter):
    """The forms of a cell are the ones the ttnn kernels select (helpers/matmul_prod_keys.py): no fused post flip, which ttnn does not run."""

    def convert_to_cpp(self) -> str:
        return "#define MATMUL_PROD_KEYS"


@dataclass
class MATMUL_FORCE_FORM(TemplateParameter):
    """The math MOP kind (0 tile MOP, 1 rows MOP of the default form, 2 row MOP: also pass MATMUL_ROW_MOP) and the hoisted unpack
    of a cell, whatever the forms table says. Measures every form of a cell so any key can be judged without another run.

    Both fields are report columns: the forms of a cell share every other sweep parameter, so without them the rows of the forms
    would collide on one key."""

    form_kind: int = 0
    form_hoist: int = 0

    def __post_init__(self):
        self.form_hoist = int(self.form_hoist)

    def convert_to_cpp(self) -> str:
        return f"#define MATMUL_FORCE_FORM {2 * self.form_kind + self.form_hoist}"


@dataclass
class MATMUL_MIXED_STRIDE(TemplateParameter):
    """Blackhole matmul functional test with operands of different formats: operand A (in0) is strided by the tile size of unpacker 1."""

    def convert_to_cpp(self) -> str:
        return "#define MATMUL_MIXED_STRIDE"


@dataclass
class MATMUL_UNPACK_TTSYNC(TemplateParameter):
    """Blackhole matmul unpack with the row base addresses written through GPRs and WRCFG under Auto TTSync."""

    def convert_to_cpp(self) -> str:
        return "#define MATMUL_UNPACK_TTSYNC"

    @staticmethod
    def for_block(formats, dest_acc, math_fidelity, ct_dim: int, rt_dim: int, kt_dim: int, dest_sync=None) -> list:
        """[MATMUL_UNPACK_TTSYNC()] for the blocks of up to 2x2 tiles that the K loop entry does not take and where the GPR form measured 0 slower than the context poll."""
        if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
            return []
        from helpers.matmul_prod_keys import mixed_main_form

        if mixed_main_form(math_fidelity, kt_dim, dest_acc == DestAccumulation.Yes or is_dest_acc_needed(formats), formats.input_format, formats.input_format_B, formats.output_format, dest_sync == DestSync.Full, ct_dim, rt_dim):
            return []
        if MATMUL_K_LOOP.for_block(formats, dest_acc, math_fidelity, ct_dim, rt_dim, kt_dim):
            return []
        if matmul_reuse_dir_for(formats, math_fidelity, ct_dim, rt_dim, dest_sync):
            return []
        # the cells below measured 0 slower than the context poll over 1440 small-block rows (ABBA, both orders) and are the
        # same cells bmm_unpack_auto_ttsync takes in the metal2 matmul kernel
        half = dest_sync == DestSync.Half
        fid = math_fidelity
        one = rt_dim == 1 and ct_dim == 1
        f32_acc = dest_acc == DestAccumulation.Yes
        in_f, out_f = formats.input_format, formats.output_format
        if kt_dim >= 5:
            take = one and fid in (MathFidelity.HiFi2, MathFidelity.LoFi)
        elif kt_dim >= 2:
            take = (one and (fid in (MathFidelity.HiFi2, MathFidelity.LoFi) or (fid == MathFidelity.HiFi3 and half)) and not (
                fid == MathFidelity.HiFi3 and in_f == DataFormat.Float16_b and out_f == DataFormat.Float32 and not f32_acc
            )) or (fid == MathFidelity.LoFi and half and rt_dim <= 2 and ct_dim <= 2)
            if fid == MathFidelity.LoFi and rt_dim == 1 and ct_dim == 2 and in_f == DataFormat.Float32 and out_f == DataFormat.Float16_b:
                take = False
            if rt_dim == 2 and ct_dim == 2 and in_f == DataFormat.Float32:
                take = False
            # a 2x1 block of Float32 to Float32 with a 16-bit DEST reads 0.3 cycles slower than the poll form
            if fid == MathFidelity.LoFi and rt_dim == 2 and ct_dim == 1 and in_f == out_f == DataFormat.Float32 and not f32_acc:
                take = False
        else:
            take = (
                one
                and half
                and fid in (MathFidelity.HiFi2, MathFidelity.LoFi)
                and not (in_f == DataFormat.Float32 and out_f == DataFormat.Float32 and not f32_acc)
                and not (fid == MathFidelity.HiFi2 and in_f == DataFormat.Bfp8_b and out_f == DataFormat.Float32 and f32_acc)
                and not (fid == MathFidelity.HiFi2 and in_f == DataFormat.Float16_b and out_f == DataFormat.Float32 and not f32_acc)
            )
        return [MATMUL_UNPACK_TTSYNC()] if take else []


@dataclass
class MATH_TRANSPOSE_FACES(TemplateParameter):
    math_transpose_faces: Transpose

    def convert_to_cpp(self) -> str:
        return f"constexpr bool MATH_TRANSPOSE_FACES = {str(self.math_transpose_faces.value).lower()};"


@dataclass
class STOCHASTIC_ROUNDING(TemplateParameter):
    stochastic_rounding: StochasticRounding

    def convert_to_cpp(self) -> str:
        return f"constexpr auto STOCHASTIC_RND = ckernel::{self.stochastic_rounding.value};"


@dataclass
class DATA_COPY_TYPE(TemplateParameter):
    data_copy_type: DataCopyType

    def convert_to_cpp(self) -> str:
        return f"constexpr auto DATA_COPY_TYPE = ckernel::DataCopyType::{self.data_copy_type.value};"


@dataclass
class BROADCAST_TYPE(TemplateParameter):
    broadcast_type: BroadcastType

    def convert_to_cpp(self) -> str:
        return f"constexpr auto BROADCAST_TYPE = ckernel::BroadcastType::{self.broadcast_type.value};"


@dataclass
class SFPU_BCAST_DIM(TemplateParameter):
    """Dest-side SFPU row/col broadcast for ``sfpu_binary_bcast_test.cpp``.

    Distinct from :class:`BROADCAST_TYPE`, which selects unpack-A broadcast on
    the pairing kernel. ``None_`` is unused by the 3-tile kernel; pairing and
    add_top_row pass it so every binary-SFPU variant emits the same CSV column.
    """

    sfpu_bcast_dim: BroadcastType = BroadcastType.None_

    def convert_to_cpp(self) -> str:
        return f"constexpr auto BCAST_DIM = ckernel::BroadcastType::{self.sfpu_bcast_dim.value};"


@dataclass
class ACC_TO_DEST(TemplateParameter):
    acc_to_dest: bool

    def convert_to_cpp(self) -> str:
        return f"constexpr bool ACC_TO_DEST = {str(self.acc_to_dest).lower()};"


@dataclass
class REUSE_DEST_TYPE(TemplateParameter):
    reuse_dest_type: EltwiseBinaryReuseDestType

    def convert_to_cpp(self) -> str:
        return f"constexpr auto REUSE_DEST_TYPE = ckernel::EltwiseBinaryReuseDestType::{self.reuse_dest_type.name};"


@dataclass
class EN_DEST_REUSE(TemplateParameter):
    def convert_to_cpp(self) -> str:
        return "#define EN_DEST_REUSE"


@dataclass
class SFPU_INT_OP(TemplateParameter):
    """Emit a #define to select the integer SFPU operation in a shared C++ test source.

    Supported values: "MUL", "GT", "LT", "LE", "GE".  When omitted the C++ source
    falls through to its default (add_int) path.
    """

    int_op: str = ""

    def convert_to_cpp(self) -> str:
        if self.int_op:
            return f"#define SFPU_INT_OP_{self.int_op.upper()}"
        return ""


@dataclass
class SFPU_BINARY_OP(TemplateParameter):
    """Select the consolidated Quasar binary-SFPU op at compile time.

    Emits ``constexpr ckernel::BinaryOp SFPU_BINARY_OP = ckernel::BinaryOp::<op>;``,
    consumed by ``sfpu_operations_quasar.h``. ``op`` is one of:
    ADD, MUL, DIV, GT, LT, LE, GE, MAX, MIN, ATAN2, COPY_DEST, GCD (reusing the
    LLK BinaryOp enum, like Blackhole — int vs float MUL is disambiguated
    by the math format in the cpp; COPY_DEST forwards the Dest encoding so
    the sfpmem mode matches).
    """

    op: str = "ADD"

    def convert_to_cpp(self) -> str:
        return f"constexpr ckernel::BinaryOp SFPU_BINARY_OP = ckernel::BinaryOp::{self.op};"


def _generate_operation_constants(mathop: MathOperation) -> list[str]:
    """Generate the appropriate operation constants based on the math operation type."""
    constants = []

    if mathop in SFPU_UNARY_OPERATIONS:
        constants.append(
            f"constexpr auto SFPU_UNARY_OPERATION = SfpuType::{mathop.cpp_enum_value};"
        )
    elif mathop in SFPU_BINARY_OPERATIONS:
        constants.append(
            f"constexpr auto SFPU_BINARY_OPERATION = ckernel::BinaryOp::{mathop.cpp_enum_value};"
        )
    elif mathop in FPU_BINARY_OPERATIONS:
        constants.append(
            f"constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::{mathop.cpp_enum_value};"
        )

    return constants


@dataclass
class MATH_OP(TemplateParameter):
    mathop: MathOperation = None
    unary_extra: MathOperation = None
    pool_type: ReducePool = None

    def convert_to_cpp(self) -> str:
        temp_header = []
        if self.mathop:
            temp_header.append("\n// Math operation configuration")
            temp_header.extend(_generate_operation_constants(self.mathop))

            # Handle reduce operations
            if self.mathop in REDUCE_OPERATIONS:
                temp_header.append(
                    f"constexpr auto REDUCE_DIM = ckernel::ReduceDim::{self.mathop.cpp_enum_value};"
                )
                if self.pool_type:
                    temp_header.append(
                        f"constexpr auto POOL_TYPE = ckernel::PoolType::{self.pool_type.value};"
                    )

        # Optional extra unary operation (used when both a binary and unary op
        # need to be present in the same kernel, e.g. binary-eltwise followed by
        # SFPU unary).  If 'unary_op' exists, append its constant.
        # Only add if we haven't already added a unary operation from the main mathop
        if self.unary_extra and (
            self.mathop is None or self.mathop not in SFPU_UNARY_OPERATIONS
        ):
            temp_header.extend(
                [
                    "\n// Additional SFPU unary operation",
                    f"constexpr auto SFPU_UNARY_OPERATION = SfpuType::{self.unary_extra.cpp_enum_value};",
                ]
            )

        return "\n".join(temp_header)


@dataclass
class SFPU_TERNARY_OP(TemplateParameter):
    """Select the ternary SFPU op at compile time.

    Emits ``constexpr auto SFPU_TERNARY_OPERATION = SfpuType::<op>;`` consumed by
    ``sfpu_operations.h``. ``ternary_mathop.cpp_enum_value`` must match the
    ``SfpuType`` enumerator name (e.g. ``addcmul``/``addcdiv``).
    """

    ternary_mathop: MathOperation = None

    def convert_to_cpp(self) -> str:
        return f"constexpr auto SFPU_TERNARY_OPERATION = SfpuType::{self.ternary_mathop.cpp_enum_value};"


@dataclass
class SFPU_TERNARY_SCALAR(TemplateParameter):
    """Scalar multiplier for addcmul/addcdiv, passed as a raw fp32 bit pattern.

    The ternary addc kernels take a ``std::uint32_t value`` reinterpreted as float in
    the SFPU. Emit the bit pattern so the C++ and torch golden agree exactly.
    """

    ternary_scalar_bits: int = 0x40000000  # 2.0f

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t SFPU_TERNARY_SCALAR = {self.ternary_scalar_bits}u;"


@dataclass
class SFPU_CLOBBER_LREGS(TemplateParameter):
    """Fill the SFPU registers with garbage before the op's init (``clobber_sfpu_lregs()``).

    Honoured by ``sfpu_ternary_test.cpp`` only; other harnesses ignore it.
    """

    def convert_to_cpp(self) -> str:
        return "#define SFPU_CLOBBER_LREGS"


@dataclass
class SFPU_BINOP_MODE(TemplateParameter):
    """Select the float unary-with-scalar binop at compile time.

    Emits ``constexpr int SFPU_BINOP_MODE = <n>;`` consumed by
    ``sfpu_binop_scalar_{test,perf}.cpp``, matching the BINOP_MODE enum in
    ``ckernel_sfpu_binop_with_unary.h`` (ADD=0, SUB=1, MUL=2, DIV=3, RSUB=4).
    """

    # Maps MathOperation.cpp_enum_value -> the kernel's BINOP_MODE integer.
    _MODE = {"ADD": 0, "SUB": 1, "MUL": 2, "DIV": 3, "RSUB": 4}

    binop_mathop: MathOperation = None

    def convert_to_cpp(self) -> str:
        return f"constexpr int SFPU_BINOP_MODE = {self._MODE[self.binop_mathop.cpp_enum_value]};"


@dataclass
class SFPU_UNARY_SCALAR(TemplateParameter):
    """Scalar operand for the float unary-with-scalar binops, as raw fp32 bits.

    ``calculate_binop_with_scalar`` decodes it via ``Converter::as_float``; emit
    the bit pattern so the C++ and torch golden agree exactly. For DIV this is
    the host-inverted divisor (1/divisor), since the kernel multiplies.
    """

    value_bits: int = 0x40000000  # 2.0f

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t SFPU_UNARY_SCALAR = {self.value_bits}u;"


@dataclass
class SFPU_UNARY_THRESHOLD(TemplateParameter):
    """Threshold operand for a unary op that clamps against a scalar, as raw fp32 bits.

    Same contract as :class:`SFPU_UNARY_SCALAR` -- emit the bit pattern rather than a
    decimal literal so the kernel and the torch golden agree exactly. Separate from
    SFPU_UNARY_SCALAR so a driver can carry both at once, which
    relu_min_lreg2_regression_test.cpp does: the scalar is its LREG2 poison and the
    threshold is the value relu_min is supposed to clamp to.
    """

    # Field name is deliberately not `value_bits`: test_perf_header_gate requires every
    # parameter field name in the tree to be unique, because the perf CSV takes its column
    # headers from them and SFPU_UNARY_SCALAR already owns `value_bits`.
    threshold_bits: int = 0x40A00000  # 5.0f

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t SFPU_UNARY_THRESHOLD = {self.threshold_bits}u;"


@dataclass
class RAND_RANGE(TemplateParameter):
    """rand output interval ``[from, from + scale]`` as fp32 bits; emitted as macros over the dispatcher defaults."""

    rand_from_bits: int = 0x3F800000  # 1.0f
    rand_scale_bits: int = 0x40000000  # 2.0f

    def convert_to_cpp(self) -> str:
        return (
            f"#define RAND_FROM_BITS {self.rand_from_bits:#010x}u\n"
            f"#define RAND_SCALE_BITS {self.rand_scale_bits:#010x}u"
        )


@dataclass
class RAND_SEED(TemplateParameter):
    """rand PRNG seed; emitted as a macro over the dispatcher default."""

    rand_seed: int = 0x12345678

    def convert_to_cpp(self) -> str:
        return f"#define RAND_SEED {self.rand_seed:#010x}u"


@dataclass
class SFPU_RELU_MIN_INT_THRESHOLD(TemplateParameter):
    """Integer threshold for relu_min's vInt branch, as a two's-complement uint32.

    Emitted as a macro rather than a constexpr for the same reason as
    :class:`SFPU_SHIFT_AMOUNT`: sfpu_operations.h selects on ``#ifdef``, the header is
    shared by every unary test, and only the int32 relu_min sweep sets this, so the rest
    have to keep compiling without it. Unset means the kernel's fixed 5.

    Takes a *signed* Python int and emits its two's-complement pattern, because
    ``_relu_min_`` declares the parameter ``std::uint32_t`` and immediately
    ``static_cast<int>``s it. A negative value is the only way to reach the sign+magnitude
    re-encoding branch inside that wrapper.
    """

    threshold: int = 5

    def convert_to_cpp(self) -> str:
        return f"#define SFPU_RELU_MIN_INT_THRESHOLD {self.threshold & 0xFFFFFFFF}u"


@dataclass
class SFPU_RELU_MAX_THRESHOLD(TemplateParameter):
    """Float threshold for relu_max, emitted as its fp32 bit pattern.

    Emitted as a macro rather than a constexpr for the same reason as
    :class:`SFPU_RELU_MIN_INT_THRESHOLD`: sfpu_operations.h selects on ``#ifdef``, the
    header is shared by every unary test, and only the relu_max threshold sweep sets this.
    Unset means the kernel's fixed 5.0 (RELU_MAX_THRESHOLD on the golden side).

    Takes a Python float and emits its IEEE-754 single bits, which is the encoding
    ``relu_max_tile`` takes (relu6 passes ``0x40c00000u``), so 0.0, -0.0 and a negative
    threshold are all expressible.

    The field is ``relu_max_threshold``, not ``threshold``: parameter field names become
    perf-CSV headers and must be unique across classes (test_perf_header_gate.py), and
    :class:`SFPU_RELU_MIN_INT_THRESHOLD` already owns ``threshold``. The default is the
    golden's RELU_MAX_THRESHOLD, so only the C++ fallback in sfpu_operations.h is a copy.
    """

    relu_max_threshold: float = RELU_MAX_THRESHOLD

    def convert_to_cpp(self) -> str:
        bits = struct.unpack("<I", struct.pack("<f", self.relu_max_threshold))[0]
        return f"#define SFPU_RELU_MAX_THRESHOLD {bits:#010x}u"


@dataclass
class SFPU_UNARY_MAX_MIN_SCALAR(TemplateParameter):
    """Unary max/min scalar, emitted as fp32 bits (float) or two's-complement int32 (int).
    A macro because the dispatcher selects on ``#ifdef``; unset keeps its 0.0f / 1000 default.
    """

    max_min_scalar: float | int = 0.0

    def convert_to_cpp(self) -> str:
        value = self.max_min_scalar
        bits = _fp32_bits(value) if isinstance(value, float) else value & 0xFFFFFFFF
        return f"#define SFPU_UNARY_MAX_MIN_SCALAR {bits:#010x}u"


@dataclass
class SFPU_UNARY_MAX_MIN_SIGN_MAGNITUDE(TemplateParameter):
    """Int32 unary max/min on a sign-magnitude Dest (macro, like SFPU_UNARY_MAX_MIN_SCALAR)."""

    max_min_sign_magnitude: bool = False

    def convert_to_cpp(self) -> str:
        if not self.max_min_sign_magnitude:
            return ""
        return "#define SFPU_UNARY_MAX_MIN_SIGN_MAGNITUDE 1"


@dataclass
class SFPU_SHIFT_AMOUNT(TemplateParameter):
    """Shift amount for the *unary* shift ops (LeftShift / RightShift).

    Emitted as a macro rather than a constexpr because sfpu_operations.h selects on
    ``#ifdef SFPU_SHIFT_AMOUNT``: the header is shared by every unary test, and only the shift
    sweep sets this, so the others have to keep compiling without it. The binary shift ops take
    their amount as a second operand and need none of this.
    """

    shift_amount: int = 3

    def convert_to_cpp(self) -> str:
        return f"#define SFPU_SHIFT_AMOUNT {self.shift_amount}u"


@dataclass
class SFPU_POLYGAMMA_ORDER(TemplateParameter):
    """Order n for polygamma, emitted as the (n, scale) fp32 bit patterns ttnn passes.

    ``scale`` is (-1)^(n+1) * n!, exact in fp32 for every supported order (1..11). Macros
    rather than constexprs for the same reason as :class:`SFPU_SHIFT_AMOUNT`: sfpu_operations.h
    selects on ``#ifdef``, and a test that does not set this keeps trigamma (n = 1).
    """

    polygamma_order: int = 1

    def convert_to_cpp(self) -> str:
        n = self.polygamma_order
        if not 1 <= n <= 11:
            raise ValueError(f"polygamma supports orders 1 to 11, got {n}")
        scale = (-1) ** (n + 1) * math.factorial(n)
        return (
            f"#define SFPU_POLYGAMMA_N_BITS {_fp32_bits(n):#x}u\n"
            f"#define SFPU_POLYGAMMA_SCALE_BITS {_fp32_bits(scale):#x}u"
        )


@dataclass
class SFPU_DROPOUT_PARAMS(TemplateParameter):
    """Dropout probability (p * INT_MAX), seed and scale, as macros (the header selects on ``#ifdef``).

    Fields are ``dropout_``-prefixed because every param field name is a perf-CSV column.
    """

    dropout_probability: int = 0
    dropout_seed: int = DROPOUT_SEED
    dropout_scale: float = DROPOUT_SCALE

    def __post_init__(self):
        assert (
            0 <= self.dropout_probability <= DROPOUT_PROBABILITY_MAX
        ), f"dropout probability {self.dropout_probability:#x} outside 0 .. {DROPOUT_PROBABILITY_MAX:#x}"
        assert (
            0 <= self.dropout_seed <= 0xFFFFFFFF
        ), f"dropout seed {self.dropout_seed:#x} is not 32-bit"

    def convert_to_cpp(self) -> str:
        scale_bits = struct.unpack("<I", struct.pack("<f", self.dropout_scale))[0]
        return "\n".join(
            (
                f"#define SFPU_DROPOUT_PROBABILITY {self.dropout_probability:#010x}u",
                f"#define SFPU_DROPOUT_SEED {self.dropout_seed:#010x}u",
                f"#define SFPU_DROPOUT_SCALE_BITS {scale_bits:#010x}u",
            )
        )


@dataclass
class DISABLE_SRC_ZERO_FLAG(TemplateParameter):
    disable_src_zero_flag: bool

    def convert_to_cpp(self) -> str:
        return f"constexpr bool disable_src_zero_flag = {str(self.disable_src_zero_flag).lower()};"


@dataclass
class MATH_FIDELITY(TemplateParameter):
    math_fidelity: MathFidelity

    def convert_to_cpp(self) -> str:
        return f"constexpr ckernel::MathFidelity MATH_FIDELITY = {self.math_fidelity.cpp_enum_value};"


@dataclass
class APPROX_MODE(TemplateParameter):
    approx_mode: ApproximationMode = ApproximationMode.No

    def convert_to_cpp(self) -> str:
        return f"constexpr bool APPROX_MODE = {self.approx_mode.cpp_enum_value};"


@dataclass
class SFPU_SRCS_IMPL(TemplateParameter):
    sfpu_srcs_impl: SfpuSrcsImpl

    def convert_to_cpp(self) -> str:
        return f"constexpr bool SFPU_SRCS_LOADMACRO = {self.sfpu_srcs_impl.cpp_enum_value};"


@dataclass
class SFPU_TYPED_BF16_STORE(TemplateParameter):
    typed_bf16_store: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool SFPU_TYPED_BF16_STORE = {str(self.typed_bf16_store).lower()};"


@dataclass
class SFPU_INPUT_SCALE(TemplateParameter):
    """``calculate_add_rsqrt``'s ``INPUT_SCALE``: the fp32 factor on x, as raw bits."""

    input_scale_bits: int = 0x3F800000  # 1.0f

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t SFPU_INPUT_SCALE = {self.input_scale_bits}u;"


@dataclass
class SFPU_FAST_APPROX(TemplateParameter):
    """The sqrt/rsqrt family's ``FAST_APPROX`` template argument.

    Emits ``constexpr bool SFPU_FAST_APPROX = <bool>;``. Distinct from
    :class:`APPROX_MODE`: ``APPROX_MODE`` selects which approximation *body* runs
    (SQRT_10-bits vs SQRT_23-bits), while this flag only drops the trailing
    ``v_if(x < 0) -> NaN`` guard in ``_calculate_sqrt_body_``. It is therefore
    unobservable unless the stimuli reach a negative argument.

    Surfaced as ``fast_and_approx`` on the compute API's ``add_rsqrt_tile``.
    """

    fast_approx: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool SFPU_FAST_APPROX = {str(self.fast_approx).lower()};"


@dataclass
class CUSTOM_MM_UNINIT(TemplateParameter):
    """The custom_mm_block_uninit / compressed_custom_mm_block_uninit configuration.

    Emits the three compile-time switches ``custom_mm_uninit_restore_test.cpp`` reads:

    ``UNINIT_DENSE_PACKING``   the ``dense_packing`` template argument, applied to both
                               the run-0 init (W-stride -> 32 rows) and the uninit
                               (W-stride -> 64 rows). The two must agree: a block packed
                               dense and torn down non-dense is not a supported call.
    ``UNINIT_COMPACT_PACKING`` the ``compact_packing`` template argument, likewise applied
                               to both: Z-stride -> 8 rows and W-stride -> 16 rows, then
                               back to 16 and 64. Excludes ``dense_packing``.
    ``UNINIT_RESTORE_MOP``     the ``restore_tile_pack_mop`` template argument, i.e.
                               whether the uninit reinstalls the Default tile-pack MOP.
    ``UNINIT_SKIP``            negative control -- drop the uninit entirely. Not a
                               supported configuration; it exists to prove the restores
                               are load-bearing rather than incidentally redundant.
    ``BLOCK_MOP_NUM_FACES``    the tile geometry the run-0 block-contiguous MOP is
                               programmed with. The pack MOP bakes in tile geometry, so
                               this decides whether the MOP restore is observable at all:
                               at 4 (same geometry as the run-1 pack) restoring and not
                               restoring are indistinguishable, while at 2 (a 16x32 tiny
                               tile) the un-restored MOP packs the wrong face count --
                               the hazard the uninit's comment describes.
    """

    dense_packing: bool = False
    compact_packing: bool = False
    restore_mop: bool = False
    skip: bool = False
    block_mop_num_faces: int = 2

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr bool UNINIT_DENSE_PACKING = {str(self.dense_packing).lower()};",
                f"constexpr bool UNINIT_COMPACT_PACKING = {str(self.compact_packing).lower()};",
                f"constexpr bool UNINIT_RESTORE_MOP = {str(self.restore_mop).lower()};",
                f"constexpr bool UNINIT_SKIP = {str(self.skip).lower()};",
                f"constexpr std::uint32_t BLOCK_MOP_NUM_FACES = {self.block_mop_num_faces}u;",
            ]
        )


@dataclass
class CUSTOM_MM_LAYOUT(TemplateParameter):
    """``CUSTOM_MM_LAYOUT``: the ckernel::CustomMmDestLayout a custom_mm test kernel runs, as its underlying value."""

    dest_layout: CustomMmDestLayout = CustomMmDestLayout.Dense

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint8_t CUSTOM_MM_LAYOUT = {self.dest_layout.value}u; "
            f"// CustomMmDestLayout::{self.dest_layout.name}"
        )


@dataclass
class CUSTOM_MM_CALLS(TemplateParameter):
    """Split a custom_mm test kernel's K over back-to-back execute calls that accumulate into one DEST.

    ``CUSTOM_MM_NUM_CALLS``  number of execute calls; 1 is the single-call kernel. With more than one,
                             the unpacker first leaves a non-zero SrcA clear value (a both-bank SrcB
                             clear only corrupts the SrcA writes it overlaps while unpacker 0's last
                             SrcA clear value is non-zero, and that value outlives kernels), and math
                             waits 2000 cycles before each call, so the unpacker reaches the next call
                             while math still holds the previous call's banks.
    ``CUSTOM_MM_REARM``      compressed kernel only: leave a -inf SrcA clear value after every call,
                             as a max-reduce or top-k between calls would. It undoes the end-of-call
                             stall's clear to 0, which otherwise hides a both-bank clear in the execute.
    """

    num_calls: int = 1
    rearm: bool = False

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr std::uint32_t CUSTOM_MM_NUM_CALLS = {self.num_calls}u;",
                f"constexpr bool CUSTOM_MM_REARM = {str(self.rearm).lower()};",
            ]
        )


@dataclass
class SAMPLING_PRGM0_HAZARD(TemplateParameter):
    """Cross-op vConstFloatPrgm0 hazard switches for ``sfpu_sampling_test.cpp``.

    Emits ``#define SAMPLING_POLLUTE_PRGM0`` and/or ``#define SAMPLING_SKIP_RECIP_INIT``.

    ``pollute``  run ``log_init`` first, standing in for an earlier op in the same kernel
                 that owns vConstFloatPrgm0 (log sets it to ~8.3e-8; the non-legacy
                 reciprocal needs 2.0f).
    ``skip_init`` drop ``sampling_recip_init``. Not a supported call -- it exists so the
                 test can show the init is load-bearing rather than merely present.
    """

    pollute: bool = False
    skip_init: bool = False

    def convert_to_cpp(self) -> str:
        lines = []
        if self.pollute:
            lines.append("#define SAMPLING_POLLUTE_PRGM0")
        if self.skip_init:
            lines.append("#define SAMPLING_SKIP_RECIP_INIT")
        return "\n".join(lines)


@dataclass
class DST_WRITE_ADDR_OFFSET(TemplateParameter):
    """Compile-time knobs for ``set_dst_write_addr_offset_test.cpp``.

    ``OFFSET_ENABLED``   whether the SFPU body calls
                         ``ckernel::sfpu::set_dst_write_addr_offset`` at all. False is the
                         control the helper is measured against: with it absent,
                         ``SFPU_DST_INDEX`` alone decides where the negate lands, so
                         ``helper(N * 64) at dst_index 0`` and ``no helper at dst_index N``
                         must agree bit for bit.
    ``OFFSET_ROWS``      the argument, in Dst ROWS. One 32x32 tile is 64 rows
                         (``DstTileSizeLog2[Tile32x32] == 6``), which is what makes the two
                         real call patterns land where they do: 64 is a whole-tile rebase
                         (deepseek_top32_rm's ``tile_offset``) and 2 is the column-group
                         flip (topk_xl's ``odd_col_offset``).
    ``SFPU_DST_INDEX``   the ``dst_index`` handed to
                         ``_llk_math_eltwise_unary_sfpu_params_``, i.e. where
                         ``_llk_math_eltwise_sfpu_start_`` points the write pointer before
                         the body runs -- and therefore what the helper overwrites.
    ``SFPU_ENABLED``     drop the SFPU op entirely, giving the datacopy-only baseline the
                         negated variants are diffed against.
    """

    offset_enabled: bool = True
    offset_rows: int = 0
    sfpu_dst_index: int = 0
    sfpu_enabled: bool = True

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr bool OFFSET_ENABLED = {str(self.offset_enabled).lower()};",
                f"constexpr std::uint32_t OFFSET_ROWS = {self.offset_rows}u;",
                f"constexpr std::uint32_t SFPU_DST_INDEX = {self.sfpu_dst_index}u;",
                f"constexpr bool SFPU_ENABLED = {str(self.sfpu_enabled).lower()};",
            ]
        )


@dataclass
class PACK_NUM_TILES(TemplateParameter):
    """Tile count for the block/per-tile pack drivers.

    Emits ``constexpr std::uint32_t PACK_NUM_TILES = <n>;``. Distinct from the runtime
    ``TILE_COUNT``: the pack loops here are compile-time bounded so the block MOP's
    outer-loop patching and the per-tile loop stay in step.
    """

    num_tiles: int = 4

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t PACK_NUM_TILES = {self.num_tiles}u;"


@dataclass
class RMSNORM_DEST_REUSE(TemplateParameter):
    """Compile-time knobs for ``rmsnorm_bcast_scalar_dest_reuse_test.cpp``.

    The first four are template arguments (or a template-fixed runtime argument) on the
    LLK pair, so none of them can be a runtime parameter; the last two are driver hooks
    that perturb the state the inits must re-establish:

    ``rmsnorm_num_tiles``
        Outer-loop count of the math MOP *and* the unpack MOP -- one
        ``_llk_unpack_A_`` call walks this many tiles. Bounded by DEST half-sync capacity.
    ``rmsnorm_num_faces``
        Runtime argument to both ``_init_``s, but it sizes the MOP loops, so a variant
        must be built per value. Only 1, 2 and 4 are accepted (``LLK_ASSERT``).
    ``clear_dest``
        Template argument on the math execute; gates the ``ZEROACC`` between the MOVD2B
        and the MOP.
    ``unpack_full_transpose``
        Drives both ``transpose_of_faces`` and ``within_face_16x16_transpose`` on the
        unpack init. This axis exists only because blaze's version of the header won the
        reconciliation, so it is new reachable surface. Its replay-buffer path is
        restricted to one tile and four faces by ``LLK_ASSERT``.
    ``rmsnorm_clobber_unpa_x_end``
        Before the rmsnorm unpack init, the unpack thread leaves unpacker A's X span at one
        row (as ``_llk_unpack_A_top32_rm_init_`` does). The init must re-program it: its
        MOP issues every real UNPACR on SrcA.
    ``rmsnorm_preserve_src_zero_flag``
        Math sets the Src zero flag to keep before the seed datacopy, as ``copy_init``
        does. The rmsnorm math init must restore the operand-driven value.

    The ``rmsnorm_`` prefixed fields match the constants they emit and stay globally
    unique: ``test_perf_header_gate.py`` requires that no two parameter classes declare the
    same field name, because a test passing both would produce two perf-CSV columns with
    the same header. Bare ``num_tiles`` belongs to ``PACK_NUM_TILES`` and bare
    ``num_faces`` to ``NUM_FACES``.
    """

    rmsnorm_num_tiles: int = 1
    rmsnorm_num_faces: int = 4
    clear_dest: bool = False
    unpack_full_transpose: bool = False
    rmsnorm_clobber_unpa_x_end: bool = False
    rmsnorm_preserve_src_zero_flag: bool = False

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr std::uint32_t RMSNORM_NUM_TILES = {self.rmsnorm_num_tiles}u;",
                f"constexpr std::uint32_t RMSNORM_NUM_FACES = {self.rmsnorm_num_faces}u;",
                f"constexpr bool RMSNORM_CLEAR_DEST = {str(self.clear_dest).lower()};",
                "constexpr bool RMSNORM_UNPACK_FULL_TRANSPOSE = "
                f"{str(self.unpack_full_transpose).lower()};",
                "constexpr bool RMSNORM_CLOBBER_UNPA_X_END = "
                f"{str(self.rmsnorm_clobber_unpa_x_end).lower()};",
                "constexpr bool RMSNORM_PRESERVE_SRC_ZERO_FLAG = "
                f"{str(self.rmsnorm_preserve_src_zero_flag).lower()};",
            ]
        )


@dataclass
class REDUCE_BLOCK_CT_DIM(TemplateParameter):
    """Compile-time block width (in tiles) for the block-based reduce_block_max_row LLKs.

    A standalone one-line constant, deliberately *not* routed through the matmul-centric
    ``INPUT_DIMENSIONS`` bundle: this pure block-reduce test has no use for that bundle's
    other fields (``FULL_RT_DIM`` / ``FULL_CT_DIM`` / ``BLOCK_RT_DIM``) or its
    ``generate_input_dim`` tile-shape validation, and it needs only a plain compile-time
    block width. The distinct name (``REDUCE_BLOCK_CT_DIM``, not ``BLOCK_CT_DIM``) also
    preempts a redefinition clash if ``INPUT_DIMENSIONS`` — the sole emitter of
    ``BLOCK_CT_DIM`` — is ever added to this test: the header generator concatenates every
    param's ``convert_to_cpp()`` with no de-dup, so two same-named ``constexpr`` lines would
    fail to compile. (This test does not currently emit ``BLOCK_CT_DIM``.)
    """

    reduce_block_ct_dim: int

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t REDUCE_BLOCK_CT_DIM = {self.reduce_block_ct_dim};"
        )


@dataclass
class USE_RUNTIME(TemplateParameter):
    """Selects the runtime (dynamic block_ct_dim) reduce_block_max_row LLK family."""

    use_runtime: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool USE_RUNTIME = {str(self.use_runtime).lower()};"


@dataclass
class REINIT_MODE(TemplateParameter):
    """Selects the reduce_block_max_row re-arm path: 0=none, 1=short (MOP + addrmods),
    2=minimal (ADDR_MOD_1/2/6), 3=addrmod-only reinit (ADDR_MOD_1/2/3/6)."""

    reinit_mode: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int REINIT_MODE = {self.reinit_mode};"


@dataclass
class CLOBBER_OP(TemplateParameter):
    """Op run between reduce init and reinit to overwrite the reduce MOP/addrmods
    (reconfig-escape guard for the reinit paths):
    0=none, 1=eltwise binary (all addrmods + MOP), 2=minimal_safe (ADDR_MOD_1/2/6 only),
    3=addrmod_all (ADDR_MOD_1/2/3/6).
    """

    clobber_op: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int CLOBBER_OP = {self.clobber_op};"


@dataclass
class RESPECT_TRIGGER(TemplateParameter):
    """Enable the reduce_block_max_row producer/consumer trigger handshake.

    When true, the unpack splits the block reduce into two half-width MOP runs
    separated by a HW semaphore wait (FPU_SFPU), so the reduce can start on the
    first half of the block before the second half is signalled. The test's PACK
    thread plays the producer (posts the tokens). Requires an even block_ct_dim
    (the unpack MOP outerloop is block_ct_dim / 2)."""

    respect_trigger: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool RESPECT_TRIGGER = {str(self.respect_trigger).lower()};"


@dataclass
class OVERLAP_FIRST_HALF(TemplateParameter):
    """Overlap the first-half reduce with the second-half pack (runtime family only).

    When true (and RESPECT_TRIGGER + USE_RUNTIME), the unpack's first half gates on
    the early UNPACK_MATH_DONE token instead of FPU_SFPU, so run()#1 overlaps the
    second-half pack. Ignored by the compile-time unpack family (no overlap path)."""

    overlap_first_half: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool OVERLAP_FIRST_HALF = {str(self.overlap_first_half).lower()};"


@dataclass
class ITERATIONS(TemplateParameter):
    iterations: int = 8

    def convert_to_cpp(self) -> str:
        return f"constexpr int ITERATIONS = {self.iterations};"


@dataclass
class FAST_MODE(TemplateParameter):
    fast_mode: FastMode = FastMode.No

    def convert_to_cpp(self) -> str:
        return f"constexpr bool FAST_MODE = {str(self.fast_mode.value).lower()};"


@dataclass
class CLAMP_NEGATIVE(TemplateParameter):
    clamp_negative: bool = True

    def convert_to_cpp(self) -> str:
        return f"constexpr bool CLAMP_NEGATIVE = {str(self.clamp_negative).lower()};"


@dataclass
class STABLE_SORT(TemplateParameter):
    stable_sort: StableSort = StableSort.No

    def convert_to_cpp(self) -> str:
        return f"constexpr bool STABLE_SORT = {str(self.stable_sort.value).lower()};"


@dataclass
class FUSED_SORT(TemplateParameter):
    fused_sort: FusedSort = FusedSort.No

    def convert_to_cpp(self) -> str:
        return f"constexpr bool FUSED_SORT = {str(self.fused_sort.value).lower()};"


@dataclass
class DEST_SYNC(TemplateParameter):
    dest_sync: DestSync = DestSync.Half

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr auto dest_sync = ckernel::DstSync::Sync{self.dest_sync.name};"
        )


@dataclass
class TILIZE(TemplateParameter):
    tilize: Tilize = Tilize.No

    def convert_to_cpp(self) -> str:
        return f"constexpr bool tilize_en = {str(self.tilize.value).lower()};"


@dataclass
class PACK_BLOCK(TemplateParameter):
    """Pack every dest block with one _llk_pack_block_ run (Blackhole)."""

    pack_block: bool = False  # 2 and 3 are ints: the block calls llk_pack_block makes, not the opt-in fast forms; 3: the W offset forms

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr bool pack_block_en = {str(bool(self.pack_block)).lower()};\n"
            f"constexpr bool pack_block_default = {str(self.pack_block == 2).lower()};\n"
            f"constexpr bool pack_block_w = {str(self.pack_block == 3).lower()};"
        )


class PACK_RMW_DEST(TemplateParameter):
    """Opt-in byte-wise destination write of the default packs (Blackhole), see _llk_pack_rmw_dest_init_; on when LLK_PACK_RMW=1."""

    def convert_to_cpp(self) -> str:
        return "#define PACK_RMW_DEST" if os.environ.get("LLK_PACK_RMW") == "1" else ""


class MATMUL_PACK_BLOCK(TemplateParameter):
    """Output pack form of the matmul kernels (see matmul_pack_form), picked from columns the test already reports, so
    it adds no column: 0 per tile, 1 one block run, 2 the closed block, 4 the dense replay run of small tiles."""

    def __init__(self, pack_block: int = 0):
        self.matmul_pack_block = int(pack_block)

    def __repr__(self) -> str:
        return f"MATMUL_PACK_BLOCK({self.matmul_pack_block})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, MATMUL_PACK_BLOCK)
            and other.matmul_pack_block == self.matmul_pack_block
        )

    def __hash__(self) -> int:
        return hash(self.matmul_pack_block)

    def convert_to_cpp(self) -> str:
        return f"#define MATMUL_PACK_FORM {self.matmul_pack_block}"


@dataclass
class TINY_PACK_MODE(TemplateParameter):
    """Tiny-tile block pack (Blackhole): 0 block-contiguous run, 1 closed run per tile, 2 _llk_pack_ per tile, 3 dense replay run"""

    tiny_pack_mode: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int tiny_pack_mode = {self.tiny_pack_mode};"


@dataclass
class ENABLE_2X_FORMAT(TemplateParameter):
    enable_2x_format: bool = False

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr bool ENABLE_2X_FORMAT = {str(self.enable_2x_format).lower()};"
        )


@dataclass
class ENABLE_DIRECT_INDEXING(TemplateParameter):
    enable_direct_indexing: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool ENABLE_DIRECT_INDEXING = {str(self.enable_direct_indexing).lower()};"


@dataclass
class UNPACKER_ENGINE_SEL(TemplateParameter):
    unpacker_engine_sel: UnpackerEngine = UnpackerEngine.UnpA

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t UNPACKER_ENGINE_SEL = p_unpacr::{self.unpacker_engine_sel.value};"


@dataclass
class VECTOR_MODE(TemplateParameter):
    vector_mode: VectorMode = VectorMode.RC

    def convert_to_cpp(self) -> str:
        return f"constexpr auto VECTOR_MODE = {self.vector_mode.cpp_enum_value};"


@dataclass
class PERF_RUN_TYPE(TemplateParameter):
    perf_run_type: PerfRunType

    def convert_to_cpp(self) -> str:
        return (
            f"\nconstexpr auto PERF_RUN_TYPE = PerfRunType::{self.perf_run_type.name};"
        )


@dataclass
class REDUCE_POOL_TYPE(TemplateParameter):
    reduce_pool_type: ReducePool

    def convert_to_cpp(self) -> str:
        return f"constexpr auto POOL_TYPE = ckernel::PoolType::{self.reduce_pool_type.value};"


@dataclass
class REDUCE_ORDER(TemplateParameter):
    """Order of the chained SFPU reduce passes in sfpu_reduce_multidim_test.cpp, all under one
    shared init_reduce (see ReduceOrder; the kernel names the values REDUCE_ORDER_*).
    """

    reduce_order: ReduceOrder = ReduceOrder.ColRow

    def convert_to_cpp(self) -> str:
        return f"constexpr int REDUCE_ORDER = {self.reduce_order.value};"


@dataclass
class SDPA_OP(TemplateParameter):
    sdpa_op: SdpaOp = SdpaOp.RecipIter

    def convert_to_cpp(self) -> str:
        return f"constexpr int SDPA_OP = {self.sdpa_op.value};"


@dataclass
class SDPA_EXP_SCALE(TemplateParameter):
    scale_bf16: int = 0x3F80  # bf16(1.0)

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint16_t EXP_SCALE_BF16 = {self.scale_bf16}u;"


@dataclass
class SDPA_SOFTPLUS_PARAMS(TemplateParameter):
    softplus_beta_bits: int = 0x3F800000  # 1.0f
    softplus_beta_reciprocal_bits: int = 0x3F800000  # 1.0f
    softplus_threshold_bits: int = 0x41A00000  # 20.0f

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr std::uint32_t SOFTPLUS_BETA_BITS = {self.softplus_beta_bits}u;",
            f"constexpr std::uint32_t SOFTPLUS_BETA_RECIPROCAL_BITS = {self.softplus_beta_reciprocal_bits}u;",
            f"constexpr std::uint32_t SOFTPLUS_THRESHOLD_BITS = {self.softplus_threshold_bits}u;",
        ]
        return "\n".join(lines)


@dataclass
class SDPA_FW_OP(TemplateParameter):
    sdpa_fw_op: SdpaFwOp = SdpaFwOp.Recip

    def convert_to_cpp(self) -> str:
        return f"constexpr int SDPA_FW_OP = {self.sdpa_fw_op.value};"


@dataclass
class TOPK(TemplateParameter):
    topk_k: int = 0
    topk_matrix_width: int = 0
    topk_sort_direction: TopKSortDirection = TopKSortDirection.Descending
    topk_stable_sort: bool = False
    # Fused-key stable mode: [bf16 value | u16 index] packed 32-bit keys sorted by the unstable
    # network (requires dest_acc=Yes; mutually exclusive with topk_stable_sort).
    topk_fused_stable: bool = False
    # Rank-stamped stable mode: sign-conditioned local-rank tags in the value words' lo16, true
    # indices riding index tracking; the unstable network sorts the tagged keys (requires
    # dest_acc=Yes; mutually exclusive with both other stable modes).
    topk_rank_stamped: bool = False
    # Rank-stamped only: rank tag field width in the value word's low bits (16 for bf16 values,
    # narrower for fp32 keys whose low mantissa bits are zero).
    topk_tag_bits: int = 16

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr std::uint32_t TOPK_K = {self.topk_k};",
            f"constexpr std::uint32_t TOPK_LOGK = {int(math.log2(self.topk_k))};",
            f"constexpr std::uint32_t TOPK_NUM_ITERATIONS = {int(math.log2(self.topk_matrix_width // TILE_DIMENSIONS[1] // 2))};",
            f"constexpr std::uint32_t TOPK_SORT_DIRECTION = {self.topk_sort_direction.value};",
            f"constexpr bool TOPK_STABLE_SORT = {str(self.topk_stable_sort).lower()};",
            f"constexpr bool TOPK_FUSED_STABLE = {str(self.topk_fused_stable).lower()};",
            f"constexpr bool TOPK_RANK_STAMPED = {str(self.topk_rank_stamped).lower()};",
            f"constexpr std::uint32_t TOPK_TAG_BITS = {self.topk_tag_bits};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "std::uint32_t TOPK_K;",
            "std::uint32_t TOPK_LOGK;",
            "std::uint32_t TOPK_NUM_ITERATIONS;",
            "std::uint32_t TOPK_SORT_DIRECTION;",
            "bool TOPK_STABLE_SORT;",
            "bool TOPK_FUSED_STABLE;",
            "bool TOPK_RANK_STAMPED;",
            "std::uint32_t TOPK_TAG_BITS;",
        ]
        return "\n".join(lines), "IIII???I"


@dataclass
class GENERALIZED_MOE_GATE(TemplateParameter):
    """Compile-time configuration for the generalized_moe_gate test.

    ``read_base``/``from_*``/``to_*`` are SFPU column offsets; a run occupies the pair ``{lo, hi}``.
    ``row_src``/``row_dst``/``srcb`` name copy4rows' 4-row DEST blocks and its SrcB scratch window.
    ``eps`` and ``scale`` are float bit patterns, as the LLK takes them.
    ``sections`` is how many DEST sections the kernel runs; 2 puts the second in the upper
    half under DstSync::Half and packs it to buffer_Res[4..7].
    ``sigmoid`` selects the op's enable_sigmoid front-end: transpose, sigmoid, then a RELOAD
    binary reading SrcA back out of DEST.
    """

    mode: int = 0
    sub_op: int = 0
    grouped: bool = False
    topk: int = 8
    softmax: bool = False
    produce_run: bool = False
    reload: bool = False
    eps: int = 0
    scale: int = 0
    read_base: int = 0
    from_lo: int = 0
    from_hi: int = 2
    to_lo: int = 0
    to_hi: int = 2
    field: int = 0
    idx_offset: int = 0
    row_src: int = 0
    row_dst: int = 4
    srcb: int = 16
    second_copy: bool = False
    pre_copy4rows: bool = False
    row_src_2: int = 0
    row_dst_2: int = 8
    srcb_2: int = 20
    d2b_dst: int = 0
    b2d_base: int = 0
    sections: int = 1
    sigmoid: bool = False
    transpose_of_faces: bool = True
    do_extra_scale: bool = False
    extra_scale: int = 0x3F800000
    output_tiles: int = 3

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr int GMG_MODE = {self.mode};",
            f"constexpr int GMG_SUB_OP = {self.sub_op};",
            f"constexpr bool GMG_GROUPED = {str(self.grouped).lower()};",
            f"constexpr std::uint32_t GMG_TOPK = {self.topk};",
            f"constexpr bool GMG_SOFTMAX = {str(self.softmax).lower()};",
            f"constexpr bool GMG_PRODUCE_RUN = {str(self.produce_run).lower()};",
            f"constexpr bool GMG_RELOAD = {str(self.reload).lower()};",
            f"constexpr std::uint32_t GMG_EPS = {self.eps};",
            f"constexpr std::uint32_t GMG_SCALE = {self.scale};",
            f"constexpr std::uint32_t GMG_READ_BASE = {self.read_base};",
            f"constexpr std::uint32_t GMG_FROM_LO = {self.from_lo};",
            f"constexpr std::uint32_t GMG_FROM_HI = {self.from_hi};",
            f"constexpr std::uint32_t GMG_TO_LO = {self.to_lo};",
            f"constexpr std::uint32_t GMG_TO_HI = {self.to_hi};",
            f"constexpr std::uint32_t GMG_FIELD = {self.field};",
            f"constexpr std::uint32_t GMG_IDX_OFFSET = {self.idx_offset};",
            f"constexpr std::uint32_t GMG_ROW_SRC = {self.row_src};",
            f"constexpr std::uint32_t GMG_ROW_DST = {self.row_dst};",
            f"constexpr std::uint32_t GMG_SRCB = {self.srcb};",
            f"constexpr bool GMG_SECOND_COPY = {str(self.second_copy).lower()};",
            f"constexpr bool GMG_PRE_COPY4ROWS = {str(self.pre_copy4rows).lower()};",
            f"constexpr std::uint32_t GMG_ROW_SRC_2 = {self.row_src_2};",
            f"constexpr std::uint32_t GMG_ROW_DST_2 = {self.row_dst_2};",
            f"constexpr std::uint32_t GMG_SRCB_2 = {self.srcb_2};",
            f"constexpr std::uint32_t GMG_D2B_DST = {self.d2b_dst};",
            f"constexpr std::uint32_t GMG_B2D_BASE = {self.b2d_base};",
            f"constexpr std::uint32_t GMG_SECTIONS = {self.sections};",
            f"constexpr bool GMG_SIGMOID = {str(self.sigmoid).lower()};",
            f"constexpr bool GMG_TRANSPOSE_OF_FACES = {str(self.transpose_of_faces).lower()};",
            f"constexpr bool GMG_DO_EXTRA_SCALE = {str(self.do_extra_scale).lower()};",
            f"constexpr std::uint32_t GMG_EXTRA_SCALE = {self.extra_scale};",
            f"constexpr std::uint32_t GMG_OUTPUT_TILES = {self.output_tiles};",
        ]
        return "\n".join(lines)


@dataclass
class DEEPSEEK_MOE_GATE(TemplateParameter):
    """Compile-time configuration for the deepseek_moe_gate test."""

    dmg_mode: int = 0
    dmg_sub_op: int = 0
    dmg_sigmoid: bool = False
    dmg_reload: bool = False
    dmg_eps: int = 0
    dmg_scale: int = 0

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr int DMG_MODE = {self.dmg_mode};",
            f"constexpr int DMG_SUB_OP = {self.dmg_sub_op};",
            f"constexpr bool DMG_SIGMOID = {str(self.dmg_sigmoid).lower()};",
            f"constexpr bool DMG_RELOAD = {str(self.dmg_reload).lower()};",
            f"constexpr std::uint32_t DMG_EPS = {self.dmg_eps};",
            f"constexpr std::uint32_t DMG_SCALE = {self.dmg_scale};",
        ]
        return "\n".join(lines)


@dataclass
class HADAMARD(TemplateParameter):
    """Compile-time configuration for the H128 Hadamard test."""

    hadamard_normalize: bool = True
    h16_tile_index: int = 0

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr bool HADAMARD_NORMALIZE = {str(self.hadamard_normalize).lower()};",
            f"constexpr std::uint32_t HADAMARD_H16_TILE_INDEX = {self.h16_tile_index};",
        ]
        return "\n".join(lines)


@dataclass
class ROPE(TemplateParameter):
    ht: int = 1
    wt: int = 1
    x_base: int = 0
    x_stride: int = 64
    cos_base: int = 64
    sin_base: int = 128
    cs_stride: int = 64
    fused_cos_sin: bool = False
    tile_h: int = 1
    cos_sin_per_row: bool = False
    has_scale: bool = False
    scale_fp32: int = 0

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr bool ROPE_FUSED_COS_SIN = {str(self.fused_cos_sin).lower()};",
            f"constexpr std::uint32_t ROPE_TILE_H = {self.tile_h};",
            f"constexpr bool ROPE_COS_SIN_PER_ROW = {str(self.cos_sin_per_row).lower()};",
            f"constexpr std::uint32_t ROPE_HT = {self.ht};",
            f"constexpr std::uint32_t ROPE_WT = {self.wt};",
            f"constexpr std::uint32_t ROPE_X_BASE = {self.x_base};",
            f"constexpr std::uint32_t ROPE_X_STRIDE = {self.x_stride};",
            f"constexpr std::uint32_t ROPE_COS_BASE = {self.cos_base};",
            f"constexpr std::uint32_t ROPE_SIN_BASE = {self.sin_base};",
            f"constexpr std::uint32_t ROPE_CS_STRIDE = {self.cs_stride};",
            f"constexpr bool ROPE_HAS_SCALE = {str(self.has_scale).lower()};",
            f"constexpr std::uint32_t ROPE_SCALE_FP32 = {hex(self.scale_fp32)};",
        ]
        return "\n".join(lines)


@dataclass
class TOPK_XL(TemplateParameter):
    k: int = 512
    num_chunks: int = 1
    tail_elements: int = 512
    num_rows: int = 1
    index_op: TopKXLIndexOp = TopKXLIndexOp.RowMajor
    group_id: int = 0
    group_shift: int = 16
    core_id: int = 0
    sort_direction: TopKSortDirection = TopKSortDirection.Descending
    fused_reduce: bool = False
    chunk_base_mode: TopKXLChunkBaseMode = TopKXLChunkBaseMode.Static
    chunk_base: int = 0
    fused_e2e: bool = False
    seg_base: int = 0
    sort_mode: TopKXLSortMode = TopKXLSortMode.Dispatch
    lsb_row_major: bool = False
    reinit_after_copy: bool = False
    blaze_compat: bool = False

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr std::uint32_t TOPK_XL_K = {self.k};",
            f"constexpr std::uint32_t TOPK_XL_NUM_CHUNKS = {self.num_chunks};",
            f"constexpr std::uint32_t TOPK_XL_TAIL_ELEMENTS = {self.tail_elements};",
            f"constexpr std::uint32_t TOPK_XL_NUM_ROWS = {self.num_rows};",
            f"constexpr std::uint32_t TOPK_XL_INDEX_OP = {self.index_op.value};",
            f"constexpr std::uint32_t TOPK_XL_GROUP_ID = {self.group_id};",
            f"constexpr std::uint32_t TOPK_XL_GROUP_SHIFT = {self.group_shift};",
            f"constexpr std::uint32_t TOPK_XL_CORE_ID = {self.core_id};",
            f"constexpr bool TOPK_XL_ASCENDING = {str(self.sort_direction == TopKSortDirection.Ascending).lower()};",
            f"constexpr bool TOPK_XL_FUSED_REDUCE = {str(self.fused_reduce).lower()};",
            f"constexpr std::uint32_t TOPK_XL_CHUNK_BASE_MODE = {self.chunk_base_mode.value};",
            f"constexpr std::uint32_t TOPK_XL_CHUNK_BASE = {self.chunk_base};",
            f"constexpr bool TOPK_XL_FUSED_E2E = {str(self.fused_e2e).lower()};",
            f"constexpr std::uint32_t TOPK_XL_SEG_BASE = {self.seg_base};",
            f"constexpr std::uint32_t TOPK_XL_SORT_MODE = {self.sort_mode.value};",
            f"constexpr bool TOPK_XL_LSB_ROW_MAJOR = {str(self.lsb_row_major).lower()};",
            f"constexpr bool TOPK_XL_REINIT_AFTER_COPY = {str(self.reinit_after_copy).lower()};",
        ]
        if self.blaze_compat:
            lines.append("#define TOPK_XL_BLAZE_COMPAT 1")
        return "\n".join(lines)


@dataclass
class TOP32_RM(TemplateParameter):
    """Compile-time knobs for ``top32_rm_test.cpp`` (the DeepSeek top32_rm family).

    ``row_elements``  length of the row being reduced to its top 32. The driver walks it in
                      64-element chunks, so a value that is 32 (mod 64) is what reaches the
                      ``num_faces=2`` tail path -- the same shape the Metal dev test covers
                      with row=160.
    ``datum_bytes``   L1 datum size of both operands, which is the only thing the driver
                      needs to turn a chunk index into an address: one chunk is
                      ``64 * datum_bytes`` bytes, and unpacker addresses count 16-byte words.
                      Must match the format the stimuli are written in.
    ``top_min``       ``_bitonic_top32_merge_``'s template polarity. False (the consumer's
                      value) keeps the max half of each compare-exchange, i.e. a top-32.
    ``top32_mode``    0 walks the row in 64-element chunks through this family's own unpack
                      (``top32_rm_dev_compute.cpp``); 1 is the pre-sorted path, which
                      transposes whole 1024-element tiles into Dest and runs
                      ``_bitonic_top32_of_1024_rm_pre_sorted_{prep,combine,final}_``
                      (``top32_rm_dev_compute_v2.cpp``), then finishes any remainder in
                      64-element chunks the way that kernel does. Mode 1 requires the input to
                      be pre-sorted into descending runs of 32.
    ``via_wrappers``  route every SFPU call through the Metal wrapper layer
                      (``experimental/llk_sfpu/llk_math_deepseek_top32_rm.h``) instead of
                      calling the ``ckernel::sfpu::`` primitives directly. The wrappers are
                      thin -- each is the same ``_llk_math_eltwise_unary_sfpu_params_`` call --
                      so this is the same computation through one more layer, and the only thing
                      in the tree that calls those 7 entry points at all.
    """

    row_elements: int = 64
    datum_bytes: int = 2
    top_min: bool = False
    via_wrappers: bool = False
    # Named for the constant it emits rather than bare `mode`: field names have to be unique
    # across every parameter class, or two params in one test produce duplicate perf-CSV
    # columns (test_perf_header_gate.py). `mode` is already taken by GENERALIZED_MOE_GATE.
    top32_mode: int = 0

    def convert_to_cpp(self) -> str:
        if self.row_elements % 32 != 0:
            raise ValueError(
                f"row_elements must be a multiple of 32, got {self.row_elements}"
            )
        if self.top32_mode not in (0, 1):
            raise ValueError(
                f"top32_mode must be 0 (plain) or 1 (pre-sorted), got {self.top32_mode}"
            )
        if self.top32_mode == 1 and self.row_elements < 1024:
            raise ValueError(
                "the pre-sorted mode needs at least one whole 1024-element chunk, got "
                f"{self.row_elements}"
            )
        return "\n".join(
            [
                f"constexpr std::uint32_t TOP32_ROW_ELEMENTS = {self.row_elements}u;",
                f"constexpr std::uint32_t TOP32_DATUM_BYTES = {self.datum_bytes}u;",
                f"constexpr bool TOP32_TOP_MIN = {str(self.top_min).lower()};",
                f"constexpr std::uint32_t TOP32_MODE = {self.top32_mode}u;",
                f"#define TOP32_VIA_WRAPPERS {int(self.via_wrappers)}",
            ]
        )


@dataclass
class ADD_TOP_ROW(TemplateParameter):
    add_top_row: bool

    def convert_to_cpp(self) -> str:
        return f"constexpr bool ADD_TOP_ROW = {str(self.add_top_row).lower()};"


@dataclass
class TO_FROM_INT8(TemplateParameter):
    to_from_int8: bool

    def convert_to_cpp(self) -> str:
        return f"constexpr bool TO_FROM_INT8 = {str(self.to_from_int8).lower()};"


@dataclass
class IS_MAX_OP(TemplateParameter):
    """Compile-time flag: true for element-wise max, false for min."""

    is_max_op: bool = True

    def convert_to_cpp(self) -> str:
        return f"constexpr bool IS_MAX_OP = {str(self.is_max_op).lower()};"


@dataclass
class SFPU_SCALE_EN(TemplateParameter):
    """Compile-time SCALE_EN flag for SFPU kernels that optionally pre-scale their input.

    Pairs with ``SFPU_UNARY_SCALAR``, which carries the scale itself. Kernels in the
    exp family take the scale as a *bfloat16* bit pattern (e.g. 0x3F80 for 1.0f,
    ``p_sfpu::kCONST_1_FP16B``), not an fp32 one.
    """

    scale_en: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool SFPU_SCALE_EN = {str(self.scale_en).lower()};"


@dataclass
class SOFTMAX_K(TemplateParameter):
    """Number of valid lanes ``k`` for the softmax_k SFPU entry (``_softmax_k_<k>``).

    ``k`` counts values per row inside face 0's 16 columns; columns >= k must be
    exactly 0.0 in DEST (the kernel's predication treats 0.0 as padding).
    """

    softmax_k: int = 16

    def convert_to_cpp(self) -> str:
        return f"constexpr int SOFTMAX_K = {self.softmax_k};"


@dataclass
class SAMPLING_OP(TemplateParameter):
    """Select which ckernel_sfpu_sampling.h entry point the sampling test drives.

    Emits ``#define SAMPLING_OP_<NAME>`` consumed by ``sfpu_sampling_test.cpp``.
    """

    sampling_op: str = "recip_scalar"

    def convert_to_cpp(self) -> str:
        return f"#define SAMPLING_OP_{self.sampling_op.upper()}"


@dataclass
class MOE_GATE_TOPK(TemplateParameter):
    """Compile-time configuration of the generic MoE-gate top-k SFPU entry.

    Defaults match the compute API. ``generate_indices=False`` preserves the
    caller's expert IDs, while ``scores_include_bias=True`` returns biased
    scores as well as using them to select experts. Extra scaling stays disabled
    in this driver.
    """

    num_selected_experts: int = 8
    num_total_experts: int = 256
    normalize: bool = False
    zero_tail: bool = False
    full_sort: bool = False
    generate_indices: bool = True
    scores_include_bias: bool = False

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr int MOE_GATE_NUM_SELECTED_EXPERTS = {self.num_selected_experts};",
            f"constexpr int MOE_GATE_NUM_TOTAL_EXPERTS = {self.num_total_experts};",
            f"constexpr bool MOE_GATE_NORMALIZE = {str(self.normalize).lower()};",
            f"constexpr bool MOE_GATE_ZERO_TAIL = {str(self.zero_tail).lower()};",
            f"constexpr bool MOE_GATE_FULL_SORT = {str(self.full_sort).lower()};",
            f"constexpr bool MOE_GATE_GENERATE_INDICES = {str(self.generate_indices).lower()};",
            f"constexpr bool MOE_GATE_SCORES_INCLUDE_BIAS = {str(self.scores_include_bias).lower()};",
        ]
        return "\n".join(lines)


@dataclass
class MOE_GATE_NORMALIZE_PARAMS(TemplateParameter):
    """``eps`` / ``scale`` for the MoE-gate normalize step, as raw fp32 bit patterns.

    Both are decoded on device via ``Converter::as_float``, so emitting the bit
    patterns keeps the kernel and the torch golden exactly aligned.
    """

    eps_bits: int = 0x00000000  # 0.0f
    scale_bits: int = 0x3F800000  # 1.0f
    extra_scale_bits: int = 0x3F800000  # 1.0f, identity for the do_extra_scale path

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr std::uint32_t MOE_GATE_EPS_BITS = {self.eps_bits}u;",
            f"constexpr std::uint32_t MOE_GATE_SCALE_BITS = {self.scale_bits}u;",
            f"constexpr std::uint32_t MOE_GATE_EXTRA_SCALE_BITS = {self.extra_scale_bits}u;",
        ]
        return "\n".join(lines)


# === RUNTIME PARAMETER IMPLEMENTATIONS ===


@dataclass
class BLOCK_MAX8_PARAMS(RuntimeParameter):
    """Pass the valid prefix and optional DST sign-bit probe to the driver."""

    valid_scores: int
    inject_negative_zero: int

    def convert_to_cpp(self):
        """Provide the equivalent constant for harness template mode."""
        return (
            f"constexpr unsigned VALID_SCORES = {self.valid_scores};"
            f"constexpr unsigned INJECT_NEGATIVE_ZERO = {self.inject_negative_zero};"
        )

    def convert_to_struct_fields(self):
        """Serialize the valid prefix and probe flag as unsigned runtime arguments."""
        return "unsigned VALID_SCORES; unsigned INJECT_NEGATIVE_ZERO;", "II"


def generate_input_dim(
    srcA: tuple[int],
    srcB: tuple[int],
    block_ct_dim: int = None,
    block_rt_dim: int = None,
    tile_dimensions: tuple[int, int] = (32, 32),
):
    num_rows, num_cols = tile_dimensions
    validate_tile_dimensions(srcA[0], num_rows)
    validate_tile_dimensions(srcA[1], num_cols)
    validate_tile_dimensions(srcB[0], num_rows)
    validate_tile_dimensions(srcB[1], num_cols)

    full_ct_dim = srcB[1] // num_cols
    full_rt_dim = srcA[0] // num_rows

    block_ct_dim = full_ct_dim if block_ct_dim is None else block_ct_dim
    block_rt_dim = full_rt_dim if block_rt_dim is None else block_rt_dim

    return INPUT_DIMENSIONS(full_rt_dim, full_ct_dim, block_ct_dim, block_rt_dim)


@dataclass
class INPUT_DIMENSIONS(RuntimeParameter):
    full_rt_dim: int = 0
    full_ct_dim: int = 0
    block_ct_dim: int = 0
    block_rt_dim: int = 0

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr std::uint32_t FULL_RT_DIM = {self.full_rt_dim};",
            f"constexpr std::uint32_t FULL_CT_DIM = {self.full_ct_dim};",
            f"constexpr std::uint32_t BLOCK_CT_DIM = {self.block_ct_dim};",
            f"constexpr std::uint32_t BLOCK_RT_DIM = {self.block_rt_dim};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            f"std::uint32_t FULL_RT_DIM;",
            f"std::uint32_t FULL_CT_DIM;",
            f"std::uint32_t BLOCK_CT_DIM;",
            f"std::uint32_t BLOCK_RT_DIM;",
        ]
        return "\n".join(lines), "IIII"


@dataclass
class LOOP_FACTOR(RuntimeParameter):
    loop_factor: int = 1

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t LOOP_FACTOR = {self.loop_factor};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"std::uint32_t LOOP_FACTOR;", "I"


@dataclass
class UNPACK_TRANS_FACES(RuntimeParameter):
    unpack_transpose_faces: Transpose = Transpose.No

    def convert_to_cpp(self) -> str:
        return f"constexpr bool UNPACK_TRANSPOSE_FACES = {str(self.unpack_transpose_faces.value).lower()};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"bool UNPACK_TRANSPOSE_FACES;", "?"


@dataclass
class UNPACK_TRANS_WITHIN_FACE(RuntimeParameter):
    unpack_transpose_within_face: Transpose = Transpose.No

    def convert_to_cpp(self) -> str:
        return f"constexpr bool UNPACK_TRANSPOSE_WITHIN_FACE = {str(self.unpack_transpose_within_face.value).lower()};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"bool UNPACK_TRANSPOSE_WITHIN_FACE;", "?"


@dataclass
class NARROW_TILE(RuntimeParameter):
    narrow_tile: NarrowTile = NarrowTile.No

    def convert_to_cpp(self) -> str:
        return f"constexpr bool NARROW_TILE = {str(self.narrow_tile.value).lower()};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"bool NARROW_TILE;", "?"


@dataclass
class DEST_INDEX(RuntimeParameter):
    dst_index: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int DST_INDEX = {self.dst_index};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"int DST_INDEX;", "i"


@dataclass
class SFPU_TILE_INDICES(RuntimeParameter):
    src0_tile_idx: int = 0
    src1_tile_idx: int = 1
    dst_tile_idx: int = 0

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr int SRC0_TILE_IDX = {self.src0_tile_idx};",
            f"constexpr int SRC1_TILE_IDX = {self.src1_tile_idx};",
            f"constexpr int DST_TILE_IDX = {self.dst_tile_idx};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines = [
            "int SRC0_TILE_IDX;",
            "int SRC1_TILE_IDX;",
            "int DST_TILE_IDX;",
        ]
        return "\n".join(lines), "iii"


@dataclass
class ZERO_POINT(RuntimeParameter):
    """fp32 bit-pattern of the quant-family zero-point, passed to the binary SFPU
    init at runtime. DEQUANT expects the bits of -zero_point (the init negates the
    contract by loading these bits directly). Ignored by non-quant binary ops."""

    zero_point_bits: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t ZERO_POINT = {self.zero_point_bits}u;"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "std::uint32_t ZERO_POINT;", "I"


@dataclass
class MAX_POOL_WITH_INDICES(TemplateParameter):
    """Compile-time knobs of the Quasar max_pool_with_indices SFPU kernel.

    ``max_pool_num_rows`` is the kernel's 9-versus-32 row dispatch selector, ``max_pool_row_major`` picks
    ``DataLayout::ROW_MAJOR`` over ``DataLayout::TILE``, and ``max_pool_accumulate`` carries the
    running max across chunks in the Dest tiles above the operands."""

    max_pool_num_rows: int = 9
    max_pool_row_major: bool = False
    max_pool_accumulate: bool = False

    def convert_to_cpp(self) -> str:
        layout = "ROW_MAJOR" if self.max_pool_row_major else "TILE"
        lines = [
            f"constexpr int MAX_POOL_NUM_ROWS = {self.max_pool_num_rows};",
            f"constexpr ckernel::DataLayout MAX_POOL_LAYOUT = ckernel::DataLayout::{layout};",
            f"constexpr bool MAX_POOL_ACCUMULATE = {str(self.max_pool_accumulate).lower()};",
        ]
        return "\n".join(lines)


@dataclass
class MAX_POOL_CHUNK(RuntimeParameter):
    """Index of the max_pool_with_indices call in its accumulation chain; chunk 0 seeds
    the running max, later chunks fold into it. Ignored unless accumulate is set."""

    max_pool_chunk: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t MAX_POOL_CHUNK = {self.max_pool_chunk}u;"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "std::uint32_t MAX_POOL_CHUNK;", "I"


@dataclass
class SIGN_MAGNITUDE_FORMAT(TemplateParameter):
    """SMAG32 Dest datapath toggle; read by the quant binary ops, int ADD/MUL, signed comparisons, GCD and LCM."""

    sign_magnitude: bool = False

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr bool SFPU_SIGN_MAGNITUDE = {str(self.sign_magnitude).lower()};"
        )


@dataclass
class SFPU_DST_ROUNDING_MODE(TemplateParameter):
    """Selects the bf16 narrowing mode for binary SFPU ADD/SUB results."""

    dst_rounding: DstRoundingMode = DstRoundingMode.Default

    def convert_to_cpp(self) -> str:
        return f"constexpr ckernel::DstRoundingMode SFPU_DST_ROUNDING_MODE = {self.dst_rounding.cpp_enum_value};"


@dataclass
class L1_ACC(RuntimeParameter):
    l1_acc: L1Accumulation = L1Accumulation.No

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr int L1_ACC = {1 if self.l1_acc == L1Accumulation.Yes else 0};"
        )

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"int L1_ACC;", "i"


@dataclass
class TILE_COUNT(RuntimeParameter):
    tile_cnt: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t TILE_CNT = {self.tile_cnt};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"std::uint32_t TILE_CNT;", "I"


@dataclass
class NUM_GUARD_TILES(RuntimeParameter):
    count: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t NUM_GUARD_TILES = {self.count};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"std::uint32_t NUM_GUARD_TILES;", "I"


@dataclass
class INPUT_TILE_CNT(RuntimeParameter):
    input_tile_cnt: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int INPUT_TILE_CNT = {self.input_tile_cnt};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "int INPUT_TILE_CNT;", "i"


@dataclass
class OUTPUT_TILE_CNT(RuntimeParameter):
    output_tile_cnt: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int OUTPUT_TILE_CNT = {self.output_tile_cnt};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "int OUTPUT_TILE_CNT;", "i"


@dataclass
class REDUCE_TO_ONE(RuntimeParameter):
    is_reduce_to_one: bool = False

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr bool IS_REDUCE_TO_ONE = {str(self.is_reduce_to_one).lower()};"
        )

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "bool IS_REDUCE_TO_ONE;", "?"


@dataclass
class SRCA_REUSE_COUNT(RuntimeParameter):
    srca_reuse_count: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int SRCA_REUSE_COUNT = {self.srca_reuse_count};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return f"int SRCA_REUSE_COUNT;", "i"


@dataclass
class PARTIAL_FACE(RuntimeParameter):
    partial_a: bool = False
    partial_face_pack: bool = False
    partial_b: bool = False
    partial_face_math: bool = False

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr bool PARTIAL_FACE_A = {str(self.partial_a).lower()};",
            f"constexpr bool PARTIAL_FACE_PACK = {str(self.partial_face_pack).lower()};",
            f"constexpr bool PARTIAL_FACE_B = {str(self.partial_b).lower()};",
            f"constexpr bool PARTIAL_FACE_MATH = {str(self.partial_face_math).lower()};",
        ]

        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "bool PARTIAL_FACE_A;",
            "bool PARTIAL_FACE_PACK;",
            "bool PARTIAL_FACE_B;",
            "bool PARTIAL_FACE_MATH;",
        ]
        return "\n".join(lines), "????"


@dataclass
class CRK_TILE_DIMM(RuntimeParameter):
    c_dimm: c_uint32 = 0
    r_dimm: c_uint32 = 0
    k_dimm: c_uint32 = 0

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr std::uint32_t RT_DIM = {self.r_dimm};",
            f"constexpr std::uint32_t CT_DIM = {self.c_dimm};",
            f"constexpr std::uint32_t KT_DIM = {self.k_dimm};",
        ]

        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "std::uint32_t CT_DIM;",
            "std::uint32_t RT_DIM;",
            "std::uint32_t KT_DIM;",
        ]
        return "\n".join(lines), "III"


@dataclass
class NUM_TILES_IN_BLOCK(RuntimeParameter):
    num_tiles_in_block: int = 1
    input_num_tiles_in_block: int = None
    output_num_tiles_in_block: int = None

    def __post_init__(self):
        if self.input_num_tiles_in_block is None:
            self.input_num_tiles_in_block = self.num_tiles_in_block
        if self.output_num_tiles_in_block is None:
            self.output_num_tiles_in_block = self.num_tiles_in_block

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr std::uint32_t NUM_TILES_IN_BLOCK = {self.num_tiles_in_block};",
            f"constexpr std::uint32_t INPUT_NUM_TILES_IN_BLOCK = {self.input_num_tiles_in_block};",
            f"constexpr std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK = {self.output_num_tiles_in_block};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines = [
            "std::uint32_t NUM_TILES_IN_BLOCK;",
            "std::uint32_t INPUT_NUM_TILES_IN_BLOCK;",
            "std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK;",
        ]
        return "\n".join(lines), "III"


@dataclass
class NUM_BLOCKS(RuntimeParameter):
    num_blocks: int = 1
    input_num_blocks: int = None
    output_num_blocks: int = None

    def __post_init__(self):
        if self.input_num_blocks is None:
            self.input_num_blocks = self.num_blocks
        if self.output_num_blocks is None:
            self.output_num_blocks = self.num_blocks

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr int NUM_BLOCKS = {self.num_blocks};",
            f"constexpr int INPUT_NUM_BLOCKS = {self.input_num_blocks};",
            f"constexpr int OUTPUT_NUM_BLOCKS = {self.output_num_blocks};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines = [
            "int NUM_BLOCKS;",
            "int INPUT_NUM_BLOCKS;",
            "int OUTPUT_NUM_BLOCKS;",
        ]
        return "\n".join(lines), "iii"


@dataclass
class NUM_FACES(RuntimeParameter):
    num_faces: int = 4  # Number of active faces for result matrix
    num_faces_A: int = 4  # Number of active faces for matrix A
    num_faces_B: int = 4  # Number of active faces for matrix B

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr std::uint32_t num_faces = {self.num_faces};",
            f"constexpr std::uint32_t num_faces_A = {self.num_faces_A};",
            f"constexpr std::uint32_t num_faces_B = {self.num_faces_B};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "std::uint32_t num_faces;",
            "std::uint32_t num_faces_A;",
            "std::uint32_t num_faces_B;",
        ]
        return "\n".join(lines), "III"


@dataclass
class NUM_FACES_R_DIM(RuntimeParameter):
    num_faces_r_dim_A: int = 2  # Number of faces in row dimension for matrix A
    num_faces_r_dim_B: int = 2  # Number of faces in row dimension for matrix B

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            (
                f"constexpr int num_faces_r_dim_A = {self.num_faces_r_dim_A};"
                if self.num_faces_r_dim_A
                else ""
            ),
            (
                f"constexpr int num_faces_r_dim_B = {self.num_faces_r_dim_B};"
                if self.num_faces_r_dim_B
                else ""
            ),
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "int num_faces_r_dim_A;",
            "int num_faces_r_dim_B;",
        ]
        return "\n".join(lines), "ii"


@dataclass
class NUM_FACES_C_DIM(RuntimeParameter):
    num_faces_c_dim_A: int = 2  # Number of faces in column dimension for matrix A
    num_faces_c_dim_B: int = 2  # Number of faces in column dimension for matrix B

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            (
                f"constexpr int num_faces_c_dim_A = {self.num_faces_c_dim_A};"
                if self.num_faces_c_dim_A
                else ""
            ),
            (
                f"constexpr int num_faces_c_dim_B = {self.num_faces_c_dim_B};"
                if self.num_faces_c_dim_B
                else ""
            ),
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "int num_faces_c_dim_A;",
            "int num_faces_c_dim_B;",
        ]
        return "\n".join(lines), "ii"


# NOTE: If IN_FACE_DIMS parameter is propagated throughout test-infra, it can replace
# other variables used to pass input face dimensions (eg. TEST_FACE_DIMS).
@dataclass
class IN_FACE_DIMS(RuntimeParameter):
    in0_face_r_dim: int = 16
    in0_face_c_dim: int = 16
    in1_face_r_dim: int = 16
    in1_face_c_dim: int = 16

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr int in0_face_r_dim = {self.in0_face_r_dim};",
            f"constexpr int in0_face_c_dim = {self.in0_face_c_dim};",
            f"constexpr int in1_face_r_dim = {self.in1_face_r_dim};",
            f"constexpr int in1_face_c_dim = {self.in1_face_c_dim};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "int in0_face_r_dim;",
            "int in0_face_c_dim;",
            "int in1_face_r_dim;",
            "int in1_face_c_dim;",
        ]
        return "\n".join(lines), "iiii"


@dataclass
class TEST_FACE_DIMS(RuntimeParameter):
    face_r_dim: int = 16
    face_c_dim: int = 16

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr std::uint32_t TEST_FACE_R_DIM = {self.face_r_dim};",
            f"constexpr std::uint32_t TEST_FACE_C_DIM = {self.face_c_dim};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "std::uint32_t TEST_FACE_R_DIM;",
            "std::uint32_t TEST_FACE_C_DIM;",
        ]
        return "\n".join(lines), "II"


@dataclass
class IN_TILE_DIMS(RuntimeParameter):
    in0_r_dim: int = 32
    in0_c_dim: int = 32
    in1_r_dim: int = 32
    in1_c_dim: int = 32

    def convert_to_cpp(self) -> str:
        lines: list[str] = [
            f"constexpr std::uint32_t in0_tile_r_dim = {self.in0_r_dim};",
            f"constexpr std::uint32_t in0_tile_c_dim = {self.in0_c_dim};",
            f"constexpr std::uint32_t in1_tile_r_dim = {self.in1_r_dim};",
            f"constexpr std::uint32_t in1_tile_c_dim = {self.in1_c_dim};",
        ]
        return "\n".join(lines)

    def convert_to_struct_fields(self) -> tuple[str, str]:
        lines: list[str] = [
            "std::uint32_t in0_tile_r_dim;",
            "std::uint32_t in0_tile_c_dim;",
            "std::uint32_t in1_tile_r_dim;",
            "std::uint32_t in1_tile_c_dim;",
        ]
        return "\n".join(lines), "IIII"


@dataclass
class RELU_CONFIG(RuntimeParameter):
    """Packer ReLU config: packed 32-bit value (mode in low 2 bits, threshold in bits 16–31)."""

    relu_config: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int RELU_CONFIG = {self.relu_config};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "int RELU_CONFIG;", "i"


@dataclass
class NUM_ROWS_TO_PACK(RuntimeParameter):
    num_rows_to_pack: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t NUM_ROWS_TO_PACK = {self.num_rows_to_pack};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "std::uint32_t NUM_ROWS_TO_PACK;", "I"


@dataclass
class EMA_ALPHA_BETA(TemplateParameter):
    """Alpha/beta smoothing weights for the EMA entry, as raw fp32 bit patterns.

    ``_load_alpha_beta_`` loads each as the fp32 representation into LREG5 (alpha)
    and LREG6 (beta); the kernel computes ``EMA_new = alpha*EMA_old + beta*input``.
    Emitting the bit patterns keeps the C++ and torch golden exactly aligned.
    """

    alpha_bits: int = 0x3E800000  # 0.25f
    beta_bits: int = 0x3F400000  # 0.75f

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr std::uint32_t EMA_ALPHA_BITS = {self.alpha_bits}u;",
            f"constexpr std::uint32_t EMA_BETA_BITS = {self.beta_bits}u;",
        ]
        return "\n".join(lines)


@dataclass
class EMA_INTERLEAVED_INIT(TemplateParameter):
    """Whether sfpu_ema_test.cpp issues an eltwise binary init between the datacopy and ema_tile."""

    binary_init_before_ema: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool EMA_BINARY_INIT_BEFORE_EMA = {str(self.binary_init_before_ema).lower()};"


@dataclass
class TILE_DST_CT_OFFSET(TemplateParameter):
    offset: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t TILE_DST_CT_OFFSET = {self.offset};"


@dataclass
class CONFIGURE_TEST_RUN_IDX(RuntimeParameter):
    configure_test_run_idx: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t CONFIGURE_TEST_RUN_IDX = {self.configure_test_run_idx};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "std::uint32_t CONFIGURE_TEST_RUN_IDX;", "I"


@dataclass
class HOST_IS_STREAM_PRODUCER(RuntimeParameter):
    host_is_stream_producer: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool HOST_IS_STREAM_PRODUCER = {str(self.host_is_stream_producer).lower()};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "bool HOST_IS_STREAM_PRODUCER;", "?"


@dataclass
class HOST_IS_STREAM_CONSUMER(RuntimeParameter):
    host_is_stream_consumer: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool HOST_IS_STREAM_CONSUMER = {str(self.host_is_stream_consumer).lower()};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "bool HOST_IS_STREAM_CONSUMER;", "?"


@dataclass
class FILL_INT_FORMAT(TemplateParameter):
    data_format: DataFormat = DataFormat.Int32

    def convert_to_cpp(self) -> str:
        return f"constexpr auto FILL_INT_FORMAT = DataFormat::{self.data_format.name};"


@dataclass
class WELFORDS(TemplateParameter):
    """Compile-time schedule of sfpu_welfords_quasar_test.cpp (tiles_per_block 0 = one Dest section)."""

    reciprocal_size: int = 0
    partial_last_tile: bool = False
    welfords_start_row: int = 0
    welfords_num_rows: int = 32
    face_layout: bool = False
    final_grouped: bool = False
    final_group_id: int = 0
    final_dst: int = 0
    save_restore: bool = False
    state_grouped: bool = False
    state_group_id: int = 0
    state_dst: int = 0
    save_after_tiles: int = 0
    tiles_per_block: int = 0

    def convert_to_cpp(self) -> str:
        def b(value: bool) -> str:
            return str(value).lower()

        lines: list[str] = [
            f"constexpr std::size_t WELFORDS_RECIP_SIZE = {self.reciprocal_size};",
            f"constexpr bool WELFORDS_PARTIAL_LAST_TILE = {b(self.partial_last_tile)};",
            f"constexpr std::uint32_t WELFORDS_START_ROW = {self.welfords_start_row};",
            f"constexpr std::uint32_t WELFORDS_NUM_ROWS = {self.welfords_num_rows};",
            f"constexpr bool WELFORDS_FACE_LAYOUT = {b(self.face_layout)};",
            f"constexpr bool WELFORDS_FINAL_GROUPED = {b(self.final_grouped)};",
            f"constexpr std::uint32_t WELFORDS_FINAL_GROUP_ID = {self.final_group_id};",
            f"constexpr std::uint32_t WELFORDS_FINAL_DST = {self.final_dst};",
            f"constexpr bool WELFORDS_SAVE_RESTORE = {b(self.save_restore)};",
            f"constexpr bool WELFORDS_STATE_GROUPED = {b(self.state_grouped)};",
            f"constexpr std::uint32_t WELFORDS_STATE_GROUP_ID = {self.state_group_id};",
            f"constexpr std::uint32_t WELFORDS_STATE_DST = {self.state_dst};",
            f"constexpr std::uint32_t WELFORDS_SAVE_AFTER_TILES = {self.save_after_tiles};",
            f"constexpr std::uint32_t WELFORDS_TILES_PER_BLOCK = {self.tiles_per_block};",
        ]
        return "\n".join(lines)


@dataclass
class TWO_PASS_STATS(TemplateParameter):
    """Compile-time schedule of sfpu_welfords_two_pass_quasar_test.cpp.

    two_pass_mode: 0 STREAM, 1 COMBINE, 2 SWITCH; two_pass_finalize: 0 row, 1 raw, 2 split, 3 combined, 4 row w/o mean.
    """

    two_pass_mode: int = 0
    two_pass_dual: bool = True
    two_pass_finalize: int = 0
    two_pass_retain_anchor: bool = False
    two_pass_average_variance: bool = True
    two_pass_group_a: int = 0
    two_pass_group_b: int = 0
    two_pass_final_dst: int = 0
    two_pass_state_dst: int = 0
    two_pass_tiles_per_block: int = 0
    two_pass_block_tiles: int = 1
    two_pass_partial_last_tile: bool = False
    two_pass_start_row: int = 0
    two_pass_num_rows: int = 32

    def convert_to_cpp(self) -> str:
        def b(value: bool) -> str:
            return str(value).lower()

        lines: list[str] = [
            f"constexpr std::uint32_t TWO_PASS_MODE = {self.two_pass_mode};",
            f"constexpr bool TWO_PASS_DUAL = {b(self.two_pass_dual)};",
            f"constexpr std::uint32_t TWO_PASS_FINALIZE = {self.two_pass_finalize};",
            f"constexpr bool TWO_PASS_RETAIN_ANCHOR = {b(self.two_pass_retain_anchor)};",
            f"constexpr bool TWO_PASS_AVERAGE_VARIANCE = {b(self.two_pass_average_variance)};",
            f"constexpr std::uint32_t TWO_PASS_GROUP_A = {self.two_pass_group_a};",
            f"constexpr std::uint32_t TWO_PASS_GROUP_B = {self.two_pass_group_b};",
            f"constexpr std::uint32_t TWO_PASS_FINAL_DST = {self.two_pass_final_dst};",
            f"constexpr std::uint32_t TWO_PASS_STATE_DST = {self.two_pass_state_dst};",
            f"constexpr std::uint32_t TWO_PASS_TILES_PER_BLOCK = {self.two_pass_tiles_per_block};",
            f"constexpr std::uint32_t TWO_PASS_BLOCK_TILES = {self.two_pass_block_tiles};",
            f"constexpr bool TWO_PASS_PARTIAL_LAST_TILE = {b(self.two_pass_partial_last_tile)};",
            f"constexpr std::uint32_t TWO_PASS_START_ROW = {self.two_pass_start_row};",
            f"constexpr std::uint32_t TWO_PASS_NUM_ROWS = {self.two_pass_num_rows};",
        ]
        return "\n".join(lines)


@dataclass
class TYPECAST_FORMATS(TemplateParameter):
    """Compile-time config for the SFPU typecast test kernel.

    Emits the logical input/output ``DataFormat`` enum values consumed by
    ``typecast_tile<IN, OUT>`` (mirrored by the typecast dispatch in
    ``sfpu_operations.h``, reached via ``SfpuType::typecast``).
    """

    input_format: DataFormat = DataFormat.Float32
    output_format: DataFormat = DataFormat.Float16_b

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr auto TYPECAST_IN_FORMAT = DataFormat::{self.input_format.name};",
            f"constexpr auto TYPECAST_OUT_FORMAT = DataFormat::{self.output_format.name};",
        ]
        return "\n".join(lines)


@dataclass
class CUSTOM_MM_REUSE_CFG(TemplateParameter):
    """Compile-time chain geometry for the custom_mm_reuse_dest_srcb test.

    The reuse LLK is the second matmul of a fused chain: its in0 operand is moved
    out of DEST (where a preceding Dense-layout custom_mm left its output) into
    SrcB via MOVD2B; only the weights are unpacked into SrcA. The C++ driver runs
    the whole chain, so these four constants size both the producer custom_mm and
    the reuse consumer:

    ``in0_tile_r_dim``  height of the in0 tile (1/2/4/8); selects the MOVD2B pattern.
    ``producer_kt``     producer inner dim in tiles (its kt_dim).
    ``reuse_kt``        consumer inner dim in tiles == producer ct_dim == number of
                        DEST-resident in0 tiles the reuse op reduces over.
    ``reuse_nt``        consumer output width in tiles (1 to 16).
    """

    in0_tile_r_dim: int = 8
    producer_kt: int = 2
    reuse_kt: int = 2
    reuse_nt: int = 1

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr std::uint32_t IN0_TILE_R_DIM = {self.in0_tile_r_dim}u;",
            f"constexpr std::uint32_t PRODUCER_KT = {self.producer_kt}u;",
            f"constexpr std::uint32_t REUSE_KT = {self.reuse_kt}u;",
            f"constexpr std::uint32_t REUSE_NT = {self.reuse_nt}u;",
        ]
        return "\n".join(lines)


@dataclass
class SDPA_CUSTOM_MM_FLAGS(TemplateParameter):
    """Compile-time knobs for the experimental sdpa_custom_mm block matmul test.

    Emitted as #defines (not constexpr) because the C++ driver both consumes
    SIGNAL_GRANULARITY as a non-type template argument and guards each name with an
    #ifndef default, so a standalone compile of the source still works.

    signal_granularity : FPU->SFPU post cadence (llk_math_sdpa_custom_mm template arg).
                         Must divide ct_dim. Purely a signalling cadence; does not change
                         the numeric result.
    read_transposed    : selects the transposed SrcA (in1) L1 walk in the unpack LLK.
    mm_transpose       : the `transpose` init flag threaded through the unpack/math inits
                         (addr_mod SrcA increment + Haloize_mode).
    sdpa_row_stride    : physical SrcA row width in tiles for read_transposed. None
                         omits the LLK argument; zero and kt_dim retain tight packing.
    sdpa_input_tile_offset : leading SrcA tiles skipped before the matmul input.
    preserve_src_zero_flag : math leaves the Src zero flag at keep before the init, as a preceding
                         copy_tile_init does.
    """

    signal_granularity: int = 1
    read_transposed: bool = False
    mm_transpose: bool = False
    sdpa_row_stride: int | None = None
    sdpa_input_tile_offset: int = 0
    preserve_src_zero_flag: bool = False

    def convert_to_cpp(self) -> str:
        lines = [
            f"#define SIGNAL_GRANULARITY {self.signal_granularity}",
            f"#define READ_TRANSPOSED {str(self.read_transposed).lower()}",
            f"#define MM_TRANSPOSE {str(self.mm_transpose).lower()}",
            f"#define SDPA_INPUT_TILE_OFFSET {self.sdpa_input_tile_offset}",
            f"#define SDPA_PRESERVE_SRC_ZERO_FLAG {str(self.preserve_src_zero_flag).lower()}",
        ]
        if self.sdpa_row_stride is not None:
            lines.append(f"#define SDPA_ROW_STRIDE {self.sdpa_row_stride}")
        return "\n".join(lines)


@dataclass
class SDPA_CUSTOM_MM_REUSE_DEST(TemplateParameter):
    """Compile-time dims for the experimental sdpa_custom_mm_reuse_dest_srcb OV matmul.

    kt_dim: number of K tiles (softmax-score / V K tiles). The unpack MOP requires an
            even kt_dim >= 2 (see the "kt_dim: even number from 2 to 256" note in
            llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb.h); kt_dim < 2 underflows the
            MOP iteration count and hangs the unpacker.
    nt_dim: number of V head-dim output tiles per K iteration (1..16).
    """

    kt_dim: int = 2
    nt_dim: int = 1

    def convert_to_cpp(self) -> str:
        lines = [
            f"constexpr std::uint32_t KT_DIM = {self.kt_dim};",
            f"constexpr std::uint32_t NT_DIM = {self.nt_dim};",
        ]
        return "\n".join(lines)


@dataclass
class SDPA_REDUCE_ROW_POOL(TemplateParameter):
    """Select MAX vs SUM for the experimental sdpa_reduce_row op.

    Emits ``constexpr int SDPA_REDUCE_POOL = 0|1;`` consumed by
    sources/sdpa_reduce_row_test.cpp (0 == MAX -> calculate_sdpa_reduce_max_row,
    1 == SUM -> calculate_sdpa_reduce_sum_row). The op only supports MAX and SUM.
    """

    reduce_pool: ReducePool = ReducePool.Max

    def convert_to_cpp(self) -> str:
        pool_to_int = {ReducePool.Max: 0, ReducePool.Sum: 1}
        try:
            value = pool_to_int[self.reduce_pool]
        except KeyError:
            raise ValueError(
                f"sdpa_reduce_row supports only MAX and SUM pools, got {self.reduce_pool}"
            )
        return f"constexpr int SDPA_REDUCE_POOL = {value};"


@dataclass
class MUL_REDUCE_SCALAR_CHUNK_SIZE(RuntimeParameter):
    chunk_size: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t CHUNK_SIZE = {self.chunk_size};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "std::uint32_t CHUNK_SIZE;", "I"


@dataclass
class ZERO_PAD_ROWS(TemplateParameter):
    """SFPU row bounds for _zero_pad_tile_."""

    valid_rows: int = 0
    total_rows: int = 32

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr int ZERO_PAD_VALID_ROWS = {self.valid_rows};\n"
            f"constexpr int ZERO_PAD_TOTAL_ROWS = {self.total_rows};"
        )


@dataclass
class SPARSE_K_CONFIG(TemplateParameter):
    sparse_k_iterations: int = 32
    bank_mask: int = 0x3F
    my_bank: int = 0
    global_bank_shift: int = 14
    within_bank_mask: int = 0x3FFF
    out_shift: int = 0

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr int SPARSE_K_ITERATIONS = {self.sparse_k_iterations};\n"
            f"constexpr std::uint32_t SPARSE_K_BANK_MASK = {self.bank_mask}u;\n"
            f"constexpr std::uint32_t SPARSE_K_MY_BANK = {self.my_bank}u;\n"
            f"constexpr std::uint32_t SPARSE_K_GLOBAL_BANK_SHIFT = {self.global_bank_shift}u;\n"
            f"constexpr std::uint32_t SPARSE_K_WITHIN_BANK_MASK = {self.within_bank_mask}u;\n"
            f"constexpr std::uint32_t SPARSE_K_OUT_SHIFT = {self.out_shift}u;"
        )


@dataclass
class CLAMPED_SILU_PARAMS(TemplateParameter):
    clamped_silu_op: str = "GATE"
    scalar0: float = 1.0
    scalar1: float = 1.0

    def convert_to_cpp(self) -> str:
        return (
            f"#define CLAMPED_SILU_OP_{self.clamped_silu_op}\n"
            f"constexpr std::uint32_t CLAMPED_SILU_SCALAR0 = {_fp32_bits(self.scalar0)}u;\n"
            f"constexpr std::uint32_t CLAMPED_SILU_SCALAR1 = {_fp32_bits(self.scalar1)}u;"
        )


@dataclass
class GATED_REDUCE_PARAMS(TemplateParameter):
    gate: str
    up: str
    scale_flags: GatedReduceScale
    live_rows: int = 32

    def convert_to_cpp(self) -> str:
        return (
            f"#define GATED_REDUCE_GATE ckernel::sfpu::GatedReduceGate::{self.gate}\n"
            f"#define GATED_REDUCE_UP ckernel::sfpu::GatedReduceUp::{self.up}\n"
            f"constexpr bool GATED_REDUCE_GATE_SCALE = {str(bool(self.scale_flags & GatedReduceScale.Gate)).lower()};\n"
            f"constexpr bool GATED_REDUCE_UP_SCALE = {str(bool(self.scale_flags & GatedReduceScale.Up)).lower()};\n"
            f"constexpr bool GATED_REDUCE_OUT_SCALE = {str(bool(self.scale_flags & GatedReduceScale.Out)).lower()};\n"
            f"constexpr int GATED_REDUCE_ROWS = {self.live_rows};"
        )


@dataclass
class GATED_REDUCE_SCALARS(RuntimeParameter):
    gated_scale_bits: int
    gated_out_scale_bits: int
    gated_limit_bits: int
    gated_alpha_bits: int

    def convert_to_cpp(self) -> str:
        return "\n".join(
            f"constexpr std::uint32_t {name.upper()} = {value}u;"
            for name, value in vars(self).items()
        )

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return (
            "\n".join(f"std::uint32_t {name.upper()};" for name in vars(self)),
            "IIII",
        )


@dataclass
class PACK_SRC_FROM_OUTPUT(TemplateParameter):
    """The packer input of a block float input from a 16-bit DEST follows the output format family, as in production (data_format.cpp)."""

    pack_src_from_output = True

    def convert_to_cpp(self) -> str:
        return ""
