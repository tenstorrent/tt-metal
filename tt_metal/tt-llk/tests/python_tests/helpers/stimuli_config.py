# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import os
import shutil
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

import torch

from .device_io import read_from_device, write_to_device
from .format_config import DataFormat
from .golden_generators import GeneratorProxy, ProxyMode
from .llk_params import format_tile_sizes
from .logger import logger
from .pack import (
    pack_bfp2_b,
    pack_bfp4_b,
    pack_bfp8_b,
    pack_bfp16,
    pack_fp8_e4m3,
    pack_fp16,
    pack_fp32,
    pack_int8,
    pack_int16,
    pack_int32,
    pack_mxfp4,
    pack_mxfp8p,
    pack_mxfp8r,
    pack_mxint2,
    pack_mxint4,
    pack_mxint8,
    pack_uint8,
    pack_uint16,
    pack_uint32,
)
from .tile_constants import FACE_C_DIM, MAX_TILE_ELEMENTS, calculate_tile_size_bytes
from .unpack import unpack_res_tiles


class StimuliConfig:

    # === STATIC VARIABLES ===
    STIMULI_L1_ADDRESS_PERF = 0x21000
    STIMULI_L1_ADDRESS_DEBUG = 0x70000

    # Optional L1 buffers between B and Res, in layout order.
    _OPTIONAL_OPERAND_SPECS = (
        ("S", "buffer_S", "stimuli_S_format", "tile_count_S"),
        ("T", "buffer_T", "stimuli_T_format", "tile_count_T"),
        ("C", "buffer_C", "stimuli_C_format", "tile_count_C"),
    )

    _CACHE_BUFFER_ATTRS = (
        "buffer_A",
        "buffer_B",
        "buffer_S",
        "buffer_T",
        "buffer_C",
    )

    WITH_COVERAGE: ClassVar[bool] = False

    OFFSET_DICT: ClassVar[dict[str, list[int]]]
    STIMULI_CACHE_ROOT: ClassVar[Path]

    @classmethod
    def initialize_cache(cls, folder_path: Path):
        GeneratorProxy.STIMULI_CACHE_ROOT = cls.STIMULI_CACHE_ROOT = folder_path
        if GeneratorProxy.MODE == ProxyMode.CACHE_GOLDEN:
            # Clean entire folder if there already was some stimuli cached
            shutil.rmtree(cls.STIMULI_CACHE_ROOT, ignore_errors=True)
            os.makedirs(cls.STIMULI_CACHE_ROOT, exist_ok=True)

    def __init__(
        self,
        buffer_A: torch.Tensor,
        stimuli_A_format: DataFormat,
        buffer_B: torch.Tensor,
        stimuli_B_format: DataFormat,
        stimuli_res_format: DataFormat,
        tile_count_A: int = 1,
        tile_count_B: int = None,
        tile_count_res: int = 1,
        buffer_C: torch.Tensor = None,
        stimuli_C_format: DataFormat = None,
        tile_count_C: int = None,
        buffer_S: torch.Tensor = None,
        stimuli_S_format: DataFormat = None,
        tile_count_S: int = None,
        buffer_T: torch.Tensor = None,
        stimuli_T_format: DataFormat = None,
        tile_count_T: int = None,
        srcs_layout_operands: frozenset[str] = None,
        num_faces: int = 4,
        face_r_dim: int = 16,
        tile_dimensions: list[int] = [32, 32],
        sfpu=False,
        write_full_tiles: bool = False,
        use_dense_tile_dimensions: bool = False,
        operand_res_tile_size: int = None,
        twos_complement: bool = False,
    ):

        # Fields init
        self.buffer_A = buffer_A
        self.stimuli_A_format = stimuli_A_format
        self.tile_count_A = tile_count_A
        self.buffer_B = buffer_B
        self.stimuli_B_format = stimuli_B_format
        self.tile_count_B = tile_count_B
        self.buffer_C = buffer_C
        self.stimuli_C_format = stimuli_C_format
        self.tile_count_C = tile_count_C
        self.buffer_S = buffer_S
        self.stimuli_S_format = stimuli_S_format
        self.tile_count_S = tile_count_S
        self.buffer_T = buffer_T
        self.stimuli_T_format = stimuli_T_format
        self.tile_count_T = tile_count_T
        self.srcs_layout_operands = srcs_layout_operands
        self.stimuli_res_format = stimuli_res_format
        self.tile_count_res = tile_count_res
        self.num_faces = num_faces
        self.face_r_dim = face_r_dim
        self.tile_dimensions = tile_dimensions
        self.sfpu = sfpu
        self.write_full_tiles = write_full_tiles
        self.use_dense_tile_dimensions = use_dense_tile_dimensions
        self.operand_res_tile_size = operand_res_tile_size
        self.twos_complement = twos_complement

        # Hardware flags injected by TestConfig via set_use_srcs() / set_dest_acc()
        self.use_srcs = False
        self._dest_acc_32b = False

        self._calculate_tile_sizes()

    def _operand_use_srcs(self, operand: str) -> bool:
        # Per-operand SrcS L1 layout when unpack_to_srcs=True but not every buffer uses SrcS (e.g. parallel matmul + exp).
        if self.srcs_layout_operands is not None:
            return operand in self.srcs_layout_operands
        return self.use_srcs

    @staticmethod
    def _buf_addr_attr(name: str) -> str:
        return f"buf_{name.lower()}_addr"

    @staticmethod
    def _tile_size_attr(name: str) -> str:
        return f"tile_size_{name}_bytes"

    def _active_optional_operands(self):
        for name, buf_attr, fmt_attr, cnt_attr in self._OPTIONAL_OPERAND_SPECS:
            if getattr(self, buf_attr) is not None:
                yield {
                    "name": name,
                    "buffer": getattr(self, buf_attr),
                    "format": getattr(self, fmt_attr),
                    "tile_count": getattr(self, cnt_attr),
                }

    def _operand_addr(self, name: str) -> int:
        return getattr(self, self._buf_addr_attr(name))

    def _operand_tile_size_bytes(self, name: str) -> int:
        return getattr(self, self._tile_size_attr(name))

    def _write_optional_operand(self, op, location: str, *, dense: bool):
        pack_function = StimuliConfig.get_packer(op["format"])
        if not pack_function:
            raise ValueError(
                f"Unsupported data format for operand {op['name']}: {op['format'].name}"
            )

        name = op["name"]
        common_args = (
            op["buffer"],
            op["tile_count"],
            pack_function,
            self._operand_addr(name),
            self._operand_tile_size_bytes(name),
            self.num_faces,
            self.face_r_dim,
            location,
        )
        if dense:
            StimuliConfig.write_matrix_w_tile_dimensions(
                *common_args,
                self.tile_dimensions,
                use_srcs=self._operand_use_srcs(name),
                twos_complement=self.twos_complement,
            )
        else:
            StimuliConfig.write_matrix(
                *common_args,
                self.write_full_tiles,
                use_srcs=self._operand_use_srcs(name),
                twos_complement=self.twos_complement,
            )

    def _calculate_tile_sizes(self):
        """Compute tile sizes and L1 buffer addresses from current flags."""
        self.tile_size_A_bytes = calculate_tile_size_bytes(
            self.stimuli_A_format,
            self.tile_dimensions,
            format_tile_sizes,
            use_srcs=self._operand_use_srcs("A"),
        )
        self.tile_size_B_bytes = calculate_tile_size_bytes(
            self.stimuli_B_format,
            self.tile_dimensions,
            format_tile_sizes,
            use_srcs=self._operand_use_srcs("B"),
        )

        self.buf_a_addr = 0
        if StimuliConfig.WITH_COVERAGE:
            self.buf_a_addr = StimuliConfig.STIMULI_L1_ADDRESS_DEBUG
        else:
            self.buf_a_addr = StimuliConfig.STIMULI_L1_ADDRESS_PERF

        self.buf_b_addr = self.buf_a_addr + self.tile_size_A_bytes * self.tile_count_A

        next_addr = self.buf_b_addr + self.tile_size_B_bytes * self.tile_count_B

        self.tile_size_S_bytes = 0
        self.buf_s_addr = 0
        self.tile_size_T_bytes = 0
        self.buf_t_addr = 0
        self.tile_size_C_bytes = 0
        self.buf_c_addr = 0

        for op in self._active_optional_operands():
            tile_size = calculate_tile_size_bytes(
                op["format"],
                self.tile_dimensions,
                format_tile_sizes,
                use_srcs=self._operand_use_srcs(op["name"]),
            )
            if op["name"] == "S":
                self.tile_size_S_bytes = tile_size
                self.buf_s_addr = next_addr
            elif op["name"] == "T":
                self.tile_size_T_bytes = tile_size
                self.buf_t_addr = next_addr
            elif op["name"] == "C":
                self.tile_size_C_bytes = tile_size
                self.buf_c_addr = next_addr
            next_addr += tile_size * op["tile_count"]

        self.buf_res_addr = next_addr

        if self.operand_res_tile_size is not None:
            self.buf_res_tile_size = self.operand_res_tile_size
        else:
            self.buf_res_tile_size = calculate_tile_size_bytes(
                self.stimuli_res_format,
                self.tile_dimensions,
                format_tile_sizes,
                use_srcs=self._operand_use_srcs("Res"),
                dest_acc=self._dest_acc_32b,
            )

    def set_use_srcs(self, unpack_to_srcs: bool):
        """Enable SrcS-interleaved L1 layout. Called by TestConfig."""
        self.use_srcs = unpack_to_srcs
        self._calculate_tile_sizes()

    def set_dest_acc(self, dest_acc):
        """Set 32-bit dest accumulation mode. Called by TestConfig."""
        from .llk_params import DestAccumulation

        self._dest_acc_32b = dest_acc == DestAccumulation.Yes
        self._calculate_tile_sizes()

    def __str__(self) -> str:
        lines = (
            "StimuliConfig:"
            f"  buffer_A: {self.buffer_A}"
            f"  stimuli_A_format: {self.stimuli_A_format}"
            f"  tile_count_A: {self.tile_count_A}"
            f"  buffer_B: {self.buffer_B}"
            f"  stimuli_B_format: {self.stimuli_B_format}"
            f"  tile_count_B: {self.tile_count_B}"
            f"  buffer_C: {self.buffer_C}"
            f"  stimuli_C_format: {self.stimuli_C_format}"
            f"  tile_count_C: {self.tile_count_C}"
            f"  buffer_S: {self.buffer_S}"
            f"  stimuli_S_format: {self.stimuli_S_format}"
            f"  tile_count_S: {self.tile_count_S}"
            f"  buffer_T: {self.buffer_T}"
            f"  stimuli_T_format: {self.stimuli_T_format}"
            f"  tile_count_T: {self.tile_count_T}"
            f"  srcs_layout_operands: {self.srcs_layout_operands}"
            f"  stimuli_res_format: {self.stimuli_res_format}"
            f"  tile_count_res: {self.tile_count_res}"
            f"  num_faces: {self.num_faces}"
            f"  face_r_dim: {self.face_r_dim}"
            f"  tile_dimensions: {self.tile_dimensions}"
            f"  sfpu: {self.sfpu}"
            f"  write_full_tiles: {self.write_full_tiles}"
            f"  use_dense_tile_dimensions: {self.use_dense_tile_dimensions}"
            f"  use_srcs: {self.use_srcs}"
            f"  dest_acc_32b: {self._dest_acc_32b}"
            f"  operand_res_tile_size: {self.operand_res_tile_size}"
            f"  buf_a_addr: 0x{self.buf_a_addr:08X}"
            f"  buf_b_addr: 0x{self.buf_b_addr:08X}"
            f"  buf_res_addr: 0x{self.buf_res_addr:08X}"
        )
        for op in self._active_optional_operands():
            lines += (
                f"  {self._buf_addr_attr(op['name'])}:"
                f" 0x{self._operand_addr(op['name']):08X}"
            )
        return lines

    def generate_runtime_operands_values(self) -> list:
        values = [
            self.buf_a_addr,
            self.tile_size_A_bytes,
            self.buf_b_addr,
            self.tile_size_B_bytes,
        ]

        for op in self._active_optional_operands():
            if op["name"] in ("S", "T"):
                values.extend(
                    [
                        self._operand_addr(op["name"]),
                        self._operand_tile_size_bytes(op["name"]),
                    ]
                )

        values.extend(
            [
                self.buf_res_addr,
                self.buf_res_tile_size,
            ]
        )

        if self.buffer_C is not None:
            values.extend([self.buf_c_addr, self.tile_size_C_bytes])

        return values

    def generate_runtime_struct_fields(self) -> tuple[list[str], str]:
        lines: list[str] = [
            "Operand buffer_A;",
            "Operand buffer_B;",
        ]
        pack_formats = "IIII"

        for op in self._active_optional_operands():
            if op["name"] in ("S", "T"):
                lines.append(f"Operand buffer_{op['name']};")
                pack_formats += "II"

        lines.append("Operand buffer_Res;")
        pack_formats += "II"

        if self.buffer_C is not None:
            lines.append("Operand buffer_C;")
            pack_formats += "II"

        return lines, pack_formats

    def generate_stimuli_header_addresses(self) -> list[str]:
        lines: list[str] = [
            f"constexpr Operand buffer_A({hex(self.buf_a_addr)}, {self.tile_size_A_bytes});",
            f"constexpr Operand buffer_B({hex(self.buf_b_addr)}, {self.tile_size_B_bytes});",
        ]

        for op in self._active_optional_operands():
            if op["name"] in ("S", "T"):
                lines.append(
                    f"constexpr Operand buffer_{op['name']}("
                    f"{hex(self._operand_addr(op['name']))}, "
                    f"{self._operand_tile_size_bytes(op['name'])});"
                )

        lines.append(
            f"constexpr Operand buffer_Res({hex(self.buf_res_addr)}, {self.buf_res_tile_size});"
        )

        if self.buffer_C is not None:
            lines.append(
                f"constexpr Operand buffer_C({hex(self.buf_c_addr)}, {self.tile_size_C_bytes});"
            )

        return lines

    @staticmethod
    def get_packer(data_format):
        packers = {
            DataFormat.Float16: pack_fp16,
            DataFormat.Float16_b: pack_bfp16,
            DataFormat.Float32: pack_fp32,
            DataFormat.Bfp8_b: pack_bfp8_b,
            DataFormat.Bfp4_b: pack_bfp4_b,
            DataFormat.Bfp2_b: pack_bfp2_b,
            DataFormat.Int32: pack_int32,
            DataFormat.MxFp8R: pack_mxfp8r,
            DataFormat.MxFp8P: pack_mxfp8p,
            DataFormat.MxFp4: pack_mxfp4,
            DataFormat.MxInt8: pack_mxint8,
            DataFormat.MxInt4: pack_mxint4,
            DataFormat.MxInt2: pack_mxint2,
            DataFormat.Fp8_e4m3: pack_fp8_e4m3,
            DataFormat.UInt32: pack_uint32,
            DataFormat.Int16: pack_int16,
            DataFormat.UInt16: pack_uint16,
            DataFormat.Int8: pack_int8,
            DataFormat.UInt8: pack_uint8,
        }
        return packers.get(data_format)

    @staticmethod
    def write_matrix(
        buffer,
        tile_count: int,
        pack_function,
        base_address: int,
        tile_size: int,
        num_faces: int,
        face_r_dim: int,
        location: str = "0,0",
        write_full_tiles: bool = False,
        use_srcs: bool = False,
        twos_complement: bool = False,
    ):
        """
        Original backward-compatible write_matrix.
        - Always strides through buffer at MAX_TILE_ELEMENTS (1024) intervals
        - Packs either full tiles (1024 elements) or partial tiles (num_faces * face_r_dim * 16)
        """
        addresses = []
        packed_data_list = []

        # Elements to pack per tile:
        # - For tilize tests (write_full_tiles=True): write all 1024 elements
        # - For other tests: write only the faces we care about
        if write_full_tiles:
            tile_elements = MAX_TILE_ELEMENTS
        else:
            tile_elements = num_faces * face_r_dim * FACE_C_DIM

        def _pack_tile(buffer_tile):
            if pack_function in (
                pack_mxfp8r,
                pack_mxfp8p,
                pack_mxfp4,
                pack_mxint8,
                pack_mxint4,
                pack_mxint2,
            ):
                return pack_function(
                    buffer_tile,
                    num_faces=num_faces,
                    face_r_dim=face_r_dim,
                    use_srcs=use_srcs,
                )
            if pack_function in (pack_bfp8_b, pack_bfp4_b, pack_bfp2_b):
                return pack_function(
                    buffer_tile, num_faces=num_faces, face_r_dim=face_r_dim
                )
            if twos_complement and pack_function in (pack_int32, pack_int16, pack_int8):
                return pack_function(buffer_tile, twos_complement=True)
            return pack_function(buffer_tile)

        for ind in range(tile_count):
            # Always stride at MAX_TILE_ELEMENTS (1024) for backward compatibility
            start_idx = MAX_TILE_ELEMENTS * ind
            tile_data = buffer[start_idx : start_idx + tile_elements]
            packed_data = _pack_tile(tile_data)
            addresses.append(base_address + ind * tile_size)
            packed_data_list.append(packed_data)

        for addr, data in zip(addresses, packed_data_list):
            write_to_device(location, addr, data)

        # laneJN bit-exact sweep instrumentation: expose the exact bytes that
        # landed in L1 so an exhaustive sweep can verify its input coverage.
        return b"".join(packed_data_list)

    @staticmethod
    def write_matrix_w_tile_dimensions(
        buffer,
        tile_count: int,
        pack_function,
        base_address: int,
        tile_size: int,
        num_faces: int,
        face_r_dim: int,
        tile_dimensions: list[int],
        location: str = "0,0",
        use_srcs: bool = False,
        twos_complement: bool = False,
    ):
        """
        New write_matrix for variable tile dimensions with dense L1 data.
        - Strides through buffer based on actual tile_dimensions (tile_r * tile_c)
        - Always writes all elements for the given tile dimensions
        """
        addresses = []
        packed_data_list = []

        tile_r, tile_c = tile_dimensions
        tile_elements = tile_r * tile_c  # Dense: use actual tile dimensions

        def _pack_tile(buffer_tile):
            if pack_function in (
                pack_mxfp8r,
                pack_mxfp8p,
                pack_mxfp4,
                pack_mxint8,
                pack_mxint4,
                pack_mxint2,
            ):
                return pack_function(
                    buffer_tile,
                    num_faces=num_faces,
                    face_r_dim=face_r_dim,
                    use_srcs=use_srcs,
                )
            if pack_function in (pack_bfp8_b, pack_bfp4_b, pack_bfp2_b):
                return pack_function(
                    buffer_tile, num_faces=num_faces, face_r_dim=face_r_dim
                )
            if twos_complement and pack_function in (pack_int32, pack_int16, pack_int8):
                return pack_function(buffer_tile, twos_complement=True)
            return pack_function(buffer_tile)

        for ind in range(tile_count):
            start_idx = tile_elements * ind
            tile_data = buffer[start_idx : start_idx + tile_elements]
            packed_data = _pack_tile(tile_data)
            addresses.append(base_address + ind * tile_size)
            packed_data_list.append(packed_data)

        for addr, data in zip(addresses, packed_data_list):
            write_to_device(location, addr, data)

        # laneJN bit-exact sweep instrumentation (see write_matrix).
        return b"".join(packed_data_list)

    def write(self, location: str = "0,0"):
        """
        Write method that dispatches to appropriate implementation.
        - If use_dense_tile_dimensions=True: uses write_matrix_w_tile_dimensions (for new tests)
        - Otherwise: uses write_matrix (backward compatible)
        """
        _DIM = "\033[2m"
        _BOLD = "\033[1m"
        _CYAN, _YELLOW, _MAGENTA, _GREEN, _RST = (
            "\033[36m",
            "\033[33m",
            "\033[35m",
            "\033[32m",
            "\033[0m",
        )
        sep = f"{_DIM}{'─' * 52}{_RST}"
        rows = [
            f"  {_CYAN}A    0x{self.buf_a_addr:08X}{_RST}  {_DIM}{self.tile_count_A} × {self.tile_size_A_bytes} B{_RST}",
            f"  {_YELLOW}B    0x{self.buf_b_addr:08X}{_RST}  {_DIM}{self.tile_count_B} × {self.tile_size_B_bytes} B{_RST}",
        ]
        _operand_row_colors = {"C": _MAGENTA}
        for op in self._active_optional_operands():
            color = _operand_row_colors.get(op["name"], "")
            rows.append(
                f"  {color}{op['name']}    0x{self._operand_addr(op['name']):08X}{_RST}"
                f"  {_DIM}{op['tile_count']} × {self._operand_tile_size_bytes(op['name'])} B{_RST}"
            )
        rows.append(f"  {_GREEN}Res  0x{self.buf_res_addr:08X}{_RST}")
        logger.debug(
            "\n{}\n  {}L1 layout @ {}{}\n{}\n{}",
            sep,
            _BOLD,
            location,
            _RST,
            "\n".join(rows),
            sep,
        )

        if self.use_dense_tile_dimensions:
            self._write_dense_tile_dimensions(location)
        else:
            self._write_backward_compatible(location)

    def _write_backward_compatible(self, location: str = "0,0"):
        """
        Original backward-compatible write method.
        Uses write_matrix which always strides at 1024 elements.
        """
        pack_function_A = StimuliConfig.get_packer(self.stimuli_A_format)
        pack_function_B = StimuliConfig.get_packer(self.stimuli_B_format)

        # Validate pack functions for A and B
        if not pack_function_A or not pack_function_B:
            raise ValueError(
                f"Unsupported data formats: srcA({self.stimuli_A_format.name}), srcB({self.stimuli_B_format.name})"
            )

        # laneJN bit-exact sweep raw-injection point: when the test set
        # `lanejn_raw_a` (exact per-tile packed payload bytes for operand A,
        # concatenated tile 0..N-1), write THOSE bytes to L1 instead of the
        # packed generated tensor.  This bypasses the host float pack path so
        # every 16-bit/32-bit input pattern — NaN payloads included — is
        # deliverable to the kernel exactly as production L1 traffic could
        # deliver it.  Inert unless the attribute is set.
        _lanejn_raw_a = getattr(self, "lanejn_raw_a", None)
        if _lanejn_raw_a is not None:
            if len(_lanejn_raw_a) % self.tile_count_A != 0:
                raise ValueError(
                    f"lanejn_raw_a length {len(_lanejn_raw_a)} not divisible by "
                    f"tile_count_A {self.tile_count_A}"
                )
            _payload = len(_lanejn_raw_a) // self.tile_count_A
            if _payload > self.tile_size_A_bytes:
                raise ValueError(
                    f"lanejn_raw_a per-tile payload {_payload} exceeds "
                    f"tile_size_A_bytes {self.tile_size_A_bytes}"
                )
            for _ind in range(self.tile_count_A):
                write_to_device(
                    location,
                    self.buf_a_addr + _ind * self.tile_size_A_bytes,
                    _lanejn_raw_a[_ind * _payload : (_ind + 1) * _payload],
                )
            self.lanejn_src_a_raw = bytes(_lanejn_raw_a)
        else:
            self.lanejn_src_a_raw = StimuliConfig.write_matrix(
                self.buffer_A,
                self.tile_count_A,
                pack_function_A,
                self.buf_a_addr,
                self.tile_size_A_bytes,
                self.num_faces,
                self.face_r_dim,
                location,
                self.write_full_tiles,
                use_srcs=self._operand_use_srcs("A"),
                twos_complement=self.twos_complement,
            )

        StimuliConfig.write_matrix(
            self.buffer_B,
            self.tile_count_B,
            pack_function_B,
            self.buf_b_addr,
            self.tile_size_B_bytes,
            self.num_faces,
            self.face_r_dim,
            location,
            self.write_full_tiles,
            use_srcs=self._operand_use_srcs("B"),
            twos_complement=self.twos_complement,
        )

        for op in self._active_optional_operands():
            self._write_optional_operand(op, location, dense=False)

    def _write_dense_tile_dimensions(self, location: str = "0,0"):
        """
        New write method for variable tile dimensions with dense L1 data.
        Uses write_matrix_w_tile_dimensions which strides based on actual tile dimensions.
        """
        pack_function_A = StimuliConfig.get_packer(self.stimuli_A_format)
        pack_function_B = StimuliConfig.get_packer(self.stimuli_B_format)

        # Validate pack functions for A and B
        if not pack_function_A or not pack_function_B:
            raise ValueError(
                f"Unsupported data formats: srcA({self.stimuli_A_format.name}), srcB({self.stimuli_B_format.name})"
            )

        if getattr(self, "lanejn_raw_a", None) is not None:
            # laneJN raw injection is only implemented for the
            # backward-compatible tile layout; fail loudly rather than sweep
            # the wrong bytes.
            raise RuntimeError(
                "lanejn_raw_a injection is unsupported on the dense "
                "tile-dimensions write path"
            )
        self.lanejn_src_a_raw = StimuliConfig.write_matrix_w_tile_dimensions(
            self.buffer_A,
            self.tile_count_A,
            pack_function_A,
            self.buf_a_addr,
            self.tile_size_A_bytes,
            self.num_faces,
            self.face_r_dim,
            self.tile_dimensions,
            location,
            use_srcs=self._operand_use_srcs("A"),
            twos_complement=self.twos_complement,
        )
        StimuliConfig.write_matrix_w_tile_dimensions(
            self.buffer_B,
            self.tile_count_B,
            pack_function_B,
            self.buf_b_addr,
            self.tile_size_B_bytes,
            self.num_faces,
            self.face_r_dim,
            self.tile_dimensions,
            location,
            use_srcs=self._operand_use_srcs("B"),
            twos_complement=self.twos_complement,
        )

        for op in self._active_optional_operands():
            self._write_optional_operand(op, location, dense=True)

    def _collect_operand_tiles(
        self,
        operand: str,
        addr: int,
        fmt,
        count: int,
        sfpu: bool,
        *,
        debug_label: str | None = None,
        location="0,0",
    ):
        use_srcs = self._operand_use_srcs(operand)
        tile_size_bytes = calculate_tile_size_bytes(
            fmt,
            self.tile_dimensions,
            format_tile_sizes,
            use_srcs=use_srcs,
            dest_acc=self._dest_acc_32b,
        )
        read_bytes_cnt = tile_size_bytes * count

        if debug_label is not None:
            _GREEN, _DIM, _RST = "\033[32m", "\033[2m", "\033[0m"
            logger.debug(
                "Reading {}{}{}  0x{:08X}{} {}← {} B{}",
                _GREEN,
                debug_label,
                _RST,
                addr,
                _RST,
                _DIM,
                read_bytes_cnt,
                _RST,
            )

        read_data = read_from_device(location, addr, num_bytes=read_bytes_cnt)

        # laneJN bit-exact sweep instrumentation: stash the raw L1 bytes of
        # every operand read so sweeps can compare results bit-for-bit without
        # going through the (payload-canonicalizing) float unpack.
        if not hasattr(self, "lanejn_raw_reads"):
            self.lanejn_raw_reads = {}
        self.lanejn_raw_reads[operand] = bytes(read_data)

        # Pass explicit tile_stride_bytes when tiles are densely packed
        # (use_dense_tile_dimensions or use_srcs). For the backward-compatible
        # path, pass None so unpack_res_tiles strides at the full 32×32 tile
        # size and extracts only the needed faces.
        stride_bytes = (
            tile_size_bytes if (self.use_dense_tile_dimensions or use_srcs) else None
        )
        return unpack_res_tiles(
            read_data,
            fmt,
            count,
            sfpu,
            self.num_faces,
            self.face_r_dim,
            tile_stride_bytes=stride_bytes,
            use_srcs=use_srcs,
            dest_acc=self._dest_acc_32b,
            twos_complement=self.twos_complement,
        )

    def collect_results(self, location="0,0"):
        return self._collect_operand_tiles(
            "Res",
            self.buf_res_addr,
            self.stimuli_res_format,
            self.tile_count_res,
            self.sfpu,
            debug_label="Res",
            location=location,
        )

    def _operand_tile_stride_bytes(self, operand: str, fmt) -> int:
        """Per-tile L1 stride of an operand, as ``_collect`` computes it.

        Deliberately recomputed rather than reusing ``buf_res_tile_size``: that
        field honours the ``operand_res_tile_size`` override, while the read path
        does not. Sharing this helper keeps the region we clear and the region we
        read byte-for-byte identical.
        """
        return calculate_tile_size_bytes(
            fmt,
            self.tile_dimensions,
            format_tile_sizes,
            use_srcs=self._operand_use_srcs(operand),
            dest_acc=self._dest_acc_32b,
        )

    def _collect_raw(self, operand: str, addr: int, fmt, count: int, location) -> bytes:
        """Read an operand's *meaningful* bytes from L1, without decoding.

        Returns the packed bytes exactly as the packer wrote them, which is the
        right representation for a bit-identity comparison across repeated runs
        (decoding to floats would make NaN payloads compare unequal to
        themselves and hide real determinism issues).

        Only the bytes the kernel actually writes are returned. A tile occupies a
        full stride in L1, but for the backward-compatible layout the kernel
        populates just ``num_faces`` faces per tile and leaves the remaining faces
        untouched. Mirroring ``unpack_res_tiles``, we skip that padding so it
        can't cause spurious mismatches (the padding retains whatever happened to
        be in L1 before the run). The padding is skipped by reading each tile's
        prefix separately rather than reading it and discarding it, which at
        num_faces 1 and 2 leaves three quarters / half of the bytes on the device.
        """
        tile_stride = self._operand_tile_stride_bytes(operand, fmt)

        # Dense / SrcS layouts pack the whole tile densely, so every byte is
        # meaningful. The backward-compatible layout only writes the first
        # num_faces faces of each (full-size) tile slot.
        meaningful_per_tile = (
            tile_stride
            if (self.use_dense_tile_dimensions or self._operand_use_srcs(operand))
            else fmt.num_bytes_per_tile(self.num_faces * self.face_r_dim * FACE_C_DIM)
        )

        if meaningful_per_tile >= tile_stride:
            return read_from_device(location, addr, num_bytes=tile_stride * count)

        return b"".join(
            read_from_device(
                location, addr + tile * tile_stride, num_bytes=meaningful_per_tile
            )
            for tile in range(count)
        )

    @property
    def result_buffer_num_bytes(self) -> int:
        """Size of the result region in L1, including per-tile padding."""
        return (
            self._operand_tile_stride_bytes("Res", self.stimuli_res_format)
            * self.tile_count_res
        )

    def collect_raw_result_bytes(self, location="0,0") -> bytes:
        """Raw meaningful bytes of the Res output buffer."""
        return self._collect_raw(
            "Res",
            self.buf_res_addr,
            self.stimuli_res_format,
            self.tile_count_res,
            location,
        )

    def collect_raw_buffer_c_bytes(self, location="0,0") -> bytes:
        """Raw meaningful bytes of buffer_C, which some tests use as an output."""
        if self.buffer_C is None:
            raise ValueError("buffer_C is not configured")
        return self._collect_raw(
            "C",
            self.buf_c_addr,
            self.stimuli_C_format,
            self.tile_count_C,
            location,
        )

    @property
    def input_region_num_bytes(self) -> int:
        """Bytes spanned by the read-only input operands.

        Operands are laid out contiguously from ``buf_a_addr`` in the order
        A, B, S, C, Res. buffer_C is excluded because some tests use it as a
        second *output* (see ``collect_buffer_c_results``), so it legitimately
        changes during a run and must not be treated as corrupted input.
        """
        end = self.buf_c_addr if self.buffer_C is not None else self.buf_res_addr
        return end - self.buf_a_addr

    def read_input_region(self, location="0,0") -> bytes:
        """Raw L1 bytes of the input operands, to check they survived a run."""
        return read_from_device(
            location, self.buf_a_addr, num_bytes=self.input_region_num_bytes
        )

    def clear_result_buffer(self, location="0,0", fill_byte: int = 0xA5) -> None:
        """Overwrite the result region with a sentinel before a re-run.

        Ensures a repeated run genuinely recomputes the output instead of us
        re-reading stale bytes left in L1 by the previous run.
        """
        write_to_device(
            location,
            self.buf_res_addr,
            bytes([fill_byte]) * self.result_buffer_num_bytes,
        )

    def collect_buffer_c_results(self, location="0,0"):
        if self.buffer_C is None:
            raise ValueError("buffer_C is not configured")

        return self._collect_operand_tiles(
            "C",
            self.buf_c_addr,
            self.stimuli_C_format,
            self.tile_count_C,
            sfpu=False,
            location=location,
        )

    def save_to_cache(self):
        stimuli_id = sha256(
            os.environ.get("PYTEST_CURRENT_TEST", "").encode()
        ).hexdigest()
        os.makedirs(StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id, exist_ok=True)

        for buf_attr in self._CACHE_BUFFER_ATTRS:
            buffer = getattr(self, buf_attr)
            if buffer is None:
                continue
            cache_path = (
                StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id / f"{buf_attr}.pt"
            )
            logger.debug(cache_path)
            torch.save(buffer, cache_path)

        if GeneratorProxy.TEMP_RESULT is not None:
            logger.debug(StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id / "golden.pt")
            torch.save(
                GeneratorProxy.TEMP_RESULT,
                StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id / "golden.pt",
            )

    def load_from_cache(self):
        stimuli_id = sha256(
            os.environ.get("PYTEST_CURRENT_TEST", "").encode()
        ).hexdigest()
        cache_dir = StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id

        def _load(name: str, buffer):
            if buffer is None:
                return None
            cache_path = cache_dir / f"{name}.pt"
            logger.debug(cache_path)
            return torch.load(cache_path, weights_only=True)

        self.buffer_A = _load("buffer_A", self.buffer_A)
        self.buffer_B = _load("buffer_B", self.buffer_B)
        self.buffer_S = _load("buffer_S", self.buffer_S)
        self.buffer_T = _load("buffer_T", self.buffer_T)
        self.buffer_C = _load("buffer_C", self.buffer_C)
