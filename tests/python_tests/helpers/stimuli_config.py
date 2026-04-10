# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import os
import shutil
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

import torch
from ttexalens.tt_exalens_lib import (
    read_from_device,
    write_to_device,
)

from .format_config import DataFormat
from .golden_generators import GeneratorProxy, ProxyMode
from .llk_params import format_tile_sizes
from .logger import logger
from .pack import (
    pack_bfp4_b,
    pack_bfp8_b,
    pack_bfp16,
    pack_fp8_e4m3,
    pack_fp16,
    pack_fp32,
    pack_int8,
    pack_int16,
    pack_int32,
    pack_mxfp8p,
    pack_mxfp8r,
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
        num_faces: int = 4,
        face_r_dim: int = 16,
        tile_dimensions: list[int] = [32, 32],
        sfpu=False,
        write_full_tiles: bool = False,
        use_dense_tile_dimensions: bool = False,
        operand_res_tile_size: int = None,
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
        self.stimuli_res_format = stimuli_res_format
        self.tile_count_res = tile_count_res
        self.num_faces = num_faces
        self.face_r_dim = face_r_dim
        self.tile_dimensions = tile_dimensions
        self.sfpu = sfpu
        self.write_full_tiles = write_full_tiles
        self.use_dense_tile_dimensions = use_dense_tile_dimensions
        self.operand_res_tile_size = operand_res_tile_size

        # Hardware flags injected by TestConfig via set_use_srcs() / set_dest_acc()
        self.use_srcs = False
        self._dest_acc_32b = False

        self._calculate_tile_sizes()

    def _calculate_tile_sizes(self):
        """Compute tile sizes and L1 buffer addresses from current flags."""
        self.tile_size_A_bytes = calculate_tile_size_bytes(
            self.stimuli_A_format,
            self.tile_dimensions,
            format_tile_sizes,
            use_srcs=self.use_srcs,
        )
        self.tile_size_B_bytes = calculate_tile_size_bytes(
            self.stimuli_B_format,
            self.tile_dimensions,
            format_tile_sizes,
            use_srcs=self.use_srcs,
        )

        self.buf_a_addr = 0
        if StimuliConfig.WITH_COVERAGE:
            self.buf_a_addr = StimuliConfig.STIMULI_L1_ADDRESS_DEBUG
        else:
            self.buf_a_addr = StimuliConfig.STIMULI_L1_ADDRESS_PERF

        self.buf_b_addr = self.buf_a_addr + self.tile_size_A_bytes * self.tile_count_A

        if self.buffer_C is not None:
            self.tile_size_C_bytes = calculate_tile_size_bytes(
                self.stimuli_C_format,
                self.tile_dimensions,
                format_tile_sizes,
                use_srcs=self.use_srcs,
            )
            self.buf_c_addr = (
                self.buf_b_addr + self.tile_size_B_bytes * self.tile_count_B
            )
            self.buf_res_addr = (
                self.buf_c_addr + self.tile_size_C_bytes * self.tile_count_C
            )
        else:
            self.buf_res_addr = (
                self.buf_b_addr + self.tile_size_B_bytes * self.tile_count_B
            )

        if self.operand_res_tile_size is not None:
            self.buf_res_tile_size = self.operand_res_tile_size
        else:
            self.buf_res_tile_size = calculate_tile_size_bytes(
                self.stimuli_res_format,
                self.tile_dimensions,
                format_tile_sizes,
                use_srcs=self.use_srcs,
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
        if self.buffer_C is not None:
            lines += f"  buf_c_addr: 0x{self.buf_c_addr:08X}"
        return lines

    def generate_runtime_operands_values(self) -> list:
        values = [
            self.buf_a_addr,
            self.tile_size_A_bytes,
            self.buf_b_addr,
            self.tile_size_B_bytes,
            self.buf_res_addr,
            self.buf_res_tile_size,
        ]

        if self.buffer_C is not None:
            values.extend([self.buf_c_addr, self.tile_size_C_bytes])

        return values

    def generate_runtime_struct_fields(self) -> tuple[list[str], str]:
        lines: list[str] = [
            "Operand buffer_A;",
            "Operand buffer_B;",
            "Operand buffer_Res;",
        ]
        pack_formats = "IIIIII"

        if self.buffer_C is not None:
            lines.append("Operand buffer_C;")
            pack_formats += "II"

        return lines, pack_formats

    def generate_stimuli_header_addresses(self) -> list[str]:
        lines: list[str] = [
            f"constexpr Operand buffer_A({hex(self.buf_a_addr)}, {self.tile_size_A_bytes});",
            f"constexpr Operand buffer_B({hex(self.buf_b_addr)}, {self.tile_size_B_bytes});",
            f"constexpr Operand buffer_Res({hex(self.buf_res_addr)}, {self.buf_res_tile_size});",
        ]

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
            DataFormat.Int32: pack_int32,
            DataFormat.MxFp8R: pack_mxfp8r,
            DataFormat.MxFp8P: pack_mxfp8p,
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
            if pack_function in (pack_mxfp8r, pack_mxfp8p):
                return pack_function(
                    buffer_tile,
                    num_faces=num_faces,
                    face_r_dim=face_r_dim,
                    use_srcs=use_srcs,
                )
            if pack_function in (pack_bfp8_b, pack_bfp4_b):
                return pack_function(
                    buffer_tile, num_faces=num_faces, face_r_dim=face_r_dim
                )
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
            if pack_function in (pack_mxfp8r, pack_mxfp8p):
                return pack_function(
                    buffer_tile,
                    num_faces=num_faces,
                    face_r_dim=face_r_dim,
                    use_srcs=use_srcs,
                )
            if pack_function in (pack_bfp8_b, pack_bfp4_b):
                return pack_function(
                    buffer_tile, num_faces=num_faces, face_r_dim=face_r_dim
                )
            return pack_function(buffer_tile)

        for ind in range(tile_count):
            start_idx = tile_elements * ind
            tile_data = buffer[start_idx : start_idx + tile_elements]
            packed_data = _pack_tile(tile_data)
            addresses.append(base_address + ind * tile_size)
            packed_data_list.append(packed_data)

        for addr, data in zip(addresses, packed_data_list):
            write_to_device(location, addr, data)

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
        if self.buffer_C is not None:
            rows.append(
                f"  {_MAGENTA}C    0x{self.buf_c_addr:08X}{_RST}  {_DIM}{self.tile_count_C} × {self.tile_size_C_bytes} B{_RST}"
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

        StimuliConfig.write_matrix(
            self.buffer_A,
            self.tile_count_A,
            pack_function_A,
            self.buf_a_addr,
            self.tile_size_A_bytes,
            self.num_faces,
            self.face_r_dim,
            location,
            self.write_full_tiles,
            use_srcs=self.use_srcs,
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
            use_srcs=self.use_srcs,
        )

        if self.buffer_C is not None:
            pack_function_C = StimuliConfig.get_packer(self.stimuli_C_format)
            if not pack_function_C:
                raise ValueError(
                    f"Unsupported data format for operand C: srcA({self.stimuli_C_format.name})"
                )
            StimuliConfig.write_matrix(
                self.buffer_C,
                self.tile_count_C,
                pack_function_C,
                self.buf_c_addr,
                self.tile_size_C_bytes,
                self.num_faces,
                self.face_r_dim,
                location,
                self.write_full_tiles,
                use_srcs=self.use_srcs,
            )

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

        StimuliConfig.write_matrix_w_tile_dimensions(
            self.buffer_A,
            self.tile_count_A,
            pack_function_A,
            self.buf_a_addr,
            self.tile_size_A_bytes,
            self.num_faces,
            self.face_r_dim,
            self.tile_dimensions,
            location,
            use_srcs=self.use_srcs,
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
            use_srcs=self.use_srcs,
        )

        if self.buffer_C is not None:
            pack_function_C = StimuliConfig.get_packer(self.stimuli_C_format)
            if not pack_function_C:
                raise ValueError(
                    f"Unsupported data format for operand C: srcA({self.stimuli_C_format.name})"
                )
            StimuliConfig.write_matrix_w_tile_dimensions(
                self.buffer_C,
                self.tile_count_C,
                pack_function_C,
                self.buf_c_addr,
                self.tile_size_C_bytes,
                self.num_faces,
                self.face_r_dim,
                self.tile_dimensions,
                location,
                use_srcs=self.use_srcs,
            )

    def collect_results(self, location="0,0"):
        # Read tiles based on actual tile dimensions
        tile_size_res_bytes = calculate_tile_size_bytes(
            self.stimuli_res_format,
            self.tile_dimensions,
            format_tile_sizes,
            use_srcs=self.use_srcs,
            dest_acc=self._dest_acc_32b,
        )
        read_bytes_cnt = tile_size_res_bytes * self.tile_count_res

        _GREEN, _DIM, _RST = "\033[32m", "\033[2m", "\033[0m"
        logger.debug(
            "Reading {}Res  0x{:08X}{} {}← {} B{}",
            _GREEN,
            self.buf_res_addr,
            _RST,
            _DIM,
            read_bytes_cnt,
            _RST,
        )

        read_data = read_from_device(
            location, self.buf_res_addr, num_bytes=read_bytes_cnt
        )

        # Pass explicit tile_stride_bytes when tiles are densely packed
        # (use_dense_tile_dimensions or use_srcs).  For the backward-compatible
        # path, pass None so unpack_res_tiles strides at the full 32×32 tile
        # size and extracts only the needed faces.
        stride_bytes = (
            tile_size_res_bytes
            if (self.use_dense_tile_dimensions or self.use_srcs)
            else None
        )
        res_from_L1 = unpack_res_tiles(
            read_data,
            self.stimuli_res_format,
            self.tile_count_res,
            self.sfpu,
            self.num_faces,
            self.face_r_dim,
            tile_stride_bytes=stride_bytes,
            use_srcs=self.use_srcs,
            dest_acc=self._dest_acc_32b,
        )
        return res_from_L1

    def save_to_cache(self):
        stimuli_id = sha256(
            os.environ.get("PYTEST_CURRENT_TEST", "").encode()
        ).hexdigest()
        os.makedirs(StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id, exist_ok=True)

        if self.buffer_A is not None:
            logger.debug(StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id / "buffer_A.pt")
            torch.save(
                self.buffer_A,
                StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id / "buffer_A.pt",
            )

        if self.buffer_B is not None:
            logger.debug(StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id / "buffer_B.pt")
            torch.save(
                self.buffer_B,
                StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id / "buffer_B.pt",
            )

        if self.buffer_C is not None:
            logger.debug(StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id / "buffer_C.pt")
            torch.save(
                self.buffer_C,
                StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id / "buffer_C.pt",
            )

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
        if self.buffer_A is not None:
            logger.debug(StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id / "buffer_A.pt")
            self.buffer_A = torch.load(
                StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id / "buffer_A.pt"
            )

        if self.buffer_B is not None:
            logger.debug(StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id / "buffer_B.pt")
            self.buffer_B = torch.load(
                StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id / "buffer_B.pt"
            )

        if self.buffer_C is not None:
            logger.debug(StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id / "buffer_C.pt")
            self.buffer_C = torch.load(
                StimuliConfig.STIMULI_CACHE_ROOT / stimuli_id / "buffer_C.pt"
            )
