# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


from ttexalens.tt_exalens_lib import (
    read_from_device,
    write_to_device,
)

from .format_config import DataFormat
from .llk_params import format_tile_sizes
from .pack import (
    pack_bfp8_b,
    pack_bfp16,
    pack_fp16,
    pack_fp32,
    pack_int8,
    pack_int32,
    pack_mxfp8p,
    pack_mxfp8r,
    pack_uint8,
    pack_uint16,
    pack_uint32,
)
from .tile_constants import FACE_C_DIM, MAX_TILE_ELEMENTS, calculate_tile_size_bytes
from .unpack import (
    unpack_res_tiles,
)


class StimuliConfig:

    # === STATIC VARIABLES ===
    STIMULI_L1_ADDRESS = 0x65000

    def __init__(
        self,
        buffer_A,
        stimuli_A_format: DataFormat,
        buffer_B,
        stimuli_B_format: DataFormat,
        stimuli_res_format: DataFormat,
        tile_count_A: int = 1,
        tile_count_B: int = None,
        tile_count_res: int = 1,
        buffer_C=None,
        stimuli_C_format: DataFormat = None,
        tile_count_C: int = None,
        num_faces: int = 4,
        face_r_dim: int = 16,
        tile_dimensions: list[int] = [32, 32],
        sfpu=False,
        write_full_tiles: bool = False,
        use_dense_tile_dimensions: bool = False,
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

        # Stimuli addresses calculation
        # Use actual tile size based on tile_dimensions for memory-efficient allocation
        self.tile_size_A_bytes = calculate_tile_size_bytes(
            self.stimuli_A_format, self.tile_dimensions, format_tile_sizes
        )
        self.tile_size_B_bytes = calculate_tile_size_bytes(
            self.stimuli_B_format, self.tile_dimensions, format_tile_sizes
        )

        self.buf_a_addr = StimuliConfig.STIMULI_L1_ADDRESS
        self.buf_b_addr = self.buf_a_addr + self.tile_size_A_bytes * self.tile_count_A

        if self.buffer_C is not None:
            self.tile_size_C_bytes = calculate_tile_size_bytes(
                self.stimuli_C_format, self.tile_dimensions, format_tile_sizes
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

    def generate_stimuli_header_addresses(self, formats) -> list[str]:
        # Use actual tile sizes based on tile_dimensions
        input_format = DataFormat.Float16_b if formats is None else formats.input_format
        output_format = (
            DataFormat.Float16_b if formats is None else formats.output_format
        )

        buf_a_tile_size = calculate_tile_size_bytes(
            input_format, self.tile_dimensions, format_tile_sizes
        )
        buf_b_tile_size = calculate_tile_size_bytes(
            input_format, self.tile_dimensions, format_tile_sizes
        )
        buf_res_tile_size = calculate_tile_size_bytes(
            output_format, self.tile_dimensions, format_tile_sizes
        )

        lines: list[str] = [
            f"constexpr Operand buffer_A({hex(self.buf_a_addr)}, {buf_a_tile_size});",
            f"constexpr Operand buffer_B({hex(self.buf_b_addr)}, {buf_b_tile_size});",
            f"constexpr Operand buffer_Res({hex(self.buf_res_addr)}, {buf_res_tile_size});",
        ]

        if self.buffer_C is not None:
            buf_c_tile_size = calculate_tile_size_bytes(
                input_format, self.tile_dimensions, format_tile_sizes
            )

            lines.append(
                f"constexpr Operand buffer_C({hex(self.buf_c_addr)}, {buf_c_tile_size});"
            )

        return lines

    @staticmethod
    def get_packer(data_format):
        packers = {
            DataFormat.Float16: pack_fp16,
            DataFormat.Float16_b: pack_bfp16,
            DataFormat.Float32: pack_fp32,
            DataFormat.Bfp8_b: pack_bfp8_b,
            DataFormat.Int32: pack_int32,
            DataFormat.MxFp8R: pack_mxfp8r,
            DataFormat.MxFp8P: pack_mxfp8p,
            DataFormat.UInt32: pack_uint32,
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

        pack_function_lambda = lambda buffer_tile: (
            pack_function(buffer_tile, num_faces=num_faces, face_r_dim=face_r_dim)
            if pack_function in [pack_bfp8_b, pack_mxfp8r, pack_mxfp8p]
            else pack_function(buffer_tile)
        )

        for ind in range(tile_count):
            # Always stride at MAX_TILE_ELEMENTS (1024) for backward compatibility
            start_idx = MAX_TILE_ELEMENTS * ind
            tile_data = buffer[start_idx : start_idx + tile_elements]
            packed_data = pack_function_lambda(tile_data)
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

        pack_function_lambda = lambda buffer_tile: (
            pack_function(buffer_tile, num_faces=num_faces, face_r_dim=face_r_dim)
            if pack_function in [pack_bfp8_b, pack_mxfp8r, pack_mxfp8p]
            else pack_function(buffer_tile)
        )

        for ind in range(tile_count):
            start_idx = tile_elements * ind
            tile_data = buffer[start_idx : start_idx + tile_elements]
            packed_data = pack_function_lambda(tile_data)
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
            )

    def collect_results(self, location="0,0"):
        # Read tiles based on actual tile dimensions
        tile_size_res_bytes = calculate_tile_size_bytes(
            self.stimuli_res_format, self.tile_dimensions, format_tile_sizes
        )
        read_bytes_cnt = tile_size_res_bytes * self.tile_count_res

        read_data = read_from_device(
            location, self.buf_res_addr, num_bytes=read_bytes_cnt
        )
        # Only pass tile_stride_bytes for dense tile dimensions.
        # For backward-compatible path (use_dense_tile_dimensions=False),
        # let unpack_res_tiles default to format_tile_sizes which strides at
        # full 32x32 tile size and extracts only the needed faces.
        stride_bytes = tile_size_res_bytes if self.use_dense_tile_dimensions else None
        res_from_L1 = unpack_res_tiles(
            read_data,
            self.stimuli_res_format,
            self.tile_count_res,
            self.sfpu,
            self.num_faces,
            self.face_r_dim,
            tile_stride_bytes=stride_bytes,
        )
        return res_from_L1
