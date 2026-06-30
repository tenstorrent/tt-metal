# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mapping of ``tt::DataFormat`` enum codes to names.

The generated ``chlkc_descriptors.h`` stores per-circular-buffer formats as raw
``uint8_t`` codes from ``tt::DataFormat`` (see
``tt_metal/api/tt-metalium/tt_backend_api_types.hpp``). This table mirrors that
enum so the analyzer can report human-readable format names.
"""

from __future__ import annotations

DATA_FORMAT_BY_CODE: dict[int, str] = {
    0: "Float32",
    1: "Float16",
    2: "Bfp8",
    3: "Bfp4",
    4: "Tf32",
    5: "Float16_b",
    6: "Bfp8_b",
    7: "Bfp4_b",
    8: "Int32",
    9: "UInt16",
    10: "Lf8",
    11: "Bfp2",
    12: "MxInt8",
    13: "Int16",
    14: "Int8",
    15: "Bfp2_b",
    16: "MxInt4",
    17: "MxInt2",
    18: "MxFp8R",
    19: "MxFp6R",
    20: "MxFp8P",
    21: "MxFp6P",
    22: "MxFp4",
    24: "UInt32",
    26: "Fp8_e4m3",
    27: "MxFp4_2x_A",
    29: "MxFp4_2x_B",
    30: "UInt8",
    240: "RawUInt8",
    241: "RawUInt16",
    242: "RawUInt32",
    255: "Invalid",
}

# Code used for unconfigured / unused circular-buffer slots.
UNUSED_FORMAT_CODE = 255


def data_format_name(code: int) -> str:
    return DATA_FORMAT_BY_CODE.get(code, f"Unknown({code})")
