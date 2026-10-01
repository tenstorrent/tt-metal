# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf sweep of the Blackhole compressed_custom_mm LLK pair (sources/compressed_custom_mm_perf.cpp, the perf twin of
test_compressed_custom_mm.py); the figures are per weight tile, zero tiles counted."""

from dataclasses import dataclass

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.compressed_utils import FMT_CODE, encode_tile_meta
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    IN_FACE_DIMS,
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    TemplateParameter,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b

# The L1 format the harness reserves the in1 buffer with; the LLK strides the stream by its own per-format tile size.
DECLARED_IN1 = {
    "bfp8": DataFormat.Bfp8_b,
    "bfp4": DataFormat.Bfp4_b,
    "bfp2": DataFormat.Bfp2_b,
    "alt84": DataFormat.Bfp8_b,
    "alt82": DataFormat.Bfp8_b,
    "zero50_4": DataFormat.Bfp4_b,
    "zero50_8": DataFormat.Bfp8_b,
}


def tile_formats(pattern, kt, ct):
    """Row-major kt x ct list of FMT_CODEs for a named metadata pattern."""
    codes = []
    for k in range(kt):
        for c in range(ct):
            i = k * ct + c
            if pattern in ("bfp8", "bfp4", "bfp2"):
                codes.append(FMT_CODE[pattern])
            elif pattern == "alt84":
                codes.append(FMT_CODE["bfp8"] if i % 2 == 0 else FMT_CODE["bfp4"])
            elif pattern == "alt82":
                codes.append(FMT_CODE["bfp8"] if i % 2 == 0 else FMT_CODE["bfp2"])
            elif pattern == "zero50_4":
                codes.append(FMT_CODE["bfp4"] if c % 2 == 0 else FMT_CODE["bfp0"])
            elif pattern == "zero50_8":
                codes.append(FMT_CODE["bfp8"] if c % 2 == 0 else FMT_CODE["bfp0"])
            else:
                raise ValueError(pattern)
    return codes


@dataclass
class COMPRESSED_MM_META(TemplateParameter):
    """The metadata words of a kt x ct pattern and the non-zero tile count per k row, baked into the build header."""

    pattern: str = "bfp8"
    kt: int = 2
    ct: int = 1

    def convert_to_cpp(self) -> str:
        codes = tile_formats(self.pattern, self.kt, self.ct)
        meta = encode_tile_meta(codes, self.ct)
        words = [int.from_bytes(meta[i : i + 4], "little") for i in range(0, len(meta), 4)]
        nz = [sum(1 for c in range(self.ct) if codes[k * self.ct + c] != 0) for k in range(self.kt)]
        return "\n".join(
            [
                f"constexpr std::uint32_t META_WORDS = {len(words)};",
                "#define META {" + ", ".join(f"0x{w:08x}u" for w in words) + "}",
                "#define META_NZ {" + ", ".join(str(n) for n in nz) + "}",
            ]
        )


# (pattern, M, ct, kt): the patterns at the DeepSeek decode call shape, and the one-tile and full-width k rows.
CASES = [(p, 8, 8, 16) for p in ("bfp8", "bfp4", "bfp2", "alt84", "alt82", "zero50_4", "zero50_8")]
CASES += [("bfp8", 8, 1, 16), ("bfp8", 8, 16, 16), ("bfp4", 8, 1, 16), ("bfp4", 8, 16, 16), ("bfp4", 1, 8, 16)]


@pytest.mark.perf
@parametrize(case=CASES)
def test_perf_compressed_custom_mm(perf_report, case):
    # a single parametrized axis arrives as a one-tuple
    pattern, in0_face_r_dim, ct, kt = case[0]
    in1_format = DECLARED_IN1[pattern]
    configuration = PerfConfig(
        "sources/compressed_custom_mm_perf.cpp",
        InputOutputFormat(BF16, BF16, in1_format),
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.UNPACK_ISOLATE, PerfRunType.MATH_ISOLATE],
        templates=[
            CRK_TILE_DIMM(c_dimm=ct, r_dimm=1, k_dimm=kt),
            COMPRESSED_MM_META(pattern=pattern, kt=kt, ct=ct),
        ],
        runtimes=[
            NUM_FACES(num_faces=2, num_faces_A=2, num_faces_B=4),
            IN_FACE_DIMS(in0_face_r_dim=in0_face_r_dim),
            TILE_COUNT(kt * ct),
            LOOP_FACTOR(64),
        ],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            in1_format,
            BF16,
            tile_count_A=kt,
            tile_count_B=kt * ct,
            tile_count_res=ct,
        ),
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
