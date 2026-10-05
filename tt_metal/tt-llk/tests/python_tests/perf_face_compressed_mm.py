# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf sweep of the Blackhole face_compressed_mm LLK pair (sources/face_compressed_mm_perf.cpp, the perf twin of
test_matmul_face_compressed.py); the figures are per weight tile, zero faces counted."""

import math
from dataclasses import dataclass

import numpy as np
import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.compressed_utils import (
    DEEPSEEK_T420,
    FMT_CODE,
    assign_clustered,
    generate_refined_face_assignment,
)
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
from test_matmul_face_compressed import (
    COMPRESSION_GRANULARITY,
    encode_meta,
    pack_b,
    promote_assignment,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b
MAN_WORDS = {FMT_CODE["bfp2"]: 4, FMT_CODE["bfp4"]: 8}


def face_assignment(pattern, K, N):
    if pattern in ("bfp4", "bfp2"):
        return assign_clustered(K, N, (pattern,), COMPRESSION_GRANULARITY)
    if pattern == "deepseek":
        return generate_refined_face_assignment(
            K, N, DEEPSEEK_T420, switch_mult=2.0, seed=0
        )
    raise ValueError(pattern)


def build_meta(pattern, ct, kt):
    """The meta words, the non-zero face count per activation block and the packed weight bytes of a kt x ct call."""
    K, N = kt * 32, ct * 32
    ct_f, kt_f = N // 16, K // 16
    assignment = promote_assignment(face_assignment(pattern, K, N), ct_f)
    faces = [
        (code, b"\x00" * (16 + 16 * MAN_WORDS[code])) if code != 0 else (0, b"")
        for code in assignment
    ]
    packed_b, chunk_info = pack_b(faces)
    words = np.frombuffer(
        encode_meta(assignment, ct_f, kt_f, chunk_info), dtype=np.uint32
    )
    nz_blocks = [
        sum(1 for i in range(b * 4 * ct_f, (b + 1) * 4 * ct_f) if assignment[i] != 0)
        for b in range(kt_f // 4)
    ]
    return words, nz_blocks, len(packed_b)


@dataclass
class FACE_COMPRESSED_MM_META(TemplateParameter):
    """The meta buffer of a kt x ct call and the non-zero face count per activation block, baked into the build header."""

    face_pattern: str = "bfp4"
    face_kt: int = 2
    face_ct: int = 1
    chained: bool = False

    def convert_to_cpp(self) -> str:
        words, nz_blocks, _ = build_meta(self.face_pattern, self.face_ct, self.face_kt)
        return "\n".join(
            [
                f"constexpr std::uint32_t META_WORDS = {len(words)};",
                "#define META {" + ", ".join(f"0x{int(w):08x}u" for w in words) + "}",
                "#define META_NZ_BLOCKS {" + ", ".join(str(n) for n in nz_blocks) + "}",
                f"constexpr bool CHAINED = {str(self.chained).lower()};",
            ]
        )


# (pattern, M, ct, kt): one, two and eight output tiles per call, the bfp2 data bound, the DeepSeek face distribution
# with doubled format switches, and M 1.
CASES = [
    ("bfp4", 8, 1, 16),
    ("bfp4", 8, 2, 16),
    ("bfp4", 8, 8, 16),
    ("bfp2", 8, 8, 16),
    ("deepseek", 8, 2, 16),
    ("deepseek", 8, 8, 16),
    ("bfp4", 1, 8, 16),
]


@pytest.mark.perf
@parametrize(case=CASES, chained=[False, True])
def test_perf_face_compressed_mm(perf_report, case, chained):
    pattern, in0_face_r_dim, ct, kt = case
    _, _, packed_len = build_meta(pattern, ct, kt)
    configuration = PerfConfig(
        "sources/face_compressed_mm_perf.cpp",
        InputOutputFormat(BF16, BF16, DataFormat.Bfp8_b),
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
        ],
        templates=[
            CRK_TILE_DIMM(c_dimm=ct, r_dimm=1, k_dimm=kt),
            FACE_COMPRESSED_MM_META(
                face_pattern=pattern, face_kt=kt, face_ct=ct, chained=chained
            ),
        ],
        runtimes=[
            NUM_FACES(num_faces=2, num_faces_A=2, num_faces_B=4),
            IN_FACE_DIMS(in0_face_r_dim=in0_face_r_dim),
            TILE_COUNT(kt * ct),
            LOOP_FACTOR(16),
        ],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            DataFormat.Bfp8_b,
            BF16,
            tile_count_A=kt,
            tile_count_B=math.ceil(packed_len / 1024) + 1,
            tile_count_res=ct,
        ),
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
