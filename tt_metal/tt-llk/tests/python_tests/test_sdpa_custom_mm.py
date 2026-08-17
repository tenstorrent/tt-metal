# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Blackhole-only unit test for the experimental SDPA custom matmul LLK
(api/compute/experimental/sdpa_custom_mm.h -> sdpa_custom_mm_block, promoted by tt-metal
#53295, item 3).

The C++ driver (sources/sdpa_custom_mm_test.cpp) replicates the body of
``sdpa_custom_mm_block(...)`` call-for-call from the low-level LLKs (a tt-llk test cannot
include tt_metal/hw/inc/api/compute). See that file's header comment for the full pipeline
and the golden derivation.

GOLDEN
------
sdpa_custom_mm is the SDPA QK^T / PV matmul. operand0 (in0) -> SrcB (lhs), operand1 (in1)
-> SrcA (rhs), so:

    out[M, N] = in0[M, K] @ in1[K, N]           (LoFi only)

with the custom_mm layout limits:
    in0 (SrcB) tile shape [M, 32], M in {1,2,4,8}
    in1 (SrcA) tile shape [32, 32]
    ct_dim = N/32 in [1,16], kt_dim = K/32 even in [2,256]

The math LLK zeroes DEST (mask_chunk=false path) then accumulates the full kt walk, so the
result is a plain LoFi tiled matmul. The FPU->SFPU semaphore posts (signal_granularity
cadence) are pure signalling and do not change the numbers, so signal_granularity is a
compile-time-only axis here.

DEFINED LANES
-------------
in0 only has M rows, so each output tile has M defined rows and 32-M rows of undefined
padding. The device packs ct_dim tiles; within a tile the two 16-col faces (M x FACE_C_DIM,
row-major) sit contiguously, then padding out to the full 32-row tile. We drop the padding
and validate ONLY the M x N defined region against the golden (same reorder as
compressed_utils.run_compressed).

The mask_chunk=true path is NOT exercised: it unpacks a mask tile into SrcB and
MOVB2D-broadcasts it, which needs an SFPU-produced mask CB and a downstream SFPU consumer of
the FPU_SFPU semaphore -- out of scope for a compile+golden math test on a host without a BH
card.
"""

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import MatmulGolden
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.pack import pack_bfp16
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    IN_FACE_DIMS,
    NUM_FACES,
    SDPA_CUSTOM_MM_FLAGS,
)
from helpers.tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
    FACE_C_DIM,
)
from helpers.tilize_untilize import tilize
from helpers.utils import passed_test
from ttexalens.tt_exalens_lib import write_to_device

pytestmark = [skip_for_wormhole, skip_for_quasar]


class _SdpaCustomMMStimuli(StimuliConfig):
    """Writes the two pre-packed operand buffers exactly as the LLK expects.

    buffer_A = in1 (SrcA, rhs): the [K, N] matrix as kt*ct standard [32,32] Float16_b tiles.
    buffer_B = in0 (SrcB, lhs): the [M, K] matrix packed face-by-face -- 2*kt faces of
               M x FACE_C_DIM, contiguous (identical layout to compressed_utils' packed_a).
    """

    def __init__(self, kt, ct, packed_in1, packed_in0):
        super().__init__(
            buffer_A=torch.zeros(
                1, dtype=torch.float32
            ),  # placeholder (raw bytes below)
            stimuli_A_format=DataFormat.Float16_b,
            tile_count_A=kt * ct,
            buffer_B=torch.zeros(1, dtype=torch.float32),  # placeholder
            stimuli_B_format=DataFormat.Float16_b,
            tile_count_B=kt,
            stimuli_res_format=DataFormat.Float16_b,
            tile_count_res=ct,
        )
        self.packed_in1 = packed_in1
        self.packed_in0 = packed_in0

    def write(self, location: str = "0,0"):
        write_to_device(location, self.buf_a_addr, self.packed_in1)
        write_to_device(location, self.buf_b_addr, self.packed_in0)


def _pack_in0(torch_a, kt):
    """in0 [M, K] -> 2*kt contiguous faces of M x FACE_C_DIM (SrcB narrow-tile layout)."""
    out = b""
    for i in range(kt * 2):
        out += pack_bfp16(torch_a[:, i * FACE_C_DIM : (i + 1) * FACE_C_DIM])
    return out


def _pack_in1(torch_b, kt, ct):
    """in1 [K, N] -> kt*ct standard [32,32] Float16_b tiles (face-major, row-major grid)."""
    out = b""
    for k in range(kt):
        for c in range(ct):
            blk = torch_b[
                k * DEFAULT_TILE_R_DIM : (k + 1) * DEFAULT_TILE_R_DIM,
                c * DEFAULT_TILE_C_DIM : (c + 1) * DEFAULT_TILE_C_DIM,
            ]
            faces = tilize(
                blk.reshape(-1),
                tile_dimensions=[DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM],
            )
            out += pack_bfp16(faces)
    return out


def _run(M, K, N, signal_granularity, read_transposed, mm_transpose):
    kt, ct = K // DEFAULT_TILE_R_DIM, N // DEFAULT_TILE_C_DIM
    assert M in {1, 2, 4, 8}, "in0 row count M must be in {1,2,4,8}"
    assert kt >= 2 and kt % 2 == 0, "kt_dim must be an even number >= 2"
    assert 1 <= ct <= 16, "ct_dim must be in [1, 16]"
    assert (
        ct % signal_granularity == 0
    ), "ct_dim must be divisible by signal_granularity"

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N), dtype=torch.bfloat16)

    # Golden: LoFi tiled matmul A@B, row-major (tilize handled device-side). Instantiate
    # MatmulGolden directly (not via get_golden_generator) so the compile-producer's dummy
    # generator does not break the narrow-M reshape -- same reason as compressed_utils.
    golden = MatmulGolden()(
        torch_a,
        torch_b,
        DataFormat.Float16_b,
        MathFidelity.LoFi,
        input_A_dimensions=[M, K],
        input_B_dimensions=[K, N],
        tilize=False,
        input_A_format=DataFormat.Float16_b,
        input_B_format=DataFormat.Float16_b,
    ).reshape(M, N)

    configuration = TestConfig(
        "sources/sdpa_custom_mm_test.cpp",
        InputOutputFormat(
            input_format=DataFormat.Float16_b,
            output_format=DataFormat.Float16_b,
        ),
        templates=[
            CRK_TILE_DIMM(c_dimm=ct, r_dimm=1, k_dimm=kt),
            SDPA_CUSTOM_MM_FLAGS(
                signal_granularity=signal_granularity,
                read_transposed=read_transposed,
                mm_transpose=mm_transpose,
            ),
        ],
        runtimes=[
            # in1 (SrcA) = [32,32] -> 4 faces of 16 rows; in0 (SrcB) = [M,32] -> 2 faces of M
            # rows; result tile is [M,32] -> 2 faces. in1_face_r_dim stays at the 16 default.
            NUM_FACES(num_faces=2, num_faces_A=4, num_faces_B=2),
            IN_FACE_DIMS(in0_face_r_dim=M),
        ],
        variant_stimuli=_SdpaCustomMMStimuli(
            kt, ct, _pack_in1(torch_b, kt, ct), _pack_in0(torch_a, kt)
        ),
        dest_acc=DestAccumulation.No,
    )

    res = configuration.run().result

    # Device packs ct = N//32 result tiles; within each tile the two 16-col faces
    # (M x FACE_C_DIM, row-major) are contiguous, then padded to a full 32-row tile.
    # Drop per-tile padding and reorder to row-major (M, N).
    res_tensor = torch.as_tensor(res, dtype=torch.bfloat16)
    faces_per_tile = DEFAULT_TILE_C_DIM // FACE_C_DIM
    per_tile = res_tensor.reshape(ct, -1)[:, : M * DEFAULT_TILE_C_DIM]
    res_tensor = (
        per_tile.reshape(ct, faces_per_tile, M, FACE_C_DIM)
        .permute(2, 0, 1, 3)
        .reshape(M, N)
    )

    # K-aware absolute floor (same rationale as compressed_utils.run_compressed): the LoFi
    # MVMUL accumulates the K-deep sum in a bf16 dest, so noise grows ~linearly per K-tile.
    FLOAT16B_DEFAULT_ATOL = 0.05
    ACC_ATOL_PER_KT = 0.005
    active = golden.abs()
    active = active[active > 0]
    mean_active = active.mean().item() if active.numel() else 0.0
    acc_atol = max(FLOAT16B_DEFAULT_ATOL, ACC_ATOL_PER_KT * kt * mean_active)

    assert passed_test(
        golden,
        res_tensor,
        DataFormat.Float16_b,
        custom_atol=acc_atol,
        print_pcc=True,
    ), f"sdpa_custom_mm failed for (M={M}, K={K}, N={N}, sg={signal_granularity}, read_transposed={read_transposed})"


# Shapes: kt = K/32 (even, >=2), ct = N/32 (1..16), M in {1,2,4,8}. Kept small so the sweep
# stays tractable while crossing single-/multi-c-tile (ct=1 vs ct>1, which pick different
# MOP template modes in the unpack LLK) and multi-K accumulation.
SHAPES = [
    (1, 64, 32),  # M=1, kt=2, ct=1  (ct==1 UNPACR_B MOP mode)
    (2, 64, 64),  # M=2, kt=2, ct=2  (ct>=2 UNPACR_A1/2/3 MOP mode)
    (4, 128, 96),  # M=4, kt=4, ct=3  (odd ct, first/second-half split)
    (8, 256, 128),  # M=8, kt=8, ct=4  (deeper K accumulation)
]


@parametrize(
    shape=SHAPES,
)
def test_sdpa_custom_mm(shape):
    """Default configuration: signal_granularity=1 (post per c-tile), no read transpose."""
    # A single-name parametrize axis binds the value as-is (a 1-tuple wrapping the shape),
    # so unwrap one level before destructuring.
    (M, K, N) = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
    _run(M, K, N, signal_granularity=1, read_transposed=False, mm_transpose=False)


@parametrize(
    shape=SHAPES,
)
def test_sdpa_custom_mm_read_transposed(shape):
    """read_transposed=True changes the SrcA L1 walk (block/inner increments) but not the
    numeric result for a symmetric [K,N] operand read in transposed order; the golden is the
    same A@B. This exercises the read_transposed unpack path for a clean compile + PCC.
    """
    (M, K, N) = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
    _run(M, K, N, signal_granularity=1, read_transposed=True, mm_transpose=False)


@parametrize(
    # signal_granularity must divide ct_dim; pick shapes where a non-unit granularity is
    # legal. sg == ct takes the header's single-post fast path; the result is identical.
    shape_sg=[
        ((2, 64, 64), 2),  # sg == ct : fast path
        ((8, 256, 128), 2),  # sg = 2, ct = 4 : per-2-c-tile cadence
        ((8, 256, 128), 4),  # sg == ct : fast path
    ],
)
def test_sdpa_custom_mm_signal_granularity(shape_sg):
    """signal_granularity only changes the FPU->SFPU post cadence, never the numbers, so all
    legal granularities must reproduce the same A@B golden."""
    # Single-name parametrize binds the value as-is (a 1-tuple wrapping (shape, sg)).
    (shape, sg) = (
        shape_sg[0]
        if len(shape_sg) == 1 and isinstance(shape_sg[0], tuple)
        else shape_sg
    )
    M, K, N = shape
    _run(M, K, N, signal_granularity=sg, read_transposed=False, mm_transpose=False)
