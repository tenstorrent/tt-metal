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
    ct_dim = N/32 in [1,16], kt_dim = K/32 even, >= 2

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

The mask re-entry cases preload the mask from the host, then run masked and unmasked
matmuls without reinitializing the unpacker. PACK consumes the FPU_SFPU notifications.
"""

from dataclasses import dataclass

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.custom_mm_utils import face_walk_operands
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import MatmulGolden
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.pack import pack_bfp16
from helpers.param_config import parametrize
from helpers.perf.blaze import execute_llk
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    IN_FACE_DIMS,
    NUM_FACES,
    SDPA_CUSTOM_MM_FLAGS,
    TILE_COUNT,
    TemplateParameter,
)
from helpers.tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
    FACE_C_DIM,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test
from ttexalens.tt_exalens_lib import write_to_device

pytestmark = [skip_for_wormhole, skip_for_quasar]

# The simulator exits the process on these, so xfail cannot mark them.
_SIM_CFG_CONTEXT_REASON = (
    "simulator does not support tensix_unpacr with cfg_context_id=1, "
    "used by the mask/restore sequence"
)
_SIM_MOP_LOOP_COUNT_REASON = (
    "simulator does not support a MOP loop count above 31, which kt_dim 256 needs"
)


def _skip_on_simulator(request, reason):
    """Skip on the simulator only; silicon runs normally."""
    if request.config.getoption("--run-simulator"):
        pytest.skip(reason)


class _SdpaCustomMMStimuli(StimuliConfig):
    """Writes the two pre-packed operand buffers exactly as the LLK expects.

    buffer_A = in1 (SrcA, rhs): [K, N] as standard [32,32] Float16_b tiles,
               optionally with poisoned leading tiles and inter-row padding.
    buffer_B = in0 (SrcB, lhs): the [M, K] matrix packed face-by-face -- 2*kt faces of
               M x FACE_C_DIM, contiguous (identical layout to compressed_utils' packed_a).
    """

    def __init__(self, kt, ct, packed_in1, packed_in0, mask=None):
        super().__init__(
            buffer_A=torch.zeros(
                1, dtype=torch.float32
            ),  # placeholder (raw bytes below)
            stimuli_A_format=DataFormat.Float16_b,
            tile_count_A=len(packed_in1) // 2048 + (mask is not None),
            buffer_B=torch.zeros(1, dtype=torch.float32),  # placeholder
            stimuli_B_format=DataFormat.Float16_b,
            # in0 is packed densely; reserve only its bytes
            tile_count_B=max(1, -(-len(packed_in0) // 2048)),
            stimuli_res_format=DataFormat.Float16_b,
            tile_count_res=ct * (2 if mask is not None else 1),
        )
        if mask is not None:
            # One linear mask row: SrcB row i supplies the i-th 16-column face.
            packed_mask = pack_bfp16(mask)
            packed_in1 += packed_mask + bytes(2048 - len(packed_mask))
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


def _pack_in1(
    torch_b, kt, ct, read_transposed=False, row_stride=None, input_tile_offset=0
):
    """Pack in1 [K, N] as face-major Float16_b tiles with optional poison padding.

    Tile ordering must match the order the unpack LLK walks SrcA from L1
    (llk_unpack_AB_sdpa_custom_mm.h ``_llk_unpack_AB_sdpa_custom_mm_``):
      * read_transposed=False: contiguous read (block_increment == inner_increment ==
        tile_size), i.e. the walk reads tile linear-index ``k*ct + c`` -> pack K-MAJOR.
      * read_transposed=True: block_increment = row_stride*tile_size, inner_increment =
        -(((ct-1)*row_stride)-1)*tile_size, so the walk reads ``c*row_stride + k``
        -> pack C-MAJOR (outer c, inner k) so the transposed read reconstructs the exact
        same [K,N] matrix and the golden stays A@B. (The buffer, not the golden, absorbs
        the transpose; read_transposed is a SrcA L1 access-pattern axis, not a math change.)
    """
    tiles = {}
    for k in range(kt):
        for c in range(ct):
            blk = torch_b[
                k * DEFAULT_TILE_R_DIM : (k + 1) * DEFAULT_TILE_R_DIM,
                c * DEFAULT_TILE_C_DIM : (c + 1) * DEFAULT_TILE_C_DIM,
            ]
            faces = tilize_block(
                blk,
                dimensions=[DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM],
            ).flatten()
            tiles[(k, c)] = pack_bfp16(faces)

    # Finite nonzero poison makes using kt instead of the physical row stride, or
    # dropping tile_index_a, numerically visible without relying on NaN propagation.
    poison_tile = pack_bfp16(torch.full((32, 32), 16.0, dtype=torch.bfloat16))
    out = poison_tile * input_tile_offset
    if read_transposed:
        row_stride = row_stride or kt
        assert row_stride >= kt
        # Buffer linear index = input_tile_offset + c*row_stride + k.
        for c in range(ct):
            for k in range(kt):
                out += tiles[(k, c)]
            out += poison_tile * (row_stride - kt)
    else:
        for k in range(kt):  # buffer linear index = k*ct + c
            for c in range(ct):
                out += tiles[(k, c)]
    return out


def _run(
    M,
    K,
    N,
    signal_granularity,
    read_transposed,
    mm_transpose,
    row_stride=None,
    input_tile_offset=0,
    preserve_zero_flag=False,
    operands=None,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    """Run sdpa_custom_mm and check it against the golden; return the (M, N) result.

    ``operands`` optionally supplies ``(A, B)`` instead of the seeded random operands.
    """
    kt, ct = K // DEFAULT_TILE_R_DIM, N // DEFAULT_TILE_C_DIM
    assert M in {1, 2, 4, 8}, "in0 row count M must be in {1,2,4,8}"
    assert kt >= 2 and kt % 2 == 0, "kt_dim must be an even number >= 2"
    assert 1 <= ct <= 16, "ct_dim must be in [1, 16]"
    assert (
        ct % signal_granularity == 0
    ), "ct_dim must be divisible by signal_granularity"

    if operands is None:
        torch.manual_seed(0)
        torch_a = torch.randn((M, K), dtype=torch.bfloat16)
        torch_b = torch.randn((K, N), dtype=torch.bfloat16)
    else:
        torch_a, torch_b = operands

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

    outcome = execute_llk(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        test_config_kwargs=dict(
            test_name="sources/sdpa_custom_mm_test.cpp",
            formats=InputOutputFormat(
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
            ),
            templates=[
                CRK_TILE_DIMM(c_dimm=ct, r_dimm=1, k_dimm=kt),
                SDPA_CUSTOM_MM_FLAGS(
                    signal_granularity=signal_granularity,
                    read_transposed=read_transposed,
                    mm_transpose=mm_transpose,
                    sdpa_row_stride=row_stride,
                    sdpa_input_tile_offset=input_tile_offset,
                    preserve_src_zero_flag=preserve_zero_flag,
                ),
            ],
            runtimes=[
                # in1 (SrcA) = [32,32] -> 4 faces of 16 rows; in0 (SrcB) = [M,32] -> 2 faces of M
                # rows; result tile is [M,32] -> 2 faces. in1_face_r_dim stays at the 16 default.
                NUM_FACES(num_faces=2, num_faces_A=4, num_faces_B=2),
                IN_FACE_DIMS(in0_face_r_dim=M),
                TILE_COUNT(ct),
            ],
            variant_stimuli=_SdpaCustomMMStimuli(
                kt,
                ct,
                _pack_in1(
                    torch_b, kt, ct, read_transposed, row_stride, input_tile_offset
                ),
                _pack_in0(torch_a, kt),
            ),
            dest_acc=DestAccumulation.No,
        ),
    )
    if is_perf:
        return None
    res = outcome.result

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
    return res_tensor


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


def test_sdpa_custom_mm_face_walk_exact(request):
    """kt_dim = 256 as two SrcB face walks, compared exactly against A @ B."""
    _skip_on_simulator(request, _SIM_MOP_LOOP_COUNT_REASON)
    M, kt, ct = 8, 256, 1
    torch_a, torch_b = (t.to(torch.bfloat16) for t in face_walk_operands(M, kt, ct))
    res = _run(
        M,
        kt * DEFAULT_TILE_R_DIM,
        ct * DEFAULT_TILE_C_DIM,
        signal_granularity=1,
        read_transposed=False,
        mm_transpose=False,
        operands=(torch_a, torch_b),
    )
    golden = (torch_a.float() @ torch_b.float()).to(res.dtype)
    assert torch.equal(res, golden), (
        f"sdpa_custom_mm kt={kt}: {(res != golden).sum().item()} of {res.numel()} "
        "outputs differ from the exact golden"
    )


@parametrize(
    shape=SHAPES,
)
def test_sdpa_custom_mm_read_transposed(shape):
    """read_transposed=True changes the SrcA L1 read order (block_increment = kt*tile,
    inner_increment jumps back one tile), so the walk reads tile linear-index ``c*kt + k``
    instead of ``k*ct + c``. ``_pack_in1`` packs the buffer C-MAJOR to match, so the
    transposed read reconstructs the same [K,N] matrix and the golden stays A@B -- the
    buffer, not the golden, absorbs the transpose. Exercises the read_transposed unpack
    path for a clean compile + PCC.
    """
    (M, K, N) = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
    # ct>=2 shapes provide the intended transposed-read coverage.
    if N // DEFAULT_TILE_C_DIM == 1:
        pytest.skip(
            "read_transposed with ct_dim==1 selects the wrong input tile (#60119)"
        )
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


@dataclass
class SDPA_MASK_REENTRY(TemplateParameter):
    def convert_to_cpp(self):
        return "#define SDPA_MASK_REENTRY true"


@parametrize(
    shape=[(1, 64, 96), (8, 128, 128)],
    stride_mode=["implicit", "zero", "tight", "padded"],
    read_transposed=lambda stride_mode: (
        [False, True] if stride_mode == "padded" else [True]
    ),
    input_tile_offset=lambda stride_mode: ([0, 1] if stride_mode == "padded" else [0]),
)
def test_sdpa_custom_mm_row_stride(
    shape, stride_mode, read_transposed, input_tile_offset
):
    """Contract only K tiles while stepping over poisoned merged-K/V row padding.

    Omitted, zero, and explicit kt strides retain tight packing. A stride argument
    must not change contiguous reads. Leading poison also checks tile-index addressing.
    """
    M, K, N = shape
    kt = K // DEFAULT_TILE_R_DIM
    row_stride = {"implicit": None, "zero": 0, "tight": kt, "padded": kt + 2}[
        stride_mode
    ]
    _run(
        M,
        K,
        N,
        signal_granularity=1,
        read_transposed=read_transposed,
        mm_transpose=False,
        row_stride=row_stride,
        input_tile_offset=input_tile_offset,
    )


@parametrize(
    M=[1, 8],
    ct=[1, 3, 8],
    # The ct==1 fast path requires contiguous reads, as in the existing
    # test_sdpa_custom_mm_read_transposed contract above.
    read_transposed=lambda ct: [False, True] if ct > 1 else [False],
    padded_stride=lambda read_transposed: (
        [False, True] if read_transposed else [False]
    ),
)
def test_sdpa_custom_mm_mask_extent_restore(
    request, M, ct, read_transposed, padded_stride
):
    """Mask every output face, then reuse the restored SrcB geometry on the next matmul."""
    _skip_on_simulator(request, _SIM_CFG_CONTEXT_REASON)
    kt, K, N = 2, 64, ct * 32
    row_stride = kt + 2 if padded_stride else None
    input_tile_offset = 1 if padded_stride else 0
    generator = torch.Generator().manual_seed(73)
    lhs = (torch.randint(-2, 3, (M, K), generator=generator).float() / 4).to(
        torch.bfloat16
    )
    rhs = (torch.randint(-2, 3, (K, N), generator=generator).float() / 4).to(
        torch.bfloat16
    )
    # Distinct nonzero face patterns reveal a truncated mask extent. Dyadic
    # inputs keep both the mask-seeded and zero-seeded sums exactly representable.
    mask = ((torch.arange(N) % 4 - 1) * 2).to(torch.bfloat16)
    outcome = execute_llk(
        test_config_kwargs=dict(
            test_name="sources/sdpa_custom_mm_test.cpp",
            formats=InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
            templates=[
                CRK_TILE_DIMM(c_dimm=ct, r_dimm=1, k_dimm=kt),
                SDPA_CUSTOM_MM_FLAGS(
                    signal_granularity=ct,
                    read_transposed=read_transposed,
                    sdpa_row_stride=row_stride,
                    sdpa_input_tile_offset=input_tile_offset,
                ),
                SDPA_MASK_REENTRY(),
            ],
            runtimes=[
                NUM_FACES(num_faces=2, num_faces_A=4, num_faces_B=2),
                IN_FACE_DIMS(in0_face_r_dim=M),
            ],
            variant_stimuli=_SdpaCustomMMStimuli(
                kt,
                ct,
                _pack_in1(rhs, kt, ct, read_transposed, row_stride, input_tile_offset),
                _pack_in0(lhs, kt),
                mask=mask,
            ),
            dest_acc=DestAccumulation.No,
        )
    )
    packed = torch.as_tensor(outcome.result).reshape(2, ct, -1)
    result = (
        packed[:, :, : M * 32]
        .reshape(2, ct, 2, M, 16)
        .permute(0, 3, 1, 2, 4)
        .reshape(2, M, N)
    )
    product = lhs.float() @ rhs.float()
    golden = torch.stack([product + mask.float(), product]).to(torch.bfloat16)
    assert torch.equal(
        result, golden
    ), "Mask extent or SrcB geometry restoration changed the matmul"


# Smallest positive bf16 denormal, 2^-133; bf16 denormals are its multiples 1..127 (7 mantissa bits).
BF16_DENORMAL_MIN = 2.0**-133
BF16_DENORMAL_MANTISSAS = 1 << 7
# B is scaled up so a kept denormal gives a visibly nonzero product; the normal rows of A are scaled down
# by the same amount so their products stay O(1).
OPERAND_SCALE = 2.0**100


@pytest.mark.parametrize("N", [32, 64])
def test_sdpa_custom_mm_flushes_denormal_srcb_after_keep_flag(N):
    """The init must restore the Src zero flag a preceding datacopy left at keep.

    Rows 0-3 of A (SrcB) are bf16 denormals, which a bf16 MVMUL flushes to zero; scaled by B they give nonzero
    products if kept, so those rows must come out exactly 0. The golden cannot tell flushed rows from kept
    ones at its tolerance, so only the exact-zero check guards them. Rows 4-7 are normal and check the rest
    of the product.
    """
    M, K = 8, 64
    denormal_rows = 4

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.float32) / OPERAND_SCALE
    torch_a[:denormal_rows] = (
        BF16_DENORMAL_MIN
        * torch.randint(1, BF16_DENORMAL_MANTISSAS, (denormal_rows, K)).float()
    )
    torch_b = torch.randn((K, N), dtype=torch.float32) * OPERAND_SCALE

    res = _run(
        M,
        K,
        N,
        signal_granularity=1,
        read_transposed=False,
        mm_transpose=False,
        preserve_zero_flag=True,
        operands=(torch_a.to(torch.bfloat16), torch_b.to(torch.bfloat16)),
    )
    kept = res[:denormal_rows].float()
    assert (
        torch.count_nonzero(kept) == 0
    ), f"denormal SrcB rows were not flushed: max |value| {kept.abs().max().item():.3e}"
