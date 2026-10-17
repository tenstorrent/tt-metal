# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Plain (non-compressed) custom_mm matmul on the Blackhole-only experimental LLK
pipeline (llk_{unpack_AB,math}_custom_mm.h, promoted by tt-metal #52727).

``llk_math_custom_mm.h`` had no tt-llk coverage before this test. The compressed
sibling (``test_matmul_custom_compressed.py``) exercises the compressed variant; this is
the first driver of the PLAIN path, and its harness is the compressed one with the
BFP quantization / meta stripped out.

What the LLK computes
---------------------
A standard tiled matmul ``C[M,N] = A[M,K] @ B[K,N]`` with the header's tile-shape limits:

  in0 (A)  -> SrcB, tile shape ``[{1,2,4,8}, 32]``: only the top two faces, each
              ``face_r_dim = M`` rows tall. So M is restricted to {1,2,4,8}.
  in1 (B)  -> SrcA, full ``[32,32]`` tiles.
  rt_dim = 1, ct_dim in [1,16], kt_dim in [1,256], LoFi only.

``split_acc=false`` / ``finalize=false``, so there is no finalization merge and DEST
holds the plain accumulated product (custom_mm.h: finalize must be false when split_acc
is false). The result is packed as ``ct_dim`` output tiles; inside each tile the two
16-col faces (M x 16, row-major) sit contiguously, then pad out to a full 32-row tile.

Only defined lanes are asserted
-------------------------------
The LLK writes ONLY the top ``M`` rows of each output face. The remaining ``32 - M`` rows
of every 32-row DEST tile are undefined, so the reorder below drops the per-tile padding
and compares ONLY the ``M`` defined rows of each tile against the golden ``A @ B``.

The ``ct in {7,9,11}`` question
-------------------------------
The unpack MOP splits each k-tile's ``ct_dim`` output tiles into
``first_half = ceil(ct/2)`` and ``second_half = floor(ct/2)`` replay ranges
(llk_unpack_AB_custom_mm.h, the block near lines 130-160). Odd ct_dim is the case where
the two halves differ in length, so this sweep includes the odd widths 7, 9, 11 (as well
as 1 and even widths) to settle whether that odd-ct split multiplies correctly.

Blackhole only: these LLKs live only in the Blackhole tree and cannot run on WH/Quasar.
This test writes a correct-by-construction golden; runtime pass/fail is a BH-card check.
"""

from dataclasses import dataclass

import pytest
import torch
from conftest import blackhole_only, skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import MatmulGolden
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.pack import pack_bfp4_b, pack_bfp8_b, pack_bfp16, pack_fp32
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    CUSTOM_MM_CALLS,
    CUSTOM_MM_IN1_WINDOW,
    IN_FACE_DIMS,
    NUM_FACES,
    TemplateParameter,
)
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM, FACE_C_DIM
from helpers.tilize_untilize import tilize, untilize
from helpers.unpack import unpack_bfp4_b, unpack_bfp8_b
from helpers.utils import passed_test

pytestmark = [skip_for_wormhole, skip_for_quasar]

# in0 (A/SrcB) row count. The header restricts this to {1,2,4,8}; 8 is the largest and
# exercises both faces at full height, 1/2/4 cover the narrow-M cases.
SUPPORTED_M = (1, 2, 4, 8)

# Packer for the plain B/result path, keyed on format. Float16_b packs as bf16, Float32
# as raw fp32; both round-trip through torch below so the golden folds in the same
# representation the device sees.
_PACKERS = {
    DataFormat.Float16_b: pack_bfp16,
    DataFormat.Float32: pack_fp32,
    DataFormat.Bfp8_b: lambda tensor, num_faces=4: bytes(
        pack_bfp8_b(tensor, num_faces=num_faces)
    ),
    DataFormat.Bfp4_b: lambda tensor, num_faces=4: bytes(
        pack_bfp4_b(tensor, num_faces=num_faces)
    ),
}

_BFP_UNPACKERS = {
    DataFormat.Bfp8_b: unpack_bfp8_b,
    DataFormat.Bfp4_b: unpack_bfp4_b,
}


@dataclass
class CUSTOM_MM_PRESERVE_SRC_ZERO_FLAG(TemplateParameter):
    """Math sets the Src zero flag to keep before the custom_mm init, as a preceding copy_tile_init does."""

    preserve: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool CUSTOM_MM_PRESERVE_SRC_ZERO_FLAG = {str(self.preserve).lower()};"


class CustomMMStimuliConfig(StimuliConfig):
    """A into buffer_A (kt*2 faces of [M,16]), B tilized into buffer_B (kt*ct full tiles).

    Mirrors compressed_utils.CompressedStimuliConfig but has no meta / buffer_C. The
    focused odd-K cases also use a uniform BFP4_b or BFP8_b stream for B.
    """

    def __init__(
        self,
        kt,
        ct,
        in0_format,
        in1_format,
        out_format,
        packed_a,
        packed_b,
        in1_tiles=None,
    ):
        super().__init__(
            buffer_A=torch.zeros(
                1, dtype=torch.float32
            ),  # placeholder; real bytes below
            stimuli_A_format=in0_format,
            tile_count_A=kt,
            buffer_B=torch.zeros(1, dtype=torch.float32),  # placeholder
            stimuli_B_format=in1_format,
            tile_count_B=in1_tiles or kt * ct,
            stimuli_res_format=out_format,
            tile_count_res=ct,
        )
        self.packed_a = packed_a
        self.packed_b = packed_b

    def write(self, location: str = "0,0"):
        from ttexalens.tt_exalens_lib import write_to_device

        write_to_device(location, self.buf_a_addr, self.packed_a)
        write_to_device(location, self.buf_b_addr, self.packed_b)


def _run_custom_mm(
    M,
    kt,
    ct,
    formats,
    dest_acc,
    calls=CUSTOM_MM_CALLS(),
    preserve_zero_flag=False,
    operands=None,
    in1_window=CUSTOM_MM_IN1_WINDOW(),
    in1_tile_w=DEFAULT_TILE_C_DIM,
):
    """Run custom_mm and check it against the golden; return the (M, N) result.

    ``operands`` optionally supplies ``(A, B)`` instead of the seeded random operands.
    ``in1_window`` lays B out ``stride`` tiles wide and computes the ct-tile column window at ``col_start``.
    With ``read_transposed`` B is column-major instead, ``stride`` (default kt) tiles per column, the slots past
    kt holding filler tiles, and one extra column after the window so a read past it lands on live data.
    ``in1_tile_w`` 16 uses [32, 16] B tiles, giving [M, 16] output tiles.
    """
    K = kt * DEFAULT_TILE_R_DIM
    N = ct * in1_tile_w
    col_start = in1_window.col_start
    read_transposed = in1_window.in1_read_transposed
    in1_row_tiles = col_start + ct + 1 if read_transposed else in1_window.stride or ct
    in1_col_tiles = in1_window.stride or kt if read_transposed else kt
    in1_tile_dims = [DEFAULT_TILE_R_DIM, in1_tile_w]
    in1_faces = 2 * in1_tile_w // FACE_C_DIM
    in0_format = formats.input_format
    in1_format = formats.input_format_B
    out_format = formats.output_format
    in0_packer = _PACKERS[in0_format]
    in1_packer = _PACKERS[in1_format]

    if operands is None:
        torch.manual_seed(0)
        torch_a = torch.randn((M, K), dtype=torch.float32)
        torch_b = torch.randn(
            (in1_col_tiles * DEFAULT_TILE_R_DIM, in1_row_tiles * in1_tile_w),
            dtype=torch.float32,
        )
    else:
        torch_a, torch_b = operands

    # in0 (A -> SrcB): kt*2 faces of [M, 16], column-face order along K, contiguous.
    packed_a = b""
    for i in range(kt * 2):
        packed_a += in0_packer(torch_a[:, i * FACE_C_DIM : (i + 1) * FACE_C_DIM])

    # in1 (B -> SrcA): in1_col_tiles x in1_row_tiles full [32,32] tiles, k-major / c-minor -- the order the SrcA
    # CFGSHIFTMASK walk reads them (read_transposed=false) -- or c-major / k-minor with read_transposed. The kernel
    # reads a ct-wide window of each row. Rows past kt are filler that only a misaddressed read reaches.
    packed_tiles = {}
    golden_b = torch_b.clone()
    for r in range(in1_col_tiles):
        for c in range(in1_row_tiles):
            blk = torch_b[
                r * DEFAULT_TILE_R_DIM : (r + 1) * DEFAULT_TILE_R_DIM,
                c * in1_tile_w : (c + 1) * in1_tile_w,
            ]
            # Face-major (tilized) layout inside each 32x32 tile. Tilize in the input
            # format so the returned dtype matches the packer (pack_fp32 rejects a
            # bfloat16 tensor, which is tilize's default Float16_b dtype).
            tiled_blk = tilize(
                blk.reshape(-1),
                stimuli_format=in1_format,
                tile_dimensions=in1_tile_dims,
            )
            packed_tile = (
                in1_packer(tiled_blk, num_faces=in1_faces)
                if in1_format in _BFP_UNPACKERS
                else in1_packer(tiled_blk)
            )
            packed_tiles[r, c] = packed_tile
            if in1_format in _BFP_UNPACKERS:
                golden_b[
                    r * DEFAULT_TILE_R_DIM : (r + 1) * DEFAULT_TILE_R_DIM,
                    c * in1_tile_w : (c + 1) * in1_tile_w,
                ] = untilize(
                    _BFP_UNPACKERS[in1_format](packed_tile, num_faces=in1_faces),
                    tile_dimensions=in1_tile_dims,
                ).reshape(
                    DEFAULT_TILE_R_DIM, in1_tile_w
                )
    tile_order = (
        [(r, c) for c in range(in1_row_tiles) for r in range(in1_col_tiles)]
        if read_transposed
        else [(r, c) for r in range(in1_col_tiles) for c in range(in1_row_tiles)]
    )
    packed_b = b"".join(packed_tiles[rc] for rc in tile_order)
    golden_b = golden_b[:K]

    # Golden: plain A @ B (LoFi), row-major, NOT tilized -- the device-side layout is
    # undone in the result reorder below. Instantiate MatmulGolden directly (not via
    # get_golden_generator) so the compile-producer's dummy generator does not break the
    # narrow-M reshape.
    window_cols = slice(col_start * in1_tile_w, (col_start + ct) * in1_tile_w)
    golden = MatmulGolden()(
        torch_a,
        golden_b[:, window_cols].contiguous(),
        out_format,
        MathFidelity.LoFi,
        input_A_dimensions=[M, K],
        input_B_dimensions=[K, N],
        tilize=False,
        input_A_format=in0_format,
        # BFP inputs were round-tripped through the exact bytes written to L1 above.
        input_B_format=(
            DataFormat.Float16_b if in1_format in _BFP_UNPACKERS else in1_format
        ),
    ).reshape(M, N)

    configuration = TestConfig(
        "sources/custom_mm_test.cpp",
        formats,
        templates=[
            CRK_TILE_DIMM(c_dimm=ct, r_dimm=1, k_dimm=kt),
            calls,
            CUSTOM_MM_PRESERVE_SRC_ZERO_FLAG(preserve_zero_flag),
            in1_window,
        ],
        runtimes=[
            # in0 uses 2 faces (M x 16 each); in1 (B) 4, or 2 for [32, 16]; the result one face per 16 columns.
            NUM_FACES(
                num_faces=in1_tile_w // FACE_C_DIM, num_faces_A=2, num_faces_B=in1_faces
            ),
            IN_FACE_DIMS(in0_face_r_dim=M),
        ],
        variant_stimuli=CustomMMStimuliConfig(
            kt,
            ct,
            in0_format,
            in1_format,
            out_format,
            packed_a,
            packed_b,
            in1_col_tiles * in1_row_tiles,
        ),
        dest_acc=dest_acc,
    )

    res = configuration.run().result
    torch_out = torch.float32 if out_format == DataFormat.Float32 else torch.bfloat16
    res_tensor = torch.as_tensor(res, dtype=torch_out)

    # Device packs ct tiles; within a tile the two 16-col faces (M x FACE_C_DIM, row-major)
    # sit contiguously, padded out to a full 32-row tile. Drop the per-tile padding
    # (span = len // ct) and reorder to row-major (M, N). Only the M defined rows survive.
    faces_per_tile = in1_tile_w // FACE_C_DIM
    per_tile = res_tensor.reshape(ct, -1)[:, : M * in1_tile_w]
    res_tensor = (
        per_tile.reshape(ct, faces_per_tile, M, FACE_C_DIM)
        .permute(2, 0, 1, 3)
        .reshape(M, N)
    )

    golden = golden.reshape(M, N).to(torch_out)

    # K-aware absolute floor (Float16_b / bf16 dest only): the single LoFi MVMUL
    # accumulates the K-deep sum in a bf16 dest, so rounding noise grows ~linearly per
    # k-tile -- a floor on small outputs that Float16_b's default atol (0.05) is too tight
    # for at kt>=4. Scale it by kt * mean|nonzero golden| (never below default; rtol/PCC
    # unchanged, PCC is the real gate and stays ~0.99999). This mirrors the proven
    # compressed sibling (compressed_utils.run_compressed). Float32 dest accumulates in
    # fp32, so its noise is negligible and it keeps the default (tighter) atol.
    custom_atol = None
    if out_format == DataFormat.Float16_b:
        FLOAT16B_DEFAULT_ATOL = 0.05
        ACC_ATOL_PER_KT = 0.005
        active_golden = golden.abs()
        active_golden = active_golden[active_golden > 0]
        mean_active = active_golden.mean().item() if active_golden.numel() else 0.0
        custom_atol = max(FLOAT16B_DEFAULT_ATOL, ACC_ATOL_PER_KT * kt * mean_active)
        if in1_window.split_acc:
            # split_acc rounds its two partial sums to bf16 apart before the finalize merges them, so an
            # output that cancels to near zero is off by up to a bf16 step of a partial (2^-7 of its size).
            custom_atol = max(custom_atol, golden.abs().max().item() * 2**-7)

    assert passed_test(
        golden, res_tensor, out_format, custom_atol=custom_atol, print_pcc=True
    ), f"custom_mm matmul failed for M={M} kt={kt} ct={ct} formats={formats}"
    return res_tensor


# LoFi-only per the header ("fidelity: LoFi only"). Float16_b (16-bit dest) and Float32
# (32-bit dest) cover both DEST widths.
CUSTOM_MM_FORMATS = input_output_formats(
    [DataFormat.Float16_b, DataFormat.Float32], same=True
)

# kt is in [1,256]; small even values cover the paired MOP path.
KT_DIMS = [2, 4]

# ct in [1,16]. Include 1, an even width, and the odd widths 7/9/11 that the header's
# first_half/second_half MOP split treats asymmetrically (the ct in {7,9,11} question).
CT_DIMS = [1, 2, 7, 9, 11, 16]


def _dest_acc_for(formats):
    # Float32 operands require a 32-bit dest; Float16_b uses the 16-bit dest.
    if formats.input_format == DataFormat.Float32:
        return DestAccumulation.Yes
    return DestAccumulation.No


@blackhole_only
@parametrize(
    formats=CUSTOM_MM_FORMATS,
    M=list(SUPPORTED_M),
    kt=KT_DIMS,
    ct=CT_DIMS,
)
def test_custom_mm(formats, M, kt, ct):
    _run_custom_mm(M, kt, ct, formats, _dest_acc_for(formats))


ODD_K_CASES = [
    pytest.param(
        1,
        1,
        1,
        InputOutputFormat(
            DataFormat.Float16_b, DataFormat.Float16_b, DataFormat.Bfp4_b
        ),
        id="k1-ct1-bfp4-post1",
    ),
    pytest.param(
        1,
        9,
        1,
        InputOutputFormat(
            DataFormat.Float16_b, DataFormat.Float16_b, DataFormat.Bfp8_b
        ),
        id="k9-ct1-bfp8-post0",
    ),
    pytest.param(
        1,
        9,
        2,
        InputOutputFormat(
            DataFormat.Float16_b, DataFormat.Float16_b, DataFormat.Bfp4_b
        ),
        id="k9-ct2-bfp4-post1",
    ),
    pytest.param(
        8,
        9,
        3,
        InputOutputFormat(
            DataFormat.Float16_b, DataFormat.Float16_b, DataFormat.Bfp8_b
        ),
        id="k9-ct3-bfp8-post0",
    ),
]


@blackhole_only
@pytest.mark.parametrize("M,kt,ct,formats", ODD_K_CASES)
def test_custom_mm_odd_k(formats, M, kt, ct):
    """Exercise the single-K replay tail with both unpack tunings."""
    _run_custom_mm(M, kt, ct, formats, DestAccumulation.No)


# Multi-call: K splits over back-to-back calls that accumulate into one DEST, the way the blaze
# Matmul drives row blocks. The kernel holds math back before each call, so the unpacker enters
# the next call while math still holds SrcB; a both-bank SrcB clear issued there by an execute
# (rather than once by the init) drops the SrcA writes it overlaps.
MULTI_CALL_CASES = [
    # (id, M, kt, ct, calls, in1 format)
    ("m1-ct1", 1, 32, 1, 16, DataFormat.Float16_b),
    ("m8-ct1", 8, 32, 1, 16, DataFormat.Float16_b),
    ("m1-ct2", 1, 16, 2, 8, DataFormat.Float16_b),
    ("m1-ct1-bfp8", 1, 32, 1, 16, DataFormat.Bfp8_b),
]


@blackhole_only
@pytest.mark.parametrize(
    "M,kt,ct,num_calls,in1_format",
    [pytest.param(*case[1:], id=case[0]) for case in MULTI_CALL_CASES],
)
def test_custom_mm_multi_call(M, kt, ct, num_calls, in1_format):
    """Back-to-back calls into one DEST, checked against the full-K golden."""
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b, in1_format)
    _run_custom_mm(
        M, kt, ct, formats, DestAccumulation.No, CUSTOM_MM_CALLS(num_calls=num_calls)
    )


# Column windows of a wider in1, read through in1_stride: an output wider than 16 tiles (one DEST pass)
# is produced window by window this way. ct == 1 takes its row step on the full-unpack tile, and odd kt
# runs the single-K tail, so both sit next to the paired MOP path.
IN1_WINDOW_CASES = [
    # (id, M, kt, ct, stride, col_start, calls, in1 format)
    ("ct1", 1, 4, 1, 3, 2, 1, DataFormat.Float16_b),
    ("ct2", 4, 4, 2, 5, 1, 1, DataFormat.Float16_b),
    ("ct16-of-40", 1, 2, 16, 40, 16, 1, DataFormat.Float16_b),
    ("odd-k-ct3", 8, 3, 3, 7, 4, 1, DataFormat.Bfp8_b),
    ("odd-k-ct1-bfp4", 1, 5, 1, 4, 3, 1, DataFormat.Bfp4_b),
    ("two-calls-ct4", 1, 8, 4, 12, 8, 2, DataFormat.Float16_b),
]


@blackhole_only
@pytest.mark.parametrize(
    "M,kt,ct,stride,col_start,num_calls,in1_format",
    [pytest.param(*case[1:], id=case[0]) for case in IN1_WINDOW_CASES],
)
def test_custom_mm_in1_window(M, kt, ct, stride, col_start, num_calls, in1_format):
    """A ct-wide column window of a stride-tile-wide in1, checked against that window's golden."""
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b, in1_format)
    _run_custom_mm(
        M,
        kt,
        ct,
        formats,
        DestAccumulation.No,
        CUSTOM_MM_CALLS(num_calls=num_calls),
        in1_window=CUSTOM_MM_IN1_WINDOW(stride=stride, col_start=col_start),
    )


# Column-major in1 read with read_transposed: K steps one tile within an output column, and in1_stride is the
# tile step between columns. ct == 1 takes its K step on the full-unpack tile, which must stay one tile
# whatever the column stride is.
READ_TRANSPOSED_CASES = [
    # (id, M, kt, ct, stride, col_start, in1 format)
    ("ct1-odd-k", 1, 3, 1, 0, 0, DataFormat.Float16_b),
    ("ct1-stride", 4, 3, 1, 5, 2, DataFormat.Float16_b),
    ("ct1-bfp4", 1, 4, 1, 0, 1, DataFormat.Bfp4_b),
    ("ct3", 8, 4, 3, 0, 0, DataFormat.Float16_b),
    ("ct2-stride-odd-k-bfp8", 2, 3, 2, 5, 1, DataFormat.Bfp8_b),
    ("ct16", 1, 2, 16, 0, 0, DataFormat.Float16_b),
]


@blackhole_only
@pytest.mark.parametrize(
    "M,kt,ct,stride,col_start,in1_format",
    [pytest.param(*case[1:], id=case[0]) for case in READ_TRANSPOSED_CASES],
)
def test_custom_mm_read_transposed(M, kt, ct, stride, col_start, in1_format):
    """A ct-wide window of a column-major in1, read with read_transposed, against that window's golden."""
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b, in1_format)
    _run_custom_mm(
        M,
        kt,
        ct,
        formats,
        DestAccumulation.No,
        in1_window=CUSTOM_MM_IN1_WINDOW(
            stride=stride, col_start=col_start, in1_read_transposed=True
        ),
    )


# [32, 16] in1 tiles: 16-wide output tiles, so an output width of any multiple of 16 columns. split_acc
# finalizes on the last call; full-tile split_acc cases sit alongside as the reference for that path.
SPLIT_AND_NARROW_CASES = [
    # (id, M, kt, ct, in1 tile width, split_acc, stride, col_start, calls, in1 format)
    ("n16-m1", 1, 4, 1, 16, False, 0, 0, 1, DataFormat.Float16_b),
    ("n16-m8-split", 8, 4, 1, 16, True, 0, 0, 1, DataFormat.Float16_b),
    ("n48-m4-odd-k-split", 4, 3, 3, 16, True, 0, 0, 1, DataFormat.Float16_b),
    ("n80-bfp8-split", 1, 8, 5, 16, True, 0, 0, 1, DataFormat.Bfp8_b),
    ("n16-bfp4-odd-k-split", 1, 5, 1, 16, True, 0, 0, 1, DataFormat.Bfp4_b),
    ("n256-m8-split", 8, 2, 16, 16, True, 0, 0, 1, DataFormat.Float16_b),
    ("n32-window-split", 2, 4, 2, 16, True, 5, 2, 1, DataFormat.Float16_b),
    ("n48-two-calls-split", 1, 8, 3, 16, True, 0, 0, 2, DataFormat.Float16_b),
    ("full-m8-split", 8, 4, 3, 32, True, 0, 0, 1, DataFormat.Float16_b),
    ("full-m1-odd-k-bfp4-split", 1, 5, 2, 32, True, 0, 0, 1, DataFormat.Bfp4_b),
]


@blackhole_only
@pytest.mark.parametrize(
    "M,kt,ct,in1_tile_w,split_acc,stride,col_start,num_calls,in1_format",
    [pytest.param(*case[1:], id=case[0]) for case in SPLIT_AND_NARROW_CASES],
)
def test_custom_mm_split_acc_and_narrow_in1(
    M, kt, ct, in1_tile_w, split_acc, stride, col_start, num_calls, in1_format
):
    """[32, 16] in1 tiles and split accumulation with finalization, against the plain golden."""
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b, in1_format)
    _run_custom_mm(
        M,
        kt,
        ct,
        formats,
        DestAccumulation.No,
        CUSTOM_MM_CALLS(num_calls=num_calls),
        in1_window=CUSTOM_MM_IN1_WINDOW(
            stride=stride, col_start=col_start, split_acc=split_acc
        ),
        in1_tile_w=in1_tile_w,
    )


# Smallest positive bf16 denormal, 2^-133; bf16 denormals are its multiples 1..127 (7 mantissa bits).
BF16_DENORMAL_MIN = 2.0**-133
BF16_DENORMAL_MANTISSAS = 1 << 7
# B is scaled up so a kept denormal gives a visibly nonzero product; the normal rows of A are scaled down
# by the same amount so their products stay O(1).
OPERAND_SCALE = 2.0**100


@blackhole_only
@pytest.mark.parametrize("ct", [1, 2])
def test_custom_mm_flushes_denormal_srcb_after_keep_flag(ct):
    """The init must restore the Src zero flag a preceding datacopy left at keep.

    Rows 0-3 of A (SrcB) are bf16 denormals, which a bf16 MVMUL flushes to zero; scaled by B they give nonzero
    products if kept, so those rows must come out exactly 0. The golden cannot tell flushed rows from kept
    ones at its tolerance, so only the exact-zero check guards them. Rows 4-7 are normal and check the rest
    of the product.
    """
    M, kt = 8, 2
    K, N = kt * DEFAULT_TILE_R_DIM, ct * DEFAULT_TILE_C_DIM
    denormal_rows = 4

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.float32) / OPERAND_SCALE
    torch_a[:denormal_rows] = (
        BF16_DENORMAL_MIN
        * torch.randint(1, BF16_DENORMAL_MANTISSAS, (denormal_rows, K)).float()
    )
    torch_b = torch.randn((K, N), dtype=torch.float32) * OPERAND_SCALE

    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    res = _run_custom_mm(
        M,
        kt,
        ct,
        formats,
        DestAccumulation.No,
        preserve_zero_flag=True,
        operands=(torch_a, torch_b),
    )
    kept = res[:denormal_rows].float()
    assert (
        torch.count_nonzero(kept) == 0
    ), f"denormal SrcB rows were not flushed: max |value| {kept.abs().max().item():.3e}"
