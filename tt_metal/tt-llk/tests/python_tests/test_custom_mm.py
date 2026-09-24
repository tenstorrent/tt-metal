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
  rt_dim = 1, ct_dim in [1,16], kt_dim even in [2,256], LoFi only.

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

import torch
from conftest import blackhole_only, skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.pack import pack_bfp16, pack_fp32
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import CRK_TILE_DIMM, IN_FACE_DIMS, NUM_FACES
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM, FACE_C_DIM
from helpers.tilize_untilize import tilize
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
}


class CustomMMStimuliConfig(StimuliConfig):
    """A into buffer_A (kt*2 faces of [M,16]), B tilized into buffer_B (kt*ct full tiles).

    Mirrors compressed_utils.CompressedStimuliConfig but plain: no meta / buffer_C, and B
    is a plain (Float16_b or Float32) tile stream rather than a packed BFP one.
    """

    def __init__(self, kt, ct, in_format, out_format, packed_a, packed_b):
        super().__init__(
            buffer_A=torch.zeros(
                1, dtype=torch.float32
            ),  # placeholder; real bytes below
            stimuli_A_format=in_format,
            tile_count_A=kt,
            buffer_B=torch.zeros(1, dtype=torch.float32),  # placeholder
            stimuli_B_format=in_format,
            tile_count_B=kt * ct,
            stimuli_res_format=out_format,
            tile_count_res=ct,
        )
        self.packed_a = packed_a
        self.packed_b = packed_b

    def write(self, location: str = "0,0"):
        from ttexalens.tt_exalens_lib import write_to_device

        write_to_device(location, self.buf_a_addr, self.packed_a)
        write_to_device(location, self.buf_b_addr, self.packed_b)


def _run_custom_mm(M, kt, ct, formats, dest_acc):
    K = kt * DEFAULT_TILE_R_DIM
    N = ct * DEFAULT_TILE_C_DIM
    in_format = formats.input_format
    out_format = formats.output_format
    packer = _PACKERS[in_format]

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.float32)
    torch_b = torch.randn((K, N), dtype=torch.float32)

    # in0 (A -> SrcB): kt*2 faces of [M, 16], column-face order along K, contiguous.
    packed_a = b""
    for i in range(kt * 2):
        packed_a += packer(torch_a[:, i * FACE_C_DIM : (i + 1) * FACE_C_DIM])

    # in1 (B -> SrcA): kt*ct full [32,32] tiles, k-major / c-minor -- the order the SrcA
    # CFGSHIFTMASK walk reads them (read_transposed=false: contiguous tiles).
    packed_b = b""
    for r in range(kt):
        for c in range(ct):
            blk = torch_b[
                r * DEFAULT_TILE_R_DIM : (r + 1) * DEFAULT_TILE_R_DIM,
                c * DEFAULT_TILE_C_DIM : (c + 1) * DEFAULT_TILE_C_DIM,
            ]
            # Face-major (tilized) layout inside each 32x32 tile. Tilize in the input
            # format so the returned dtype matches the packer (pack_fp32 rejects a
            # bfloat16 tensor, which is tilize's default Float16_b dtype).
            packed_b += packer(
                tilize(
                    blk.reshape(-1),
                    stimuli_format=in_format,
                    tile_dimensions=[DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM],
                )
            )

    # Golden: plain A @ B (LoFi), row-major, NOT tilized -- the device-side layout is
    # undone in the result reorder below. Instantiate MatmulGolden directly (not via
    # get_golden_generator) so the compile-producer's dummy generator does not break the
    # narrow-M reshape.
    golden = MatmulGolden()(
        torch_a,
        torch_b,
        out_format,
        MathFidelity.LoFi,
        input_A_dimensions=[M, K],
        input_B_dimensions=[K, N],
        tilize=False,
        input_A_format=in_format,
        input_B_format=in_format,
    ).reshape(M, N)

    configuration = TestConfig(
        "sources/custom_mm_test.cpp",
        formats,
        templates=[
            CRK_TILE_DIMM(c_dimm=ct, r_dimm=1, k_dimm=kt),
        ],
        runtimes=[
            # Result / in0 use 2 faces (M x 16 each); in1 (B) uses 4 full faces.
            NUM_FACES(num_faces=2, num_faces_A=2, num_faces_B=4),
            IN_FACE_DIMS(in0_face_r_dim=M),
        ],
        variant_stimuli=CustomMMStimuliConfig(
            kt, ct, in_format, out_format, packed_a, packed_b
        ),
        dest_acc=dest_acc,
    )

    res = configuration.run().result
    torch_out = torch.float32 if out_format == DataFormat.Float32 else torch.bfloat16
    res_tensor = torch.as_tensor(res, dtype=torch_out)

    # Device packs ct tiles; within a tile the two 16-col faces (M x FACE_C_DIM, row-major)
    # sit contiguously, padded out to a full 32-row tile. Drop the per-tile padding
    # (span = len // ct) and reorder to row-major (M, N). Only the M defined rows survive.
    faces_per_tile = DEFAULT_TILE_C_DIM // FACE_C_DIM
    per_tile = res_tensor.reshape(ct, -1)[:, : M * DEFAULT_TILE_C_DIM]
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

    assert passed_test(
        golden, res_tensor, out_format, custom_atol=custom_atol, print_pcc=True
    ), f"custom_mm matmul failed for M={M} kt={kt} ct={ct} formats={formats}"


# LoFi-only per the header ("fidelity: LoFi only"). Float16_b (16-bit dest) and Float32
# (32-bit dest) cover both DEST widths.
CUSTOM_MM_FORMATS = input_output_formats(
    [DataFormat.Float16_b, DataFormat.Float32], same=True
)

# kt is even in [2,256]; small values keep L1 in budget while still accumulating over K.
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
