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
  rt_dim = 1, ct_dim in [1,34] (bounded by DEST capacity), kt_dim in [1,256], LoFi only.

The SplitAcc layouts accumulate the second k-face apart and finalize on the last call,
merging it back; the other layouts need no finalization. Either way DEST ends up holding the
plain product. The result is packed as ``ct_dim`` output tiles; inside each tile the two
16-col faces (M x 16, row-major) sit contiguously, then pad out to a full 32-row tile.

Only defined lanes are asserted
-------------------------------
The LLK writes ONLY the top ``M`` rows of each output face. The remaining ``32 - M`` rows
of every 32-row DEST tile are undefined, so the reorder below drops the per-tile padding
and compares ONLY the ``M`` defined rows of each tile against the golden ``A @ B``.

Unpack MOP slot splits
----------------------
The unpack MOP covers one k-tile per iteration and spreads its ``ct_dim - 1`` reuses over
one, two or four template slots (llk_unpack_AB_custom_mm.h, _llk_unpack_AB_custom_mm_mop_config_):
A0 alone for ct = 1, A0 + B for ct 2 and 3, A0..A3 for ct 4 to 34. Odd widths are the ones
whose slots differ in length, so the sweeps include odd ct (3, 7, 25) as well as 1, even
widths and both ends of each slot configuration.

Blackhole only: these LLKs live only in the Blackhole tree and cannot run on WH/Quasar.
This test writes a correct-by-construction golden; runtime pass/fail is a BH-card check.
"""

from dataclasses import dataclass

import pytest
import torch
from conftest import blackhole_only, skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import MatmulGolden
from helpers.llk_params import (
    CustomMmDestLayout,
    DestAccumulation,
    DestSync,
    MathFidelity,
)
from helpers.pack import pack_bfp4_b, pack_bfp8_b, pack_bfp16, pack_fp32
from helpers.param_config import input_output_formats, max_dest_tiles, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    CUSTOM_MM_CALLS,
    CUSTOM_MM_LAYOUT,
    DEST_SYNC,
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
    DataFormat.Bfp8_b: lambda tensor: bytes(pack_bfp8_b(tensor)),
    DataFormat.Bfp4_b: lambda tensor: bytes(pack_bfp4_b(tensor)),
}

_BFP_UNPACKERS = {
    DataFormat.Bfp8_b: unpack_bfp8_b,
    DataFormat.Bfp4_b: unpack_bfp4_b,
}


@dataclass
class CUSTOM_MM_PRESERVE_SRC_ZERO_FLAG(TemplateParameter):
    """Math sets the Src zero flag to keep before the custom_mm init, as a preceding copy_init does."""

    preserve: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool CUSTOM_MM_PRESERVE_SRC_ZERO_FLAG = {str(self.preserve).lower()};"


class CustomMMStimuliConfig(StimuliConfig):
    """A into buffer_A (kt*2 faces of [M,16]), B tilized into buffer_B (kt*ct full tiles).

    Mirrors compressed_utils.CompressedStimuliConfig but has no meta / buffer_C. The
    focused odd-K cases also use a uniform BFP4_b or BFP8_b stream for B.
    """

    def __init__(self, kt, ct, in0_format, in1_format, out_format, packed_a, packed_b):
        super().__init__(
            buffer_A=torch.zeros(
                1, dtype=torch.float32
            ),  # placeholder; real bytes below
            stimuli_A_format=in0_format,
            tile_count_A=kt,
            buffer_B=torch.zeros(1, dtype=torch.float32),  # placeholder
            stimuli_B_format=in1_format,
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


def _build_custom_mm(
    M,
    kt,
    ct,
    formats,
    dest_acc,
    torch_a,
    torch_b,
    calls=None,
    preserve_zero_flag=False,
    layout=CustomMmDestLayout.Dense,
    dest_sync=DestSync.Half,
):
    """Build (but do not run) one custom_mm variant; return it with its (M, N) golden."""
    K = kt * DEFAULT_TILE_R_DIM
    N = ct * DEFAULT_TILE_C_DIM
    in0_format = formats.input_format
    in1_format = formats.input_format_B
    out_format = formats.output_format
    in0_packer = _PACKERS[in0_format]
    in1_packer = _PACKERS[in1_format]

    # in0 (A -> SrcB): kt*2 faces of [M, 16], column-face order along K, contiguous.
    packed_a = b""
    for i in range(kt * 2):
        packed_a += in0_packer(torch_a[:, i * FACE_C_DIM : (i + 1) * FACE_C_DIM])

    # in1 (B -> SrcA): kt*ct full [32,32] tiles, k-major / c-minor -- the order the SrcA
    # CFGSHIFTMASK walk reads them (read_transposed=false: contiguous tiles).
    packed_b = b""
    golden_b = torch_b.clone()
    for r in range(kt):
        for c in range(ct):
            blk = torch_b[
                r * DEFAULT_TILE_R_DIM : (r + 1) * DEFAULT_TILE_R_DIM,
                c * DEFAULT_TILE_C_DIM : (c + 1) * DEFAULT_TILE_C_DIM,
            ]
            # Face-major (tilized) layout inside each 32x32 tile. Tilize in the input
            # format so the returned dtype matches the packer (pack_fp32 rejects a
            # bfloat16 tensor, which is tilize's default Float16_b dtype).
            tiled_blk = tilize(
                blk.reshape(-1),
                stimuli_format=in1_format,
                tile_dimensions=[DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM],
            )
            packed_tile = in1_packer(tiled_blk)
            packed_b += packed_tile
            if in1_format in _BFP_UNPACKERS:
                golden_b[
                    r * DEFAULT_TILE_R_DIM : (r + 1) * DEFAULT_TILE_R_DIM,
                    c * DEFAULT_TILE_C_DIM : (c + 1) * DEFAULT_TILE_C_DIM,
                ] = untilize(
                    _BFP_UNPACKERS[in1_format](packed_tile),
                    tile_dimensions=[DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM],
                ).reshape(
                    DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM
                )

    # Golden: plain A @ B (LoFi), row-major, NOT tilized -- the device-side layout is
    # undone in _device_rows. Instantiate MatmulGolden directly (not via
    # get_golden_generator) so the compile-producer's dummy generator does not break the
    # narrow-M reshape.
    golden = MatmulGolden()(
        torch_a,
        golden_b,
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
            calls if calls is not None else CUSTOM_MM_CALLS(),
            CUSTOM_MM_PRESERVE_SRC_ZERO_FLAG(preserve_zero_flag),
            CUSTOM_MM_LAYOUT(layout),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[
            # Result / in0 use 2 faces (M x 16 each); in1 (B) uses 4 full faces.
            NUM_FACES(num_faces=2, num_faces_A=2, num_faces_B=4),
            IN_FACE_DIMS(in0_face_r_dim=M),
        ],
        variant_stimuli=CustomMMStimuliConfig(
            kt, ct, in0_format, in1_format, out_format, packed_a, packed_b
        ),
        dest_acc=dest_acc,
    )
    return configuration, golden


def _device_rows(res, M, ct, out_format):
    """Undo the device result layout into row-major (M, N).

    The device packs ct tiles; within a tile the two 16-col faces (M x FACE_C_DIM, row-major)
    sit contiguously, padded out to a full 32-row tile. Drop the per-tile padding
    (span = len // ct) and reorder to row-major (M, N). Only the M defined rows survive.
    """
    torch_out = torch.float32 if out_format == DataFormat.Float32 else torch.bfloat16
    res_tensor = torch.as_tensor(res, dtype=torch_out)
    faces_per_tile = DEFAULT_TILE_C_DIM // FACE_C_DIM
    per_tile = res_tensor.reshape(ct, -1)[:, : M * DEFAULT_TILE_C_DIM]
    return (
        per_tile.reshape(ct, faces_per_tile, M, FACE_C_DIM)
        .permute(2, 0, 1, 3)
        .reshape(M, ct * DEFAULT_TILE_C_DIM)
    )


def _seeded_operands(M, kt, ct):
    torch.manual_seed(0)
    torch_a = torch.randn((M, kt * DEFAULT_TILE_R_DIM), dtype=torch.float32)
    torch_b = torch.randn(
        (kt * DEFAULT_TILE_R_DIM, ct * DEFAULT_TILE_C_DIM), dtype=torch.float32
    )
    return torch_a, torch_b


def _run_custom_mm(
    M,
    kt,
    ct,
    formats,
    dest_acc,
    calls=CUSTOM_MM_CALLS(),
    preserve_zero_flag=False,
    operands=None,
    layout=CustomMmDestLayout.Dense,
):
    """Run custom_mm and check it against the golden; return the (M, N) result.

    ``operands`` optionally supplies ``(A, B)`` instead of the seeded random operands.
    """
    torch_a, torch_b = operands if operands is not None else _seeded_operands(M, kt, ct)
    out_format = formats.output_format
    configuration, golden = _build_custom_mm(
        M,
        kt,
        ct,
        formats,
        dest_acc,
        torch_a,
        torch_b,
        calls,
        preserve_zero_flag,
        layout,
    )
    res_tensor = _device_rows(configuration.run().result, M, ct, out_format)

    torch_out = torch.float32 if out_format == DataFormat.Float32 else torch.bfloat16
    golden = golden.reshape(M, ct * DEFAULT_TILE_C_DIM).to(torch_out)

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
    return res_tensor


# LoFi-only per the header ("fidelity: LoFi only").
# kt = 2 accumulates over several unpack MOP iterations; kt = 1 is covered by the compact sweep.
KT_DIMS = [2]

# Dense layout, 32 DEST rows per output tile: 16 tiles fit a 16-bit SyncHalf DEST (512 rows), 8 a 32-bit one
# (256 rows). 1 is the one-slot unpack MOP, 2 and 3 the two-slot ones, 4 the narrowest four-slot one and 7 an uneven
# four-slot split.
CT_DIMS_16BIT = [1, 2, 3, 4, 7, 16]
CT_DIMS_32BIT = [1, 3, 4, 7]


@blackhole_only
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    M=list(SUPPORTED_M),
    kt=KT_DIMS,
    ct=CT_DIMS_16BIT,
)
def test_custom_mm(formats, M, kt, ct):
    _run_custom_mm(M, kt, ct, formats, DestAccumulation.No)


@blackhole_only
@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    M=list(SUPPORTED_M),
    kt=KT_DIMS,
    ct=CT_DIMS_32BIT,
)
def test_custom_mm_fp32_dest(formats, M, kt, ct):
    _run_custom_mm(M, kt, ct, formats, DestAccumulation.Yes)


# A Float16_b output from a 32-bit DEST packs from a Float16_b source. The packer counts its DEST strides in pack
# source datums, not DEST datums, so the layout strides must follow the pack source.
NARROW_PACK_SRC_FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
]


@blackhole_only
@parametrize(
    formats=NARROW_PACK_SRC_FORMATS,
    M=[1, 8],
    kt=KT_DIMS,
    ct=[1, 4, 7],
)
def test_custom_mm_fp32_dest_narrow_pack_src(formats, M, kt, ct):
    _run_custom_mm(M, kt, ct, formats, DestAccumulation.Yes)


# The SplitAcc layouts accumulate the second k-face apart and finalize on the last call. kt = 1 finalizes with no
# MOP iteration before it; the widths cover the one-, two- and four-slot unpack MOPs and fit the Default layout's
# 8 tiles.
SPLIT_ACC_LAYOUTS = [
    CustomMmDestLayout.DefaultSplitAcc,
    CustomMmDestLayout.DenseSplitAcc,
]


@blackhole_only
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    layout=SPLIT_ACC_LAYOUTS,
    M=[1, 8],
    kt=[1, 2, 9],
    ct=[1, 3, 7],
)
def test_custom_mm_split_acc(formats, layout, M, kt, ct):
    _run_custom_mm(M, kt, ct, formats, DestAccumulation.No, layout=layout)


@blackhole_only
@parametrize(layout=SPLIT_ACC_LAYOUTS, ct=[1, 2])
def test_custom_mm_split_acc_multi_call(layout, ct):
    """Only the last of several calls into the same tiles finalizes."""
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    _run_custom_mm(
        8,
        8,
        ct,
        formats,
        DestAccumulation.No,
        CUSTOM_MM_CALLS(num_calls=4),
        layout=layout,
    )


# kt above CUSTOM_MM_MAX_MOP_ITERATIONS (128) splits the unpack over several TT_MOPs, with the one-slot MOP at ct = 1
# and the four-slot one at ct = 4 (Bfp8_b weights keep that B within L1). 256 is the widest kt_dim.
LONG_K_CASES = [
    pytest.param(
        8,
        200,
        1,
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        id="k200-ct1",
    ),
    pytest.param(
        1,
        256,
        1,
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        id="k256-ct1",
    ),
    pytest.param(
        8,
        130,
        4,
        InputOutputFormat(
            DataFormat.Float16_b, DataFormat.Float16_b, DataFormat.Bfp8_b
        ),
        id="k130-ct4-bfp8",
    ),
]


@blackhole_only
@pytest.mark.parametrize("M,kt,ct,formats", LONG_K_CASES)
def test_custom_mm_long_k(formats, M, kt, ct):
    """kt_dim past one TT_MOP's 128 iterations, at the one- and four-slot unpack MOP widths."""
    _run_custom_mm(M, kt, ct, formats, DestAccumulation.No)


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
        2,
        InputOutputFormat(
            DataFormat.Float16_b, DataFormat.Float16_b, DataFormat.Bfp4_b
        ),
        id="k9-ct2-bfp4-post1",
    ),
]


@blackhole_only
@pytest.mark.parametrize("M,kt,ct,formats", ODD_K_CASES)
def test_custom_mm_odd_k(formats, M, kt, ct):
    """The post1 unpack tuning (Bfp4_b weights) at the one- and two-slot MOP widths, on a single and an odd kt."""
    _run_custom_mm(M, kt, ct, formats, DestAccumulation.No)


# Compact layout: 16 DEST rows per output tile, so a 16-bit SyncHalf DEST (512 rows) holds 32 tiles and a
# 32-bit one (256 rows) 16. 3 is the widest two-slot unpack MOP and 4 the narrowest four-slot one; 25 is an uneven
# four-slot split and 32 the widest DEST-legal width.
COMPACT_CT_DIMS_16BIT = [1, 3, 4, 7, 25, 32]
COMPACT_CT_DIMS_32BIT = [1, 16]


def _run_compact_against_default(M, kt, ct, formats, dest_acc, dest_sync=DestSync.Half):
    """Run custom_mm in the Compact layout and require it bit-identical to Default-layout runs.

    The reference packs the same B in column slices narrow enough for the default 64-row
    layout (8 tiles in a 16-bit SyncHalf DEST, 4 in a 32-bit one); only the compact run
    takes ``dest_sync``. Both do the same MVMULs
    in the same order, so any difference is a layout or unpack-walk error. A tolerance
    against the fp32 golden cannot tell those apart from bf16 accumulation rounding at these
    widths. A PCC check against the golden still guards the case of both being wrong alike.
    """
    out_format = formats.output_format
    torch_a, torch_b = _seeded_operands(M, kt, ct)
    max_default_ct = max_dest_tiles(dest_acc)

    compact, golden = _build_custom_mm(
        M,
        kt,
        ct,
        formats,
        dest_acc,
        torch_a,
        torch_b,
        layout=CustomMmDestLayout.Compact,
        dest_sync=dest_sync,
    )
    slices = []
    for c0 in range(0, ct, max_default_ct):
        width = min(max_default_ct, ct - c0)
        cols = slice(c0 * DEFAULT_TILE_C_DIM, (c0 + width) * DEFAULT_TILE_C_DIM)
        reference, _ = _build_custom_mm(
            M,
            kt,
            width,
            formats,
            dest_acc,
            torch_a,
            torch_b[:, cols],
            layout=CustomMmDestLayout.Default,
        )
        slices.append((width, reference))

    # Build every variant first: in the compile-producer phase the first run() skips the test.
    compact.prepare()
    for _, reference in slices:
        reference.prepare()

    result = _device_rows(compact.run().result, M, ct, out_format)
    expected = torch.cat(
        [
            _device_rows(reference.run().result, M, width, out_format)
            for width, reference in slices
        ],
        dim=1,
    )

    bits = torch.int32 if out_format == DataFormat.Float32 else torch.int16
    mismatched = result.view(bits) != expected.view(bits)
    assert not mismatched.any(), (
        f"compact custom_mm differs from the default layout in {int(mismatched.sum())} of "
        f"{mismatched.numel()} outputs (first at {mismatched.nonzero()[0].tolist()}) for "
        f"M={M} kt={kt} ct={ct} formats={formats}"
    )

    pcc = torch.corrcoef(
        torch.stack([result.float().flatten(), golden.float().flatten()])
    )[0, 1]
    assert pcc > 0.999, f"compact custom_mm PCC {pcc:.6f} against the golden"


@blackhole_only
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    M=[1, 8],
    kt=[1, 3],
    ct=COMPACT_CT_DIMS_16BIT,
)
def test_custom_mm_compact(formats, M, kt, ct):
    _run_compact_against_default(M, kt, ct, formats, DestAccumulation.No)


@blackhole_only
@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    M=[1, 8],
    kt=[2],
    ct=COMPACT_CT_DIMS_32BIT,
)
def test_custom_mm_compact_fp32_dest(formats, M, kt, ct):
    _run_compact_against_default(M, kt, ct, formats, DestAccumulation.Yes)


@blackhole_only
@parametrize(
    formats=NARROW_PACK_SRC_FORMATS,
    M=[1, 8],
    kt=[2],
    ct=COMPACT_CT_DIMS_32BIT,
)
def test_custom_mm_compact_fp32_dest_narrow_pack_src(formats, M, kt, ct):
    """Compact strides from a Float16_b pack source on a 32-bit DEST, half the DEST row width."""
    _run_compact_against_default(M, kt, ct, formats, DestAccumulation.Yes)


# SyncFull gives a 16-bit DEST 1024 rows, room for 34 compact tiles: the widest unpack MOP, whose four slots carry
# 8, 8, 8 and 8 reuses at ct = 33 and 8, 8, 8 and 9 at ct = 34.
COMPACT_SYNC_FULL_CT_DIMS = [33, 34]


@blackhole_only
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    M=[1, 8],
    kt=[2],
    ct=COMPACT_SYNC_FULL_CT_DIMS,
)
def test_custom_mm_compact_sync_full(formats, M, kt, ct):
    _run_compact_against_default(
        M, kt, ct, formats, DestAccumulation.No, dest_sync=DestSync.Full
    )


COMPACT_WIDE_BFP_CASES = [
    pytest.param(
        8,
        3,
        3,
        InputOutputFormat(
            DataFormat.Float16_b, DataFormat.Float16_b, DataFormat.Bfp4_b
        ),
        id="k3-ct3-bfp4-post1",
    ),
    pytest.param(
        8,
        2,
        32,
        InputOutputFormat(
            DataFormat.Float16_b, DataFormat.Float16_b, DataFormat.Bfp4_b
        ),
        id="k2-ct32-bfp4-post1",
    ),
]


@blackhole_only
@pytest.mark.parametrize("M,kt,ct,formats", COMPACT_WIDE_BFP_CASES)
def test_custom_mm_compact_wide_bfp(formats, M, kt, ct):
    """The post1 unpack tuning (Bfp4_b weights) at the widest two-slot MOP and the widest four-slot one."""
    _run_compact_against_default(M, kt, ct, formats, DestAccumulation.No)


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
