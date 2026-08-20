# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Plain (uncompressed) custom_mm matmul -- Blackhole only.

The plain custom_mm family was promoted to ``experimental/`` by tt-metal #52727 and had
**no test at all**: all four entry points were uncalled anywhere under ``tests/sources``.
The asymmetry is easy to miss -- ``test_matmul_custom_compressed.py`` covers the
*compressed* variant, so the harder path was exercised and the simpler one was not, and
``test_matmul_custom.py`` drives ``llk_math_matmul_custom_no_mop.h``, an unrelated family.
This closes the "no coverage at all" part; what it still leaves uncovered -- transpose,
split_acc/finalize, and the top of the documented kt_dim range -- is listed in the header
comment of ``sources/matmul_custom_mm_test.cpp``.

The op is not a general matmul. Operand a is a full 32x32 4-face tile, operand b a narrow
``[{1,2,4,8}, 32]`` tile using only its top two faces, so the computation is
``(M x K) @ (K x N)`` with M in {1,2,4,8} -- a narrow activation row times a full weight
block -- accumulating over ``kt_dim`` K-tiles into ``ct_dim`` output tiles, all from a
single call per thread.

Sweep, and why these bounds
---------------------------
``kt_dim`` is **even only**, which is a hard constraint rather than a style choice:
``_llk_unpack_AB_custom_mm_run_`` issues ``TT_MOP(0, (kt_dim / 2) - 1, 0)``, so an odd
kt_dim would run the wrong number of MOP iterations. The doc tables say "even number from
2 to 256" and this is where that comes from.

``ct_dim`` is swept including the **odd middle values** the doc tables leave unverified
(they claim any integer 1..16). This test reaches 7, which is the largest odd value that
fits: the ct output tiles are all resident in DEST at once, and DEST half-sync holds 8
bf16 tiles, so ct_dim > 8 is not reachable from a single call in this configuration at all.
That is worth knowing against the documented 1..16 -- the upper half of that range needs
either DstSync::SyncFull or a caller that splits the block, and neither is what the tables
imply.

Not covered yet, deliberately: ``transpose``, and ``split_acc`` / ``finalize`` -- both of
which ARE forwarded on this family, unlike the compressed one, and so are the obvious next
increment. Establishing the family under test is the expensive step; widening it is not.

Golden and tolerance follow the compressed sibling (``helpers/compressed_utils.py``), for
the same reasons: the device packs ct tiles whose two 16-column faces sit contiguously
followed by padding, so the result is reordered to row-major before comparing, and the
absolute floor is scaled by kt because the K-deep sum accumulates in a bf16 dest.
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
    CUSTOM_MM_FLAGS,
    IN_FACE_DIMS,
    NUM_FACES,
)
from helpers.tile_constants import DEFAULT_TILE_C_DIM, FACE_C_DIM
from helpers.tilize_untilize import tilize
from helpers.utils import passed_test
from ttexalens.tt_exalens_lib import write_to_device

pytestmark = [skip_for_wormhole, skip_for_quasar]

# Narrow activation rows the family supports -- operand b carries only its top two faces.
SUPPORTED_M = [1, 2, 4, 8]

# kt_dim must be even (TT_MOP(0, (kt_dim / 2) - 1, 0)).
KT_DIMS = [2, 4]

# ct output tiles are all live in DEST, and half-sync holds 8 bf16 tiles. 3 and 7 are the
# odd middle values the doc tables never pinned.
CT_DIMS = [1, 3, 7, 8]

FLOAT16B_DEFAULT_ATOL = 0.05
ACC_ATOL_PER_KT = 0.005


class _PlainStimuli(StimuliConfig):
    """Raw-bytes stimuli for the plain family: narrow A rows, full B tiles.

    Mirrors ``CompressedStimuliConfig`` -- the harness's normal tensor path assumes one
    tile layout for both operands, and this op does not have that -- but both buffers are
    plain Float16_b, with no metadata buffer since the plain kernel reads none.
    """

    def __init__(self, kt, ct, packed_a, packed_b):
        super().__init__(
            buffer_A=torch.zeros(1, dtype=torch.float32),  # placeholder, see write()
            stimuli_A_format=DataFormat.Float16_b,
            tile_count_A=kt,
            buffer_B=torch.zeros(1, dtype=torch.float32),  # placeholder, see write()
            stimuli_B_format=DataFormat.Float16_b,
            tile_count_B=kt * ct,
            stimuli_res_format=DataFormat.Float16_b,
            tile_count_res=ct,
        )
        self.packed_a = packed_a
        self.packed_b = packed_b

    def write(self, location: str = "0,0"):
        write_to_device(location, self.buf_a_addr, self.packed_a)
        write_to_device(location, self.buf_b_addr, self.packed_b)


def _run_custom_mm(M, kt, ct, transpose=False, split_acc=False, finalize=False):
    """Drive one variant and return (golden, device) as (M, N) bf16 tensors."""
    K = kt * DEFAULT_TILE_C_DIM
    N = ct * DEFAULT_TILE_C_DIM

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N), dtype=torch.bfloat16)

    # Operand b (narrow): M rows split into kt*2 faces of FACE_C_DIM columns, exactly as
    # the compressed harness packs it -- the unpacker takes one face per instruction.
    packed_a = b""
    for i in range(kt * 2):
        packed_a += pack_bfp16(torch_a[:, i * FACE_C_DIM : (i + 1) * FACE_C_DIM])

    # Operand a (full tiles): kt*ct tilized 32x32 tiles, row-major over (k, c).
    packed_b = b""
    for k in range(kt):
        for c in range(ct):
            blk = torch_b[
                k * DEFAULT_TILE_C_DIM : (k + 1) * DEFAULT_TILE_C_DIM,
                c * DEFAULT_TILE_C_DIM : (c + 1) * DEFAULT_TILE_C_DIM,
            ]
            packed_b += pack_bfp16(
                tilize(
                    blk.reshape(-1),
                    tile_dimensions=[DEFAULT_TILE_C_DIM, DEFAULT_TILE_C_DIM],
                )
            )

    # MatmulGolden instantiated directly rather than via get_golden_generator: under
    # --compile-producer the harness swaps in a DummyGoldenGenerator returning zeros(1024),
    # which would break the reshape for narrow M. Same reason as the compressed helper.
    # transpose acts on in1 -- the full tiles -- and does so per TILE, so the golden operand
    # is torch_b with each 32x32 tile transposed in place, not torch_b.T.
    golden_b = torch_b
    if transpose:
        golden_b = torch_b.clone()
        for k in range(kt):
            for c in range(ct):
                rows = slice(k * DEFAULT_TILE_C_DIM, (k + 1) * DEFAULT_TILE_C_DIM)
                cols = slice(c * DEFAULT_TILE_C_DIM, (c + 1) * DEFAULT_TILE_C_DIM)
                golden_b[rows, cols] = torch_b[rows, cols].transpose(0, 1)

    golden = MatmulGolden()(
        torch_a,
        golden_b,
        DataFormat.Float16_b,
        MathFidelity.LoFi,
        input_A_dimensions=[M, K],
        input_B_dimensions=[K, N],
        tilize=False,
        input_A_format=DataFormat.Float16_b,
        input_B_format=DataFormat.Float16_b,
    ).reshape(M, N)

    configuration = TestConfig(
        "sources/matmul_custom_mm_test.cpp",
        InputOutputFormat(
            input_format=DataFormat.Float16_b,
            output_format=DataFormat.Float16_b,
        ),
        templates=[
            CRK_TILE_DIMM(c_dimm=ct, r_dimm=1, k_dimm=kt),
            CUSTOM_MM_FLAGS(
                transpose=transpose, split_acc=split_acc, finalize=finalize
            ),
        ],
        runtimes=[
            NUM_FACES(num_faces=2, num_faces_A=2, num_faces_B=4),
            IN_FACE_DIMS(in0_face_r_dim=M),
        ],
        variant_stimuli=_PlainStimuli(kt, ct, packed_a, packed_b),
        dest_acc=DestAccumulation.No,
    )

    res = configuration.run().result

    # Device packs ct tiles; within a tile the two 16-column faces (M x FACE_C_DIM,
    # row-major) sit contiguously, then padding out to a full 32-row tile. Drop the
    # padding and reorder to row-major (M, N).
    res_tensor = torch.as_tensor(res, dtype=torch.bfloat16)
    faces_per_tile = DEFAULT_TILE_C_DIM // FACE_C_DIM
    per_tile = res_tensor.reshape(ct, -1)[:, : M * DEFAULT_TILE_C_DIM]
    res_tensor = (
        per_tile.reshape(ct, faces_per_tile, M, FACE_C_DIM)
        .permute(2, 0, 1, 3)
        .reshape(M, N)
    )

    return golden, res_tensor


def _assert_matches(golden, device, kt, label):
    # K-aware absolute floor: the LoFi MVMUL accumulates the K-deep sum in a bf16 dest, so
    # noise grows ~linearly per K-tile. Same calibration as the compressed sibling.
    active = golden.abs()
    active = active[active > 0]
    mean_active = active.mean().item() if active.numel() else 0.0
    acc_atol = max(FLOAT16B_DEFAULT_ATOL, ACC_ATOL_PER_KT * kt * mean_active)

    assert passed_test(
        golden,
        device,
        DataFormat.Float16_b,
        custom_atol=acc_atol,
        print_pcc=True,
    ), f"plain custom_mm failed for {label}"


@parametrize(
    M=SUPPORTED_M,
    kt=KT_DIMS,
    ct=CT_DIMS,
)
def test_matmul_custom_mm(M, kt, ct):
    """(M x K) @ (K x N) through the plain custom_mm family, one call per thread."""
    golden, device = _run_custom_mm(M, kt, ct)
    _assert_matches(golden, device, kt, f"M={M}, kt={kt}, ct={ct}")


@parametrize(
    M=SUPPORTED_M,
    kt=KT_DIMS,
    ct=[1, 8],
)
def test_matmul_custom_mm_transpose(M, kt, ct):
    """``transpose=True``: in1's tiles arrive transposed, one tile at a time.

    The golden transposes each 32x32 tile of the full operand in place rather than
    transposing the whole [K, N] matrix -- the flag acts on what SrcA holds per MVMUL, and
    getting that wrong is the easy mistake here. ``ct`` is trimmed to its ends because the
    axis under test is the flag, not the block width, which the main sweep already covers.
    """
    golden, device = _run_custom_mm(M, kt, ct, transpose=True)
    _assert_matches(golden, device, kt, f"transpose, M={M}, kt={kt}, ct={ct}")


@parametrize(
    M=SUPPORTED_M,
    kt=KT_DIMS,
    ct=[1, 8],
)
def test_matmul_custom_mm_split_acc_finalize(M, kt, ct):
    """``split_acc`` + ``finalize`` must reproduce the plain result exactly.

    That equality *is* the specification: split_acc scatters the inner dimension's partials
    to Dest rows 8/24 instead of accumulating them in place, and finalize is the replay that
    ELWADDs them back. So the same golden as the plain sweep, and a failure here means either
    the scatter or the merge is wrong -- with no third possibility, since nothing else in the
    call changes.

    Only the paired form is swept. ``finalize`` without ``split_acc`` would merge rows that
    are not partials, and ``CUSTOM_MM_FLAGS`` rejects that combination at build time rather
    than letting a test assert on it; ``split_acc`` without ``finalize`` leaves the partials
    unmerged in Dest, which no golden describes.
    """
    golden, device = _run_custom_mm(M, kt, ct, split_acc=True, finalize=True)
    _assert_matches(golden, device, kt, f"split_acc+finalize, M={M}, kt={kt}, ct={ct}")
