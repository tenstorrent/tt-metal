# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Plain (uncompressed) custom_mm matmul -- Blackhole only.

The plain custom_mm family was promoted to ``experimental/`` by tt-metal #52727 and had
**no test at all**: all four entry points were uncalled anywhere under ``tests/sources``.
The asymmetry is easy to miss -- ``test_matmul_custom_compressed.py`` covers the
*compressed* variant, so the harder path was exercised and the simpler one was not, and
``test_matmul_custom.py`` drives ``llk_math_matmul_custom_no_mop.h``, an unrelated family.
Tracked as A1 in ``REMAINING_WORK.md``; this closes the "no coverage at all" part of it.

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


@parametrize(
    M=SUPPORTED_M,
    kt=KT_DIMS,
    ct=CT_DIMS,
)
def test_matmul_custom_mm(M, kt, ct):
    """(M x K) @ (K x N) through the plain custom_mm family, one call per thread."""
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
        "sources/matmul_custom_mm_test.cpp",
        InputOutputFormat(
            input_format=DataFormat.Float16_b,
            output_format=DataFormat.Float16_b,
        ),
        templates=[
            CRK_TILE_DIMM(c_dimm=ct, r_dimm=1, k_dimm=kt),
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

    # K-aware absolute floor: the LoFi MVMUL accumulates the K-deep sum in a bf16 dest, so
    # noise grows ~linearly per K-tile. Same calibration as the compressed sibling.
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
    ), f"plain custom_mm failed for M={M}, kt={kt}, ct={ct} (K={K}, N={N})"
