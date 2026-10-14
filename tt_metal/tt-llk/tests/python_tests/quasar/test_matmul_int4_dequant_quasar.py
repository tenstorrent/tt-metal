# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Group-wise dequant of a wint4_aint8 linear, run in parallel with the matmul:
# the FPU computes each K group's int8 x int4 partial into Int32 Dest, the TRISC3
# SFPU rewrites it as float(partial) * s_w[g, n], PACK0 sums the groups in fp32 in
# L1, and a final TRISC3 pass multiplies by the per-token scale s_a[m].

import pytest
import torch
from helpers.chip_architecture import is_4row_arch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    PerfRunType,
    Transpose,
    format_dict,
)
from helpers.param_config import DEST_SYNC_TILE_LIMITS, parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    ENABLE_2X_FORMAT,
    ENABLE_DIRECT_INDEXING,
    GROUP_KT_DIM,
    IMPLIED_MATH_FORMAT,
    IN_FACE_DIMS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from helpers.tile_shape import construct_tile_shape
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test
from quasar.test_matmul_int4_quasar import (
    MATMUL_INT4_FORMATS,
    TILE,
    generate_int_matrix,
)

TILE_ELEMENTS = TILE * TILE

# (ct, rt, kt, group_kt): group_kt K tiles share one scale. group_kt == kt is per-channel;
# group_kt == 4 is the group-128 case the requester uses.
MATMUL_GROUP_DIMS = [
    (1, 1, 1, 1),
    (1, 1, 4, 4),
    (1, 1, 4, 1),
    (1, 1, 8, 4),
    (2, 2, 8, 4),
    (4, 1, 4, 2),
    (1, 2, 8, 2),
]


def matmul_group_dims(dest_sync):
    # 32-bit Dest halves the tile capacity.
    max_tiles = DEST_SYNC_TILE_LIMITS[dest_sync] // 2
    return [dims for dims in MATMUL_GROUP_DIMS if dims[0] * dims[1] <= max_tiles]


def dequant_register_format_hints():
    # Int8_2x exists only on the 4-row variant.
    return [None, DataFormat.Int8_2x] if is_4row_arch() else [None]


def _pad_to_tiles(values):
    # The stimuli writer works in whole tiles.
    pad = (-values.numel()) % TILE_ELEMENTS
    return torch.cat([values, torch.zeros(pad, dtype=values.dtype)])


@pytest.mark.quasar
@parametrize(
    format=MATMUL_INT4_FORMATS,
    dest_sync_mode=[DestSync.Half],
    matmul_dims=lambda dest_sync_mode: matmul_group_dims(dest_sync_mode),
    register_format_hint=dequant_register_format_hints(),
)
def test_matmul_int4_dequant_quasar(
    format,
    dest_sync_mode,
    matmul_dims,
    register_format_hint,
    run_types=(PerfRunType.L1_TO_L1,),
    loop_factor=1,
    *,
    is_perf=False,
    perf_report=None,
):
    format = InputOutputFormat(
        format.input_format,
        DataFormat.Float32,
        input_format_B=format.input_format_B,
        register_format_hint=register_format_hint,
    )
    ct_dim, rt_dim, kt_dim, group_kt_dim = matmul_dims
    dest_acc = DestAccumulation.Yes
    tile_dimensions = [TILE, TILE]
    tile_shape = construct_tile_shape(tile_dimensions)

    # A = activations (M x K), B = weights (K x N).
    input_A_dimensions = [rt_dim * TILE, kt_dim * TILE]
    input_B_dimensions = [kt_dim * TILE, ct_dim * TILE]
    output_dimensions = [rt_dim * TILE, ct_dim * TILE]
    tile_cnt_A = rt_dim * kt_dim
    tile_cnt_B = kt_dim * ct_dim
    output_tile_cnt = rt_dim * ct_dim

    src_A = generate_int_matrix(format.input_format, input_A_dimensions, seed=1)
    src_B = generate_int_matrix(format.input_format_B, input_B_dimensions, seed=2)

    # Group-wise scales: out[m, n] = s_a[m] * sum_g s_w[g, n] * partial_g[m, n].
    # The device sums float(partial_g) * s_w[g, n] in fp32 in group order (L1 acc), then
    # multiplies by s_a once, so the reference does the same.
    num_groups = kt_dim // group_kt_dim
    group_k = group_kt_dim * TILE
    generator = torch.Generator().manual_seed(3)
    s_a = torch.empty(output_dimensions[0]).uniform_(0.005, 0.02, generator=generator)
    s_w = torch.empty(num_groups, output_dimensions[1]).uniform_(
        0.01, 0.05, generator=generator
    )
    A = src_A.to(torch.int64).view(input_A_dimensions)
    B = src_B.to(torch.int64).view(input_B_dimensions)
    acc = torch.zeros(output_dimensions, dtype=torch.float32)
    error_bound = torch.zeros(output_dimensions, dtype=torch.float32)
    for g in range(num_groups):
        ks = slice(g * group_k, (g + 1) * group_k)
        partial = torch.matmul(A[:, ks], B[ks, :]).to(torch.float32)
        term = partial * s_w[g].unsqueeze(0)
        acc = term if g == 0 else acc + term
        error_bound += term.abs()
    golden = acc * s_a.unsqueeze(1)
    error_bound *= s_a.unsqueeze(1)

    # One SrcS slice (4 rows x 16 cols) per (group, tile column, 16-column half):
    # the 16 column scales repeated down the 4 rows.
    w_scale_slices = (
        s_w.view(num_groups, ct_dim, 2, 1, 16).expand(-1, -1, -1, 4, -1).flatten()
    )
    w_scale_slices = _pad_to_tiles(w_scale_slices)
    # s_a expanded to one full tile per output tile.
    a_scale_tiles = tilize_block(
        s_a.view(rt_dim * TILE, 1).expand(-1, ct_dim * TILE).flatten(),
        dimensions=output_dimensions,
        stimuli_format=DataFormat.Float32,
        tile_dimensions=tile_dimensions,
    ).flatten()
    error_bound = tilize_block(
        error_bound.flatten(),
        dimensions=output_dimensions,
        stimuli_format=DataFormat.Float32,
        tile_dimensions=tile_dimensions,
    ).flatten()

    golden_tensor = tilize_block(
        golden.flatten(),
        dimensions=output_dimensions,
        stimuli_format=format.output_format,
        tile_dimensions=tile_dimensions,
    ).flatten()

    tilized_A = tilize_block(
        src_A,
        dimensions=input_A_dimensions,
        stimuli_format=format.input_format,
        tile_dimensions=tile_dimensions,
    ).flatten()
    tilized_B = tilize_block(
        src_B,
        dimensions=input_B_dimensions,
        stimuli_format=format.input_format_B,
        tile_dimensions=tile_dimensions,
    ).flatten()

    num_faces = tile_shape.total_num_faces()
    test_config_kwargs = {
        "test_name": "sources/quasar/matmul_int4_dequant_quasar_test.cpp",
        "formats": format,
        "templates": [
            MATH_FIDELITY(MathFidelity.LoFi),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            ENABLE_2X_FORMAT(format.register_format_hint is not None),
            ENABLE_DIRECT_INDEXING(False),
            DEST_SYNC(dest_sync_mode),
            UNPACK_TRANS_FACES(Transpose.No),
        ],
        "runtimes": [
            CRK_TILE_DIMM(ct_dim, rt_dim, kt_dim),
            TILE_COUNT(output_tile_cnt * kt_dim),
            NUM_FACES(num_faces, num_faces, num_faces),
            IN_FACE_DIMS(
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
            ),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim, tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim, tile_shape.num_faces_c_dim),
            LOOP_FACTOR(loop_factor),
            GROUP_KT_DIM(group_kt_dim),
        ],
        "variant_stimuli": StimuliConfig(
            tilized_A,
            format.input_format,
            tilized_B,
            format.input_format_B,
            format.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=output_tile_cnt,
            buffer_C=w_scale_slices,
            stimuli_C_format=DataFormat.Float32,
            tile_count_C=w_scale_slices.numel() // TILE_ELEMENTS,
            # Scratch tile that closes the SrcS bank cycle after each scale load.
            buffer_S=torch.zeros(TILE_ELEMENTS),
            stimuli_S_format=DataFormat.Float32,
            tile_count_S=1,
            buffer_T=a_scale_tiles,
            stimuli_T_format=DataFormat.Float32,
            tile_count_T=output_tile_cnt,
            num_faces=num_faces,
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        "unpack_to_dest": False,
        "dest_acc": dest_acc,
    }
    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")
    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=list(run_types),
        test_config_kwargs=test_config_kwargs,
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[format.output_format])

    # Every int32 -> fp32 cast is exact (|partial| < 2^24); each scale multiply and each
    # fp32 accumulation rounds by <= 1/2 ulp, so bound the error by a few ulp of the sum of
    # |terms| (robust to cancellation) - far tighter than the Float32 default tolerance.
    tolerance = 2**-21 * (kt_dim // group_kt_dim + 2) * error_bound
    max_excess = ((res_tensor - golden_tensor).abs() - tolerance).max()
    assert (
        max_excess <= 0
    ), f"Dequant mismatch beyond ulp bound by {max_excess}, max abs err {(res_tensor - golden_tensor).abs().max()}"
    assert passed_test(
        golden_tensor, res_tensor, format.output_format
    ), "Assert against golden failed"
