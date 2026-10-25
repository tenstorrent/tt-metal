# Local sweep extension (not committed), on perf_math_matmul's kernel: the tile shapes the swept grids never vary
# (in1 32x16, in0 32x16 with in1 16x32, in0 16x32 or r<16 with in1 32x16, an in1 partial face) and tiny in0 tiles
# (face height 1 to 16) with in1 transpose. Bfp4_b only at face height >= 4.
from dataclasses import replace
from itertools import product
import perf_math_matmul as pmm
import pytest
from helpers.format_config import DataFormat, InputOutputFormat, is_dest_acc_needed
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity, StochasticRounding, Transpose
from helpers.matmul_sweep import FaceLayoutConfig, MatmulConfig, generate_tile_dims

F = DataFormat
FMTS = [InputOutputFormat(F.Float16_b, F.Float16_b), InputOutputFormat(F.Bfp8_b, F.Bfp8_b),
        InputOutputFormat(F.Float32, F.Float32), InputOutputFormat(F.Bfp4_b, F.Float16_b)]
# (name, in0 r, in0 c, in1 r, in1 c, faces in0, faces in1, faces out, partial in0, partial in1)
SHAPES = [("in1_32x16", 32, 32, 32, 16, 4, 2, 2, False, False),
          ("in0_32x16_in1_16x32", 32, 16, 16, 32, 2, 2, 4, False, False),
          ("in0_16x32_in1_32x16", 16, 32, 32, 16, 2, 2, 1, False, False),
          ("in0_8x32_in1_32x16", 8, 32, 32, 16, 2, 2, 1, True, False),
          ("in0_32x8_in1_8x32", 32, 8, 8, 32, 2, 2, 4, False, True)]


def cfg(fmt, acc, sync, rt, ct, kt, shape, tr, dst=0):
    _, r0, c0, r1, c1, f0, f1, fo, p0, p1 = shape
    td = generate_tile_dims(([rt * 32, kt * 32], [kt * 32, ct * 32]), in0_tile_r_dim=r0)
    td = replace(td, in0_tile_c_dim=c0, in1_tile_r_dim=r1, in1_tile_c_dim=c1)
    fl = FaceLayoutConfig(unpack_transpose_faces=tr, unpack_transpose_within_face=tr, num_faces_in0=f0, num_faces_in1=f1,
                          num_faces=fo, partial_face_in0=p0, partial_face_in1=p1, partial_face_math=(r0 < 16),
                          partial_face_pack=(r0 < 16))
    return MatmulConfig(tile_dimensions=td, face_layout_config=fl, formats=fmt, stochastic_rnd=StochasticRounding.No,
                        dst_index=dst, dest_sync=sync, dest_acc=acc)


def accs(fmt):
    return [DestAccumulation.Yes] if is_dest_acc_needed(fmt) or fmt.input_format == F.Float32 else [DestAccumulation.No, DestAccumulation.Yes]


PARAMS = []
for fmt, sync in product(FMTS, (DestSync.Half, DestSync.Full)):
    for acc in accs(fmt):
        for shape in SHAPES:
            if fmt.input_format == F.Bfp4_b and min(shape[1], shape[3]) < 4:
                continue
            for rt, ct, kt in ((1, 1, 1), (1, 4, 1), (1, 4, 4), (2, 2, 4)):
                tr = (Transpose.No,) if shape[4] == 16 else (Transpose.No, Transpose.Yes)
                for t in tr:
                    for mf in pmm.MATH_FIDELITIES:
                        PARAMS.append((mf, cfg(fmt, acc, sync, rt, ct, kt, shape, t), 0, 1))
        for r0 in (1, 2, 4, 8, 16):
            if fmt.input_format == F.Bfp4_b and r0 < 4:
                continue
            tiny = ("tiny", r0, 32, 32, 32, 2, 4, 2, True, False)
            for ct in (1, 4, 8):
                for mf in pmm.MATH_FIDELITIES:
                    PARAMS.append((mf, cfg(fmt, acc, sync, 1, ct, 1, tiny, Transpose.Yes), 0, 1))


# a nonzero DEST tile index (the last that fits the section), full 32x32 tiles
FULL = ("full", 32, 32, 32, 32, 4, 4, 4, False, False)
CAP = {(DestSync.Half, False): 8, (DestSync.Full, False): 16, (DestSync.Half, True): 4, (DestSync.Full, True): 8}
for fmt, sync in product(FMTS[:2], (DestSync.Half, DestSync.Full)):
    for acc in accs(fmt):
        for rt, ct, kt in ((1, 1, 1), (1, 2, 4), (2, 1, 4)):
            for mf in (MathFidelity.LoFi, MathFidelity.HiFi4):
                dst = CAP[(sync, acc == DestAccumulation.Yes)] - rt * ct
                PARAMS.append((mf, cfg(fmt, acc, sync, rt, ct, kt, FULL, Transpose.No, dst), 0, 1))


@pytest.mark.perf
@pytest.mark.parametrize("math_fidelity,matmul_config,throttle,num_blocks", PARAMS)
def test_perf_math_matmul_cov(math_fidelity, matmul_config, throttle, num_blocks, perf_report):
    pmm.test_perf_math_matmul(math_fidelity, matmul_config, throttle, num_blocks, perf_report)
