# Local sweep extension (not committed): perf_math_matmul's grid (regular, transpose, throttle 5, dest hand-off, tiny
# tiles with partial faces) for Bfp4_b inputs, reduced like the main sweep (transpose at throttle 0 and one block,
# throttle 5 at kt 4 with LoFi and HiFi4).
from itertools import product
import perf_math_matmul as pmm
import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.matmul_sweep import sweep_tiny_tiles_matmul

F = DataFormat
B4 = [InputOutputFormat(F.Bfp4_b, o) for o in (F.Float16_b, F.Bfp8_b, F.Bfp4_b, F.Float32)]
pmm.MATMUL_FORMATS = B4
REG = pmm.generate_perf_matmul_combinations()
TINY = [c for c in sweep_tiny_tiles_matmul(B4, pmm.DEST_ACC_MODES, pmm.STOCHASTIC_ROUNDING_MODES, pmm.DEST_SYNC_MODES,
                                           math_matmul=True) if pmm._fits_tiny_perf_tile_shape(c)]


def _tr(c):
    return c.face_layout_config.unpack_transpose_faces.name == "Yes"


PARAMS = (
    [(f, c, 0, 1) for f, c in product(pmm.MATH_FIDELITIES, REG)]
    + [(f, c, 5, 1) for f, c in product(pmm.MATH_FIDELITIES, REG)
       if not _tr(c) and c.tile_dimensions.kt_dim == 4 and f.name in ("LoFi", "HiFi4")]
    + [(f, c, 0, 1) for f, c in product(pmm.MATH_FIDELITIES, TINY)]
    + [(f, c, 0, pmm.DEST_HANDOFF_NUM_BLOCKS) for f, c in product(pmm.MATH_FIDELITIES, REG)
       if not _tr(c) and c.tile_dimensions.kt_dim == 1]
)


@pytest.mark.perf
@pytest.mark.parametrize("math_fidelity,matmul_config,throttle,num_blocks", PARAMS)
def test_perf_math_matmul_bfp4(math_fidelity, matmul_config, throttle, num_blocks, perf_report):
    pmm.test_perf_math_matmul(math_fidelity, matmul_config, throttle, num_blocks, perf_report)
