# dyn3/dyn4 coverage for the rerouted shapes (2-tile, 3-tile, 1x7, 12-tile), LoFi and HiFi2, Half and Full
from perf_matmul_dyn_base import DestSync, F, InputOutputFormat, MathFidelity

CASES = [
    (fmt, sync, dims, mf)
    for fmt in (
        InputOutputFormat(F.Float16_b, F.Float16_b),
        InputOutputFormat(F.Bfp8_b, F.Bfp8_b),
        InputOutputFormat(F.Float16_b, F.Bfp4_b),
    )
    for sync in (DestSync.Half, DestSync.Full)
    for dims in ((1, 2, 1), (2, 1, 1), (3, 1, 1), (1, 7, 3), (3, 4, 1))
    for mf in (MathFidelity.LoFi, MathFidelity.HiFi2)
    if not (sync == DestSync.Half and dims == (3, 4, 1))
]
