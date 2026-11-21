# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reduce perf sweep shared by WH/BH (perf_reduce.py) and Quasar (perf_reduce_quasar.py).

Both sweeps take their formats, reduce dims, pool types and fidelities from here, so
compare_test_and_perf.py --cross-arch lines the reports up axis by axis. Rows pair
where both architectures accept the config; the rest stay one-sided:

- Blackhole only: the Bfp8_b formats.
- Quasar only: the 1x32 and 16x16 tiles, and MxFp4.
"""

from .format_config import DataFormat, InputOutputFormat
from .llk_params import DestAccumulation, MathFidelity, ReduceDimension, ReducePool

PERF_TILE_COUNT = 16

CORE_FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
]
CORE_REDUCE_DIMS = [ReduceDimension.Row, ReduceDimension.Column]
CORE_POOL_TYPES = [ReducePool.Max, ReducePool.Sum]
CORE_TILE_DIMENSIONS = [(32, 32)]

# Formats both architectures add on top of the core.
SHARED_EXTENSION_FORMATS = [
    InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b),
]
# Block-float is WH/BH only; Quasar has no Bfp8_b.
WHBH_EXTENSION_FORMATS = SHARED_EXTENSION_FORMATS + [
    InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Bfp8_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Bfp8_b),
]
# Quasar picks its tile shapes per format in reduce_tile_dimensions (32x16 included).
WHBH_EXTENSION_TILE_DIMENSIONS = [(32, 16)]
EXTENSION_POOL_TYPES = [ReducePool.Average]

PERF_POOL_TYPES = CORE_POOL_TYPES + EXTENSION_POOL_TYPES
PERF_MATH_FIDELITIES = [MathFidelity.LoFi, MathFidelity.HiFi4]

# Scalar reduce runs the column and row passes back to back, so one config is
# enough to track it; it is a separate test so it is not crossed with the matrix.
# Both fidelities run on every arch so the scalar rows pair. On Quasar, HiFi4 goes
# past the functional Float16_b LoFi cap on purpose: init drops the phases Float16_b
# cannot use, so the pair shows what HiFi4 costs on WH/BH and not on Quasar.
SCALAR_REPRESENTATIVE = dict(
    formats=[InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)],
    dest_acc=[DestAccumulation.No],
    reduce_dim=[ReduceDimension.Scalar],
    pool_type=[ReducePool.Sum],
    math_fidelity=PERF_MATH_FIDELITIES,
)


# The core matrix already measures PERF_TILE_COUNT tiles; this sweep adds the other
# block lengths on one representative config.
BLOCK_LENGTH_TILE_COUNTS = [1, 8, 32]
BLOCK_LENGTH_REPRESENTATIVE = dict(
    formats=[InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)],
    dest_acc=[DestAccumulation.No],
    reduce_dim=[ReduceDimension.Row],
    pool_type=[ReducePool.Sum],
)


def reduce_to_one_modes(reduce_dim, pool_type):
    """Production REDUCE_ROW Max/Sum folds a whole row of tiles into one DEST tile."""
    if reduce_dim == ReduceDimension.Row and pool_type in (
        ReducePool.Max,
        ReducePool.Sum,
    ):
        return [False, True]
    return [False]


def perf_math_fidelities(pool_type, valid_fidelities=None):
    """GMPOOL (Max) ignores fidelity; LoFi and HiFi4 bracket the fidelity phase count.

    ``valid_fidelities`` caps the result to what the format supports (Quasar passes
    ``get_valid_math_fidelities(formats)``). Wrap calls in a lambda inside ``parametrize``:
    it treats every parameter name, defaulted or not, as a dependency.
    """
    if pool_type == ReducePool.Max:
        return [MathFidelity.LoFi]
    return [
        fidelity
        for fidelity in PERF_MATH_FIDELITIES
        if valid_fidelities is None or fidelity in valid_fidelities
    ]


def perf_input_dimensions(tile_dimensions, tile_count=PERF_TILE_COUNT):
    """Stack ``tile_count`` tiles vertically so every tile shape runs the same tile count."""
    return [tile_dimensions[0] * tile_count, tile_dimensions[1]]
