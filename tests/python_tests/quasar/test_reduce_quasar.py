# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import pytest
import torch
from helpers.device import collect_results, write_stimuli_to_l1
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    ReduceGapoolGolden,
    ReduceGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    ImpliedMathFormat,
    MathFidelity,
    MathOperation,
    ReduceDimension,
    ReducePool,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.utils import passed_test

# Helper dictionary to map reduce dimensions to math operations
mathop_mapping = {
    ReduceDimension.Row: MathOperation.ReduceRow,
    ReduceDimension.Column: MathOperation.ReduceColumn,
    ReduceDimension.Scalar: MathOperation.ReduceScalar,
}

MATH_FIDELITY_MODES = [
    MathFidelity.LoFi,
    MathFidelity.HiFi2,
    MathFidelity.HiFi3,
    MathFidelity.HiFi4,
]
POOL_TYPES = [ReducePool.Max, ReducePool.Sum, ReducePool.Average]


def generate_pool_type_and_math_fidelity_combinations():
    def is_valid_combination(pool_type, math_fidelity):
        # Max pool only supports LoFi
        if pool_type == ReducePool.Max:
            return math_fidelity == MathFidelity.LoFi
        # Sum and Average support all fidelities
        return True

    return [
        combo
        for combo in product(POOL_TYPES, MATH_FIDELITY_MODES)
        if is_valid_combination(*combo)
    ]


@pytest.mark.quasar
@parametrize(
    test_name="reduce_quasar_test",
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
        ],
    ),
    dest_acc=[DestAccumulation.No],
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    pool_type_and_math_fidelity=generate_pool_type_and_math_fidelity_combinations(),
    implied_math_format=[ImpliedMathFormat.No, ImpliedMathFormat.Yes],
)
def test_reduce_quasar(
    test_name,
    formats,
    dest_acc,
    reduce_dim,
    pool_type_and_math_fidelity,
    implied_math_format,
):

    pool_type, math_fidelity = pool_type_and_math_fidelity

    input_dimensions = [64, 64]

    src_A, _, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format, input_dimensions=input_dimensions
    )

    if pool_type in [
        ReducePool.Max,
        ReducePool.Sum,
    ]:  # result in srcA should be multiplied by 1
        src_B = torch.full((1024,), 1)
    else:
        # reduce average divides by length of elements in array we reduce
        src_B = torch.full((1024,), 1 / 32)

    if pool_type == ReducePool.Max:
        generate_golden = get_golden_generator(ReduceGolden)
        golden_tensor = generate_golden(
            src_A, reduce_dim, pool_type, formats.output_format, tile_cnt
        )
    else:
        generate_golden = get_golden_generator(ReduceGapoolGolden)
        golden_tensor = generate_golden(
            src_A, src_B, formats.output_format, reduce_dim, math_fidelity, tile_cnt
        )

    mathop = mathop_mapping[reduce_dim]

    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "reduce_dim": reduce_dim,
        "pool_type": pool_type,
        "mathop": mathop,
        "math_fidelity": math_fidelity,
        "implied_math_format": implied_math_format,
        "tile_cnt": tile_cnt,
    }

    res_address = write_stimuli_to_l1(
        test_config,
        src_A,
        src_B,
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt,
        tile_count_B=1,
    )

    run_test(test_config)

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
