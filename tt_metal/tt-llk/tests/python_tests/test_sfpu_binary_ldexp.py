# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    VectorMode,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    VECTOR_MODE,
    generate_input_dim,
)

pytestmark = [skip_for_quasar]

FORMATS = input_output_formats([DataFormat.Float32], same=True)

ELEMENTS_PER_TILE = 1024

INF = float("inf")
NAN = float("nan")
TINY = torch.finfo(torch.float32).tiny

_PAIRS = {
    "inf_nan_a": [
        (INF, 3.0),
        (-INF, 3.0),
        (INF, -128.0),
        (-INF, -128.0),
        (INF, -150.0),
        (NAN, 3.0),
        (-NAN, -128.0),
    ],
    "inf_nan_b": [
        (1.0, NAN),
        (1.0, -NAN),
        (-2.0, INF),
        (1.0, -INF),
        (INF, INF),
        (INF, -INF),
        (0.0, INF),
    ],
    "overflow": [
        (1.0, 128.0),
        (1.5, 127.0),
        (2.0**100, 100.0),
        (-3.0, 200.0),
        (TINY, 128.0),
        (2.0**-100, 200.0),
        (0.0, 128.0),
    ],
    "underflow": [
        (1.0, -127.0),
        (1.0, -150.0),
        (3.0, -200.0),
        (-5.0, -160.0),
        (2.0**127, -150.0),
        (2.0**100, -226.0),
        (-TINY, -1.0),
    ],
    "near_min_normal": [
        (8.0, -128.0),
        (7.0, -127.0),
        (100.0, -128.0),
        (2.0**30, -149.0),
        (1.5 * 2.0**23, -140.0),
        (1e30, -136.0),
        (-1e30, -130.0),
    ],
    "normal": [
        (5.0, 3.0),
        (-0.75, 10.0),
        (1.25, -20.0),
        (-0.0, 3.0),
        (3.0, 0.0),
        (TINY, 1.0),
        (2.0**-60, 64.0),
    ],
}


def _tiles(pairs):
    a = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
    b = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
    reps = ELEMENTS_PER_TILE // len(pairs) + 1
    return (
        a.repeat(reps)[:ELEMENTS_PER_TILE].contiguous(),
        b.repeat(reps)[:ELEMENTS_PER_TILE].contiguous(),
    )


def _golden(a, b):
    z = torch.ldexp(a, b)
    return torch.where(z.abs() < TINY, torch.zeros_like(z), z)


@parametrize(
    formats=FORMATS,
    inputs=list(_PAIRS),
)
def test_sfpu_binary_ldexp(formats, inputs):
    src_A, src_B = _tiles(_PAIRS[inputs])

    configuration = TestConfig(
        "sources/sfpu_binary_ldexp_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            APPROX_MODE(ApproximationMode.No),
            VECTOR_MODE(VectorMode.RC),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=formats.input_format.is_32_bit(),
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result[:ELEMENTS_PER_TILE]
    torch_format = format_dict[formats.output_format]
    got = torch.tensor(res_from_L1, dtype=torch_format).flatten().to(torch.float32)
    want = _golden(src_A, src_B)

    bad = torch.isnan(got) != torch.isnan(want)
    finite = ~torch.isnan(want)
    bad[finite] |= got[finite] != want[finite]
    if bad.any():
        rows = sorted(
            {
                (src_A[i].item(), src_B[i].item(), want[i].item(), got[i].item())
                for i in torch.nonzero(bad).flatten().tolist()
            },
            key=str,
        )
        detail = "\n".join(
            f"  ldexp({a}, {b}) = {g}, expected {w}" for a, b, w, g in rows
        )
        raise AssertionError(
            f"{int(bad.sum())}/{ELEMENTS_PER_TILE} lanes disagree with torch.ldexp:\n{detail}"
        )
