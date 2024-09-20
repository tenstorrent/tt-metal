# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# Legacy file calling new test files to keep old commands behavior.

import pytest

import ttnn
from tests.didt.test_ff1_matmul import test_ff1_matmul, test_specific_chip_ff1_matmul

MATH_FIDELITY = ttnn.MathFidelity.LoFi
GELU = False


@pytest.mark.parametrize(
    # to keep the legacy commands that used this parametrization id working
    "ff1_hang_dummy_param",
    (None,),
    ids=["ff1-hang"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(1, id="1chips"),
        pytest.param(2, id="2chips"),
        pytest.param(8, id="8chips"),
    ],
    indirect=["mesh_device"],
)
def test_reproduce_matmul_2d_hang(mesh_device, ff1_hang_dummy_param, iterations, use_program_cache):
    test_ff1_matmul(mesh_device, GELU, MATH_FIDELITY, iterations, -1, use_program_cache)


@pytest.mark.parametrize(
    "logical_chip_index",
    [0, 1, 2, 3, 4, 5, 6, 7],
    ids=[
        "logical_chip0",
        "logical_chip1",
        "logical_chip2",
        "logical_chip3",
        "logical_chip4",
        "logical_chip5",
        "logical_chip6",
        "logical_chip7",
    ],
)
def test_specific_chip_reproduce_matmul_2d_hang(mesh_device, logical_chip_index, iterations, use_program_cache):
    test_specific_chip_ff1_matmul(
        mesh_device, logical_chip_index, GELU, MATH_FIDELITY, iterations, -1, use_program_cache
    )


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(1, id="1chips"),
        pytest.param(2, id="2chips"),
        pytest.param(8, id="8chips"),
    ],
    indirect=["mesh_device"],
)
def test_determinism(mesh_device, iterations, determinism_check_iterations, use_program_cache):
    if determinism_check_iterations == -1:
        determinism_check_iterations = 1

    test_ff1_matmul(mesh_device, GELU, MATH_FIDELITY, iterations, determinism_check_iterations, use_program_cache)


@pytest.mark.parametrize(
    "logical_chip_index",
    [0, 1, 2, 3, 4, 5, 6, 7],
    ids=[
        "logical_chip0",
        "logical_chip1",
        "logical_chip2",
        "logical_chip3",
        "logical_chip4",
        "logical_chip5",
        "logical_chip6",
        "logical_chip7",
    ],
)
def test_determinism_specific_chip(
    mesh_device, logical_chip_index, iterations, determinism_check_iterations, use_program_cache
):
    if determinism_check_iterations == -1:
        determinism_check_iterations = 1

    test_specific_chip_ff1_matmul(
        mesh_device,
        logical_chip_index,
        GELU,
        MATH_FIDELITY,
        iterations,
        determinism_check_iterations,
        use_program_cache,
    )
