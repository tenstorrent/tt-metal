# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# Legacy file calling new test files to keep old commands behavior.

import pytest

from tests.didt.test_lm_head_matmul import test_lm_head_matmul, test_specific_chip_lm_head_matmul


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(1, id="1chips"),
        pytest.param(2, id="2chips"),
        pytest.param(8, id="8chips"),
    ],
    indirect=["mesh_device"],
)
def test_reproduce_lm_head_nd_32(mesh_device, iterations, use_program_cache):
    test_lm_head_matmul(mesh_device, iterations, -1, use_program_cache)


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
def test_specific_chip_lm_head_nd_32(mesh_device, logical_chip_index, iterations, use_program_cache):
    test_specific_chip_lm_head_matmul(mesh_device, logical_chip_index, iterations, -1, use_program_cache)


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

    test_lm_head_matmul(mesh_device, iterations, determinism_check_iterations, use_program_cache)


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

    test_specific_chip_lm_head_matmul(
        mesh_device, logical_chip_index, iterations, determinism_check_iterations, use_program_cache
    )
