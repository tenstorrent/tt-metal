import pytest
import torch
import random
import os
import numpy as np
from functools import partial
from itertools import chain
from operator import contains, eq, getitem
from pathlib import Path


@pytest.fixture(scope="function")
def reset_seeds():
    torch.manual_seed(213919)
    np.random.seed(213919)
    random.seed(213919)

    yield


@pytest.fixture(scope="function")
def function_level_defaults(reset_seeds):
    yield


@pytest.fixture(scope="session")
def is_dev_env():
    return os.environ.get("TT_METAL_ENV", "") == "dev"


@pytest.fixture(scope="session")
def model_location_generator():
    def model_location_generator_(rel_path):
        internal_weka_path = Path("/mnt/MLPerf")
        has_internal_weka = (internal_weka_path / "bit_error_tests").exists()

        if has_internal_weka:
            return Path("/mnt/MLPerf") / rel_path
        else:
            return Path("/opt/tt-metal-models") / rel_path

    return model_location_generator_


ALL_ARCHS = set(
    [
        "grayskull",
        "wormhole_b0",
    ]
)


def pytest_addoption(parser):
    parser.addoption(
        "--tt-arch",
        choices=[*ALL_ARCHS],
        default="grayskull",
        help="Target arch, ex. grayskull, wormhole_b0",
    )


def pytest_generate_tests(metafunc):
    tt_arch = metafunc.config.getoption("--tt-arch")

    silicon_arch_specific_fixture_name_to_avail_archs = {
        "silicon_arch_grayskull": set(
            [
                "grayskull",
            ]
        ),
        "silicon_arch_wormhole_b0": set(
            [
                "wormhole_b0",
            ]
        ),
    }

    check_uses_silicon_arch_specific_fixture = partial(
        contains, silicon_arch_specific_fixture_name_to_avail_archs
    )
    test_requested_silicon_arch_fixtures = tuple(
        filter(check_uses_silicon_arch_specific_fixture, metafunc.fixturenames)
    )
    is_test_requesting_specific_silicon_archs = (
        len(test_requested_silicon_arch_fixtures) > 0
    )
    get_archs_for_silicon_arch_specific_fixture = partial(
        getitem, silicon_arch_specific_fixture_name_to_avail_archs
    )
    test_requested_silicon_archs = chain.from_iterable(
        map(
            get_archs_for_silicon_arch_specific_fixture,
            test_requested_silicon_arch_fixtures,
        )
    )

    available_archs = (
        test_requested_silicon_archs
        if is_test_requesting_specific_silicon_archs
        else ALL_ARCHS
    )
    matches_user_requested_silicon_arch = partial(eq, tt_arch)
    available_archs = tuple(
        filter(matches_user_requested_silicon_arch, available_archs)
    )

    uses_silicon_arch = "silicon_arch_name" in metafunc.fixturenames

    # sanity
    if is_test_requesting_specific_silicon_archs and not uses_silicon_arch:
        raise Exception(
            f"{metafunc.function} requesting a specific silicon target, but doesn't use silicon_arch_name fixture"
        )

    if uses_silicon_arch:
        metafunc.parametrize("silicon_arch_name", available_archs)
        for test_requested_silicon_arch_fixture in test_requested_silicon_arch_fixtures:
            # These arch-specific fixtures should not be used in the test function,
            # so use any parameters
            metafunc.parametrize(test_requested_silicon_arch_fixture, [True])
