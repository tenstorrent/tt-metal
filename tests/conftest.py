import pytest
import torch
import random
import os
import numpy as np
from functools import partial
from itertools import chain
from operator import contains, eq, getitem
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/../")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")
from loguru import logger

from tests.scripts.common import run_process_and_get_result


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
    """
    This is not a standard docstring.

    We will explain the non-standard fixtures that pytest_generate_tests is
    creating here.

    silicon_arch_name and silicon_arch_<ARCH_NAME>
    ----------------------------------------------

    This is how tests should be requesting accelerator architecture names.
    Tests which aim to run on silicon should request a silicon_arch_name
    fixture. Just that single fixture will parametrize the test to run on the
    provided architecture name from the command line through the --tt-arch
    option. The value of the fixture will be the string value of the
    architecture name. For example,

    @pytest.mark.post_commit
    def test_model_silicon(silicon_arch_name):
        # silicon_arch_name will be one of grayskull, wormhole_b0 etc.
        run_model_on_silicon(silicon_arch_name)
        ...

    If you want to restrict a test to only a specific architecture, you can
    provide an additional fixture in the form of silicon_arch_<ARCH_NAME>. This
    will limit the range of possible values for silicon_arch_name to only be
    ARCH_NAME.

    @pytest.mark.post_commit
    def test_model_silicon_grayskull_only(
        silicon_arch_name,
        silicon_arch_grayskull,
    ):
        # silicon_arch_name can only be grayskull or empty
        run_model_on_silicon(silicon_arch_name)
        ...

    If --tt-arch specifies an architecture that's not ARCH_NAME, the test will
    be skipped. We ensure skipping by providing an empty list parametrization
    for silicon_arch_name, and with the empty_parameter_set_mark config option
    for pytest, will skip any tests with an empty list parametrization.

    Note that you must provide silicon_arch_name as a fixture if you want to
    use the silicon_arch_<ARCH_NAME> fixture.

    Note that if tests want to use the ARCH value from the API, tests should
    create their own separate fixture which will convert the string value
    provided from silicon_arch_name into ARCH. We keep it as strings here
    because these fixtures will be used in tests which do not have access to
    any Python APIs.
    """

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
    test_requested_silicon_archs = ALL_ARCHS.intersection(
        *map(
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
            # The values of these arch-specific fixtures should not be used in
            # the test function, so use any parameters, like [True]
            metafunc.parametrize(test_requested_silicon_arch_fixture, [True])


# Report stashing to get outcomes etc
phase_report_key = pytest.StashKey()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    # store test results for each phase of a call, which can
    # be "setup", "call", "teardown"
    item.stash.setdefault(phase_report_key, {})[rep.when] = rep


@pytest.fixture(scope="function")
def reset_tensix(request):
    yield

    report = request.node.stash[phase_report_key]

    test_failed = ("call" not in report) or report["call"].failed

    if test_failed:
        logger.debug("Test failed - resetting with tensix-reset script")
        result = run_process_and_get_result(
            "./tt_metal/device/bin/silicon/tensix-reset"
        )
        assert result.returncode == 0, "Tensix reset script raised error"


@pytest.fixture(scope="function")
def use_program_cache():
    from libs import tt_lib as ttl
    ttl.program_cache.enable()
    yield
    ttl.program_cache.disable_and_clear()
