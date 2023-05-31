import pytest
from functools import partial
from tests.scripts.common import (
    TestEntry,
    generate_test_entry_id,
    run_single_test,
    SpecificReturnCodes,
)

from loguru import logger

SILICON_DRIVER_TEST_ENTRIES = set(
    [
        TestEntry("llrt/tests/test_silicon_driver", "test_silicon_driver"),
        TestEntry(
            "llrt/tests/test_silicon_driver_dram_sweep",
            "test_silicon_driver_dram_sweep",
        ),
        TestEntry(
            "llrt/tests/test_silicon_driver_l1_sweep", "test_silicon_driver_l1_sweep"
        ),
    ]
)

SHORT_SILICON_DRIVER_TEST_ENTRIES = set(
    [
        TestEntry("llrt/tests/test_silicon_driver", "test_silicon_driver"),
        TestEntry(
            "llrt/tests/test_silicon_driver_dram_sweep",
            "test_silicon_driver_dram_sweep",
            "--short",
        ),
        TestEntry(
            "llrt/tests/test_silicon_driver_l1_sweep",
            "test_silicon_driver_l1_sweep",
            "--short",
        ),
    ]
)

LLRT_TEST_ENTRIES = set(
    [
        TestEntry(
            "llrt/tests/test_run_blank_brisc_triscs", "test_run_blank_brisc_triscs"
        ),
        TestEntry("llrt/tests/test_run_risc_read_speed", "test_run_risc_read_speed"),
        TestEntry("llrt/tests/test_run_risc_write_speed", "test_run_risc_write_speed"),
        TestEntry("llrt/tests/test_run_eltwise_sync", "test_run_eltwise_sync"),
        TestEntry("llrt/tests/test_run_sync", "test_run_sync"),
        TestEntry("llrt/tests/test_run_sync_db", "test_run_sync_db"),
        TestEntry("llrt/tests/test_run_dataflow_cb_test", "test_run_dataflow_cb_test"),
        TestEntry("llrt/tests/test_run_test_debug_print", "test_run_test_debug_print"),
        TestEntry(
            "llrt/tests/test_run_datacopy_switched_riscs",
            "test_run_datacopy_switched_riscs",
        ),
        TestEntry("llrt/tests/test_dispatch_v1", "test_dispatch_v1"),
    ]
) - set(
    [
        TestEntry("llrt/tests/test_dispatch_v1", "test_dispatch_v1"),
    ]
)


SKIP_LLRT_WORMHOLE_ENTRIES = set(
    [
        TestEntry("llrt/tests/test_run_eltwise_sync", "test_run_eltwise_sync"),
        TestEntry("llrt/tests/test_run_test_debug_print", "test_run_test_debug_print"),
        TestEntry(
            "llrt/tests/test_run_datacopy_switched_riscs",
            "test_run_datacopy_switched_riscs",
        ),
    ]
)


POST_COMMIT_LLRT_TEST_ENTRIES = SHORT_SILICON_DRIVER_TEST_ENTRIES | LLRT_TEST_ENTRIES

POST_COMMIT_LLRT_TEST_WH_B0_ENTRIES = (
    POST_COMMIT_LLRT_TEST_ENTRIES - SKIP_LLRT_WORMHOLE_ENTRIES
)

LONG_LLRT_TEST_ENTRIES = SILICON_DRIVER_TEST_ENTRIES | LLRT_TEST_ENTRIES


def run_single_llrt_test(test_entry, timeout=600, tt_arch="grayskull"):
    run_test = partial(run_single_test, "llrt", timeout=timeout, tt_arch=tt_arch)

    test_process_result = run_test(test_entry)

    return_code = test_process_result.returncode

    assert (
        return_code != SpecificReturnCodes.TIMEOUT_RETURN_CODE.value
    ), f"{test_entry} seems to have timed out - TIMEOUT ERROR"
    assert (
        return_code == SpecificReturnCodes.PASSED_RETURN_CODE.value
    ), f"{test_entry} failed with an error return code of {return_code}"


@pytest.mark.post_commit
@pytest.mark.frequent
@pytest.mark.parametrize(
    "llrt_test_entry", POST_COMMIT_LLRT_TEST_ENTRIES, ids=generate_test_entry_id
)
def test_run_llrt_test_grayskull(
    silicon_arch_grayskull, silicon_arch_name, llrt_test_entry, llrt_fixtures
):
    run_single_llrt_test(llrt_test_entry, tt_arch=silicon_arch_name)


@pytest.mark.post_commit
@pytest.mark.frequent
@pytest.mark.parametrize(
    "llrt_test_entry", POST_COMMIT_LLRT_TEST_WH_B0_ENTRIES, ids=generate_test_entry_id
)
def test_run_llrt_test_wormhole_b0(
    silicon_arch_name, silicon_arch_wormhole_b0, llrt_test_entry, llrt_fixtures
):
    run_single_llrt_test(llrt_test_entry, tt_arch=silicon_arch_name)


@pytest.mark.long
@pytest.mark.parametrize(
    "llrt_test_entry", LONG_LLRT_TEST_ENTRIES, ids=generate_test_entry_id
)
def test_run_llrt_test_long(
    silicon_arch_name, silicon_arch_grayskull, llrt_test_entry, llrt_fixtures
):
    run_single_llrt_test(llrt_test_entry, tt_arch=silicon_arch_name)
