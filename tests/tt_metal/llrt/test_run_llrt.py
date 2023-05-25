import pytest
from functools import partial
from tests.scripts.common import TestEntry, generate_test_entry_id

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
)


SKIP_LLRT_ENTRIES = set(
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

LONG_LLRT_TEST_ENTRIES = SILICON_DRIVER_TEST_ENTRIES | LLRT_TEST_ENTRIES


def detect_llrt_skip(test_entry, silicon_arch_name):
    assert isinstance(test_entry, TestEntry)

    if test_entry in SKIP_LLRT_WORMHOLE_ENTRIES and "wormhole" in silicon_arch_name:
        pytest.skip(f"Test {test_entry} not supported in wormhole")

    if test_entry in SKIP_LLRT_ENTRIES:
        pytest.skip(f"Test {test_entry} not functioning right now")


@pytest.fixture
def llrt_to_dos():
    from loguru import logger

    logger.warning("Need to do timeouts")
    logger.warning("Need to do soft resets")
    logger.warning("Need to do short versions")
    logger.warning(
        "Need to do cmd line parameterize pipeline type, arch, maybe module?"
    )
    logger.warning("Need to do randomization?")
    logger.warning("Need to do multiple threads for build_kernels")


@pytest.mark.post_commit
@pytest.mark.hyperquotidian
@pytest.mark.parametrize(
    "llrt_test_entry", POST_COMMIT_LLRT_TEST_ENTRIES, ids=generate_test_entry_id
)
def test_run_llrt_test(silicon_arch_name, llrt_test_entry, llrt_to_dos):
    detect_llrt_skip(llrt_test_entry, silicon_arch_name)

    from loguru import logger

    logger.info(llrt_test_entry.test_name)


@pytest.mark.long
@pytest.mark.parametrize(
    "llrt_test_entry", LONG_LLRT_TEST_ENTRIES, ids=generate_test_entry_id
)
def test_run_llrt_test_long(
    silicon_arch_name, silicon_arch_grayskull, llrt_test_entry, llrt_to_dos
):
    detect_llrt_skip(llrt_test_entry, silicon_arch_name)

    from loguru import logger

    logger.info(llrt_test_entry.test_name)
