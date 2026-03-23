# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_coverage
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.perf import PerfConfig
from helpers.profiler import Profiler
from helpers.test_config import TestConfig, TestMode


def get_expected_overhead():
    match get_chip_architecture():
        case ChipArchitecture.WORMHOLE:
            return 29
        case ChipArchitecture.BLACKHOLE:
            return 30
        case _:
            raise ValueError("Unsupported chip architecture")


# Coverage uses different linker script, that doesn't utilize local data memory at all, only L1
# Because of this, measured overhead is at 2.3k instead of ~23 cycles
@skip_for_coverage
def test_profiler_overhead(workers_tensix_coordinates):

    # This is a test of the profiler itself and doesn't use configuration.run method at all,
    # therefore it can't leverage default producer-consumer separation of compile and execute phases.
    # In order to avoid compiling the test elf twice we run it in only one of two phases - the consumer/execute phase,
    # where everything is done.
    if TestConfig.MODE == TestMode.PRODUCE or TestConfig.WITH_COVERAGE:
        pytest.skip()

    configuration = PerfConfig("sources/profiler_overhead_test.cpp")

    configuration.generate_variant_hash()
    configuration.build_elfs()
    configuration.run_elf_files(workers_tensix_coordinates)

    runtime = Profiler.get_data(
        configuration.test_name, configuration.variant_id, workers_tensix_coordinates
    )

    # filter out all zones that don't have marker "OVERHEAD"

    overhead_zones = runtime.zones().marker("OVERHEAD").frame()
    assert (
        len(overhead_zones) == 32
    ), f"Expected 32 overhead zones, got {len(overhead_zones)}"

    # the first iteration is inconsistent, because code is not in icache
    overhead_zones = overhead_zones.iloc[1:].reset_index(drop=True)

    expected_overhead = get_expected_overhead()
    for i, zone in overhead_zones.iterrows():
        calculated_duration = 10 * (i + 9)
        overhead = zone["duration"] - calculated_duration

        assert overhead == pytest.approx(expected_overhead, abs=5), (
            f"iterations: {i}, runtime: {zone['duration']}/{calculated_duration} "
            f"(actual/calculated), overhead {overhead}/{expected_overhead} (actual/expected)"
        )
