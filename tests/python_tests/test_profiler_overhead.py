# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.profiler import Profiler
from helpers.test_config import ProfilerBuild, run_test


def get_expected_overhead():
    match get_chip_architecture():
        case ChipArchitecture.WORMHOLE:
            return 29
        case ChipArchitecture.BLACKHOLE:
            return 30
        case _:
            raise ValueError("Unsupported chip architecture")


def test_profiler_overhead():

    test_config = {
        "testname": "profiler_overhead_test",
    }

    run_test(test_config, profiler_build=ProfilerBuild.Yes)

    runtime = Profiler.get_data(test_config["testname"])

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
