# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.profiler import Profiler
from helpers.test_config import ProfilerBuild, run_test


def get_expected_overhead():
    match get_chip_architecture():
        case ChipArchitecture.WORMHOLE:
            return 36
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

    # filter out all zones that dont have marker "OVERHEAD"
    overhead_zones = [x for x in runtime.unpack if x.full_marker.marker == "OVERHEAD"]
    assert (
        len(overhead_zones) == 32
    ), f"Expected 32 overhead zones, got {len(overhead_zones)}"

    # the first iteration is inconsistent, because code is not in icache
    overhead_zones.pop(0)

    for i, zone in enumerate(
        overhead_zones, 9
    ):  # enumerate from 9 because the first iteration is ignored
        calculated_duration = 10 * i
        overhead = zone.duration - calculated_duration

        expected_overhead = get_expected_overhead()
        assert overhead == pytest.approx(expected_overhead, abs=5), (
            f"iterations: {i}, runtime: {zone.duration}/{i * 10} "
            f"(actual/calculated), overhead {overhead}/{expected_overhead} (actual/expected)"
        )
