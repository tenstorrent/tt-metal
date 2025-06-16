# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.device import (
    run_elf_files,
    wait_for_tensix_operations_finished,
)
from helpers.profiler import Profiler, build_with_profiler


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

    profiler_meta = build_with_profiler(test_config)
    assert profiler_meta is not None, "Profiler metadata should not be None"

    run_elf_files("profiler_overhead_test")
    wait_for_tensix_operations_finished()

    runtime = Profiler.get_data(profiler_meta)

    # filter out all zones that dont have marker "OVERHEAD"
    overhead_zones = [x for x in runtime.unpack if x.full_marker.marker == "OVERHEAD"]
    assert (
        len(overhead_zones) == 30
    ), f"Expected 32 overhead zones, got {len(overhead_zones)}"

    for loop_iterations, zone in enumerate(overhead_zones, 10):
        calculated_duration = 10 * loop_iterations
        overhead = zone.duration - calculated_duration

        assert overhead == pytest.approx(
            get_expected_overhead(), abs=5
        ), f"iterations: {loop_iterations}, runtime: {zone.duration}/{calculated_duration} (actual/calculated), overhead {overhead}/{EXPECTED_OVERHEAD} (actual/expected) "
