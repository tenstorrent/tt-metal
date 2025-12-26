# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_coverage
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.profiler import ProfilerConfig
from helpers.stimuli_config import StimuliConfig
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

    if TestConfig.MODE == TestMode.PRODUCE:
        pytest.skip()

    configuration = ProfilerConfig(
        "sources/profiler_overhead_test.cpp",
        input_output_formats([DataFormat.Float16])[0],
        variant_stimuli=StimuliConfig(
            [], DataFormat.Float16, [], DataFormat.Float16, DataFormat.Float16, 1, 1
        ),
    )

    configuration.generate_variant_hash()
    configuration.build_elfs()
    configuration.run_elf_files(workers_tensix_coordinates)

    runtime = configuration.get_data(workers_tensix_coordinates)

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
