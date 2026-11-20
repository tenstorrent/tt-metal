# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The RISC-V monitor of every TRISC counts a loop of known length: run with --perf-riscv-pmc."""

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.counters import RISCV_BANK_PREFIX, read_riscv_counters
from helpers.llk_params import PerfRunType
from helpers.perf.core import COUNT_INIT, COUNT_TILE_LOOP, PerfConfig
from helpers.profiler import Profiler
from helpers.test_config import BuildMode, TestConfig
from tt_llk_harness import DataFormat, input_output_formats

# Kept equal to the counts in riscv_pmc_count_test.cpp; each iteration retires ten instructions.
SHORT_ITERATIONS = 100
LONG_ITERATIONS = 1100
INSTRUCTIONS_PER_ITERATION = 10

PROFILER_THREAD = {"UNPACK": "unpack", "MATH": "math", "PACK": "pack", "SFPU": "sfpu"}


def _counts(rows, zone, bank):
    values = {
        r["counter_name"]: r["count"]
        for r in rows
        if r["zone"] == zone and r["bank"] == bank
    }
    return values["cycles"], values["instructions"]


def test_riscv_pmc_counts_a_known_loop():
    if get_chip_architecture() == ChipArchitecture.WORMHOLE:
        pytest.skip("Wormhole RISC-V cores have no performance counter CSRs")
    if TestConfig.PERF_RISCV_EVENTS is None:
        pytest.skip("needs --perf-riscv-pmc")
    # Built and run in one phase, like the profiler overhead test: the check reads the device directly.
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

    configuration = PerfConfig(
        "sources/riscv_pmc_count_test.cpp",
        input_output_formats([DataFormat.Float16_b])[0],
        run_types=[PerfRunType.L1_TO_L1],
    )
    if not configuration.run_configs:
        pytest.skip("LLK_PERF_RUN_TYPES selects none of this test's run types")
    configuration.templates, configuration.runtimes, configuration.current_run_type = (
        configuration.run_configs[0]
    )
    configuration.generate_variant_hash()
    configuration.build_elfs()
    configuration.counted_zones = COUNT_INIT | COUNT_TILE_LOOP
    configuration.write_runtimes_to_L1()
    configuration.run_elf_files()
    configuration.wait_for_tensix_operations_finished()

    zones = Profiler.get_data(
        configuration.test_name, configuration.variant_id, TestConfig.TENSIX_LOCATION
    )
    rows = read_riscv_counters(
        TestConfig.TENSIX_LOCATION,
        configuration.counted_zones,
        configuration.launch_token,
    )
    banks = sorted({r["bank"] for r in rows})
    assert banks, "no RISC-V monitor rows"

    extra = (LONG_ITERATIONS - SHORT_ITERATIONS) * INSTRUCTIONS_PER_ITERATION
    report = []
    for bank in banks:
        thread = PROFILER_THREAD[bank[len(RISCV_BANK_PREFIX) :]]
        short_cycles, short_instructions = _counts(rows, "ZONE_0", bank)
        long_cycles, long_instructions = _counts(rows, "ZONE_1", bank)
        frame = zones.zones().frame()
        frame = frame[frame["thread"] == thread]
        short_wall = int(frame[frame["marker"] == "INIT"]["duration"].iloc[0])
        long_wall = int(frame[frame["marker"] == "TILE_LOOP"]["duration"].iloc[0])
        report.append(
            f"{bank}: instructions {short_instructions}/{long_instructions}, cycles {short_cycles}/{long_cycles}, "
            f"zone wall {short_wall}/{long_wall}"
        )
        # The same capture code surrounds both loops, so only the extra iterations remain.
        assert long_instructions - short_instructions == extra, report[-1]
        assert (
            short_instructions >= SHORT_ITERATIONS * INSTRUCTIONS_PER_ITERATION
        ), report[-1]
        # The RISC-V window encloses the zone's timestamps.
        assert short_cycles >= short_wall and long_cycles >= long_wall, report[-1]
        assert long_cycles - short_cycles == pytest.approx(
            long_wall - short_wall, rel=0.02
        ), report[-1]
    print("\n".join(report))
