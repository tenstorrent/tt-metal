# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The NoC interface and router counters count the host's polls of a long launch: run with --perf-noc-counters (and
--perf-router-counters on Blackhole)."""

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.counters import NOC_BANK_PREFIX, ROUTER_BANK_PREFIX, read_tile_counters
from helpers.llk_params import PerfRunType
from helpers.perf.core import COUNT_TILE_LOOP, PerfConfig
from helpers.test_config import BuildMode, TestConfig
from tt_llk_harness import DataFormat, input_output_formats


def _per_bank(rows, prefix, counter=None):
    totals = {}
    for r in rows:
        if r["bank"].startswith(prefix) and (
            counter is None or r["counter_name"] == counter
        ):
            totals[r["bank"]] = totals.get(r["bank"], 0) + r["count"]
    return totals


def test_tile_counters_count_the_host_polls():
    if get_chip_architecture() == ChipArchitecture.QUASAR:
        pytest.skip("the Quasar harness has no BRISC to read the tile counters")
    if TestConfig.PERF_NOC_GROUPS is None:
        pytest.skip("needs --perf-noc-counters")
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

    configuration = PerfConfig(
        "sources/tile_counters_count_test.cpp",
        input_output_formats([DataFormat.Float16_b])[0],
        run_types=[PerfRunType.L1_TO_L1],
    )
    configuration.templates, configuration.runtimes, configuration.current_run_type = (
        configuration.run_configs[0]
    )
    configuration.generate_variant_hash()
    configuration.build_elfs()
    configuration.counted_zones = COUNT_TILE_LOOP
    configuration.write_runtimes_to_L1()
    configuration.run_elf_files()
    polls = 0

    def count_poll():
        nonlocal polls
        polls += 1

    configuration.wait_for_tensix_operations_finished(
        timeout=10, poll_callback=count_poll
    )
    rows = read_tile_counters(
        TestConfig.TENSIX_LOCATION,
        configuration.counted_zones,
        configuration.launch_token,
        TestConfig.PERF_NOC_GROUPS,
        TestConfig.PERF_ROUTER_OPTION,
    )
    requests = _per_bank(rows, NOC_BANK_PREFIX, "SLV_RD_REQ_RECEIVED")
    words = _per_bank(rows, NOC_BANK_PREFIX, "SLV_RD_DATA_WORD_SENT")
    routers = _per_bank(rows, ROUTER_BANK_PREFIX)
    report = f"host polls {polls}, read requests {requests}, data words {words}, router out {routers}"
    print(report)

    # Every poll but the last (it reads the launch done after the window) is one single-word read inside the window;
    # the first may land before BRISC opens it.
    assert polls > 100, report
    assert polls - 2 <= sum(requests.values()) <= polls - 1, report
    assert sum(words.values()) == sum(requests.values()), report
    if TestConfig.PERF_ROUTER_OPTION is not None:
        # Each poll's response leaves through the router of the NoC the request came in on.
        noc = max(requests, key=requests.get)[len(NOC_BANK_PREFIX) :]
        assert (
            routers[f"{ROUTER_BANK_PREFIX}{noc}"] >= requests[f"{NOC_BANK_PREFIX}{noc}"]
        ), report
