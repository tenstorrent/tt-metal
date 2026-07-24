# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import pytest
from conftest import skip_for_coverage, skip_for_quasar
from helpers.perf import PerfConfig
from helpers.profiler import EntryType, Profiler
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import TemplateParameter
from ttexalens.tt_exalens_lib import read_words_from_device


@dataclass
class OVERRUN_FILL(TemplateParameter):
    """Fill level for the overrun reproducer, injected as #defines into build.h."""

    filler_count: int = 501
    nest_depth: int = 20

    def convert_to_cpp(self) -> str:
        return f"#define FILLER_COUNT {self.filler_count}\n#define NEST_DEPTH {self.nest_depth}"


@skip_for_coverage
@skip_for_quasar
@pytest.mark.parametrize("filler_count, nest_depth", [(501, 20), (400, 40)])
def test_profiler_buffer_overrun_into_neighbor(filler_count, nest_depth):
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip()

    config = PerfConfig(
        "sources/profiler_stress_overrun_test.cpp",
        templates=[OVERRUN_FILL(filler_count, nest_depth)],
    )
    config.generate_variant_hash()
    config.build_elfs()
    config.run_elf_files()
    config.wait_for_tensix_operations_finished()

    # reading unpack's buffer over the NoC flushes its data cache to L1, so any spill is
    # visible when we read the math buffer next.
    read_words_from_device(
        addr=TestConfig.THREAD_PERFORMANCE_DATA_BUFFER[0],
        word_count=0x400,
        location=TestConfig.TENSIX_LOCATION,
    )

    words = read_words_from_device(
        addr=TestConfig.THREAD_PERFORMANCE_DATA_BUFFER[1],
        word_count=16,
        location=TestConfig.TENSIX_LOCATION,
    )

    entries = []
    i = 0
    while i < len(words):
        word = int(words[i])
        if not (word & Profiler.ENTRY_EXISTS_BIT):
            break
        kind = (word & Profiler.ENTRY_TYPE_MASK) >> Profiler.ENTRY_TYPE_SHAMT
        marker_id = (word & Profiler.ENTRY_ID_MASK) >> Profiler.ENTRY_ID_SHAMT
        entries.append((i, marker_id))
        i += 4 if kind == EntryType.TIMESTAMP_DATA.value else 2

    word0_kind = (int(words[0]) & Profiler.ENTRY_TYPE_MASK) >> Profiler.ENTRY_TYPE_SHAMT
    kernel_id = (int(words[0]) & Profiler.ENTRY_ID_MASK) >> Profiler.ENTRY_ID_SHAMT
    foreign = [(idx, mid) for idx, mid in entries if mid != kernel_id]

    assert (
        word0_kind == EntryType.ZONE_START.value and not foreign
    ), f"math buffer corrupted: word0 kind=0x{word0_kind:x}, foreign entries={foreign}"


@skip_for_coverage
@skip_for_quasar
def test_profiler_overrun_crashes_normal_read():
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip()

    config = PerfConfig(
        "sources/profiler_stress_overrun_test.cpp",
        templates=[OVERRUN_FILL()],
    )
    config.generate_variant_hash()
    config.build_elfs()
    config.run_elf_files()
    config.wait_for_tensix_operations_finished()

    runtime = Profiler.get_data(
        config.test_name, config.variant_id, TestConfig.TENSIX_LOCATION
    )

    math_markers = set(str(m) for m in runtime.math().raw()["marker"])
    assert {"NEST", "FILLER"}.isdisjoint(
        math_markers
    ), f"math buffer contains unpack's markers {math_markers}"
