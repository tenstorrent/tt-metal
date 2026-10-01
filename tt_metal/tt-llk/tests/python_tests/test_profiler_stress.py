# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import pytest
from conftest import skip_for_coverage
from helpers.device_io import read_words_from_device
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.profiler import EntryType, Profiler
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import TemplateParameter


@dataclass
class OVERRUN_FILL(TemplateParameter):
    """Fill level for the overrun reproducer, injected as constexprs into build.h."""

    filler_count: int = 501
    nest_depth: int = 20

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t FILLER_COUNT = {self.filler_count};\n"
            f"constexpr std::uint32_t NEST_DEPTH = {self.nest_depth};"
        )


def _build_and_run(config):
    # This is a test of the profiler itself and doesn't use configuration.run method at all,
    # therefore it can't leverage default producer-consumer separation of compile and execute phases.
    # In order to avoid compiling the test elf twice we run it in only one of two phases - the consumer/execute phase,
    # where everything is done.
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip()

    config.generate_variant_hash()
    config.build_elfs()
    config.run_elf_files()
    config.wait_for_tensix_operations_finished()


@skip_for_coverage
@parametrize(
    filler_count=[501, 400],
    nest_depth=lambda filler_count: 20 if filler_count == 501 else 40,
)
def test_profiler_buffer_overflow_into_neighbor(filler_count, nest_depth):
    config = PerfConfig(
        "sources/profiler_stress_overrun_test.cpp",
        templates=[OVERRUN_FILL(filler_count, nest_depth)],
    )
    _build_and_run(config)

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
def test_profiler_overflow_absent_from_math_read():
    config = PerfConfig(
        "sources/profiler_stress_overrun_test.cpp",
        templates=[OVERRUN_FILL()],
    )
    _build_and_run(config)

    runtime = Profiler.get_data(
        config.test_name, config.variant_id, TestConfig.TENSIX_LOCATION
    )

    math_markers = set(str(m) for m in runtime.math().raw()["marker"])
    assert {"NEST", "FILLER"}.isdisjoint(
        math_markers
    ), f"math buffer contains unpack's markers {math_markers}"
