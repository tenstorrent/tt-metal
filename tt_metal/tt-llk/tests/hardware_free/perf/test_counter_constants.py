# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Check the host/device launch-word and Quasar register ABI without hardware."""

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import counters, test_config
from helpers.chip_architecture import ChipArchitecture
from helpers.perf import core as perf_core
from helpers.perf.core import COUNT_INIT, COUNT_TILE_LOOP, PerfConfig
from helpers.test_config import TestConfig

LLK = Path(__file__).resolve().parents[3]


def _compile_values(tmp_path, source, arch):
    # CXX may target the device; this probe must run on the host.
    compiler = shlex.split(os.environ.get("CXX_FOR_BUILD", "c++"))
    assert compiler and shutil.which(
        compiler[0]
    ), "Host C++ compiler is unavailable; install c++ or set CXX_FOR_BUILD to a host compiler"
    source_file = tmp_path / "constants.cpp"
    binary = tmp_path / "constants"
    source_file.write_text(source)
    result = subprocess.run(
        [
            *compiler,
            "-std=c++17",
            f"-DARCH_{arch.name}",
            f"-I{LLK / 'tools/include'}",
            str(source_file),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return list(map(int, subprocess.check_output([str(binary)], text=True).split()))


def _launch_constants(tmp_path, arch):
    header = counters.LLK_COUNTERS_HEADER.read_text()
    perf = (LLK / "tests/helpers/include/perf.h").read_text()
    layout = re.findall(r"^#define PERF_COUNTERS_.*$", perf, re.MULTILINE)
    layout += re.findall(
        r"constexpr std::uint32_t PERF_COUNTERS_(?!PENDING_ZONE_ADDR)\w+\s*=[^;]+;",
        header,
    )
    launch = re.search(
        r"constexpr std::uint32_t LAUNCH_TOKEN_SHIFT\s*=.*?"
        r"(?=inline __attribute__\(\(always_inline\)\) std::uint32_t launch_word\(\))",
        header,
        re.DOTALL,
    )
    assert launch, "Cannot locate launch-word declarations before launch_word()"
    zones = re.findall(
        r"constexpr std::uint32_t ZONE_ID_(?:INIT|TILE_LOOP)\s*=[^;]+;", header
    )
    assert len(zones) == 2, "Cannot locate INIT and TILE_LOOP zone declarations"
    return _compile_values(
        tmp_path,
        "#include <cstdint>\n#include <iostream>\n"
        + "\n".join(layout)
        + "\n"
        + launch.group()
        + "\n".join(zones)
        + '\nint main() { std::cout << COUNTED_ZONES_REG << " " << LAUNCH_TOKEN_SHIFT << " " << LAUNCH_WORD_MASK'
        + ' << " " << (1u << ZONE_ID_INIT) << " " << (1u << ZONE_ID_TILE_LOOP); }\n',
        arch,
    )


@pytest.mark.parametrize("arch", list(ChipArchitecture))
def test_launch_word_address(tmp_path, monkeypatch, arch):
    """Compile the device declarations, including Quasar's derived L1 address."""
    address, _, _, init_mask, tile_loop_mask = _launch_constants(tmp_path, arch)
    assert (COUNT_INIT, COUNT_TILE_LOOP) == (init_mask, tile_loop_mask)
    monkeypatch.setattr(TestConfig, "CHIP_ARCH", arch, raising=False)
    monkeypatch.setattr(TestConfig, "runs_on_ttsim", staticmethod(lambda: False))
    assert address == TestConfig.counted_zones_address()


@pytest.mark.parametrize(
    "arch", [ChipArchitecture.WORMHOLE, ChipArchitecture.BLACKHOLE]
)
def test_launch_word_matches_metal(monkeypatch, arch):
    """Pin the host/device scratch address to metal's hardware register definitions."""
    header_path = (
        LLK.parent
        / "hw/inc/internal/tt-1xx"
        / arch.name.lower()
        / "noc/noc_overlay_parameters.h"
    )
    if not (LLK.parent / "api/tt-metalium/host_api.hpp").is_file():
        pytest.skip("Standalone LLK checkout has no metal hardware tree")
    assert header_path.is_file(), f"Missing metal register header: {header_path}"
    header = header_path.read_text()
    values = {}
    for name in ("NOC_OVERLAY_START_ADDR", "STREAM_SCRATCH_1_REG_INDEX"):
        match = re.search(
            rf"^#define {name}\s+(0x[0-9a-fA-F]+|\d+)\s*$", header, re.MULTILINE
        )
        assert match, f"Cannot read {name} from {header_path}"
        values[name] = int(match[1], 0)
    monkeypatch.setattr(TestConfig, "CHIP_ARCH", arch, raising=False)
    monkeypatch.setattr(TestConfig, "runs_on_ttsim", staticmethod(lambda: False))
    assert TestConfig.counted_zones_address() == (
        values["NOC_OVERLAY_START_ADDR"] + 4 * values["STREAM_SCRATCH_1_REG_INDEX"]
    )


def test_quasar_host_registers(tmp_path, monkeypatch):
    """Pin host NEO0 configure writes to the device's registers, mode and period."""
    values = _compile_values(
        tmp_path,
        """
#include <iostream>
#include "perf_counters/registers.h"
int main() {
    using namespace llk::perf;
    std::cout << neo_window(0) << " " << REF_PERIOD_MAX << " " << MODE_CONTINUOUS;
    for (unsigned i = 0; i < NUM_BANKS; ++i) {
        const auto regs = bank_regs(static_cast<Bank>(i), neo_window(0));
        if (regs.control) std::cout << " " << i << " " << regs.ref_period << " " << regs.mode;
    }
}
""",
        ChipArchitecture.QUASAR,
    )
    window, period, mode, *registers = values
    assert window == counters._QUASAR_NEO0_WINDOW
    assert {
        bank: (ref_period - window, mode - window)
        for bank, ref_period, mode in zip(
            registers[::3], registers[1::3], registers[2::3]
        )
    } == counters._QUASAR_BANK_REGS
    writes = []
    monkeypatch.setattr(
        counters,
        "write_words_to_device",
        lambda location, addr, words, safe_mode: writes.append(
            (location, addr, words, safe_mode)
        ),
    )
    counters.configure_quasar_counters("0,0")
    assert writes == [
        ("0,0", address, [value], False)
        for ref_period, mode_reg in zip(registers[1::3], registers[2::3])
        for address, value in ((ref_period, period), (mode_reg, mode))
    ]


@pytest.mark.parametrize("arch", list(ChipArchitecture))
def test_launch_word_readback(tmp_path, monkeypatch, arch):
    """Write a launch word, accept its readout, reject stale tokens and check wraparound."""
    address, shift, mask, _, _ = _launch_constants(tmp_path, arch)
    max_token = mask >> shift
    monkeypatch.setattr(TestConfig, "CHIP_ARCH", arch, raising=False)
    monkeypatch.setattr(TestConfig, "TENSIX_LOCATION", "0,0", raising=False)
    monkeypatch.setattr(TestConfig.TEST_TARGET, "run_simulator", False)
    writes = []

    def write(location, addr, words, safe_mode):
        writes.append((location, addr, words, safe_mode))

    memory = {}

    def read(location, addr, word_count):
        assert location == "0,0"
        words = memory[addr]
        assert len(words) == word_count
        return words

    monkeypatch.setattr(test_config, "write_words_to_device", write)
    monkeypatch.setattr(counters, "write_words_to_device", write)
    monkeypatch.setattr(counters, "read_words_from_device", read)
    sync_address = counters._zone_sync_ctrl_addr(1)
    memory[counters._zone_config_addr(1)] = [0x80000000] + [0] * (
        counters.COUNTER_SLOT_COUNT - 1
    )
    memory[counters._zone_data_addr(1)] = [101] * 5 + [17]
    for previous in (0, 0x1234, max_token - 1, max_token):
        monkeypatch.setattr(TestConfig, "_LAUNCH_TOKEN", previous)
        launch = SimpleNamespace(launch_token=0)
        writes.clear()
        TestConfig._write_launch_word(launch, 1 << 1)
        token = previous % max_token + 1
        assert launch.launch_token == token
        assert writes[0] == ("0,0", address, [(token << shift) | (1 << 1)], False)
        memory[sync_address] = [(token << shift) | 0xFF]
        result = counters.read_counters("0,0", zones=1 << 1, token=token)
        assert result[["zone", "cycles", "count"]].to_dict("records") == [
            {"zone": "ZONE_1", "cycles": 101, "count": 17}
        ]
        memory[sync_address] = [((token ^ 1) << shift) | 0xFF]
        with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
            counters.CounterReadoutMissing, match="not this launch's readout"
        ):
            counters.read_counters("0,0", zones=1 << 1, token=token)


@pytest.mark.parametrize(
    "arch,run_simulator,zones,write_count,configure_count",
    [
        (ChipArchitecture.QUASAR, True, COUNT_TILE_LOOP, 0, 0),
        (ChipArchitecture.QUASAR, False, 0, 1, 0),
        (ChipArchitecture.QUASAR, False, COUNT_TILE_LOOP, 1, 1),
        (ChipArchitecture.WORMHOLE, False, COUNT_TILE_LOOP, 1, 0),
        (ChipArchitecture.BLACKHOLE, False, COUNT_TILE_LOOP, 1, 0),
    ],
)
def test_counter_launch_io_guards(
    monkeypatch, arch, run_simulator, zones, write_count, configure_count
):
    """Suppress ttsim I/O and configure Quasar only for a counted launch."""
    monkeypatch.setattr(TestConfig, "CHIP_ARCH", arch, raising=False)
    monkeypatch.setattr(TestConfig, "TENSIX_LOCATION", "0,0", raising=False)
    monkeypatch.setattr(TestConfig, "_LAUNCH_TOKEN", 0)
    monkeypatch.setattr(TestConfig.TEST_TARGET, "run_simulator", run_simulator)
    monkeypatch.setenv("TT_METAL_SIMULATOR", "/unused/libttsim.so")
    writes, configures, reads, results = [], [], [], []
    monkeypatch.setattr(
        test_config, "write_words_to_device", lambda *a, **k: writes.append((a, k))
    )
    monkeypatch.setattr(counters, "configure_quasar_counters", configures.append)
    monkeypatch.setattr(
        perf_core, "read_counters", lambda **kwargs: reads.append(kwargs) or {}
    )
    launch = SimpleNamespace(
        launch_token=0,
        counted_zones=zones,
        _has_counter_readback=PerfConfig._has_counter_readback,
    )
    TestConfig._write_launch_word(launch, zones)
    PerfConfig._read_counters_of_launch(launch, results, run_index=0)
    assert len(writes) == write_count
    assert configures == ["0,0"] * configure_count
    assert (
        reads
        == [dict(location="0,0", zones=zones, token=launch.launch_token)] * write_count
    )
    assert results == [{"run_index": 0}] * write_count
