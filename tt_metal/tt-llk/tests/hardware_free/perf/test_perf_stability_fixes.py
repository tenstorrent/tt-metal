# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Each test fails when a merged LLK perf stability fix is removed. If you change one on purpose,
# show in the same PR that two runs of one commit stay bit-exact, then update the test.

import pathlib
import re

import yaml

_TESTS = pathlib.Path(__file__).parents[2]
_REPO = pathlib.Path(__file__).parents[5]
_RUNNERS = ("run_llk_perf_wormhole.sh", "run_llk_perf_blackhole.sh")
_INCLUDE = _TESTS / "helpers" / "include"


def _read(path):
    return path.read_text()


def test_measuring_pass_takes_one_test_at_a_time():
    for runner in (*_RUNNERS, "run_llk_perf_placement.sh"):
        passes = [
            l for l in _read(_TESTS / runner).splitlines() if "--compile-consumer" in l
        ]
        assert passes and all("--dist loadgroup" in l for l in passes), (
            f"{runner}: the measuring pass (--compile-consumer) lost '--dist loadgroup' (#57015). "
            "Without it xdist gives each worker a block of tests, so a test runs after other tests "
            "on its core in each run and measures differently: 6,078 BH points moved on one commit."
        )


def test_scheduling_chunk_stays_small():
    chunk = re.search(
        r"--maxschedchunk[= ](\d+)", _read(_TESTS / "python_tests" / "pytest.ini")
    )
    assert chunk and int(chunk.group(1)) <= 200, (
        "pytest.ini: --maxschedchunk must stay at 200 or less. #53642 (fd8a72b6cbd) raised it from "
        "10 to 2000 and BH lost repeatability (per-core test order); #57015 holds the measuring pass."
    )


def test_perf_runs_build_without_asserts():
    for runner in (*_RUNNERS, "run_llk_perf_placement.sh"):
        assert re.search(
            r'^export TT_LLK_DISABLE_ASSERTS="\$\{TT_LLK_DISABLE_ASSERTS:-1\}"',
            _read(_TESTS / runner),
            re.M,
        ), (
            f"{runner}: export TT_LLK_DISABLE_ASSERTS=1 (#56746). Assert builds compile other INIT code "
            "and give false INIT moves of 1-8 cycles."
        )
    impl = yaml.safe_load(_read(_REPO / ".github/workflows/llk-perf-impl.yaml"))
    steps = [s for job in impl["jobs"].values() for s in job.get("steps", [])]
    for name in ("Compile", "Run tests"):
        step = next(s for s in steps if s.get("name") == name)
        assert step.get("env", {}).get("TT_LLK_DISABLE_ASSERTS") == "1", (
            f"llk-perf-impl.yaml step '{name}': set TT_LLK_DISABLE_ASSERTS: \"1\" (#56746). Assert builds "
            "compile other INIT code and give false INIT moves."
        )


def test_debug_feature_bits_are_cleared_before_each_launch():
    brisc = _read(_TESTS / "helpers" / "src" / "brisc.cpp")
    clear = brisc.find("llk_perf::clear_debug_features();")
    assert 0 <= clear < brisc.find("clear_trisc_soft_reset();"), (
        "brisc.cpp: call clear_debug_features() before the TRISCs start, in both builds (#60892). A debug "
        "feature bit left set by an earlier kernel made configs measure fast or slow by test history (#45727)."
    )
    llk = _TESTS.parent
    writers = [
        p.relative_to(llk)
        for arch in ("tt_llk_wormhole_b0", "tt_llk_blackhole")
        for p in (llk / arch).rglob("*.h")
        if "DBG_FEATURE_DISABLE" in _read(p) and not p.name.startswith("llk_math")
    ]
    assert not writers, (
        f"{writers}: only the math thread may write DBG_FEATURE_DISABLE (#45727). An unpacker write "
        "without read-modify-write leaked bit 11 from kernel to kernel."
    )


def test_one_barrier_for_both_builds():
    for header in ("profiler.h", "counters.h"):
        text = _read(_INCLUDE / header)
        assert not re.search(r"\b(sync_point|sem_rendezvous)\b", text), (
            f"{header}: use llk_barrier::rendezvous (barrier.h, #52442). A second rendezvous gave each "
            "zone a different head start: -1 cycle per tile on perf_unpack_tilize."
        )
    assert "llk_barrier::rendezvous" in _read(_INCLUDE / "profiler.h")


def test_zone_records_are_written_after_the_end_read():
    text = _read(_INCLUDE / "profiler.h")
    start = text.index("class zone_scoped")
    ctor, dtor = text.index("zone_scoped()", start), text.index("~zone_scoped()", start)
    end = text.index("\n    }\n", dtor)
    body = text[dtor:end]
    assert "write_entry" not in text[ctor:dtor] and body.index(
        "read_wall_clock()"
    ) < body.index("write_entry(EntryType::ZONE_START"), (
        "profiler.h: write both zone records after the closing wall clock read (#63817). A record "
        "written inside the window is an L1 store in every measured window."
    )


def test_launch_ends_without_l1_polling():
    brisc = _read(_TESTS / "helpers" / "src" / "brisc.cpp")
    device = _read(_TESTS / "python_tests" / "helpers" / "device.py")
    assert (
        "load_blocking(pc_buf)" in brisc
        and "host_signal::launch_done()" in brisc
        and "def launch_done_address" in device
    ), (
        "brisc.cpp/device.py: BRISC waits on the TRISC PC buffers and marks the launch done in an overlay "
        "register the host polls (#63818). Other polls during a kernel move values on the same build: "
        "the soft reset register moved 37 of 122,760 WH and 85 of 462,266 BH values."
    )
