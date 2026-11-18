# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free checks of the LLK perf harness' RISC-V monitor, NoC interface and router counter families: the option
parsing, the compile defines, the L1 layout shared with counters.h and the decoding of the blocks the device writes.
"""

import re

import pandas as pd
import pytest
from helpers import counters
from helpers.counters import (
    LLK_COUNTERS_HEADER,
    CounterReadoutMissing,
    decode_riscv_block,
    decode_tile_block,
    parse_noc_groups,
    parse_riscv_events,
    parse_router_option,
    read_family_counters,
)
from helpers.llk_params import PerfRunType
from helpers.metrics import export_counters
from helpers.test_config import TestConfig
from tt_llk_perf import headers

TOKEN = 0x1234
FLAG = 0xFF | TOKEN << 8


@pytest.fixture
def families_off():
    saved = (
        TestConfig.PERF_RISCV_EVENTS,
        TestConfig.PERF_NOC_GROUPS,
        TestConfig.PERF_ROUTER_OPTION,
        TestConfig.ENABLE_PERF_COUNTERS,
    )
    TestConfig.PERF_RISCV_EVENTS = None
    TestConfig.PERF_NOC_GROUPS = None
    TestConfig.PERF_ROUTER_OPTION = None
    yield
    (
        TestConfig.PERF_RISCV_EVENTS,
        TestConfig.PERF_NOC_GROUPS,
        TestConfig.PERF_ROUTER_OPTION,
        TestConfig.ENABLE_PERF_COUNTERS,
    ) = saved


# --- option parsing -------------------------------------------------------------------------------------------------


def test_riscv_events_by_name_number_and_or():
    stall = headers.riscv_event_select("blackhole", "baby", ["BP_PC_STALL_ANY"])
    assert parse_riscv_events("BP_PC_STALL_ANY", "blackhole") == (stall, 0)
    assert parse_riscv_events("bp_pc_stall_any,0xA0000000", "blackhole") == (
        stall,
        0xA0000000,
    )
    both = headers.riscv_event_select(
        "blackhole", "baby", ["BP_PC_STALL_ANY", "EX_INST_RET"]
    )
    assert both == 0x80001
    assert parse_riscv_events("BP_PC_STALL_ANY+EX_INST_RET", "blackhole") == (both, 0)
    # the stall union of every pipeline domain is one select
    union = parse_riscv_events(
        "BP_PC_STALL_ANY+IF_STALL_ANY+DE_STALL_ANY+EX_STALL_ANY+LS_STALL_ANY",
        "blackhole",
    )
    assert union == (0x110111, 0)


def test_riscv_events_use_the_trisc_table_on_quasar():
    select = headers.riscv_event_select("quasar", "trisc", ["MEM_INS_BUF_WRITE"])
    assert parse_riscv_events("MEM_INS_BUF_WRITE", "quasar") == (select, 0)


def test_riscv_events_refuse_wormhole_unknown_names_and_bad_counts():
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="Wormhole RISC-V cores implement no"
    ):
        parse_riscv_events("0x1", "wormhole")
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="same pipeline domain"
    ):
        parse_riscv_events("BP_PC_STALL_ANY+BP_STALL_NO_ID_RTR", "blackhole")
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="not an event"
    ):
        parse_riscv_events("NOT_AN_EVENT", "blackhole")
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="one or two events"
    ):
        parse_riscv_events("0x1,0x2,0x3", "blackhole")
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="32 bits"
    ):
        parse_riscv_events("0x100000000", "blackhole")


def test_noc_groups():
    assert parse_noc_groups("MASTER", "blackhole") == 1
    assert parse_noc_groups("master+atomic", "wormhole") == 0b101
    assert parse_noc_groups("ALL", "blackhole") == 0b111
    assert parse_noc_groups("0x2", "blackhole") == 2
    for bad in ("0", "0x8", "READS"):
        with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
            ValueError, match="--perf-noc-counters"
        ):
            parse_noc_groups(bad, "blackhole")
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="Quasar"
    ):
        parse_noc_groups("ALL", "quasar")


def test_router_option():
    consts = headers.router_constants()
    assert parse_router_option("ALL", "blackhole") == (
        consts["VC_MASK_ALL"] | consts["BANKS_ALL"]
    )
    assert parse_router_option("0x10001", "blackhole") == 0x10001
    # 2 NoCs x 2 banks x 16 channels kept apart is 64 values, the record holds 32
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="at most"
    ):
        parse_router_option("ALL+PER_VC", "blackhole")
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="selects no bank"
    ):
        parse_router_option("0x1", "blackhole")
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="outside the router option mask"
    ):
        parse_router_option("0x80000000", "blackhole")
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="no router counters"
    ):
        parse_router_option("0x10001", "wormhole")
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        ValueError, match="Quasar"
    ):
        parse_router_option("ALL", "quasar")


# --- compile defines ------------------------------------------------------------------------------------------------


def test_defines_and_artefact_suffix(families_off, monkeypatch):
    assert TestConfig.perf_riscv_flags() == ""
    assert TestConfig.perf_tile_counter_flags() == ""
    TestConfig.ENABLE_PERF_COUNTERS = True
    base = TestConfig._perf_counters_suffix()
    TestConfig.PERF_RISCV_EVENTS = (0x10, 0xA0000000)
    TestConfig.PERF_NOC_GROUPS = 0b101
    TestConfig.PERF_ROUTER_OPTION = 0x10001
    assert TestConfig.perf_riscv_flags() == (
        " -DLLK_PERF_RISCV_PMC -DLLK_PERF_RISCV_EV3=0x10 -DLLK_PERF_RISCV_EV4=0xa0000000"
    )
    assert TestConfig.perf_tile_counter_flags() == (
        "-DLLK_PERF_NOC_GROUPS=0x5 -DLLK_PERF_ROUTER_OPTION=0x10001 "
    )
    # a family changes the compiled code, so its artefacts must not mix with another selection's
    suffix = TestConfig._perf_counters_suffix()
    assert suffix.startswith(base) and suffix != base
    assert "riscv10-a0000000" in suffix and "noc5" in suffix and "router10001" in suffix


# --- L1 layout shared with counters.h -------------------------------------------------------------------------------


def _header_value(text, name):
    match = re.search(rf"{name}\s*=\s*([^;]+);", text)
    assert match is not None, name
    return match.group(1)


def test_layout_matches_the_device_header():
    perf_h = (LLK_COUNTERS_HEADER.parent / "perf.h").read_text()

    def define(name):
        return int(
            re.search(rf"#define\s+{name}\s+(0x[0-9A-Fa-f]+|\d+)", perf_h).group(1), 0
        )

    assert define("PERF_COUNTERS_BASE_ADDR") == TestConfig.PERF_COUNTERS_BASE_ADDR
    assert (
        define("PERF_COUNTERS_CONFIG_WORDS") == TestConfig._PERF_COUNTERS_CONFIG_WORDS
    )
    assert define("PERF_COUNTERS_DATA_WORDS") == TestConfig._PERF_COUNTERS_DATA_WORDS
    assert (
        define("PERF_COUNTERS_RISCV_THREADS") == TestConfig._PERF_COUNTERS_RISCV_THREADS
    )
    assert (
        define("PERF_COUNTERS_RISCV_THREAD_WORDS")
        == TestConfig._PERF_COUNTERS_RISCV_THREAD_WORDS
    )
    assert (
        define("PERF_COUNTERS_TILE_VALUE_WORDS")
        == TestConfig._PERF_COUNTERS_TILE_VALUE_WORDS
    )
    # the launch block is the NoC and router flags and selects, then two value arrays
    assert re.search(
        r"PERF_COUNTERS_LAUNCH_BLOCK_WORDS\s+\(4 \+ 2 \* PERF_COUNTERS_TILE_VALUE_WORDS\)",
        perf_h,
    )
    header = LLK_COUNTERS_HEADER.read_text()
    assert "PERF_COUNTERS_ZONE_DATA_BYTES + 40 + PERF_COUNTERS_RISCV_BYTES" in header
    assert (
        TestConfig.PERF_COUNTERS_ZONE_SIZE
        == TestConfig._PERF_COUNTERS_ZONE_DATA_BYTES
        + 40
        + TestConfig._PERF_COUNTERS_RISCV_BYTES
    )


def test_layout_fits_below_the_profiler_region():
    end = (
        TestConfig.PERF_COUNTERS_LAUNCH_BLOCK_ADDR
        + (4 + 2 * TestConfig._PERF_COUNTERS_TILE_VALUE_WORDS) * 4
    )
    assert end <= 0x16AFF0
    assert TestConfig.PERF_COUNTERS_LAUNCH_BLOCK_ADDR == (
        TestConfig.PERF_COUNTERS_ZONES_BASE
        + TestConfig.PERF_COUNTERS_MAX_ZONES * TestConfig.PERF_COUNTERS_ZONE_SIZE
        + 4
    )


@pytest.mark.parametrize("arch", ("blackhole", "wormhole", "quasar"))
def test_the_tensix_inventory_fits_the_slots(arch):
    tables = headers.bank_tables(arch)
    l1 = max(
        (sum(1 for e in tables["L1"] if e.l1_mux == mux) for mux in range(8)), default=0
    )
    total = sum(len(v) for bank, v in tables.items() if bank != "L1") + max(l1, 1)
    assert total <= TestConfig._PERF_COUNTERS_CONFIG_WORDS
    assert total <= TestConfig._PERF_COUNTERS_DATA_WORDS


# --- decoding -------------------------------------------------------------------------------------------------------


def _riscv_words(ev3, ev4, cycles, instret, pmc3, pmc4, flag=FLAG):
    return [flag, ev3, ev4, cycles, instret, pmc3, pmc4, 0]


def test_riscv_block_rows_are_named_and_carry_cycles():
    stall = headers.riscv_event_select("blackhole", "baby", ["BP_PC_STALL_ANY"])
    rows = decode_riscv_block(
        _riscv_words(stall, 0, 1000, 600, 350, 0), "blackhole", "MATH", "ZONE_1", TOKEN
    )
    by_name = {r["counter_name"]: r for r in rows}
    assert set(by_name) == {"cycles", "instructions", "BP_PC_STALL_ANY"}
    assert all(r["bank"] == "RISCV_MATH" and r["zone"] == "ZONE_1" for r in rows)
    assert by_name["cycles"]["count"] == 1000
    assert by_name["instructions"]["count"] == 600
    assert by_name["BP_PC_STALL_ANY"]["count"] == 350
    assert by_name["BP_PC_STALL_ANY"]["counter_id"] == stall
    assert all(r["cycles"] == 1000 for r in rows)


def test_riscv_block_without_this_launchs_token_raises():
    stale = 0xFF | (TOKEN - 1) << 8
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        CounterReadoutMissing, match="RISC-V monitor block"
    ):
        decode_riscv_block(
            _riscv_words(0, 0, 1, 1, 0, 0, flag=stale),
            "blackhole",
            "UNPACK",
            "ZONE_1",
            TOKEN,
        )
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        CounterReadoutMissing
    ):
        decode_riscv_block(
            _riscv_words(0, 0, 1, 1, 0, 0, flag=0),
            "blackhole",
            "UNPACK",
            "ZONE_1",
            TOKEN,
        )


def _tile_words(noc_values=(), router_values=(), noc_select=0, router_select=0):
    n = TestConfig._PERF_COUNTERS_TILE_VALUE_WORDS
    words = [FLAG, noc_select, FLAG, router_select]
    words += list(noc_values) + [0] * (n - len(noc_values))
    words += list(router_values) + [0] * (n - len(router_values))
    return words


def test_tile_block_names_noc_and_router_values():
    selected = headers.noc_selected_counters("blackhole", 0b001)
    nocs = headers.noc_counter_table("blackhole").num_nocs
    values = list(range(1, len(selected) * nocs + 1))
    option = 0x10001
    router_labels = ["OUT_PORT1"]
    router_values = [7, 9]
    words = _tile_words(values, router_values, 0b001, option)
    rows = decode_tile_block(words, "blackhole", "ZONE_1", 0b001, option, TOKEN)
    assert {r["zone"] for r in rows} == {"LAUNCH_ZONE_1"}
    noc0 = {r["counter_name"]: r["count"] for r in rows if r["bank"] == "NOC0"}
    assert noc0[selected[0].name] == 1
    assert noc0[selected[-1].name] == len(selected)
    noc1 = {r["counter_name"]: r["count"] for r in rows if r["bank"] == "NOC1"}
    assert noc1[selected[0].name] == len(selected) + 1
    assert any(name.endswith("bytes") or "bytes" in name for name in noc0)
    router = {
        (r["bank"], r["counter_name"]): r["count"]
        for r in rows
        if r["bank"].startswith("ROUTER")
    }
    assert router == {
        ("ROUTER0", router_labels[0]): 7,
        ("ROUTER1", router_labels[0]): 9,
    }


def test_tile_block_with_another_selection_raises():
    selected = headers.noc_selected_counters("blackhole", 0b001)
    nocs = headers.noc_counter_table("blackhole").num_nocs
    words = _tile_words([0] * len(selected) * nocs, noc_select=0b010)
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        RuntimeError, match="group mask 0x2"
    ):
        decode_tile_block(words, "blackhole", "ZONE_1", 0b001, None, TOKEN)
    words = _tile_words(router_values=[1, 2], router_select=0x10002)
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        RuntimeError, match="option 0x10002"
    ):
        decode_tile_block(words, "blackhole", "ZONE_1", None, 0x10001, TOKEN)
    words = _tile_words(noc_select=0b001)
    words[0] = 0
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        CounterReadoutMissing, match="NoC counter block"
    ):
        decode_tile_block(words, "blackhole", "ZONE_1", 0b001, None, TOKEN)


def test_a_family_that_is_off_reads_nothing(families_off, monkeypatch):
    def no_device(*args, **kwargs):
        raise AssertionError("the device was read with every family off")

    monkeypatch.setattr(counters, "read_words_from_device", no_device)
    assert read_family_counters("0,0", 0b10, TOKEN).empty


def test_read_family_counters_reads_blocks_from_the_layout(families_off, monkeypatch):
    stall = headers.riscv_event_select("blackhole", "baby", ["BP_PC_STALL_ANY"])
    TestConfig.PERF_RISCV_EVENTS = (stall, 0)
    TestConfig.PERF_NOC_GROUPS = 0b001
    memory = {}
    for thread in range(3):
        addr = (
            TestConfig.PERF_COUNTERS_ZONES_BASE
            + 1 * TestConfig.PERF_COUNTERS_ZONE_SIZE
            + TestConfig._PERF_COUNTERS_RISCV_ZONE_OFFSET
            + thread * TestConfig._PERF_COUNTERS_RISCV_THREAD_WORDS * 4
        )
        memory[addr] = _riscv_words(stall, 0, 100 + thread, 50, 10 + thread, 0)
    selected = headers.noc_selected_counters("blackhole", 0b001)
    nocs = headers.noc_counter_table("blackhole").num_nocs
    memory[TestConfig.PERF_COUNTERS_LAUNCH_BLOCK_ADDR] = _tile_words(
        [3] * len(selected) * nocs, noc_select=0b001
    )

    def fake_read(location, addr, word_count):
        return memory[addr][:word_count]

    monkeypatch.setattr(counters, "read_words_from_device", fake_read)
    monkeypatch.setattr(counters, "get_chip_architecture", lambda: "blackhole")
    frame = read_family_counters("0,0", 0b10, TOKEN)
    riscv = frame[frame["bank"].str.startswith("RISCV_")]
    assert sorted(riscv["bank"].unique()) == [
        "RISCV_MATH",
        "RISCV_PACK",
        "RISCV_UNPACK",
    ]
    assert (
        riscv[riscv["bank"] == "RISCV_PACK"].set_index("counter_name")["count"][
            "cycles"
        ]
        == 102
    )
    assert (frame[frame["bank"] == "NOC0"]["count"] == 3).any()
    assert (frame["zone"].isin(["ZONE_1", "LAUNCH_ZONE_1"])).all()


def test_an_isolate_run_type_reads_its_measured_thread_only(families_off, monkeypatch):
    stall = headers.riscv_event_select("blackhole", "baby", ["BP_PC_STALL_ANY"])
    TestConfig.PERF_RISCV_EVENTS = (stall, 0)
    math = (
        TestConfig.PERF_COUNTERS_ZONES_BASE
        + 1 * TestConfig.PERF_COUNTERS_ZONE_SIZE
        + TestConfig._PERF_COUNTERS_RISCV_ZONE_OFFSET
        + 1 * TestConfig._PERF_COUNTERS_RISCV_THREAD_WORDS * 4
    )
    memory = {math: _riscv_words(stall, 0, 100, 50, 10, 0)}

    def fake_read(location, addr, word_count):
        return memory[addr][:word_count]  # any other thread's block is a KeyError

    monkeypatch.setattr(counters, "read_words_from_device", fake_read)
    monkeypatch.setattr(counters, "get_chip_architecture", lambda: "blackhole")
    frame = read_family_counters("0,0", 0b10, TOKEN, PerfRunType.MATH_ISOLATE)
    assert list(frame["bank"].unique()) == ["RISCV_MATH"]
    assert counters.riscv_threads_of(PerfRunType.L1_TO_L1, "blackhole") == [0, 1, 2]
    assert counters.riscv_threads_of(PerfRunType.L1_TO_L1, "quasar") == [0, 1, 2, 3]
    assert counters.riscv_threads_of(PerfRunType.SFPU_ISOLATE, "quasar") == [3]


# --- report ---------------------------------------------------------------------------------------------------------


def test_family_rows_reach_the_counters_csv_columns():
    stall = headers.riscv_event_select("blackhole", "baby", ["BP_PC_STALL_ANY"])
    rows = decode_riscv_block(
        _riscv_words(stall, 0, 1000, 600, 350, 0), "blackhole", "MATH", "ZONE_1", TOKEN
    )
    selected = headers.noc_selected_counters("blackhole", 0b001)
    nocs = headers.noc_counter_table("blackhole").num_nocs
    rows += decode_tile_block(
        _tile_words([4] * len(selected) * nocs, noc_select=0b001),
        "blackhole",
        "ZONE_1",
        0b001,
        None,
        TOKEN,
    )
    frame = pd.DataFrame(rows)
    frame["run_index"] = 0
    out = export_counters(frame, "L1_TO_L1", ["INIT", "TILE_LOOP"])
    assert set(out["marker"]) == {"TILE_LOOP", "LAUNCH_TILE_LOOP"}
    columns = set(out.columns)
    assert "L1_TO_L1_RISCV_MATH.cycles" in columns
    assert "L1_TO_L1_RISCV_MATH.instructions" in columns
    assert "L1_TO_L1_RISCV_MATH.BP_PC_STALL_ANY" in columns
    assert f"L1_TO_L1_NOC0.{selected[0].name}" in columns
    row = out[out["marker"] == "TILE_LOOP"].iloc[0]
    assert row["L1_TO_L1_RISCV_MATH.BP_PC_STALL_ANY"] == 350
