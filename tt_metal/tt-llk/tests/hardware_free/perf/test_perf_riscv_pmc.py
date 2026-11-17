# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free checks of the RISC-V performance monitor tables and decoder in tt_llk_perf."""

import re
from pathlib import Path

import pytest
from helpers.perf.marker_rows import marker_row, tracy_analysis
from tt_llk_perf import headers, riscv_metrics, riscv_pmc


@pytest.fixture(scope="module")
def event_names():
    return headers.riscv_event_names()


def test_enum_names_are_unique_and_start_with_undef(event_names):
    assert event_names[0] == "UNDEF"
    assert len(set(event_names)) == len(event_names)


@pytest.mark.parametrize("arch", ("blackhole", "quasar"))
def test_every_table_name_is_declared_and_selects_are_unique(arch, event_names):
    for kind, entries in headers.riscv_event_tables(arch).items():
        names = [n for n, _ in entries]
        selects = [s for _, s in entries]
        assert set(names) <= set(event_names), kind
        assert len(set(names)) == len(names), kind
        assert len(set(selects)) == len(selects), kind
        assert all(s != 0 for s in selects), kind


def test_baby_selects_are_one_nibble_per_domain():
    for _, select in headers.riscv_event_tables("blackhole")["baby"]:
        position = (select.bit_length() - 1) // 4 * 4
        assert select >> position < 16 and select & ((1 << position) - 1) == 0


def test_dm_selects_are_one_mask_bit_of_one_set():
    for _, select in headers.riscv_event_tables("quasar")["dm"]:
        mask = select >> 8
        assert mask and mask & (mask - 1) == 0
        assert select & 0xFF in (0, 1, 2)


def test_quasar_trisc_is_the_baby_table_plus_two_if_events():
    baby = dict(headers.riscv_event_tables("blackhole")["baby"])
    trisc = dict(headers.riscv_event_tables("quasar")["trisc"])
    assert len(trisc) == len(baby) + 2
    assert {k: v for k, v in trisc.items() if k in baby} == baby
    assert set(trisc) - set(baby) == {"IF_STALL_NO_IF_RTS", "IF_ID_STALL_LD"}
    assert trisc["IF_STALL_NO_IF_RTS"] == 0x70 and trisc["IF_ID_STALL_LD"] == 0x80


def test_known_selects_decode_to_names():
    assert headers.riscv_select_events("blackhole", "baby", 0xA0000000) == [
        "MEM_INS_BUF_WRITE"
    ]
    assert headers.riscv_select_events("blackhole", "baby", 0x0000D000) == [
        "DE1_VALID_TT_INSTR"
    ]
    assert headers.riscv_select_events("blackhole", "baby", 0x10110111) == [
        "BP_PC_STALL_ANY",
        "IF_STALL_ANY",
        "DE_STALL_ANY",
        "EX_STALL_ANY",
        "LS_STALL_ANY",
        "MEM_STALL_ANY",
    ]
    assert headers.riscv_select_events("blackhole", "baby", 0) == []
    assert headers.riscv_select_events("quasar", "dm", 0x201) == [
        "DM_PIPE_LONG_LATENCY"
    ]
    assert headers.riscv_select_events("quasar", "dm", 0x1801) == [
        "DM_PIPE_ICACHE_BLOCKED",
        "DM_PIPE_DCACHE_BLOCKED",
    ]
    assert headers.riscv_select_events("quasar", "dm", 0x202) == ["DM_MEM_DCACHE_MISS"]


def test_select_round_trip():
    for arch in ("blackhole", "quasar"):
        for kind, entries in headers.riscv_event_tables(arch).items():
            for name, select in entries:
                assert headers.riscv_event_select(arch, kind, [name]) == select
                assert name in headers.riscv_select_events(arch, kind, select)


def test_events_that_share_a_select_field_combine():
    select = headers.riscv_event_select(
        "quasar", "dm", ["DM_PIPE_LOAD_USE", "DM_PIPE_FLUSH"]
    )
    assert select == 0x8101
    assert headers.riscv_select_events("quasar", "dm", select) == [
        "DM_PIPE_LOAD_USE",
        "DM_PIPE_FLUSH",
    ]


@pytest.mark.parametrize(
    "arch, kind, names",
    [
        ("blackhole", "baby", ["BP_PC_STALL_ANY", "BP_STALL_NO_ID_RTR"]),
        ("quasar", "dm", ["DM_COMMIT_LOAD", "DM_PIPE_FLUSH"]),
    ],
)
def test_events_that_need_different_values_of_one_field_raise(arch, kind, names):
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        headers.riscv_event_select(arch, kind, names)


def test_unknown_kind_and_event_raise():
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        headers.riscv_select_events("blackhole", "dm", 1)
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        headers.riscv_event_select("blackhole", "baby", ["NOPE"])


def test_unknown_select_bits_are_labelled_and_counted():
    # IF has no event 15 on Blackhole, and DM set 1 has no mask bit 10.
    launch = riscv_pmc.unpack_launch(
        (7 << 32) | 9, (100 << 32) | 1000, (0x40001 << 32) | 0xA00000F0
    )
    decoded = riscv_pmc.decode_launch("blackhole", "baby", launch)
    assert decoded["event3"] == "MEM_INS_BUF_WRITE|UNKNOWN_0xf0"
    assert decoded["metrics"][
        "riscv_mem_ins_buf_write_or_unknown_0xf0_pct"
    ] == pytest.approx(0.9)
    dm = riscv_pmc.decode_launch("quasar", "dm", launch)
    assert dm["event4"] == "UNKNOWN_0x40001"
    assert dm["metrics"]["riscv_unknown_0x40001_pct"] == pytest.approx(0.7)


def test_stall_pct_needs_the_whole_stall_union():
    union = "|".join(sorted(riscv_metrics.STALL_UNION))
    assert (
        riscv_metrics.compute_riscv_metrics(1000, 500, {union: 400})["riscv_stall_pct"]
        == 40.0
    )
    one = riscv_metrics.compute_riscv_metrics(1000, 500, {"LS_STALL_ANY": 400})
    assert one["riscv_stall_pct"] is None and one["riscv_ls_stall_any_pct"] == 40.0


def test_core_kind():
    assert riscv_pmc.core_kind("blackhole", "TRISC_1") == "baby"
    assert riscv_pmc.core_kind("blackhole", "ERISC") == "baby"
    assert riscv_pmc.core_kind("blackhole", "UNKNOWN") is None
    assert riscv_pmc.core_kind("quasar", "QUASAR_NEO2_TRISC3") == "trisc"
    assert riscv_pmc.core_kind("quasar", "QUASAR_DM0") == "dm"
    assert riscv_pmc.core_kind("wormhole", "BRISC") is None


def test_decode_launch_blackhole():
    launch = riscv_pmc.unpack_launch(
        (100 << 32) | 400, (500 << 32) | 1000, (0xA0000000 << 32) | 0x10110111
    )
    decoded = riscv_pmc.decode_launch("blackhole", "baby", launch)
    assert decoded["event4"] == "MEM_INS_BUF_WRITE"
    m = decoded["metrics"]
    assert m["riscv_ipc_ratio"] == 0.5
    assert m["riscv_stall_pct"] == 40.0
    assert m["riscv_mem_ins_buf_write_pct"] == 10.0
    assert m["riscv_mem_ins_buf_write_per_instr_ratio"] == 0.2


def test_decode_launch_dm_with_idle_counters():
    launch = riscv_pmc.unpack_launch(0, (300 << 32) | 600, 0)
    decoded = riscv_pmc.decode_launch("quasar", "dm", launch)
    assert decoded["event3"] == "DISABLED" and decoded["event4"] == "DISABLED"
    assert decoded["event5"] == "DISABLED" and decoded["event6"] == "DISABLED"
    assert decoded["metrics"]["riscv_ipc_ratio"] == 0.5
    assert "event5" not in riscv_pmc.decode_launch(
        "blackhole", "baby", riscv_pmc.unpack_launch(0, 0, 0)
    )


def test_decode_launch_dm_with_four_counters():
    launch = riscv_pmc.unpack_launch(
        (40 << 32) | 30,
        (300 << 32) | 600,
        (0x801 << 32) | 0x201,
        (60 << 32) | 50,
        (0x402 << 32) | 0x202,
    )
    assert launch.pmc5 == 50 and launch.pmc6 == 60
    assert launch.event5 == 0x202 and launch.event6 == 0x402
    decoded = riscv_pmc.decode_launch("quasar", "dm", launch)
    assert decoded["event5"] == "DM_MEM_DCACHE_MISS"
    assert decoded["event6"] == "DM_MEM_DCACHE_RELEASE"
    m = decoded["metrics"]
    assert m["riscv_dm_mem_dcache_miss_pct"] == pytest.approx(50 / 600 * 100)
    assert m["riscv_dm_mem_dcache_release_per_instr_ratio"] == 0.2
    assert m["riscv_dm_pipe_long_latency_pct"] == pytest.approx(30 / 600 * 100)


def test_mem_stall_per_tti_push_needs_both_counters():
    both = riscv_metrics.compute_riscv_metrics(
        1000, 500, {"MEM_STALL_ANY": 30, "MEM_INS_BUF_WRITE": 10}
    )
    assert both["riscv_mem_stall_per_tti_push_ratio"] == 3.0
    only = riscv_metrics.compute_riscv_metrics(1000, 500, {"MEM_STALL_ANY": 30})
    assert only["riscv_mem_stall_per_tti_push_ratio"] is None


def test_metrics_absent_inputs_read_none_not_zero():
    m = riscv_metrics.compute_riscv_metrics(0, 0, {"LS_LOAD_OP": 5})
    assert m["riscv_ipc_ratio"] is None
    assert (
        m["riscv_ls_load_op_pct"] is None
        and m["riscv_ls_load_op_per_instr_ratio"] is None
    )


def test_tracy_decoder_pairs_streams_and_names_events():
    analysis = tracy_analysis()
    rows = []
    for run, (c3, c4, cyc, ins) in enumerate(
        [(400, 100, 1000, 500), (50, 25, 100, 40)], start=7
    ):
        t = run * 100
        rows += [
            marker_row(run, "TRISC_1", 9092, t, (c4 << 32) | c3),
            marker_row(run, "TRISC_1", 9093, t + 1, (ins << 32) | cyc),
            marker_row(run, "TRISC_1", 9094, t + 2, (0xA0000000 << 32) | 0x10110111),
        ]
    rows.append(marker_row(7, "BRISC", 38106, 5, 0))
    df = analysis.decode_riscv_pmc_markers(rows, "blackhole")
    assert list(df["run_host_id"]) == [7, 8]
    assert set(df["event4"]) == {"MEM_INS_BUF_WRITE"}
    assert list(df["riscv_ipc_ratio"]) == [0.5, 0.4]
    assert list(df["mcycle"]) == [1000, 100]


def test_tracy_decoder_keeps_chips_apart():
    analysis = tracy_analysis()
    rows = []
    for device, (cyc, ins) in enumerate([(1000, 500), (2000, 600)]):
        rows += [
            marker_row(7, "TRISC_1", 9092, 100, 0, device=device),
            marker_row(7, "TRISC_1", 9093, 101, (ins << 32) | cyc, device=device),
            marker_row(7, "TRISC_1", 9094, 102, 0, device=device),
        ]
    df = analysis.decode_riscv_pmc_markers(rows, "blackhole")
    assert list(df["device_id"]) == [0, 1]
    assert list(df["mcycle"]) == [1000, 2000]
    assert list(df["minstret"]) == [500, 600]


def test_tracy_decoder_skips_cores_without_the_monitor():
    analysis = tracy_analysis()
    rows = [marker_row(1, "BRISC", i, 1, 0) for i in (9092, 9093, 9094)]
    assert analysis.decode_riscv_pmc_markers(rows, "wormhole") is None
    rows = [marker_row(1, "ERISC", i, 1, 0) for i in (9092, 9093, 9094)]
    assert analysis.decode_riscv_pmc_markers(rows, "wormhole") is None


def test_tracy_decoder_names_blackhole_ethernet_events():
    analysis = tracy_analysis()
    rows = [
        marker_row(3, "ERISC", 9092, 10, (7 << 32) | 9),
        marker_row(3, "ERISC", 9093, 11, (400 << 32) | 1000),
        marker_row(3, "ERISC", 9094, 12, (0xA0000000 << 32) | 0x10110111),
    ]
    df = analysis.decode_riscv_pmc_markers(rows, "blackhole")
    assert list(df["risc_type"]) == ["ERISC"]
    assert df["event3"][0].startswith("BP_PC_STALL_ANY|")
    assert df["mcycle"][0] == 1000 and df["riscv_ipc_ratio"][0] == 0.4


def test_tracy_decoder_keeps_ethernet_cores_apart():
    analysis = tracy_analysis()
    rows = []
    for core, cycles in (((9, 1), 1000), ((9, 6), 2000)):
        rows += [
            marker_row(4, "ERISC", 9092, 10, (7 << 32) | 9, core),
            marker_row(4, "ERISC", 9093, 11, (400 << 32) | cycles, core),
            marker_row(4, "ERISC", 9094, 12, (0xA0000000 << 32) | 0x10110111, core),
        ]
    df = analysis.decode_riscv_pmc_markers(rows, "blackhole")
    assert sorted(df["mcycle"]) == [1000, 2000]
    assert set(df["risc_type"]) == {"ERISC"}


def test_tracy_decoder_pairs_the_upper_dm_markers():
    analysis = tracy_analysis()
    rows = []
    for n in range(2):
        t = 100 * (n + 1)
        rows += [
            marker_row(5, "QUASAR_DM2", 9092, t, (n << 32) | 3),
            marker_row(5, "QUASAR_DM2", 9093, t + 1, (300 << 32) | 600),
            marker_row(5, "QUASAR_DM2", 9094, t + 2, (0x801 << 32) | 0x201),
            marker_row(5, "QUASAR_DM2", 9095, t + 3, ((7 + n) << 32) | 9),
            marker_row(5, "QUASAR_DM2", 9096, t + 4, (0x402 << 32) | 0x202),
        ]
    df = analysis.decode_riscv_pmc_markers(rows, "quasar")
    assert len(df) == 2
    assert set(df["event5"]) == {"DM_MEM_DCACHE_MISS"}
    assert set(df["event6"]) == {"DM_MEM_DCACHE_RELEASE"}
    assert list(df["riscv_dm_mem_dcache_release_pct"]) == [
        pytest.approx(7 / 6),
        pytest.approx(8 / 6),
    ]


def test_tracy_decoder_without_upper_markers_leaves_them_idle():
    analysis = tracy_analysis()
    rows = [
        marker_row(1, "QUASAR_DM3", 9092, 10, 3),
        marker_row(1, "QUASAR_DM3", 9093, 11, (300 << 32) | 600),
        marker_row(1, "QUASAR_DM3", 9094, 12, 0x201),
    ]
    df = analysis.decode_riscv_pmc_markers(rows, "quasar")
    assert df["event5"][0] == "DISABLED" and df["event6"][0] == "DISABLED"


def test_tracy_decoder_short_stream_pairs_what_it_can():
    analysis = tracy_analysis()
    rows = [
        marker_row(1, "QUASAR_DM0", 9092, 10, 3),
        marker_row(1, "QUASAR_DM0", 9093, 11, (300 << 32) | 600),
        marker_row(1, "QUASAR_DM0", 9094, 12, 0x201),
        marker_row(1, "QUASAR_DM0", 9092, 20, 3),
    ]
    df = analysis.decode_riscv_pmc_markers(rows, "quasar")
    assert len(df) == 1 and df["event3"][0] == "DM_PIPE_LONG_LATENCY"


def _firmware(name):
    path = Path(__file__).resolve().parents[4] / "hw/firmware/src/tt-1xx" / name
    if not path.is_file():
        pytest.skip("not inside a tt-metal checkout")
    return path.read_text()


@pytest.mark.parametrize(
    "firmware, hooked",
    (
        ("brisc.cc", True),
        ("ncrisc.cc", True),
        ("trisc.cc", True),
        ("active_erisc.cc", True),
        ("idle_erisc.cc", True),
        ("subordinate_erisc.cc", False),
        ("drisc.cc", False),
    ),
)
def test_firmware_hooks_match_the_cores_the_host_reads(firmware, hooked):
    text = _firmware(firmware)
    assert ("RiscvPmcBegin()" in text) == hooked
    assert ("RiscvPmcEnd()" in text) == hooked
