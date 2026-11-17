# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free checks of the ethernet counter tables and the ethernet traffic decoder in tt_llk_perf."""

import re
from pathlib import Path

import pytest
from tt_llk_perf import eth_counters, headers

TX_QUEUE = {
    "TXQ_TRANSFER_CNT": 0x30,
    "TXQ_PKT_START_CNT": 0x34,
    "TXQ_PKT_END_CNT": 0x3C,
    "TXQ_WORD_CNT": 0x40,
    "TXQ_RESEND_CNT": 0x68,
}
RX_QUEUE_COMMON = {
    "RXQ_WORD_CNT": 0x14,
    "RXQ_PKT_START_CNT": 0x24,
    "RXQ_PKT_END_CNT": 0x28,
    "RXQ_PKT_DROP_CNT": 0x4C,
}
NO_QUEUE = eth_counters.NO_QUEUE
EXPECTED_L1 = {"blackhole": 25, "wormhole": 4, "quasar": 0}


@pytest.fixture(scope="module")
def names():
    return headers.eth_counter_names()


def test_enum_names_are_unique_and_start_with_undef(names):
    assert names[0] == "UNDEF"
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("arch", ("blackhole", "wormhole"))
def test_queue_tables(arch):
    assert dict(headers.eth_tx_queue_table(arch)) == TX_QUEUE
    rx = dict(headers.eth_rx_queue_table(arch))
    assert {k: rx[k] for k in RX_QUEUE_COMMON} == RX_QUEUE_COMMON
    assert set(rx) - set(RX_QUEUE_COMMON) <= {"RXQ_BYTE_CNT"}


def test_blackhole_counts_payload_bytes_and_wormhole_does_not():
    assert dict(headers.eth_rx_queue_table("blackhole"))["RXQ_BYTE_CNT"] == 0x04
    assert "RXQ_BYTE_CNT" not in dict(headers.eth_rx_queue_table("wormhole"))


def test_quasar_tables_are_empty():
    assert headers.eth_tx_queue_table("quasar") == []
    assert headers.eth_rx_queue_table("quasar") == []
    assert headers.eth_l1_table("quasar") == []
    assert headers.eth_layout("quasar") == (0, 0, 0, 0, 0, 0, 0)


def test_layouts():
    assert headers.eth_layout("blackhole") == (
        0xFFB90000,
        0xFFB94000,
        0x1000,
        3,
        3,
        96,
        96,
    )
    assert headers.eth_layout("wormhole_b0") == (
        0xFFB90000,
        0xFFB92000,
        0x1000,
        2,
        2,
        16,
        16,
    )


@pytest.mark.parametrize("arch", sorted(EXPECTED_L1))
def test_l1_table_sizes(arch):
    assert len(headers.eth_l1_table(arch)) == EXPECTED_L1[arch]


@pytest.mark.parametrize("arch", ("blackhole", "wormhole"))
def test_every_table_name_is_declared(arch, names):
    used = {n for n, _ in headers.eth_tx_queue_table(arch)}
    used |= {n for n, _ in headers.eth_rx_queue_table(arch)}
    used |= {e.name for e in headers.eth_l1_table(arch)}
    assert used <= set(names)


def test_blackhole_l1_values_do_not_overlap_and_queue_copies_are_complete():
    entries = headers.eth_l1_table("blackhole")
    spans = sorted((e.address, e.address + 4 * e.words) for e in entries)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end <= start
    assert all(
        e.address >= 0x7CE00 and e.address + 4 * e.words <= 0x7CED0 for e in entries
    )
    assert not any(e.high_first for e in entries)
    for name in ("FW_TXQ_RESEND_CNT", "FW_RXQ_PKT_DROP"):
        assert sorted(e.queue for e in entries if e.name == name) == [0, 1, 2]


def test_blackhole_frame_and_byte_counters_are_64_bit():
    entries = {e.name: e for e in headers.eth_l1_table("blackhole")}
    for name, entry in entries.items():
        if name.startswith(("FW_FRAMES", "FW_BYTES", "FW_CORR", "FW_UNCORR")):
            assert entry.words == 2
        if name in (
            "FW_RETRAIN_COUNT",
            "FW_LINK_FLAP_COUNT",
            "FW_LINK_POLL_ALIVE_COUNT",
        ):
            assert entry.words == 1


def test_wormhole_l1_codeword_words_are_high_first():
    entries = {e.name: e for e in headers.eth_l1_table("wormhole")}
    assert entries["FW_CORR_CW"].high_first and entries["FW_CORR_CW"].words == 2
    assert entries["FW_UNCORR_CW"].address == 0x1F98
    assert entries["FW_RETRAIN_COUNT"].address == 0x1EDC
    assert entries["FW_CRC_ERR"].address == 0x1F7C


def _rows(*items, device=0, program=7, seq=0, flags=0, elapsed=1_000_000):
    return [
        eth_counters.Row(
            device, program, seq, flags, elapsed, core, 3, counter, queue, delta
        )
        for core, counter, queue, delta in items
    ]


def test_per_program_converts_words_to_bytes_over_queues_and_cores():
    rows = _rows(
        ((25, 25), "TXQ_WORD_CNT", 0, 100),
        ((25, 25), "TXQ_WORD_CNT", 1, 20),
        ((25, 25), "RXQ_WORD_CNT", 0, 8),
        ((26, 25), "TXQ_WORD_CNT", 0, 30),
        ((26, 25), "TXQ_RESEND_CNT", 0, 2),
        ((26, 25), "RXQ_PKT_DROP_CNT", 1, 1),
    )
    (p,) = eth_counters.per_program(rows, "blackhole")
    assert p.tx_bytes == 150 * 96 and p.rx_bytes == 8 * 96
    assert p.cores[(25, 25)].tx_bytes == 120 * 96
    assert p.cores[(25, 25)].queues[("TXQ_WORD_CNT", 1)] == 20
    assert p.resends == 2 and p.drops == 1
    assert p.tx_gbps == pytest.approx(150 * 96 / 1_000_000)


def test_wormhole_word_is_16_bytes():
    rows = _rows(((25, 17), "TXQ_WORD_CNT", 0, 10))
    assert eth_counters.per_program(rows, "wormhole")[0].tx_bytes == 160


def test_idle_rate_applies_to_registers_only_and_clamps():
    rows = _rows(
        ((25, 25), "TXQ_WORD_CNT", 0, 1000),
        ((25, 25), "TXQ_PKT_END_CNT", 0, 5),
        ((25, 25), "FW_FRAMES_TXD", NO_QUEUE, 40),
        elapsed=2_000_000_000,
    )
    idle = {
        (0, (25, 25), "TXQ_WORD_CNT", 0): 100.0,
        (0, (25, 25), "TXQ_PKT_END_CNT", 0): 100.0,
        (0, (25, 25), "FW_FRAMES_TXD", NO_QUEUE): 100.0,
    }
    (p,) = eth_counters.per_program(rows, "blackhole", idle)
    core = p.cores[(25, 25)]
    assert core.get("TXQ_WORD_CNT") == 800
    assert core.get("TXQ_PKT_END_CNT") == 0
    assert core.get("FW_FRAMES_TXD") == 40


def test_launches_stay_separate_and_none_row_is_empty_program():
    rows = _rows(((25, 25), "TXQ_WORD_CNT", 0, 4), seq=0, program=1) + _rows(
        ((-1, -1), "NONE", -1, 0), seq=1, program=1
    )
    first, second = eth_counters.per_program(rows, "blackhole")
    assert first.tx_bytes == 384
    assert second.tx_bytes == 0 and second.cores == {}


def test_flags_decode():
    rows = _rows(
        ((25, 25), "TXQ_WORD_CNT", 0, 1),
        flags=eth_counters.FLAG_OVERLAPPED | eth_counters.FLAG_RESET,
    )
    (p,) = eth_counters.per_program(rows, "blackhole")
    assert p.overlapped and p.reset


def test_stale_flag_decodes():
    rows = _rows(((25, 25), "TXQ_WORD_CNT", 0, 1), flags=eth_counters.FLAG_STALE)
    (p,) = eth_counters.per_program(rows, "blackhole")
    assert p.stale and not p.reset


def test_quasar_has_no_word_size():
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        eth_counters.per_program(_rows(((25, 25), "TXQ_WORD_CNT", 0, 1)), "quasar")


def test_csv_round_trip(tmp_path):
    header = "device_id,program_id,launch_seq,flags,elapsed_ns,eth_x,eth_y,channel,counter,queue,delta\n"
    (tmp_path / "eth_traffic.csv").write_text(
        header
        + "0,9,0,0,5000,25,25,3,TXQ_WORD_CNT,0,64\n"
        + "0,9,0,0,5000,-1,-1,-1,NONE,-1,0\n"
    )
    (tmp_path / "eth_idle.csv").write_text(
        "device_id,eth_x,eth_y,channel,counter,queue,counts_per_second\n"
        "0,25,25,3,TXQ_WORD_CNT,0,0\n"
    )
    (tmp_path / "eth_run.csv").write_text(
        "device_id,run_id,flags,elapsed_ns,eth_x,eth_y,channel,counter,queue,delta\n"
        "0,5,0,4000000000,25,25,3,RXQ_WORD_CNT,2,10\n"
    )
    rows = eth_counters.read_traffic(tmp_path / "eth_traffic.csv")
    idle = eth_counters.read_idle(tmp_path / "eth_idle.csv")
    (p,) = eth_counters.per_program(rows, "blackhole", idle)
    assert p.tx_bytes == 64 * 96
    run = eth_counters.per_run(
        eth_counters.read_run(tmp_path / "eth_run.csv"), "blackhole"
    )
    assert run[0].rx_bytes == 960 and run[0].seq == 5
    assert eth_counters.main([str(tmp_path)]) == 0


# Cross-checks against the metal firmware headers, which are not part of a standalone tt-llk checkout.
_METAL_HW_INC = Path(__file__).resolve().parents[4] / "hw" / "inc" / "internal"


def _metal_header(*parts):
    path = _METAL_HW_INC.joinpath(*parts)
    if not path.is_file():
        pytest.skip(f"{path} is not part of this checkout")
    return path.read_text()


def _define(text, name):
    match = re.search(rf"#define\s+{name}\s+(0[xX][0-9a-fA-F]+|\d+)\b", text)
    assert match is not None, name
    return int(match.group(1), 0)


def _struct_body(text, name):
    match = re.search(rf"struct\s+{name}\s*\{{(.*?)\n\}};", text, re.S)
    assert match is not None, name
    return match.group(1)


def _array_length(expression):
    # Lengths in the firmware header are a number or the difference of two.
    match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", expression.strip())
    assert match is not None, expression
    return int(match.group(1)) - int(match.group(2) or 0)


def _live_status_offsets(text):
    offsets, position = {}, 0
    for width, name, count in re.findall(
        r"[u]int(32|64)_t\s+(\w+)(?:\[([^\]]+)\])?\s*;",
        _struct_body(text, "eth_live_status_t"),
    ):
        size = int(width) // 8
        if count:
            position += size * _array_length(count)
        else:
            offsets[name] = position
            position += size
    return offsets


def test_blackhole_l1_table_matches_the_firmware_mirror_layout():
    text = _metal_header("tt-1xx", "blackhole", "eth_fw_api.h")
    sizes = dict(re.findall(r"static_assert\(sizeof\((\w+)\) == (\d+),", text))
    live = _define(text, "MEM_SYSENG_BOOT_RESULTS_BASE") + sum(
        int(sizes[s]) for s in ("eth_status_t", "serdes_results_t", "macpcs_results_t")
    )
    fields = _live_status_offsets(text)
    by_name = {
        "FW_RETRAIN_COUNT": ("retrain_count",),
        "FW_LINK_FLAP_COUNT": ("link_flap_count",),
        "FW_LINK_POLL_ALIVE_COUNT": ("link_poll_alive_count",),
        "FW_FRAMES_TXD": ("frames_txd",),
        "FW_FRAMES_TXD_OK": ("frames_txd_ok",),
        "FW_FRAMES_TXD_BADFCS": ("frames_txd_badfcs",),
        "FW_BYTES_TXD": ("bytes_txd",),
        "FW_BYTES_TXD_OK": ("bytes_txd_ok",),
        "FW_BYTES_TXD_BADFCS": ("bytes_txd_badfcs",),
        "FW_FRAMES_RXD": ("frames_rxd",),
        "FW_FRAMES_RXD_OK": ("frames_rxd_ok",),
        "FW_FRAMES_RXD_BADFCS": ("frames_rxd_badfcs",),
        "FW_FRAMES_RXD_DROPPED": ("frames_rxd_dropped",),
        "FW_BYTES_RXD": ("bytes_rxd",),
        "FW_BYTES_RXD_OK": ("bytes_rxd_ok",),
        "FW_BYTES_RXD_BADFCS": ("bytes_rxd_badfcs",),
        "FW_BYTES_RXD_DROPPED": ("bytes_rxd_dropped",),
        "FW_CORR_CW": ("corr_cw",),
        "FW_UNCORR_CW": ("uncorr_cw",),
    }
    for entry in headers.eth_l1_table("blackhole"):
        if entry.name in by_name:
            assert entry.address == live + fields[by_name[entry.name][0]], entry.name
        elif entry.name == "FW_TXQ_RESEND_CNT":
            assert entry.address == live + fields[f"txq{entry.queue}_resend_cnt"]
        elif entry.name == "FW_RXQ_PKT_DROP":
            assert entry.address == live + fields[f"rxq{entry.queue}_pkt_drop"]
        else:
            raise AssertionError(entry.name)


def test_wormhole_l1_table_matches_the_firmware_address_map():
    text = _metal_header("tt-1xx", "wormhole", "eth_l1_address_map.h")
    expected = {
        "FW_RETRAIN_COUNT": "RETRAIN_COUNT_ADDR",
        "FW_CRC_ERR": "CRC_ERR_ADDR",
        "FW_CORR_CW": "CORR_CW_HI_ADDR",
        "FW_UNCORR_CW": "UNCORR_CW_HI_ADDR",
    }
    for entry in headers.eth_l1_table("wormhole"):
        match = re.search(rf"{expected[entry.name]}\s*=\s*(0[xX][0-9a-fA-F]+)", text)
        assert match is not None and entry.address == int(match.group(1), 0)


def test_tx_queue_offsets_match_the_register_defines():
    text = _metal_header("ethernet", "tt_eth_ss_regs.h")
    defines = {
        "TXQ_TRANSFER_CNT": "ETH_TXQ_TRANSFER_CNT",
        "TXQ_PKT_START_CNT": "ETH_TXQ_PKT_START_CNT",
        "TXQ_PKT_END_CNT": "ETH_TXQ_PKT_END_CNT",
        "TXQ_WORD_CNT": "ETH_TXQ_WORD_CNT",
        "TXQ_RESEND_CNT": "ETH_TXQ_RESEND_CNT",
    }
    for arch in ("blackhole", "wormhole"):
        for name, offset in headers.eth_tx_queue_table(arch):
            assert offset == _define(text, defines[name])
        layout = headers.eth_layout(arch)
        assert layout.txq_base == _define(text, "ETH_TXQ0_REGS_START")
        assert layout.queue_stride == _define(text, "ETH_TXQ_REGS_SIZE")
