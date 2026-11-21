# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free checks of the DRAM tile counter tables and the DRAM traffic decoder in tt_llk_perf."""

import pytest
from tt_llk_perf import dram_counters, headers

EXPECTED_COUNTS = {"blackhole": 6, "wormhole": 6, "quasar": 0}


@pytest.fixture(scope="module")
def names():
    return headers.dram_counter_names()


def test_enum_names_are_unique_and_start_with_undef(names):
    assert names[0] == "UNDEF"
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("arch", sorted(EXPECTED_COUNTS))
def test_table_sizes(arch):
    assert len(headers.dram_counter_table(arch)) == EXPECTED_COUNTS[arch]


@pytest.mark.parametrize("arch", ("blackhole", "wormhole"))
def test_table_names_are_declared_and_offsets_unique(arch, names):
    entries = headers.dram_counter_table(arch)
    assert set(n for n, _ in entries) <= set(names)
    offsets = [o for _, o in entries]
    assert len(set(offsets)) == len(offsets)
    assert all(o % 4 == 0 for o in offsets)


def test_niu_offsets_follow_the_status_register_indices():
    index = {
        "NIU_SLV_RD_DATA_WORD_SENT": 0x33,
        "NIU_SLV_RD_REQ_RECEIVED": 0x35,
        "NIU_SLV_NONPOSTED_WR_DATA_WORD_RECEIVED": 0x38,
        "NIU_SLV_POSTED_WR_DATA_WORD_RECEIVED": 0x39,
        "NIU_SLV_NONPOSTED_WR_REQ_RECEIVED": 0x3A,
        "NIU_SLV_POSTED_WR_REQ_RECEIVED": 0x3B,
    }
    for arch in ("blackhole", "wormhole"):
        got = dict(headers.dram_counter_table(arch))
        assert got == {n: 0x200 + 4 * i for n, i in index.items()}


def test_layouts():
    assert headers.dram_layout("blackhole") == (0xFFB20000, 0x10000, 0, 1, 0, 64, 2)
    assert headers.dram_layout("wormhole_b0") == (
        0x100080000,
        0x8000,
        0x10000,
        3,
        2,
        32,
        2,
    )
    assert headers.dram_layout("quasar").word_bytes == 0


def test_blackhole_address_is_the_worker_one():
    for y in (0, 1, 2, 11):
        assert headers.dram_niu_address("blackhole", 0, y, 0x2CC) == 0xFFB202CC
        assert headers.dram_niu_address("blackhole", 1, y, 0x2CC) == 0xFFB302CC


def test_wormhole_dram_tile_addresses_follow_the_register_instances():
    rd = dict(headers.dram_counter_table("wormhole"))["NIU_SLV_RD_DATA_WORD_SENT"]
    assert rd == 0x2CC
    # y % 3 == 1 -> instance 0, 2 -> 1, 0 -> 2
    assert headers.dram_niu_address("wormhole", 0, 1, rd) == 0x1000802CC
    assert headers.dram_niu_address("wormhole", 0, 2, rd) == 0x1000902CC
    assert headers.dram_niu_address("wormhole", 0, 0, rd) == 0x1000A02CC
    assert headers.dram_niu_address("wormhole", 0, 3, rd) == 0x1000A02CC
    assert headers.dram_niu_address("wormhole", 1, 1, rd) == 0x1000882CC
    assert headers.dram_niu_address("wormhole", 1, 0, rd) == 0x1000A82CC
    assert headers.dram_niu_address("wormhole", 0, 4, rd) == 0x1000802CC


def _rows(*items, device=0, program=7, seq=0, flags=0, elapsed=1_000_000):
    return [
        dram_counters.Row(
            device, program, seq, flags, elapsed, tile, noc, counter, w, 0
        )
        for tile, noc, counter, w in items
    ]


def test_per_program_sums_words_to_bytes_over_nocs_and_tiles():
    rows = _rows(
        ((0, 1), 0, "NIU_SLV_RD_DATA_WORD_SENT", 100),
        ((0, 1), 1, "NIU_SLV_RD_DATA_WORD_SENT", 20),
        ((0, 1), 1, "NIU_SLV_NONPOSTED_WR_DATA_WORD_RECEIVED", 8),
        ((0, 1), 1, "NIU_SLV_POSTED_WR_DATA_WORD_RECEIVED", 2),
        ((0, 2), 0, "NIU_SLV_RD_DATA_WORD_SENT", 30),
    )
    (p,) = dram_counters.per_program(rows, "blackhole")
    assert p.read_bytes == 150 * 64 and p.write_bytes == 10 * 64
    assert p.tiles[(0, 1)].read_words == 120 and p.tiles[(0, 1)].write_words == 10
    assert p.read_gbps == pytest.approx(150 * 64 / 1_000_000)


def test_wormhole_word_is_32_bytes():
    rows = _rows(((0, 1), 0, "NIU_SLV_RD_DATA_WORD_SENT", 10))
    assert dram_counters.per_program(rows, "wormhole")[0].read_bytes == 320


def test_idle_drift_is_subtracted_and_clamped():
    rows = _rows(
        ((0, 1), 0, "NIU_SLV_RD_DATA_WORD_SENT", 1000),
        ((0, 2), 0, "NIU_SLV_RD_DATA_WORD_SENT", 5),
        elapsed=2_000_000_000,
    )
    idle = {
        (0, (0, 1), 0, "NIU_SLV_RD_DATA_WORD_SENT"): dram_counters.Idle(100.0, 200.0),
        (0, (0, 2), 0, "NIU_SLV_RD_DATA_WORD_SENT"): dram_counters.Idle(100.0, 0.0),
    }
    (p,) = dram_counters.per_program(rows, "blackhole", idle)
    assert p.tiles[(0, 1)].read_words == 1000 - 100 - 400
    assert p.tiles[(0, 2)].read_words == 0


def test_launches_stay_separate_and_none_row_is_empty_program():
    rows = _rows(((0, 1), 0, "NIU_SLV_RD_DATA_WORD_SENT", 4), seq=0, program=1) + _rows(
        ((-1, -1), -1, "NONE", 0), seq=1, program=1
    )
    first, second = dram_counters.per_program(rows, "blackhole")
    assert first.read_bytes == 256
    assert second.read_bytes == 0 and second.tiles == {}


def test_flags_decode():
    rows = _rows(
        ((0, 1), 0, "NIU_SLV_RD_DATA_WORD_SENT", 1),
        flags=dram_counters.FLAG_OVERLAPPED,
    )
    (p,) = dram_counters.per_program(rows, "blackhole")
    assert p.overlapped and not p.unsettled


def test_quasar_has_no_word_size():
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        dram_counters.per_program(
            _rows(((0, 1), 0, "NIU_SLV_RD_DATA_WORD_SENT", 1)), "quasar"
        )


def test_csv_round_trip(tmp_path):
    (tmp_path / "dram_traffic.csv").write_text(
        "device_id,program_id,launch_seq,flags,elapsed_ns,dram_x,dram_y,noc,counter,delta_words,self_words\n"
        "0,9,0,0,5000,17,12,0,NIU_SLV_RD_DATA_WORD_SENT,64,42\n"
        "0,9,0,0,5000,-1,-1,-1,NONE,0,0\n"
    )
    (tmp_path / "dram_idle.csv").write_text(
        "device_id,dram_x,dram_y,noc,counter,words_per_snapshot_pair,words_per_second\n"
        "0,17,12,0,NIU_SLV_RD_DATA_WORD_SENT,4,0\n"
    )
    rows = dram_counters.read_traffic(tmp_path / "dram_traffic.csv")
    idle = dram_counters.read_idle(tmp_path / "dram_idle.csv")
    (p,) = dram_counters.per_program(rows, "blackhole", idle)
    assert p.read_bytes == 60 * 64
    assert dram_counters.main([str(tmp_path)]) == 0


CONTROLLER_HEADER = "device_id,program_id,launch_seq,flags,elapsed_ns,channel,dram_x,dram_y,counter,value\n"


def test_controller_commands_add_over_channels(tmp_path):
    (tmp_path / "dram_controller.csv").write_text(
        CONTROLLER_HEADER + "0,9,0,0,5000,0,17,12,RD_CMD,553\n"
        "0,9,0,0,5000,0,17,12,RD_AP_CMD,613\n"
        "0,9,0,0,5000,0,17,12,WR_CMD,708\n"
        "0,9,0,0,5000,0,17,12,ACT_CMD,783\n"
        "0,9,0,4,5000,3,26,2,WR_AP_CMD,170\n"
        "0,9,0,4,5000,3,26,2,ACT_CMD,10\n"
        "0,9,1,0,700,0,17,12,RD_AP_CMD,8\n"
    )
    first, second = dram_counters.per_program_controller(
        dram_counters.read_controller(tmp_path / "dram_controller.csv")
    )
    assert (first.read_commands, first.write_commands, first.activates) == (
        1166,
        878,
        793,
    )
    assert first.channels[3] == {"WR_AP_CMD": 170, "ACT_CMD": 10}
    assert first.wrapped and not second.wrapped
    assert first.command_bytes == (1166 + 878) * 64
    assert first.row_hit_rate == pytest.approx(1 - 793 / 2044)
    assert second.read_commands == 8 and second.row_hit_rate == 1.0


def test_controller_log_is_printed_when_present(tmp_path, capsys):
    (tmp_path / "dram_traffic.csv").write_text(
        "device_id,program_id,launch_seq,flags,elapsed_ns,dram_x,dram_y,noc,counter,delta_words,self_words\n"
        "0,9,0,0,5000,-1,-1,-1,NONE,0,0\n"
    )
    (tmp_path / "dram_controller.csv").write_text(
        CONTROLLER_HEADER + "0,9,0,0,5000,0,17,12,RD_CMD,5\n"
    )
    assert dram_counters.main([str(tmp_path), "--no-idle"]) == 0
    assert "rd_cmds" in capsys.readouterr().out
