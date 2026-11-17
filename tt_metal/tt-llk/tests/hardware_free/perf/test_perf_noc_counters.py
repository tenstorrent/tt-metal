# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free checks of the NoC interface counter tables, decoder and metrics in tt_llk_perf."""

import re
from pathlib import Path

import pytest
from tt_llk_perf import headers, noc_counters, noc_metrics

EXPECTED = {
    # arch: (data word bytes, NoC count)
    "blackhole": (64, 2),
    "wormhole": (32, 2),
    "quasar": (256, 1),
}
GROUP_COUNTS = {"MASTER": 6, "SLAVE": 6, "ATOMIC": 4}
METAL_NOC_HEADERS = {
    "blackhole": "tt-1xx/blackhole/noc/noc_parameters.h",
    "wormhole": "tt-1xx/wormhole/noc/noc_parameters.h",
    "quasar": "tt-2xx/quasar/noc/noc_parameters.h",
}


@pytest.fixture(scope="module")
def counter_names():
    return headers.noc_counter_names()


def test_enum_names_are_unique_and_start_with_undef(counter_names):
    assert counter_names[0] == "UNDEF"
    assert len(set(counter_names)) == len(counter_names)


@pytest.mark.parametrize("arch", sorted(EXPECTED))
def test_table_shape(arch, counter_names):
    table = headers.noc_counter_table(arch)
    assert (table.word_bytes, table.num_nocs) == EXPECTED[arch]
    names = [e.name for e in table.entries]
    assert set(names) <= set(counter_names) and len(set(names)) == len(names)
    assert len({e.reg for e in table.entries}) == len(table.entries)
    for group, count in GROUP_COUNTS.items():
        assert sum(e.group == group for e in table.entries) == count


@pytest.mark.parametrize("arch", sorted(EXPECTED))
def test_every_group_has_an_even_counter_count(arch):
    for mask in range(1, 8):
        per_noc = len(headers.noc_selected_counters(arch, mask))
        assert per_noc * EXPECTED[arch][1] % 2 == 0
        assert per_noc * EXPECTED[arch][1] // 2 <= noc_counters.MAX_PAIRS


def test_blackhole_word_is_the_measured_64_bytes():
    assert headers.noc_counter_table("blackhole").word_bytes == 64


@pytest.mark.parametrize("arch", sorted(EXPECTED))
def test_registers_match_the_metal_noc_headers(arch):
    root = Path(__file__).resolve().parents[4] / "hw" / "inc" / "internal"
    path = root / METAL_NOC_HEADERS[arch]
    if not path.is_file():
        pytest.skip("not inside a tt-metal checkout")
    defines = {
        n: int(v, 16)
        for n, v in re.findall(
            r"#define\s+(NIU_\w+)\s+(0x[0-9A-Fa-f]+)\b", path.read_text()
        )
    }
    for entry in headers.noc_counter_table(arch).entries:
        assert defines[f"NIU_{entry.name}"] == entry.reg, entry.name


def test_register_address_constants():
    for arch, base in (
        ("blackhole", 0xFFB20200),
        ("wormhole", 0xFFB20200),
        ("quasar", 0x02000200),
    ):
        text = (headers.find_include_dir() / f"noc_counters_{arch}.h").read_text()
        assert (
            int(re.search(r"NIU_STATUS_BASE\s*=\s*(0x[0-9A-Fa-f]+)", text).group(1), 16)
            == base
        )
        assert (
            int(re.search(r"NIU_NOC_STRIDE\s*=\s*(0x[0-9A-Fa-f]+)", text).group(1), 16)
            == 0x10000
        )


def test_selected_counters_follow_the_group_mask():
    assert [e.name for e in headers.noc_selected_counters("blackhole", 1)] == [
        "MST_RD_REQ_SENT",
        "MST_RD_DATA_WORD_RECEIVED",
        "MST_NONPOSTED_WR_REQ_SENT",
        "MST_NONPOSTED_WR_DATA_WORD_SENT",
        "MST_POSTED_WR_REQ_SENT",
        "MST_POSTED_WR_DATA_WORD_SENT",
    ]
    assert len(headers.noc_selected_counters("blackhole", 7)) == 16
    assert headers.noc_selected_counters("blackhole", 0) == []


def test_marker_ids_do_not_overlap_other_profiler_ids():
    ids = set(noc_counters.MARKER_IDS)
    assert min(ids) == 9100 and max(ids) == 9116 and len(ids) == 17
    assert not ids & {9090, 9091, 9092, 9093, 9094, 9095, 9096}


def test_marker_ids_match_the_firmware_header():
    path = (
        Path(__file__).resolve().parents[4]
        / "hw"
        / "inc"
        / "internal"
        / "profiler"
        / "noc_counters.hpp"
    )
    if not path.is_file():
        pytest.skip("not inside a tt-metal checkout")
    text = path.read_text()
    consts = {
        n: int(v) for n, v in re.findall(r"constexpr [\w:]+ (\w+) = (\d+);", text)
    }
    assert consts["NOC_COUNTERS_HEADER_PROFILER_ID"] == noc_counters.HEADER_MARKER_ID
    assert (
        consts["NOC_COUNTERS_PAIR_PROFILER_ID_BASE"] == noc_counters.PAIR_MARKER_ID_BASE
    )
    assert consts["NOC_COUNTERS_MAX_PAIRS"] == noc_counters.MAX_PAIRS


def test_capture_core():
    assert noc_counters.capture_core("blackhole", "BRISC")
    assert noc_counters.capture_core("wormhole", "BRISC")
    assert noc_counters.capture_core("quasar", "QUASAR_DM0")
    assert not noc_counters.capture_core("blackhole", "NCRISC")
    assert not noc_counters.capture_core("blackhole", "ERISC")
    assert not noc_counters.capture_core("quasar", "QUASAR_DM1")


def _pairs(flat):
    return {k: (flat[2 * k + 1] << 32) | flat[2 * k] for k in range(len(flat) // 2)}


def _eltwise_binary_launch():
    """The silicon capture of metal_example_eltwise_binary (64 tiles of 2 KiB per buffer) on Blackhole, group mask 3."""
    names = [e.name for e in headers.noc_selected_counters("blackhole", 3)]
    noc0 = dict.fromkeys(names, 0)
    noc1 = dict.fromkeys(names, 0)
    noc0["MST_RD_REQ_SENT"] = 128
    noc0["MST_RD_DATA_WORD_RECEIVED"] = 4096
    noc1["MST_NONPOSTED_WR_REQ_SENT"] = 64
    noc1["MST_NONPOSTED_WR_DATA_WORD_SENT"] = 2048
    return [noc0[n] for n in names] + [noc1[n] for n in names]


def test_decode_launch_matches_the_silicon_capture():
    flat = _eltwise_binary_launch()
    launch = noc_counters.unpack_launch((2 << 32) | 3, _pairs(flat))
    decoded = noc_counters.decode_noc_launch("blackhole", launch)
    assert decoded["word_bytes"] == 64
    m0, m1 = decoded["metrics"][0], decoded["metrics"][1]
    assert m0["noc_issued_read_bytes"] == 2 * 128 * 1024
    assert m0["noc_bytes_in"] == 2 * 128 * 1024 and m0["noc_bytes_out"] == 0
    assert m0["noc_issued_read_reqs"] == 128
    assert m0["noc_bytes_per_issued_read_req_ratio"] == 2048.0
    assert m1["noc_issued_write_bytes"] == 128 * 1024
    assert m1["noc_bytes_out"] == 128 * 1024 and m1["noc_write_bytes"] == 128 * 1024
    assert m1["noc_bytes_per_issued_write_req_ratio"] == 2048.0
    assert decoded["counts"][1]["MST_NONPOSTED_WR_REQ_SENT"] == 64


def test_pairs_unpack_low_counter_first():
    launch = noc_counters.unpack_launch((1 << 32) | 1, {0: (7 << 32) | 5, 1: 9})
    assert launch.counters == [5, 7, 9, 0]


def test_missing_pair_and_wrong_length_raise():
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        noc_counters.unpack_launch((1 << 32) | 1, {1: 5})
    short = noc_counters.unpack_launch(
        (2 << 32) | 3, _pairs(_eltwise_binary_launch()[:-2])
    )
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        noc_counters.decode_launch("blackhole", short)


def test_metrics_of_a_group_that_was_off_read_none():
    m = noc_metrics.compute_noc_metrics({"MST_RD_DATA_WORD_RECEIVED": 10}, 64)
    assert m["noc_issued_read_bytes"] == 640
    assert m["noc_issued_write_bytes"] is None
    assert m["noc_read_bytes"] is None and m["noc_bytes_in"] is None
    assert m["noc_bytes_per_issued_read_req_ratio"] is None


def test_metrics_split_posted_and_nonposted_and_both_sides():
    counts = {
        "MST_NONPOSTED_WR_DATA_WORD_SENT": 3,
        "MST_POSTED_WR_DATA_WORD_SENT": 5,
        "MST_RD_DATA_WORD_RECEIVED": 7,
        "SLV_RD_DATA_WORD_SENT": 11,
        "SLV_NONPOSTED_WR_DATA_WORD_RECEIVED": 13,
        "SLV_POSTED_WR_DATA_WORD_RECEIVED": 17,
    }
    m = noc_metrics.compute_noc_metrics(counts, 32)
    assert m["noc_write_bytes"] == (3 + 5 + 13 + 17) * 32
    assert m["noc_read_bytes"] == (7 + 11) * 32
    assert m["noc_bytes_in"] == (7 + 13 + 17) * 32
    assert m["noc_bytes_out"] == (3 + 5 + 11) * 32


def test_metric_labels():
    assert noc_metrics.noc_metric_label("noc_bytes_in") == "NoC Bytes In"
    assert noc_metrics.noc_is_ratio_key("noc_bytes_per_issued_read_req_ratio")
    assert not noc_metrics.noc_is_ratio_key("noc_bytes_in")


def _tracy_analysis():
    import importlib.util

    pytest.importorskip("pandas")
    pytest.importorskip("loguru")
    path = (
        Path(__file__).resolve().parents[5]
        / "tools"
        / "tracy"
        / "perf_counter_analysis.py"
    )
    if not path.is_file():
        pytest.skip("not inside a tt-metal checkout")
    spec = importlib.util.spec_from_file_location(
        "perf_counter_analysis_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _marker(run, risc, marker_id, time, data, core=(1, 2)):
    return {
        "run_host_id": run,
        "core_x": core[0],
        "core_y": core[1],
        "risc_type": risc,
        "marker_id": marker_id,
        "record_time": time,
        "data": data,
    }


def _launch_rows(run, risc, time, flat, mask=3, nocs=2, core=(1, 2)):
    rows = [_marker(run, risc, 9100, time, (nocs << 32) | mask, core)]
    rows += [
        _marker(run, risc, 9101 + k, time, v, core) for k, v in _pairs(flat).items()
    ]
    return rows


def test_tracy_decoder_makes_one_row_per_launch_and_noc():
    analysis = _tracy_analysis()
    rows = _launch_rows(7, "BRISC", 1000, _eltwise_binary_launch())
    rows += _launch_rows(8, "BRISC", 2000, [0] * 24)
    rows.append(_marker(7, "NCRISC", 38106, 5, 0))
    df = analysis.decode_noc_counter_markers(rows, "blackhole")
    assert len(df) == 4
    first = df[(df["run_host_id"] == 7) & (df["noc"] == 0)].iloc[0]
    assert first["MST_RD_DATA_WORD_RECEIVED"] == 4096
    assert first["noc_read_bytes"] == 2 * 128 * 1024
    assert first["word_bytes"] == 64
    second = df[(df["run_host_id"] == 7) & (df["noc"] == 1)].iloc[0]
    assert second["noc_write_bytes"] == 128 * 1024


def test_tracy_decoder_skips_other_cores_and_broken_launches():
    analysis = _tracy_analysis()
    other_core = _launch_rows(1, "NCRISC", 10, _eltwise_binary_launch())
    assert analysis.decode_noc_counter_markers(other_core, "blackhole") is None
    broken = _launch_rows(1, "BRISC", 10, _eltwise_binary_launch())[:-1]
    assert analysis.decode_noc_counter_markers(broken, "blackhole") is None


def test_tracy_decoder_quasar_has_one_noc():
    analysis = _tracy_analysis()
    names = [e.name for e in headers.noc_selected_counters("quasar", 1)]
    flat = [0] * len(names)
    flat[names.index("MST_RD_DATA_WORD_RECEIVED")] = 4
    df = analysis.decode_noc_counter_markers(
        _launch_rows(1, "QUASAR_DM0", 10, flat, mask=1, nocs=1), "quasar"
    )
    assert len(df) == 1 and df["noc_issued_read_bytes"][0] == 4 * 256
