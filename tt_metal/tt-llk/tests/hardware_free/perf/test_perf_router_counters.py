# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free checks of the router counter tables, option decoding and marker decoder in tt_llk_perf."""

import re
from pathlib import Path

import pytest
from tt_llk_perf import headers, router_counters, router_metrics

# arch: (bank offsets, NoC count, counter bits, what one count is)
EXPECTED = {
    "blackhole": ([0x500, 0x580], 2, 64, "HEAD_FLIT"),
    "wormhole": ([], 2, 0, "NONE"),
    "quasar": (
        [0x600, 0x640, 0x680, 0x6C0, 0x700, 0x740, 0x780, 0x7C0],
        1,
        32,
        "ALL_FLIT",
    ),
}
METAL_NOC_HEADERS = {
    "blackhole": "tt-1xx/blackhole/noc/noc_parameters.h",
    "wormhole": "tt-1xx/wormhole/noc/noc_parameters.h",
    "quasar": "tt-2xx/quasar/noc/noc_parameters.h",
}
ON = 0x00FFFFFF
PER_VC = 0x01000000


def _internal_dir():
    return Path(__file__).resolve().parents[4] / "hw" / "inc" / "internal"


@pytest.fixture(scope="module")
def consts():
    return headers.router_constants()


def test_port_names_are_unique_and_start_with_undef():
    names = headers.router_port_names()
    assert names[0] == "UNDEF" and len(set(names)) == len(names)


def test_option_layout_constants(consts):
    assert consts["NUM_VCS"] == 16
    assert consts["VC_MASK_ALL"] == 0xFFFF
    assert consts["BANK_SHIFT"] == 16
    assert consts["BANKS_ALL"] == 0xFF0000
    assert consts["PER_VC"] == PER_VC
    assert consts["MAX_VALUES"] == 2 * router_counters.MAX_PAIRS


@pytest.mark.parametrize("arch", sorted(EXPECTED))
def test_table_shape(arch):
    offsets, nocs, bits, kind = EXPECTED[arch]
    table = headers.router_counter_table(arch)
    assert [b.offset for b in table.banks] == offsets
    assert (table.num_nocs, table.counter_bits, table.count_kind) == (nocs, bits, kind)
    ports = [b.port for b in table.banks]
    assert set(ports) <= set(headers.router_port_names()) and len(set(ports)) == len(
        ports
    )


def test_blackhole_has_two_output_banks_and_no_input_side():
    ports = [b.port for b in headers.router_counter_table("blackhole").banks]
    assert ports == ["OUT_PORT1", "OUT_PORT2"]


def test_quasar_has_four_output_and_four_input_banks():
    ports = [b.port for b in headers.router_counter_table("quasar").banks]
    assert ports == [f"OUT_PORT{n}" for n in range(1, 5)] + [
        f"IN_PORT{n}" for n in range(1, 5)
    ]


def test_banks_do_not_overlap(consts):
    for arch, (offsets, _, bits, _) in EXPECTED.items():
        stride = bits // 8
        spans = sorted((o, o + consts["NUM_VCS"] * stride) for o in offsets)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), arch


@pytest.mark.parametrize("arch", ["blackhole", "quasar"])
def test_registers_match_the_metal_noc_headers(arch, consts):
    path = _internal_dir() / METAL_NOC_HEADERS[arch]
    if not path.is_file():
        pytest.skip("not inside a tt-metal checkout")
    text = path.read_text()
    found = re.findall(
        r"#define\s+PORT(\d)_(OUT_|IN_)?FLIT_COUNTER(_LOWER)?\(vc\)\s+\(NOC_REGS_START_ADDR \+ (0x[0-9A-Fa-f]+) \+ \(\(vc\) \* (\d+)\)\)",
        text,
    )
    offsets = {}
    for port, side, lower, offset, stride in found:
        offsets[f"{'IN' if side == 'IN_' else 'OUT'}_PORT{port}"] = (
            int(offset, 16),
            int(stride),
        )
    table = headers.router_counter_table(arch)
    assert {b.port: b.offset for b in table.banks} == {
        p: o for p, (o, _) in offsets.items()
    }
    assert {s for _, s in offsets.values()} == {table.counter_bits // 8}
    assert re.search(r"#define\s+NOC_VCS\s+16\b", text)


def test_wormhole_metal_header_has_no_router_counters():
    path = _internal_dir() / METAL_NOC_HEADERS["wormhole"]
    if not path.is_file():
        pytest.skip("not inside a tt-metal checkout")
    assert "FLIT_COUNTER" not in path.read_text()


def test_register_base_constants():
    for arch, base in (
        ("blackhole", 0xFFB20000),
        ("wormhole", 0xFFB20000),
        ("quasar", 0x02000000),
    ):
        text = (headers.find_include_dir() / f"router_counters_{arch}.h").read_text()
        assert (
            int(
                re.search(r"ROUTER_REGS_BASE\s*=\s*(0x[0-9A-Fa-f]+)", text).group(1), 16
            )
            == base
        )
        assert (
            int(
                re.search(r"ROUTER_NOC_STRIDE\s*=\s*(0x[0-9A-Fa-f]+)", text).group(1),
                16,
            )
            == 0x10000
        )


def test_virtual_channel_stride_is_the_counter_width():
    for arch in ("blackhole", "quasar"):
        text = (headers.find_include_dir() / f"router_counters_{arch}.h").read_text()
        stride = int(re.search(r"ROUTER_VC_STRIDE\s*=\s*(\d+)", text).group(1))
        assert stride == EXPECTED[arch][2] // 8


def test_option_selects_banks_and_channels():
    sel = headers.router_option("blackhole", ON)
    assert [b.port for b in sel.banks] == ["OUT_PORT1", "OUT_PORT2"]
    assert sel.vcs == list(range(16)) and not sel.per_vc
    sel = headers.router_option("blackhole", 0x20000 | 0x0012 | PER_VC)
    assert (
        [b.port for b in sel.banks] == ["OUT_PORT2"]
        and sel.vcs == [1, 4]
        and sel.per_vc
    )


def test_bank_bits_past_the_table_are_dropped():
    assert [b.port for b in headers.router_option("blackhole", ON).banks] == [
        "OUT_PORT1",
        "OUT_PORT2",
    ]
    assert len(headers.router_option("quasar", ON).banks) == 8
    assert headers.router_option("wormhole", ON).banks == []


def test_value_labels_sum_and_per_vc():
    assert router_counters.value_labels("blackhole", ON) == ["OUT_PORT1", "OUT_PORT2"]
    assert router_counters.value_labels("blackhole", 0x10000 | 0x0012 | PER_VC) == [
        "OUT_PORT1_VC1",
        "OUT_PORT1_VC4",
    ]


def test_option_without_banks_or_channels_raises():
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        router_counters.value_labels("wormhole", ON)
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        router_counters.value_labels("blackhole", 0x10000)


def test_default_option_fits_in_the_record_budget(consts):
    for arch, (offsets, nocs, _, _) in EXPECTED.items():
        assert nocs * len(offsets) <= consts["MAX_VALUES"], arch
    assert (
        2 * 2 * 16 > consts["MAX_VALUES"]
    )  # every channel apart on Blackhole does not fit


def test_marker_ids_do_not_overlap_other_profiler_ids():
    ids = set(router_counters.MARKER_IDS)
    assert min(ids) == 9120 and max(ids) == 9136 and len(ids) == 17
    assert not ids & set(range(9090, 9097))
    assert not ids & set(range(9100, 9117))


def test_marker_ids_match_the_firmware_header():
    path = _internal_dir() / "profiler" / "router_counters.hpp"
    if not path.is_file():
        pytest.skip("not inside a tt-metal checkout")
    consts = {
        n: int(v)
        for n, v in re.findall(r"constexpr [\w:]+ (\w+) = (\d+);", path.read_text())
    }
    assert (
        consts["ROUTER_COUNTERS_HEADER_PROFILER_ID"] == router_counters.HEADER_MARKER_ID
    )
    assert (
        consts["ROUTER_COUNTERS_PAIR_PROFILER_ID_BASE"]
        == router_counters.PAIR_MARKER_ID_BASE
    )
    assert consts["ROUTER_COUNTERS_MAX_PAIRS"] == router_counters.MAX_PAIRS


def test_capture_core():
    assert router_counters.capture_core("blackhole", "BRISC")
    assert router_counters.capture_core("quasar", "QUASAR_DM0")
    assert not router_counters.capture_core("wormhole", "BRISC")
    assert not router_counters.capture_core("blackhole", "NCRISC")
    assert not router_counters.capture_core("quasar", "QUASAR_DM1")


def _pairs(flat):
    flat = list(flat) + [0] * (len(flat) % 2)
    return {k: (flat[2 * k + 1] << 32) | flat[2 * k] for k in range(len(flat) // 2)}


def test_decode_blackhole_sum_mode():
    # NoC 0 port 1 and 2, then NoC 1 port 1 and 2
    launch = router_counters.unpack_launch((2 << 32) | ON, _pairs([128, 0, 0, 64]))
    decoded = router_counters.decode_router_launch("blackhole", launch)
    assert decoded["count_kind"] == "HEAD_FLIT"
    assert decoded["counts"][0] == {"OUT_PORT1": 128, "OUT_PORT2": 0}
    assert decoded["counts"][1] == {"OUT_PORT1": 0, "OUT_PORT2": 64}
    assert decoded["metrics"][0] == {"router_out_count": 128, "router_in_count": None}
    assert decoded["metrics"][1]["router_out_count"] == 64


def test_decode_blackhole_per_vc_mode():
    option = 0x30000 | 0x0012 | PER_VC
    flat = [5, 0, 0, 7, 1, 2, 3, 4]
    launch = router_counters.unpack_launch((2 << 32) | option, _pairs(flat))
    counts = router_counters.decode_launch("blackhole", launch)
    assert counts[0] == {
        "OUT_PORT1_VC1": 5,
        "OUT_PORT1_VC4": 0,
        "OUT_PORT2_VC1": 0,
        "OUT_PORT2_VC4": 7,
    }
    assert counts[1]["OUT_PORT2_VC4"] == 4
    assert router_metrics.compute_router_metrics(counts[0])["router_out_count"] == 12


def test_decode_quasar_odd_value_count_pads_the_last_pair():
    option = 0x70000 | 0xFFFF  # three output ports summed: one NoC, three values
    launch = router_counters.unpack_launch((1 << 32) | option, _pairs([4, 5, 6]))
    assert launch.values == [4, 5, 6, 0]
    counts = router_counters.decode_launch("quasar", launch)
    assert counts[0] == {"OUT_PORT1": 4, "OUT_PORT2": 5, "OUT_PORT3": 6}


def test_quasar_input_and_output_totals():
    launch = router_counters.unpack_launch(
        (1 << 32) | ON, _pairs([1, 2, 3, 4, 10, 20, 30, 40])
    )
    decoded = router_counters.decode_router_launch("quasar", launch)
    assert decoded["count_kind"] == "ALL_FLIT"
    assert decoded["metrics"][0] == {"router_out_count": 10, "router_in_count": 100}


def test_missing_pair_wrong_length_and_wormhole_raise():
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        router_counters.unpack_launch((2 << 32) | ON, {1: 5})
    short = router_counters.unpack_launch((2 << 32) | ON, _pairs([1, 2]))
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        router_counters.decode_launch("blackhole", short)
    stray = router_counters.unpack_launch((2 << 32) | ON, _pairs([1, 2, 3, 4, 5, 6]))
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        router_counters.decode_launch("blackhole", stray)
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error in LLK suite
        router_counters.decode_launch("wormhole", router_counters.Launch(ON, 2, [0, 0]))


def test_pairs_unpack_low_value_first():
    launch = router_counters.unpack_launch((1 << 32) | ON, {0: (7 << 32) | 5})
    assert launch.values == [5, 7]


def test_metric_labels():
    assert (
        router_metrics.router_metric_label("router_out_count") == "Router Output Count"
    )
    assert router_metrics.router_metric_label("other") == "other"


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


def _launch_rows(run, risc, time, flat, option=ON, nocs=2):
    rows = [_marker(run, risc, 9120, time, (nocs << 32) | option)]
    rows += [_marker(run, risc, 9121 + k, time, v) for k, v in _pairs(flat).items()]
    return rows


def test_tracy_decoder_makes_one_row_per_launch_and_noc():
    analysis = _tracy_analysis()
    rows = _launch_rows(7, "BRISC", 1000, [128, 0, 0, 64])
    rows += _launch_rows(8, "BRISC", 2000, [0, 0, 0, 0])
    rows.append(_marker(7, "BRISC", 9100, 1000, 5))
    df = analysis.decode_router_counter_markers(rows, "blackhole")
    assert len(df) == 4
    first = df[(df["run_host_id"] == 7) & (df["noc"] == 0)].iloc[0]
    assert first["OUT_PORT1"] == 128 and first["router_out_count"] == 128
    assert first["count_kind"] == "HEAD_FLIT"
    second = df[(df["run_host_id"] == 7) & (df["noc"] == 1)].iloc[0]
    assert second["OUT_PORT2"] == 64


def test_tracy_decoder_skips_other_cores_and_broken_launches():
    analysis = _tracy_analysis()
    assert (
        analysis.decode_router_counter_markers(
            _launch_rows(1, "NCRISC", 10, [1, 2, 3, 4]), "blackhole"
        )
        is None
    )
    broken = _launch_rows(1, "BRISC", 10, [1, 2, 3, 4])[:-1]
    assert analysis.decode_router_counter_markers(broken, "blackhole") is None
    assert (
        analysis.decode_router_counter_markers(
            _launch_rows(1, "BRISC", 10, [1, 2, 3, 4]), "wormhole"
        )
        is None
    )


def test_tracy_decoder_quasar_has_one_noc():
    analysis = _tracy_analysis()
    flat = [1, 2, 3, 4, 10, 20, 30, 40]
    df = analysis.decode_router_counter_markers(
        _launch_rows(1, "QUASAR_DM0", 10, flat, nocs=1), "quasar"
    )
    assert (
        len(df) == 1
        and df["router_in_count"][0] == 100
        and df["count_kind"][0] == "ALL_FLIT"
    )
