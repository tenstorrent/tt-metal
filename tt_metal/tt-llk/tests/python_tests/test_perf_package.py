# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free checks of the shared tt_llk_perf package (tools/python).

Imports nothing from helpers, so it runs without ttexalens or a device: the
header parsers must read both arch tables and the enum, and the metric engine
must accept a synthetic CounterView.
"""

import pytest
from tt_llk_perf import headers, metrics

ARCHES = ("blackhole", "wormhole", "quasar")
# Quasar has no L1 counter bank; every other bank must parse on every arch.
EXPECTED_ENTRY_COUNTS = {
    "quasar": {"INSTRN": 51, "FPU": 3, "TDMA_UNPACK": 18, "TDMA_PACK": 5, "L1": 0},
}


class _View:
    """CounterView over a flat {counter_name: count} dict with one shared cycle count."""

    def __init__(self, values, cycles=1000.0, blackhole=False):
        self._values = values
        self._cycles = cycles
        self._blackhole = blackhole

    def count(self, bank, name):
        return float(self._values.get(name, 0.0))

    def cycles(self, bank):
        return self._cycles if self._values else 0.0

    def has(self, name):
        return name in self._values

    def is_blackhole(self):
        return self._blackhole


@pytest.fixture(scope="module")
def enum_names():
    return headers.counter_type_names()


@pytest.fixture(scope="module", params=ARCHES)
def arch_tables(request):
    return request.param, headers.bank_tables(request.param)


def test_enum_is_dense_from_undef(enum_names):
    assert enum_names[0] == "UNDEF"
    assert sorted(enum_names) == list(range(len(enum_names)))
    assert len(enum_names) >= 228
    assert len(set(enum_names.values())) == len(enum_names)


def test_quasar_block_follows_the_tt1xx_ordinals(enum_names):
    # Appended after the last tt-1xx enumerator so the 8-bit tt-1xx records keep their ordinals.
    assert enum_names[194] == "L1_1_UNPACKER1_EXT_IF_3_GRANT"
    assert enum_names[195] == "CFG_INSTRN_AVAILABLE_3"
    assert enum_names[226] == "QUASAR_L1_CLIENT_EVENT"
    assert enum_names[227] == "UNPACK2_BUSY_THREAD0"


def test_every_bank_parses_for_every_arch(arch_tables):
    arch, tables = arch_tables
    assert set(tables) == set(headers.BANK_KEYS)
    for bank in headers.BANK_KEYS:
        if bank == "L1" and arch == "quasar":
            assert tables[bank] == []
        else:
            assert tables[bank], (arch, bank)
    expected = EXPECTED_ENTRY_COUNTS.get(arch)
    if expected is not None:
        assert {bank: len(entries) for bank, entries in tables.items()} == expected


def test_table_names_are_enum_members(arch_tables, enum_names):
    _, tables = arch_tables
    known = set(enum_names.values())
    for bank, entries in tables.items():
        for entry in entries:
            assert entry.name in known, (bank, entry)


def test_l1_entries_carry_a_mux_and_others_do_not(arch_tables):
    arch, tables = arch_tables
    assert all(e.l1_mux is not None for e in tables["L1"])
    expected_muxes = {"blackhole": 6, "wormhole": 2, "quasar": 0}[arch]
    assert {e.l1_mux for e in tables["L1"]} == set(range(expected_muxes))
    for bank in ("INSTRN", "FPU", "TDMA_UNPACK", "TDMA_PACK"):
        assert all(e.l1_mux is None for e in tables[bank])


def test_selects_are_unique_within_a_bank(arch_tables):
    _, tables = arch_tables
    for bank, entries in tables.items():
        keys = [(e.select, e.l1_mux) for e in entries]
        assert len(keys) == len(set(keys)), bank


def test_arch_aliases_and_unknown_arch():
    assert headers.bank_tables("wormhole_b0") == headers.bank_tables("wormhole")
    assert headers.bank_tables("BLACKHOLE") == headers.bank_tables("blackhole")
    assert headers.bank_tables("QUASAR") == headers.bank_tables("quasar")
    with pytest.raises(ValueError):
        headers.bank_tables("grayskull")


def test_table_parser_tolerates_a_section_attribute():
    # quasar.h places its arrays with LLK_PERF_TABLE_SECTION between the name and the "=".
    text = (
        "constexpr std::array<Entry, 2> fpu_counters LLK_PERF_TABLE_SECTION = "
        "{{{PerfCounterType::FPU_COUNTER, 0}, {PerfCounterType::MATH_COUNTER, 257}}};"
    )
    assert headers.parse_tables(text)["FPU"] == [
        headers.CounterEntry("FPU_COUNTER", 0, None),
        headers.CounterEntry("MATH_COUNTER", 257, None),
    ]


def test_quasar_tables_use_the_quasar_enumerators(enum_names):
    tables = headers.bank_tables("quasar")
    instrn = {e.name: e.select for e in tables["INSTRN"]}
    assert instrn["CFG_INSTRN_AVAILABLE_0"] == 0
    assert instrn["THREAD_STALLS_3"] == 35
    assert instrn["SRCA_STALL_MATH"] == 50
    assert instrn["THREAD_INSTRUCTIONS_3"] == 259
    assert "XSEARCH_INSTRN_AVAILABLE_0" not in instrn
    unpack = {e.name: e.select for e in tables["TDMA_UNPACK"]}
    assert unpack["UNPACK2_BUSY_THREAD0"] == 9
    assert {e.select for e in tables["TDMA_PACK"]} == {11, 18, 267, 271, 272}


def test_l1_client_selection_space():
    # 37 subports x 8 events; event 0 and the THCON SBank events cannot carry data.
    valid = [s for s in range(37 * 8) if metrics.quasar_l1_client_selection_is_valid(s)]
    assert len(valid) == 256
    assert not metrics.quasar_l1_client_selection_is_valid(37 * 8)
    assert len({metrics.quasar_l1_client_label(s) for s in valid}) == 256
    assert (
        metrics.quasar_l1_client_label(5 * 8 + 1)
        == "L1_CLIENT_UNPACK0_IF0_SBANK0_SBANK_POP"
    )
    key = metrics.l1_client_metric_key(metrics.quasar_l1_client_label(2 * 8 + 6))
    assert key == "l1_client_trisc2_pending_reqs_carry_ratio"
    assert (
        metrics.metric_label(key)
        == "L1_CLIENT_TRISC2_PENDING_REQS_CARRY Mean Outstanding"
    )


def test_metric_keys_have_a_family_suffix_and_a_label():
    out = metrics.compute_metrics(_View({}))
    assert set(out) == set(metrics.METRIC_LABELS)
    assert all(
        k.endswith("_pct") or k.endswith("_ratio") for k in metrics.METRIC_LABELS
    )
    assert all(v is None for v in out.values())
    # The dynamic l1_client family stays out of the static vocabulary.
    assert not any(k.startswith("l1_client_") for k in metrics.METRIC_LABELS)


def test_quasar_metrics_gate_on_their_counters():
    out = metrics.compute_metrics(
        _View({"THREAD_STALLS_3": 500.0, "THREAD_INSTRUCTIONS_3": 900.0})
    )
    assert out["thread3_stall_pct"] == 50.0
    assert out["thread3_instrn_per_ready_cycle_ratio"] == 1.8
    assert out["srca_stall_math_pct"] is None
    out = metrics.compute_metrics(_View({"FPU_COUNTER": 250.0}))
    assert out["thread3_stall_pct"] is None


def test_synthetic_view_drives_the_engine():
    out = metrics.compute_metrics(
        _View(
            {
                "FPU_COUNTER": 250.0,
                "MATH_COUNTER": 500.0,
                "PACKER_BUSY": 800.0,
                "PACKER0_DEST_READ_REQ": 200.0,
            }
        )
    )
    assert out["fpu_utilization_pct"] == 25.0
    assert out["compute_utilization_pct"] == 50.0
    assert out["pack_utilization_pct"] == 80.0
    assert out["pack_dest_eff_pct"] == 25.0
    assert out["unpack_thread_stall_pct"] is None
