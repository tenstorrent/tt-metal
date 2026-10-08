# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free checks of the shared tt_llk_perf package (tools/python).
Imports nothing from helpers, so it runs without ttexalens or a device.
"""

import pytest
from tt_llk_perf import headers, metrics

ARCHES = ("blackhole", "wormhole")


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
    assert len(enum_names) >= 195
    assert len(set(enum_names.values())) == len(enum_names)


def test_every_bank_parses_for_both_arches(arch_tables):
    arch, tables = arch_tables
    assert set(tables) == set(headers.BANK_KEYS)
    for bank in headers.BANK_KEYS:
        assert tables[bank], (arch, bank)


def test_table_names_are_enum_members(arch_tables, enum_names):
    _, tables = arch_tables
    known = set(enum_names.values())
    for bank, entries in tables.items():
        for entry in entries:
            assert entry.name in known, (bank, entry)


def test_l1_entries_carry_a_mux_and_others_do_not(arch_tables):
    arch, tables = arch_tables
    assert all(e.l1_mux is not None for e in tables["L1"])
    expected_muxes = 6 if arch == "blackhole" else 2
    assert {e.l1_mux for e in tables["L1"]} == set(range(expected_muxes))
    for bank in ("INSTRN", "FPU", "TDMA_UNPACK", "TDMA_PACK"):
        assert all(e.l1_mux is None for e in tables[bank])


def test_selects_are_unique_within_a_bank(arch_tables):
    _, tables = arch_tables
    for bank, entries in tables.items():
        keys = [(e.select, e.l1_mux) for e in entries]
        assert len(keys) == len(set(keys)), bank


def test_arch_aliases_and_quasar():
    assert headers.bank_tables("wormhole_b0") == headers.bank_tables("wormhole")
    assert headers.bank_tables("BLACKHOLE") == headers.bank_tables("blackhole")
    assert headers.bank_tables("quasar") == {}
    # allow-pytest.raises: the tt-llk suite does not load the metal root conftest, so the expect_error
    # fixture is not in scope, and a header parser has no device error for the CI triager to match.
    with pytest.raises(ValueError):  # allow-pytest.raises
        headers.bank_tables("grayskull")


def test_metric_keys_have_a_family_suffix_and_a_label():
    out = metrics.compute_metrics(_View({}))
    assert set(out) == set(metrics.METRIC_LABELS)
    assert all(
        k.endswith("_pct") or k.endswith("_ratio") for k in metrics.METRIC_LABELS
    )
    assert all(v is None for v in out.values())


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


def test_table_parser_reads_hex_and_rejects_what_it_cannot_parse():
    decl = "inline constexpr std::array<Entry, {n}> fpu_counters = {{{{{body}}}}};"
    hex_body = "{PerfCounterType::FPU_COUNTER, 0x101}, {PerfCounterType::SFPU_COUNTER, 7}"
    entries = headers.parse_tables(decl.format(n=2, body=hex_body))["FPU"]
    assert [(e.name, e.select) for e in entries] == [
        ("FPU_COUNTER", 257),
        ("SFPU_COUNTER", 7),
    ]
    named = "{PerfCounterType::FPU_COUNTER, GRANT_BASE | 1}"
    # allow-pytest.raises: same reason as test_arch_aliases_and_quasar.
    with pytest.raises(ValueError, match="not an integer literal"):  # allow-pytest.raises
        headers.parse_tables(decl.format(n=1, body=named))
    with pytest.raises(ValueError, match="declares 3"):  # allow-pytest.raises
        headers.parse_tables(decl.format(n=3, body=hex_body))


def test_a_missed_select_reads_missing_not_zero():
    assert metrics.bounded(float("nan")) is None
    assert metrics.bounded(1.5) == 1.0
    assert metrics.bounded(-0.5) == 0.0


def test_srcb_write_metrics_mirror_srca():
    out = metrics.compute_metrics(
        _View(
            {
                "SRCB_WRITE_REQ": 100.0,
                "SRCB_WRITE_NOT_BLOCKED_PORT": 80.0,
                "SRCB_WRITE_NOT_BLOCKED_OVR": 40.0,
            }
        )
    )
    assert out["srcb_write_eff_pct"] == 80.0
    assert out["srcb_write_ovr_blocked_pct"] == 60.0
