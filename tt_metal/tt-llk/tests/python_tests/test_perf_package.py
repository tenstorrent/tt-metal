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
    with pytest.raises(ValueError):
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
