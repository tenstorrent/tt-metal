#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from tracy import perf_metrics_common as mc
from tracy.perf_counter_analysis import COUNTER_TYPE_NAMES, PERF_COUNTER_CSV_HEADERS


class _View:
    """CounterView over a flat {counter_name: value} dict with one shared cycle count."""

    def __init__(self, values, cycles=2000.0, blackhole=False):
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


def test_every_metric_key_has_a_label_and_a_family_suffix():
    out = mc.compute_metrics(_View({}))
    assert set(out) == set(mc.METRIC_LABELS)
    assert all(k.endswith("_pct") or k.endswith("_ratio") for k in out)
    assert mc.RATIO_KEYS == {k for k in out if k.endswith("_ratio")}


def test_empty_view_yields_none_not_zero():
    out = mc.compute_metrics(_View({}))
    assert all(v is None for v in out.values())


def test_full_view_percentages_stay_bounded():
    names = set(COUNTER_TYPE_NAMES.values()) - {"UNDEF"}
    out = mc.compute_metrics(_View({n: 1000.0 for n in names}))
    for key, value in out.items():
        assert value is not None, key
        if key.endswith("_pct"):
            assert 0.0 <= value <= 100.0, (key, value)


def test_cross_bank_stalls_gate_on_missing_pack_counters():
    # Unpack group captured without the pack group: absent counters read 0, and an ungated
    # complement would report a bogus 100% stall instead of N/A.
    out = mc.compute_metrics(_View({"MATH_INSTRN_AVAILABLE": 1500.0}))
    assert out["math_scoreboard_stall_pct"] is None
    assert out["math_dest_wr_port_stall_pct"] is None


def test_per_engine_packers_gate_on_wormhole_only_counters():
    out = mc.compute_metrics(_View({"PACKER_BUSY": 800.0}))
    assert out["packer0_util_pct"] is None
    assert out["packer1_util_pct"] is None
    assert out["packer2_util_pct"] is None
    assert out["packer3_util_pct"] == 40.0


def test_mean_port_util_averages_only_present_ports():
    view = _View({"L1_0_NOC_RING0_OUTGOING_0": 500.0, "L1_0_NOC_RING0_OUTGOING_1": 1500.0})
    assert mc.mean_port_util(view, "L1", mc.L1_RING0, 2000.0) == 0.5
    assert mc.mean_port_util(view, "L1", mc.L1_EXT_PACK, 2000.0) is None


def test_enum_parser_matches_the_compiled_ordinals():
    # UNDEF anchors ordinal 0 and the table is dense from there.
    assert COUNTER_TYPE_NAMES[0] == "UNDEF"
    assert sorted(COUNTER_TYPE_NAMES) == list(range(len(COUNTER_TYPE_NAMES)))
    enum_names = set(COUNTER_TYPE_NAMES.values())
    assert all(n in enum_names for n in mc.L1_ALL)


def test_csv_headers_cover_every_label_with_four_stats():
    assert len(PERF_COUNTER_CSV_HEADERS) == 4 * len(mc.METRIC_LABELS)
    assert "Avg FPU util on full grid (%)" in PERF_COUNTER_CSV_HEADERS
    assert "Stall Overlap T0 Min (ratio)" in PERF_COUNTER_CSV_HEADERS


def test_formulas_with_distinct_values():
    v = _View(
        {
            "FPU_COUNTER": 500.0,
            "MATH_COUNTER": 800.0,
            "SRCA_WRITE_ACTUAL": 300.0,
            "UNPACK0_BUSY_THREAD0": 600.0,
            "UNPACK1_BUSY_THREAD0": 400.0,
            "PACKER_BUSY": 1000.0,
            "PACKER_DEST_READ_AVAILABLE": 250.0,
            "FPU_INSTRN_AVAILABLE_1": 1000.0,
        },
        cycles=2000.0,
    )
    out = mc.compute_metrics(v)
    assert out["fpu_utilization_pct"] == 25.0
    assert out["compute_utilization_pct"] == 40.0
    assert out["unpack0_write_eff_pct"] == 50.0
    assert out["pack_dest_eff_pct"] == 25.0
    assert out["fpu_exec_eff_pct"] == 50.0
    assert out["compute_to_unpack_ratio"] == 0.8


def test_partial_captures_read_none_not_zero():
    # INSTRN-only capture: the FPU numerator was never captured.
    out = mc.compute_metrics(_View({"FPU_INSTRN_AVAILABLE_1": 900.0, "THREAD_STALLS_0": 10.0}))
    assert out["fpu_exec_eff_pct"] is None
    # UNPACK-only capture: no FPU bank behind the compute-to-unpack ratio.
    out = mc.compute_metrics(_View({"UNPACK0_BUSY_THREAD0": 500.0, "UNPACK1_BUSY_THREAD0": 500.0}))
    assert out["compute_to_unpack_ratio"] is None
    # L1_1-only capture: the L1_0 composites have no inputs.
    out = mc.compute_metrics(_View({"L1_1_EXT_UNPACKER_1": 100.0}))
    assert out["l1_total_bw_pct"] is None
    assert out["risc_core_l1_util_pct"] is None
    assert out["noc_vs_compute_balance_pct"] is None


def test_idle_packer_engine_is_full_imbalance():
    out = mc.compute_metrics(
        _View({"PACKER_BUSY_0": 10.0, "PACKER_BUSY_1": 10.0, "PACKER_BUSY_2": 0.0, "PACKER_BUSY": 10.0})
    )
    assert out["packer_load_imbalance_pct"] == 100.0
    out = mc.compute_metrics(_View({"PACKER_BUSY": 10.0}))
    assert out["packer_load_imbalance_pct"] is None


def test_dest_write_port_stall_needs_a_ticking_counter():
    out = mc.compute_metrics(_View({"MATH_INSTRN_AVAILABLE": 1000.0, "MATH_NOT_STALLED_DEST_WR_PORT": 0.0}))
    assert out["math_dest_wr_port_stall_pct"] is None
    out = mc.compute_metrics(_View({"MATH_INSTRN_AVAILABLE": 1000.0, "MATH_NOT_STALLED_DEST_WR_PORT": 750.0}))
    assert out["math_dest_wr_port_stall_pct"] == 25.0


def test_port1_side_of_the_read_write_split_follows_the_arch():
    counts = {
        "L1_0_UNPACKER_0": 100.0,
        "L1_0_UNPACKER_1_ECC_PACK1": 100.0,
        "L1_0_NOC_RING0_OUTGOING_0": 100.0,
        "L1_0_NOC_RING0_INCOMING_0": 100.0,
    }
    assert mc.compute_metrics(_View(counts, blackhole=True))["l1_read_write_ratio_pct"] == 75.0
    assert mc.compute_metrics(_View(counts, blackhole=False))["l1_read_write_ratio_pct"] == 50.0


def test_shipped_counter_type_table_matches_the_header():
    import json
    from pathlib import Path

    shipped = json.loads((Path(mc.__file__).with_name("perf_counter_type_names.json")).read_text())
    assert {int(k): v for k, v in shipped.items()} == mc.perf_counter_type_names()
