#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import pytest
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
    # Every counter present and equal: no _pct may leave 0..100 (cross-bank clamps are tested separately).
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


def test_per_engine_packers_gate_on_wormhole_only_counters():
    out = mc.compute_metrics(_View({"PACKER_BUSY": 800.0}))
    assert out["packer0_util_pct"] is None
    assert out["packer1_util_pct"] is None
    assert out["packer2_util_pct"] is None
    assert out["pack_utilization_pct"] == 40.0


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
    # Four statistics per metric, plus the three grid-wide averages that only a host+device run can fill.
    assert len(PERF_COUNTER_CSV_HEADERS) == 4 * len(mc.METRIC_LABELS) + 3
    assert "FPU Util Avg (%)" in PERF_COUNTER_CSV_HEADERS
    assert "Avg FPU util on full grid (%)" in PERF_COUNTER_CSV_HEADERS
    assert "Stall Overlap T0 Min (ratio)" in PERF_COUNTER_CSV_HEADERS


def test_every_metric_is_none_when_any_of_its_inputs_is_missing():
    # Remove one counter at a time from a full capture: a metric may lose its value (None) or change because
    # a port dropped out of a mean, but it must never collapse to a fake 0 or 100.
    names = [n for n in COUNTER_TYPE_NAMES.values() if n not in ("UNDEF", "QUASAR_L1_CLIENT_EVENT")]
    for trial in range(5):
        # Varied, deterministic counts in 200..900; grant <= request keeps the clamped L1 metrics away from
        # their saturation points.
        full = {n: 200.0 + (37 * i + 101 * trial) % 700 for i, n in enumerate(names)}
        for i, n in enumerate(names):
            if n.endswith("_GRANT") and n[: -len("_GRANT")] in full:
                full[n] = full[n[: -len("_GRANT")]] * (0.5 + ((13 * i + 7 * trial) % 50) / 100.0)
        base = mc.compute_metrics(_View(full, cycles=1000.0))
        for gone in names:
            reduced = dict(full)
            del reduced[gone]
            out = mc.compute_metrics(_View(reduced, cycles=1000.0))
            for key, value in out.items():
                if base[key] is not None and value is not None and value != base[key]:
                    assert value not in (0.0, 100.0), (key, gone)


def test_ratio_family_is_unbounded_and_raw():
    # Flow above 1: THCON srcA writes land while the unpacker is idle.
    out = mc.compute_metrics(_View({"SRCA_WRITE_REQ": 1500.0, "UNPACK0_BUSY_THREAD0": 1000.0}, cycles=2000.0))
    assert out["unpack_to_math_flow0_ratio"] == 1.5
    assert out["unpack_to_math_flow_ratio"] == 1.5
    assert out["unpack_to_math_flow1_ratio"] is None
    for key in mc.METRIC_LABELS:
        assert key.endswith("_pct") or key.endswith("_ratio"), key


def test_formulas_with_distinct_values():
    v = _View(
        {
            "FPU_COUNTER": 500.0,
            "MATH_COUNTER": 800.0,
            "SRCA_WRITE_NOT_BLOCKED_PORT": 300.0,
            "UNPACK0_BUSY_THREAD0": 600.0,
            "UNPACK1_BUSY_THREAD0": 400.0,
            "PACKER_BUSY": 1000.0,
            "PACKER0_DEST_READ_REQ": 250.0,
            "MATH_INSTRN_AVAILABLE_1": 1000.0,
        },
        cycles=2000.0,
    )
    out = mc.compute_metrics(v)
    assert out["fpu_utilization_pct"] == 25.0
    assert out["compute_utilization_pct"] == 40.0
    assert out["pack_dest_eff_pct"] == 25.0
    assert out["fpu_exec_eff_ratio"] == 0.5  # a ratio: raw value, not a percentage
    assert out["compute_to_unpack_ratio"] == 0.8


def test_partial_captures_read_none_not_zero():
    # INSTRN-only capture: the FPU numerator was never captured.
    out = mc.compute_metrics(_View({"MATH_INSTRN_AVAILABLE_1": 900.0, "THREAD_STALLS_0": 10.0}))
    assert out["fpu_exec_eff_ratio"] is None
    # UNPACK-only capture: no FPU bank behind the compute-to-unpack ratio.
    out = mc.compute_metrics(_View({"UNPACK0_BUSY_THREAD0": 500.0, "UNPACK1_BUSY_THREAD0": 500.0}))
    assert out["compute_to_unpack_ratio"] is None
    # L1_1-only capture: the L1_0 composites have no inputs.
    out = mc.compute_metrics(_View({"L1_1_UNPACKER1_EXT_IF_1": 100.0}))
    assert out["l1_total_bw_pct"] is None
    assert out["l1_port2_util_pct"] is None
    assert out["noc_vs_compute_balance_pct"] is None


def test_idle_packer_engine_is_full_imbalance():
    out = mc.compute_metrics(
        _View({"PACKER_BUSY_0": 10.0, "PACKER_BUSY_1": 10.0, "PACKER_BUSY_2": 0.0, "PACKER_BUSY": 10.0})
    )
    assert out["packer_load_imbalance_pct"] == 100.0
    out = mc.compute_metrics(_View({"PACKER_BUSY": 10.0}))
    assert out["packer_load_imbalance_pct"] is None


def test_scoreboard_stall_is_gated_and_clamped():
    # Cross-bank: the not-stalled count can exceed the unpack-bank denominator, which must read 0, not negative.
    out = mc.compute_metrics(_View({"MATH_INSTRN_AVAILABLE": 1000.0}))
    assert out["math_scoreboard_stall_pct"] is None
    out = mc.compute_metrics(_View({"MATH_INSTRN_AVAILABLE": 1000.0, "MATH_NOT_SCOREBOARD_STALLED": 750.0}))
    assert out["math_scoreboard_stall_pct"] == 25.0
    out = mc.compute_metrics(_View({"MATH_INSTRN_AVAILABLE": 1000.0, "MATH_NOT_SCOREBOARD_STALLED": 1100.0}))
    assert out["math_scoreboard_stall_pct"] == 0.0


def test_instrn_wait_rates_need_their_counter():
    # A capture without the WAITING_FOR_* counters (tt-2xx) must read None, not 0%.
    out = mc.compute_metrics(_View({"THREAD_STALLS_0": 100.0, "THREAD_INSTRUCTIONS_0": 900.0}))
    for key in (
        "math_wait_srca_pct",
        "math_thread_stall_pct",
        "math_sem_wait_pct",
        "pack_sem_wait_pct",
        "any_thread_stall_pct",
        "move_instrn_avail_t0_pct",
        "stall_overlap_t0_ratio",
    ):
        assert out[key] is None, key
    assert out["unpack0_thread1_share_pct"] is None
    out = mc.compute_metrics(_View({"THREAD_STALLS_0": 100.0, "WAITING_FOR_SRCA_VALID": 50.0}, cycles=1000.0))
    assert out["math_wait_srca_pct"] == 5.0


def test_per_arch_port_names_feed_one_metric():
    for name in ("L1_0_UNPACKER_1_ECC_PACK1", "L1_0_UNPACKER_1_ECC"):
        out = mc.compute_metrics(_View({name: 250.0, mc.L1_PORT1_GRANT[name]: 200.0}, cycles=1000.0))
        assert out["l1_port1_util_pct"] == 25.0
        assert out["l1_port1_backpressure_pct"] == pytest.approx(20.0)
    for name in ("L1_1_TDMA_PACKER_2", "L1_1_PACKER_IF_0"):
        assert mc.compute_metrics(_View({name: 100.0}, cycles=1000.0))["l1_packer_port8_util_pct"] == 10.0


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

    shipped = json.loads((Path(mc.__file__).with_name("perf_counter_type_names.json")).read_text())
    assert {int(k): v for k, v in shipped.items()} == mc.perf_counter_type_names()


def test_l1_grant_ratios_stay_bounded_when_ready_exceeds_requests():
    # The tt-1xx L1 grant counter is the interface ready line, so it can exceed the request count.
    out = mc.compute_metrics(
        _View(
            {
                "L1_0_UNPACKER_0": 100.0,
                "L1_0_UNPACKER_0_GRANT": 900.0,
                "L1_0_NOC_RING0_OUTGOING_0": 10.0,
                "L1_0_NOC_RING0_OUTGOING_0_GRANT": 500.0,
            }
        )
    )
    assert out["l1_unpacker_backpressure_pct"] == 0.0
    assert out["noc_ring0_grant_eff_pct"] == 100.0
    assert out["noc_ring0_out_backpressure_pct"] == 0.0
    assert out["l1_contention_index_pct"] == 0.0


def test_tech_report_catalogue_matches_metric_labels_exactly():
    # One catalogue row per engine metric, no stale rows, labels identical to the engine's.
    report = (Path(__file__).resolve().parents[3] / "tech_reports" / "PerfCounters" / "perf-counters.md").read_text()
    catalogue = {}
    for label_unit, key in re.findall(r"^\| (.+?) \| `([a-z0-9_]+)` \| `", report, re.M):
        assert key not in catalogue, f"duplicate catalogue row for {key}"
        catalogue[key] = re.sub(r" \((%|ratio)\)$", "", label_unit)
    assert catalogue == mc.METRIC_LABELS


def test_quasar_families_gate_on_their_counters():
    # A tt-1xx capture has no thread 3, no INSTISSUE class and no thread-ORed stall reasons: all None.
    out = mc.compute_metrics(_View({"THREAD_STALLS_0": 100.0, "THREAD_INSTRUCTIONS_0": 900.0}))
    assert out["thread3_stall_pct"] is None
    assert out["thread3_ipc_pct"] is None
    assert out["instissue_instrn_avail_t0_pct"] is None
    assert out["srca_stall_math_pct"] is None
    assert out["srca_stall_math_share_pct"] is None
    assert out["unpack2_busy_t0_pct"] is None
    assert out["thread0_instrn_per_ready_cycle_ratio"] == 900.0 / 1900.0
    # The per-ready-cycle ratio needs the thread's stall counter for its denominator.
    assert mc.compute_metrics(_View({"THREAD_INSTRUCTIONS_0": 900.0}))["thread0_instrn_per_ready_cycle_ratio"] is None
    out = mc.compute_metrics(
        _View(
            {
                "THREAD_STALLS_3": 500.0,
                "THREAD_INSTRUCTIONS_3": 900.0,
                "INSTISSUE_INSTRN_AVAILABLE_0": 200.0,
                "UNPACK2_BUSY_THREAD0": 1000.0,
                "MATH_SRC_DATA_READY": 400.0,
            }
        )
    )
    assert out["thread3_stall_pct"] == 25.0
    assert out["thread3_ipc_pct"] == 45.0
    assert out["instissue_instrn_avail_t0_pct"] == 10.0
    assert out["unpack2_busy_t0_pct"] == 50.0
    assert out["math_src_data_ready_pct"] == 20.0
    # 900 instructions over the 1500 cycles thread 3 was not stalled.
    assert out["thread3_instrn_per_ready_cycle_ratio"] == 0.6


def test_stall_reason_shares_need_two_reasons():
    out = mc.compute_metrics(_View({"SRCA_STALL_MATH": 300.0}))
    assert out["srca_stall_math_pct"] == 15.0
    assert out["srca_stall_math_share_pct"] is None
    out = mc.compute_metrics(_View({"SRCA_STALL_MATH": 300.0, "DEST_STALL_PACK": 100.0}))
    assert out["srca_stall_math_share_pct"] == 75.0
    assert out["dest_stall_pack_share_pct"] == 25.0
    # Every reason has a rate and a share in the vocabulary; the fifteen match the Quasar INSTRN readout.
    assert len(mc.STALL_REASON_COUNTERS) == 15
    for stem in mc.STALL_REASON_COUNTERS:
        assert f"{stem}_pct" in mc.METRIC_LABELS and f"{stem}_share_pct" in mc.METRIC_LABELS


def test_fpu_sfpu_overlap_and_thread1_write_shares():
    out = mc.compute_metrics(_View({"FPU_COUNTER": 800.0, "SFPU_COUNTER": 600.0, "MATH_COUNTER": 1000.0}))
    assert out["fpu_sfpu_overlap_pct"] == 20.0  # (800 + 600 - 1000) / 2000
    out = mc.compute_metrics(_View({"FPU_COUNTER": 800.0, "MATH_COUNTER": 1000.0}))
    assert out["fpu_sfpu_overlap_pct"] is None
    out = mc.compute_metrics(
        _View(
            {
                "SRCA_WRITE_THREAD0": 300.0,
                "SRCA_WRITE_THREAD1": 100.0,
                "SRCB_WRITE_THREAD0": 0.0,
                "SRCB_WRITE_THREAD1": 50.0,
            }
        )
    )
    assert out["srca_write_thread0_share_pct"] == 75.0
    assert out["srca_write_thread1_share_pct"] == 25.0
    assert out["srcb_write_thread1_share_pct"] == 100.0


def test_l1_client_rates_are_dynamic_and_round_trip_their_labels():
    names = ["L1_CLIENT_UNPACK0_IF0_LANE0_SBANK_POP", "L1_CLIENT_UNPACK0_IF0_LANE0_ISSUE_STALL_CARRY"]
    view = _View({n: 100.0 for n in names}, cycles=10000.0)
    out = mc.compute_l1_client_metrics(view, names + ["FPU_COUNTER"])
    assert out == {
        "l1_client_unpack0_if0_lane0_sbank_pop_pct": 1.0,
        "l1_client_unpack0_if0_lane0_issue_stall_carry_pct": 4.0,  # carries pulse once per four lanes
    }
    for key in out:
        assert mc.metric_label(key).endswith(" Rate")
        assert mc.metric_label(key)[: -len(" Rate")] in names
    assert mc.metric_label("fpu_utilization_pct") == "FPU Util"
    # Nothing dynamic leaks into the static vocabulary the CSV headers and the LLK gate derive from.
    assert not any(k.startswith("l1_client_") for k in mc.METRIC_LABELS)
    assert mc.compute_l1_client_metrics(_View({}), names) == {}


def test_l1_client_labels_cover_every_subport_range():
    assert mc.quasar_l1_client_label(0) == "L1_CLIENT_TRISC0_UNUSED"
    assert mc.quasar_l1_client_label(4 * 8 + 7) == "L1_CLIENT_THCON_ORDER_FIFO_ACTIVE"
    assert mc.quasar_l1_client_label(5 * 8 + 1) == "L1_CLIENT_UNPACK0_IF0_LANE0_SBANK_POP"
    assert mc.quasar_l1_client_label(24 * 8 + 3) == "L1_CLIENT_UNPACK2_IF0_LANE3_ISSUE_WORK_CARRY"
    assert mc.quasar_l1_client_label(36 * 8 + 6) == "L1_CLIENT_PACK1_IF0_LANE3_PENDING_REQS_CARRY"
