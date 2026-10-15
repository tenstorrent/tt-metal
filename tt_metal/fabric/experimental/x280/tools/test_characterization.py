# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_characterization import command_for, record_host_label, run_one, validate_test_report, write_attempt
from tabulate_suites import (
    WARMUP_SOURCE,
    ParseError,
    campaign_report,
    conversion,
    endpoint_difference,
    floor_index_percentile,
    hops,
    parse_jsonl,
    tabulate,
    ticks_to_ns,
    validate_campaign_records,
    validate_clock,
)


def core_sync_case():
    """One fixed 4x4 case and its three raw release records, including large integers."""
    config = {"id": "sync", "member": "core_sync/x280_director_characterization", "environment": {}}
    rows = [
        {
            "suite": "core_sync",
            "member": "x280_director_characterization",
            "release": release,
            "run_nonce": 1,
            "chip_id": 0,
            "origins_comparable": False,
            "measurement_schema": 1,
            "wall_counter": "tensix_live_timer",
            "warmup_source": "none",
            "participants": [
                {"logical_x": x, "logical_y": y, "raw_timestamp": 2**53 + x * 4 + y + 1, "outcome": 0}
                for x in range(4)
                for y in range(4)
            ],
        }
        for release in ("null", "unicast", "multicast")
    ]
    return config, rows


def packet_aggregate(run):
    """A clean packet aggregate record joined to `run`, as the suite writes it."""
    return dict(
        run,
        record="aggregate",
        aggregate_start_mtime=100,
        aggregate_end_mtime=600,
        aggregate_start_rdcycle=7,
        aggregate_end_rdcycle=17507,
        empty_visits=0,
        measurement_schema=2,
        wall_counter="x280_mtime",
        cpu_counter="x280_rdcycle",
        aggregate_error=None,
        clock={"domain": "x280_mtime", "expected_hz": 50_000_000, "conversion_hz": 50_000_000, "errors": []},
    )


X280_MTIME_CLOCK = {"domain": "x280_mtime", "expected_hz": 50_000_000, "conversion_hz": 50_000_000, "errors": []}


def compute_record(**changes):
    """An instrumented X280 route_compute record over four items."""
    record = {
        "suite": "pipeline_stage",
        "member": "route_compute",
        "kind": "summary",
        "backend_id": 0x10200001,
        "run_nonce": 1,
        "packet_header_bytes": 96,
        "route_buffer_capacity": 35,
        "instrumentation": "per_sample",
        "fixture": "direction_table_v1",
        "header_count": 16,
        "prepared_input_base": 0,
        "prepared_input_bytes": 4288,
        "wayenable": 1,
        "pinned_way": 1,
        "waymask_bits": 1,
        "warmup": 8,
        "timed_count": 4,
        "stage_witness": 0,
        "subject": "next-hop direction-table lookup",
        "cpu_samples": {"count": 4, "units": "ticks", "raw_ticks": [9, 7, 8, 30]},
        "measurement_schema": 2,
        "wall_counter": "x280_mtime",
        "cpu_counter": "x280_rdcycle",
        "raw_wall_ticks": 200,
        "cpu_cycles": 350,
        "items": 4,
        "clock": X280_MTIME_CLOCK,
    }
    record.update(changes)
    return record


def tabulated(records):
    """The single table the records render to, as {head: cell} per row, and its constants."""
    text = "\n".join(json.dumps(record) for record in records)
    rows = parse_jsonl("records", text)
    (table,) = tabulate(rows[0].member, rows)
    return [dict(zip(table.heads, cells)) for cells in table.cells], table.constant


class Tables(unittest.TestCase):
    def test_a_pipeline_record_gets_its_per_item_and_sample_figures(self):
        (row,), _ = tabulated([compute_record()])
        # 200 mtime ticks at 50 MHz over 4 items; 350 rdcycle cycles over 4 items.
        self.assertEqual((row["ns"], row["ns/item"], row["cpu/item"]), ("4000.000", "1000.000", "87.500"))
        self.assertEqual(
            (
                row["cpu sample mean cycles"],
                row["cpu sample p50 cycles"],
                row["cpu sample p90 cycles"],
                row["cpu sample p99 cycles"],
            ),
            ("13.500", "8", "9", "9"),
        )
        with self.assertRaisesRegex(
            ParseError, r"^records:1: \[pipeline_stage route_compute\] ns/item: " r"ratio with a zero denominator"
        ):
            tabulated([compute_record(items=0)])

    def test_a_failed_clock_converts_with_a_warning_and_a_malformed_one_stays_raw(self):
        failed = dict(X280_MTIME_CLOCK, errors=["current_frequency_mismatch"])
        (row,), _ = tabulated([compute_record(clock=failed)])
        self.assertEqual((row["wall ticks"], row["ns"], row["ns/item"]), ("200", "4000.000", "1000.000"))
        self.assertEqual(
            row["clock"],
            "warning: unvalidated, converted at the measured 50000000 Hz: " "clock evidence contains errors",
        )
        malformed = dict(X280_MTIME_CLOCK, errors=None)
        (row,), _ = tabulated([compute_record(clock=malformed)])
        self.assertEqual((row["wall ticks"], row["cpu/item"], row["ns"], row["ns/item"]), ("200", "87.500", "", ""))
        self.assertEqual(row["clock"], "raw: clock evidence has no errors list")

    def test_a_record_matches_exactly_one_table_of_its_member(self):
        forward = dict(
            compute_record(member="route_forward", backend_id=0x10200002, instrumentation="off"),
            source_placement="pinned_zero_device",
            source_base=0x0A000AC0,
            source_bytes=96,
            hop_count=4,
            boundary="decision",
            subject="router forward decision",
        )
        tabulated([forward])
        with self.assertRaisesRegex(
            ParseError, r"matches 0 tables; nearest \(X280 decide off\) needs " r"hop_count 3 not in \[4\]$"
        ):
            tabulated([dict(forward, hop_count=3)])
        with self.assertRaisesRegex(ParseError, "matches 0 tables"):
            tabulated([dict(forward, backend_id=0x10200009)])
        del forward["items"]
        with self.assertRaisesRegex(
            ParseError, r"^records:1: \[pipeline_stage route_forward\] ns/item: " r"record has no items$"
        ):
            tabulated([forward])

    def test_pooled_populations_join_every_worker_of_the_run(self):
        run = {
            "suite": "packet_injection",
            "member": "tensix_to_x280",
            "record": "run",
            "successful": True,
            "mode": 1,
            "run_value": 17,
            "family": "brisc",
            "tile_count": 1,
            "worker_count": 2,
            "S": 16,
            "B": 1,
            "C": 1,
            "K": 1,
            "W": 4,
            "N": 3,
            "measurement_schema": 2,
            "clock": X280_MTIME_CLOCK,
        }
        workers = [
            dict(
                run,
                record="worker",
                worker_index=i,
                risc=i,
                logical_x=0,
                logical_y=0,
                latency={"count": 3, "units": "ticks", "raw_ticks": ticks},
            )
            for i, ticks in enumerate(([1, 2, 3], [10, 20, 30]))
        ]
        rows = parse_jsonl("records", "\n".join(json.dumps(r) for r in [run] + workers))
        pooled, per_worker = tabulate("packet_injection tensix_to_x280", rows)
        # The six samples' floor-index p50 is index 2 of [1, 2, 3, 10, 20, 30].
        self.assertEqual(dict(zip(pooled.heads, pooled.cells[0]))["latency p50 ticks"], "3")
        self.assertEqual([dict(zip(per_worker.heads, c))["latency p50 ticks"] for c in per_worker.cells], ["2", "20"])
        with self.assertRaisesRegex(ParseError, r"run_value 17 joins worker_index \[0\], not 0..2 - 1"):
            tabulate("packet_injection tensix_to_x280", rows[:2])
        twice = parse_jsonl("records", "\n".join(json.dumps(r) for r in [run, workers[0], workers[0]]))
        with self.assertRaisesRegex(ParseError, r"joins worker_index \[0, 0\]"):
            tabulate("packet_injection tensix_to_x280", twice)

    def test_a_record_with_a_duplicate_key_or_a_non_finite_number_is_refused(self):
        line = json.dumps(compute_record())
        with self.assertRaisesRegex(ParseError, r"^records:1: not JSON: duplicate key 'run_nonce'$"):
            parse_jsonl("records", line[:-1] + ', "run_nonce": 2}')
        with self.assertRaisesRegex(ParseError, r"^records:1: not JSON: non-standard numeric constant NaN$"):
            parse_jsonl("records", line.replace('"raw_wall_ticks": 200', '"raw_wall_ticks": NaN'))

    def test_a_core_sync_record_without_participants_is_refused(self):
        _, rows = core_sync_case()
        with self.assertRaisesRegex(
            ParseError,
            r"^records:1: \[core_sync x280_director_characterization\] "
            r"workers: participants is not a nonempty list$",
        ):
            tabulated([dict(rows[0], participants=[])])


class CampaignValidation(unittest.TestCase):
    def setUp(self):
        self.config, self.rows = core_sync_case()

    def test_exact_release_and_participant_sets(self):
        validate_campaign_records(self.config, self.rows)
        with self.assertRaises(ValueError):
            validate_campaign_records(self.config, self.rows[:2])
        self.rows[2]["release"] = "unicast"
        with self.assertRaises(ValueError):
            validate_campaign_records(self.config, self.rows)
        self.rows[2]["release"] = "multicast"
        self.rows[0]["participants"][1] = self.rows[0]["participants"][0]
        with self.assertRaises(ValueError):
            validate_campaign_records(self.config, self.rows)

    def test_core_sync_records_carry_no_cross_core_duration(self):
        validate_campaign_records(self.config, [dict(r, cpu_cycles=None) for r in self.rows])
        with self.assertRaisesRegex(ValueError, "common counter origin"):
            validate_campaign_records(self.config, [dict(r, cpu_cycles=7) for r in self.rows])

    def test_skipped_or_wrong_gtest_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.xml"
            path.write_text(
                '<testsuites><testsuite><testcase classname="Suite" name="Test" '
                'status="run" result="completed"/></testsuite></testsuites>'
            )
            validate_test_report(path, "Suite.Test")
            with self.assertRaises(ValueError):
                validate_test_report(path, "Suite.Other")
            path.write_text(
                '<testsuites><testsuite><testcase classname="Suite" name="Test" '
                'status="notrun" result="skipped"><skipped/></testcase></testsuite></testsuites>'
            )
            with self.assertRaises(ValueError):
                validate_test_report(path, "Suite.Test")

    def test_p2p_counts_come_from_requested_samples(self):
        config = {
            "id": "peer",
            "member": "p2p_latency/x280_same_tile",
            "environment": {"P2P_LATENCY_X280_SAME_TILE_N": "2"},
        }
        population = {"count": 2, "raw_ticks": [10, 11]}
        rows = [
            {"participant": p, "phase": phase, "samples": 2, "wall": population, "cpu_cycles": population}
            for p in ("A", "B")
            for phase in (0, 1)
        ]
        validate_campaign_records(config, rows)
        config["environment"]["P2P_LATENCY_X280_SAME_TILE_N"] = "3"
        with self.assertRaises(ValueError):
            validate_campaign_records(config, rows)

    def test_packet_worker_count_comes_from_family_and_tiles(self):
        config = {
            "id": "packet",
            "member": "packet_injection/tensix_to_x280",
            "environment": {"PACKET_INJECTION_T": "2", "PACKET_INJECTION_FAMILY": "paired", "PACKET_INJECTION_N": "3"},
        }
        run = {
            "record": "run",
            "successful": True,
            "run_value": 17,
            "N": 3,
            "worker_count": 4,
            "warmup_source": "explicit",
        }
        rows = [run, packet_aggregate(run)] + [dict(run, record="worker", worker_index=i) for i in range(4)]
        validate_campaign_records(config, rows)
        with self.assertRaisesRegex(ValueError, "set mismatch"):
            validate_campaign_records(config, [dict(r, worker_count=3) for r in rows])
        with self.assertRaisesRegex(ValueError, "set mismatch"):
            validate_campaign_records(config, rows[:-1] + [dict(rows[-1], worker_index=0)])
        config["environment"]["PACKET_INJECTION_C"] = "8"
        with self.assertRaises(ValueError):
            validate_campaign_records(config, [dict(r, C=4) for r in rows])
        validate_campaign_records(config, [dict(r, C=8) for r in rows])
        config["environment"]["PACKET_INJECTION_FAMILY"] = "brisc"
        with self.assertRaises(ValueError):
            validate_campaign_records(config, rows)

    def test_packet_aggregate_is_one_raw_x280_bracket(self):
        config = {
            "id": "packet",
            "member": "packet_injection/tensix_to_x280",
            "environment": {"PACKET_INJECTION_T": "1", "PACKET_INJECTION_FAMILY": "paired", "PACKET_INJECTION_N": "3"},
        }
        run = {
            "record": "run",
            "successful": True,
            "run_value": 17,
            "N": 3,
            "worker_count": 2,
            "warmup_source": "explicit",
        }
        workers = [dict(run, record="worker", worker_index=i) for i in range(2)]
        validate_campaign_records(config, [run, packet_aggregate(run)] + workers)
        self.assertEqual(conversion(packet_aggregate(run)), (50_000_000, None, None))
        with self.assertRaisesRegex(ValueError, "set mismatch"):
            validate_campaign_records(config, [run, dict(workers[1])] + workers)
        wrong = {
            "unusable": {"aggregate_error": "invalid_interval"},
            "x280_mtime": {"clock": {"domain": "aiclk"}},
            "counters": {"wall_counter": "tensix_live_timer"},
            "forward interval": {"aggregate_end_mtime": 100},
            "unrelated": {"run_value": 18},
        }
        for message, change in wrong.items():
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                validate_campaign_records(config, [run, dict(packet_aggregate(run), **change)] + workers)

    def test_erisc_writer_count_comes_from_the_filter_or_the_run(self):
        def config(point):
            return {
                "id": "erisc",
                "member": "packet_injection/erisc_to_x280_header",
                "argv": ["bin", "--gtest_filter=Axes/PacketInjectionEriscToX280Header.Run/" + point],
                "environment": {"PACKET_INJECTION_N": "3"},
            }

        run = {
            "record": "run",
            "successful": True,
            "run_value": 17,
            "N": 3,
            "worker_count": 2,
            "B": 1,
            "warmup_source": "calibrated",
        }
        rows = [run, packet_aggregate(run)] + [dict(run, record="worker", worker_index=i) for i in range(2)]
        validate_campaign_records(config("writers2_b1"), rows)
        validate_campaign_records(config("writers_all_b1"), rows)
        for wrong in ("writers4_b1", "writers2_b4"):
            with self.assertRaises(ValueError):
                validate_campaign_records(config(wrong), rows)

    def test_erisc_consumer_settings_come_from_the_run(self):
        config = {
            "id": "erisc",
            "member": "packet_injection/erisc_to_x280_header",
            "argv": ["bin", "--gtest_filter=Axes/PacketInjectionEriscToX280Header.Run/writers1_b1"],
            "environment": {"PACKET_INJECTION_N": "3", "PACKET_INJECTION_C": "4", "PACKET_INJECTION_K": "8"},
        }
        run = {
            "record": "run",
            "successful": True,
            "run_value": 17,
            "N": 3,
            "worker_count": 1,
            "B": 1,
            "S": 16,
            "C": 4,
            "K": 8,
            "W": 4096,
            "warmup_source": "calibrated",
        }
        rows = [run, packet_aggregate(run), dict(run, record="worker", worker_index=0)]
        validate_campaign_records(config, rows)
        run["K"] = 16
        with self.assertRaises(ValueError):
            validate_campaign_records(config, rows)

    def test_packet_inflight_comes_from_the_run_and_defaults_to_one(self):
        config = {
            "id": "erisc",
            "member": "packet_injection/erisc_to_x280_header",
            "argv": ["bin", "--gtest_filter=Axes/PacketInjectionEriscToX280Header.Run/writers1_b1"],
            "environment": {"PACKET_INJECTION_N": "3", "PACKET_INJECTION_INFLIGHT": "1"},
        }
        run = {
            "record": "run",
            "successful": True,
            "run_value": 17,
            "N": 3,
            "worker_count": 1,
            "B": 1,
            "S": 16,
            "C": 1,
            "K": 1,
            "W": 4096,
            "warmup_source": "calibrated",
        }
        rows = [run, packet_aggregate(run), dict(run, record="worker", worker_index=0)]
        validate_campaign_records(config, rows)
        config["environment"]["PACKET_INJECTION_INFLIGHT"] = "4"
        with self.assertRaises(ValueError):
            validate_campaign_records(config, rows)
        run["inflight"] = 4
        validate_campaign_records(config, rows)

    def test_pipeline_filter_requires_its_sample_population(self):
        config = {
            "id": "pipeline",
            "member": "pipeline_stage/route_compute",
            "environment": {"PIPELINE_STAGE_N": "2"},
            "argv": ["binary", "--gtest_filter=Suite.Test/direction_table_instrumented", "--gtest_color=no"],
        }
        row = {"raw_wall_ticks": 20, "cpu_cycles": 10, "items": 2, "cpu_samples": {"count": 2, "raw_ticks": [2, 3]}}
        validate_campaign_records(config, [row])
        del row["cpu_samples"]
        with self.assertRaises(ValueError):
            validate_campaign_records(config, [row])

    def test_a_malformed_clock_fails_and_a_failed_check_only_warns(self):
        config = {"id": "bw", "member": "transport_bandwidth/tensix_to_x280", "environment": {}}
        row = {
            "measurement_schema": 2,
            "raw_wall_ticks": 20,
            "cpu_cycles": 10,
            "warmup_source": "calibrated",
            "clock": {"domain": "x280_mtime", "expected_hz": 50_000_000, "conversion_hz": 50_000_000, "errors": []},
        }
        validate_campaign_records(config, [row])
        self.assertEqual(ticks_to_ns(20, conversion(row)[0]), 400)
        row["clock"]["errors"] = ["current_frequency_mismatch"]
        validate_campaign_records(config, [row])
        self.assertEqual(conversion(row), (50_000_000, None, "clock evidence contains errors"))
        del row["clock"]["conversion_hz"]
        with self.assertRaisesRegex(ValueError, "clock conversion_hz must equal a positive expected_hz"):
            validate_campaign_records(config, [row])

    def test_every_clocked_record_states_its_members_warm_up_source(self):
        manifest = json.loads((Path(__file__).resolve().parent.parent / "characterization_manifest.json").read_text())
        self.assertEqual({c["member"] for c in manifest["configurations"]}, set(WARMUP_SOURCE))
        row = {"measurement_schema": 2, "raw_wall_ticks": 20, "cpu_cycles": 10, "items": 2, "clock": X280_MTIME_CLOCK}
        cases = [
            ("transport_bandwidth/tensix_to_x280", {}, "calibrated"),
            ("transport_bandwidth/tensix_to_x280", {"TRANSPORT_BW_W": "128"}, "explicit"),
            ("transport_bandwidth/x280_to_tensix", {}, "explicit"),
            ("pipeline_stage/route_forward", {"PIPELINE_STAGE_W": "128"}, "explicit"),
        ]
        for member, environment, source in cases:
            config = {"id": "c", "member": member, "argv": [], "environment": dict(environment, PIPELINE_STAGE_N="2")}
            with self.subTest(member=member, environment=environment):
                validate_campaign_records(config, [dict(row, warmup_source=source)])
                for wrong in ({"warmup_source": "none"}, {}):
                    with self.assertRaisesRegex(ValueError, f"warm-up source is .*, not '{source}'"):
                        validate_campaign_records(config, [dict(row, **wrong)])
        config = {
            "id": "c",
            "member": "packet_injection/erisc_to_x280_header",
            "argv": ["bin", "--gtest_filter=Axes/PacketInjectionEriscToX280Header.Run/writers1_b1"],
            "environment": {"PACKET_INJECTION_N": "3"},
        }
        run = {
            "record": "run",
            "successful": True,
            "run_value": 17,
            "N": 3,
            "worker_count": 1,
            "B": 1,
            "warmup_source": "calibrated",
        }
        rows = [run, packet_aggregate(run), dict(run, record="worker", worker_index=0)]
        validate_campaign_records(config, rows)
        rows[1] = dict(rows[1], warmup_source="explicit")
        with self.assertRaisesRegex(ValueError, "not 'calibrated'"):
            validate_campaign_records(config, rows)

    def test_eth_credit_runs_end_clean_and_each_window_follows_its_first_batch(self):
        config = {"id": "ec", "member": "eth_credit/eth_credit", "argv": [], "environment": {}}
        window = {
            "suite": "eth_credit",
            "member": "eth_credit",
            "case": "throughput_sweep",
            "p": 0,
            "BATCH": 4,
            "packets": 20,
            "first_batch_packets": 4,
            "receive_window": {"interval_error": None},
        }
        attribution = {
            "member": "eth_credit_attribution",
            "case": "throughput_sweep",
            "origins": [],
            "stalled": [],
            "missing": [],
        }
        link = {
            "member": "eth_credit_link",
            "case": "throughput_sweep",
            "chip": 0,
            "metal_failure": "none",
            "undrained": "none",
        }
        validate_campaign_records(config, [window, attribution, link])
        wrong = {
            "no eth_credit throughput or round-trip record": [attribution, link],
            "no round trip timed": [
                {"member": "eth_credit_round_trip", "case": "round_trip_hop", "p": 0, "round_trips": 0}
            ],
            "first batch 20 of 20": [dict(window, first_batch_packets=20), attribution],
            "first batch 0 of 20": [dict(window, first_batch_packets=0), attribution],
            "first batch 5 of 20": [dict(window, first_batch_packets=5), attribution],
            "first batch 1 of 1": [dict(window, first_batch_packets=1, packets=1), attribution],
            "receive window run_failed": [dict(window, receive_window={"interval_error": "run_failed"})],
            "attributes a failure": [window, dict(attribution, stalled=[{"chip": 0}])],
            "ended finish, none": [window, dict(link, metal_failure="finish")],
        }
        for message, rows in wrong.items():
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_campaign_records(config, rows)

    def test_explicit_environment_and_literal_substitution(self):
        config = dict(self.config, argv=["${HOST_BUILD}/binary"], environment={"PACKET_INJECTION_N": "3"})
        manifest = {"common_environment": {}}
        with patch.dict(
            os.environ,
            {
                "PACKET_INJECTION_N": "999",
                "PIPELINE_STAGE_N": "999",
                "TT_METAL_WATCHER": "1",
                "TT_METAL_SLOW_DISPATCH_MODE": "1",
            },
        ):
            argv, environment = command_for(manifest, config, {"HOST_BUILD": "/tmp/$(never-executed)"})
        self.assertEqual(argv, ["/tmp/$(never-executed)/binary"])
        self.assertEqual(environment["PACKET_INJECTION_N"], "3")
        self.assertNotIn("PIPELINE_STAGE_N", environment)
        self.assertNotIn("TT_METAL_WATCHER", environment)
        self.assertNotIn("TT_METAL_SLOW_DISPATCH_MODE", environment)

    def test_a_null_environment_value_unsets_the_variable(self):
        config = dict(self.config, argv=["binary"], environment={"TT_VISIBLE_DEVICES": None})
        manifest = {"common_environment": {"TT_VISIBLE_DEVICES": "${DEVICE}"}}
        with patch.dict(os.environ, {"TT_VISIBLE_DEVICES": "3"}):
            _, environment = command_for(manifest, config, {"DEVICE": "0"})
        self.assertNotIn("TT_VISIBLE_DEVICES", environment)


class Reducers(unittest.TestCase):
    def test_percentiles_take_the_floor_index_of_the_sorted_raw_samples(self):
        self.assertEqual(floor_index_percentile([30, 10, 20], 0.5), 20)
        self.assertEqual(floor_index_percentile([30, 10, 20], 0.99), 20)
        self.assertEqual(floor_index_percentile(list(range(1, 101)), 0.9), 90)
        self.assertIsNone(floor_index_percentile([], 0.5))

    def test_an_interval_wraps_modulo_two_to_the_64(self):
        self.assertEqual(endpoint_difference(2**64 - 5, 4), 9)


class Hops(unittest.TestCase):
    def test_matches_the_harness_inventory_cases(self):
        # tests/host/test_harness_inventory.cpp CountsDirectionalRingDistance, on the 17x12 grid.
        l2cpu0, l2cpu1, l2cpu2, dram, tensix_a, tensix_b = (8, 3), (8, 9), (8, 5), (9, 3), (1, 11), (16, 2)
        cases = [
            (l2cpu0, l2cpu0, 0, 0),
            (l2cpu0, dram, 1, 16),
            (l2cpu0, l2cpu2, 2, 10),
            (l2cpu2, l2cpu0, 10, 2),
            (l2cpu0, l2cpu1, 6, 6),
            (tensix_b, (1, 2), 2, 15),
            (tensix_a, (1, 0), 1, 11),
            ((0, 0), (16, 11), 27, 2),
        ]
        for src, dst, noc0, noc1 in cases:
            self.assertEqual((hops(src, dst, 0), hops(src, dst, 1)), (noc0, noc1), (src, dst))
        with self.assertRaises(ParseError):
            hops((17, 3), l2cpu0, 0)


class ClockValidation(unittest.TestCase):
    # Evidence from an ERISC header-ingress run, sampled on the ERISC's own wall clock.
    ERISC_CLOCK = json.loads(
        """
    {
        "domain": "aiclk",
        "expected_hz": 1350000000,
        "expected_source": "cmfw_aiclk_limit_max",
        "reference_counter": "erisc_wall_clock",
        "reference_core": [20, 25],
        "sample_gap_ns": 50000000,
        "conversion_hz": 1350000000,
        "start": {
            "samples": [
                [
                    6529531444499707,
                    71923254684,
                    6529531444523677
                ],
                [
                    6529531494630395,
                    71990942954,
                    6529531494663995
                ]
            ],
            "rate_lower_hz": 1349331819.4808228,
            "rate_upper_hz": 1350882131.2144213,
            "telemetry": {
                "host_before_ns": 6529531444475687,
                "host_after_ns": 6529531444499487,
                "aiclk_limit_max_mhz": 1350,
                "aiclk_mhz": 1350,
                "aiclk_arb_max_raw": 525638,
                "thermal_trip_count": 0,
                "update_period_ms": 100
            }
        },
        "end": {
            "samples": [
                [
                    6529531525197910,
                    72032171314,
                    6529531525203700
                ],
                [
                    6529531575305187,
                    72099853502,
                    6529531575339308
                ]
            ],
            "rate_lower_hz": 1349826504.637944,
            "rate_upper_hz": 1350901780.6198049,
            "telemetry": {
                "host_before_ns": 6529531675482542,
                "host_after_ns": 6529531675513032,
                "aiclk_limit_max_mhz": 1350,
                "aiclk_mhz": 1350,
                "aiclk_arb_max_raw": 525638,
                "thermal_trip_count": 0,
                "update_period_ms": 100
            }
        },
        "elapsed_rate_lower_hz": 1348508445.521346,
        "elapsed_rate_upper_hz": 1350248076.6059642,
        "elapsed_window_ns": 30533915,
        "rate_tolerance_ppm": 10000,
        "maximum_endpoint_bound_width_ppm": 2500,
        "maximum_elapsed_bound_width_ppm": 5000,
        "errors": []
    }
    """
    )

    def test_erisc_wall_clock_reference_is_accepted_and_unknown_ones_are_not(self):
        validate_clock(self.ERISC_CLOCK)
        with self.assertRaisesRegex(ValueError, "unknown AICLK reference counter"):
            validate_clock(dict(self.ERISC_CLOCK, reference_counter="drisc_wall_clock"))

    def test_the_reference_core_is_a_translated_coordinate_pair(self):
        for core in (None, [20], [20, -1], [20, 25.0], "20,25"):
            with self.subTest(core=core), self.assertRaisesRegex(ValueError, "AICLK reference core is malformed"):
                validate_clock(dict(self.ERISC_CLOCK, reference_core=core))

    def test_aiclk_evidence_samples_the_counter_its_record_converts(self):
        config = {
            "id": "p2p",
            "member": "p2p_latency/erisc_peer_tile",
            "argv": [],
            "environment": {"P2P_LATENCY_ERISC_PEER_TILE_N": "2"},
        }
        population = {"count": 2, "raw_ticks": [2, 3]}
        rows = [
            {
                "participant": participant,
                "phase": phase,
                "samples": 2,
                "wall": population,
                "cpu_cycles": population,
                "measurement_schema": 2,
                "warmup_source": "calibrated",
                "clock": self.ERISC_CLOCK,
            }
            for participant in ("X280", "ERISC")
            for phase in (0, 1)
        ]
        tensix = dict(self.ERISC_CLOCK, reference_counter="tensix_live_timer")
        validate_campaign_records(config, [dict(row, wall_counter="erisc_wall_clock") for row in rows])
        validate_campaign_records(
            config, [dict(row, wall_counter="risc_debug_wall_clock", clock=tensix) for row in rows]
        )
        for wall, clock in (("erisc_wall_clock", tensix), ("risc_debug_wall_clock", self.ERISC_CLOCK)):
            with self.subTest(wall=wall), self.assertRaisesRegex(ValueError, "AICLK evidence sampled on"):
                validate_campaign_records(config, [dict(row, wall_counter=wall, clock=clock) for row in rows])


class AttemptLogging(unittest.TestCase):
    def setUp(self):
        self.config, self.rows = core_sync_case()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.engine = Path(__file__).resolve().parents[2]
        (self.root / "attempts").mkdir()
        self.script = self.root / "fake_benchmark.py"
        self.script.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\nfrom pathlib import Path\n"
            "mode = os.environ.get('FAKE_MODE', 'pass')\n"
            "print('fake benchmark:', mode, flush=True)\n"
            "if mode == 'exit': sys.exit(7)\n"
            "xml = Path(next(a.split('xml:', 1)[1] for a in sys.argv if a.startswith('--gtest_output=')))\n"
            'case = \'<testcase classname="Suite" name="Test" status="run" result="completed"/>\'\n'
            'if mode == \'skip\': case = \'<testcase classname="Suite" name="Test" status="notrun" result="skipped"><skipped/></testcase>\'\n'
            "if mode == 'zero': case = ''\n"
            "xml.write_text('<testsuites><testsuite>' + case + '</testsuite></testsuites>')\n"
            "raw = Path(os.environ['X280_CHARACTERIZATION_OUTPUT_DIR'])\nraw.mkdir()\n"
            f"rows = {self.rows!r}\n"
            "if mode == 'partial': rows = rows[:2]\n"
            "if mode == 'duplicate': rows[2]['release'] = 'unicast'\n"
            "lines = [json.dumps(r) for r in rows]\n"
            "if mode == 'duplicate-key': lines[0] = lines[0][:-1] + ', \"release\": \"null\"}'\n"
            "if mode == 'schema3': lines[0] = lines[0].replace('\"measurement_schema\": 1', '\"measurement_schema\": 3')\n"
            "(raw / 'records.jsonl').write_text(''.join(line + '\\n' for line in lines))\n"
        )
        self.script.chmod(0o700)
        self.config.update(argv=[str(self.script), "--gtest_filter=Suite.Test"])
        self.manifest = {
            "common_environment": {"X280_CHARACTERIZATION_OUTPUT_DIR": "${CAMPAIGN}/raw"},
            "configurations": [self.config],
        }
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_a_campaign_keeps_the_host_label_it_was_started_with(self):
        record_host_label(self.root, "host-a")
        record_host_label(self.root, "host-a")
        with self.assertRaisesRegex(ValueError, "collected on host 'host-a', not 'host-b'"):
            record_host_label(self.root, "host-b")
        self.assertTrue(run_one(self.manifest, self.config, {}, self.root, self.engine, "host-a"))
        attempt = next((self.root / "attempts").iterdir())
        self.assertEqual(json.loads((attempt / "attempt.json").read_text())["host_label"], "host-a")
        # An unlabelled run on a labelled campaign goes through and records no label.
        record_host_label(self.root, None)
        self.assertEqual(json.loads((self.root / "host.json").read_text()), {"host_label": "host-a"})

    def test_an_unlabelled_campaign_records_no_host(self):
        record_host_label(self.root, None)
        self.assertFalse((self.root / "host.json").exists())
        self.assertTrue(run_one(self.manifest, self.config, {}, self.root, self.engine))
        attempt = next((self.root / "attempts").iterdir())
        self.assertNotIn("host_label", json.loads((attempt / "attempt.json").read_text()))

    def test_a_malformed_host_record_is_refused_as_a_value_error(self):
        for content in ("not json", "[]", "{}", '{"host_label": 7}'):
            with self.subTest(content=content):
                (self.root / "host.json").write_text(content)
                with self.assertRaises(ValueError):
                    record_host_label(self.root, "host-a")

    def test_one_record_and_one_hash_list_round_trip(self):
        self.assertTrue(run_one(self.manifest, self.config, {}, self.root, self.engine))
        attempt = next((self.root / "attempts").iterdir())
        self.assertEqual({p.name for p in attempt.iterdir()}, {"attempt.json", "run.log", "gtest.xml", "raw"})
        record = json.loads((attempt / "attempt.json").read_text())
        self.assertEqual(record["status"], "passed")
        self.assertNotIn("raw_sha256", record)
        self.assertEqual(set(record["artifacts_sha256"]), {"run.log", "gtest.xml", "raw/records.jsonl"})
        # Historical prose is not an executable report input.
        (self.root / "provenance.json").write_text("not JSON")
        self.assertTrue(campaign_report(self.root, self.root / "report"))
        summary = json.loads((self.root / "report" / "campaign.json").read_text())
        self.assertEqual(summary["passed_configurations"], 1)
        self.assertNotIn("provenance", summary)
        self.assertNotIn("complete", summary)
        master = [json.loads(line) for line in (self.root / "report" / "master_results.jsonl").read_text().splitlines()]
        self.assertEqual(master[0]["row"]["participants"][0]["raw_timestamp"], 2**53 + 1)
        with (attempt / "raw" / "records.jsonl").open("a") as output:
            output.write("{}\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            campaign_report(self.root, self.root / "corrupt-report")

    def test_the_report_counts_an_attempt_it_cannot_read_as_failed(self):
        for _ in range(3):
            self.assertTrue(run_one(self.manifest, self.config, {}, self.root, self.engine))
        attempts = sorted((self.root / "attempts").iterdir())
        for attempt, corrupt in zip(
            attempts,
            (
                lambda line: line[:-1] + ', "release": "null"}',
                lambda line: line.replace('"chip_id": 0', '"chip_id": NaN'),
            ),
        ):
            raw = attempt / "raw" / "records.jsonl"
            lines = raw.read_text().splitlines()
            raw.write_text("\n".join([corrupt(lines[0])] + lines[1:]) + "\n")
            record = json.loads((attempt / "attempt.json").read_text())
            record["artifacts_sha256"]["raw/records.jsonl"] = hashlib.sha256(raw.read_bytes()).hexdigest()
            (attempt / "attempt.json").write_text(json.dumps(record))
        self.assertTrue(campaign_report(self.root, self.root / "report"))
        coverage = json.loads((self.root / "report" / "coverage.json").read_text())["sync"]
        self.assertEqual(coverage["status"], "passed")
        self.assertEqual([a["status"] for a in coverage["attempts"]], ["failed", "failed", "passed"])
        self.assertRegex(
            coverage["attempts"][0]["error"], r"^not reportable: .*records.jsonl:1: not JSON: duplicate key 'release'$"
        )
        self.assertRegex(
            coverage["attempts"][1]["error"],
            r"^not reportable: .*records.jsonl:1: not JSON: non-standard numeric constant NaN$",
        )

    def test_report_states_repeated_runs(self):
        self.assertTrue(run_one(self.manifest, self.config, {}, self.root, self.engine))
        self.assertTrue(campaign_report(self.root, self.root / "once"))
        self.assertIn("Between-run variation is unavailable.", (self.root / "once" / "MASTER_RESULTS.md").read_text())
        self.assertTrue(run_one(self.manifest, self.config, {}, self.root, self.engine))
        self.assertTrue(campaign_report(self.root, self.root / "twice"))
        text = (self.root / "twice" / "MASTER_RESULTS.md").read_text()
        self.assertIn("Passed runs per covered configuration: 2;", text)
        self.assertNotIn("Between-run variation is unavailable.", text)

    def test_failed_executions_remain_in_the_ledger(self):
        for mode in ("exit", "skip", "zero", "partial", "duplicate", "duplicate-key", "schema3"):
            with self.subTest(mode=mode):
                self.config["environment"] = {"FAKE_MODE": mode}
                self.assertFalse(run_one(self.manifest, self.config, {}, self.root, self.engine))
        records = [json.loads(p.read_text()) for p in (self.root / "attempts").glob("*/attempt.json")]
        self.assertEqual(len(records), 7)
        self.assertTrue(all(r["status"] == "failed" and r["error"] for r in records))
        self.assertFalse(campaign_report(self.root, self.root / "failed-report"))

    def test_missing_binary_is_recorded_before_launch(self):
        self.config["argv"][0] = str(self.root / "missing")
        self.assertFalse(run_one(self.manifest, self.config, {}, self.root, self.engine))
        record = json.loads(next((self.root / "attempts").glob("*/attempt.json")).read_text())
        self.assertEqual(record["status"], "failed")
        self.assertIn("missing", record["error"])

    def test_timeout_is_failed_and_interruption_keeps_running_record(self):
        with patch("run_characterization.subprocess.check_output", return_value="a" * 40), patch(
            "run_characterization.subprocess.run", side_effect=subprocess.TimeoutExpired(["fake"], 180)
        ):
            self.assertFalse(run_one(self.manifest, self.config, {}, self.root, self.engine))
        with patch("run_characterization.subprocess.check_output", return_value="a" * 40), patch(
            "run_characterization.subprocess.run", side_effect=KeyboardInterrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                run_one(self.manifest, self.config, {}, self.root, self.engine)
        records = [json.loads(p.read_text()) for p in (self.root / "attempts").glob("*/attempt.json")]
        self.assertEqual({r["status"] for r in records}, {"failed", "running"})
        self.assertFalse(campaign_report(self.root, self.root / "interrupted-report"))

    def test_failed_record_update_preserves_previous_state(self):
        path = self.root / "attempt.json"
        write_attempt(path, {"status": "running"})
        with patch("pathlib.Path.replace", side_effect=OSError("write failed")):
            with self.assertRaisesRegex(OSError, "write failed"):
                write_attempt(path, {"status": "passed"})
        self.assertEqual(json.loads(path.read_text()), {"status": "running"})


if __name__ == "__main__":
    unittest.main()
