# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import unittest

from perf_analysis import counter_delta, expected_migration_positions, interval_metrics, migration_intervals


class TimingEvidenceTests(unittest.TestCase):
    def test_overlapping_transfers_are_not_added_to_critical_path(self):
        result = interval_metrics([(0, 3), (4, 6)], [(1, 5), (2, 7)])
        self.assertEqual(result["transfer_sum_s"], 9)
        self.assertEqual(result["transfer_active_s"], 6)
        self.assertEqual(result["overlap_s"], 4)
        self.assertEqual(result["nonoverlap_prefill_s"], 1)
        self.assertEqual(result["nonoverlap_transfer_s"], 2)
        self.assertEqual(result["initial_prefill_s"], 1)
        self.assertEqual(result["final_transfer_tail_s"], 1)

    def make_log(self):
        return "\n".join(
            [
                "[2026-10-21 00:00:00.000] migrate command: migration_id=42 layers=[0,2) src_positions=[0,1024)",
                "[2026-10-21 00:00:00.200] migration_id=42 layer=0 src_positions=[0,1024): completed successful=true reason= releasing 1 local slab(s)",
                "[2026-10-21 00:00:00.300] migration_id=42 layer=1 src_positions=[0,1024): completed successful=true reason= releasing 1 local slab(s)",
            ]
        )

    def test_same_migration_id_on_multiple_layers(self):
        result = migration_intervals(self.make_log(), 2, [(0, 1024)])
        self.assertEqual(len(result), 2)

    def test_child_success_is_not_a_second_operation_completion(self):
        log = (
            self.make_log()
            + "\n[2026-10-21 00:00:00.400] migration_id=42 layer=0 src_positions=[0,1024): child completed successful=true terminal=true"
        )
        self.assertEqual(len(migration_intervals(log, 2, [(0, 1024)])), 2)

    def test_does_not_hide_failure_among_successes(self):
        with self.assertRaisesRegex(ValueError, "fault"):
            migration_intervals(self.make_log() + "\nsuccessful=false", 2, [(0, 1024)])

    def test_rejects_missing_completion(self):
        with self.assertRaisesRegex(ValueError, "unmatched"):
            migration_intervals("\n".join(self.make_log().splitlines()[:-1]), 2, [(0, 1024)])

    def test_rejects_wrong_positions_and_duplicate_completion(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            migration_intervals(self.make_log(), 2, [(0, 2048)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            migration_intervals(self.make_log() + "\n" + self.make_log().splitlines()[-1], 2, [(0, 1024)])

    def test_retries_and_local_child_dispatch_keep_first_start(self):
        lines = self.make_log().splitlines()
        repeated = lines[0].replace("00:00:00.000", "00:00:00.050")
        deferred = "[2026-10-21 00:00:00.020] migrate command: migration_id=42: deferred reason=local_staging_full"
        rows = migration_intervals("\n".join([lines[0], deferred, repeated, *lines[1:]]), 2, [(0, 1024)])
        self.assertEqual(rows[0]["command_log_count"], 2)
        self.assertAlmostEqual(rows[0]["duration_s"], 0.2, places=5)

    def test_rejects_repeated_command_with_changed_range_or_after_completion(self):
        lines = self.make_log().splitlines()
        with self.assertRaisesRegex(ValueError, "conflicting"):
            migration_intervals("\n".join([lines[0], lines[0].replace("0,1024", "0,2048"), *lines[1:]]), 2, [(0, 1024)])
        with self.assertRaisesRegex(ValueError, "after completion"):
            migration_intervals(self.make_log() + "\n" + lines[0], 2, [(0, 1024)])

    def test_final_prompt_block_is_computed_by_decoder(self):
        self.assertEqual(
            expected_migration_positions([(0, 32768), (32768, 65536)], 65536, 32), [(0, 32768), (32768, 65504)]
        )
        self.assertEqual(expected_migration_positions([(32, 1024)], 1024, 32), [(32, 992)])

    def test_wire_bytes_do_not_count_receiver_drain_twice(self):
        before = 'kv_manager_bytes_moved_total{leg="wire"} 100\nkv_manager_bytes_moved_total{leg="drain"} 100'
        after = 'kv_manager_bytes_moved_total{leg="wire"} 500\nkv_manager_bytes_moved_total{leg="drain"} 500'
        self.assertEqual(counter_delta(before, after, "kv_manager_bytes_moved_total", {"leg": "wire"}), 400)
        with self.assertRaisesRegex(ValueError, "reset"):
            counter_delta(after, before, "kv_manager_bytes_moved_total", {"leg": "wire"})


if __name__ == "__main__":
    unittest.main()
