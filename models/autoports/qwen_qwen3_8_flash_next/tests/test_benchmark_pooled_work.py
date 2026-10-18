# SPDX-License-Identifier: Apache-2.0
"""Host-only ownership accounting checks; no TTNN or hardware imports."""

import importlib.util
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "tools/benchmark_pooled_work.py"
SPEC = importlib.util.spec_from_file_location("pooled_work_host_test", SOURCE)
WORK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORK)


class PooledMetadataTests(unittest.TestCase):
    def account(self, end, kind, cores=120, sequence=1):
        return WORK.gather_metadata_reads(end - sequence, sequence, kind=kind, work=8192, num_cores=cores)

    def test_explicit_partition_matches_formula_and_covers_prefix(self):
        for cores in (1, 2, 7, 120, 130):
            small, extra = divmod(8192, cores)
            for end in (1, 3, 4, 127, 128, 129, 132, 2048, 4096, 262144):
                with self.subTest(cores=cores, end=end):
                    live = 4 * ((end // 4 + 31) // 32)
                    contiguous = [
                        range(i * small + min(i, extra), i * small + min(i, extra) + small + (i < extra))
                        for i in range(cores)
                    ]
                    cyclic = [range(i, 8192, cores) for i in range(cores)]
                    for kind, partition in (("contiguous", contiguous), ("cyclic", cyclic)):
                        assigned = [[job for job in jobs if job < live] for jobs in partition]
                        result = self.account(end, kind, cores)
                        self.assertEqual(sorted(job for jobs in assigned for job in jobs), list(range(live)))
                        self.assertEqual(result["active_workers"], sum(bool(jobs) for jobs in assigned))

    def test_empty_prefix_still_reads_positions_on_every_core(self):
        for end in (1, 2, 3):
            for kind in ("contiguous", "cyclic"):
                result = self.account(end, kind)
                self.assertEqual(result["positions_read_bytes"], 15360)
                self.assertEqual(result["page_table_read_bytes"], 0)
                self.assertEqual(result["live_jobs"], 0)

    def test_decode_b1_b32_inactive_lanes_and_alignment(self):
        old, new = self.account(4096, "contiguous"), self.account(4096, "cyclic")
        self.assertEqual((old["active_workers"], new["active_workers"]), (2, 120))
        self.assertEqual((old["page_table_read_bytes"], new["page_table_read_bytes"]), (1024, 61440))
        # One active row plus31 physical inactive rows at position0.
        rows = [new] + [self.account(1, "cyclic")] * 31
        self.assertEqual(sum(row["positions_read_bytes"] for row in rows), 32 * 15360)
        self.assertEqual(sum(row["page_table_read_bytes"] for row in rows), 61440)
        self.assertEqual(self.account(4097, "cyclic")["table_bytes_per_active_worker"], 544)
        self.assertEqual(self.account(262144, "cyclic")["table_bytes_per_active_worker"], 32768)

    def test_prefill_uses_end_position_and_does_not_change_matrix_work(self):
        self.assertEqual(self.account(4096, "cyclic", sequence=256), self.account(4096, "cyclic"))
        self.assertNotIn("flops", self.account(4096, "cyclic"))

    def test_unknown_or_missing_geometry_fails_closed(self):
        valid = dict(position=0, sequence=1, kind="cyclic", work=8192, num_cores=120)
        for changes in (
            dict(position=-1),
            dict(sequence=0),
            dict(position=262144),
            dict(work=4096),
            dict(num_cores=0),
            dict(num_cores=8193),
            dict(num_cores=True),
            dict(kind="unknown"),
        ):
            with self.subTest(changes=changes):
                try:
                    WORK.gather_metadata_reads(**(valid | changes))
                except ValueError:
                    pass
                else:
                    self.fail("Unsupported geometry was accepted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
