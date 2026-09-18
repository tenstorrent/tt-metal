"""Host admission boundaries; charged pagecache is reported, never called process RSS."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from capacity_resources import admit, observe

GIB = 1024**3


def sample():
    return dict(
        mem_available_bytes=504 * GIB,
        disk_free_bytes=15000 * GIB,
        cgroups=[dict(max=None, high=None, current=450 * GIB, file_bytes=400 * GIB, inactive_file_bytes=300 * GIB)],
    )


class ResourceTests(unittest.TestCase):
    # Large charged current with unlimited limits does not incorrectly become a low-memory verdict.
    def test_unlimited_current_not_rss(self):
        result = admit(sample(), 128 * GIB, 32 * GIB)
        self.assertIsNone(result["conservative_cgroup_charged_headroom_bytes"])
        self.assertFalse(result["cgroup_current_is_process_rss"])

    # Either finite memory.high or memory.max is an admission boundary; fail with reclaimability context.
    def test_finite_hierarchy_limits(self):
        row = sample()
        row["cgroups"][0].update(max=600 * GIB, high=590 * GIB)
        self.assertEqual(admit(row, 128 * GIB, 32 * GIB)["conservative_cgroup_charged_headroom_bytes"], 140 * GIB)
        row["cgroups"].append(dict(max=500 * GIB, high=None, current=400 * GIB))
        with self.assertRaisesRegex(ValueError, "includes pagecache"):
            admit(row, 128 * GIB, 32 * GIB)

    # Host and disk admission fail independently at one byte below the explicit floors.
    def test_host_and_disk_floors(self):
        for key, floor in (("mem_available_bytes", 128 * GIB), ("disk_free_bytes", 32 * GIB)):
            row = sample()
            row[key] = floor - 1
            with self.assertRaises(ValueError):
                admit(row, 128 * GIB, 32 * GIB)

    # Read leaf and ancestor constraints; root often exposes memory.current without memory.max/high.
    def test_os_observer_reads_ancestor_limits(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            proc = root / "proc"
            group = root / "cgroup"
            leaf = group / "slice" / "job"
            (proc / "self").mkdir(parents=True)
            leaf.mkdir(parents=True)
            (proc / "meminfo").write_text("MemTotal: 600000000 kB\nMemAvailable: 500000000 kB\n")
            (proc / "self/cgroup").write_text("0::/slice/job\n")
            for p in (leaf, group):
                (p / "memory.current").write_text("4096")
                (p / "memory.stat").write_text("file 2048\ninactive_file 1024\n")
            (leaf / "memory.max").write_text("999999999999")
            (leaf / "memory.high").write_text("max")
            with patch("capacity_resources.shutil.disk_usage", return_value=SimpleNamespace(free=100 * GIB)):
                row = observe(root, proc=proc, cgroup_root=group)
            self.assertEqual(len(row["cgroups"]), 2)
            self.assertEqual(row["cgroups"][0]["file_bytes"], 2048)
            self.assertEqual(row["cgroups"][0]["max"], 999999999999)
            self.assertIsNone(row["cgroups"][1]["max"])


if __name__ == "__main__":
    unittest.main()
