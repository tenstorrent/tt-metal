import tempfile, unittest, os
from pathlib import Path
from stage_checkpoint import capacity, owned_tree, entries, TOTAL, CACHE_ESTIMATE, RESERVE


class Admission(unittest.TestCase):
    def test_initial_capacity(self):
        self.assertEqual(capacity(TOTAL + CACHE_ESTIMATE + RESERVE, 0), TOTAL + CACHE_ESTIMATE + RESERVE)

    def test_one_byte_short(self):
        with self.assertRaises(ValueError):
            capacity(TOTAL + CACHE_ESTIMATE + RESERVE - 1, 0)

    def test_resume_allocated_credit(self):
        self.assertEqual(capacity(CACHE_ESTIMATE + RESERVE, TOTAL), CACHE_ESTIMATE + RESERVE)

    def test_oversized_namespace(self):
        with self.assertRaises(ValueError):
            capacity(10**15, TOTAL + 100000001)

    def test_manifest_wrong(self):
        with self.assertRaises(ValueError):
            entries(b"{}")

    def test_symlink_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "link").symlink_to("/tmp")
            with self.assertRaises(ValueError):
                owned_tree(p, os.getuid())

    def test_foreign_uid_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                owned_tree(Path(d), os.getuid() + 1)

    def test_own_regular(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "x").write_bytes(b"x" * 4096)
            self.assertEqual(owned_tree(p, os.getuid()), (p / "x").stat().st_blocks * 512)


if __name__ == "__main__":
    unittest.main()
