import unittest
from unittest.mock import patch
from guarded_noc_estimator import GuardedEstimator


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.est = GuardedEstimator()

    def test_boundaries_and_interpolation(self):
        for n in (1, 2, 16, 100, 256):
            for size in (64, 128, 7500, 65536):
                self.assertTrue(self.est.coverage(1, n, size)["eligible"])

    def test_out_of_range_never_calls_estimator(self):
        with patch("guarded_noc_estimator.subprocess.run") as run:
            for n, size in ((0, 2048), (257, 2048), (586, 2048), (16, 32), (16, 65537)):
                self.assertIsNone(self.est.estimate(1, n, size)["latency_us"])
            run.assert_not_called()

    def test_missing_core_axis_never_calls_estimator(self):
        with patch("guarded_noc_estimator.subprocess.run") as run:
            for cores in (7, 44, 88, 110):
                self.assertIsNone(self.est.estimate(cores, 38, 2048)["latency_us"])
            run.assert_not_called()

    def test_missing_corner(self):
        self.est.data["entries"] = [
            e
            for e in self.est.data["entries"]
            if not (e["key"].get("arch", 2) == 3 and e["key"].get("memory", 0) == 1 and e["key"].get("pattern", 1) == 2)
        ]
        self.assertFalse(self.est.coverage(1, 16, 2048)["eligible"])

    def test_real_bounded_query(self):
        result = self.est.estimate(1, 100, 2048)
        self.assertEqual(result["source"], "bounded_estimator")
        self.assertGreater(result["latency_us"], 0)
        self.assertEqual(tuple(result["coverage"]["transaction_bounds"]), (64, 256))


if __name__ == "__main__":
    unittest.main()
