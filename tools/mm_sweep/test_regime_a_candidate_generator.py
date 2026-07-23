#!/usr/bin/env python3

import unittest

import regime_a_candidate_generator as cg


class CandidateGeneratorTest(unittest.TestCase):
    def test_known_mt8_winners_are_retained(self):
        known = {
            (256, 2048, 1024): (1, 4, 2, 2, 2),
            (256, 6144, 768): (1, 12, 1, 2, 1),
            (256, 6144, 2304): (1, 12, 1, 2, 1),
            (256, 6144, 4608): (1, 12, 1, 2, 1),
        }
        for shape, winner in known.items():
            got, _reasons, stats = cg.select_candidates(*shape, budget=128, audit=0)
            self.assertIn(winner, {g.cfg for g in got}, (shape, stats))

    def test_output_is_feasible_unique_and_bounded(self):
        got, reasons, stats = cg.select_candidates(128, 6080, 4640, budget=96, audit=8)
        cfgs = [g.cfg for g in got]
        self.assertEqual(len(cfgs), len(set(cfgs)))
        self.assertLessEqual(len(cfgs), 96)
        self.assertEqual(stats["selected"], len(cfgs))
        self.assertTrue(all(cg.geometry(128, 6080, 4640, c) is not None for c in cfgs))
        self.assertTrue(all(c in reasons for c in cfgs))

    def test_explicit_config_is_retained(self):
        cfg = (1, 4, 2, 2, 4)
        got, reasons, _stats = cg.select_candidates(
            256, 2048, 1024, budget=32, audit=0, include=[cfg]
        )
        self.assertIn(cfg, {g.cfg for g in got})
        self.assertIn("explicit", reasons[cfg])

    def test_invalid_small_n_has_no_candidates(self):
        got, reasons, stats = cg.select_candidates(32, 2048, 288)
        self.assertEqual(got, [])
        self.assertEqual(reasons, {})
        self.assertEqual(stats["selected"], 0)


if __name__ == "__main__":
    unittest.main()
