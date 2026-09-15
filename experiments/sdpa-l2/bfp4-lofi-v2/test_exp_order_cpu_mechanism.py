"""Small CPU unit controls; skips explicitly if Torch is unavailable."""

import math
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    import exp_order_cpu_mechanism as model


@unittest.skipIf(torch is None, "CPU Torch is required; no device dependency")
class ExpOrderMechanism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_scalar_identity_and_native_octave(self):
        records = model.scalar_algebra_checks()
        native = [r for r in records if r["family"] == "continuous_native"]
        self.assertGreater(abs(native[0]["product_to_direct_ratio"] - 1), 0.04)
        self.assertAlmostEqual(native[1]["product_to_direct_ratio"], 1, places=12)
        x = torch.tensor([-2.3, -1.7, -0.1, 0.0], dtype=torch.float64)
        self.assertTrue(
            torch.allclose(
                model.score_exp(x - math.log(2), "continuous_native"),
                model.score_exp(x, "continuous_native") / 2,
                rtol=1e-12,
                atol=1e-12,
            )
        )

    def test_two_block_order_effect_and_controls(self):
        scores = torch.cat([torch.zeros(model.CHUNK), torch.full((model.CHUNK,), 0.4)]).double()[None, :]
        values = torch.cat([torch.ones((model.CHUNK, model.DIM)), torch.zeros((model.CHUNK, model.DIM))]).double()
        reference = torch.softmax(scores, dim=1) @ values
        for family in ["exact", "constant_bias", "continuous_native"]:
            outputs = {}
            for schedule in ["running_max", "global_max"]:
                pair = [model.online(scores, values, order, family, schedule, True)[0] for order in [[0, 1], [1, 0]]]
                outputs[schedule] = pair
                if family != "continuous_native":
                    for output in pair:
                        self.assertTrue(torch.allclose(output, reference, rtol=1e-12, atol=1e-12))
                if schedule == "global_max":
                    self.assertTrue(torch.allclose(*pair, rtol=1e-12, atol=1e-12))
            if family == "continuous_native":
                self.assertGreater(float((outputs["running_max"][0] - outputs["running_max"][1]).abs().max()), 1e-4)

    def test_constant_values_preserved_with_matched_denominator(self):
        scores = torch.linspace(-1.5, 1.5, 2 * model.CHUNK, dtype=torch.float64)[None, :]
        values = torch.full((2 * model.CHUNK, model.DIM), 3.25, dtype=torch.float64)
        for family in ["exact", "constant_bias", "continuous_native"]:
            for schedule in ["running_max", "global_max"]:
                for order in [[0, 1], [1, 0]]:
                    output, _ = model.online(scores, values, order, family, schedule, True)
                    self.assertTrue(torch.allclose(output, torch.full_like(output, 3.25), rtol=1e-12, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
