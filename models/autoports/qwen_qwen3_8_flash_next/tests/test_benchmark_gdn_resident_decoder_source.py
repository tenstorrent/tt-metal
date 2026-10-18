# SPDX-License-Identifier: Apache-2.0
"""Host-only scope and strict-comparison checks for whole-decoder qualification."""
import inspect
import types
import unittest

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_gdn_resident_decoder_probe as probe


class DecoderProbeTests(unittest.TestCase):
    def test_private_binding_preserves_every_original_fallback(self):
        seen = []

        def original(*values):
            seen.append("original")
            return "original"

        def selected(self, *values):
            seen.append("candidate")
            return "candidate"

        attention = types.SimpleNamespace(_recurrence=original, prefill_qk_rm_enabled=True, recurrence_broadcast=True)
        view = types.SimpleNamespace(kind="linear_attention", attention=attention)
        module = types.SimpleNamespace(RECURRENCE=selected, gdn_method=lambda f: f)
        probe.install_candidate(view, module)
        for batch, sequence in ((1, 127), (1, 128), (1, 129), (1, 256), (1, 512), (1, 1024), (32, 1)):
            value = types.SimpleNamespace(shape=(batch, sequence, 2560))
            expected = "candidate" if batch == 1 and sequence in (128, 256, 512) else "original"
            self.assertEqual(attention._recurrence(value, None, None, None, None), expected)
        self.assertEqual(attention._resident_probe_shapes, {128, 256, 512})
        attention.prefill_qk_rm_enabled = False
        value = types.SimpleNamespace(shape=(1, 512, 2560))
        self.assertEqual(attention._recurrence(value, None, None, None, None), "original")
        self.assertIs(module.RECURRENCE, selected)

    def test_qsa_is_an_unchanged_control(self):
        attention = types.SimpleNamespace()
        view = types.SimpleNamespace(kind="indexed_attention", attention=attention)
        probe.install_candidate(view, object())
        self.assertEqual(vars(attention), {})

    def test_baseline_geometry_is_same_production_hybrid(self):
        source = inspect.getsource(probe.run_layer)
        self.assertIn('"original": _decoder(owner, {}, CANDIDATE_PREPARED)', source)
        self.assertIn('"candidate": _decoder(owner, {}, CANDIDATE_PREPARED)', source)
        self.assertIn('install_candidate(views["candidate"], args.candidate)', source)
        self.assertIn('require_exact_tree(raw["original"], raw["candidate"])', source)
        self.assertEqual(probe.LENGTHS, (127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4096))

    def test_exact_byte_comparison_supports_fp8_and_signed_zero(self):
        value = torch.tensor([1.0, 2.0], dtype=torch.float8_e4m3fn)
        probe.require_exact_tree({"x": [value]}, {"x": [value.clone()]})
        for left, right in (
            (torch.tensor([0.0]), torch.tensor([-0.0])),
            (torch.tensor([float("nan")]), torch.tensor([float("nan")])),
        ):
            with self.subTest(left=left, right=right):
                try:
                    probe.require_exact_tree(left, right)
                except AssertionError:
                    pass
                else:
                    self.fail("Invalid comparison unexpectedly passed")


if __name__ == "__main__":
    unittest.main()
