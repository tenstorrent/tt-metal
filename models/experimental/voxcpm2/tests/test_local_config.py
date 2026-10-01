# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host contract checks only; these are not device accuracy tests."""

import unittest

from models.experimental.voxcpm2.tt.local import make_local_config


class TestLocalConfig(unittest.TestCase):
    def setUp(self):
        self.base = {
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "num_hidden_layers": 24,
            "vocab_size": 32000,
            "rope_scaling": {"long_factor": [1.0, 2.0]},
            "scale_depth": 1.4,
            "no_rope": False,
        }
        self.component = {"hidden_dim": 1024, "ffn_dim": 4096, "num_heads": 16, "num_layers": 4}

    def test_inherits_numerically_significant_settings_without_mutation(self):
        local = make_local_config(self.base, self.component)
        self.assertEqual(local["num_key_value_heads"], 2)
        self.assertEqual(local["scale_depth"], 1.4)
        self.assertEqual(local["vocab_size"], 0)
        self.assertEqual(local["hidden_size"], 1024)
        self.assertIsNone(local["kv_channels"])
        local["rope_scaling"]["long_factor"][0] = 99.0
        self.assertEqual(self.base["rope_scaling"]["long_factor"][0], 1.0)
        self.assertEqual(self.base["vocab_size"], 32000)

    def test_rejects_incompatible_inherited_kv_heads(self):
        self.component["num_heads"] = 3
        with self.assertRaisesRegex(ValueError, "KV heads"):
            make_local_config(self.base, self.component)

    def test_explicit_kv_channels_allows_distinct_attention_width(self):
        self.component.update(hidden_dim=1025, kv_channels=64)
        self.assertEqual(make_local_config(self.base, self.component)["kv_channels"], 64)

    def test_rejects_boolean_as_layer_count(self):
        self.component["num_layers"] = True
        with self.assertRaisesRegex(ValueError, "num_layers"):
            make_local_config(self.base, self.component)

    def test_rejects_nondivisible_hidden_without_kv_channels(self):
        self.component["hidden_dim"] = 1025
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            make_local_config(self.base, self.component)


if __name__ == "__main__":
    unittest.main()
