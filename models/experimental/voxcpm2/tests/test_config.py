# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import unittest
from models.experimental.voxcpm2.config import validate_minicpm_config


def config():
    return dict(hidden_size=128, intermediate_size=256, num_attention_heads=4,
                num_key_value_heads=2, num_hidden_layers=2, rms_norm_eps=1e-6,
                max_position_embeddings=256, rope_theta=10000.0,
                rope_scaling=dict(original_max_position_embeddings=128,
                                  long_factor=[2.0]*16, short_factor=[1.0]*16))


class ConfigTests(unittest.TestCase):
    def test_gqa_channels_and_separate_projection_width(self):
        value = config()
        value.update(kv_channels=32, hidden_size=96)
        self.assertEqual(validate_minicpm_config(value), 32)

    def test_invalid_checkpoint_contract(self):
        for mutation in [{'num_key_value_heads': 3}, {'hidden_size': 127},
                         {'num_hidden_layers': 0}, {'kv_channels': 31},
                         {'rope_scaling': {}}]:
            with self.subTest(mutation=mutation):
                value = config()
                value.update(mutation)
                with self.assertRaises(ValueError):
                    validate_minicpm_config(value)


if __name__ == '__main__':
    unittest.main()
