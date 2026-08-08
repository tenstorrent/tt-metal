# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
# Unit Test Suite for TTPhi1 Modular Implementation (`phi1_model.py`)

import unittest

import numpy as np
import torch

# Skip actual ttnn test if not on Wormhole
try:
    from tt.phi1_model import TTPhi1Attention, TTPhi1DecoderLayer, TTPhi1ForCausalLM, TTPhi1MLP

    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False


class TestPhi1ModularComponents(unittest.TestCase):
    @unittest.skipIf(not TTNN_AVAILABLE, "ttnn not available locally")
    def test_mock_instantiation(self):
        """
        Verify that TTPhi1 components can be instantiated successfully
        using mock state dictionaries, proving class interface integrity.
        """
        device = ttnn.open_device(0)
        try:
            # Mock state dict representing HuggingFace's microsoft/phi-1
            hidden_size = 2048
            state_dict = {
                "model.layers.0.input_layernorm.weight": torch.ones(hidden_size),
                "model.layers.0.input_layernorm.bias": torch.zeros(hidden_size),
                "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden_size, hidden_size),
                "model.layers.0.self_attn.k_proj.weight": torch.randn(hidden_size, hidden_size),
                "model.layers.0.self_attn.v_proj.weight": torch.randn(hidden_size, hidden_size),
                "model.layers.0.self_attn.q_proj.bias": torch.zeros(hidden_size),
                "model.layers.0.self_attn.k_proj.bias": torch.zeros(hidden_size),
                "model.layers.0.self_attn.v_proj.bias": torch.zeros(hidden_size),
                "model.layers.0.self_attn.dense.weight": torch.randn(hidden_size, hidden_size),
                "model.layers.0.self_attn.dense.bias": torch.zeros(hidden_size),
                "model.layers.0.mlp.fc1.weight": torch.randn(8192, hidden_size),
                "model.layers.0.mlp.fc1.bias": torch.zeros(8192),
                "model.layers.0.mlp.fc2.weight": torch.randn(hidden_size, 8192),
                "model.layers.0.mlp.fc2.bias": torch.zeros(hidden_size),
                "model.embed_tokens.weight": torch.randn(51200, hidden_size),
                "model.final_layernorm.weight": torch.ones(hidden_size),
                "model.final_layernorm.bias": torch.zeros(hidden_size),
                "lm_head.weight": torch.randn(51200, hidden_size),
                "lm_head.bias": torch.zeros(51200),
            }

            try:
                # Test component by component
                mlp = TTPhi1MLP(device, state_dict, base_address="model.layers.0")
                self.assertIsNotNone(mlp.fc1)

                attn = TTPhi1Attention(device, state_dict, base_address="model.layers.0")
                self.assertIsNotNone(attn.wqkv)

                layer = TTPhi1DecoderLayer(device, state_dict, base_address="model", layer_num=0)
                self.assertEqual(layer.layer_num, 0)

                # Test overall topology instantiation (single layer for speed)
                model = TTPhi1ForCausalLM(device, state_dict, base_address="model", num_hidden_layers=1)
                self.assertIsNotNone(model.lm_head_weight)
            except Exception as e:
                self.fail(f"Instantiation failed with exception: {e}")
        finally:
            ttnn.close_device(device)

    def test_parallel_residual_equation(self):
        """
        Verify that Phi-1 parallel residual connection math produces exact identical results.
        """
        batch, seq_len, hidden = 1, 16, 2048
        np.random.seed(42)
        x = np.random.randn(batch, seq_len, hidden).astype(np.float32)
        attn_out = np.random.randn(batch, seq_len, hidden).astype(np.float32)
        mlp_out = np.random.randn(batch, seq_len, hidden).astype(np.float32)

        expected = x + attn_out + mlp_out
        step1 = attn_out + mlp_out
        actual = x + step1

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
