# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Small CPU mutation/policy probes, not model-scale accuracy evidence."""

import unittest

import torch
from transformers import LlamaConfig

from models.demos.llama_3p1_8b_d_p.tests.full_model import reference


class NativeInputMutationTests(unittest.TestCase):
    # A strong synthetic residual fixture detects omitted or swapped layer math using the same-input oracle.
    def test_synthetic_skip_and_wrong_weight_sensitivity(self):
        config = LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16384,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            rope_scaling={
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        )
        shapes = {
            "input_layernorm.weight": (32,),
            "post_attention_layernorm.weight": (32,),
            "self_attn.q_proj.weight": (32, 32),
            "self_attn.k_proj.weight": (16, 32),
            "self_attn.v_proj.weight": (16, 32),
            "self_attn.o_proj.weight": (32, 32),
            "mlp.gate_proj.weight": (64, 32),
            "mlp.up_proj.weight": (64, 32),
            "mlp.down_proj.weight": (32, 64),
        }
        generator = torch.Generator().manual_seed(73001)
        layers = []
        for _ in range(4):
            layers.append(
                {
                    name: (torch.ones(shape) if len(shape) == 1 else torch.randn(shape, generator=generator) * 0.15)
                    for name, shape in shapes.items()
                }
            )
        actual_input = torch.randn((17, 32), generator=generator) * 0.2
        for layer_idx in range(4):
            expected, _, _ = reference.reference_layer(actual_input, layers[layer_idx], config)
            swapped, _, _ = reference.reference_layer(actual_input, layers[(layer_idx + 1) % 4], config)
            for mutation in (actual_input, swapped):
                pcc, nl2 = reference.metrics(expected, mutation)
                self.assertFalse(pcc >= 0.999 and nl2 <= 0.025, (layer_idx, pcc, nl2))
            actual_input = expected


if __name__ == "__main__":
    unittest.main(verbosity=2)
