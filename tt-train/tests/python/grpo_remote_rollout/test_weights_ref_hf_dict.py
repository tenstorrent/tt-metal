# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One-device check of ``LlamaHfExport.weights_ref_hf_dict``."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

import ttnn
import ttml
from ttml.models import WeightTyingType
from ttml.models.llama import LlamaConfig

from grpo_remote_rollout.utils.llama_overrides import LlamaHfExport

pytestmark = pytest.mark.requires_device

HIDDEN = 64
NUM_HEADS = 2
NUM_KV_HEADS = 1
KV_DIM = NUM_KV_HEADS * HIDDEN // NUM_HEADS
INTERMEDIATE = 128
NUM_LAYERS = 2


def to_numpy(tensor: ttnn.Tensor) -> np.ndarray:
    return ttml.autograd.create_tensor(tensor).to_numpy(ttnn.DataType.FLOAT32)


def bf16_block(rng: np.random.Generator, rows: int) -> np.ndarray:
    """A (1, 1, rows, HIDDEN) block, rounded through bf16 so it survives the trip to the device unchanged."""
    return rng.standard_normal((1, 1, rows, HIDDEN), dtype=np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


def test_weights_ref_hf_dict_slices_fused_projections():
    config = LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        vocab_size=64,
        max_position_embeddings=64,
        weight_tying=WeightTyingType.Enabled,
    )
    model = LlamaHfExport(config)
    params = model.parameters()
    rng = np.random.default_rng(0)
    # Every gamma starts at ones, which would let a swapped pair of norms pass.
    for name, param in params.items():
        if name.endswith("/gamma"):
            ttml.autograd.assign_numpy(param, bf16_block(rng, 1))

    expected: dict[str, np.ndarray] = {}
    for i in range(NUM_LAYERS):
        p = f"Llama/blocks/{i}"
        hf = f"model.layers.{i}"
        q, k, v = bf16_block(rng, HIDDEN), bf16_block(rng, KV_DIM), bf16_block(rng, KV_DIM)
        ttml.autograd.assign_numpy(params[f"{p}/attention/qkv_linear/weight"], np.concatenate([q, k, v], axis=2))
        gate, up = bf16_block(rng, INTERMEDIATE), bf16_block(rng, INTERMEDIATE)
        ttml.autograd.assign_numpy(params[f"{p}/mlp/w_gate_up/weight"], np.concatenate([gate, up], axis=2))

        expected[f"{hf}.self_attn.q_proj.weight"] = q
        expected[f"{hf}.self_attn.k_proj.weight"] = k
        expected[f"{hf}.self_attn.v_proj.weight"] = v
        expected[f"{hf}.mlp.gate_proj.weight"] = gate
        expected[f"{hf}.mlp.up_proj.weight"] = up
        # The rest are passed through as they are.
        for hf_name, name in [
            ("self_attn.o_proj", "attention/out_linear"),
            ("mlp.down_proj", "mlp/w2"),
            ("input_layernorm", "attention_norm"),
            ("post_attention_layernorm", "mlp_norm"),
        ]:
            param = "gamma" if hf_name.endswith("layernorm") else "weight"
            expected[f"{hf}.{hf_name}.weight"] = params[f"{p}/{name}/{param}"].to_numpy(ttnn.DataType.FLOAT32)

    fc = params["Llama/fc/weight"].to_numpy(ttnn.DataType.FLOAT32)
    expected["model.embed_tokens.weight"] = fc
    expected["lm_head.weight"] = fc
    expected["model.norm.weight"] = params["Llama/ln_fc/gamma"].to_numpy(ttnn.DataType.FLOAT32)

    exported = model.weights_ref_hf_dict()

    assert sorted(exported) == sorted(expected)
    for key, value in expected.items():
        np.testing.assert_array_equal(to_numpy(exported[key]), value, err_msg=key)
