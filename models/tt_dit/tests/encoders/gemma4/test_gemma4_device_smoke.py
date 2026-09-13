# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device smoke test for the Gemma-4 encoder.

Runs a narrowed stack (small hidden/vocab, one sliding plus one global layer) that
keeps the head geometry the real checkpoint uses: head_dim 256 sliding against 512
global, a single global KV head fanned out to all 16 query heads, and V tied to K.
Those are the shapes the ttnn ops are most likely to reject, so exercising them does
not need the full 12B model. Numerical parity is a separate test.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[6]))

import pytest
import torch

import ttnn
from models.tt_dit.encoders.gemma4.model_gemma import FULL_ATTENTION, SLIDING_ATTENTION, Gemma4Config, Gemma4Encoder
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager

VOCAB = 1000
HIDDEN = 1024
INTERMEDIATE = 2048
NUM_LAYERS = 6
SEQ_LEN = 128


def _random_state_dict(config: Gemma4Config) -> dict[str, torch.Tensor]:
    """Checkpoint-shaped random weights, using the packed file's flat ``model.`` prefix."""
    state = {
        "model.embed_tokens.weight": torch.randn(config.vocab_size, config.hidden_size) * 0.02,
        "model.norm.weight": torch.randn(config.hidden_size) * 0.02,
    }
    for idx in range(config.num_hidden_layers):
        is_global = config.is_global(idx)
        head_dim = config.attn_head_dim(is_global)
        kv_heads = config.attn_kv_heads(is_global)
        prefix = f"model.layers.{idx}."
        state[prefix + "layer_scalar"] = torch.tensor([1.0 + 0.01 * idx])
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ]:
            state[prefix + norm + ".weight"] = torch.randn(config.hidden_size) * 0.02
        attn = prefix + "self_attn."
        state[attn + "q_norm.weight"] = torch.randn(head_dim) * 0.02
        state[attn + "k_norm.weight"] = torch.randn(head_dim) * 0.02
        state[attn + "q_proj.weight"] = torch.randn(config.num_attention_heads * head_dim, config.hidden_size) * 0.02
        state[attn + "k_proj.weight"] = torch.randn(kv_heads * head_dim, config.hidden_size) * 0.02
        # Global layers tie V to K, so the checkpoint has no v_proj for them.
        if not (config.attention_k_eq_v and is_global):
            state[attn + "v_proj.weight"] = torch.randn(kv_heads * head_dim, config.hidden_size) * 0.02
        state[attn + "o_proj.weight"] = torch.randn(config.hidden_size, config.num_attention_heads * head_dim) * 0.02
        mlp = prefix + "mlp."
        state[mlp + "gate_proj.weight"] = torch.randn(config.intermediate_size, config.hidden_size) * 0.02
        state[mlp + "up_proj.weight"] = torch.randn(config.intermediate_size, config.hidden_size) * 0.02
        state[mlp + "down_proj.weight"] = torch.randn(config.hidden_size, config.intermediate_size) * 0.02
    return state


def _narrow_config() -> Gemma4Config:
    return Gemma4Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        max_position_embeddings=SEQ_LEN,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_gemma4_encoder_runs(*, mesh_device):
    torch.manual_seed(0)
    config = _narrow_config()
    assert [i for i in range(NUM_LAYERS) if config.is_global(i)] == [5]
    assert config.layer_types[0] == SLIDING_ATTENTION and config.layer_types[5] == FULL_ATTENTION

    parallel_config = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    encoder = Gemma4Encoder(config, mesh_device, ccl_manager, parallel_config, max_seq_len=SEQ_LEN)
    encoder.load_torch_state_dict(_random_state_dict(config))

    token_ids = torch.randint(0, VOCAB, (1, SEQ_LEN))
    tt_ids = ttnn.from_torch(token_ids, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    hidden_states = encoder(tt_ids)

    # embedding + one per layer + final norm
    assert len(hidden_states) == NUM_LAYERS + 2
    for idx, tensor in enumerate(hidden_states):
        host = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()
        assert host.shape[-2:] == (SEQ_LEN, HIDDEN), f"state {idx} has shape {tuple(host.shape)}"
        assert torch.isfinite(host).all(), f"state {idx} contains non-finite values"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_gemma4_rejects_prompts_past_the_window(*, mesh_device, expect_error):
    """The sliding layers run as plain causal attention, which is only equivalent to the
    reference while the prompt fits inside the window."""
    config = _narrow_config()
    config.sliding_window = 32

    parallel_config = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    encoder = Gemma4Encoder(config, mesh_device, ccl_manager, parallel_config, max_seq_len=SEQ_LEN)

    token_ids = torch.randint(0, VOCAB, (1, SEQ_LEN))
    tt_ids = ttnn.from_torch(token_ids, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    with expect_error(NotImplementedError, "sliding window"):
        encoder(tt_ids)
