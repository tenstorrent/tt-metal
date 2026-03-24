# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Qwen3.5-9B HF→internal weight remapping."""
import pytest

from models.demos.blackhole.qwen3_5_9b.tt.weight_mapping import remap_qwen35_state_dict

CHECKPOINT_DIR = "/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"
HIDDEN_SIZE = 4096
NUM_LAYERS = 32
LINEAR_KEY_DIM = 2048  # 16 heads × 128 head_dim
LINEAR_VALUE_DIM = 4096  # 32 heads × 128 head_dim
FULL_ATTN_Q_DIM = 8192  # 16 heads × 256 head_dim × 2 (query + gate)
FULL_ATTN_KV_DIM = 1024  # 4 heads × 256 head_dim


def _load_raw_state_dict():
    """Load raw HF state dict using safetensors."""
    import glob

    from safetensors import safe_open

    state_dict = {}
    for path in sorted(glob.glob(f"{CHECKPOINT_DIR}/model.safetensors-*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


@pytest.fixture(scope="module")
def raw_state_dict():
    return _load_raw_state_dict()


@pytest.fixture(scope="module")
def remapped(raw_state_dict):
    return remap_qwen35_state_dict(raw_state_dict)


class TestPrefixStripping:
    def test_no_model_language_model_prefix(self, remapped):
        for key in remapped:
            assert not key.startswith("model.language_model."), f"Prefix not stripped: {key}"

    def test_no_visual_keys(self, remapped):
        for key in remapped:
            assert "visual" not in key, f"Vision key not filtered: {key}"

    def test_no_mtp_keys(self, remapped):
        for key in remapped:
            assert "mtp" not in key.split(".")[0], f"MTP key not filtered: {key}"


class TestTopLevelWeights:
    def test_embed_tokens(self, remapped):
        assert "tok_embeddings.weight" in remapped
        assert remapped["tok_embeddings.weight"].shape == (248320, HIDDEN_SIZE)

    def test_lm_head(self, remapped):
        assert "output.weight" in remapped
        assert remapped["output.weight"].shape == (248320, HIDDEN_SIZE)

    def test_final_norm(self, remapped):
        assert "norm.weight" in remapped
        assert remapped["norm.weight"].shape == (HIDDEN_SIZE,)


class TestDeltaNetLayerWeights:
    """Test layer 0 (a DeltaNet/linear attention layer)."""

    def test_qkv_split(self, remapped):
        q = remapped["layers.0.linear_attn.q_proj.weight"]
        k = remapped["layers.0.linear_attn.k_proj.weight"]
        v = remapped["layers.0.linear_attn.v_proj.weight"]
        assert q.shape == (LINEAR_KEY_DIM, HIDDEN_SIZE)
        assert k.shape == (LINEAR_KEY_DIM, HIDDEN_SIZE)
        assert v.shape == (LINEAR_VALUE_DIM, HIDDEN_SIZE)

    def test_conv1d_split(self, remapped):
        q_conv = remapped["layers.0.linear_attn.q_conv.weight"]
        k_conv = remapped["layers.0.linear_attn.k_conv.weight"]
        v_conv = remapped["layers.0.linear_attn.v_conv.weight"]
        assert q_conv.shape == (LINEAR_KEY_DIM, 1, 4)
        assert k_conv.shape == (LINEAR_KEY_DIM, 1, 4)
        assert v_conv.shape == (LINEAR_VALUE_DIM, 1, 4)

    def test_decay_projections(self, remapped):
        a = remapped["layers.0.linear_attn.in_proj_a.weight"]
        b = remapped["layers.0.linear_attn.in_proj_b.weight"]
        assert a.shape == (32, HIDDEN_SIZE)
        assert b.shape == (32, HIDDEN_SIZE)

    def test_gate_projection(self, remapped):
        z = remapped["layers.0.linear_attn.in_proj_z.weight"]
        assert z.shape == (HIDDEN_SIZE, HIDDEN_SIZE)

    def test_output_proj(self, remapped):
        o = remapped["layers.0.linear_attn.out_proj.weight"]
        assert o.shape == (HIDDEN_SIZE, HIDDEN_SIZE)

    def test_a_log_and_dt_bias(self, remapped):
        assert remapped["layers.0.linear_attn.A_log"].shape == (32,)
        assert remapped["layers.0.linear_attn.dt_bias"].shape == (32,)

    def test_norm(self, remapped):
        assert remapped["layers.0.linear_attn.norm.weight"].shape == (128,)

    def test_mlp(self, remapped):
        assert remapped["layers.0.mlp.gate_proj.weight"].shape == (12288, HIDDEN_SIZE)
        assert remapped["layers.0.mlp.up_proj.weight"].shape == (12288, HIDDEN_SIZE)
        assert remapped["layers.0.mlp.down_proj.weight"].shape == (HIDDEN_SIZE, 12288)

    def test_layernorms(self, remapped):
        assert remapped["layers.0.input_layernorm.weight"].shape == (HIDDEN_SIZE,)
        assert remapped["layers.0.post_attention_layernorm.weight"].shape == (HIDDEN_SIZE,)


class TestGatedAttentionLayerWeights:
    """Test layer 3 (a Gated Full Attention layer)."""

    def test_q_proj(self, remapped):
        q = remapped["layers.3.self_attn.q_proj.weight"]
        assert q.shape == (FULL_ATTN_Q_DIM, HIDDEN_SIZE)

    def test_kv_proj(self, remapped):
        k = remapped["layers.3.self_attn.k_proj.weight"]
        v = remapped["layers.3.self_attn.v_proj.weight"]
        assert k.shape == (FULL_ATTN_KV_DIM, HIDDEN_SIZE)
        assert v.shape == (FULL_ATTN_KV_DIM, HIDDEN_SIZE)

    def test_o_proj(self, remapped):
        o = remapped["layers.3.self_attn.o_proj.weight"]
        assert o.shape == (HIDDEN_SIZE, HIDDEN_SIZE)

    def test_qk_norm(self, remapped):
        assert remapped["layers.3.self_attn.q_norm.weight"].shape == (256,)
        assert remapped["layers.3.self_attn.k_norm.weight"].shape == (256,)

    def test_mlp(self, remapped):
        assert remapped["layers.3.mlp.gate_proj.weight"].shape == (12288, HIDDEN_SIZE)

    def test_layernorms(self, remapped):
        assert remapped["layers.3.input_layernorm.weight"].shape == (HIDDEN_SIZE,)
        assert remapped["layers.3.post_attention_layernorm.weight"].shape == (HIDDEN_SIZE,)


class TestAllLayersPresent:
    def test_all_32_layers_have_mlp(self, remapped):
        for i in range(32):
            assert f"layers.{i}.mlp.gate_proj.weight" in remapped, f"Missing MLP for layer {i}"

    def test_deltanet_layers_count(self, remapped):
        deltanet_layers = [i for i in range(32) if f"layers.{i}.linear_attn.q_proj.weight" in remapped]
        assert len(deltanet_layers) == 24

    def test_full_attn_layers_count(self, remapped):
        attn_layers = [i for i in range(32) if f"layers.{i}.self_attn.q_proj.weight" in remapped]
        assert len(attn_layers) == 8

    def test_full_attn_at_correct_positions(self, remapped):
        expected = [3, 7, 11, 15, 19, 23, 27, 31]
        for i in expected:
            assert f"layers.{i}.self_attn.q_proj.weight" in remapped, f"Layer {i} should be full attention"
