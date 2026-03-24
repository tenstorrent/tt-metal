# models/demos/blackhole/qwen3_5_9b/tests/test_model_config.py
"""Tests for Qwen3.5-9B model config loading."""
import pytest

CHECKPOINT_DIR = "/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"


class TestQwen35ModelArgs:
    @pytest.fixture(scope="class")
    def args(self):
        from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs

        return Qwen35ModelArgs(mesh_device=None, checkpoint_dir=CHECKPOINT_DIR)

    def test_core_dimensions(self, args):
        assert args.dim == 4096
        assert args.n_layers == 32
        assert args.n_heads == 16
        assert args.n_kv_heads == 4
        assert args.head_dim == 256
        assert args.hidden_dim == 12288
        assert args.vocab_size == 248320
        assert args.norm_eps == 1e-6

    def test_rope_config(self, args):
        assert args.rope_theta == 10_000_000
        assert args.partial_rotary_factor == 0.25
        assert args.rope_head_dim == 64

    def test_deltanet_config(self, args):
        assert args.linear_num_key_heads == 16
        assert args.linear_num_value_heads == 32
        assert args.linear_key_head_dim == 128
        assert args.linear_value_head_dim == 128
        assert args.linear_conv_kernel_dim == 4

    def test_attention_type_list(self, args):
        assert len(args.attention_type_list) == 32
        expected = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8
        assert args.attention_type_list == expected

    def test_full_attention_layer_indices(self, args):
        full_attn = [i for i, t in enumerate(args.attention_type_list) if t == "full_attention"]
        assert full_attn == [3, 7, 11, 15, 19, 23, 27, 31]

    def test_is_full_attention_layer(self, args):
        assert args.is_full_attention_layer(3) is True
        assert args.is_full_attention_layer(0) is False

    def test_is_deltanet_layer(self, args):
        assert args.is_deltanet_layer(0) is True
        assert args.is_deltanet_layer(3) is False

    def test_derived_dims(self, args):
        assert args.linear_q_dim == 2048
        assert args.linear_k_dim == 2048
        assert args.linear_v_dim == 4096
