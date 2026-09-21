# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Qwen3.5 / 3.6 HF -> internal weight remapping.

Every expected shape is derived from the loaded checkpoint's config via the ``dims``
fixture, so this file runs unchanged against 2B / 9B / 27B / 35B-A3B. Hardcoding the
9B geometry here previously made the suite fail on every other checkpoint.
"""
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.tp_common import replicate_kv_weight
from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict


def _load_raw_state_dict(checkpoint_dir):
    """Load raw HF state dict using safetensors."""
    import glob

    from safetensors import safe_open

    # Qwen ships both spellings: ``model-000NN-of-000MM`` and ``model.safetensors-000NN-of-000MM``.
    paths = sorted(glob.glob(f"{checkpoint_dir}/model.safetensors-*.safetensors")) or sorted(
        glob.glob(f"{checkpoint_dir}/model-*.safetensors")
    )
    assert paths, f"No safetensors shards found in {checkpoint_dir}"
    state_dict = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


@pytest.fixture(scope="module")
def args():
    return Qwen36ModelArgs(mesh_device=None)


@pytest.fixture(scope="module")
def dims(args):
    """Expected weight shapes, derived from the checkpoint config rather than hardcoded."""
    layer_types = args.attention_type_list
    return SimpleNamespace(
        dim=args.dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        head_dim=args.head_dim,
        linear_k_dim=args.linear_k_dim,
        linear_v_dim=args.linear_v_dim,
        # a/b decay projections, A_log and dt_bias are all per value-head.
        n_v_heads=args.linear_num_value_heads,
        v_head_dim=args.linear_value_head_dim,
        conv_k=args.linear_conv_kernel_dim,
        # q_proj is 2x wide: query and the output gate are fused into one projection.
        full_attn_q_dim=args.n_heads * args.head_dim * 2,
        full_attn_kv_dim=args.n_kv_heads * args.head_dim,
        gdn_layers=[i for i, t in enumerate(layer_types) if t == "linear_attention"],
        full_attn_layers=[i for i, t in enumerate(layer_types) if t == "full_attention"],
    )


@pytest.fixture(scope="module")
def gdn_layer(dims):
    """Index of the first DeltaNet layer (0 on every checkpoint so far)."""
    return dims.gdn_layers[0]


@pytest.fixture(scope="module")
def attn_layer(dims):
    """Index of the first full-attention layer (3 at full_attention_interval=4)."""
    return dims.full_attn_layers[0]


@pytest.fixture(scope="module")
def raw_state_dict(args):
    return _load_raw_state_dict(args.CKPT_DIR)


@pytest.fixture(scope="module")
def remapped(raw_state_dict):
    return remap_qwen36_state_dict(raw_state_dict)


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
    def test_embed_tokens(self, remapped, dims):
        assert "tok_embeddings.weight" in remapped
        assert remapped["tok_embeddings.weight"].shape == (dims.vocab_size, dims.dim)

    def test_lm_head(self, remapped, dims):
        """Present on untied checkpoints from lm_head.weight, and on tied ones
        (Qwen3.5-2B) via the embedding alias in ``_tie_output_weight``."""
        assert "output.weight" in remapped
        assert remapped["output.weight"].shape == (dims.vocab_size, dims.dim)

    def test_final_norm(self, remapped, dims):
        assert "norm.weight" in remapped
        assert remapped["norm.weight"].shape == (dims.dim,)


class TestDeltaNetLayerWeights:
    """Test the first DeltaNet / linear-attention layer."""

    def test_qkv_combined_only(self, remapped, dims, gdn_layer):
        qkv = remapped[f"layers.{gdn_layer}.linear_attn.qkv_proj.weight"]
        assert qkv.shape == (2 * dims.linear_k_dim + dims.linear_v_dim, dims.dim)
        # Split q/k/v_proj are no longer emitted (the op uses the combined weight)
        for name in ("q_proj", "k_proj", "v_proj"):
            assert f"layers.{gdn_layer}.linear_attn.{name}.weight" not in remapped

    def test_conv1d_split(self, remapped, dims, gdn_layer):
        pre = f"layers.{gdn_layer}.linear_attn"
        assert remapped[f"{pre}.q_conv.weight"].shape == (dims.linear_k_dim, 1, dims.conv_k)
        assert remapped[f"{pre}.k_conv.weight"].shape == (dims.linear_k_dim, 1, dims.conv_k)
        assert remapped[f"{pre}.v_conv.weight"].shape == (dims.linear_v_dim, 1, dims.conv_k)

    def test_decay_projections(self, remapped, dims, gdn_layer):
        pre = f"layers.{gdn_layer}.linear_attn"
        assert remapped[f"{pre}.in_proj_a.weight"].shape == (dims.n_v_heads, dims.dim)
        assert remapped[f"{pre}.in_proj_b.weight"].shape == (dims.n_v_heads, dims.dim)

    def test_gate_projection(self, remapped, dims, gdn_layer):
        assert remapped[f"layers.{gdn_layer}.linear_attn.in_proj_z.weight"].shape == (dims.linear_v_dim, dims.dim)

    def test_output_proj(self, remapped, dims, gdn_layer):
        assert remapped[f"layers.{gdn_layer}.linear_attn.out_proj.weight"].shape == (dims.dim, dims.linear_v_dim)

    def test_a_log_and_dt_bias(self, remapped, dims, gdn_layer):
        pre = f"layers.{gdn_layer}.linear_attn"
        assert remapped[f"{pre}.A_log"].shape == (dims.n_v_heads,)
        assert remapped[f"{pre}.dt_bias"].shape == (dims.n_v_heads,)

    def test_norm(self, remapped, dims, gdn_layer):
        assert remapped[f"layers.{gdn_layer}.linear_attn.norm.weight"].shape == (dims.v_head_dim,)

    def test_mlp(self, remapped, dims, gdn_layer):
        pre = f"layers.{gdn_layer}.mlp"
        assert remapped[f"{pre}.gate_proj.weight"].shape == (dims.hidden_dim, dims.dim)
        assert remapped[f"{pre}.up_proj.weight"].shape == (dims.hidden_dim, dims.dim)
        assert remapped[f"{pre}.down_proj.weight"].shape == (dims.dim, dims.hidden_dim)

    def test_layernorms(self, remapped, dims, gdn_layer):
        assert remapped[f"layers.{gdn_layer}.input_layernorm.weight"].shape == (dims.dim,)
        assert remapped[f"layers.{gdn_layer}.post_attention_layernorm.weight"].shape == (dims.dim,)


class TestGatedAttentionLayerWeights:
    """Test the first Gated Full Attention layer."""

    def test_q_proj(self, remapped, dims, attn_layer):
        assert remapped[f"layers.{attn_layer}.self_attn.q_proj.weight"].shape == (dims.full_attn_q_dim, dims.dim)

    def test_kv_proj(self, remapped, dims, attn_layer):
        pre = f"layers.{attn_layer}.self_attn"
        assert remapped[f"{pre}.k_proj.weight"].shape == (dims.full_attn_kv_dim, dims.dim)
        assert remapped[f"{pre}.v_proj.weight"].shape == (dims.full_attn_kv_dim, dims.dim)

    def test_o_proj(self, remapped, dims, attn_layer):
        assert remapped[f"layers.{attn_layer}.self_attn.o_proj.weight"].shape == (dims.dim, dims.dim)

    def test_qk_norm(self, remapped, dims, attn_layer):
        pre = f"layers.{attn_layer}.self_attn"
        assert remapped[f"{pre}.q_norm.weight"].shape == (dims.head_dim,)
        assert remapped[f"{pre}.k_norm.weight"].shape == (dims.head_dim,)

    def test_mlp(self, remapped, dims, attn_layer):
        assert remapped[f"layers.{attn_layer}.mlp.gate_proj.weight"].shape == (dims.hidden_dim, dims.dim)

    def test_layernorms(self, remapped, dims, attn_layer):
        assert remapped[f"layers.{attn_layer}.input_layernorm.weight"].shape == (dims.dim,)
        assert remapped[f"layers.{attn_layer}.post_attention_layernorm.weight"].shape == (dims.dim,)


class TestAllLayersPresent:
    def test_all_layers_have_mlp(self, remapped, dims):
        for i in range(dims.n_layers):
            assert f"layers.{i}.mlp.gate_proj.weight" in remapped, f"Missing MLP for layer {i}"

    def test_deltanet_layers_count(self, remapped, dims):
        found = [i for i in range(dims.n_layers) if f"layers.{i}.linear_attn.qkv_proj.weight" in remapped]
        assert found == dims.gdn_layers

    def test_full_attn_layers_count(self, remapped, dims):
        found = [i for i in range(dims.n_layers) if f"layers.{i}.self_attn.q_proj.weight" in remapped]
        assert found == dims.full_attn_layers

    def test_full_attn_at_correct_positions(self, remapped, dims):
        for i in dims.full_attn_layers:
            assert f"layers.{i}.self_attn.q_proj.weight" in remapped, f"Layer {i} should be full attention"


class TestReplicateKVWeight:
    """KV-head replication for TP > n_kv_heads (27B: 4 KV heads on TP=8).

    No fixtures / no checkpoint / no device — pure host tensor reshaping, so this runs
    anywhere. Guards the invariant the TP attention loader depends on: device ``d`` must
    receive exactly the KV head that its GQA query group attends to.
    """

    # Qwen3.6-27B full-attention geometry.
    N_HEADS = 24
    N_KV_HEADS = 4
    HEAD_DIM = 256
    IN_DIM = 512  # narrowed from 5120; the helper is agnostic to the input width

    def _kv_weight(self):
        """[n_kv_heads*head_dim, in] where every row of head h holds the value h."""
        return torch.cat([torch.full((self.HEAD_DIM, self.IN_DIM), float(h)) for h in range(self.N_KV_HEADS)], dim=0)

    def _heads_per_device(self, replicated, tp):
        """The KV head index each device ends up with."""
        rows = max(1, self.N_KV_HEADS // tp) * self.HEAD_DIM
        return [int(replicated[d * rows : (d + 1) * rows].unique().item()) for d in range(tp)]

    @pytest.mark.parametrize("tp", [1, 2, 4])
    def test_noop_when_tp_fits_kv_heads(self, tp):
        """tp <= n_kv_heads needs no replication — must return the very same object, so
        TP<=4 weight prep stays bit-identical."""
        w = self._kv_weight()
        assert replicate_kv_weight(w, self.N_KV_HEADS, tp, self.HEAD_DIM) is w

    def test_tp8_shape(self):
        out = replicate_kv_weight(self._kv_weight(), self.N_KV_HEADS, 8, self.HEAD_DIM)
        assert out.shape == (8 * self.HEAD_DIM, self.IN_DIM)

    def test_tp8_pairs_devices_on_one_head(self):
        out = replicate_kv_weight(self._kv_weight(), self.N_KV_HEADS, 8, self.HEAD_DIM)
        assert self._heads_per_device(out, 8) == [0, 0, 1, 1, 2, 2, 3, 3]

    @pytest.mark.parametrize("tp", [4, 8])
    def test_matches_gqa_query_grouping(self, tp):
        """The head each device gets must be the one its local query heads map to under
        GQA — otherwise attention silently reads the wrong K/V."""
        out = replicate_kv_weight(self._kv_weight(), self.N_KV_HEADS, tp, self.HEAD_DIM)
        q_per_device = self.N_HEADS // tp
        heads_per_kv = self.N_HEADS // self.N_KV_HEADS
        expected = [(d * q_per_device) // heads_per_kv for d in range(tp)]
        assert self._heads_per_device(out, tp) == expected

    def test_every_head_still_reachable(self):
        """Replication must not drop a head: all n_kv_heads appear across the mesh."""
        out = replicate_kv_weight(self._kv_weight(), self.N_KV_HEADS, 8, self.HEAD_DIM)
        assert sorted(set(self._heads_per_device(out, 8))) == list(range(self.N_KV_HEADS))
