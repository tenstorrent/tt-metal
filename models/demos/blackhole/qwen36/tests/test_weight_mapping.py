# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Qwen3.6 HF -> internal weight remapping.

Every expected shape is derived from the checkpoint's own config.json / Qwen36ModelArgs, so the
same assertions hold for any Qwen3.5/3.6 variant (the 27B P0 target included). Needs the HF
checkpoint on disk (HF_MODEL); no device. Tensors are loaded as shape-only meta tensors, so the
27B checkpoint (~54 GB) never touches host RAM.
"""

import glob
import json
import os

import pytest
import torch

from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.tp_common import replicate_kv_weight
from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict


def _load_raw_state_dict(checkpoint_dir):
    """Raw HF state dict from the sharded safetensors (model-0000N-of-0000M.safetensors), as
    shape-only meta tensors: the remap is pure shape/naming logic, so no data is needed."""
    from safetensors import safe_open

    paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    assert paths, f"no safetensors under {checkpoint_dir}"
    state_dict = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = torch.empty(f.get_slice(key).get_shape(), dtype=torch.bfloat16, device="meta")
    return state_dict


@pytest.fixture(scope="module")
def args():
    return Qwen36ModelArgs(mesh_device=None)


@pytest.fixture(scope="module")
def geom(args):
    """Expected geometry, all from the checkpoint config (no variant-specific constants)."""
    with open(os.path.join(args.CKPT_DIR, "config.json")) as f:
        cfg = json.load(f)
    text = cfg.get("text_config", cfg)
    layer_types = list(args.attention_type_list)
    return dict(
        hidden=args.dim,
        vocab=args.vocab_size,
        n_layers=len(layer_types),
        full_attn_layers=[i for i, t in enumerate(layer_types) if t == "full_attention"],
        linear_layers=[i for i, t in enumerate(layer_types) if t == "linear_attention"],
        intermediate=text["intermediate_size"],
        head_dim=args.head_dim,
        q_dim=args.n_heads * args.head_dim * 2,  # query + gate (2x wide q_proj)
        o_in=args.n_heads * args.head_dim,
        kv_dim=args.n_kv_heads * args.head_dim,
        lin_k=args.linear_k_dim,
        lin_v=args.linear_v_dim,
        lin_v_heads=args.linear_num_value_heads,
        lin_v_head_dim=args.linear_value_head_dim,
        conv_k=text.get("linear_conv_kernel_dim", 4),
    )


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

    def test_mtp_keys_present(self, remapped, geom):
        """MTP (spec-decode drafter) weights are KEPT verbatim through remap."""
        expected = {
            "mtp.fc.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.norm.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.layers.0.mlp.gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight",
            "mtp.layers.0.mlp.down_proj.weight",
        }
        missing = expected - set(remapped)
        assert not missing, f"MTP keys dropped by remap: {sorted(missing)}"
        # fc (eh_proj) maps concat(token_emb, hidden) -> hidden, i.e. [hidden, 2*hidden].
        h = geom["hidden"]
        assert remapped["mtp.fc.weight"].shape == (h, 2 * h)


class TestTopLevelWeights:
    def test_embed_tokens(self, remapped, geom):
        assert remapped["tok_embeddings.weight"].shape == (geom["vocab"], geom["hidden"])

    def test_lm_head(self, remapped, geom):
        assert remapped["output.weight"].shape == (geom["vocab"], geom["hidden"])

    def test_final_norm(self, remapped, geom):
        assert remapped["norm.weight"].shape == (geom["hidden"],)


class TestDeltaNetLayerWeights:
    """The first linear-attention (GDN) layer."""

    @pytest.fixture(scope="class")
    def L(self, geom):
        return geom["linear_layers"][0]

    def test_qkv_combined_only(self, remapped, geom, L):
        qkv = remapped[f"layers.{L}.linear_attn.qkv_proj.weight"]
        assert qkv.shape == (geom["lin_k"] + geom["lin_k"] + geom["lin_v"], geom["hidden"])
        # Split q/k/v_proj are no longer emitted (the op uses the combined weight)
        for n in ("q_proj", "k_proj", "v_proj"):
            assert f"layers.{L}.linear_attn.{n}.weight" not in remapped

    def test_conv1d_split(self, remapped, geom, L):
        assert remapped[f"layers.{L}.linear_attn.q_conv.weight"].shape == (geom["lin_k"], 1, geom["conv_k"])
        assert remapped[f"layers.{L}.linear_attn.k_conv.weight"].shape == (geom["lin_k"], 1, geom["conv_k"])
        assert remapped[f"layers.{L}.linear_attn.v_conv.weight"].shape == (geom["lin_v"], 1, geom["conv_k"])

    def test_decay_projections(self, remapped, geom, L):
        assert remapped[f"layers.{L}.linear_attn.in_proj_a.weight"].shape == (geom["lin_v_heads"], geom["hidden"])
        assert remapped[f"layers.{L}.linear_attn.in_proj_b.weight"].shape == (geom["lin_v_heads"], geom["hidden"])

    def test_gate_projection(self, remapped, geom, L):
        assert remapped[f"layers.{L}.linear_attn.in_proj_z.weight"].shape == (geom["lin_v"], geom["hidden"])

    def test_output_proj(self, remapped, geom, L):
        assert remapped[f"layers.{L}.linear_attn.out_proj.weight"].shape == (geom["hidden"], geom["lin_v"])

    def test_a_log_and_dt_bias(self, remapped, geom, L):
        assert remapped[f"layers.{L}.linear_attn.A_log"].shape == (geom["lin_v_heads"],)
        assert remapped[f"layers.{L}.linear_attn.dt_bias"].shape == (geom["lin_v_heads"],)

    def test_norm(self, remapped, geom, L):
        assert remapped[f"layers.{L}.linear_attn.norm.weight"].shape == (geom["lin_v_head_dim"],)

    def test_mlp(self, remapped, geom, L):
        I, h = geom["intermediate"], geom["hidden"]
        assert remapped[f"layers.{L}.mlp.gate_proj.weight"].shape == (I, h)
        assert remapped[f"layers.{L}.mlp.up_proj.weight"].shape == (I, h)
        assert remapped[f"layers.{L}.mlp.down_proj.weight"].shape == (h, I)

    def test_layernorms(self, remapped, geom, L):
        assert remapped[f"layers.{L}.input_layernorm.weight"].shape == (geom["hidden"],)
        assert remapped[f"layers.{L}.post_attention_layernorm.weight"].shape == (geom["hidden"],)


class TestGatedAttentionLayerWeights:
    """The first gated full-attention layer."""

    @pytest.fixture(scope="class")
    def L(self, geom):
        return geom["full_attn_layers"][0]

    def test_q_proj(self, remapped, geom, L):
        assert remapped[f"layers.{L}.self_attn.q_proj.weight"].shape == (geom["q_dim"], geom["hidden"])

    def test_kv_proj(self, remapped, geom, L):
        assert remapped[f"layers.{L}.self_attn.k_proj.weight"].shape == (geom["kv_dim"], geom["hidden"])
        assert remapped[f"layers.{L}.self_attn.v_proj.weight"].shape == (geom["kv_dim"], geom["hidden"])

    def test_o_proj(self, remapped, geom, L):
        assert remapped[f"layers.{L}.self_attn.o_proj.weight"].shape == (geom["hidden"], geom["o_in"])

    def test_qk_norm(self, remapped, geom, L):
        assert remapped[f"layers.{L}.self_attn.q_norm.weight"].shape == (geom["head_dim"],)
        assert remapped[f"layers.{L}.self_attn.k_norm.weight"].shape == (geom["head_dim"],)

    def test_mlp(self, remapped, geom, L):
        assert remapped[f"layers.{L}.mlp.gate_proj.weight"].shape == (geom["intermediate"], geom["hidden"])

    def test_layernorms(self, remapped, geom, L):
        assert remapped[f"layers.{L}.input_layernorm.weight"].shape == (geom["hidden"],)
        assert remapped[f"layers.{L}.post_attention_layernorm.weight"].shape == (geom["hidden"],)


class TestAllLayersPresent:
    def test_all_layers_have_mlp(self, remapped, geom):
        for i in range(geom["n_layers"]):
            assert f"layers.{i}.mlp.gate_proj.weight" in remapped, f"Missing MLP for layer {i}"

    def test_deltanet_layers(self, remapped, geom):
        got = [i for i in range(geom["n_layers"]) if f"layers.{i}.linear_attn.qkv_proj.weight" in remapped]
        assert got == geom["linear_layers"]

    def test_full_attn_layers(self, remapped, geom):
        got = [i for i in range(geom["n_layers"]) if f"layers.{i}.self_attn.q_proj.weight" in remapped]
        assert got == geom["full_attn_layers"]


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
