# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HF exporter tests.

Steps 1-3 run on CPU only (no hardware).
Step 4 (TestRoundTripFidelity) requires @pytest.mark.requires_device and
TINYLLAMA_HF_DIR env var pointing to a local TinyLlama HF model directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file as load_safetensors

import ttnn
import ttml
from ttml.models.llama import Llama, LlamaConfig, load_from_safetensors, save_to_hf_format
from ttml.models.llama.hf_exporter import (
    _build_hf_state_dict,
    _normalize_config,
    _permute_proj_rows,
)
from ttml.models.llama.safetensors_loader import _unpermute_proj_rows


# TinyLlama 1.1B — used for the device round-trip test (Step 4)
TINYLLAMA_CONFIG = LlamaConfig(
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=22,
    num_attention_heads=32,
    num_key_value_heads=4,
    vocab_size=32000,
    max_position_embeddings=2048,
    rope_theta=10000.0,
)

# Tiny config for CPU-only shape tests — all dims tile-aligned (multiples of 32)
_TINY_CFG = LlamaConfig(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    vocab_size=64,
    max_position_embeddings=64,
)


def _make_synthetic_params(config: LlamaConfig) -> dict[str, np.ndarray]:
    """2D numpy params in Python Llama key schema for CPU-only shape/fidelity tests."""
    H = config.hidden_size
    A = config.num_attention_heads
    KV = config.num_key_value_heads
    HD = H // A
    IS = config.intermediate_size
    V = config.vocab_size
    L = config.num_hidden_layers
    rng = np.random.default_rng(0)

    params: dict[str, np.ndarray] = {
        "Llama/tok_emb/weight": rng.standard_normal((V, H)).astype(np.float32),
        "Llama/ln_fc/gamma": rng.standard_normal((H,)).astype(np.float32),
        "Llama/fc/weight": rng.standard_normal((V, H)).astype(np.float32),
    }
    for i in range(L):
        pfx = f"Llama/blocks/{i}"
        params.update(
            {
                f"{pfx}/attention_norm/gamma": rng.standard_normal((H,)).astype(np.float32),
                f"{pfx}/mlp_norm/gamma": rng.standard_normal((H,)).astype(np.float32),
                f"{pfx}/attention/q_linear/weight": rng.standard_normal((A * HD, H)).astype(np.float32),
                f"{pfx}/attention/kv_linear/weight": rng.standard_normal((2 * KV * HD, H)).astype(np.float32),
                f"{pfx}/attention/out_linear/weight": rng.standard_normal((H, H)).astype(np.float32),
                f"{pfx}/mlp/w1/weight": rng.standard_normal((IS, H)).astype(np.float32),
                f"{pfx}/mlp/w3/weight": rng.standard_normal((IS, H)).astype(np.float32),
                f"{pfx}/mlp/w2/weight": rng.standard_normal((H, IS)).astype(np.float32),
            }
        )
    return params


# ---------------------------------------------------------------------------
# Step 1 — Q/K permutation round-trip (CPU only)
# ---------------------------------------------------------------------------


class TestPermuteProjRowsRoundtrip:
    @pytest.mark.parametrize(
        "num_heads,head_dim,hidden",
        [
            (2, 32, 64),
            (4, 16, 64),
            (32, 64, 2048),
        ],
    )
    def test_permute_is_inverse_of_unpermute(self, num_heads, head_dim, hidden):
        R = num_heads * head_dim
        w = np.random.default_rng(42).standard_normal((R, hidden)).astype(np.float32)
        w_roundtrip = _permute_proj_rows(_unpermute_proj_rows(w, num_heads), num_heads)
        assert np.allclose(w, w_roundtrip, atol=1e-6)


# ---------------------------------------------------------------------------
# Step 2 — KV split round-trip (CPU only)
# ---------------------------------------------------------------------------


class TestKvSplitRoundtrip:
    def test_kv_split_is_inverse_of_concat(self):
        num_kv_heads, head_dim, hidden = 4, 64, 2048
        kv_rows = num_kv_heads * head_dim
        rng = np.random.default_rng(42)
        k = rng.standard_normal((kv_rows, hidden)).astype(np.float32)
        v = rng.standard_normal((kv_rows, hidden)).astype(np.float32)
        kv = np.concatenate([k, v], axis=0)
        k_split, v_split = np.split(kv, 2, axis=0)
        assert np.allclose(k, k_split, atol=1e-6)
        assert np.allclose(v, v_split, atol=1e-6)


# ---------------------------------------------------------------------------
# Step 3 — HF state dict shape correctness (CPU only)
# ---------------------------------------------------------------------------


class TestBuildHfStateDictShapes:
    @pytest.fixture(scope="class")
    def hf_dict(self):
        cfg_dict = _normalize_config(_TINY_CFG)
        params = _make_synthetic_params(_TINY_CFG)
        return _build_hf_state_dict(params, cfg_dict), _TINY_CFG

    def test_embed_tokens_shape(self, hf_dict):
        hf, cfg = hf_dict
        assert hf["model.embed_tokens.weight"].shape == (cfg.vocab_size, cfg.hidden_size)

    def test_lm_head_shape(self, hf_dict):
        hf, cfg = hf_dict
        assert hf["lm_head.weight"].shape == (cfg.vocab_size, cfg.hidden_size)

    def test_q_proj_shape(self, hf_dict):
        hf, cfg = hf_dict
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        assert hf["model.layers.0.self_attn.q_proj.weight"].shape == (
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
        )

    def test_k_proj_shape(self, hf_dict):
        hf, cfg = hf_dict
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        kv_rows = cfg.num_key_value_heads * head_dim
        assert hf["model.layers.0.self_attn.k_proj.weight"].shape == (kv_rows, cfg.hidden_size)

    def test_v_proj_shape(self, hf_dict):
        hf, cfg = hf_dict
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        kv_rows = cfg.num_key_value_heads * head_dim
        assert hf["model.layers.0.self_attn.v_proj.weight"].shape == (kv_rows, cfg.hidden_size)

    def test_mlp_shapes(self, hf_dict):
        hf, cfg = hf_dict
        assert hf["model.layers.0.mlp.gate_proj.weight"].shape == (cfg.intermediate_size, cfg.hidden_size)
        assert hf["model.layers.0.mlp.up_proj.weight"].shape == (cfg.intermediate_size, cfg.hidden_size)
        assert hf["model.layers.0.mlp.down_proj.weight"].shape == (cfg.hidden_size, cfg.intermediate_size)

    def test_all_expected_keys_present(self, hf_dict):
        hf, cfg = hf_dict
        expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
        for i in range(cfg.num_hidden_layers):
            expected.update(
                {
                    f"model.layers.{i}.input_layernorm.weight",
                    f"model.layers.{i}.post_attention_layernorm.weight",
                    f"model.layers.{i}.self_attn.q_proj.weight",
                    f"model.layers.{i}.self_attn.k_proj.weight",
                    f"model.layers.{i}.self_attn.v_proj.weight",
                    f"model.layers.{i}.self_attn.o_proj.weight",
                    f"model.layers.{i}.mlp.gate_proj.weight",
                    f"model.layers.{i}.mlp.up_proj.weight",
                    f"model.layers.{i}.mlp.down_proj.weight",
                }
            )
        assert expected.issubset(hf.keys())


# ---------------------------------------------------------------------------
# Step 4 — Round-trip fidelity with TinyLlama (requires device)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def exported_tinyllama(tmp_path_factory):
    """Load TinyLlama into TTML via device, export to HF format, return (out_dir, hf_dir).

    Skips if TINYLLAMA_HF_DIR env var is not set.
    Resets the autograd graph on teardown.
    """
    hf_dir_str = os.environ.get("TINYLLAMA_HF_DIR")
    if not hf_dir_str:
        pytest.skip("TINYLLAMA_HF_DIR not set")
    hf_dir = Path(hf_dir_str)

    model = Llama(TINYLLAMA_CONFIG)
    load_from_safetensors(model, hf_dir, TINYLLAMA_CONFIG)

    merged: dict[str, np.ndarray] = {}
    for name, param in model.parameters().items():
        tensor = param.tensor if hasattr(param, "tensor") else param
        merged[name] = tensor.to_numpy(ttnn.DataType.FLOAT32)

    out_dir = tmp_path_factory.mktemp("hf_export")
    save_to_hf_format(merged, out_dir, TINYLLAMA_CONFIG)

    yield out_dir, hf_dir

    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.requires_device
class TestRoundTripFidelity:
    """Step 4: HF TinyLlama -> TTML -> HF, verify output files and weight fidelity."""

    def test_output_files_exist(self, exported_tinyllama):
        out_dir, _ = exported_tinyllama
        assert (out_dir / "model.safetensors").exists()
        assert (out_dir / "config.json").exists()
        assert (out_dir / "generation_config.json").exists()

    def test_config_json_fields(self, exported_tinyllama):
        out_dir, _ = exported_tinyllama
        with open(out_dir / "config.json") as f:
            cfg = json.load(f)

        assert cfg["architectures"] == ["LlamaForCausalLM"]
        assert cfg["hidden_size"] == TINYLLAMA_CONFIG.hidden_size
        assert cfg["num_attention_heads"] == TINYLLAMA_CONFIG.num_attention_heads
        assert cfg["num_key_value_heads"] == TINYLLAMA_CONFIG.num_key_value_heads
        assert cfg["vocab_size"] == TINYLLAMA_CONFIG.vocab_size
        assert cfg["num_hidden_layers"] == TINYLLAMA_CONFIG.num_hidden_layers

    def test_embed_tokens_shape(self, exported_tinyllama):
        out_dir, _ = exported_tinyllama
        exp = load_safetensors(str(out_dir / "model.safetensors"))
        assert exp["model.embed_tokens.weight"].shape == (
            TINYLLAMA_CONFIG.vocab_size,
            TINYLLAMA_CONFIG.hidden_size,
        )

    def test_kv_proj_shapes(self, exported_tinyllama):
        out_dir, _ = exported_tinyllama
        exp = load_safetensors(str(out_dir / "model.safetensors"))
        head_dim = TINYLLAMA_CONFIG.hidden_size // TINYLLAMA_CONFIG.num_attention_heads
        kv_rows = TINYLLAMA_CONFIG.num_key_value_heads * head_dim
        assert exp["model.layers.0.self_attn.k_proj.weight"].shape == (kv_rows, TINYLLAMA_CONFIG.hidden_size)
        assert exp["model.layers.0.self_attn.v_proj.weight"].shape == (kv_rows, TINYLLAMA_CONFIG.hidden_size)

    def test_weight_fidelity(self, exported_tinyllama):
        out_dir, hf_dir = exported_tinyllama

        orig: dict[str, np.ndarray] = {}
        for st_file in sorted(hf_dir.glob("*.safetensors")):
            for k, v in load_safetensors(str(st_file)).items():
                orig[k] = v.astype(np.float32)

        exp = load_safetensors(str(out_dir / "model.safetensors"))

        failures = []
        for key, exp_arr in exp.items():
            if key not in orig:
                failures.append(f"{key!r}: not found in original HF model")
                continue
            max_diff = float(np.abs(orig[key] - exp_arr).max())
            if max_diff >= 1e-3:
                failures.append(f"{key}: max_diff={max_diff:.6f} (threshold 1e-3)")

        assert not failures, "Weight fidelity failures:\n" + "\n".join(failures)
