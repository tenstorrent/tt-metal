# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks of load_pplx_state_dict against the real checkpoint; needs ~60 GB of host RAM."""

import json

import pytest
import torch
from safetensors import safe_open

from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict
from models.demos.pplx_decider_v1_27b.tt.weight_mapping import load_pplx_state_dict, resolve_checkpoint

VOCAB_SIZE = 248320
DIM = 5120
NUM_LAYERS = 64
FULL_ATTENTION_LAYERS = {i for i in range(NUM_LAYERS) if i % 4 == 3}


@pytest.fixture(scope="module")
def ckpt_dir():
    try:
        path = resolve_checkpoint()
    except Exception as exc:
        pytest.skip(f"pplx-decider checkpoint not available: {exc}")
    if not (path / "readout.safetensors").is_file():
        pytest.skip(f"{path} is not a pplx-decider checkpoint")
    return path


@pytest.fixture(scope="module")
def token_ids(ckpt_dir):
    with open(ckpt_dir / "decision_config.json") as f:
        return tuple(json.load(f)["token_ids"])


@pytest.fixture(scope="module")
def state_dict(ckpt_dir, token_ids):
    return load_pplx_state_dict(ckpt_dir, token_ids, VOCAB_SIZE, DIM)


def _raw(ckpt_dir, key):
    with open(ckpt_dir / "model.safetensors.index.json") as f:
        filename = json.load(f)["weight_map"][key]
    with safe_open(str(ckpt_dir / filename), framework="pt") as sf:
        return sf.get_tensor(key)


def test_key_set_matches_qwen36_remap(ckpt_dir, state_dict):
    with open(ckpt_dir / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    # Placeholders only: the remap needs 3-D tensors for conv1d slicing and ignores values otherwise.
    qwen_layout = {"model." + key: torch.zeros(1, 1, 1) for key in weight_map if key.startswith("language_model.")}
    qwen_layout["lm_head.weight"] = torch.zeros(1, 1, 1)
    expected = set(remap_qwen36_state_dict(qwen_layout))

    assert set(state_dict) == expected
    assert len(state_dict) == 947
    layer_keys = [k for k in state_dict if k.startswith("layers.")]
    assert len(layer_keys) == 944
    assert {"tok_embeddings.weight", "norm.weight", "output.weight"} <= set(state_dict)

    gdn_layers = {i for i in range(NUM_LAYERS) if i not in FULL_ATTENTION_LAYERS}
    assert len(gdn_layers) == 48 and len(FULL_ATTENTION_LAYERS) == 16
    for i in gdn_layers:
        for name in ("qkv_proj", "q_conv", "k_conv", "v_conv"):
            assert f"layers.{i}.linear_attn.{name}.weight" in state_dict
    for i in FULL_ATTENTION_LAYERS:
        assert f"layers.{i}.self_attn.q_proj.weight" in state_dict


def test_sampled_tensors_bit_equal(ckpt_dir, state_dict):
    prefix = "language_model."
    pairs = [
        ("tok_embeddings.weight", prefix + "embed_tokens.weight"),
        ("layers.0.linear_attn.qkv_proj.weight", prefix + "layers.0.linear_attn.in_proj_qkv.weight"),
        ("layers.3.self_attn.q_proj.weight", prefix + "layers.3.self_attn.q_proj.weight"),
        ("layers.62.mlp.down_proj.weight", prefix + "layers.62.mlp.down_proj.weight"),
        ("norm.weight", prefix + "norm.weight"),
    ]
    for new_key, raw_key in pairs:
        assert torch.equal(state_dict[new_key], _raw(ckpt_dir, raw_key)), new_key

    conv = torch.cat(
        [state_dict[f"layers.1.linear_attn.{n}_conv.weight"] for n in ("q", "k", "v")],
        dim=0,
    )
    assert torch.equal(conv, _raw(ckpt_dir, prefix + "layers.1.linear_attn.conv1d.weight"))


def test_output_weight_is_readout_at_token_ids(ckpt_dir, state_dict, token_ids):
    with safe_open(str(ckpt_dir / "readout.safetensors"), framework="pt") as sf:
        readout = sf.get_tensor("weight")
    output = state_dict["output.weight"]
    assert output.shape == (VOCAB_SIZE, DIM)

    index = torch.tensor(token_ids, dtype=torch.long)
    assert torch.equal(output[index], readout)
    other = torch.ones(VOCAB_SIZE, dtype=torch.bool)
    other[index] = False
    assert not output[other].any()


def test_dtypes_and_excluded_keys(state_dict):
    assert all(t.dtype == torch.bfloat16 for t in state_dict.values())
    assert not any("visual" in k or "mtp" in k for k in state_dict)
