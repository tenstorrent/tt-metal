# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free tests for input packing and raw-HF checkpoint contracts."""

import copy
import json

import pytest
import torch
from safetensors.torch import save_file

from models.demos.llama_3p1_8b_d_p.tt.input import pack_token_ids
from models.demos.llama_3p1_8b_d_p.tt.weights import CheckpointWeights, validate_checkpoint_config


def _config():
    return {
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
        "max_position_embeddings": 131072,
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    }


# Absolute-position tags expose wrong SP rotation, wrong encounter order, and padding overwrites.
# The expected mapping is enumerated independently for every row rather than using the packer.
@pytest.mark.parametrize(
    "start,end",
    [
        (0, 1),
        (0, 31),
        (0, 32),
        (0, 33),
        (0, 1024),
        (32, 65),
        (224, 257),
        (480, 511),
        (480, 512),
        (480, 513),
        (992, 1033),
        (1024, 1025),
        (1024, 1537),
        (2016, 2048),
    ],
)
def test_pack_token_ids_absolute_rows_and_true_length(start, end):
    ids = torch.arange(start + 100, end + 100, dtype=torch.int64)
    before = ids.clone()
    actual = pack_token_ids(ids, actual_start=start, actual_end=end, pad_id=17).reshape(4, 256)
    for sp in range(4):
        expected = []
        for absolute in range(start, start + 1024):
            if (absolute // 256) % 4 == sp:
                expected.append(absolute + 100 if absolute < end else 17)
        assert torch.equal(actual[sp], torch.tensor(expected))
    assert torch.equal(ids, before)


# Invalid true lengths and IDs must fail before transfer or cache mutation. Booleans and floating
# values are rejected explicitly instead of being converted into plausible token IDs.
@pytest.mark.parametrize(
    "ids,start,end,pad,error",
    [
        ([1], 0, 0, 0, ValueError),
        ([1], 1, 2, 0, ValueError),
        ([1], 0, 2, 0, ValueError),
        ([1], 2048, 2049, 0, ValueError),
        ([1], False, 1, 0, TypeError),
        ([128256], 0, 1, 0, ValueError),
        ([-1], 0, 1, 0, ValueError),
        ([True], 0, 1, 0, ValueError),
        ([1.0], 0, 1, 0, ValueError),
        (torch.tensor([1.0]), 0, 1, 0, ValueError),
        ([1], 0, 1, 128256, ValueError),
    ],
)
def test_pack_token_ids_rejects_invalid_input(ids, start, end, pad, error):
    with pytest.raises(error):  # allow-pytest.raises: device-free test also runs outside repository conftest
        pack_token_ids(ids, actual_start=start, actual_end=end, pad_id=pad)


# A checkpoint with GPT-OSS-style or different Llama mathematics cannot enter this wrapper merely
# because some tensor dimensions happen to match. The exact Llama3 frequency policy is checked too.
@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 8192),
        ("rms_norm_eps", 1e-6),
        ("hidden_act", "gelu"),
        ("tie_word_embeddings", True),
        ("attention_bias", True),
        ("mlp_bias", True),
        ("sliding_window", 128),
        ("rope_scaling", {"rope_type": "yarn", "factor": 8.0}),
    ],
)
def test_checkpoint_config_rejects_architecture_substitutions(field, value):
    correct = _config()
    validate_checkpoint_config(correct)
    changed = copy.deepcopy(correct)
    changed[field] = value
    with pytest.raises(ValueError):  # allow-pytest.raises: device-free test also runs outside repository conftest
        validate_checkpoint_config(changed)


# Asymmetric Q/K rows must survive disk loading exactly. This small shard checks the generic read
# boundary directly; the real-checkpoint gate separately verifies all production layer dimensions.
def test_checkpoint_read_preserves_raw_qk_rows(tmp_path):
    q = torch.arange(32).reshape(8, 4).bfloat16()
    k = -q - 10
    names = {"model.layers.0.self_attn.q_proj.weight": q, "model.layers.0.self_attn.k_proj.weight": k}
    save_file(names, tmp_path / "weights.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(_config()))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "weights.safetensors" for name in names}})
    )
    loader = CheckpointWeights(tmp_path)
    loaded = loader._read({name: (8, 4) for name in names})
    for name, expected in names.items():
        assert torch.equal(loaded[name], expected)
    with pytest.raises(ValueError):  # allow-pytest.raises: device-free test also runs outside repository conftest
        loader._read({next(iter(names)): (4, 8)})
