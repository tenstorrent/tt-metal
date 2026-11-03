# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only check that GptOssLazyStateDict matches the eager from_pretrained load (#60193).

Builds a tiny MXFP4 GPT-OSS checkpoint with random weights, then loads it both ways through
ModelArgs.load_state_dict and requires the same keys and bit-identical tensors.
"""

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import GptOssConfig, GptOssForCausalLM

from models.demos.gpt_oss.tt.model_config import ModelArgs
from models.demos.gpt_oss.utils.lazy_state_dict import GptOssLazyStateDict
from models.demos.gpt_oss.utils.substate import substate

HIDDEN = 128
INTERMEDIATE = 64
NUM_EXPERTS = 4
MXFP4_BLOCK = 32  # values per scale; a block packs two FP4 values per byte


def _write_mxfp4_checkpoint(path, num_shards):
    config = GptOssConfig(
        num_hidden_layers=2,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_local_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        vocab_size=256,
        layer_types=["sliding_attention", "full_attention"],
    )
    torch.manual_seed(0)
    model = GptOssForCausalLM(config).to(torch.bfloat16)
    tensors = {}
    for key, value in model.state_dict().items():
        if key.endswith("mlp.experts.gate_up_proj") or key.endswith("mlp.experts.down_proj"):
            # HF layout is [E, in, out]; the checkpoint stores blocks [E, out, in / 32, 16] + scales [E, out, in / 32]
            experts, in_features, out_features = value.shape
            groups = in_features // MXFP4_BLOCK
            tensors[f"{key}_blocks"] = torch.randint(
                0, 256, (experts, out_features, groups, MXFP4_BLOCK // 2), dtype=torch.uint8
            )
            tensors[f"{key}_scales"] = torch.randint(122, 131, (experts, out_features, groups), dtype=torch.uint8)
        else:
            tensors[key] = value.contiguous()

    # Spread keys round-robin over the shards, keeping each _blocks/_scales pair in one shard as HF does
    bases = sorted({k.removesuffix("_blocks").removesuffix("_scales") for k in tensors})
    shard_of = {base: i % num_shards for i, base in enumerate(bases)}
    weight_map = {}
    for shard in range(num_shards):
        name = f"model-{shard:05d}-of-{num_shards:05d}.safetensors" if num_shards > 1 else "model.safetensors"
        shard_keys = [k for k in tensors if shard_of[k.removesuffix("_blocks").removesuffix("_scales")] == shard]
        save_file({k: tensors[k] for k in shard_keys}, str(path / name))
        weight_map.update({k: name for k in shard_keys})
    if num_shards > 1:
        (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))

    config_dict = config.to_dict()
    config_dict["quantization_config"] = {
        "quant_method": "mxfp4",
        "modules_to_not_convert": [
            "model.layers.*.self_attn",
            "model.layers.*.mlp.router",
            "model.embed_tokens",
            "lm_head",
        ],
    }
    (path / "config.json").write_text(json.dumps(config_dict))
    return config


@pytest.mark.parametrize("num_shards", [1, 3])
def test_lazy_state_dict_matches_from_pretrained(tmp_path, monkeypatch, num_shards):
    config = _write_mxfp4_checkpoint(tmp_path, num_shards)

    monkeypatch.setenv("GPT_OSS_EAGER_HF_LOAD", "1")
    eager = ModelArgs.load_state_dict(str(tmp_path), convert_to_meta_format=True)
    monkeypatch.delenv("GPT_OSS_EAGER_HF_LOAD")
    lazy = ModelArgs.load_state_dict(str(tmp_path), convert_to_meta_format=True)

    assert isinstance(lazy, GptOssLazyStateDict)
    assert set(lazy) == set(eager)
    for key in eager:
        assert lazy[key].dtype == eager[key].dtype, key
        assert lazy[key].shape == eager[key].shape, key
        assert torch.equal(lazy[key], eager[key]), key

    # Model construction reads the lazy dict only through substate(); it must return plain dicts
    # with the same contents as on the eager dict.
    for layer_idx in range(config.num_hidden_layers):
        lazy_layer = substate(lazy, f"model.layers.{layer_idx}")
        eager_layer = substate(eager, f"model.layers.{layer_idx}")
        assert type(lazy_layer) is dict
        assert lazy_layer.keys() == eager_layer.keys()
        assert all(torch.equal(lazy_layer[k], eager_layer[k]) for k in eager_layer)
