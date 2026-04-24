# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from safetensors import safe_open

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.converter import convert_hf_checkpoint
from models.demos.deepseek_v4_flash.fp4 import (
    FP4_E2M1_TABLE,
    dequantize_fp4_packed,
    pack_fp4_indices,
    unpack_fp4_indices,
)
from models.demos.deepseek_v4_flash.key_mapping import expert_packed_key, normalize_hf_key
from models.demos.deepseek_v4_flash.manifest import TT_MANIFEST_SCHEMA_VERSION, load_tt_manifest, validate_tt_manifest
from models.demos.deepseek_v4_flash.synthetic import (
    generate_tiny_hf_checkpoint,
    tiny_config_dict,
    tiny_inference_config_dict,
)


def test_key_map_accepts_v4_and_hf_aliases():
    assert normalize_hf_key("embed.weight").canonical == "embed.weight"
    assert normalize_hf_key("model.embed_tokens.weight").canonical == "embed.weight"
    assert normalize_hf_key("lm_head.weight").canonical == "head.weight"
    assert normalize_hf_key("model.layers.2.self_attn.q_a_proj.weight").canonical == "layers.2.attn.wq_a.weight"
    assert (
        normalize_hf_key("model.layers.2.self_attn.q_b_proj.weight_scale_inv").canonical == "layers.2.attn.wq_b.scale"
    )
    assert (
        normalize_hf_key("model.layers.2.self_attn.compressor.wgate.weight").canonical
        == "layers.2.attn.compressor.wgate.weight"
    )
    assert (
        normalize_hf_key("model.layers.2.self_attn.indexer.compressor.wkv.weight").canonical
        == "layers.2.attn.indexer.compressor.wkv.weight"
    )
    assert normalize_hf_key("model.layers.3.mlp.gate.weight").canonical == "layers.3.ffn.gate.weight"
    assert normalize_hf_key("model.layers.3.mlp.gate.e_score_correction_bias").canonical == "layers.3.ffn.gate.bias"

    mapped = normalize_hf_key("model.layers.3.mlp.experts.2.gate_proj.weight")
    assert mapped.canonical == "layers.3.ffn.experts.2.w1.weight"
    assert mapped.category == "expert"
    assert mapped.layer == 3
    assert mapped.expert == 2
    assert mapped.projection == "w1"
    assert expert_packed_key(mapped.canonical) == "layers.3.ffn.experts.2.w1.weight_packed"


def test_manifest_version_validation():
    manifest = {
        "schema_version": TT_MANIFEST_SCHEMA_VERSION,
        "model_name": "deepseek_v4_flash",
        "config": {
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "compress_rope_theta": 160000,
            "compress_ratios": [0],
        },
        "artifacts": {
            "non_expert_safetensors": [],
            "expert_safetensors": [],
            "metadata_safetensors": "metadata.safetensors",
        },
        "expert_format": {
            "abi": "deepseek_v4_flash.fp4_e2m1fn_x2.block32.v1",
            "block_size": 32,
        },
    }
    validate_tt_manifest(manifest)
    manifest["schema_version"] = 999
    with pytest.raises(ValueError, match="Unsupported DeepSeek V4 Flash manifest"):
        validate_tt_manifest(manifest)


def test_fp4_pack_unpack_and_dequantize():
    indices = torch.tensor([[0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15]], dtype=torch.int32)
    packed = pack_fp4_indices(indices)
    assert packed.dtype == torch.uint8
    torch.testing.assert_close(unpack_fp4_indices(packed), indices.to(torch.uint8))

    scale = torch.tensor([[2.0, 4.0], [0.5, 1.0]], dtype=torch.float32)
    dequant = dequantize_fp4_packed(packed, scale, block_size=4, dtype=torch.float32)
    expected = FP4_E2M1_TABLE[indices.long()] * scale.repeat_interleave(4, dim=-1)
    torch.testing.assert_close(dequant, expected)


def test_config_rejects_compress_rope_theta_mismatch():
    config = tiny_config_dict(num_hidden_layers=1)
    inference_config = tiny_inference_config_dict(config)
    inference_config["compress_rope_theta"] = 40000
    with pytest.raises(ValueError, match="compress_rope_theta"):
        DeepSeekV4FlashConfig.from_hf_configs(config, inference_config)


def test_generate_and_convert_tiny_one_layer_checkpoint(tmp_path):
    source = generate_tiny_hf_checkpoint(tmp_path / "source", num_hidden_layers=1)
    output = convert_hf_checkpoint(source, tmp_path / "tt_preprocessed")

    manifest = load_tt_manifest(output)
    assert manifest["config"]["num_hidden_layers"] == 1
    assert manifest["config"]["compress_rope_theta"] == 160000
    assert manifest["expert_format"]["abi"] == "deepseek_v4_flash.fp4_e2m1fn_x2.block32.v1"
    assert manifest["artifacts"]["copied_files"] == [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "inference_config.json",
    ]

    non_expert_artifact = output / manifest["artifacts"]["non_expert_safetensors"][0]
    expert_artifact = output / manifest["artifacts"]["expert_safetensors"][0]
    with safe_open(non_expert_artifact, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        assert "embed.weight" in keys
        assert "layers.0.ffn.gate.weight" in keys
        assert "layers.0.ffn.shared_experts.w1.weight" in keys
        assert "layers.0.ffn.experts.0.w1.weight" not in keys
    with safe_open(expert_artifact, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        assert "layers.0.ffn.experts.0.w1.weight_packed" in keys
        assert "layers.0.ffn.experts.0.w1.scale" in keys
        assert handle.get_tensor("layers.0.ffn.experts.0.w1.weight_packed").dtype == torch.uint8
    with safe_open(output / manifest["artifacts"]["metadata_safetensors"], framework="pt", device="cpu") as handle:
        torch.testing.assert_close(handle.get_tensor("compress_ratios"), torch.tensor([0], dtype=torch.int32))


def test_generate_and_convert_tiny_three_layer_fixture(tmp_path):
    source = generate_tiny_hf_checkpoint(tmp_path / "source", num_hidden_layers=3)
    output = convert_hf_checkpoint(source, tmp_path / "tt_preprocessed")
    manifest = load_tt_manifest(output)
    assert manifest["config"]["num_hidden_layers"] == 3
    assert manifest["config"]["num_hash_layers"] == 3
    assert manifest["config"]["compress_ratios"] == [0, 0, 4]
    assert manifest["counts"]["expert_tensors"] == 3 * 4 * 3 * 2

    non_expert_artifact = output / manifest["artifacts"]["non_expert_safetensors"][0]
    with safe_open(non_expert_artifact, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        assert "layers.0.ffn.gate.weight" in keys
        assert "layers.0.ffn.gate.tid2eid" in keys
        assert "layers.2.ffn.gate.weight" in keys
        assert "layers.2.ffn.gate.tid2eid" in keys
        assert "layers.2.attn.compressor.ape" in keys
        assert "layers.2.attn.compressor.norm.weight" in keys
        assert "layers.2.attn.compressor.wgate.weight" in keys
        assert "layers.2.attn.compressor.wkv.weight" in keys
        assert "layers.2.attn.indexer.wq_b.weight" in keys

    with (source / "model.safetensors.index.json").open("r", encoding="utf-8") as handle:
        source_index = json.load(handle)
    assert "layers.2.attn.compressor.ape" in source_index["weight_map"]


def test_generate_and_convert_tiny_compressed_stack_fixture(tmp_path):
    source = generate_tiny_hf_checkpoint(tmp_path / "source", num_hidden_layers=2, compress_ratios=(4, 4))
    output = convert_hf_checkpoint(source, tmp_path / "tt_preprocessed")
    manifest = load_tt_manifest(output)

    assert manifest["config"]["num_hidden_layers"] == 2
    assert manifest["config"]["compress_ratios"] == [4, 4]

    non_expert_artifact = output / manifest["artifacts"]["non_expert_safetensors"][0]
    with safe_open(non_expert_artifact, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        assert "layers.0.attn.compressor.ape" in keys
        assert "layers.0.attn.indexer.wq_b.weight" in keys
        assert "layers.1.attn.compressor.ape" in keys
        assert "layers.1.attn.indexer.wq_b.weight" in keys

    with safe_open(output / manifest["artifacts"]["metadata_safetensors"], framework="pt", device="cpu") as handle:
        torch.testing.assert_close(handle.get_tensor("compress_ratios"), torch.tensor([4, 4], dtype=torch.int32))

    with pytest.raises(ValueError, match="compress_ratios length"):
        tiny_config_dict(num_hidden_layers=2, compress_ratios=(4,))
