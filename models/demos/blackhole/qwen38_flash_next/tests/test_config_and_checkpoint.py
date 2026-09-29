# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Config, Qwen38Placement

CHECKPOINT = Path(os.environ.get("QWEN38_CHECKPOINT", "/nonexistent/Qwen3.8-Flash-Next"))


def test_exact_release_config_and_layer_schedule():
    config = Qwen38Config.from_checkpoint(CHECKPOINT)

    assert config.hidden_size == 2560
    assert config.residual_width == 10240
    assert config.hidden_act == "silu"
    assert config.num_hidden_layers == 48
    assert config.layer_types == ("linear_attention", "linear_attention", "linear_attention", "full_attention") * 12
    assert config.ple_checkpoint_layer == 1
    assert config.gdn_qk_heads == 16
    assert config.gdn_value_heads == 48
    assert config.gdn_output_gate == "sigmoid"
    assert config.qsa_query_heads == 24
    assert config.qsa_kv_heads == 2
    assert config.num_experts == 512
    assert config.top_k == 10
    assert config.norm_topk_prob is True
    assert config.mtp_layers == 1


def test_config_fails_closed_on_shape_only_substitution(tmp_path, expect_error):
    source = json.loads((CHECKPOINT / "config.json").read_text())
    source["text_config"]["linear_num_value_heads"] = 32
    (tmp_path / "config.json").write_text(json.dumps(source))

    with expect_error(ValueError, "linear_num_value_heads"):
        Qwen38Config.from_checkpoint(tmp_path)


def test_exact_tp4_placement_contract():
    config = Qwen38Config.from_checkpoint(CHECKPOINT)
    placement = Qwen38Placement(config, mesh_shape=(1, 4), physical_ids=(0, 1, 2, 3))

    assert placement.hidden_ranges == ((0, 640), (640, 1280), (1280, 1920), (1920, 2560))
    assert placement.vocab_ranges[-1] == (186240, 248320)
    assert placement.expert_ranges == ((0, 128), (128, 256), (256, 384), (384, 512))
    assert placement.gdn_qk_head_ranges[-1] == (12, 16)
    assert placement.gdn_value_head_ranges[-1] == (36, 48)
    assert placement.qsa_query_head_ranges == ((0, 6), (6, 12), (12, 18), (18, 24))
    assert placement.qsa_kv_device_groups == ((0, 1), (2, 3))
    assert placement.index_query_head_ranges == ((0, 1), (1, 2), (2, 3), (3, 4))
    assert placement.index_key_replicas == (0, 1, 2, 3)
    assert placement.ple_result_ranges == placement.hidden_ranges


@pytest.mark.parametrize(
    ("shape", "physical_ids", "message"),
    [((1, 2), (0, 1), "exactly four"), ((2, 2), (0, 1, 2, 3), "1x4"), ((1, 4), (0, 1, 1, 3), "distinct")],
)
def test_placement_rejects_wrong_or_ambiguous_mesh(shape, physical_ids, message, expect_error):
    config = Qwen38Config.from_checkpoint(CHECKPOINT)
    with expect_error(ValueError, message):
        Qwen38Placement(config, mesh_shape=shape, physical_ids=physical_ids)


def test_checkpoint_headers_and_exact_release_totals():
    checkpoint = Qwen38Checkpoint(CHECKPOINT)
    summary = checkpoint.validate_all_headers()

    assert summary.shard_count == 131
    assert summary.tensor_count == 1658
    assert summary.tensor_bytes == 359_999_963_128
    assert summary.file_bytes == 360_000_192_888
    assert summary.int64_elements == 35

    metadata = checkpoint.metadata("model.language_model.layers.0.linear_attn.A_log")
    assert metadata.shape == (48,)
    assert metadata.dtype == "BF16"


def test_checkpoint_lazy_slice_does_not_materialize_full_expert_tensor():
    checkpoint = Qwen38Checkpoint(CHECKPOINT)
    name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    metadata = checkpoint.metadata(name)
    assert metadata.shape == (512, 1280, 2560)

    expert_zero = checkpoint.tensor_slice(name, (slice(0, 1), slice(None), slice(None)))
    assert expert_zero.shape == (1, 1280, 2560)
    assert expert_zero.dtype == torch.bfloat16


def test_ple_table_is_identified_as_host_resident():
    config = Qwen38Config.from_checkpoint(CHECKPOINT)
    placement = Qwen38Placement(config, mesh_shape=(1, 4), physical_ids=(0, 1, 2, 3))
    checkpoint = Qwen38Checkpoint(CHECKPOINT)
    ple_names = checkpoint.names_with_prefix("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_")

    assert len(ple_names) == 128
    assert all(placement.classify_tensor(name) == "host_ple" for name in ple_names)
