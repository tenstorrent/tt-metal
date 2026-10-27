# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.checkpoint import (
    HF_MODEL_ID,
    PINNED_CHECKPOINT_REVISION,
    Qwen38Checkpoint,
    checkpoint_directory,
)
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Config, Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE, TP_SIZE

CHECKPOINT = checkpoint_directory(missing=Path("/nonexistent/Qwen3.8-Flash-Next"))


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


def _tp_ranges(total: int) -> tuple[tuple[int, int], ...]:
    width = total // TP_SIZE
    return tuple((index * width, (index + 1) * width) for index in range(TP_SIZE))


def test_exact_tp_placement_contract():
    config = Qwen38Config.from_checkpoint(CHECKPOINT)
    placement = Qwen38Placement(config, mesh_shape=MESH_SHAPE, physical_ids=tuple(range(TP_SIZE)))
    half = TP_SIZE // 2

    assert placement.hidden_ranges == _tp_ranges(2560)
    assert placement.vocab_ranges[-1] == (248320 - 248320 // TP_SIZE, 248320)
    assert placement.expert_ranges == _tp_ranges(512)
    assert placement.gdn_qk_head_ranges[-1] == (16 - 16 // TP_SIZE, 16)
    assert placement.gdn_value_head_ranges[-1] == (48 - 48 // TP_SIZE, 48)
    assert placement.qsa_query_head_ranges == _tp_ranges(24)
    assert placement.qsa_kv_device_groups == (tuple(range(half)), tuple(range(half, TP_SIZE)))
    stride = TP_SIZE // 4
    assert placement.index_query_head_ranges == tuple(
        (device // stride, device // stride + (device % stride == 0)) for device in range(TP_SIZE)
    )
    assert placement.index_key_replicas == tuple(range(TP_SIZE))
    assert placement.ple_result_ranges == placement.hidden_ranges
    if TP_SIZE == 4:
        assert placement.hidden_ranges == ((0, 640), (640, 1280), (1280, 1920), (1920, 2560))
        assert placement.vocab_ranges[-1] == (186240, 248320)
        assert placement.expert_ranges == ((0, 128), (128, 256), (256, 384), (384, 512))
        assert placement.qsa_query_head_ranges == ((0, 6), (6, 12), (12, 18), (18, 24))
        assert placement.qsa_kv_device_groups == ((0, 1), (2, 3))
        assert placement.index_query_head_ranges == ((0, 1), (1, 2), (2, 3), (3, 4))


@pytest.mark.parametrize(
    ("shape", "physical_ids", "message"),
    [
        ((1, TP_SIZE // 2), tuple(range(TP_SIZE // 2)), f"exactly {TP_SIZE}"),
        ((2, TP_SIZE // 2), tuple(range(TP_SIZE)), f"1x{TP_SIZE}"),
        (MESH_SHAPE, (0, 1, 1) + tuple(range(3, TP_SIZE)), "distinct"),
    ],
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


def test_checkpoint_directory_uses_an_existing_directory_as_is(tmp_path, monkeypatch):
    import huggingface_hub

    def fail(*args, **kwargs):
        raise AssertionError("snapshot_download must not be called for a directory")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)

    assert checkpoint_directory({"HF_MODEL": str(tmp_path)}) == tmp_path


def test_checkpoint_directory_resolves_a_repo_id_in_the_local_hub_cache(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)

    assert checkpoint_directory({"HF_MODEL": "org/model"}, revision="abc") == tmp_path / "snapshot"
    assert calls == [(("org/model",), {"revision": "abc", "local_files_only": True})]


def test_checkpoint_directory_defaults_to_the_release_repo_id(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)

    assert checkpoint_directory({}) == tmp_path
    assert calls == [((HF_MODEL_ID,), {"revision": PINNED_CHECKPOINT_REVISION, "local_files_only": True})]


def test_checkpoint_directory_passes_no_revision_through(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)

    assert checkpoint_directory({}, revision=None) == tmp_path
    assert calls == [((HF_MODEL_ID,), {"revision": None, "local_files_only": True})]


def test_checkpoint_directory_raises_when_nothing_resolves(monkeypatch, expect_error):
    import huggingface_hub

    def fail(*args, **kwargs):
        raise RuntimeError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)

    with expect_error(FileNotFoundError, "HF_MODEL"):
        checkpoint_directory({"HF_MODEL": "org/model"})
    assert checkpoint_directory({"HF_MODEL": "org/model"}, missing=Path("/x")) == Path("/x")
