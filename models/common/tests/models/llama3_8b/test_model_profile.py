# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pure semantic snapshots for the Llama-3.1-8B architecture/SKU composition."""

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.common.models.llama3_8b.model import (
    LazyWeight,
    Llama31DecoderPrecision,
    TransformerBlock1D,
    TransformerBlock1DConfig,
    _make_llama31_8b_rope_config,
    _resolve_llama31_8b_architecture_profile,
    _use_distributed_prefill_rmsnorm,
    build_llama3_transformer_1d_config,
)
from models.common.modules.rope.rope_1d import RotarySetup1D


def _single_device(device_id, *, count=1):
    return SimpleNamespace(id=lambda: device_id, get_num_devices=lambda: count)


def _cache_weight(device):
    return LazyWeight(
        source=torch.zeros(1),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_llama_single_device_lane_reuses_equivalent_legacy_cache(tmp_path):
    lane = _cache_weight(_single_device(2))
    exact_path = lane._get_cache_fill_path(tmp_path, "weight")
    assert exact_path is not None
    portable_path = Path(str(exact_path).replace("device_2", "device_1"))
    portable_path.write_bytes(b"portable-host-tensor")

    assert lane._get_cache_fill_path(tmp_path, "weight") == portable_path


def test_llama_single_device_lane_prefers_its_exact_legacy_cache(tmp_path):
    lane = _cache_weight(_single_device(2))
    exact_path = lane._get_cache_fill_path(tmp_path, "weight")
    assert exact_path is not None
    portable_path = Path(str(exact_path).replace("device_2", "device_1"))
    portable_path.write_bytes(b"portable-host-tensor")
    exact_path.write_bytes(b"exact-host-tensor")

    assert lane._get_cache_fill_path(tmp_path, "weight") == exact_path


def test_llama_multi_device_cache_does_not_reuse_another_device_identity(tmp_path):
    lane = _cache_weight(_single_device(2, count=4))
    exact_path = lane._get_cache_fill_path(tmp_path, "weight")
    assert exact_path is not None
    portable_path = Path(str(exact_path).replace("device_2", "device_1"))
    portable_path.write_bytes(b"different-mesh-tensor")

    assert lane._get_cache_fill_path(tmp_path, "weight") == exact_path


@pytest.mark.parametrize(
    ("device_name", "model_name", "expected_cutoff"),
    [
        ("N150", "Llama-3.1-8B-Instruct", 512),
        ("N150", "other-model", 1024),
        ("T3K", "Llama-3.1-8B-Instruct", 1024),
    ],
)
def test_wormhole_profile_preserves_existing_semantics(device_name, model_name, expected_cutoff):
    profile = _resolve_llama31_8b_architecture_profile(
        arch=ttnn.device.Arch.WORMHOLE_B0,
        cluster_type=ttnn.cluster.ClusterType.T3K,
        device_name=device_name,
        model_name=model_name,
        dram_grid_width=8,
    )

    assert profile.rms_packer_l1_acc is False
    assert profile.rms_distributed_at_dim_4096 is True
    assert profile.mlp_prefill_len_cutoff == expected_cutoff
    assert profile.mlp_prefill_dram_shard_grid_width == 8
    assert profile.mlp_prefill_ff1_ff3_grid == (8, 8)
    assert profile.mlp_prefill_ff2_grid == (8, 8)
    assert profile.attention_prefill_qkv_grid == (8, 8)
    assert profile.attention_decode_create_qkv_head_grid is None
    assert profile.attention_decode_transformation_core_grid is None
    assert profile.enable_minimal_qkv is False
    assert profile.enable_minimal_ff2 is False
    assert profile.lm_head_max_columns_per_device is None


def test_blackhole_p150x4_profile_semantic_snapshot():
    profile = _resolve_llama31_8b_architecture_profile(
        arch=ttnn.device.Arch.BLACKHOLE,
        cluster_type=ttnn.cluster.ClusterType.P150_X4,
        device_name="P150x4",
        model_name="Llama-3.1-8B-Instruct",
        dram_grid_width=8,
    )

    assert profile.rms_packer_l1_acc is True
    # Multi-device Llama-8B receives 4096 / num_devices hidden slices from
    # the sharded embedding; using local RMSNorm would pair those slices with
    # a replicated 4096-element gamma and fail device validation.
    assert profile.rms_distributed_at_dim_4096 is True
    assert profile.mlp_prefill_len_cutoff == 512
    assert profile.mlp_prefill_dram_shard_grid_width == 8
    assert profile.mlp_prefill_ff1_ff3_grid == (8, 8)
    assert profile.mlp_prefill_ff2_grid == (8, 8)
    assert profile.attention_prefill_qkv_grid == (8, 10)
    assert (profile.attention_decode_create_qkv_head_grid.x, profile.attention_decode_create_qkv_head_grid.y) == (
        8,
        4,
    )
    assert (
        profile.attention_decode_transformation_core_grid.x,
        profile.attention_decode_transformation_core_grid.y,
    ) == (8, 8)
    assert profile.enable_minimal_qkv is True
    assert profile.enable_minimal_ff2 is True
    assert profile.lm_head_max_columns_per_device == 4008


@pytest.mark.parametrize(
    ("arch", "cluster_type", "device_name", "num_devices", "expected"),
    [
        (ttnn.device.Arch.BLACKHOLE, ttnn.cluster.ClusterType.P150_X4, "P150", 1, False),
        (ttnn.device.Arch.BLACKHOLE, ttnn.cluster.ClusterType.P150_X2, "P300", 2, True),
        (ttnn.device.Arch.BLACKHOLE, ttnn.cluster.ClusterType.P150_X4, "P150x4", 4, True),
        (ttnn.device.Arch.WORMHOLE_B0, ttnn.cluster.ClusterType.T3K, "N150", 1, False),
        (ttnn.device.Arch.WORMHOLE_B0, ttnn.cluster.ClusterType.T3K, "N300", 2, True),
    ],
)
def test_effective_prefill_rmsnorm_policy(arch, cluster_type, device_name, num_devices, expected):
    profile = _resolve_llama31_8b_architecture_profile(
        arch=arch,
        cluster_type=cluster_type,
        device_name=device_name,
        model_name="Llama-3.1-8B-Instruct",
        dram_grid_width=8,
    )

    assert (
        _use_distributed_prefill_rmsnorm(
            num_devices=num_devices,
            dim=4096,
            architecture_profile=profile,
        )
        is expected
    )


def test_blackhole_batch32_rope_uses_attention_decode_grid():
    """Keep fused Q/K rotary's 64 shards on the attention program's 8x8 cores."""
    profile = _resolve_llama31_8b_architecture_profile(
        arch=ttnn.device.Arch.BLACKHOLE,
        cluster_type=ttnn.cluster.ClusterType.P150_X4,
        device_name="P150",
        model_name="Llama-3.1-8B-Instruct",
        dram_grid_width=8,
    )
    mesh_device = MagicMock()
    physical_grid = ttnn.CoreCoord(12, 10)
    mesh_device.compute_with_storage_grid_size.return_value = physical_grid
    decode_grid = profile.attention_decode_transformation_core_grid or physical_grid

    rope_config = _make_llama31_8b_rope_config(
        rope_cos=torch.zeros(1, 1, 2048, 128),
        rope_sin=torch.zeros(1, 1, 2048, 128),
        max_batch_size=32,
        head_dim=128,
        mesh_device=mesh_device,
        decode_transformation_core_grid=decode_grid,
    )

    assert rope_config.use_qk_fused is True
    assert rope_config.max_batch_size * 2 == 64
    assert (rope_config.core_grid.x, rope_config.core_grid.y) == (8, 8)
    assert rope_config.core_grid != physical_grid

    resolved = RotarySetup1D.from_config(rope_config).config
    assert resolved.batch_size_per_device_group == 64
    assert (resolved.batch_grid.bounding_box().grid_size().x, resolved.batch_grid.bounding_box().grid_size().y) == (
        8,
        8,
    )
    # The failing 12x10-derived placement used cores x=8..11 but stopped at
    # y=5. Fused Q/K uses y=0..7 at x=0..7, and the runtime failure was first
    # observed at (0, 6).
    assert resolved.batch_grid.contains(ttnn.CoreCoord(0, 6))
    assert not resolved.batch_grid.contains(ttnn.CoreCoord(8, 0))
    assert resolved.decode_trans_mat_mem_config.shard_spec.grid == resolved.batch_grid


@pytest.mark.parametrize(
    ("device_name", "expected_max_columns"),
    [("P100", 16032), ("P150", 16032), ("P300", 16032), ("P150x4", 4008), ("P150x8", 1002)],
)
def test_blackhole_lm_head_split_policy_matches_tttv1(device_name, expected_max_columns):
    profile = _resolve_llama31_8b_architecture_profile(
        arch=ttnn.device.Arch.BLACKHOLE,
        cluster_type=ttnn.cluster.ClusterType.P150_X8,
        device_name=device_name,
        model_name="Llama-3.1-8B-Instruct",
        dram_grid_width=8,
    )

    assert profile.lm_head_max_columns_per_device == expected_max_columns


def test_architecture_profile_selection_fails_closed(expect_error):
    unsupported_arch = object()
    with expect_error(ValueError, "Unsupported Llama 3.1 8B architecture"):
        _resolve_llama31_8b_architecture_profile(
            arch=unsupported_arch,
            cluster_type=ttnn.cluster.ClusterType.T3K,
            device_name="unknown",
            model_name="Llama-3.1-8B-Instruct",
            dram_grid_width=8,
        )


def test_performance_precision_preserves_layer_31_exception():
    precision = Llama31DecoderPrecision.performance(32, "Llama-3.1-8B-Instruct")

    assert precision._tensor_precision[0]["ff1_ff3"] == "bfp4"
    assert precision._op_fidelity[0]["li_ff1_ff3"] == "lofi"
    assert precision._tensor_precision[31]["ff1_ff3"] == "bfp8"
    assert precision._op_fidelity[31]["li_ff1_ff3"] == "hifi2fp16"
    assert precision._op_fidelity[31]["li_ff2"] == "hifi2fp16"


def test_accuracy_precision_keeps_all_six_attention_and_four_mlp_slot_recipes():
    precision = Llama31DecoderPrecision.accuracy(1, "Llama-3.1-8B-Instruct")

    assert precision._op_fidelity[0] == {
        "li_ff1_ff3": "hifi2fp16",
        "li_ff2": "hifi2fp16",
        "li_qkv_decode": "hifi2",
        "sdpa_decode": "hifi2",
        "li_o_decode": "hifi2",
        "li_qkv_prefill": "hifi2",
        "sdpa_prefill": "hifi4",
        "li_o_prefill": "hifi2",
        "accuracy": "hifi4fp32",
    }


def test_builder_reads_mesh_architecture_once():
    source = inspect.getsource(build_llama3_transformer_1d_config)

    assert source.count("mesh_device.arch()") == 1
    assert source.count("ttnn.cluster.get_cluster_type()") == 1


def test_sampling_uses_the_same_tile_padded_rows_as_decode_logits():
    source = inspect.getsource(build_llama3_transformer_1d_config)

    assert "max_batch_size=tile_padded_batch_rows" in source


def test_transformer_block_consumes_only_common_configs(monkeypatch):
    common = {
        "attention_norm": object(),
        "attention": object(),
        "ff_norm": object(),
        "mlp": object(),
    }
    config = TransformerBlock1DConfig(
        attention_norm_config=common["attention_norm"],
        attention_config=common["attention"],
        ff_norm_config=common["ff_norm"],
        mlp_config=common["mlp"],
    )
    rms_from_config = MagicMock(side_effect=[object(), object()])
    attention_from_config = MagicMock(return_value=object())
    mlp_from_config = MagicMock(return_value=object())
    monkeypatch.setattr("models.common.models.llama3_8b.model.RMSNorm1D.from_config", rms_from_config)
    monkeypatch.setattr("models.common.models.llama3_8b.model.Attention1D.from_config", attention_from_config)
    monkeypatch.setattr("models.common.models.llama3_8b.model.MLP1D.from_config", mlp_from_config)

    TransformerBlock1D.from_config(config)

    assert config.attention_config is common["attention"]
    assert config.mlp_config is common["mlp"]
    assert [call.args[0] for call in rms_from_config.call_args_list] == [
        common["attention_norm"],
        common["ff_norm"],
    ]
    attention_from_config.assert_called_once_with(common["attention"])
    mlp_from_config.assert_called_once_with(common["mlp"])
