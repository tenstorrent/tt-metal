# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.converter import convert_hf_checkpoint
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint
from models.demos.deepseek_v4_flash.weight_inventory import (
    build_weight_inventory_report,
    estimate_decode_cache_nbytes,
    estimate_max_seq_len_supported,
    plan_weight_placements,
    read_weight_tensor_records,
)


def test_weight_inventory_matches_converted_manifest_counts(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=2, compress_ratios=(4, 4))
    preprocessed = convert_hf_checkpoint(snapshot, tmp_path / "tt_preprocessed")
    manifest = load_tt_manifest(preprocessed)

    report = build_weight_inventory_report(preprocessed, mesh_shape=(2, 4), large_tensor_threshold=128)
    payload = report.to_mapping()

    assert payload["mesh_shape"] == [2, 4]
    assert payload["tensor_count"] == (
        manifest["counts"]["non_expert_tensors"] + manifest["counts"]["expert_tensors"] + 6
    )
    assert payload["counts_by_category"]["non_expert"] == manifest["counts"]["non_expert_tensors"]
    assert payload["counts_by_category"]["expert"] == manifest["counts"]["expert_tensors"]
    assert payload["counts_by_category"]["metadata"] == 6
    assert payload["counts_by_strategy"]["expert_home_device"] == manifest["counts"]["expert_tensors"]
    assert payload["max_device_weight_nbytes"] > 0


def test_weight_inventory_places_experts_on_home_devices(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=1, num_routed_experts=8)
    preprocessed = convert_hf_checkpoint(snapshot, tmp_path / "tt_preprocessed")
    records = read_weight_tensor_records(preprocessed)
    placements = {placement.key: placement for placement in plan_weight_placements(records, mesh_shape=(2, 4))}

    assert placements["layers.0.ffn.experts.0.w1.weight_packed"].devices == ((0, 0),)
    assert placements["layers.0.ffn.experts.7.w3.scale"].devices == ((1, 3),)
    assert placements["embed.weight"].strategy in ("tp_shard_replicate_ep", "replicate_all")


def test_decode_cache_limit_estimate_uses_configured_sequence_length(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=3, compress_ratios=(0, 0, 4))
    preprocessed = convert_hf_checkpoint(snapshot, tmp_path / "tt_preprocessed")
    config = DeepSeekV4FlashConfig.from_model_path(snapshot)
    inventory = build_weight_inventory_report(preprocessed, mesh_shape=(2, 4), large_tensor_threshold=128)

    full_cache_bytes = estimate_decode_cache_nbytes(config, seq_len=1024, dtype_bytes=2)
    assert full_cache_bytes > 0
    assert (
        estimate_max_seq_len_supported(
            config,
            inventory,
            device_dram_bytes=inventory.max_device_weight_nbytes + full_cache_bytes + (1 << 30),
            safety_margin_bytes=0,
        )
        == 1024
    )
    assert (
        estimate_max_seq_len_supported(
            config,
            inventory,
            device_dram_bytes=inventory.max_device_weight_nbytes,
            safety_margin_bytes=0,
        )
        == 0
    )
