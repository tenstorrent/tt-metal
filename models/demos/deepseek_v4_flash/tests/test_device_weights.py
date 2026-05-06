# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from models.demos.deepseek_v4_flash.converter import convert_hf_checkpoint
from models.demos.deepseek_v4_flash.device_weights import PreprocessedWeightIndex, TtDeviceWeightOwner
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint


def test_preprocessed_weight_index_loads_canonical_tensors(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=1)
    preprocessed = convert_hf_checkpoint(snapshot, tmp_path / "tt_preprocessed")
    index = PreprocessedWeightIndex(preprocessed)

    embed = index.load_torch("embed.weight")
    expert = index.load_torch("layers.0.ffn.experts.0.w1.weight_packed")

    assert tuple(embed.shape) == (64, 32)
    assert expert.dtype.is_floating_point is False
    assert "layers.0.ffn.experts.0.w1.weight_packed" in index.keys()
    with pytest.raises(KeyError, match="not present"):
        index.record("missing.weight")


def test_device_weight_owner_reports_placements_without_loading(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=1, num_routed_experts=8)
    preprocessed = convert_hf_checkpoint(snapshot, tmp_path / "tt_preprocessed")
    owner = TtDeviceWeightOwner(preprocessed, mesh_device=object(), mesh_shape=(2, 4))

    assert owner.placement("embed.weight").strategy in ("tp_shard_replicate_ep", "replicate_all")
    expert_placement = owner.placement("layers.0.ffn.experts.7.w3.scale")
    assert expert_placement.strategy == "expert_home_device"
    assert expert_placement.devices == ((1, 3),)
    assert owner.owned_keys == ()
