# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Equivalence tests between ``prepare_*`` and ``CacheWeightProvider`` load paths."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file

import ttnn
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights
from models.demos.deepseek_v3_b1.demo import weight_provider as weight_provider_mod
from models.demos.deepseek_v3_b1.demo.weight_provider import CacheWeightProvider
from models.demos.deepseek_v3_b1.prepare_weights import (
    CURRENT_TRANSFORM_VERSION,
    DeepSeekV3MoELayerWeights,
    prepare_moe_layer_weights,
)
from models.demos.deepseek_v3_b1.tensor_cache import CacheConfig, CacheContext, TensorCache
from models.demos.deepseek_v3_b1.tests.unit_tests.test_prepare_weights import (
    NUM_ROUTED_EXPERTS_FOR_TESTS,
    _assert_overlapped_tensors_match,
    _deallocate_layer,
    _layer_state_dict,
    _skip_unless_4x2_mesh,
)


def _write_hf_model_dir(model_dir: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Minimal HF layout: one shard + model.safetensors.index.json for LazyStateDict."""
    model_dir.mkdir(parents=True, exist_ok=True)
    shard_name = "model.safetensors"
    save_file(tensors, str(model_dir / shard_name))
    index = {"metadata": {}, "weight_map": {k: shard_name for k in tensors}}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")


def _test_cache_context(mesh_shape: tuple[int, int] = (4, 2)) -> CacheContext:
    return CacheContext(
        schema_version=1,
        hf_model_id="test-model",
        hf_revision="test-rev",
        transform_version=CURRENT_TRANSFORM_VERSION,
        mesh_shape=mesh_shape,
    )


def _assert_moe_layer_weights_equivalent(
    a: DeepSeekV3MoELayerWeights,
    b: DeepSeekV3MoELayerWeights,
    *,
    mesh_device: ttnn.MeshDevice,
) -> None:
    """Structural + value parity for MoE layer (overlapped metadata and host tensor equality)."""
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    def _tt_close(x: ttnn.Tensor, y: ttnn.Tensor, *, rtol: float = 1e-2, atol: float = 0.2) -> None:
        tx = ttnn.to_torch(x, mesh_composer=composer)
        ty = ttnn.to_torch(y, mesh_composer=composer)
        assert tx.shape == ty.shape, f"shape mismatch {tx.shape} vs {ty.shape}"
        assert torch.allclose(tx, ty, rtol=rtol, atol=atol), "tensor value mismatch"

    for name in (
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj",
        "o_proj",
        "gate_mm",
        "attn_norm",
        "q_norm",
        "kv_norm",
        "ffn_norm",
        "kv_b1_proj",
        "kv_b2_proj",
        "shared_gate_proj",
        "shared_up_proj",
    ):
        _assert_overlapped_tensors_match(getattr(a, name), getattr(b, name))
        _tt_close(getattr(a, name).fused_tensor, getattr(b, name).fused_tensor)

    _tt_close(a.shared_down_proj, b.shared_down_proj)
    _tt_close(a.gate_bias, b.gate_bias)

    assert len(a.routed_gate_proj) == len(b.routed_gate_proj)
    for tg, ug in zip(a.routed_gate_proj, b.routed_gate_proj, strict=True):
        _tt_close(tg, ug)
    for tu, uu in zip(a.routed_up_proj, b.routed_up_proj, strict=True):
        _tt_close(tu, uu)
    for td, ud in zip(a.routed_down_proj, b.routed_down_proj, strict=True):
        _tt_close(td, ud)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_moe_layer_matches_cache_weight_provider_moe_load(bh_2d_mesh_device, tmp_path):
    """``prepare_moe_layer_weights`` vs ``CacheWeightProvider.load_moe_layer`` (cache + structural/value parity)."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))

    layer_idx = 2
    state = _layer_state_dict(layer_idx, is_moe=True, seed=91)
    model_dir = tmp_path / "hf_model"
    _write_hf_model_dir(model_dir, state)

    cache_dir = tmp_path / "tensor_cache"
    cache_config = CacheConfig(cache=TensorCache(cache_dir), context=_test_cache_context())
    bdw = BlitzDecodeWeights(submesh)

    with mock.patch.object(weight_provider_mod, "NUM_ROUTED_EXPERTS", NUM_ROUTED_EXPERTS_FOR_TESTS):
        via_prepare = prepare_moe_layer_weights(
            bdw,
            state,
            layer_idx,
            num_routed_experts=NUM_ROUTED_EXPERTS_FOR_TESTS,
            move_to_device=True,
            cache_config=cache_config,
        )
        provider = CacheWeightProvider(
            cache_dir,
            model_dir,
            hf_model_id="test-model",
            hf_revision="test-rev",
            schema_version=1,
        )
        via_provider = provider.load_moe_layer(layer_idx, submesh)

    try:
        _assert_moe_layer_weights_equivalent(via_prepare, via_provider, mesh_device=submesh)
    finally:
        _deallocate_layer(via_prepare)
        _deallocate_layer(via_provider)
