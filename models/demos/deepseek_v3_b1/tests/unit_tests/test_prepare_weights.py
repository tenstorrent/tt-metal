# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for prepare_weights: per-layer prepare on 4x2 mesh and TensorCache integration.

- test_prepare_dense_layer_single_layer_4x2 / test_prepare_moe_layer_single_layer_4x2: one layer on 4x2 mesh.
- test_prepare_attention_weights_*_4x2, test_prepare_shared_expert_weights_*_4x2, test_prepare_routed_expert_weights_*_4x2: per-group prepare on 4x2 mesh.
- test_prepare_embedding_weights_4x2 / test_prepare_lm_head_weights_4x2: global weights on 4x2 mesh.
- test_cache_routed_experts_miss_and_hit_4x2: TensorCache routed expert caching roundtrip.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights
from models.demos.deepseek_v3_b1.prepare_weights import (
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MoELayerWeights,
    DenseRoutedExpertWeights,
    MoERoutedExpertWeights,
    prepare_attention_weights,
    prepare_dense_layer_weights,
    prepare_embedding_weights,
    prepare_lm_head_weights,
    prepare_moe_layer_weights,
    prepare_routed_expert_weights,
    prepare_shared_expert_weights,
)


def _assert_on_device(tensor: ttnn.Tensor) -> None:
    """Assert the tensor storage type is DEVICE."""
    assert tensor.storage_type() == ttnn.StorageType.DEVICE, f"Expected DEVICE storage, got {tensor.storage_type()}"


def _skip_unless_4x2_mesh(bh_2d_mesh_device):
    """Skip test if mesh device does not have enough devices for a 4x2 submesh."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires 8 devices (4x2 mesh)")


# HF state dict shapes (out_features, in_features) for linears; full logical for 4x2 mesh. See DEEPSEEK_PREPARE_WEIGHTS_DESIGN_DOC.md §5.
HF_Q_B_FULL_LOGICAL = (24576, 1536)
HF_O_PROJ_FULL_LOGICAL = (7168, 16384)
HF_KV_B_FULL_LOGICAL = (32768, 512)
HF_SHARED_GATE_UP_FULL_LOGICAL = (2048, 7168)

# Don't use all the experts in tests to avoid taking too long
NUM_ROUTED_EXPERTS = 4


def _layer_state_dict(
    layer_idx: int,
    *,
    is_moe: bool,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Build a minimal state_dict for one layer (HF key convention, random weights).

    Uses full logical shapes for 4x2 mesh; prepare_weights passes them to blitz, which shards across the mesh.
    """
    g = torch.Generator().manual_seed(seed)
    q_b_hf = HF_Q_B_FULL_LOGICAL
    o_proj_hf = HF_O_PROJ_FULL_LOGICAL
    kv_b_hf = HF_KV_B_FULL_LOGICAL
    shared_hf = HF_SHARED_GATE_UP_FULL_LOGICAL

    state = {
        f"model.layers.{layer_idx}.self_attn.q_a_proj.weight": torch.randn(
            1536, 7168, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.q_b_proj.weight": torch.randn(*q_b_hf, generator=g, dtype=torch.bfloat16),
        f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight": torch.randn(
            576, 7168, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight": torch.randn(
            *kv_b_hf, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.o_proj.weight": torch.randn(*o_proj_hf, generator=g, dtype=torch.bfloat16),
        f"model.layers.{layer_idx}.input_layernorm.weight": torch.randn(7168, generator=g, dtype=torch.bfloat16),
        f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight": torch.randn(
            1536, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight": torch.randn(
            512, generator=g, dtype=torch.bfloat16
        ),
        f"model.layers.{layer_idx}.post_attention_layernorm.weight": torch.randn(
            7168, generator=g, dtype=torch.bfloat16
        ),
    }
    if is_moe:
        state[f"model.layers.{layer_idx}.mlp.gate.weight"] = torch.randn(256, 7168, generator=g, dtype=torch.bfloat16)
        state[f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"] = torch.randn(
            256, generator=g, dtype=torch.bfloat16
        )
        state[f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"] = torch.randn(
            *shared_hf, generator=g, dtype=torch.bfloat16
        )
        state[f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"] = torch.randn(
            *shared_hf, generator=g, dtype=torch.bfloat16
        )
        # shared down: HF (7168, 2048*tp) full logical
        shared_down_rows = 7168
        shared_down_cols = shared_hf[0]  # 2048 for 4x2
        state[f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"] = torch.randn(
            shared_down_rows, shared_down_cols, generator=g, dtype=torch.bfloat16
        )
        for e in range(NUM_ROUTED_EXPERTS):
            state[f"model.layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
                2048, 7168, generator=g, dtype=torch.bfloat16
            )
            state[f"model.layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"] = torch.randn(
                2048, 7168, generator=g, dtype=torch.bfloat16
            )
            state[f"model.layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"] = torch.randn(
                7168, 2048, generator=g, dtype=torch.bfloat16
            )
    else:
        # Dense MLP: HF (out, in) gate/up (18432, 7168), down (7168, 18432)
        state[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = torch.randn(
            18432, 7168, generator=g, dtype=torch.bfloat16
        )
        state[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = torch.randn(
            18432, 7168, generator=g, dtype=torch.bfloat16
        )
        state[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = torch.randn(
            7168, 18432, generator=g, dtype=torch.bfloat16
        )
    return state


def _add_global_weights(state: dict[str, torch.Tensor], seed: int = 42) -> None:
    """Add embedding, final norm, and lm_head to state (in place)."""
    g = torch.Generator().manual_seed(seed)
    state["model.embed_tokens.weight"] = torch.randn(129280, 7168, generator=g, dtype=torch.bfloat16)
    state["model.norm.weight"] = torch.randn(7168, generator=g, dtype=torch.bfloat16)
    state["lm_head.weight"] = torch.randn(129280, 7168, generator=g, dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_attention_weights_dense_4x2(bh_2d_mesh_device):
    """Prepare attention weights only for a dense layer on 4x2 mesh; verify shapes and fusion group sharing."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=False)
    bdw = BlitzDecodeWeights(submesh)
    attn = prepare_attention_weights(bdw, state, 0, is_moe=False)
    assert attn.gate_mm is None
    assert attn.q_a_proj.tensor_shape == (3584, 3072)
    assert attn.q_b_proj.tensor_shape == (1536, 12288)
    assert attn.kv_a_proj.tensor_shape == (7168, 576)
    assert attn.o_proj.tensor_shape == (8192, 7168)
    assert attn.attn_norm.tensor_shape == (1, 7168)
    assert attn.kv_b1_proj.tensor_shape == (8192, 512)
    assert attn.kv_b2_proj.tensor_shape == (512, 8192)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_attention_weights_moe_4x2(bh_2d_mesh_device):
    """Prepare attention weights only for an MoE layer on 4x2 mesh; verify shapes and gate_mm present."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=True, seed=43)
    bdw = BlitzDecodeWeights(submesh)
    attn = prepare_attention_weights(bdw, state, 0, is_moe=True)
    assert attn.gate_mm is not None
    assert attn.gate_mm.tensor_shape == (7168, 256)
    assert attn.gate_bias is not None
    assert attn.gate_bias.shape == (16, 16)
    assert attn.q_a_proj.tensor_shape == (3584, 3072)
    assert attn.o_proj.tensor_shape == (8192, 7168)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_shared_expert_weights_dense_4x2(bh_2d_mesh_device):
    """Prepare shared expert weights only for a dense layer on 4x2 mesh; verify shapes."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=False)
    bdw = BlitzDecodeWeights(submesh)
    shared = prepare_shared_expert_weights(bdw, state, 0, is_moe=False)
    assert shared.shared_gate_proj.tensor_shape is not None
    assert shared.shared_up_proj.tensor_shape is not None
    assert shared.shared_down_proj.shape is not None


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_shared_expert_weights_moe_4x2(bh_2d_mesh_device):
    """Prepare shared expert weights only for an MoE layer on 4x2 mesh; verify shapes."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=True, seed=43)
    bdw = BlitzDecodeWeights(submesh)
    shared = prepare_shared_expert_weights(bdw, state, 0, is_moe=True)
    assert shared.shared_gate_proj.tensor_shape == (7168, 256)
    assert shared.shared_up_proj.tensor_shape == (7168, 256)
    assert shared.shared_down_proj.shape is not None


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_routed_expert_weights_dense_4x2(bh_2d_mesh_device):
    """Prepare routed expert weights only for a dense layer on 4x2 mesh; verify shapes."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=False)
    bdw = BlitzDecodeWeights(submesh)
    routed = prepare_routed_expert_weights(bdw, state, 0, is_moe=False)
    assert isinstance(routed, DenseRoutedExpertWeights)
    assert routed.routed_gate_proj.shape is not None
    assert routed.routed_up_proj.shape is not None
    assert routed.routed_down_proj.shape is not None


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_routed_expert_weights_moe_4x2(bh_2d_mesh_device):
    """Prepare routed expert weights only for an MoE layer on 4x2 mesh; verify shapes and expert count."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=True, seed=43)
    bdw = BlitzDecodeWeights(submesh)
    routed = prepare_routed_expert_weights(
        bdw,
        state,
        0,
        is_moe=True,
        num_routed_experts=NUM_ROUTED_EXPERTS,
        move_to_device=True,
    )
    assert isinstance(routed, MoERoutedExpertWeights)
    assert len(routed.routed_gate_proj) == NUM_ROUTED_EXPERTS
    assert len(routed.routed_up_proj) == NUM_ROUTED_EXPERTS
    assert len(routed.routed_down_proj) == NUM_ROUTED_EXPERTS
    routed.validate_contiguous_dram()


def _moe_routed_expert_stacked_tensors(seed: int = 43):
    """Build (num_experts, K, N) stacked gate/up and (num_experts, K_down, N_down) down for get_tt_moe_routed_expert_weights."""
    g = torch.Generator().manual_seed(seed)
    # MoE expert shapes: gate/up (K=7168, N=2048), down (2048, 7168)
    gate_stacked = torch.randn(NUM_ROUTED_EXPERTS, 7168, 2048, generator=g, dtype=torch.bfloat16)
    up_stacked = torch.randn(NUM_ROUTED_EXPERTS, 7168, 2048, generator=g, dtype=torch.bfloat16)
    down_stacked = torch.randn(NUM_ROUTED_EXPERTS, 2048, 7168, generator=g, dtype=torch.bfloat16)
    return gate_stacked, up_stacked, down_stacked


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_dense_layer_single_layer_4x2(bh_2d_mesh_device):
    """Build one dense layer on 4x2 mesh; verify type and shapes (MLA TP=2)."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=False)
    bdw = BlitzDecodeWeights(submesh)
    t0 = time.perf_counter()
    layer = prepare_dense_layer_weights(bdw, state, 0)
    elapsed = time.perf_counter() - t0
    logger.info("prepare_dense_layer_weights (1 dense layer, 4x2 mesh): {:.3f} s", elapsed)
    assert isinstance(layer, DeepSeekV3DenseLayerWeights)
    assert layer.q_a_proj.tensor_shape == (3584, 3072)
    assert layer.q_b_proj.tensor_shape == (1536, 12288)
    assert layer.kv_a_proj.tensor_shape == (7168, 576)
    assert layer.o_proj.tensor_shape == (8192, 7168)
    assert layer.attn_norm.tensor_shape == (1, 7168)
    assert layer.q_norm.tensor_shape == (1, 1536)
    assert layer.kv_norm.tensor_shape == (1, 512)
    assert layer.ffn_norm.tensor_shape == (1, 7168)
    assert layer.kv_b1_proj.tensor_shape == (8192, 512)
    assert layer.kv_b2_proj.tensor_shape == (512, 8192)
    assert hasattr(layer, "shared_gate_proj") and layer.shared_gate_proj is not None
    assert hasattr(layer, "shared_up_proj") and layer.shared_up_proj is not None
    assert hasattr(layer, "routed_gate_proj") and layer.routed_gate_proj is not None
    assert hasattr(layer, "routed_up_proj") and layer.routed_up_proj is not None
    assert hasattr(layer, "routed_down_proj") and layer.routed_down_proj is not None


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_moe_layer_single_layer_4x2(bh_2d_mesh_device):
    """Build one MoE layer on 4x2 mesh; verify type and shapes (MLA TP=2, MoE TP=8)."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = _layer_state_dict(0, is_moe=True, seed=43)
    bdw = BlitzDecodeWeights(submesh)
    logger.info(f"State dict prepared")
    t0 = time.perf_counter()
    logger.info(f"Preparing weights...")
    layer = prepare_moe_layer_weights(bdw, state, 0, num_routed_experts=NUM_ROUTED_EXPERTS)
    logger.info(f"Weights prepared")
    elapsed = time.perf_counter() - t0
    logger.info("prepare_moe_layer_weights (1 MoE layer, 4x2 mesh): {:.3f} s", elapsed)
    assert isinstance(layer, DeepSeekV3MoELayerWeights)
    assert layer.q_a_proj.tensor_shape == (3584, 3072)
    assert layer.q_b_proj.tensor_shape == (1536, 12288)
    assert layer.kv_a_proj.tensor_shape == (7168, 576)
    assert layer.o_proj.tensor_shape == (8192, 7168)
    assert layer.gate_mm.tensor_shape == (7168, 256)
    assert layer.gate_bias.shape == (16, 16)
    assert layer.attn_norm.tensor_shape == (1, 7168)
    assert layer.q_norm.tensor_shape == (1, 1536)
    assert layer.kv_norm.tensor_shape == (1, 512)
    assert layer.ffn_norm.tensor_shape == (1, 7168)
    assert layer.kv_b1_proj.tensor_shape == (8192, 512)
    assert layer.kv_b2_proj.tensor_shape == (512, 8192)
    assert layer.shared_gate_proj.tensor_shape == (7168, 256)
    assert layer.shared_up_proj.tensor_shape == (7168, 256)
    assert hasattr(layer, "shared_down_proj")
    assert len(layer.routed_gate_proj) == NUM_ROUTED_EXPERTS
    assert len(layer.routed_up_proj) == NUM_ROUTED_EXPERTS
    assert len(layer.routed_down_proj) == NUM_ROUTED_EXPERTS


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_embedding_weights_4x2(bh_2d_mesh_device):
    """Prepare embedding weights on 4x2 mesh; verify shape. Tensors stay on host until load_embedding_weights."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = {}
    _add_global_weights(state)
    weights = prepare_embedding_weights(state, submesh)
    assert isinstance(weights, DeepSeekV3EmbeddingLayerWeights)
    assert weights.embedding.shape is not None
    assert weights.embedding.shape == (
        129280,
        7168,
    ), f"Expected embedding shape (129280, 7168), got {weights.embedding.shape}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_prepare_lm_head_weights_4x2(bh_2d_mesh_device):
    """Prepare LM head and final norm weights on 4x2 mesh; verify shapes. LM head is vocab-sharded on device (TP=8)."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    state = {}
    _add_global_weights(state)
    weights = prepare_lm_head_weights(state, submesh)
    assert isinstance(weights, DeepSeekV3LMHeadWeights)
    assert weights.lm_head.shape is not None
    assert weights.lm_head.shape == (7168, 16160), f"Expected lm_head shape (7168, 16160), got {weights.lm_head.shape}"
    assert weights.final_norm.shape is not None
    assert weights.final_norm.shape == (1, 7168), f"Expected final_norm shape (1, 7168), got {weights.final_norm.shape}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_cache_routed_experts_miss_and_hit_4x2(bh_2d_mesh_device, tmp_path):
    """Test TensorCache.get_or_create_tensor_list: miss creates and dumps expert tensors, hit loads from sentinel-guarded dir."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))

    from models.demos.deepseek_v3_b1.tensor_cache import Fingerprint, TensorCache

    cache = TensorCache(tmp_path / "tensor_cache")
    mesh_shape = (submesh.shape[0], submesh.shape[1])
    fp = Fingerprint(
        schema_version=1,
        hf_model_id="test-model",
        hf_revision="test-rev",
        transform_version=1,
        mesh_shape=mesh_shape,
        group_name="routed_gate_proj",
        layer_idx=0,
        spec_fingerprints=(),
    )

    gate_stacked, up_stacked, down_stacked = _moe_routed_expert_stacked_tensors(seed=44)
    bdw = BlitzDecodeWeights(submesh)
    create_called = [0]

    def _create_gate():
        create_called[0] += 1
        g, _u, _d = bdw.get_tt_moe_routed_expert_weights(gate_stacked, up_stacked, down_stacked)
        return list(g)

    result1 = cache.get_or_create_tensor_list(fp, create=_create_gate, device=submesh)
    assert create_called[0] == 1, "Expected create callback on cache miss"
    assert isinstance(result1, list)
    assert len(result1) == NUM_ROUTED_EXPERTS
    shapes1 = [result1[e].shape for e in range(NUM_ROUTED_EXPERTS)]

    list_dir = cache._artifact_dir(fp.artifact_id()) / "list"
    assert (list_dir / "_complete").exists(), "Sentinel marker must exist after miss"
    assert len(list(list_dir.glob("*.tensorbin"))) == NUM_ROUTED_EXPERTS

    for t in result1:
        ttnn.deallocate(t, force=True)
    del result1

    result2 = cache.get_or_create_tensor_list(fp, create=_create_gate, device=submesh)
    assert create_called[0] == 1, "create callback must NOT be called on cache hit"
    assert isinstance(result2, list)
    assert len(result2) == NUM_ROUTED_EXPERTS
    for e in range(NUM_ROUTED_EXPERTS):
        assert result2[e].shape == shapes1[e]
        _assert_on_device(result2[e])
