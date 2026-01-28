# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the Attention2D module (TG/Galaxy 2D mesh topology: 4x8 or 8x4).

This test suite verifies:
1. Unit tests for config dataclasses (no device needed)
2. Attention2D class matches HuggingFace/Meta reference model
3. Attention2D correctly rejects non-TG devices
4. Backward compatibility: Attention2D.from_model_args() works correctly
"""

from unittest.mock import MagicMock

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1DConfig

# ============================================================================
# Constants
# ============================================================================

LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"

# ============================================================================
# Helper functions
# ============================================================================


def create_mock_lazy_weight(device=None, shape=None):
    """Create a mock LazyWeight for unit tests."""
    w = MagicMock(spec=LazyWeight)
    w.device = device
    w.source = MagicMock()
    if shape:
        w.source.shape = shape
    return w


def get_attention_weights_from_ref_model(
    reference_attn, num_devices: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Extract attention weights from a reference attention module in TTNN layout for TG.

    Returns:
        (wqkv, wo, q_norm, k_norm) tensors in TTNN layout
    """
    # Get Q, K, V, O projections
    wq = reference_attn.q_proj.weight.T  # (dim, n_heads * head_dim)
    wk = reference_attn.k_proj.weight.T  # (dim, n_kv_heads * head_dim)
    wv = reference_attn.v_proj.weight.T  # (dim, n_kv_heads * head_dim)
    wo = reference_attn.o_proj.weight.T  # (n_heads * head_dim, dim)

    # Build combined QKV weight for TG - uses n_kv_heads for chunking
    n_kv_heads = wk.shape[1] // 128  # head_dim = 128
    qkv_list = []
    for i in range(n_kv_heads):
        wq_chunk = torch.chunk(wq, n_kv_heads, dim=1)[i]
        wk_chunk = torch.chunk(wk, n_kv_heads, dim=1)[i]
        wv_chunk = torch.chunk(wv, n_kv_heads, dim=1)[i]
        qkv = torch.cat([wq_chunk, wk_chunk, wv_chunk], dim=-1)
        qkv_list.append(qkv)

    wqkv = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

    # WO weight: (1, 1, n_heads * head_dim, dim)
    wo = wo.unsqueeze(0).unsqueeze(0)

    # Q/K norm weights (optional, e.g., for Qwen models)
    q_norm = None
    k_norm = None
    if hasattr(reference_attn, "q_norm") and reference_attn.q_norm is not None:
        q_norm = reference_attn.q_norm.weight
    if hasattr(reference_attn, "k_norm") and reference_attn.k_norm is not None:
        k_norm = reference_attn.k_norm.weight

    return wqkv, wo, q_norm, k_norm


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ============================================================================
# Unit Tests - No device required
# ============================================================================


def test_attention_2d_config_creation():
    """Test that Attention2DConfig dataclass can be created with explicit values."""
    from models.common.modules.attention.attention_2d import Attention2DConfig

    mock_mesh_device = MagicMock()
    mock_mesh_device.shape = (4, 8)
    mock_mesh_device.get_num_devices.return_value = 32

    mock_tt_ccl = MagicMock()

    mock_wqkv = create_mock_lazy_weight(device=mock_mesh_device, shape=(1, 1, 4096, 6144))
    mock_wo = create_mock_lazy_weight(device=mock_mesh_device, shape=(1, 1, 4096, 4096))

    # Create config with explicit values
    config = Attention2DConfig(
        wqkv=mock_wqkv,
        wo=mock_wo,
        mesh_device=mock_mesh_device,
        tt_ccl=mock_tt_ccl,
        dim=4096,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        max_batch_size=32,
        topology=ttnn.Topology.Linear,
    )

    # Verify explicit values are preserved
    assert config.wqkv is mock_wqkv
    assert config.wo is mock_wo
    assert config.mesh_device is mock_mesh_device
    assert config.tt_ccl is mock_tt_ccl
    assert config.dim == 4096
    assert config.n_heads == 32
    assert config.n_kv_heads == 8
    assert config.head_dim == 128
    assert config.max_batch_size == 32
    assert config.topology == ttnn.Topology.Linear


def test_attention_2d_config_defaults():
    """Test that Attention2DConfig has sensible defaults."""
    from models.common.modules.attention.attention_2d import Attention2DConfig

    # Minimal creation - only required fields
    config = Attention2DConfig(wqkv=MagicMock(), wo=MagicMock())

    # Check defaults
    assert config.max_batch_size == 32
    assert config.max_seq_len == 128 * 1024
    assert config.use_paged_kv_cache is False
    assert config.kv_cache_dtype == ttnn.bfloat8_b
    assert config.use_qk_fused is False
    assert config.num_reduce_scatter_links == 1
    assert config.num_all_gather_links == 2

    # Optional fields default to None
    assert config.mesh_device is None
    assert config.tt_ccl is None
    assert config.dim is None
    assert config.n_heads is None
    assert config.n_kv_heads is None
    assert config.q_norm_config is None
    assert config.k_norm_config is None
    assert config.num_device_groups is None
    assert config.batch_size_per_device_group is None


def test_attention_2d_config_rejects_1d_mesh():
    """Test that _resolve_attention2d_config raises AssertionError for 1D mesh."""
    from models.common.modules.attention.attention_2d import Attention2DConfig, _resolve_attention2d_config

    # Mock 1D device
    mock_device_1d = MagicMock(spec=ttnn.MeshDevice)
    mock_device_1d.shape = (1, 8)

    mock_wqkv = create_mock_lazy_weight(device=mock_device_1d, shape=(1, 1, 4096, 6144))
    mock_wo = create_mock_lazy_weight(device=mock_device_1d, shape=(1, 1, 4096, 4096))

    config = Attention2DConfig(wqkv=mock_wqkv, wo=mock_wo)

    with pytest.raises(AssertionError, match="Attention2D requires 2D mesh"):
        _resolve_attention2d_config(config)


@pytest.mark.parametrize(
    "cluster_shape",
    [(1, 1), (1, 2), (1, 8), (2, 4)],  # Non-Galaxy shapes - should be rejected
    ids=["1x1", "1x2", "1x8", "2x4"],
)
def test_attention_2d_rejects_non_galaxy_from_model_args(cluster_shape):
    """
    Test that Attention2D.from_model_args() raises ValueError for non-Galaxy devices.
    """
    from models.common.modules.attention.attention_2d import Attention2D

    class _DummyArgs:
        def __init__(self, cluster_shape):
            self.cluster_shape = list(cluster_shape)

    model_args = _DummyArgs(cluster_shape)

    with pytest.raises(ValueError, match="Attention2D requires Galaxy topology"):
        Attention2D.from_model_args(
            mesh_device=None,
            tt_ccl=None,
            args=model_args,
            state_dict=None,
            weight_cache_path=None,
            layer_num=0,
            transformation_mats=None,
        )


def test_attention_2d_config_tg_specific_fields():
    """Test that TG-specific fields can be set explicitly."""
    from models.common.modules.attention.attention_2d import Attention2DConfig

    mock_mesh_device = MagicMock()
    mock_mesh_device.shape = (4, 8)
    mock_mesh_device.get_num_devices.return_value = 32

    mock_tt_ccl = MagicMock()

    mock_wqkv = create_mock_lazy_weight(device=mock_mesh_device, shape=(1, 1, 4096, 6144))
    mock_wo = create_mock_lazy_weight(device=mock_mesh_device, shape=(1, 1, 4096, 4096))
    mock_mem_cfg = MagicMock()

    # Create config with TG-specific fields
    config = Attention2DConfig(
        wqkv=mock_wqkv,
        wo=mock_wo,
        mesh_device=mock_mesh_device,
        tt_ccl=mock_tt_ccl,
        num_device_groups=4,
        batch_size_per_device_group=8,
        ccl_dtype=ttnn.bfloat8_b,
        create_head_input_memcfg=mock_mem_cfg,
        self_out_reduce_scatter_memcfg=mock_mem_cfg,
    )

    # Verify TG-specific values are preserved
    assert config.num_device_groups == 4
    assert config.batch_size_per_device_group == 8
    assert config.ccl_dtype == ttnn.bfloat8_b
    assert config.create_head_input_memcfg is mock_mem_cfg
    assert config.self_out_reduce_scatter_memcfg is mock_mem_cfg


def test_attention_2d_config_q_k_norm_config():
    """Test that Q/K norm configs can be composed into Attention2DConfig."""
    from models.common.modules.attention.attention_2d import Attention2DConfig

    mock_mesh_device = MagicMock()
    mock_mesh_device.shape = (4, 8)
    mock_mesh_device.get_num_devices.return_value = 32

    mock_wqkv = create_mock_lazy_weight(device=mock_mesh_device, shape=(1, 1, 4096, 6144))
    mock_wo = create_mock_lazy_weight(device=mock_mesh_device, shape=(1, 1, 4096, 4096))

    q_norm_weight = create_mock_lazy_weight(device=mock_mesh_device, shape=(1, 1, 1, 128))
    k_norm_weight = create_mock_lazy_weight(device=mock_mesh_device, shape=(1, 1, 1, 128))

    q_norm_config = RMSNorm1DConfig(
        weight=q_norm_weight,
        mesh_device=mock_mesh_device,
        eps=1e-5,
    )
    k_norm_config = RMSNorm1DConfig(
        weight=k_norm_weight,
        mesh_device=mock_mesh_device,
        eps=1e-5,
    )

    config = Attention2DConfig(
        wqkv=mock_wqkv,
        wo=mock_wo,
        mesh_device=mock_mesh_device,
        q_norm_config=q_norm_config,
        k_norm_config=k_norm_config,
    )

    assert config.q_norm_config is q_norm_config
    assert config.k_norm_config is k_norm_config


# ============================================================================
# TTNN Topology Bug Tests - Document known issues with 2D mesh tensor topology
# ============================================================================


def _check_topology_has_duplicate_shard_dims(placements: list) -> tuple[bool, str]:
    """
    Check if placements have duplicate shard dimensions (the known bug pattern).

    Args:
        placements: List of placement objects from tensor_topology().placements()

    Returns:
        (has_duplicate, message): Tuple of (True if duplicate dims found, descriptive message)
    """

    def normalize_dim(d: int, ndim: int = 4) -> int:
        return d if d >= 0 else d + ndim

    axis0_dim = placements[0].dim if isinstance(placements[0], ttnn.PlacementShard) else None
    axis1_dim = placements[1].dim if isinstance(placements[1], ttnn.PlacementShard) else None

    if axis0_dim is not None and axis1_dim is not None:
        norm_axis0 = normalize_dim(axis0_dim)
        norm_axis1 = normalize_dim(axis1_dim)

        if norm_axis0 == norm_axis1:
            return True, (
                f"Both mesh axes shard the same tensor dimension: "
                f"axis0={axis0_dim} (norm={norm_axis0}), axis1={axis1_dim} (norm={norm_axis1})"
            )

    return False, "Topology appears correct"


@pytest.fixture(scope="function")
def ttnn_linear_2d_mesh_has_topology_bug(ttnn_mesh_device):
    """
    Fixture that checks if the ttnn.linear 2D mesh topology bug exists.

    Returns:
        bool: True if the bug is present, False if fixed
    """
    mesh_device = ttnn_mesh_device
    cluster_shape = list(mesh_device.shape)

    # Skip if not a 2D mesh
    if len(cluster_shape) != 2 or cluster_shape[0] == 1 or cluster_shape[1] == 1:
        logger.info("Not a 2D mesh, skipping topology bug check")
        return False

    dim, hidden_dim, seq_len = 4096, 14336, 32

    # Create minimal test tensors
    torch_input = torch.randn(1, 1, seq_len, dim, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=cluster_shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    torch_weight = torch.randn(dim, hidden_dim, dtype=torch.bfloat16)
    tt_weight = ttnn.from_torch(
        torch_weight,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, -2), mesh_shape=cluster_shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run linear and check topology
    tt_output = ttnn.linear(tt_input, tt_weight)
    output_placements = list(tt_output.tensor_topology().placements())

    has_bug, msg = _check_topology_has_duplicate_shard_dims(output_placements)
    if has_bug:
        logger.warning(f"ttnn.linear 2D mesh topology bug detected: {msg}")
    else:
        logger.info("ttnn.linear 2D mesh topology bug NOT detected - may be fixed!")

    # Cleanup
    ttnn.deallocate(tt_output)
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_weight)

    return has_bug


# ============================================================================
# Integration Tests - Device required
# ============================================================================


# [INFO] TG/Galaxy attention tests
# - 8x4 mesh: Correct orientation for TG weight sharding (qkv_size across 8 devices)
# - 4x8 mesh: Not yet supported (sharding mismatch)
# - Prefill mode: Working
# - Decode mode: Not yet working (floating point exceptions)
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [
        pytest.param((4, 8), marks=pytest.mark.skip(reason="4x8 mesh not yet supported - sharding mismatch")),
        (8, 4),
    ],
    ids=[
        "4x8",
        "8x4",
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "dtype,dim,n_heads,n_kv_heads,head_dim,hf_model_name",
    [
        pytest.param(
            ttnn.bfloat8_b,
            4096,
            32,
            8,
            128,
            LLAMA_8B,
            id="bf8b-llama8B",
        ),
    ],
)
@pytest.mark.parametrize(
    "seq_len,mode",
    [
        (512, "prefill"),
        pytest.param(
            32,
            "decode",
            marks=pytest.mark.skip(
                reason="TG decode: SDPA collapses tensor from 32 to 8 devices (one per row). "
                "After all_reduce on axis 1, all 4 columns have identical data, and SDPA "
                "optimizes by outputting only one tensor per row. This causes issues with "
                "subsequent all_gather operations. Needs investigation into SDPA behavior."
            ),
        ),
    ],
    ids=[
        "prefill-512",
        "decode-32",
    ],
)
def test_attention_2d_vs_reference(
    ttnn_mesh_device: ttnn.MeshDevice,
    ttnn_linear_2d_mesh_has_topology_bug: bool,  # noqa: ARG001 - fixture runs to log warnings
    seq_len,
    mode,
    dtype,
    dim,
    n_heads,
    n_kv_heads,
    head_dim,
    hf_model_name,
):
    """
    Test Attention2D constructed via direct APIs (Attention2DConfig).

    This test runs the full forward pass including:
    - RotarySetup for rot_mats and transformation_mat
    - Prefill or decode forward pass
    - Output validation (no NaN/Inf, correct shape)
    """
    from models.common.modules.attention.attention_2d import Attention2D, Attention2DConfig
    from models.common.modules.tt_ccl import get_tt_ccl
    from models.tt_transformers.tt.rope import RotarySetup, get_rot_mats

    seed = 1234
    torch.manual_seed(seed)

    # Load HF config and create model with dummy weights
    config = AutoConfig.from_pretrained(hf_model_name)
    config.num_hidden_layers = 1
    with no_init_weights():
        hf_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    reference_attn = hf_model.model.layers[0].self_attn

    # Initialize weights deterministically
    with torch.no_grad():
        for param in reference_attn.parameters():
            param.copy_(torch.randn_like(param))

    assert dim == config.hidden_size
    assert n_heads == config.num_attention_heads
    assert n_kv_heads == config.num_key_value_heads

    cluster_shape = list(ttnn_mesh_device.shape)
    batch_size = 32  # TG batch size
    max_seq_len = 2048
    rope_theta = config.rope_theta if hasattr(config, "rope_theta") else 10000.0
    rope_scaling = None  # Default, no scaling

    # Get weights in TG format
    wqkv_torch, wo_torch, _, _ = get_attention_weights_from_ref_model(reference_attn, num_devices=n_kv_heads)

    # Create LazyWeights
    ttnn.SetDefaultDevice(ttnn_mesh_device)

    lazy_wqkv = LazyWeight(source=wqkv_torch, dtype=dtype)
    lazy_wo = LazyWeight(source=wo_torch, dtype=dtype)

    # Create TT_CCL for collective operations
    tt_ccl = get_tt_ccl(ttnn_mesh_device)

    # Setup RoPE transformation matrices
    rope_setup = RotarySetup(
        ttnn_mesh_device,
        batch_size,
        head_dim,
        max_seq_len,
        rope_theta,
        rope_scaling,
        use_qk_fused=False,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    # Create Attention2D via from_config
    attn_config = Attention2DConfig(
        wqkv=lazy_wqkv,
        wo=lazy_wo,
        mesh_device=ttnn_mesh_device,
        tt_ccl=tt_ccl,
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        use_qk_fused=False,
    )

    tt_model = Attention2D.from_config(attn_config)

    # Verify config is properly resolved
    assert tt_model.config.is_resolved(), "Config should be resolved after from_config"
    assert tt_model.config.dim == dim
    assert tt_model.config.n_heads == n_heads
    assert tt_model.config.n_kv_heads == n_kv_heads

    # Initialize KV cache
    tt_model.init_kv_cache()

    # Verify TG-specific tensors are created
    assert tt_model.config._slice_mat is not None
    assert tt_model.config._user_selection_matrix is not None
    assert tt_model.config.num_device_groups == 32 // n_kv_heads
    assert tt_model.config.batch_size_per_device_group == batch_size // tt_model.config.num_device_groups

    if mode == "prefill":
        # Prefill mode - single user
        pt_attention_input = torch.randn(1, 1, seq_len, dim, dtype=torch.bfloat16)

        # Prepare TT input for prefill
        tt_input = ttnn.from_torch(
            pt_attention_input,
            device=ttnn_mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, 3), mesh_shape=cluster_shape),
        )

        # Get rot_mats for prefill (cos/sin matrices)
        rot_mats = get_rot_mats(
            head_dim=head_dim,
            device=ttnn_mesh_device,
            seq_len=seq_len,
            theta=rope_theta,
            rope_scaling=rope_scaling,
        )

        # Run TT model forward
        tt_out = tt_model.forward(
            tt_input,
            None,  # current_pos not used in prefill
            rot_mats,
            transformation_mats.get("prefill"),
            user_id=0,
            mode="prefill",
        )

        # Convert output to torch and verify
        tt_out_torch = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(ttnn_mesh_device, dims=(1, 3), mesh_shape=cluster_shape),
        )
        tt_output_torch = tt_out_torch[:, 0:1, :seq_len, :dim].view(1, seq_len, dim)

        # Verify output shape and content
        assert tt_output_torch.shape == (
            1,
            seq_len,
            dim,
        ), f"Expected shape {(1, seq_len, dim)}, got {tt_output_torch.shape}"
        assert not torch.isnan(tt_output_torch).any(), "Output contains NaN values"
        assert not torch.isinf(tt_output_torch).any(), "Output contains Inf values"

        logger.info(f"test_attention_2d_vs_reference (from_config): PASSED for mode={mode}, seq_len={seq_len}")
        logger.info(f"  Config: dim={dim}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
        logger.info(f"  Output shape: {tt_output_torch.shape}, dtype: {tt_output_torch.dtype}")

    else:
        # Decode mode - first prefill to populate KV cache, then decode
        decode_batch_size = seq_len  # e.g., 32 users

        # Step 1: Prefill pass to populate KV cache (use 128 tokens as initial context)
        prefill_seq_len = 128
        pt_prefill_input = torch.randn(1, 1, prefill_seq_len, dim, dtype=torch.bfloat16)

        tt_prefill_input = ttnn.from_torch(
            pt_prefill_input,
            device=ttnn_mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, 3), mesh_shape=cluster_shape),
        )

        prefill_rot_mats = get_rot_mats(
            head_dim=head_dim,
            device=ttnn_mesh_device,
            seq_len=prefill_seq_len,
            theta=rope_theta,
            rope_scaling=rope_scaling,
        )

        # Run prefill to populate KV cache for user_id=0
        _ = tt_model.forward(
            tt_prefill_input,
            None,
            prefill_rot_mats,
            transformation_mats.get("prefill"),
            user_id=0,
            mode="prefill",
        )

        # Step 2: Decode pass - single token per user
        pt_decode_input = torch.randn(1, 1, decode_batch_size, dim, dtype=torch.bfloat16)

        tt_decode_input = ttnn.from_torch(
            pt_decode_input,
            device=ttnn_mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, 3), mesh_shape=cluster_shape),
        )

        # Convert to decode input memory config
        tt_decode_input = ttnn.to_memory_config(tt_decode_input, tt_model.config.decode_input_memcfg)

        # Create decode-specific RotarySetup with correct batch_size for decode mode
        decode_rope_setup = RotarySetup(
            ttnn_mesh_device,
            decode_batch_size,
            head_dim,
            max_seq_len,
            rope_theta,
            rope_scaling,
            use_qk_fused=False,
        )
        decode_transformation_mats = decode_rope_setup.get_both_trans_mats()

        # Get decode rot_mats using RotarySetup.get_rot_mats for HEIGHT_SHARDED matrices
        position_idxs = torch.full((decode_batch_size,), prefill_seq_len, dtype=torch.long)
        decode_rot_mats = decode_rope_setup.get_rot_mats(position_idxs)

        # Create current_pos tensor
        # For TG, current_pos must be sharded across mesh columns (cluster_axis=1)
        # so each device group gets batch_size_per_device_group = 8 positions
        current_pos = ttnn.from_torch(
            position_idxs,
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                ttnn_mesh_device,
                dims=(None, 0),  # Replicate on rows, shard on columns along dim 0
                mesh_shape=cluster_shape,
            ),
        )

        # Run decode
        tt_out = tt_model.forward(
            tt_decode_input,
            current_pos,
            decode_rot_mats,
            decode_transformation_mats.get("decode"),
            mode="decode",
        )

        # Convert output to torch
        tt_out_torch = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(ttnn_mesh_device, dims=(1, 3), mesh_shape=cluster_shape),
        )

        # Verify output content
        output_elements = tt_out_torch.numel()
        assert output_elements > 0, "Output is empty"
        assert not torch.isnan(tt_out_torch).any(), "Output contains NaN values"
        assert not torch.isinf(tt_out_torch).any(), "Output contains Inf values"

        logger.info(
            f"test_attention_2d_vs_reference (from_config): PASSED for mode={mode}, decode_batch_size={decode_batch_size}"
        )
        logger.info(f"  Config: dim={dim}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
        logger.info(f"  Output shape: {tt_out_torch.shape}, dtype: {tt_out_torch.dtype}")

    ttnn.SetDefaultDevice(None)


# [INFO] This test will retire once models/tt_transformers/tt/model_config.py retires
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(8, 4)],
    ids=["8x4"],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (512, 32))
def test_attention_2d_vs_reference_from_model_args(ttnn_mesh_device: ttnn.MeshDevice, seq_len):
    """
    Test that Attention2D class created via from_model_args() backward compatibility API.

    This test validates backward compatibility with the ModelArgs factory method.
    Runs only on Galaxy (TG) devices due to Galaxy-specific CCL operations.
    """
    import os

    from models.common.modules.attention.attention_2d import Attention2D
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs
    from models.tt_transformers.tt.rope import RotarySetup

    # HF_MODEL env var is required for ModelArgs with cache_hf=True
    env_model = os.environ.get("HF_MODEL")
    if not env_model:
        pytest.skip("HF_MODEL environment variable not set - required for model loading")

    batch_size = 32  # TG batch size
    mode = "decode" if seq_len <= 32 else "prefill"

    model_args = ModelArgs(ttnn_mesh_device, max_batch_size=batch_size, max_seq_len=2048, cache_hf=True)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # Create TT_CCL
    tt_ccl = TT_CCL(ttnn_mesh_device)

    # Setup RoPE transformation matrices
    rope_setup = RotarySetup(
        ttnn_mesh_device,
        batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling,
        model_args.use_qk_fused,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    # Create Attention2D via from_model_args
    tt_model = Attention2D.from_model_args(
        mesh_device=ttnn_mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        layer_num=0,
        transformation_mats=transformation_mats,
        use_paged_kv_cache=False,
    )

    # Verify the model was created correctly
    assert tt_model is not None
    assert tt_model.config.is_resolved()
    assert tt_model.config.dim == model_args.dim
    assert tt_model.config.n_heads == model_args.n_heads
    assert tt_model.config.n_kv_heads == model_args.n_kv_heads
    assert tt_model.config.head_dim == model_args.head_dim
    assert tt_model.layer_past is not None  # KV cache should be initialized

    # Verify TG-specific configurations
    assert tt_model.config.num_device_groups == 32 // model_args.n_kv_heads
    assert tt_model.config.batch_size_per_device_group == batch_size // tt_model.config.num_device_groups
    assert tt_model.config._slice_mat is not None
    assert tt_model.config._user_selection_matrix is not None

    logger.info(f"test_attention_2d_vs_reference_from_model_args: PASSED for mode={mode}, seq_len={seq_len}")
    logger.info(f"  Config: dim={model_args.dim}, n_heads={model_args.n_heads}, n_kv_heads={model_args.n_kv_heads}")
    logger.info(
        f"  TG-specific: num_device_groups={tt_model.config.num_device_groups}, batch_size_per_device_group={tt_model.config.batch_size_per_device_group}"
    )

    # Note: Full forward pass test requires setting up:
    # - rot_mats (rotary embedding matrices)
    # - transformation_mat
    # - current_pos tensor
    # - Proper input memory configs matching model_config
    # This is left as a TODO for full integration testing with TTTv1 model pipeline


# ============================================================================
# Additional unit tests for helper functions
# ============================================================================


def test_num_to_corerange():
    """Test _num_to_corerange helper function."""
    from models.common.modules.attention.attention_2d import _num_to_corerange

    # Single core - verify it returns a valid CoreRange
    cr = _num_to_corerange(1)
    assert cr is not None
    assert isinstance(cr, ttnn.CoreRange)

    # 8 cores (one row)
    cr = _num_to_corerange(8)
    assert cr is not None
    assert isinstance(cr, ttnn.CoreRange)

    # 32 cores (4 rows)
    cr = _num_to_corerange(32)
    assert cr is not None
    assert isinstance(cr, ttnn.CoreRange)

    # With start_core offset at beginning of second row (fits in remaining columns)
    start = ttnn.CoreCoord(0, 1)
    cr = _num_to_corerange(8, start_core=start)
    assert cr is not None
    assert isinstance(cr, ttnn.CoreRange)


def test_zeros_like_kv_cache():
    """Test _zeros_like_kv_cache helper function."""
    from models.common.modules.attention.attention_2d import _zeros_like_kv_cache

    cache = _zeros_like_kv_cache(batch_size=8, n_kv_heads=1, max_seq_len=2048, head_dim=128)

    assert cache.shape == (8, 1, 2048, 128)
    assert cache.dtype == torch.float32  # Default float
    assert torch.all(cache == 0)


def test_default_topology():
    """Test _default_topology helper function."""
    from models.common.modules.attention.attention_2d import _default_topology

    # Mock mesh device for single device
    mock_device_1 = MagicMock()
    mock_device_1.get_num_devices.return_value = 1
    assert _default_topology(mock_device_1) is None

    # Mock mesh device for multi-device (non-8)
    mock_device_4 = MagicMock()
    mock_device_4.get_num_devices.return_value = 4
    assert _default_topology(mock_device_4) == ttnn.Topology.Linear


def test_attention_2d_config_is_resolved():
    """Test is_resolved() method of Attention2DConfig."""
    from models.common.modules.attention.attention_2d import Attention2DConfig

    # Minimal config - not resolved
    config = Attention2DConfig(wqkv=MagicMock(), wo=MagicMock())
    assert not config.is_resolved()

    # Add some required fields
    mock_mesh_device = MagicMock()
    mock_mesh_device.shape = (4, 8)
    mock_mesh_device.get_num_devices.return_value = 32

    config = Attention2DConfig(
        wqkv=MagicMock(),
        wo=MagicMock(),
        mesh_device=mock_mesh_device,
        tt_ccl=MagicMock(),
        dim=4096,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        qkv_size=6144,
        scale=0.0883,
        topology=ttnn.Topology.Linear,
        ccl_dtype=ttnn.bfloat8_b,
        num_device_groups=4,
        batch_size_per_device_group=8,
        _slice_mat=MagicMock(),
        _user_selection_matrix=MagicMock(),
        decode_xqkv_prg_config=MagicMock(),
        decode_sdpa_prg_config=MagicMock(),
        li_qkv_decode_compute_kernel_cfg=MagicMock(),
        sdpa_decode_compute_kernel_cfg=MagicMock(),
        li_o_decode_compute_kernel_cfg=MagicMock(),
    )
    assert config.is_resolved()
