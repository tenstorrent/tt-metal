# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the MLP2D module (TG/Galaxy 2D mesh topology).

This test suite verifies:
1. Unit tests for config dataclasses (no device needed)
2. MLP2D class matches HuggingFace/Meta reference model
3. MLP2D correctly rejects non-TG devices
4. Backward compatibility: MLP2D.from_model_args() works correctly
"""

from unittest.mock import MagicMock

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.mlp.mlp_2d import MLP2D, MLP2DConfig, _resolve_mlp2d_config
from models.common.utility_functions import comp_allclose, comp_pcc

# ============================================================================
# Unit Tests - No device required
# ============================================================================


def create_mock_lazy_weight(device=None, shape=None):
    w = MagicMock(spec=LazyWeight)
    w.device = device
    w.source = MagicMock()
    if shape:
        w.source.shape = shape
    return w


def test_mlp_2d_config_creation():
    """Test that MLP2DConfig dataclass can be created with explicit values.

    Note: _resolve_mlp2d_config is tested via integration tests (test_mlp_2d_vs_reference)
    since it requires real devices and tt_ccl. This test only verifies dataclass creation.
    """

    # Mock device
    mock_device = MagicMock(spec=ttnn.MeshDevice)
    mock_device.shape = (4, 8)
    mock_device.get_num_devices.return_value = 32
    mock_device.dram_grid_size.return_value = ttnn.CoreCoord(12, 1)

    # Mock tt_ccl (required for unit tests since we can't create real semaphores)
    mock_tt_ccl = MagicMock()

    # Mock weights
    w1 = create_mock_lazy_weight(device=mock_device, shape=(8192, 28672))
    w2 = create_mock_lazy_weight(device=mock_device, shape=(28672, 8192))
    w3 = create_mock_lazy_weight(device=mock_device, shape=(8192, 28672))

    # Create config with explicit values (like MLP1D unit test pattern)
    config = MLP2DConfig(
        w1=w1,
        w2=w2,
        w3=w3,
        mesh_device=mock_device,
        tt_ccl=mock_tt_ccl,
        dim=8192,
        hidden_dim=28672,
        max_batch_size=32,
    )

    # Verify explicit values are preserved
    assert config.w1 is w1
    assert config.w2 is w2
    assert config.w3 is w3
    assert config.mesh_device is mock_device
    assert config.tt_ccl is mock_tt_ccl
    assert config.dim == 8192
    assert config.hidden_dim == 28672
    assert config.max_batch_size == 32

    # Verify defaults for optional fields
    assert config.w1_w3_dtype is None  # Will be resolved to bfloat8_b
    assert config.topology is None  # Will be auto-detected


def test_mlp_2d_config_rejects_1d_mesh():
    """Test that MLP2DConfig raises assertion error for 1D mesh (requires 2D mesh)."""

    # Mock 1D device
    mock_device_1d = MagicMock(spec=ttnn.MeshDevice)
    mock_device_1d.shape = (1, 8)

    w1 = create_mock_lazy_weight(device=mock_device_1d, shape=(4096, 14336))
    w2 = create_mock_lazy_weight(device=mock_device_1d, shape=(14336, 4096))
    w3 = create_mock_lazy_weight(device=mock_device_1d, shape=(4096, 14336))

    config = MLP2DConfig(w1=w1, w2=w2, w3=w3)

    with pytest.raises(AssertionError, match="MLP2D requires 2D mesh"):
        _resolve_mlp2d_config(config)


def test_mlp_2d_optimization_config():
    """Test MLP2D optimization settings can be explicitly set.

    Note: _resolve_mlp2d_config is tested via integration tests. This test only
    verifies that optimization config fields can be explicitly set on the dataclass.
    """

    mock_device = MagicMock(spec=ttnn.MeshDevice)
    mock_device.shape = (4, 8)
    mock_device.get_num_devices.return_value = 32

    mock_tt_ccl = MagicMock()

    w1 = create_mock_lazy_weight(device=mock_device, shape=(8192, 28672))
    w2 = create_mock_lazy_weight(device=mock_device, shape=(28672, 8192))
    w3 = create_mock_lazy_weight(device=mock_device, shape=(8192, 28672))

    # Create config with explicit dtype overrides
    config = MLP2DConfig(
        w1=w1,
        w2=w2,
        w3=w3,
        mesh_device=mock_device,
        tt_ccl=mock_tt_ccl,
        dim=8192,
        hidden_dim=28672,
        w1_w3_dtype=ttnn.bfloat16,
        activation_dtype=ttnn.bfloat16,
    )

    # Verify explicit values are preserved
    assert config.w1_w3_dtype == ttnn.bfloat16
    assert config.activation_dtype == ttnn.bfloat16
    assert config.w2_dtype is None  # Will be resolved to bfloat8_b default


@pytest.mark.parametrize(
    "cluster_shape",
    [(1, 1), (1, 2), (1, 8), (2, 4)],  # Non-Galaxy shapes - should be rejected by from_model_args
    ids=["1x1", "1x2", "1x8", "2x4"],
)
def test_mlp_2d_rejects_non_galaxy_from_model_args(cluster_shape):
    """
    Test that MLP2D.from_model_args() raises ValueError for non-Galaxy devices.
    """

    class _DummyArgs:
        def __init__(self, cluster_shape):
            self.cluster_shape = list(cluster_shape)

    model_args = _DummyArgs(cluster_shape)

    with pytest.raises(ValueError, match="MLP2D requires Galaxy topology"):
        MLP2D.from_model_args(
            mesh_device=None,
            tt_ccl=None,
            args=model_args,
            state_dict=None,
            weight_cache_path=None,
            layer_num=0,
        )


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

    This fixture runs a minimal topology check and returns the result.
    Other tests can use this to decide whether to apply workarounds.

    Note: scope="function" because ttnn_mesh_device may vary per test parametrization.
    The check is fast so the overhead is minimal.

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


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(8, 4)],
    ids=["8x4"],
    indirect=True,
)
@pytest.mark.xfail(
    reason="TTNN bug: ttnn.linear produces invalid topology where both mesh axes shard the same dimension. "
    "See test docstring for details. Remove xfail once TTNN issue is fixed.",
    strict=True,  # Fail if the bug is accidentally fixed (so we know to update)
)
def test_ttnn_linear_2d_mesh_topology_bug(ttnn_linear_2d_mesh_has_topology_bug: bool):
    """
    Document the TTNN bug where ttnn.linear produces incorrect topology metadata
    for 2D mesh matmul operations.

    Setup (in fixture):
        - Input x: shape [1, 1, 32, 4096], topology [Replicated, Shard(3)]
        - Weight w: shape [4096, 14336], topology [Shard(-1), Shard(-2)]

    Expected output topology after x @ w:
        - [Shard(3), PartialSum] or similar

    Actual (buggy) output topology:
        - [Shard(-1), Shard(3)] - both axes claim to shard the same dimension!

    TODO: File TTNN issue and remove xfail once fixed.
    """
    if ttnn_linear_2d_mesh_has_topology_bug:
        pytest.fail(
            "ttnn.linear produces invalid topology: both mesh axes shard the same dimension. "
            "Expected different dimensions or [Shard, PartialSum/Replicate]."
        )


# [INFO] currently tt_transformers is not testing 2D mesh MLP in CI -- existing TG tests are DP only that runs 1D MLPs in parallel
# todo)) add more targeted unit tests like the ones in test_mlp_1d.py when relevant model are implemented
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [
        (4, 8),
        (8, 4),
    ],
    ids=[
        "4x8",
        "8x4",
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "dtype,batch_size,dim,hidden_dim,hf_model_name",
    [
        pytest.param(
            ttnn.bfloat8_b,
            1,
            4096,
            14336,
            "meta-llama/Llama-3.1-8B-Instruct",
            id="bf8b-bs1-default-hf",
        ),
    ],
)
@pytest.mark.parametrize(
    "seq_len,mode",
    [
        (512, "prefill"),
        (32, "decode"),
    ],
    ids=[
        "prefill-512",
        "decode-32",
    ],
)
def test_mlp_2d_vs_reference(
    ttnn_mesh_device: ttnn.MeshDevice,
    ttnn_linear_2d_mesh_has_topology_bug: bool,
    seq_len,
    mode,
    dtype,
    batch_size,
    dim,
    hidden_dim,
    hf_model_name,
):
    """
    Test MLP2D constructed via direct APIs (MLP2DConfig) matches HF reference MLP.
    """

    seed = 1234
    torch.manual_seed(seed)

    # Load HF config and create model with dummy weights
    config = AutoConfig.from_pretrained(hf_model_name)
    config.num_hidden_layers = 1
    with no_init_weights():
        hf_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    reference_mlp = hf_model.model.layers[0].mlp

    # Initialize only the MLP submodule deterministically.
    with torch.no_grad():
        for param in reference_mlp.parameters():
            param.copy_(torch.randn_like(param))

    assert dim == config.hidden_size
    assert hidden_dim == config.intermediate_size
    cluster_shape = list(ttnn_mesh_device.shape)

    # TT expects weights in (input_dim, output_dim) layout.
    w1_torch = reference_mlp.gate_proj.weight.T  # (dim, hidden_dim)
    w3_torch = reference_mlp.up_proj.weight.T  # (dim, hidden_dim)
    w2_torch = reference_mlp.down_proj.weight.T  # (hidden_dim, dim)
    # [INFO] PyTorch's nn.Linear operates on the last dimension regardless of tensor rank.
    torch_input = torch.randn(batch_size, 1, seq_len, dim, dtype=torch.bfloat16)

    # Create LazyWeights
    ttnn.SetDefaultDevice(ttnn_mesh_device)
    lazy_w1 = LazyWeight(source=w1_torch, dtype=dtype)
    lazy_w2 = LazyWeight(source=w2_torch, dtype=dtype)
    lazy_w3 = LazyWeight(source=w3_torch, dtype=dtype)

    # Create MLP2D directly with weights
    tt_model = MLP2D(lazy_w1, lazy_w2, lazy_w3)

    # Run HF reference MLP
    with torch.no_grad():
        reference_output = reference_mlp(torch_input)

    # Run TT model
    # [INFO] we use LazyWeight on input for the benefit of faster testing (cached input); in production, the input is already a ttnn tensor.
    tt_input = LazyWeight(source=torch_input, dtype=ttnn.bfloat8_b)
    tt_output = tt_model.forward(tt_input, mode)
    ttnn.SetDefaultDevice(None)

    # WORKAROUND: ttnn.linear produces incorrect topology metadata for 2D mesh matmul.
    # The output topology shows [Shard(-1), Shard(3)] but the correct data layout after
    # the final all-reduce on axis 0 is [Replicated, Shard(3)]:
    # expected: [ttnn.PlacementReplicate, ttnn.PlacementShard(3)]
    # got: [ttnn.PlacementShard(-1), ttnn.PlacementShard(3)]
    #   - Axis 0 (size 8): Replicated (all-reduced/gathered)
    #   - Axis 1 (size 4): Sharded on dim 3
    # The fixture `ttnn_linear_2d_mesh_has_topology_bug` checks this once per module.
    if ttnn_linear_2d_mesh_has_topology_bug:
        # Bug present: use explicit mesh_composer with correct topology
        expected_composer_cfg = ttnn.MeshComposerConfig(
            dims=[0, 3],  # axis 0: replicated (dim ignored), axis 1: shard on dim 3
            mesh_shape_override=ttnn.MeshShape([1, cluster_shape[1]]),  # [1, 4]: skip axis 0, concat axis 1
        )
        mesh_composer = ttnn.create_mesh_composer(ttnn_mesh_device, expected_composer_cfg)
        tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)
    else:
        raise RuntimeError("Bug fixed: use auto_compose -- tt_output_torch = to_torch_auto_compose(tt_output)")

    # Compare
    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"MLP2D (direct API) vs HF reference: {pcc_message}")

    assert passing, f"MLP2D output does not meet PCC requirement {pcc_required}: {pcc_message}."
    logger.info(f"MLP2D (direct API) vs HF reference: PASSED for mode={mode}, seq_len={seq_len}")


# [INFO] this test will retire once models/tt_transformers/tt/model_config.py retires
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(8, 4)],
    ids=["8x4"],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (512, 32))
def test_mlp_2d_vs_reference_from_model_args(ttnn_mesh_device: ttnn.MeshDevice, seq_len):
    """
    Test that MLP2D class matches the HuggingFace/Meta reference model.

    Runs only on Galaxy (TG) devices due to Galaxy-specific CCL operations.
    """

    from models.tt_transformers.tests.test_utils import get_ref_model_dype
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    batch_size = 1
    mode = "decode" if seq_len <= 32 else "prefill"

    model_args = ModelArgs(ttnn_mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    model_config = model_args.get_model_config()

    # Load reference model
    first_layer_prefix = model_args.get_state_dict_prefix("MLP", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
    }
    reference_model = model_args.reference_mlp()
    reference_model.load_state_dict(partial_state_dict)

    # Create MLP2D
    tt_ccl = TT_CCL(ttnn_mesh_device)
    tt_model = MLP2D.from_model_args(
        mesh_device=ttnn_mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        layer_num=0,
    )

    # Create input
    torch_input = torch.randn(
        1, 1, seq_len, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
    )

    # Run reference
    reference_output = reference_model(torch_input)

    # Run TT model - use model_config for proper input sharding
    input_mem_config = model_config["MLP_ACT_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

    tt_input = ttnn.from_torch(
        torch_input,
        device=ttnn_mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, 3), mesh_shape=model_args.cluster_shape),
        dtype=ttnn.bfloat8_b,
        memory_config=input_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = tt_model.forward(tt_input, mode)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(ttnn_mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    tt_output_torch = tt_output_torch[:, :1, :, :]

    # Compare
    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"MLP2D vs reference: {pcc_message}")

    assert passing, f"MLP2D output does not meet PCC requirement {pcc_required}: {pcc_message}."
    logger.info(f"MLP2D vs reference: PASSED for mode={mode}, seq_len={seq_len}")
