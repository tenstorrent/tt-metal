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

from functools import cached_property

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.mlp.mlp_2d import MLP2D, MeshContext2D, MLP2DConfig
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.ccl import TT_CCL

# ============================================================================
# Unit Tests - No device required
# ============================================================================


def test_mlp_2d_config_creation():
    """Test that MLP2DConfig can be created with expected values and nested configs."""
    from dataclasses import dataclass

    from models.common.modules.mlp.mlp_2d import (
        MeshContext2D,
        MLP2DConfig,
        MLP2DDecodeConfigs,
        MLP2DOptimizationConfig,
        MLP2DPrefillConfigs,
    )

    # Create a mock MeshContext2D for testing (no real device needed)
    @dataclass
    class MockMeshContext2D(MeshContext2D):
        """Mock MeshContext2D for unit tests without real devices."""

        mesh_device: object = None
        tt_ccl: object = None
        _num_devices: int = 32
        _cluster_shape: list = None

        def __post_init__(self):
            if self._cluster_shape is None:
                self._cluster_shape = [4, 8]  # 2D mesh for TG

        def num_devices(self) -> int:
            return self._num_devices

        def cluster_shape(self) -> list:
            return self._cluster_shape

        def dram_grid_size(self):
            # Return a mock CoreCoord
            class MockCoreCoord:
                x = 12
                y = 1

            return MockCoreCoord()

    mock_ctx = MockMeshContext2D(_num_devices=32, _cluster_shape=[4, 8])

    config = MLP2DConfig(
        dim=8192,
        hidden_dim=28672,
        mesh_ctx=mock_ctx,
    )

    assert config.dim == 8192
    assert config.hidden_dim == 28672
    assert config.cluster_shape == [4, 8]
    assert config.num_devices == 32

    # Nested configs should be auto-created via cached_property
    assert isinstance(config.decode_config, MLP2DDecodeConfigs)
    assert isinstance(config.prefill_config, MLP2DPrefillConfigs)
    assert isinstance(config.optimization_config, MLP2DOptimizationConfig)

    # Each sub-config should have a reference back to the parent
    assert config.decode_config.cfg is config
    assert config.prefill_config.cfg is config
    assert config.optimization_config.cfg is config

    # Sub-configs should have callable methods
    assert callable(config.decode_config.ff1_3_prg_config)
    assert callable(config.prefill_config.w1_w3_prg_config)
    assert callable(config.optimization_config.ff1_3_dtype)


def test_mlp_2d_config_rejects_1d_mesh():
    """Test that MLP2DConfig raises assertion error for 1D mesh (requires 2D mesh)."""
    from dataclasses import dataclass

    from models.common.modules.mlp.mlp_2d import MeshContext2D, MLP2DConfig

    @dataclass
    class MockMeshContext(MeshContext2D):
        """Mock MeshContext for testing rejection."""

        mesh_device: object = None
        tt_ccl: object = None
        _num_devices: int = 8
        _cluster_shape: list = None

        def __post_init__(self):
            if self._cluster_shape is None:
                self._cluster_shape = [1, 8]

        def num_devices(self) -> int:
            return self._num_devices

        def cluster_shape(self) -> list:
            return self._cluster_shape

    # Test 1D mesh rejection (cluster_shape[0] == 1)
    mock_ctx_1d = MockMeshContext(_num_devices=8, _cluster_shape=[1, 8])
    with pytest.raises(AssertionError, match="MLP2D requires 2D mesh"):
        MLP2DConfig(
            dim=4096,
            hidden_dim=14336,
            mesh_ctx=mock_ctx_1d,
        )

    # Test 1D mesh rejection (cluster_shape[1] == 1)
    mock_ctx_1d_alt = MockMeshContext(_num_devices=8, _cluster_shape=[8, 1])
    with pytest.raises(AssertionError, match="MLP2D requires 2D mesh"):
        MLP2DConfig(
            dim=4096,
            hidden_dim=14336,
            mesh_ctx=mock_ctx_1d_alt,
        )

    # 2x4 should be accepted by MLP2DConfig (it's a valid 2D mesh)
    # Note: from_model_args() would reject it because model_config.py is Galaxy-specific
    mock_ctx_2x4 = MockMeshContext(_num_devices=8, _cluster_shape=[2, 4])
    config = MLP2DConfig(
        dim=4096,
        hidden_dim=14336,
        mesh_ctx=mock_ctx_2x4,
    )
    assert config.cluster_shape == [2, 4]


def test_mlp_2d_configs_methods():
    """Test that MLP2DDecodeConfigs and MLP2DPrefillConfigs methods work."""
    from dataclasses import dataclass

    from models.common.modules.mlp.mlp_2d import MeshContext2D, MLP2DConfig

    @dataclass
    class MockMeshContext2D(MeshContext2D):
        mesh_device: object = None
        tt_ccl: object = None
        _num_devices: int = 32
        _cluster_shape: list = None

        def __post_init__(self):
            if self._cluster_shape is None:
                self._cluster_shape = [4, 8]

        def num_devices(self) -> int:
            return self._num_devices

        def cluster_shape(self) -> list:
            return self._cluster_shape

        def dram_grid_size(self):
            class MockCoreCoord:
                x = 12
                y = 1

            return MockCoreCoord()

    mock_ctx = MockMeshContext2D(_num_devices=32, _cluster_shape=[4, 8])

    config = MLP2DConfig(
        dim=8192,
        hidden_dim=28672,
        mesh_ctx=mock_ctx,
    )

    # Decode should have TG-specific methods
    assert callable(config.decode_config.ff1_3_prg_config)
    assert callable(config.decode_config.ff2_prg_config)
    assert callable(config.decode_config.ff1_out_reduce_scatter_memcfg)
    assert callable(config.decode_config.ff1_out_gathered_memcfg)
    assert callable(config.decode_config.ff2_out_reduce_scatter_memcfg)
    assert callable(config.decode_config.sharded_attn_input_memcfg)

    # Prefill should have prefill-specific methods (that take seq_len)
    assert callable(config.prefill_config.w1_w3_prg_config)
    assert callable(config.prefill_config.w2_prg_config)

    # Test that prefill methods can be called and return values
    prefill_w1_w3 = config.prefill_config.w1_w3_prg_config(seq_len=512)
    assert prefill_w1_w3 is not None


def test_mlp_2d_optimization_config():
    """Test MLP2DOptimizationConfig default values."""
    from dataclasses import dataclass

    from models.common.modules.mlp.mlp_2d import MeshContext2D, MLP2DConfig

    @dataclass
    class MockMeshContext2D(MeshContext2D):
        mesh_device: object = None
        tt_ccl: object = None

        def num_devices(self) -> int:
            return 32

        def cluster_shape(self) -> list:
            return [4, 8]

        def dram_grid_size(self):
            class MockCoreCoord:
                x = 12
                y = 1

            return MockCoreCoord()

    mock_ctx = MockMeshContext2D()

    config = MLP2DConfig(
        dim=8192,
        hidden_dim=28672,
        mesh_ctx=mock_ctx,
    )

    opt = config.optimization_config
    assert opt.ff1_3_dtype() == ttnn.bfloat8_b
    assert opt.ff2_dtype() == ttnn.bfloat8_b
    assert opt.activation_dtype() is None
    assert opt.li_ff1_3_compute_kernel_cfg() is not None
    assert opt.li_ff2_compute_kernel_cfg() is not None


@pytest.mark.parametrize(
    "cluster_shape",
    [(1, 1), (1, 2), (1, 8), (2, 4)],  # Non-Galaxy shapes - should be rejected
    ids=["1x1", "1x2", "1x8", "2x4"],
)
def test_mlp_2d_rejects_non_galaxy(cluster_shape):
    """
    Test that MLP2D.from_model_args() raises ValueError for non-Galaxy devices.

    MLP2D requires Galaxy topology (4x8 or 8x4) due to Galaxy-specific CCL operations.
    """
    from models.common.modules.mlp.mlp_2d import MLP2D

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

    logger.info("MLP2D correctly rejects non-Galaxy devices")


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

    This test uses MLP2DConfig directly instead of from_model_args() to verify
    the low-level API works correctly. Loads HF model config and uses dummy weights.
    """

    seed = 1234
    torch.manual_seed(seed)

    # Load HF config and create model with dummy weights
    config = AutoConfig.from_pretrained(hf_model_name)
    config.num_hidden_layers = 1  # Only need 1 layer for MLP test
    with no_init_weights():
        hf_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    reference_mlp = hf_model.model.layers[0].mlp

    # Initialize only the MLP submodule deterministically.
    with torch.no_grad():
        for param in reference_mlp.parameters():
            param.copy_(torch.randn_like(param))

    # Extract dimensions from HF config
    # [INFO] we included dim and hidden_dim as parameters to increase visibility of them
    assert dim == config.hidden_size
    assert hidden_dim == config.intermediate_size
    cluster_shape = list(ttnn_mesh_device.shape)

    # TT expects weights in (input_dim, output_dim) layout.
    w1_torch = reference_mlp.gate_proj.weight.T.contiguous()  # (dim, hidden_dim)
    w3_torch = reference_mlp.up_proj.weight.T.contiguous()  # (dim, hidden_dim)
    w2_torch = reference_mlp.down_proj.weight.T.contiguous()  # (hidden_dim, dim)
    # [INFO] PyTorch's nn.Linear operates on the last dimension regardless of tensor rank.
    torch_input = torch.randn(batch_size, 1, seq_len, dim, dtype=torch.bfloat16)

    # Create TT_CCL and MeshContext2D directly
    mesh_ctx = MeshContext2D(mesh_device=ttnn_mesh_device, tt_ccl=TT_CCL(ttnn_mesh_device))

    # Create LazyWeight instances
    def make_lazy_weight(tensor: torch.Tensor, shard_dims: tuple[int, int]) -> LazyWeight:
        return LazyWeight(
            source=tensor,
            dtype=dtype,
            device=ttnn_mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=shard_dims, mesh_shape=cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Create MLP2DConfig subclass with lazy weights
    class TestMLP2DConfig(MLP2DConfig):
        @property
        def lazy_w1(self) -> LazyWeight:
            return make_lazy_weight(w1_torch, self.w1_shard_dims)

        @cached_property
        def lazy_w2(self) -> LazyWeight:
            return make_lazy_weight(w2_torch, self.w2_shard_dims)

        @cached_property
        def lazy_w3(self) -> LazyWeight:
            return make_lazy_weight(w3_torch, self.w1_shard_dims)

    mlp_config = TestMLP2DConfig(
        dim=dim,
        hidden_dim=hidden_dim,
        mesh_ctx=mesh_ctx,
        max_batch_size=batch_size,
    )

    mlp_config.lazy_w1 = make_lazy_weight(w1_torch, mlp_config.w1_shard_dims)
    mlp_config.lazy_w2 = make_lazy_weight(w2_torch, mlp_config.w2_shard_dims)
    mlp_config.lazy_w3 = make_lazy_weight(w3_torch, mlp_config.w1_shard_dims)

    # Create MLP2D directly with config
    tt_model = MLP2D(mlp_config)

    # Run HF reference MLP
    with torch.no_grad():
        reference_output = reference_mlp(torch_input)

    input_mem_config = ttnn.L1_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

    # Run TT model
    tt_input = ttnn.from_torch(
        torch_input,
        device=ttnn_mesh_device,
        # NOTE: keep explicit positive dim index here; negative dims can confuse MeshToTensor composition
        # and produce replicated outputs with an extra mesh-axis dimension.
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, 3), mesh_shape=cluster_shape),
        dtype=ttnn.bfloat8_b,
        memory_config=input_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = tt_model.forward(tt_input, mode)

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


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [
        (8, 4),
    ],
    ids=["8x4"],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (512, 32))
def test_mlp_2d_dim_ge_8192_paths(
    ttnn_mesh_device: ttnn.MeshDevice, ttnn_linear_2d_mesh_has_topology_bug: bool, seq_len
):
    """
    Exercise the dim >= 8192 TG execution paths (reduce_scatter + all_gather, composite all-reduce).

    We intentionally avoid a HF reference model here (70B-size dims/weights would be too heavy),
    and instead compare against a pure torch reference MLP:
        y = (SiLU(x @ w1) * (x @ w3)) @ w2
    """
    from functools import cached_property

    from models.common.modules.lazy_weight import LazyWeight
    from models.common.modules.mlp.mlp_2d import MeshContext2D, MLP2DConfig
    from models.tt_transformers.tt.ccl import TT_CCL

    # Force dim >= 8192 branch; keep hidden_dim modest to keep test memory/time reasonable.
    dim = 8192
    hidden_dim = 4096
    assert dim % 32 == 0 and hidden_dim % 32 == 0

    dtype = ttnn.bfloat8_b
    batch_size = 1
    mode = "decode" if seq_len <= 32 else "prefill"
    cluster_shape = list(ttnn_mesh_device.shape)

    # Deterministic weights + input
    torch.manual_seed(1234)
    w1_torch = torch.randn(dim, hidden_dim, dtype=torch.bfloat16)
    w3_torch = torch.randn(dim, hidden_dim, dtype=torch.bfloat16)
    w2_torch = torch.randn(hidden_dim, dim, dtype=torch.bfloat16)
    torch_input = torch.randn(1, 1, seq_len, dim, dtype=torch.bfloat16)

    # Torch reference
    with torch.no_grad():
        ref_w1 = torch.matmul(torch_input, w1_torch)
        ref_w3 = torch.matmul(torch_input, w3_torch)
        ref_hidden = torch.nn.functional.silu(ref_w1) * ref_w3
        reference_output = torch.matmul(ref_hidden, w2_torch)

    # TT model setup
    tt_ccl = TT_CCL(ttnn_mesh_device)
    mesh_ctx = MeshContext2D(mesh_device=ttnn_mesh_device, tt_ccl=tt_ccl)

    w1_shard_dims = (-1, -2)
    w2_shard_dims = (-2, -1)

    def make_lazy_weight(tensor: torch.Tensor, shard_dims: tuple[int, int]) -> LazyWeight:
        return LazyWeight(
            source=tensor,
            dtype=dtype,
            device=ttnn_mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=shard_dims, mesh_shape=cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    lazy_w1 = make_lazy_weight(w1_torch, w1_shard_dims)
    lazy_w2 = make_lazy_weight(w2_torch, w2_shard_dims)
    lazy_w3 = make_lazy_weight(w3_torch, w1_shard_dims)

    class TestMLP2DConfig(MLP2DConfig):
        @cached_property
        def lazy_w1(self) -> LazyWeight:
            return lazy_w1

        @cached_property
        def lazy_w2(self) -> LazyWeight:
            return lazy_w2

        @cached_property
        def lazy_w3(self) -> LazyWeight:
            return lazy_w3

    mlp_config = TestMLP2DConfig(
        dim=dim,
        hidden_dim=hidden_dim,
        mesh_ctx=mesh_ctx,
        max_batch_size=batch_size,
    )
    tt_model = MLP2D(mlp_config)

    tt_input = ttnn.from_torch(
        torch_input,
        device=ttnn_mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, 3), mesh_shape=cluster_shape),
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = tt_model.forward(tt_input, mode)

    # WORKAROUND: ttnn.linear produces incorrect topology metadata for 2D mesh matmul.
    # See comment in test_mlp_2d_vs_reference for details.
    # The fixture `ttnn_linear_2d_mesh_has_topology_bug` checks this once per module.
    if ttnn_linear_2d_mesh_has_topology_bug:
        # Bug present: use explicit mesh_composer with correct topology
        expected_composer_cfg = ttnn.MeshComposerConfig(
            dims=[0, 3],
            mesh_shape_override=ttnn.MeshShape([1, cluster_shape[1]]),
        )
        mesh_composer = ttnn.create_mesh_composer(ttnn_mesh_device, expected_composer_cfg)
        tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)
    else:
        raise RuntimeError("Bug fixed: use auto_compose: tt_output_torch = to_torch_auto_compose(tt_output)")

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"MLP2D dim>=8192 path vs torch reference: {pcc_message}")
    assert passing, f"MLP2D output does not meet PCC requirement {pcc_required}: {pcc_message}."


# ============================================================================
# Integration Tests - Require TG device
# ============================================================================


# todo)) retire this test once models/tt_transformers/tt/model_config.py retires
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [
        (8, 4),
    ],
    ids=[
        "8x4",
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (512, 32))
def test_mlp_2d_vs_reference_from_model_args(
    ttnn_mesh_device: ttnn.MeshDevice,
    seq_len,
    is_ci_env,
    is_ci_v2_env,
):
    """
    Test that MLP2D class matches the HuggingFace/Meta reference model.

    Runs only on Galaxy (TG) devices due to Galaxy-specific CCL operations.
    """
    if is_ci_env or is_ci_v2_env:
        pytest.skip("CI only runs unit tests")

    from models.tt_transformers.tests.test_utils import get_ref_model_dype
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    dtype = ttnn.bfloat8_b
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
        weight_cache_path=model_args.weight_cache_path(dtype),
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
