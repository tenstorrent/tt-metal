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

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

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


# ============================================================================
# Integration Tests - Require TG device
# ============================================================================


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(4, 8), (8, 4)],  # Galaxy shapes only (32 devices)
    ids=["4x8", "8x4"],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (512, 32))
def test_mlp_2d_vs_reference(ttnn_mesh_device: ttnn.MeshDevice, seq_len):
    """
    Test that MLP2D class matches the HuggingFace/Meta reference model.

    Runs only on Galaxy (TG) devices due to Galaxy-specific CCL operations.
    """
    from models.common.modules.mlp.mlp_2d import MLP2D
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


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1), (1, 2), (1, 8), (2, 4)],  # Non-Galaxy shapes - should be rejected
    ids=["1x1", "1x2", "1x8", "2x4"],
    indirect=True,
)
def test_mlp_2d_rejects_non_galaxy(ttnn_mesh_device: ttnn.MeshDevice):
    """
    Test that MLP2D.from_model_args() raises ValueError for non-Galaxy devices.

    MLP2D requires Galaxy topology (4x8 or 8x4) due to Galaxy-specific CCL operations.
    """
    from models.common.modules.mlp.mlp_2d import MLP2D
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    model_args = ModelArgs(ttnn_mesh_device, max_batch_size=1, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1

    # Non-Galaxy shapes should always be rejected
    state_dict = model_args.load_state_dict()
    tt_ccl = TT_CCL(ttnn_mesh_device)

    with pytest.raises(ValueError, match="MLP2D requires Galaxy topology"):
        MLP2D.from_model_args(
            mesh_device=ttnn_mesh_device,
            tt_ccl=tt_ccl,
            args=model_args,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
            layer_num=0,
        )

    logger.info("MLP2D correctly rejects non-Galaxy devices")


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(4, 8), (8, 4)],  # Galaxy shapes only (32 devices)
    ids=["4x8", "8x4"],
    indirect=True,
)
def test_mlp_2d_config_attributes(ttnn_mesh_device: ttnn.MeshDevice):
    """
    Test that MLP2D created via from_model_args has correct config attributes.

    Runs only on Galaxy (TG) devices due to Galaxy-specific CCL operations.
    """
    from models.common.modules.mlp.mlp_2d import MLP2D
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    model_args = ModelArgs(ttnn_mesh_device, max_batch_size=1, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1

    state_dict = model_args.load_state_dict()
    tt_ccl = TT_CCL(ttnn_mesh_device)

    mlp = MLP2D.from_model_args(
        mesh_device=ttnn_mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        layer_num=0,
    )

    # Verify config values match model_args
    assert mlp.config.dim == model_args.dim
    assert mlp.config.hidden_dim == model_args.hidden_dim
    assert mlp.config.cluster_shape == model_args.cluster_shape
    assert mlp.config.num_devices == model_args.num_devices
    assert mlp.config.prefill_len_cutoff == model_args.prefill_len_cutoff

    # Verify weights were loaded
    assert mlp.w1 is not None
    assert mlp.w2 is not None
    assert mlp.w3 is not None

    # Verify 2D shard dims
    assert mlp.config.w1_shard_dims == (-1, -2)
    assert mlp.config.w2_shard_dims == (-2, -1)

    logger.info("MLP2D config attributes test PASSED!")


# ============================================================================
# Run unit tests standalone
# ============================================================================

if __name__ == "__main__":
    print("Running MLP2D unit tests (no device required)...")

    test_mlp_2d_config_creation()
    print("  ✓ test_mlp_2d_config_creation")

    test_mlp_2d_config_rejects_1d_mesh()
    print("  ✓ test_mlp_2d_config_rejects_1d_mesh")

    test_mlp_2d_configs_methods()
    print("  ✓ test_mlp_2d_configs_methods")

    test_mlp_2d_optimization_config()
    print("  ✓ test_mlp_2d_optimization_config")

    print("\nAll MLP2D unit tests passed! ✓")
    print("\nTo run device tests (requires 2D mesh), use pytest:")
    print("  pytest models/common/tests/modules/mlp/test_mlp_2d.py -v")
