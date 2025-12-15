# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the MLP1D module (1D mesh topology: N150, N300, T3K).

This test suite verifies:
1. Unit tests for config dataclasses (no device needed)
2. MLP1D class matches HuggingFace/Meta reference model
3. MLP1D correctly rejects TG/Galaxy devices
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.utility_functions import comp_allclose, comp_pcc

# ============================================================================
# Unit Tests - No device required
# ============================================================================


def test_mlp_1d_config_creation():
    """Test that MLP1DConfig can be created with expected values and nested configs."""
    from dataclasses import dataclass

    from models.common.modules.mlp.mlp_1d import (
        MeshContext,
        MLP1DConfig,
        MLP1DDecodeConfigs,
        MLP1DOptimizationConfig,
        MLP1DPrefillConfigs,
    )

    # Create a mock MeshContext for testing (no real device needed)
    @dataclass
    class MockMeshContext(MeshContext):
        """Mock MeshContext for unit tests without real devices."""

        mesh_device: object = None
        tt_ccl: object = None
        _num_devices: int = 8
        _cluster_shape: list = None

        def __post_init__(self):
            if self._cluster_shape is None:
                self._cluster_shape = [1, self._num_devices]

        def num_devices(self) -> int:
            return self._num_devices

        def cluster_shape(self) -> list:
            return self._cluster_shape

    mock_ctx = MockMeshContext(_num_devices=8, _cluster_shape=[1, 8])

    config = MLP1DConfig(
        dim=4096,
        hidden_dim=14336,
        mesh_ctx=mock_ctx,
    )

    assert config.dim == 4096
    assert config.hidden_dim == 14336
    assert config.cluster_shape == [1, 8]
    assert config.num_devices == 8

    # Nested configs should be auto-created via cached_property with parent reference
    assert isinstance(config.decode_config, MLP1DDecodeConfigs)
    assert isinstance(config.prefill_config, MLP1DPrefillConfigs)
    assert isinstance(config.optimization_config, MLP1DOptimizationConfig)

    # Each sub-config should have a reference back to the parent
    assert config.decode_config.cfg is config
    assert config.prefill_config.cfg is config
    assert config.optimization_config.cfg is config

    # Sub-configs should have callable methods
    assert callable(config.decode_config.w1_w3_prg_config)
    assert callable(config.prefill_config.w1_w3_prg_config)
    assert callable(config.optimization_config.ff1_3_dtype)


def test_mlp_1d_config_rejects_2d_mesh():
    """Test that MLP1DConfig raises assertion error for 2D mesh."""
    from dataclasses import dataclass

    from models.common.modules.mlp.mlp_1d import MeshContext, MLP1DConfig

    @dataclass
    class MockMeshContext2D(MeshContext):
        """Mock 2D MeshContext for testing rejection."""

        mesh_device: object = None
        tt_ccl: object = None
        _num_devices: int = 32
        _cluster_shape: list = None

        def __post_init__(self):
            if self._cluster_shape is None:
                self._cluster_shape = [4, 8]  # 2D mesh - should be rejected

        def num_devices(self) -> int:
            return self._num_devices

        def cluster_shape(self) -> list:
            return self._cluster_shape

    mock_ctx = MockMeshContext2D(_num_devices=32, _cluster_shape=[4, 8])

    with pytest.raises(AssertionError, match="MLPNonTG only supports 1D meshes"):
        MLP1DConfig(
            dim=8192,
            hidden_dim=28672,
            mesh_ctx=mock_ctx,
        )


def test_mlp_1d_configs_methods():
    """Test that MLP1DDecodeConfigs and MLP1DPrefillConfigs methods work."""
    from dataclasses import dataclass

    from models.common.modules.mlp.mlp_1d import MeshContext, MLP1DConfig

    @dataclass
    class MockMeshContext(MeshContext):
        mesh_device: object = None
        tt_ccl: object = None
        _num_devices: int = 8
        _cluster_shape: list = None

        def __post_init__(self):
            if self._cluster_shape is None:
                self._cluster_shape = [1, self._num_devices]

        def num_devices(self) -> int:
            return self._num_devices

        def cluster_shape(self) -> list:
            return self._cluster_shape

    mock_ctx = MockMeshContext(_num_devices=8, _cluster_shape=[1, 8])

    config = MLP1DConfig(
        dim=4096,
        hidden_dim=14336,
        mesh_ctx=mock_ctx,
    )

    # Decode should have decode-specific methods
    assert callable(config.decode_config.w1_w3_prg_config)
    assert callable(config.decode_config.w2_prg_config)
    assert callable(config.decode_config.sharded_mlp2_input_memcfg)
    assert callable(config.decode_config.decode_residual_memcfg)

    # Prefill should have prefill-specific methods (that take seq_len)
    assert callable(config.prefill_config.w1_w3_prg_config)
    assert callable(config.prefill_config.w2_prg_config)

    # Neither should have TG-specific methods
    assert not hasattr(config.decode_config, "ff1_3_tg_progcfg")
    assert not hasattr(config.prefill_config, "ff1_3_tg_progcfg")

    # Test that methods can be called and return values
    decode_w1_w3 = config.decode_config.w1_w3_prg_config()
    assert decode_w1_w3 is not None

    prefill_w1_w3 = config.prefill_config.w1_w3_prg_config(seq_len=512)
    assert prefill_w1_w3 is not None


def test_mlp_1d_optimization_config():
    """Test MLP1DOptimizationConfig default values."""
    from dataclasses import dataclass

    from models.common.modules.mlp.mlp_1d import MeshContext, MLP1DConfig

    @dataclass
    class MockMeshContext(MeshContext):
        mesh_device: object = None
        tt_ccl: object = None

        def num_devices(self) -> int:
            return 8

        def cluster_shape(self) -> list:
            return [1, 8]

    mock_ctx = MockMeshContext()

    config = MLP1DConfig(
        dim=4096,
        hidden_dim=14336,
        mesh_ctx=mock_ctx,
    )

    opt = config.optimization_config
    assert opt.ff1_3_dtype() == ttnn.bfloat8_b
    assert opt.ff2_dtype() == ttnn.bfloat8_b
    assert opt.activation_dtype() is None
    assert opt.li_ff1_3_compute_kernel_cfg() is not None
    assert opt.li_ff2_compute_kernel_cfg() is not None


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [
        (1, 1),  # single device
        (1, 2),  # 1D mesh, 2 devices
        (1, 4),  # 1D mesh, 4 devices
        (1, 8),  # 1D mesh, 8 devices
    ],
    ids=[
        "1x1",
        "1x2",
        "1x4",
        "1x8",
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (512, 32))
def test_mlp_1d_vs_reference(ttnn_mesh_device: ttnn.MeshDevice, seq_len):
    """
    Test MLP1D constructed via direct APIs (MLP1DConfig) matches HF reference MLP.

    Uses HF_MODEL env (defaults to Llama 3.2 1B) and dummy weights.
    """
    import math
    import os
    from functools import cached_property

    from transformers import AutoConfig, AutoModelForCausalLM

    from models.common.modules.lazy_weight import LazyWeight
    from models.common.modules.mlp.mlp_1d import MLP1D, MeshContext, MLP1DConfig, MLP1DPrefillConfigs, _matmul_config
    from models.tt_transformers.tt.ccl import TT_CCL

    dtype = ttnn.bfloat8_b
    batch_size = 1
    mode = "decode" if seq_len <= 32 else "prefill"

    # HF model (default small) for reference
    hf_model_name = os.getenv("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
    config = AutoConfig.from_pretrained(hf_model_name)
    config.num_hidden_layers = 1
    hf_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    reference_mlp = hf_model.model.layers[0].mlp

    dim = config.hidden_size
    hidden_dim = config.intermediate_size
    cluster_shape = list(ttnn_mesh_device.shape)

    # Deterministic random weights + input (match test_mlp_2d_vs_reference pattern).
    # TT expects weights in (input_dim, output_dim) layout.
    torch.manual_seed(1234)
    w1_torch = torch.randn(dim, hidden_dim, dtype=torch.bfloat16)  # (dim, hidden_dim)
    w3_torch = torch.randn(dim, hidden_dim, dtype=torch.bfloat16)  # (dim, hidden_dim)
    w2_torch = torch.randn(hidden_dim, dim, dtype=torch.bfloat16)  # (hidden_dim, dim)
    torch_input = torch.randn(1, 1, seq_len, dim, dtype=torch.bfloat16)

    # HF Linear weights are stored as (output_dim, input_dim); copy transposed.
    with torch.no_grad():
        reference_mlp.gate_proj.weight.copy_(w1_torch.T.contiguous())
        reference_mlp.up_proj.weight.copy_(w3_torch.T.contiguous())
        reference_mlp.down_proj.weight.copy_(w2_torch.T.contiguous())

    tt_ccl = TT_CCL(ttnn_mesh_device)
    mesh_ctx = MeshContext(mesh_device=ttnn_mesh_device, tt_ccl=tt_ccl)

    class TestMLP1DConfig(MLP1DConfig):
        @cached_property
        def prefill_config(self) -> MLP1DPrefillConfigs:
            # NOTE: For multi-device runs, W2 consumes the per-device activation shard
            # (hidden_dim // num_devices). The default MLP1D prefill config uses K=hidden_dim,
            # which is valid for small meshes but can violate matmul constraints on 1x32.
            class _Prefill(MLP1DPrefillConfigs):
                def w2_prg_config(inner_self, seq_len: int):
                    n_w2 = inner_self.cfg.dim
                    dram_shard_grid_width = 8
                    return _matmul_config(
                        m=min(seq_len, inner_self.cfg.prefill_len_cutoff),
                        k=inner_self.cfg.hidden_dim // inner_self.cfg.num_devices,
                        n=n_w2,
                        grid_size=inner_self._mlp2_grid(seq_len),
                        per_core_n=math.ceil(n_w2 / (inner_self.cfg.tile_size * dram_shard_grid_width)),
                    )

            return _Prefill(self)

        @cached_property
        def lazy_w1(self) -> LazyWeight:
            return LazyWeight(
                source=w1_torch,
                dtype=dtype,
                device=ttnn_mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(ttnn_mesh_device, dim=-1),
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.w1_w3_mem_config(),
            )

        @cached_property
        def lazy_w2(self) -> LazyWeight:
            return LazyWeight(
                source=w2_torch,
                dtype=dtype,
                device=ttnn_mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(ttnn_mesh_device, dim=-2),
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.w2_mem_config(),
            )

        @cached_property
        def lazy_w3(self) -> LazyWeight:
            return LazyWeight(
                source=w3_torch,
                dtype=dtype,
                device=ttnn_mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(ttnn_mesh_device, dim=-1),
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.w1_w3_mem_config(),
            )

    mlp_config = TestMLP1DConfig(
        dim=dim,
        hidden_dim=hidden_dim,
        mesh_ctx=mesh_ctx,
        max_batch_size=batch_size,
    )

    tt_model = MLP1D(mlp_config)

    with torch.no_grad():
        reference_output = reference_mlp(torch_input)

    input_mem_config = (
        mlp_config.decode_config.mlp_input_memcfg()
        if mode == "decode"
        else mlp_config.prefill_config.mlp_input_memcfg()
    )

    tt_input = ttnn.from_torch(
        torch_input,
        device=ttnn_mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=cluster_shape),
        dtype=ttnn.bfloat8_b,
        memory_config=input_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = tt_model.forward(tt_input, mode)

    tt_output_torch = to_torch_auto_compose(tt_output)

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"MLP1D (direct API) vs HF reference: {pcc_message}")

    assert passing, f"MLP1D output does not meet PCC requirement {pcc_required}: {pcc_message}."
    logger.info(f"MLP1D (direct API) vs HF reference: PASSED for mode={mode}, seq_len={seq_len}")


# ============================================================================
# Integration Tests - Require device
# ============================================================================


# [INFO] this test will retire once models/tt_transformers/tt/model_config.py retires
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [
        (1, 1),  # single device
        (1, 2),  # 1D mesh, 2 devices
        (1, 4),  # 1D mesh, 4 devices
        (1, 8),  # 1D mesh, 8 devices
    ],
    ids=["1x1", "1x2", "1x4", "1x8"],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (512, 32))
def test_mlp_1d_vs_reference_from_model_args(ttnn_mesh_device: ttnn.MeshDevice, seq_len):
    """
    Test that MLP1D class matches the HuggingFace/Meta reference model.
    """
    from models.common.modules.mlp.mlp_1d import MLP1D
    from models.tt_transformers.tests.test_utils import get_ref_model_dype
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    dtype = ttnn.bfloat8_b
    batch_size = 1
    mode = "decode" if seq_len <= 32 else "prefill"

    model_args = ModelArgs(ttnn_mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1

    if model_args.is_galaxy:
        pytest.skip("MLP1D test only runs on non-TG devices")

    state_dict = model_args.load_state_dict()
    model_config = model_args.get_model_config()

    # Load reference model
    first_layer_prefix = model_args.get_state_dict_prefix("MLP", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
    }

    reference_model = model_args.reference_mlp()
    reference_model.load_state_dict(partial_state_dict)

    # Create MLP1D
    def topology_aware_cache_path(dtype):
        if model_args.instruct:
            return (
                model_args.model_cache_path
                / {
                    ttnn.bfloat16: f"tensor_cache_instruct_bf16_{ttnn_mesh_device.shape}",
                    ttnn.bfloat8_b: f"tensor_cache_instruct_bfp8_{ttnn_mesh_device.shape}",
                }[dtype]
            )
        else:
            return (
                model_args.model_cache_path
                / {
                    ttnn.bfloat16: f"tensor_cache_bf16_{ttnn_mesh_device.shape}",
                    ttnn.bfloat8_b: f"tensor_cache_bfp8_{ttnn_mesh_device.shape}",
                }[dtype]
            )

    tt_ccl = TT_CCL(ttnn_mesh_device)
    tt_model = MLP1D.from_model_args(
        mesh_device=ttnn_mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=topology_aware_cache_path(dtype),
        layer_num=0,
    )

    # Create input
    torch_input = torch.randn(
        1, 1, seq_len, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
    )

    # Run reference
    reference_output = reference_model(torch_input)

    # Run TT model
    input_mem_config = model_config["SHARDED_MLP_INPUT_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

    tt_input = ttnn.from_torch(
        torch_input,
        device=ttnn_mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
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
    logger.info(f"MLP1D vs reference: {pcc_message}")

    assert passing, f"MLP1D output does not meet PCC requirement {pcc_required}: {pcc_message}."
    logger.info(f"MLP1D vs reference: PASSED for mode={mode}, seq_len={seq_len}")
