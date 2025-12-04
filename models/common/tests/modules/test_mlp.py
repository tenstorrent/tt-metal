# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the modular MLP implementation.

This test suite verifies:
1. Unit tests for config dataclasses (no device needed)
2. Backward compatibility: MLP.from_model_args() produces identical outputs to original
3. Forward compatibility: MLP with explicit configs works correctly
4. Reference model comparison: outputs match HuggingFace/Meta reference
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

# ============================================================================
# Unit Tests - No device required
# ============================================================================


def test_mlp_config_creation():
    """Test that MLPConfig can be created with expected values."""
    from models.common.modules.mlp.mlp import MLPConfig

    config = MLPConfig(
        dim=4096,
        hidden_dim=14336,
        is_galaxy=False,
        cluster_shape=[1, 1],
        num_devices=1,
        prefill_len_cutoff=1024,
    )

    assert config.dim == 4096
    assert config.hidden_dim == 14336
    assert config.is_galaxy == False
    assert config.cluster_shape == [1, 1]
    assert config.num_devices == 1
    assert config.prefill_len_cutoff == 1024
    # Check defaults
    assert config.dummy_weights == False
    assert config.num_reduce_scatter_links == 1
    assert config.num_all_gather_links == 2


def test_mlp_config_post_init():
    """Test that __post_init__ correctly sets unpadded_hidden_dim."""
    from models.common.modules.mlp.mlp import MLPConfig

    # When unpadded_hidden_dim is not provided, it should be set to hidden_dim
    config = MLPConfig(
        dim=4096,
        hidden_dim=14336,
        is_galaxy=False,
        cluster_shape=[1, 1],
        num_devices=1,
        prefill_len_cutoff=1024,
    )
    assert config.unpadded_hidden_dim == 14336

    # When provided, it should keep the provided value
    config2 = MLPConfig(
        dim=4096,
        hidden_dim=14336,
        is_galaxy=False,
        cluster_shape=[1, 1],
        num_devices=1,
        prefill_len_cutoff=1024,
        unpadded_hidden_dim=14000,
    )
    assert config2.unpadded_hidden_dim == 14000


def test_mlp_program_configs_creation():
    """Test that MLPProgramConfigs can be created with defaults."""
    from models.common.modules.mlp.mlp import MLPProgramConfigs

    program_configs = MLPProgramConfigs()

    # All should be None by default
    assert program_configs.decode_mlp_w1_w3_prg_config is None
    assert program_configs.decode_mlp_w2_prg_config is None
    assert program_configs.ff1_3_tg_progcfg is None
    assert program_configs.ff2_tg_progcfg is None
    assert program_configs.prefill_mlp_w1_w3_prg_config is None
    assert program_configs.prefill_mlp_w2_prg_config is None


def test_mlp_optimization_config_creation():
    """Test that MLPOptimizationConfig can be created with defaults."""
    from models.common.modules.mlp.mlp import MLPOptimizationConfig

    opt_config = MLPOptimizationConfig()

    assert opt_config.ff1_3_dtype == ttnn.bfloat8_b
    assert opt_config.ff2_dtype == ttnn.bfloat8_b
    assert opt_config.activation_dtype is None
    assert opt_config.li_ff1_3_compute_kernel_cfg is None
    assert opt_config.li_ff2_compute_kernel_cfg is None


def test_ccl_topology_linear():
    """Test the default CCL topology function."""
    from models.common.modules.mlp.mlp import ccl_topology

    topology = ccl_topology(2)
    assert topology == ttnn.Topology.Linear


def test_pad_to_size():
    """Test the pad_to_size utility function."""
    from models.common.modules.mlp.mlp import pad_to_size

    # Test padding on last dimension
    x = torch.randn(1, 1, 32, 100)
    padded = pad_to_size(x, dim=-1, size=128)
    assert padded.shape == (1, 1, 32, 128)

    # Original data should be preserved
    assert torch.allclose(padded[:, :, :, :100], x)

    # Padding should be zeros
    assert torch.allclose(padded[:, :, :, 100:], torch.zeros(1, 1, 32, 28))

    # Test no padding needed
    x2 = torch.randn(1, 1, 32, 128)
    padded2 = pad_to_size(x2, dim=-1, size=128)
    assert torch.equal(padded2, x2)

    # Test padding on different dimension
    x3 = torch.randn(1, 1, 24, 128)
    padded3 = pad_to_size(x3, dim=-2, size=32)
    assert padded3.shape == (1, 1, 32, 128)


def test_mlp_config_galaxy_detection():
    """Test different is_galaxy configurations."""
    from models.common.modules.mlp.mlp import MLPConfig

    # Galaxy config (32 devices)
    galaxy_config = MLPConfig(
        dim=8192,
        hidden_dim=28672,
        is_galaxy=True,
        cluster_shape=[4, 8],
        num_devices=32,
        prefill_len_cutoff=512,
    )
    assert galaxy_config.is_galaxy == True

    # T3K config (8 devices)
    t3k_config = MLPConfig(
        dim=4096,
        hidden_dim=14336,
        is_galaxy=False,
        cluster_shape=[1, 8],
        num_devices=8,
        prefill_len_cutoff=1024,
    )
    assert t3k_config.is_galaxy == False


# ============================================================================
# Integration Tests - Require device
# ============================================================================


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (512, 32),  # One prefill, one decode
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mlp_backward_compat(seq_len, batch_size, mesh_device, reset_seeds, ensure_gc):
    """
    Test that MLP.from_model_args() produces identical outputs to the original MLP.

    This test:
    1. Creates original MLP from tt_transformers
    2. Creates modular MLP using from_model_args()
    3. Runs both on the same input
    4. Verifies outputs match with very high PCC (should be identical)
    """
    from models.common.modules.mlp.mlp import MLP as ModularMLP
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.mlp import MLP as OriginalMLP
    from models.tt_transformers.tt.model_config import ModelArgs

    dtype = ttnn.bfloat8_b
    mode = "decode" if seq_len <= 32 else "prefill"

    # Set up model args
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    model_config = model_args.get_model_config()

    # Create CCL handler
    tt_ccl = TT_CCL(mesh_device)

    # Create original MLP
    original_mlp = OriginalMLP(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_config,
    )

    # Create modular MLP using from_model_args (backward compatible interface)
    modular_mlp = ModularMLP.from_model_args(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        model_config=model_config,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
    )

    # Create input tensor
    torch_input = torch.randn(1, 1, seq_len, model_args.dim)

    # Get memory config for input
    input_mem_config = (
        (model_config["MLP_ACT_MEMCFG"] if model_args.is_galaxy else model_config["SHARDED_MLP_INPUT_MEMCFG"])
        if mode == "decode"
        else ttnn.DRAM_MEMORY_CONFIG
    )

    # Prepare input for original MLP
    tt_input_original = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(None, 3) if model_args.is_galaxy else (None, None), mesh_shape=model_args.cluster_shape
        ),
        dtype=ttnn.bfloat8_b,
        memory_config=input_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )

    # Prepare input for modular MLP (same as original)
    tt_input_modular = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(None, 3) if model_args.is_galaxy else (None, None), mesh_shape=model_args.cluster_shape
        ),
        dtype=ttnn.bfloat8_b,
        memory_config=input_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run original MLP
    logger.info("Running original MLP...")
    original_output = original_mlp(tt_input_original, mode)

    # Run modular MLP
    logger.info("Running modular MLP...")
    modular_output = modular_mlp(tt_input_modular, mode)

    # Convert to torch for comparison
    original_output_torch = ttnn.to_torch(
        original_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )

    modular_output_torch = ttnn.to_torch(
        modular_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )

    # Compare outputs - should be identical since same weights and computation
    pcc_required = 0.9999
    passing, pcc_message = comp_pcc(original_output_torch, modular_output_torch, pcc_required)

    logger.info(comp_allclose(original_output_torch, modular_output_torch))
    logger.info(f"PCC between original and modular: {pcc_message}")

    if passing:
        logger.info(f"Backward compatibility test PASSED for mode={mode}, seq_len={seq_len}")
    else:
        logger.warning(f"Backward compatibility test FAILED for mode={mode}, seq_len={seq_len}")

    assert passing, f"ModularMLP output does not match original MLP. PCC: {pcc_message}"


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (512, 32),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mlp_vs_reference(seq_len, mesh_device, reset_seeds, ensure_gc):
    """
    Test that modular MLP matches the HuggingFace/Meta reference model.

    This mirrors the original test_mlp.py test but uses the modular MLP.
    """
    from models.common.modules.mlp.mlp import MLP
    from models.tt_transformers.tests.test_utils import get_ref_model_dype
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    dtype = ttnn.bfloat8_b
    batch_size = 1
    mode = "decode" if seq_len <= 32 else "prefill"

    # Set up model args
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
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

    # Create TT model using from_model_args
    tt_ccl = TT_CCL(mesh_device)
    tt_model = MLP.from_model_args(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        model_config=model_config,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
    )

    # Create input
    torch_input = torch.randn(
        1, 1, seq_len, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
    )

    # Run reference
    reference_output = reference_model(torch_input)

    # Get memory config for input
    input_mem_config = (
        (model_config["MLP_ACT_MEMCFG"] if model_args.is_galaxy else model_config["SHARDED_MLP_INPUT_MEMCFG"])
        if mode == "decode"
        else ttnn.DRAM_MEMORY_CONFIG
    )

    # Run TT model
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(None, 3) if model_args.is_galaxy else (None, None), mesh_shape=model_args.cluster_shape
        ),
        dtype=ttnn.bfloat8_b,
        memory_config=input_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = tt_model(tt_input, mode)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    tt_output_torch = tt_output_torch[:, :1, :, :]

    # Compare
    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC vs reference: {pcc_message}")

    if passing:
        logger.info(f"MLP vs reference: PASSED for mode={mode}, seq_len={seq_len}")
    else:
        logger.warning(f"MLP vs reference: FAILED for mode={mode}, seq_len={seq_len}")

    assert passing, f"MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mlp_explicit_config(mesh_device, reset_seeds, ensure_gc):
    """
    Test MLP instantiation with explicit config objects (not using from_model_args).

    This tests the new explicit interface where all dependencies are passed in.
    """
    from models.common.modules.mlp.mlp import MLP, MLPConfig, MLPOptimizationConfig, MLPProgramConfigs
    from models.tt_transformers.tests.test_utils import get_ref_model_dype
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs, OpGroup, TensorGroup

    dtype = ttnn.bfloat8_b
    seq_len = 512
    batch_size = 1
    mode = "prefill"

    # Set up model args for reference/weight loading
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    model_config = model_args.get_model_config()

    # Build explicit config objects
    config = MLPConfig(
        dim=model_args.dim,
        hidden_dim=model_args.hidden_dim,
        is_galaxy=model_args.is_galaxy,
        cluster_shape=model_args.cluster_shape,
        num_devices=model_args.num_devices,
        prefill_len_cutoff=model_args.prefill_len_cutoff,
        dummy_weights=model_args.dummy_weights,
        unpadded_hidden_dim=model_args.unpadded_hidden_dim,
        ccl_dtype=model_args.ccl_dtype,
        num_reduce_scatter_links=model_args.num_reduce_scatter_links,
        num_all_gather_links=model_args.num_all_gather_links,
        mlp_activation_type=getattr(model_args, "mlp_activation_type", ttnn.UnaryOpType.SILU),
        w1_w3_mem_config=model_args.create_dram_sharded_mem_config(
            model_args.dim, model_args.hidden_dim // model_args.num_devices
        ),
        w2_mem_config=model_args.create_dram_sharded_mem_config(
            model_args.hidden_dim // model_args.num_devices, model_args.dim
        ),
    )

    program_configs = MLPProgramConfigs(
        decode_mlp_w1_w3_prg_config=model_config.get("DECODE_MLP_W1_W3_PRG_CONFIG"),
        decode_mlp_w2_prg_config=model_config.get("DECODE_MLP_W2_PRG_CONFIG"),
        ff1_3_tg_progcfg=model_config.get("FF1_3_TG_PROGCFG"),
        ff2_tg_progcfg=model_config.get("FF2_TG_PROGCFG"),
        prefill_mlp_w1_w3_prg_config=model_config.get("PREFILL_MLP_W1_W3_PRG_CONFIG"),
        prefill_mlp_w2_prg_config=model_config.get("PREFILL_MLP_W2_PRG_CONFIG"),
        ff1_out_reduce_scatter_memcfg=model_config.get("FF1_OUT_REDUCE_SCATTER_MEMCFG"),
        ff1_out_gathered_memcfg=model_config.get("FF1_OUT_GATHERED_MEMCFG"),
        sharded_mlp2_input_memcfg=model_config.get("SHARDED_MLP2_INPUT_MEMCFG"),
        ff2_out_reduce_scatter_memcfg=model_config.get("FF2_OUT_REDUCE_SCATTER_MEMCFG"),
        sharded_attn_input_memcfg=model_config.get("SHARDED_ATTN_INPUT_MEMCFG"),
        decode_residual_memcfg=model_config.get("DECODE_RESIDUAL_MEMCFG"),
    )

    decoders_opt = model_config.get("DECODERS_OPTIMIZATIONS")
    optimization_config = MLPOptimizationConfig(
        ff1_3_dtype=decoders_opt.get_tensor_dtype(decoder_id=0, tensor=TensorGroup.FF1_FF3),
        ff2_dtype=decoders_opt.get_tensor_dtype(decoder_id=0, tensor=TensorGroup.FF2),
        activation_dtype=decoders_opt.get_tensor_dtype(decoder_id=0, tensor=TensorGroup.ACTIVATION),
        li_ff1_3_compute_kernel_cfg=decoders_opt.get_math_fidelity(
            decoder_id=0, op=OpGroup.LI_FF1_FF3, configuration=model_args
        ),
        li_ff2_compute_kernel_cfg=decoders_opt.get_math_fidelity(
            decoder_id=0, op=OpGroup.LI_FF2, configuration=model_args
        ),
    )

    # Create MLP with explicit config
    tt_ccl = TT_CCL(mesh_device)
    state_dict_prefix = model_args.get_state_dict_prefix("MLP", 0)

    tt_model = MLP(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        config=config,
        program_configs=program_configs,
        optimization_config=optimization_config,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        state_dict_prefix=state_dict_prefix,
        ccl_topology=model_args.ccl_topology(),
    )

    # Verify against reference
    first_layer_prefix = model_args.get_state_dict_prefix("MLP", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
    }

    reference_model = model_args.reference_mlp()
    reference_model.load_state_dict(partial_state_dict)

    torch_input = torch.randn(
        1, 1, seq_len, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
    )
    reference_output = reference_model(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(None, 3) if model_args.is_galaxy else (None, None), mesh_shape=model_args.cluster_shape
        ),
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = tt_model(tt_input, mode)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    tt_output_torch = tt_output_torch[:, :1, :, :]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC vs reference (explicit config): {pcc_message}")

    assert passing, f"MLP with explicit config does not meet PCC requirement {pcc_required}: {pcc_message}."
    logger.info("MLP explicit config test PASSED!")


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
# @pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mlp_config_attributes_match(mesh_device, reset_seeds, ensure_gc):
    """
    Test that MLP created via from_model_args has correct config attributes.
    """
    from models.common.modules.mlp.mlp import MLP
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    dtype = ttnn.bfloat8_b
    batch_size = 1

    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    model_config = model_args.get_model_config()

    tt_ccl = TT_CCL(mesh_device)
    mlp = MLP.from_model_args(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        model_config=model_config,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
    )

    # Verify config values match model_args
    assert mlp.config.dim == model_args.dim, f"dim mismatch: {mlp.config.dim} != {model_args.dim}"
    assert mlp.config.hidden_dim == model_args.hidden_dim
    assert mlp.config.is_galaxy == model_args.is_galaxy
    assert mlp.config.cluster_shape == model_args.cluster_shape
    assert mlp.config.num_devices == model_args.num_devices
    assert mlp.config.prefill_len_cutoff == model_args.prefill_len_cutoff

    # Verify convenience aliases
    assert mlp.dim == model_args.dim
    assert mlp.is_galaxy == model_args.is_galaxy

    # Verify weights were loaded
    assert mlp.w1 is not None
    assert mlp.w2 is not None
    assert mlp.w3 is not None

    logger.info("Config attributes test PASSED!")


# ============================================================================
# TTTv2-style split forward function tests
# ============================================================================


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (512, 32),  # One prefill, one decode
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mlp_forward_non_tg(seq_len, mesh_device, reset_seeds, ensure_gc):
    """
    Test that forward_non_tg() produces identical outputs to forward() on non-TG devices.

    This validates the TTTv2-style split forward function where TG branching is eliminated.
    The test:
    1. Creates MLP using from_model_args()
    2. Runs forward() (original with if-else)
    3. Runs forward_non_tg() (flattened, no TG if-else)
    4. Verifies outputs are identical (PCC > 0.9999)

    Only runs on non-TG devices (N150, N300, T3K). Skips on TG.
    """
    from models.common.modules.mlp.mlp import MLP
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    dtype = ttnn.bfloat8_b
    batch_size = 1
    mode = "decode" if seq_len <= 32 else "prefill"

    # Set up model args
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1

    # Skip if TG - this test is only for non-TG devices
    if model_args.is_galaxy:
        pytest.skip("forward_non_tg test only runs on non-TG devices (N150, N300, T3K)")

    state_dict = model_args.load_state_dict()
    model_config = model_args.get_model_config()

    # Create CCL handler
    tt_ccl = TT_CCL(mesh_device)

    # Create MLP
    mlp = MLP.from_model_args(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        model_config=model_config,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
    )

    # Create input tensor
    torch_input = torch.randn(1, 1, seq_len, model_args.dim)

    # Get memory config for input
    input_mem_config = model_config["SHARDED_MLP_INPUT_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

    # Prepare input for original forward
    tt_input_original = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        dtype=ttnn.bfloat8_b,
        memory_config=input_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )

    # Prepare input for forward_non_tg (same tensor, different allocation)
    tt_input_non_tg = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        dtype=ttnn.bfloat8_b,
        memory_config=input_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run original forward
    logger.info(f"Running forward() for mode={mode}, seq_len={seq_len}...")
    original_output = mlp.forward(tt_input_original, mode)

    # Run forward_non_tg
    logger.info(f"Running forward_non_tg() for mode={mode}, seq_len={seq_len}...")
    non_tg_output = mlp.forward_non_tg(tt_input_non_tg, mode)

    # Convert to torch for comparison
    original_output_torch = ttnn.to_torch(
        original_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )

    non_tg_output_torch = ttnn.to_torch(
        non_tg_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )

    # Compare outputs - should be identical since same code path for non-TG
    pcc_required = 0.9999
    passing, pcc_message = comp_pcc(original_output_torch, non_tg_output_torch, pcc_required)

    logger.info(comp_allclose(original_output_torch, non_tg_output_torch))
    logger.info(f"PCC between forward() and forward_non_tg(): {pcc_message}")

    if passing:
        logger.info(f"forward_non_tg test PASSED for mode={mode}, seq_len={seq_len}")
    else:
        logger.warning(f"forward_non_tg test FAILED for mode={mode}, seq_len={seq_len}")

    assert passing, f"forward_non_tg() output does not match forward(). PCC: {pcc_message}"


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (512, 32),  # One prefill, one decode
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mlp_forward_non_tg_vs_reference(seq_len, mesh_device, reset_seeds, ensure_gc):
    """
    Test that forward_non_tg() matches the HuggingFace/Meta reference model.

    This validates that the flattened forward function produces correct results,
    not just that it matches the original forward().
    """
    from models.common.modules.mlp.mlp import MLP
    from models.tt_transformers.tests.test_utils import get_ref_model_dype
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    dtype = ttnn.bfloat8_b
    batch_size = 1
    mode = "decode" if seq_len <= 32 else "prefill"

    # Set up model args
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1

    # Skip if TG
    if model_args.is_galaxy:
        pytest.skip("forward_non_tg test only runs on non-TG devices")

    state_dict = model_args.load_state_dict()
    model_config = model_args.get_model_config()

    # Load reference model
    first_layer_prefix = model_args.get_state_dict_prefix("MLP", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
    }

    reference_model = model_args.reference_mlp()
    reference_model.load_state_dict(partial_state_dict)

    # Create TT model
    tt_ccl = TT_CCL(mesh_device)
    tt_model = MLP.from_model_args(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        model_config=model_config,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
    )

    # Create input
    torch_input = torch.randn(
        1, 1, seq_len, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
    )

    # Run reference
    reference_output = reference_model(torch_input)

    # Get memory config for input
    input_mem_config = model_config["SHARDED_MLP_INPUT_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

    # Run TT model with forward_non_tg
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        dtype=ttnn.bfloat8_b,
        memory_config=input_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = tt_model.forward_non_tg(tt_input, mode)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    tt_output_torch = tt_output_torch[:, :1, :, :]

    # Compare
    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"forward_non_tg vs reference: {pcc_message}")

    if passing:
        logger.info(f"forward_non_tg vs reference: PASSED for mode={mode}, seq_len={seq_len}")
    else:
        logger.warning(f"forward_non_tg vs reference: FAILED for mode={mode}, seq_len={seq_len}")

    assert passing, f"forward_non_tg output does not meet PCC requirement {pcc_required}: {pcc_message}."


# ============================================================================
# MLPNonTG class tests (separate tightened module)
# ============================================================================


def test_mlp_non_tg_config_creation():
    """Test that MLPNonTGConfig can be created with expected values and nested configs."""
    from models.common.modules.mlp.mlp_non_tg import (
        MLPNonTGConfig,
        MLPNonTGDecodeConfigs,
        MLPNonTGOptimizationConfig,
        MLPNonTGPrefillConfigs,
    )

    config = MLPNonTGConfig(
        dim=4096,
        hidden_dim=14336,
        cluster_shape=[1, 8],
        num_devices=8,
        prefill_len_cutoff=1024,
    )

    assert config.dim == 4096
    assert config.hidden_dim == 14336
    assert config.cluster_shape == [1, 8]
    assert config.num_devices == 8
    # No is_galaxy field - this is a non-TG only config
    assert not hasattr(config, "is_galaxy") or config.__dict__.get("is_galaxy") is None

    # Nested configs should be auto-created by __post_init__ with parent reference
    assert isinstance(config.decode, MLPNonTGDecodeConfigs)
    assert isinstance(config.prefill, MLPNonTGPrefillConfigs)
    assert isinstance(config.optimization, MLPNonTGOptimizationConfig)

    # Each sub-config should have a reference back to the parent
    assert config.decode.cfg is config
    assert config.prefill.cfg is config
    assert config.optimization.cfg is config

    # Sub-configs should have callable methods
    assert callable(config.decode.w1_w3_prg_config)
    assert callable(config.prefill.w1_w3_prg_config)
    assert callable(config.optimization.ff1_3_dtype)


def test_mlp_non_tg_configs_creation():
    """Test that MLPNonTGDecodeConfigs and MLPNonTGPrefillConfigs work with parent config."""
    from models.common.modules.mlp.mlp_non_tg import MLPNonTGConfig

    # Create a parent config first
    parent_config = MLPNonTGConfig(
        dim=4096,
        hidden_dim=14336,
        cluster_shape=[1, 8],
        num_devices=8,
        prefill_len_cutoff=1024,
    )

    # Decode should have decode-specific methods
    assert callable(parent_config.decode.w1_w3_prg_config)
    assert callable(parent_config.decode.w2_prg_config)
    assert callable(parent_config.decode.sharded_mlp2_input_memcfg)
    assert callable(parent_config.decode.decode_residual_memcfg)

    # Prefill should have prefill-specific methods (that take seq_len)
    assert callable(parent_config.prefill.w1_w3_prg_config)
    assert callable(parent_config.prefill.w2_prg_config)

    # Neither should have TG-specific methods
    assert not hasattr(parent_config.decode, "ff1_3_tg_progcfg")
    assert not hasattr(parent_config.prefill, "ff1_3_tg_progcfg")

    # Test that methods can be called and return values
    decode_w1_w3 = parent_config.decode.w1_w3_prg_config()
    assert decode_w1_w3 is not None

    prefill_w1_w3 = parent_config.prefill.w1_w3_prg_config(seq_len=512)
    assert prefill_w1_w3 is not None


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (512, 32),  # One prefill, one decode
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mlp_non_tg_class_vs_original(seq_len, mesh_device, reset_seeds, ensure_gc):
    """
    Test that MLPNonTG class produces identical outputs to MLP.forward_non_tg().

    This validates that the separate MLPNonTG class works correctly.
    """
    from models.common.modules.mlp.mlp import MLP
    from models.common.modules.mlp.mlp_non_tg import MLPNonTG
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    dtype = ttnn.bfloat8_b
    batch_size = 1
    mode = "decode" if seq_len <= 32 else "prefill"

    # Set up model args
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1

    # Skip if TG
    if model_args.is_galaxy:
        pytest.skip("MLPNonTG test only runs on non-TG devices")

    state_dict = model_args.load_state_dict()
    model_config = model_args.get_model_config()

    # Create CCL handler
    tt_ccl = TT_CCL(mesh_device)

    # Create original MLP
    original_mlp = MLP.from_model_args(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        model_config=model_config,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
    )

    # Create MLPNonTG (no model_config needed - uses internal defaults)
    non_tg_mlp = MLPNonTG.from_model_args(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
    )

    # Create input tensor
    torch_input = torch.randn(1, 1, seq_len, model_args.dim)

    input_mem_config = model_config["SHARDED_MLP_INPUT_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

    # Prepare inputs
    tt_input_original = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        dtype=ttnn.bfloat8_b,
        memory_config=input_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_input_non_tg = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        dtype=ttnn.bfloat8_b,
        memory_config=input_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run original forward_non_tg
    logger.info(f"Running MLP.forward_non_tg() for mode={mode}...")
    original_output = original_mlp.forward_non_tg(tt_input_original, mode)

    # Run MLPNonTG.forward
    logger.info(f"Running MLPNonTG.forward() for mode={mode}...")
    non_tg_output = non_tg_mlp.forward(tt_input_non_tg, mode)

    # Convert to torch
    original_output_torch = ttnn.to_torch(
        original_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )

    non_tg_output_torch = ttnn.to_torch(
        non_tg_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )

    # Compare - should be identical
    pcc_required = 0.9999
    passing, pcc_message = comp_pcc(original_output_torch, non_tg_output_torch, pcc_required)

    logger.info(comp_allclose(original_output_torch, non_tg_output_torch))
    logger.info(f"PCC between MLP.forward_non_tg() and MLPNonTG.forward(): {pcc_message}")

    assert passing, f"MLPNonTG.forward() does not match MLP.forward_non_tg(). PCC: {pcc_message}"
    logger.info(f"MLPNonTG class test PASSED for mode={mode}, seq_len={seq_len}")


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [
        (1, 1),  # single device # [INFO] apply auto_compose on single device would incur error in c++ code
        (1, 2),  # 1D mesh, 2 devices
        (1, 8),  # 1D mesh, 8 devices
        # todo)) 2D mesh is not supported yet
        # (2, 4),  # 2D mesh, 8 devices
    ],
    ids=[
        "1x1",
        "1x2",
        "1x8",
        # "2x4",
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (512, 32))
def test_mlp_non_tg_class_vs_reference(ttnn_mesh_device: ttnn.MeshDevice, seq_len):
    """
    Test that MLPNonTG class matches the HuggingFace/Meta reference model.
    """
    from models.common.modules.mlp.mlp_non_tg import MLPNonTG
    from models.tt_transformers.tests.test_utils import get_ref_model_dype
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    dtype = ttnn.bfloat8_b
    batch_size = 1
    mode = "decode" if seq_len <= 32 else "prefill"

    model_args = ModelArgs(ttnn_mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1

    if model_args.is_galaxy:
        pytest.skip("MLPNonTG test only runs on non-TG devices")

    state_dict = model_args.load_state_dict()
    model_config = model_args.get_model_config()  # Need for proper input memory config

    # Load reference model
    first_layer_prefix = model_args.get_state_dict_prefix("MLP", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
    }

    reference_model = model_args.reference_mlp()
    reference_model.load_state_dict(partial_state_dict)

    # Create MLPNonTG
    def topology_aware_cache_path(dtype):
        # todo)) use LazyWeight
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
    tt_model = MLPNonTG.from_model_args(
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

    # Run TT model - use model_config for proper input sharding
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
    logger.info(f"MLPNonTG vs reference: {pcc_message}")

    assert passing, f"MLPNonTG output does not meet PCC requirement {pcc_required}: {pcc_message}."
    logger.info(f"MLPNonTG vs reference: PASSED for mode={mode}, seq_len={seq_len}")


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_mlp_non_tg_rejects_galaxy(mesh_device, reset_seeds, ensure_gc):
    """
    Test that MLPNonTG.from_model_args() raises ValueError for Galaxy devices.
    """
    from models.common.modules.mlp.mlp_non_tg import MLPNonTG
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    model_args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1

    if not model_args.is_galaxy:
        pytest.skip("This test only runs on TG devices to verify rejection")

    state_dict = model_args.load_state_dict()
    tt_ccl = TT_CCL(mesh_device)

    with pytest.raises(ValueError, match="MLPNonTG cannot be used for Galaxy devices"):
        MLPNonTG.from_model_args(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=model_args,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
            layer_num=0,
        )

    logger.info("MLPNonTG correctly rejects Galaxy devices")


# ============================================================================
# Run unit tests standalone
# ============================================================================

if __name__ == "__main__":
    print("Running unit tests (no device required)...")
    test_mlp_config_creation()
    print("  ✓ test_mlp_config_creation")

    test_mlp_config_post_init()
    print("  ✓ test_mlp_config_post_init")

    test_mlp_program_configs_creation()
    print("  ✓ test_mlp_program_configs_creation")

    test_mlp_optimization_config_creation()
    print("  ✓ test_mlp_optimization_config_creation")

    test_ccl_topology_linear()
    print("  ✓ test_ccl_topology_linear")

    test_pad_to_size()
    print("  ✓ test_pad_to_size")

    test_mlp_config_galaxy_detection()
    print("  ✓ test_mlp_config_galaxy_detection")

    # MLPNonTG unit tests
    test_mlp_non_tg_config_creation()
    print("  ✓ test_mlp_non_tg_config_creation")

    test_mlp_non_tg_configs_creation()
    print("  ✓ test_mlp_non_tg_configs_creation")

    print("\nAll unit tests passed! ✓")
    print("\nTo run device tests, use pytest:")
    print("  pytest models/common/modules/mlp/test_mlp.py -v")
    print("\nTo run MLPNonTG tests specifically:")
    print("  pytest models/common/modules/mlp/test_mlp.py -v -k 'mlp_non_tg'")
