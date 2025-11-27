# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Backward compatibility test for ModularMLP.

This test verifies that ModularMLP produces identical outputs to the
original MLP implementation for the same inputs and configurations.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.modules.mlp.modular_attempt import ModularMLP
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import ModelArgs


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
def test_modular_mlp_backward_compat(seq_len, batch_size, mesh_device, reset_seeds, ensure_gc):
    """
    Test that ModularMLP produces identical outputs to the original MLP.

    This test:
    1. Creates both original MLP and ModularMLP with same config
    2. Runs both on the same input
    3. Verifies outputs match with high PCC
    """
    dtype = ttnn.bfloat8_b
    mode = "decode" if seq_len <= 32 else "prefill"

    # Set up model args
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # Create CCL handler
    tt_ccl = TT_CCL(mesh_device)

    # Create original MLP
    original_mlp = MLP(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
    )

    # Create ModularMLP with same config
    modular_mlp = ModularMLP(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
    )

    # Create input tensor
    torch_input = torch.randn(1, 1, seq_len, model_args.dim)

    # Prepare input for original MLP
    tt_input_original = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(None, 3) if model_args.is_galaxy else (None, None), mesh_shape=model_args.cluster_shape
        ),
        dtype=ttnn.bfloat8_b,
        memory_config=(
            (
                original_mlp.model_config["MLP_ACT_MEMCFG"]
                if model_args.is_galaxy
                else model_args.model_config["SHARDED_MLP_INPUT_MEMCFG"]
            )
            if mode == "decode"
            else ttnn.DRAM_MEMORY_CONFIG
        ),
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
        memory_config=(
            (
                model_args.model_config["MLP_ACT_MEMCFG"]
                if model_args.is_galaxy
                else model_args.model_config["SHARDED_MLP_INPUT_MEMCFG"]
            )
            if mode == "decode"
            else ttnn.DRAM_MEMORY_CONFIG
        ),
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

    # Compare outputs
    pcc_required = 0.9999  # Very high PCC since implementations should be identical
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
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_modular_mlp_vs_reference(mesh_device, reset_seeds, ensure_gc):
    """
    Test that ModularMLP matches the HuggingFace/Meta reference model.

    This is the same test as test_mlp.py but using ModularMLP.
    """
    dtype = ttnn.bfloat8_b
    seq_len = 512
    batch_size = 1
    mode = "prefill"

    # Set up model args
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # Load reference model
    first_layer_prefix = model_args.get_state_dict_prefix("MLP", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
    }

    reference_model = model_args.reference_mlp()
    reference_model.load_state_dict(partial_state_dict)

    # Create TT model
    tt_ccl = TT_CCL(mesh_device)
    tt_model = ModularMLP(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
    )

    # Create input
    torch_input = torch.randn(
        1, 1, seq_len, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
    )

    # Run reference
    reference_output = reference_model(torch_input)

    # Run TT model
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

    # Compare
    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC vs reference: {pcc_message}")

    if passing:
        logger.info("ModularMLP vs reference: PASSED!")
    else:
        logger.warning("ModularMLP vs reference: FAILED!")

    assert passing, f"ModularMLP output does not meet PCC requirement {pcc_required}: {pcc_message}."


# Unit tests for config and strategy components (no device needed)


def test_mlp_config_creation():
    """Test that MLPConfig can be created with expected values."""
    from models.common.modules.mlp.modular_attempt.mlp_config import HardwareTopology, MLPConfig

    config = MLPConfig(
        dim=4096,
        hidden_dim=14336,
        topology=HardwareTopology.SINGLE_CHIP,
    )

    assert config.dim == 4096
    assert config.hidden_dim == 14336
    assert config.is_galaxy == False
    assert config.topology == HardwareTopology.SINGLE_CHIP


def test_hardware_topology_enum():
    """Test HardwareTopology enum values."""
    from models.common.modules.mlp.modular_attempt.mlp_config import HardwareTopology

    assert HardwareTopology.SINGLE_CHIP.name == "SINGLE_CHIP"
    assert HardwareTopology.LINEAR_1D.name == "LINEAR_1D"
    assert HardwareTopology.GALAXY_2D.name == "GALAXY_2D"


def test_ccl_strategy_factory():
    """Test that CCL strategy factory returns correct types."""
    from models.common.modules.mlp.modular_attempt.ccl_strategies import (
        GalaxyTopologyStrategy,
        LinearTopologyStrategy,
        SingleChipStrategy,
        create_ccl_strategy,
    )
    from models.common.modules.mlp.modular_attempt.mlp_config import HardwareTopology

    # We can't fully test without a device, but we can check the factory logic
    # by checking the class mapping
    class MockDevice:
        pass

    class MockCCL:
        pass

    class MockArgs:
        pass

    strategy = create_ccl_strategy(HardwareTopology.SINGLE_CHIP, MockDevice(), MockCCL(), MockArgs())
    assert isinstance(strategy, SingleChipStrategy)

    strategy = create_ccl_strategy(HardwareTopology.LINEAR_1D, MockDevice(), MockCCL(), MockArgs())
    assert isinstance(strategy, LinearTopologyStrategy)

    strategy = create_ccl_strategy(HardwareTopology.GALAXY_2D, MockDevice(), MockCCL(), MockArgs())
    assert isinstance(strategy, GalaxyTopologyStrategy)


if __name__ == "__main__":
    # Run unit tests without device
    test_mlp_config_creation()
    test_hardware_topology_enum()
    test_ccl_strategy_factory()
    print("All unit tests passed!")
