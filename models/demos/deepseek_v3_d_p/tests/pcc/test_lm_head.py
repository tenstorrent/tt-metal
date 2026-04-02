# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
PCC test for TtLMHead module.

Compares torch.nn.Linear (reference) against TtLMHead (multi-chip TTNN)
to verify correctness with DeepSeek 671B LM head dimensions.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import extract_mesh_config
from models.demos.deepseek_v3_d_p.tt.tt_lm_head import TtLMHead
from tests.ttnn.utils_for_testing import assert_with_pcc

# Mapping from torch dtypes to corresponding ttnn dtypes
TORCH_TO_TTNN_DTYPE = {
    torch.bfloat16: ttnn.bfloat16,
    torch.float32: ttnn.float32,
}


def random_weights(config, emb_dim: int, vocab_size: int):
    """
    Generate random weights for LM head testing.

    Args:
        config: HuggingFace config
        emb_dim: Embedding dimension
        vocab_size: Vocabulary size

    Returns:
        Tuple of (config, weights_dict) in bfloat16
    """
    torch.manual_seed(42)
    std = config.initializer_range

    weights = {
        "lm_head.weight": (torch.randn(vocab_size, emb_dim) * std).to(torch.bfloat16),
    }

    logger.info(f"Generated random LM head weight: {weights['lm_head.weight'].shape}")
    return config, weights


@pytest.mark.parametrize(
    "batch_seq_len, emb_dim, vocab_size, run_pcc_check",
    [
        # fmt: off
        pytest.param(32, 1024, 10240, True, id="small"),
        pytest.param(3200, DeepSeekV3Config.EMB_SIZE, DeepSeekV3Config.VOCAB_SIZE, False, id="full-no-pcc"),
        # fmt: on
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (1, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
            1,
            ttnn.Topology.Ring,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 4), topology="ring"),
            id="ring-4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_lm_head(
    mesh_device,
    device_params,
    config_only,
    batch_seq_len: int,
    emb_dim: int,
    vocab_size: int,
    run_pcc_check: bool,
    num_links: int,
    topology: ttnn.Topology,
):
    """
    Test TtLMHead PCC against torch.nn.Linear reference.

    Torch dtypes are set inline; TTNN dtypes are derived automatically.
    """
    # Derive TTNN dtypes from torch dtypes
    torch_activations_dtype = torch.bfloat16
    torch_weights_dtype = torch.bfloat16
    activations_dtype = TORCH_TO_TTNN_DTYPE[torch_activations_dtype]
    weights_dtype = TORCH_TO_TTNN_DTYPE[torch_weights_dtype]

    num_devices = mesh_device.get_num_devices()
    mesh_shape = mesh_device.shape
    logger.debug(f"Testing with mesh_shape={mesh_shape}, num_devices={num_devices}")
    logger.debug(f"batch_seq_len={batch_seq_len}, emb_dim={emb_dim}, vocab_size={vocab_size}")

    # Create input tensor
    mesh_config = extract_mesh_config(mesh_device)
    dispatch_group_size = mesh_config.dispatch_group_size
    torch_input = torch.randn(dispatch_group_size, batch_seq_len, emb_dim).to(torch_activations_dtype)
    logger.debug(f"Created torch input: {torch_input.shape}, dtype={torch_input.dtype}")

    # Run the reference model in case the PCC check is enabled
    weights = None
    if run_pcc_check:
        config, weights = random_weights(config_only, emb_dim, vocab_size)

        # Create PyTorch reference model (Linear without bias), matching dtypes
        logger.debug("Creating torch.nn.Linear reference")
        torch_model = torch.nn.Linear(emb_dim, vocab_size, bias=False)
        torch_model.weight.data = weights["lm_head.weight"]

        # Run reference forward pass to get expected output
        logger.debug("Running torch forward pass")
        torch_output = torch_model(torch_input)
        logger.debug(f"Torch output shape: {torch_output.shape}")

    # Create TTNN LM head model
    logger.debug("Creating TtLMHead")
    tt_model = TtLMHead(
        mesh_device=mesh_device,
        emb_dim=emb_dim,
        vocab_size=vocab_size,
        torch_weights=weights,
        num_links=num_links,
        topology=topology,
        activations_dtype=activations_dtype,
        weights_dtype=weights_dtype,
    )

    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=activations_dtype,
    )
    logger.debug(f"Created ttnn input (sp and tp sharding): {tt_input.shape}")

    logger.debug("Running ttnn forward pass")
    tt_output = tt_model(tt_input)
    logger.debug(f"TTNN output shape (sharded): {tt_output.shape}")

    if not run_pcc_check:
        logger.debug("run_pcc_check=False, skipping PCC validation")
        return

    # Convert and compare
    logger.debug("Converting TTNN output to torch for comparison")
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=(0, -1)),
    )
    logger.debug(f"TTNN output converted to torch: {tt_output_torch.shape}")

    logger.debug("Comparing outputs with PCC")
    pcc_passed, pcc_message = assert_with_pcc(
        torch_output.to(torch.float32),
        tt_output_torch.to(torch.float32),
        pcc=0.9999,
    )

    logger.debug(f"PCC comparison: {pcc_message}")
    assert pcc_passed, f"PCC test failed: {pcc_message}"

    logger.debug("PCC test passed!")
