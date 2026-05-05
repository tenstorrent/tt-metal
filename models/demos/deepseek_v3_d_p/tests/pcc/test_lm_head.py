# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

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


def random_weights(config, emb_dim: int, vocab_size: int, dtype: torch.dtype):
    """
    Generate random weights for LM head testing.

    Args:
        config: HuggingFace config
        emb_dim: Embedding dimension
        vocab_size: Vocabulary size
        dtype: Desired torch dtype for weights

    Returns:
        Tuple of (config, weights_dict) in the specified dtype
    """
    torch.manual_seed(42)
    std = config.initializer_range

    weights = {
        "lm_head.weight": (torch.randn(vocab_size, emb_dim) * std).to(dtype),
    }

    logger.info(f"Generated random LM head weight: {weights['lm_head.weight'].shape}")
    return config, weights


@pytest.mark.parametrize("is_balanced", [False, True], ids=["sequential", "balanced"])
@pytest.mark.parametrize(
    "batch_seq_len, emb_dim, vocab_size, run_full_pcc_check",
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
            id="1x4-ring",
        ),
        pytest.param(
            (2, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
            1,
            ttnn.Topology.Ring,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="ring"),
            id="2x2-ring",
        ),
        pytest.param(
            (2, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="linear"),
            id="2x4-linear",
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
    run_full_pcc_check: bool,
    num_links: int,
    topology: ttnn.Topology,
    is_balanced: bool,
):
    """
    Test TtLMHead PCC against torch.nn.Linear reference.

    Torch dtypes are set inline; TTNN dtypes are derived automatically.
    """
    if batch_seq_len != ttnn.TILE_SIZE and run_full_pcc_check:
        pytest.skip("PCC check is only run for seq_len == TILE_SIZE to avoid slicing complexities")

    # Derive TTNN dtypes from torch dtypes
    torch_activations_dtype = torch.bfloat16
    torch_weights_dtype = torch.bfloat16
    ttnn_activations_dtype = ttnn.bfloat16
    ttnn_weights_dtype = ttnn.bfloat16

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
    if run_full_pcc_check:
        config, weights = random_weights(config_only, emb_dim, vocab_size, torch_weights_dtype)

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
        torch_weight=weights["lm_head.weight"] if weights else None,
        num_links=num_links,
        topology=topology,
        activations_dtype=ttnn_activations_dtype,
        weights_dtype=ttnn_weights_dtype,
        is_balanced=is_balanced,
    )

    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn_activations_dtype,
    )
    logger.debug(f"Created ttnn input (sp and tp sharding): {tt_input.shape}")

    logger.debug("Running ttnn forward pass")
    global_token_id = batch_seq_len * dispatch_group_size - 1
    tt_output, token_offset = tt_model(tt_input, global_token_id=global_token_id)
    logger.debug(f"TTNN output shape (sharded): {tt_output.shape}")

    # For now, we only run the full PCC check on input tensors with seq_len == TILE_SIZE to avoid slicing
    # because we have yet to decide in which order tokens will be distributed in seq_len (See Zigzag
    # attention for more details).
    if not run_full_pcc_check:
        logger.debug("run_full_pcc_check=False, skipping full PCC validation")
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


def test_global_to_local_token_id():
    """Verify token mapping for both balanced and sequential modes."""
    from models.demos.deepseek_v3_d_p.tt.mla.utils import global_to_local_token_id

    sp_factor = 4
    seq_len = 1024
    chunk_size_balanced = seq_len // (2 * sp_factor)  # 128
    chunk_size_sequential = seq_len // sp_factor  # 256

    # === is_balanced=True (zigzag) ===
    # Token 0 -> device 0, offset 0
    device_id, local_id = global_to_local_token_id(0, sp_factor, seq_len, is_balanced=True)
    assert device_id == 0 and local_id == 0, f"Expected (0, 0), got ({device_id}, {local_id})"

    # Token at start of chunk 1 -> device 1, offset 0
    device_id, local_id = global_to_local_token_id(chunk_size_balanced, sp_factor, seq_len, is_balanced=True)
    assert device_id == 1 and local_id == 0, f"Expected (1, 0), got ({device_id}, {local_id})"

    # Token at start of chunk 7 (last chunk) -> device 0, with offset chunk_size
    # Chunk 7 maps to device: num_chunks - 1 - chunk_id = 8 - 1 - 7 = 0
    device_id, local_id = global_to_local_token_id(7 * chunk_size_balanced, sp_factor, seq_len, is_balanced=True)
    assert (
        device_id == 0 and local_id == chunk_size_balanced
    ), f"Expected (0, {chunk_size_balanced}), got ({device_id}, {local_id})"

    # === is_balanced=False (sequential) ===
    # Token 0 -> device 0, offset 0
    device_id, local_id = global_to_local_token_id(0, sp_factor, seq_len, is_balanced=False)
    assert device_id == 0 and local_id == 0, f"Expected (0, 0), got ({device_id}, {local_id})"

    # Token 256 -> device 1, offset 0
    device_id, local_id = global_to_local_token_id(chunk_size_sequential, sp_factor, seq_len, is_balanced=False)
    assert device_id == 1 and local_id == 0, f"Expected (1, 0), got ({device_id}, {local_id})"

    # Token 512 -> device 2, offset 0
    device_id, local_id = global_to_local_token_id(2 * chunk_size_sequential, sp_factor, seq_len, is_balanced=False)
    assert device_id == 2 and local_id == 0, f"Expected (2, 0), got ({device_id}, {local_id})"

    # Last token (1023) -> device 3, offset 255
    device_id, local_id = global_to_local_token_id(seq_len - 1, sp_factor, seq_len, is_balanced=False)
    assert (
        device_id == 3 and local_id == chunk_size_sequential - 1
    ), f"Expected (3, {chunk_size_sequential - 1}), got ({device_id}, {local_id})"

    logger.info("test_global_to_local_token_id passed!")
