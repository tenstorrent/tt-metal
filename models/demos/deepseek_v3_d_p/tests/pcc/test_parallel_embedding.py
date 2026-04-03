# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
PCC test for TtParallelEmbedding module.

Verifies that the TTNN parallel embedding produces the same output as
torch.nn.functional.embedding for SP+TP distributed configurations.
"""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import get_tp_mesh_composer
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from tests.ttnn.utils_for_testing import comp_pcc


@pytest.mark.parametrize(
    "seq_len, vocab_size, emb_dim",
    [
        (128, 1024, 256),
        (512, DeepSeekV3Config.VOCAB_SIZE, DeepSeekV3Config.EMB_SIZE),
    ],
    ids=["small", "deepseek"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (1, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 4), topology="linear"),
            id="linear-4",
        ),
        pytest.param(
            (2, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="mesh-2x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_parallel_embedding(mesh_device, seq_len, vocab_size, emb_dim):
    """Test TtParallelEmbedding against torch reference."""

    sp_factor = mesh_device.shape[0]
    tp_factor = mesh_device.shape[1]
    seq_per_chip = seq_len // sp_factor

    signpost(f"embedding-{mesh_device.shape}-seq{seq_len}-v{vocab_size}-e{emb_dim}")

    logger.debug(
        f"Config: {mesh_device.shape=}, {sp_factor=}, {tp_factor=}, " f"{seq_per_chip=}, {vocab_size=}, {emb_dim=}"
    )

    assert seq_len % sp_factor == 0, f"seq_len ({seq_len}) must be divisible by sp_factor ({sp_factor})"
    assert (
        seq_per_chip % ttnn.TILE_SIZE == 0
    ), f"seq_per_chip ({seq_per_chip}) must be a multiple of TILE_SIZE ({ttnn.TILE_SIZE})"
    assert emb_dim % tp_factor == 0, f"emb_dim ({emb_dim}) must be divisible by tp_factor ({tp_factor})"

    ttnn.visualize_mesh_device(mesh_device)

    # ========================================
    # Reference: torch embedding
    # ========================================
    torch.manual_seed(42)
    torch_weight = torch.randn(vocab_size, emb_dim, dtype=torch.float32)

    # Tokens shaped as [sp_factor, 1, seq_per_chip] — one chunk per SP device
    torch_tokens = torch.randint(0, vocab_size, (sp_factor, 1, seq_per_chip))

    # Reference output: [sp_factor, 1, seq_per_chip, emb_dim]
    torch_output = F.embedding(torch_tokens, torch_weight)
    logger.debug(f"Torch reference output: {torch_output.shape}")

    # ========================================
    # TTNN: create module and shard inputs
    # ========================================
    tt_emb = TtParallelEmbedding(
        mesh_device=mesh_device,
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        torch_weight=torch_weight,
    )

    # Shard tokens: dim 0 across SP axis (rows), replicate across TP axis (cols)
    token_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, None),
    )
    tt_tokens = ttnn.from_torch(
        torch_tokens,
        mesh_mapper=token_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint32,
    )
    logger.debug(f"TT tokens shape: {tt_tokens.shape}")

    # ========================================
    # Forward
    # ========================================
    tt_output = tt_emb(tt_tokens)
    logger.debug(f"TT output shape: {tt_output.shape}")

    # ========================================
    # Compose back to host and compare
    # ========================================
    composer = get_tp_mesh_composer(mesh_device)
    tt_host = ttnn.to_torch(tt_output, mesh_composer=composer, dtype=torch.bfloat16)
    logger.debug(f"TT host output shape: {tt_host.shape}")

    threshold = 0.999
    _, pcc = comp_pcc(torch_output.float(), tt_host.float())
    logger.debug(f"PCC: {pcc:.6f} (threshold: {threshold})")

    assert pcc > threshold, f"PCC {pcc:.6f} below threshold {threshold}"
