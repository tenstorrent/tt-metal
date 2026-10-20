# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

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
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.mistral_small_4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_x_device_params,
    torus_xy_device_params,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import get_tp_mesh_composer
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from tests.ttnn.utils_for_testing import comp_pcc


@pytest.mark.parametrize(
    "isl_per_chip, vocab_size, emb_dim",
    [
        (PREFILL_CHUNK_TOKENS_PER_CHIP, DeepSeekV3Config.VOCAB_SIZE, DeepSeekV3Config.EMB_SIZE),
        (PREFILL_CHUNK_TOKENS_PER_CHIP, MistralSmall4Config.VOCAB_SIZE, MistralSmall4Config.EMB_SIZE),
        (PREFILL_CHUNK_TOKENS_PER_CHIP, KimiK3Config.VOCAB_SIZE, KimiK3Config.EMB_SIZE),
    ],
    ids=["isl_5k", "mistral4", "kimi_k3"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, sp_axis",
    [
        pytest.param(
            (1, 4),
            torus_x_device_params(),
            0,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 4), topology="ring"),
            id="torus-x-1x4",
        ),
        pytest.param(
            (2, 4),
            fabric2d_device_params(),
            0,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-2x4",
        ),
        # The production shape. This test previously topped out at 2x4, so no model was covering
        # the embedding at the mesh it actually runs on.
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            0,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
        # One 4x4 prefill stage (a Galaxy tray half): tokens over the axis-1 ring, the table over axis 0.
        pytest.param(
            (4, 4),
            torus_x_device_params(),
            1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4"),
            id="torus-x-4x4-sp1",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_parallel_embedding(mesh_device, sp_axis, isl_per_chip, vocab_size, emb_dim):
    """Test TtParallelEmbedding against torch reference."""

    tp_axis = 1 - sp_axis
    sp_factor = mesh_device.shape[sp_axis]
    tp_factor = mesh_device.shape[tp_axis]

    signpost(f"embedding-{mesh_device.shape}-isl_per_chip{isl_per_chip}-v{vocab_size}-e{emb_dim}")

    logger.debug(
        f"Config: {mesh_device.shape=}, {sp_factor=}, {tp_factor=}, " f"{isl_per_chip=}, {vocab_size=}, {emb_dim=}"
    )

    assert (
        isl_per_chip % ttnn.TILE_SIZE == 0
    ), f"isl_per_chip ({isl_per_chip}) must be a multiple of TILE_SIZE ({ttnn.TILE_SIZE})"
    assert emb_dim % tp_factor == 0, f"emb_dim ({emb_dim}) must be divisible by tp_factor ({tp_factor})"

    ttnn.visualize_mesh_device(mesh_device)

    # ========================================
    # Reference: torch embedding
    # ========================================
    torch.manual_seed(42)
    torch_weight = torch.randn(vocab_size, emb_dim, dtype=torch.float32)

    # Tokens shaped as [sp_factor, 1, isl_per_chip] — one chunk per SP device
    torch_tokens = torch.randint(0, vocab_size, (sp_factor, 1, isl_per_chip))

    # Reference output: [sp_factor, 1, isl_per_chip, emb_dim]
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
        sp_axis=sp_axis,
        tp_axis=tp_axis,
    )

    # Shard tokens: dim 0 across the SP axis, replicate across the TP axis
    token_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, None) if sp_axis == 0 else (None, 0),
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
    composer = get_tp_mesh_composer(mesh_device, sp_axis)
    tt_host = ttnn.to_torch(tt_output, mesh_composer=composer, dtype=torch.bfloat16)
    logger.debug(f"TT host output shape: {tt_host.shape}")

    threshold = 0.999
    _, pcc = comp_pcc(torch_output.float(), tt_host.float())
    logger.debug(f"PCC: {pcc:.6f} (threshold: {threshold})")

    assert pcc > threshold, f"PCC {pcc:.6f} below threshold {threshold}"
