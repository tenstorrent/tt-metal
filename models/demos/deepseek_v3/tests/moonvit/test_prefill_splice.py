# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Tests for the vision-text splice helpers in
`models/demos/deepseek_v3/tt/moonvit/prefill_splice.py`.

(1) Host-only test: pure-torch correctness — synthesize text embeddings,
    place a known set of image-token positions, run the splice, verify
    the output has vision-token values exactly at those positions and
    text-embedding values everywhere else.

(2) Device roundtrip: push text embeddings to device, run the wrapper,
    pull back, compare against the host-only result. Verifies that the
    gather → splice → push-back round-trip preserves the splice
    semantics through ttnn dtype/layout conversions.

These tests don't load the HF reference — they're focused unit tests of
the splice mechanism, independent of the actual MoonViT vision tower or
the DeepSeek LLM.

Run:
    MESH_DEVICE=N150 \\
        pytest models/demos/deepseek_v3/tests/moonvit/test_prefill_splice.py -v
"""
from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.deepseek_v3.tt.moonvit.prefill_splice import (
    splice_vision_into_text_embeddings,
    splice_vision_via_host,
)

# ----------------------------------------------------------------------
# (1) Pure-torch correctness


@torch.no_grad()
@pytest.mark.parametrize(
    "seq_len, n_image_tokens, image_positions",
    [
        # All image tokens in one contiguous block (typical: image embeds replace
        # a run of placeholder tokens emitted by the HF processor).
        (256, 16, list(range(10, 26))),
        # Two image blocks (multi-image prompt).
        (256, 32, list(range(10, 26)) + list(range(100, 116))),
        # Single image token (degenerate case).
        (64, 1, [32]),
        # No image tokens at all — splice should be a no-op.
        (64, 0, []),
    ],
)
def test_splice_host_correctness(seq_len, n_image_tokens, image_positions):
    hidden = 64
    image_token_id = 9999

    # Build tokens: non-image positions hold a sentinel (e.g. 1), image positions hold image_token_id.
    tokens = torch.ones(1, 1, seq_len, dtype=torch.int64)
    for p in image_positions:
        tokens[0, 0, p] = image_token_id

    # Text embeddings: each position gets a unique deterministic vector so we can
    # tell whether it was clobbered. Use position-as-row to make values obvious.
    text_emb = torch.arange(seq_len, dtype=torch.float32).view(1, 1, seq_len, 1).expand(1, 1, seq_len, hidden).clone()
    # Vision tokens: distinct sentinel range so collisions with text values are impossible.
    vision = torch.full((n_image_tokens, hidden), -1.0, dtype=torch.float32)
    for i in range(n_image_tokens):
        vision[i] = -float(i + 1)  # -1, -2, -3, ...

    fused = splice_vision_into_text_embeddings(text_emb, tokens, vision, image_token_id)

    assert fused.shape == text_emb.shape
    # Image positions should hold the corresponding vision vector, in order.
    for vis_i, pos in enumerate(image_positions):
        expected_row = -float(vis_i + 1)
        actual = fused[0, 0, pos]
        assert torch.all(
            actual == expected_row
        ), f"position {pos}: expected vision row {vis_i} = {expected_row}, got {actual[0].item()}"
    # Non-image positions should be untouched.
    for pos in range(seq_len):
        if pos in image_positions:
            continue
        expected = float(pos)
        actual = fused[0, 0, pos]
        assert torch.all(actual == expected), f"position {pos}: text expected {expected}, got {actual[0].item()}"


@torch.no_grad()
def test_splice_host_shape_mismatch():
    """Mismatched counts must raise (not silently truncate or pad)."""
    hidden = 32
    seq_len = 10
    image_token_id = 9999
    tokens = torch.ones(1, 1, seq_len, dtype=torch.int64)
    tokens[0, 0, 3] = image_token_id
    tokens[0, 0, 4] = image_token_id  # two image positions
    text_emb = torch.zeros(1, 1, seq_len, hidden, dtype=torch.float32)
    vision = torch.zeros(5, hidden, dtype=torch.float32)  # wrong: 5 rows for 2 positions
    with pytest.raises(ValueError, match="2 occurrences"):
        splice_vision_into_text_embeddings(text_emb, tokens, vision, image_token_id)


# ----------------------------------------------------------------------
# (2) Device roundtrip


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
def test_splice_device_roundtrip(mesh_device):
    """Device roundtrip preserves the host splice result (modulo bf16 noise)."""
    pcc_threshold = 0.999  # bf16 quantization on push-back, otherwise identical
    seq_len = 256
    hidden = 64  # small to keep the test fast
    image_token_id = 9999
    image_positions = list(range(20, 36))  # 16 contiguous image positions
    n_image_tokens = len(image_positions)

    torch.manual_seed(0)
    tokens = torch.ones(1, 1, seq_len, dtype=torch.int64)
    for p in image_positions:
        tokens[0, 0, p] = image_token_id

    text_emb = torch.randn(1, 1, seq_len, hidden, dtype=torch.float32)
    vision = torch.randn(n_image_tokens, hidden, dtype=torch.float32)

    # Host reference splice (pure torch).
    host_fused = splice_vision_into_text_embeddings(text_emb, tokens, vision, image_token_id)

    # Push text embeddings to device.
    is_mesh = type(mesh_device).__name__ == "MeshDevice"
    text_tt = ttnn.from_torch(
        text_emb.to(torch.bfloat16).contiguous(),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )

    # Device roundtrip splice.
    fused_tt = splice_vision_via_host(
        mesh_device=mesh_device,
        text_embedded_tt=text_tt,
        tokens=tokens,
        vision_tokens=vision,
        image_token_id=image_token_id,
        dtype=ttnn.bfloat16,
    )
    device_fused = ttnn.to_torch(
        fused_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    if is_mesh and device_fused.shape[0] != 1:
        device_fused = device_fused[:1]
    device_fused = device_fused.view(1, 1, seq_len, hidden)

    passing, pcc_msg = comp_pcc(host_fused, device_fused, pcc_threshold)
    logger.info(f"[device roundtrip] {comp_allclose(host_fused, device_fused)} {pcc_msg}")
    assert passing, f"device-roundtrip splice diverges from host splice: {pcc_msg}"
