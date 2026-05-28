# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Test for `MultimodalPrefillDriver` — the end-to-end multimodal prefill
orchestrator.

The driver composes:
  pixel_patches + grid_hws ─► MoonViT ─► vision tokens
  tokens ─► embed_text_fn ─► text embeddings
                                splice
                                  ▼
                  forward_from_embeddings_fn ─► output

For this test we mock the two callables (embed_text_fn,
forward_from_embeddings_fn) so we can verify the orchestration
logic without depending on a live DeepSeek-V3 LLM.

embed_text_fn stub: produces a deterministic per-position tensor with a
sentinel value at each seq position, so we can detect splice errors.

forward_from_embeddings_fn stub: identity passthrough — returns whatever
it received. The test then inspects that tensor to verify the splice
landed vision tokens at the right positions.

Run:
    HF_HOME=/localdev/zbaczewski/hf_cache MESH_DEVICE=N150 \\
        pytest models/demos/deepseek_v3/tests/moonvit/test_multimodal_prefill.py -v
"""
from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tt.moonvit.multimodal_prefill import build_driver


def _make_stub_embed_text_fn(mesh_device, text_hidden: int, dtype=ttnn.bfloat16):
    """Build a stub embed_text_fn that produces a known per-position tensor.

    For position i (0 <= i < seq_len), the resulting embedding row is
    filled with `i + 0.5` (so non-zero, distinct, and easy to spot).
    Vision-token rows are signed differently in real outputs, so when
    splice succeeds the image-position rows will hold MoonViT values
    instead of these sentinels.
    """
    is_mesh = type(mesh_device).__name__ == "MeshDevice"

    def embed_fn(tokens_tt: ttnn.Tensor) -> ttnn.Tensor:
        tokens_pt = ttnn.to_torch(
            tokens_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
        )
        if is_mesh and tokens_pt.shape[0] != 1:
            tokens_pt = tokens_pt[:1]
        tokens_pt = tokens_pt.view(-1)
        seq_len = tokens_pt.shape[0]

        emb = (
            (torch.arange(seq_len, dtype=torch.float32) + 0.5)
            .view(1, 1, seq_len, 1)
            .expand(1, 1, seq_len, text_hidden)
            .to(torch.bfloat16)
            .contiguous()
        )
        return ttnn.from_torch(
            emb,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
        )

    return embed_fn


def _identity_forward(x_embedded: ttnn.Tensor) -> ttnn.Tensor:
    """Stub LLM forward: returns the input unchanged."""
    return x_embedded


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize(
    "grid_hws, prefix_tokens, suffix_tokens",
    [
        ([[16, 16]], 8, 20),  # one image (256 patches → 64 vision tokens), 28 text tokens
        ([[16, 16], [16, 16]], 4, 12),  # two images (128 vision tokens), 16 text tokens
    ],
)
def test_driver_orchestrates_full_pipeline(mesh_device, model_args, grid_hws, prefix_tokens, suffix_tokens):
    """Driver orchestrates MoonViT + embed + splice + LLM-stub correctly."""

    text_hidden = model_args.text_hidden_size
    # Synthetic image_token_id (not collision with anything in model_args).
    image_token_id = 99_999

    # Build the driver against the real MoonViT (loads HF weights) but with our
    # synthetic image_token_id (we override the one from model_args).
    driver = build_driver(model_args=model_args, mesh_device=mesh_device, dtype=ttnn.bfloat16)
    driver.image_token_id = image_token_id  # override for the synthetic test

    # Set up the inputs.
    grid_tensor = torch.tensor(grid_hws, dtype=torch.long)
    L_patches = int(grid_tensor.prod(dim=1).sum().item())
    L_vision = sum((h // 2) * (w // 2) for h, w in grid_hws)
    seq_len = prefix_tokens + L_vision + suffix_tokens
    image_positions = list(range(prefix_tokens, prefix_tokens + L_vision))

    tokens = torch.full((1, 1, seq_len), fill_value=1, dtype=torch.int64)
    for p in image_positions:
        tokens[0, 0, p] = image_token_id

    torch.manual_seed(0)
    pixel_patches = torch.randn(L_patches, 3, 14, 14, dtype=torch.float32)

    embed_fn = _make_stub_embed_text_fn(mesh_device, text_hidden=text_hidden, dtype=ttnn.bfloat16)

    # Run the driver.
    fused_tt = driver.run(
        tokens=tokens,
        pixel_patches=pixel_patches,
        grid_hws=grid_tensor,
        embed_text_fn=embed_fn,
        forward_from_embeddings_fn=_identity_forward,
    )

    # Inspect the spliced tensor that the LLM stub received.
    is_mesh = type(mesh_device).__name__ == "MeshDevice"
    fused_pt = ttnn.to_torch(
        fused_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    if is_mesh and fused_pt.shape[0] != 1:
        fused_pt = fused_pt[:1]
    fused_pt = fused_pt.view(seq_len, text_hidden).to(torch.float32)

    # Non-image positions should still hold the stub embed_fn's sentinel (i + 0.5).
    for pos in range(seq_len):
        if pos in image_positions:
            continue
        expected = pos + 0.5
        actual_max = fused_pt[pos].max().item()
        actual_min = fused_pt[pos].min().item()
        # bf16 quantization on a constant per-row: tolerance ~ |val| * 2^-7.
        tol = max(abs(expected) * 1.0 / 128.0, 0.01)
        assert (
            abs(actual_min - expected) < tol and abs(actual_max - expected) < tol
        ), f"position {pos}: expected sentinel {expected}, got min={actual_min} max={actual_max}"

    # Image positions should hold MoonViT-produced vision tokens. We don't know
    # the exact values without re-running MoonViT, but they MUST NOT equal the
    # text sentinel (which would mean the splice didn't happen).
    for vis_i, pos in enumerate(image_positions):
        sentinel = pos + 0.5
        row = fused_pt[pos]
        # At least one element should differ meaningfully from the sentinel.
        max_diff = (row - sentinel).abs().max().item()
        assert max_diff > 0.1, (
            f"image position {pos} (vision idx {vis_i}): row matches text sentinel "
            f"{sentinel} too closely (max diff {max_diff}) — splice may have failed"
        )

    logger.info(f"[grid_hws={grid_hws} seq={seq_len} L_vision={L_vision}] driver pipeline OK")
