# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Per-submodule PCC tests for DeepSeek-V4-Flash.

Today only the input embedding is exercised: we load ``embed.weight`` from the
safetensors checkpoint, push it through a stock ``torch.nn.functional.embedding``
on CPU as the reference, run ``ttnn.embedding`` on device with the same weight
and the same token-id batch, then compare with PCC. As more submodules are
ported they should slot in alongside this test using the same pattern.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.deepseek_v4_flash.tt.embedding import DeepSeekV4Flash
from models.experimental.deepseek_v4_flash.tt.weight_loader import (
    DeepseekV4WeightLoader,
    resolve_snapshot_dir,
)


DEFAULT_MODEL_DIR = Path("/home/ttuser/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash")
PCC_THRESHOLD = 0.99


def _checkpoint_available() -> bool:
    try:
        resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return False
    return True


@pytest.fixture(scope="module")
def embed_weight() -> torch.Tensor:
    """Real bf16 embedding table loaded from the V4-Flash checkpoint."""
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    return loader.get_tensor("embed_tokens.weight")  # [vocab_size, hidden_size]


@pytest.mark.skipif(
    not _checkpoint_available(),
    reason=f"V4-Flash checkpoint not found under {DEFAULT_MODEL_DIR}",
)
@torch.no_grad()
@pytest.mark.parametrize("batch_size", (1, 4))
@pytest.mark.parametrize("seq_len", (32, 128))
def test_embed_tokens_pcc(
    device,
    reset_seeds,
    embed_weight: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> None:
    """Embed a random token-id batch on CPU and on ttnn; compare with PCC.

    Both sides see the same bf16 weight table and the same integer token-ids,
    so any discrepancy comes from the ttnn kernel path (layout / dtype
    casting). A PCC >= 0.99 is the standard tt-transformers threshold for
    embedding parity.
    """
    vocab_size, hidden_size = embed_weight.shape
    logger.info(f"embed_weight: shape={tuple(embed_weight.shape)} dtype={embed_weight.dtype}")

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int64)

    reference_output = torch.nn.functional.embedding(input_ids, embed_weight)
    logger.info(f"reference_output: shape={tuple(reference_output.shape)}")

    tt_weight = ttnn.from_torch(
        embed_weight,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_input = ttnn.from_torch(
        input_ids.to(torch.int32),
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    tt_output = ttnn.embedding(tt_input, tt_weight, layout=ttnn.TILE_LAYOUT)
    tt_output_torch = ttnn.to_torch(tt_output).reshape(reference_output.shape).to(reference_output.dtype)
    logger.info(f"tt_output_torch: shape={tuple(tt_output_torch.shape)}")

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc=PCC_THRESHOLD)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"embed_tokens PCC < {PCC_THRESHOLD} (batch={batch_size}, seq={seq_len}): {pcc_message}"


@pytest.mark.skipif(
    not _checkpoint_available(),
    reason=f"V4-Flash checkpoint not found under {DEFAULT_MODEL_DIR}",
)
@torch.no_grad()
@pytest.mark.parametrize("batch_size", (1, 4))
@pytest.mark.parametrize("seq_len", (32, 128))
def test_deepseek_v4_flash_forward_pcc(
    device,
    reset_seeds,
    batch_size: int,
    seq_len: int,
) -> None:
    """End-to-end forward parity for the ``DeepSeekV4Flash`` model object.

    Builds the model (which lazily materialises its embedding table from the
    safetensors checkpoint), pushes a random token-id batch through
    ``model(input_ids, attention_mask)`` on device, and compares against a CPU
    ``torch.nn.functional.embedding`` reference using the same weight table.

    The model is a stub today (``forward`` just returns the input embeddings),
    so this exercises the object wiring + embedding path end-to-end. As more
    submodules land, the reference should grow alongside the model.
    """
    model = DeepSeekV4Flash(config={}, weights_dir=DEFAULT_MODEL_DIR, device=device)

    embed_weight = model.weight_loader.get_tensor("embed_tokens.weight")
    vocab_size, hidden_size = embed_weight.shape
    logger.info(f"embed_weight: shape={tuple(embed_weight.shape)} dtype={embed_weight.dtype}")

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int64)

    reference_output = torch.nn.functional.embedding(input_ids, embed_weight)
    logger.info(f"reference_output: shape={tuple(reference_output.shape)}")

    tt_input_ids = ttnn.from_torch(
        input_ids.to(torch.int32),
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    tt_output = model(tt_input_ids, None)
    tt_output_torch = ttnn.to_torch(tt_output).reshape(reference_output.shape).to(reference_output.dtype)
    logger.info(f"tt_output_torch: shape={tuple(tt_output_torch.shape)}")

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc=PCC_THRESHOLD)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"DeepSeekV4Flash forward PCC < {PCC_THRESHOLD} (batch={batch_size}, seq={seq_len}): {pcc_message}"
