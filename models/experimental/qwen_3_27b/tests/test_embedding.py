# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Component tests for TtQwen36Embedding.

Hardware agnostic: the only device the tests ask for is a (1, 1) mesh, which
every machine has. A single chip already *is* a 1x1 mesh, so the same file runs
unchanged on one Wormhole and on a Blackhole box -- and gains mesh shapes here
once the module learns to shard its table.

Run:
    pytest models/experimental/qwen_3_27b/tests/test_embedding.py -v
"""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.qwen_3_27b.tt.tt_embedding import TtQwen36Embedding

PCC_THRESHOLD = 0.999

# (vocab_size, dim, seq_len)
#   small_*  -- fast iteration
#   small_t48 -- T is NOT a multiple of 32, so ttnn.embedding silently falls back
#                off the fused-tilize path. Values must still be correct.
#   full     -- the real Qwen3.6-27B table: 248320 x 5120, ~2.4 GiB in bf16
CONFIGS = [
    (1024, 512, 32),
    (1024, 512, 48),
    (248320, 5120, 32),
]
CONFIG_IDS = ["small_t32", "small_t48", "full"]


def _run_embedding(mesh_device, table, ids):
    """Push table + ids through the module, return the result as a torch tensor."""
    tt_model = TtQwen36Embedding(mesh_device, table)
    tt_ids = ttnn.from_torch(
        ids,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )
    return tt_model(tt_ids)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("vocab_size, dim, seq_len", CONFIGS, ids=CONFIG_IDS)
def test_embedding_pcc(mesh_device, vocab_size, dim, seq_len, reset_seeds):
    """Gathered rows match torch.nn.functional.embedding."""
    table = torch.randn(vocab_size, dim, dtype=torch.bfloat16)
    ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.int32)

    # Reference on the bf16 table (not upcast): the device table is bf16 too, so
    # this isolates the gather from a dtype difference.
    reference = F.embedding(ids.long(), table).float()

    tt_out = ttnn.to_torch(_run_embedding(mesh_device, table, ids)).float()

    assert tt_out.shape == reference.shape, f"{tt_out.shape} != {reference.shape}"
    passing, pcc = comp_pcc(reference, tt_out, PCC_THRESHOLD)
    logger.info(f"embedding PCC (V={vocab_size}, D={dim}, T={seq_len}): {pcc}")
    assert passing, f"embedding PCC below {PCC_THRESHOLD}: {pcc}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_embedding_output_contract(mesh_device, reset_seeds):
    """Output is rank-3 [B, T, D], TILE layout, bfloat16 -- what the blocks expect."""
    vocab_size, dim, seq_len = 1024, 512, 32
    table = torch.randn(vocab_size, dim, dtype=torch.bfloat16)
    ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.int32)

    tt_out = _run_embedding(mesh_device, table, ids)

    assert tuple(tt_out.shape) == (1, seq_len, dim), f"unexpected shape {tt_out.shape}"
    assert tt_out.layout == ttnn.TILE_LAYOUT, f"unexpected layout {tt_out.layout}"
    assert tt_out.dtype == ttnn.bfloat16, f"unexpected dtype {tt_out.dtype}"
