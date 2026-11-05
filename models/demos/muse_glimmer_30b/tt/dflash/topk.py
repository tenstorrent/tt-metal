# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-device top-k of vocab-sharded logits (shared by the DFlash2 draft head and the target verify pass)."""

import torch

import ttnn
from models.demos.muse_glimmer_30b.tt.common import as_tt, shard_mapper
from models.demos.muse_glimmer_30b.tt.dflash.selector import merge_device_topk

TOPK_PER_DEVICE = 32  # per-device top-k width (vocab-sharded logits)


def make_vocab_bias(mesh_device, padded_vocab, vocab_size):
    """bf16 [1,1,1,padded_vocab] bias sharded on the last dim: 0 for real columns, -10000 for padding columns."""
    bias = torch.zeros(1, 1, 1, padded_vocab, dtype=torch.bfloat16)
    bias[..., vocab_size:] = -10000.0
    return as_tt(bias, mesh_device, dtype=ttnn.bfloat16, mesh_mapper=shard_mapper(mesh_device, -1))


def device_topk(logits, vocab_bias, k=TOPK_PER_DEVICE):
    """logits: per-device [1,1,32,vocab_local] bf16 (not freed). Returns (vals, ids) [1,1,32,k] per device, local ids.
    Device ops only (traceable)."""
    src = logits
    if logits.memory_config().is_sharded():
        src = ttnn.sharded_to_interleaved(logits, ttnn.DRAM_MEMORY_CONFIG)
    biased = ttnn.add(src, vocab_bias)
    if src is not logits:
        src.deallocate(True)
    vals, ids = ttnn.topk(biased, k=k, dim=-1)
    biased.deallocate(True)
    return vals, ids


def read_topk(vals_tt, ids_tt, rows, *, vocab_local, vocab_size, k):
    """Host: read all device shards, merge. Returns (values [len(rows), k] f32, global ids [len(rows), k] i64),
    sorted descending."""
    rows = list(rows)
    vals = torch.stack([ttnn.to_torch(d).float()[0, 0] for d in ttnn.get_device_tensors(vals_tt)])
    ids = torch.stack([ttnn.to_torch(d).long()[0, 0] for d in ttnn.get_device_tensors(ids_tt)])
    return merge_device_topk(vals[:, rows], ids[:, rows], vocab_local=vocab_local, vocab_size=vocab_size, k=k)
