# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The DFlash drafter has no LM head of its own -- it shares the TARGET's embed_tokens/
lm_head (tie_word_embeddings=true on both the drafter's and the target's own config; no
lm_head.* or embed_tokens.* tensors exist in the drafter's checkpoint, confirmed in
weight_mapping.py). This loads just that one weight directly from the target's real
safetensors shard (mirrors weight_mapping.py's pattern) rather than instantiating the
full 31B target model.

Softcap here is the exact same tanh clamp as models/demos/gemma4/tt/model.py:1275-1279,
applied per vocab-shard before gathering -- elementwise-safe on sharded vocab (see that
file's own comment) so it doesn't need the full-vocab all-gather first.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allgather

DEFAULT_TARGET_MODEL = "google/gemma-4-31B-it"
_EMBED_TOKENS_KEY = "model.language_model.embed_tokens.weight"


def load_target_lm_head_state_dict(model_path: str = DEFAULT_TARGET_MODEL) -> torch.Tensor:
    """[vocab_size, hidden_size] real embed_tokens/lm_head weight (tied), read directly
    from the target's safetensors shard via its index."""
    import json
    import os

    from safetensors import safe_open

    if os.path.isdir(model_path):
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)["weight_map"]
        shard_path = os.path.join(model_path, index[_EMBED_TOKENS_KEY])
    else:
        from huggingface_hub import hf_hub_download

        index_path = hf_hub_download(model_path, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)["weight_map"]
        shard_path = hf_hub_download(model_path, index[_EMBED_TOKENS_KEY])

    with safe_open(shard_path, framework="pt") as f:
        return f.get_tensor(_EMBED_TOKENS_KEY)


def load_gemma4_lm_head_weight(mesh_device, mesh_config, model_path: str = DEFAULT_TARGET_MODEL) -> ttnn.Tensor:
    """Vocab-sharded (column-parallel) ttnn tensor, [1,1,hidden_size,vocab_size/tp] per device."""
    weight = load_target_lm_head_state_dict(model_path)  # [vocab, hidden]
    weight_t = weight.transpose(-2, -1).unsqueeze(0).unsqueeze(0)  # [1,1,hidden,vocab]
    mapper = mesh_config.column_parallel(mesh_device) if mesh_config.tp > 1 else None
    # bfloat16, not bfloat8_b: a 262144-wide vocab has many logits close enough that
    # bfp8's lower precision flips greedy argmax at near-ties (confirmed: 15/16 draft
    # tokens matched exactly at bfp8, one position was a genuine near-tie that flipped).
    return ttnn.as_tensor(
        weight_t,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def compute_dflash_logits(
    hidden: ttnn.Tensor, lm_head_weight: ttnn.Tensor, mesh_device, softcap: float | None
) -> torch.Tensor:
    """final_norm'd hidden -> vocab-sharded logits -> softcap (per-shard) -> gathered to host.
    Returns torch logits [.., vocab_size] (full, gathered) for a simple host argmax.

    Kept for reference/inspection use (e.g. logit-tie debugging); the hot generation path
    uses ``compute_dflash_argmax`` below instead, which never reads the full vocab to host."""
    logits_sharded = ttnn.linear(hidden, lm_head_weight)  # [.., vocab/tp] per device
    if softcap is not None and softcap > 0:
        logits_sharded = ttnn.multiply(logits_sharded, 1.0 / softcap)
        logits_sharded = ttnn.tanh(logits_sharded)
        logits_sharded = ttnn.multiply(logits_sharded, softcap)
    logits = ttnn.to_torch(logits_sharded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    return logits.float()


def argmax_last_dim(logits: ttnn.Tensor, rows: int) -> ttnn.Tensor:
    """On-device argmax over the last (vocab) dim. Returns [1,1,rows] uint32, ON DEVICE --
    no host readback of the (potentially 262144-wide) logits.

    Ported from spec_decode.py's ``SpeculativeDecoder._argmax_last``: ``ttnn.argmax``'s fast
    multicore path is ROW-PARALLEL and only correct when the row (batch) dim is EXACTLY one
    tile (32) -- verified there by a correctness probe. Rows are processed in 32-row chunks
    (pad up to 32, untilize, argmax, slice back) accordingly; DFlash's block_size (16) always
    fits in a single chunk, but this stays general in case that config ever changes."""
    R32 = 32
    if rows > R32:
        vocab = logits.shape[-1]
        chunks = []
        off = 0
        while off < rows:
            n = min(R32, rows - off)
            part = ttnn.slice(logits, [0, 0, off, 0], [1, 1, off + n, vocab])
            chunks.append(argmax_last_dim(part, n))
            ttnn.deallocate(part)
            off += n
        out = ttnn.concat(chunks, dim=2)
        for c in chunks:
            ttnn.deallocate(c)
        return out

    src = logits
    padded = None
    if rows < R32:
        padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, R32 - rows), (0, 0)], value=0.0)
        src = padded
    u = ttnn.untilize(src, use_multicore=True)
    if padded is not None:
        ttnn.deallocate(padded)
    idx = ttnn.argmax(u, dim=-1, keepdim=False)  # [1,1,32 or rows] uint32 ROW_MAJOR
    ttnn.deallocate(u)
    if rows < R32:
        sliced = ttnn.slice(idx, [0, 0, 0], [1, 1, rows])
        ttnn.deallocate(idx)
        idx = sliced
    return idx


def compute_dflash_argmax(
    hidden: ttnn.Tensor, lm_head_weight: ttnn.Tensor, mesh_device, mesh_config, ccl_manager, softcap: float | None
) -> ttnn.Tensor:
    """final_norm'd hidden -> vocab-sharded logits -> softcap (per-shard) -> on-device
    all-gather -> on-device argmax. Returns [1,1,rows] uint32 token ids, ON DEVICE -- the
    only host readback needed afterward is that small id tensor, not the full vocab."""
    logits_sharded = ttnn.linear(hidden, lm_head_weight)  # [.., vocab/tp] per device
    if softcap is not None and softcap > 0:
        logits_sharded = ttnn.multiply(logits_sharded, 1.0 / softcap)
        logits_sharded = ttnn.tanh(logits_sharded)
        logits_sharded = ttnn.multiply(logits_sharded, softcap)
    logits_full = ccl_allgather(logits_sharded, mesh_config, ccl_manager)
    rows = logits_full.shape[-2]
    return argmax_last_dim(logits_full, rows)
