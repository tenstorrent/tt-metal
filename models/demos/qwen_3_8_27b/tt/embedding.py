# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Token embedding and LM head.

Embedding: the table is sharded on the hidden dim across the TP columns (``[vocab, hidden/tp]`` per
chip, vocab replicated) — minimax_m3's 1D "emb-on-TP" mode. Each row looks up its own SP token slice,
then a TP all-gather rebuilds the full hidden (the residual stream is TP-replicated). Kept bf16 (the
activation dtype; an embedding is a lookup, not a matmul).

LM head: column-parallel on the vocab (``[hidden, vocab/tp]`` per chip, 62080 = 1940 tiles), weight
dtype per spec. Prefill's product is the KV cache, so the head runs only when logits are asked for.
"""

import torch

import ttnn
from models.demos.qwen_3_8_27b.tt.common import hifi4_fp32, upload


class TtEmbedding:
    def __init__(self, mesh_config, weight: torch.Tensor | None, *, cache=None):
        self.mc = mesh_config
        self.weight = upload(
            None if weight is None else weight.to(torch.bfloat16)[None, None],
            mesh_config.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mapper=mesh_config.shard(None, 3),
            cache=cache,
            name="embed_tokens",
        )

    def make_tokens(self, token_ids: torch.Tensor):
        """``[T]`` int token ids -> device ``[1, T/sp]`` uint32 per chip (row r: tokens [r*s, (r+1)*s))."""
        sp = self.mc.sp
        T = token_ids.numel()
        assert T % (32 * sp) == 0, f"{T} tokens is not a multiple of 32*sp"
        return ttnn.from_torch(
            token_ids.to(torch.int32).reshape(sp, T // sp),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mc.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mc.shard(0, None),
        )

    def __call__(self, tokens):
        w = ttnn.reshape(self.weight, list(self.weight.shape)[-2:])
        x = ttnn.embedding(tokens, w, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [1, s, H/tp]
        x = ttnn.reshape(x, [1, 1, x.shape[-2], x.shape[-1]])
        full = self.mc.all_gather_tp(x, dim=3)
        ttnn.deallocate(x)
        return full


class TtLMHead:
    def __init__(self, mesh_config, weight: torch.Tensor | None, *, weight_dtype, cache=None):
        """weight: HF ``lm_head.weight`` [vocab, hidden]."""
        self.mc = mesh_config
        self.w = upload(
            None if weight is None else weight.T.contiguous()[None, None],
            mesh_config.mesh_device,
            dtype=weight_dtype,
            mapper=mesh_config.shard(None, 3),
            cache=cache,
            name="lm_head",
        )
        self.ckc = hifi4_fp32()

    def __call__(self, x):
        """x [1,1,S,H] -> per-chip logits [1,1,S,vocab/tp] (vocab column-sharded)."""
        return ttnn.linear(x, self.w, compute_kernel_config=self.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
