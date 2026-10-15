# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full MiMo-V2.6-Flash text model on one TP=8 replica (a 1x8 Galaxy row).

Embedding (replicated) -> 48 decoder layers -> final RMSNorm -> LM head (vocab-parallel: each chip
holds 152576 / 8 = 19072 columns; the host concatenates the shards). The paged KV cache has one
K and one V tensor per layer, ``[num_blocks, 1, block_size, 192]`` per chip.

Decode always runs ``max_batch`` slots (default 32). A slot with position -1 is empty: the cache
update and SDPA skip it, and its MoE routing weights are masked to zero so it does not widen the
set of experts computed.
"""

import os
import time

import torch
from loguru import logger

import ttnn

from ..reference.model import load_layer_weights
from .common import TILE, MeshPlan, as_weight, cache_name, compute_config, pad_to
from .decoder import TtMiMoDecoderLayer, TtRMSNorm
from .rope import MiMoRotarySetup


def prefill_padded_len(seq_len):
    """Tokens are padded so the prefill SDPA chunk (32/128/256) divides the length."""
    n = pad_to(seq_len, TILE)
    if n >= 2048:
        return pad_to(n, 256)
    if n >= 128:
        return pad_to(n, 128)
    return n


class TtMiMoModel:
    def __init__(
        self,
        mesh_device,
        cfg,
        ckpt=None,
        cache_dir=None,
        num_layers=None,
        max_seq_len=4096,
        max_batch=32,
        block_size=64,
        attn_dtype=ttnn.bfloat8_b,
        mlp_dtype=ttnn.bfloat8_b,
        expert_dtype=ttnn.bfloat4_b,
        lm_head_dtype=ttnn.bfloat8_b,
        kv_dtype=ttnn.bfloat8_b,
        num_links=1,
    ):
        """``ckpt`` may be None when ``cache_dir`` already holds every tensor (a later DP replica)."""
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.plan = plan = MeshPlan(mesh_device, num_links=num_links)
        self.num_layers = num_layers or cfg.num_hidden_layers
        self.max_seq_len = max_seq_len
        self.max_batch = max_batch
        self.block_size = block_size
        self.kv_dtype = kv_dtype
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

        head_dim = cfg.head_dim
        rope_dim = int(head_dim * cfg.partial_rotary_factor)
        self.rope_setups = {
            "swa": MiMoRotarySetup(mesh_device, max_batch, head_dim, rope_dim, max_seq_len, cfg.swa_rope_theta),
            "full": MiMoRotarySetup(mesh_device, max_batch, head_dim, rope_dim, max_seq_len, cfg.rope_theta),
        }

        def warm(tag):
            return cache_dir is not None and os.path.exists(cache_name(cache_dir, tag, "complete"))

        def mark(tag):
            if cache_dir is not None:
                open(cache_name(cache_dir, tag, "complete"), "w").close()

        top = "top"
        need = not warm(top)
        assert ckpt is not None or not need, "cold cache needs a checkpoint"
        emb = ckpt.embedding().bfloat16() if need else None
        self.embedding = as_weight(
            emb, plan, plan.replicate(), ttnn.bfloat16, cache_name(cache_dir, "embedding"), layout=ttnn.ROW_MAJOR_LAYOUT
        )
        del emb
        self.final_norm = TtRMSNorm(
            plan, ckpt.final_norm() if need else None, cfg.layernorm_epsilon, cache_name(cache_dir, "final_norm")
        )
        lm = ckpt.lm_head().T.contiguous() if need else None
        self.vocab_local = cfg.vocab_size // plan.tp
        self.lm_head = as_weight(lm, plan, plan.shard(-1), lm_head_dtype, cache_name(cache_dir, f"lm_head_tp{plan.tp}"))
        del lm
        mark(top)

        self.layers = []
        for i in range(self.num_layers):
            t0 = time.time()
            tag = f"layer{i}_tp{plan.tp}"
            weights = None if warm(tag) else load_layer_weights(ckpt, cfg, i)
            self.layers.append(
                TtMiMoDecoderLayer(
                    plan,
                    cfg,
                    i,
                    weights,
                    self.rope_setups,
                    cache_dir=cache_dir,
                    attn_dtype=attn_dtype,
                    mlp_dtype=mlp_dtype,
                    expert_dtype=expert_dtype,
                )
            )
            del weights
            mark(tag)
            logger.info(
                f"[{tuple(mesh_device.shape)} id={id(mesh_device):x}] layer {i} ready in {time.time() - t0:.1f}s"
            )

        self.lm_config = compute_config(ttnn.MathFidelity.HiFi2)
        self.kv_caches = None
        self.page_table_tt = None

    # ------------------------------------------------------------------ setup
    def allocate_kv_cache(self, num_blocks=None):
        blocks_per_user = (self.max_seq_len + self.block_size - 1) // self.block_size
        num_blocks = num_blocks or blocks_per_user * self.max_batch
        self.kv_caches = [
            layer.attn.allocate_kv_cache(num_blocks, self.block_size, self.kv_dtype) for layer in self.layers
        ]
        return num_blocks

    def set_page_table(self, page_table):
        """page_table: torch int32 [max_batch, blocks_per_user] (slot -> physical blocks)."""
        assert page_table.shape[0] == self.max_batch
        if self.page_table_tt is not None:
            self.page_table_tt.deallocate(True)
        self.page_table_tt = ttnn.from_torch(
            page_table.to(torch.int32),
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.plan.replicate(),
        )

    def _to_device_tokens(self, tokens):
        return ttnn.from_torch(
            tokens.reshape(1, -1).to(torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.plan.replicate(),
        )

    def _embed(self, tokens):
        tt_tokens = self._to_device_tokens(tokens)
        x = ttnn.embedding(tt_tokens, self.embedding, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_tokens.deallocate(True)
        return ttnn.reshape(x, (1, 1, tokens.numel(), self.cfg.hidden_size))

    def _logits(self, x):
        h = self.final_norm(x)
        logits = ttnn.linear(
            h,
            self.lm_head,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.lm_config,
        )
        h.deallocate(True)
        return logits

    # ------------------------------------------------------------------ prefill
    def prefill_enqueue(self, tokens, slot, return_hidden=False):
        """Enqueue prefill of one user (``tokens`` 1D torch) into batch ``slot``; returns a pending handle."""
        seq_len = tokens.numel()
        padded = prefill_padded_len(seq_len)
        assert padded <= self.max_seq_len
        toks = torch.nn.functional.pad(tokens.to(torch.int32), (0, padded - seq_len))
        x = self._embed(toks)
        hidden = []
        for i, layer in enumerate(self.layers):
            y = layer.forward_prefill(x, slot, self.page_table_tt, self.kv_caches[i])
            x.deallocate(True)
            x = y
            if return_hidden:
                hidden.append(ttnn.clone(x))
        tile_start = ((seq_len - 1) // TILE) * TILE
        last = ttnn.slice(x, (0, 0, tile_start, 0), (1, 1, tile_start + TILE, self.cfg.hidden_size))
        x.deallocate(True)
        logits = self._logits(last)
        last.deallocate(True)
        return {"logits": logits, "row": seq_len - 1 - tile_start, "hidden": hidden, "seq_len": seq_len}

    def prefill_read(self, pending):
        logits = self.plan.to_torch_tp_concat(pending["logits"], dim=-1).float()
        pending["logits"].deallocate(True)
        out = logits.reshape(-1, logits.shape[-1])[pending["row"]]
        hidden = [self.plan.to_torch_replicated(h).float()[0, 0, : pending["seq_len"]] for h in pending["hidden"]]
        return out, hidden

    def prefill(self, tokens, slot, return_hidden=False):
        return self.prefill_read(self.prefill_enqueue(tokens, slot, return_hidden))

    # ------------------------------------------------------------------ decode
    def decode_enqueue(self, tokens, positions):
        """tokens, positions: torch [max_batch]; position -1 marks an empty slot."""
        assert tokens.numel() == self.max_batch and positions.numel() == self.max_batch
        x = self._embed(tokens)
        cur_pos = ttnn.from_torch(
            positions.to(torch.int32),
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.plan.replicate(),
        )
        rope_pos = positions.clamp(min=0)
        rot_mats = {
            True: self.rope_setups["swa"].get_rot_mats(rope_pos),
            False: self.rope_setups["full"].get_rot_mats(rope_pos),
        }
        mask = (positions >= 0).to(torch.bfloat16).reshape(1, 1, -1, 1)
        token_mask = ttnn.from_torch(
            mask,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self.plan.replicate(),
        )
        for i, layer in enumerate(self.layers):
            y = layer.forward_decode(x, rot_mats, cur_pos, self.page_table_tt, self.kv_caches[i], token_mask)
            x.deallocate(True)
            x = y
        logits = self._logits(x)
        x.deallocate(True)
        for t in (cur_pos, token_mask, *rot_mats[True], *rot_mats[False]):
            t.deallocate(True)
        return logits

    def decode_read(self, logits):
        out = self.plan.to_torch_tp_concat(logits, dim=-1).float()
        logits.deallocate(True)
        return out.reshape(self.max_batch, -1)

    def decode(self, tokens, positions):
        return self.decode_read(self.decode_enqueue(tokens, positions))
