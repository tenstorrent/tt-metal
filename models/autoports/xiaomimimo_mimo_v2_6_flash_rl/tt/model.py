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
from . import kv as kv_lib
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
            "swa": MiMoRotarySetup(mesh_device, max_batch, head_dim, rope_dim, cfg.swa_rope_theta),
            "full": MiMoRotarySetup(mesh_device, max_batch, head_dim, rope_dim, cfg.rope_theta),
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
            # Bump a layer's tag when its cached tensor set changes, so a warm cache from an older layout
            # is rebuilt instead of loaded with tensors missing (global layers gained grouped decode weights).
            tag = f"layer{i}_tp{plan.tp}" + ("" if cfg.is_swa(i) else "_kvsplit")
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
        self.decode_trace = None

    # ------------------------------------------------------------------ setup
    def setup_kv_cache(self, num_users):
        """Allocate paged KV for ``num_users`` users in decode slots [0, num_users).

        SWA layers get a 512-token ring per user, global layers the full ``max_seq_len``; on TP=8 a global
        layer's chip pair splits the users (kv.py), so that page table differs per chip. Slots past
        ``num_users`` are never written (their position is -1)."""
        assert num_users <= self.max_batch
        self.kv_caches, tables = [], {}
        for layer in self.layers:
            per_user = layer.attn.kv_tokens_per_user(self.max_seq_len)
            assert per_user % self.block_size == 0
            table, num_blocks = kv_lib.page_tables(
                num_users, self.max_batch, per_user // self.block_size, self.plan.tp, layer.attn.rep
            )
            self.kv_caches.append(layer.attn.allocate_kv_cache(num_blocks, self.block_size, self.kv_dtype))
            tables.setdefault(layer.is_swa, table)
        self.set_page_tables(tables)

    def set_page_tables(self, tables):
        """tables: {is_swa: int32 [tp, max_batch, blocks_per_user]} (per chip slot -> physical block)."""
        if self.page_table_tt is not None:
            for t in self.page_table_tt.values():
                t.deallocate(True)
        self.page_table_tt = {
            is_swa: kv_lib.per_chip(self.plan, table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            for is_swa, table in tables.items()
        }

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
            y = layer.forward_prefill(x, slot, self.page_table_tt, self.kv_caches[i], real_len=seq_len)
            x.deallocate(True)
            x = y
            if return_hidden:
                hidden.append(ttnn.clone(x))
        tile_start = ((seq_len - 1) // TILE) * TILE
        if padded == TILE:
            last = x  # a slice covering the whole tensor shares its buffer, so it must not be freed separately
        else:
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
    def prepare_decode_inputs_host(self, tokens, positions):
        """Host tensors for one step: tokens [1, B] u32, cur_pos [B] i32, cos/sin [1, 1, B, 192] for the SWA and
        the global theta, MoE token mask,
        and the global layers' per-chip view: owned positions [B] i32 and owner mask [1, 1, B, 1]."""
        assert tokens.numel() == self.max_batch and positions.numel() == self.max_batch
        rep = self.plan.replicate()
        tp, kv_rep = self.plan.tp, kv_lib.kv_replication(self.plan.tp, self.cfg.num_key_value_heads)

        def host(t, dtype, layout):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=rep)

        return (
            host(tokens.reshape(1, -1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            host(positions.to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            *self.rope_setups["swa"].decode_host_inputs(positions),
            *self.rope_setups["full"].decode_host_inputs(positions),
            host((positions >= 0).to(torch.bfloat16).reshape(1, 1, -1, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            kv_lib.per_chip(
                self.plan, kv_lib.owned_positions(positions, tp, kv_rep), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, False
            ),
            kv_lib.per_chip(
                self.plan, kv_lib.owner_mask(positions, tp, kv_rep), ttnn.bfloat16, ttnn.TILE_LAYOUT, False
            ),
        )

    def decode_forward_device(
        self, tokens_tt, cur_pos_tt, cos_swa, sin_swa, cos_full, sin_full, mask_tt, pos_owned_tt, owner_mask_tt
    ):
        """One decode step from device-resident inputs only (trace-capturable). Returns [1, 1, B, vocab / tp]."""
        x = ttnn.embedding(tokens_tt, self.embedding, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        x = ttnn.reshape(x, (1, 1, self.max_batch, self.cfg.hidden_size))
        rot_mats = {
            True: self.rope_setups["swa"].decode_rot_mats(cos_swa, sin_swa),
            False: self.rope_setups["full"].decode_rot_mats(cos_full, sin_full),
        }
        positions = {"pos": cur_pos_tt, "pos_owned": pos_owned_tt, "owner_mask": owner_mask_tt}
        for i, layer in enumerate(self.layers):
            y = layer.forward_decode(x, rot_mats, positions, self.page_table_tt, self.kv_caches[i], mask_tt)
            x.deallocate(True)
            x = y
        logits = self._logits(x)
        x.deallocate(True)
        for t in (*rot_mats[True], *rot_mats[False]):
            t.deallocate(True)
        return logits

    def capture_decode_trace(self):
        """Compile, then capture one decode step. Later decode calls refresh the inputs and replay it.

        Warm-up and capture run with every slot empty (position -1), so they write no KV-cache entries
        and are safe to run after users have been prefilled."""
        assert self.decode_trace is None
        empty = torch.full((self.max_batch,), -1, dtype=torch.int32)
        host = self.prepare_decode_inputs_host(torch.zeros(self.max_batch, dtype=torch.int32), empty)
        inputs = [ttnn.to_device(t, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in host]
        self.decode_forward_device(*inputs).deallocate(True)
        ttnn.synchronize_device(self.mesh_device)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        output = self.decode_forward_device(*inputs)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        self.decode_trace = {"id": trace_id, "inputs": inputs, "output": output}

    def release_decode_trace(self):
        if self.decode_trace is not None:
            ttnn.release_trace(self.mesh_device, self.decode_trace["id"])
            self.decode_trace = None

    def decode_enqueue(self, tokens, positions):
        """tokens, positions: torch [max_batch]; position -1 marks an empty slot. Non-blocking."""
        host = self.prepare_decode_inputs_host(tokens, positions)
        if self.decode_trace is not None:
            for h, d in zip(host, self.decode_trace["inputs"]):
                ttnn.copy_host_to_device_tensor(h, d)
            ttnn.execute_trace(self.mesh_device, self.decode_trace["id"], cq_id=0, blocking=False)
            return self.decode_trace["output"]
        inputs = [ttnn.to_device(t, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in host]
        logits = self.decode_forward_device(*inputs)
        for t in inputs:
            t.deallocate(True)
        return logits

    def decode_read(self, logits):
        out = self.plan.to_torch_tp_concat(logits, dim=-1).float()
        if self.decode_trace is None or logits is not self.decode_trace["output"]:
            logits.deallocate(True)
        return out.reshape(self.max_batch, -1)

    def decode(self, tokens, positions):
        return self.decode_read(self.decode_enqueue(tokens, positions))
