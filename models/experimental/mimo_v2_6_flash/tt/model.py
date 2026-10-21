# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full MiMo-V2.6-Flash text model on one TP=8 replica (a 1x8 Galaxy row).

Embedding (replicated) -> 48 decoder layers -> final RMSNorm -> LM head (vocab-parallel: each chip
holds 152576 / 8 = 19072 columns; the host concatenates the shards). The paged KV cache has one
K and one V tensor per layer, ``[num_blocks, 1, block_size, 192]`` per chip (V 128 wide on the global
layers).

Prefill runs one user at a time, in chunks of at most ``max_prefill_chunk`` tokens; each chunk attends to
the earlier ones through the KV cache (attention.py). A prefill can also start at ``start_pos > 0`` when
the cache already holds the user's first ``start_pos`` tokens.

Decode always runs ``max_batch`` slots (default 32). A slot with position -1 is empty: the cache
update and SDPA skip it, and its MoE routing weights are masked to zero so it does not widen the
set of experts computed.
"""

import contextlib
import os
import time

import torch
from loguru import logger

import ttnn

from ..reference.model import load_layer_weights
from ..reference.spec_decode import SPEC_SLOTS, serial_positions
from . import kv as kv_lib
from .attention import TtMiMoAttention
from .common import TILE, MeshPlan, as_weight, cache_name, compute_config, pad_to
from .decoder import TtMiMoDecoderLayer, TtRMSNorm
from .greedy_head import GreedyHead
from .moe import expert_quant
from .moe_stream import zero_output
from .mtp import TtMiMoMTP
from .rope import MiMoRotarySetup, partial_rope_cos_sin


def prefill_padded_len(seq_len):
    """Tokens are padded so the prefill SDPA chunk (32/128/256) divides the length: 32, 64, 96, then multiples of 128,
    then (from 2048) of 256. Every prefill program keys on one of these lengths (attention.py), so a chunk is padded
    to one even at the end of the context: the padding past the page table lands in a scratch block
    (kv.PrefillTables)."""
    n = pad_to(seq_len, TILE)
    if n >= 2048:
        return pad_to(n, 256)
    if n >= 128:
        return pad_to(n, 128)
    return n


def prefill_warmup_chunks(max_tokens, max_seq_len, block_size, window):
    """(tokens, start) of chunks that together run every prefill program a chunk of up to ``max_tokens`` tokens can:
    one per padded length at start 0, and at the later starts that bring something the programs key on (attention.py)
    that an earlier start did not: the number of window blocks a SWA layer reads, or the SDPA chunk size that divides
    the start (TtMiMoAttention._prefill_sdpa_program, which global layers run chunked MLA prefill with). Only chunks
    that fit in the context are listed."""
    shortest = {}  # padded length -> the fewest tokens padded to it
    for n in range(1, max_tokens + 1):
        shortest.setdefault(prefill_padded_len(n), n)
    out = []
    for padded, n in sorted(shortest.items()):
        if n > max_seq_len:
            continue
        out.append((n, 0))
        leads, chunks = set(), set()
        # The window blocks stop changing past the window; the SDPA chunk sizes (at most 256) repeat every 256.
        for start in range(block_size, window + 256 + 1, block_size):
            if start + n > max_seq_len:
                break
            lead = start - max(0, start - window) // block_size * block_size
            chunk = TtMiMoAttention._prefill_sdpa_program(None, padded, start).q_chunk_size
            if lead not in leads or chunk not in chunks:
                leads.add(lead)
                chunks.add(chunk)
                out.append((n, start))
    return out


def prefill_warmup_batches(max_tokens, max_users, max_padded, max_seq_len):
    """(users, tokens) of first-chunk batches (TtMiMoModel.prefill_enqueue_batch) that together run every program a
    batch of 1 to ``max_users`` users and ``max_tokens`` real tokens can: one per user count and padded length of at
    most ``max_padded``, each user with the fewest tokens padded to that length. Batch programs key on those two only
    (every chunk starts at 0)."""
    shortest = {}
    for n in range(1, min(max_tokens, max_padded, max_seq_len) + 1):
        shortest.setdefault(prefill_padded_len(n), n)
    return [
        (users, n)
        for users in range(1, max_users + 1)
        for padded, n in sorted(shortest.items())
        if padded <= max_padded and users * n <= max_tokens
    ]


class TtMiMoModel:
    DECODE_TOPK = 32  # candidates per chip read back each decode step
    DECODE_HISTORY = 64  # device-resident greedy steps between token reads
    LAYER_CACHE_LAYOUT = "kvsplit-gu-ds"  # split-user global KV, fused gate|up, DRAM-sharded qkv/router

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
        max_prefill_chunk=4096,
        attn_dtype=ttnn.bfloat8_b,
        mlp_dtype=ttnn.bfloat8_b,
        expert_dtype=ttnn.bfloat4_b,
        lm_head_dtype=ttnn.bfloat8_b,
        kv_dtype=ttnn.bfloat8_b,
        logits_dtype=None,
        num_links=2,
    ):
        """``ckpt`` may be None when ``cache_dir`` already holds every tensor (a later DP replica).
        ``max_prefill_chunk``: the default prefill chunk (a multiple of ``block_size``).
        ``logits_dtype``: fp32 keeps near-tied logits apart; in bf16 (a step of 0.125 at the usual top logit of
        16-32) they often tie exactly and the lowest token id wins. Default fp32 (MIMO_LOGITS_FP32=0: bf16)."""
        assert max_prefill_chunk % block_size == 0
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.plan = plan = MeshPlan(mesh_device, num_links=num_links)
        self.num_layers = num_layers or cfg.num_hidden_layers
        self.max_seq_len = max_seq_len
        self.max_batch = max_batch
        self.block_size = block_size
        self.max_prefill_chunk = max_prefill_chunk
        self.kv_dtype = kv_dtype
        # Width of the context table a later prefill chunk of a global layer reads (kv.PrefillTables.context): every
        # block up to the context end plus the most padding a chunk gets, as whole 32-byte rows of the page table.
        self.prefill_context_blocks = pad_to(-(-(max_seq_len + kv_lib.PREFILL_MAX_PAD) // block_size), 8)
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

        # Prepared bfloat4_b experts are their own cache files (moe.TtMiMoMoE), so their own layer tag.
        quant = expert_quant(expert_dtype)
        expert_tag = f"_{quant}" if quant else ""
        self.layers = []
        for i in range(self.num_layers):
            t0 = time.time()
            # Bump LAYER_CACHE_LAYOUT when a layer's cached tensor set changes, so a warm cache from an older
            # layout is rebuilt instead of loaded with tensors missing. Unchanged tensors (e.g. experts) are
            # still loaded from their cache files; lazy experts are only dequantized when they are missing.
            tag = f"layer{i}_tp{plan.tp}_{self.LAYER_CACHE_LAYOUT}{expert_tag}"
            weights = None if warm(tag) else load_layer_weights(ckpt, cfg, i, lazy_experts=True)
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

        if logits_dtype is None:
            logits_dtype = ttnn.float32 if os.environ.get("MIMO_LOGITS_FP32", "1") != "0" else ttnn.bfloat16
        self.logits_dtype = logits_dtype
        # HiFi2 with fp32 accumulation: the bfloat8_b LM head weights, not the fidelity, bound the logit error.
        self.lm_config = compute_config(ttnn.MathFidelity.HiFi2)
        self.kv_caches = None
        self.page_table_tt = None
        # Host copies of the model's own page tables ({is_swa: int32 [tp, max_batch, blocks]}) and each one's scratch
        # block, which kv.PrefillTables builds a chunk's block tables from (the SWA layers' copy is swa_table's).
        self.page_table_host = None
        self.table_scratch = None
        self.decode_trace = None
        self.verify_trace = None  # speculative verify step (capture_verify_trace)
        self.mtp = None  # the MTP drafter (setup_mtp)
        self.greedy_head = None
        self.prefill_traces = {}  # (users, padded length) -> traced prefill batch (setup_prefill_batch_trace)
        self.rope_tables = None  # decode cos / sin of every position (setup_decode_rope_tables)
        self.decode_users = None
        self.norm_buffers = None
        self.swa_table = None
        self.swa_qkv = None
        # vLLM-owned KV (setup_vllm_kv_cache): each layer's KV cache group, and the host copy of every group's table.
        self.layer_groups = None
        self.vllm_tables = None
        # Chip c's vocab shard starts at c * vocab_local: added to its local top-k ids for the device greedy.
        offsets = (torch.arange(plan.tp, dtype=torch.int32) * self.vocab_local).reshape(plan.tp, 1, 1, 1)
        self.vocab_offset = ttnn.from_torch(
            offsets.expand(plan.tp, 1, max_batch, self.DECODE_TOPK).contiguous(),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=plan.shard(0),
        )

    # ------------------------------------------------------------------ setup
    def setup_kv_cache(self, num_users):
        """Allocate paged KV for ``num_users`` users in decode slots [0, num_users).

        Every page table is indexed by absolute position over ``max_seq_len``. Global layers back every block; on
        TP=8 a global layer's chip pair splits the users (kv.py), so that page table differs per chip. SWA layers
        back only the blocks a sliding window can still read (kv.SwaBlockTable, shared by the SWA layers and
        updated before each prefill and decode step). Slots past ``num_users`` are never written (position -1)."""
        assert num_users <= self.max_batch
        assert self.max_seq_len % self.block_size == 0
        # The MoE streams, GreedyHead, SwaDecodeQkv and the SWA block pools are sized for the decode users once; a new
        # cache drops the old one, which a decode trace would still address.
        assert self.decode_users in (None, num_users), f"decode users already set to {self.decode_users}"
        assert self.decode_trace is None, "release the decode trace before reallocating the KV cache"
        if self.kv_caches is not None:
            for kv in self.kv_caches:
                for t in kv:
                    t.deallocate(True)
        self.decode_users = num_users
        for layer in self.layers:
            if layer.is_moe:
                layer.mlp.set_decode_users(num_users)
        self.kv_caches, tables = [], {}
        max_blocks = self.max_seq_len // self.block_size
        swa_layers = [layer for layer in self.layers if layer.is_swa]
        self.swa_table = None
        if swa_layers:
            window = swa_layers[0].attn.sliding_window
            self.swa_table = kv_lib.SwaBlockTable(
                num_users,
                self.max_batch,
                max_blocks,
                self.block_size,
                window,
                kv_lib.swa_blocks_per_user(window, self.block_size),
                # Without SwaDecodeQkv the SWA layers update their cache with paged_update_cache, which needs a cache
                # at least as wide as the table.
                min_blocks=0 if self._uses_swa_qkv() else max_blocks,
            )
        self.table_scratch = {}
        for layer in self.layers:
            if layer.is_swa:
                table, num_blocks = self.swa_table.host_table(self.plan.tp), self.swa_table.num_blocks
                scratch = 0
            else:
                table, num_blocks = kv_lib.page_tables(
                    num_users, self.max_batch, max_blocks, self.plan.tp, layer.attn.rep
                )
                # kv.page_tables adds one scratch block (the last) when a chip pair splits the users; with rep 1 every
                # block is a user's, so nothing may land past the table.
                scratch = num_blocks - 1 if layer.attn.rep > 1 else None
            self.kv_caches.append(layer.attn.allocate_kv_cache(num_blocks, self.block_size, self.kv_dtype))
            tables.setdefault(layer.is_swa, table)
            self.table_scratch.setdefault(layer.is_swa, scratch)
        self.set_page_tables(tables)
        if self.swa_table is not None:
            self.swa_table.dirty = False
        self._setup_fused_norms()

    def setup_vllm_kv_cache(self, num_users, kv_caches, layer_groups, table_width):
        """Adopt KV caches and page tables that vLLM owns, for ``num_users`` users in decode slots [0, num_users).

        ``kv_caches``: [K, V] per layer, allocated by the caller. ``layer_groups``: each layer's vLLM KV cache group;
        the layers of a group share one page table [max_batch, table_width] (``set_vllm_page_tables``), indexed by
        absolute position. A SWA group's table is vLLM's SlidingWindowSpec table: the blocks no window can read are
        vLLM's null block 0, where this model's own tables point at their scratch block (kv.SwaBlockTable).

        Every chip gets the same tables. vLLM can hand one block to requests in slots of either parity (prefix
        caching), so a global layer's chip pair keeps both copies of its KV head: decode writes every slot's K / V on
        both chips (``pos_update``) and still splits the attention by slot."""
        assert num_users <= self.max_batch
        assert self.decode_users in (None, num_users), f"decode users already set to {self.decode_users}"
        assert self.decode_trace is None and self.kv_caches is None, "the KV cache is already set up"
        assert len(kv_caches) == len(layer_groups) == len(self.layers)
        self.decode_users = num_users
        for layer in self.layers:
            if layer.is_moe:
                layer.mlp.set_decode_users(num_users)
        self.kv_caches = kv_caches
        self.layer_groups = list(layer_groups)
        empty = torch.zeros(self.max_batch, table_width, dtype=torch.int32)
        self.vllm_tables = {g: empty.clone() for g in sorted(set(self.layer_groups))}
        # Decode's device-sampled greedy token (decode_forward_device); persistent buffers, made before any capture.
        self.greedy_head = GreedyHead(self.plan, self.vocab_local, self.decode_users, self.max_batch)
        self.page_table_tt = {
            g: ttnn.from_torch(
                empty,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.plan.replicate(),
            )
            for g in self.vllm_tables
        }
        self._setup_fused_norms()

    def set_vllm_page_tables(self, rows):
        """rows: {slot: {group: int32 [<= table_width] block ids}} (vLLM's table row of the user in that slot).
        Copies every group table that changed to the device, in queue order with the work already enqueued."""
        changed = set()
        for slot, tables in rows.items():
            for g, row in tables.items():
                host = self.vllm_tables[g]
                assert row.numel() <= host.shape[1], f"group {g}: table row of {row.numel()} > {host.shape[1]}"
                new = torch.zeros(host.shape[1], dtype=torch.int32)
                new[: row.numel()] = row.to(torch.int32)
                if not torch.equal(new, host[slot]):
                    host[slot] = new
                    changed.add(g)
        for g in sorted(changed):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    self.vllm_tables[g],
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=self.plan.replicate(),
                ),
                self.page_table_tt[g],
            )

    def _layer_page_table(self, i):
        """Layer ``i``'s page table: {is_swa: table} for the model's own tables, else its vLLM group's table."""
        return self.page_table_tt if self.layer_groups is None else self.page_table_tt[self.layer_groups[i]]

    def _table_key(self, i):
        """Layer ``i``'s page table: its vLLM KV cache group, else is_swa (the model's own tables)."""
        return self.layers[i].is_swa if self.layer_groups is None else self.layer_groups[i]

    def _host_table(self, key):
        """(host copy int32 [tp, max_batch, W], scratch block) of the page table ``key`` (``_table_key``)."""
        tp = self.plan.tp
        if self.layer_groups is not None:
            return self.vllm_tables[key].unsqueeze(0).expand(tp, -1, -1), 0  # vLLM's null block
        if key:
            return self.swa_table.table.unsqueeze(0).expand(tp, -1, -1), 0
        return self.page_table_host[key], self.table_scratch[key]

    def _window_blocks(self, key, slot, start_pos):
        """Physical blocks of positions [window start, start_pos) of ``slot`` in the SWA page table ``key``: the keys a
        later SWA chunk reads (kv.SwaBlockTable.window_blocks)."""
        window = next(layer.attn.sliding_window for layer in self.layers if layer.is_swa)
        first, end = max(0, start_pos - window) // self.block_size, start_pos // self.block_size
        blocks = [int(b) for b in self._host_table(key)[0][0, slot, first:end]]
        assert 0 not in blocks, f"slot {slot}: the SWA window before {start_pos} is not in page table {key}"
        return blocks

    def _prefill_tables(self, slot, start_pos, padded, layers):
        """({page table: kv.PrefillTables}, start tensor or None) of one chunk, for the tables ``layers`` read."""
        later = start_pos > 0
        start = None
        if later:
            start = kv_lib.replicated(self.plan, torch.tensor([start_pos], dtype=torch.int32), ttnn.int32)
        tables = {}
        for i in layers:
            key = self._table_key(i)
            if key in tables:
                continue
            swa = self.layers[i].is_swa
            table, scratch = self._host_table(key)
            tables[key] = kv_lib.PrefillTables(
                self.plan,
                table,
                slot,
                start_pos,
                padded,
                self.block_size,
                scratch,
                context_blocks=self.prefill_context_blocks if later and not swa else None,
                window=self._window_blocks(key, slot, start_pos) if later and swa else None,
                start_tensor=start,
            )
        return tables, start

    def prefill_warmup_layers(self):
        """The first layer of each kind (attention type, dense MLP or MoE). Layers of one kind run the same prefill
        programs, so a chunk through these compiles everything a chunk of its length and start runs."""
        kinds = {}
        for i, layer in enumerate(self.layers):
            kinds.setdefault((layer.is_swa, layer.is_moe), i)
        return sorted(kinds.values())

    def _sync_swa_table(self):
        """Copy the SWA block table to its device tensor in place if it changed (ordered with the queued work, so
        it lands between trace replays)."""
        if self.swa_table is None or not self.swa_table.dirty:
            return
        host = kv_lib.per_chip(
            self.plan, self.swa_table.host_table(self.plan.tp), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, False
        )
        ttnn.copy_host_to_device_tensor(host, self.page_table_tt[True])
        self.swa_table.dirty = False

    def _advance_swa(self, positions):
        """Before a decode step at ``positions`` ([max_batch], -1 = idle slot): blocks for the written positions."""
        if self.swa_table is None:
            return
        for slot, p in enumerate(positions.tolist()):
            if p >= 0:
                self.swa_table.advance(slot, int(p))
        self._sync_swa_table()

    def _setup_fused_norms(self):
        """Let every compact-decode layer's block reductions also compute the following RMSNorm (the post-attention
        norm, then the next layer's input norm or the final norm), through two persistent buffers."""
        if self.norm_buffers is None:
            self.norm_buffers = (
                zero_output(self.plan, self.cfg.hidden_size),
                zero_output(self.plan, self.cfg.hidden_size),
            )
        for i, layer in enumerate(self.layers):
            if layer.attn_reduce is None and self.decode_users <= 4 and layer.can_compact_attention():
                layer.set_compact_attention(self.plan)
            if layer.fused_norm is not None or not layer.can_fuse_norms():
                continue
            nxt = self.layers[i + 1].input_norm if i + 1 < len(self.layers) else self.final_norm
            layer.set_fused_norms(self.plan, nxt, self.norm_buffers)
        self._setup_swa_qkv()

    def _uses_swa_qkv(self):
        """Every SWA layer decodes through the shared SwaDecodeQkv: compact decode (the BFP4 streaming MoE, at most 4
        users) and not turned off (MIMO_SWA_QKV=0 keeps the ttnn ops)."""
        if os.environ.get("MIMO_SWA_QKV", "1") != "1" or self.decode_users > 4:
            return False
        return all(layer.can_fuse_norms() for layer in self.layers if layer.is_swa)

    def _setup_swa_qkv(self):
        """One SwaDecodeQkv shared by the SWA layers (see ``_uses_swa_qkv``)."""
        if not self._uses_swa_qkv():
            return
        for layer in self.layers:
            if not layer.is_swa or layer.attn.swa_qkv is not None:
                continue
            if self.swa_qkv is None:
                self.swa_qkv = layer.attn.make_swa_qkv(self.decode_users, self.block_size)
            layer.attn.swa_qkv = self.swa_qkv

    def set_page_tables(self, tables):
        """tables: {is_swa: int32 [tp, max_batch, blocks_per_user]} (per chip slot -> physical block)."""
        if self.page_table_tt is not None:
            for t in self.page_table_tt.values():
                t.deallocate(True)
        self.page_table_host = dict(tables)
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

    def _logits(self, x, h=None, prefill=False):
        """``h``: final_norm(x) when the last layer's reduction already computed it (a persistent buffer).
        ``prefill``: x holds prefill rows (the final norm takes its row-invariant prefill form, TtRMSNorm)."""
        own_h = h is None
        if own_h:
            h = self.final_norm(x, prefill=prefill)
        logits = ttnn.linear(
            h,
            self.lm_head,
            dtype=self.logits_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.lm_config,
        )
        if own_h:
            h.deallocate(True)
        return logits

    # ------------------------------------------------------------------ prefill
    def _prefill_chunk(self, tokens, slot, start_pos, hidden, layers=None):
        """One chunk (``tokens`` 1D torch) at ``start_pos`` through the layers ``layers`` (indices, default all);
        returns the last one's output [1, 1, padded, hidden] and appends (copy, real rows) of every layer's output to
        ``hidden`` when given."""
        n = tokens.numel()
        padded = prefill_padded_len(n)
        assert start_pos + n <= self.max_seq_len, f"chunk ends at {start_pos + n} > {self.max_seq_len}"
        layers = range(len(self.layers)) if layers is None else layers
        if self.swa_table is not None:
            self.swa_table.prefill(slot, start_pos, start_pos + n, start_pos + padded)
            self._sync_swa_table()
        tables, start = self._prefill_tables(slot, start_pos, padded, layers)
        for setup in self.rope_setups.values():
            setup.begin_prefill_chunk()
        x = self._embed(torch.nn.functional.pad(tokens.to(torch.int32), (0, padded - n)))
        with self._prefill_reduce(padded):
            for k, i in enumerate(layers):
                y = self.layers[i].forward_prefill(
                    x,
                    slot,
                    self._layer_page_table(i),
                    self.kv_caches[i],
                    real_len=n,
                    start_pos=start_pos,
                    tables=tables[self._table_key(i)],
                )
                x.deallocate(True)
                x = y
                if hidden is not None:
                    hidden[k].append((ttnn.clone(x), n))
        # Enqueued: freeing them now is ordered with the work that reads them.
        for setup in self.rope_setups.values():
            setup.end_prefill_chunk()
        for t in tables.values():
            t.release()
        if start is not None:
            start.deallocate(True)
        return x, padded

    def prefill_enqueue(self, tokens, slot, return_hidden=False, start_pos=0, chunk_size=None, layers=None):
        """Enqueue prefill of one user (``tokens`` 1D torch, positions start_pos ..) into batch ``slot``, in chunks
        of ``chunk_size`` tokens (default ``max_prefill_chunk``; a multiple of the block size, as is ``start_pos``).
        ``start_pos > 0`` continues a user whose first start_pos tokens are in the cache. ``layers``: run only these
        layers (indices; warmup). Nothing is read back; returns a pending handle for ``prefill_read``."""
        n = tokens.numel()
        chunk = chunk_size or self.max_prefill_chunk
        assert n > 0 and chunk % self.block_size == 0 and start_pos % self.block_size == 0
        hidden = [[] for _ in (range(len(self.layers)) if layers is None else layers)] if return_hidden else None
        for s in range(0, n, chunk):
            x, padded = self._prefill_chunk(tokens[s : s + chunk], slot, start_pos + s, hidden, layers)
            if s + chunk < n:
                x.deallocate(True)
        seq_len = n - s
        tile_start = ((seq_len - 1) // TILE) * TILE
        if padded == TILE:
            last = x  # a slice covering the whole tensor shares its buffer, so it must not be freed separately
        else:
            # A tensor-args slice: keyed on the padded length, not on the row the last token is in.
            at = kv_lib.replicated(self.plan, torch.tensor([0, 0, tile_start, 0], dtype=torch.int32), ttnn.uint32)
            last = ttnn.slice(x, at, at, slice_dim=2, num_devices=padded // TILE)
            at.deallocate(True)
            x.deallocate(True)
        logits = self._logits(last, prefill=True)
        last.deallocate(True)
        return {"logits": logits, "row": seq_len - 1 - tile_start, "hidden": hidden}

    # Longest padded length a chunk of a prefill batch can have (``prefill_enqueue_batch``): shorter chunks dispatch
    # as many ops as long ones, so batching them is what pays. MIMO_PREFILL_BATCH=0 turns batching off.
    PREFILL_BATCH_MAX_PADDED = 512

    def _prefill_reduce(self, padded, users=1):
        """The row reductions of a prefill pass of ``users`` chunks padded to ``padded``. A chunk short enough to be
        batched reduces as it would alone whether it runs in a batch or not (MeshPlan.prefill_reduce), so a user's
        prefill does not depend on what it was batched with; longer chunks always run alone."""
        if padded <= self.PREFILL_BATCH_MAX_PADDED:
            return self.plan.prefill_reduce(users, padded)
        return contextlib.nullcontext()

    def prefill_enqueue_batch(self, chunks, layers=None):
        """First chunks of one or more users in one pass. ``chunks``: [(tokens 1D torch, slot[, cached blocks])], each
        from position 0, every one padded to the longest one's padded length S (at most PREFILL_BATCH_MAX_PADDED). The users' rows
        run user after user ([1, 1, R * S, hidden]) through the norms, projections and MoE, and through attention as
        a batch of R sequences. Replays the pass's trace when one was captured (``capture_prefill_batch_trace``).
        Only with vLLM's page tables (``setup_vllm_kv_cache``). Returns a pending handle for ``prefill_read``, whose
        logits have one row per user."""
        assert self.layer_groups is not None, "a prefill batch writes through vLLM's page tables"
        num, padded = len(chunks), max(prefill_padded_len(c[0].numel()) for c in chunks)
        assert padded <= self.PREFILL_BATCH_MAX_PADDED and padded <= self.max_seq_len, padded
        trace = self.prefill_traces.get((num, padded)) if layers is None else None
        if trace is not None and trace["id"] is not None:
            return self._replay_prefill_batch(trace, chunks)
        layers = range(len(self.layers)) if layers is None else layers
        keys = list(dict.fromkeys(self._table_key(i) for i in layers))
        tokens, fills, starts, picks = self._prefill_batch_host(chunks, padded, keys)
        rows = kv_lib.replicated(self.plan, torch.arange(num, dtype=torch.int32), ttnn.uint32)
        tables = {
            k: kv_lib.BatchFillTables(kv_lib.per_chip(self.plan, fills[k], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), rows)
            for k in keys
        }
        ats = [kv_lib.replicated(self.plan, torch.tensor([0, 0, s, 0], dtype=torch.int32), ttnn.uint32) for s in starts]
        picks_tt = kv_lib.replicated(self.plan, torch.tensor(picks, dtype=torch.int32).reshape(1, -1), ttnn.uint32)
        tokens_tt = self._to_device_tokens(tokens)
        for setup in self.rope_setups.values():
            setup.begin_prefill_chunk()
        logits = self._prefill_batch_device(tokens_tt, tables, ats, picks_tt, num, padded, layers)
        for setup in self.rope_setups.values():
            setup.end_prefill_chunk()
        # Enqueued: freeing them now is ordered with the work that reads them.
        for t in [*tables.values()]:
            t.release()
        for t in [rows, tokens_tt, picks_tt, *ats]:
            t.deallocate(True)
        return {"logits": logits, "row": list(range(num)), "hidden": None}

    def _prefill_batch_host(self, chunks, padded, keys):
        """Host inputs of a prefill batch: tokens int32 [R * S]; per page table key its fill rows [tp, R, blocks];
        each user's last-token tile start in the pass's rows; each user's row in the logits. A chunk's optional third
        entry is how many of its first blocks the pass must not write (cached by a prefix hit): their fill entries
        are vLLM's null block 0."""
        tokens = torch.cat([torch.nn.functional.pad(c[0].to(torch.int32), (0, padded - c[0].numel())) for c in chunks])
        slots = [c[1] for c in chunks]
        fills = {}
        for key in keys:
            table, scratch = self._host_table(key)
            fills[key] = kv_lib.batch_fill_rows(table, slots, padded, self.block_size, scratch)
            for r, c in enumerate(chunks):
                if len(c) > 2 and c[2]:
                    fills[key][:, r, : c[2]] = 0
        starts = [r * padded + (c[0].numel() - 1) // TILE * TILE for r, c in enumerate(chunks)]
        picks = [r * TILE + (c[0].numel() - 1) % TILE for r, c in enumerate(chunks)]
        return tokens, fills, starts, picks

    def _prefill_batch_device(self, tokens_tt, tables, ats, picks_tt, num, padded, layers):
        """A prefill batch from device inputs only (trace-capturable when the RoPE mats are pinned): tokens [1, R * S]
        uint32, {page table key: kv.BatchFillTables}, each user's last-token tile start (uint32 [4] slice starts), and
        each user's last-token row among those tiles (uint32 [1, R]). Returns the users' last-token logits, row-major
        [1, 1, R, vocab / tp]: the LM head and the readback see only those rows."""
        x = ttnn.embedding(tokens_tt, self.embedding, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        x = ttnn.reshape(x, (1, 1, num * padded, self.cfg.hidden_size))
        with self._prefill_reduce(padded, num):
            for i in layers:
                y = self.layers[i].forward_prefill(
                    x,
                    None,
                    self._layer_page_table(i),
                    self.kv_caches[i],
                    start_pos=0,
                    tables=tables[self._table_key(i)],
                    users=num,
                )
                x.deallocate(True)
                x = y
        if num * padded == TILE:
            tiles = x  # a slice covering the whole tensor shares its buffer, so it must not be freed separately
        else:
            # Each user's last token: the tile row holding it, through a tensor-args slice (keyed on the length).
            parts = [ttnn.slice(x, at, at, slice_dim=2, num_devices=num * padded // TILE) for at in ats]
            x.deallocate(True)
            tiles = ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG) if num > 1 else parts[0]
            if num > 1:
                for t in parts:
                    t.deallocate(True)
        # Then the row itself: gather the R rows (an exact copy), so the LM head and the readback have R rows, not
        # 32 R (the logits of a tile row of every user were most of a pass's host read).
        rows_rm = ttnn.reshape(ttnn.to_layout(tiles, ttnn.ROW_MAJOR_LAYOUT), (TILE * num, self.cfg.hidden_size))
        tiles.deallocate(True)
        last = ttnn.embedding(picks_tt, rows_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        rows_rm.deallocate(True)
        last = ttnn.reshape(last, (1, 1, num, self.cfg.hidden_size))
        logits = self._logits(last, prefill=True)
        last.deallocate(True)
        out = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        logits.deallocate(True)
        return out

    # ------------------------------------------------------------------ traced prefill batches
    def setup_prefill_batch_trace(self, num, padded):
        """The persistent buffers of a traced prefill pass of ``num`` first chunks padded to ``padded``: its inputs
        (tokens, fill tables, last-token slice starts, the RoPE mats of positions 0 .. padded - 1) and the logits it
        copies its output into. Allocate before any trace capture: a buffer allocated after a capture can sit where
        that trace's replays write."""
        assert (num, padded) not in self.prefill_traces
        keys = list(dict.fromkeys(self._table_key(i) for i in range(len(self.layers))))
        blocks = -(-padded // self.block_size)
        rows = kv_lib.replicated(self.plan, torch.arange(num, dtype=torch.int32), ttnn.uint32)
        zero_fill = torch.zeros(self.plan.tp, num, blocks, dtype=torch.int32)
        self.prefill_traces[(num, padded)] = {
            "id": None,
            "tokens": self._to_device_tokens(torch.zeros(num * padded, dtype=torch.int32)),
            "rows": rows,
            "tables": {
                k: kv_lib.BatchFillTables(
                    kv_lib.per_chip(self.plan, zero_fill, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), rows
                )
                for k in keys
            },
            "ats": [kv_lib.replicated(self.plan, torch.zeros(4, dtype=torch.int32), ttnn.uint32) for _ in range(num)],
            "picks": kv_lib.replicated(self.plan, torch.zeros(1, num, dtype=torch.int32), ttnn.uint32),
            "rope": {name: setup.get_prefill_rot_mats(0, padded) for name, setup in self.rope_setups.items()},
            "logits": ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, num, self.vocab_local]),
                self.logits_dtype,
                ttnn.ROW_MAJOR_LAYOUT,
                self.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            ),
        }

    def _prefill_batch_trace_body(self, trace, num, padded):
        """The traced pass: reads only the trace's persistent inputs, writes the trace's persistent logits."""
        for name, setup in self.rope_setups.items():
            setup.begin_prefill_chunk(pinned={(0, padded): trace["rope"][name]})
        logits = self._prefill_batch_device(
            trace["tokens"], trace["tables"], trace["ats"], trace["picks"], num, padded, range(len(self.layers))
        )
        for setup in self.rope_setups.values():
            setup.end_prefill_chunk()
        ttnn.copy(logits, trace["logits"])
        logits.deallocate(True)

    def warm_prefill_batch_trace(self, num, padded):
        """Run the traced pass eagerly once (its zero fill tables write only vLLM's null block 0), compiling every
        program it runs: do it before the decode capture, so capturing the pass later compiles nothing."""
        self._prefill_batch_trace_body(self.prefill_traces[(num, padded)], num, padded)

    def capture_prefill_batch_trace(self, num, padded):
        """Capture the pass (after ``warm_prefill_batch_trace``). Every buffer the capture makes is freed by its end,
        so nothing it allocated is alive at another trace's replay."""
        trace = self.prefill_traces[(num, padded)]
        assert trace["id"] is None
        ttnn.synchronize_device(self.mesh_device)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._prefill_batch_trace_body(trace, num, padded)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        trace["id"] = trace_id

    def _replay_prefill_batch(self, trace, chunks):
        """Copy a batch's inputs into its trace's buffers and replay it (non-blocking)."""
        num, padded = len(chunks), max(prefill_padded_len(c[0].numel()) for c in chunks)
        tokens, fills, starts, picks = self._prefill_batch_host(chunks, padded, list(trace["tables"]))
        ttnn.copy_host_to_device_tensor(
            kv_lib.replicated(self.plan, tokens.reshape(1, -1), ttnn.uint32, device=False), trace["tokens"]
        )
        for key, table in trace["tables"].items():
            host = kv_lib.per_chip(self.plan, fills[key], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, device=False)
            ttnn.copy_host_to_device_tensor(host, table.fill)
        for at, s in zip(trace["ats"], starts):
            host = kv_lib.replicated(
                self.plan, torch.tensor([0, 0, s, 0], dtype=torch.int32), ttnn.uint32, device=False
            )
            ttnn.copy_host_to_device_tensor(host, at)
        picks_host = torch.tensor(picks, dtype=torch.int32).reshape(1, -1)
        ttnn.copy_host_to_device_tensor(
            kv_lib.replicated(self.plan, picks_host, ttnn.uint32, device=False), trace["picks"]
        )
        ttnn.execute_trace(self.mesh_device, trace["id"], cq_id=0, blocking=False)
        return {"logits": trace["logits"], "row": list(range(num)), "hidden": None, "persistent": True}

    def prefill_read(self, pending):
        """-> (last-token logits [vocab], per-layer hidden states [tokens, hidden] when requested). A batch's pending
        handle (``prefill_enqueue_batch``) gives [users, vocab]."""
        logits = self.plan.to_torch_tp_concat(pending["logits"], dim=-1).float()
        if not pending.get("persistent"):  # a traced pass's logits buffer is the trace's
            pending["logits"].deallocate(True)
        out = logits.reshape(-1, logits.shape[-1])[pending["row"]]
        hidden = []
        for chunks in pending["hidden"] or []:
            parts = []
            for h, n in chunks:
                parts.append(self.plan.to_torch_replicated(h).float()[0, 0, :n])
                h.deallocate(True)
            hidden.append(torch.cat(parts))
        return out, hidden

    def prefill(self, tokens, slot, return_hidden=False, start_pos=0, chunk_size=None):
        return self.prefill_read(self.prefill_enqueue(tokens, slot, return_hidden, start_pos, chunk_size))

    # ------------------------------------------------------------------ decode
    def prepare_decode_inputs_host(self, tokens, positions, serial=False):
        """Host tensors for one step: tokens [1, B] u32, cur_pos [B] i32, cos/sin [1, 1, B, 192] for the SWA and
        the global theta, MoE token mask,
        and the global layers' per-chip view: owned positions [B] i32 and owner mask [1, 1, B, 1].
        ``serial``: also SPEC_SLOTS position tensors [B] i32, each keeping one slot's position (a verify step writes
        K / V slot by slot, reference/spec_decode.py)."""
        assert tokens.numel() == self.max_batch and positions.numel() == self.max_batch
        rep = self.plan.replicate()
        tp, kv_rep = self.plan.tp, kv_lib.kv_replication(self.plan.tp, self.cfg.num_key_value_heads)

        def host(t, dtype, layout):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=rep)

        serial_inputs = []
        if serial:
            for one in serial_positions(positions.to(torch.int32).tolist(), SPEC_SLOTS):
                serial_inputs.append(host(torch.tensor(one, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT))
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
            *(
                [host(positions.clamp(min=0).reshape(1, -1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)]
                if self.rope_tables is not None
                else []
            ),
            *serial_inputs,
        )

    def setup_decode_rope_tables(self):
        """Device cos / sin tables of every position (both thetas), for decode steps that read the trace's own
        positions (``rope_pos`` input, advanced by every replay) instead of uploading their cos / sin: a steady step
        then copies nothing. Each table row is the host's ``decode_host_inputs`` value of that position, so the
        values are bitwise the same. max_seq_len x 192 x 2 bytes x 4 tables per chip (50 MB at 32K)."""
        assert self.decode_trace is None, "the tables must exist before the decode capture"
        positions = torch.arange(self.max_seq_len)
        tables = []
        for name in ("swa", "full"):
            setup = self.rope_setups[name]
            cos, sin = partial_rope_cos_sin(positions, setup.head_dim, setup.rope_dim, setup.theta)
            for t in (cos, sin):
                tables.append(
                    ttnn.from_torch(
                        t,
                        device=self.mesh_device,
                        dtype=setup.datatype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=self.plan.replicate(),
                    )
                )
        self.rope_tables = tables

    def _decode_inputs(self, inputs, greedy_only=False, serial=0):
        """decode_forward_device's arguments from the trace's input list (with a ``rope_pos`` input after the nine
        when the decode RoPE tables are set up, then ``serial`` per-slot position inputs)."""
        base, slots = inputs[: len(inputs) - serial], list(inputs[len(inputs) - serial :])
        rope = {"rope_pos_tt": base[9]} if len(base) > 9 else {}
        return self.decode_forward_device(*base[:9], greedy_only=greedy_only, serial_pos=slots or None, **rope)

    def decode_forward_device(
        self,
        tokens_tt,
        cur_pos_tt,
        cos_swa,
        sin_swa,
        cos_full,
        sin_full,
        mask_tt,
        pos_owned_tt,
        owner_mask_tt,
        rope_pos_tt=None,
        greedy_only=False,
        serial_pos=None,
    ):
        """One decode step from device-resident inputs only (trace-capturable). Returns [1, 1, B, vocab / tp].

        ``greedy_only``: skip the per-chip top-k candidates; GreedyHead computes the greedy tokens and also writes
        them into ``tokens_tt`` (the next step's input). ``rope_pos_tt`` (with ``rope_tables``): the cos / sin come
        from the device tables at these positions, not from the cos / sin inputs. ``serial_pos``: per-slot position
        tensors; every layer writes K / V one slot at a time (a speculative verify, reference/spec_decode.py)."""
        x = ttnn.embedding(tokens_tt, self.embedding, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        x = ttnn.reshape(x, (1, 1, self.max_batch, self.cfg.hidden_size))
        made = []
        if rope_pos_tt is not None:
            for t in self.rope_tables:
                row = ttnn.embedding(rope_pos_tt, t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
                made.append(ttnn.reshape(row, (1, 1, self.max_batch, t.shape[-1])))
            cos_swa, sin_swa, cos_full, sin_full = made
        rot_mats = {
            # SWA: the sharded mats plus the raw cos / sin rows (SwaDecodeQkv reads those).
            True: (*self.rope_setups["swa"].decode_rot_mats(cos_swa, sin_swa), cos_swa, sin_swa),
            False: self.rope_setups["full"].decode_rot_mats(cos_full, sin_full),
        }
        positions = {"pos": cur_pos_tt, "pos_owned": pos_owned_tt, "owner_mask": owner_mask_tt}
        if self.layer_groups is not None:
            positions["pos_update"] = cur_pos_tt  # vLLM's blocks: both chips of a pair keep every slot's K / V
        if serial_pos:
            positions["serial"] = serial_pos
        h = None
        for i, layer in enumerate(self.layers):
            y, h = layer.forward_decode(
                x, rot_mats, positions, self._layer_page_table(i), self.kv_caches[i], mask_tt, h=h, return_norm=True
            )
            x.deallocate(True)
            x = y
        logits = self._logits(x, h)
        x.deallocate(True)
        for t in (*rot_mats[True][:2], *rot_mats[False], *made):
            t.deallocate(True)
        if greedy_only:
            next_tokens = self.greedy_head(logits, token_input=tokens_tt)
            return {"logits": logits, "next_tokens": next_tokens}
        rows = None
        if self.layer_groups is not None:
            # vLLM: host sampling reads the users' rows (row-major) instead of 32 rows of tiles; device sampling is
            # GreedyHead's greedy token (the lowest id on ties, like the host's argmax), written into the next step's
            # token input. Nothing reads per-chip top-k candidates.
            rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
            rows = ttnn.slice(rm, (0, 0, 0, 0), (1, 1, self.decode_users, self.vocab_local))
            rm.deallocate(True)
            next_tokens = self.greedy_head(logits, token_input=tokens_tt)
            out = {"logits": logits, "next_tokens": next_tokens, "logits_rows": rows}
            if h is not None:
                out["hidden"] = h  # the final norm's persistent buffer: the MTP drafter's input (mtp.py)
            return out
        # Each chip's top-k over its vocab shard: the host can merge 8 x k candidates per user instead of
        # reading 32 x 152576 logits every step.
        values, indices = ttnn.topk(logits, k=self.DECODE_TOPK, dim=-1, largest=True, sorted=True)
        # Greedy token on device: gather every chip's candidates with their global vocab ids across the row,
        # then one top-1 whose label is the global id.
        gid = ttnn.add(ttnn.typecast(indices, ttnn.int32), self.vocab_offset)
        all_values = self.plan.all_gather(values, dim=-1)
        all_ids = self.plan.all_gather(gid, dim=-1)
        gid.deallocate(True)
        _, best = ttnn.topk(all_values, k=1, dim=-1, largest=True, sorted=True, indices_tensor=all_ids)
        all_values.deallocate(True)
        all_ids.deallocate(True)
        next_tokens = ttnn.typecast(
            ttnn.reshape(ttnn.to_layout(best, ttnn.ROW_MAJOR_LAYOUT), (1, self.max_batch)), ttnn.uint32
        )
        best.deallocate(True)
        out = {"logits": logits, "values": values, "indices": indices, "next_tokens": next_tokens}
        if rows is not None:
            out["logits_rows"] = rows
        return out

    def _feed_back(self, inputs, output, tokens_written=False):
        """Device-side step advance: the greedy token becomes the next input; active positions move by one."""
        tokens_tt, cur_pos_tt, pos_owned_tt = inputs[0], inputs[1], inputs[7]
        if not tokens_written:
            ttnn.copy(output["next_tokens"], tokens_tt)
        ttnn.plus_one(cur_pos_tt, skip_negative_entries=True)
        ttnn.plus_one(pos_owned_tt, skip_negative_entries=True)
        if len(inputs) > 9:
            # The RoPE table index: idle slots start at 0 and advance too, at most as far as an active slot does,
            # so the index stays inside the table until the next reload.
            ttnn.plus_one(inputs[9])

    def capture_decode_trace(self, greedy_only=False):
        """Compile, then capture one decode step. Later decode calls refresh the inputs and replay it.

        Warm-up and capture run with every slot empty (position -1), so they write no KV-cache entries
        and are safe to run after users have been prefilled. ``greedy_only`` captures the device-greedy step
        (GreedyHead, no top-k candidates): ``decode_steps_*`` work, ``decode_read_topk`` does not."""
        assert self.decode_trace is None
        if greedy_only and self.greedy_head is None:
            # Persistent buffers, allocated before any capture.
            self.greedy_head = GreedyHead(self.plan, self.vocab_local, self.decode_users, self.max_batch)
        written = greedy_only or self.layer_groups is not None  # GreedyHead wrote the tokens into the input
        # Per-step token slots for device-resident runs, allocated before capture: buffers allocated after a
        # capture may overlap the trace's replayed intermediates.
        history = [
            ttnn.from_torch(
                torch.zeros(1, self.max_batch, dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.plan.replicate(),
            )
            for _ in range(self.DECODE_HISTORY)
        ]
        empty = torch.full((self.max_batch,), -1, dtype=torch.int32)
        host = self.prepare_decode_inputs_host(torch.zeros(self.max_batch, dtype=torch.int32), empty)
        inputs = [ttnn.to_device(t, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in host]
        warm = self._decode_inputs(inputs, greedy_only=greedy_only)
        self._feed_back(inputs, warm, tokens_written=written)  # every slot is empty: positions stay -1
        for name, t in warm.items():
            if not (written and name == "next_tokens"):  # GreedyHead's buffer is persistent
                t.deallocate(True)
        ttnn.synchronize_device(self.mesh_device)
        fabric = getattr(self.plan, "fabric_gather", None)
        phase = fabric.phase if fabric is not None else None
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        output = self._decode_inputs(inputs, greedy_only=greedy_only)
        self._feed_back(inputs, output, tokens_written=written)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        # Every replay reuses the gather phases baked in at capture: back to back, replays only alternate the
        # gather buffers if the step made an even number of fabric reductions (FabricRowGather.advance).
        fabric = getattr(self.plan, "fabric_gather", None)
        assert fabric is None or fabric.phase == (phase if phase is not None else 0), "odd fabric reductions per step"
        ttnn.synchronize_device(self.mesh_device)
        self.decode_trace = {
            "id": trace_id,
            "inputs": inputs,
            "output": output,
            "history": history,
            "greedy_only": greedy_only,
        }

    def release_decode_trace(self):
        if self.decode_trace is not None:
            ttnn.release_trace(self.mesh_device, self.decode_trace["id"])
            self.decode_trace = None

    # ------------------------------------------------------------------ speculative decoding (reference/spec_decode.py)
    def setup_mtp(self, ckpt, cache_dir, layer=0):
        """Load MTP layer ``layer`` as the speculative drafter (its cache and trace: TtMiMoMTP.setup_cache,
        capture_spec_traces)."""
        assert self.mtp is None
        self.mtp = TtMiMoMTP(self, ckpt, cache_dir, layer)

    def capture_verify_trace(self):
        """Compile, then capture the verify step: the decode step with every layer's K / V written one slot at a time
        and no feed-back (a verify always reloads its inputs). Every slot idle while capturing: no K / V written."""
        assert self.verify_trace is None and self.layer_groups is not None, "a verify serves vLLM's KV blocks"
        empty = torch.full((self.max_batch,), -1, dtype=torch.int32)
        host = self.prepare_decode_inputs_host(torch.zeros(self.max_batch, dtype=torch.int32), empty, serial=True)
        inputs = [ttnn.to_device(t, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in host]
        self._release(self._decode_inputs(inputs, serial=SPEC_SLOTS))
        ttnn.synchronize_device(self.mesh_device)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        output = self._decode_inputs(inputs, serial=SPEC_SLOTS)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        # The MTP draft program reads the step's final hidden from the norm buffer it was captured on.
        assert output.get("hidden") is self.norm_buffers[0], "the final norm is not the persistent fused-norm buffer"
        self.verify_trace = {"id": trace_id, "inputs": inputs, "output": output}

    def capture_spec_traces(self):
        """The verify step and the MTP draft program, after the decode trace (persistent buffers already exist)."""
        self.capture_verify_trace()
        self.mtp.capture_trace()

    def release_spec_traces(self):
        if self.verify_trace is not None:
            ttnn.release_trace(self.mesh_device, self.verify_trace["id"])
            self.verify_trace = None
        if self.mtp is not None:
            self.mtp.release_trace()

    def verify_enqueue(self, tokens, positions):
        """A verify step (tokens, positions: torch [max_batch], -1 an idle slot). Non-blocking; returns the verify
        trace's output (its ``next_tokens`` are the slots' greedy tokens)."""
        assert int(positions.max()) < self.max_seq_len, f"position {int(positions.max())} >= {self.max_seq_len}"
        for h, d in zip(self.prepare_decode_inputs_host(tokens, positions, serial=True), self.verify_trace["inputs"]):
            ttnn.copy_host_to_device_tensor(h, d)
        ttnn.execute_trace(self.mesh_device, self.verify_trace["id"], cq_id=0, blocking=False)
        return self.verify_trace["output"]

    def read_next_tokens(self, handle):
        """The greedy token of every slot of a decode or verify step, int64 [max_batch] (blocks)."""
        return self.plan.to_torch_replicated(handle["next_tokens"]).reshape(-1).to(torch.int64)

    def release_prefill_traces(self):
        """Release the captured prefill traces (their buffers stay, for a later capture)."""
        for trace in self.prefill_traces.values():
            if trace["id"] is not None:
                ttnn.release_trace(self.mesh_device, trace["id"])
                trace["id"] = None

    def decode_enqueue(self, tokens, positions):
        """tokens, positions: torch [max_batch]; position -1 marks an empty slot. Non-blocking.
        Returns a handle for ``decode_read`` (full logits) or ``decode_read_topk`` (per-chip candidates)."""
        assert int(positions.max()) < self.max_seq_len, f"position {int(positions.max())} >= {self.max_seq_len}"
        host = self.prepare_decode_inputs_host(tokens, positions)
        self._advance_swa(positions)
        if self.decode_trace is not None:
            for h, d in zip(host, self.decode_trace["inputs"]):
                ttnn.copy_host_to_device_tensor(h, d)
            ttnn.execute_trace(self.mesh_device, self.decode_trace["id"], cq_id=0, blocking=False)
            return self.decode_trace["output"]
        inputs = [ttnn.to_device(t, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in host]
        out = self._decode_inputs(inputs)
        for t in inputs:
            t.deallocate(True)
        return out

    def decode_enqueue_resident(self, positions):
        """A decode step on the tokens and positions the last replay fed back (``_feed_back``): copies only the
        RoPE inputs of ``positions`` (torch [max_batch], what the trace's own positions now hold), then replays the
        trace. Non-blocking; returns the trace's output."""
        assert self.decode_trace is not None, "a resident decode step replays the decode trace"
        assert int(positions.max()) < self.max_seq_len, f"position {int(positions.max())} >= {self.max_seq_len}"
        self._advance_swa(positions)
        if self.rope_tables is None:  # else the trace reads its cos / sin from the tables at its own positions
            host = (
                *self.rope_setups["swa"].decode_host_inputs(positions),
                *self.rope_setups["full"].decode_host_inputs(positions),
            )
            for h, d in zip(host, self.decode_trace["inputs"][2:6]):
                ttnn.copy_host_to_device_tensor(h, d)
        ttnn.execute_trace(self.mesh_device, self.decode_trace["id"], cq_id=0, blocking=False)
        return self.decode_trace["output"]

    def _release(self, handle):
        traced = [t["output"] for t in (self.decode_trace, self.verify_trace) if t is not None]
        if not any(handle is out for out in traced):
            persistent = [self.greedy_head.next_tokens] if self.greedy_head is not None else []
            persistent += list(self.norm_buffers or ())
            for t in handle.values():
                if not any(t is p for p in persistent):
                    t.deallocate(True)

    def decode_read(self, handle):
        """Full logits [max_batch, vocab] (reads every vocab shard)."""
        out = self.plan.to_torch_tp_concat(handle["logits"], dim=-1).float()
        self._release(handle)
        return out.reshape(self.max_batch, -1)

    def decode_read_rows(self, handle):
        """Logits of the decode users' slots [decode_users, vocab] (vLLM mode: reads only those rows)."""
        out = self.plan.to_torch_tp_concat(handle["logits_rows"], dim=-1).float()
        self._release(handle)
        return out.reshape(self.decode_users, -1)

    def decode_read_topk(self, handle):
        """Per-user candidates: values [max_batch, tp * k] and global vocab ids [max_batch, tp * k]."""
        values = self.plan.to_torch_tp_concat(handle["values"], dim=-1).float().reshape(self.max_batch, -1)
        local = self.plan.to_torch_tp_concat(handle["indices"], dim=-1).to(torch.int64).reshape(self.max_batch, -1)
        shard = torch.arange(self.plan.tp).repeat_interleave(self.DECODE_TOPK) * self.vocab_local
        self._release(handle)
        return values, local + shard[None, :]

    def decode(self, tokens, positions):
        return self.decode_read(self.decode_enqueue(tokens, positions))

    def decode_steps_host_inputs(self, tokens, positions, steps):
        """Host tensors for a device-resident greedy run: step 0's full inputs, then each later step's cos/sin.

        Only cos/sin change between steps on the host side: tokens and positions advance on device."""
        # Positions advance on device, past any host check: every step must stay inside the context (the SWA table is
        # advanced on the host before each replay, so a step past the end would stop the run with replays in flight).
        assert (
            int(positions.max()) + steps <= self.max_seq_len
        ), f"{steps} steps from position {int(positions.max())} run past max_seq_len {self.max_seq_len}"
        first = self.prepare_decode_inputs_host(tokens, positions)
        later = []
        active = positions >= 0
        step_positions = [torch.where(active, positions + s, positions) for s in range(steps)]
        for s in range(1, steps):
            p = step_positions[s]
            later.append(
                (*self.rope_setups["swa"].decode_host_inputs(p), *self.rope_setups["full"].decode_host_inputs(p))
            )
        return first, later, step_positions

    def decode_steps_enqueue(self, host_inputs, step):
        """Enqueue step ``step`` of a device-resident greedy run (non-blocking): refresh what the host owns
        (everything at step 0, cos/sin later), replay the trace, keep the step's tokens in a history slot.
        At most DECODE_HISTORY steps between ``decode_steps_read``."""
        first, later, step_positions = host_inputs
        self._advance_swa(step_positions[step])
        inputs = self.decode_trace["inputs"]
        if step == 0:
            for h, d in zip(first, inputs):
                ttnn.copy_host_to_device_tensor(h, d)
        else:
            for h, d in zip(later[step - 1], inputs[2:6]):
                ttnn.copy_host_to_device_tensor(h, d)
        ttnn.execute_trace(self.mesh_device, self.decode_trace["id"], cq_id=0, blocking=False)
        ttnn.copy(self.decode_trace["output"]["next_tokens"], self.decode_trace["history"][step % self.DECODE_HISTORY])

    def decode_steps_read(self, first, count):
        """Greedy tokens of steps [first, first + count) of a device-resident run, int64 [count, max_batch].
        Blocks until they finished."""
        assert count <= self.DECODE_HISTORY
        hist = self.decode_trace["history"]
        return torch.stack(
            [
                self.plan.to_torch_replicated(hist[(first + s) % self.DECODE_HISTORY]).reshape(-1).to(torch.int64)
                for s in range(count)
            ]
        )
