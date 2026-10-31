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

from ..reference import kv_pair_split
from ..reference.model import load_layer_weights
from ..reference.mtp_backfill import HISTORY_ROWS
from ..reference.spec_decode import SERIAL_WRITES, SPEC_SLOTS, serial_positions
from . import kv as kv_lib
from . import attention as attention_lib
from .attention import (
    GLOBAL_SDPA_MIN_START,
    TtMiMoAttention,
    later_chunk_split,
    page_table_width,
    verify_variants,
)
from .common import TILE, MeshPlan, as_weight, cache_fill, cache_name, compute_config, pad_to
from .decoder import TtMiMoDecoderLayer, TtRMSNorm
from .device_sampler import DeviceSampler
from .greedy_head import GreedyHead
from .moe import expert_quant
from .moe_stream import ReduceExpand, fabric_row_gather, user_rows, zero_output
from .mtp import TtMiMoMTP
from .pair_split import PairSplit
from .profiling import zone
from .rope import MiMoRotarySetup, partial_rope_cos_sin, partial_rope_table
from .dflash import TtDFlash
from .spec_inputs import SpecInputs
from .verify_sampler import GreedyVerify, VerifySampler

# Rows of the decode RoPE tables past max_seq_len - 1: a position derived on device before the host checks it (a
# verify or draft chained on device, a few positions ahead) still reads a row of the table, not another buffer.
ROPE_TABLE_MARGIN = 32  # one tile

# MIMO_PREFILL_CHUNK_TRACE (default 1; 0 opts out): a later prefill chunk (start > 0) of a padded length set up with
# ``setup_prefill_chunk_traces`` replays a trace of the whole chunk instead of dispatching its ~2,500 ops (about 0.4 s of
# host time per chunk, which several rows or vLLM lanes on one host thread pay one after another). A setup or capture
# that fails (the trace region too small, say) drops the traces with a warning and later chunks run eagerly.
PREFILL_CHUNK_TRACE = os.environ.get("MIMO_PREFILL_CHUNK_TRACE", "1") == "1"
# The slack a later-chunk capture needs in the trace region over the largest one captured before it (the traces of a
# replica differ only in their SDPA programs).
CHUNK_TRACE_HEADROOM = 1.1
# What the first later-chunk capture of a replica is checked against (there is no earlier one to go by): a 4096-token
# chunk's trace takes 20.8 MB per device (TP=8 x DP=4). Only an early out: a capture that does not fit still fails
# cleanly in tt-metal's own check (MeshTrace::populate_mesh_buffer, which counts a trace against the region only once
# it fits) and drops the traces.
CHUNK_TRACE_FIRST_ESTIMATE = 21 << 20


def prefill_padded_len(seq_len, start_pos=0, pair_split=False):
    """Tokens are padded so the prefill SDPA chunk (32/128/256) divides the length: 32, 64, 96, then multiples of 128,
    then (from 2048) of 256. Every prefill program keys on one of these lengths (attention.py), so a chunk is padded
    to one even at the end of the context: the padding past the page table lands in a scratch block
    (kv.PrefillTables). ``pair_split`` (the KV pair split, a chunk after the first): at least 128, as pair-split
    chunked MLA prefill reads whole 128-token blocks."""
    n = pad_to(seq_len, TILE)
    if n >= 2048:
        return pad_to(n, 256)
    if n >= 128 or (pair_split and start_pos > 0):
        return pad_to(n, 128)
    return n


def prefill_warmup_chunks(max_tokens, max_seq_len, block_size, window, pair_split=False):
    """(tokens, start) of chunks that together run every prefill program a chunk of up to ``max_tokens`` tokens can:
    one per padded length at start 0, and at the later starts that bring something the programs key on (attention.py)
    that an earlier start did not: the number of window blocks a SWA layer reads, or the SDPA chunk size that divides
    the start (TtMiMoAttention._prefill_sdpa_program, which global layers run chunked MLA prefill with). Only chunks
    that fit in the context are listed.

    From GLOBAL_SDPA_MIN_START a global layer's later chunk can run the grouped / K split program and its merge
    (attention.later_chunk_split): one start there per padded length that takes it.

    ``pair_split`` (the KV pair split): later chunks start on 2 * block_size and are padded to at least 128
    (``prefill_padded_len``); the global layers' pair-split chunked MLA prefill keys on the same SDPA chunk size."""
    align = 2 * block_size if pair_split else block_size  # a later chunk's start alignment
    shortest = {}  # padded length -> the fewest tokens padded to it
    for n in range(1, max_tokens + 1):
        shortest.setdefault(prefill_padded_len(n), n)
    later = {}  # a later chunk's padded length -> the fewest tokens padded to it
    for n in range(1, max_tokens + 1):
        later.setdefault(prefill_padded_len(n, align, pair_split), n)
    out = []
    for padded in sorted(set(shortest) | set(later)):
        if padded in shortest and shortest[padded] <= max_seq_len:
            out.append((shortest[padded], 0))
        n = later.get(padded)
        if n is None or n > max_seq_len:
            continue
        leads, chunks = set(), set()
        # The window blocks stop changing past the window; the SDPA chunk sizes (at most 256) repeat every 256.
        for start in range(align, window + 256 + 1, align):
            if start + n > max_seq_len:
                break
            lead = start - max(0, start - window) // block_size * block_size
            chunk = TtMiMoAttention._prefill_sdpa_program(None, padded, start).q_chunk_size
            if lead not in leads or chunk not in chunks:
                leads.add(lead)
                chunks.add(chunk)
                out.append((n, start))
        start = pad_to(GLOBAL_SDPA_MIN_START, max(align, 256))
        chunk = TtMiMoAttention._prefill_sdpa_program(None, padded, start).q_chunk_size
        if start + n <= max_seq_len and later_chunk_split(chunk, start) is not None:
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
        replicate_global_kv=False,
        kv_pair_split=False,
        decode_users=None,
    ):
        """``ckpt`` may be None when ``cache_dir`` already holds every tensor (a later DP replica).
        ``max_prefill_chunk``: the default prefill chunk (a multiple of ``block_size``).
        ``replicate_global_kv``: ``setup_kv_cache`` keeps every user's global-layer K / V on both chips of a pair, as
        vLLM's caches do (twice the global KV memory), so a later prefill chunk splits its heads over the pair.
        ``kv_pair_split``: ``setup_kv_cache`` splits every 128-position global block over the pair instead, each chip
        keeping its 64-position halves, as vLLM's caches do with MIMO_KV_PAIR_SPLIT (tt/pair_split.py): the global
        layers run the vLLM adapter's pair-split prefill, decode and merge on the model's own tables.
        ``logits_dtype``: fp32 keeps near-tied logits apart; in bf16 (a step of 0.125 at the usual top logit of
        16-32) they often tie exactly and the lowest token id wins. Default fp32 (MIMO_LOGITS_FP32=0: bf16).
        ``decode_users``: the decode users the caller will set up (setup_kv_cache / setup_vllm_kv_cache), known now so
        that the fused reductions' fabric gather buffer is made before anything else takes L1 (``_set_decode_users``).
        """
        assert max_prefill_chunk % block_size == 0
        assert not (replicate_global_kv and kv_pair_split), "the pair either keeps both KV copies or splits them"
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
        # The same through a pair-split global table: one block per 2 * block_size positions.
        self.pair_context_blocks = pad_to(-(-(max_seq_len + kv_lib.PREFILL_MAX_PAD) // (2 * block_size)), 8)
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        if decode_users is not None and decode_users <= 16 and self._fabric_gather_enabled():
            # The first L1 allocation: the gather buffer then sits at the top of L1 (see _set_decode_users).
            fabric_row_gather(
                plan,
                ReduceExpand(plan, cfg.hidden_size, plan.tp, None, rows=user_rows(decode_users)),
            )

        head_dim = cfg.head_dim
        rope_dim = int(head_dim * cfg.partial_rotary_factor)
        self.rope_setups = {
            "swa": MiMoRotarySetup(mesh_device, max_batch, head_dim, rope_dim, cfg.swa_rope_theta),
            "full": MiMoRotarySetup(mesh_device, max_batch, head_dim, rope_dim, cfg.rope_theta),
        }

        with cache_fill(cache_dir, "top") as need:
            assert ckpt is not None or not need, "cold cache needs a checkpoint"
            emb = ckpt.embedding().bfloat16() if need else None
            self.embedding = as_weight(
                emb,
                plan,
                plan.replicate(),
                ttnn.bfloat16,
                cache_name(cache_dir, "embedding"),
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            del emb
            self.final_norm = TtRMSNorm(
                plan, ckpt.final_norm() if need else None, cfg.layernorm_epsilon, cache_name(cache_dir, "final_norm")
            )
            lm = ckpt.lm_head().T.contiguous() if need else None
            self.vocab_local = cfg.vocab_size // plan.tp
            self.lm_head = as_weight(
                lm, plan, plan.shard(-1), lm_head_dtype, cache_name(cache_dir, f"lm_head_tp{plan.tp}")
            )
            del lm

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
            with cache_fill(cache_dir, tag) as need:
                weights = load_layer_weights(ckpt, cfg, i, lazy_experts=True) if need else None
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
        self.verify_trace = None  # speculative verify step (capture_verify_trace): the one over every group
        # The verify captured per attended-group count (attention.verify_variants; {None: the trace} ungrouped), the
        # count the next capture attends, and the variant the last verify_enqueue replayed (its accept trace follows).
        self.verify_traces = {}
        self._verify_groups = None
        self._verify_active = None
        self.mtp = None  # the MTP drafter (setup_mtp)
        self.dflash = None  # the DFlash drafter (setup_dflash, tt/dflash.py)
        self.spec_inputs = None  # the packed verify / draft uploads (setup_mtp / setup_dflash, tt/spec_inputs.py)
        self.greedy_head = None
        self.sampler = None  # vLLM mode: temperature / top-k / top-p decode sampling (setup_vllm_kv_cache)
        self.verify_sampler = None  # vLLM mode with MTP: a verify's acceptance on device (setup_vllm_kv_cache)
        self.spec_chain = None  # the draft chained behind a verify, and its result page (TtMiMoMTP.setup_cache)
        self.prefill_traces = {}  # (users, padded length) -> traced prefill batch (setup_prefill_batch_trace)
        # Traced later chunks (setup_prefill_chunk_traces): key (_chunk_trace_key) -> trace; their inputs per padded
        # length; and per slot the persistent logits row a replay's output is copied into.
        self.chunk_traces = {}
        self.chunk_inputs = {}
        self.chunk_logits = None
        self.rope_tables = None  # decode cos / sin of every position (setup_decode_rope_tables)
        self.rope_table_tails = None  # with compact tables: the 1 / 0 the rows' rotated dims are followed by
        self.decode_users = None
        self.norm_buffers = None
        self.swa_table = None
        self.swa_qkv = None
        # vLLM-owned KV (setup_vllm_kv_cache): each layer's KV cache group, and the host copy of every group's table.
        self.layer_groups = None
        self.replicate_global_kv = replicate_global_kv
        self.kv_pair_split = kv_pair_split
        self.global_kv_both = False  # both chips of a global layer's pair keep every slot's K / V
        self.vllm_tables = None
        # The KV pair split (tt/pair_split.py; setup_vllm_kv_cache with pair_split, or setup_kv_cache with
        # kv_pair_split): its PairSplit, and the page tables of the global layers (``_table_key``: their vLLM KV cache
        # groups, or False for the model's own global table), which have one block per 2 * block_size positions.
        self.pair = None
        self.pair_groups = set()
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
    def _set_decode_users(self, num_users):
        """Size the decode programs for ``num_users`` users: the MoE streams and the compact attention rows (4 up to 4
        users, 8 up to 8)."""
        rows = user_rows(num_users) if num_users <= 16 else 4
        for layer in self.layers:
            if layer.is_moe:
                layer.mlp.set_decode_users(num_users)
            layer.attn.set_decode_rows(rows)
        # The fused reductions' fabric gather buffer (persistent L1, 64 KB per reduce core, 128 KB with 8 user rows),
        # made now if __init__ did not (``decode_users``): made at the first decode step it lands below the buffers that
        # step leaves, where a later prefill program's static circular buffers reach (the prefill SDPA's end at
        # 1179008, 3 KB under it with 4 users); made after the weights it may still land under a hole they left.
        red = next(
            (layer.mlp.reduce_expand for layer in self.layers if layer.is_moe and layer.mlp.gate_up_stream), None
        )
        if self._fabric_gather_enabled() and red is not None:
            gather = fabric_row_gather(self.plan, red)
            assert gather.tiles_per_core == red.tiles_per_core, "the gather buffer was made for other decode users"

    def _fabric_gather_enabled(self):
        return os.environ.get("MIMO_FABRIC_GATHER", "1") == "1" and tuple(self.mesh_device.shape) == (1, self.plan.tp)

    def setup_kv_cache(self, num_users):
        """Allocate paged KV for ``num_users`` users in decode slots [0, num_users).

        Every page table is indexed by absolute position over ``max_seq_len``. Global layers back every block; on
        TP=8 a global layer's chip pair splits the users (kv.py), so that page table differs per chip. SWA layers
        back only the blocks a sliding window can still read (kv.SwaBlockTable, shared by the SWA layers and
        updated before each prefill and decode step). Slots past ``num_users`` are never written (position -1).

        ``replicate_global_kv`` / ``kv_pair_split`` (__init__): every chip gets every user's global blocks instead (one
        table for the row), each a block_size positions or, under the pair split, 2 * block_size positions of which
        each chip keeps its block_size halves."""
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
        self._set_decode_users(num_users)
        self.kv_caches, tables = [], {}
        self.pair, self.pair_groups = None, set()
        max_blocks = page_table_width(self.max_seq_len // self.block_size)
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
                both = self.replicate_global_kv or self.kv_pair_split
                rep = 1 if both else layer.attn.rep
                span = 2 * self.block_size if self.kv_pair_split else self.block_size  # positions per table entry
                blocks = page_table_width(-(-self.max_seq_len // span))
                table, num_blocks = kv_lib.page_tables(num_users, self.max_batch, blocks, self.plan.tp, rep)
                # kv.page_tables adds one scratch block (the last) when a chip pair splits the users; with rep 1 every
                # block is a user's, so nothing may land past the table.
                scratch = num_blocks - 1 if rep > 1 else None
                if both:  # a scratch block for a later chunk's context table past the width
                    scratch, num_blocks = num_blocks, num_blocks + 1
            self.kv_caches.append(layer.attn.allocate_kv_cache(num_blocks, self.block_size, self.kv_dtype))
            tables.setdefault(layer.is_swa, table)
            self.table_scratch.setdefault(layer.is_swa, scratch)
        self.set_page_tables(tables)
        if self.swa_table is not None:
            self.swa_table.dirty = False
        if self.replicate_global_kv:
            self._keep_global_kv_on_both_chips()
        if self.kv_pair_split:
            self._setup_pair_split({False})
        self._setup_fused_norms()
        self._setup_lead_zeros()

    def _setup_pair_split(self, keys):
        """The KV pair split on the global layers, whose page tables ``keys`` (``_table_key``) have one block per
        2 * block_size positions: the PairSplit (persistent: the positions program's buffers and the merge's per-chip
        constants, before any capture) on every global layer."""
        assert all(layer.attn.split_users for layer in self.layers if not layer.is_swa), "no chip pair to split"
        assert self.block_size == kv_pair_split.SUB, self.block_size
        assert self.max_prefill_chunk % (2 * self.block_size) == 0, self.max_prefill_chunk
        spec = self.mtp is not None or self.dflash is not None
        self.pair = PairSplit(self.plan, self.max_batch, serial_slots=SERIAL_WRITES if spec else 0)
        self.pair_groups = set(keys)
        for layer in self.layers:
            if not layer.is_swa:
                layer.attn.pair = self.pair

    def _setup_lead_zeros(self):
        """The zero query rows a later SWA chunk puts in front of its cached window (TtMiMoAttention.lead_zeros), one
        tensor per window length, shared by the SWA layers: persistent, so a captured later chunk reads them."""
        swa = [layer.attn for layer in self.layers if layer.is_swa]
        if not swa or swa[0].lead_zeros:
            return
        window = swa[0].sliding_window
        zeros = {
            k * self.block_size: swa[0].make_lead_zeros(k * self.block_size)
            for k in range(1, -(-window // self.block_size) + 1)
        }
        for attn in swa:
            attn.lead_zeros = zeros

    def _keep_global_kv_on_both_chips(self):
        """Both chips of a global layer's pair keep every slot's K / V: decode writes it on both, and a later prefill
        chunk splits the group's heads over the pair (TtMiMoAttention.keep_kv_on_both_chips)."""
        self.global_kv_both = True
        for layer in self.layers:
            if layer.attn.split_users:
                layer.attn.keep_kv_on_both_chips()

    def setup_vllm_kv_cache(self, num_users, kv_caches, layer_groups, table_width, pair_split=False):
        """Adopt KV caches and page tables that vLLM owns, for ``num_users`` users in decode slots [0, num_users).

        ``kv_caches``: [K, V] per layer, allocated by the caller. ``layer_groups``: each layer's vLLM KV cache group;
        the layers of a group share one page table [max_batch, table_width] (``set_vllm_page_tables``; ``table_width``
        an int, or {group: width}), indexed by absolute position. A SWA group's table is vLLM's SlidingWindowSpec
        table: the blocks no window can read are vLLM's null block 0, where this model's own tables point at their
        scratch block (kv.SwaBlockTable).

        Every chip gets the same tables. vLLM can hand one block to requests in slots of either parity (prefix
        caching), so a global layer's chip pair cannot split the users by slot. ``pair_split`` (MIMO_KV_PAIR_SPLIT,
        the adapter's default): the global group's blocks hold 2 * block_size positions and each chip of the pair keeps
        its block_size halves (reference/kv_pair_split.py), every chip attends every slot over its own keys and the
        pair merges by LSE (tt/pair_split.py). Otherwise the pair keeps both copies of its KV head: decode writes every
        slot's K / V on both chips (``pos_update``) and still splits the attention by slot, and a later prefill chunk
        splits the group's heads over the pair (_keep_global_kv_on_both_chips)."""
        assert num_users <= self.max_batch
        assert self.decode_users in (None, num_users), f"decode users already set to {self.decode_users}"
        assert self.decode_trace is None and self.kv_caches is None, "the KV cache is already set up"
        assert len(kv_caches) == len(layer_groups) == len(self.layers)
        self.decode_users = num_users
        self._set_decode_users(num_users)
        self.kv_caches = kv_caches
        self.layer_groups = list(layer_groups)
        groups = sorted(set(self.layer_groups))
        widths = table_width if isinstance(table_width, dict) else {g: table_width for g in groups}
        self.vllm_tables = {g: torch.zeros(self.max_batch, widths[g], dtype=torch.int32) for g in groups}
        self.pair, self.pair_groups = None, set()
        if pair_split:
            global_groups = {g for g, layer in zip(self.layer_groups, self.layers) if not layer.is_swa}
            swa_groups = {g for g, layer in zip(self.layer_groups, self.layers) if layer.is_swa}
            assert not global_groups & swa_groups, f"a KV cache group mixes global and SWA layers: {self.layer_groups}"
            self._setup_pair_split(global_groups)
        else:
            self._keep_global_kv_on_both_chips()
        # Per (group, slot): how far the host row was last written; entries past it are zero (set_vllm_page_tables).
        self.vllm_table_lengths = {(g, s): 0 for g in self.vllm_tables for s in range(self.max_batch)}
        # Decode's device-sampled greedy token and sampled tokens (decode_forward_device); persistent buffers, made
        # before any capture.
        self.greedy_head = GreedyHead(self.plan, self.vocab_local, self.decode_users, self.max_batch)
        # The device sampler serves up to 8 users (its candidate pages); more decode users sample on the host.
        self.sampler = None
        if self.decode_users <= 8:
            self.sampler = DeviceSampler(self.plan, self.vocab_local, self.decode_users, self.max_batch)
        if self.mtp is not None or self.dflash is not None:
            # A verify's acceptance on device (the plugin's fused_sample, capture_verify_trace): its persistent
            # inputs and output, like the samplers' buffers, before any capture. A verify with more slots than the
            # device sampler's users (or no device sampler: more than 8 decode users) accepts greedy requests only.
            if self.sampler is not None and self.sampler.users >= SPEC_SLOTS:
                self.verify_sampler = VerifySampler(self.sampler, self.cfg.vocab_size)
            else:
                self.verify_sampler = GreedyVerify(self.plan, self.max_batch)
        self.page_table_tt = {
            g: ttnn.from_torch(
                table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.plan.replicate(),
            )
            for g, table in self.vllm_tables.items()
        }
        self._setup_fused_norms()
        self._setup_lead_zeros()

    def set_vllm_page_tables(self, rows):
        """rows: {slot: {group: int32 block ids}} (vLLM's table row of the user in that slot). A row wider than its
        group's table may only carry zeros past it (the plugin pads every group's rows to one width; a pair-split
        global table is half as wide). Copies every group table that changed to the device, in queue order with the
        work already enqueued."""
        changed = set()
        lengths = self.vllm_table_lengths
        for slot, tables in rows.items():
            for g, row in tables.items():
                host = self.vllm_tables[g]
                n = row.numel()
                if n > host.shape[1]:
                    tail = row[host.shape[1] :]
                    if bool(tail.any()):
                        raise ValueError(
                            f"group {g}: slot {slot}'s table row has blocks past the table's {host.shape[1]} entries: "
                            f"{tail[tail != 0][:8].tolist()}"
                        )
                    row, n = row[: host.shape[1]], host.shape[1]
                row = row.to(torch.int32)
                # Compare in place, without a full-width padded copy of every row on every step.
                written = lengths[(g, slot)]
                if written <= n and torch.equal(host[slot, :n], row):
                    continue
                host[slot, :n] = row
                host[slot, n:written] = 0
                lengths[(g, slot)] = n
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

    def _pair_key(self, key):
        """Whether the page table ``key`` (``_table_key``: a vLLM group, or is_swa for the model's own tables) is a
        pair-split global table."""
        return key in self.pair_groups

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
            pair = self._pair_key(key)
            table, scratch = self._host_table(key)
            context = self.pair_context_blocks if pair else self.prefill_context_blocks
            tables[key] = kv_lib.PrefillTables(
                self.plan,
                table,
                slot,
                start_pos,
                padded,
                self.block_size,
                scratch,
                context_blocks=context if later and not swa else None,
                window=self._window_blocks(key, slot, start_pos) if later and swa else None,
                start_tensor=start,
                pair_split=pair,
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
            if layer.attn_reduce is None and self.decode_users <= 16 and layer.can_compact_attention():
                layer.set_compact_attention(self.plan)
            if layer.fused_norm is not None or not layer.can_fuse_norms():
                continue
            nxt = self.layers[i + 1].input_norm if i + 1 < len(self.layers) else self.final_norm
            layer.set_fused_norms(self.plan, nxt, self.norm_buffers)
        self._setup_swa_qkv()

    def _uses_swa_qkv(self):
        """Every SWA layer decodes through the shared SwaDecodeQkv: compact decode (the BFP4 streaming MoE, at most 8
        users) and not turned off (MIMO_SWA_QKV=0 keeps the ttnn ops)."""
        if os.environ.get("MIMO_SWA_QKV", "1") != "1" or self.decode_users > 8:
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
    @contextlib.contextmanager
    def _prefill_rope(self, pinned=None):
        """Each RoPE setup's shared prefill cos/sin for one chunk or batch (MiMoRotarySetup.begin_prefill_chunk),
        ended however the pass exits: one that raises (a failed warm pass or capture) leaves no chunk open, so the next
        prefill can begin its own. ``pinned``: {setup name: {(start, length): [cos, sin]}} the caller owns."""
        begun = []
        try:
            for name, setup in self.rope_setups.items():
                setup.begin_prefill_chunk(pinned=None if pinned is None else pinned[name])
                begun.append(setup)
            yield
        finally:
            for setup in begun:
                setup.end_prefill_chunk()

    def _prefill_chunk(self, tokens, slot, start_pos, hidden, layers=None, draft_group=0, after_layer=None):
        """One chunk (``tokens`` 1D torch) at ``start_pos`` through the layers ``layers`` (indices, default all);
        returns the last one's output [1, 1, padded, hidden] and appends (copy, real rows) of every layer's output to
        ``hidden`` when given. With the DFlash drafter the chunk's last rows also go into draft group ``draft_group``'s
        context ring (TtDFlash.prefill_context; a warm-up through a subset of the layers runs the same programs).
        ``after_layer(i)`` runs after layer ``i`` is enqueued (profiling: tests/perf/profile_prefill.py reads the
        device profiler there)."""
        n = tokens.numel()
        padded = prefill_padded_len(n, start_pos, self.pair is not None)
        assert start_pos + n <= self.max_seq_len, f"chunk ends at {start_pos + n} > {self.max_seq_len}"
        layers = range(len(self.layers)) if layers is None else layers
        if self.swa_table is not None:
            self.swa_table.prefill(slot, start_pos, start_pos + n, start_pos + padded)
            self._sync_swa_table()
        tables, start = self._prefill_tables(slot, start_pos, padded, layers)
        # The shared cos/sin are freed as the chunk ends (enqueued: ordered with the work that reads them).
        with self._prefill_rope():
            with zone("embed"):
                x = self._embed(torch.nn.functional.pad(tokens.to(torch.int32), (0, padded - n)))
            dflash_acc = None
            with self._prefill_reduce(padded):
                for k, i in enumerate(layers):
                    layer = self.layers[i]
                    with zone(f"layer{i:02d}_{'swa' if layer.is_swa else 'global'}{'' if layer.is_moe else '_dense'}"):
                        y = layer.forward_prefill(
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
                    if self.dflash is not None:
                        dflash_acc = self.dflash.prefill_tap(i, x, dflash_acc)
                    if hidden is not None:
                        hidden[k].append((ttnn.clone(x), n))
                    if after_layer is not None:
                        after_layer(i)
            if self.dflash is not None:
                if dflash_acc is None:
                    dflash_acc = self.dflash.prefill_tap(self.dflash.tap_layers[0], x, None)
                self.dflash.prefill_context(dflash_acc, start_pos, n, draft_group)
        for t in tables.values():
            t.release()
        if start is not None:
            start.deallocate(True)
        return x, padded

    def prefill_enqueue(
        self,
        tokens,
        slot,
        return_hidden=False,
        start_pos=0,
        chunk_size=None,
        layers=None,
        history_continues=False,
        draft_group=0,
        allow_trace=True,
    ):
        """Enqueue prefill of one user (``tokens`` 1D torch, positions start_pos ..) into batch ``slot``, in chunks
        of ``chunk_size`` tokens (default ``max_prefill_chunk``; a multiple of the block size, as is ``start_pos``).
        ``start_pos > 0`` continues a user whose first start_pos tokens are in the cache. ``layers``: run only these
        layers (indices; warmup). Nothing is read back; returns a pending handle for ``prefill_read``. With the MTP
        history each chunk also writes its last rows' final normed hidden into the slot's history
        (tt/mtp_backfill.py); ``history_continues``: the first chunk continues the request's earlier chunks (their
        entries stay), else it is the request's first and the slot's other entries are invalidated.
        ``draft_group``: the request's draft group (its DFlash context ring). ``allow_trace=False``: every chunk runs
        eagerly, even one a captured later-chunk trace covers (vLLM's prefill_forward runs a row that would mix
        replays and eager chunks all eager: ``chunk_traced``)."""
        n = tokens.numel()
        chunk = chunk_size or self.max_prefill_chunk
        align = self.block_size * (2 if self.pair is not None else 1)  # pair split: 128-token global blocks
        assert n > 0 and chunk % align == 0 and start_pos % align == 0, (n, chunk, start_pos, align)
        hidden = [[] for _ in (range(len(self.layers)) if layers is None else layers)] if return_hidden else None
        hist = self._history()
        x = replayed = None
        for s in range(0, n, chunk):
            piece = tokens[s : s + chunk]
            writes = None
            if hist is not None and slot < SPEC_SLOTS:
                writes = hist.chunk_writes(slot, start_pos + s, piece, 0, history_continues or s > 0)
            eager = not allow_trace or hidden is not None or layers is not None
            trace = None if eager else self._chunk_trace(slot, start_pos + s, piece)
            if trace is not None:
                replayed = self._replay_prefill_chunk(trace, piece, slot, start_pos + s, writes or [])
                continue
            replayed = None
            x, padded = self._prefill_chunk(piece, slot, start_pos + s, hidden, layers, draft_group)
            if writes is not None:
                write_list = hist.prefill_list(writes)
                hist.prefill(x, self.final_norm, write_list, len(writes))
                write_list.deallocate(True)
            if s + chunk < n:
                x.deallocate(True)
        if replayed is not None:  # the last chunk was a replay: its logits row is the slot's persistent one
            return replayed
        seq_len = n - s
        tile_start = ((seq_len - 1) // TILE) * TILE
        if (
            hidden is None
            and self.layer_groups is None
            and any(t["id"] is not None for t in self.chunk_traces.values())
        ):
            # With later-chunk traces captured, a pending eager output could be alive while another slot's chunk
            # replays (the model's own callers enqueue several prompts before reading); the trace's intermediates may
            # sit where it is. So the logits row goes into the slot's persistent row, as a replay's does. (vLLM's
            # prefill_forward instead replays before it runs any eager pass of the lane.)
            at = kv_lib.replicated(self.plan, torch.tensor([0, 0, tile_start, 0], dtype=torch.int32), ttnn.uint32)
            pick = kv_lib.replicated(
                self.plan, torch.tensor([[seq_len - 1 - tile_start]], dtype=torch.int32), ttnn.uint32
            )
            logits = self._pick_logits(x, [at], pick, 1, padded)
            ttnn.copy(logits, self.chunk_logits[slot])
            for t in (logits, at, pick):
                t.deallocate(True)
            return {"logits": self.chunk_logits[slot], "row": 0, "hidden": None, "persistent": True}
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
        hist, history_list = self._history(), None
        if hist is not None:
            history_list = hist.prefill_list(self._prefill_batch_writes(chunks, padded))
        rows = kv_lib.replicated(self.plan, torch.arange(num, dtype=torch.int32), ttnn.uint32)
        tables = {
            k: kv_lib.BatchFillTables(kv_lib.per_chip(self.plan, fills[k], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), rows)
            for k in keys
        }
        ats = [kv_lib.replicated(self.plan, torch.tensor([0, 0, s, 0], dtype=torch.int32), ttnn.uint32) for s in starts]
        picks_tt = kv_lib.replicated(self.plan, torch.tensor(picks, dtype=torch.int32).reshape(1, -1), ttnn.uint32)
        tokens_tt = self._to_device_tokens(tokens)
        with self._prefill_rope():
            logits = self._prefill_batch_device(tokens_tt, tables, ats, picks_tt, num, padded, layers, history_list)
        # Enqueued: freeing them now is ordered with the work that reads them.
        for t in [*tables.values()]:
            t.release()
        for t in [rows, tokens_tt, picks_tt, *ats, *([history_list] if history_list is not None else [])]:
            t.deallocate(True)
        return {"logits": logits, "row": list(range(num)), "hidden": None}

    def _prefill_batch_writes(self, chunks, padded):
        """The MTP history's write list of a prefill batch: each user's first chunk (a request's first: the slot's
        other entries invalidated), its rows at r * padded of the pass. Slots past the history's write nothing."""
        hist, writes = self._history(), []
        for r, c in enumerate(chunks):
            if c[1] < SPEC_SLOTS:
                writes += hist.chunk_writes(c[1], 0, c[0], r * padded, False)
        return writes

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
            fills[key] = kv_lib.batch_fill_rows(table, slots, padded, self.block_size, scratch, self._pair_key(key))
            for r, c in enumerate(chunks):
                if len(c) > 2 and c[2]:
                    fills[key][:, r, : c[2]] = 0
        starts = [r * padded + (c[0].numel() - 1) // TILE * TILE for r, c in enumerate(chunks)]
        picks = [r * TILE + (c[0].numel() - 1) % TILE for r, c in enumerate(chunks)]
        return tokens, fills, starts, picks

    def _prefill_batch_device(self, tokens_tt, tables, ats, picks_tt, num, padded, layers, history_list=None):
        """A prefill batch from device inputs only (trace-capturable when the RoPE mats are pinned): tokens [1, R * S]
        uint32, {page table key: kv.BatchFillTables}, each user's last-token tile start (uint32 [4] slice starts), and
        each user's last-token row among those tiles (uint32 [1, R]). Returns the users' last-token logits, row-major
        [1, 1, R, vocab / tp]: the LM head and the readback see only those rows. ``history_list``: the MTP history's
        write list over the pass's rows (int32 [4 * R * HISTORY_ROWS], ``_prefill_batch_writes``)."""
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
        if history_list is not None:
            self._history().prefill(x, self.final_norm, history_list, history_list.shape[-1] // 4)
        return self._pick_logits(x, ats, picks_tt, num, padded)

    def _pick_logits(self, x, ats, picks_tt, num, padded):
        """The last-token logits of a pass's R users (x [1, 1, R * S, hidden], freed): row-major [1, 1, R, vocab / tp]
        (see ``_prefill_batch_device``)."""
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
            # The MTP history's write list (tt/mtp_backfill.py): every user's HISTORY_ROWS entries, rewritten per
            # replay.
            "history": (self._history().prefill_list([], num * HISTORY_ROWS) if self._history() is not None else None),
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
        with self._prefill_rope({name: {(0, padded): rope} for name, rope in trace["rope"].items()}):
            logits = self._prefill_batch_device(
                trace["tokens"],
                trace["tables"],
                trace["ats"],
                trace["picks"],
                num,
                padded,
                range(len(self.layers)),
                trace["history"],
            )
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
        if trace["history"] is not None:
            writes = self._prefill_batch_writes(chunks, padded)
            host = self._history().prefill_list(writes, num * HISTORY_ROWS, device=False)
            ttnn.copy_host_to_device_tensor(host, trace["history"])
        ttnn.execute_trace(self.mesh_device, trace["id"], cq_id=0, blocking=False)
        return {"logits": trace["logits"], "row": list(range(num)), "hidden": None, "persistent": True}

    # ------------------------------------------------------------------ traced later chunks
    def _swa_window(self):
        return next((layer.attn.sliding_window for layer in self.layers if layer.is_swa), None)

    def _owner_rep(self):
        """The global pair's replication when it splits the users (a later chunk reads its slot's owner mask), else 0."""
        for layer in self.layers:
            attn = layer.attn
            if not layer.is_swa and attn.split_users and attn.pair is None and not attn.kv_both_chips:
                return attn.rep
        return 0

    def _chunk_trace_key(self, slot, start_pos, padded):
        """What a later chunk's programs key on besides its padded length (attention.py): the global layers' SDPA chunk
        size and whether they take the grouped / K split program (TtMiMoAttention._prefill_sdpa_program,
        later_chunk_split), the SWA layers' cached window rows, and the slot's parity when the pair splits the users
        (its owner mask). Everything else a later chunk reads is a device input of the trace."""
        chunk = TtMiMoAttention._prefill_sdpa_program(None, padded, start_pos).q_chunk_size
        split = later_chunk_split(chunk, start_pos) is not None
        window = self._swa_window()
        lead = 0 if window is None else start_pos - max(0, start_pos - window) // self.block_size * self.block_size
        rep = self._owner_rep()
        return (padded, chunk, split, lead, slot % rep if rep else 0)

    def _chunk_trace_keys(self, padded):
        """The keys a later chunk of ``padded`` tokens can have once its start is past the SWA window: every SDPA chunk
        size and program the start can bring (the sizes repeat every 256 positions; the grouped program from
        GLOBAL_SDPA_MIN_START), every slot parity. Each with the first start that has it (the start the trace is
        warmed and captured at, which the Python of a chunk sees; a replay reads its start from the device)."""
        align = self.block_size * (2 if self.pair is not None else 1)
        window = self._swa_window() or 0
        first = pad_to(max(window, align), align)
        late = pad_to(GLOBAL_SDPA_MIN_START, max(align, 256))
        starts = list(range(first, first + 512, align)) + list(range(late, late + 512, align))
        rep = self._owner_rep() or 1
        keys = {}
        for start in starts:
            if start + padded > self.max_seq_len:
                continue
            for parity in range(rep):
                keys.setdefault(self._chunk_trace_key(parity, start, padded), start)
        return keys

    def setup_prefill_chunk_traces(self, padded_lengths, sdpa_chunks=None):
        """The persistent buffers of the traced later chunks of each padded length in ``padded_lengths``
        (``_chunk_trace_keys``; ``sdpa_chunks``: only the keys whose global SDPA chunk size is in it). Allocate before
        any trace capture (a buffer allocated after a capture can sit where that trace's replays write); then
        ``warm_prefill_chunk_traces`` and, after the decode capture, ``capture_prefill_chunk_traces``. Returns the
        keys set up. Refuses (before allocating anything) on a build whose chunked MLA prefill takes no runtime start
        (``chunk_traces_supported``)."""
        assert self.kv_caches is not None and self.dflash is None, "traced later chunks need the KV cache, no DFlash"
        if not self.chunk_traces_supported():
            raise RuntimeError(
                "traced later prefill chunks need chunked_flash_mla_prefill's chunk_start_idx_tensor (this ttnn build "
                "keys the global layers' later chunks on a scalar start, which a replay at another start cannot "
                "change); rebuild ttnn or set MIMO_PREFILL_CHUNK_TRACE=0"
            )
        if self.chunk_logits is None:
            self.chunk_logits = [self._logits_row() for _ in range(self.max_batch)]
        made = []
        for padded in padded_lengths:
            assert padded == prefill_padded_len(padded, 1, self.pair is not None), f"{padded} is not a padded length"
            for key, start in self._chunk_trace_keys(padded).items():
                if key in self.chunk_traces or (sdpa_chunks is not None and key[1] not in sdpa_chunks):
                    continue
                inputs = self._chunk_inputs(padded, start)
                self.chunk_traces[key] = {
                    "id": None,
                    "key": key,
                    "start": start,
                    "tables": self._chunk_tables(padded, key[3], inputs["start"]),
                    "logits": self._logits_row(),
                }
                made.append(key)
        return made

    def chunk_traces_supported(self):
        """Whether a later chunk's trace can replay at a start other than its capture's: every global layer reads the
        chunk's start from a device input. The KV pair split's chunked MLA prefill always does; the other global
        later-chunk paths only on a build whose chunked_flash_mla_prefill takes chunk_start_idx_tensor
        (attention.MLA_RUNTIME_START), else they key their program on the scalar start the trace was captured at."""
        return attention_lib.MLA_RUNTIME_START or all(
            layer.is_swa or layer.attn.pair is not None for layer in self.layers
        )

    def _logits_row(self):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 1, self.vocab_local]),
            self.logits_dtype,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

    def _chunk_inputs(self, padded, start):
        """The inputs the traced later chunks of ``padded`` tokens share (a replay rewrites them before it runs):
        tokens, the start position, the RoPE cos / sin of the chunk's positions, the last token's tile start and row,
        the MTP history's write list."""
        if padded not in self.chunk_inputs:
            hist = self._history()
            self.chunk_inputs[padded] = {
                "tokens": self._to_device_tokens(torch.zeros(padded, dtype=torch.int32)),
                "start": kv_lib.replicated(self.plan, torch.tensor([start], dtype=torch.int32), ttnn.int32),
                "rope": {name: setup.get_prefill_rot_mats(start, padded) for name, setup in self.rope_setups.items()},
                "at": kv_lib.replicated(self.plan, torch.zeros(4, dtype=torch.int32), ttnn.uint32),
                "pick": kv_lib.replicated(self.plan, torch.zeros(1, 1, dtype=torch.int32), ttnn.uint32),
                "history": hist.prefill_list([], HISTORY_ROWS) if hist is not None else None,
            }
        return self.chunk_inputs[padded]

    def _chunk_tables(self, padded, lead, start):
        """{page table key: kv.ChunkTables} of a traced later chunk whose SWA layers read ``lead`` cached rows."""
        tables = {}
        for i, layer in enumerate(self.layers):
            key = self._table_key(i)
            if key in tables:
                continue
            pair = self._pair_key(key)
            scratch = self._host_table(key)[1] or 0
            context = None if layer.is_swa else (self.pair_context_blocks if pair else self.prefill_context_blocks)
            window = lead // self.block_size if layer.is_swa else 0
            tables[key] = kv_lib.ChunkTables(self.plan, padded, self.block_size, scratch, context, window, start)
        return tables

    def _chunk_trace_body(self, trace):
        """The traced later chunk: reads only its persistent inputs, writes its persistent logits row."""
        padded, parity = trace["key"][0], trace["key"][4]
        start, inputs = trace["start"], self.chunk_inputs[padded]
        with self._prefill_rope({name: {(start, padded): rope} for name, rope in inputs["rope"].items()}):
            x = ttnn.embedding(inputs["tokens"], self.embedding, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            x = ttnn.reshape(x, (1, 1, padded, self.cfg.hidden_size))
            with self._prefill_reduce(padded):
                for i, layer in enumerate(self.layers):
                    y = layer.forward_prefill(
                        x,
                        parity,
                        self._layer_page_table(i),
                        self.kv_caches[i],
                        start_pos=start,
                        tables=trace["tables"][self._table_key(i)],
                    )
                    x.deallocate(True)
                    x = y
            if inputs["history"] is not None:
                self._history().prefill(x, self.final_norm, inputs["history"], HISTORY_ROWS)
            logits = self._pick_logits(x, [inputs["at"]], inputs["pick"], 1, padded)
        ttnn.copy(logits, trace["logits"])
        logits.deallocate(True)

    def warm_prefill_chunk_traces(self):
        """Run every traced later chunk not yet captured eagerly once (on its scratch tables: it writes only scratch
        blocks), compiling every program it runs, and the copy of its logits row."""
        for trace in self.chunk_traces.values():
            if trace["id"] is None:
                self._chunk_trace_body(trace)
                ttnn.copy(trace["logits"], self.chunk_logits[0])
        ttnn.synchronize_device(self.mesh_device)

    def capture_prefill_chunk_traces(self):
        """Capture every traced later chunk not yet captured (after ``warm_prefill_chunk_traces``). Every buffer a
        capture makes is freed by its end. The traces are optional: when the trace region has no room for the next one
        (the largest captured so far plus CHUNK_TRACE_HEADROOM) or a capture fails, every later-chunk trace is dropped
        (``drop_prefill_chunk_traces``, with a warning) and later chunks run eagerly. Capture them after every trace the
        replica needs. Returns whether they are all captured."""
        largest = 0
        for trace in self.chunk_traces.values():
            if trace["id"] is not None:
                continue
            used, total = self.trace_region_bytes()
            need = largest or CHUNK_TRACE_FIRST_ESTIMATE
            if total and total - used < need * CHUNK_TRACE_HEADROOM:
                self.drop_prefill_chunk_traces(
                    f"the trace region has {(total - used) / 2**20:.0f} MB of {total / 2**20:.0f} MB free per device "
                    f"and a later-chunk trace takes {'' if largest else 'about '}{need / 2**20:.0f} MB"
                )
                return False
            trace_id = None
            try:
                ttnn.synchronize_device(self.mesh_device)
                trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                try:
                    self._chunk_trace_body(trace)
                finally:
                    ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
                ttnn.synchronize_device(self.mesh_device)
            # e.g. "Creating trace buffers of size ... but only ... is allocated for trace region"
            except Exception as e:
                if trace_id is not None:
                    try:
                        ttnn.release_trace(self.mesh_device, trace_id)
                    except Exception:
                        pass
                self.drop_prefill_chunk_traces(f"capturing a later-chunk trace failed: {e}")
                return False
            trace["id"] = trace_id
            largest = max(largest, self.trace_region_bytes()[0] - used)
        used, total = self.trace_region_bytes()
        if total and self.chunk_traces:
            logger.info(
                f"MiMo: {len(self.chunk_traces)} later-chunk prefill traces captured (up to {largest / 2**20:.1f} MB "
                f"each); trace region {used / 2**20:.0f} MB of {total / 2**20:.0f} MB used per device"
            )
        return True

    def trace_region_bytes(self):
        """(allocated, total) bytes of the trace region per device; (0, 0) when traces live in DRAM
        (trace_region_size 0) or the allocator cannot say."""
        try:
            view = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.TRACE)
        except Exception:
            return 0, 0
        return view.total_bytes_allocated_per_bank * view.num_banks, view.total_bytes_per_bank * view.num_banks

    def drop_prefill_chunk_traces(self, reason):
        """Give up the traced later chunks (a setup or capture that failed): release every captured one, free their
        buffers (dropping the last reference frees a device tensor), warn. Later chunks then run eagerly
        (``_chunk_trace`` finds no trace)."""
        logger.warning(
            f"MiMo: traced later prefill chunks disabled, later chunks run eagerly ({reason}). Raise trace_region_size "
            "or set MIMO_PREFILL_CHUNK_TRACE=0 to silence this."
        )
        for trace in self.chunk_traces.values():
            if trace["id"] is not None:
                try:
                    ttnn.release_trace(self.mesh_device, trace["id"])
                except Exception:
                    pass
        self.chunk_traces, self.chunk_inputs, self.chunk_logits = {}, {}, None

    def release_prefill_chunk_traces(self):
        """Release the captured later-chunk traces (their buffers stay, for a later capture)."""
        for trace in self.chunk_traces.values():
            if trace["id"] is not None:
                ttnn.release_trace(self.mesh_device, trace["id"])
                trace["id"] = None

    def _chunk_trace(self, slot, start_pos, tokens):
        """The captured trace a later chunk of ``tokens`` at ``start_pos`` of ``slot`` replays, or None (eager)."""
        if not self.chunk_traces or start_pos == 0 or self.dflash is not None:
            return None
        padded = prefill_padded_len(tokens.numel(), start_pos, self.pair is not None)
        trace = self.chunk_traces.get(self._chunk_trace_key(slot, start_pos, padded))
        return trace if trace is not None and trace["id"] is not None else None

    def chunk_traced(self, slot, start_pos, num_tokens, chunk_size=None):
        """Whether every chunk ``prefill_enqueue`` splits a later prefill of ``num_tokens`` tokens at ``start_pos``
        into (``chunk_size``, default max_prefill_chunk) replays a trace. A prefill longer than one chunk (an
        unaligned start moved back past the chunk budget) can have chunks that replay and chunks that run eagerly;
        it is not traced, and vLLM's prefill_forward runs it with ``allow_trace=False``."""
        chunk = chunk_size or self.max_prefill_chunk
        return num_tokens > 0 and all(
            self._chunk_trace(slot, start_pos + s, torch.empty(min(chunk, num_tokens - s))) is not None
            for s in range(0, num_tokens, chunk)
        )

    def _replay_prefill_chunk(self, trace, tokens, slot, start_pos, writes):
        """Copy a later chunk's inputs into its trace's buffers, replay it (non-blocking) and copy its logits row into
        the slot's own (a later replay of the trace rewrites the trace's). ``writes``: its MTP history writes."""
        n = tokens.numel()
        padded = trace["key"][0]
        inputs = self.chunk_inputs[padded]
        if self.swa_table is not None:  # the host bookkeeping of the eager chunk (its device copy waits for decode)
            self.swa_table.prefill(slot, start_pos, start_pos + n, start_pos + padded)
        host = torch.nn.functional.pad(tokens.to(torch.int32), (0, padded - n)).reshape(1, -1)
        ttnn.copy_host_to_device_tensor(kv_lib.replicated(self.plan, host, ttnn.uint32, False), inputs["tokens"])
        start = torch.tensor([start_pos], dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(kv_lib.replicated(self.plan, start, ttnn.int32, False), inputs["start"])
        positions = torch.arange(start_pos, start_pos + padded)
        for name, setup in self.rope_setups.items():
            cos, sin = partial_rope_cos_sin(positions, setup.head_dim, setup.rope_dim, setup.theta)
            for value, t in zip((cos, sin), inputs["rope"][name]):
                ttnn.copy_host_to_device_tensor(setup._host(value, False), t)
        for key, tables in trace["tables"].items():
            table, scratch = self._host_table(key)
            pair, swa = self._pair_key(key), tables.context is None
            context = None if swa else (self.pair_context_blocks if pair else self.prefill_context_blocks)
            fill, ctx = kv_lib.PrefillTables.host_rows(
                table, slot, start_pos, padded, self.block_size, scratch, context, pair
            )
            window = self._window_blocks(key, slot, start_pos) if swa else None
            tables.write(fill, ctx, window)
        tile_start = (n - 1) // TILE * TILE
        at = torch.tensor([0, 0, tile_start, 0], dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(kv_lib.replicated(self.plan, at, ttnn.uint32, False), inputs["at"])
        pick = torch.tensor([[(n - 1) % TILE]], dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(kv_lib.replicated(self.plan, pick, ttnn.uint32, False), inputs["pick"])
        if inputs["history"] is not None:
            host = self._history().prefill_list(writes, HISTORY_ROWS, device=False)
            ttnn.copy_host_to_device_tensor(host, inputs["history"])
        ttnn.execute_trace(self.mesh_device, trace["id"], cq_id=0, blocking=False)
        out = self.chunk_logits[slot]
        ttnn.copy(trace["logits"], out)
        return {"logits": out, "row": 0, "hidden": None, "persistent": True}

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
        ``serial``: also SERIAL_WRITES position tensors [B] i32, each keeping the positions of one column of the draft
        groups (a verify step writes K / V one column at a time, reference/spec_decode.py serial_positions)."""
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

    def decode_rope_tables_host(self, compact=False):
        """The torch tables setup_decode_rope_tables uploads: cos / sin of the SWA theta, then of the global theta, over
        max_seq_len + ROPE_TABLE_MARGIN positions, 192 (``compact``: 64) dims, each row computed in the host's
        per-step call (rope.partial_rope_table over max_batch positions). The same for every replica of a mesh."""
        rows = self.max_seq_len + ROPE_TABLE_MARGIN
        tables = []
        for name in ("swa", "full"):
            setup = self.rope_setups[name]
            width = setup.rope_dim if compact else setup.head_dim
            tables += partial_rope_table(rows, setup.head_dim, setup.rope_dim, setup.theta, width, self.max_batch)
        return tables

    def setup_decode_rope_tables(self, compact=False, host=None):
        """Device cos / sin tables of every position (both thetas: cos / sin of the SWA theta, then of the global one),
        for the decode, verify and MTP draft steps, which read the rows of their own position inputs
        (``rope_table_rows``) instead of uploading their cos / sin: a steady step then copies nothing. Each row is the
        host's ``decode_host_inputs`` value of that position (``decode_rope_tables_host``, or ``host`` when the caller
        made them once for several replicas), so the values are bitwise the same.
        ``compact``: the tables keep only the rope_dim (64) rotated dims of a row, and each read appends the constant
        1 / 0 of the other head_dim - rope_dim dims (one concat per table read). Per chip, max_seq_len x 4 tables x
        192 (64 compact) x 2 bytes: 50 MB at 32K, 201 MB (67 MB) at 128K, 1.61 GB (537 MB) at 1M. The tables run
        ROPE_TABLE_MARGIN rows past max_seq_len - 1."""
        assert self.decode_trace is None, "the tables must exist before the decode capture"
        # Through the class: the device tests call this on a stub of the model's attributes.
        host = host if host is not None else TtMiMoModel.decode_rope_tables_host(self, compact)
        setup = self.rope_setups["swa"]
        assert all((s.head_dim, s.rope_dim) == (setup.head_dim, setup.rope_dim) for s in self.rope_setups.values())
        width = setup.rope_dim if compact else setup.head_dim
        assert [tuple(t.shape) for t in host] == [(self.max_seq_len + ROPE_TABLE_MARGIN, width)] * 4
        rep = self.plan.replicate()
        self.rope_tables = [
            ttnn.from_torch(
                t, device=self.mesh_device, dtype=setup.datatype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
            )
            for t in host
        ]
        self.rope_table_tails = None
        if compact:
            shape = (1, 1, self.max_batch, setup.head_dim - setup.rope_dim)
            self.rope_table_tails = [
                ttnn.from_torch(
                    torch.full(shape, value),
                    device=self.mesh_device,
                    dtype=setup.datatype,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=rep,
                )
                for value in (1.0, 0.0)
            ]

    def rope_table_rows(self, idx_tt, tables):
        """The rows of ``tables`` (cos / sin pairs of ``rope_tables``) at the positions ``idx_tt`` (uint32 [1, B]):
        [1, 1, B, head_dim] tiles, what ``decode_host_inputs`` makes of those positions. The caller frees them.
        Trace-capturable; with compact tables each row is the table's 64 dims followed by the cos / sin tail."""
        out = []
        for k, t in enumerate(tables):
            row = ttnn.embedding(idx_tt, t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            row = ttnn.reshape(row, (1, 1, self.max_batch, t.shape[-1]))
            if self.rope_table_tails is not None:
                whole = ttnn.concat([row, self.rope_table_tails[k % 2]], dim=-1)
                row.deallocate(True)
                row = whole
            out.append(row)
        return out

    def _decode_inputs(self, inputs, greedy_only=False, sample=False, serial=0, verify_groups=None):
        """decode_forward_device's arguments from the trace's input list (with a ``rope_pos`` input after the nine
        when the decode RoPE tables are set up, then ``serial`` per-slot position inputs)."""
        base, slots = inputs[: len(inputs) - serial], list(inputs[len(inputs) - serial :])
        rope = {"rope_pos_tt": base[9]} if len(base) > 9 else {}
        return self.decode_forward_device(
            *base[:9],
            greedy_only=greedy_only,
            sample=sample,
            serial_pos=slots or None,
            verify_groups=verify_groups,
            **rope,
        )

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
        sample=False,
        serial_pos=None,
        verify_groups=None,
    ):
        """One decode step from device-resident inputs only (trace-capturable). Returns [1, 1, B, vocab / tp].

        ``greedy_only``: skip the per-chip top-k candidates; GreedyHead computes the greedy tokens and also writes
        them into ``tokens_tt`` (the next step's input). ``rope_pos_tt`` (with ``rope_tables``): the cos / sin come
        from the device tables at these positions, not from the cos / sin inputs. ``sample`` (vLLM mode): the
        DeviceSampler draws each row's token with its temperature / top-k / top-p (greedy rows take the argmax, as
        GreedyHead). ``serial_pos``: per-slot position tensors; every layer writes K / V one slot at a time (a
        speculative verify, reference/spec_decode.py). ``verify_groups``: the grouped verify attends groups 0 ..
        verify_groups - 1 only (attention.verify_variants)."""
        x = ttnn.embedding(tokens_tt, self.embedding, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        x = ttnn.reshape(x, (1, 1, self.max_batch, self.cfg.hidden_size))
        made = []
        if rope_pos_tt is not None:
            made = self.rope_table_rows(rope_pos_tt, self.rope_tables)
            cos_swa, sin_swa, cos_full, sin_full = made
        rot_mats = {
            # SWA: the sharded mats plus the raw cos / sin rows (SwaDecodeQkv reads those).
            True: (*self.rope_setups["swa"].decode_rot_mats(cos_swa, sin_swa), cos_swa, sin_swa),
            False: self.rope_setups["full"].decode_rot_mats(cos_full, sin_full),
        }
        positions = {"pos": cur_pos_tt, "pos_owned": pos_owned_tt, "owner_mask": owner_mask_tt}
        sdpa_pos = pos_owned_tt
        if self.pair is not None:
            # The KV pair split: each chip's write and SDPA positions from the global ones (which the trace advances),
            # on device; the owned positions and owner mask are not read.
            write, sdpa_pos, serial = self.pair.positions(cur_pos_tt)
            positions.update(pair_write=write, pair_sdpa=sdpa_pos)
            if serial_pos:
                positions["pair_serial"] = serial
        elif self.global_kv_both:
            positions["pos_update"] = cur_pos_tt  # both chips of a pair keep every slot's K / V
        if serial_pos:
            positions["serial"] = serial_pos
        if verify_groups is not None:
            positions["verify_groups"] = verify_groups
        # Global layers' SDPA over the decode users' rows only (see attention._forward_decode_split): with fewer
        # users than table rows the kernel would spread its cores over 32 rows of mostly idle slots.
        compact_sdpa = self.decode_users < self.max_batch and os.environ.get("MIMO_COMPACT_SDPA", "1") == "1"
        sdpa_tables = {}
        if compact_sdpa:
            positions["pos_sdpa"] = ttnn.slice(sdpa_pos, (0,), (self.decode_users,))
        h = None
        for i, layer in enumerate(self.layers):
            layer_positions = positions
            if compact_sdpa and not layer.is_swa:
                key = self._table_key(i)
                if key not in sdpa_tables:
                    table = self.page_table_tt[key]
                    sdpa_tables[key] = ttnn.slice(table, (0, 0), (self.decode_users, table.shape[-1]))
                layer_positions = dict(positions, page_table_sdpa=sdpa_tables[key])
            y, h = layer.forward_decode(
                x,
                rot_mats,
                layer_positions,
                self._layer_page_table(i),
                self.kv_caches[i],
                mask_tt,
                h=h,
                return_norm=True,
            )
            x.deallocate(True)
            x = y
            if self.dflash is not None:
                self.dflash.tap(i, y)  # DFlash's context: the target layers' outputs it reads (tt/dflash.py)
        logits = self._logits(x, h)
        x.deallocate(True)
        for t in (*rot_mats[True][:2], *rot_mats[False], *made, *sdpa_tables.values()):
            t.deallocate(True)
        if compact_sdpa:
            positions["pos_sdpa"].deallocate(True)
        if greedy_only:
            next_tokens = self.greedy_head(logits, token_input=tokens_tt)
            return {"logits": logits, "next_tokens": next_tokens}
        rows = None
        if self.layer_groups is not None:
            # vLLM: host sampling reads the users' rows (row-major) instead of 32 rows of tiles; device sampling is
            # GreedyHead's greedy token (the lowest id on ties, like the host's argmax), or with ``sample`` the
            # DeviceSampler's draw, written into the next step's token input.
            rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
            rows = ttnn.slice(rm, (0, 0, 0, 0), (1, 1, self.decode_users, self.vocab_local))
            rm.deallocate(True)
            head = self.sampler if sample else self.greedy_head
            next_tokens = head(logits, token_input=tokens_tt)
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
        (GreedyHead, no top-k candidates): ``decode_steps_*`` work, ``decode_read_topk`` does not.

        With a DeviceSampler (vLLM mode) a second trace on the same inputs, ``sampled``, also draws the sampled rows'
        tokens; a step replays it only when a row samples, so greedy batches skip the sampler. Both steps compile
        before either capture: a program compiled after a capture may keep buffers the replays overwrite."""
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
        variants = [False] + ([True] if self.sampler is not None and not greedy_only else [])
        for sample in variants:
            warm = self._decode_step(inputs, greedy_only, sample)
            self._feed_back(inputs, warm, tokens_written=written)  # every slot is empty: positions stay -1
            if sample and self.verify_sampler is not None:
                # Compiles SampleVerify before any capture (its steps 1-2 are the sampled step's own programs, which
                # key on nothing the verify changes), so the accept trace (capture_verify_trace) compiles nothing
                # after the decode trace. It writes only its own output page.
                self.verify_sampler(warm["logits"], warm["next_tokens"])
            # Keeps the persistent buffers: GreedyHead's and the sampler's tokens and the final norm's "hidden" (the
            # MTP drafter's input).
            self._release(warm)
        if self.verify_sampler is not None and not isinstance(self.verify_sampler, VerifySampler):
            # The greedy accept step, which no sampled step compiled above.
            self.verify_sampler(None, self.greedy_head.next_tokens)
        if self.spec_chain is not None:
            # The chain and drafts steps (tt/spec_chain.py) of the accept and MTP traces, and the verify trace's relay
            # step, compile here too, on the persistent buffers their captures use; so do the history's kernel steps
            # (tt/mtp_backfill.py).
            self.spec_chain.warm(self.verify_sampler.ctrl)
        if self._history() is not None:
            self._history().warm(self.mtp.inputs)
        if self.spec_chain is not None:
            # The drafter's programs too (its setup made the chain): they compile before any capture (TtMiMoMTP.warm,
            # TtDFlash.warm).
            (self.mtp if self.mtp is not None else self.dflash).warm()
        ttnn.synchronize_device(self.mesh_device)
        captured = []
        for sample in variants:
            fabric = getattr(self.plan, "fabric_gather", None)
            phase = fabric.phase if fabric is not None else None
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            output = self._decode_step(inputs, greedy_only, sample)
            self._feed_back(inputs, output, tokens_written=written)
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
            # Every replay reuses the gather phases baked in at capture: back to back, replays only alternate the
            # gather buffers if the step made an even number of fabric reductions (FabricRowGather.advance).
            fabric = getattr(self.plan, "fabric_gather", None)
            assert fabric is None or fabric.phase == (
                phase if phase is not None else 0
            ), "odd fabric reductions per step"
            captured.append({"id": trace_id, "output": output})
        ttnn.synchronize_device(self.mesh_device)
        self.decode_trace = {
            **captured[0],
            "inputs": inputs,
            "history": history,
            "greedy_only": greedy_only,
            "sampled": captured[1] if len(captured) > 1 else None,
        }

    def _history(self):
        """The MTP drafter's target-hidden history (tt/mtp_backfill.py), or None."""
        return self.mtp.history if self.mtp is not None else None

    def _decode_step(self, inputs, greedy_only, sample):
        """The decode trace's step: with the MTP history, its input tokens kept at the start and its slots' final
        norm rows written into the history at the end (before the positions advance)."""
        hist = self._history()
        if hist is not None:
            hist.stage(inputs[0])
        output = self._decode_inputs(inputs, greedy_only=greedy_only, sample=sample)
        if hist is not None:
            assert output.get("hidden") is self.norm_buffers[0], "the history reads the final norm's buffer"
            hist.decode(inputs[1], output["hidden"])
        return output

    def release_decode_trace(self):
        if self.decode_trace is not None:
            ttnn.release_trace(self.mesh_device, self.decode_trace["id"])
            if self.decode_trace["sampled"] is not None:
                ttnn.release_trace(self.mesh_device, self.decode_trace["sampled"]["id"])
            self.decode_trace = None

    def _decode_replay(self, sample):
        """Replays the decode trace (``sample``: the sampled one); returns its output."""
        trace = self.decode_trace["sampled"] if sample else self.decode_trace
        assert trace is not None, "a sampled decode step needs the DeviceSampler's trace (vLLM mode)"
        if sample:
            # The sampled step replays only on an empty queue. Replayed behind commands still in flight (the last
            # step's token read, a reload's copies), it now and then breaks the replica's decode for good: every
            # slot's logits turn flat or inf from then on, new requests included, while its inputs, page tables and
            # sampling parameters read back intact and the sampler draws correctly from what it is given (4x8
            # Galaxy, SPEED-Bench 1k at c=16, T=1: 14 of 64 requests garbled). The greedy trace never does; any
            # extra small copy ahead of the sampled one makes it more likely. The fault is below the model. The
            # drain costs nothing measurable: the same run then garbles 0 of 64, streaming 897 chunks/s against 836.
            ttnn.synchronize_device(self.mesh_device)
        ttnn.execute_trace(self.mesh_device, trace["id"], cq_id=0, blocking=False)
        return trace["output"]

    # ------------------------------------------------------------------ speculative decoding (reference/spec_decode.py)
    def setup_mtp(self, ckpt, cache_dir, layers=(0,), backfill=False):
        """Load the MTP layers ``layers`` (one, or one per draft: TtMiMoMTP) as the speculative drafter (its caches
        and trace: TtMiMoMTP.setup_cache, capture_spec_traces), and make the packed uploads of the verify and draft
        traces (persistent: before any capture). ``backfill``: the target-hidden history and the ring backfill."""
        assert self.mtp is None and self.decode_trace is None
        self.mtp = TtMiMoMTP(self, ckpt, cache_dir, layers, backfill=backfill)
        kv_rep = kv_lib.kv_replication(self.plan.tp, self.cfg.num_key_value_heads)
        self.spec_inputs = SpecInputs(self.plan, self.max_batch, kv_rep)

    def setup_dflash(self, model_path, cache_dir, load=True):
        """Load the DFlash draft model (``model_path``/dflash, tt/dflash.py) as the speculative drafter (its buffers
        and trace: TtDFlash.setup_cache, capture_spec_traces), and make the verify's packed upload (persistent: before
        any capture). ``load``: read the checkpoint (False: a later replica, from the tensor cache)."""
        assert self.mtp is None and self.dflash is None and self.decode_trace is None
        self.dflash = TtDFlash(self, model_path, cache_dir, load=load)
        kv_rep = kv_lib.kv_replication(self.plan.tp, self.cfg.num_key_value_heads)
        self.spec_inputs = SpecInputs(self.plan, self.max_batch, kv_rep)

    def capture_verify_trace(self):
        """Compile, then capture the verify step: the decode step with every layer's K / V written one slot at a time
        and no feed-back (a verify always reloads its inputs). Every slot idle while capturing: no K / V written.

        The trace starts with the relay step (tt/spec_chain.py: a chained verify's tokens and positions from the
        lane's last result page; a no-op on an upload the host built) and SpecInputs.derive_verify, which writes every
        input but the cos / sin from the packed upload (``verify_enqueue``); the decode step after them is the one
        captured before, followed by the MTP history's verify write (``_verify_step``), which reads the tokens and
        positions the relay left."""
        assert self.verify_trace is None and self.layer_groups is not None, "a verify serves vLLM's KV blocks"
        empty = torch.full((self.max_batch,), -1, dtype=torch.int32)
        host = self.prepare_decode_inputs_host(torch.zeros(self.max_batch, dtype=torch.int32), empty, serial=True)
        inputs = [ttnn.to_device(t, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in host]
        # One capture per attended-group count (attention.verify_variants), every one on the same inputs; the widest
        # last, which is self.verify_trace.
        for variant in verify_variants():
            self._verify_groups = variant
            self._capture_verify_variant(inputs)
            self.verify_traces[variant] = self.verify_trace
            self.verify_trace = None
        self._verify_groups = None
        self.verify_trace = self.verify_traces[verify_variants()[-1]]
        self._verify_active = verify_variants()[-1]

    def _capture_verify_variant(self, inputs):
        """capture_verify_trace for one verify variant (``_verify_groups``) on the trace inputs ``inputs``."""
        rope = self.rope_tables is not None
        # The packed uploads still hold the last served step (a recapture): idle them, so the runs below write no K / V.
        self.spec_inputs.reset()
        self._relay()
        self.spec_inputs.derive_verify(inputs, rope)
        self._release(self._verify_step(inputs))
        ttnn.synchronize_device(self.mesh_device)
        fabric = getattr(self.plan, "fabric_gather", None)
        phase = fabric.phase if fabric is not None else None
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._relay()
        self.spec_inputs.derive_verify(inputs, rope)
        output = self._verify_step(inputs)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        # As the decode traces (capture_decode_trace): an even number of fabric reductions, so verify and decode
        # replays interleave on the gather phases they were captured with.
        fabric = getattr(self.plan, "fabric_gather", None)
        assert fabric is None or fabric.phase == (phase if phase is not None else 0), "odd fabric reductions per step"
        ttnn.synchronize_device(self.mesh_device)
        # The MTP draft program reads the step's final hidden from the norm buffer it was captured on.
        assert output.get("hidden") is self.norm_buffers[0], "the final norm is not the persistent fused-norm buffer"
        self.verify_trace = {"id": trace_id, "inputs": inputs, "output": output, "accept": None}
        if self.verify_sampler is not None:
            # The accept step (VerifySampler) in its own trace on the verify trace's output logits and greedy tokens,
            # then the chain step (tt/spec_chain.py: the result page, and the MTP draft's inputs when a draft chains
            # behind the pass), replayed after an argmax_ids or fused_sample verify; a logits verify replays the
            # verify trace alone. Their programs compiled in capture_decode_trace; this eager run hits the cache.
            assert self.spec_chain is not None and output["next_tokens"] is self.greedy_head.next_tokens
            self.verify_sampler(output["logits"], output["next_tokens"])
            self.spec_chain.chain()
            ttnn.synchronize_device(self.mesh_device)
            fabric = getattr(self.plan, "fabric_gather", None)
            phase = fabric.phase if fabric is not None else None
            accept_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            self.verify_sampler(output["logits"], output["next_tokens"])
            self.spec_chain.chain()
            ttnn.end_trace_capture(self.mesh_device, accept_id, cq_id=0)
            # No fabric reduction in the accept step: replays leave the gather phases where the verify left them.
            fabric = getattr(self.plan, "fabric_gather", None)
            assert fabric is None or fabric.phase == (phase if phase is not None else 0), "fabric phase moved"
            ttnn.synchronize_device(self.mesh_device)
            self.verify_trace["accept"] = accept_id

    def _relay(self):
        """The verify trace's relay step (tt/spec_chain.py), when the replica accepts on device (VerifySampler)."""
        if self.spec_chain is not None and self.verify_sampler is not None:
            self.spec_chain.relay(self.verify_sampler.ctrl)

    def _verify_step(self, inputs):
        """The verify trace's step: with the MTP history, its slots' final norm rows written into their requests'
        rings (the verify upload's chain row names them) at the end."""
        output = self._decode_inputs(inputs, serial=SERIAL_WRITES, verify_groups=self._verify_groups)
        hist = self._history()
        if hist is not None:
            hist.verify(self.spec_inputs.verify_packed, output["hidden"])
        return output

    def capture_spec_traces(self):
        """The verify step and the drafter's program (MTP or DFlash), after the decode trace (persistent buffers
        already exist)."""
        self.capture_verify_trace()
        if self.mtp is not None:
            self.mtp.capture_trace()
        if self.dflash is not None:
            self.dflash.capture_trace()

    def release_spec_traces(self):
        for trace in self.verify_traces.values():
            ttnn.release_trace(self.mesh_device, trace["id"])
            if trace["accept"] is not None:
                ttnn.release_trace(self.mesh_device, trace["accept"])
        self.verify_traces = {}
        self.verify_trace = None
        if self.mtp is not None:
            self.mtp.release_trace()
        if self.dflash is not None:
            self.dflash.release_trace()

    def verify_enqueue(self, tokens, positions, chain=None, accept=None, groups=None):
        """A verify step (tokens, positions: torch [max_batch], -1 an idle slot). Non-blocking; returns the verify
        trace's output (its ``next_tokens`` are the slots' greedy tokens). Copies the packed tokens and positions
        (the trace derives its other inputs from them) with the chain row ``chain`` (reference.spec_chain.chain_row;
        None: no draft chains behind the pass), and the RoPE cos / sin unless the trace reads them from the device
        tables. ``accept``: the pass's (slots, groups) for the accept trace (``verify_accept_enqueue``), staged first:
        the verify trace's relay step writes a chained pass's drafts into the control page. ``groups``: the verify
        variant to replay (attention.verify_variant of the pass's highest used group; None: every group)."""
        assert int(positions.max()) < self.max_seq_len, f"position {int(positions.max())} >= {self.max_seq_len}"
        variant = verify_variants()[-1] if groups is None else groups
        trace = self.verify_traces[variant]
        self._verify_active = variant
        if accept is not None:
            self.verify_sampler.stage(*accept)
        spec = self.spec_inputs
        ttnn.copy_host_to_device_tensor(spec.verify_host(tokens, positions, chain), spec.verify_packed)
        if self.rope_tables is None:
            host = (
                *self.rope_setups["swa"].decode_host_inputs(positions),
                *self.rope_setups["full"].decode_host_inputs(positions),
            )
            for h, d in zip(host, trace["inputs"][2:6]):
                ttnn.copy_host_to_device_tensor(h, d)
        ttnn.execute_trace(self.mesh_device, trace["id"], cq_id=0, blocking=False)
        return trace["output"]

    def verify_accept_enqueue(self, slots=None, groups=None):
        """After ``verify_enqueue``: the pass's acceptance on device (VerifySampler) with its sampled slots' parameters
        and its requests (reference.sample_verify.pass_inputs; staged here unless ``verify_enqueue`` staged them),
        then the chain step. Non-blocking; ``spec_chain.read`` returns the pass's result page (after
        ``mtp.enqueue_chained``, with the drafts chained behind it)."""
        trace = self.verify_traces.get(self._verify_active)
        assert trace is not None and trace["accept"] is not None, "no accept trace captured"
        if groups is not None:
            self.verify_sampler.stage(slots, groups)
        ttnn.execute_trace(self.mesh_device, trace["accept"], cq_id=0, blocking=False)

    def read_next_tokens(self, handle):
        """The greedy token of every slot of a decode or verify step, int64 [max_batch] (blocks)."""
        return self.plan.to_torch_replicated(handle["next_tokens"]).reshape(-1).to(torch.int64)

    def release_prefill_traces(self):
        """Release the captured prefill traces (their buffers stay, for a later capture)."""
        for trace in self.prefill_traces.values():
            if trace["id"] is not None:
                ttnn.release_trace(self.mesh_device, trace["id"])
                trace["id"] = None

    def decode_enqueue(self, tokens, positions, sample=False):
        """tokens, positions: torch [max_batch]; position -1 marks an empty slot. Non-blocking.
        Returns a handle for ``decode_read`` (full logits) or ``decode_read_topk`` (per-chip candidates).
        ``sample`` (vLLM mode): the DeviceSampler draws the tokens (its parameters and uniforms set beforehand)."""
        assert int(positions.max()) < self.max_seq_len, f"position {int(positions.max())} >= {self.max_seq_len}"
        host = self.prepare_decode_inputs_host(tokens, positions)
        self._advance_swa(positions)
        if self.decode_trace is not None:
            for h, d in zip(host, self.decode_trace["inputs"]):
                ttnn.copy_host_to_device_tensor(h, d)
            return self._decode_replay(sample)
        inputs = [ttnn.to_device(t, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in host]
        out = self._decode_inputs(inputs, sample=sample)
        for t in inputs:
            t.deallocate(True)
        return out

    def decode_enqueue_resident(self, positions, sample=False):
        """A decode step on the tokens and positions the last replay fed back (``_feed_back``): copies only the
        RoPE inputs of ``positions`` (torch [max_batch], what the trace's own positions now hold), then replays the
        trace (``sample``: the sampled one). Non-blocking; returns the trace's output."""
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
        return self._decode_replay(sample)

    def _release(self, handle):
        trace = self.decode_trace
        traced = (trace, trace["sampled"] if trace is not None else None, *self.verify_traces.values())
        if not any(t is not None and handle is t["output"] for t in traced):
            persistent = [h.next_tokens for h in (self.greedy_head, self.sampler) if h is not None]
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
