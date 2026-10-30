# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-9B text model for Blackhole P150.

tok_embeddings -> 32 x Qwen36DecoderLayer -> RMSNorm -> LM Head.
Hybrid state: KV cache (8 attn layers) + recurrent state (24 DeltaNet layers).
"""

import contextlib
import math
import os
from collections import OrderedDict

import torch
from loguru import logger
from tqdm import tqdm
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt import tp_common as _tpc
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.prefix_cache import alloc_zeros_like
from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup
from models.demos.blackhole.qwen36.tt.spec_masks import MASK_NEG, MASK_ROWS, SHARD_WORDS
from models.demos.blackhole.qwen36.tt.tp_common import aliases as _tpc_aliases
from models.demos.blackhole.qwen36.tt.tp_common import kv_cache_bf8_enabled, kv_write_masked_enabled
from models.demos.blackhole.qwen36.tt.tp_common import lm_head_kwargs as _lm_head_kwargs
from models.demos.blackhole.qwen36.tt.tp_common import (
    pack_u32_rows,
    prefill_masked_big_enabled,
    rope_row_index,
    spec_hostfix_enabled,
    spec_hostset_enabled,
    spec_hostset_trace_enabled,
    verify_attn_lean_enabled,
    verify_dar_enabled,
    verify_device_topk_m,
)
from models.tt_transformers.tt.common import Mode, get_block_size, num_blocks_in_seq


def serve_device_decode_env_enabled():
    """QWEN36_SERVE_DEVICE_DECODE (default "1"): device-resident serving decode (see Qwen36Model._serve_device_decode).
    "0" keeps the host-reload decode (host-packed RoPE, [B,1] token input, no feedback / async) byte for byte."""
    return os.environ.get("QWEN36_SERVE_DEVICE_DECODE", "1") == "1"


class VerifyHandle:
    """What ``verify_after_external_sync`` needs from ``verify_stage_and_enqueue``."""

    __slots__ = ("rows", "B", "read_logits", "read_topk", "read_col0", "fused", "bucket", "post_pending", "mask_keep")

    def __init__(self, rows, B, read_logits, read_topk, read_col0, fused, bucket=None, post_pending=False):
        self.rows, self.B = rows, B
        self.read_logits, self.read_topk, self.read_col0, self.fused = read_logits, read_topk, read_col0, fused
        self.bucket = bucket  # the row bucket b < B the step replayed (QWEN36_SPEC_BUCKETS), None = the B-row traces
        self.post_pending = (
            post_pending  # forward_only enqueue: the main trace is queued, verify_enqueue_post is still due
        )
        self.mask_keep = None  # the ttnn host tensor of this step's mask write (kept until the step's event; phase 3)


class _VfyBucket:
    """One verify row bucket b < B (QWEN36_SPEC_BUCKETS): its traces and bucket-local buffers. The state it runs on (GDN ring,
    window pair, ctrl / aux / token / page-table buffers, KV) is the shared B-row state; only the b-row outputs live here.
    ``handoff`` holds the destinations of the in-trace hand-off into the shared B-row outputs: stand-ins while the bucket is
    warmed (the live B-row outputs do not exist yet), the live ``_vfy_*_out`` tensors while it is captured."""

    def __init__(self, b, rows):
        self.b, self.rows = b, rows
        self.dl = None  # _DeviceDecodeLoop(batch=rows)
        self.trace_id = self.logits_trace_id = self.topk_trace_id = None
        self.stats_trace_id = (
            None  # the stats trace (QWEN36_SPEC_DEVICE_LOGPROBS / _TOPP), see _forward_verify_stats_body
        )
        self.mask_trace_id = None  # the mask trace (QWEN36_SPEC_DEVICE_MASK), see _forward_verify_mask_body
        self.pen_trace_id = None  # the pen trace (QWEN36_SPEC_DEVICE_PENALTIES), see prepare_verify_pen
        self.shard_spec = (
            None  # (shape, dtype, layout, memory_config) of this bucket's shard logits (stats warm stand-in)
        )
        self.normed = self.shard_logits = self.logits_out = self.logits_rm_out = None  # bucket-local trace outputs
        self.narrow_ready = False
        self.handoff = {}  # "ids", "rows", "topk_vals", "topk_ids" -> destination tensor
        self.standins = []  # tensors freed once the bucket is captured


class _ShortPrefillTrace:
    """Persistent device buffers + outputs of ONE captured bucket short-prompt prefill trace (any length <= bucket).

    Every per-request input lives in a persistent (replicated) device tensor whose address is baked into
    the trace; a request overwrites them with copy_host_to_device_tensor and replays the trace. The
    host-side caches below hold the request-independent host tensors (re-DMA'd each request, so the
    persistent copies never have to survive other traces' replays)."""

    def __init__(self, bucket, buf_blocks, blocks_per_bucket):
        self.bucket = bucket
        self.buf_blocks = buf_blocks  # width of the full page-table buffer (a 32-multiple)
        self.blocks_per_bucket = blocks_per_bucket
        self.trace_id = None
        # Persistent inputs (device).
        self.tok = None  # [1, bucket] uint32 RM
        self.csi = None  # [1] int32 RM: chunk_start_idx (always 0 for a from-scratch short prompt)
        self.full_pt = None  # [1, buf_blocks] int32 RM
        self.chunk_pt = None  # [1, blocks_per_bucket] int32 RM
        self.cos = None  # [1, 1, bucket, rope_head_dim] bf16 TILE
        self.sin = None
        self.sel = None  # [1, 1, 1, bucket] one-hot (row valid_len-1) selecting the last real position
        self.sel_dtype = None
        # Arbitrary-length GDN masks (persistent, rewritten per request): see the short-prefill section comment.
        self.m_bg = None  # [1, bucket, 1] fp32 TILE: 1 for t < valid_len (multiplied into beta and g)
        self.conv_sel = None  # [1, K-1, bucket] bf16 TILE one-hot of the last K-1 real conv-input rows
        self.m_qkv = None  # [1, bucket, 1] bf16 TILE q/k/v mask (QWEN36_PREFILL_TRACE_QKV_MASK=1 only)
        self.gdn_masks = None  # dict handed to every GDN layer (None: exact-bucket legacy trace, no KDA conv)
        self.masked = False  # True when the trace serves any length <= bucket
        self.n_hist = None  # K-1
        self.block_size = None
        self.gdn_refs = ()  # the GDN rec_state tensors bound at capture (binding check at replay)
        # Trace outputs (device): per-shard argmax index + max value, and the pre-gather vocab-sharded logits.
        self.idx = None
        self.val = None
        self.logits = None  # [1, 1, 1, vocab/TP]
        # Spec variant (QWEN36_SPEC_FAST_PREFILL, MTPSpecEngine): the body also feeds the MTP drafter over the
        # segment (its KV warm-up) and keeps the drafter feed row of the last real position (slot Tp-1).
        self.spec = False
        self.mtp_tok = None  # [1, bucket] uint32 RM: MTP input tokens (slot i fused with prompt token i+1)
        self.mtp_full_pt = None  # [1, buf_blocks] int32 RM: the MTP page table (padding -> the MTP scratch block)
        self.mtp_chunk_pt = None  # [1, blocks_per_bucket] int32 RM
        self.last_feed = None  # output [1, 1, 1, dim/TP] bf16: spec_feed_rows(hidden) at row valid_len-1
        # Host caches (request-independent).
        self.csi_host = None
        self.sel_host = None
        self.text_rope_host = None  # (cos_host, sin_host) for text-only requests (plain 1D RoPE)
        self.host_cache = OrderedDict()  # valid_len -> host mask tensors (LRU)


class _DeviceDecodeLoop:
    """Persistent device buffers of the device-resident single-user greedy decode loop
    (Qwen36Model.setup_device_decode_loop). Everything here is allocated BEFORE the decode trace is captured and its
    address is baked into the trace; the loop state (token / position / RoPE index / history cursor) is advanced by
    the trace itself. ``host_refs`` keeps the host tensors of the last arm() alive until the device has consumed
    them."""

    def __init__(self, **buffers):
        self.__dict__.update(buffers)
        self.host_refs = []

    def release(self):
        self.host_refs = []
        for name, buf in list(vars(self).items()):
            if isinstance(buf, ttnn.Tensor):
                ttnn.deallocate(buf)
                setattr(self, name, None)


def spec_kv_write_conflicts(page_tables, positions, T, block_size):
    """Cross-user collisions of the grouped verify KV write (see _spec_kv_write_conflicts_old for the rule).
    QWEN36_SPEC_HOSTFIX (default 1): a [B, nb] tensor is read at the <= B*T needed entries instead of tolist() of the
    whole table (~33k ints at nb = 4097); same result and same errors. Returns [(j, u_first, u_second, block, tile)]."""
    if not (
        spec_hostfix_enabled()
        and isinstance(page_tables, torch.Tensor)
        and page_tables.dim() == 2
        and len(positions) <= page_tables.shape[0]
    ):
        return _spec_kv_write_conflicts_old(page_tables, positions, T, block_size)
    nb = page_tables.shape[1]
    users = [u for u, p in enumerate(positions) if int(p) >= 0]
    if not users:
        return []
    base = torch.tensor([int(positions[u]) for u in users], dtype=torch.int64)
    pos = base[None, :] + torch.arange(T, dtype=torch.int64)[:, None]  # [T, U]: call j, active user
    col = pos // block_size
    over = col >= nb
    if bool(over.any()):
        j, i = (int(x) for x in over.nonzero()[0])  # first in the old (j, u) order
        raise IndexError(f"user {users[i]} writes position {int(pos[j, i])}, past its {nb}-block page table")
    blks = page_tables[torch.tensor(users)[None, :].expand(T, -1), col].tolist()
    tiles = ((pos % block_size) // ttnn.TILE_SIZE).tolist()
    conflicts = []
    for j in range(T):
        owner = {}  # (block, tile) -> first user that writes it in call j
        for i, u in enumerate(users):
            blk = int(blks[j][i])
            if blk == 0:
                continue
            key = (blk, tiles[j][i])
            first = owner.setdefault(key, u)
            if first != u:
                conflicts.append((j, first, u, *key))
    return conflicts


def _spec_kv_write_conflicts_old(page_tables, positions, T, block_size):
    """Cross-user collisions of the grouped verify KV write. Call j writes candidate j of every
    user (one user per core) to the 32-row cache tile (page_tables[u][pos // block_size],
    (pos % block_size) // 32) at pos = positions[u] + j; two users on the same tile in the same
    call race (read-modify-write, one update lost). Writes to block 0 (vLLM's null block: padding rows,
    padded columns past a row's reservation) are junk by design and skipped. Users with a negative position
    are skipped.
    Returns [(j, u_first, u_second, block, tile)], empty when the write is safe."""
    rows = page_tables.tolist() if isinstance(page_tables, torch.Tensor) else [list(r) for r in page_tables]
    conflicts = []
    for j in range(T):
        owner = {}  # (block, tile) -> first user that writes it in call j
        for u, p in enumerate(positions):
            if int(p) < 0:
                continue
            pos = int(p) + j
            if pos // block_size >= len(rows[u]):
                raise IndexError(f"user {u} writes position {pos}, past its {len(rows[u])}-block page table")
            blk = int(rows[u][pos // block_size])
            if blk == 0:
                continue
            key = (blk, (pos % block_size) // ttnn.TILE_SIZE)
            first = owner.setdefault(key, u)
            if first != u:
                conflicts.append((j, first, u, *key))
    return conflicts


def _page_table_rows(page_tables):
    """int32 contiguous [B, nb] torch rows from a tensor or array-like (1-D is one user's row)."""
    pt = torch.as_tensor(page_tables)
    return (pt.reshape(1, -1) if pt.dim() == 1 else pt).to(torch.int32).contiguous()


class Qwen36Model:
    """Qwen3.5-9B text LM on Blackhole P150. HF_MODEL env var selects checkpoint."""

    _rope_perm = False  # QWEN36_ROPE_PERM, resolved per instance in __init__ (class default for partial test doubles)

    # QWEN36_SPEC_BUCKETS row buckets (see _VfyBucket): None / empty = only the B-row verify trace exists.
    _vfy_bk = None
    _vfy_last_bucket = None

    def __init__(self, mesh_device, args, state_dict, tensor_cache_path=None):
        self.args = args
        self.device = mesh_device
        self.mesh_device = mesh_device  # Generator reads model.mesh_device
        self.num_devices = mesh_device.get_num_devices()
        # CCL for multi-device all-reduce; None on single device (ops no-op).
        if self.num_devices > 1:
            from models.tt_transformers.tt.ccl import TT_CCL

            self.tt_ccl = TT_CCL(mesh_device)
        else:
            self.tt_ccl = None
        self.configuration = args  # Generator reads model.configuration.max_seq_len
        self.sampling_dp = 1
        # QWEN36_SERVE_DEVICE_DECODE (default 1), TP meshes only (the RoPE lookup / in-trace increments are the
        # device-loop recipe of change 1.6, which needs a multi-device mesh): the serving decode (Generator ->
        # prepare_decode_inputs_host / ttnn_decode_forward) is device-resident, like qwen38_27b_qb2:
        #   * token input = a [1,1,1,32] uint32 ROW_MAJOR buffer (B's layout; sliced to the B active rows in-trace) that
        #     the Generator's sampler overwrites in place with the sampled tokens (token feedback);
        #   * RoPE cos/sin are looked up in-trace from rope_idx (= max(pos, 0) + rope_delta, uint32 RM [1, B]) in the
        #     replicated decode tables -- no per-step host cos/sin;
        #   * the device-sampling variant of the trace advances cur_pos (skip_negative_entries) and rope_idx once per
        #     replay, so a steady step needs no host input reload.
        # Flag 0 (or a single-device mesh): host-packed RoPE, [B,1] token input, no feedback (every step is reloaded).
        self._serve_device_decode = serve_device_decode_env_enabled() and self.num_devices > 1
        self._tt_supports_decode_token_feedback = self._serve_device_decode
        # Reuses the vocab-sharded lm_head as the sampler's shard: needs divisible vocab; 64K = top-k limit.
        mesh_shape = tuple(int(dim) for dim in mesh_device.shape)
        self._supports_on_device_sampling = (
            mesh_shape in ((1, 4), (1, 8))
            and args.vocab_size % self.num_devices == 0
            and (args.vocab_size // self.num_devices <= 64 * 1024)
        )
        if self._supports_on_device_sampling:
            from models.common.sampling.generator import SamplingGenerator

            # vocab/num_devices isn't a power of 2; the multi-device TopK kernel needs it padded.
            args.pad_logits_to_power_of_2 = True
            # force_argmax (the cheap 1-all-gather greedy path) is enabled on the base
            # SAMPLING_AG_CONFIG in model_config.py and runs IN-TRACE (faster than eager). Decode
            # bucketing is made compatible with the in-trace sampler by namespacing the sampling
            # trace per bucket width (SamplingGenerator.set_trace_bucket, driven from
            # qwen36_vllm.decode_forward) — see generator._validate_trace_inputs.
            # vLLM v1 already gives each n>1 child its own seed (seed + i), so concurrent requests that share a seed are
            # independent requests that must produce the same stream; per-slot salting would make them diverge.
            args.salt_duplicate_seeds = False
            self.sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=self.tt_ccl)
        else:
            self.sampling = None

        # Framework Embedding (mesh-aware; replicates on 1-device mesh).
        from models.tt_transformers.tt.embedding import Embedding

        self.embd = Embedding(
            mesh_device=mesh_device,
            args=args,
            weight_cache_path=tensor_cache_path,
            state_dict=state_dict,
            dtype=ttnn.bfloat16,
        )

        # RoPE setup (for gated attention layers only)
        self.rope = Qwen36RoPESetup(mesh_device, args)
        # QWEN36_ROPE_PERM (default on): the base model's TP full-attention layers hold permuted q/k channels and take hd-wide
        # permuted cos/sin (rope_tp.widen_cos_sin). The MTP drafter is unchanged (64-wide). Must match Qwen36DecoderLayer.
        from models.demos.blackhole.qwen36.tt.attention.rope_tp import rope_perm_env_on, rope_perm_supported

        self._rope_perm = bool(
            self.num_devices > 1 and rope_perm_env_on() and rope_perm_supported(args.head_dim, args.rope_head_dim)
        )
        self.rope.wide_hd = args.head_dim if self._rope_perm else None
        if self._serve_device_decode:
            # The serving decode trace looks cos/sin up in these tables: allocate them now, outside any trace.
            self.rope.decode_rm_tables()

        # layer_indices (from from_pretrained) picks checkpoint layers; else 0..n_layers-1.
        # Each layer uses its real checkpoint index for weights and type (DeltaNet vs attn).
        self.layer_indices = getattr(args, "layer_indices", None) or list(range(args.n_layers))

        # Per-request vision grid (t,h,w), stashed by get_image_features / get_video_features so the
        # prefill paths can build the multimodal 3D RoPE (M-RoPE) position ids without threading
        # grid_thw through every prefill signature. Exactly one is non-None for a multimodal request
        # (image XOR video); both None => text-only. The active one also selects which placeholder
        # token id (image_token_id vs video_token_id) the vision-splice paths look for.
        self._req_image_grid_thw = None
        self._req_video_grid_thw = None

        # QWEN36_DECODE_ALLREDUCE=1 (decode only): replicated 40-core L1 residual + all_reduce_async (see
        # tp_common.DecodeResidualAllReduce). Built here so its persistent L1 buffer is allocated at
        # construction (never inside a trace); the layers below pick up the same shared instance. None when
        # off/unsupported -> the fractured-residual decode path is unchanged. Prefill never uses it.
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        self._decode_ar = tpc.get_decode_allreduce(args, self.tt_ccl) if self.num_devices > 1 else None

        # Transformer layers
        logger.info(f"Loading {len(self.layer_indices)} transformer layers (indices={self.layer_indices})...")
        self.layers = []
        for i in tqdm(self.layer_indices, desc="Loading layers"):
            layer = Qwen36DecoderLayer(mesh_device, args, state_dict, i, tensor_cache_path, tt_ccl=self.tt_ccl)
            self.layers.append(layer)

        # Framework RMSNorm (add_unit_offset=True). Single device: is_distributed=None.
        # 27B TP: hidden is sharded -> pass is_distributed + tt_ccl or use DistributedNorm.
        self.norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            weight_key="norm",
            weight_cache_path=tensor_cache_path,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=args.norm_eps,
            **(
                dict(is_distributed=args.is_distributed_norm, ccl_topology=args.ccl_topology(), tt_ccl=self.tt_ccl)
                if self.num_devices > 1
                else {}
            ),
        )
        # Speculative-decode feed contract (see spec_feed_rows): the MTP drafter is fed the OUTPUT of
        # this final norm, not the residual stream before it. spec_norm is that norm with the
        # trailing all-gather off, so its output keeps the fractured [1,1,*,dim/tp] shape/dtype the
        # drafter already consumes.
        self.spec_norm = None
        if self.num_devices > 1:
            # TP: DistributedNorm all-gathers fractured hidden for LM head.
            from models.tt_transformers.tt.distributed_norm import DistributedNorm

            self.norm = DistributedNorm(self.norm, args, tt_ccl=self.tt_ccl, TG=args.is_galaxy)
            # Same RMSNorm object (no weight duplication), all-gather disabled. Only valid when the
            # PREFILL norm is the distributed (stats-gathered) form, whose inner output is already
            # fractured to dim/tp; a gather-then-norm PREFILL would hand back the full width instead.
            if args.is_distributed_norm(Mode.PREFILL):
                self.spec_norm = DistributedNorm(
                    self.norm.norm, args, tt_ccl=self.tt_ccl, TG=args.is_galaxy, enable_all_gather=False
                )
        else:
            # Single device: the plain RMSNorm's full-dim output IS the fractured form.
            self.spec_norm = self.norm
        assert self.spec_norm is not None, (
            "the spec feed contract needs the fractured-output distributed final norm "
            "(args.is_distributed_norm(Mode.PREFILL) is False on this mesh/dim)"
        )

        # LM head [in,out]. Mesh: vocab-sharded (dim=-1); _lm_head all-gathers logits.
        # M=1 decode is weight-read-bound (~1.3GB/token), so sharding cuts bandwidth;
        # gather moves only the logit row. REPLICATED fallback if vocab indivisible.
        lm_head_weight = state_dict["output.weight"]  # [vocab_size, dim]; transposed on cache miss only
        vocab_rows = lm_head_weight.shape[0]
        self._lmhead_vocab_sharded = self.num_devices > 1 and vocab_rows % self.num_devices == 0
        if self.num_devices > 1 and not self._lmhead_vocab_sharded:
            logger.warning(
                f"LM-head vocab {vocab_rows} not divisible by num_devices "
                f"{self.num_devices}; falling back to replicated LM head."
            )
        if self._lmhead_vocab_sharded:
            # Separate cache (.vshard): as_tensor ignores mesh_mapper on reload.
            lm_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
            lm_cache = tensor_cache_path / "output.weight.vshard" if tensor_cache_path else None
        else:
            lm_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if self.num_devices > 1 else None
            lm_cache = tensor_cache_path / "output.weight" if tensor_cache_path else None
        self.lm_head_weight = ttnn.as_tensor(
            lm_head_weight,
            preprocess=lambda t: t.T.contiguous(),  # [dim, vocab_size]
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=lm_cache,
            **(dict(mesh_mapper=lm_mapper) if lm_mapper is not None else {}),
        )

        self.vocab_size = args.vocab_size
        # True: return pre-gather vocab-sharded logits for per-shard argmax + host combine.
        self._ondev_argmax = False
        self._paged_kv_caches = None
        # Positions in self.layers of full-attn layers (not checkpoint indices); drives KV cache bind.
        self._attention_layer_indices = [pos for pos, layer in enumerate(self.layers) if layer.is_full_attention]
        # Chunk-outer trace: one all-layer chunk captured, replayed per chunk via DMA inputs.
        # Persistent buffers below; addresses baked into trace.
        self._chunked_trace_id = None
        self._chunked_prepared_key = None
        self._gdn_slot_ops_warmed = False
        # Per-slot traces of the whole GDN slot write (capture_gdn_slot_write_traces): slot -> trace id.
        self._gdn_slot_write_traces = {}
        self._gdn_slot_write_fp = None
        self._gdn_slot_write_fmt = None  # GDN conv format the slot-write traces were captured under ("packed" / "both")
        self._gdn_slot_write_warned = False
        self._gdn_slot_write_replayed = False
        self._chunked_trace_output = None
        self._chunked_chunk_size = None
        self._chunk_token_buf = None
        self._chunk_start_idx_tensor = None
        self._chunk_page_table_buf = None
        self._chunk_full_page_table_buf = None
        self._chunk_cos_buf = None
        self._chunk_sin_buf = None
        # QWEN36_PREFILL_BIG_CHUNK (default 0 = off): a second chunk trace of this many rows (a multiple of 2048) that
        # replays the full blocks of a prefill segment before the 2048-row chunks; the paged SDPA splits its work over
        # heads x Q-chunks only, so the bigger chunk keeps far more cores busy at depth. The 2048 set stays active.
        self._prefill_big_chunk = int(os.environ.get("QWEN36_PREFILL_BIG_CHUNK", "0") or 0)
        assert self._prefill_big_chunk % 2048 == 0, "QWEN36_PREFILL_BIG_CHUNK must be 0 or a multiple of 2048"
        self._chunk_sets = {}  # inactive chunk-trace state by label ("small" / "big"); the active one is on self
        self._chunk_set_active = "small"
        self._last_chunk = None  # (trace output, rows) of the last replayed chunk (either set)
        # Traced batched short-prompt (bucket) prefill: one B=1 full-bucket trace replayed
        # once per user (see capture_prefill_trace_bucket / prefill_traced_bucket_batched).
        self._bucket_trace_id = None
        self._bucket_trace_output = None
        self._bucket_size = None
        self._bucket_token_buf = None
        self._bucket_start_idx_tensor = None
        self._bucket_page_table_buf = None
        self._bucket_full_page_table_buf = None
        self._bucket_cos_buf = None
        self._bucket_sin_buf = None
        self._gdn_batched_prev = None  # batched GDN bindings saved during bucket-trace capture
        # Traced B=1 EXACT-bucket short-prompt prefill (QWEN36_PREFILL_BUCKET_TRACE): bucket -> _ShortPrefillTrace.
        # Captured in the untimed warmup phase (capture_prefill_trace_short), replayed by prefill_short_traced.
        self._short_traces = {}
        # Prepared (inputs allocated, programs compiled) but not yet captured short traces: prepare_prefill_traces_short
        # -> record_prefill_traces_short (the spec engine splits the two so every allocation precedes every capture).
        self._short_traces_pending = {}
        self._spec_sel_cc = None  # exact one-hot row gather (HiFi4, fp32 acc) of the spec short trace's feed row
        self._logits_read_comp = None  # lazily built ConcatMeshToTensor(dim=-1) for the short trace's logits
        # Block that absorbs the padded K/V writes of a short trace (blocks >= ceil(valid_len/block) of its bucket).
        # vLLM: block 0 is the null block (never allocated to a request). Demos/tests with an identity page table
        # must set a spare block (a block no request owns) before replaying a trace.
        self.prefill_trash_block = 0
        self._argmax_read_comp = None  # lazily built ConcatMeshToTensor(dim=0) for read_shard_argmax_token
        # Persistent B=1 GDN prefill scratch (batched serving): allocated once at warmup, its buffer
        # addresses are baked into the chunk-prefill trace and reused by every prefill_paged_slots
        # replay, so it is never freed/reallocated (only zeroed in place). See _bind_gdn_prefill_scratch.
        self._gdn_prefill_scratch = None
        # GDN prefix-state cache (APC, see prefix_cache.py); built by the vLLM wrapper before warmup. None = off.
        self._prefix_cache = None
        # Chunked-prefill continuation store (same snapshot class, own slots; built by the vLLM wrapper before warmup
        # when vLLM can split a prompt across engine steps). None = off. _chunk_budget: (per-step token budget,
        # long_prefill_token_threshold, prefill_chunk_alignment); budget None = unknown (snapshot every aligned end).
        self._chunk_state_cache = None
        self._chunk_budget = (None, 0, 128)
        # Per-decode-slot M-RoPE delta (batched serving, max_batch_size > 1): decode rope row = kv_pos + delta[slot].
        # Set at each user's prefill (0 for text), moved with the GDN state on a slot_remap. B=1 keeps rope.rope_delta.
        self._slot_rope_delta = torch.zeros(max(int(args.max_batch_size), 1), dtype=torch.int64)
        # True while the short-trace init buffers (dn._prefill_init_*) hold a carried state instead of zeros.
        self._prefill_init_dirty = False

        # Optional vision tower (DropInVisionTransformer), attached lazily by
        # init_vision_model() for the multimodal serving path. None on the text-only path.
        self.vision_model = None
        self.vision_args = None

        # Trace-safe vision splice (traced serving path). The chunk/masked-bucket forwards run a
        # FIXED-shape ttnn.where(mask, vision, text) over these persistent buffers — compiled once
        # at warmup, then updated per request via copy_host_to_device, so no per-request program
        # ever compiles to clobber a parked trace. Allocated (single device only) in
        # capture_prefill_trace_chunked; None means "no traced path" -> the where is skipped.
        self._vis_buf = None  # [1, chunk_size, dim] bf16, image rows placed at their positions
        self._vis_mask_buf = None  # [1, chunk_size, 1] bf16, 1 at image positions else 0
        self._vis_zero_mask_host = None  # cached host zero mask for the clear (text/tail) path

        # MTP (multi-token prediction) drafter head for speculative decode. Built only when the
        # checkpoint ships mtp.* weights AND the config declares an MTP head. Its own paged KV
        # cache is allocated alongside the base caches in allocate_kv_caches.
        self.mtp = None
        self._mtp_kv_cache = None
        if getattr(args, "has_mtp", False) and "mtp.fc.weight" in state_dict:
            from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP

            logger.info("Building MTP (multi-token prediction) drafter head...")
            self.mtp = Qwen36MTP(
                mesh_device, args, state_dict, parent=self, tensor_cache_path=tensor_cache_path, tt_ccl=self.tt_ccl
            )

    def init_vision_model(self, reference_visual=None, vision_args=None, dtype=ttnn.bfloat8_b, debug=False):
        """Build and attach the TT vision tower (DropInVisionTransformer).

        Runs on the SAME mesh as the text model. Still needs the HF reference visual for patch embed
        / positional-interpolation (not ported to TT); if ``reference_visual`` is None it is loaded
        via ``VisionModelArgs.reference_vision_model``. Idempotent — returns the existing tower if
        already built. ``vision_args`` / ``dtype`` default internally; ``debug`` runs the reference
        path alongside and logs PCC.
        """
        if self.vision_model is not None:
            return self.vision_model
        from models.demos.blackhole.qwen36.tt.vision.model import DropInVisionTransformer
        from models.demos.blackhole.qwen36.tt.vision.vision_model_config import VisionModelArgs

        if vision_args is None:
            vision_args = VisionModelArgs(
                self.mesh_device,
                max_batch_size=self.args.max_batch_size,
                max_seq_len=self.args.max_seq_len,
            )
        if reference_visual is None:
            reference_visual = vision_args.reference_vision_model(depth=vision_args.hf_config.vision_config.depth)
        self.vision_args = vision_args
        self.vision_model = DropInVisionTransformer(reference_visual, vision_args, dtype=dtype, debug=debug)
        return self.vision_model

    def get_image_features(self, pixel_values, image_grid_thw):
        """Run the vision tower over a single user's images.

        Mirrors HF ``get_image_features``: pixel patches in, packed image embeddings out — one row
        per image-placeholder token, ready for ``_scatter_vision_tokens``. ``pixel_values`` is
        ``[num_patches, patch_dim]``; ``image_grid_thw`` is per-image ``(t,h,w)`` ``[num_images, 3]``.
        Returns ``[num_image_tokens, H]``, hidden-fractured on a mesh like the text embeddings.
        """
        assert self.vision_model is not None, "init_vision_model() must be called before get_image_features()"
        # Stash the grid (as an IMAGE grid) so the prefill paths can build M-RoPE position ids for
        # this request (the splice positions in input_ids + the (t,h,w) grid are all M-RoPE needs).
        # Clear any stale video grid so the modality (and thus the placeholder token id) is image.
        self._req_image_grid_thw = image_grid_thw
        self._req_video_grid_thw = None
        image_features = self.vision_model.forward(pixel_values, grid_thw=image_grid_thw, host_trim=True)
        # The vision tower returns [1, B, S, H]; flatten the leading (batch/seq) dims to the
        # packed [num_image_tokens, H] rows the text-model splice (_scatter_vision_tokens /
        # _set_vision_merge) expects. The hidden dim is unchanged so the mesh hidden-fracture
        # is preserved. B == 1 for now.
        # host_trim=True already returns the packed 2D [num_image_tokens, H] tensor (shape-stable: no device op
        # depends on the exact request size).
        return image_features

    def get_video_features(self, pixel_values_videos, video_grid_thw):
        """Run the vision tower over a single user's video frames.

        Same forward as ``get_image_features`` on the video pixels/grid. Downstream differs: M-RoPE
        treats the grid as VIDEO (split per frame by timestamps, modality==2), and embeddings splice
        into ``video_token_id`` placeholders. Stashing the grid here as a video grid selects both.
        ``pixel_values_videos`` ``[num_patches, patch_dim]``; ``video_grid_thw`` ``[num_videos, 3]``.
        Returns ``[num_video_tokens, H]``, hidden-fractured on a mesh like the text embeddings.
        """
        assert self.vision_model is not None, "init_vision_model() must be called before get_video_features()"
        # Stash the grid as a VIDEO grid; clear any stale image grid so the modality (and the
        # placeholder token id) is video.
        self._req_video_grid_thw = video_grid_thw
        self._req_image_grid_thw = None
        video_features = self.vision_model.forward(pixel_values_videos, grid_thw=video_grid_thw, host_trim=True)
        return video_features  # packed 2D [num_video_tokens, H] (see get_image_features)

    def _vision_placeholder_token_id(self):
        """The input-id the current request's vision embeddings splice into: ``video_token_id`` for
        a video request (video grid stashed), else ``image_token_id``. The vision-splice paths
        (_scatter_vision_tokens / _set_vision_merge / _vis_row_offset_for) use this to locate the
        placeholder positions, mirroring HF's ``input_ids == image_token_id`` /
        ``input_ids == video_token_id`` masks."""
        if self._req_video_grid_thw is not None:
            return int(self.args.hf_config.video_token_id)
        return int(self.args.hf_config.image_token_id)

    def _build_request_rope(self, token_ids, vision_tokens):
        """Stage the per-request RoPE for this prefill: M-RoPE (3D position ids + rope_delta) when
        the request is multimodal (vision_tokens present -> use the grid stashed by
        get_image_features / get_video_features), else clear to ordinary 1D RoPE. Call once at a
        prefill entry point with the REAL token ids (token_ids[:, :actual_len]); the chunk/tail
        seams then slice the staged table by sequence position and decode offsets by rope_delta."""
        image_grid = self._req_image_grid_thw if vision_tokens is not None else None
        video_grid = self._req_video_grid_thw if vision_tokens is not None else None
        self.rope.build_request_rope(token_ids, image_grid_thw=image_grid, video_grid_thw=video_grid)

    def _alloc_vision_merge_buffers(self, device, chunk_size):
        """Allocate the persistent vision-splice buffers used by the traced prefill path.

        ``_vis_buf`` holds image embeddings at their token positions (zeros elsewhere);
        ``_vis_mask_buf`` is the 0/1 mask. Both start zero so the captured ttnn.where is identity
        until a multimodal request stages them. Allocate before warmup so the where compiles in the
        warmup pass (then captured), never at request time. Shapes match the consuming activations:
        single device vis [1, chunk_size, dim] / mask [1, chunk_size, 1]; TP vis [1,1,chunk_size,dim]
        HIDDEN-SHARDED like embd (ShardTensor2dMesh dims=(None,-1)), mask [1,1,chunk_size,1] REPLICATED
        (broadcasts over sharded hidden). DropInVisionTransformer fractures the same way so columns line up.
        """
        if self._vis_buf is not None:
            return
        H = self.args.dim
        if self.num_devices > 1:
            shard = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -1), mesh_shape=self.args.cluster_shape)
            rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
            self._vis_buf = ttnn.from_torch(
                torch.zeros(1, 1, chunk_size, H, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=shard,
            )
            self._vis_mask_buf = ttnn.from_torch(
                torch.zeros(1, 1, chunk_size, 1, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )
            self._vis_zero_mask_host = ttnn.from_torch(
                torch.zeros(1, 1, chunk_size, 1, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=None,
                mesh_mapper=rep,
            )
            return
        self._vis_buf = ttnn.from_torch(
            torch.zeros(1, chunk_size, H, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._vis_mask_buf = ttnn.from_torch(
            torch.zeros(1, chunk_size, 1, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._vis_zero_mask_host = ttnn.from_torch(
            torch.zeros(1, chunk_size, 1, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def _apply_vision_merge(self, x, length):
        """Final text-vs-vision SELECTION of the trace-safe splice: where(mask, vision, x).

        Not a scatter: host ``_set_vision_merge`` already placed the n vision rows at their image
        positions, so ``_vis_buf`` is a FULL [1, chunk_size, dim] buffer the same length as ``x``,
        not the raw [n, dim] vision tensor. The [.,1] mask is 1 only at those n positions, so
        out == vision there and out == x elsewhere (mask==0 is identity). Placement is on host
        because aligning variable n is variable-shape; any on-device scatter/pad/slice-copy would
        recompile per request and clobber the parked trace. ``length`` selects the segment
        (chunk_size vs bucket); no-op when buffers are unallocated (text-only / device-scatter path).
        """
        if self._vis_buf is None:
            return x
        if self._vis_buf.shape.rank == 4:
            # TP: buffers are [1, 1, chunk_size, dim(/TP)], matching the 4D TP activations.
            full = self._vis_buf.shape[2] == length
            mask = self._vis_mask_buf if full else self._vis_mask_buf[:, :, :length, :]
            vis = self._vis_buf if full else self._vis_buf[:, :, :length, :]
        else:
            full = self._vis_buf.shape[1] == length
            mask = self._vis_mask_buf if full else self._vis_mask_buf[:, :length, :]
            vis = self._vis_buf if full else self._vis_buf[:, :length, :]
        out = ttnn.where(mask, vis, x)
        ttnn.deallocate(x)
        return out

    def _set_vision_merge(self, ids_host, vision_tokens, vis_row_offset=0):
        """Stage the persistent vision buffers for the next forward (host -> device copy only; no
        program compiles). ``vision_tokens`` None clears the mask (where becomes identity); otherwise
        packed image rows are read back to host, placed at their token positions in a zero
        [1, chunk_size, dim] buffer, and uploaded with the 0/1 mask. ``ids_host`` locates image
        placeholders (== hf_config.image_token_id). ``vis_row_offset`` is the count of image
        placeholders before this segment, so a large image spanning chunks splices
        ``vis_host[vis_row_offset : vis_row_offset + n]``. A segment with no placeholders clears
        the mask (identity merge).
        """
        if self._vis_buf is None:
            # Fail loudly rather than silently drop the image: a multimodal request must run on a
            # path with the trace-safe buffers (capture_prefill_trace_chunked, single device) or
            # the on-device scatter (non-traced prefill_paged).
            assert vision_tokens is None, "vision merge requested but the trace-safe buffers are not allocated"
            return
        if vision_tokens is None:
            ttnn.copy_host_to_device_tensor(self._vis_zero_mask_host, self._vis_mask_buf)
            return
        tp = self.num_devices > 1
        cs = self._vis_buf.shape[-2]  # seq dim: dim 1 (3D single) / dim 2 (4D TP)
        Hg = self.args.dim  # global hidden (the buffer's last dim is dim/TP on a mesh)
        flat = ids_host.reshape(-1)
        pos = torch.nonzero(flat[:cs] == self._vision_placeholder_token_id(), as_tuple=False).reshape(-1)
        n = int(pos.numel())
        # No image placeholders in this segment (text-only chunk, or a tail that holds none of the
        # image rows): the merge is the identity, so just clear the mask.
        if n == 0:
            ttnn.copy_host_to_device_tensor(self._vis_zero_mask_host, self._vis_mask_buf)
            return
        assert vis_row_offset + n <= int(vision_tokens.shape[0]), (
            f"vision splice out of range: row offset {vis_row_offset} + {n} image positions in this "
            f"segment exceeds {int(vision_tokens.shape[0])} packed vision rows"
        )
        # Gather the (hidden-fractured on a mesh) vision rows to full [num_image_tokens, Hg] on
        # host, then take this segment's slice. The placement is along the SEQ dim, orthogonal to
        # the hidden fracture, so the round-trip gather->place->reshard preserves the per-device
        # columns. ConcatMeshToTensor(dim=1) over the 2D [rows, dim/TP] is the inverse of the
        # dims=(None,-1) hidden shard used on re-upload.
        if tp:
            vis_host = ttnn.to_torch(vision_tokens, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=1)).to(
                torch.bfloat16
            )
        else:
            vis_host = ttnn.to_torch(vision_tokens).to(torch.bfloat16)  # [num_image_tokens, Hg]
        seg = vis_host[vis_row_offset : vis_row_offset + n]
        if tp:
            shard = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -1), mesh_shape=self.args.cluster_shape)
            rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
            vis_full = torch.zeros(1, 1, cs, Hg, dtype=torch.bfloat16)
            vis_full[0, 0, pos] = seg
            mask = torch.zeros(1, 1, cs, 1, dtype=torch.bfloat16)
            mask[0, 0, pos, 0] = 1.0
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(vis_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=shard),
                self._vis_buf,
            )
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep),
                self._vis_mask_buf,
            )
            return
        vis_full = torch.zeros(1, cs, Hg, dtype=torch.bfloat16)
        vis_full[0, pos] = seg
        mask = torch.zeros(1, cs, 1, dtype=torch.bfloat16)
        mask[0, pos, 0] = 1.0
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(vis_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self._vis_buf
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self._vis_mask_buf
        )

    def _vis_row_offset_for(self, token_ids, chunk_start):
        """Packed-vision-row offset for the prefill segment starting at absolute position
        ``chunk_start``: the number of image-placeholder tokens before it. The vision rows are
        packed in image-placeholder order, so this is the index of the first row this segment
        owns — used to splice a large image whose placeholders span multiple chunks / the tail."""
        if chunk_start <= 0:
            return 0
        return int((token_ids[:, :chunk_start] == self._vision_placeholder_token_id()).sum())

    def switch_mode(self, mode):
        """Generator mode-change hook; no-op (no prefetcher)."""
        return None

    def _lm_head(self, x, out_dtype=None, gather=True):
        """LM-head matmul. Vocab-sharded mesh: partial logits + all-gather to full replicated.
        Single device: plain matmul. ``gather=False`` returns the per-device vocab shard [.., vocab/TP]
        (the verify's sharded argmax consumes it without the all-gather).

        out_dtype: leave None (default) for the base/verify path — losslessness is defined by the
        BASE argmax, so that call must stay byte-identical. The MTP drafter passes float32: its
        argmax only has to rank candidates, and at bf16 4.7-5.5% of draft rejections at 8k/32k were
        EXACT ties in the drafter's own logits, i.e. tokens thrown away to a rounding coin flip."""
        kw, ccl_kw = _lm_head_kwargs(), {}  # QWEN36_LMHEAD_CFG (opt-in) for the base path
        if out_dtype is not None:
            # HiFi2 matches the bfloat8_b weight; fp32 dest accumulation is what actually keeps the
            # extra bits. packer_l1_acc is off simply because that is the combination measured.
            kw = dict(
                dtype=out_dtype,
                compute_kernel_config=ttnn.init_device_compute_kernel_config(
                    self.device.arch(),
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                ),
            )
            # tt_all_gather defaults its CCL dtype to bfloat16 and TYPECASTS anything else on the
            # way in. That is load-bearing twice over: it would put the bf16 ties straight back, and
            # the fp32 -> bf16 cast on this [1,1,B,vocab/tp] tensor actually returns garbage (the
            # drafter's argmax came back with an out-of-range id). Gather at the head's own dtype.
            ccl_kw = dict(dtype=out_dtype)
        logits = ttnn.linear(x, self.lm_head_weight, **kw)
        if self._lmhead_vocab_sharded and gather:
            from models.tt_transformers.tt.ccl import tt_all_gather

            logits = tt_all_gather(
                logits,
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=None,
                dim=len(logits.shape) - 1,
                topology=self.args.ccl_topology(),
                **ccl_kw,
            )
        return logits

    # Multi-core max over a vocab shard: reduce dim=-1 parallelizes over tile-ROWS (R/32 cores), so a
    # tall/narrow grid (C=32 -> R~1952 -> ~61 cores) beats a square one (8 tile-rows -> 8 cores).
    _MAXVAL_C = 32

    def _shard_max_value(self, sharded_logits):
        """Per-device max over the last dim of vocab-sharded logits [1,1,1,vocab/TP] -> [1,1,1,1]."""
        per_shard = self.vocab_size // self.num_devices
        max_c = self._MAXVAL_C
        max_r = (((per_shard + max_c - 1) // max_c) + 31) // 32 * 32
        padded = ttnn.pad(sharded_logits, [(0, 0), (0, 0), (0, 0), (0, max_r * max_c - per_shard)], value=-1e30)
        grid = ttnn.reshape(padded, (1, 1, max_r, max_c))
        part = ttnn.max(grid, dim=-1)
        part_row = ttnn.reshape(part, (1, 1, 1, max_r))
        val = ttnn.max(part_row, dim=-1)
        ttnn.deallocate(padded)
        ttnn.deallocate(grid)
        ttnn.deallocate(part)
        ttnn.deallocate(part_row)
        return val

    def shard_argmax_max(self, sharded_logits):
        """Greedy first stage over the PRE-gather vocab-sharded logits [1,1,1,vocab/TP]: per-device local
        argmax + max value (the host picks the winning shard in read_shard_argmax_token). Skips the
        full-vocab all-gather and readback; shared by the traced decode loop (text_demo) and the traced
        short-prompt prefill. Trace-safe (device ops only)."""
        logits_rm = ttnn.to_layout(sharded_logits, ttnn.ROW_MAJOR_LAYOUT)  # untilize for the multi-core argmax path
        idx = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
        ttnn.deallocate(logits_rm)
        return idx, self._shard_max_value(sharded_logits)

    def read_shard_argmax_token(self, idx_t, val_t):
        """Host second stage of shard_argmax_max: winner device = argmax of the per-shard max values; the
        global token id = d * (vocab/TP) + local_idx[d]. One mesh-composed D2H per tensor (2 total) rather
        than one per shard; idx and val share the device order, so d maps to vocab shard d."""
        if self._argmax_read_comp is None:
            self._argmax_read_comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        per_shard = self.vocab_size // self.num_devices
        idxs = ttnn.to_torch(idx_t, mesh_composer=self._argmax_read_comp).reshape(-1)
        vals = ttnn.to_torch(val_t, mesh_composer=self._argmax_read_comp).reshape(-1)
        d = int(torch.argmax(vals).item())
        return d * per_shard + int(idxs[d].item())

    # ------------------------------------------------------------------------------------------------------- #
    # Device-resident single-user greedy decode loop (QWEN36_DECODE_DEVICE_LOOP; driven by demo/text_demo.py)
    # ------------------------------------------------------------------------------------------------------- #
    # The host loop pays, per token: build token / position / RoPE cos+sin on the host, 3 H2D copies, execute_trace,
    # synchronize, 2 blocking D2H reads and a host combine of the per-shard argmax (~0.5 ms of a ~31 ms step). Here ONE
    # captured trace performs the whole step AND feeds itself, so the host only enqueues traces back to back
    # (non-blocking) and reads the generated tokens once at the end -- the design of models/demos/qwen38_27b_qb2
    # (generator._model_step / _sampling_step / _append_history):
    #   * RoPE: cos/sin rows come from a replicated bf16 ROW_MAJOR [max_seq_len, rope_dim] table via
    #     ttnn.embedding(rope_idx, table) (rope_idx = KV position + rope_delta) -- the same [1,1,1,rope_dim] TILE
    #     tensors the host packed before (the table rows are the host values, bf16-rounded the same way);
    #   * token feedback: the global greedy token is picked on device from the per-shard (argmax, max) pair and copied
    #     IN PLACE into the token buffer the next step's embedding reads;
    #   * position and RoPE index advance with ttnn.plus_one at the end of the trace; the token is appended to a device
    #     history buffer at a device-side cursor (ttnn.indexed_fill + plus_one) and read back once after the loop.
    # Greedy only (no sampling / penalties / teacher forcing): those need the host in the loop.
    #
    # Why per-shard argmax + a tiny gather and NOT the on-device sampler (model.sampling, force-argmax): the sampler
    # gathers the FULL logits (pads B=1 to 32 rows -> a ~4 MB/device all-gather, then untilizes and arg-maxes the
    # [32, vocab] row: roughly 0.3-0.5 ms of device time, which would cancel the 0.5 ms saved; the batched demo
    # measured its "sample" mode slower than "shard"), while the per-shard argmax + max (shard_argmax_max) is already
    # in the trace. Here its two scalars per device are widened to one tile row each, all-gathered
    # ([1,1,1,32] -> [1,1,1,128], one tile per device) and reduced with the tie-break recipe of models/common/sampling
    # (_adjust_values_for_tiebreak): lowest GLOBAL index among the tied maxima, int32 min (exact; fp32/uint32 reduces
    # go through the FPU and round indices above 2**11) -- identical to torch.argmax on the full logits and to
    # read_shard_argmax_token.
    _DL_LANES = 32  # tile width: one candidate row per device, the same candidate replicated over all 32 lanes
    # Added to the index of every non-maximum candidate: a power of two (exact in bf16) above any vocab id.
    _DL_TIE_SENTINEL = 1 << 24

    def device_decode_loop_unsupported_reason(self, first_position, num_steps, batch=1):
        """None when the device-resident greedy decode loop can run, else a short human-readable reason.
        ``batch`` = number of users decoded together (1 = the single-user loop; 2..32 = the batched loop, every user
        at ``first_position``)."""
        if not 1 <= int(batch) <= self._DL_LANES:
            return f"batch {batch} is outside 1..{self._DL_LANES}"
        if self.num_devices < 2 or self.tt_ccl is None:
            return "single-device mesh"
        if not self._lmhead_vocab_sharded:
            return "LM head is not vocab-sharded"
        shape = tuple(int(dim) for dim in self.mesh_device.shape)
        if len(shape) != 2 or shape[0] != 1:
            return f"mesh {shape} is not (1, N)"
        if self.vocab_size >= self._DL_TIE_SENTINEL:
            return f"vocab {self.vocab_size} exceeds the tie-break sentinel {self._DL_TIE_SENTINEL}"
        first_rope = int(first_position) + int(self.rope.rope_delta)
        if first_rope < 0 or first_rope + int(num_steps) > self.rope.max_seq_len:
            return (
                f"RoPE index range [{first_rope}, {first_rope + int(num_steps)}) is outside the "
                f"{self.rope.max_seq_len}-row decode table"
            )
        return None

    def setup_device_decode_loop(self, num_steps, batch=1):
        """Allocate the persistent buffers of the device-resident decode loop. MUST run before the decode trace is
        captured (their addresses are baked into it). ``num_steps`` = number of decode steps the history must hold,
        ``batch`` = number of users (lanes 0..batch-1 of tokbuf / every history row; positions [batch]).
        Returns a _DeviceDecodeLoop; call device_decode_loop_arm to load the initial state."""
        batch = int(batch)
        assert 1 <= batch <= self._DL_LANES, f"batch {batch} outside 1..{self._DL_LANES}"
        mesh = self.mesh_device
        rep = ttnn.ReplicateTensorToMesh(mesh)
        lanes = self._DL_LANES
        n = self.num_devices
        per_shard = self.vocab_size // n
        # Replicated RoPE lookup tables (once per model; allocated here, outside any trace).
        self.rope.decode_rm_tables()

        def up(t, dtype, layout, mapper=rep):
            return ttnn.from_torch(
                t.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        i32 = torch.int32
        RM, TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
        return _DeviceDecodeLoop(
            num_steps=int(num_steps),
            batch=batch,
            # Loop state, advanced in-trace. tokbuf: lanes 0..batch-1 = the token each user's next step embeds (B's
            # [1,1,1,32] layout, so the same buffer is also the row appended to the history).
            tokbuf=up(torch.zeros(1, 1, 1, lanes, dtype=i32), ttnn.uint32, RM),
            # KV position per user (paged SDPA / cache-update cur_pos)
            pos=up(torch.zeros(batch, dtype=i32), ttnn.int32, RM),
            rope_idx=up(torch.zeros(1, batch, dtype=i32), ttnn.uint32, RM),  # row of the RoPE tables, per user
            cursor=up(torch.zeros(1, dtype=i32), ttnn.uint32, RM),  # next history row
            history=up(torch.zeros(int(num_steps) + 1, 1, 1, lanes, dtype=i32), ttnn.uint32, RM),
            # Constants of the token combine.
            zeros_bf16=up(torch.zeros(1, 1, 1, lanes, dtype=torch.bfloat16), ttnn.bfloat16, TILE),
            zeros_u32=up(torch.zeros(1, 1, 1, lanes, dtype=i32), ttnn.uint32, TILE),
            # Per-device vocab offset d * (vocab/TP), one [1,1,1,32] row per device (sharded: device d holds d's row).
            shard_base=up(
                (torch.arange(n, dtype=i32) * per_shard).reshape(n, 1, 1, 1).expand(n, 1, 1, lanes),
                ttnn.uint32,
                RM,
                mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            ),
        )

    def device_decode_loop_arm(self, dl, token, position):
        """(Re)load the loop's input state into its persistent buffers: the token to embed next, the KV position, the
        RoPE index (= position + rope_delta) and a zeroed history cursor. Async H2D copies on CQ 0, ordered before the
        next execute_trace; the host tensors are kept alive in dl.host_refs until the caller has drained the device.
        Single user: ``token`` / ``position`` are ints. Batched (dl.batch > 1): sequences of ``dl.batch`` ints (or an
        int position shared by all users)."""
        rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
        nb = dl.batch
        if nb > 1:
            tokens = [int(t) for t in token]
            positions = [int(p) for p in position] if hasattr(position, "__len__") else [int(position)] * nb
            assert len(tokens) == nb and len(positions) == nb, (len(tokens), len(positions), nb)
            tok = torch.zeros(1, 1, 1, self._DL_LANES, dtype=torch.int32)
            tok[0, 0, 0, :nb] = torch.tensor(tokens, dtype=torch.int32)
            pos_t = torch.tensor(positions, dtype=torch.int32)
            rope_t = (pos_t + int(self.rope.rope_delta)).reshape(1, nb)

            def host_b(t, dtype):
                return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep)

            staged = [
                (host_b(tok, ttnn.uint32), dl.tokbuf),
                (host_b(pos_t, ttnn.int32), dl.pos),
                (host_b(rope_t, ttnn.uint32), dl.rope_idx),
                (host_b(torch.zeros(1, dtype=torch.int32), ttnn.uint32), dl.cursor),
            ]
            for host_t, dev_t in staged:
                ttnn.copy_host_to_device_tensor(host_t, dev_t)
            dl.host_refs = [host_t for host_t, _ in staged]
            return
        tok = torch.zeros(1, 1, 1, self._DL_LANES, dtype=torch.int32)
        tok[0, 0, 0, 0] = int(token)

        def host(t, dtype):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep)

        position = int(position)
        staged = [
            (host(tok, ttnn.uint32), dl.tokbuf),
            (host(torch.tensor([position], dtype=torch.int32), ttnn.int32), dl.pos),
            (host(torch.tensor([[position + int(self.rope.rope_delta)]], dtype=torch.int32), ttnn.uint32), dl.rope_idx),
            (host(torch.zeros(1, dtype=torch.int32), ttnn.uint32), dl.cursor),
        ]
        for host_t, dev_t in staged:
            ttnn.copy_host_to_device_tensor(host_t, dev_t)
        dl.host_refs = [host_t for host_t, _ in staged]

    def _rope_wide_rows(self, rope_idx):
        """QWEN36_ROPE_PERM: (cos, sin) [1, B, 1, head_dim] TILE bf16 (one tile row per user, the layout DecodeRope shards)
        looked up on device from the permuted hd-wide tables: the same rows as the 64-wide lookup, widened."""
        B, hd = rope_idx.shape[-1], self.args.head_dim
        out = []
        for table in self.rope.decode_rm_tables_wide(hd):
            row = ttnn.embedding(rope_idx, table, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            row = ttnn.reshape(row, (1, B, 1, hd))  # RM: a view
            out.append(ttnn.to_layout(row, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        return out[0], out[1]

    def _rot_mats_decode_base(self, positions):
        """rope_tp.rot_mats_decode for the BASE model's decode: rd-wide [1,B,1,rd], or (QWEN36_ROPE_PERM) the permuted
        hd-wide [1,B,1,hd] DecodeRope input. The MTP drafter calls rot_mats_decode directly (64-wide)."""
        from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode, rot_mats_decode_wide

        if self._rope_perm:
            return rot_mats_decode_wide(
                self.device, self.args.rope_head_dim, self.args.head_dim, self.args.rope_theta, positions
            )
        return rot_mats_decode(
            self.device, self.args.rope_head_dim, self.args.max_seq_len, self.args.rope_theta, positions
        )

    def _rope_decode_arg(self, cos, sin):
        """What the layer loop of a BASE decode / verify / seed body passes as cos / sin. QWEN36_ROPE_PERM: one DecodeRope
        (hd-wide cos/sin [1,B,1,hd] height-sharded once for all full-attention layers); caller frees it after the loop
        (rope_tp.DecodeRope.free). Off: the tensors unchanged and None."""
        if not self._rope_perm:
            return cos, sin, None
        from models.demos.blackhole.qwen36.tt.attention.rope_tp import DecodeRope

        dr = DecodeRope(self.mesh_device, cos, sin, self.args.head_dim)
        return dr, dr, dr

    def _dl_rope_cos_sin(self, rope_idx, batch=1, wide=False):
        """cos/sin of the RoPE rows `rope_idx` ([1,batch] uint32), looked up on device. batch == 1: [1,1,1,rope_dim]
        TILE bf16. batch > 1: [1,1,batch,rope_dim] TILE bf16 (row u = user u): the layout apply_partial_rope_decode
        reshapes the host-packed [1,batch,1,rope_dim] cos/sin to before use (same values, same final tensor; the
        lookup output is already in it, so that reshape becomes a no-op and the per-user tile re-layout is skipped).
        wide (QWEN36_ROPE_PERM; the BASE model's callers pass self._rope_perm): the permuted [1,batch,1,head_dim] rows
        instead (_rope_wide_rows). The MTP drafter's callers keep the default (64-wide)."""
        if wide:
            return self._rope_wide_rows(rope_idx)
        rd = self.args.rope_head_dim
        out = []
        for table in self.rope.decode_rm_tables():
            row = ttnn.embedding(rope_idx, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # [1,batch,rd] TILE -> [1,1,batch,rd] / [1,1,rd] -> [1,1,1,rd]: a pure view
            out.append(ttnn.reshape(row, (1, 1, batch, rd)))
        return out[0], out[1]

    def _shard_max_value_rows(self, sharded_logits, rows):
        """Per-device, per-row max of vocab-sharded logits [1,1,rows,vocab/TP] -> [1,1,rows] (the batched
        generalization of _shard_max_value; same tile-parallel reduce as the batched demo's _maxval_dev_b)."""
        per_shard = self.vocab_size // self.num_devices
        max_c = self._MAXVAL_C
        max_r = (((per_shard + max_c - 1) // max_c) + 31) // 32 * 32
        padded = ttnn.pad(sharded_logits, [(0, 0), (0, 0), (0, 0), (0, max_r * max_c - per_shard)], value=-1e30)
        grid = ttnn.reshape(padded, (1, rows, max_r, max_c))
        part = ttnn.max(grid, dim=-1)
        part_row = ttnn.reshape(part, (1, 1, rows, max_r))
        val = ttnn.max(part_row, dim=-1)
        ttnn.deallocate(padded)
        ttnn.deallocate(grid)
        ttnn.deallocate(part)
        ttnn.deallocate(part_row)
        return val

    def shard_argmax_max_rows(self, sharded_logits, rows):
        """Batched shard_argmax_max: per-device, per-user (row) local argmax idx [1,1,rows] uint32 RM + max
        [1,1,rows] bf16 TILE of the pre-gather logits [1,1,rows,vocab/TP]. Trace-safe."""
        if _tpc.argmax_topk_pick(sharded_logits, rows):  # QWEN36_ARGMAX_TOPK_MIN_ROWS: one stable topk, column 0
            i0, v0 = _tpc.local_topk_argmax(sharded_logits, rows)
            i32 = ttnn.typecast(i0, ttnn.uint32)  # uint16 -> uint32 (what the callers get from argmax)
            idx = ttnn.reshape(ttnn.to_layout(i32, ttnn.ROW_MAJOR_LAYOUT), (1, 1, rows))
            val = ttnn.reshape(v0, (1, 1, rows))
            return idx, val
        logits_rm = ttnn.to_layout(sharded_logits, ttnn.ROW_MAJOR_LAYOUT)
        idx = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
        ttnn.deallocate(logits_rm)
        return idx, self._shard_max_value_rows(sharded_logits, rows)

    def _dl_greedy_token_row(self, idx, val, dl):
        """Global greedy token from the per-shard (argmax idx [1,1,1] uint32 RM, max val [1,1,1] bf16 TILE), on device.
        Returns the token replicated over a [1,1,1,32] uint32 ROW_MAJOR row (the layout of tokbuf / a history row)."""
        # Per-device candidate (max value, GLOBAL vocab index = local idx + device * vocab/TP), each widened to one
        # tile-aligned row: the gather moves [1,1,1,32] tensors (one tile per device).
        val_row = ttnn.add(ttnn.reshape(val, (1, 1, 1, 1)), dl.zeros_bf16)
        idx_row = ttnn.add(ttnn.reshape(idx, (1, 1, 1, 1)), dl.shard_base)  # RM uint32 scalar-broadcast add (exact)
        idx_row = ttnn.to_layout(idx_row, ttnn.TILE_LAYOUT)
        gathered_val = ttnn.all_gather(val_row, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,1,32*TP] bf16
        gathered_idx = ttnn.all_gather(idx_row, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,1,32*TP] uint32
        # Lowest global index among the tied maxima: models/common/sampling TTSampling._adjust_values_for_tiebreak.
        # Value domain bf16 (exact), index domain int32 (the reduce MUST be int32: FPU reduces round ids above 2**11).
        idx_i32 = ttnn.typecast(gathered_idx, ttnn.int32)
        max_val = ttnn.max(gathered_val, dim=3, keepdim=True)
        not_max = ttnn.lt(gathered_val, max_val)  # 1.0 strictly below the max, else 0.0
        offset = ttnn.typecast(ttnn.multiply(not_max, float(self._DL_TIE_SENTINEL)), ttnn.int32)
        token = ttnn.min(ttnn.add(idx_i32, offset), dim=3, keepdim=True)  # [1,1,1,1] int32
        token = ttnn.add(ttnn.typecast(token, ttnn.uint32), dl.zeros_u32)  # widen to [1,1,1,32] uint32 TILE
        return ttnn.to_layout(token, ttnn.ROW_MAJOR_LAYOUT)

    def _dl_greedy_token_row_batched(self, idx, val, dl):
        """Batched _dl_greedy_token_row: global greedy token of each of the dl.batch users from the per-shard
        (argmax idx [1,1,B] uint32 RM, max val [1,1,B] bf16 TILE), on device. Returns the tokens in lanes 0..B-1 of a
        [1,1,1,32] uint32 ROW_MAJOR row (the layout of tokbuf / a history row; lanes >= B hold 0).

        Each device widens its B candidates to one 32-lane tile row (user u on lane u; pad lanes -1e30 / index 0),
        the two rows are all-gathered along the lane axis ([1,1,1,32] -> [1,1,1,32*TP], device d's block at lanes
        d*32 .. d*32+31) and every user's TP candidates are combined with the same tie-break as the single-user
        recipe: max value, then the LOWEST global index among tied maxima (non-max candidates get + 2**24, int32 min)
        == torch.argmax on the full logits. TP is small, so the cross-device reduce is a chain of elementwise
        max / min over the TP lane blocks (slices at multiples of 32 lanes)."""
        lanes = self._DL_LANES
        B = dl.batch
        n = self.num_devices
        val_row = ttnn.reshape(val, (1, 1, 1, B))
        idx_row = ttnn.reshape(idx, (1, 1, 1, B))
        if B < lanes:
            val_row = ttnn.pad(val_row, [(0, 0), (0, 0), (0, 0), (0, lanes - B)], value=-1e30)
            idx_row = ttnn.pad(idx_row, [(0, 0), (0, 0), (0, 0), (0, lanes - B)], value=0)
        idx_row = ttnn.add(idx_row, dl.shard_base)  # RM uint32, + device * vocab/TP (exact)
        idx_row = ttnn.to_layout(idx_row, ttnn.TILE_LAYOUT)
        gathered_val = ttnn.all_gather(val_row, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,1,32*TP] bf16
        gathered_idx = ttnn.all_gather(idx_row, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,1,32*TP] uint32

        def block(t, d):
            return ttnn.slice(t, [0, 0, 0, d * lanes], [1, 1, 1, (d + 1) * lanes])

        # Per-user global max over the TP blocks (bf16: exact), replicated back over the TP blocks.
        max_val = block(gathered_val, 0)
        for d in range(1, n):
            max_val = ttnn.maximum(max_val, block(gathered_val, d))
        max_full = ttnn.concat([max_val] * n, dim=3)
        not_max = ttnn.lt(gathered_val, max_full)  # 1.0 strictly below the user's max, else 0.0
        offset = ttnn.typecast(ttnn.multiply(not_max, float(self._DL_TIE_SENTINEL)), ttnn.int32)
        cand = ttnn.add(ttnn.typecast(gathered_idx, ttnn.int32), offset)  # int32: exact
        token = block(cand, 0)
        for d in range(1, n):
            token = ttnn.minimum(token, block(cand, d))  # [1,1,1,32] int32, lane u = user u's token
        token = ttnn.typecast(token, ttnn.uint32)
        return ttnn.to_layout(token, ttnn.ROW_MAJOR_LAYOUT)

    def device_decode_step(self, dl, page_table):
        """One decode step of the device-resident loop: everything the host loop did around the trace, in-trace.
        Eager and trace-capturable (device ops only, no allocation that outlives the step: every persistent buffer is in
        `dl`, allocated by setup_device_decode_loop). `page_table`: the persistent device page table (dev[3])."""
        lanes = self._DL_LANES
        nb = dl.batch
        # Token of this step = lane 0 (lanes 0..batch-1 when batched) of the token buffer (B's idiom: reshape [1,32],
        # slice [1,B]). The batched embedding of [1,B] ids == the [B,1] ids of the host loop: same rows, same order.
        ids = ttnn.slice(ttnn.reshape(dl.tokbuf, (1, lanes)), [0, 0], [1, nb], [1, 1])
        cos, sin = self._dl_rope_cos_sin(dl.rope_idx, nb, wide=self._rope_perm)
        logits = self._forward_decode(ids, cos, sin, dl.pos, page_table, sharded_lm_head=True)
        if nb == 1:
            idx, val = self.shard_argmax_max(logits)
            ttnn.deallocate(logits)
            token_row = self._dl_greedy_token_row(idx, val, dl)
        else:
            idx, val = self.shard_argmax_max_rows(logits, nb)
            ttnn.deallocate(logits)
            token_row = self._dl_greedy_token_row_batched(idx, val, dl)
        # Token feedback (in place) + history append (B's _append_history).
        ttnn.copy(token_row, dl.tokbuf)
        updated = ttnn.indexed_fill(dl.cursor, dl.history, token_row, dim=0)
        ttnn.copy(updated, dl.history)
        ttnn.deallocate(updated)
        ttnn.plus_one(dl.cursor)
        # Advance the loop state for the next step (after every reader of pos / rope_idx above).
        ttnn.plus_one(dl.pos)
        ttnn.plus_one(dl.rope_idx)

    def read_device_decode_history(self, dl, num_tokens):
        """The first `num_tokens` tokens the loop appended (one blocking D2H of the history buffer): a list of ints
        (single user) or, for a batched loop (dl.batch > 1), a list of dl.batch per-user token lists."""
        history = ttnn.to_torch(ttnn.get_device_tensors(dl.history)[0])  # [num_steps + 1, 1, 1, lanes]
        if dl.batch > 1:
            return history[:num_tokens, 0, 0, : dl.batch].to(torch.int64).t().tolist()
        return history[:num_tokens, 0, 0, 0].to(torch.int64).tolist()

    def _final_norm_decode(self, x):
        """Final RMSNorm before the LM head (TP decode).

        The bare `self.norm(x, DECODE)` runs plain ttnn.rms_norm on a DRAM-interleaved [32,dim]
        tensor -> single tile-row -> 1 core (~80us/token). Passing the framework's 'lm_head' norm
        config runs the sharded multi-core norm across lm_head_core_grid instead; output_mem_config
        is forced back to DRAM so the LM-head matmul input is byte-identical (layout-only change).

        QWEN36_DECODE_ALLREDUCE=1: x is the replicated RES40 residual (last dim == dim) -> local sharded norm, DRAM out.
        """
        if self._decode_ar is not None and x.shape[-1] == self.args.dim:
            return self._decode_ar.final_norm(x, self.norm.norm)
        if self.num_devices > 1:
            nc = dict(self.args.get_norm_config("lm_head", Mode.DECODE))
            nc["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
            return self.norm(x, mode=Mode.DECODE, norm_config=nc)
        return self.norm(x, mode=Mode.DECODE)

    @classmethod
    def from_pretrained(
        cls,
        device,
        max_batch_size=1,
        max_seq_len=2048,
        n_layers=None,
        layer_indices=None,
        hf_model=None,
        enable_mtp=None,
        state_dict=None,
    ):
        # HF_MODEL env var (hub or local path) is canonical; hf_model sets it for back-compat.
        if hf_model is not None:
            import os

            os.environ["HF_MODEL"] = hf_model

        # enable_mtp: None -> QWEN36_MTP env if set, else on for dense / off for MoE checkpoints;
        # False -> no MTP head/weights/KV.
        args = Qwen36ModelArgs(
            mesh_device=device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            enable_mtp=enable_mtp,
        )

        # layer_indices: run only these checkpoint layers (e.g. [0,3,31]) for profiling.
        # Each keeps its real type via full attention_type_list; n_layers below takes the first N layers instead.
        if layer_indices is not None:
            layer_indices = list(layer_indices)
            assert layer_indices, "layer_indices must be non-empty"
            assert all(
                0 <= i < len(args.attention_type_list) for i in layer_indices
            ), f"layer_indices {layer_indices} out of range [0, {len(args.attention_type_list)})"
            args.layer_indices = layer_indices
            args.n_layers = len(layer_indices)
        elif n_layers is not None:
            # Keep attention_type_list whole: n_layers alone decides how many layers get built
            # (list(range(args.n_layers))); each kept index's type is unchanged either way, and MTP
            # needs a real full_attention index from this list -- truncating below 4 layers drops it.
            args.n_layers = n_layers

        # NOTE: the warm-ttnn-cache HF-load skip is DISABLED for qwen3.6.
        # Its Gated-DeltaNet loader consumes conv weights on the host without a cache_file_name --
        # gdn/weights.py::load_conv_weight does ttnn.from_torch(state_dict[name], ...) for q/k/v_conv in
        # every DeltaNet layer, and gdn/tp.py derives taps the same way -- so a dataless placeholder
        # feeds those layers garbage while the HF load is skipped. This is the same failure that made
        # the vision demo emit token soup, on the text path. Re-enabling needs those conv weights either
        # cache-backed or captured to the sidecar via an is_host_weight predicate. (#45400 review)
        cache_path = args.weight_cache_path()
        if state_dict is None:
            logger.info("Loading + remapping weights via Qwen36ModelArgs.load_state_dict()...")
            state_dict = args.load_state_dict()
        else:
            logger.info("Using preloaded state dict")

        model = cls(device, args, state_dict, tensor_cache_path=cache_path)
        return model

    def prefill_tp(self, token_ids, valid_len=None, vision_tokens=None):
        """Tensor-parallel full-model prefill (num_devices>1). Stateless: runs the
        whole sequence from scratch through the fractured-residual TP layers and
        returns the next-token logits at position valid_len-1.

        token_ids: torch [1, T] (pad T to a multiple of 128 for the GDN chunk
        kernel; right-padding does not affect the causal logit at valid_len-1).
        Returns ttnn logits [1, 1, 1, vocab_size] (host).
        """
        B, T = token_ids.shape
        assert B == 1, "prefill_tp is single-sequence"
        valid_len = valid_len or T

        # Stage the per-request RoPE (M-RoPE for multimodal, 1D for text), then build cos/sin from
        # that staged sequence table — same source as the traced TP path (_rope_tp_cos_sin_torch).
        self._build_request_rope(token_ids[:, :valid_len], vision_tokens)
        tok = ttnn.from_torch(
            token_ids.to(torch.int32),
            dtype=ttnn.uint32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x = self.embd(tok)  # [1, T, dim_frac] (hidden dim sharded across mesh)
        x = self._scatter_vision_tokens(x, token_ids, vision_tokens)
        x = ttnn.reshape(x, (1, 1, T, x.shape[-1]))
        cos_t, sin_t = self._rope_tp_cos_sin_torch(0, T)
        rep = ttnn.ReplicateTensorToMesh(self.device)
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)

        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="prefill", chunk_size=128, valid_len=valid_len)

        # Last real position via one-hot matmul (not slice): bare slice breaks at long T (~49k+).
        sel = torch.zeros(1, 1, 1, T, dtype=torch.float32)
        sel[0, 0, 0, valid_len - 1] = 1.0
        sel_tt = ttnn.from_torch(
            sel,
            dtype=x.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x_last = ttnn.matmul(sel_tt, x)  # [1,1,1,dim_frac]
        ttnn.deallocate(sel_tt)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = self.norm(x_last, mode=Mode.PREFILL)  # DistributedNorm on selected row
        logits = self._lm_head(x_last)
        # Replicated logits; read one replica -> torch [vocab_size].
        lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
        return lt[0].reshape(-1)[: self.vocab_size]

    def reset_tp(self):
        """Reset TP layer KV cache / GDN state for a new sequence."""
        for layer in self.layers:
            layer.attention.reset_state()

    def decode_tp(self, token_id, pos):
        """Single-token TP decode at position `pos` (B=1). Uses KV + GDN from prefill/decode."""

        tok = ttnn.from_torch(
            torch.tensor([[int(token_id)]], dtype=torch.int32),
            dtype=ttnn.uint32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x = self.embd(tok)  # [1,1,dim_frac]
        x = ttnn.reshape(x, (1, 1, 1, x.shape[-1]))  # [1,1,B=1,dim_frac]
        if self._decode_ar is not None:
            x = self._decode_ar.enter(x)  # QWEN36_DECODE_ALLREDUCE=1: one AG -> replicated RES40 residual
        # RoPE position offset by rope_delta for multimodal (KV position cur_pos_tt stays `pos`).
        cos, sin = self._rot_mats_decode_base(torch.tensor([pos + self.rope.rope_delta], dtype=torch.int32))
        cur_pos_tt = ttnn.from_torch(
            torch.tensor([pos], dtype=torch.int32),
            dtype=ttnn.int32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        cos, sin, _dr = self._rope_decode_arg(cos, sin)
        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="decode", position_tensor=cur_pos_tt)
        if _dr is not None:
            _dr.free()
        x = self._final_norm_decode(x)
        logits = self._lm_head(x)
        lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
        return lt[0].reshape(-1)[: self.vocab_size]

    def generate_tp(self, prompt_ids, max_new_tokens=20):
        """TP greedy generation: prefill prompt, then decode. Returns new token ids."""
        import math as _math

        self.reset_tp()
        T = len(prompt_ids)
        T_pad = max(128, _math.ceil(T / 128) * 128)
        padded = prompt_ids + [0] * (T_pad - T)
        logits = self.prefill_tp(torch.tensor([padded], dtype=torch.long), valid_len=T)
        nxt = int(torch.argmax(logits).item())
        out = [nxt]
        for pos in range(T, T + max_new_tokens - 1):
            logits = self.decode_tp(nxt, pos)
            nxt = int(torch.argmax(logits).item())
            out.append(nxt)
        return out

    def _scatter_vision_tokens(self, x, token_ids, vision_tokens):
        """Splice vision-model embeddings into the text token embeddings, on device.

        On-device HF ``masked_scatter``: flatten ``x`` to ``[rows, H]``, ``ttnn.scatter`` packed
        ``vision_tokens`` into a zero buffer at image-placeholder rows, then ``ttnn.where(mask, vision, text)``.
        Embeddings/vision never leave the device. No-op if ``vision_tokens`` is None or the prompt has no
        image tokens. Mask/placement come from host token ids (``input_ids == image_token_id``); uploaded
        are the ``[n, H]`` scatter index (hidden-sharded like activations) and the ``[rows, 1]``
        where-predicate (replicated; broadcasts over sharded hidden). ``x`` is ``[B,T,H]`` from
        ``self.embd``; ``vision_tokens`` is ``[num_image_tokens, H]``. Returns same shape/layout/sharding as ``x``.
        """
        if vision_tokens is None:
            return x

        orig_shape = tuple(x.shape)
        hidden = orig_shape[-1]
        rows = 1
        for d in orig_shape[:-1]:
            rows *= d

        # special_image_mask = input_ids == image_token_id, computed on host from the
        # token ids and uploaded. torch.nonzero gives the placement positions directly.
        flat_ids = token_ids.reshape(-1)
        mask_bool = flat_ids == self._vision_placeholder_token_id()
        pos = torch.nonzero(mask_bool, as_tuple=False).reshape(-1)
        n = int(pos.numel())
        if n == 0:
            return x
        assert n == int(
            vision_tokens.shape[0]
        ), f"input_ids has {n} image-token positions but vision_tokens has {int(vision_tokens.shape[0])} rows"

        # Placement index: the dim-0 rows of the flattened [rows, H] embedding to fill,
        # repeated across the hidden dim so the whole hidden vector at each row is written.
        # ttnn.scatter mirrors torch.scatter: out[index[i, h], h] = src[i, h], with
        # index/src/input the same rank.
        index = pos.view(n, 1).expand(n, hidden).contiguous().to(torch.int32)
        # where-predicate: [rows, 1], broadcasts over hidden in ttnn.where.
        mask_col = mask_bool.view(rows, 1)

        if self.num_devices > 1:
            # Shard the index along hidden the same way the embedding shards its
            # activations, so each device's [n, H/TP] index matches its local x/vision
            # shard (the hidden columns are identical, so splitting is free). The predicate
            # broadcasts over the sharded hidden dim, so it is replicated.
            index_tt = ttnn.from_torch(
                index,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(None, 1), mesh_shape=self.args.cluster_shape
                ),
            )
            mask_tt = ttnn.from_torch(
                mask_col,
                dtype=x.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        else:
            index_tt = ttnn.from_torch(index, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
            mask_tt = ttnn.from_torch(mask_col, dtype=x.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)

        # ttnn.scatter requires input.dtype == src.dtype.
        src = vision_tokens if vision_tokens.dtype == x.dtype else ttnn.typecast(vision_tokens, x.dtype)

        # Place the packed vision rows into a zero buffer, then select per row:
        # vision at image positions, original text embedding everywhere else.
        x_2d = ttnn.reshape(x, (rows, hidden))
        vision_placed = ttnn.scatter(ttnn.zeros_like(x_2d), 0, index_tt, src)
        ttnn.deallocate(index_tt)
        out = ttnn.where(mask_tt, vision_placed, x_2d)
        ttnn.deallocate(vision_placed)
        ttnn.deallocate(mask_tt)
        return ttnn.reshape(out, orig_shape)

    def prefill(self, token_ids, vision_tokens=None):
        B, T = token_ids.shape

        # Stage the per-request RoPE (M-RoPE for multimodal, 1D for text) before any cos/sin seam.
        self._build_request_rope(token_ids, vision_tokens)

        if T > 1024:
            return self.prefill_layer_chunked(token_ids, chunk_size=2048, vision_tokens=vision_tokens)

        # Short sequences (<=1024)
        self.reset_state()

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = self.embd(token_ids_ttnn)
        x = self._scatter_vision_tokens(x, token_ids, vision_tokens)

        cos, sin = self.rope.get_prefill_rot_mats(0, T)

        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="prefill")

        x = self.norm(x, mode=Mode.PREFILL)

        x_last = x[:, -1:, :]
        logits = self._lm_head(x_last)

        return logits

    def prefill_layer_chunked(self, token_ids, chunk_size=2048, page_table=None, vision_tokens=None):
        """Prefill long sequences using layer-at-a-time chunked processing.

        DeltaNet uses larger chunk_size (256 vs 64) to limit Neumann-series error
        (4096 tokens -> 16 sub-chunks, PCC >0.98). page_table enables paged prefill."""
        B, T = token_ids.shape
        self.reset_state()

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = self.embd(token_ids_ttnn)
        x = self._scatter_vision_tokens(x, token_ids, vision_tokens)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(token_ids_ttnn)

        # Attn layers: chunk_size>=4096 (no Neumann limit; fewer SDPA compilations).
        attn_chunk_size = max(chunk_size, 4096)

        page_table_tt = None
        if page_table is not None:
            page_table_tt = ttnn.from_torch(
                page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )

        for layer_idx, layer in enumerate(self.layers):
            layer_chunk_size = attn_chunk_size if layer.is_full_attention else chunk_size

            chunks_out = []
            for chunk_start in range(0, T, layer_chunk_size):
                chunk_end = min(chunk_start + layer_chunk_size, T)

                x_chunk = x[:, chunk_start:chunk_end, :]
                x_chunk = ttnn.to_layout(x_chunk, ttnn.TILE_LAYOUT)

                if layer.is_full_attention and page_table is not None:
                    # Paged prefill path. M-RoPE-aware cos/sin for sequence positions of this chunk
                    # (slices the staged per-request table for multimodal; 1D RoPE otherwise).
                    cos, sin = self.rope.get_prefill_rot_mats(chunk_start, chunk_end - chunk_start)

                    block_size = 64
                    chunk_blocks_end = math.ceil(chunk_end / block_size)
                    chunk_page_table = page_table[:, chunk_start // block_size : chunk_blocks_end]
                    chunk_page_table_tt = ttnn.from_torch(
                        chunk_page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                    )

                    x_chunk = layer.forward(
                        x_chunk,
                        cos=cos,
                        sin=sin,
                        mode="prefill",
                        page_table=page_table_tt,
                        chunk_page_table=chunk_page_table_tt,
                        chunk_start_idx=chunk_start,
                    )

                elif layer.is_full_attention:
                    # Original concat path (non-paged prefill). M-RoPE-aware cos/sin (per-request
                    # table slice for multimodal; 1D RoPE otherwise).
                    cos, sin = self.rope.get_prefill_rot_mats(chunk_start, chunk_end - chunk_start)
                    x_chunk = layer.forward(x_chunk, cos=cos, sin=sin, mode="prefill")
                else:
                    x_chunk = layer.forward(
                        x_chunk,
                        cos=None,
                        sin=None,
                        mode="prefill",
                        chunk_size=layer.attention.long_prefill_chunk_size,
                    )

                chunks_out.append(x_chunk)

            # Last layer: save last token from last chunk before concat (avoids L1 clash on long T).
            is_last_layer = layer_idx == len(self.layers) - 1
            if is_last_layer:
                x_last = chunks_out[-1][:, -1:, :]
                x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)

            if len(chunks_out) == 1:
                x_new = chunks_out[0]
            else:
                x_new = ttnn.concat(chunks_out, dim=1)
                for c in chunks_out:
                    ttnn.deallocate(c)
            x_new = ttnn.to_memory_config(x_new, ttnn.DRAM_MEMORY_CONFIG)

            ttnn.deallocate(x)
            x = x_new

        x_last = self.norm(x_last, mode=Mode.PREFILL)
        logits = self._lm_head(x_last)
        ttnn.deallocate(x)

        return logits

    def decode(self, token_ids, current_pos):
        B = token_ids.shape[0]

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = self.embd(token_ids_ttnn)
        ttnn.deallocate(token_ids_ttnn)

        # RoPE position is offset by rope_delta for a multimodal request (image tokens compress the
        # position space); the KV/cache position (cur_pos_tensor below) stays the true sequence pos.
        position_ids = torch.full((B, 1), current_pos + self.rope.rope_delta, dtype=torch.long)
        cos, sin = self.rope.get_rot_mats(position_ids)

        # cur_pos for SDPA decode + paged_update_cache ([B*n_kv] after cache reshape).
        n_kv = self.args.n_kv_heads
        cur_pos_tensor = ttnn.from_torch(
            torch.full((B * n_kv,), current_pos, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        for i, layer in enumerate(self.layers):
            x = layer.forward(x, cos=cos, sin=sin, mode="decode", position_tensor=cur_pos_tensor)

        x = self._final_norm_decode(x)
        if self._ondev_argmax:
            # Pre-gather vocab-sharded logits; caller argmaxes shards, skips all-gather + readback.
            logits = ttnn.linear(x, self.lm_head_weight, **_lm_head_kwargs())
        else:
            logits = self._lm_head(x)
        ttnn.deallocate(x)

        return logits

    def _forward_decode(
        self, token_ids_buf, cos, sin, cur_pos_tensor, page_table, sharded_lm_head=False, return_hidden=False
    ):
        """Trace-safe paged decode. All inputs are device tensors.

        sharded_lm_head=True: return the pre-gather vocab-sharded logits (no all-gather)
        for the on-device sampler, which does its own cross-device top-k + gather.
        return_hidden=True: also return the last-decoder-layer output (pre final norm),
        fractured [1,1,B,dim/tp] — the seed the MTP drafter head consumes. Returns
        (logits, hidden) in that case.
        """
        x = self.embd(token_ids_buf)
        if self.num_devices > 1:
            # TP expects [1,1,B,dim_frac]; embd yields [B,1,dim_frac].
            x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1], x.shape[-1]))
            if self._decode_ar is not None and not return_hidden:
                # QWEN36_DECODE_ALLREDUCE=1: one AG -> replicated RES40 residual. Not with return_hidden: the MTP drafter
                # consumes the last-layer hidden FRACTURED [1,1,B,dim/tp], so that decode keeps the fractured residual.
                x = self._decode_ar.enter(x)
        _cap = getattr(self, "_capture_layer", None)
        cos, sin, _dr = self._rope_decode_arg(cos, sin) if self.num_devices > 1 else (cos, sin, None)
        for _li, layer in enumerate(self.layers):
            if layer.is_full_attention:
                x = layer.forward(x, cos, sin, position_tensor=cur_pos_tensor, page_table=page_table, mode="decode")
            else:
                x = layer.forward(x, mode="decode")
            if _cap is not None and _li == _cap:  # debug: capture hidden after layer _cap
                self._cap_rec = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if _dr is not None:
            _dr.free()
        # Independent clone of the live last-layer output: the MTP head consumes this hidden on a
        # LATER forward and all-gathers it, so it must survive the final norm + lm_head below (a
        # to_memory_config is a no-op alias when x is already DRAM, so the norm would free it).
        hidden = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG) if return_hidden else None
        x = self._final_norm_decode(x)
        if sharded_lm_head or self._ondev_argmax:
            # Pre-gather vocab-sharded logits (on-device sampling / greedy argmax).
            logits = ttnn.linear(x, self.lm_head_weight, **_lm_head_kwargs())
        else:
            logits = self._lm_head(x)
        ttnn.deallocate(x)
        if return_hidden:
            return logits, hidden
        return logits

    def ttnn_mtp_prefill_forward(
        self, hidden_states, tokens, chunk_start, page_table_torch, valid_len=None, flex_sdpa=True
    ):
        """Warm the MTP drafter's KV over a whole prompt CHUNK in ONE forward.

        The drafter is a single full-attention decoder layer, so its KV fills like the base model's —
        one call per 2048-token chunk instead of one decode step per token (what the reference
        proposer does). ``hidden_states`` [1,1,S,dim/tp] fractured is the base per-position
        pre-final-norm hidden for this chunk. ``tokens`` torch [1,S] int32: slot i is fused with
        the token at i+1 (the draft loop's shift pairing). ``chunk_start`` is the absolute position
        of row 0. ``page_table_torch`` [1, num_blocks] is the MTP layer's own KV page table.
        ``valid_len``: real rows in this chunk; chunk blocks past them point at the MTP scratch block
        so padding rows never write K/V into real blocks. ``flex_sdpa``: device chunk_start tensor,
        one SDPA program per bucket for any chunk position.
        """
        assert self.mtp is not None, "MTP head not built (has_mtp/config?)"
        S = tokens.shape[-1]
        dev = self.device
        rep = ttnn.ReplicateTensorToMesh(dev)
        tok_tt = ttnn.from_torch(
            tokens.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, mesh_mapper=rep
        )
        # Same absolute positions as the base chunk, so the same RoPE table slice applies.
        cos_t, sin_t = self._rope_tp_cos_sin_torch(chunk_start, S, wide=False)  # the drafter keeps 64-wide RoPE
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=rep)
        block_size = get_block_size(self._paged_kv_caches)
        blk0 = chunk_start // block_size
        blkN = num_blocks_in_seq(chunk_start + S, block_size)
        pt_full = ttnn.from_torch(
            page_table_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, mesh_mapper=rep
        )
        pt_chunk_torch = page_table_torch[:, blk0:blkN].contiguous()
        if valid_len is not None:
            pt_chunk_torch = pt_chunk_torch.clone()
            keep = max(0, -(-(chunk_start + valid_len) // block_size) - blk0)
            pt_chunk_torch[:, keep:] = int(self.mtp.attention.paged_k.shape[0]) - 1  # scratch block
        pt_chunk = ttnn.from_torch(
            pt_chunk_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, mesh_mapper=rep
        )
        csi = None
        if flex_sdpa:
            assert chunk_start % 128 == 0, f"flex SDPA needs chunk_start % 128 == 0, got {chunk_start}"
            csi = ttnn.from_torch(
                torch.tensor([chunk_start], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                mesh_mapper=rep,
            )
        out = self.mtp.forward_prefill(
            hidden_states,
            tok_tt,
            cos,
            sin,
            pt_full,
            chunk_page_table=pt_chunk,
            chunk_start_idx=chunk_start,
            chunk_start_idx_tensor=csi,
        )
        for t in (tok_tt, cos, sin, pt_full, pt_chunk, csi, out):
            if t is not None:
                ttnn.deallocate(t)

    def plan_spec_prefix(self, tokens_row, L, start_pos, force_points=(), exact_resume=None, nonfinal=False):
        """Prefix-cache plan for ONE spec prefill (None = the unchanged from-0 path): (s, (cache, slot) | None, points).

        start_pos is vLLM's num_computed_tokens (None: caller does not take part in prefix caching). s = the largest
        snapshot position <= spec_resume_cap(start_pos, L) (0 = recompute everything; always correct): the snapshot is
        restored into the B=1 scratch and [s, L) is recomputed (main KV rewritten, drafter KV rewritten). `points`
        are the positions to snapshot while prefilling (see prefix_cache.spec_save_points). ``force_points``: extra
        128-aligned positions p (s < p <= L-3) to snapshot (MTP chunked prefill: the chunk end - 128); positions that do not
        fit are dropped.

        ``exact_resume`` = e (MTP chunked prefill Option B, start_pos == e, snapshot at e saved by the previous chunk): resume
        at e itself when a snapshot of tokens[:e] exists (needs L - e >= 3), else the normal cap. ``nonfinal``: this call
        is a non-final chunk whose end the caller snapshots itself: no end-of-prompt split point."""
        cache = self._prefix_cache
        if cache is None or start_pos is None or self.args.max_batch_size <= 1:
            return None
        from models.demos.blackhole.qwen36.tt.prefix_cache import (
            SpecPoints,
            spec_learn_positions,
            spec_resume_cap,
            spec_save_points,
        )

        cap = spec_resume_cap(start_pos, L)
        s, src = 0, None
        if exact_resume is not None and int(exact_resume) > 0 and L - int(exact_resume) >= 3:
            p, slot = cache.lookup(tokens_row, int(exact_resume))
            if slot is not None and p == int(exact_resume):
                s, src = p, (cache, slot)
        if src is None and cap > 0:
            p, slot = cache.lookup(tokens_row, cap)
            if slot is not None:
                s, src = p, (cache, slot)
        points = spec_save_points(
            L,
            s,
            spec_learn_positions(start_pos, L),
            chunk=self._chunked_chunk_size or 2048,
            min_len=int(os.environ.get("QWEN36_PREFIX_SNAPSHOT_MIN_LEN", "8192")),
            stride=int(os.environ.get("QWEN36_PREFIX_SNAPSHOT_STRIDE", "8192")),
            end_point=not nonfinal,
        )
        forced = [int(p) for p in force_points if s < int(p) <= L - 3 and int(p) % 128 == 0]
        if forced:
            low = points.low - set(forced)
            points = SpecPoints(sorted(set(points) | set(forced)))
            points.low = low
        if s == 0 and not points and not nonfinal:
            return None
        return s, src, points

    def prefill_for_spec(
        self,
        token_ids,
        page_table,
        actual_len,
        on_chunk,
        slot=0,
        plan=None,
        vision_tokens=None,
        final=True,
        save_end=False,
    ):
        """Chunked prompt prefill for speculative decode; hands each chunk's hidden to ``on_chunk``.

        Uses the same 2048-token chunking as the demo. Prefilling the whole prompt as one masked
        bucket put every length above the largest bucket outside the validated envelope (L1
        circular-buffer clash at 2642, slice overflow at 8192), so spec could not run 4k/8k/16k.
        ``on_chunk(hidden, chunk_start, valid_len)`` runs before each chunk's hidden is freed, so
        the MTP drafter can warm KV one forward per chunk. hidden is [1,1,bucket,dim/tp]; rows >=
        valid_len are padding. Returns next-token logits at actual_len-1 (device, replicated).

        ``slot``: GDN decode row this user's state ends up in; ``page_table`` is that user's [1, nb] row
        either way. The prefill itself is B=1, so slot > 0 (or any batched GDN binding) runs it on the
        persistent B=1 scratch and copies the result on device into row ``slot`` with TPGatedDeltaNet.write_slot,
        leaving the other users' rows intact (as prefill_paged_slots does). A B=1 model
        (max_batch_size == 1, slot 0) prefills in place.

        ``vision_tokens`` (multimodal prompt: packed vision rows, see Qwen36ForCausalLM._compute_vision_tokens): the M-RoPE
        table is built with the grid and each chunk splices its own rows. A multimodal prompt is never prefix-cached
        (``plan`` must be None).

        ``final`` False (MTP chunked prefill, a non-final chunk; needs ``plan``): the GDN state is only snapshotted at the plan's
        save points (+ the chunk end L when ``save_end``: Option B exact resume); nothing is written to decode row ``slot``
        and no logits are computed (returns None).
        """
        assert self.num_devices > 1, "prefill_for_spec is the TP path"
        assert final or plan is not None, "a non-final prefill chunk needs a prefix plan"
        assert vision_tokens is None or plan is None, "a multimodal prompt bypasses the prefix cache"
        assert 0 <= slot < self.args.max_batch_size, f"slot {slot} out of range [0,{self.args.max_batch_size})"
        if self.args.max_batch_size > 1:
            dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]
            dram = ttnn.DRAM_MEMORY_CONFIG
            prev = self._bind_gdn_prefill_scratch()
            try:
                if plan is None:
                    logits = self._prefill_for_spec_b1(
                        token_ids, page_table, actual_len, on_chunk, vision_tokens=vision_tokens
                    )
                else:
                    logits = self._prefill_for_spec_resumable(
                        token_ids, page_table, actual_len, on_chunk, plan, want_logits=final, save_end=save_end
                    )
                if not final:
                    return None
                snap = [
                    (
                        ttnn.clone(dn.rec_state, memory_config=dram),
                        [ttnn.clone(c, memory_config=dram) for c in dn.conv_states],
                    )
                    for dn in dn_layers
                ]
            finally:
                self._unbind_gdn_prefill_scratch(prev)
            for dn, (rec, convs) in zip(dn_layers, snap):
                dn.write_slot(slot, rec, convs)  # consumes the clones
            return logits
        return self._prefill_for_spec_b1(token_ids, page_table, actual_len, on_chunk, vision_tokens=vision_tokens)

    def _prefill_for_spec_resumable(self, token_ids, page_table, L, on_chunk, plan, want_logits=True, save_end=False):
        """Prefix-cached spec prefill on the BOUND B=1 scratch (plan from plan_spec_prefix): restore the snapshot at s
        (or zero the state when s == 0), prefill [s, p1), [p1, p2) ... [pk, L) at absolute positions with the same
        2048-chunk eager loop as _prefill_for_spec_b1 (``on_chunk`` gets every chunk, so the drafter KV is warmed over
        [s, L)), and snapshot the scratch after the segments that end at a save point. Returns device logits at L-1."""
        cache = self._prefix_cache
        s, src, points = plan
        tokens = token_ids[:, :L].to(torch.int32)
        self._build_request_rope(tokens, None)
        if s > 0:
            src[0].restore(src[1])
        else:
            self._reset_gdn_state_for_new_sequence()
        saved = []
        logits, seg_start = None, s
        bounds = list(points) + [L]
        for i, seg_end in enumerate(bounds):
            last = i == len(bounds) - 1
            logits = self._spec_prefill_segment(
                tokens, page_table, seg_start, seg_end, on_chunk, want_logits=last and want_logits
            )
            if not last:
                cache.save(tokens[0], seg_end, low_prio=seg_end in getattr(points, "low", ()))
                saved.append(seg_end)
            seg_start = seg_end
        if save_end:  # non-final chunk, exact resume: the next chunk starts from the state after tokens[:L]
            cache.save(tokens[0], L)
            saved.append(L)
        cache.log(L, -1, s, saved)
        return logits

    def _spec_prefill_segment(self, token_ids, page_table, start, end, on_chunk, want_logits):
        """tokens[start:end] (start % 128 == 0, GDN state carried from `start`) in 2048-token eager chunks + a masked
        tail; hands every chunk's hidden to on_chunk. Device logits at end-1 when want_logits, else None."""
        chunk_size = self._chunked_chunk_size or 2048
        n = end - start
        num_full, tail_real = n // chunk_size, n % chunk_size
        last_hidden = None
        logits = None
        for c in range(num_full):
            cs = start + c * chunk_size
            if last_hidden is not None:
                ttnn.deallocate(last_hidden)
            last_hidden = self._forward_prefill_chunk_masked_tp(
                token_ids[:, cs : cs + chunk_size],
                chunk_size,
                cs,
                page_table,
                chunk_size,
                flex_sdpa=True,
                full_chunk=True,
            )
            ttnn.synchronize_device(self.device)
            if c < num_full - 1 or tail_real > 0:
                on_chunk(last_hidden, cs, chunk_size)
        if tail_real > 0:
            if last_hidden is not None:
                ttnn.deallocate(last_hidden)
            cs = start + num_full * chunk_size
            logits, hidden = self.prefill_masked_bucket(
                token_ids[:, cs:end], page_table, actual_len=tail_real, chunk_start=cs, return_hidden=True
            )
            on_chunk(hidden, cs, tail_real)
            ttnn.deallocate(hidden)
        else:
            if want_logits:
                logits = self._masked_bucket_logits_tp(last_hidden, chunk_size, chunk_size)
            on_chunk(last_hidden, start + (num_full - 1) * chunk_size, chunk_size)
            ttnn.deallocate(last_hidden)
        if not want_logits and logits is not None:
            ttnn.deallocate(logits)
            logits = None
        return logits

    def _prefill_for_spec_b1(self, token_ids, page_table, actual_len, on_chunk, vision_tokens=None):
        """prefill_for_spec's body: one B=1 prompt into whatever GDN state is currently bound."""
        chunk_size = self._chunked_chunk_size or 2048
        num_full = actual_len // chunk_size
        tail_real = actual_len - num_full * chunk_size
        self._build_request_rope(token_ids[:, :actual_len], vision_tokens)

        if num_full == 0:
            logits, hidden = self.prefill_masked_bucket(
                token_ids[:, :actual_len],
                page_table,
                actual_len=actual_len,
                chunk_start=0,
                vision_tokens=vision_tokens,
                return_hidden=True,
            )
            on_chunk(hidden, 0, actual_len)
            ttnn.deallocate(hidden)
            return logits

        self._reset_gdn_state_for_new_sequence()
        last_hidden = None
        for c in range(num_full):
            cs = c * chunk_size
            if last_hidden is not None:
                ttnn.deallocate(last_hidden)
            # Stage this chunk's slice of the packed vision rows (clears the mask for text: no stale rows).
            self._set_vision_merge(
                token_ids[:, cs : cs + chunk_size], vision_tokens, self._vis_row_offset_for(token_ids, cs)
            )
            last_hidden = self._forward_prefill_chunk_masked_tp(
                token_ids[:, cs : cs + chunk_size],
                chunk_size,
                cs,
                page_table,
                chunk_size,
                flex_sdpa=True,
                full_chunk=True,
            )
            ttnn.synchronize_device(self.device)
            # Not the last chunk, or there is a tail after it: the base hidden is consumed here.
            if c < num_full - 1 or tail_real > 0:
                on_chunk(last_hidden, cs, chunk_size)
        if tail_real > 0:
            ttnn.deallocate(last_hidden)
            cs = num_full * chunk_size
            logits, hidden = self.prefill_masked_bucket(
                token_ids[:, cs:actual_len],
                page_table,
                actual_len=tail_real,
                chunk_start=cs,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, cs),
                return_hidden=True,
            )
            on_chunk(hidden, cs, tail_real)
            ttnn.deallocate(hidden)
            return logits
        # Exact multiple of chunk_size: the last chunk supplies both the logits and the last hidden.
        logits = self._masked_bucket_logits_tp(last_hidden, chunk_size, chunk_size)
        on_chunk(last_hidden, (num_full - 1) * chunk_size, chunk_size)
        ttnn.deallocate(last_hidden)
        return logits

    def ttnn_mtp_decode_forward(
        self,
        hidden_states,
        token_id,
        position,
        page_table,
        sharded_lm_head=False,
        need_logits=True,
        rope_deltas=None,
    ):
        """One MTP draft step over B rows at absolute ``position``. Builds the partial-RoPE cos/sin
        for the MTP positions (base + rope_delta), then runs the drafter head.
        rope_deltas: one M-RoPE delta per row (the rows' own); None = the scalar ``rope.rope_delta``.

        hidden_states : fractured [1,1,B,dim/tp] — the base's drafter feed (spec_feed_rows: the
                        fractured final-norm output) or the previous MTP step's next_hidden
                        (chaining).
        token_id      : the token at each row's ``position``. int (B=1), list[int] of length B, or a
                        device [B,1] uint32 tensor (on-device draft chaining).
        position      : int (all rows at the same slot) or list[int] of length B (one per user).
        page_table    : device int32 page table [B, nb] for the MTP layer's own KV cache.
        Returns (logits, next_hidden).
        """
        from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode

        assert self.mtp is not None, "MTP head not built (has_mtp/config?)"
        pos_list = [int(p) for p in position] if isinstance(position, (list, tuple, torch.Tensor)) else [int(position)]
        cos, sin = rot_mats_decode(
            self.device,
            self.args.rope_head_dim,
            self.args.max_seq_len,
            self.args.rope_theta,
            torch.tensor(
                [
                    p + (self.rope.rope_delta if rope_deltas is None else int(rope_deltas[i]))
                    for i, p in enumerate(pos_list)
                ],
                dtype=torch.int32,
            ),
        )
        rep = ttnn.ReplicateTensorToMesh(self.device)
        # A device [B,1] uint32 token_id (on-device draft chaining) is used as is, skipping a host round trip.
        if isinstance(token_id, ttnn.Tensor):
            tok = token_id
        else:
            _ids = [int(t) for t in token_id] if isinstance(token_id, (list, tuple, torch.Tensor)) else [int(token_id)]
            assert len(_ids) == len(pos_list), f"{len(_ids)} tokens for {len(pos_list)} positions"
            tok = ttnn.from_torch(
                torch.tensor(_ids, dtype=torch.int32).reshape(len(_ids), 1),
                dtype=ttnn.uint32,
                device=self.device,
                mesh_mapper=rep,
            )
        cur_pos_tt = ttnn.from_torch(
            torch.tensor(pos_list, dtype=torch.int32), dtype=ttnn.int32, device=self.device, mesh_mapper=rep
        )
        return self.mtp.forward_decode(
            hidden_states,
            tok,
            cur_pos_tt,
            cos,
            sin,
            page_table,
            sharded_lm_head=sharded_lm_head,
            need_logits=need_logits,
        )

    def spec_feed_rows(self, rows):
        """Map base per-position hidden rows to what the MTP drafter is fed (the spec "feed contract").

        ``rows`` is [1,1,*,dim/tp] fractured bf16: residual leaving the last decoder layer, BEFORE
        the final norm. Every site that hands base hidden to the drafter (prompt warm, seed, verify
        rows -> reseed / anchor) goes through here. Returns a NEW tensor — the OUTPUT of the base
        final norm, kept fractured to dim/tp by ``spec_norm`` (same weights, no trailing all-gather).
        Matching chain-side half: Qwen36MTP.forward_decode feeds back mtp.norm's output. Caller owns
        both tensors and frees ``rows`` when the result ``is not rows``. Trace-safe (fixed-shape norm).
        """
        return self.spec_norm(rows, mode=Mode.PREFILL)

    # ------------------------------------------------------------------------------------------
    # Traced speculative verify (the eager-loop fix). The verify forward is a fixed-shape forward over
    # exactly B users x T = K+1 candidate tokens, so — like the per-user bucket prefill trace
    # (capture_prefill_trace_bucket) — it can be captured once and replayed, turning the loop's
    # dominant cost from host-launched-per-op (eager) to a single execute_trace. The GDN recurrent
    # path is trace-safe (persistent pad + fixed-address GDN state buffers + a fixed per-token
    # state ring); B*T is constant across replays (unlike variable-length prompts), so every shape
    # in it bakes into the trace (there is no GDN mask: no padded rows).
    #
    # Row layout (the invariant the whole spec stack shares): B*T <= TILE_SIZE rows of one decode
    # tile, user-major, so row r = u*T + j is user u's candidate j. Full attention sees every row as
    # a pseudo-user (per-row KV write at its own absolute position) folded into per-user SDPA
    # groups; GDN reshapes the rows to [B, T, C], a plain reshape because the rows are user-major.
    #
    # The commit is a deferred select, not a device phase: after readback the host writes each
    # user's accepted-prefix index mi_u into two small shared device selectors (_vfy_state_idx,
    # _vfy_conv_sel) before the next replay. The recurrent kernel then starts user u from ring
    # block mi_u and the conv1d from window row mi_u, so nothing is copied between iterations.
    # ------------------------------------------------------------------------------------------
    def _spec_row_deltas(self, n):
        """Per-ENGINE-row M-RoPE deltas [n] (python ints; 0 for text and for rows past the table): what the spec prefill
        recorded for each row (set_slot_rope_delta). The spec verify / draft / join builders use these instead of the
        scalar ``rope.rope_delta`` (the last prefill's), so rows of different requests keep their own delta. A model without
        the per-slot vector (a test fake) falls back to the scalar for every row."""
        sd = getattr(self, "_slot_rope_delta", None)
        if sd is None:
            return [int(self.rope.rope_delta)] * int(n)
        v = [int(x) for x in sd[: int(n)].tolist()]
        return v + [0] * (int(n) - len(v))

    def _spec_pos_deltas(self, n_pos):
        """Per-position deltas for a verify-layout vector of ``n_pos`` = users * T rows (user-major): each user's delta T times."""
        T = int(self._vfy_T)
        out = []
        for d in Qwen36Model._spec_row_deltas(self, int(n_pos) // T):
            out.extend([d] * T)
        return out

    def _rope_tp_cos_sin_decode_torch(self, positions, deltas=None, wide=False):
        """Torch cos/sin [1, B, 1, rope_head_dim] in the DECODE rope layout — one rotation per ROW.

        Byte-identical to what prepare_decode_inputs_host builds for a decode step at those positions
        (same inv_freq / outer / cat / bf16 cast). The hybrid verify pushes its candidates through the
        DECODE attention kernel as B pseudo-users, so it needs this per-row table, not the
        prefill [1,1,S,rd] one.
        wide (QWEN36_ROPE_PERM; the BASE model's callers pass self._rope_perm): permuted hd-wide [1, B, 1, head_dim]
        (rope_tp.widen_cos_sin). The MTP drafter's callers keep the default (64-wide)."""
        pos_vec = positions.to(torch.int32).reshape(-1)
        # RoPE position is the KV position offset by rope_delta (0 for text); same as decode. ``deltas``: one delta per
        # position (the rows' own); None = the scalar request delta.
        rope_pos_vec = pos_vec + (
            self.rope.rope_delta if deltas is None else torch.as_tensor(deltas, dtype=torch.int32)
        )
        B = rope_pos_vec.shape[0]
        cos, sin = self._rope_cos_sin_rows(rope_pos_vec)
        if wide:
            cos, sin = self._rope_widen(cos, sin)
        return cos.reshape(1, B, 1, -1), sin.reshape(1, B, 1, -1)

    def _rope_widen(self, cos, sin):
        """widen_cos_sin for bf16 cos/sin rows [..., rope_head_dim] -> permuted [..., head_dim] (QWEN36_ROPE_PERM)."""
        from models.demos.blackhole.qwen36.tt.attention.rope_tp import widen_cos_sin

        return widen_cos_sin(cos, sin, self.args.head_dim)

    def _rope_cos_sin_rows(self, rope_pos_vec):
        """bf16 (cos, sin) [len(rope_pos_vec), rope_head_dim] for RoPE positions (rope_delta already applied)."""
        rd = self.args.rope_head_dim
        inv_freq = 1.0 / (self.args.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
        freqs = torch.outer(rope_pos_vec.float(), inv_freq)  # [B, rd/2], per-row rotation
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)

    def _rope_tp_cos_sin_decode_cached(self, positions, deltas=None, wide=False):
        """_rope_tp_cos_sin_decode_torch served from a host bf16 table built once (QWEN36_VERIFY_STAGING_CACHE):
        row p = _rope_cos_sin_rows(p), so the result is bit-identical; positions outside the table fall back to
        the direct computation. Saves the per-step outer/cos/sin on the host."""
        tab = getattr(self, "_rope_host_table", None)
        if tab is None:
            tab = self._rope_host_table = self._rope_cos_sin_rows(torch.arange(self.rope.max_seq_len))
        rope_pos = positions.to(torch.int32).reshape(-1) + (
            self.rope.rope_delta if deltas is None else torch.as_tensor(deltas, dtype=torch.int32)
        )
        if rope_pos.numel() == 0 or int(rope_pos.min()) < 0 or int(rope_pos.max()) >= tab[0].shape[0]:
            return self._rope_tp_cos_sin_decode_torch(positions, deltas, wide)
        B = rope_pos.shape[0]
        idx = rope_pos.long()
        cos, sin = tab[0][idx], tab[1][idx]
        if wide:
            cos, sin = self._rope_widen(cos, sin)
        return cos.reshape(1, B, 1, -1), sin.reshape(1, B, 1, -1)

    @staticmethod
    def _vfy_pt_unchanged(prev, new):
        """True when host page table ``new`` equals the one last staged on device (``prev``): same shape and
        values, so the verify's page-table H2D copies can be skipped."""
        return prev is not None and prev.shape == new.shape and bool(torch.equal(prev, new))

    def _spec_tail(self, x, sharded=False):
        """Shared tail of the verify and seed bodies: residual ``x`` [1,1,R,dim] (freed here) ->
        (logits [1,1,R,vocab] replicated per device, feed [1,1,R,dim/tp] fractured).
        ``sharded=True`` (QWEN36_VERIFY_SHARDED_ARGMAX verify trace): returns (logits shard
        [1,1,R,vocab/TP] with NO all-gather, feed, normed [1,1,R,dim] kept alive for the logits trace).

        The bucket is exactly the R real rows, so there is no row-select. ``feed`` is the post-norm
        drafter feed (spec_feed_rows); its programs are the inner norm's, already compiled by the
        self.norm call."""
        if self._decode_ar is not None and x.shape[-1] == self.args.dim:
            return self._spec_tail_replicated(x, sharded)
        rows = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)
        normed = self.norm(rows, mode=Mode.PREFILL)  # -> full [1,1,R,dim]
        logits = self._lm_head(normed, gather=not sharded)
        if not sharded:
            ttnn.deallocate(normed)
        feed = self.spec_feed_rows(rows)
        if feed is not rows:
            ttnn.deallocate(rows)
        if sharded:
            return logits, feed, normed
        return logits, feed

    def _spec_tail_replicated(self, x, sharded):
        """_spec_tail for the replicated RES40 residual (QWEN36_VERIFY_DAR): the final norm is the plain decode's local
        sharded norm (_final_norm_decode), so the LM head sees what a decode step feeds it; no stats / hidden all-gathers.
        The drafter feed is this device's dim/tp span of that normed row (mesh_partition, the inverse of the gather: the
        MTP chain does the same, Qwen36MTP.forward_decode) instead of a second distributed norm over the fractured rows.
        Returns what _spec_tail returns: (logits, feed) or, sharded, (logits shard, feed, normed)."""
        normed = self._final_norm_decode(x)  # [1,1,R,dim] DRAM, replicated
        ttnn.deallocate(x)
        logits = self._lm_head(normed, gather=not sharded)
        feed = ttnn.mesh_partition(normed, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if sharded:
            return logits, feed, normed
        ttnn.deallocate(normed)
        return logits, feed

    def _verify_dar(self, rows):
        """The DecodeResidualAllReduce the traced verify runs on (QWEN36_VERIFY_DAR, default on), or None for the fractured
        residual of the old verify. Needs the decode all-reduce itself (QWEN36_DECODE_ALLREDUCE=1 and a supported mesh,
        ``self._decode_ar``), a bucket of at most one tile of rows (RES40 shard height) and the fused gdn_spec_step on every
        GDN layer (the unfused verify keeps its fp32 out-proj reduce-scatter). Lossless: the per-row reduction is the
        very all_reduce_async of plain decode, so verify rows and decode rows round the same way (the old reduce-scatter
        + all-gather summed the TP partials in another order)."""
        if not verify_dar_enabled() or self._decode_ar is None or rows > ttnn.TILE_SIZE:
            return None
        if not all(
            getattr(layer.attention, "_spec_fused", False) for layer in self.layers if not layer.is_full_attention
        ):
            return None
        return self._decode_ar

    # ------------------------------------------------------------------------------------------------------- #
    # QWEN36_SPEC_HOSTSET: packed verify inputs (note 39). The per-replay kvpos / RoPE index / GDN ctrl page of the
    # verify are ONE uint32 host buffer ([3, W], row 0 = row positions, row 1 = RoPE table row per verify row,
    # row 2 = the ctrl page) = ONE H2D copy; the trace slices it and looks cos/sin up from the resident RoPE tables.
    # ------------------------------------------------------------------------------------------------------- #
    def _vfy_inputs(self, rows):
        """(kvpos, cos, sin, ctrl, owned): the verify body's per-replay inputs. Without the packed layout these are the
        persistent buffers (owned = []). With it (``_vfy_hostset``) they are derived from ``_vfy_aux_buf`` inside the
        body: kvpos = typecast(slice row 0) int32 [rows]; cos / sin = ttnn.embedding(slice row 1, RoPE tables)
        [1,1,rows,rope_dim] TILE (_dl_rope_cos_sin, the lookup the batched decode loop runs); ctrl = slice row 2 [1,N]
        uint32 (fused GDN only). ``owned`` are freed by the body when it is done."""
        if not getattr(self, "_vfy_hostset", False):
            return (
                self._vfy_kvpos_buf,
                self._vfy_cos_buf,
                self._vfy_sin_buf,
                getattr(self, "_vfy_ctrl", None),
                [],
            )
        aux = self._vfy_aux_buf
        pos_u = ttnn.slice(aux, [0, 0], [1, rows])
        kvpos_i = ttnn.typecast(pos_u, ttnn.int32)
        ttnn.deallocate(pos_u)
        kvpos = ttnn.reshape(kvpos_i, (rows,))
        ridx = ttnn.slice(aux, [1, 0], [2, rows])
        cos, sin = self._dl_rope_cos_sin(ridx, rows, wide=self._rope_perm)
        ctrl = None
        owned = [kvpos, cos, sin, ridx]
        if getattr(self, "_vfy_fused", False):
            ctrl = ttnn.slice(aux, [2, 0], [3, self._vfy_ctrl_words])
            owned.append(ctrl)
        return kvpos, cos, sin, ctrl, owned

    def _vfy_pos_masked(self, kvpos, rows, T):
        """T masked position tensors for the masked KV write (QWEN36_KV_WRITE_MASKED), built once per trace and shared by every
        full-attention layer: entry j = kvpos where row % T == j, else -1 (paged_fused_update_cache skips a -1 row), as device
        int32 ops on the persistent 0/1 masks m_j: kvpos * m_j + (m_j - 1). Returns None when the masks were not allocated
        for this row count (the layers then keep the grouped write). Each entry is a [rows] int32 ROW_MAJOR DRAM tensor.
        """
        ms = getattr(self, "_vfy_kvmask", {}).get(rows)
        if not ms:
            return None
        _dram = ttnn.DRAM_MEMORY_CONFIG
        p_rm = ttnn.reshape(kvpos, (1, rows))
        p_t = ttnn.to_layout(p_rm, ttnn.TILE_LAYOUT, memory_config=_dram)
        out = []
        for m, nm in ms:
            x = ttnn.multiply(p_t, m, memory_config=_dram)
            y = ttnn.add(x, nm, memory_config=_dram)
            ttnn.deallocate(x)
            z = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
            ttnn.deallocate(y)
            out.append(ttnn.reshape(z, (rows,)))
        ttnn.deallocate(p_t)
        return out

    def _expand_row_deltas(self, rope_delta, n_pos):
        """``rope_delta`` (an int, or one delta per engine row) as the per-position value rope_row_index takes for a
        verify-layout vector of ``n_pos`` = users * T rows: an int stays a scalar, a list is repeated T times per user.
        """
        if not isinstance(rope_delta, (list, tuple, torch.Tensor)):
            return int(rope_delta)
        d = [int(x) for x in torch.as_tensor(rope_delta).reshape(-1).tolist()]
        T = int(self._vfy_T)
        assert len(d) * T == int(n_pos), (len(d), T, n_pos)
        return [x for x in d for _ in range(T)]

    def _vfy_aux_host(self, row_positions, mi_prev, parity=None, rope_delta=None):
        """Host form of ``_vfy_aux_buf``: torch.int32 [3 (2 without the fused GDN op), W]. ``parity`` (default: the layers'
        current GDN window parity): the parity word of the ctrl page, for a step whose replay was enqueued earlier.
        ``rope_delta`` (default: each row's own delta): an int = every row the same; a list of one delta per engine row
        (each repeated for the row's T positions, user-major), as the fused / chain device bodies build it.
        """
        from models.demos.blackhole.qwen36.tt.gdn.tp import spec_ctrl_page

        n = len(row_positions)
        segs = [
            row_positions,
            rope_row_index(
                row_positions,
                (
                    Qwen36Model._spec_pos_deltas(self, n)
                    if rope_delta is None
                    else Qwen36Model._expand_row_deltas(self, rope_delta, n)
                ),
                self.rope.max_seq_len,
            ),
        ]
        if getattr(self, "_vfy_fused", False):
            if parity is None:
                _pars = {dn._spec_win_par for dn in self._vfy_gdn}
                assert len(_pars) == 1, f"GDN window parities out of step: {sorted(_pars)}"
                parity = _pars.pop()
            segs.append(spec_ctrl_page(mi_prev, self._vfy_B, self._vfy_Nv, parity=parity).reshape(-1))
        return pack_u32_rows(segs)

    def _vfy_stage_aux(self, aux_host):
        """Copy the packed aux inputs to the device unless the device already holds exactly these words (the host copy
        of what the buffer holds is ``_vfy_aux_staged``, written only here). Returns True when a copy was made."""
        prev = getattr(self, "_vfy_aux_staged", None)
        if prev is not None and prev.shape == aux_host.shape and bool(torch.equal(prev, aux_host)):
            return False
        rep = ttnn.ReplicateTensorToMesh(self.device)
        _h = ttnn.from_torch(aux_host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep)
        ttnn.copy_host_to_device_tensor(_h, self._vfy_aux_buf)
        self._vfy_aux_staged = aux_host.clone()
        return True

    def vfy_gdn_parity(self):
        """The GDN conv-window parity the NEXT verify replay reads (one value for every layer; it flips once per replay, in
        verify_after_external_sync). The chain overlay (spec_chain.pack_ovr_host) carries it as its parity word."""
        _pars = {dn._spec_win_par for dn in self._vfy_gdn}
        assert len(_pars) == 1, f"GDN window parities out of step: {sorted(_pars)}"
        return _pars.pop()

    def prestage_verify_aux(self, positions, mi_prev):
        """Stage the NEXT verify's packed aux inputs for the predicted ``positions`` / ``mi_prev`` (B ints each) while the
        drafter trace runs on the device. Safe by construction: it writes the same words verify_traced would build
        for these arguments, and verify_traced compares its own words with ``_vfy_aux_staged`` and copies again when
        they differ, so a wrong prediction costs one extra tiny copy and nothing else. The drafter trace reads none
        of these buffers; the copy is queued behind it. Returns True when it staged, False when the packed layout is
        off or the arguments are not a valid verify's."""
        if not getattr(self, "_vfy_hostset", False) or getattr(self, "_vfy_trace_id", None) is None:
            return False
        T = self._vfy_T
        if len(positions) != self._vfy_B or len(mi_prev) != self._vfy_B:
            return False
        if any(int(x) < 0 for x in positions) or any(not 0 <= int(x) < T for x in mi_prev):
            return False
        self._vfy_stage_aux(self._vfy_aux_host(self._vfy_row_positions(positions), mi_prev))
        return True

    def _vfy_narrow_logits(self, logits_rm, b=None):
        """Column 0 of every user of the ROW_MAJOR verify logits [1,1,B*T,V] -> [B,1,V] (rows u*T), on the device: a
        view-reshape to [B,T,V] and a slice of the T axis, so a narrow (no-draft) step reads B rows (1/T of the bytes)
        instead of B*T. Pure data movement: the values are the full read's rows u*T."""
        B, T = (self._vfy_B if b is None else b), self._vfy_T
        v = ttnn.reshape(logits_rm, (B, T, int(logits_rm.shape[-1])))
        return ttnn.slice(v, [0, 0, 0], [B, 1, int(logits_rm.shape[-1])])

    def _vfy_warm_narrow(self, logits_rm, b=None):
        """Compile the narrow-read slice before any trace is captured (a program compiled while a trace is parked can
        land in memory the replay writes over). Any failure leaves the narrow read off and the full read in place.
        ``b``: a bucket's row count (the bucket's own flag ``narrow_ready`` is returned instead of set on the model)."""
        ready = False
        if spec_hostset_enabled() and self._vfy_T >= 2:
            try:
                out = self._vfy_narrow_logits(logits_rm, b)
                ttnn.synchronize_device(self.device)
                ttnn.deallocate(out)
                ready = True
            except Exception as e:  # noqa: BLE001 - an optimisation: never block startup
                logger.warning(f"QWEN36_SPEC_HOSTSET narrow logits read disabled (compile failed: {e})")
        if b is None:
            self._vfy_narrow_ready = ready
        return ready

    def _forward_verify_bucket_tp(self, n_users, bk=None):
        """Trace body: recurrent verify over the persistent B*T-row bucket buffers -> per-position
        logits + hidden. Reads only fixed-address buffers so it is trace-capturable. Full-attention
        layers run the decode flash kernel with the rows as pseudo-users (see capture_verify_trace).
        GDN stays on seq-dim recurrent verify per user (batch dim is not a valid recurrence axis) and
        takes its per-user starting state through the two shared selectors. Returns (logits, rows,
        ids, logits_rm): ``rows`` is the drafter feed [1,1,B*T,dim/tp] bf16 (spec_feed_rows).
        ``logits_rm`` is the ROW_MAJOR bf16 copy the in-trace argmax needs, kept for the sampling path.

        With ``self._vfy_sharded`` (QWEN36_VERIFY_SHARDED_ARGMAX) the tail skips the logits all-gather,
        untilize and full-vocab argmax: ids come from the per-shard argmax + max combined on device
        (shard_argmax_max_rows + _dl_greedy_token_row_batched, as the batched decode loop does), and the
        return is (None, rows, ids [1,1,1,32] uint32 lanes 0..R-1, None, normed, shard_logits). The full
        logits are then produced only by the separate logits trace (_forward_verify_logits_body) when a
        sampled request asks. ``shard_logits`` ([1,1,R,vocab/TP] bf16 TILE, the tensor the argmax read) is
        kept alive as a sixth persistent output only under QWEN36_VERIFY_DEVICE_TOPK (``self._vfy_topk``),
        for the topk trace (_forward_verify_topk_body); otherwise it is freed here and the slot is None.

        ``n_users < self._vfy_B`` (a row bucket, QWEN36_SPEC_BUCKETS; ``bk`` is its _VfyBucket) runs the first ``n_users``
        rows of the shared B-row state: the token buffer and the three page tables are sliced to the bucket's rows, the
        aux / ctrl words are already sliced by ``rows`` (_vfy_inputs), the GDN op strides its ring by B (max_users). The tail
        hands its ids and drafter-feed rows over to the shared B-row outputs (``bk.handoff``) inside the trace and returns
        (None, None, None, None, normed, shard_logits): normed / shard_logits are bucket-local (read by the bucket's logits
        / topk traces). ``n_users == B`` is the unchanged B-row body.
        """
        T = self._vfy_T
        rows = n_users * T
        bucket = n_users < self._vfy_B
        if bucket:
            assert bk is not None and bk.b == n_users, f"bucket body of {n_users} rows needs its _VfyBucket"
            assert self._vfy_sharded, "row buckets need the sharded verify"
        kvpos, cos, sin, ctrl, owned = self._vfy_inputs(rows)
        pt_rows, pt_users, pt_groups = self._vfy_kvpt_buf, self._vfy_kvpt_users, self._vfy_kvpt1_buf
        if bucket:
            nb = self._vfy_nb
            tok = ttnn.slice(self._vfy_token_buf, [0, 0], [1, rows])
            pt_rows = ttnn.slice(self._vfy_kvpt_buf, [0, 0], [rows, nb])
            pt_users = ttnn.slice(self._vfy_kvpt_users, [0, 0], [n_users, nb])
            G8 = self._vfy_spec_groups
            if (
                G8 >= self._vfy_B and G8 % self._vfy_B == 0
            ):  # the fused spec SDPA's per-group table: users 0..b-1 are the first rows
                _g = G8 // self._vfy_B * n_users
                assert _g == int(self._vfy_att0_layer().spec_sdpa_groups(T, n_users)), "spec SDPA group count"
                pt_groups = ttnn.slice(self._vfy_kvpt1_buf, [0, 0], [_g, nb])
            else:
                pt_groups = None  # fused SDPA off: the table is unread
            x = self.embd(tok)
            ttnn.deallocate(tok)
        else:
            x = self.embd(self._vfy_token_buf)
        x = ttnn.reshape(x, (1, 1, rows, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        _dar = self._verify_dar(rows)
        if _dar is not None:
            x = _dar.enter(x)  # QWEN36_VERIFY_DAR: one AG -> replicated RES40 residual, as the plain decode
        # QWEN36_VERIFY_ATTN_LEAN: cos/sin in the prefill-shaped layout apply_partial_rope_decode wants ([1,1,R,rd]; the
        # staged buffers are [1,R,1,rd]), built ONCE here instead of 4 reshapes per attention layer; and the T strided
        # position slices of the grouped KV write, likewise shared by every attention layer.
        # With the packed aux (QWEN36_SPEC_HOSTSET) cos/sin/kvpos are the trace-derived tensors from _vfy_inputs (cos/sin
        # already [1,1,R,rd] from the RoPE table lookup): one source, no second reshape; the T position slices read kvpos.
        _rd = self.args.rope_head_dim
        _pos_groups, _tmp, _pos_masked = None, [], None
        _cos, _sin, _dr = self._rope_decode_arg(
            cos, sin
        )  # QWEN36_ROPE_PERM: one DecodeRope for all layers (cos is [1,rows,1,hd])
        if verify_attn_lean_enabled():
            if _dr is None and tuple(cos.shape) != (1, 1, rows, _rd):
                _cos = ttnn.reshape(cos, (1, 1, rows, _rd))
                _sin = ttnn.reshape(sin, (1, 1, rows, _rd))
                _tmp += [(_cos, cos), (_sin, sin)]
            from models.demos.blackhole.qwen36.tt.attention.tp import KV_GROUP_WRITE_OK

            if KV_GROUP_WRITE_OK and T > 1:
                _pos_groups = [ttnn.slice(kvpos, (j,), (rows,), (T,)) for j in range(T)]
            _pos_masked = self._vfy_pos_masked(kvpos, rows, T) if KV_GROUP_WRITE_OK and T > 1 else None
        for layer in self.layers:
            if layer.is_full_attention:
                # Hybrid verify: every row is a pseudo-user of its own sequence. cur_pos for row u*T+j is
                # p_u+j, so the decode SDPA's causal bound gives that row exactly [0, p_u+j]; the candidate
                # rows above it are written but masked out. alias_kv_write=True because a user's T rows
                # share one page table (see TPAttention._write_kv_aliased). At n_users > 1 the write goes
                # out per candidate (T calls of n_users rows, one per user) and _vfy_kvpt_users, the
                # [B, nb] per-user table, is reused by every group.
                x_new = layer.forward(
                    x,
                    cos=_cos,
                    sin=_sin,
                    mode="decode",
                    position_tensor=kvpos,
                    page_table=pt_rows,
                    alias_kv_write=True,
                    spec_user_page_table=pt_users,
                    # The SDPA read folds those rows into a few batch rows (spec_multi_pos_tiles: groups
                    # of 4 candidates, n_users x the single-user split), so the KV cache streams out of
                    # DRAM once per group per layer instead of once per row. The op wants one aliased
                    # page-table row per group; the per-row KV write above still needs the full row-count
                    # alias table. A T with no L1-fitting split (7, 11, ...) uses the legacy per-row call.
                    spec_verify_mode=True,
                    spec_page_table=(pt_groups if pt_groups is not None else self._vfy_kvpt1_buf),
                    n_users=n_users,
                    spec_pos_groups=_pos_groups,
                    spec_pos_masked=_pos_masked,
                )
            else:
                x_new = layer.forward(
                    x,
                    mode="prefill",
                    chunk_size=self.args.gdn_chunk_size,
                    valid_len=rows,
                    gdn_recurrent=True,
                    decode_cfg=True,
                    n_users=n_users,
                    state_blk_idx=self._vfy_state_idx,
                    conv_sel=self._vfy_conv_sel,
                    spec_ctrl=ctrl,  # QWEN36_VERIFY_SPEC_STEP=1: the fused op's page
                )
            ttnn.deallocate(x)
            x = x_new
        if _dr is not None:
            _dr.free()
        for _new, _old in _tmp:  # per-trace temporaries (a reshape that aliased its source is not freed twice)
            if _new is not _old and not _tpc_aliases(_new, _old):
                ttnn.deallocate(_new)
        for _g in _pos_groups or ():
            ttnn.deallocate(_g)
        for _g in _pos_masked or ():
            ttnn.deallocate(_g)
        if bucket:  # the sliced page tables are per-trace temporaries (never a full-span alias: b < B)
            for t in (pt_rows, pt_users, pt_groups):
                if t is not None:
                    ttnn.deallocate(t)
        if self._vfy_sharded:
            shard_logits, rows_out, normed = self._spec_tail(x, sharded=True)
            idx, val = self.shard_argmax_max_rows(shard_logits, rows)
            if getattr(self, "_vfy_topk", False):
                keep = shard_logits  # the topk trace ranks this exact tensor (QWEN36_VERIFY_DEVICE_TOPK)
            else:
                ttnn.deallocate(shard_logits)
                keep = None
            ids = self._dl_greedy_token_row_batched(idx, val, bk.dl if bucket else self._vfy_dl)  # [1,1,1,32] uint32 RM
            for t in owned:
                ttnn.deallocate(t)
            if bucket:
                # In-trace hand-off into the shared B-row outputs the accept / chain / drafter / host read. ids lanes >= rows
                # are 0 (the argmax pad); the drafter-feed rows >= rows of the shared buffer keep their content.
                from models.demos.blackhole.qwen36.tt.spec_bucket import handoff_rows_concat

                ttnn.copy(ids, bk.handoff["ids"])
                handoff_rows_concat(
                    rows_out, bk.handoff["rows"], rows
                )  # in place (ttnn.copy): slice_write on a TILE dst re-binds the dst tensor to a new buffer
                ttnn.deallocate(ids)
                ttnn.deallocate(rows_out)
                return None, None, None, None, normed, keep
            return None, rows_out, ids, None, normed, keep
        assert not bucket, "row buckets need the sharded verify"
        logits, rows_out = self._spec_tail(x)  # [1,1,B*T,vocab] replicated, drafter feed [1,1,B*T,dim/tp]
        # Greedy acceptance compares token ids, so argmax the verify rows inside the trace and read back
        # B*T uint32 instead of B*T x 151936 floats. ttnn.argmax needs ROW_MAJOR input (a TILE tensor
        # takes a single-core internal untilize that is hopeless at this vocab width), so untilize first.
        # No 32-row pad: the multicore argmax is correct below a full tile of rows, and padding would
        # multiply the bytes untilize and argmax touch (see mtp.argmax_last).
        u = ttnn.untilize(logits, use_multicore=True)
        ids = ttnn.argmax(u, dim=-1, keepdim=False)  # [1,1,B*T] uint32 ROW_MAJOR
        # `u` is NOT deallocated: it is handed back as the ROW_MAJOR logits copy. The sampling path
        # reads THAT back instead of the TILE `logits`, which costs an untilize on host.
        for t in owned:
            ttnn.deallocate(t)
        return logits, rows_out, ids, u, None, None

    def _vfy_att0_layer(self):
        """The first full-attention layer's attention module (its spec SDPA plan names the group counts)."""
        return getattr(self, "_vfy_att0", None) or next(
            layer.attention for layer in self.layers if layer.is_full_attention
        )

    def _forward_verify_logits_body(self, bk=None):
        """Second verify trace (sharded-argmax mode only): full replicated logits [1,1,R,vocab] + the ROW_MAJOR
        copy, recomputed from the first trace's persistent ``_vfy_normed_out`` (same linear program and input
        as the in-trace shard logits, so the values are the ones the argmax saw). Replayed right after the
        main trace only when read_logits=True (sampled requests, spec_mode == "logits"), or lazily by
        fetch_verify_logits when a topk read needs the full row (spec_mode == "topk" fallback).
        ``bk``: a row bucket's _VfyBucket (reads its own ``normed``)."""
        logits = self._lm_head(self._vfy_normed_out if bk is None else bk.normed)
        u = ttnn.untilize(logits, use_multicore=True)
        return logits, u

    def _forward_verify_topk_body(self, bk=None):
        """Third verify trace (QWEN36_VERIFY_DEVICE_TOPK, sharded-argmax mode only): per-device top-M of the
        main trace's persistent shard logits ``_vfy_shard_logits_out`` [1,1,R,vocab/TP] bf16 TILE, gathered
        over the TP devices. Returns (values [1,1,32,TP*M] bf16, ids [1,1,32,TP*M] uint32); device d's block
        is columns d*M .. d*M+M-1, sorted (value desc, LOCAL id asc) by ``stable=True``; the host adds
        d * vocab/TP and merges (spec_engine.merge_shard_topk). Rows past R and lanes past vocab/TP are
        -1e30 pads (ttnn.topk wants 32 rows and a power-of-two width; the pads never rank above a real
        logit, so a local id >= vocab/TP can only come back from a pad row). The values are the shard
        logits' own bf16 values, unchanged, so they equal what the logits trace's RM readback holds.
        Replayed right after the main trace when read_topk=True (spec_mode == "topk").
        ``bk``: a row bucket's _VfyBucket: ranks its own ``shard_logits`` and copies the (same-shape, 32-row) results into the
        shared topk buffers (``bk.handoff``); returns (None, None) then."""
        x = self._vfy_shard_logits_out if bk is None else bk.shard_logits
        rows = int(x.shape[2])
        per_shard = int(x.shape[3])
        pad_rows = ttnn.TILE_SIZE - rows
        p = ttnn.pad(x, [(0, 0), (0, 0), (0, pad_rows), (0, self._vfy_topk_w - per_shard)], value=-1e30)
        vals, idx = ttnn.topk(p, k=self._vfy_topk_m, dim=-1, largest=True, sorted=True, stable=True)
        ttnn.deallocate(p)
        # uint16 -> uint32 before the gather: the uint32 TILE all-gather is the proven path
        # (_dl_greedy_token_row_batched); the decode sampler gathers the uint16 and widens afterwards.
        idx32 = ttnn.typecast(idx, ttnn.uint32)
        ttnn.deallocate(idx)
        g_vals = ttnn.all_gather(vals, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g_idx = ttnn.all_gather(idx32, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(vals)
        ttnn.deallocate(idx32)
        if bk is not None:
            ttnn.copy(g_vals, bk.handoff["topk_vals"])
            ttnn.copy(g_idx, bk.handoff["topk_ids"])
            ttnn.deallocate(g_vals)
            ttnn.deallocate(g_idx)
            return None, None
        return g_vals, g_idx

    def _forward_verify_stats_body(self, bk=None, x=None, uv=None):
        """Fourth verify trace (QWEN36_SPEC_DEVICE_LOGPROBS / QWEN36_SPEC_DEVICE_TOPP, design note 67 section 3): per engine
        row r, over the WHOLE vocabulary (TP shards x vocab/TP),
            L_raw[r] = sum exp(x - M_r)            (raw logsumexp = M_r + log L_raw)
            L_T[r]   = sum exp((x - M_r) / T_r)    (top-p normaliser of the row's temperature)
        with M_r the row's global maximum, read from the topk trace's union ``_vfy_topk_vals_out`` (it holds every shard's
        maximum, so max over its TP*M columns is the global maximum) and T_r the row's temperature from the engine's samp_in row 0
        (greedy / non-positive temperature rows use T = 1). Everything is fp32 (accurate exp, HiFi4, fp32 destination
        accumulation: bf16 accumulation is off by 0.25 in lse); the two sums are stacked on the row axis and reduced by ONE
        ttnn.sum, then all-gathered over the shards and added. The result is written in place into ``_vfy_stats_out`` fp32
        [1,1,64,1]: rows 0..31 = L_raw, rows 32..63 = L_T (row r = engine row r; rows past R hold finite or garbage values the
        host must ignore). Replayed by verify_stage_and_enqueue(execute_stats=True) right after the topk trace; reads that trace's
        union, so the topk trace must have run in the same step.
        ``bk``: a row bucket (reads its ``shard_logits``, R = b*T rows, padded to 32). ``x`` / ``uv``: explicit inputs (warm-up
        stand-ins)."""
        TL, f32 = ttnn.TILE_LAYOUT, ttnn.float32
        T = self._vfy_T
        if x is None:
            x = self._vfy_shard_logits_out if bk is None else bk.shard_logits
        if uv is None:
            uv = self._vfy_topk_vals_out
        rows = int(x.shape[2])
        nu = rows // T
        sc = self._vfy_stats_consts
        # 1 / T per engine row [1,1,32,1]
        samp = ttnn.to_layout(self._vfy_stats_samp, TL)  # [4+B, W] fp32
        sampt = ttnn.transpose(samp, 0, 1)  # [W, 4+B]
        ttnn.deallocate(samp)
        temp = ttnn.slice(sampt, [0, 0], [nu, 1])
        greedy = ttnn.slice(sampt, [0, 3], [nu, 4])
        ttnn.deallocate(sampt)
        bad = ttnn.maximum(greedy, ttnn.typecast(ttnn.le(temp, sc["zero"]), f32))
        t_eff = ttnn.add(ttnn.multiply(temp, ttnn.rsub(bad, 1.0)), bad)
        inv = ttnn.reciprocal(t_eff)  # [nu, 1]
        inv_r = ttnn.reshape(ttnn.multiply(inv, sc["ones_t"]), (rows, 1))
        if rows < 32:
            inv_r = ttnn.pad(inv_r, [(0, 32 - rows), (0, 0)], value=1.0)
        inv_r = ttnn.reshape(inv_r, (1, 1, 32, 1))
        # logits - M, in fp32
        xp = ttnn.pad(x, [(0, 0), (0, 0), (0, 32 - rows), (0, 0)], value=-1e30) if rows < 32 else x
        x32 = ttnn.typecast(xp, f32)
        mg = ttnn.typecast(ttnn.max(uv, dim=-1, keepdim=True), f32)  # [1,1,32,1]
        d = ttnn.subtract(x32, mg)
        ttnn.deallocate(x32)
        dT = ttnn.multiply(d, inv_r)
        e_raw = ttnn.exp(d, fast_and_approximate_mode=False)
        ttnn.deallocate(d)
        e_T = ttnn.exp(dT, fast_and_approximate_mode=False)
        ttnn.deallocate(dT)
        st = ttnn.concat([e_raw, e_T], dim=2)  # [1,1,64,W]
        ttnn.deallocate(e_raw)
        ttnn.deallocate(e_T)
        s = ttnn.sum(st, dim=-1, keepdim=True, compute_kernel_config=self._vfy_stats_cc)  # [1,1,64,1]
        ttnn.deallocate(st)
        g = ttnn.all_gather(s, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,TP*64,1]
        ttnn.deallocate(s)
        g = ttnn.reshape(g, (1, self.num_devices, 64, 1))
        tot = ttnn.sum(g, dim=1, keepdim=True, compute_kernel_config=self._vfy_stats_cc)  # [1,1,64,1]
        ttnn.copy(tot, self._vfy_stats_out)
        ttnn.deallocate(tot)
        ttnn.deallocate(g)
        for t in (inv, inv_r, mg, temp, greedy, bad, t_eff):
            ttnn.deallocate(t)

    def prepare_verify_stats(self, samp_in):
        """Warm the stats trace bodies (B-row and every prepared row bucket) and allocate their persistent buffers, after
        prepare_verify_trace / prepare_verify_buckets and before ANY capture (capture forbids cache misses). ``samp_in`` = the
        engine's sampled-accept staging tensor (fp32 ROW_MAJOR [4+B, W]); the stats traces read its temperature / greedy rows
        and it must stay allocated at this address. Needs the verify topk trace (``_vfy_topk``). Idempotent. record_verify_trace /
        record_verify_bucket then capture the traces. Raises when the stats ops cannot be built (no silent downgrade).
        """
        if getattr(self, "_vfy_stats_ready", False):
            return
        assert getattr(self, "_vfy_prepared_key", None) is not None, "call prepare_verify_trace first"
        if not getattr(self, "_vfy_topk", False):
            raise ValueError("the stats trace needs the verify topk trace (QWEN36_VERIFY_DEVICE_TOPK=1)")
        dev = self.device
        rep = ttnn.ReplicateTensorToMesh(dev)
        B, T = self._vfy_B, self._vfy_T
        f32, TL = ttnn.float32, ttnn.TILE_LAYOUT
        samp_shape = tuple(int(d) for d in samp_in.shape)
        if len(samp_shape) != 2 or samp_shape[0] != 4 + B or samp_in.dtype != f32:
            raise ValueError(f"stats samp_in must be fp32 [4+{B}, W], got {samp_shape} {samp_in.dtype}")

        def dev_t(t, dt=f32):
            return ttnn.from_torch(
                t, dtype=dt, layout=TL, device=dev, mesh_mapper=rep, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        self._vfy_stats_samp = samp_in
        self._vfy_stats_cc = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,  # REQUIRED: bf16 accumulation of the 31040-term sums is off by 0.25 in lse (S0)
            packer_l1_acc=False,
        )
        self._vfy_stats_consts = {"zero": dev_t(torch.zeros(1, 1)), "ones_t": dev_t(torch.ones(1, T))}
        self._vfy_stats_out = dev_t(torch.zeros(1, 1, 64, 1))
        self._vfy_stats_standins = []
        specs = [(None, self._vfy_specs["shard_logits"])] + [
            (bk, bk.shard_spec)
            for _, bk in sorted((getattr(self, "_vfy_bk", None) or {}).items(), reverse=True)
            if bk.shard_spec is not None
        ]
        tv_shape, tv_dt, tv_layout, tv_mc = self._vfy_specs["topk_vals"]
        uv = ttnn.from_torch(
            torch.zeros(*tv_shape, dtype=torch.bfloat16),
            dtype=tv_dt,
            layout=tv_layout,
            device=dev,
            memory_config=tv_mc,
            mesh_mapper=rep,
        )
        try:
            for bk, (shape, dt, layout, mc) in specs:
                x = ttnn.from_torch(
                    torch.zeros(*shape, dtype=torch.bfloat16),
                    dtype=dt,
                    layout=layout,
                    device=dev,
                    memory_config=mc,
                    mesh_mapper=rep,
                )
                self._forward_verify_stats_body(bk, x=x, uv=uv)
                ttnn.deallocate(x)
            ttnn.synchronize_device(dev)
        finally:
            ttnn.deallocate(uv)
        self._vfy_stats_ready = True

    def _capture_stats_trace(self, bk=None):
        """Capture the stats trace of the B-row traces (bk None) or of row bucket ``bk``; returns the trace id. No-op (None)
        when prepare_verify_stats did not run."""
        if not getattr(self, "_vfy_stats_ready", False):
            return None
        tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._forward_verify_stats_body(bk)
        ttnn.end_trace_capture(self.device, tid, cq_id=0)
        return tid

    def prepare_verify_pen(self, pen):
        """Warm the pen bodies (phase 3 part 2, tt/spec_penalties.py): ``pen`` is the engine's PenaltyDevice (state, parameters and the
        update trace's result buffer bound). Runs the apply body once over a stand-in of the B-row shard logits and of every prepared
        row bucket's, the update body and the eager writes, so every program is compiled before ANY capture (capture forbids cache
        misses). record_verify_trace / record_verify_bucket then capture the pen traces. Needs the verify topk trace. Idempotent.
        Raises when the ops cannot be built (no silent downgrade)."""
        if getattr(self, "_vfy_pen_ready", False):
            return
        assert getattr(self, "_vfy_prepared_key", None) is not None, "call prepare_verify_trace first"
        if not getattr(self, "_vfy_topk", False):
            raise ValueError("the pen trace needs the verify topk trace (QWEN36_VERIFY_DEVICE_TOPK=1)")
        dev = self.device
        rep = ttnn.ReplicateTensorToMesh(dev)
        specs = [self._vfy_specs["shard_logits"]] + [
            bk.shard_spec
            for _, bk in sorted((getattr(self, "_vfy_bk", None) or {}).items(), reverse=True)
            if bk.shard_spec is not None
        ]
        xs = [
            ttnn.from_torch(
                torch.zeros(*shape, dtype=torch.bfloat16),
                dtype=dt,
                layout=layout,
                device=dev,
                memory_config=mc,
                mesh_mapper=rep,
            )
            for shape, dt, layout, mc in specs
        ]
        try:
            pen.warm(xs)
        finally:
            for x in xs:
                ttnn.deallocate(x)
        self._vfy_pen = pen
        self._vfy_pen_ready = True

    def _capture_pen_trace(self, bk=None):
        """Capture the pen trace of the B-row traces (bk None) or of row bucket ``bk``; returns the trace id. No-op (None) when
        prepare_verify_pen did not run."""
        if not getattr(self, "_vfy_pen_ready", False):
            return None
        tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._vfy_pen.apply_body(self._vfy_shard_logits_out if bk is None else bk.shard_logits)
        ttnn.end_trace_capture(self.device, tid, cq_id=0)
        return tid

    def release_verify_pen(self):
        """Free the pen state buffers (the traces themselves are released with their verify trace)."""
        pen = getattr(self, "_vfy_pen", None)
        if pen is not None:
            for t in pen.tensors():
                ttnn.deallocate(t)
        self._vfy_pen, self._vfy_pen_ready = None, False

    def zero_verify_stats(self):
        """Write zeros into the stats buffer in place (trace capture runs no op, so it would hold uninitialized memory until the
        first stats replay). No-op without a stats trace."""
        if not getattr(self, "_vfy_stats_ready", False):
            return
        h = ttnn.from_torch(
            torch.zeros(1, 1, 64, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        ttnn.copy_host_to_device_tensor(h, self._vfy_stats_out)

    def fetch_verify_stats(self, rows=None):
        """Synchronize and read the stats buffer of the LAST stats replay: (L_raw [rows], L_T [rows]) float32 torch, rows <= 32
        (default every row of the B-row shape). Tests and the audit; the serving path reads it with the step outputs."""
        assert getattr(self, "_vfy_stats_ready", False), "no stats trace"
        ttnn.synchronize_device(self.device)
        t = ttnn.to_torch(ttnn.get_device_tensors(self._vfy_stats_out)[0]).reshape(64).to(torch.float32)
        n = self._vfy_B * self._vfy_T if rows is None else int(rows)
        return t[:n].clone(), t[32 : 32 + n].clone()

    def release_verify_stats(self):
        """Free the stats trace buffers (the traces themselves are released with their verify trace)."""
        for t in (getattr(self, "_vfy_stats_out", None), *(getattr(self, "_vfy_stats_consts", None) or {}).values()):
            if t is not None:
                ttnn.deallocate(t)
        self._vfy_stats_out, self._vfy_stats_consts, self._vfy_stats_samp = None, None, None
        self._vfy_stats_ready = False

    def _forward_verify_mask_body(self, bk=None, x=None):
        """Fifth verify trace (QWEN36_SPEC_DEVICE_MASK, design note 80 section 3.2 and 81, 85): edit the shard logits IN PLACE
        with a packed token mask, between the main trace and the topk trace. ``_vfy_mask_in`` int32 ROW_MAJOR [1,32,970] per
        device (row r = engine row r; the host sends the PERMUTED words: bit k of word w of shard s is token s*31040 + k*970 + w).
        Chain (P0b "W", bitwise equal to the host ``x + where(allowed, 0, MASK_NEG)`` in bf16): slice the rows of the bucket,
        reshape [R,1,970], bitwise_right_shift by arange(32) [1,32,1] (broadcast), bitwise_and 1, reshape [1,1,R,31040], TILE,
        where(bit, 0, MASK_NEG), add_ into ``x``. MASK_NEG = -2**99 is exact in bf16 (a truncated scalar would differ from the host
        value); adding it to a bf16 logit gives it back, adding 0 keeps the logit. 0.3 ms for 32 rows (a captured trace: eager
        dispatch is 1.7 ms). Rows of an unconstrained mask row are all ones and stay unchanged.
        ``bk``: a row bucket (edits its ``shard_logits``); ``x``: an explicit tensor (warm-up stand-in)."""
        if x is None:
            x = self._vfy_shard_logits_out if bk is None else bk.shard_logits
        rows, width = int(x.shape[2]), int(x.shape[3])
        assert width == 32 * SHARD_WORDS and rows <= MASK_ROWS, (rows, width)
        m = self._vfy_mask_in
        cut = rows < MASK_ROWS
        if cut:
            m = ttnn.slice(m, [0, 0, 0], [1, rows, SHARD_WORDS])
        s = ttnn.bitwise_right_shift(ttnn.reshape(m, (rows, 1, SHARD_WORDS)), self._vfy_mask_ar)  # [R,32,970]
        if cut:
            ttnn.deallocate(m)
        b = ttnn.bitwise_and(s, 1)
        ttnn.deallocate(s)
        t = ttnn.to_layout(ttnn.reshape(b, (1, 1, rows, width)), ttnn.TILE_LAYOUT)
        ttnn.deallocate(b)
        add = ttnn.where(t, 0.0, MASK_NEG)
        ttnn.deallocate(t)
        ttnn.add_(x, add)
        ttnn.deallocate(add)

    def prepare_verify_mask(self):
        """Warm the mask trace bodies (B-row and every prepared row bucket) and allocate their persistent buffers, after
        prepare_verify_trace / prepare_verify_buckets and before ANY capture (capture forbids cache misses). Needs the verify topk
        trace. Idempotent. record_verify_trace / record_verify_bucket then capture the traces. Raises when the ops cannot be built
        (no silent downgrade)."""
        if getattr(self, "_vfy_mask_ready", False):
            return
        assert getattr(self, "_vfy_prepared_key", None) is not None, "call prepare_verify_trace first"
        if not getattr(self, "_vfy_topk", False):
            raise ValueError("the mask trace needs the verify topk trace (QWEN36_VERIFY_DEVICE_TOPK=1)")
        dev = self.device
        n = int(self.num_devices)
        if 32 * SHARD_WORDS * n != int(self.vocab_size):
            raise ValueError(
                f"the mask layout needs vocab {32 * SHARD_WORDS * n} = {n} x {SHARD_WORDS} words x 32 bits, got {self.vocab_size}"
            )
        rep = ttnn.ReplicateTensorToMesh(dev)
        RM, i32, DRAM = ttnn.ROW_MAJOR_LAYOUT, ttnn.int32, ttnn.DRAM_MEMORY_CONFIG
        self._vfy_mask_in = ttnn.from_torch(
            torch.full((n, MASK_ROWS, SHARD_WORDS), -1, dtype=torch.int32),
            dtype=i32,
            layout=RM,
            device=dev,
            mesh_mapper=ttnn.ShardTensorToMesh(dev, dim=0),
            memory_config=DRAM,
        )
        self._vfy_mask_ar = ttnn.from_torch(
            torch.arange(32, dtype=torch.int32).reshape(1, 32, 1),
            dtype=i32,
            layout=RM,
            device=dev,
            mesh_mapper=rep,
            memory_config=DRAM,
        )
        specs = [self._vfy_specs["shard_logits"]] + [
            bk.shard_spec
            for _, bk in sorted((getattr(self, "_vfy_bk", None) or {}).items(), reverse=True)
            if bk.shard_spec is not None
        ]
        for shape, dt, layout, mc in specs:
            x = ttnn.from_torch(
                torch.zeros(*shape, dtype=torch.bfloat16),
                dtype=dt,
                layout=layout,
                device=dev,
                memory_config=mc,
                mesh_mapper=rep,
            )
            self._forward_verify_mask_body(x=x)
            ttnn.deallocate(x)
        ttnn.synchronize_device(dev)
        self._vfy_mask_ready = True

    def _capture_mask_trace(self, bk=None):
        """Capture the mask trace of the B-row traces (bk None) or of row bucket ``bk``; returns the trace id. No-op (None) when
        prepare_verify_mask did not run."""
        if not getattr(self, "_vfy_mask_ready", False):
            return None
        tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._forward_verify_mask_body(bk)
        ttnn.end_trace_capture(self.device, tid, cq_id=0)
        return tid

    def reset_verify_mask(self):
        """Write all ones (no constraint) into the persistent mask input in place (trace capture runs no op, so it would hold
        uninitialized memory until the first mask write; a replay then would edit the logits at random). No-op without a mask
        trace."""
        if not getattr(self, "_vfy_mask_ready", False):
            return
        self.verify_stage_mask(torch.full((int(self.num_devices), MASK_ROWS, SHARD_WORDS), -1, dtype=torch.int32))

    def verify_stage_mask(self, mask):
        """H2D of the mask input: ``mask`` int32 [num_devices, 32, 970] contiguous (spec_masks.device_mask_input / the permuted
        staging ring), sharded on dim 0 (one [1,32,970] block per device; P0b: 0.03 ms host build, 0.04 ms call). Returns the ttnn
        host tensor: the caller keeps it (and ``mask``) alive until the step's event. Ordered in the queue after every enqueued
        trace, before the next one."""
        assert getattr(self, "_vfy_mask_ready", False), "no mask trace"
        host = ttnn.from_torch(
            mask, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0)
        )
        ttnn.copy_host_to_device_tensor(host, self._vfy_mask_in)
        return host

    def release_verify_mask(self):
        """Free the mask trace buffers (the traces themselves are released with their verify trace)."""
        for t in (getattr(self, "_vfy_mask_in", None), getattr(self, "_vfy_mask_ar", None)):
            if t is not None:
                ttnn.deallocate(t)
        self._vfy_mask_in = self._vfy_mask_ar = None
        self._vfy_mask_ready = False

    def _vfy_setup_device_topk(self):
        """Resolve QWEN36_VERIFY_DEVICE_TOPK (default ON) and QWEN36_VERIFY_TOPK_M (default 64) for this verify
        bucket: ``_vfy_topk`` (bool), ``_vfy_topk_m`` (M candidates per shard) and ``_vfy_topk_w`` (the
        power-of-two width the shard logits are padded to). Needs the sharded-argmax verify (the shard logits
        exist only there); M <= 64 is the multi-core ttnn.topk ceiling and keeps 16-bit indices."""
        self._vfy_topk = False
        self._vfy_topk_m = self._vfy_topk_w = None
        m = verify_device_topk_m()
        if m == 0:
            return
        if not self._vfy_sharded:
            logger.warning("QWEN36_VERIFY_DEVICE_TOPK ignored: the sharded-argmax verify is off")
            return
        per_shard = self.vocab_size // self.num_devices
        w = 1 << (per_shard - 1).bit_length()
        if not 1 <= m <= 64 or m > per_shard or w > 32768:
            raise ValueError(
                f"QWEN36_VERIFY_TOPK_M={m} unsupported: need 1 <= M <= 64 and M <= vocab/TP={per_shard}, "
                f"and a padded shard width {w} <= 32768 for the multi-core stable top-k"
            )
        self._vfy_topk, self._vfy_topk_m, self._vfy_topk_w = True, m, w

    def fetch_verify_logits(self):
        """Lazy full logits of the LAST verify replay: replays the logits trace over the persistent
        ``_vfy_normed_out`` the main trace wrote, synchronizes and reads the ROW_MAJOR copy -> torch
        [B*T, vocab] bf16 (what verify_traced(read_logits=True) returns). The topk read's fallback when a
        row cannot be served from its candidates. Valid until the next replay of any verify or seed body
        (the drafter does not touch ``_vfy_normed_out``)."""
        assert getattr(self, "_vfy_logits_trace_id", None) is not None, "no logits trace (sharded verify off)"
        dev = self.device
        rows = self._vfy_B * self._vfy_T
        bk = getattr(self, "_vfy_last_bucket", None)  # the last replay ran a row bucket: its own logits trace / rows
        if bk is not None:
            ttnn.execute_trace(dev, bk.logits_trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            lt = ttnn.to_torch(ttnn.get_device_tensors(bk.logits_rm_out)[0]).reshape(-1, self.vocab_size)[: bk.rows]
            return self._vfy_pad_rows(lt, rows)
        ttnn.execute_trace(dev, self._vfy_logits_trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        lt = ttnn.to_torch(ttnn.get_device_tensors(self._vfy_logits_rm_out)[0])
        return lt.reshape(-1, self.vocab_size)[:rows]

    @staticmethod
    def _vfy_pad_rows(t, rows):
        """Zero-pad a bucket's [b_rows, W] host readback to the ``rows`` rows the engine expects (the B-row shape)."""
        n = int(t.shape[0])
        if n >= rows:
            return t[:rows]
        return torch.cat([t, torch.zeros(rows - n, *t.shape[1:], dtype=t.dtype)], dim=0)

    def _vfy_setup_sharded_argmax(self, rows):
        """Resolve QWEN36_VERIFY_SHARDED_ARGMAX (default ON) for this verify bucket and allocate the one
        constant it needs (per-device vocab offset), outside any trace. Falls back to the full-gather argmax
        (flag off or unsupported mesh/vocab) with a log line.

        Tie-break: the combine returns the LOWEST global index among tied maxima (per-shard first-index
        argmax, then min index over the shards whose max equals the global max), which is torch.argmax on
        the full logits."""
        self._vfy_sharded = False
        old = getattr(self, "_vfy_dl", None)
        if old is not None:
            old.release()
            self._vfy_dl = None
        if not bool(int(os.environ.get("QWEN36_VERIFY_SHARDED_ARGMAX", "1"))):
            return
        why = self.device_decode_loop_unsupported_reason(0, 0, batch=rows)
        if why is not None and "RoPE" not in why:  # the RoPE range check is about the decode loop, not verify
            logger.warning(f"QWEN36_VERIFY_SHARDED_ARGMAX ignored: {why}")
            return
        self._vfy_dl = self._vfy_new_dl(rows)
        self._vfy_sharded = True

    def _vfy_new_dl(self, rows):
        """The sharded-argmax constants of a verify bucket of ``rows`` rows (per-device vocab offset), outside any trace."""
        mesh = self.mesh_device
        n = self.num_devices
        per_shard = self.vocab_size // n
        shard_base = ttnn.from_torch(
            (torch.arange(n, dtype=torch.int32) * per_shard).reshape(n, 1, 1, 1).expand(n, 1, 1, self._DL_LANES),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        return _DeviceDecodeLoop(batch=rows, shard_base=shard_base)

    def _vfy_row_positions(self, positions):
        """Per-ROW absolute positions for the user-major bucket: row u*T + j is positions[u] + j."""
        T = self._vfy_T
        if spec_hostset_enabled():  # QWEN36_SPEC_HOSTSET: one broadcast add instead of B*T Python ints
            pos = torch.as_tensor([int(p) for p in positions], dtype=torch.int64)
            return (pos[:, None] + torch.arange(T)).reshape(-1).to(torch.int32)
        return torch.tensor([int(positions[u]) + j for u in range(self._vfy_B) for j in range(T)], dtype=torch.int32)

    def _vfy_page_table_host(self, pt):
        """Host forms (torch int32, matching the device buffers) of the verify's
        persistent page-table buffers (_vfy_kvpt_buf, _vfy_kvpt_users, _vfy_kvpt1_buf), in that
        order. ``pt``: int32 [B, nb], one row per user.

        * per-row [B*T, nb]: row u*T+j is user u's block table (repeat_interleave, since the rows are
          user-major); read by the per-row KV write and the legacy per-row SDPA.
        * per-user [B, nb]: row u = user u's blocks, for the grouped aliased KV write. Group j holds
          rows {u*T + j}, one per user, so one persistent buffer serves all T groups
          (TPAttention._write_kv_aliased).
        * per-group [spec_sdpa_groups, nb]: one row per candidate group, for the fused spec SDPA
          (spec_multi_pos_tiles), which splits each user's T candidates across spec_sdpa_groups(T, B)/B
          batch rows and wants one aliased page-table row per group. The per-row KV write must not
          slice a narrower table full-span (a full-span ttnn.slice aliases its input), so both
          row-count forms are kept.
        """
        T, B, G = self._vfy_T, self._vfy_B, self._vfy_spec_groups
        pt = pt.to(torch.int32)
        if G >= B and G % B == 0:
            pt1 = pt.repeat_interleave(G // B, dim=0)
        else:
            pt1 = pt[:1].repeat(G, 1)  # fused path off: the buffer exists but is unread
        return pt.repeat_interleave(T, dim=0).contiguous(), pt.contiguous(), pt1.contiguous()

    def _vfy_check_kv_writes(self, pt, positions):
        """Assert the grouped verify KV write has no same-tile, same-call collision between users
        for block tables ``pt`` [B, nb] and first verify slots ``positions`` (spec_kv_write_conflicts)."""
        bad = spec_kv_write_conflicts(pt, positions, self._vfy_T, get_block_size(self._paged_kv_caches))
        assert not bad, (
            f"the grouped spec KV write needs distinct (block, 32-row tile) per user within each "
            f"candidate call, but {len(bad)} collide; first (call j, user, user, block, tile): "
            f"{bad[:4]}. Set QWEN36_SPEC_KV_GROUP_WRITE=0 to fall back to the per-row write."
        )

    def release_verify_trace(self):
        """Drop the captured verify trace. No-op when nothing is captured.

        The trace bakes in the paged KV caches' addresses and every GDN layer's _spec_ring /
        _verify_win_buf, all of which allocate_kv_caches + prepare_spec_verify re-allocate, so it
        must never outlive them: free_kv_caches calls this, and so does a re-capture.
        """
        if getattr(self, "_vfy_trace_id", None) is not None:
            ttnn.release_trace(self.mesh_device, self._vfy_trace_id)
            self._vfy_trace_id = None
        if getattr(self, "_vfy_logits_trace_id", None) is not None:
            ttnn.release_trace(self.mesh_device, self._vfy_logits_trace_id)
            self._vfy_logits_trace_id = None
        if getattr(self, "_vfy_topk_trace_id", None) is not None:
            ttnn.release_trace(self.mesh_device, self._vfy_topk_trace_id)
            self._vfy_topk_trace_id = None
        if getattr(self, "_vfy_stats_trace_id", None) is not None:
            ttnn.release_trace(self.mesh_device, self._vfy_stats_trace_id)
            self._vfy_stats_trace_id = None
        if getattr(self, "_vfy_mask_trace_id", None) is not None:
            ttnn.release_trace(self.mesh_device, self._vfy_mask_trace_id)
            self._vfy_mask_trace_id = None
        if getattr(self, "_vfy_pen_trace_id", None) is not None:
            ttnn.release_trace(self.mesh_device, self._vfy_pen_trace_id)
            self._vfy_pen_trace_id = None
        self.release_verify_buckets()
        self.release_verify_stats()
        self.release_verify_mask()
        self.release_verify_pen()

    def release_verify_buckets(self):
        """Release every row bucket's traces and bucket-local buffers (dl constants, stand-ins, trace outputs). No-op when
        there are none. release_verify_trace calls it, so shutdown and a re-prepare drop the buckets with the B-row trace.
        """
        bks = getattr(self, "_vfy_bk", None) or {}
        for b in sorted(bks):
            bk = bks[b]
            for name in (
                "trace_id",
                "logits_trace_id",
                "topk_trace_id",
                "stats_trace_id",
                "mask_trace_id",
                "pen_trace_id",
            ):
                if getattr(bk, name) is not None:
                    ttnn.release_trace(self.mesh_device, getattr(bk, name))
                    setattr(bk, name, None)
            for name in ("normed", "shard_logits", "logits_out", "logits_rm_out"):
                if getattr(bk, name) is not None:
                    ttnn.deallocate(getattr(bk, name))
                    setattr(bk, name, None)
            for t in bk.standins:
                ttnn.deallocate(t)
            bk.standins, bk.handoff = [], {}
            if bk.dl is not None:
                bk.dl.release()
                bk.dl = None
        self._vfy_bk = {}
        self._vfy_last_bucket = None

    @staticmethod
    def _fit_spec_page_table(pt, nb):
        """Host page table -> int32 contiguous [B, nb]: [w] is one row; zero-padded when w < nb (never
        touched while positions stay inside the real table), clipped when w > nb."""
        pt = pt if isinstance(pt, torch.Tensor) else torch.as_tensor(pt)
        pt = pt.reshape(1, -1) if pt.dim() == 1 else pt
        pt = pt.to(torch.int32)
        w = int(pt.shape[1])
        if w < nb:
            pt = torch.nn.functional.pad(pt, (0, nb - w))
        elif w > nb:
            pt = pt[:, :nb]
        return pt.contiguous()

    def capture_verify_trace(self, page_tables, T, warm_positions):
        """One-shot wrapper: release any trace, force a re-prepare, then prepare_verify_trace +
        record_verify_trace. Prefer preparing and recording once at startup."""
        self.release_verify_trace()
        self._vfy_prepared_key = None
        self.prepare_verify_trace(page_tables, T, warm_positions)
        self.record_verify_trace()

    def prepare_verify_trace(self, page_tables, T, warm_positions):
        """Allocate and compile everything the verify trace needs without capturing (warmup phase 1);
        record_verify_trace captures it (phase 2); verify_traced replays it.
        Bucket is a fixed B x T (T = len([pending] + drafts) per user, B = max_batch_size users).
        GDN state is carried (not reset): the trace advances a per-token ring in place and the host
        points the next replay at each user's accepted slot (mi) instead of copying anything back.
        ``page_tables``: torch int32 [B, num_blocks], one row per user, staged here; verify_traced
        keeps them unless it is handed new ``page_tables`` (continuous batching).
        ``warm_positions``: B absolute positions the two throwaway warmup/capture passes write KV at.
        Capture happens after prompt prefill (so those programs cannot clobber the parked trace), so
        point the writes past each user's prompt frontier; the first real verify overwrites them.
        Idempotent for the same (B, T, nb); a different key releases any live trace first.
        """
        from models.demos.blackhole.qwen36.tt.attention.tp import KV_GROUP_WRITE_OK
        from models.demos.blackhole.qwen36.tt.gdn.tp import spec_conv_sel, spec_ctrl_page, spec_state_blk_idx

        assert self.num_devices > 1, "traced verify is the TP path"
        assert self._paged_kv_caches is not None, "allocate_kv_caches first"
        pt = _page_table_rows(page_tables)
        B = int(pt.shape[0])
        nb = int(pt.shape[1])
        # B is the model's decode width, not a per-request choice: GDN's fused recurrent decode and
        # its batched state are both allocated at max_batch_size, and the verify shares them.
        assert B == self.args.max_batch_size, (
            f"verify runs the full decode width: page_tables has {B} rows, max_batch_size is "
            f"{self.args.max_batch_size}"
        )
        assert len(warm_positions) == B, f"warm_positions needs one position per user ({B})"
        rows = B * T
        # One decode tile carries every row of every user: the hard cap on B*(K+1) (nlp_create /
        # concat_heads_decode cap at 32 users, and every decode matmul config is a single tile row).
        assert rows <= ttnn.TILE_SIZE, f"verify bucket {B} users x T={T} = {rows} rows exceeds one tile"
        # The TILE_SIZE bucket runs the DECODE matmul config so every layer takes the DRAM-sharded
        # decode kernels and the weight load is amortized across the B*T tokens (a 32-row M-tile
        # costs the same as a 1-row one; the 128-row prefill bucket is compute-bound even at T=1).
        # Not bit-exact with a prefill-config verify (different matmul kernels round near-ties
        # differently), so spec need not reproduce target greedy exactly.
        # Hybrid verify skips chunked prefill SDPA (6 cores serially scanning KV; it int-divides
        # chunk_start by 32, so an unaligned anchor drops up to 31 recent tokens). Candidates go
        # through decode flash as pseudo-users: write all K/V, then one SDPA-decode with per-row
        # cur_pos so row u*T+j attends [0, p_u+j]. Causality is free and the KV scan spreads over
        # the full grid.
        key = (B, T, nb)
        if getattr(self, "_vfy_prepared_key", None) == key and getattr(self, "_vfy_token_buf", None) is not None:
            return
        dev = self.device
        rep = ttnn.ReplicateTensorToMesh(dev)

        # A re-prepare must not leave the old trace parked: its buffers are about to be replaced
        # (prepare_spec_verify reallocates on a shape change) and its id would leak.
        self.release_verify_trace()
        self._vfy_prepared_key = None
        self._vfy_T, self._vfy_B = T, B
        self._vfy_nb = nb
        self._vfy_setup_sharded_argmax(rows)
        self._vfy_setup_device_topk()
        # QWEN36_SPEC_HOSTSET (+ _TRACE): packed aux inputs + in-trace RoPE lookup (see _vfy_inputs). Decided here, so
        # the captured trace and every replay agree. The RoPE tables are allocated now, outside any trace.
        self._vfy_hostset = spec_hostset_trace_enabled()
        self._vfy_aux_staged = None
        if self._vfy_hostset:
            self.rope.decode_rm_tables()

        # Persistent per-replay input buffers (addresses baked into the trace).
        self._vfy_token_buf = ttnn.from_torch(
            torch.zeros(1, rows, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=rep,
        )
        # Positions are staged per replay; they are only allocated here, at the captured width, and
        # re-staged per replay by verify_traced (same convention as the prefill traces).
        # KV is written position-exact (paged_update_cache at absolute slots), not block-aligned.
        warm = self._vfy_row_positions(warm_positions)
        if self._vfy_hostset:
            self._vfy_kvpos_buf = None  # derived in the trace from _vfy_aux_buf
        else:
            self._vfy_kvpos_buf = ttnn.from_torch(
                warm, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, mesh_mapper=rep
            )
        # Grouped aliased KV write (TPAttention._write_kv_aliased): checked here for the warm passes and on
        # every replay (verify_traced), while the tables are still torch; see spec_kv_write_conflicts.
        self._vfy_pt_host = pt.clone()
        # QWEN36_VERIFY_STAGING_CACHE (default ON): skip the page-table H2D when unchanged, and take RoPE
        # cos/sin from a host table instead of recomputing them per replay.
        self._vfy_staging_cache = bool(int(os.environ.get("QWEN36_VERIFY_STAGING_CACHE", "1")))
        if self._vfy_staging_cache:
            self._rope_tp_cos_sin_decode_cached(warm)  # builds the table now, not inside the first replay
        self._vfy_kv_grouped = B > 1 and T > 1 and KV_GROUP_WRITE_OK
        if self._vfy_kv_grouped:
            self._vfy_check_kv_writes(pt, warm_positions)
        # The three persistent page-table buffers (see _vfy_page_table_host) are allocated before the
        # warmup so their addresses bake into the trace; verify_traced(page_tables=...) re-stages them
        # in place.
        _att0 = next(layer.attention for layer in self.layers if layer.is_full_attention)
        _spec_groups = int(_att0.spec_sdpa_groups(T, B))
        self._vfy_spec_groups = _spec_groups
        self._vfy_kvpt_buf, self._vfy_kvpt_users, self._vfy_kvpt1_buf = (
            ttnn.from_torch(h, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, mesh_mapper=rep)
            for h in self._vfy_page_table_host(pt)
        )
        # QWEN36_KV_WRITE_MASKED: constant 0/1 candidate masks for the masked KV write (see _vfy_pos_masked); persistent
        # (allocated here, outside any trace) for every row count n_users * T a verify body can run at.
        for _old in getattr(self, "_vfy_kvmask", {}).values():
            for _pair in _old:
                for _t in _pair:
                    ttnn.deallocate(_t)
        self._vfy_kvmask = {}
        if kv_write_masked_enabled() and self._vfy_kv_grouped:
            for _n in range(1, B + 1):
                _r = _n * T
                _ms = []
                for _j in range(T):
                    _m = torch.tensor([[1 if (i % T) == _j else 0 for i in range(_r)]], dtype=torch.int32)
                    _ms.append(
                        tuple(
                            ttnn.from_torch(h, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=rep)
                            for h in (_m, _m - 1)
                        )
                    )
                self._vfy_kvmask[_r] = _ms
        # RoPE tables. Hybrid verify feeds the DECODE attention kernel, whose cos/sin are
        # [1, rows, 1, rope_dim] (one rotation per ROW at that row's own position) — not the prefill
        # [1, 1, S, rope_dim] table. Same host math the plain decode step uses, so the two paths'
        # rotations are byte-identical and near-ties round the same way.
        if self._vfy_hostset:
            self._vfy_cos_buf = self._vfy_sin_buf = None  # looked up in the trace from the resident RoPE tables
        else:
            cos_t, sin_t = self._rope_tp_cos_sin_decode_torch(warm, wide=self._rope_perm)
            self._vfy_cos_buf = ttnn.from_torch(
                cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=rep
            )
            self._vfy_sin_buf = ttnn.from_torch(
                sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=rep
            )

        # The deferred commit's two selectors, shared by all GDN layers (pure index / one-hot data, so one
        # pair of buffers is staged per iteration instead of one per layer). state_blk_idx picks each
        # (user, v-head)'s starting block in the per-token state ring; conv_sel picks each user's conv
        # window rows out of [E_prev | new qkv]. Both start at mi = 0 for every user, as the seed leaves them.
        gdn = [layer.attention for layer in self.layers if not layer.is_full_attention]
        assert gdn, "the hybrid verify needs GDN layers"
        Nv, Kc = gdn[0].Nv, gdn[0].K
        self._vfy_Nv, self._vfy_Kc = Nv, Kc
        self._vfy_gdn = gdn
        _mi0 = [0] * B
        # QWEN36_VERIFY_SPEC_STEP=1: the fused GDN spec op reads ONE ctrl page {parity, mi, ring block | HOLD} instead
        # of the two selectors (allocated below, once the layers' window pairs exist and have a parity).
        self._vfy_fused = bool(getattr(gdn[0], "_spec_fused", False))
        assert all(
            bool(getattr(dn, "_spec_fused", False)) == self._vfy_fused for dn in gdn
        ), "GDN layers disagree on QWEN36_VERIFY_SPEC_STEP"
        if getattr(self, "_vfy_ctrl", None) is not None:  # a re-capture: drop the previous page (the trace is released)
            ttnn.deallocate(self._vfy_ctrl)
        self._vfy_ctrl = None
        if self._vfy_fused:
            self._vfy_state_idx = self._vfy_conv_sel = None
        else:
            self._vfy_state_idx = ttnn.from_torch(
                spec_state_blk_idx(_mi0, B, Nv),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                mesh_mapper=rep,
            )
            self._vfy_conv_sel = ttnn.from_torch(
                spec_conv_sel(_mi0, B, T, Kc),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=rep,
            )

        # Allocate each GDN layer's spec-verify buffers (per-token state ring + conv window) before
        # the warmup pass: the trace bakes their addresses in, and nothing may allocate later.
        for dn in gdn:
            dn.prepare_spec_verify(B, T)
        if self._vfy_hostset:
            from models.demos.blackhole.qwen36.tt.gdn.tp import spec_ctrl_words

            if getattr(self, "_vfy_aux_buf", None) is not None:  # a re-capture: drop the previous buffer
                ttnn.deallocate(self._vfy_aux_buf)
            self._vfy_ctrl_words = spec_ctrl_words(B, Nv)
            _aux = self._vfy_aux_host(warm, _mi0)
            self._vfy_aux_buf = ttnn.from_torch(
                _aux, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, mesh_mapper=rep
            )
            self._vfy_aux_staged = _aux.clone()
        elif self._vfy_fused:
            # The shared ctrl page, at the layers' CURRENT window parity (a re-capture at the same spec shape keeps
            # the pair and its parity; a fresh prepare_spec_verify resets it to 0). The page is data the op reads
            # at replay, so the parity it is captured with does not bind later replays -- verify_traced restages
            # it every iteration and flips every layer's parity afterwards, in lockstep.
            _pars = {dn._spec_win_par for dn in gdn}
            assert len(_pars) == 1, f"GDN window parities out of step: {sorted(_pars)}"
            self._vfy_ctrl = ttnn.from_torch(
                spec_ctrl_page(_mi0, B, Nv, parity=_pars.pop()),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                mesh_mapper=rep,
            )

        # Throwaway seed, only to fill the program cache. The spec loop calls seed_spec_state() again after
        # this capture (the warmup/capture passes below overwrite the ring and the window), and that call
        # must be a pure cache hit: a program compiled while a trace is parked lands its kernel binaries
        # in memory the replay writes over. Safe now: rec_state and the conv window are valid and the
        # verify writes neither, so this only fills the spec buffers with contents the real seed overwrites.
        for dn in gdn:
            dn.seed_spec_state()
        ttnn.synchronize_device(dev)

        # Warmup outside the trace compiles every program (a compile during replay clobbers the trace);
        # the capture pass runs in record_verify_trace. Both passes overwrite the ring and the conv window;
        # the spec loop calls dn.seed_spec_state() after capture, which re-initialises token-slot 0 from
        # the live decode state, and the first replay then runs at mi = 0.
        outs = self._forward_verify_bucket_tp(n_users=B)
        # Shape / dtype / layout / memory config of the shared B-row outputs, for the row buckets' warm stand-ins.
        self._vfy_specs = {"ids": self._tensor_spec(outs[2]), "rows": self._tensor_spec(outs[1])}
        if self._vfy_sharded:
            # Compile the logits trace's programs too (cache misses are disallowed during capture).
            self._vfy_normed_out = outs[4]
            _lg = self._forward_verify_logits_body()
            self._vfy_warm_narrow(_lg[1])
            for t in _lg:
                ttnn.deallocate(t)
            ttnn.deallocate(self._vfy_normed_out)
            self._vfy_normed_out = None
            if self._vfy_topk:
                # ... and the topk trace's (pad, stable top-k, typecast, two all-gathers) on the kept shard logits.
                self._vfy_shard_logits_out = outs[5]
                _tk = self._forward_verify_topk_body()
                self._vfy_specs["topk_vals"], self._vfy_specs["topk_ids"] = (self._tensor_spec(t) for t in _tk)
                self._vfy_specs["shard_logits"] = self._tensor_spec(self._vfy_shard_logits_out)
                for t in _tk:
                    ttnn.deallocate(t)
                ttnn.deallocate(self._vfy_shard_logits_out)
                self._vfy_shard_logits_out = None
        if not self._vfy_sharded:
            self._vfy_warm_narrow(outs[3])
        for t in outs[:4]:
            if t is not None:
                ttnn.deallocate(t)
        ttnn.synchronize_device(dev)
        self._vfy_att0 = _att0
        self._vfy_prepared_key = key

    def record_verify_trace(self):
        """Capture the verify trace over the buffers prepare_verify_trace allocated (warmup phase 2).
        Idempotent: a live trace is kept."""
        assert getattr(self, "_vfy_prepared_key", None) is not None, "call prepare_verify_trace first"
        if getattr(self, "_vfy_trace_id", None) is not None:
            return
        dev = self.device
        B, T = self._vfy_B, self._vfy_T
        rows = B * T
        _att0 = self._vfy_att0
        _spec_groups = self._vfy_spec_groups
        _kv_grouped = self._vfy_kv_grouped

        self._vfy_trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
        (
            self._vfy_logits_out,
            self._vfy_rows_out,
            self._vfy_ids_out,
            self._vfy_logits_rm_out,
            self._vfy_normed_out,
            self._vfy_shard_logits_out,
        ) = self._forward_verify_bucket_tp(n_users=B)
        ttnn.end_trace_capture(dev, self._vfy_trace_id, cq_id=0)
        if self._vfy_sharded:
            # Sampled requests (read_logits) need the full logits, which the main trace no longer gathers:
            # a second, small trace (lm_head + all-gather + untilize on the persistent normed rows).
            self._vfy_logits_trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
            self._vfy_logits_out, self._vfy_logits_rm_out = self._forward_verify_logits_body()
            ttnn.end_trace_capture(dev, self._vfy_logits_trace_id, cq_id=0)
            if self._vfy_topk:
                # QWEN36_VERIFY_DEVICE_TOPK: a third, smaller trace replaces the logits trace on sampled steps
                # whose rows can be served from the exact top-M candidates (read_topk; the logits trace stays
                # the fallback, replayed lazily by fetch_verify_logits).
                self._vfy_topk_trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
                self._vfy_topk_vals_out, self._vfy_topk_ids_out = self._forward_verify_topk_body()
                ttnn.end_trace_capture(dev, self._vfy_topk_trace_id, cq_id=0)
                # prepare_verify_stats ran (QWEN36_SPEC_DEVICE_LOGPROBS / _TOPP): a fourth, small trace (fp32 sums over the vocabulary)
                self._vfy_stats_trace_id = self._capture_stats_trace()
                # prepare_verify_mask ran (QWEN36_SPEC_DEVICE_MASK): a fifth trace that edits the shard logits in place (replayed
                # between the main and the topk trace of a masked step only)
                self._vfy_mask_trace_id = self._capture_mask_trace()
                # prepare_verify_pen ran (QWEN36_SPEC_DEVICE_PENALTIES): a sixth, small trace edits the shard logits in place
                self._vfy_pen_trace_id = self._capture_pen_trace()
        self._vfy_capture_count = getattr(self, "_vfy_capture_count", 0) + 1
        if getattr(self, "_vfy_bk", None):
            self._log_trace_memory(f"V{B}")
        # Fused grouped SDPA read (spec_multi_pos_tiles) or the legacy per-row call; a T with no
        # L1-fitting split (7, 11, ...) stays legacy.
        _fused = _att0.spec_sdpa_enabled(T)
        _how = (
            f"FUSED spec_multi_pos_tiles ({_spec_groups} groups x Tg={rows // _spec_groups})"
            if _fused
            else f"legacy {rows} pseudo-users"
        )
        _kvw = (
            f"GROUPED ({T} calls x {B} rows per K/V per layer)"
            if self._vfy_kv_grouped
            else f"per-row ({rows} calls per K/V per layer)"
        )
        logger.info(
            f"Verify trace (B={B} users x T={T} = {rows} rows) captured "
            f"successfully! verify SDPA: {_how}; verify KV write: {_kvw}"
        )

    @staticmethod
    def _tensor_spec(t):
        return (tuple(t.shape), t.dtype, t.layout, t.memory_config())

    def _log_trace_memory(self, label):
        """Log (and keep in ``_vfy_mem_log``) the trace-region and DRAM bytes in use after the capture ``label``:
        ttnn.get_memory_view(mesh, TRACE / DRAM).total_bytes_allocated_per_bank x num_banks. Never raises."""
        try:
            tv = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.TRACE)
            dv = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM)
            trace_b = int(tv.total_bytes_allocated_per_bank) * int(tv.num_banks)
            trace_cap = int(tv.total_bytes_per_bank) * int(tv.num_banks)
            dram_b = int(dv.total_bytes_allocated_per_bank) * int(dv.num_banks)
        except Exception as e:  # noqa: BLE001 - a log line only
            logger.warning(f"[spec_bucket] trace memory view unavailable after {label}: {e}")
            return None
        log = self.__dict__.setdefault("_vfy_mem_log", {})
        prev = list(log.values())[-1] if log else (0, 0)
        log[label] = (trace_b, dram_b)
        logger.info(
            f"[spec_bucket] memory after {label}: trace {trace_b / 2**20:.2f} MiB of {trace_cap / 2**20:.0f} MiB "
            f"(+{(trace_b - prev[0]) / 2**20:.2f}), DRAM allocated {dram_b / 2**20:.1f} MiB (+{(dram_b - prev[1]) / 2**20:.1f}) per device-sum"
        )
        return log[label]

    def prepare_verify_buckets(self, buckets):
        """Warm every row bucket b < B in ``buckets`` (descending), after prepare_verify_trace and before ANY capture."""
        for b in sorted((int(x) for x in buckets if int(x) < self._vfy_B), reverse=True):
            self.prepare_verify_bucket(b)

    def record_verify_buckets(self, buckets):
        """Capture every warmed bucket in ``buckets``, descending (V4, V2, V1), after record_verify_trace (V8)."""
        for b in sorted((int(x) for x in buckets if int(x) < self._vfy_B), reverse=True):
            self.record_verify_bucket(b)

    def prepare_verify_bucket(self, b):
        """Warm the b-row verify bucket (b < B) on the shared B-row state: allocate its persistent buffers (sharded-argmax
        constants, hand-off stand-ins shaped like the live B-row outputs) and run its main / logits / topk bodies once so
        every program is compiled before any capture (capture_decode forbids cache misses). Idempotent. The spec loop
        seeds the GDN state after warmup, as for the B-row trace."""
        assert getattr(self, "_vfy_prepared_key", None) is not None, "call prepare_verify_trace first"
        B, T = self._vfy_B, self._vfy_T
        assert 1 <= b < B, f"row bucket {b} must be in [1, {B})"
        assert self._vfy_sharded, "row buckets need the sharded verify (QWEN36_VERIFY_SHARDED_ARGMAX)"
        if self._vfy_bk is None:
            self._vfy_bk = {}
        if b in self._vfy_bk:
            return
        dev = self.device
        rep = ttnn.ReplicateTensorToMesh(dev)
        bk = _VfyBucket(b, b * T)
        self._vfy_bk[b] = bk  # registered first: release_verify_buckets frees a half-built bucket
        bk.dl = self._vfy_new_dl(bk.rows)
        torch_dt = {ttnn.bfloat16: torch.bfloat16, ttnn.uint32: torch.int32}
        for name, (shape, dt, layout, mc) in self._vfy_specs.items():
            if name == "shard_logits":  # the stats warm stand-in spec, not a hand-off target
                continue
            bk.handoff[name] = ttnn.from_torch(
                torch.zeros(*shape, dtype=torch_dt[dt]),
                dtype=dt,
                layout=layout,
                device=dev,
                memory_config=mc,
                mesh_mapper=rep,
            )
        bk.standins = list(bk.handoff.values())
        outs = self._forward_verify_bucket_tp(n_users=b, bk=bk)
        bk.normed = outs[4]
        _lg = self._forward_verify_logits_body(bk)
        bk.narrow_ready = self._vfy_warm_narrow(_lg[1], b)
        for t in _lg:
            ttnn.deallocate(t)
        ttnn.deallocate(bk.normed)
        bk.normed = None
        if self._vfy_topk:
            bk.shard_logits = outs[5]
            bk.shard_spec = self._tensor_spec(outs[5])
            self._forward_verify_topk_body(bk)
            ttnn.deallocate(bk.shard_logits)
            bk.shard_logits = None
        ttnn.synchronize_device(dev)

    def record_verify_bucket(self, b):
        """Capture the b-row verify bucket's traces (main, logits, topk) after record_verify_trace: the B-row outputs it
        hands over to must exist, and they only overlap the B-row trace's own intermediates when captured right after it.
        Capture order: V8, then V4, V2, V1 (record_verify_buckets), each main + logits + topk, all before the drafter and
        accept captures: the bucket-local outputs (normed / shard logits / logits) are read lazily by fetch_verify_logits
        after those traces replayed, so nothing captured later may reuse their memory. Logs the trace / DRAM memory."""
        bk = (getattr(self, "_vfy_bk", None) or {}).get(b)
        assert bk is not None, f"call prepare_verify_bucket({b}) first"
        assert (
            getattr(self, "_vfy_trace_id", None) is not None
        ), "capture the B-row verify trace (record_verify_trace) first"
        if bk.trace_id is not None:
            return
        dev = self.device
        bk.handoff = {
            "ids": self._vfy_ids_out,
            "rows": self._vfy_rows_out,
            "topk_vals": getattr(self, "_vfy_topk_vals_out", None),
            "topk_ids": getattr(self, "_vfy_topk_ids_out", None),
        }
        bk.trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
        outs = self._forward_verify_bucket_tp(n_users=b, bk=bk)
        ttnn.end_trace_capture(dev, bk.trace_id, cq_id=0)
        bk.normed, bk.shard_logits = outs[4], outs[5]
        bk.logits_trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
        bk.logits_out, bk.logits_rm_out = self._forward_verify_logits_body(bk)
        ttnn.end_trace_capture(dev, bk.logits_trace_id, cq_id=0)
        if self._vfy_topk:
            bk.topk_trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
            self._forward_verify_topk_body(bk)
            ttnn.end_trace_capture(dev, bk.topk_trace_id, cq_id=0)
            bk.stats_trace_id = self._capture_stats_trace(bk)
            bk.mask_trace_id = self._capture_mask_trace(bk)
            bk.pen_trace_id = self._capture_pen_trace(bk)
        for t in bk.standins:  # warm-pass hand-off targets, used by no trace
            ttnn.deallocate(t)
        bk.standins = []
        logger.info(f"Verify bucket trace (b={b} users x T={self._vfy_T} = {bk.rows} rows) captured")
        self._log_trace_memory(f"V{b}")

    def verify_traced(
        self,
        tokens,
        positions,
        mi_prev,
        read_logits=False,
        page_tables=None,
        inactive=None,
        read_topk=False,
        read_col0=False,
        bucket=None,
    ):
        """Stage, replay, sync and read one verify step: ``verify_stage_and_enqueue`` then ``verify_sync_and_read``.

        Replay the captured verify trace for B users x T candidates.

        ``tokens``: B lists of T ids ([pending] + drafts per user). ``positions``: B ints, user u's
        first verify slot this iteration (row u*T+j lands at positions[u]+j). ``mi_prev``: B ints,
        each user's accepted-prefix index from the previous iteration (0 right after the seed). It
        is the whole commit: it selects user u's starting recurrent state (ring block) and conv
        window rows, so no state is copied between iterations.

        Returns (ids: B*T host ints, feed_rows: the trace's persistent [1,1,B*T,dim/tp] drafter feed,
        readback: None, logits torch [B*T, vocab] bf16, or the top-k pair). ``read_logits`` pulls the full
        logits to host (for the sampler; greedy needs only the in-trace argmax ids). It reads the trace's
        ROW_MAJOR copy (_vfy_logits_rm_out) with no cast, which avoids a host untilize; the sampler converts
        to float32 lazily per row. ``read_topk`` (QWEN36_VERIFY_DEVICE_TOPK) replays the topk trace instead
        and returns ``(values [B*T, TP*M] bf16, local_ids [B*T, TP*M] int)`` from _forward_verify_topk_body
        (spec_engine.merge_shard_topk turns them into the exact global top-M per row); the full logits
        stay available lazily through fetch_verify_logits. The two reads are exclusive.

        ``page_tables``: int32 [B, num_blocks] block tables (array-like; 1-D only for B == 1), one
        row per user. When given, every persistent page-table buffer is re-staged for this replay,
        with columns clipped or padded with block 0 (vLLM's null block) to the captured width.
        None keeps the tables staged last. A grouped KV write that would put two users on the same
        (block, 32-row tile) in the same call (block 0 excepted) raises AssertionError before anything is staged.
        The MTP drafter's own table (``SpeculativeDecoder.mtp_pt``) is a separate buffer this argument does
        not touch. ``inactive``: length-B bools; those users are skipped by the KV-write collision check only
        (their rows write the null block).

        ``read_col0`` (with ``read_logits``; QWEN36_SPEC_HOSTSET): read only column 0 of each user, as ``[B, vocab]``
        (``self._vfy_last_col0`` is True then); the full ``[B*T, vocab]`` read is the fallback (flag off, T == 1, or the
        slice did not compile).

        ``feed_rows`` is the trace's own output buffer: do not deallocate it, and finish with it
        before the next replay (the spec loop reads its anchor rows and reseeds the drafter within
        the iteration).
        """
        handle = self.verify_stage_and_enqueue(
            tokens,
            positions,
            mi_prev,
            read_logits=read_logits,
            page_tables=page_tables,
            inactive=inactive,
            read_topk=read_topk,
            read_col0=read_col0,
            bucket=bucket,
        )
        return self.verify_sync_and_read(handle)

    def verify_stage_and_enqueue(
        self,
        tokens,
        positions,
        mi_prev,
        read_logits=False,
        page_tables=None,
        inactive=None,
        read_topk=False,
        read_col0=False,
        execute_topk=False,
        execute_stats=False,
        execute_pen=False,
        skip_tokens_h2d=False,
        skip_aux_h2d=False,
        skip_kv_check=False,
        flip_parity=False,
        bucket=None,
        execute_mask=False,
        mask_host=None,
        forward_only=False,
    ):
        """First half of ``verify_traced`` (same arguments): validate, stage every per-replay input, and enqueue the
        main trace plus the optional logits / topk traces with ``blocking=False``. No synchronize and no readback.
        ``execute_topk`` (QWEN36_SPEC_FUSED stage S2): enqueue the topk trace but do not read it back (the handle says
        read_topk=False); the fused sampled accept trace consumes ``_vfy_topk_vals_out`` / ``_vfy_topk_ids_out`` on the device.
        ``execute_stats`` (phase 2, design note 67): also enqueue the stats trace right after the topk trace (needs a topk
        replay in the same call and prepare_verify_stats before capture); results stay on the device (fetch_verify_stats).
        ``execute_pen`` (phase 3 part 2, tt/spec_penalties.py): enqueue the pen trace right after the main trace, before the topk
        trace, so the topk / stats / accept traces see the penalized shard logits (needs a topk replay in the same call and
        prepare_verify_pen before capture). The caller replays it only when a live row has a non-neutral penalty.
        ``skip_tokens_h2d`` / ``skip_aux_h2d`` (QWEN36_SPEC_CHAIN, stage S3c): the chain trace already wrote the token buffer / the
        packed aux buffer on the device, so the host copy is not made (the arguments still feed the page-table and KV-write checks);
        skipping the aux copy forgets the host mirror ``_vfy_aux_staged`` (it no longer describes the device words).
        ``skip_kv_check`` (deferred chain step, S4b): the positions are only upper bounds, so the per-step same-tile collision check is
        replaced by the caller's invariant (no non-null block shared across active rows). ``flip_parity`` (deferred serving):
        flip the GDN window parity HERE, once per replay, because no ``verify_after_external_sync`` follows (the next step's overlay
        must know its parity word at submit time); the returned handle then has ``fused=False`` so a later after-sync call cannot flip again.

        ``bucket`` (QWEN36_SPEC_BUCKETS): the row bucket b whose traces to replay (None or B = the B-row traces). Everything staged
        is the same B-row host state; the b-row traces use rows 0..b-1 of it and hand their outputs to the shared B-row buffers. The
        caller keeps every live row below b (the engine parks the others).

        ``execute_mask`` / ``mask_host`` (phase 3, QWEN36_SPEC_DEVICE_MASK, design note 85): write the packed token mask
        ``mask_host`` (int32 [num_devices, 32, 970], the permuted words) into the mask input and replay the mask trace between the
        main trace and the topk trace (order: main, mask, topk, stats). Needs a topk replay (or ``forward_only``) and a captured mask
        trace. ``forward_only`` (the split step, ``verify_enqueue_post``): stage everything and enqueue the MAIN trace only (no
        logits / mask / topk / stats trace); the returned handle has ``post_pending`` True and ``verify_enqueue_post`` enqueues the
        rest. The GDN parity flip (``flip_parity``) is done here either way.

        Returns a ``VerifyHandle`` for ``verify_sync_and_read`` (blocking read) or ``verify_after_external_sync``
        (the caller already synchronized the device). Exactly one of those must follow each call: the GDN window
        parity flips there, once per replay.
        """
        from models.demos.blackhole.qwen36.tt.gdn.tp import spec_conv_sel, spec_ctrl_page, spec_state_blk_idx

        assert getattr(self, "_vfy_trace_id", None) is not None, "call record_verify_trace first"
        bk = None
        if bucket is not None and int(bucket) != self._vfy_B:
            bk = (getattr(self, "_vfy_bk", None) or {}).get(int(bucket))
            assert bk is not None and bk.trace_id is not None, f"verify bucket {bucket} is not captured"
        _main_tid = self._vfy_trace_id if bk is None else bk.trace_id
        _logits_tid = getattr(self, "_vfy_logits_trace_id", None) if bk is None else bk.logits_trace_id
        _topk_tid = getattr(self, "_vfy_topk_trace_id", None) if bk is None else bk.topk_trace_id
        if read_topk:
            if read_logits:
                raise ValueError("verify_traced: read_logits and read_topk are exclusive reads")
        if read_topk or execute_topk:
            if _topk_tid is None:
                raise ValueError("verify_traced(read_topk=True) needs QWEN36_VERIFY_DEVICE_TOPK=1 at capture")
        _stats_tid = getattr(self, "_vfy_stats_trace_id", None) if bk is None else bk.stats_trace_id
        if execute_stats:
            if not (read_topk or execute_topk):
                raise ValueError("execute_stats needs a topk replay in the same step (the stats read its union)")
            if _stats_tid is None:
                raise ValueError("execute_stats needs a captured stats trace (prepare_verify_stats before capture)")
        self._vfy_check_post(bk, read_topk or execute_topk, execute_mask, mask_host, forward_only)
        _pen_tid = getattr(self, "_vfy_pen_trace_id", None) if bk is None else bk.pen_trace_id
        self._vfy_check_pen(_pen_tid, read_topk or execute_topk or forward_only, execute_pen)
        T, B = self._vfy_T, self._vfy_B
        rows = B * T
        assert len(tokens) == B, f"expected {B} users, got {len(tokens)}"
        assert all(len(t) == T for t in tokens), f"expected {T} tokens per user"
        assert len(positions) == B and len(mi_prev) == B, "one position and one mi per user"
        dev = self.device
        rep = ttnn.ReplicateTensorToMesh(dev)

        # Page tables: normalise to the captured width (clip, or pad with the null block 0) and check
        # the grouped KV write before staging anything, so a refused replay leaves the device alone.
        pt = self._vfy_pt_host
        if page_tables is not None:
            pt = torch.as_tensor(page_tables)
            assert pt.dim() == 2 or (pt.dim() == 1 and B == 1), "page_tables must be [B, nb] (1-D only for B == 1)"
            pt = _page_table_rows(pt)
            assert pt.shape[0] == B, f"page_tables has {pt.shape[0]} rows, expected {B} users"
            W = int(self._vfy_pt_host.shape[1])
            pt = pt[:, :W]
            if pt.shape[1] < W:
                pt = torch.cat([pt, torch.zeros(B, W - pt.shape[1], dtype=torch.int32)], dim=1)
            pt = pt.contiguous()
        if self._vfy_kv_grouped and not skip_kv_check:
            chk = positions if inactive is None else [-1 if x else int(p) for p, x in zip(positions, inactive)]
            self._vfy_check_kv_writes(pt, chk)
        if page_tables is not None:
            # _vfy_pt_host mirrors what the three device buffers hold (set at prepare and here), so an equal
            # table needs no copy. Stored as a clone: the caller may mutate its tensor in place.
            if not (self._vfy_staging_cache and self._vfy_pt_unchanged(self._vfy_pt_host, pt)):
                bufs = (self._vfy_kvpt_buf, self._vfy_kvpt_users, self._vfy_kvpt1_buf)
                for host, buf in zip(self._vfy_page_table_host(pt), bufs):
                    _h = ttnn.from_torch(
                        host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep
                    )
                    ttnn.copy_host_to_device_tensor(_h, buf)
                self._vfy_pt_host = pt.clone()

        # Stage per-replay inputs into the persistent buffers (addresses preserved). Row-major flatten
        # of the B x T token lists is the user-major row order.
        if not skip_tokens_h2d:
            tok = torch.tensor([int(t) for row in tokens for t in row], dtype=torch.int32).reshape(1, rows)
            _h = ttnn.from_torch(tok, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep)
            ttnn.copy_host_to_device_tensor(_h, self._vfy_token_buf)
        # Absolute cache slot for each row: positions[u] + j. This doubles as the decode SDPA's
        # per-row cur_pos, which is what makes the hybrid verify causal.
        pos = self._vfy_row_positions(positions)
        _hostset = bool(getattr(self, "_vfy_hostset", False))
        if not _hostset:
            _h = ttnn.from_torch(pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep)
            ttnn.copy_host_to_device_tensor(_h, self._vfy_kvpos_buf)
            if self._vfy_staging_cache:
                cos_t, sin_t = self._rope_tp_cos_sin_decode_cached(
                    pos, self._spec_pos_deltas(len(pos)), wide=self._rope_perm
                )
            else:
                cos_t, sin_t = self._rope_tp_cos_sin_decode_torch(
                    pos, self._spec_pos_deltas(len(pos)), wide=self._rope_perm
                )
            _h = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep)
            ttnn.copy_host_to_device_tensor(_h, self._vfy_cos_buf)
            _h = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep)
            ttnn.copy_host_to_device_tensor(_h, self._vfy_sin_buf)
        # The deferred commit (see prepare_verify_trace): where each user's recurrence and conv
        # window resume. Two small host-built tensors, shared by every GDN layer.
        _fused = bool(getattr(self, "_vfy_fused", False))
        if _fused:
            # QWEN36_VERIFY_SPEC_STEP=1: ONE ctrl page {parity, mi, ring block} for all GDN layers, at their current
            # window parity (the half holding E_prev; the op writes the other half). No HOLD: a row left out of a
            # step is parked on host by the engine (MTPSpecEngine.verify), and padding rows run at mi = 0 over the
            # null block like the composite path's.
            assert all(0 <= int(mi_prev[u]) < T for u in range(B)), f"mi_prev {list(mi_prev)} out of range [0,{T})"
            _pars = {dn._spec_win_par for dn in self._vfy_gdn}
            assert len(_pars) == 1, f"GDN window parities out of step: {sorted(_pars)}"
            if not _hostset:
                _h = ttnn.from_torch(
                    spec_ctrl_page(mi_prev, B, self._vfy_Nv, parity=_pars.pop()),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=None,
                    mesh_mapper=rep,
                )
                ttnn.copy_host_to_device_tensor(_h, self._vfy_ctrl)
        else:
            _h = ttnn.from_torch(
                spec_state_blk_idx(mi_prev, B, self._vfy_Nv),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
                mesh_mapper=rep,
            )
            ttnn.copy_host_to_device_tensor(_h, self._vfy_state_idx)
            _h = ttnn.from_torch(
                spec_conv_sel(mi_prev, B, T, self._vfy_Kc),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=None,
                mesh_mapper=rep,
            )
            ttnn.copy_host_to_device_tensor(_h, self._vfy_conv_sel)

        if _hostset:
            # kvpos + RoPE rows + ctrl page: one packed copy, skipped when the next-verify prestage already staged these
            # exact words (prestage_verify_aux); the trace slices them and looks cos/sin up on the device.
            if skip_aux_h2d:
                self._vfy_aux_staged = None
            else:
                self._vfy_stage_aux(self._vfy_aux_host(pos, mi_prev))
        ttnn.execute_trace(dev, _main_tid, cq_id=0, blocking=False)
        self._vfy_last_bucket = bk
        mask_keep = None
        if not forward_only:
            if read_logits and self._vfy_sharded:
                ttnn.execute_trace(dev, _logits_tid, cq_id=0, blocking=False)
            # Trace order (phase 3 merge): main, [logits], mask, pen, topk, stats. Mask and penalties edit the same shard logits
            # in place and commute (a masked -inf stays -inf under every penalty step), the topk trace reads the result.
            if execute_mask:  # phase 3: the mask edits the shard logits the topk trace reads
                mask_keep = self.verify_stage_mask(mask_host)
                ttnn.execute_trace(
                    dev, bk.mask_trace_id if bk is not None else self._vfy_mask_trace_id, cq_id=0, blocking=False
                )
            if execute_pen:  # phase 3 part 2: the penalties edit the same shard logits
                ttnn.execute_trace(dev, _pen_tid, cq_id=0, blocking=False)
            if read_topk or execute_topk:
                ttnn.execute_trace(dev, _topk_tid, cq_id=0, blocking=False)
            if execute_stats:
                ttnn.execute_trace(dev, _stats_tid, cq_id=0, blocking=False)
        if flip_parity and _fused:
            for dn in self._vfy_gdn:
                dn._spec_win_par ^= 1
        handle = VerifyHandle(
            rows=rows,
            B=B,
            read_logits=read_logits,
            read_topk=read_topk,
            read_col0=read_col0,
            fused=_fused and not flip_parity,
            bucket=None if bk is None else bk.b,
            post_pending=bool(forward_only),
        )
        handle.mask_keep = mask_keep
        return handle

    def _vfy_check_post(self, bk, topk, execute_mask, mask_host, forward_only):
        """Validation of the mask arguments of ``verify_stage_and_enqueue`` / ``verify_enqueue_post`` (before anything is staged)."""
        if not execute_mask:
            return
        tid = getattr(self, "_vfy_mask_trace_id", None) if bk is None else bk.mask_trace_id
        if tid is None:
            raise ValueError("execute_mask needs a captured mask trace (prepare_verify_mask before capture)")
        if mask_host is None:
            raise ValueError("execute_mask needs mask_host (int32 [num_devices, 32, 970])")
        if not (topk or forward_only):
            raise ValueError(
                "execute_mask needs a topk replay in the same step (the mask edits the logits the topk trace reads)"
            )

    def _vfy_check_pen(self, pen_tid, topk, execute_pen):
        """Validation of ``execute_pen`` of ``verify_stage_and_enqueue`` / ``verify_enqueue_post`` (before anything is staged)."""
        if not execute_pen:
            return
        if not topk:
            raise ValueError(
                "execute_pen needs a topk replay in the same step (the topk trace ranks the penalized logits)"
            )
        if pen_tid is None:
            raise ValueError("execute_pen needs a captured pen trace (prepare_verify_pen before capture)")

    def verify_enqueue_post(
        self, handle, execute_topk=True, execute_stats=False, execute_mask=False, mask_host=None, execute_pen=False
    ):
        """Second half of a ``forward_only`` enqueue: enqueue [mask write + mask trace], the topk trace and the optional stats
        trace behind the main trace of ``handle`` (same bucket, same order as the one-call path: main, mask, pen, topk, stats; ``execute_pen`` as in ``verify_stage_and_enqueue``). Clears
        ``handle.post_pending``; ``handle.mask_keep`` holds the host tensor of the mask write until the step's event. No synchronize,
        no readback."""
        if not handle.post_pending:
            raise ValueError("verify_enqueue_post: the handle has no pending second half")
        bk = None if handle.bucket is None else self._vfy_bk[handle.bucket]
        _topk_tid = getattr(self, "_vfy_topk_trace_id", None) if bk is None else bk.topk_trace_id
        _stats_tid = getattr(self, "_vfy_stats_trace_id", None) if bk is None else bk.stats_trace_id
        if execute_topk and _topk_tid is None:
            raise ValueError("verify_enqueue_post(execute_topk=True) needs QWEN36_VERIFY_DEVICE_TOPK=1 at capture")
        if execute_stats and (not execute_topk or _stats_tid is None):
            raise ValueError("execute_stats needs a topk replay and a captured stats trace")
        self._vfy_check_post(bk, execute_topk, execute_mask, mask_host, False)
        _pen_tid = getattr(self, "_vfy_pen_trace_id", None) if bk is None else bk.pen_trace_id
        self._vfy_check_pen(_pen_tid, execute_topk, execute_pen)
        dev = self.device
        if execute_mask:
            handle.mask_keep = self.verify_stage_mask(mask_host)
            ttnn.execute_trace(
                dev, bk.mask_trace_id if bk is not None else self._vfy_mask_trace_id, cq_id=0, blocking=False
            )
        if execute_pen:
            ttnn.execute_trace(dev, _pen_tid, cq_id=0, blocking=False)
        if execute_topk:
            ttnn.execute_trace(dev, _topk_tid, cq_id=0, blocking=False)
        if execute_stats:
            ttnn.execute_trace(dev, _stats_tid, cq_id=0, blocking=False)
        handle.post_pending = False
        return handle

    def verify_sync_and_read(self, handle):
        """Second half of ``verify_traced``: synchronize the device, then ``verify_after_external_sync``."""
        ttnn.synchronize_device(self.device)
        return self.verify_after_external_sync(handle)

    def verify_after_external_sync(self, handle):
        """``verify_sync_and_read`` minus the synchronize (the caller synced after enqueueing). Flips the GDN window
        parity (once per replay) and reads ids / logits / topk. Returns what ``verify_traced`` returns."""
        if handle.post_pending:
            raise ValueError("verify_after_external_sync: the second half of a forward_only step was never enqueued")
        rows, B = handle.rows, handle.B
        read_logits, read_topk, read_col0, _fused = handle.read_logits, handle.read_topk, handle.read_col0, handle.fused
        if _fused:
            # The replay rebuilt every user's conv window into win[1-par], so that half is now E_prev for every GDN
            # layer; flip them together (verify_win_cur / the next page follow).
            for dn in self._vfy_gdn:
                dn._spec_win_par ^= 1

        # Replicated ids (the trace argmaxed the replicated logits): one replica is the whole answer,
        # and it is B*T uint32 instead of B*T x vocab floats.
        ids = ttnn.to_torch(ttnn.get_device_tensors(self._vfy_ids_out)[0]).reshape(-1)
        ids = [int(v) for v in ids[:rows]]
        lt = None
        self._vfy_last_col0 = False
        bk = None
        if getattr(handle, "bucket", None) is not None:
            # A row bucket replayed: ids / topk come from the shared B-row buffers (rows >= b*T hold zeros / pads), the logits from
            # the bucket's own b-row copy. Every readback is zero-padded to the B-row shape the engine expects.
            bk = self._vfy_bk[handle.bucket]
            if read_logits:
                if read_col0 and bk.narrow_ready:
                    sl = self._vfy_narrow_logits(bk.logits_rm_out, bk.b)
                    lt = ttnn.to_torch(ttnn.get_device_tensors(sl)[0])
                    ttnn.deallocate(sl)
                    lt = self._vfy_pad_rows(lt.reshape(bk.b, -1)[:, : self.vocab_size], B)
                    self._vfy_last_col0 = True
                else:
                    lt = ttnn.to_torch(ttnn.get_device_tensors(bk.logits_rm_out)[0])
                    lt = self._vfy_pad_rows(lt.reshape(-1, self.vocab_size)[: bk.rows], rows)
            if read_topk:
                width = self.num_devices * self._vfy_topk_m
                vals = ttnn.to_torch(ttnn.get_device_tensors(self._vfy_topk_vals_out)[0]).reshape(-1, width)[: bk.rows]
                idx = ttnn.to_torch(ttnn.get_device_tensors(self._vfy_topk_ids_out)[0]).reshape(-1, width)[: bk.rows]
                lt = (self._vfy_pad_rows(vals, rows), self._vfy_pad_rows(idx, rows))
            return ids, self._vfy_rows_out, lt
        if read_logits:
            if read_col0 and getattr(self, "_vfy_narrow_ready", False):
                # QWEN36_SPEC_HOSTSET narrow step: only column 0 of each user is real. Slice it on the device and read
                # [B, V] (1/T of the bytes); ``_vfy_last_col0`` tells the caller which shape it got.
                sl = self._vfy_narrow_logits(self._vfy_logits_rm_out)
                lt = ttnn.to_torch(ttnn.get_device_tensors(sl)[0])
                ttnn.deallocate(sl)
                lt = lt.reshape(B, -1)[:, : self.vocab_size]
                self._vfy_last_col0 = True
            else:
                lt = ttnn.to_torch(ttnn.get_device_tensors(self._vfy_logits_rm_out)[0])
                lt = lt.reshape(-1, self.vocab_size)[:rows]
        if read_topk:
            # Replicated by the all-gather: one replica holds every device's block (32 KB + 64 KB at M=64).
            width = self.num_devices * self._vfy_topk_m
            vals = ttnn.to_torch(ttnn.get_device_tensors(self._vfy_topk_vals_out)[0]).reshape(-1, width)[:rows]
            idx = ttnn.to_torch(ttnn.get_device_tensors(self._vfy_topk_ids_out)[0]).reshape(-1, width)[:rows]
            lt = (vals, idx)
        return ids, self._vfy_rows_out, lt

    def _forward_seed_bucket_tp(self, token_buf, kvpos, kvpt, cos, sin, n_users):
        """``_forward_verify_bucket_tp`` at T = 1: the seed's device body, one row per user.

        The same forward as the verify trace body, layer for layer, with two differences that T = 1
        forces:
          * GDN takes ``gdn_seed=True`` -> forward_seed_recurrent, which advances the durable state;
            the ring and E_prev it would otherwise read do not exist yet (capture_verify_trace
            allocates them after the seed).
          * the rows are B different users, one candidate each, so the KV write is the ordinary
            batched paged_update_cache (``alias_kv_write=False``): the per-row loop exists only
            because a user's T candidates share a page table and collide on a cache tile.
            ``spec_verify_mode`` is moot: spec_sdpa_enabled(1) is False, so the verify body would
            take the legacy per-row SDPA call here too.

        Everything else (decode-config norms, the per-row decode SDPA with per-row cur_pos, the final
        norm + LM head + spec_feed_rows tail) is the verify body's. Runs eagerly and before any trace
        capture, so every program it compiles is safe.

        Returns (logits [1,1,B,vocab] replicated, feed [1,1,B,dim/tp] fractured).
        """
        x = self.embd(token_buf)
        x = ttnn.reshape(x, (1, 1, n_users, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        # The seed must be the verify body's arithmetic (the drafter is handed its feed rows), and the verify
        # never takes the lean attention decode path, so neither does the seed: on the lean path B=1 draft
        # acceptance drops from 8.5 to 3.25 of 11. Eager, before any capture, so toggling the flag is safe.
        attns = [layer.attention for layer in self.layers if layer.is_full_attention]
        for attn in attns:
            attn.lean_decode = False
        try:
            x = self._seed_bucket_layers(x, kvpos, kvpt, cos, sin, n_users)
        finally:
            for attn in attns:
                attn.lean_decode = True
        return self._spec_tail(x)

    def _seed_bucket_layers(self, x, kvpos, kvpt, cos, sin, n_users):
        """The layer loop of _forward_seed_bucket_tp (consumes x, returns the residual after the last layer)."""
        cos, sin, _dr = self._rope_decode_arg(cos, sin)
        for layer in self.layers:
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=cos,
                    sin=sin,
                    mode="decode",
                    position_tensor=kvpos,
                    page_table=kvpt,
                    n_users=n_users,
                )
            else:
                x_new = layer.forward(
                    x,
                    mode="prefill",
                    chunk_size=self.args.gdn_chunk_size,
                    valid_len=n_users,
                    gdn_recurrent=True,
                    gdn_seed=True,
                    decode_cfg=True,
                    n_users=n_users,
                )
            ttnn.deallocate(x)
            x = x_new
        if _dr is not None:
            _dr.free()
        return x

    def seed_spec_step(self, tokens, positions, page_tables):
        """One eager verify-style forward over B users, T = 1: the spec loop's seed, and the only
        place the base model runs outside the verify trace once a generation starts.

        Consumes each user's first predicted token at its own absolute position and returns
        (logits: torch [B, vocab] host, feed_rows: ttnn [1,1,B,dim/tp]). The inputs are staged
        as capture_verify_trace stages the trace's, at one row per user instead of T:
        tokens [1,B] uint32, cur_pos/KV slot [B] int32, page table [B,nb], per-row decode RoPE from
        _rope_tp_cos_sin_decode_torch. The feed rows go through spec_feed_rows, so the drafter is
        handed the tensor the verify's rows would have produced.

        The seed is verify-style rather than a plain decode step because it writes the GDN state the
        whole generation resumes from, and the decode path builds its conv differently from the
        verify: a K-tap shift-register MAC versus the packed KDA conv over each user's [K, C] window
        (gdn _kda_conv_packed). The two are not bit-equal, so a decode-formulated seed hands the loop
        a state its own formulation would never have produced, and greedy near-ties fork off the
        trajectory the verify would have taken.
        """
        assert self.num_devices > 1, "seed_spec_step is the TP path"
        assert not self._ondev_argmax, "seed_spec_step reads gathered logits; on-device argmax is off on the spec path"
        B = len(tokens)
        assert B == self.args.max_batch_size, f"seed runs the full decode width ({self.args.max_batch_size}), got {B}"
        assert len(positions) == B, "one position per user"
        assert B <= ttnn.TILE_SIZE, f"the seed is one decode tile: {B} users"
        assert self._paged_kv_caches is not None, "allocate_kv_caches first"
        pt = _page_table_rows(page_tables)
        assert pt.shape[0] == B, f"page_tables must have one row per user, got {tuple(pt.shape)}"

        dev = self.device
        rep = ttnn.ReplicateTensorToMesh(dev)
        toks = torch.tensor([int(t) for t in tokens], dtype=torch.int32).reshape(1, B)
        pos = torch.tensor([int(p) for p in positions], dtype=torch.int32)
        token_buf = ttnn.from_torch(toks, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, mesh_mapper=rep)
        # Absolute cache slot per row, doubling as the decode SDPA's cur_pos (the trace's _vfy_kvpos_buf at T = 1).
        kvpos = ttnn.from_torch(pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, mesh_mapper=rep)
        kvpt = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, mesh_mapper=rep)
        cos_t, sin_t = self._rope_tp_cos_sin_decode_torch(pos, self._spec_row_deltas(len(pos)), wide=self._rope_perm)
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=rep)

        logits, feed = self._forward_seed_bucket_tp(token_buf, kvpos, kvpt, cos, sin, B)
        # _lm_head all-gathers, so the logits are replicated across the TP mesh: read one replica
        # instead of concatenating every rank (the idiom process_output_decode uses).
        lt = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).reshape(-1, self.vocab_size)[:B].float()
        ttnn.deallocate(logits)
        for t in (token_buf, kvpos, kvpt, cos, sin):
            ttnn.deallocate(t)
        return lt, feed

    def decode_step_paged(self, token_id, pos, page_table):
        """One paged base decode at absolute ``pos`` (B=1), returning host logits + the pre-final-norm
        hidden. Advances GDN (recurrent) + writes paged KV at ``pos``.

        page_table : torch [1, num_blocks] identity table.
        Returns (logits [vocab] host float, hidden [1,1,1,dim/tp] fractured device — the MTP seed).
        """

        rep = ttnn.ReplicateTensorToMesh(self.device) if self.num_devices > 1 else None
        mk = dict(mesh_mapper=rep) if rep is not None else {}
        tok = ttnn.from_torch(
            torch.tensor([[int(token_id)]], dtype=torch.int32), dtype=ttnn.uint32, device=self.device, **mk
        )
        cos, sin = self._rot_mats_decode_base(torch.tensor([pos + self.rope.rope_delta], dtype=torch.int32))
        cur_pos_tt = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, device=self.device, **mk)
        pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        logits, hidden = self._forward_decode(tok, cos, sin, cur_pos_tt, pt, return_hidden=True)
        if self._ondev_argmax:
            # _forward_decode returned the PRE-gather vocab-sharded logits; the ranks are different
            # vocab slices, so they really do have to be concatenated to rebuild the full row.
            lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
        else:
            # _lm_head all-gathered: the logits are REPLICATED, so read one replica instead of
            # pulling every rank back only to drop all but the first (process_output_decode's idiom).
            lt = ttnn.to_torch(ttnn.get_device_tensors(logits)[0])
        ttnn.deallocate(logits)
        lt = lt.reshape(-1)[: self.vocab_size].float()
        return lt, hidden

    def _forward_prefill_chunk(
        self, token_buf, cos_buf, sin_buf, chunk_start_idx_tensor, full_page_table, chunk_page_table
    ):
        """Trace-safe single-chunk prefill. Updates paged KV + GDN state in place.
        Returns last-layer hidden [1, chunk_size, hidden_size]."""
        x = self.embd(token_buf)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        # Trace-safe vision splice (fixed-shape where over persistent buffers; identity when the
        # mask buffer is zero, which is the case for every text-only chunk and request). The caller
        # stages the buffers before replaying chunk 0 of a multimodal prompt; chunks>0 are cleared.
        x = self._apply_vision_merge(x, length=x.shape[1])
        for layer in self.layers:
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=cos_buf,
                    sin=sin_buf,
                    mode="prefill",
                    page_table=full_page_table,
                    chunk_page_table=chunk_page_table,
                    chunk_start_idx_tensor=chunk_start_idx_tensor,
                )
            else:
                x_new = layer.forward(x, mode="prefill", chunk_size=layer.attention.long_prefill_chunk_size)
            ttnn.deallocate(x)
            x = x_new
        return x

    def _rope_tp_cos_sin_torch(self, start, length, wide=None):
        """Torch cos/sin tables [1, 1, length, rope_head_dim] for SEQUENCE positions
        [start, start+length), in the rope_tp (HF split-halves) format consumed by
        apply_partial_rope_prefill. Single source of truth for the TP masked-bucket and
        traced chunk-outer prefill paths (so the captured trace's cos/sin are byte-identical
        to the eager path's). M-RoPE-aware: when a multimodal request staged a per-sequence
        table (build_request_rope) this slices it; otherwise it is ordinary 1D RoPE at
        positions [start, start+length) — byte-identical to the pre-M-RoPE behaviour.
        wide (default: self._rope_perm, the base model): permuted hd-wide [1, 1, length, head_dim] (QWEN36_ROPE_PERM; the M-RoPE
        rows are widened the same way). The MTP drafter's own call passes wide=False; a drafter handed the base's hd-wide
        buffers narrows them in TPAttention._rope_prefill."""
        rd = self.args.rope_head_dim
        cos_t, sin_t = self.rope.prefill_cos_sin_torch(start, length)  # [length, rd] bf16
        if self._rope_perm if wide is None else wide:
            cos_t, sin_t = self._rope_widen(cos_t, sin_t)
            rd = self.args.head_dim
        cos = cos_t.reshape(1, 1, length, rd)
        sin = sin_t.reshape(1, 1, length, rd)
        return cos, sin

    def _forward_prefill_chunk_tp(
        self,
        token_buf,
        cos_buf,
        sin_buf,
        chunk_start_idx_tensor,
        full_page_table,
        chunk_page_table,
        seq_len=None,
        gdn_masks=None,
    ):
        """TP trace-safe single-chunk prefill (replicated persistent buffers).
        Full chunk (valid_len==chunk_size); flexible SDPA via device chunk_start_idx.
        seq_len: positions in token_buf (default: the captured chunk size; the short-prompt trace passes its bucket).
        gdn_masks: persistent device masks of the short trace (see _ShortPrefillTrace); None = full chunk.
        Returns hidden [1,1,chunk_size,dim]."""
        chunk_size = seq_len if seq_len is not None else self._chunked_chunk_size
        x = self.embd(token_buf)
        x = ttnn.reshape(x, (1, 1, chunk_size, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        # Trace-safe vision splice (fixed-shape where over the hidden-sharded persistent buffers;
        # identity when the mask is zero, i.e. every text-only chunk). The caller stages the
        # buffers before replaying chunk 0 of a multimodal prompt; later chunks are cleared.
        x = self._apply_vision_merge(x, length=chunk_size)
        for layer in self.layers:
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=cos_buf,
                    sin=sin_buf,
                    mode="prefill",
                    page_table=full_page_table,
                    chunk_page_table=chunk_page_table,
                    chunk_start_idx_tensor=chunk_start_idx_tensor,
                )
            else:
                # valid_len=None: no GDN mask; trace-safe static conv capture (matches valid_len==chunk_size).
                x_new = layer.forward(
                    x, mode="prefill", chunk_size=self.args.gdn_chunk_size, valid_len=None, gdn_masks=gdn_masks
                )
            ttnn.deallocate(x)
            x = x_new
        return x

    _CHUNK_SET_ATTRS = (
        "_chunk_token_buf",
        "_chunk_start_idx_tensor",
        "_chunk_full_page_table_buf",
        "_chunk_page_table_buf",
        "_chunk_cos_buf",
        "_chunk_sin_buf",
        "_chunked_trace_id",
        "_chunked_trace_output",
        "_chunked_chunk_size",
        "_chunked_prepared_key",
    )

    def _activate_chunk_set(self, label):
        """Swap the chunk-trace state on self to the set `label` ("small" = the 2048 serving set, "big")."""
        if label == self._chunk_set_active:
            return
        self._chunk_sets[self._chunk_set_active] = {a: getattr(self, a) for a in self._CHUNK_SET_ATTRS}
        for a, v in self._chunk_sets.pop(label, dict.fromkeys(self._CHUNK_SET_ATTRS)).items():
            setattr(self, a, v)
        self._chunk_set_active = label

    @contextlib.contextmanager
    def _big_chunk_set(self):
        self._activate_chunk_set("big")
        try:
            yield
        finally:
            self._activate_chunk_set("small")

    def big_chunk_rows(self):
        """Rows of the recorded big chunk trace, or 0 (off / not recorded)."""
        big = self._chunk_sets.get("big") if self._chunk_set_active == "small" else None
        return int(big["_chunked_chunk_size"]) if big and big["_chunked_trace_id"] is not None else 0

    def masked_big_buckets(self, page_table):
        """Masked short-trace buckets beyond 2048 for the spec fast prefill (QWEN36_PREFILL_MASKED_BIG=1 with
        QWEN36_PREFILL_BIG_CHUNK=8192 and the big set prepared): (4096, 8192) limited to what the page table holds, else
        (). Pass them to prepare_prefill_traces_short(spec=True) with the default buckets."""
        if not prefill_masked_big_enabled() or self._prefill_big_chunk != 8192 or "big" not in self._chunk_sets:
            return ()
        cap = page_table.shape[-1] * get_block_size(self._paged_kv_caches)
        return tuple(b for b in self._MASKED_BIG_BUCKETS if cap >= b)

    def prepare_prefill_trace_chunked_big(self, device, page_table, warm_cb=None):
        """Allocate + compile the QWEN36_PREFILL_BIG_CHUNK chunk set (after the 2048 one, before any capture). One more
        eager big-chunk forward then compiles the tail-free last-row logits program on its real output and hands that
        output to ``warm_cb(hidden)`` (the spec engine compiles its per-chunk drafter warm at this size). No-op when off.
        The masked buckets are warmed by the 2048 prepare."""
        if not self._prefill_big_chunk:
            return
        block_size = get_block_size(self._paged_kv_caches)
        if page_table.shape[-1] * block_size < self._prefill_big_chunk:
            # A table shorter than one big chunk makes the paged SDPA pad it with a device ttnn.zeros: a host write,
            # illegal inside the capture. Such a context never has a full big block, so the set is not needed.
            logger.info(
                f"Big prefill chunk set skipped: the page table holds {page_table.shape[-1] * block_size} tokens "
                f"< {self._prefill_big_chunk}"
            )
            return
        tp = self.num_devices > 1
        with self._big_chunk_set():
            self.prepare_prefill_trace_chunked(device, page_table, self._prefill_big_chunk, warmup_masked_buckets=False)
            if tp:
                out = self._forward_prefill_chunk_tp(
                    self._chunk_token_buf,
                    self._chunk_cos_buf,
                    self._chunk_sin_buf,
                    self._chunk_start_idx_tensor,
                    self._chunk_full_page_table_buf,
                    self._chunk_page_table_buf,
                )
                big = self._prefill_big_chunk
                ttnn.deallocate(self._masked_bucket_logits_tp(out, big, big))
                if warm_cb is not None:
                    warm_cb(out)
                ttnn.deallocate(out)
                ttnn.synchronize_device(device)
            self._reset_gdn_state_for_new_sequence()
        logger.info(f"Big prefill chunk set prepared: {self._prefill_big_chunk} rows")

    def record_prefill_trace_chunked_big(self, device):
        """Capture the big chunk trace (after record_prefill_trace_chunked). No-op when off or not prepared."""
        if "big" not in self._chunk_sets:  # off, or prepare skipped it
            return
        with self._big_chunk_set():
            self.record_prefill_trace_chunked(device)

    def capture_prefill_trace_chunked(
        self, device, page_table, chunk_size=2048, warmup_masked_buckets=True, capture_chunk_trace=True
    ):
        """Capture one chunk's all-layer prefill as a trace; replayed per chunk.

        Chunk-outer prefill stays under the 4 GiB trace limit at long context.
        Flexible SDPA (runtime chunk_start) makes one trace serve all chunk positions.

        capture_chunk_trace=False warms the masked-bucket programs (prepare) but skips parking the chunk trace: a
        prompt < chunk_size never replays it when the short-prompt trace (capture_prefill_traces_short) serves it,
        and its trace-region bytes then go to the short trace."""
        self.prepare_prefill_trace_chunked(device, page_table, chunk_size, warmup_masked_buckets)
        if capture_chunk_trace:
            self.record_prefill_trace_chunked(device)

    def prepare_prefill_trace_chunked(self, device, page_table, chunk_size=2048, warmup_masked_buckets=True):
        """Allocate + compile everything the chunk trace needs. No-op if already prepared for these args."""
        key = (chunk_size, tuple(page_table.shape), bool(warmup_masked_buckets))
        if self._chunked_prepared_key == key:
            return
        if self._chunked_trace_id is not None:
            ttnn.release_trace(device, self._chunked_trace_id)
            self._chunked_trace_id = None
        if self.num_devices > 1:
            self._prepare_prefill_trace_chunked_tp(device, page_table, chunk_size, warmup_masked_buckets)
        else:
            self._prepare_prefill_trace_chunked_single(device, page_table, chunk_size, warmup_masked_buckets)
        self._chunked_prepared_key = key

    def record_prefill_trace_chunked(self, device):
        """Capture the chunk trace on prepare_prefill_trace_chunked's buffers; allocates nothing outside capture."""
        assert self._chunked_prepared_key is not None, "Call prepare_prefill_trace_chunked first"
        if self._chunked_trace_id is not None:
            ttnn.release_trace(device, self._chunked_trace_id)
            self._chunked_trace_id = None
        tp = self.num_devices > 1
        self._reset_gdn_state_for_new_sequence()
        forward = self._forward_prefill_chunk_tp if tp else self._forward_prefill_chunk
        # The trace output is read right after each replay, before any other trace can run.
        with trace_allocation_tracker.corruptible_allocation_scope(device):
            self._chunked_trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            self._chunked_trace_output = forward(
                self._chunk_token_buf,
                self._chunk_cos_buf,
                self._chunk_sin_buf,
                self._chunk_start_idx_tensor,
                self._chunk_full_page_table_buf,
                self._chunk_page_table_buf,
            )
            ttnn.end_trace_capture(device, self._chunked_trace_id, cq_id=0)
        logger.info(f"Chunked prefill trace{' (TP)' if tp else ''} captured successfully!")

    def _prepare_prefill_trace_chunked_single(self, device, page_table, chunk_size, warmup_masked_buckets):
        assert self._paged_kv_caches is not None, "Call allocate_kv_caches first"
        assert chunk_size % 128 == 0, f"chunk_size {chunk_size} must be a multiple of 128"
        B = 1
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_chunk = chunk_size // block_size

        self._chunked_chunk_size = chunk_size

        # Allocate the vision-splice buffers BEFORE warmup so the fixed-shape ttnn.where in
        # _forward_prefill_chunk / _forward_prefill_chunk_masked compiles in the warmup pass (and
        # is captured), never at request time. Zero-initialised -> identity for text-only.
        self._alloc_vision_merge_buffers(device, chunk_size)

        # ---- Persistent per-chunk input buffers (addresses baked into the trace) ----
        self._chunk_token_buf = ttnn.from_torch(
            torch.zeros(B, chunk_size, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self._chunk_start_idx_tensor = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self._chunk_full_page_table_buf = ttnn.from_torch(
            page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self._chunk_page_table_buf = ttnn.from_torch(
            page_table[:, :blocks_per_chunk].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        # TP handoff: add ReplicateTensorToMesh for cos/sin (parity with tt/rope.py).
        self._chunk_cos_buf = ttnn.from_torch(
            self.rope.cos_cpu[:chunk_size].unsqueeze(0).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self._chunk_sin_buf = ttnn.from_torch(
            self.rope.sin_cpu[:chunk_size].unsqueeze(0).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Warmup outside trace: compile per-chunk programs.
        self._reset_gdn_state_for_new_sequence()
        warmup_out = self._forward_prefill_chunk(
            self._chunk_token_buf,
            self._chunk_cos_buf,
            self._chunk_sin_buf,
            self._chunk_start_idx_tensor,
            self._chunk_full_page_table_buf,
            self._chunk_page_table_buf,
        )
        ttnn.deallocate(warmup_out)
        ttnn.synchronize_device(device)

        # Warmup masked-bucket programs outside trace (same GDN mode as serving).
        # Dummy prefills dirty state/KV; reset below before capture.
        if warmup_masked_buckets:
            self.warmup_prefill_masked_buckets(page_table)
        self._reset_gdn_state_for_new_sequence()

    def _prepare_prefill_trace_chunked_tp(self, device, page_table, chunk_size, warmup_masked_buckets):
        """TP fork of _prepare_prefill_trace_chunked_single.

        Replicated persistent buffers; rope_tp cos/sin; GDN uses its module-owned state.
        Trace replays _forward_prefill_chunk_tp."""
        assert self._paged_kv_caches is not None, "Call allocate_kv_caches first"
        assert chunk_size % 128 == 0, f"chunk_size {chunk_size} must be a multiple of 128"
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_chunk = chunk_size // block_size

        self._chunked_chunk_size = chunk_size

        # Allocate the hidden-sharded vision-splice buffers BEFORE warmup so the fixed-shape
        # ttnn.where in the TP forwards compiles in the warmup pass (and is captured), never at
        # request time. Zero-initialised -> identity for text-only.
        self._alloc_vision_merge_buffers(device, max(chunk_size, self._prefill_big_chunk))

        rep = ttnn.ReplicateTensorToMesh(device)
        B = 1
        # Persistent per-chunk inputs (replicated; addresses baked into trace).
        self._chunk_token_buf = ttnn.from_torch(
            torch.zeros(B, chunk_size, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        self._chunk_start_idx_tensor = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        self._chunk_full_page_table_buf = ttnn.from_torch(
            page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, mesh_mapper=rep
        )
        self._chunk_page_table_buf = ttnn.from_torch(
            page_table[:, :blocks_per_chunk].contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        cos_t, sin_t = self._rope_tp_cos_sin_torch(0, chunk_size)
        self._chunk_cos_buf = ttnn.from_torch(
            cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=rep
        )
        self._chunk_sin_buf = ttnn.from_torch(
            sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=rep
        )

        # Warmup outside trace: compile per-chunk programs.
        self._reset_gdn_state_for_new_sequence()
        warmup_out = self._forward_prefill_chunk_tp(
            self._chunk_token_buf,
            self._chunk_cos_buf,
            self._chunk_sin_buf,
            self._chunk_start_idx_tensor,
            self._chunk_full_page_table_buf,
            self._chunk_page_table_buf,
        )
        ttnn.deallocate(warmup_out)
        ttnn.synchronize_device(device)

        # Warmup masked-bucket/tail programs outside trace (same GDN mode; avoids trace clobber).
        if warmup_masked_buckets:
            self.warmup_prefill_masked_buckets(page_table)
        self._reset_gdn_state_for_new_sequence()

    # ----------------------------------------------------------------------- #
    # Traced B=1 short-prompt prefill (ANY length < 2048) + on-device first token / pre-gather logits
    # ----------------------------------------------------------------------- #
    # A short prompt (< chunk_size) otherwise runs prefill_masked_bucket: an EAGER ~3.3k-op forward whose
    # GDN layers upload a per-layer host one-hot (valid_len masks), preceded by 288 eager GDN-state reset
    # copies, followed by an all-gather + full-logit readback of all replicas. Here the whole request is one
    # trace per BUCKET Tb (smallest of 128/256/512/1024/2048 >= length):
    #   [zero GDN carries] -> embed -> 64 layers -> one-hot last real row -> norm -> LM head -> (argmax, max, logits)
    # replayed with every per-request input in a persistent (replicated) device tensor rewritten by
    # copy_host_to_device_tensor. Arbitrary valid_len <= Tb needs only five length-dependent inputs:
    #   * m_bg   [1,Tb,1] fp32  1 for t < valid_len. GDN multiplies beta and g by it, so the padded steps are the
    #                           identity on the recurrent state (final_state == state at valid_len). The q/k/v
    #                           mask (QWEN36_PREFILL_TRACE_QKV_MASK=1, default 0) is bit-parity only: the padded
    #                           rows come AFTER the real ones and have beta = g = 0.
    #   * conv_sel [1,3,Tb] bf16  one-hot of the last 3 REAL conv-input rows (replaces the static tail slice);
    #                           rows with valid_len-3+j < 0 are all-zero (exact: the from-scratch history is zero).
    #   * sel    [1,1,1,Tb]     one-hot of the last real row for the LM head.
    #   * chunk page table      entries >= ceil(valid_len/block) point at a TRASH block, so the padded K/V of the
    #                           bucket never lands in another request's blocks (vLLM: block 0 = the null block,
    #                           never allocated to a request; demo: a spare block, see prefill_trash_block).
    #   * tokens / cos / sin    zero-padded / bucket-long.
    # Outputs kept per trace: per-shard (argmax, max) for the greedy demo, and the pre-gather vocab-sharded last-row
    # logits [1,1,1,V/TP] that vLLM reads once with ConcatMeshToTensor(dim=-1) (no all-gather CCL).
    # Prompts >= chunk_size (2048) keep the chunk trace + eager tail (out of scope).

    _SHORT_TRACE_DEFAULT_BUCKETS = (128, 256, 512, 1024, 2048)
    _MASKED_BIG_BUCKETS = (4096, 8192)  # QWEN36_PREFILL_MASKED_BIG, spec short path only (see masked_big_buckets)

    @staticmethod
    def short_prefill_trace_enabled():
        """QWEN36_PREFILL_BUCKET_TRACE (default "1"); "0" restores the eager masked-bucket short prefill."""
        return os.environ.get("QWEN36_PREFILL_BUCKET_TRACE", "1") != "0"

    @staticmethod
    def short_prefill_trace_qkv_mask_enabled():
        """QWEN36_PREFILL_TRACE_QKV_MASK (default "0"): also multiply q/k/v by the length mask in the trace (bit parity
        with the eager masked path; real outputs are unaffected either way)."""
        return os.environ.get("QWEN36_PREFILL_TRACE_QKV_MASK", "0") == "1"

    def short_prefill_trace_buckets(self):
        """Buckets to capture by default: QWEN36_PREFILL_TRACE_BUCKETS (comma list, default 128,256,512,1024,2048)."""
        raw = os.environ.get("QWEN36_PREFILL_TRACE_BUCKETS", "")
        if raw.strip():
            vals = tuple(int(x) for x in raw.split(",") if x.strip())
        else:
            vals = self._SHORT_TRACE_DEFAULT_BUCKETS
        return tuple(sorted(b for b in set(vals) if b in self._PREFILL_MASK_BUCKETS))

    def _short_trace_masks_supported(self):
        """The arbitrary-length trace needs the KDA conv in every GDN layer (the FIR conv builds host one-hots)."""
        return all(getattr(l.attention, "_gdn_kda_conv", False) for l in self.layers if not l.is_full_attention)

    def short_prefill_trace_bucket(self, length, chunk_size=2048):
        """The bucket to capture for a `length`-token prompt, or None when the traced short prefill does not
        apply (disabled, single device / replicated LM head, length >= chunk_size, or a non-exact length without
        the KDA conv)."""
        if not self.short_prefill_trace_enabled():
            return None
        if self.num_devices <= 1 or not self._lmhead_vocab_sharded:
            return None
        if length < 1 or length >= chunk_size:
            return None
        bucket = self._mask_bucket_for(length)
        if length != bucket and not self._short_trace_masks_supported():
            return None
        return bucket

    def _gdn_binding_refs(self):
        """The GDN state tensors currently bound on every GDN layer (a short trace bakes their addresses)."""
        return tuple(l.attention.rec_state for l in self.layers if not l.is_full_attention)

    def _short_trace_for(self, length, chunk_size=2048, bucket=None):
        """The captured trace serving a `length`-token prompt, or None (flag off, length >= chunk_size, bucket not
        captured, or the GDN state bound now is not the one the trace was captured against, e.g. the decode buffers
        vs the batched prefill scratch -> the caller falls back to the eager masked-bucket path).
        ``bucket`` (QWEN36_PREFILL_MASKED_BIG only; 4096 or 8192): the masked big bucket the spec fast prefill chose for a
        remainder of `length` <= bucket rows (3 <= length: the carried GDN conv history). None = the default lookup."""
        if not self._short_traces or not self.short_prefill_trace_enabled():
            return None
        if bucket is not None:
            if not prefill_masked_big_enabled() or bucket not in self._MASKED_BIG_BUCKETS:
                return None
            if not (3 <= length <= bucket):
                return None
            tr = self._short_traces.get(bucket)
            if tr is None or not tr.masked or not tr.spec:
                return None
        else:
            if not (1 <= length < chunk_size):
                return None
            tr = self._short_traces.get(self._mask_bucket_for(length))
            if tr is None or (length != tr.bucket and not tr.masked):
                return None
        refs = self._gdn_binding_refs()
        if len(refs) != len(tr.gdn_refs) or any(a is not b for a, b in zip(refs, tr.gdn_refs)):
            if not getattr(self, "_short_trace_binding_warned", False):
                self._short_trace_binding_warned = True
                logger.warning("Short prefill trace skipped: GDN state binding differs from the one captured against.")
            return None
        return tr

    def _forward_prefill_short_tp(self, tr):
        """Trace body of the short-prompt prefill (see the section comment). Pure device ops on the persistent
        buffers of `tr`; returns (idx, val, logits): per-device argmax index / max value of the vocab-sharded logits
        and the logits themselves [1,1,1,vocab/TP] (pre-gather)."""
        # Initialise the GDN carries the forward READS (recurrent state + cross-chunk conv carry) inside the trace,
        # replacing 288 eager per-request copies. The sources are the persistent init buffers: zeros for a fresh
        # sequence, the carried / restored state for a resumed one (_prepare_short_trace_gdn_init stages them).
        # conv_states need no reset: capture_state=True rewrites all K of them at the end of every GDN layer
        # (slot 0 from the zero source, 1..K-1 from this segment's tail).
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            ttnn.copy(dn._prefill_init_rec, dn.rec_state)
            ttnn.copy(dn._prefill_init_conv_carry, dn.conv_carry)
        hidden = self._forward_prefill_chunk_tp(
            tr.tok, tr.cos, tr.sin, tr.csi, tr.full_pt, tr.chunk_pt, seq_len=tr.bucket, gdn_masks=tr.gdn_masks
        )
        last_feed = None
        if tr.spec:
            # MTP drafter KV warm-up over the segment, in the trace (QWEN36_SPEC_FAST_PREFILL): the same ops the eager
            # spec prefill runs per chunk (MTPSpecEngine._prefill_row._on_chunk -> spec_feed_rows, _warm_mtp_chunk,
            # _capture_last_row), on the persistent MTP inputs. The feed row of the last real position is kept as a
            # trace output (an exact one-hot gather, like _capture_last_row); the padded rows' K/V go to the MTP
            # scratch block through mtp_chunk_pt. The base's cos/sin/chunk-start tensors are the same positions.
            feed = self.spec_feed_rows(hidden)
            last_feed = ttnn.matmul(tr.sel, feed, compute_kernel_config=self._spec_select_cc())  # [1, 1, 1, dim_frac]
            mtp_out = self.mtp.forward_prefill(
                feed,
                tr.mtp_tok,
                tr.cos,
                tr.sin,
                tr.mtp_full_pt,
                chunk_page_table=tr.mtp_chunk_pt,
                chunk_start_idx=0,  # host int only sizes the page table; the device csi tensor positions the SDPA
                chunk_start_idx_tensor=tr.csi,
            )
            ttnn.deallocate(mtp_out)
            if feed is not hidden:
                ttnn.deallocate(feed)
        # Last-position select + norm + LM head: the same ops as _masked_bucket_logits_tp, minus the logits
        # all-gather (the argmax runs per vocab shard instead).
        x_last = ttnn.matmul(tr.sel, hidden)  # [1, 1, 1, dim_frac]
        ttnn.deallocate(hidden)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = self.norm(x_last, mode=Mode.PREFILL)
        logits = ttnn.linear(x_last, self.lm_head_weight, **_lm_head_kwargs())  # [1, 1, 1, vocab/TP], pre-gather
        ttnn.deallocate(x_last)
        idx, val = self.shard_argmax_max(logits)
        return idx, val, logits, last_feed

    def _spec_select_cc(self):
        """Compute config of the spec short trace's feed-row gather: a one-hot matmul must be exact (HiFi4 + fp32
        accumulation, packer_l1_acc off), the same config MTPSpecEngine uses for its anchor / last-row selects."""
        if self._spec_sel_cc is None:
            self._spec_sel_cc = ttnn.init_device_compute_kernel_config(
                self.mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
        return self._spec_sel_cc

    def capture_prefill_trace_short(self, device, page_table, bucket=128):
        """Capture ONE bucket's short-prompt prefill trace (see capture_prefill_traces_short)."""
        return self.capture_prefill_traces_short(device, page_table, buckets=(bucket,))

    def capture_prefill_traces_short(self, device, page_table, buckets=None, spec=False):
        """Capture the B=1 short-prompt prefill trace of every bucket in `buckets` (default
        short_prefill_trace_buckets()). Call in the UNTIMED warmup phase, after allocate_kv_caches (and normally
        after capture_prefill_trace_chunked, which allocates the vision-splice buffers and warms the masked-bucket
        programs), with the GDN state bound EXACTLY as requests will run it (decode buffers for B=1 serving, the
        persistent prefill scratch for batched serving). Replay: prefill_short_traced / prefill_short_traced_logits /
        prefill_traced_chunked.

        Persistent device inputs per bucket (all replicated, addresses baked into the trace, ALL rewritten per
        request so none has to survive other traces' replays): tokens [1,Tb], chunk_start_idx, full page table
        [1,buf_blocks], chunk page table [1,Tb/block], cos/sin [1,1,Tb,rope_dim], last-row one-hot select
        [1,1,1,Tb], GDN masks m_bg [1,Tb,1] fp32 + conv_sel [1,3,Tb] bf16 (+ q/k/v mask if enabled); plus the
        vision-splice buffers shared with the chunk path. EVERY bucket's inputs are allocated before the first
        warm pass and every warm pass runs before the first capture (no compile / allocation behind a parked trace;
        later buckets' intermediates cannot alias earlier buckets' inputs).

        ``spec`` (MTPSpecEngine, QWEN36_SPEC_FAST_PREFILL): the trace body also warms the MTP drafter's KV over the
        segment and outputs the drafter feed row of the last real position (see _forward_prefill_short_tp); three more
        persistent inputs per bucket (MTP tokens, full and chunk MTP page tables, staged per request from
        _replay_short_trace's ``spec_inputs``).

        prepare_prefill_traces_short + record_prefill_traces_short are the two halves (allocate + compile, then
        capture): the spec engine runs them in separate warmup phases so that every persistent buffer of every trace is
        allocated before any trace is captured."""
        self.prepare_prefill_traces_short(device, page_table, buckets=buckets, spec=spec)
        self.record_prefill_traces_short(device)

    def prepare_prefill_traces_short(self, device, page_table, buckets=None, spec=False):
        """First half of capture_prefill_traces_short: allocate every bucket's persistent inputs and run the eager warm
        passes (every program compiled), leaving the buckets in ``_short_traces_pending`` for record_prefill_traces_short.
        Idempotent per bucket: a bucket already pending or captured is left alone."""
        assert self.num_devices > 1 and self._lmhead_vocab_sharded, "short prefill trace is the TP vocab-sharded path"
        assert self._paged_kv_caches is not None, "Call allocate_kv_caches first"
        buckets = sorted(set(self.short_prefill_trace_buckets() if buckets is None else buckets))
        assert buckets, "no buckets to capture"
        if spec:
            assert self.mtp is not None, "spec short prefill trace needs the MTP head"
            assert self.mtp.attention.use_paged and self.mtp.attention.paged_k is not None, "MTP paged KV not bound"
        todo = []
        for b in buckets:
            cur = self._short_traces_pending.get(b) or self._short_traces.get(b)
            if cur is not None and cur.spec == bool(spec):
                continue  # already prepared / captured for this variant
            todo.append(b)
        if not todo:
            return
        buckets = todo
        block_size = get_block_size(self._paged_kv_caches)
        masked = self._short_trace_masks_supported()
        use_qkv_mask = masked and self.short_prefill_trace_qkv_mask_enabled()
        for bucket in buckets:
            assert bucket % 128 == 0, f"bucket {bucket} must be a multiple of 128 (GDN sub-chunk)"
            assert bucket % block_size == 0
            assert (
                page_table.shape[-1] >= bucket // block_size
            ), f"page_table has {page_table.shape[-1]} blocks, bucket {bucket} needs {bucket // block_size}"
        n_hist = None
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                assert (
                    dn.rec_state is not None and dn.conv_carry is not None
                ), "GDN state must be the persistent in-place state (allocate_kv_caches) before capture"
                assert dn._zero_rec is not None and dn._zero_conv_carry is not None
                if dn._prefill_init_rec is None:  # no persistent scratch (B=1 serving): init buffers for this binding
                    dn._prefill_init_rec = alloc_zeros_like(dn.rec_state, dn.mesh)
                    dn._prefill_init_conv_carry = alloc_zeros_like(dn.conv_carry, dn.mesh)
                assert tuple(dn._prefill_init_rec.shape) == tuple(dn.rec_state.shape) and tuple(
                    dn._prefill_init_conv_carry.shape
                ) == tuple(dn.conv_carry.shape), "GDN prefill init buffers must have the bound (B=1) state shapes"
                assert tuple(dn._zero_rec.shape) == tuple(dn.rec_state.shape), (
                    f"GDN _zero_rec {tuple(dn._zero_rec.shape)} != bound rec_state {tuple(dn.rec_state.shape)}: "
                    "capture with the same GDN binding requests use (allocate the scratch AFTER allocate_kv_caches)"
                )
                n_hist = dn.K - 1
        for bucket in buckets:
            self.release_prefill_trace_short(bucket)  # drop a stale capture of this bucket (e.g. a re-capture)

        # Vision-splice buffers must exist BEFORE warmup so the ttnn.where compiles outside the trace.
        if self._vis_buf is None:
            self._alloc_vision_merge_buffers(device, self._chunked_chunk_size or 2048)
        assert self._vis_buf.shape[-2] >= max(buckets), "vision buffers smaller than the bucket"

        rep = ttnn.ReplicateTensorToMesh(device)
        # Full-page-table width MUST be a 32-multiple >= the SDPA's target blocks so forward_prefill_paged's
        # zero-PAD branch (ttnn.zeros + concat) never runs under capture.
        buf_blocks = max(32, ((page_table.shape[-1] + 31) // 32) * 32)
        pt = page_table.to(torch.int32).reshape(page_table.shape[0], -1)
        if pt.shape[-1] < buf_blocks:
            pt = torch.cat([pt, torch.zeros(pt.shape[0], buf_blocks - pt.shape[-1], dtype=torch.int32)], dim=1)

        def _up(t, dtype, layout):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, mesh_mapper=rep)

        # Persistent inputs of ALL buckets, allocated BEFORE any warm pass/capture.
        trs = {}
        for bucket in buckets:
            bpb = bucket // block_size
            tr = _ShortPrefillTrace(bucket, buf_blocks, bpb)
            tr.block_size = block_size
            tr.masked = masked
            tr.n_hist = n_hist
            tr.spec = bool(spec)
            tr.tok = _up(torch.zeros(1, bucket, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            tr.csi = _up(torch.zeros(1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            tr.full_pt = _up(pt.contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            tr.chunk_pt = _up(pt[:, :bpb].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            if spec:
                # The MTP cache is indexed by the same block ids (plus its own scratch block for padding rows).
                tr.mtp_tok = _up(torch.zeros(1, bucket, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                tr.mtp_full_pt = _up(pt.contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                tr.mtp_chunk_pt = _up(pt[:, :bpb].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            cos_t, sin_t = self._rope_tp_cos_sin_torch(0, bucket)
            tr.cos = _up(cos_t, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            tr.sin = _up(sin_t, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            if masked:
                bg, conv_sel, _ = self._short_trace_host_torch(bucket, bucket, n_hist)
                tr.m_bg = _up(bg, ttnn.float32, ttnn.TILE_LAYOUT)
                tr.conv_sel = _up(conv_sel, ttnn.bfloat16, ttnn.TILE_LAYOUT)
                if use_qkv_mask:
                    tr.m_qkv = _up(bg, ttnn.bfloat16, ttnn.TILE_LAYOUT)
                tr.gdn_masks = {"bg": tr.m_bg, "conv_sel": tr.conv_sel}
                if tr.m_qkv is not None:
                    tr.gdn_masks["qkv"] = tr.m_qkv
            tr.gdn_refs = self._gdn_binding_refs()
            trs[bucket] = tr

        # Warm pass 1 (eager, smallest bucket): compiles the layer programs and gives the hidden dtype for the
        # one-hot select (the eager _masked_bucket_logits_tp builds it with dtype=hidden.dtype).
        b0 = buckets[0]
        t0 = trs[b0]
        hidden = self._forward_prefill_chunk_tp(
            t0.tok, t0.cos, t0.sin, t0.csi, t0.full_pt, t0.chunk_pt, seq_len=b0, gdn_masks=t0.gdn_masks
        )
        sel_dtype = hidden.dtype
        ttnn.deallocate(hidden)
        for bucket, tr in trs.items():
            tr.sel_dtype = sel_dtype
            _, _, sel = self._short_trace_host_torch(bucket, bucket, n_hist or 3)
            tr.sel = _up(sel, sel_dtype, ttnn.TILE_LAYOUT)
        # Warm passes 2-3 of EVERY bucket (eager, the full trace body incl. reset/select/norm/LM head/argmax):
        # compile every remaining program before the first trace is parked, and take the steady-state path once.
        for bucket, tr in trs.items():
            for _ in range(2):
                outs = self._forward_prefill_short_tp(tr)
                ttnn.synchronize_device(device)
                for o in outs:
                    if o is not None:
                        ttnn.deallocate(o)
            self._short_traces_pending[bucket] = tr
        # The warm passes advanced GDN state and wrote KV blocks [0, bucket/block): zero the state for whatever
        # runs next (a request's replay re-zeroes it in-trace and rewrites those KV blocks).
        self._reset_gdn_state_for_new_sequence()

    def record_prefill_traces_short(self, device):
        """Second half of capture_prefill_traces_short: capture every prepared (pending) bucket, with the GDN state bound
        as at prepare time. Allocates nothing but the trace outputs (read right after each replay, before any other trace
        runs). No-op when nothing is pending."""
        if not self._short_traces_pending:
            return
        for bucket, tr in sorted(self._short_traces_pending.items()):
            refs = self._gdn_binding_refs()
            assert len(refs) == len(tr.gdn_refs) and all(
                a is b for a, b in zip(refs, tr.gdn_refs)
            ), f"bucket {bucket}: GDN state binding differs from the one prepared against"
            tr.trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            tr.idx, tr.val, tr.logits, tr.last_feed = self._forward_prefill_short_tp(tr)
            ttnn.end_trace_capture(device, tr.trace_id, cq_id=0)
            ttnn.synchronize_device(device)
            self._short_traces[bucket] = tr
            logger.info(
                f"Short-prompt prefill trace (TP, bucket {bucket}, any length <= {bucket}"
                f"{'' if tr.masked else ' [exact only: no KDA conv]'}, on-device argmax + pre-gather logits"
                f"{' + MTP KV warm-up and feed row' if tr.spec else ''}) captured."
            )
        self._short_traces_pending = {}
        # The capture passes advanced GDN state and wrote KV blocks [0, bucket/block): zero the state for
        # whatever runs next (a request's replay re-zeroes it in-trace and rewrites those KV blocks).
        self._reset_gdn_state_for_new_sequence()

    def release_prefill_trace_short(self, bucket=None):
        """Release the short-prompt prefill trace for `bucket` (all of them when None) and its persistent
        input buffers. Called on re-capture and from free_kv_caches (the traces bake the KV/GDN addresses)."""
        buckets = sorted(set(self._short_traces) | set(self._short_traces_pending)) if bucket is None else [bucket]
        for b in buckets:
            tr = self._short_traces.pop(b, None)
            if tr is None:
                tr = self._short_traces_pending.pop(b, None)  # prepared, never captured
            if tr is None:
                continue
            if tr.trace_id is not None:
                ttnn.release_trace(self.device, tr.trace_id)
                tr.trace_id = None
            for buf in (
                tr.tok,
                tr.csi,
                tr.full_pt,
                tr.chunk_pt,
                tr.cos,
                tr.sin,
                tr.sel,
                tr.m_bg,
                tr.conv_sel,
                tr.m_qkv,
                tr.mtp_tok,
                tr.mtp_full_pt,
                tr.mtp_chunk_pt,
            ):
                if buf is not None:
                    ttnn.deallocate(buf)
            tr.idx = tr.val = tr.logits = tr.last_feed = None  # trace outputs: freed with their last reference
            tr.gdn_refs = ()
            tr.host_cache.clear()

    @staticmethod
    def _short_trace_host_torch(bucket, valid_len, n_hist):
        """Torch (bg [1,Tb,1] fp32, conv_sel [1,n_hist,Tb] fp32 0/1, sel [1,1,1,Tb] fp32 0/1) for a prompt of
        `valid_len` real tokens at bucket Tb. conv_sel[j, valid_len-n_hist+j] = 1 (rows before position 0 stay zero)."""
        bg = torch.zeros(1, bucket, 1, dtype=torch.float32)
        bg[:, :valid_len, :] = 1.0
        conv_sel = torch.zeros(1, n_hist, bucket, dtype=torch.float32)
        for j in range(n_hist):
            i = valid_len - n_hist + j
            if i >= 0:
                conv_sel[0, j, i] = 1.0
        sel = torch.zeros(1, 1, 1, bucket, dtype=torch.float32)
        sel[0, 0, 0, valid_len - 1] = 1.0
        return bg, conv_sel, sel

    _SHORT_TRACE_HOST_CACHE = 64  # per-trace LRU of the per-length host mask tensors

    def _short_trace_host_inputs(self, tr, valid_len):
        """Host (device=None) replicated tensors of the length-dependent inputs of `tr` for `valid_len`, cached
        per length (a few hundred KB each at the largest bucket)."""
        hit = tr.host_cache.get(valid_len)
        if hit is not None:
            tr.host_cache.move_to_end(valid_len)
            return hit
        rep = ttnn.ReplicateTensorToMesh(self.device)

        def _h(t, dtype):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep)

        bg, conv_sel, sel = self._short_trace_host_torch(tr.bucket, valid_len, tr.n_hist or 3)
        host = {"sel": _h(sel, tr.sel_dtype)}
        if tr.masked:
            host["bg"] = _h(bg, ttnn.float32)
            host["conv_sel"] = _h(conv_sel, ttnn.bfloat16)
            if tr.m_qkv is not None:
                host["qkv"] = _h(bg, ttnn.bfloat16)
        tr.host_cache[valid_len] = host
        while len(tr.host_cache) > self._SHORT_TRACE_HOST_CACHE:
            tr.host_cache.popitem(last=False)
        return host

    def _prepare_short_trace_gdn_init(self, carry):
        """Stage the GDN init buffers the short trace copies into rec_state/conv_carry (see _forward_prefill_short_tp).
        carry=True (resumed segment: start > 0 or a restored snapshot): the currently bound state becomes the trace's
        init and the buffers are marked dirty. carry=False (fresh sequence): zeros, copied only if a previous resumed
        replay left the buffers dirty, so the common no-prefix-cache path costs nothing extra."""
        if not carry and not self._prefill_init_dirty:
            return
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            if carry:
                ttnn.copy(dn.rec_state, dn._prefill_init_rec)
                ttnn.copy(dn.conv_carry, dn._prefill_init_conv_carry)
            else:
                ttnn.copy(dn._zero_rec, dn._prefill_init_rec)
                ttnn.copy(dn._zero_conv_carry, dn._prefill_init_conv_carry)
        self._prefill_init_dirty = carry

    def _replay_short_trace(
        self,
        token_ids,
        page_table,
        actual_len,
        vision_tokens=None,
        trash_block=None,
        strict=True,
        chunk_start=0,
        carry_gdn=False,
        spec_inputs=None,
        vis_row_offset=0,
        bucket=None,
    ):
        """Stage every per-request input of the bucket trace serving `actual_len` tokens, replay it, drain.
        Returns the _ShortPrefillTrace (outputs readable) or None when no trace serves this request (strict=False
        also returns None when the trash block collides with the request's real blocks).

        spec_inputs (required for a spec trace, see capture_prefill_traces_short(spec=True)): host int32 torch tensors
        (mtp_tokens [1, bucket], mtp_full_table [1, buf_blocks], mtp_chunk_table [1, bucket/block]) staged into the
        MTP inputs; MTPSpecEngine builds them with mtp_prefill_inputs.

        chunk_start (default 0; a multiple of 128): the segment's ABSOLUTE start position, for a prefix-cached /
        resumed prefill. token_ids then holds just the segment's tokens, page_table the request's FULL row (the
        segment writes blocks [chunk_start/block, ...) and attends over everything before); rope, the SDPA chunk
        offset and the chunk page table follow chunk_start. carry_gdn: continue from the bound GDN state instead of
        zeros (resumed segment; needs actual_len >= 3 so the conv history of the next step comes from this segment).

        vis_row_offset (default 0): image/video placeholder tokens before this segment (a segment with chunk_start > 0
        of a multimodal prompt splices vision_tokens[vis_row_offset : ...]; see _vis_row_offset_for).

        bucket (default None): the masked big bucket (4096 / 8192, QWEN36_PREFILL_MASKED_BIG) serving `actual_len` rows.
        """
        tr = self._short_trace_for(actual_len, bucket=bucket)
        if tr is None:
            return None
        assert token_ids.shape[0] == 1 and token_ids.shape[1] >= actual_len, "short traced prefill is B=1"
        trash = self.prefill_trash_block if trash_block is None else int(trash_block)
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        cb0 = chunk_start // tr.block_size  # first block of this segment
        assert chunk_start % tr.block_size == 0
        nreal = num_blocks_in_seq(chunk_start + actual_len, tr.block_size)
        assert pt.shape[-1] >= nreal, "page_table shorter than the prompt (zero-padding would corrupt block 0)"
        row = pt.reshape(1, -1).to(torch.int32)
        # The segment's chunk window [cb0, cb0 + blocks_per_bucket) may reach past the full-table width.
        row_w = max(tr.buf_blocks, cb0 + tr.blocks_per_bucket)
        if row.shape[1] < row_w:
            row = torch.cat([row, torch.zeros(1, row_w - row.shape[1], dtype=torch.int32)], dim=1)
        elif row.shape[1] > row_w:
            row = row[:, :row_w]
        row = row.clone()
        if bool((row[0, :nreal] == trash).any()):
            if strict:
                raise AssertionError(
                    f"trash block {trash} is one of the request's own blocks {row[0, :nreal].tolist()}; set "
                    "model.prefill_trash_block to a block no request owns (vLLM: 0 = the null block)"
                )
            return None
        # Padded blocks (>= ceil(valid_len/block)) of the bucket: K/V writes AND SDPA reads go to the trash block.
        if nreal < cb0 + tr.blocks_per_bucket:
            row[0, nreal : cb0 + tr.blocks_per_bucket] = trash
        chunk_row = row[:, cb0 : cb0 + tr.blocks_per_bucket].contiguous()
        row = row[:, : tr.buf_blocks].contiguous()
        rep = ttnn.ReplicateTensorToMesh(self.device)

        # Per-request RoPE: M-RoPE for multimodal, else clears to 1D RoPE with rope_delta = 0 (decode reads it).
        # A resumed segment (chunk_start > 0) keeps the table the caller staged for the whole prompt.
        if chunk_start == 0:
            self._build_request_rope(token_ids[:, :actual_len], vision_tokens)
        self._prepare_short_trace_gdn_init(carry_gdn)
        toks = token_ids[:, :actual_len].to(torch.int32)
        if tr.bucket > actual_len:
            tok_buf = torch.cat([toks, torch.zeros(1, tr.bucket - actual_len, dtype=torch.int32)], dim=1)
        else:
            tok_buf = toks

        # Keep every host tensor alive until the trace has run (the copies may still be in flight).
        host_refs = []

        def _stage(host_t, dev_buf):
            host_refs.append(host_t)
            ttnn.copy_host_to_device_tensor(host_t, dev_buf)

        _stage(
            ttnn.from_torch(tok_buf, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep),
            tr.tok,
        )
        _stage(
            ttnn.from_torch(row, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep),
            tr.full_pt,
        )
        _stage(
            ttnn.from_torch(
                chunk_row,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
                mesh_mapper=rep,
            ),
            tr.chunk_pt,
        )
        # cos/sin over the whole bucket: plain 1D RoPE for text (host tensors cached per bucket), the staged
        # M-RoPE table otherwise.
        if chunk_start > 0:
            cos_t, sin_t = self._rope_tp_cos_sin_torch(chunk_start, tr.bucket)
            cos_host, sin_host = (
                ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep)
                for t in (cos_t, sin_t)
            )
        elif getattr(self.rope, "_req_cos", None) is None:
            if tr.text_rope_host is None:
                cos_t, sin_t = self._rope_tp_cos_sin_torch(0, tr.bucket)
                tr.text_rope_host = tuple(
                    ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep)
                    for t in (cos_t, sin_t)
                )
            cos_host, sin_host = tr.text_rope_host
        else:
            cos_t, sin_t = self._rope_tp_cos_sin_torch(0, tr.bucket)
            cos_host, sin_host = (
                ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep)
                for t in (cos_t, sin_t)
            )
        _stage(cos_host, tr.cos)
        _stage(sin_host, tr.sin)
        if tr.csi_host is None:
            tr.csi_host = ttnn.from_torch(
                torch.zeros(1, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
                mesh_mapper=rep,
            )
        if chunk_start > 0:
            _stage(
                ttnn.from_torch(
                    torch.tensor([chunk_start], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=None,
                    mesh_mapper=rep,
                ),
                tr.csi,
            )
        else:
            _stage(tr.csi_host, tr.csi)  # request-independent, re-DMA'd anyway so it never depends on other traces
        host = self._short_trace_host_inputs(tr, actual_len)
        _stage(host["sel"], tr.sel)
        if tr.masked:
            _stage(host["bg"], tr.m_bg)
            _stage(host["conv_sel"], tr.conv_sel)
            if tr.m_qkv is not None:
                _stage(host["qkv"], tr.m_qkv)
        if tr.spec:
            if spec_inputs is None:
                raise ValueError(f"bucket {tr.bucket} is a spec short trace: _replay_short_trace needs spec_inputs")
            mtp_tok, mtp_full, mtp_chunk = spec_inputs
            want = ((1, tr.bucket), (1, tr.buf_blocks), (1, tr.blocks_per_bucket))
            got = tuple(tuple(t.shape) for t in (mtp_tok, mtp_full, mtp_chunk))
            if got != want:
                raise ValueError(f"spec_inputs shapes {got} != {want} (mtp tokens, mtp full table, mtp chunk table)")
            for t, buf, dt in (
                (mtp_tok, tr.mtp_tok, ttnn.uint32),
                (mtp_full, tr.mtp_full_pt, ttnn.int32),
                (mtp_chunk, tr.mtp_chunk_pt, ttnn.int32),
            ):
                _stage(
                    ttnn.from_torch(
                        t.to(torch.int32).contiguous(),
                        dtype=dt,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=None,
                        mesh_mapper=rep,
                    ),
                    buf,
                )
        elif spec_inputs is not None:
            raise ValueError(f"bucket {tr.bucket} is not a spec short trace, but spec_inputs were given")
        # Vision-splice mask (host->device copy only; clears it for text-only requests).
        self._set_vision_merge(toks, vision_tokens, vis_row_offset)

        ttnn.execute_trace(self.device, tr.trace_id, cq_id=0, blocking=False)
        # Drain before reading (and before releasing the host tensors the staging copies read from).
        ttnn.synchronize_device(self.device)
        host_refs.clear()
        return tr

    def prefill_short_traced(self, token_ids, page_table, actual_len, vision_tokens=None, trash_block=None):
        """Greedy first token of a prompt of `actual_len` tokens (any length < 2048 whose bucket was captured) via
        the short-prompt trace. Returns the token id (python int) read from two tiny per-shard tensors, or None
        when no trace serves this length — the caller then falls back to prefill_traced_chunked (which returns
        logits). The caller keeps the full-logits path when sampling.

        Per request: stage the rope (also sets rope_delta for decode), DMA tokens / page tables (padded blocks ->
        the trash block) / cos / sin / csi / one-hot / GDN masks / vision mask into the persistent buffers,
        execute_trace, read idx+max. The GDN state reset happens inside the trace; the paged KV + GDN state are
        left exactly as the eager masked-bucket path leaves them (KV for blocks [0, ceil(len/block)), recurrent +
        conv state at position `actual_len`). `trash_block` (default model.prefill_trash_block) must not be one of
        the request's own blocks."""
        tr = self._replay_short_trace(token_ids, page_table, actual_len, vision_tokens, trash_block)
        if tr is None:
            return None
        token = self.read_shard_argmax_token(tr.idx, tr.val)
        assert 0 <= token < self.vocab_size, f"short prefill produced out-of-vocab token id {token}"
        return token

    def prefill_short_traced_logits(
        self, token_ids, page_table, actual_len, vision_tokens=None, trash_block=None, strict=True
    ):
        """Like prefill_short_traced but returns the last-row logits as a host torch float tensor [1, 1, vocab]
        (the vocab shards are read once with ConcatMeshToTensor(dim=-1): no all-gather, one D2H of the pre-gather
        [1,1,1,V/TP] output), or None when no trace serves this request."""
        tr = self._replay_short_trace(token_ids, page_table, actual_len, vision_tokens, trash_block, strict=strict)
        if tr is None:
            return None
        if self._logits_read_comp is None:
            self._logits_read_comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)
        lg = ttnn.to_torch(tr.logits, mesh_composer=self._logits_read_comp)
        return lg.reshape(1, 1, -1)[..., : self.vocab_size].float()

    # ----------------------------------------------------------------------- #
    # Traced batched SHORT-prompt prefill (B=32 / ISL<=128)
    # ----------------------------------------------------------------------- #
    # The traced chunk body and GDN chunk-seq kernel are B=1 (the kernel caps
    # BH=B*Nv_tp at ~32 => B<=4 at TP=4). So capture ONE B=1 full-bucket(128) trace and
    # replay it once per user: each replay DMAs the user's token slice + page-table row
    # into persistent buffers, execute_trace, then copy the B=1 GDN state into row u of
    # the batched [B,...] decode buffer. Full bucket => valid_len=None (trace-safe; a
    # one-hot mask in-trace TT_FATALs). Avoids per-layer host dispatch and per-op from_torch.

    def _alloc_gdn_scratch_b1(self):
        """Allocate a dedicated B=1 GDN state set on every GDN layer, distinct from the
        batched [B,...] decode buffer. Returns the prior batched bindings for the caller to
        restore for decode. MUST run before trace capture (allocates buffers)."""
        prev = []
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            prev.append(
                (
                    dn,
                    dn.B,
                    dn.rec_state,
                    dn.conv_states,
                    dn.conv_carry,
                    dn._zero_conv0,
                    # _zero_rec is B-shaped like _zero_conv0 and travels with the binding: reset_state_inplace
                    # copies it into rec_state, and a [B,...] zero over a [1,...] scratch (or the reverse) is a
                    # hard shape TT_FATAL.
                    dn._zero_rec,
                )
            )
            # reset_state allocates against self.B, so set B=1 first.
            dn.B = 1
            dn.reset_state()  # builds rec_state [1,Nv,Dk,Dv], conv_states[*] [1,1,D], conv_carry, _zero_conv0
        return prev

    def _restore_gdn_batched(self, prev):
        """Restore the batched [B,...] GDN bindings saved by _alloc_gdn_scratch_b1 and free
        the B=1 scratch, so decode reads the assembled batched state."""
        for dn, B_b, rec_b, conv_b, carry_b, zero0_b, zerorec_b in prev:
            # Free the B=1 scratch allocated for the prefill trace.
            if dn.rec_state is not None:
                ttnn.deallocate(dn.rec_state)
            for cs in dn.conv_states or []:
                ttnn.deallocate(cs)
            if dn.conv_carry is not None:
                ttnn.deallocate(dn.conv_carry)
            if dn._zero_conv0 is not None:
                ttnn.deallocate(dn._zero_conv0)
            if dn._zero_rec is not None:
                ttnn.deallocate(dn._zero_rec)
            # Rebind the batched decode buffers.
            dn.B = B_b
            dn.rec_state = rec_b
            dn.conv_states = conv_b
            dn.conv_carry = carry_b
            dn._zero_conv0 = zero0_b
            dn._zero_rec = zerorec_b

    def _ensure_gdn_prefill_scratch(self):
        """Allocate the PERSISTENT B=1 GDN prefill scratch once (idempotent).

        Unlike _alloc_gdn_scratch_b1 (throwaway, freed by _restore_gdn_batched), this scratch lives
        for the server lifetime: the batched chunk-prefill trace bakes its buffer addresses at warmup
        and every prefill_paged_slots replay reuses them, so it must never be freed/reallocated (only
        zeroed in place via _reset_gdn_state_for_new_sequence). Allocate at warmup so no device buffer
        is allocated at request time (which would be unsafe under the parked decode trace)."""
        if self._gdn_prefill_scratch is not None:
            return
        scratch = []
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            # reset_state allocates fresh B=1 buffers and assigns them WITHOUT freeing the current
            # (batched) ones, so save+restore the batched bindings and keep the scratch handles alive.
            # _zero_rec is saved and restored with the rest of the binding (see _alloc_gdn_scratch_b1).
            saved = (
                dn.B,
                dn.rec_state,
                dn.conv_states,
                dn.conv_carry,
                dn._zero_conv0,
                dn._zero_rec,
            )
            dn.B = 1
            dn.reset_state()  # builds rec_state [1,Nv,Dk,Dv], conv_states[*] [1,1,D], conv_carry, _zero_conv0
            scratch.append((dn, dn.rec_state, dn.conv_states, dn.conv_carry, dn._zero_conv0, dn._zero_rec))
            if dn._prefill_init_rec is None:  # short-trace init buffers (B=1 scratch shapes), zeros
                dn._prefill_init_rec = alloc_zeros_like(dn.rec_state, dn.mesh)
                dn._prefill_init_conv_carry = alloc_zeros_like(dn.conv_carry, dn.mesh)
            (
                dn.B,
                dn.rec_state,
                dn.conv_states,
                dn.conv_carry,
                dn._zero_conv0,
                dn._zero_rec,
            ) = saved
        self._gdn_prefill_scratch = scratch

    def _bind_gdn_prefill_scratch(self):
        """Bind the persistent B=1 prefill scratch onto every GDN layer (prefill runs B=1); returns the
        saved batched decode bindings for _unbind_gdn_prefill_scratch. Allocates the scratch on first use.
        The drop-in analogue of _alloc_gdn_scratch_b1 that reuses one persistent scratch instead of
        allocating a throwaway per call (so the chunk trace's baked addresses stay valid)."""
        self._ensure_gdn_prefill_scratch()
        prev = []
        for dn, rec, conv, carry, zero0, zerorec in self._gdn_prefill_scratch:
            prev.append(
                (
                    dn,
                    dn.B,
                    dn.rec_state,
                    dn.conv_states,
                    dn.conv_carry,
                    dn._zero_conv0,
                    dn._zero_rec,
                )
            )
            dn.B = 1
            dn.rec_state = rec
            dn.conv_states = conv
            dn.conv_carry = carry
            dn._zero_conv0 = zero0
            # The scratch's own [1,...] zero state: reset_state_inplace copies it into the bound rec_state.
            dn._zero_rec = zerorec
        return prev

    def _unbind_gdn_prefill_scratch(self, prev):
        """Rebind the batched [B,...] decode buffers saved by _bind_gdn_prefill_scratch, WITHOUT freeing
        the persistent scratch (unlike _restore_gdn_batched, whose scratch is throwaway)."""
        for dn, B_b, rec_b, conv_b, carry_b, zero0_b, zerorec_b in prev:
            dn.B = B_b
            dn.rec_state = rec_b
            dn.conv_states = conv_b
            dn.conv_carry = carry_b
            dn._zero_conv0 = zero0_b
            dn._zero_rec = zerorec_b

    def _snapshot_gdn_scratch(self):
        """Snapshot the B=1 GDN scratch (host torch) to restore around the throwaway capture run."""
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        out = []
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            out.append(
                (
                    ttnn.to_torch(dn.rec_state, mesh_composer=comp),
                    [ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states],
                )
            )
        return out

    def _restore_gdn_scratch(self, snap):
        """Restore the B=1 GDN scratch in place (preserving the addresses the trace baked in)
        from a _snapshot_gdn_scratch result."""
        mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)

        def _back(t, dtype):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device, mesh_mapper=mapper)

        for layer, (rec, convs) in zip((l for l in self.layers if not l.is_full_attention), snap):
            dn = layer.attention
            r = _back(rec, dn.rec_state.dtype)
            ttnn.copy(r, dn.rec_state)
            ttnn.deallocate(r)
            for j, c in enumerate(convs):
                cc = _back(c, dn.conv_states[j].dtype)
                ttnn.copy(cc, dn.conv_states[j])
                ttnn.deallocate(cc)

    def capture_prefill_trace_bucket(self, device, page_table, bucket=128):
        """Capture ONE B=1 full-bucket prefill trace (all-layer forward, valid_len=None) for batched
        serving of prompts whose length is EXACTLY the bucket. GDN points at a B=1 scratch. Replay
        once per user via prefill_traced_bucket_batched. Only full-bucket prompts are traced:
        valid_len cannot be masked inside a trace, so a short prompt would pad through the GDN
        recurrence and corrupt decode state. Callers route actual_len < bucket to eager
        prefill_paged_peruser. ``page_table`` torch [1, bpu] int32 — one user's row (buffer width
        fixed; each replay DMAs a different user's row). ``bucket`` must be a multiple of 128.
        """
        assert self.num_devices > 1, "capture_prefill_trace_bucket is the TP (num_devices>1) path"
        assert self._paged_kv_caches is not None, "Call allocate_kv_caches first"
        assert bucket % 128 == 0, f"bucket {bucket} must be a multiple of 128 (GDN sub-chunk)"
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_bucket = bucket // block_size

        if getattr(self, "_bucket_trace_id", None) is not None:
            ttnn.release_trace(device, self._bucket_trace_id)
            self._bucket_trace_id = None
        self._bucket_size = bucket
        # _forward_prefill_chunk_tp sizes its reshape/loop from _chunked_chunk_size; point it
        # at the bucket. Save the prior value so release_prefill_trace_bucket can restore it
        # (a later chunked prefill on the same model must not be left at 128).
        self._chunked_chunk_size_prebucket = self._chunked_chunk_size
        self._chunked_chunk_size = bucket

        # Swap GDN to a dedicated B=1 scratch for capture + replay (the trace writes B=1).
        self._gdn_batched_prev = self._alloc_gdn_scratch_b1()

        rep = ttnn.ReplicateTensorToMesh(device)
        B = 1
        # Full-page-table buffer width MUST be a 32-multiple >= the SDPA's target_blocks so
        # forward_prefill_paged's zero-PAD branch (ttnn.zeros + ttnn.concat — a host write that
        # TT_FATALs in a trace) never runs during replay. Replays' _fit_pt_row pad to this width.
        buf_blocks = max(32, ((page_table.shape[-1] + 31) // 32) * 32)
        if page_table.shape[-1] < buf_blocks:
            page_table = torch.cat(
                [page_table, torch.zeros(page_table.shape[0], buf_blocks - page_table.shape[-1], dtype=torch.int32)],
                dim=1,
            )
        self._bucket_buf_blocks = buf_blocks
        # Persistent per-replay input buffers (replicated; addresses baked into the trace).
        self._bucket_token_buf = ttnn.from_torch(
            torch.zeros(B, bucket, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        self._bucket_start_idx_tensor = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        self._bucket_full_page_table_buf = ttnn.from_torch(
            page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, mesh_mapper=rep
        )
        self._bucket_page_table_buf = ttnn.from_torch(
            page_table[:, :blocks_per_bucket].contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        cos_t, sin_t = self._rope_tp_cos_sin_torch(0, bucket)
        self._bucket_cos_buf = ttnn.from_torch(
            cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=rep
        )
        self._bucket_sin_buf = ttnn.from_torch(
            sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=rep
        )

        # Warmup OUTSIDE the trace: compile every program the capture + replays run (a compile
        # during replay would clobber the parked trace). Two sets: (a) the full-bucket forward
        # (the trace body); (b) the logit-select run eagerly after each replay.
        self._reset_gdn_state_for_new_sequence()
        warmup_out = self._forward_prefill_chunk_tp(
            self._bucket_token_buf,
            self._bucket_cos_buf,
            self._bucket_sin_buf,
            self._bucket_start_idx_tensor,
            self._bucket_full_page_table_buf,
            self._bucket_page_table_buf,
        )
        # Warm the logit-select at actual_len=bucket. Its program is fixed per bucket; actual_len
        # only changes the one-hot values (a host write), so one warmup covers every actual_len.
        warm_logits = self._masked_bucket_logits_tp(warmup_out, bucket, bucket)
        ttnn.deallocate(warm_logits)
        ttnn.deallocate(warmup_out)
        ttnn.synchronize_device(device)

        # Capture: snapshot the B=1 scratch, run the throwaway compile+capture passes, restore
        # so the addresses the trace baked in stay valid (both passes advance the in-place GDN
        # recurrence; KV at block 0 is harmlessly overwritten by the first real replay).
        self._reset_gdn_state_for_new_sequence()
        gdn_snap = self._snapshot_gdn_scratch()
        self._forward_prefill_chunk_tp(
            self._bucket_token_buf,
            self._bucket_cos_buf,
            self._bucket_sin_buf,
            self._bucket_start_idx_tensor,
            self._bucket_full_page_table_buf,
            self._bucket_page_table_buf,
        )
        self._bucket_trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self._bucket_trace_output = self._forward_prefill_chunk_tp(
            self._bucket_token_buf,
            self._bucket_cos_buf,
            self._bucket_sin_buf,
            self._bucket_start_idx_tensor,
            self._bucket_full_page_table_buf,
            self._bucket_page_table_buf,
        )
        ttnn.end_trace_capture(device, self._bucket_trace_id, cq_id=0)
        self._restore_gdn_scratch(gdn_snap)
        logger.info(f"Bucket({bucket}) prefill trace (TP) captured successfully!")

    def release_prefill_trace_bucket(self):
        """Release the captured bucket prefill trace + persistent buffers and restore the
        batched GDN bindings for decode. Called after prefill_traced_bucket_batched."""
        if getattr(self, "_bucket_trace_id", None) is not None:
            ttnn.release_trace(self.device, self._bucket_trace_id)
            self._bucket_trace_id = None
        for buf in (
            getattr(self, "_bucket_token_buf", None),
            getattr(self, "_bucket_start_idx_tensor", None),
            getattr(self, "_bucket_full_page_table_buf", None),
            getattr(self, "_bucket_page_table_buf", None),
            getattr(self, "_bucket_cos_buf", None),
            getattr(self, "_bucket_sin_buf", None),
        ):
            if buf is not None:
                ttnn.deallocate(buf)
        self._bucket_token_buf = None
        self._bucket_start_idx_tensor = None
        self._bucket_full_page_table_buf = None
        self._bucket_page_table_buf = None
        self._bucket_cos_buf = None
        self._bucket_sin_buf = None
        self._bucket_trace_output = None
        # Restore _chunked_chunk_size (capture pointed it at the bucket) for a later chunked prefill.
        if hasattr(self, "_chunked_chunk_size_prebucket"):
            self._chunked_chunk_size = self._chunked_chunk_size_prebucket
            del self._chunked_chunk_size_prebucket
        # Restore the batched [B,...] GDN decode buffers the prefill assembled into.
        if getattr(self, "_gdn_batched_prev", None) is not None:
            self._restore_gdn_batched(self._gdn_batched_prev)
            self._gdn_batched_prev = None

    def prefill_traced_bucket_batched(self, token_ids_list, page_table, valid_lens=None):
        """Traced batched short-prompt prefill: replay the captured B=1 full-bucket trace once per
        user, stitching each replay's B=1 GDN state into row u of the batched [B,...] decode buffer.
        Attention fills each user's physical blocks via the per-user page-table row (batch_idx=0
        baked into the trace). Returns a list of B device logits [1, 1, vocab]. Every user's
        actual_len MUST equal the captured bucket: the trace runs valid_len=None (no GDN mask), so
        a short prompt padded to the bucket would push padding through the GDN recurrence and
        corrupt decode state. Short prompts go to eager prefill_paged_peruser. Asserts
        actual_len == bucket and never pads. Call capture_prefill_trace_bucket first;
        release_prefill_trace_bucket before decode.
        """
        assert self.num_devices > 1, "prefill_traced_bucket_batched is the TP (num_devices>1) path"
        assert getattr(self, "_bucket_trace_id", None) is not None, "Call capture_prefill_trace_bucket first"
        bucket = self._bucket_size
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_bucket = bucket // block_size
        rep = ttnn.ReplicateTensorToMesh(self.device)

        B = len(token_ids_list)
        page_table_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert page_table_torch.shape[0] == B, "page_table must have one row per user"

        # Pad/clip each request's page-table row to the captured buffer width. Keep the
        # [1, buf_blocks] batch dim — copy_host_to_device_tensor requires identical logical shapes.
        buf_blocks = int(self._bucket_full_page_table_buf.shape[-1])

        def _fit_pt_row(row):
            row = row.reshape(1, -1)  # [1, bpu] -> ensure 2D
            if row.shape[1] < buf_blocks:
                row = torch.cat([row, torch.zeros(1, buf_blocks - row.shape[1], dtype=row.dtype)], dim=1)
            elif row.shape[1] > buf_blocks:
                row = row[:, :buf_blocks]
            return row.contiguous()

        # Per-user logits are read to HOST during the loop and re-uploaded at the end: each replay
        # overwrites the persistent trace output, and the post-loop assembly churns device memory.
        host_logits = []  # torch [1, 1, vocab] (one replica) per user
        # Collect each replay's B=1 GDN state to assemble into the batched decode buffer. Snapshot
        # via a host to_torch round trip (NOT ttnn.clone() — that allocates from the same general
        # device pool the captured trace's own baked-address intermediates draw from, so the next
        # user's execute_trace() silently overwrites the "cloned" snapshot; confirmed via
        # checksumming a live tensor changing value with nothing writing to it, ruled out as a race
        # since an added synchronize_device() didn't change the deterministic wrong result).
        per_user_rec = []
        per_user_conv = []
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        dn_states = [layer.attention for layer in self.layers if not layer.is_full_attention]

        try:
            for u in range(B):
                toks = token_ids_list[u]
                assert toks.shape[0] == 1, f"user {u}: token_ids must be [1, T_u]"
                actual = valid_lens[u] if valid_lens is not None else toks.shape[1]
                # CORRECTNESS: traced path serves ONLY full-bucket prompts (see docstring); short
                # prompts must be routed to eager prefill_paged_peruser.
                assert actual == bucket, (
                    f"user {u}: actual_len {actual} != bucket {bucket}; the traced bucket prefill "
                    f"only serves full-bucket prompts — route short prompts to prefill_paged_peruser"
                )

                # Zero the B=1 GDN scratch before each user (address-stable; the trace baked these in).
                self._reset_gdn_state_for_new_sequence()

                # Full bucket, no padding (actual == bucket).
                token_buf = toks[:, :bucket].to(torch.int32)

                # DMA this user's inputs into the persistent buffers (addresses preserved).
                tok_host = ttnn.from_torch(
                    token_buf, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep
                )
                ttnn.copy_host_to_device_tensor(tok_host, self._bucket_token_buf)

                row = _fit_pt_row(page_table_torch[u])
                pt_host = ttnn.from_torch(
                    row, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep
                )
                ttnn.copy_host_to_device_tensor(pt_host, self._bucket_full_page_table_buf)
                cpt_host = ttnn.from_torch(
                    row[:, :blocks_per_bucket].contiguous(),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=None,
                    mesh_mapper=rep,
                )
                ttnn.copy_host_to_device_tensor(cpt_host, self._bucket_page_table_buf)

                # chunk_start=0 and cos/sin for [0, bucket) are baked in; no per-replay DMA needed.
                ttnn.execute_trace(self.device, self._bucket_trace_id, cq_id=0, blocking=False)
                # Sync before reading this user's state/logit: the trace writes hidden/rec_state/
                # conv_states in place and the next replay would overwrite them.
                ttnn.synchronize_device(self.device)

                # Gather this replay's B=1 GDN state for assembly after the loop. The next replay
                # resets the scratch IN PLACE, so snapshot now via host to_torch (see comment above).
                per_user_rec.append([ttnn.to_torch(dn.rec_state, mesh_composer=comp) for dn in dn_states])
                per_user_conv.append(
                    [[ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states] for dn in dn_states]
                )

                # Logit at actual_len-1, read to HOST immediately (before the next replay overwrites
                # the trace output). Re-uploaded at the end.
                lg = self._masked_bucket_logits_tp(self._bucket_trace_output, actual, bucket)
                host_logits.append(ttnn.to_torch(lg, mesh_composer=comp)[0:1].clone())  # [1,1,vocab] one replica
                ttnn.deallocate(lg)

            ttnn.synchronize_device(self.device)
        finally:
            # Rebind the batched buffers regardless of success or failure (restore was deferred so
            # the loop could use the B=1 scratch) — otherwise a mid-loop assertion/exception (e.g. a
            # short prompt hitting the actual_len == bucket check) leaves every GDN layer pointed at
            # the B=1 scratch instead of the batched decode buffers.
            if getattr(self, "_gdn_batched_prev", None) is not None:
                self._restore_gdn_batched(self._gdn_batched_prev)
                self._gdn_batched_prev = None

        # Assemble the per-user states into row u in place.
        self._assemble_per_user_gdn(per_user_rec, per_user_conv)

        # Re-upload the per-user logits as stable device tensors after all allocations.
        return self._reupload_host_logits(host_logits)

    def _assemble_per_user_gdn(self, per_user_rec, per_user_conv):
        """Stitch B per-user B=1 GDN states (host torch) into row u of the batched [B,...] decode
        buffers via assemble_batched_state. The batched GDN bindings MUST already be rebound (writes
        in place). Shared by prefill_traced_bucket_batched/prefill_chunked_peruser.

        per_user_rec[u][li]:  host rec_state snapshot for user u, GDN layer li (mesh dim 0 = devices).
        per_user_conv[u][li]: list of K host conv_states snapshots for user u, GDN layer li.
        """
        B = len(per_user_rec)
        mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)
        dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]
        for li, dn in enumerate(dn_layers):
            K = dn.K
            D = dn.qkv_dim_tp
            rec_list = [
                ttnn.from_torch(
                    per_user_rec[u][li],
                    dtype=dn.rec_state.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    mesh_mapper=mapper,
                )
                for u in range(B)
            ]
            # Rebuild the [1, K-1, D] carry tensor the assembler expects from the per-slot
            # conv_states snapshot (slots 1..K-1; slot 0 is the shifted-out zero). Each slot
            # snapshot is [1, 1, D]; concat along dim 1 -> [1, K-1, D].
            conv_carry_list = []
            staging = []  # (per-slot tensors) to deallocate after assembly
            for u in range(B):
                slots = [
                    ttnn.from_torch(
                        per_user_conv[u][li][m],
                        dtype=dn.conv_states[m].dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                        mesh_mapper=mapper,
                    )
                    for m in range(1, K)
                ]
                staging.extend(slots)
                conv_carry_list.append(ttnn.concat(slots, dim=1) if K - 1 > 1 else ttnn.reshape(slots[0], (1, 1, D)))
            # assemble_batched_state takes ownership of rec_list + conv_carry_list and deallocates
            # them (gdn/tp.py); we only free the per-slot staging tensors it never sees.
            dn.assemble_batched_state(rec_list, conv_carry_list)
            for t in staging:
                ttnn.deallocate(t)

    def _reupload_host_logits(self, host_logits):
        """Re-upload per-user host logits (read to host during a batched prefill loop) as stable
        replicated device tensors [1, 1, vocab] — the prefill_paged_peruser return contract.
        Done after all per-user execute/assembly allocations so the returned tensors are stable."""
        return [
            ttnn.from_torch(
                hl,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            for hl in host_logits
        ]

    def prefill_paged_slots(
        self, token_ids_list, page_table, empty_slots, valid_lens=None, start_positions=None, vision_tokens_list=None
    ):
        """vLLM continuous-batching TP prefill: prefill each new request into ITS decode slot.

        QWEN36_PREFILL_BUCKET_TRACE=1 (default): each user's prompt runs against the persistent B=1 GDN scratch
        (traced short prefill when its bucket is captured, else the unchanged eager/chunked path), and the scratch
        state is then written into the user's slot with DEVICE ops (write_slot from the scratch tensors, no host
        round trip) before the next user overwrites the scratch. "0": the previous host-snapshot implementation
        (_prefill_paged_slots_host), byte for byte. Same contract either way (see _prefill_paged_slots_host).

        start_positions (optional, per request): number of leading tokens whose paged KV is already in the request's
        page-table row (vLLM automatic prefix caching). With a GdnPrefixStateCache attached (self._prefix_cache) the
        prefill resumes from the largest cached GDN snapshot at or below it (_prefill_with_prefix_cache); otherwise it
        is ignored and the prompt is recomputed from 0 (correct, slower). Requires the traced path (default).

        vLLM chunked prefill (supports_chunked_prefill): a long prompt arrives over several engine steps, each row
        holding the prompt from position 0, valid_lens[u] = this chunk's END and start_positions[u] = the tokens already
        prefilled. A row that may be a non-final chunk (_maybe_partial_chunk) snapshots its GDN state at its end into
        self._chunk_state_cache, and the next chunk resumes from that snapshot (same planner as the APC resume).

        vision_tokens_list (optional, per request): None (text), or a CALLABLE returning the request's packed vision
        embeddings (ttnn [n_vision_tokens, H]) -- called right before that user's prefill because the vision tower
        stashes MODEL-GLOBAL grids (_req_image_grid_thw / _req_video_grid_thw) that the prefill's M-RoPE build reads --
        and deallocated right after it. A multimodal user bypasses the prefix cache entirely (start position ignored,
        no snapshot lookup, no saves). After every user's prefill its M-RoPE delta (0 for text) is recorded per decode
        slot (set_slot_rope_delta) for the batched decode rope index."""
        if vision_tokens_list is not None and not self.short_prefill_trace_enabled():
            raise NotImplementedError("batched multimodal prefill requires QWEN36_PREFILL_BUCKET_TRACE=1")
        if not self.short_prefill_trace_enabled():
            return self._prefill_paged_slots_host(token_ids_list, page_table, empty_slots, valid_lens=valid_lens)
        assert self.num_devices > 1, "prefill_paged_slots is the TP (num_devices>1) path"
        N = len(token_ids_list)
        assert len(empty_slots) == N, "one slot per request"
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert pt.shape[0] == N, "page_table must have one row per request"
        # Tokens this engine step prefills (vLLM chunked prefill cuts a prompt only when the step's budget runs out).
        ends = [int(valid_lens[u]) if valid_lens is not None else token_ids_list[u].shape[1] for u in range(N)]
        step_tokens = sum(ends) - (sum(max(int(p), 0) for p in start_positions) if start_positions is not None else 0)
        host_logits = []
        for u in range(N):
            toks = token_ids_list[u]
            assert toks.shape[0] == 1, f"request {u}: token_ids must be [1, T_u]"
            actual = int(valid_lens[u]) if valid_lens is not None else toks.shape[1]
            assert actual >= 1, f"request {u}: empty prompt (actual_len={actual})"
            # Bind the persistent B=1 scratch for THIS user only (the traces baked its addresses at warmup);
            # always rebind the batched decode buffers (a mid-loop assert must not leave GDN on the scratch).
            vt_src = vision_tokens_list[u] if vision_tokens_list is not None else None
            vt = None
            prev = self._bind_gdn_prefill_scratch()
            try:
                if vt_src is not None:
                    # Multimodal: vision tower right before THIS user's prefill (model-global grid stash), prefix cache
                    # bypassed (plan None; the cache path is text-only and would rebuild a text rope / save snapshots).
                    vt = vt_src() if callable(vt_src) else vt_src
                    start, plan = 0, None
                else:
                    start = int(start_positions[u]) if start_positions is not None else 0
                    cont = self._maybe_partial_chunk(step_tokens, actual - max(start, 0), actual)
                    plan = self._plan_prefix_prefill(toks[:, :actual], actual, start, cont=cont)
                if vt is not None:
                    host_logits.append(
                        self.prefill_traced_chunked(
                            toks[:, :actual],
                            pt[u : u + 1],
                            actual_len=actual,
                            vision_tokens=vt,
                            return_host_logits=True,
                        )
                    )
                elif plan is None:  # cache off / nothing to resume or save: the unchanged path
                    host_logits.append(
                        self.prefill_traced_chunked(
                            toks[:, :actual], pt[u : u + 1], actual_len=actual, return_host_logits=True
                        )
                    )
                else:
                    host_logits.append(self._prefill_with_prefix_cache(toks[:, :actual], pt[u : u + 1], actual, plan))
            finally:
                self._unbind_gdn_prefill_scratch(prev)
                if vt is not None:
                    ttnn.deallocate(vt)
            # This user's M-RoPE delta (build_request_rope set it for MM, cleared it to 0 for text) follows its slot.
            self.set_slot_rope_delta(int(empty_slots[u]), self.rope.rope_delta)
            # Device-to-device: scratch (B=1) -> row empty_slots[u] of the batched buffers, other rows untouched. All
            # temporaries are consumed here, before the next user's replay can alias them.
            self._write_gdn_slot_from_scratch(int(empty_slots[u]))
        return host_logits

    # ----------------------------------------------------------------------- #
    # Prefix-cached (resumable) prefill, see prefix_cache.py
    # ----------------------------------------------------------------------- #
    # Resume offsets and snapshot positions are multiples of 128: the SDPA chunk offset uses q_chunk=128 integer
    # division, and the GDN chunk kernel works on 128-token sub-chunks. Every segment keeps >= 3 tokens at the end of
    # the prompt (the traced short path selects the conv new-state from the segment's own last 3 rows).
    _PREFIX_ALIGN = 128

    def _maybe_partial_chunk(self, step_tokens, chunk_len, end):
        """Can this prefill row be a NON-final chunk of a prompt vLLM split across engine steps (chunked prefill)? The
        model is not told. The scheduler cuts a prompt only when the step's token budget (or long_prefill_token_threshold)
        runs out, and the plugin floors a non-final chunk end to prefill_chunk_alignment (128). A wrong False costs a
        recompute on the next chunk (resumed from an earlier snapshot, or 0); a wrong True a ~3 ms snapshot."""
        if self._chunk_state_cache is None or end % self._PREFIX_ALIGN:
            return False
        budget, thr, align = self._chunk_budget
        if budget is None:  # no vLLM config (tests): snapshot every aligned end
            return True
        slack = max(int(align), self._PREFIX_ALIGN)
        return step_tokens > budget - slack or (thr > 0 and chunk_len > thr - slack)

    def _plan_prefix_prefill(self, toks, L, start, cont=False):
        """(resume_pos s, (store, slot) or None, sorted save positions, stride, vLLM start, chunk save points, cont) for
        one request, or None when there is neither a snapshot to resume from nor one to take (-> the legacy path).

        Two snapshot stores: the APC cache (self._prefix_cache, an LRU shared by all requests) and the chunked-prefill
        continuation store (self._chunk_state_cache). cont=True (a possible non-final chunk ending at L, a multiple of
        128): snapshot the state at L into the continuation store, where the next chunk (start = L) finds it."""
        cache, ccache = self._prefix_cache, self._chunk_state_cache
        cont = bool(cont) and ccache is not None and L % self._PREFIX_ALIGN == 0
        if (cache is None and ccache is None) or self._chunked_trace_id is None:
            return None
        A = self._PREFIX_ALIGN
        h128 = (max(start, 0) // A) * A
        while h128 > 0 and L - h128 < 3:
            h128 -= A
        s, src = 0, None
        if h128 > 0:
            for store in (ccache, cache):
                if store is not None:
                    p, slot = store.lookup(toks[0], h128)
                    if slot is not None and p > s:
                        s, src = p, (store, slot)
        points = set()
        chunk_pts = set()
        if cache is not None:
            if h128 > s:  # vLLM has the KV up to h128 but we hold no GDN state there: learn it for the next request
                points.add(h128)
            min_len = int(os.environ.get("QWEN36_PREFIX_SNAPSHOT_MIN_LEN", "8192"))
            b_end = (
                (L - 3) // A
            ) * A  # end-of-prompt snapshot: covers the whole-prompt-shared prefix of a later request
            # (not for a possible non-final chunk: L is not the prompt end, and the split costs an extra short replay)
            if L >= min_len and b_end > max(s, h128) and not cont:
                points.add(b_end)  # long prompts: split the prefill here (the end-of-prompt snapshot)
            elif not points:
                # Shorter prompts: no split. Save at the last full-chunk boundary through the per-chunk save_cb (a ~3 ms
                # snapshot between two chunk-trace replays) when one exists and leaves >= 3 tokens after it.
                chunk = self._chunked_chunk_size or 2048
                p = s + ((L - s) // chunk) * chunk
                if p > s and p <= L - 3:
                    chunk_pts.add(p)
        # A 1-2 token remainder after the last full chunk would run eagerly (a < 3-token resumed tail has no short
        # trace): split 128 tokens earlier so the last segment (128 + tail tokens) replays the short trace.
        chunk = self._chunked_chunk_size or 2048
        last = max(points, default=s)
        n_last = L - last
        if n_last > chunk and 0 < n_last % chunk < 3:
            points.add(last + (n_last // chunk) * chunk - A)
        stride = int(os.environ.get("QWEN36_PREFIX_SNAPSHOT_STRIDE", "8192"))
        if s == 0 and not points and not chunk_pts and not cont and (cache is None or L - 3 < stride):
            return None  # nothing to resume or save: legacy path
        return s, src, sorted(points), stride, start, chunk_pts, cont

    def _prefill_with_prefix_cache(self, toks, pt_row, L, plan):
        """Resumable text prefill of one request on the BOUND B=1 GDN scratch: restore the snapshot at s (or zero),
        run [s, p1), [p1, p2), ..., [pk, L) with _prefill_range (GDN state carries between segments), snapshot after the
        segments ending at the save points and after every `stride`-th chunk (APC cache), and at L when cont (chunked-
        prefill continuation store); returns host logits [1,1,vocab]."""
        cache = self._prefix_cache
        s, src, points, stride, vllm_start, chunk_pts, cont = plan
        tokens = toks[:1, :L].to(torch.int32)
        # Text-only: clears per-request rope (rope_delta = 0); resumed segments slice 1D RoPE by absolute position.
        self._build_request_rope(tokens, None)
        if s > 0:
            src[0].restore(src[1])
        saved = []

        def _save(p):
            if cache is not None:
                cache.save(tokens[0], p)
                saved.append(p)

        def _stride_cb(p):  # after each full chunk ending at p
            if cache is not None and p <= L - 3 and ((p - s) % stride == 0 or p in chunk_pts):
                _save(p)

        host = None
        seg_start = s
        bounds = points + [L]
        for i, seg_end in enumerate(bounds):
            last = i == len(bounds) - 1
            host = self._prefill_range(
                tokens,
                pt_row,
                seg_start,
                seg_end,
                reset_gdn=(seg_start == 0),
                want_logits=last,
                save_cb=_stride_cb,
            )
            if not last:
                _save(seg_end)
            seg_start = seg_end
        if cont:
            self._chunk_state_cache.save(tokens[0], L)
        if cache is not None:
            cache.log(L, vllm_start, s, sorted(set(saved)))
        return host

    def _prefill_range(self, tokens, page_table_row, start, end, reset_gdn, want_logits, save_cb=None):
        """Prefill tokens[start:end] at their ABSOLUTE positions on the bound B=1 GDN scratch: full 2048-token chunks
        replay the chunk trace (_prefill_traced_chunked_tp, start offset), the remainder (< 2048 tokens) replays the
        bucket's short trace with chunk_start (else the eager masked bucket). reset_gdn=True: the state starts from
        zeros (start must be 0), otherwise the bound state carries on. save_cb(p) fires after each full chunk. Returns
        host logits [1,1,vocab] at end-1 when want_logits, else None."""
        chunk = self._chunked_chunk_size or 2048
        n = end - start
        assert n >= 1 and start % self._PREFIX_ALIGN == 0 and (start == 0 or not reset_gdn)
        assert n >= 3 or (start == 0 and reset_gdn), f"segment [{start},{end}) < 3 tokens must start fresh at 0"
        num_full, tail = n // chunk, n % chunk
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        if num_full > 0:
            self._prefill_traced_chunked_tp(
                tokens,
                page_table_row,
                end,
                num_full,
                chunk,
                tail,
                start=start,
                reset_gdn=reset_gdn,
                save_cb=save_cb,
                run_tail=False,
            )
        if tail == 0:
            if not want_logits:
                return None
            lg = self._masked_bucket_logits_tp(self._chunked_trace_output, chunk, chunk)
            host = ttnn.to_torch(lg, mesh_composer=comp).reshape(-1, self.args.vocab_size)[:1].float().view(1, 1, -1)
            ttnn.deallocate(lg)
            return host
        ts = start + num_full * chunk
        fresh = reset_gdn and num_full == 0  # GDN zero at this segment's start
        tr = None
        if fresh or tail >= 3:  # a < 3-token resumed tail would lose the previous conv history (conv_sel rows < 0)
            tr = self._replay_short_trace(
                tokens[:, ts:end],
                page_table_row,
                tail,
                strict=False,
                chunk_start=ts,
                carry_gdn=not fresh,
            )
        if tr is not None:
            if not want_logits:
                return None
            if self._logits_read_comp is None:
                self._logits_read_comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)
            lg = ttnn.to_torch(tr.logits, mesh_composer=self._logits_read_comp)
            return lg.reshape(1, 1, -1)[..., : self.vocab_size].float()
        # Eager masked-bucket fallback (no short trace for this length / trash collision / < 3-token tail): the page
        # table is clipped/padded to the warmed width like the long-prompt tail path.
        pt = page_table_row
        buf_blocks = int(self._chunk_full_page_table_buf.shape[-1])
        if pt.shape[1] < buf_blocks:
            pt = torch.cat([pt, torch.zeros(pt.shape[0], buf_blocks - pt.shape[1], dtype=pt.dtype)], dim=1)
        elif pt.shape[1] > buf_blocks:
            pt = pt[:, :buf_blocks]
        lg = self.prefill_masked_bucket(tokens[:, ts:end], pt, actual_len=tail, chunk_start=ts)
        host = None
        if want_logits:
            host = ttnn.to_torch(lg, mesh_composer=comp).reshape(-1, self.args.vocab_size)[:1].float().view(1, 1, -1)
        ttnn.deallocate(lg)
        return host

    def _write_gdn_slot_from_scratch(self, slot):
        """Write the persistent B=1 prefill scratch state (device tensors, left intact) into decode `slot` of the
        batched GDN buffers via TPGatedDeltaNet.write_slot(consume=False). The batched buffers must be bound."""
        tid = self._gdn_slot_write_traces.get(slot)
        if tid is not None and self._gdn_slot_write_replayable():
            for dn, rec, _conv, _carry, _zero0, _zerorec in self._gdn_prefill_scratch:
                assert dn.rec_state is not rec, "write_slot_from_scratch needs the batched decode buffers bound"
                dn.sync_conv_taps()  # eager and a no-op unless taps are stale (write_slot does this before its writes)
            ttnn.execute_trace(self.mesh_device, tid, cq_id=0, blocking=False)
            if not self._gdn_slot_write_replayed:
                self._gdn_slot_write_replayed = True
                logger.info(f"GDN slot-write trace replay in use (first replay: slot {slot})")
            # Host-side state write_slot would have left behind in either captured format (packed: no conv_states / RM
            # history writes; "both" (spec engine): the taps and the RM history row are written by the trace itself).
            for dn, _rec, _conv, _carry, _zero0, _zerorec in self._gdn_prefill_scratch:
                dn._conv_win_stale = True
            return
        self._write_gdn_slot_eager(slot)

    def _write_gdn_slot_eager(self, slot):
        for dn, rec, conv, _carry, _zero0, _zerorec in self._gdn_prefill_scratch:
            assert dn.rec_state is not rec, "write_slot_from_scratch needs the batched decode buffers bound"
            dn.write_slot(slot, rec, conv, consume=False)

    def _gdn_slot_write_fingerprint(self, conv_fmt=None):
        """Every device address a slot-write trace bakes, per GDN layer: the batched rec_state and the B=1 scratch rec +
        conv taps, plus, under the "packed" format, the packed history and the shared pack tables, and under any other
        format (the spec engine's "both": write_slot writes conv_states and the RM history row) the K tap buffers and
        the RM history."""
        conv_fmt = self._gdn_slot_write_fmt if conv_fmt is None else conv_fmt
        fp = []
        for dn, rec, conv, _carry, _zero0, _zerorec in self._gdn_prefill_scratch:
            if conv_fmt == "packed":
                tab = dn._pack_tables()
                extra = (
                    dn._conv_hist_packed.buffer_address(),
                    tuple(
                        t.buffer_address()
                        for t in (tab["slot_to_packed"][0], tab["slot_to_packed"][1], tab["zero_row"])
                    ),
                )
            else:
                extra = (
                    tuple(c.buffer_address() for c in dn.conv_states),
                    None if dn._conv_hist_rm is None else dn._conv_hist_rm.buffer_address(),
                )
            fp.append(
                (
                    dn.rec_state.buffer_address(),
                    rec.buffer_address(),
                    tuple(c.buffer_address() for c in conv),
                    extra,
                )
            )
        return fp

    def _gdn_slot_write_replayable(self):
        """A captured slot-write trace is valid only while every layer is still in the conv format the capture ran under
        and none of the baked addresses moved (checked per replay; a mismatch warns once and the caller goes eager)."""
        fmt = self._gdn_slot_write_fmt
        if fmt is None or any(dn._conv_fmt != fmt for dn, *_ in self._gdn_prefill_scratch):
            return False
        fp = self._gdn_slot_write_fingerprint()
        if fp == self._gdn_slot_write_fp:
            return True
        if not self._gdn_slot_write_warned:
            self._gdn_slot_write_warned = True
            bad = next((i for i, (a, b) in enumerate(zip(fp, self._gdn_slot_write_fp)) if a != b), None)
            logger.warning(f"GDN slot-write trace invalidated: baked addresses moved (first at GDN layer {bad}); eager")
        return False

    def release_gdn_slot_write_traces(self):
        for tid in self._gdn_slot_write_traces.values():
            try:
                ttnn.release_trace(self.mesh_device, tid)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"release_trace failed for a slot-write trace: {e}")
        self._gdn_slot_write_traces = {}
        self._gdn_slot_write_fp = None
        self._gdn_slot_write_fmt = None

    def capture_gdn_slot_write_traces(self, conv_fmt="packed"):
        """Capture one trace per decode slot of the whole GDN slot write (every layer's in-place rec_state + conv-state
        write from the persistent B=1 scratch): ~15-30 host-dispatched ops per layer collapse into one execute_trace. Call
        after the decode traces are captured. No-op (logged once) unless the in-place path applies; any capture failure
        releases the partial traces and leaves the eager path in charge.

        conv_fmt: the GDN conv-state format the layers must be in, which selects write_slot's body: "packed" (plain
        serving decode: the in-place packed-history write) or "both" (the MTP spec engine, QWEN36_SPEC_FAST_PREFILL:
        conv_states + the RM history row, all in place). Replays are valid only while every layer stays in that format
        (_gdn_slot_write_replayable)."""
        import time

        if self._gdn_slot_write_traces or os.environ.get("QWEN36_GDN_SLOT_WRITE_TRACE", "1") != "1":
            return
        if conv_fmt not in ("packed", "both"):
            raise ValueError(f"capture_gdn_slot_write_traces: conv_fmt must be 'packed' or 'both', got {conv_fmt!r}")
        why = None
        if os.environ.get("QWEN36_GDN_INPLACE_SLOT_WRITE", "1") != "1":
            why = "QWEN36_GDN_INPLACE_SLOT_WRITE=0"
        elif not (self.num_devices > 1 and self.args.max_batch_size > 1):
            why = "not batched"
        elif self._gdn_prefill_scratch is None:
            why = "no B=1 prefill scratch"
        elif conv_fmt == "packed" and any(
            not (dn._conv_fmt == "packed" and dn._hist_hook_active() and dn._decode_B > 1)
            for dn, *_ in self._gdn_prefill_scratch
        ):
            why = "GDN layers are not all in the packed conv format"
        elif conv_fmt == "both" and any(
            not (dn._conv_fmt == "both" and dn._decode_B > 1 and dn._rec_row_layout_ok())
            for dn, *_ in self._gdn_prefill_scratch
        ):
            why = "GDN layers are not all in the 'both' conv format with a rank-4 TILE rec_state"
        if why is not None:
            logger.info(f"GDN slot-write trace capture skipped: {why}")
            return
        t0 = time.perf_counter()
        try:
            for dn, *_ in self._gdn_prefill_scratch:
                dn.sync_conv_taps()  # so the captured write_slot bodies contain no tap-sync ops
            for slot in range(self.args.max_batch_size):
                self._write_gdn_slot_eager(slot)  # warm: compiles + allocates everything before capture
                ttnn.synchronize_device(self.mesh_device)
                tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                try:
                    self._write_gdn_slot_eager(slot)
                finally:
                    ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
                self._gdn_slot_write_traces[slot] = tid
            ttnn.synchronize_device(self.mesh_device)
            self._gdn_slot_write_fmt = conv_fmt
            self._gdn_slot_write_fp = self._gdn_slot_write_fingerprint(conv_fmt)
        except Exception as e:  # noqa: BLE001
            logger.error(f"GDN slot-write trace capture failed ({e!r}); falling back to eager slot writes")
            self.release_gdn_slot_write_traces()
            return
        logger.info(
            f"Captured {len(self._gdn_slot_write_traces)} GDN slot-write traces ({conv_fmt} conv format) in "
            f"{time.perf_counter() - t0:.2f}s"
        )

    def _prefill_paged_slots_host(self, token_ids_list, page_table, empty_slots, valid_lens=None):
        """vLLM continuous-batching TP prefill: prefill each new request into ITS decode slot.

        The per-slot analogue of prefill_paged_peruser for online serving. Under vLLM a new
        request is prefilled while the other decode slots are live, so each user's B=1 state must
        land in row empty_slots[u] WITHOUT disturbing the others (GDN state is a fixed [B,...]
        buffer indexed by slot, not paged). Mirrors prefill_traced_bucket_batched's machinery —
        bind a B=1 GDN scratch, run the trace-safe pre-warmed masked-bucket prefill per user,
        snapshot its B=1 state — but writes each snapshot into its slot via write_slot (preserving
        the live rows) instead of assembling the whole batch. Attention fills each user's physical
        blocks via its page-table row (the same blocks decode reads via the decode page table).

        token_ids_list: list of N torch.Tensor [1, T_u].
        page_table:     torch.Tensor [N, max_blocks] int32 — row u = request u's blocks.
        empty_slots:    list of N ints — request u's persistent decode slot.
        valid_lens:     optional list of N real token counts (defaults to each T_u).
        Returns:        list of N host torch logits [1, 1, vocab_size] (one per request, in call order).

        Call allocate_kv_caches(batch_size=B) + the batched warmup first. Any prompt length is served:
        prefill_traced_chunked chunks long prompts via pre-warmed programs (no post-park compile).
        """
        assert self.num_devices > 1, "prefill_paged_slots is the TP (num_devices>1) path"
        N = len(token_ids_list)
        assert len(empty_slots) == N, "one slot per request"
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert pt.shape[0] == N, "page_table must have one row per request"
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        dn_states = [layer.attention for layer in self.layers if not layer.is_full_attention]
        dram = ttnn.DRAM_MEMORY_CONFIG

        host_logits = []
        for u in range(N):
            toks = token_ids_list[u]
            assert toks.shape[0] == 1, f"request {u}: token_ids must be [1, T_u]"
            actual = int(valid_lens[u]) if valid_lens is not None else toks.shape[1]
            assert actual >= 1, f"request {u}: empty prompt (actual_len={actual})"
            # Bind the persistent B=1 GDN prefill scratch: prefill runs B=1 and its per-sequence reset
            # would otherwise zero the batched [B,...] decode buffer (clobbering the live rows). This is
            # the SAME scratch whose addresses the chunk-prefill trace baked at warmup, so the traced
            # long-prompt path replays correctly; short prompts use the masked bucket on the same scratch.
            prev = self._bind_gdn_prefill_scratch()
            try:
                # prefill_traced_chunked runs short prompts in one masked-bucket forward and chunks longer ones.
                lg = self.prefill_traced_chunked(toks[:, :actual], pt[u : u + 1], actual_len=actual)
                host_logits.append(
                    ttnn.to_torch(lg, mesh_composer=comp).reshape(-1, self.args.vocab_size)[:1].float().view(1, 1, -1)
                )
                ttnn.deallocate(lg)
                # write_slot consumes these clones below, before the next request's chunk-trace replay
                # could overwrite them (see prefill_traced_bucket_batched).
                snap = [
                    (
                        ttnn.clone(dn.rec_state, memory_config=dram),
                        [ttnn.clone(c, memory_config=dram) for c in dn.conv_states],
                    )
                    for dn in dn_states
                ]
            finally:
                # Always rebind the batched decode buffers (a mid-call assert must not leave GDN on scratch).
                # Does NOT free the scratch — it persists for the next request and keeps the trace valid.
                self._unbind_gdn_prefill_scratch(prev)
            for dn, (rec, convs) in zip(dn_states, snap):
                dn.write_slot(int(empty_slots[u]), rec, convs)  # preserves the other live rows
        return host_logits

    def _write_gdn_slot(self, slot, rec_snap, conv_snap):
        """Upload one request's B=1 GDN state snapshot (host torch, per GDN layer) and write it
        into decode `slot` of the batched buffers via TPGatedDeltaNet.write_slot (preserving the
        other live rows). Shapes/mappers mirror _assemble_per_user_gdn (mesh dim 0 = devices).

        rec_snap[li]:  host [num_devices, Nv, Dk, Dv]; conv_snap[li]: list of K host [num_devices, 1, D].
        """
        mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)
        dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]
        for li, dn in enumerate(dn_layers):
            rec = ttnn.from_torch(
                rec_snap[li],
                dtype=dn.rec_state.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=mapper,
            )
            convs = [
                ttnn.from_torch(
                    conv_snap[li][m],
                    dtype=dn.conv_states[m].dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    mesh_mapper=mapper,
                )
                for m in range(dn.K)
            ]
            dn.write_slot(slot, rec, convs)

    def prepare_gdn_decode_width(self, B):
        """Make every GDN layer's decode conv state valid for the width class of decode width B (batched fused
        GDN decode: B <= SCAN_MAX reads the RM history, wider widths run the original decode on conv_states). Call
        once per width change / before each decode call, eagerly (outside trace capture/replay). A no-op unless the
        batched fused mode is on (QWEN36_GDN_FUSED_DECODE=0, max_batch_size == 1: nothing to do). Returns the number
        of layers that needed an in-place sync."""
        n_sync = 0
        n_layers = 0
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            n_layers += 1
            if layer.attention.prepare_decode_width(B):
                n_sync += 1
        if n_layers:
            fmt = next(l.attention._conv_fmt for l in self.layers if not l.is_full_attention)
            self._gdn_conv_fmt = fmt
            if n_sync:
                # rate-limited: one line per sync, at most the first 20 then every 100th
                self._gdn_sync_count = getattr(self, "_gdn_sync_count", 0) + 1
                c = self._gdn_sync_count
                if c <= 20 or c % 100 == 0:
                    logger.info(f"GDN conv-format sync #{c}: width {B} -> '{fmt}' ({n_sync}/{n_layers} layers)")
        return n_sync

    def warmup_gdn_slot_ops(self):
        """Compile the per-slot GDN state edits (write_slot per slot, remap_slots) before any trace is live."""
        if self._gdn_slot_ops_warmed:
            return
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        dn_states = [layer.attention for layer in self.layers if not layer.is_full_attention]
        prev = self._bind_gdn_prefill_scratch()
        try:
            rec = [ttnn.to_torch(dn.rec_state, mesh_composer=comp) for dn in dn_states]
            conv = [[ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states] for dn in dn_states]
        finally:
            self._unbind_gdn_prefill_scratch(prev)
        for slot in range(self.args.max_batch_size):
            self._write_gdn_slot(slot, rec, conv)
        # A full permutation touches every row, so one remap compiles every gather program.
        self._remap_gdn_slots(list(range(self.args.max_batch_size))[::-1])
        # For odd B > 1 the reversal fixes the middle row, so its 1-row paths never compile: also shift cyclically.
        _B = self.args.max_batch_size
        if _B > 1 and _B % 2 == 1:
            self._remap_gdn_slots([(i + 1) % _B for i in range(_B)])
        # The same under the PACKED conv format (what decode runs with once a decode op is active): the direct packed
        # slot write and the packed remap use different programs (embedding -> TILE, 5D slice_write, ...) that must be
        # compiled / allocated before any trace is captured, else their device buffers land behind a captured trace.
        prevs = [dn.warmup_enter_packed() for dn in dn_states]
        if any(p is not None for p in prevs):
            for slot in range(self.args.max_batch_size):
                self._write_gdn_slot(slot, rec, conv)
            self._remap_gdn_slots(list(range(self.args.max_batch_size))[::-1])
            if _B > 1 and _B % 2 == 1:  # as above: cyclic shift moves the middle row the reversal leaves fixed
                self._remap_gdn_slots([(i + 1) % _B for i in range(_B)])
            for dn, p in zip(dn_states, prevs):
                dn.warmup_exit_packed(p)
        ttnn.synchronize_device(self.mesh_device)
        self._gdn_slot_ops_warmed = True

    def set_slot_rope_delta(self, slot, delta):
        """Record decode slot `slot`'s M-RoPE delta (0 for text); read by the batched decode rope index."""
        self._slot_rope_delta[int(slot)] = int(delta)

    def remap_slot_rope_delta(self, remap):
        """Apply a vLLM slot_remap to the per-slot rope deltas (slot i takes the delta previously at slot remap[i];
        rows beyond len(remap) are unchanged). Same semantics as _remap_gdn_slots."""
        r = torch.as_tensor([int(x) for x in remap], dtype=torch.int64)
        new = self._slot_rope_delta.clone()
        new[: r.shape[0]] = self._slot_rope_delta[r]
        self._slot_rope_delta = new

    def _decode_rope_delta(self, B):
        """Rope delta for a B-wide decode: per-slot vector when batched, else the scalar request delta."""
        if self.args.max_batch_size > 1:
            return self._slot_rope_delta[:B]
        return self.rope.rope_delta

    def _remap_gdn_slots(self, remap, live=None):
        """Apply a vLLM batch-condense slot_remap to every GDN layer's batched decode state
        (device-side; slot i takes the state at slot remap[i]). Mirrors seed_manager.apply_slot_remap
        for GDN's per-slot recurrent+conv state, which the plugin's slot_remap does not itself move.
        No-op for an identity remap. `live` (optional list of bools, True = row holds a live request after the
        remap) lets the recurrent-state move skip writes into dead rows."""
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.remap_slots(remap, live=live)

    def set_gdn_fused_decode(self, enabled: bool) -> None:
        """Select the GDN decode recurrence for EVERY GDN layer: the single fused device op
        (``fused_recurrent_gated_delta_rule``) when ``enabled``, else the composite op chain.

        Spec decode requires the fused op: the traced verify advances GDN with it, and decode must
        use identical math or greedy near-ties flip between the two paths (acceptance measured
        2.82 -> 2.00 / 3 when they disagreed at ~1e-5). The fused op is also faster per layer
        (0.345 vs 0.541 ms eager) and closer to the FLA reference, but today it supports only the
        full batch width (B == max_batch_size). This is the ONE place the switch flips: it is a
        visible, model-scoped choice, never a side effect of building a SpeculativeDecoder, because
        on a shared model it changes every later plain decode as well.
        """
        if enabled and self.gdn_verify_spec_step:
            # QWEN36_VERIFY_SPEC_STEP=1: the verify (and the seed) run gdn_spec_step with gdn_decode_step's fast /
            # fp32 arithmetic, so the matching plain decode is the gdn_decode_step op, NOT fused_recurrent. Keep
            # decode there: switching it would re-open the decode/verify math gap this flag closes.
            logger.info(
                "[GDN] set_gdn_fused_decode(True) ignored: QWEN36_VERIFY_SPEC_STEP=1 keeps plain decode on "
                "gdn_decode_step (the verify's gdn_spec_step arithmetic)"
            )
            enabled = False
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.use_fused_recurrent_decode = bool(enabled)

    @property
    def gdn_fused_decode(self) -> bool:
        """True when every GDN layer decodes with the fused recurrent op (see set_gdn_fused_decode)."""
        gdn = [layer.attention for layer in self.layers if not layer.is_full_attention]
        return bool(gdn) and all(dn.use_fused_recurrent_decode for dn in gdn)

    @property
    def gdn_verify_spec_step(self) -> bool:
        """True when every GDN layer runs the spec verify / seed through gdn_spec_step (QWEN36_VERIFY_SPEC_STEP=1);
        plain decode then runs the gdn_decode_step op with the same arithmetic."""
        gdn = [layer.attention for layer in self.layers if not layer.is_full_attention]
        return bool(gdn) and all(getattr(dn, "_spec_fused", False) for dn in gdn)

    def prefill_chunked_peruser(self, token_ids_list, page_table, valid_lens=None):
        """Batched per-user LONG-prefill (TP, eager). Runs the single-user chunk-outer path
        (prefill_traced_chunked) for each user into a B=1 GDN scratch, then stitches each user's
        final GDN state into row u of the batched [B,...] decode buffers. Handles ANY prompt length
        with exact valid_len masking (no last-token padding through GDN): short -> masked bucket;
        long -> chunk-outer + masked tail. Long-prompt counterpart to prefill_traced_bucket_batched
        (full-bucket-only) and prefill_paged_peruser (single-pass). Call allocate_kv_caches(batch_size=B)
        first. ``token_ids_list`` B [1, T_u]; ``page_table`` [B, max_blocks_per_seq] — width MUST be
        a multiple of 8 (chunked SDPA reads each row as a ROW_MAJOR int32 stick; stick_size
        (width*4) % 32 == 0, else it reads the wrong KV). ``valid_lens`` optional. Returns B ttnn logits [1,1,vocab] (replicated; at valid_len-1).
        """
        assert self.num_devices > 1, "prefill_chunked_peruser is the TP (num_devices>1) path"
        assert self._paged_kv_caches is not None, "Call allocate_kv_caches first"
        # The chunked path keys its chunk math on _chunked_chunk_size (default 2048); a parked
        # bucket trace leaves it at 128, breaking num_full/tail sizing. Require it released first.
        assert getattr(self, "_bucket_trace_id", None) is None, (
            "release the bucket prefill trace before prefill_chunked_peruser " "(_chunked_chunk_size would be wrong)"
        )
        assert self._chunked_chunk_size in (None, 2048), (
            f"prefill_chunked_peruser expects the 2048-token chunk; got _chunked_chunk_size="
            f"{self._chunked_chunk_size}"
        )

        B = len(token_ids_list)
        page_table_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert page_table_torch.shape[0] == B, "page_table must have one row per user"

        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]

        # Swap every GDN layer to a B=1 scratch (per-user path is B=1); assembled into the batched
        # buffers after the loop. prev holds the batched bindings for restore.
        prev = self._alloc_gdn_scratch_b1()
        host_logits = []  # torch [1, 1, vocab] (one replica) per user
        # Snapshot each user's B=1 GDN state via a host to_torch round trip (NOT ttnn.clone() — see
        # prefill_traced_bucket_batched for why the device-side clone path is broken: it aliases
        # the captured trace's own baked-address intermediates and gets silently overwritten by
        # the next user's execute_trace()).
        per_user_rec = []  # per user, list over GDN layers of host rec_state snapshots
        per_user_conv = []  # per user, list over GDN layers of [list of K conv snapshots]
        try:
            for u in range(B):
                toks = token_ids_list[u]
                assert toks.shape[0] == 1, f"user {u}: token_ids must be [1, T_u]"
                vlen = valid_lens[u] if valid_lens is not None else toks.shape[1]
                assert 1 <= vlen <= toks.shape[1], f"user {u}: valid_len {vlen} not in [1, {toks.shape[1]}]"

                # Per-user long path (from scratch) into the B=1 scratch: carries state across
                # chunks, masks the tail exactly, writes user u's KV via the page-table row.
                lg = self.prefill_traced_chunked(toks, page_table_torch[u : u + 1].contiguous(), actual_len=vlen)
                # Read the logit to HOST immediately (the next prefill + post-loop assembly churn
                # device memory and would otherwise corrupt the returned tensor).
                ttnn.synchronize_device(self.device)
                host_logits.append(ttnn.to_torch(lg, mesh_composer=comp)[0:1].clone())  # [1,1,vocab] one replica
                ttnn.deallocate(lg)

                # Snapshot this user's B=1 GDN state for assembly after the loop (host round trip —
                # the B=1 scratch is reset IN PLACE for the next user). slot 0 of conv_states is
                # the zeroed shifted-out tap; only slots 1..K-1 carry state.
                per_user_rec.append([ttnn.to_torch(dn.rec_state, mesh_composer=comp) for dn in dn_layers])
                per_user_conv.append(
                    [[ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states] for dn in dn_layers]
                )
        finally:
            # Restore the batched [B,...] GDN decode buffers and free the B=1 scratch. The clones
            # are independent allocations, so freeing the scratch here does not touch them.
            self._restore_gdn_batched(prev)

        ttnn.synchronize_device(self.device)
        # Stitch the per-user states into row u of the (now-rebound) batched decode buffers.
        self._assemble_per_user_gdn(per_user_rec, per_user_conv)
        # Re-upload the per-user logits as stable device tensors (prefill_paged_peruser contract).
        return self._reupload_host_logits(host_logits)

    def _forward_prefill_chunk_eager(self, token_slice, chunk_start, page_table):
        """Eager final partial-chunk prefill (< chunk_size). GDN zero-pads to 128-multiple internally
        (not bucket padding). Returns hidden [1,T_tail_padded,hidden_size]."""
        T_tail = token_slice.shape[1]
        block_size = 64
        tok = ttnn.from_torch(
            token_slice.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        x = self.embd(tok)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tok)
        cos, sin = self.rope.get_prefill_rot_mats(chunk_start, T_tail)
        full_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        blk0 = chunk_start // block_size
        blkN = math.ceil((chunk_start + T_tail) / block_size)
        chunk_pt = ttnn.from_torch(
            page_table[:, blk0:blkN].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        for layer in self.layers:
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=cos,
                    sin=sin,
                    mode="prefill",
                    page_table=full_pt,
                    chunk_page_table=chunk_pt,
                    chunk_start_idx=chunk_start,
                )
            else:
                x_new = layer.forward(x, mode="prefill", chunk_size=layer.attention.long_prefill_chunk_size)
            ttnn.deallocate(x)
            x = x_new
        return x

    # Fixed buckets for masked tail/short prefill. Lengths round up here -> bounded compile set.
    # All 128-multiples (GDN sub-chunk). Masked GDN in DRAM avoids L1 clash at bucket 512.
    # Diverges from get_padded_prefill_len: 256/512 for short TTFT; GDN needs exact valid_len mask.
    _PREFILL_MASK_BUCKETS = (128, 256, 512, 1024, 2048)

    @classmethod
    def _mask_bucket_for(cls, length):
        """Smallest fixed bucket >= length (falls back to the next 128-multiple)."""
        for b in cls._PREFILL_MASK_BUCKETS:
            if length <= b:
                return b
        return ((length + 127) // 128) * 128

    def _forward_prefill_chunk_masked(
        self, token_buf, valid_len, chunk_start, page_table, bucket, flex_sdpa=True, vision_tokens=None
    ):
        """Single masked fixed-bucket prefill forward over `bucket` positions.

        First valid_len tokens real; rest padded. Attn runs full bucket; GDN masks via valid_len.
        Returns hidden [1,bucket,hidden] or [1,1,bucket,hidden] (TP)."""
        if self.num_devices > 1:
            return self._forward_prefill_chunk_masked_tp(
                token_buf, valid_len, chunk_start, page_table, bucket, flex_sdpa=flex_sdpa, vision_tokens=vision_tokens
            )
        block_size = get_block_size(self._paged_kv_caches)
        tok = ttnn.from_torch(
            token_buf.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        x = self.embd(tok)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tok)
        # Trace-safe vision splice: a fixed-shape ttnn.where over persistent buffers (compiled
        # at warmup; mask==0 -> identity, so text-only is unchanged). The caller stages the
        # buffers (prefill_masked_bucket -> _set_vision_merge). No-op until a trace is captured.
        x = self._apply_vision_merge(x, length=bucket)
        cos, sin = self.rope.get_prefill_rot_mats(chunk_start, bucket)
        full_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        blk0 = chunk_start // block_size
        # Fill K/V only for real blocks (ceil(valid_len/64)); padded writes would corrupt block 0.
        blkN = num_blocks_in_seq(chunk_start + valid_len, block_size)
        chunk_pt = ttnn.from_torch(
            page_table[:, blk0:blkN].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        # Flexible SDPA (device chunk_start): one program per bucket for any tail position.
        # Host-int chunk_start compiles per position and can clobber parked trace.
        csi_tensor = ttnn.from_torch(
            torch.tensor([chunk_start], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        for layer in self.layers:
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=cos,
                    sin=sin,
                    mode="prefill",
                    page_table=full_pt,
                    chunk_page_table=chunk_pt,
                    chunk_start_idx_tensor=csi_tensor,
                )
            else:
                x_new = layer.forward(
                    x, mode="prefill", chunk_size=layer.attention.long_prefill_chunk_size, valid_len=valid_len
                )
            ttnn.deallocate(x)
            x = x_new
        return x

    def _forward_prefill_chunk_masked_tp(
        self,
        token_buf,
        valid_len,
        chunk_start,
        page_table,
        bucket,
        flex_sdpa=True,
        vision_tokens=None,
        full_chunk=False,
    ):
        """TP (num_devices>1) masked fixed-bucket single-chunk prefill forward.

        flex_sdpa=True: flexible chunked SDPA (serving). flex_sdpa=False: host-int path (debug).
        Fills K/V for real blocks only. Returns hidden [1,1,bucket,dim]."""
        block_size = get_block_size(self._paged_kv_caches)
        tok = ttnn.from_torch(
            token_buf.to(torch.int32),
            dtype=ttnn.uint32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x = self.embd(tok)
        x = ttnn.reshape(x, (1, 1, bucket, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tok)
        # Trace-safe vision splice (fixed-shape where over the hidden-sharded persistent buffers,
        # sliced to bucket; identity when the mask is zero). The caller stages the buffers
        # (prefill_masked_bucket -> _set_vision_merge). No-op until a trace is captured.
        x = self._apply_vision_merge(x, length=bucket)
        # rope_tp cos/sin for absolute positions [chunk_start, chunk_start+bucket).
        cos_t, sin_t = self._rope_tp_cos_sin_torch(chunk_start, bucket)
        cos = ttnn.from_torch(
            cos_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        sin = ttnn.from_torch(
            sin_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        full_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        blk0 = chunk_start // block_size
        blkN = num_blocks_in_seq(chunk_start + valid_len, block_size)
        chunk_pt = ttnn.from_torch(
            page_table[:, blk0:blkN].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        csi_tensor = (
            ttnn.from_torch(
                torch.tensor([chunk_start], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
            if flex_sdpa
            else None
        )
        _cap = getattr(self, "_capture_layer", None)
        for _li, layer in enumerate(self.layers):
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=cos,
                    sin=sin,
                    mode="prefill",
                    page_table=full_pt,
                    chunk_page_table=chunk_pt,
                    chunk_start_idx=chunk_start,
                    chunk_start_idx_tensor=csi_tensor,
                    valid_len=valid_len,  # unused by full attention
                )
            else:
                x_new = layer.forward(
                    x,
                    mode="prefill",
                    chunk_size=self.args.gdn_chunk_size,
                    # full_chunk: GDN takes the traced body's conv (KDA) so eager == traced
                    valid_len=None if full_chunk else valid_len,
                )
            ttnn.deallocate(x)
            x = x_new
            if _cap is not None and _li == _cap:  # debug: capture hidden after layer _cap
                self._cap_chunk = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Deallocate per-chunk inputs; only hidden survives (avoids OOM in eager 64k loop).
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(full_pt)
        ttnn.deallocate(chunk_pt)
        if csi_tensor is not None:
            ttnn.deallocate(csi_tensor)
        return x

    def prefill_masked_bucket(
        self,
        token_ids,
        page_table,
        actual_len,
        chunk_start=0,
        bucket=None,
        flex_sdpa=True,
        vision_tokens=None,
        vis_row_offset=0,
        return_hidden=False,
    ):
        """Masked fixed-bucket prefill for a segment of `actual_len` real tokens.

        Pads to a fixed bucket, runs all layers ONCE, and masks GDN recurrent + conv state to exactly
        `actual_len` real tokens — numerically equivalent to eager exact-length (prefill_paged) but
        using one of a few bucket-sized programs, so warmup can compile every path before a trace is
        parked (avoids the compile-clobbers-trace hang). ``return_hidden`` also returns per-position
        pre-final-norm hidden [1,1,bucket,dim/tp] (MTP KV-warming; caller deallocates). ``chunk_start``
        is the segment's absolute start (0 for a short prompt; num_full*chunk_size for a long tail —
        carried GDN/KV must already be in place). ``vis_row_offset`` is image-placeholder tokens before this segment. Returns host [1,1,vocab] logits at actual_len-1.
        """
        B_batch, _ = token_ids.shape
        assert B_batch == 1, "masked-bucket prefill is single-sequence"
        if bucket is None:
            bucket = self._mask_bucket_for(actual_len)
        assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"

        if chunk_start == 0:
            # chunk_start==0: new sequence, re-zero GDN. chunk_start>0: tail, keep carried state.
            self._reset_gdn_state_for_new_sequence()
            # Stage the per-request RoPE for this segment (M-RoPE for multimodal, 1D for text).
            # Only at the sequence start; a carried tail (chunk_start>0) keeps the table the
            # long-prompt entry (prefill_traced_chunked) already staged.
            self._build_request_rope(token_ids[:, :actual_len], vision_tokens)

        real = token_ids[:, :actual_len].to(torch.int32)
        if bucket > actual_len:
            pad = torch.zeros(1, bucket - actual_len, dtype=torch.int32)
            token_buf = torch.cat([real, pad], dim=1)
        else:
            token_buf = real

        # Stage the trace-safe vision buffers (host->device copy only). A segment splices its own
        # slice of the packed vision rows (vis_row_offset); a segment with no image placeholders
        # (text-only prompt, or a tail past the image) clears the mask inside _set_vision_merge so
        # the where is the identity. No-op without buffers.
        self._set_vision_merge(token_buf, vision_tokens, vis_row_offset)

        hidden = self._forward_prefill_chunk_masked(
            token_buf, actual_len, chunk_start, page_table, bucket, flex_sdpa=flex_sdpa, vision_tokens=vision_tokens
        )
        ttnn.synchronize_device(self.device)

        if self.num_devices > 1:
            logits = self._masked_bucket_logits_tp(hidden, actual_len, bucket)
            if return_hidden:
                return logits, hidden
            ttnn.deallocate(hidden)
            return logits

        # One-hot matmul for last row (fixed program per bucket; slice would recompile per length).
        sel = torch.zeros(1, 1, bucket, dtype=torch.float32)
        sel[0, 0, actual_len - 1] = 1.0
        sel_tt = ttnn.from_torch(sel, dtype=hidden.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        x_last = ttnn.matmul(sel_tt, hidden)
        ttnn.deallocate(sel_tt)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = self.norm(x_last, mode=Mode.PREFILL)
        logits = self._lm_head(x_last)
        if return_hidden:
            return logits.cpu(), hidden
        ttnn.deallocate(hidden)
        return logits.cpu()

    def _masked_bucket_logits_tp(self, hidden, actual_len, bucket):
        """TP: one-hot select row actual_len-1, norm, lm_head. Returns replicated [1,1,vocab]."""
        sel = torch.zeros(1, 1, 1, bucket, dtype=torch.float32)
        sel[0, 0, 0, actual_len - 1] = 1.0
        sel_tt = ttnn.from_torch(
            sel,
            dtype=hidden.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x_last = ttnn.matmul(sel_tt, hidden)  # [1, 1, 1, dim]
        ttnn.deallocate(sel_tt)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = self.norm(x_last, mode=Mode.PREFILL)
        logits = self._lm_head(x_last)
        return ttnn.reshape(logits, (1, 1, logits.shape[-1]))

    def warmup_prefill_masked_buckets(self, page_table, buckets=None):
        """Compile every masked-bucket prefill program up front, so a request never compiles after
        a trace is parked (a post-park compile clobbers the trace -> second-request hang, #48536).
        Two program kinds: bucket-keyed (SDPA / GDN mask / norm / MLP-or-MoE), one per (bucket,
        is_full), warmed by a dummy forward at each bucket once masked and once full; and
        fill-width-keyed (paged_fill_cache), hashed on fill shape, warmed by
        _warmup_paged_fill_widths (no full forward). MUST run in GDN serving state, before any
        trace is parked (capture_prefill_trace_chunked calls this just before begin_trace_capture).
        page_table must cover the largest bucket.
        """
        if buckets is None:
            buckets = self._PREFILL_MASK_BUCKETS
        block_size = get_block_size(self._paged_kv_caches)

        # Bucket-keyed programs: one masked + one no-mask forward per bucket.
        seen = set()
        for bucket in sorted(buckets):
            for actual_len in (max(1, bucket // 2), bucket):
                actual_len = max(1, min(actual_len, bucket))
                key = (bucket, actual_len == bucket)
                if key in seen:
                    continue
                seen.add(key)
                toks = torch.zeros(1, actual_len, dtype=torch.int32)
                self.prefill_masked_bucket(toks, page_table, actual_len=actual_len, bucket=bucket)
        # Fill-width-keyed programs: warm every width directly (no full forward).
        self._warmup_paged_fill_widths(page_table, buckets, block_size)
        ttnn.synchronize_device(self.device)

    def _warmup_paged_fill_widths(self, page_table, buckets, block_size):
        """Warm the per-fill-width programs in TPAttention.forward_prefill_paged's KV-fill sub-path
        (ttnn.slice + paged_fill_cache) without a full-model forward. Both hash on the fill shape
        (seq = fill_blocks * block_size), so each width is a fresh program; an un-warmed width would
        compile after the trace is parked and clobber it (hang).

        The ops are shape-keyed, so warming one layer's cache serves every layer -- far cheaper than
        the old per-width all-layer forward. Cover EVERY fill-width-dependent op here; a new one is
        caught by test_prefill_warmup_no_recompile (width sweep under misses-disallowed)."""
        if not self._paged_kv_caches:
            return
        k_cache, v_cache = self._paged_kv_caches[0]
        nkv, hd = k_cache.shape[1], k_cache.shape[3]
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.num_devices > 1 else None
        seen = set()
        for bucket in sorted(buckets):
            k_full = ttnn.from_torch(
                torch.zeros(1, nkv, bucket, hd, dtype=torch.bfloat16),
                dtype=k_cache.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                # L1 to match the request-time fill's L1 K/V (slice's cache key is buffer-type-specific).
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            for w in range(1, num_blocks_in_seq(bucket, block_size) + 1):
                page_len = min(w * block_size, bucket)
                key = (bucket, page_len)
                if key in seen:
                    continue
                seen.add(key)
                pt = ttnn.from_torch(
                    page_table[:, :w].contiguous(),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    mesh_mapper=mapper,
                )
                if page_len < bucket:
                    fill = ttnn.slice(k_full, (0, 0, 0, 0), (1, nkv, page_len, hd))
                else:
                    fill = k_full
                ttnn.experimental.paged_fill_cache(k_cache, fill, pt, batch_idx=0)
                ttnn.experimental.paged_fill_cache(v_cache, fill, pt, batch_idx=0)
                if page_len < bucket:
                    ttnn.deallocate(fill)
                ttnn.deallocate(pt)
            ttnn.deallocate(k_full)

    def prefill_traced_chunked(
        self, token_ids, page_table, actual_len, vision_tokens=None, return_host_logits=False, on_chunk=None
    ):
        """Single traced-prefill entry for every prompt length (see _prefill_traced_chunked_dev for the long path).

        A prompt < chunk_size (no full chunk) whose bucket has a captured short trace (capture_prefill_traces_short,
        QWEN36_PREFILL_BUCKET_TRACE=1) is ONE trace replay with the GDN masked by persistent device inputs; otherwise
        (flag off, bucket not captured, GDN bound differently from the capture, trash block collides, or a long
        prompt) the previous eager masked-bucket / chunk-trace path runs unchanged.

        Returns the same object as before: a device tensor [1,1,vocab] (replicated; the traced short path re-uploads
        its host logits for compatibility). return_host_logits=True returns a host torch float tensor [1,1,vocab]
        instead (no re-upload; the vLLM / prefill_paged_slots callers).

        ``on_chunk(hidden, chunk_start, valid_len)`` (TP traced only; same contract as prefill_for_spec):
        called per full chunk right after its replay is enqueued, then for the tail / short prompt.
        hidden is the pre-final-norm [1,1,2048,dim/tp]; on_chunk must NOT free it (it is the trace's
        persistent output), and must enqueue its reads before returning (the next replay overwrites it).
        When on_chunk is given the short-trace shortcut is skipped (it has no per-chunk hidden to hand out)."""
        chunk_size = self._chunked_chunk_size or 2048
        if (
            on_chunk is None
            and actual_len < chunk_size
            and token_ids.shape[0] == 1
            and 1 <= actual_len <= token_ids.shape[1]
        ):
            host = self.prefill_short_traced_logits(
                token_ids, page_table, actual_len, vision_tokens=vision_tokens, strict=False
            )
            if host is not None:
                if return_host_logits:
                    return host
                return ttnn.from_torch(
                    host.to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
        lg = self._prefill_traced_chunked_dev(
            token_ids, page_table, actual_len, vision_tokens=vision_tokens, on_chunk=on_chunk
        )
        if return_host_logits:
            comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            host = ttnn.to_torch(lg, mesh_composer=comp).reshape(-1, self.args.vocab_size)[:1].float().view(1, 1, -1)
            ttnn.deallocate(lg)
            return host
        return lg

    def _prefill_traced_chunked_dev(self, token_ids, page_table, actual_len, vision_tokens=None, on_chunk=None):
        """Prefill by replaying the captured per-chunk trace for each FULL 2048-token chunk,
        then processing the final partial chunk eagerly with minimal padding.

        Only the real prompt (token_ids[:, :actual_len]) is processed; any bucket padding in
        token_ids is ignored. Full chunks (num_full = actual_len // chunk_size) are replayed
        from the trace; the remaining tail (< chunk_size tokens) is run eagerly so the GDN
        kernel zero-pads it to the next multiple of 128 (matching the non-traced path) instead
        of repeating the bucket padding through the recurrence — which corrupts the decode
        state at long context. actual_len is the real prompt length; the next-token logit is
        extracted at actual_len-1. Returns ttnn.Tensor (host) [1, 1, vocab_size].

        vision_tokens (multimodal) are spliced trace-safely: the captured forward runs a
        FIXED-shape ttnn.where(mask, vision, text) over persistent buffers (_vis_buf /
        _vis_mask_buf) that this method stages per chunk via copy_host_to_device — no program
        compiles at request time, so the parked trace is never clobbered. Each chunk (and the tail)
        splices its own slice of the packed vision rows (vis_row_offset = image tokens before the
        chunk), so a large image whose placeholders span multiple chunks is handled; a segment with
        no image tokens clears the mask (the where becomes the identity). Works on both single
        device (3D buffers) and TP (4D hidden-sharded buffers; the vision rows are gathered to full
        hidden on host in _set_vision_merge, placed along seq, then re-sharded — see
        _alloc_vision_merge_buffers).

        ``on_chunk``: see prefill_traced_chunked (A's spec hook; TP traced path only).
        """
        # Default to the standard 2048-token chunk when no trace is captured (e.g. the TP MVP,
        # which serves <=2048 prompts entirely via the masked bucket below and so needs no chunk
        # trace). The chunk trace is only required once there is at least one full chunk to replay.
        chunk_size = self._chunked_chunk_size or 2048
        B, T = token_ids.shape
        assert 1 <= actual_len <= T, f"actual_len {actual_len} not in [1, {T}]"
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_chunk = chunk_size // block_size
        num_full = actual_len // chunk_size
        tail_real = actual_len - num_full * chunk_size
        assert (
            num_full == 0 or self.num_devices > 1 or self._chunked_trace_id is not None
        ), "Call capture_prefill_trace_chunked first"

        # Stage the per-request RoPE once for the whole prompt (M-RoPE for multimodal, 1D for text).
        # The chunk-replay loops + the masked tail then slice this sequence-indexed table by chunk
        # position, and decode offsets by the stored rope_delta. (The num_full==0 short path below
        # re-stages it inside prefill_masked_bucket; that is idempotent.)
        self._build_request_rope(token_ids[:, :actual_len], vision_tokens)

        # Short prompt (no full chunks): route the whole prompt through the SAME masked
        # fixed-bucket path the long-prompt tail uses. chunk_start=0 makes prefill_masked_bucket
        # do the sequence-start GDN reset and run one masked forward — there is no trace to replay,
        # so the chunk-input plumbing below is skipped. This is the single bucketed+masked path
        # shared by short prompts and the long-prompt tail; prefill_dispatch routes every traced
        # prefill here so the short/long seam is defined once.
        if num_full == 0:
            # Pad/clip the SDPA page table to the warmed/captured width so the short-prompt forward
            # REPLAYS the pre-warmed programs instead of recompiling at request time (which clobbers
            # parked decode/chunk traces -> second-request hang). vLLM pads to its own
            # max_num_blocks_per_req, which differs from the warmed width. Trailing entries index
            # blocks past the prompt and are never read by causal SDPA (as in the long-prompt branch
            # below). No-op when no chunk buffer was captured or the widths already match.
            buf = getattr(self, "_chunk_full_page_table_buf", None)
            if buf is not None:
                buf_blocks = int(buf.shape[-1])
                if page_table.shape[1] < buf_blocks:
                    page_table = torch.cat(
                        [
                            page_table,
                            torch.zeros(page_table.shape[0], buf_blocks - page_table.shape[1], dtype=page_table.dtype),
                        ],
                        dim=1,
                    )
                elif page_table.shape[1] > buf_blocks:
                    page_table = page_table[:, :buf_blocks]
            if on_chunk is not None:
                logits, hidden = self.prefill_masked_bucket(
                    token_ids[:, :actual_len],
                    page_table,
                    actual_len=actual_len,
                    chunk_start=0,
                    vision_tokens=vision_tokens,
                    return_hidden=True,
                )
                on_chunk(hidden, 0, actual_len)
                ttnn.deallocate(hidden)
                return logits
            return self.prefill_masked_bucket(
                token_ids[:, :actual_len], page_table, actual_len=actual_len, chunk_start=0, vision_tokens=vision_tokens
            )

        assert on_chunk is None or (
            self.num_devices > 1 and self._chunked_trace_id is not None
        ), "on_chunk needs the TP traced chunk path"
        if self.num_devices > 1:
            # TP long prompt: traced replay preferred; eager masked-bucket fallback if no trace.
            if self._chunked_trace_id is not None:
                return self._prefill_traced_chunked_tp(
                    token_ids,
                    page_table,
                    actual_len,
                    num_full,
                    chunk_size,
                    tail_real,
                    vision_tokens=vision_tokens,
                    on_chunk=on_chunk,
                )
            # Eager fallback: flexible qk=64 SDPA matches traced path.
            return self._prefill_chunked_eager_tp(
                token_ids,
                page_table,
                actual_len,
                num_full,
                chunk_size,
                tail_real,
                flex_sdpa=True,
                vision_tokens=vision_tokens,
            )

        # Re-zero GDN once; carries across replays + masked tail (chunk_start>0 skips reset).
        self._reset_gdn_state_for_new_sequence()
        # Pad/clip page_table to captured buffer width (vLLM may differ). Trailing blocks unused.
        buf_blocks = int(self._chunk_full_page_table_buf.shape[-1])
        if page_table.shape[1] < buf_blocks:
            page_table = torch.cat(
                [
                    page_table,
                    torch.zeros(page_table.shape[0], buf_blocks - page_table.shape[1], dtype=page_table.dtype),
                ],
                dim=1,
            )
        elif page_table.shape[1] > buf_blocks:
            page_table = page_table[:, :buf_blocks]
        pt_host = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(pt_host, self._chunk_full_page_table_buf)

        # Replay trace for each full chunk.
        for c in range(num_full):
            cs = c * chunk_size
            tok_host = ttnn.from_torch(
                token_ids[:, cs : cs + chunk_size].to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            ttnn.copy_host_to_device_tensor(tok_host, self._chunk_token_buf)

            csi_host = ttnn.from_torch(
                torch.tensor([cs], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            ttnn.copy_host_to_device_tensor(csi_host, self._chunk_start_idx_tensor)

            blk0 = cs // block_size
            cpt_host = ttnn.from_torch(
                page_table[:, blk0 : blk0 + blocks_per_chunk].contiguous(),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(cpt_host, self._chunk_page_table_buf)

            # M-RoPE-aware per-chunk cos/sin (slices the staged per-request table for multimodal;
            # 1D RoPE for text). Updated into the persistent buffer per chunk via host->device copy,
            # so it stays trace-safe.
            cos_seq, sin_seq = self.rope.prefill_cos_sin_torch(cs, chunk_size)
            cos_host = ttnn.from_torch(
                cos_seq.unsqueeze(0).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            sin_host = ttnn.from_torch(
                sin_seq.unsqueeze(0).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(cos_host, self._chunk_cos_buf)
            ttnn.copy_host_to_device_tensor(sin_host, self._chunk_sin_buf)

            # Stage the trace-safe vision buffers: each chunk splices its own slice of the packed
            # vision rows (vis_row_offset = image tokens before cs); a chunk with no image tokens
            # clears the mask so the captured where is the identity. host->device copy only (no
            # compile), so the parked trace is untouched. Handles a large image whose placeholders
            # span multiple chunks.
            self._set_vision_merge(
                token_ids[:, cs : cs + chunk_size], vision_tokens, self._vis_row_offset_for(token_ids, cs)
            )

            ttnn.execute_trace(self.device, self._chunked_trace_id, cq_id=0, blocking=False)

        ttnn.synchronize_device(self.device)

        # Tail via masked bucket (or last full chunk hidden if exact multiple of chunk_size).
        if tail_real > 0:
            cs = num_full * chunk_size
            return self.prefill_masked_bucket(
                token_ids[:, cs:actual_len],
                page_table,
                actual_len=tail_real,
                chunk_start=cs,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, cs),
            )
        hidden = self._chunked_trace_output  # last full chunk's hidden state
        pos_in_chunk = (actual_len - 1) - (num_full - 1) * chunk_size
        ttnn.synchronize_device(self.device)

        x_last = hidden[:, pos_in_chunk : pos_in_chunk + 1, :]
        x_last = ttnn.to_layout(x_last, ttnn.TILE_LAYOUT)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = self.norm(x_last, mode=Mode.PREFILL)
        logits = self._lm_head(x_last)
        return logits.cpu()

    def _prefill_chunked_eager_tp(
        self, token_ids, page_table, actual_len, num_full, chunk_size, tail_real, flex_sdpa=True, vision_tokens=None
    ):
        """TP eager long-prompt prefill via warmed bucket=chunk_size programs.
        Returns logits [1,1,vocab] at actual_len-1."""
        # Re-zero GDN at sequence start; tail (chunk_start>0) keeps carried state.
        self._reset_gdn_state_for_new_sequence()
        last_hidden = None
        for c in range(num_full):
            cs = c * chunk_size
            if last_hidden is not None:
                ttnn.deallocate(last_hidden)
            # Stage the vision buffers: each chunk splices its own slice of the packed vision rows
            # (vis_row_offset = image tokens before cs); a chunk with no image tokens clears the
            # mask so the where is the identity (host->device copy only, no compile).
            self._set_vision_merge(
                token_ids[:, cs : cs + chunk_size], vision_tokens, self._vis_row_offset_for(token_ids, cs)
            )
            # Full chunk: valid_len == bucket == chunk_size (no padding/masking).
            last_hidden = self._forward_prefill_chunk_masked_tp(
                token_ids[:, cs : cs + chunk_size], chunk_size, cs, page_table, chunk_size, flex_sdpa=flex_sdpa
            )
            ttnn.synchronize_device(self.device)
        if tail_real > 0:
            ttnn.deallocate(last_hidden)
            cs = num_full * chunk_size
            return self.prefill_masked_bucket(
                token_ids[:, cs:actual_len],
                page_table,
                actual_len=tail_real,
                chunk_start=cs,
                flex_sdpa=flex_sdpa,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, cs),
            )
        # Exact multiple of chunk_size: logit from last full chunk.
        logits = self._masked_bucket_logits_tp(last_hidden, chunk_size, chunk_size)
        ttnn.deallocate(last_hidden)
        return logits

    def _replay_chunks(
        self, token_ids, page_table, pt_host, start, n_chunks, chunk_size, vision_tokens, save_cb, on_chunk
    ):
        """Replay the ACTIVE chunk set's trace for n_chunks chunks of chunk_size rows from absolute position start;
        returns (trace output, chunk_size) of the last replayed chunk, or None when n_chunks == 0."""
        if n_chunks == 0:
            return None
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_chunk = chunk_size // block_size
        rep = ttnn.ReplicateTensorToMesh(self.device)
        num_full = n_chunks
        ttnn.copy_host_to_device_tensor(pt_host, self._chunk_full_page_table_buf)
        # Replay trace per full chunk. Host input-prep (from_torch tilize of cos/sin + DMA copies) and
        # device trace exec share cq_id=0, so the queue already ORDERS each chunk's copies AFTER the
        # prior chunk's trace reads them (no double-buffering needed). The old code did a full
        # synchronize_device every chunk, which forced the host to wait and serialized chunk N+1's CPU
        # prep behind chunk N's device exec. Instead we hold host-tensor refs alive (so their in-flight
        # DMAs aren't GC'd) and sync only every _SYNC_EVERY chunks — the host software-pipelines chunk
        # N+1's from_torch/tilize over chunk N's device exec. Periodic (not fully removed) sync bounds
        # in-flight queue depth so very long context (e.g. traced_128k = 64 chunks) can't overrun the
        # command queue. QWEN36_PREFILL_OVERLAP=0 restores the per-chunk sync.
        _log_every = max(1, num_full // 4)
        _overlap = os.environ.get("QWEN36_PREFILL_OVERLAP", "1") != "0"
        _SYNC_EVERY = 8 if _overlap else 1
        _host_refs = []  # keep host tensors alive until the next sync frees their DMAs
        for c in range(num_full):
            cs = start + c * chunk_size
            tok_host = ttnn.from_torch(
                token_ids[:, cs : cs + chunk_size].to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
                mesh_mapper=rep,
            )
            ttnn.copy_host_to_device_tensor(tok_host, self._chunk_token_buf)

            csi_host = ttnn.from_torch(
                torch.tensor([cs], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
                mesh_mapper=rep,
            )
            ttnn.copy_host_to_device_tensor(csi_host, self._chunk_start_idx_tensor)

            blk0 = cs // block_size
            cpt_host = ttnn.from_torch(
                page_table[:, blk0 : blk0 + blocks_per_chunk].contiguous(),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
                mesh_mapper=rep,
            )
            ttnn.copy_host_to_device_tensor(cpt_host, self._chunk_page_table_buf)

            cos_t, sin_t = self._rope_tp_cos_sin_torch(cs, chunk_size)
            cos_host = ttnn.from_torch(
                cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep
            )
            sin_host = ttnn.from_torch(
                sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep
            )
            ttnn.copy_host_to_device_tensor(cos_host, self._chunk_cos_buf)
            ttnn.copy_host_to_device_tensor(sin_host, self._chunk_sin_buf)
            _host_refs += [tok_host, csi_host, cpt_host, cos_host, sin_host]

            # Stage the hidden-sharded vision buffers: each chunk splices its own slice of the
            # packed vision rows (vis_row_offset = image tokens before cs); a chunk with no image
            # tokens clears the mask so the captured where is the identity. host->device copy only
            # (no compile), so the parked trace is untouched.
            self._set_vision_merge(
                token_ids[:, cs : cs + chunk_size], vision_tokens, self._vis_row_offset_for(token_ids, cs)
            )

            ttnn.execute_trace(self.device, self._chunked_trace_id, cq_id=0, blocking=False)
            if save_cb is not None:
                save_cb(cs + chunk_size)  # device-side snapshot, ordered after this chunk's trace on cq 0
            if on_chunk is not None:
                # Enqueued before the next replay (same cq), so it reads this chunk's output; no host sync.
                on_chunk(self._chunked_trace_output, cs, chunk_size)

            # Bound in-flight depth; after a sync the completed DMAs' host tensors can be released.
            if (c + 1) % _SYNC_EVERY == 0:
                ttnn.synchronize_device(self.device)
                _host_refs.clear()
            if (c + 1) % _log_every == 0:
                logger.info(f"[TP chunk-replay] {c + 1}/{num_full} chunks")

        # Drain any still-in-flight chunk DMAs before returning, so the loop's host input tensors
        # (_host_refs) are not GC'd while a non-blocking execute_trace is still reading them — a
        # use-after-free that hangs the device. When num_full < _SYNC_EVERY the loop never synced,
        # so the no-tail return below (which issues no further blocking work) would otherwise race.
        if _host_refs:
            ttnn.synchronize_device(self.device)
            _host_refs.clear()
        return self._chunked_trace_output, chunk_size

    def _prefill_traced_chunked_tp(
        self,
        token_ids,
        page_table,
        actual_len,
        num_full,
        chunk_size,
        tail_real,
        vision_tokens=None,
        start=0,
        reset_gdn=True,
        save_cb=None,
        run_tail=True,
        on_chunk=None,
        allow_big=False,
    ):
        """TP traced chunk-outer prefill: replay the captured per-chunk trace
        (_forward_prefill_chunk_tp) for each FULL chunk, then run the partial tail through the
        masked bucket. The TP analog of the single-device loop in prefill_traced_chunked: each
        chunk's inputs are DMA'd into the REPLICATED persistent buffers via
        copy_host_to_device_tensor (no per-chunk program dispatch / device allocation — only one
        execute_trace per chunk), so GDN recurrent/conv + paged-KV state carry in place across
        replays and host pressure stays bounded at 128K. The tail's chunk_start>0 skips the GDN
        reset so the carried state continues. Returns logits [1, 1, vocab] at actual_len-1.

        Resumable prefill (prefix cache, _prefill_range): `start` is the ABSOLUTE position of the first chunk (a
        multiple of 128; token_ids / page_table are the request's full rows, chunk c covers [start + c*chunk_size, ...)),
        reset_gdn=False continues the bound GDN state instead of re-zeroing, save_cb(p) is called right after the
        replay of the chunk ending at absolute position p, and run_tail=False stops after the full chunks (returns
        None; the caller runs the tail). Defaults reproduce the original behaviour exactly.

        allow_big: the full blocks of big_chunk_rows() rows (QWEN36_PREFILL_BIG_CHUNK) replay the big chunk trace
        first, then the rest the 2048 one; same rows covered, so the tail is unchanged. self._last_chunk names the last
        replayed chunk (output, rows) for callers that read its output."""
        rep = ttnn.ReplicateTensorToMesh(self.device)

        # Re-zero GDN once; carries across replays + tail (chunk_start>0 skips reset).
        if reset_gdn:
            self._reset_gdn_state_for_new_sequence()

        # Pad/clip page_table to captured width; write once (constant across chunks).
        buf_blocks = int(self._chunk_full_page_table_buf.shape[-1])
        if page_table.shape[1] < buf_blocks:
            page_table = torch.cat(
                [
                    page_table,
                    torch.zeros(page_table.shape[0], buf_blocks - page_table.shape[1], dtype=page_table.dtype),
                ],
                dim=1,
            )
        elif page_table.shape[1] > buf_blocks:
            page_table = page_table[:, :buf_blocks]
        pt_host = ttnn.from_torch(
            page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep
        )

        rows = num_full * chunk_size
        big = self.big_chunk_rows() if allow_big else 0
        n_big = rows // big if big else 0
        last = None
        if n_big:
            with self._big_chunk_set():
                last = self._replay_chunks(
                    token_ids, page_table, pt_host, start, n_big, big, vision_tokens, save_cb, on_chunk
                )
        n_small = (rows - n_big * big) // chunk_size
        last = (
            self._replay_chunks(
                token_ids,
                page_table,
                pt_host,
                start + n_big * big,
                n_small,
                chunk_size,
                vision_tokens,
                save_cb,
                on_chunk,
            )
            or last
        )
        self._last_chunk = last

        if not run_tail:
            return None

        # Tail via masked bucket, or _masked_bucket_logits_tp if no tail (TP 4D hidden).
        if tail_real > 0:
            cs = num_full * chunk_size
            if on_chunk is not None:
                logits, hidden = self.prefill_masked_bucket(
                    token_ids[:, cs:actual_len],
                    page_table,
                    actual_len=tail_real,
                    chunk_start=cs,
                    vision_tokens=vision_tokens,
                    vis_row_offset=self._vis_row_offset_for(token_ids, cs),
                    return_hidden=True,
                )
                on_chunk(hidden, cs, tail_real)
                ttnn.deallocate(hidden)
                return logits
            return self.prefill_masked_bucket(
                token_ids[:, cs:actual_len],
                page_table,
                actual_len=tail_real,
                chunk_start=cs,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, cs),
            )
        out, last_rows = self._last_chunk
        return self._masked_bucket_logits_tp(out, last_rows, last_rows)

    def reset_state(self):
        """Reset layer state for a new sequence."""
        for layer in self.layers:
            if layer.is_full_attention:
                layer.attention.reset_cache()
            else:
                layer.attention.reset_state_inplace()

    def _reset_gdn_state_for_new_sequence(self):
        """Zero GDN recurrent+conv at sequence start.

        Trace capture runs forward twice; GDN state is non-idempotent. Must re-zero before each
        real sequence."""
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.reset_state_inplace()

    def set_paged_kv_caches(self, kv_caches):
        """Attach paged KV caches to the 8 attention layers."""
        self._paged_kv_caches = kv_caches
        for cache_idx, layer_idx in enumerate(self._attention_layer_indices):
            k_cache, v_cache = kv_caches[cache_idx]
            self.layers[layer_idx].attention.set_paged_kv_cache(k_cache, v_cache)

    def allocate_kv_caches(self, kv_cache_shape, dtype, batch_size=1):
        """Allocate caches for all 32 layers. Returns only the attention KV caches (for vLLM)."""
        assert self._paged_kv_caches is None, "allocate_kv_caches already called; deallocate first"
        # GDN reset_state frees the spec ring/E_prev a prepared verify trace points at.
        self._vfy_prepared_key = None
        # bf8 paged KV (QWEN_SDPA_BF8, or cache-only QWEN36_KV_BF8): halves KV memory and long-context SDPA reads.
        if kv_cache_bf8_enabled():
            dtype = ttnn.bfloat8_b
        if self.num_devices > 1:
            return self._allocate_kv_caches_tp(kv_cache_shape, dtype, batch_size)

        kv_caches = []
        for idx in self._attention_layer_indices:
            k_cache = ttnn.zeros(kv_cache_shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
            v_cache = ttnn.zeros(kv_cache_shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
            kv_caches.append([k_cache, v_cache])
        self.set_paged_kv_caches(kv_caches)
        self._allocate_mtp_kv_cache(kv_cache_shape, dtype, batch_size, replicate=False)
        return kv_caches

    def _allocate_mtp_kv_cache(self, kv_cache_shape, dtype, batch_size, replicate):
        """Allocate + bind the MTP drafter's own paged KV cache (one full-attention layer).

        Separate from the base caches (own page table); shape matches a base attention layer except
        for ONE EXTRA BLOCK at the end. That extra block is the spec-decode batched reseed's
        scratch: padding rows point their whole page-table row at the cache's extra last block
        (index ``num_blocks``, derived from the cache size, never from the page table's width) so
        their (discarded) KV write lands somewhere harmless. A block of its own — instead of
        borrowing the sequence's last one — lets a sequence use the full ``num_blocks x block_size``
        span (needed at 256k: 262016 prompt + 100 new vs 4096 x 64 = 262144). The page table never
        names that block; only the reseed's padding rows do. No-op with no MTP head. Fixed address + _stable_state for decode-trace reuse.
        """
        if self.mtp is None:
            return

        mtp_cache_shape = [kv_cache_shape[0] + 1, *kv_cache_shape[1:]]

        def _mk():
            if replicate:
                return ttnn.as_tensor(
                    torch.zeros(mtp_cache_shape, dtype=torch.bfloat16),
                    device=self.device,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                )
            return ttnn.zeros(mtp_cache_shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)

        mtp_k, mtp_v = _mk(), _mk()
        self.mtp.attention.set_paged_kv_cache(mtp_k, mtp_v)
        self.mtp.attention.B = batch_size
        self.mtp.attention._stable_state = True
        self._mtp_kv_cache = [mtp_k, mtp_v]

    def free_kv_caches(self):
        """Release KV caches for a fresh generation run."""
        # The verify trace bakes in the paged KV caches freed below and the GDN spec buffers that the
        # next allocate_kv_caches -> reset_state drops, so it goes even when the KV-cache marker is
        # already clear (hence before the early-out).
        eng = getattr(self, "_spec_engine", None)
        if eng is not None:
            eng.shutdown()  # also releases the verify trace
            self._spec_engine = None
        self.release_verify_trace()
        # Force prepare_verify_trace to re-allocate against the new caches.
        self._vfy_prepared_key = None
        if self._paged_kv_caches is None:
            return
        if getattr(self, "_chunked_trace_id", None) is not None:
            ttnn.release_trace(self.device, self._chunked_trace_id)
            self._chunked_trace_id = None
        self.release_prefill_trace_short()  # bakes the KV/GDN addresses freed below
        self._chunked_prepared_key = None
        if getattr(self, "_paged_kv_caches", None) is not None:
            for k_cache, v_cache in self._paged_kv_caches:
                ttnn.deallocate(k_cache)
                ttnn.deallocate(v_cache)
            self._paged_kv_caches = None
        if getattr(self, "_mtp_kv_cache", None) is not None:
            for cache in self._mtp_kv_cache:
                ttnn.deallocate(cache)
            self._mtp_kv_cache = None
            if self.mtp is not None:
                self.mtp.attention.use_paged = False

    def _allocate_kv_caches_tp(self, kv_cache_shape, dtype, batch_size):
        """TP paged KV allocation (B=1). Replicated per device; GDN self-manages state."""

        def _mk():
            return ttnn.as_tensor(
                torch.zeros(kv_cache_shape, dtype=torch.bfloat16),
                device=self.device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )

        kv_caches = [[_mk(), _mk()] for _ in self._attention_layer_indices]
        self.set_paged_kv_caches(kv_caches)  # binds via TPAttention.set_paged_kv_cache
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.B = batch_size
                layer.attention.reset_state()
        self._allocate_mtp_kv_cache(kv_cache_shape, dtype, batch_size, replicate=True)
        return kv_caches

    def _prefill_paged_tp(self, token_ids, page_table, valid_len=None, vision_tokens=None, gdn_collect=False):
        """TP (num_devices>1) paged prefill, B=1. Mirrors the demo prefill_tp but routes
        the full-attention layers through the paged KV cache (forward_prefill_paged) so
        decode can read it via page_table. GDN layers capture their recurrent/conv state
        as in the demo. Returns logits [1, 1, vocab] at position valid_len-1.
        """
        B, T = token_ids.shape
        assert B == 1, "TP prefill is single-sequence (B=1); batched serving prefills one user at a time"
        vlen = valid_len or T
        # Stage the per-request RoPE (M-RoPE for multimodal, 1D for text).
        self._build_request_rope(token_ids[:, :vlen], vision_tokens)
        pt_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        page_table_tt = ttnn.from_torch(pt_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        tok = ttnn.from_torch(
            token_ids.to(torch.int32),
            dtype=ttnn.uint32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x = self.embd(tok)
        x = self._scatter_vision_tokens(x, token_ids, vision_tokens)
        x = ttnn.reshape(x, (1, 1, T, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        cos_t, sin_t = self._rope_tp_cos_sin_torch(0, T)
        rep = ttnn.ReplicateTensorToMesh(self.device)
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        for layer in self.layers:
            x = layer.forward(
                x,
                cos=cos,
                sin=sin,
                mode="prefill",
                chunk_size=self.args.gdn_chunk_size,
                valid_len=vlen,
                page_table=page_table_tt,
                chunk_page_table=page_table_tt,
                chunk_start_idx=0,
                gdn_collect=gdn_collect,
            )
        x = self.norm(x, mode=Mode.PREFILL)
        x_last = x[:, :, vlen - 1 : vlen, :]
        logits = self._lm_head(x_last)
        ttnn.deallocate(x)
        return ttnn.reshape(logits, (1, 1, logits.shape[-1]))

    def prefill_paged_peruser(self, token_ids_list, page_table, valid_lens=None):
        """Batched per-user TP prefill (the batched serving contract).

        Prefills B users into ONE shared paged KV cache + batched GDN decode state, one user at a
        time. Full-attention layers fill each user's blocks via the per-user page-table row; each
        GDN layer collects that user's from-scratch state, stitched into row u of the batched
        decode buffers by finalize_pending(). Call allocate_kv_caches(batch_size=B) first.
        ``token_ids_list`` B [1, T_u]; ``page_table`` [B, max_blocks_per_seq]; ``valid_lens``
        optional. Returns B ttnn logits [1,1,vocab] (one per user, at valid_len-1).
        """
        assert self.num_devices > 1, "prefill_paged_peruser is the TP (num_devices>1) path"
        B = len(token_ids_list)
        page_table_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert page_table_torch.shape[0] == B, "page_table must have one row per user"

        # Fresh GDN per-user accumulators (cleared at the end by finalize_pending).
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention._pending = []

        logits = []
        for u in range(B):
            vlen = valid_lens[u] if valid_lens is not None else None
            # Each user's prefill is the validated B=1 paged path, pointed at user u's blocks.
            lg = self._prefill_paged_tp(
                token_ids_list[u], page_table_torch[u : u + 1], valid_len=vlen, gdn_collect=True
            )
            logits.append(lg)

        # Stitch every user's collected GDN state into the batched decode buffers (row u = user u).
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.finalize_pending()
        return logits

    def _alloc_gdn_scratch_b(self, bg):
        """Like _alloc_gdn_scratch_b1 but for a group of `bg` users: allocate a dedicated
        [bg,...] GDN state on every GDN layer, distinct from the real batched [B,...] decode
        buffer. Returns the prior batched bindings for _restore_gdn_batched. Used by the grouped
        batched prefill (forward_prefill_batched writes [bg,...] into these in place)."""
        prev = []
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            # Same 7-field binding as _alloc_gdn_scratch_b1 (_restore_gdn_batched unpacks both).
            prev.append(
                (
                    dn,
                    dn.B,
                    dn.rec_state,
                    dn.conv_states,
                    dn.conv_carry,
                    dn._zero_conv0,
                    dn._zero_rec,
                )
            )
            dn.B = bg
            dn.reset_state()  # builds rec_state [bg,Nv,Dk,Dv], conv_states[*] [1,bg,D], carry, zero0
        return prev

    def _assemble_groups_gdn_dev(self, group_rec_dev, group_conv_dev):
        """Assemble per-GROUP GDN states (each already batched [bg,...] on device, from
        forward_prefill_batched) into the full [B,...] batched decode buffers via device-side
        concat — no host round-trip. rec: concat groups along dim 0 -> [B,Nv,Dk,Dv]; conv_states[m]:
        concat groups along dim 1 -> [1,B,D]. The batched GDN bindings MUST already be rebound
        (writes in place). Row u == user u because groups are contiguous
        (group g = users [g*group_size : ...])."""
        dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]
        ng = len(group_rec_dev)
        for li, dn in enumerate(dn_layers):
            rec_full = ttnn.concat([group_rec_dev[g][li] for g in range(ng)], dim=0)  # [B, Nv, Dk, Dv]
            rec_src = rec_full if rec_full.dtype == dn.rec_state.dtype else ttnn.typecast(rec_full, dn.rec_state.dtype)
            ttnn.copy(rec_src, dn.rec_state)
            if rec_src is not rec_full:
                ttnn.deallocate(rec_src)
            ttnn.deallocate(rec_full)
            for g in range(ng):
                ttnn.deallocate(group_rec_dev[g][li])
            for m in range(dn.K):
                conv_full = ttnn.concat([group_conv_dev[g][li][m] for g in range(ng)], dim=1)  # [1, B, D]
                ttnn.copy(conv_full, dn.conv_states[m])
                ttnn.deallocate(conv_full)
                for g in range(ng):
                    ttnn.deallocate(group_conv_dev[g][li][m])
            # The fused decode reads its conv history from a separate RM buffer, not conv_states: full-replace it
            # from the conv_states just written (no-op unless the fused path is on and this is the decode binding).
            dn.sync_fused_conv_hist_from_states()

    def prefill_paged_grouped(self, token_ids_list, page_table, valid_lens=None, group_size=4):
        """Grouped batched SHORT-prompt prefill (single-pass, every valid_len <= one GDN bucket):
        process users in groups of <= group_size through ONE hybrid forward per group instead of B
        sequential B=1 forwards. Within a group GDN runs BATCHED (forward_prefill_batched, per-row
        valid_len masking — bit-exact, see test_gdn_tp_batched_prefill) and full-attention runs
        PER-USER (attention prefill is B=1 only). Groups of <=4 respect the GDN kernel cap
        BH=B*Nv_tp<=32. Same math as prefill_paged_peruser (including different prompts/lengths);
        it just amortizes underutilized GDN over the group. ``token_ids_list`` B [1, T_u];
        ``page_table`` [B, blocks_per_user]; ``valid_lens`` optional — every valid_len MUST be <=
        the derived bucket; longer prompts go to the chunked path. Returns B ttnn logits [1,1,vocab].
        """
        assert self.num_devices > 1, "prefill_paged_grouped is the TP (num_devices>1) path"
        assert self._paged_kv_caches is not None, "Call allocate_kv_caches first"
        B = len(token_ids_list)
        pt_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert pt_torch.shape[0] == B, "page_table must have one row per user"
        vlens = list(valid_lens) if valid_lens is not None else [int(t.shape[1]) for t in token_ids_list]

        gdn_chunk = self.args.gdn_chunk_size
        block_size = get_block_size(self._paged_kv_caches)
        # Common bucket for the group forward: round the longest prompt up to a GDN-chunk multiple.
        bucket = max(gdn_chunk, ((max(vlens) + gdn_chunk - 1) // gdn_chunk) * gdn_chunk)
        assert all(v <= bucket for v in vlens), "every valid_len must fit the single-pass bucket"

        # The fused chunk_gated_delta_rule op caps the group by its SCAN, which maps one (head,
        # v-block) row per core: BH = B*Nv_tp must stay <= the compute grid (~96-104 cores on P150).
        # With Nv_tp=12 that's B <= 8, and — unlike the old gated_delta_attn_seq kernel — it is
        # bucket-independent (SCAN L1 is state-sized, not chunk-count-sized). Validated bit-exact vs
        # per-user at B=8 for bucket 128 and 256 (test_gdn_fused_batch: ceiling + large-group). Buckets
        # >256 aren't produced here (callers route T>256 to per-user), so cap them at 1 defensively.
        gdn_max_bg = 8 if bucket <= 2 * gdn_chunk else 1
        group_size = max(1, min(group_size, gdn_max_bg))

        dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]
        rep = ttnn.ReplicateTensorToMesh(self.device)
        # cos/sin for absolute positions [0, bucket) — shared by all users (single pass from pos 0).
        cos_t, sin_t = self._rope_tp_cos_sin_torch(0, bucket)
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        csi = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )

        group_rec_dev, group_conv_dev = [], []
        host_logits = [None] * B
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        for g0 in range(0, B, group_size):
            grp = list(range(g0, min(g0 + group_size, B)))
            Bg = len(grp)
            prev = self._alloc_gdn_scratch_b(Bg)
            try:
                # Batched embedding: [1, Bg, bucket, dim] (pad each user's tokens to the bucket).
                tok_bg = torch.zeros(Bg, bucket, dtype=torch.int32)
                for i, u in enumerate(grp):
                    t = token_ids_list[u][0, : vlens[u]].to(torch.int32)
                    tok_bg[i, : t.shape[0]] = t
                tok = ttnn.from_torch(tok_bg, dtype=ttnn.uint32, device=self.device, mesh_mapper=rep)
                x = self.embd(tok)  # [Bg, bucket, d]
                d = x.shape[-1]
                # Canonical residual-stream shape [1, 1, Bg*bucket, d] (dim1==1) so the framework
                # norm / MLP / residual add see the SAME layout as the validated per-user path
                # (a [1,Bg,bucket,d] shape trips "invalid subtile broadcast" in the norm/residual).
                # Reshaped to [Bg, bucket, d] only for the batched GDN, and split to [1,Bg,bucket,d]
                # to slice each user for the per-user attention. Row order is user-major (user u owns
                # rows [u*bucket : (u+1)*bucket]), matching the group state assembly.
                x = ttnn.reshape(x, (1, 1, Bg * bucket, d))
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(tok)
                # Per-user device page tables (full + real-blocks-only for the KV fill).
                full_pts, chunk_pts = [], []
                for u in grp:
                    row = pt_torch[u : u + 1].contiguous()
                    full_pts.append(
                        ttnn.from_torch(row, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
                    )
                    blkN = num_blocks_in_seq(vlens[u], block_size)
                    chunk_pts.append(
                        ttnn.from_torch(
                            row[:, :blkN].contiguous(),
                            dtype=ttnn.int32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=self.device,
                        )
                    )

                for layer in self.layers:
                    # The DistributedNorm all-gathers to the FULL hidden dim for the module (dn), so
                    # attn_in's last dim is the full dim (!= d, the fractured residual-stream dim).
                    attn_in = layer.attention_norm(x, mode=Mode.PREFILL)  # [1, 1, Bg*bucket, full]
                    full = attn_in.shape[-1]
                    if layer.is_full_attention:
                        attn_in_b = ttnn.reshape(attn_in, (1, Bg, bucket, full))  # split users for slicing
                        outs = []
                        for i, u in enumerate(grp):
                            xi = ttnn.reshape(attn_in_b[:, i : i + 1, :, :], (1, 1, bucket, full))
                            oi = layer.attention.forward_prefill_paged(
                                xi,
                                cos,
                                sin,
                                full_pts[i],
                                chunk_page_table=chunk_pts[i],
                                chunk_start_idx=0,
                                chunk_start_idx_tensor=csi,
                                user_id=0,  # per-user page tables are single-row; blocks route via values
                            )
                            ttnn.deallocate(xi)  # per-user slice copy
                            outs.append(ttnn.reshape(oi, (1, 1, bucket, oi.shape[-1])))
                        # Concat user outputs along the seq dim -> [1, 1, Bg*bucket, d_out] (user-major).
                        attn_out = ttnn.concat(outs, dim=2) if Bg > 1 else outs[0]
                        for o in outs:
                            if o is not attn_out:
                                ttnn.deallocate(o)
                    else:
                        # Batched GDN over the group (per-row valid_len masking, from scratch).
                        gdn_in = ttnn.reshape(attn_in, (Bg, bucket, full))
                        attn_out = layer.attention.forward_prefill_batched(
                            gdn_in, chunk_size=gdn_chunk, valid_lens=[vlens[u] for u in grp], carry=False
                        )  # [1, Bg, bucket, d_out]
                        attn_out = ttnn.reshape(attn_out, (1, 1, Bg * bucket, attn_out.shape[-1]))
                    ttnn.deallocate(attn_in)
                    h = ttnn.add(x, attn_out)  # both [1, 1, Bg*bucket, d]
                    ttnn.deallocate(x)
                    ttnn.deallocate(attn_out)
                    ff_in = layer.ffn_norm(h, mode=Mode.PREFILL)
                    ff_out = layer.feed_forward.forward(ff_in, mode="prefill")
                    ttnn.deallocate(ff_in)
                    x = ttnn.add(h, ff_out)
                    ttnn.deallocate(h)
                    ttnn.deallocate(ff_out)

                # Final norm + per-user next-token logit at valid_len-1, read to host immediately.
                xn = self.norm(x, mode=Mode.PREFILL)  # [1, 1, Bg*bucket, full]
                ttnn.deallocate(x)
                xn_b = ttnn.reshape(xn, (1, Bg, bucket, xn.shape[-1]))
                for i, u in enumerate(grp):
                    x_last = xn_b[:, i : i + 1, vlens[u] - 1 : vlens[u], :]  # [1,1,1,full] (slice copy)
                    lg = ttnn.linear(x_last, self.lm_head_weight, **_lm_head_kwargs())
                    ttnn.deallocate(x_last)
                    host_logits[u] = (
                        ttnn.to_torch(lg, mesh_composer=comp).reshape(1, 1, -1)[:, :, : self.args.vocab_size].clone()
                    )
                    ttnn.deallocate(lg)
                ttnn.deallocate(xn)
                for t in full_pts + chunk_pts:
                    ttnn.deallocate(t)

                # Clone the group's batched GDN state (survives the next group's scratch reset).
                group_rec_dev.append([ttnn.clone(dn.rec_state) for dn in dn_layers])
                group_conv_dev.append([[ttnn.clone(dn.conv_states[m]) for m in range(dn.K)] for dn in dn_layers])
            finally:
                self._restore_gdn_batched(prev)

        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(csi)
        ttnn.synchronize_device(self.device)
        # Stitch the per-group states into the full [B,...] batched decode buffers (row u = user u).
        self._assemble_groups_gdn_dev(group_rec_dev, group_conv_dev)
        return self._reupload_host_logits(host_logits)

    def _fill_paged_cache_from_prefill(self, page_table):
        """Copy concat K/V into paged cache after prefill (one layer at a time to limit memory)."""
        for cache_idx, layer_idx in enumerate(self._attention_layer_indices):
            attn = self.layers[layer_idx].attention
            if attn.past_key is not None:
                k_cache, v_cache = self._paged_kv_caches[cache_idx]
                ttnn.experimental.paged_fill_cache(k_cache, attn.past_key, page_table, batch_idx=0)
                ttnn.experimental.paged_fill_cache(v_cache, attn.past_value, page_table, batch_idx=0)
                ttnn.deallocate(attn.past_key)
                ttnn.deallocate(attn.past_value)
                attn.past_key = None
                attn.past_value = None

    def prefill_paged(self, token_ids, page_table, valid_len=None, vision_tokens=None):
        """Prefill using paged attention for long sequences, concat for short.

        T > 1024: paged prefill (paged_fill_cache + chunked_sdpa) via prefill_layer_chunked with
        page_table. T <= 1024: direct concat prefill + post-hoc paged cache fill. ``token_ids``
        [B, T]; ``page_table`` [B, max_blocks_per_seq] int32. Returns logits [B, 1, vocab_size].
        """
        if self.num_devices > 1:
            return self._prefill_paged_tp(token_ids, page_table, valid_len=valid_len, vision_tokens=vision_tokens)

        B, T = token_ids.shape
        # Stage the per-request RoPE (M-RoPE for multimodal, 1D for text) before any cos/sin seam;
        # prefill_layer_chunked (T>1024) inherits the staged table.
        self._build_request_rope(token_ids[:, :valid_len] if valid_len else token_ids, vision_tokens)
        # Keep page_table as torch.Tensor for CPU slicing in prefill_layer_chunked.
        page_table_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        self.reset_state()

        # Concat-based prefill for SDPA.
        if T > 1024:
            logits = self.prefill_layer_chunked(
                token_ids, chunk_size=2048, page_table=page_table_torch, vision_tokens=vision_tokens
            )
        else:
            token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
            x = self.embd(token_ids_ttnn)
            x = self._scatter_vision_tokens(x, token_ids, vision_tokens)
            ttnn.deallocate(token_ids_ttnn)

            cos, sin = self.rope.get_prefill_rot_mats(0, T)

            for layer in self.layers:
                x = layer.forward(x, cos=cos, sin=sin, mode="prefill")

            x = self.norm(x, mode=Mode.PREFILL)
            x_last = x[:, -1:, :]
            logits = self._lm_head(x_last)
            ttnn.deallocate(x)

        # Post-prefill: paged_fill no-op if already paged (T>1024); copies concat KV otherwise.
        page_table_device = ttnn.from_torch(
            page_table_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        self._fill_paged_cache_from_prefill(page_table_device)

        return logits

    def decode_paged(self, token_ids, current_pos, page_table):
        """Single-token paged decode. Returns logits [B,1,vocab_size]."""
        B = token_ids.shape[0]
        # Accept torch or ttnn page_table.
        if isinstance(page_table, torch.Tensor):
            page_table = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = self.embd(token_ids_ttnn)
        ttnn.deallocate(token_ids_ttnn)

        # RoPE position offset by rope_delta for multimodal (KV position stays the true seq pos).
        position_ids = torch.full((B, 1), current_pos + self.rope.rope_delta, dtype=torch.long)
        cos, sin = self.rope.get_rot_mats(position_ids)

        # cur_pos [B] for paged ops (not [B*n_kv] like non-paged decode).
        cur_pos_tensor = ttnn.from_torch(
            torch.full((B,), current_pos, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        for layer in self.layers:
            if layer.is_full_attention:
                x = layer.forward(
                    x,
                    cos=cos,
                    sin=sin,
                    mode="decode",
                    position_tensor=cur_pos_tensor,
                    page_table=page_table,
                )
            else:
                x = layer.forward(x, cos=cos, sin=sin, mode="decode")

        x = self._final_norm_decode(x)
        logits = self._lm_head(x)
        ttnn.deallocate(x)

        return logits

    # Generator contract — decode

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """Build HOST decode inputs: (tokens_tt, cur_pos_tt, rope_packed, page_table_tt)."""
        from models.demos.blackhole.qwen36.tt.generator_interface import pack_rope_host

        B = tokens.shape[0]
        if self._serve_device_decode:
            return self._prepare_decode_inputs_host_resident(tokens, current_pos, page_table)
        tokens_tt = ttnn.from_torch(tokens.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        # Per-user positions: current_pos may be a [B] tensor (each user at its own position) or a
        # scalar (lockstep). Build a [B] int32 vector so cur_pos and rope carry one rotation per user.
        if isinstance(current_pos, torch.Tensor):
            pos_vec = current_pos.to(torch.int32).reshape(-1)
            assert pos_vec.shape[0] == B, f"current_pos length {pos_vec.shape[0]} != batch {B}"
        else:
            pos_vec = torch.full((B,), int(current_pos), dtype=torch.int32)
        # RoPE position is the KV position offset by rope_delta (multimodal compresses the position
        # space; post-image text has t==h==w so 1D RoPE at rope_pos is correct). cur_pos_tt below
        # stays the true KV position. rope_delta is 0 for text, so this is a no-op there.
        rope_pos_vec = pos_vec + torch.as_tensor(self._decode_rope_delta(B)).to(pos_vec.dtype)
        if self.num_devices > 1:
            # TP: rope_tp cos/sin [1,B,1,rope_dim] packed on host.
            rd = self.args.rope_head_dim
            inv_freq = 1.0 / (self.args.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
            freqs = torch.outer(rope_pos_vec.float(), inv_freq)  # [B, rd/2], per-user rotation
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().reshape(1, B, 1, rd).to(torch.bfloat16)
            sin = emb.sin().reshape(1, B, 1, rd).to(torch.bfloat16)
            if self._rope_perm:  # QWEN36_ROPE_PERM: permuted hd-wide [1,B,1,hd] (DecodeRope input)
                cos, sin = self._rope_widen(cos, sin)
            rope_packed = ttnn.from_torch(torch.cat([cos, sin], dim=0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        else:
            # Single-device decode is B=1 in this port; per-user single-device rope is out of scope.
            cos_host, sin_host = self.rope.get_cos_sin_host(int(rope_pos_vec[0]))  # HOST ttnn [1,1,rope_head_dim]
            rope_packed = pack_rope_host(cos_host, sin_host)  # torch-based (host)
        cur_pos_tt = ttnn.from_torch(pos_vec, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        page_table_tt = (
            ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            if page_table is not None
            else None
        )
        return tokens_tt, cur_pos_tt, rope_packed, page_table_tt

    _DECODE_MAX_WIDTH = 32  # token buffer width (one tile row); the sampler / decode width never exceeds it

    def _prepare_decode_inputs_host_resident(self, tokens, current_pos, page_table=None):
        """QWEN36_SERVE_DEVICE_DECODE host inputs: (tokens [1,1,1,32] u32 RM, cur_pos [B] i32 RM, rope_idx [1,B] u32 RM,
        page_table). Slot 3 stays the page table (Generator DECODE_PAGE_TABLE_INPUT_IDX).

        All values are only authoritative on a ``reload_inputs`` step (the plugin may hand stale ones otherwise, and
        the Generator then copies only slot 3), so nothing here may fail on them: tokens / rope_idx are clamped into
        their unsigned ranges instead of asserted. Inactive rows (pos == -1) keep cur_pos -1 (paged ops skip them; the
        in-trace plus_one skips negative entries) and look up RoPE row max(-1, 0) + rope_delta."""
        B = tokens.shape[0]
        assert B <= self._DECODE_MAX_WIDTH, f"decode width {B} exceeds the {self._DECODE_MAX_WIDTH}-wide token buffer"
        tokens_tt = ttnn.from_torch(
            self._resident_token_row(tokens, self._DECODE_MAX_WIDTH), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        if isinstance(current_pos, torch.Tensor):
            pos_vec = current_pos.to(torch.int32).reshape(-1)
            assert pos_vec.shape[0] == B, f"current_pos length {pos_vec.shape[0]} != batch {B}"
        else:
            pos_vec = torch.full((B,), int(current_pos), dtype=torch.int32)
        rope_idx = self._resident_rope_index(pos_vec, self._decode_rope_delta(B), self.rope.max_seq_len)
        rope_idx_tt = ttnn.from_torch(rope_idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        cur_pos_tt = ttnn.from_torch(pos_vec, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        page_table_tt = (
            ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            if page_table is not None
            else None
        )
        return tokens_tt, cur_pos_tt, rope_idx_tt, page_table_tt

    @staticmethod
    def _resident_token_row(tokens, lanes):
        """[1,1,1,lanes] int32 token row: the B decode tokens in lanes [0, B), zero padding after (pure torch)."""
        row = torch.zeros(1, 1, 1, lanes, dtype=torch.int32)
        flat = tokens.reshape(-1).to(torch.int32).clamp(min=0)
        row[0, 0, 0, : flat.shape[0]] = flat
        return row

    @staticmethod
    def _resident_rope_index(pos_vec, rope_delta, max_seq_len):
        """[1,B] int32 RoPE-table row per user = KV position + rope_delta (pure torch).

        rope_delta is 0 for text; multimodal compresses the position space. This is the position the host path fed
        rope.cos/sin: max(pos, 0) keeps idle rows (pos -1) in range; the clamp to the table also bounds a negative
        delta (or stale host positions on a non-reload step) so the in-trace embedding never reads out of range."""
        delta = rope_delta.to(torch.int64) if isinstance(rope_delta, torch.Tensor) else int(rope_delta)
        idx = pos_vec.reshape(-1).to(torch.int64).clamp(min=0) + delta  # scalar, or per-row [B] (batched serving)
        return idx.clamp(0, int(max_seq_len) - 1).to(torch.int32).reshape(1, -1)

    def _resident_rope_cos_sin(self, rope_idx):
        """cos / sin of the RoPE rows `rope_idx` ([1, B] uint32 RM), looked up on device, TILE bf16, as [1, 1, B, rope_dim].

        B == 1: [1,1,1,rd], exactly what unpack_rope handed the layer on the host path. B > 1: the host path handed
        [1,B,1,rd], which apply_partial_rope_decode immediately reshapes to [1,1,B,rd] (batch in the seq role); the
        embedding already yields that row-per-user layout ([1,B,rd] TILE -> a pure view), so the reshape is skipped.
        Rows are the host path's bf16 values bit for bit (rope.decode_rm_tables)."""
        B = rope_idx.shape[-1]
        if self._rope_perm:  # QWEN36_ROPE_PERM: permuted hd-wide [1,B,1,hd] rows (DecodeRope input)
            return self._rope_wide_rows(rope_idx)
        rd = self.args.rope_head_dim
        out = []
        for table in self.rope.decode_rm_tables():
            row = ttnn.embedding(rope_idx, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out.append(ttnn.reshape(row, (1, 1, B, rd)))
        return out[0], out[1]

    def _resident_decode_ids(self, tokens, B):
        """Token ids [1, B] uint32 RM of the B active rows of the [1,1,1,32] token buffer (B's idiom)."""
        flat = ttnn.reshape(tokens, (1, self._DECODE_MAX_WIDTH))
        if B == self._DECODE_MAX_WIDTH:
            return flat
        return ttnn.slice(flat, [0, 0], [1, B], [1, 1])

    def prepare_inputs_decode(self, tokens, current_pos, page_table=None):
        """Host-to-device transfer for decode inputs."""
        from models.tt_transformers.tt.common import copy_host_to_device

        host = self.prepare_decode_inputs_host(tokens, current_pos, page_table=page_table)
        return copy_host_to_device(host, mesh_device=self.mesh_device)

    def ttnn_decode_forward(
        self,
        tokens,
        current_pos,
        rot_mat_idxs=None,
        page_table=None,
        kv_cache=None,
        on_device_logits=False,
        **kwargs,
    ):
        """Generator decode forward. kv_cache accepted but unused (state is model-bound).

        on_device_logits=True: return the raw vocab-sharded shard for the on-device sampler.
        """
        from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope

        resident = self._serve_device_decode
        if resident:
            # rot_mat_idxs = rope_idx [1, B]; tokens = the [1,1,1,32] buffer -> the B active ids.
            cos, sin = self._resident_rope_cos_sin(rot_mat_idxs)
            tokens_in = self._resident_decode_ids(tokens, rot_mat_idxs.shape[-1])
        else:
            cos, sin = unpack_rope(rot_mat_idxs)
            tokens_in = tokens
        if on_device_logits:
            assert self.sampling is not None, "on_device_logits=True but self.sampling is None"
            logits = self._forward_decode(tokens_in, cos, sin, current_pos, page_table, sharded_lm_head=True)
            if resident:
                # Advance the resident decode state once per replay, AFTER every reader of cur_pos / rope_idx (the
                # layers above): the sampler then overwrites the token buffer in place, so the next replay consumes
                # (sampled token, pos + 1) with no host input. Idle rows (pos -1) stay -1. Only this device-sampling
                # variant advances: the host-sampling trace is reloaded by the host every step.
                ttnn.plus_one(current_pos, skip_negative_entries=True)
                ttnn.plus_one(rot_mat_idxs)
            # Sampler runs >=32-wide; pad B up to it (else shape mismatch). Extra slots unused.
            sampler_batch = self.sampling.tt_sampling.max_batch_size
            B = logits.shape[2]
            if B < sampler_batch:
                logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, sampler_batch - B), (0, 0)], value=0.0)
            # Bare tensor (not a tuple): the traced path passes this straight to capture_trace().
            return logits
        logits = self._forward_decode(tokens_in, cos, sin, current_pos, page_table)
        return logits, None

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False, is_log_probs=False):
        """Convert decode output to host torch. Host-sampling returns logits [B,S,vocab];
        on-device sampling returns sampled token ids or sampled-token log-probs.
        """
        if is_tokens or is_log_probs:
            # Sampled ids and old-path sampled-token log-probs are replicated across devices.
            if self.num_devices > 1:
                return ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).reshape(-1)[:B]
            return ttnn.to_torch(tt_out).reshape(-1)[:B]
        if self.num_devices > 1:
            # TP: read one replica (get_device_tensors[0]), not ConcatMeshToTensor (~4x readback).
            full = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
        else:
            full = ttnn.to_torch(tt_out).float()
        rows = full.reshape(-1, self.args.vocab_size)
        required_rows = B * S
        if rows.shape[0] < required_rows:
            # Decode bucketing returns only the active prefix. The shared
            # generator requests the fixed serving width, so add neutral rows
            # instead of splitting each vocabulary row during ``view(B, S, -1)``.
            rows = torch.nn.functional.pad(rows, (0, 0, 0, required_rows - rows.shape[0]))
        return rows[:required_rows].view(B, S, self.args.vocab_size)

    def _save_deltanet_states(self):
        """Snapshot GDN state to host (guard across decode-trace capture's double forward)."""
        saved = []
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                saved.append(
                    {
                        "recurrent": ttnn.to_torch(dn.recurrent_state),
                        "conv": ttnn.to_torch(dn.fused_conv_state),
                    }
                )
        return saved

    def _restore_deltanet_states(self, saved_states, device):
        """Restore GDN state via ttnn.copy (preserves trace-baked buffer addresses)."""
        idx = 0
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                saved = saved_states[idx]
                restored = ttnn.from_torch(
                    saved["recurrent"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                )
                ttnn.copy(restored, dn.recurrent_state)
                ttnn.deallocate(restored)
                restored_conv = ttnn.from_torch(
                    saved["conv"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                )
                ttnn.copy(restored_conv, dn.fused_conv_state)
                ttnn.deallocate(restored_conv)
                idx += 1
