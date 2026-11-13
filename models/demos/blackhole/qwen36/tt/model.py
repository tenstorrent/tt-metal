# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-9B text model for Blackhole P150.

tok_embeddings -> 32 x Qwen36DecoderLayer -> RMSNorm -> LM Head.
Hybrid state: KV cache (8 attn layers) + recurrent state (24 DeltaNet layers).
"""

import math
import os

import torch
from loguru import logger
from tqdm import tqdm
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.decode_allreduce import get_decode_allreduce
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup
from models.demos.blackhole.qwen36.tt.spec_argmax_fused import LANES, FusedShardArgmax
from models.demos.blackhole.qwen36.tt.spec_sample import VerifySampler, with_draws
from models.demos.blackhole.qwen36.tt.spec_step import (
    VERIFY_POSITIONS,
    VERIFY_ROPE,
    VERIFY_STEP_ROWS,
    VERIFY_TOKENS,
    pack_u32_rows,
    pack_width,
    rope_rows,
    rope_table_rows,
)
from models.tt_transformers.tt.common import Mode, get_block_size, num_blocks_in_seq


def spec_kv_write_conflicts(page_tables, positions, T, block_size):
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


class _MaskedPrefillTrace:
    """The captured masked-bucket prefill of one bucket (Qwen36Model.prepare_prefill_traces_masked): its persistent
    inputs, which a replay restages, and per variant (False: plain, True: with a prefill checkpoint) the trace id and
    the trace outputs (hidden [1, 1, bucket, dim / TP], logits [1, 1, vocab])."""

    def __init__(self, bucket, blocks):
        self.bucket = bucket
        self.blocks = blocks  # width of the chunk page table: KV blocks that the bucket's rows can fill
        self.tok = self.cos = self.sin = self.csi = self.full_pt = self.chunk_pt = self.sel = None
        self.sel_dtype = None
        self.masks = None  # gdn.tp.PrefillMasks
        self.checkpoint = None  # gdn.tp.PrefillCheckpoint.buffers, for a bucket that a KV block boundary falls inside
        self.trace_ids = {}
        self.outputs = {}

    def variants(self):
        return (False, True) if self.checkpoint is not None else (False,)

    def free(self):
        for t in (self.tok, self.cos, self.sin, self.csi, self.full_pt, self.chunk_pt, self.sel):
            if t is not None:
                ttnn.deallocate(t)
        for part in (self.masks, self.checkpoint):
            if part is not None:
                part.free()
        for out in self.outputs.values():
            for t in out:
                ttnn.deallocate(t)
        self.outputs = {}


class Qwen36Model:
    """Qwen3.5-9B text LM on Blackhole P150. HF_MODEL env var selects checkpoint."""

    def __init__(self, mesh_device, args, state_dict, tensor_cache_path=None):
        self.args = args
        self.device = mesh_device
        self.mesh_device = mesh_device  # Generator reads model.mesh_device
        self.num_devices = mesh_device.get_num_devices()
        if tpc.is_2d_mesh(mesh_device):
            self.num_devices = args.num_devices  # the TP degree (mesh axis 1); rows replicate or split the sequence
        # CCL for multi-device all-reduce; None on single device (ops no-op).
        if self.num_devices > 1:
            from models.tt_transformers.tt.ccl import TT_CCL

            self.tt_ccl = TT_CCL(mesh_device)
        else:
            self.tt_ccl = None
        # The replicated decode residual (decode_allreduce.py), or None; shared with every layer through tt_ccl.
        self._decode_ar = get_decode_allreduce(args, self.tt_ccl)
        self.configuration = args  # Generator reads model.configuration.max_seq_len
        self.sampling_dp = 1
        # Rope is host-recomputed each step, so callers explicitly request a
        # full input reload. Sampling does not alias the decode token input.
        self._tt_supports_decode_token_feedback = False
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
            lm_mapper = tpc.tp_shard_mapper(mesh_device, -1)
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
        self._chunked_trace_output = None
        self._chunked_chunk_size = None
        self._chunk_token_buf = None
        self._chunk_start_idx_tensor = None
        self._chunk_page_table_buf = None
        self._chunk_full_page_table_buf = None
        self._chunk_cos_buf = None
        self._chunk_sin_buf = None
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
        # Captured masked-bucket prefill (prepare_prefill_traces_masked): bucket -> _MaskedPrefillTrace, the GDN
        # workspace the traces read, and the host tensors of the last replay's inputs (held until the device is idle).
        self._masked_traces = {}
        self._masked_traces_recorded = False
        self._masked_trace_gdn = None
        self._masked_trace_refs = []
        # Persistent B=1 GDN prefill scratch (batched serving): allocated once at warmup, its buffer
        # addresses are baked into the chunk-prefill trace and reused by every prefill_paged_slots
        # replay, so it is never freed/reallocated (only zeroed in place). See _bind_gdn_prefill_scratch.
        self._gdn_prefill_scratch = None
        # gdn_block_state.GdnBlockState when the GDN request state lives in vLLM KV cache blocks
        # (Qwen36ForCausalLM.allocate_kv_cache_per_layer); the GDN layers then keep only a batch-1 prefill workspace.
        self.gdn_blocks = None

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
        # host_trim: every device op of the tower depends only on the padded patch bucket, and the packed
        # [num_image_tokens, H] rows (hidden-fractured) are trimmed on the host, so a new image size creates no
        # program after the serving traces are captured.
        return self.vision_model.forward(pixel_values, grid_thw=image_grid_thw, host_trim=True)

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
        return self.vision_model.forward(pixel_values_videos, grid_thw=video_grid_thw, host_trim=True)

    def _vision_placeholder_mask(self, token_ids):
        """True at the tokens that the current request's vision rows splice into: ``image_token_id`` when an image
        grid is stashed, ``video_token_id`` when a video grid is, ``image_token_id`` when neither is. The vision rows
        are packed in the order of these tokens (vision_rows). The vision-splice paths (_scatter_vision_tokens /
        _set_vision_merge / _vis_row_offset_for) use this to locate the placeholder positions, mirroring HF's
        ``input_ids == image_token_id`` / ``input_ids == video_token_id`` masks."""
        cfg = self.args.hf_config
        ids = []
        if self._req_image_grid_thw is not None or self._req_video_grid_thw is None:
            ids.append(int(cfg.image_token_id))
        if self._req_video_grid_thw is not None:
            ids.append(int(cfg.video_token_id))
        return torch.isin(token_ids, torch.tensor(ids, dtype=token_ids.dtype))

    def vision_rows(self, token_ids, image=None, video=None):
        """Run the vision tower over one prompt's images and videos and return the host rows that its prefill splices
        (vision_rows_host): one row per image or video placeholder token, in the order of the tokens, or None for a
        text prompt. ``image`` and ``video`` are (pixel values [num_patches, patch_dim], grid_thw [n, 3]) or None.
        ``token_ids`` holds the prompt from position 0 through at least its last placeholder token. Stashes both
        grids, which select the prompt's M-RoPE table (_build_request_rope) and placeholder tokens
        (_vision_placeholder_mask). Raises ValueError when a modality's rows and placeholder tokens differ in number.
        """
        cfg = self.args.hf_config
        rows = {}
        # Each forward allocates its rows on device, so each modality is read to the host before the next forward.
        if image is not None:
            rows[int(cfg.image_token_id)] = self.vision_rows_host(self.get_image_features(*image))
        if video is not None:
            rows[int(cfg.video_token_id)] = self.vision_rows_host(self.get_video_features(*video))
        self._req_image_grid_thw = None if image is None else image[1]
        self._req_video_grid_thw = None if video is None else video[1]
        if not rows:
            return None
        ids = torch.as_tensor(token_ids).reshape(-1)
        placeholders = ids[self._vision_placeholder_mask(ids)]
        for token, r in rows.items():
            n = int((placeholders == token).sum())
            if n != int(r.shape[0]):
                raise ValueError(f"the prompt has {n} placeholder tokens {token} for {int(r.shape[0])} vision rows")
        if len(rows) == 1:
            return next(iter(rows.values()))
        first = next(iter(rows.values()))
        packed = torch.empty(int(placeholders.numel()), first.shape[-1], dtype=first.dtype)
        for token, r in rows.items():
            packed[placeholders == token] = r
        return packed

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
        placeholders (_vision_placeholder_mask). ``vis_row_offset`` is the count of image
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
        pos = torch.nonzero(self._vision_placeholder_mask(flat[:cs]), as_tuple=False).reshape(-1)
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
        vis_host = self.vision_rows_host(vision_tokens)  # [num_image_tokens, Hg]
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

    def vision_rows_host(self, vision_tokens):
        """The packed vision rows as one host bf16 tensor [num_image_tokens, dim] (what _set_vision_merge splices); a
        host tensor passes through. A traced prefill reads the rows once, before its first replay: the vision tower
        allocates the device rows per request, after the traces were captured, so a replay can overwrite them before
        a later chunk splices them."""
        if vision_tokens is None or isinstance(vision_tokens, torch.Tensor):
            return vision_tokens
        if self.num_devices > 1:
            return ttnn.to_torch(vision_tokens, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=1)).to(
                torch.bfloat16
            )
        return ttnn.to_torch(vision_tokens).to(torch.bfloat16)

    def _vis_row_offset_for(self, token_ids, chunk_start):
        """Packed-vision-row offset for the prefill segment starting at absolute position
        ``chunk_start``: the number of image-placeholder tokens before it. The vision rows are
        packed in image-placeholder order, so this is the index of the first row this segment
        owns — used to splice a large image whose placeholders span multiple chunks / the tail."""
        if chunk_start <= 0:
            return 0
        return int(self._vision_placeholder_mask(token_ids[:, :chunk_start]).sum())

    def switch_mode(self, mode):
        """Generator mode-change hook; no-op (no prefetcher)."""
        return None

    def _lm_head(self, x, out_dtype=None):
        """LM-head matmul. Vocab-sharded mesh: partial logits + all-gather to full replicated.
        Single device: plain matmul.

        out_dtype: leave None (default) for the base/verify path — losslessness is defined by the
        BASE argmax, so that call must stay byte-identical. The MTP drafter passes float32: its
        argmax only has to rank candidates, and at bf16 4.7-5.5% of draft rejections at 8k/32k were
        EXACT ties in the drafter's own logits, i.e. tokens thrown away to a rounding coin flip."""
        kw, ccl_kw = {}, {}
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
        if self._lmhead_vocab_sharded:
            logits = self._gather_vocab(logits, **ccl_kw)
        return logits

    def _gather_vocab(self, shard, **ccl_kw):
        """All-gather of a vocab-sharded LM-head output to the full replicated vocab row (frees ``shard``)."""
        from models.tt_transformers.tt.ccl import tt_all_gather

        return tt_all_gather(
            shard,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=None,
            dim=len(shard.shape) - 1,
            topology=self.args.ccl_topology(),
            **ccl_kw,
        )

    def _final_norm_decode(self, x):
        """Final RMSNorm before the LM head (TP decode).

        The bare `self.norm(x, DECODE)` runs plain ttnn.rms_norm on a DRAM-interleaved [32,dim]
        tensor -> single tile-row -> 1 core (~80us/token). Passing the framework's 'lm_head' norm
        config runs the sharded multi-core norm across lm_head_core_grid instead; output_mem_config
        is forced back to DRAM so the LM-head matmul input is byte-identical (layout-only change).

        A full-width x is the replicated decode residual (decode_allreduce.py): its local norm, no gather.
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
        logger.info("Loading + remapping weights via Qwen36ModelArgs.load_state_dict()...")
        state_dict = args.load_state_dict()

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
        from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode

        tok = ttnn.from_torch(
            torch.tensor([[int(token_id)]], dtype=torch.int32),
            dtype=ttnn.uint32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x = self.embd(tok)  # [1,1,dim_frac]
        x = ttnn.reshape(x, (1, 1, 1, x.shape[-1]))  # [1,1,B=1,dim_frac]
        # RoPE position offset by rope_delta for multimodal (KV position cur_pos_tt stays `pos`).
        cos, sin = rot_mats_decode(
            self.device,
            self.args.rope_head_dim,
            self.args.max_seq_len,
            self.args.rope_theta,
            torch.tensor([pos + self.rope.rope_delta], dtype=torch.int32),
        )
        cur_pos_tt = ttnn.from_torch(
            torch.tensor([pos], dtype=torch.int32),
            dtype=ttnn.int32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="decode", position_tensor=cur_pos_tt)
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
        mask_bool = self._vision_placeholder_mask(flat_ids)
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

                    block_size = get_block_size(self._paged_kv_caches)
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
            logits = ttnn.linear(x, self.lm_head_weight)
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
        _cap = getattr(self, "_capture_layer", None)
        for _li, layer in enumerate(self.layers):
            if layer.is_full_attention:
                x = layer.forward(x, cos, sin, position_tensor=cur_pos_tensor, page_table=page_table, mode="decode")
            else:
                x = layer.forward(x, mode="decode")
            if _cap is not None and _li == _cap:  # debug: capture hidden after layer _cap
                self._cap_rec = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Independent clone of the live last-layer output: the MTP head consumes this hidden on a
        # LATER forward and all-gathers it, so it must survive the final norm + lm_head below (a
        # to_memory_config is a no-op alias when x is already DRAM, so the norm would free it).
        hidden = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG) if return_hidden else None
        x = self._final_norm_decode(x)
        if sharded_lm_head or self._ondev_argmax:
            # Pre-gather vocab-sharded logits (on-device sampling / greedy argmax).
            logits = ttnn.linear(x, self.lm_head_weight)
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
        cos_t, sin_t = self._rope_tp_cos_sin_torch(chunk_start, S)
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

    def prefill_for_spec(self, token_ids, page_table, actual_len, on_chunk, slot=0):
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
        """
        assert self.num_devices > 1, "prefill_for_spec is the TP path"
        assert 0 <= slot < self.args.max_batch_size, f"slot {slot} out of range [0,{self.args.max_batch_size})"
        if self.args.max_batch_size > 1:
            dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]
            dram = ttnn.DRAM_MEMORY_CONFIG
            prev = self._bind_gdn_prefill_scratch()
            try:
                logits = self._prefill_for_spec_b1(token_ids, page_table, actual_len, on_chunk)
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
        return self._prefill_for_spec_b1(token_ids, page_table, actual_len, on_chunk)

    def _prefill_for_spec_b1(self, token_ids, page_table, actual_len, on_chunk):
        """prefill_for_spec's body: one B=1 prompt into whatever GDN state is currently bound."""
        chunk_size = self._chunked_chunk_size or 2048
        num_full = actual_len // chunk_size
        tail_real = actual_len - num_full * chunk_size
        self._build_request_rope(token_ids[:, :actual_len], None)

        if num_full == 0:
            logits, hidden = self.prefill_masked_bucket(
                token_ids[:, :actual_len], page_table, actual_len=actual_len, chunk_start=0, return_hidden=True
            )
            on_chunk(hidden, 0, actual_len)
            self._release_prefill_hidden(hidden)
            return logits

        self._reset_gdn_state_for_new_sequence()
        last_hidden = None
        for c in range(num_full):
            cs = c * chunk_size
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
            # Not the last chunk, or there is a tail after it: the base hidden is consumed here.
            if c < num_full - 1 or tail_real > 0:
                on_chunk(last_hidden, cs, chunk_size)
        if tail_real > 0:
            ttnn.deallocate(last_hidden)
            cs = num_full * chunk_size
            logits, hidden = self.prefill_masked_bucket(
                token_ids[:, cs:actual_len], page_table, actual_len=tail_real, chunk_start=cs, return_hidden=True
            )
            on_chunk(hidden, cs, tail_real)
            self._release_prefill_hidden(hidden)
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

        hidden_states : fractured [1,1,B,dim/tp] — the base's drafter feed (spec_feed_rows: the
                        fractured final-norm output) or the previous MTP step's next_hidden
                        (chaining).
        token_id      : the token at each row's ``position``. int (B=1), list[int] of length B, or a
                        device [B,1] uint32 tensor (on-device draft chaining).
        position      : int (all rows at the same slot) or list[int] of length B (one per user).
        page_table    : device int32 page table [B, nb] for the MTP layer's own KV cache.
        rope_deltas   : one M-RoPE delta per row (its request's), or None: every row takes rope.rope_delta.
        Returns (logits, next_hidden).
        """
        from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode

        assert self.mtp is not None, "MTP head not built (has_mtp/config?)"
        pos_list = [int(p) for p in position] if isinstance(position, (list, tuple, torch.Tensor)) else [int(position)]
        deltas = [self.rope.rope_delta] * len(pos_list) if rope_deltas is None else [int(d) for d in rope_deltas]
        if len(deltas) != len(pos_list):
            raise ValueError(f"{len(deltas)} M-RoPE deltas for {len(pos_list)} positions")
        cos, sin = rot_mats_decode(
            self.device,
            self.args.rope_head_dim,
            self.args.max_seq_len,
            self.args.rope_theta,
            torch.tensor([p + d for p, d in zip(pos_list, deltas)], dtype=torch.int32),
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
    def _rope_tp_cos_sin_decode_torch(self, positions, deltas=None):
        """Torch cos/sin [1, B, 1, rope_head_dim] in the DECODE rope layout — one rotation per ROW.

        Byte-identical to what prepare_decode_inputs_host builds for a decode step at those positions
        (same inv_freq / outer / cat / bf16 cast). The hybrid verify pushes its candidates through the
        DECODE attention kernel as B pseudo-users, so it needs this per-row table, not the
        prefill table of positions 0..S-1 (_verify_cos_sin_torch lays it out as [1,1,B,rd]). ``deltas``: one M-RoPE
        delta per row (its request's), or None: every row takes rope.rope_delta."""
        rd = self.args.rope_head_dim
        pos_vec = positions.to(torch.int32).reshape(-1)
        # RoPE position is the KV position offset by rope_delta (0 for text); same as decode.
        if deltas is None:
            rope_pos_vec = pos_vec + self.rope.rope_delta
        else:
            deltas = torch.as_tensor(deltas, dtype=torch.int32).reshape(-1)
            if deltas.shape != pos_vec.shape:
                raise ValueError(f"{deltas.numel()} M-RoPE deltas for {pos_vec.numel()} positions")
            rope_pos_vec = pos_vec + deltas
        B = rope_pos_vec.shape[0]
        inv_freq = 1.0 / (self.args.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
        freqs = torch.outer(rope_pos_vec.float(), inv_freq)  # [B, rd/2], per-row rotation
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().reshape(1, B, 1, rd).to(torch.bfloat16)
        sin = emb.sin().reshape(1, B, 1, rd).to(torch.bfloat16)
        return cos, sin

    def _verify_cos_sin_torch(self, row_positions, deltas=None):
        """_rope_tp_cos_sin_decode_torch's values (with the same ``deltas``) as [1, 1, R, rope_head_dim]: the layout
        the attention layers' partial RoPE rotates in (apply_partial_rope_decode), so no layer reshapes the verify's
        cos / sin."""
        rd = self.args.rope_head_dim
        return tuple(t.reshape(1, 1, -1, rd) for t in self._rope_tp_cos_sin_decode_torch(row_positions, deltas))

    def rope_table_rows(self):
        """Rows of the device RoPE tables (rope_rm_tables), spec_step.rope_table_rows of max_seq_len."""
        return rope_table_rows(self.args.max_seq_len)

    def rope_row_deltas(self, n, deltas=None):
        """The M-RoPE delta of each of ``n`` rows for rope_rows: ``deltas`` (one per row), or rope.rope_delta for every
        row, the rule of _rope_tp_cos_sin_decode_torch."""
        if deltas is None:
            return int(self.rope.rope_delta)
        deltas = [int(d) for d in deltas]
        if len(deltas) != n:
            raise ValueError(f"{len(deltas)} M-RoPE deltas for {n} positions")
        return deltas

    def rope_rm_tables(self):
        """(cos, sin) bf16 ROW_MAJOR [rope_table_rows, rope_head_dim], replicated in DRAM. Row i holds
        _rope_tp_cos_sin_decode_torch of RoPE position i (KV position plus M-RoPE delta), so rope_lookup returns the
        bytes the host would stage. Built on first use; the callers build them before any trace capture."""
        if getattr(self, "_rope_rm", None) is None:
            rows, rd = self.rope_table_rows(), self.args.rope_head_dim
            positions = torch.arange(rows, dtype=torch.int32)
            tables = self._rope_tp_cos_sin_decode_torch(positions, torch.zeros(rows, dtype=torch.int32))
            self._rope_rm = tuple(
                ttnn.from_torch(
                    t.reshape(rows, rd).contiguous(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                )
                for t in tables
            )
        return self._rope_rm

    def rope_lookup(self, table_rows, n):
        """cos, sin [1, 1, n, rope_head_dim] bf16 TILE of the RoPE table rows ``table_rows`` ([1, n] uint32 ROW_MAJOR on
        the device), in the layout apply_partial_rope_decode rotates in. The caller frees both."""
        rd = self.args.rope_head_dim
        out = []
        for table in self.rope_rm_tables():
            row = ttnn.embedding(table_rows, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out.append(ttnn.reshape(row, (1, 1, n, rd)))
        return out[0], out[1]

    def _residual_ar(self, rows):
        """The DecodeResidualAllReduce a verify forward of ``rows`` rows runs on, or None for the fractured residual:
        the replicated residual is one tile of rows, and its GDN path exists only with the GDN state in vLLM KV cache
        blocks (bound before any verify trace is captured)."""
        if self._decode_ar is None or rows > ttnn.TILE_SIZE:
            return None
        if any(layer.attention.state_blocks is None for layer in self.layers if not layer.is_full_attention):
            return None
        return self._decode_ar

    def _spec_tail(self, x):
        """Shared tail of the verify and seed bodies: residual ``x`` [1,1,R,dim] (freed here) ->
        (logits [1,1,R,vocab] replicated per device, feed [1,1,R,dim/tp] fractured).

        The bucket is exactly the R real rows, so there is no row-select. ``feed`` is the post-norm
        drafter feed (spec_feed_rows); its programs are the inner norm's, already compiled by the
        self.norm call."""
        logits, feed, _ = self._spec_tail_ids(x, None)
        return logits, feed

    def _spec_tail_ids(self, x, shard_ids):
        """_spec_tail, and ``shard_ids(shard)`` run on this device's vocab shard of the logits before the vocab all-gather
        (vocab-sharded head only). Returns (logits, feed, ids): ids is shard_ids' result, or None without a sharded
        head or ``shard_ids``.

        A full-width x is the replicated decode residual: the final norm is its local norm and the feed is this
        device's dim/tp span of its output (mesh_partition, as Qwen36MTP.forward_decode re-fractures)."""
        if self._decode_ar is not None and x.shape[-1] == self.args.dim:
            normed = self._final_norm_decode(x)  # [1,1,R,dim] DRAM, replicated
            ttnn.deallocate(x)
            logits, ids = self._lm_head_ids(normed, shard_ids)
            feed = ttnn.mesh_partition(normed, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(normed)
            return logits, feed, ids
        rows = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)
        normed = self.norm(rows, mode=Mode.PREFILL)  # -> full [1,1,R,dim]
        logits, ids = self._lm_head_ids(normed, shard_ids)
        ttnn.deallocate(normed)
        feed = self.spec_feed_rows(rows)
        if feed is not rows:
            ttnn.deallocate(rows)
        return logits, feed, ids

    def _lm_head_ids(self, normed, shard_ids):
        """(logits, ids) of the full-width ``normed`` rows: ``shard_ids`` runs on this device's vocab shard before the
        all-gather (vocab-sharded head only); ids is None without it."""
        if shard_ids is None or not self._lmhead_vocab_sharded:
            return self._lm_head(normed), None
        shard = ttnn.linear(normed, self.lm_head_weight)
        ids = shard_ids(shard)
        return self._gather_vocab(shard), ids

    def _forward_verify_bucket_tp(self, n_users):
        """Trace body: recurrent verify over the persistent B*T-row bucket buffers -> per-position
        logits + hidden. Reads only fixed-address buffers so it is trace-capturable. Full-attention
        layers run the decode flash kernel with the rows as pseudo-users (see capture_verify_trace).
        GDN stays on seq-dim recurrent verify per user (batch dim is not a valid recurrence axis) and
        takes its per-user starting state through the two shared selectors. The rows' tokens, KV positions and RoPE
        table rows come from the step buffer _vfy_step_buf (spec_step), cos and sin from rope_lookup. Returns (logits,
        rows, ids, logits_rm): ``rows`` is the drafter feed [1,1,B*T,dim/tp] bf16 (spec_feed_rows).
        ``logits_rm`` is the ROW_MAJOR bf16 copy the in-trace argmax needs, kept for the sampling path.
        """
        from models.demos.blackhole.qwen36.tt.attention.tp import KV_GROUP_WRITE_OK

        T = self._vfy_T
        rows = n_users * T
        step = self._vfy_step_buf
        tokens = ttnn.slice(step, (VERIFY_TOKENS, 0), (VERIFY_TOKENS + 1, rows))
        positions = ttnn.slice(step, (VERIFY_POSITIONS, 0), (VERIFY_POSITIONS + 1, rows))
        kvpos = ttnn.reshape(ttnn.typecast(positions, ttnn.int32), (rows,))
        ttnn.deallocate(positions)
        rope_idx = ttnn.slice(step, (VERIFY_ROPE, 0), (VERIFY_ROPE + 1, rows))
        cos, sin = self.rope_lookup(rope_idx, rows)
        ttnn.deallocate(rope_idx)
        x = self.embd(tokens)
        ttnn.deallocate(tokens)
        x = ttnn.reshape(x, (1, 1, rows, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        dar = self._residual_ar(rows)
        if dar is not None:
            x = dar.enter(x)  # every layer and _spec_tail then run on the replicated residual
        # The positions of candidate j's rows {u*T + j}, one KV write call each: the same T slices serve every
        # attention layer (TPAttention._forward_verify).
        pos_groups = None
        if KV_GROUP_WRITE_OK and T > 1:
            pos_groups = [ttnn.slice(kvpos, (j,), (rows,), (T,)) for j in range(T)]
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
                    cos=cos,
                    sin=sin,
                    mode="decode",
                    position_tensor=kvpos,
                    page_table=self._vfy_kvpt_buf,
                    alias_kv_write=True,
                    spec_user_page_table=self._vfy_kvpt_users,
                    # The SDPA read folds those rows into a few batch rows (spec_multi_pos_tiles: groups
                    # of 4 candidates, n_users x the single-user split), so the KV cache streams out of
                    # DRAM once per group per layer instead of once per row. The op wants one aliased
                    # page-table row per group; the per-row KV write above still needs the full row-count
                    # alias table. A T with no L1-fitting split (7, 11, ...) uses the legacy per-row call.
                    spec_verify_mode=True,
                    spec_page_table=self._vfy_kvpt1_buf,
                    n_users=n_users,
                    spec_pos_groups=pos_groups,
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
                )
            ttnn.deallocate(x)
            x = x_new
        for t in (*(pos_groups or ()), kvpos, cos, sin):
            ttnn.deallocate(t)
        # Greedy acceptance compares token ids, so argmax the verify rows inside the trace and read back
        # B*T uint32 instead of B*T x vocab floats. The fused per-shard argmax (tt/spec_argmax_fused.py) takes the ids
        # from each device's vocab shard before the all-gather: [1,1,1,32] uint32 ROW_MAJOR, lane r = row r.
        logits, rows_out, ids = self._spec_tail_ids(x, self._vfy_shard_ids)  # [1,1,B*T,vocab] replicated, feed
        # The logits readback of a logits-mode verify reads the ROW_MAJOR copy. Without a fused shard argmax, ttnn.argmax
        # needs ROW_MAJOR input too (a TILE tensor takes a single-core internal untilize that is hopeless at this vocab
        # width). No 32-row pad: the multicore argmax is correct below a full tile of rows, and padding would multiply
        # the bytes untilize and argmax touch (see mtp.argmax_last).
        u = ttnn.untilize(logits, use_multicore=True)
        if ids is None:
            ids = ttnn.argmax(u, dim=-1, keepdim=False)  # [1,1,B*T] uint32 ROW_MAJOR
        # `u` is NOT deallocated: it is handed back as the ROW_MAJOR logits copy. The sampling path
        # reads THAT back instead of the TILE `logits`, which costs an untilize on host.
        return logits, rows_out, ids, u

    def _vfy_shard_ids(self, shard):
        """The verify's argmax ids from this device's vocab shard of the logits (fused per-shard argmax), or None when
        the shard cannot take it; the verify body then argmaxes the gathered logits."""
        fused = getattr(self, "_vfy_argmax_fused", None)
        if fused is None:
            fused = self._vfy_argmax_fused = FusedShardArgmax(self.mesh_device, self.vocab_size, self.num_devices)
        return fused(shard, LANES) if fused.supports(shard) is None else None

    def _vfy_row_positions(self, positions):
        """Per-ROW absolute positions for the user-major bucket: row u*T + j is positions[u] + j."""
        T = self._vfy_T
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
        if getattr(self, "_vfy_sample_trace_id", None) is not None:
            ttnn.release_trace(self.mesh_device, self._vfy_sample_trace_id)
            self._vfy_sample_trace_id = None

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

    def prepare_verify_trace(self, page_tables, T, warm_positions, sample=False):
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
        ``sample``: also compile the device sampler over the verify's logits (spec_sample.VerifySampler), which
        record_verify_trace captures as a trace of its own and verify_traced(sample_plan=...) replays.
        Idempotent for the same (B, T, nb, sample); a different key releases any live trace first.
        """
        from models.demos.blackhole.qwen36.tt.attention.tp import KV_GROUP_WRITE_OK
        from models.demos.blackhole.qwen36.tt.gdn.tp import spec_conv_sel, spec_state_blk_idx

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
        key = (B, T, nb, bool(sample))
        if getattr(self, "_vfy_prepared_key", None) == key and getattr(self, "_vfy_step_buf", None) is not None:
            return
        dev = self.device
        rep = ttnn.ReplicateTensorToMesh(dev)

        # A re-prepare must not leave the old trace parked: its buffers are about to be replaced
        # (prepare_spec_verify reallocates on a shape change) and its id would leak.
        self.release_verify_trace()
        self._vfy_prepared_key = None
        self._vfy_T, self._vfy_B = T, B
        self._vfy_nb = nb

        # The persistent per-replay input buffer (address baked into the trace): the rows' tokens, KV positions (KV is
        # written position-exact, paged_update_cache at absolute slots) and RoPE table rows, staged per replay by
        # verify_traced in one copy. The hybrid verify feeds the DECODE attention kernel one rotation per ROW at that
        # row's own position, looked up from the RoPE tables, which hold the host math of the plain decode step, so the
        # two paths' rotations are byte-identical and near-ties round the same way.
        self.rope_rm_tables()
        self._vfy_step_buf = ttnn.from_torch(
            self._vfy_step_host([[0] * T] * B, warm_positions, None),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=rep,
        )
        self._vfy_sampler = VerifySampler(dev, B, T, self.vocab_size) if sample else None
        # Grouped aliased KV write (TPAttention._write_kv_aliased): checked here for the warm passes and on
        # every replay (verify_traced), while the tables are still torch; see spec_kv_write_conflicts.
        self._vfy_pt_host = pt.clone()
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

        gdn = [layer.attention for layer in self.layers if not layer.is_full_attention]
        assert gdn, "the hybrid verify needs GDN layers"
        if self.gdn_blocks is not None:
            # The GDN state is in vLLM KV cache blocks: each GDN layer's verify reads its start state from, and
            # writes every token's state to, the blocks of the verify tables (GdnBlockState.set_verify_blocks), which
            # point at the null block until the first verify. No ring, window or selector exists.
            assert (
                self.gdn_blocks.verify_tokens == T
            ), f"the GDN verify tables are bound for T = {self.gdn_blocks.verify_tokens}, the verify trace has T = {T}"
            self._vfy_state_idx = self._vfy_conv_sel = None
            self.gdn_blocks.clear_verify_blocks()
            self._warm_verify_body(B)
            ttnn.synchronize_device(dev)
            self._vfy_att0 = _att0
            self._vfy_prepared_key = key
            return

        # The deferred commit's two selectors, shared by all GDN layers (pure index / one-hot data, so one
        # pair of buffers is staged per iteration instead of one per layer). state_blk_idx picks each
        # (user, v-head)'s starting block in the per-token state ring; conv_sel picks each user's conv
        # window rows out of [E_prev | new qkv]. Both start at mi = 0 for every user, as the seed leaves them.
        Nv, Kc = gdn[0].Nv, gdn[0].K
        self._vfy_Nv, self._vfy_Kc = Nv, Kc
        _mi0 = [0] * B
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
        self._warm_verify_body(B)
        ttnn.synchronize_device(dev)
        self._vfy_att0 = _att0
        self._vfy_prepared_key = key

    def _warm_verify_body(self, B):
        """One eager pass of the verify body, and of the device sampler over its logits, only to compile their
        programs before any capture; the outputs are dropped."""
        outs = self._forward_verify_bucket_tp(n_users=B)
        if self._vfy_sampler is not None:
            self._vfy_sampler.body(outs[0])  # writes the sampler's own persistent draws buffer
        for t in outs:
            ttnn.deallocate(t)

    def record_verify_trace(self):
        """Capture the verify trace over the buffers prepare_verify_trace allocated (warmup phase 2).
        With ``sample`` prepared, also the sampler trace, which reads the verify trace's logits.
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
        ) = self._forward_verify_bucket_tp(n_users=B)
        ttnn.end_trace_capture(dev, self._vfy_trace_id, cq_id=0)
        if self._vfy_sampler is not None:
            # A trace of its own, replayed only by a verify of a device-sampled step with a sampled row.
            self._vfy_sample_trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
            self._vfy_sampler.body(self._vfy_logits_out)
            ttnn.end_trace_capture(dev, self._vfy_sample_trace_id, cq_id=0)
        self._vfy_capture_count = getattr(self, "_vfy_capture_count", 0) + 1
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

    def verify_traced(
        self,
        tokens,
        positions,
        mi_prev,
        read_logits=False,
        page_tables=None,
        inactive=None,
        rope_deltas=None,
        sample_plan=None,
    ):
        """Replay the captured verify trace for B users x T candidates.

        ``tokens``: B lists of T ids ([pending] + drafts per user). ``positions``: B ints, user u's
        first verify slot this iteration (row u*T+j lands at positions[u]+j). ``mi_prev``: B ints,
        each user's accepted-prefix index from the previous iteration (0 right after the seed). It
        is the whole commit: it selects user u's starting recurrent state (ring block) and conv
        window rows, so no state is copied between iterations.

        Returns (ids: B*T host ints, feed_rows: the trace's persistent [1,1,B*T,dim/tp] drafter feed,
        logits: torch [B*T, vocab] or None). ``read_logits`` pulls the full logits to host (for the
        sampler; greedy needs only the in-trace argmax ids). It reads the trace's ROW_MAJOR copy
        (_vfy_logits_rm_out) with no cast, which avoids a host untilize; the sampler converts to
        float32 lazily per row.

        ``sample_plan`` (spec_sample.sampling_plan; the trace must be prepared with ``sample``): each row the plan
        samples returns the device sampler's draw in place of its argmax id.

        ``page_tables``: int32 [B, num_blocks] block tables (array-like; 1-D only for B == 1), one
        row per user. When given, every persistent page-table buffer is re-staged for this replay if the table differs
        from the staged one, with columns clipped or padded with block 0 (vLLM's null block) to the captured width.
        None keeps the tables staged last. A grouped KV write that would put two users on the same
        (block, 32-row tile) in the same call (block 0 excepted) raises AssertionError before anything is staged.
        The MTP drafter's own table (``SpeculativeDecoder.mtp_pt``) is a separate buffer this argument does
        not touch. ``inactive``: length-B bools; those users are skipped by the KV-write collision check only
        (their rows write the null block). ``rope_deltas``: B ints, user u's M-RoPE delta for all of its T rows;
        None: every row takes rope.rope_delta.

        ``feed_rows`` is the trace's own output buffer: do not deallocate it, and finish with it
        before the next replay (the spec loop reads its anchor rows and reseeds the drafter within
        the iteration).

        With the GDN state in vLLM KV cache blocks (gdn_blocks), ``mi_prev`` is not used: the caller writes the verify
        tables (GdnBlockState.set_verify_blocks) before the replay.
        """
        assert getattr(self, "_vfy_trace_id", None) is not None, "call record_verify_trace first"
        T, B = self._vfy_T, self._vfy_B
        if sample_plan is not None and self._vfy_sampler is None:
            raise ValueError("verify_traced(sample_plan=...) needs a verify trace prepared with sample=True")
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
        if self._vfy_kv_grouped:
            chk = positions if inactive is None else [-1 if x else int(p) for p, x in zip(positions, inactive)]
            self._vfy_check_kv_writes(pt, chk)
        # Only this method and prepare_verify_trace write the table buffers, so an unchanged table is not copied again.
        if page_tables is not None and not torch.equal(pt, self._vfy_pt_host):
            bufs = (self._vfy_kvpt_buf, self._vfy_kvpt_users, self._vfy_kvpt1_buf)
            for host, buf in zip(self._vfy_page_table_host(pt), bufs):
                _h = ttnn.from_torch(host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep)
                ttnn.copy_host_to_device_tensor(_h, buf)
            self._vfy_pt_host = pt.clone()  # the caller may reuse its table tensor

        # The rows' tokens, positions and RoPE table rows: one copy into the persistent step buffer.
        _h = ttnn.from_torch(
            self._vfy_step_host(tokens, positions, rope_deltas),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None,
            mesh_mapper=rep,
        )
        ttnn.copy_host_to_device_tensor(_h, self._vfy_step_buf)
        if self.gdn_blocks is None:
            self._stage_verify_selectors(mi_prev, B, rep)
        if sample_plan is not None:
            self._vfy_sampler.stage(sample_plan)

        ttnn.execute_trace(dev, self._vfy_trace_id, cq_id=0, blocking=False)
        if sample_plan is not None:
            ttnn.execute_trace(dev, self._vfy_sample_trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)

        # Replicated ids (the trace argmaxed the replicated logits): one replica is the whole answer,
        # and it is B*T uint32 instead of B*T x vocab floats.
        ids = ttnn.to_torch(ttnn.get_device_tensors(self._vfy_ids_out)[0]).reshape(-1)
        if sample_plan is not None:
            ids = with_draws(ids[:rows].reshape(B, T), self._vfy_sampler.read(), sample_plan).reshape(-1)
        ids = [int(v) for v in ids[:rows]]
        lt = None
        if read_logits:
            lt = ttnn.to_torch(ttnn.get_device_tensors(self._vfy_logits_rm_out)[0])
            lt = lt.reshape(-1, self.vocab_size)[:rows]
        return ids, self._vfy_rows_out, lt

    def _vfy_step_host(self, tokens, positions, rope_deltas):
        """The verify step buffer's rows (spec_step) of B users x T tokens: ``tokens`` B lists of T ids, ``positions`` user
        u's first verify slot (row u*T + j sits at positions[u] + j, which doubles as the decode SDPA's per-row cur_pos and
        makes the hybrid verify causal), ``rope_deltas`` B ints (None: every row takes rope.rope_delta). Rows are
        user-major, the row-major flatten of the B x T token lists."""
        T, B = self._vfy_T, self._vfy_B
        tok = torch.tensor([int(t) for row in tokens for t in row], dtype=torch.int32)
        pos = self._vfy_row_positions(positions)
        deltas = None if rope_deltas is None else [int(d) for d in rope_deltas for _ in range(T)]
        ridx = rope_rows(pos, self.rope_row_deltas(B * T, deltas), self.rope_table_rows())
        rows = [None] * VERIFY_STEP_ROWS
        rows[VERIFY_TOKENS], rows[VERIFY_POSITIONS], rows[VERIFY_ROPE] = tok, pos, ridx
        return pack_u32_rows(rows, width=pack_width(B * T))

    def _stage_verify_selectors(self, mi_prev, B, rep):
        """The deferred commit (see prepare_verify_trace): where each user's recurrence and conv window resume in the
        model-owned ring. Two small host-built tensors, shared by every GDN layer."""
        from models.demos.blackhole.qwen36.tt.gdn.tp import spec_conv_sel, spec_state_blk_idx

        T = self._vfy_T
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
        return self._spec_tail(x)

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
        # Absolute cache slot per row, doubling as the decode SDPA's cur_pos (the verify trace's positions at T = 1).
        kvpos = ttnn.from_torch(pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, mesh_mapper=rep)
        kvpt = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, mesh_mapper=rep)
        cos_t, sin_t = self._rope_tp_cos_sin_decode_torch(pos)
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
        from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode

        rep = ttnn.ReplicateTensorToMesh(self.device) if self.num_devices > 1 else None
        mk = dict(mesh_mapper=rep) if rep is not None else {}
        tok = ttnn.from_torch(
            torch.tensor([[int(token_id)]], dtype=torch.int32), dtype=ttnn.uint32, device=self.device, **mk
        )
        cos, sin = rot_mats_decode(
            self.device,
            self.args.rope_head_dim,
            self.args.max_seq_len,
            self.args.rope_theta,
            torch.tensor([pos + self.rope.rope_delta], dtype=torch.int32),
        )
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

    def _rope_tp_cos_sin_torch(self, start, length):
        """Torch cos/sin tables [1, 1, length, rope_head_dim] for SEQUENCE positions
        [start, start+length), in the rope_tp (HF split-halves) format consumed by
        apply_partial_rope_prefill. Single source of truth for the TP masked-bucket and
        traced chunk-outer prefill paths (so the captured trace's cos/sin are byte-identical
        to the eager path's). M-RoPE-aware: when a multimodal request staged a per-sequence
        table (build_request_rope) this slices it; otherwise it is ordinary 1D RoPE at
        positions [start, start+length) — byte-identical to the pre-M-RoPE behaviour."""
        rd = self.args.rope_head_dim
        cos_t, sin_t = self.rope.prefill_cos_sin_torch(start, length)  # [length, rd] bf16
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
        gdn_capture_state=True,
        gdn_carry_state=True,
        sp=None,
    ):
        """TP trace-safe single-chunk prefill (replicated persistent buffers).
        Full chunk (valid_len==chunk_size); flexible SDPA via device chunk_start_idx.
        seq_len: tokens in token_buf (default _chunked_chunk_size); another size serves a prefill-only bucket.
        gdn_capture_state=False skips the GDN decode-state (conv_states) refresh, for prefill-only traces.
        gdn_carry_state=False: GDN layers start from zero state and drop their final state (prefill-only bucket traces).
        sp: tt/sp.py SeqParallel; token_buf/cos/sin hold this row's seq_len tokens of a sequence split over mesh rows.
        Returns hidden [1,1,seq_len,dim]."""
        chunk_size = self._chunked_chunk_size if seq_len is None else seq_len
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
                    sp=sp,
                )
            else:
                # valid_len=None: no GDN mask; trace-safe static conv capture (matches valid_len==chunk_size).
                x_new = layer.forward(
                    x,
                    mode="prefill",
                    chunk_size=self.args.gdn_chunk_size,
                    valid_len=None,
                    gdn_capture_state=gdn_capture_state,
                    gdn_carry_state=gdn_carry_state,
                    sp=sp,
                )
            ttnn.deallocate(x)
            x = x_new
        return x

    def capture_prefill_trace_chunked(self, device, page_table, chunk_size=2048, warmup_masked_buckets=True):
        """Capture one chunk's all-layer prefill as a trace; replayed per chunk.

        Chunk-outer prefill stays under the 4 GiB trace limit at long context.
        Flexible SDPA (runtime chunk_start) makes one trace serve all chunk positions."""
        self.prepare_prefill_trace_chunked(device, page_table, chunk_size, warmup_masked_buckets)
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
        self._alloc_vision_merge_buffers(device, chunk_size)

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

    def prefill_paged_slots(self, token_ids_list, page_table, empty_slots, valid_lens=None):
        """vLLM continuous-batching TP prefill: prefill each new request into ITS decode slot.
        Per-slot analogue of prefill_paged_peruser: a new request is prefilled while other decode
        slots are live, so each user's B=1 state must land in row empty_slots[u] WITHOUT disturbing
        the others (GDN is a fixed [B,...] buffer indexed by slot, not paged). Each request runs B=1 on
        the persistent GDN scratch (pre-warmed masked bucket / chunk trace), then its state is cloned on
        device and written into its slot with write_slot (preserving live rows) before the next request.
        Attention fills each user's blocks via its page-table row. ``token_ids_list`` N [1,T_u];
        ``page_table`` [N, max_blocks]; ``empty_slots`` N ints; ``valid_lens`` optional. Returns N host
        torch logits [1,1,vocab] in call order. Call allocate_kv_caches(batch_size=B) + batched warmup
        first; long prompts use prefill_traced_chunked via pre-warmed programs (no post-park compile).
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

    def warmup_gdn_slot_ops(self):
        """Compile the per-slot GDN state edits (write_slot per slot, remap_slots) before any trace is live."""
        if self._gdn_slot_ops_warmed:
            return
        dn_states = [layer.attention for layer in self.layers if not layer.is_full_attention]
        dram = ttnn.DRAM_MEMORY_CONFIG
        prev = self._bind_gdn_prefill_scratch()
        try:
            # The prefill paths' device copy: clone from the bound scratch, then write_slot.
            dev_snaps = [
                [
                    (
                        ttnn.clone(dn.rec_state, memory_config=dram),
                        [ttnn.clone(c, memory_config=dram) for c in dn.conv_states],
                    )
                    for dn in dn_states
                ]
                for _ in range(self.args.max_batch_size)
            ]
        finally:
            self._unbind_gdn_prefill_scratch(prev)
        for slot in range(self.args.max_batch_size):
            for dn, (r, cs) in zip(dn_states, dev_snaps[slot]):
                dn.write_slot(slot, r, cs)
        # A full permutation touches every row, so one remap compiles every gather program.
        self._remap_gdn_slots(list(range(self.args.max_batch_size))[::-1])
        ttnn.synchronize_device(self.mesh_device)
        self._gdn_slot_ops_warmed = True

    def _remap_gdn_slots(self, remap):
        """Apply a vLLM batch-condense slot_remap to every GDN layer's batched decode state
        (device-side; slot i takes the state at slot remap[i]). Mirrors seed_manager.apply_slot_remap
        for GDN's per-slot recurrent+conv state, which the plugin's slot_remap does not itself move.
        No-op for an identity remap."""
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.remap_slots(remap)

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
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.use_fused_recurrent_decode = bool(enabled)

    @property
    def gdn_fused_decode(self) -> bool:
        """True when every GDN layer decodes with the fused recurrent op (see set_gdn_fused_decode)."""
        gdn = [layer.attention for layer in self.layers if not layer.is_full_attention]
        return bool(gdn) and all(dn.use_fused_recurrent_decode for dn in gdn)

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
        block_size = get_block_size(self._paged_kv_caches)
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
        self,
        token_buf,
        valid_len,
        chunk_start,
        page_table,
        bucket,
        flex_sdpa=True,
        vision_tokens=None,
        checkpoint_len=None,
    ):
        """Single masked fixed-bucket prefill forward over `bucket` positions.

        First valid_len tokens real; rest padded. Attn runs full bucket; GDN masks via valid_len.
        Returns hidden [1,bucket,hidden] or [1,1,bucket,hidden] (TP)."""
        if self.num_devices > 1:
            return self._forward_prefill_chunk_masked_tp(
                token_buf,
                valid_len,
                chunk_start,
                page_table,
                bucket,
                flex_sdpa=flex_sdpa,
                vision_tokens=vision_tokens,
                checkpoint_len=checkpoint_len,
            )
        if checkpoint_len:
            raise ValueError("a prefill checkpoint inside a masked-bucket forward needs a TP mesh")
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
        checkpoint_len=None,
        inputs=None,
        checkpoint=False,
    ):
        """TP (num_devices>1) masked fixed-bucket single-chunk prefill forward.

        flex_sdpa=True: flexible chunked SDPA (serving). flex_sdpa=False: host-int path (debug).
        Fills K/V for real blocks only. Returns hidden [1,1,bucket,dim].
        checkpoint_len: every GDN layer also computes its state after the first checkpoint_len tokens
        (TPGatedDeltaNet.prefill_checkpoint).
        inputs: a _MaskedPrefillTrace. The forward then reads the inputs that a replay staged in its persistent tensors
        instead of token_buf, valid_len, chunk_start, page_table and checkpoint_len, and makes no host write, so it can
        be captured; ``checkpoint`` adds the trace's persistent checkpoint. The tensors stay allocated."""
        if inputs is not None:
            tok, cos, sin, csi_tensor = inputs.tok, inputs.cos, inputs.sin, inputs.csi
            full_pt, chunk_pt = inputs.full_pt, inputs.chunk_pt
            # The device chunk start places the SDPA; the host int only sizes a page table narrower than full_pt.
            chunk_start = 0
            gdn = dict(valid_len=None, gdn_masks=inputs.masks, gdn_checkpoint=inputs.checkpoint if checkpoint else None)
        else:
            tok = ttnn.from_torch(
                token_buf.to(torch.int32),
                dtype=ttnn.uint32,
                device=self.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        x = self.embd(tok)
        x = ttnn.reshape(x, (1, 1, bucket, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        if inputs is None:
            ttnn.deallocate(tok)
        # Trace-safe vision splice (fixed-shape where over the hidden-sharded persistent buffers,
        # sliced to bucket; identity when the mask is zero). The caller stages the buffers
        # (prefill_masked_bucket -> _set_vision_merge). No-op until a trace is captured.
        x = self._apply_vision_merge(x, length=bucket)
        ck = None
        if inputs is None:
            block_size = get_block_size(self._paged_kv_caches)
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
                page_table[:, blk0:blkN].contiguous(),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
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
            if checkpoint_len:
                from models.demos.blackhole.qwen36.tt.gdn.tp import PrefillCheckpoint

                ck = PrefillCheckpoint(self.device, bucket, checkpoint_len, self.args.gdn_conv_kernel_size)
            # full_chunk: GDN takes the traced body's conv (KDA) so eager == traced
            gdn = dict(valid_len=None if full_chunk else valid_len, gdn_checkpoint=ck)
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
                x_new = layer.forward(x, mode="prefill", chunk_size=self.args.gdn_chunk_size, **gdn)
            ttnn.deallocate(x)
            x = x_new
            if _cap is not None and _li == _cap:  # debug: capture hidden after layer _cap
                self._cap_chunk = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if inputs is not None:
            return x
        # Deallocate per-chunk inputs; only hidden survives (avoids OOM in eager 64k loop).
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(full_pt)
        ttnn.deallocate(chunk_pt)
        if csi_tensor is not None:
            ttnn.deallocate(csi_tensor)
        if ck is not None:
            ck.free()
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
        stage_rope=True,
        checkpoint_len=None,
    ):
        """Masked fixed-bucket prefill for a segment of `actual_len` real tokens.

        Pads to a fixed bucket, runs all layers ONCE, and masks GDN recurrent + conv state to exactly
        `actual_len` real tokens — numerically equivalent to eager exact-length (prefill_paged) but
        using one of a few bucket-sized programs, so warmup can compile every path before a trace is
        parked (avoids the compile-clobbers-trace hang). ``return_hidden`` also returns per-position
        pre-final-norm hidden [1,1,bucket,dim/tp] (MTP KV-warming; caller deallocates). ``chunk_start``
        is the segment's absolute start (0 for a short prompt; num_full*chunk_size for a long tail —
        carried GDN/KV must already be in place). ``vis_row_offset`` is image-placeholder tokens before this segment.
        ``stage_rope`` False keeps the RoPE table the caller staged (prefill_traced_chunked, from the whole prompt).
        ``checkpoint_len`` (TP): every GDN layer also computes its state after the segment's first checkpoint_len
        tokens (TPGatedDeltaNet.prefill_checkpoint).
        TP with a captured masked-bucket trace for the bucket (record_prefill_traces_masked): the forward and the
        logits replay it with the same operations; a returned hidden is then the trace's output, which the caller
        hands to _release_prefill_hidden instead of freeing it.
        Returns host [1,1,vocab] logits at actual_len-1.
        """
        B_batch, _ = token_ids.shape
        assert B_batch == 1, "masked-bucket prefill is single-sequence"
        if bucket is None:
            bucket = self._mask_bucket_for(actual_len)
        assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"
        traced = self._masked_prefill_trace(bucket, checkpoint_len) if flex_sdpa else None

        if chunk_start == 0:
            # chunk_start==0: new sequence, re-zero GDN (a replay starts from zero state itself). chunk_start>0: tail,
            # keep carried state.
            if traced is None:
                self._reset_gdn_state_for_new_sequence()
            # Stage the per-request RoPE for this segment (M-RoPE for multimodal, 1D for text).
            # Only at the sequence start; a carried tail (chunk_start>0) keeps the table the
            # long-prompt entry (prefill_traced_chunked) already staged.
            if stage_rope:
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

        if traced is not None:
            hidden, logits = self._replay_masked_trace(
                traced, token_buf, page_table, actual_len, chunk_start, checkpoint_len
            )
            return (logits, hidden) if return_hidden else logits

        hidden = self._forward_prefill_chunk_masked(
            token_buf,
            actual_len,
            chunk_start,
            page_table,
            bucket,
            flex_sdpa=flex_sdpa,
            vision_tokens=vision_tokens,
            checkpoint_len=checkpoint_len,
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

    def _masked_bucket_logits_tp(self, hidden, actual_len, bucket, sel=None):
        """TP: one-hot select row actual_len-1, norm, lm_head. Returns replicated [1,1,vocab].
        sel: a device tensor holding that one-hot ([1,1,1,bucket], hidden's dtype) in place of actual_len (a captured
        trace); the caller owns it."""
        sel_tt = sel
        if sel is None:
            sel_host = torch.zeros(1, 1, 1, bucket, dtype=torch.float32)
            sel_host[0, 0, 0, actual_len - 1] = 1.0
            sel_tt = ttnn.from_torch(
                sel_host,
                dtype=hidden.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        x_last = ttnn.matmul(sel_tt, hidden)  # [1, 1, 1, dim]
        if sel is None:
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
        if self.gdn_blocks is not None:
            self._warmup_prefill_checkpoints(page_table, buckets, block_size)
        ttnn.synchronize_device(self.device)

    def _warmup_prefill_checkpoints(self, page_table, buckets, block_size):
        """Compile the extra GDN programs of a prefill checkpoint inside a masked-bucket forward
        (TPGatedDeltaNet.forward_prefill ``checkpoint``) with one checkpointed forward per bucket that has a KV block
        boundary inside it: vLLM places checkpoints on block boundaries. Those programs key on the bucket, not on the
        checkpoint position. The GDN layers keep the last forward's checkpoint for GdnBlockState.warmup, which compiles
        its block write."""
        for bucket in sorted(buckets):
            if bucket > block_size:
                toks = torch.zeros(1, bucket, dtype=torch.int32)
                self.prefill_masked_bucket(
                    toks, page_table, actual_len=bucket, bucket=bucket, checkpoint_len=block_size
                )

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

    # ----------------------------------------------------------------------- #
    # Captured masked-bucket prefill
    # ----------------------------------------------------------------------- #
    # An eager masked-bucket forward dispatches every operation of all layers from the host, which costs more than the
    # device time of a 2048-token forward at TP=8. One trace per bucket (and one more per bucket that a KV block
    # boundary falls inside, which also computes a prefill checkpoint) serves every length, start position, page
    # table and checkpoint position: the operations are those of the eager masked forward, and what the eager forward
    # builds on the host from those values (tokens, RoPE rows, page tables, the GDN masks, the logits row and the
    # checkpoint selectors) is staged into persistent inputs before each replay. The trace starts by copying its
    # initial GDN state (zeros or the workspace) from persistent buffers, so a new sequence costs no copies. K / V
    # rows past the real blocks of the chunk page table go to the null block 0, which holds no request's data.

    def _gdn_layers(self):
        return [layer.attention for layer in self.layers if not layer.is_full_attention]

    def _gdn_workspace(self):
        return tuple(t for dn in self._gdn_layers() for t in (dn.rec_state, dn.conv_carry))

    def _log_trace_region(self, label):
        """Log the bytes of the trace region in use; never raises."""
        try:
            view = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.TRACE)
            used = int(view.total_bytes_allocated_per_bank) * int(view.num_banks)
            cap = int(view.total_bytes_per_bank) * int(view.num_banks)
        except Exception as e:  # noqa: BLE001 - a log line only
            logger.info(f"Trace region use after {label}: unavailable ({e})")
            return
        logger.info(f"Trace region use after {label}: {used / 2**20:.1f} of {cap / 2**20:.0f} MiB")

    def prepare_prefill_traces_masked(self):
        """Allocate the persistent inputs of the captured masked-bucket prefill of every bucket in
        _PREFILL_MASK_BUCKETS, and the GDN layers' trace buffers, and compile every program of the traced forward by
        running it eagerly on them. record_prefill_traces_masked captures the traces. TP, after
        prepare_prefill_trace_chunked (the traces use its page-table width and vision-splice buffers) and before any
        trace capture, with the batch-1 GDN workspace bound as prefills run. Idempotent."""
        from models.demos.blackhole.qwen36.tt.gdn.tp import PrefillCheckpoint, PrefillMasks

        if self._masked_traces:
            return
        if self.num_devices < 2 or self._chunked_prepared_key is None:
            raise ValueError("the masked-bucket prefill traces need a TP mesh and prepare_prefill_trace_chunked first")
        device = self.device
        block_size = get_block_size(self._paged_kv_caches)
        kconv = self.args.gdn_conv_kernel_size
        buf_blocks = int(self._chunk_full_page_table_buf.shape[-1])
        buckets = sorted(self._PREFILL_MASK_BUCKETS)
        if self._vis_buf is None:
            self._alloc_vision_merge_buffers(device, self.prefill_chunk_size)
        for dn in self._gdn_layers():
            dn.alloc_prefill_trace_buffers(checkpoint=buckets[-1] > block_size)
        rep = ttnn.ReplicateTensorToMesh(device)

        def up(t, dtype, layout):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, mesh_mapper=rep)

        traces = {}
        for b in buckets:
            tr = _MaskedPrefillTrace(b, -(-b // block_size))
            tr.tok = up(torch.zeros(1, b, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            cos_t, sin_t = self._rope_tp_cos_sin_torch(0, b)
            tr.cos = up(cos_t, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            tr.sin = up(sin_t, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            tr.csi = up(torch.zeros(1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            tr.full_pt = up(torch.zeros(1, buf_blocks, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            tr.chunk_pt = up(torch.zeros(1, tr.blocks, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            tr.masks = PrefillMasks(device, b, kconv)
            if b > block_size:  # vLLM's checkpoints are on block boundaries
                tr.checkpoint = PrefillCheckpoint.buffers(device, b, kconv)
            traces[b] = tr
        # The logits row selector has the hidden's dtype, as _masked_bucket_logits_tp builds it.
        hidden = self._forward_prefill_chunk_masked_tp(None, None, 0, None, buckets[0], inputs=traces[buckets[0]])
        for tr in traces.values():
            tr.sel_dtype = hidden.dtype
            tr.sel = up(torch.zeros(1, 1, 1, tr.bucket, dtype=torch.float32), hidden.dtype, ttnn.TILE_LAYOUT)
        ttnn.deallocate(hidden)
        # Each variant twice: a capture must find every program compiled and every lazily allocated buffer allocated.
        # The eager operations around a replay (staging the initial state, copying the logits out) compile here too.
        for dn in self._gdn_layers():
            dn.stage_prefill_init(carry=True)
            dn.stage_prefill_init(carry=False)
        for tr in traces.values():
            for checkpoint in tr.variants():
                for _ in range(2):
                    hidden, logits = self._masked_trace_body(tr, checkpoint)
                    ttnn.deallocate(ttnn.clone(logits))
                    ttnn.deallocate(hidden)
                    ttnn.deallocate(logits)
        ttnn.synchronize_device(device)
        self._reset_gdn_state_for_new_sequence()
        self._masked_traces = traces
        self._masked_trace_gdn = self._gdn_workspace()
        logger.info(
            f"Masked-bucket prefill traces prepared: buckets {buckets}, with a prefill checkpoint "
            f"{[b for b in buckets if traces[b].checkpoint is not None]}"
        )

    def record_prefill_traces_masked(self):
        """Capture the traces that prepare_prefill_traces_masked prepared. Allocates only the trace outputs, which a
        replay's caller reads before the next replay. Idempotent."""
        if not self._masked_traces:
            raise ValueError("call prepare_prefill_traces_masked first")
        if self._masked_traces_recorded:
            return
        device = self.device
        with trace_allocation_tracker.corruptible_allocation_scope(device):
            for tr in self._masked_traces.values():
                for checkpoint in tr.variants():
                    tid = ttnn.begin_trace_capture(device, cq_id=0)
                    tr.outputs[checkpoint] = self._masked_trace_body(tr, checkpoint)
                    ttnn.end_trace_capture(device, tid, cq_id=0)
                    tr.trace_ids[checkpoint] = tid
        self._reset_gdn_state_for_new_sequence()
        self._masked_traces_recorded = True
        self._log_trace_region("the masked-bucket prefill traces")

    def release_prefill_traces_masked(self):
        """Release the masked-bucket prefill traces, their inputs and outputs, and the GDN layers' trace buffers; later
        masked-bucket prefills run eagerly."""
        if not getattr(self, "_masked_traces", None):
            return
        ttnn.synchronize_device(self.device)
        self._masked_trace_refs = []
        for tr in self._masked_traces.values():
            for tid in tr.trace_ids.values():
                ttnn.release_trace(self.device, tid)
            tr.free()
        for dn in self._gdn_layers():
            dn.free_prefill_trace_buffers()
        self._masked_traces = {}
        self._masked_traces_recorded = False
        self._masked_trace_gdn = None

    def _masked_trace_body(self, tr, checkpoint):
        """The captured forward and logits of tr; returns (hidden, logits)."""
        for dn in self._gdn_layers():
            dn.load_prefill_init()
        hidden = self._forward_prefill_chunk_masked_tp(None, None, 0, None, tr.bucket, inputs=tr, checkpoint=checkpoint)
        return hidden, self._masked_bucket_logits_tp(hidden, None, tr.bucket, sel=tr.sel)

    def _masked_prefill_trace(self, bucket, checkpoint_len):
        """The _MaskedPrefillTrace that serves a masked-bucket prefill of ``bucket`` with or without a checkpoint, or
        None (eager): no trace captured for it, a debug layer capture, or a GDN workspace other than the one the
        traces read (the batched prefill scratch)."""
        if not getattr(self, "_masked_traces_recorded", False):
            return None
        tr = self._masked_traces.get(bucket)
        if tr is None or bool(checkpoint_len) not in tr.trace_ids:
            return None
        if getattr(self, "_capture_layer", None) is not None:
            return None
        if any(a is not b for a, b in zip(self._gdn_workspace(), self._masked_trace_gdn, strict=True)):
            return None
        return tr

    def _replay_masked_trace(self, tr, token_buf, page_table, actual_len, chunk_start, checkpoint_len):
        """Stage tr's inputs for ``actual_len`` real tokens of token_buf from ``chunk_start`` (a new sequence at 0,
        otherwise continuing from the GDN workspace) and replay it. The caller has staged the RoPE table and the
        vision-splice buffers. Returns (hidden: the trace's output, logits: a copy the caller frees)."""
        from models.demos.blackhole.qwen36.tt.gdn.tp import PrefillCheckpoint, PrefillMasks

        bucket, kconv = tr.bucket, self.args.gdn_conv_kernel_size
        block_size = get_block_size(self._paged_kv_caches)
        if chunk_start % block_size:
            raise ValueError(f"a masked-bucket prefill from {chunk_start} starts inside a {block_size}-token KV block")
        ck_host = None
        if checkpoint_len:
            if checkpoint_len >= actual_len:
                raise ValueError(
                    f"a prefill checkpoint after {checkpoint_len} tokens needs a prefill of more than {actual_len}"
                )
            ck_host = PrefillCheckpoint.host_tensors(bucket, checkpoint_len, kconv)
        # The eager forward fills the real blocks [blk0, blkN) only; the trace fills tr.blocks blocks, the rest in
        # the null block 0.
        width = int(tr.full_pt.shape[-1])
        blk0 = chunk_start // block_size
        blkN = num_blocks_in_seq(chunk_start + actual_len, block_size)
        row = page_table[:1].to(torch.int32)
        if row.shape[1] < blkN:
            raise ValueError(f"a page table of {row.shape[1]} blocks does not hold the prefill's {blkN} blocks")
        pad = max(width, blk0 + tr.blocks) - row.shape[1]
        if pad > 0:
            row = torch.cat([row, torch.zeros(1, pad, dtype=torch.int32)], dim=1)
        chunk_row = row[:, blk0 : blk0 + tr.blocks].clone()
        chunk_row[:, blkN - blk0 :] = 0

        if self._masked_trace_refs:
            # The previous replay's input copies may still read their host tensors.
            ttnn.synchronize_device(self.device)
            self._masked_trace_refs = []
        for dn in self._gdn_layers():
            dn.stage_prefill_init(carry=chunk_start > 0)

        rep = ttnn.ReplicateTensorToMesh(self.device)
        refs = self._masked_trace_refs

        def stage(t, dtype, layout, buf):
            host = ttnn.from_torch(t, dtype=dtype, layout=layout, device=None, mesh_mapper=rep)
            ttnn.copy_host_to_device_tensor(host, buf)
            refs.append(host)

        rm, tile = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
        stage(token_buf.to(torch.int32), ttnn.uint32, rm, tr.tok)
        cos_t, sin_t = self._rope_tp_cos_sin_torch(chunk_start, bucket)
        stage(cos_t, ttnn.bfloat16, tile, tr.cos)
        stage(sin_t, ttnn.bfloat16, tile, tr.sin)
        stage(torch.tensor([chunk_start], dtype=torch.int32), ttnn.int32, rm, tr.csi)
        stage(row[:, :width].contiguous(), ttnn.int32, rm, tr.full_pt)
        stage(chunk_row, ttnn.int32, rm, tr.chunk_pt)
        for t, buf, dtype in zip(
            PrefillMasks.host_tensors(bucket, actual_len, kconv),
            (tr.masks.conv_select, tr.masks.gate_mask, tr.masks.qkv_mask),
            PrefillMasks.dtypes(),
        ):
            stage(t, dtype, tile, buf)
        sel = torch.zeros(1, 1, 1, bucket, dtype=torch.float32)
        sel[0, 0, 0, actual_len - 1] = 1.0
        stage(sel, tr.sel_dtype, tile, tr.sel)
        if ck_host is not None:
            stage(ck_host[0], ttnn.float32, tile, tr.checkpoint.gate_mask)
            stage(ck_host[1], ttnn.bfloat16, tile, tr.checkpoint.conv_select)

        variant = ck_host is not None
        if not getattr(self, "_masked_trace_logged", False):
            self._masked_trace_logged = True
            logger.info(f"Masked-bucket prefill trace replay active (first: bucket {bucket}, checkpoint {variant})")
        ttnn.execute_trace(self.device, tr.trace_ids[variant], cq_id=0, blocking=False)
        for dn in self._gdn_layers():
            dn.note_traced_prefill(variant)
        hidden, logits = tr.outputs[variant]
        return hidden, ttnn.clone(logits)

    def _release_prefill_hidden(self, hidden):
        """Free a hidden from prefill_masked_bucket(return_hidden=True), unless it is a captured trace's output."""
        for tr in self._masked_traces.values() if getattr(self, "_masked_traces", None) else ():
            if any(hidden is out[0] for out in tr.outputs.values()):
                return
        ttnn.deallocate(hidden)

    @property
    def prefill_chunk_size(self):
        """Tokens per chunk of prefill_traced_chunked: the captured chunk trace's size, 2048 before a capture."""
        return self._chunked_chunk_size or 2048

    def prefill_traced_chunked(
        self,
        token_ids,
        page_table,
        actual_len,
        vision_tokens=None,
        on_chunk=None,
        start_pos=0,
        rope_token_ids=None,
        checkpoint_pos=0,
    ):
        """Prefill by replaying the captured per-chunk trace for each FULL 2048-token chunk, then
        processing the final partial chunk as one masked-bucket forward (prefill_masked_bucket: a
        captured trace per bucket when record_prefill_traces_masked ran, else eager). Only the real prompt
        (token_ids[:, :actual_len]) is processed. The tail (< chunk_size) masks the GDN state to its real
        tokens instead of repeating bucket padding through the recurrence (which
        corrupts decode state at long context). Next-token logit is at actual_len-1; returns host
        [1,1,vocab]. vision_tokens splice trace-safely: captured forward runs a FIXED-shape
        ttnn.where(mask, vision, text) over persistent buffers staged per chunk via copy_host_to_device
        (no request-time compiles). Each chunk/tail splices its packed-vision slice (vis_row_offset =
        image tokens before it); no image tokens -> identity mask. Works on single device (3D) and TP (4D hidden-sharded; see _set_vision_merge / _alloc_vision_merge_buffers).

        ``on_chunk(hidden, chunk_start, valid_len)`` (TP traced only; same contract as prefill_for_spec):
        called per full chunk right after its replay is enqueued, then for the tail / short prompt.
        hidden is the pre-final-norm [1,1,2048,dim/tp]; on_chunk must NOT free it (it is the trace's
        persistent output), and must enqueue its reads before returning (the next replay overwrites it).

        ``start_pos > 0`` (TP) continues a prefill: it computes tokens [start_pos, actual_len) from the GDN state that
        the batch-1 workspace (rec_state, conv_carry) already holds, attending over the KV cache that the page table
        holds for [0, start_pos). token_ids still starts at position 0. start_pos must be a multiple of the KV block
        size: paged_fill_cache writes a chunk from the first row of its first block. A multimodal continuation splices
        the vision rows of its own image tokens, which the vision rows of the image tokens before start_pos precede.

        ``rope_token_ids`` (default token_ids[:, :actual_len]) are the tokens the request's RoPE table is built from:
        the M-RoPE positions of an image need all of the image's tokens, so a multimodal prefill that runs in segments
        passes the whole prompt for each segment, including one that ends inside an image.

        ``checkpoint_pos > 0`` (TP): every GDN layer also computes its state after checkpoint_pos tokens
        (TPGatedDeltaNet.prefill_checkpoint) in the masked-bucket forward of the prefill's last chunk, which must
        contain it. When that last chunk is a full one, it runs as a masked-bucket forward instead of a chunk-trace
        replay.
        """
        # Default to the standard 2048-token chunk when no trace is captured (e.g. the TP MVP,
        # which serves <=2048 prompts entirely via the masked bucket below and so needs no chunk
        # trace). The chunk trace is only required once there is at least one full chunk to replay.
        chunk_size = self.prefill_chunk_size
        vision_tokens = self.vision_rows_host(vision_tokens)  # before the first replay (see vision_rows_host)
        B, T = token_ids.shape
        assert 0 <= start_pos < actual_len <= T, f"need 0 <= start_pos {start_pos} < actual_len {actual_len} <= {T}"
        block_size = get_block_size(self._paged_kv_caches)
        if start_pos:
            assert self.num_devices > 1, "a prefill continues at start_pos > 0 only on the TP path"
            assert start_pos % block_size == 0, f"start_pos {start_pos} is not on a {block_size}-token KV block"
        blocks_per_chunk = chunk_size // block_size
        num_full = (actual_len - start_pos) // chunk_size
        tail_real = actual_len - start_pos - num_full * chunk_size
        checkpoint_len = None
        if checkpoint_pos:
            if self.num_devices < 2:
                raise ValueError("a prefill checkpoint inside the forward needs a TP mesh")
            if tail_real == 0:
                num_full, tail_real = num_full - 1, chunk_size
            tail_start = actual_len - tail_real
            if not tail_start < checkpoint_pos < actual_len:
                raise ValueError(
                    f"prefill checkpoint after {checkpoint_pos} tokens is not inside the last chunk "
                    f"[{tail_start}, {actual_len}) of a prefill from {start_pos}"
                )
            checkpoint_len = checkpoint_pos - tail_start
        assert (
            num_full == 0 or self.num_devices > 1 or self._chunked_trace_id is not None
        ), "Call capture_prefill_trace_chunked first"

        # Stage the per-request RoPE once for the whole prompt (M-RoPE for multimodal, 1D for text).
        # The chunk-replay loops + the masked tail then slice this sequence-indexed table by chunk
        # position, and decode offsets by the stored rope_delta.
        self._build_request_rope(token_ids[:, :actual_len] if rope_token_ids is None else rope_token_ids, vision_tokens)

        # Short prompt (no full chunks): route the whole prompt through the SAME masked
        # fixed-bucket path the long-prompt tail uses. chunk_start=0 makes prefill_masked_bucket
        # start from zero GDN state and run one masked forward; there is no chunk to replay,
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
            short = dict(
                actual_len=tail_real,
                chunk_start=start_pos,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, start_pos),
                stage_rope=False,
                checkpoint_len=checkpoint_len,
            )
            if on_chunk is not None:
                logits, hidden = self.prefill_masked_bucket(
                    token_ids[:, start_pos:actual_len], page_table, return_hidden=True, **short
                )
                on_chunk(hidden, start_pos, tail_real)
                self._release_prefill_hidden(hidden)
                return logits
            return self.prefill_masked_bucket(token_ids[:, start_pos:actual_len], page_table, **short)

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
                    start_pos=start_pos,
                    checkpoint_len=checkpoint_len,
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
                start_pos=start_pos,
                checkpoint_len=checkpoint_len,
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
        self,
        token_ids,
        page_table,
        actual_len,
        num_full,
        chunk_size,
        tail_real,
        flex_sdpa=True,
        vision_tokens=None,
        start_pos=0,
        checkpoint_len=None,
    ):
        """TP eager long-prompt prefill via warmed bucket=chunk_size programs, from start_pos (see
        prefill_traced_chunked). Returns logits [1,1,vocab] at actual_len-1."""
        # Re-zero GDN at sequence start; tail (chunk_start>0) keeps carried state.
        if start_pos == 0:
            self._reset_gdn_state_for_new_sequence()
        last_hidden = None
        for c in range(num_full):
            cs = start_pos + c * chunk_size
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
            cs = start_pos + num_full * chunk_size
            return self.prefill_masked_bucket(
                token_ids[:, cs:actual_len],
                page_table,
                actual_len=tail_real,
                chunk_start=cs,
                flex_sdpa=flex_sdpa,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, cs),
                checkpoint_len=checkpoint_len,
            )
        # Exact multiple of chunk_size: logit from last full chunk.
        logits = self._masked_bucket_logits_tp(last_hidden, chunk_size, chunk_size)
        ttnn.deallocate(last_hidden)
        return logits

    def _prefill_traced_chunked_tp(
        self,
        token_ids,
        page_table,
        actual_len,
        num_full,
        chunk_size,
        tail_real,
        vision_tokens=None,
        on_chunk=None,
        start_pos=0,
        checkpoint_len=None,
    ):
        """TP traced chunk-outer prefill from start_pos (see prefill_traced_chunked): replay the captured per-chunk trace
        (_forward_prefill_chunk_tp) for each FULL chunk, then run the partial tail through the
        masked bucket. The TP analog of the single-device loop in prefill_traced_chunked: each
        chunk's inputs are DMA'd into the REPLICATED persistent buffers via
        copy_host_to_device_tensor (no per-chunk program dispatch / device allocation — only one
        execute_trace per chunk), so GDN recurrent/conv + paged-KV state carry in place across
        replays and host pressure stays bounded at 128K. The tail's chunk_start>0 skips the GDN
        reset so the carried state continues. Returns logits [1, 1, vocab] at actual_len-1."""
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_chunk = chunk_size // block_size
        rep = ttnn.ReplicateTensorToMesh(self.device)

        # Re-zero GDN once; carries across replays + tail (chunk_start>0 skips reset).
        if start_pos == 0:
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
            cs = start_pos + c * chunk_size
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

        # Tail via masked bucket, or _masked_bucket_logits_tp if no tail (TP 4D hidden).
        if tail_real > 0:
            cs = start_pos + num_full * chunk_size
            if on_chunk is not None:
                logits, hidden = self.prefill_masked_bucket(
                    token_ids[:, cs:actual_len],
                    page_table,
                    actual_len=tail_real,
                    chunk_start=cs,
                    vision_tokens=vision_tokens,
                    vis_row_offset=self._vis_row_offset_for(token_ids, cs),
                    return_hidden=True,
                    checkpoint_len=checkpoint_len,
                )
                on_chunk(hidden, cs, tail_real)
                self._release_prefill_hidden(hidden)
                return logits
            return self.prefill_masked_bucket(
                token_ids[:, cs:actual_len],
                page_table,
                actual_len=tail_real,
                chunk_start=cs,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, cs),
                checkpoint_len=checkpoint_len,
            )
        return self._masked_bucket_logits_tp(self._chunked_trace_output, chunk_size, chunk_size)

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
        # QWEN_SDPA_BF8: bf8 paged KV for SDPA; halves KV memory (gated — validate PCC at long ctx).
        if os.environ.get("QWEN_SDPA_BF8", "0") == "1":
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
        self.release_prefill_traces_masked()
        if getattr(self, "_chunked_trace_id", None) is not None:
            ttnn.release_trace(self.device, self._chunked_trace_id)
            self._chunked_trace_id = None
        self._chunked_prepared_key = None
        if self.gdn_blocks is not None:
            self.gdn_blocks.release()
            self.gdn_blocks = None
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
                    lg = ttnn.linear(x_last, self.lm_head_weight)
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
        rope_pos_vec = pos_vec + self.rope.rope_delta
        if self.num_devices > 1:
            # TP: rope_tp cos/sin [1,B,1,rope_dim] packed on host.
            rd = self.args.rope_head_dim
            inv_freq = 1.0 / (self.args.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
            freqs = torch.outer(rope_pos_vec.float(), inv_freq)  # [B, rd/2], per-user rotation
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().reshape(1, B, 1, rd).to(torch.bfloat16)
            sin = emb.sin().reshape(1, B, 1, rd).to(torch.bfloat16)
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

        cos, sin = unpack_rope(rot_mat_idxs)
        if on_device_logits:
            assert self.sampling is not None, "on_device_logits=True but self.sampling is None"
            logits = self._forward_decode(tokens, cos, sin, current_pos, page_table, sharded_lm_head=True)
            # Sampler runs >=32-wide; pad B up to it (else shape mismatch). Extra slots unused.
            sampler_batch = self.sampling.tt_sampling.max_batch_size
            B = logits.shape[2]
            if B < sampler_batch:
                logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, sampler_batch - B), (0, 0)], value=0.0)
            # Bare tensor (not a tuple): the traced path passes this straight to capture_trace().
            return logits
        logits = self._forward_decode(tokens, cos, sin, current_pos, page_table)
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
