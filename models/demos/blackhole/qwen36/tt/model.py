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

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.gdn.gated_deltanet import (
    fla_fidelity_name,
    fla_scan_fid_len_max,
    fla_scan_fidelities_up_to,
    fla_scan_fidelity_for_len,
)
from models.demos.blackhole.qwen36.tt.layer import (
    Qwen36DecoderLayer,
    decode_norm_sharded,
    decode_norm_sharded_applies,
    decode_norm_sharded_out,
    make_decode_norm_sharded_config,
)
from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import flexible_sdpa_q_chunk
from models.tt_transformers.tt.common import Mode, get_block_size, num_blocks_in_seq


def _signpost(header):
    """Tracy signpost, gated on QWEN36_SIGNPOSTS=1 (unset/0 -> no-op, no tracy import at all).
    Meaningful only from an UNTRACED op sequence (see sp_prefill.py's `_sp` for the same note);
    decode_tp's per-layer python loop always re-executes eagerly, so no extra guard is needed
    here the way sp_prefill.py's traced/untraced `_run_layer_major` needs one."""
    if os.environ.get("QWEN36_SIGNPOSTS") != "1":
        return
    try:
        from tracy import signpost
    except ImportError:
        return
    signpost(header)


class Qwen36Model:
    """Qwen3.5-9B text LM on Blackhole P150. HF_MODEL env var selects checkpoint."""

    def __init__(self, mesh_device, args, state_dict, tensor_cache_path=None):
        self.args = args
        self.device = mesh_device
        self.mesh_device = mesh_device  # Generator reads model.mesh_device
        self.num_devices = mesh_device.get_num_devices()
        # Single-device (1x1, non-SP) path only: sequence_parallel at tp=1 also has num_devices == 1 but runs
        # the TP-path modules (decode_tp / prefill_tp / sp_prefill.py), so the single-device-only features
        # below (D3 decode norm, I-3 / split LM head, M3 zero biases) are gated on this, not num_devices == 1.
        self._single_device = self.num_devices == 1 and not getattr(args, "sequence_parallel", False)
        # CCL for multi-device all-reduce; None on single device (ops no-op).
        if self.num_devices > 1:
            from models.tt_transformers.tt.ccl import TT_CCL

            self.tt_ccl = TT_CCL(mesh_device)
        else:
            self.tt_ccl = None
        self.configuration = args  # Generator reads model.configuration.max_seq_len
        self.sampling_dp = 1
        # RoPE is host-recomputed each step, so refresh all decode trace inputs.
        self._tt_vllm_always_refresh_decode_trace_inputs = True
        # On-device sampling: allowlist 1x4/1x8 TP only — vocab/TP must fit Top-K's 64K shard limit (TP=2 does not).
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
        # M3 ZB (tp_common M3 table): the M1 S2 / S3 zero biases, allocated once here (weight load,
        # before any trace capture) and shared by all layers. Empty when QWEN36_M3_ZB=0.
        self._m3_zero_biases = self._m3_alloc_zero_biases(mesh_device)

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
        if self.num_devices > 1:
            # TP: DistributedNorm all-gathers fractured hidden for LM head.
            from models.tt_transformers.tt.distributed_norm import DistributedNorm

            self.norm = DistributedNorm(self.norm, args, tt_ccl=self.tt_ccl, TG=args.is_galaxy)
        # I-1 D3 (QWEN36_I1_D3, single device): the decode final norm runs width-sharded on 8 cores
        # (see _final_norm_decode); None = pre-I-1 path.
        self._decode_norm_cfg = (
            make_decode_norm_sharded_config(args.dim) if self._single_device and tpc.i1_enabled("D3") else None
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
        # I-3 LM head "A3" (QWEN36_I3_LMHEAD=A3, M4; single device, any grid tpc.i3_a3_lm_plan accepts --
        # 13x10 P150 and 11x10 p300c die): the 10 DRAM width-sharded column chunks are built ONCE here from
        # the weight just loaded, and the unsplit weight is freed (see _a3_build_lm_chunks): every
        # single-device LM-head site then runs on the chunks. None = every other LMHEAD value, or a device
        # the plan rejects (warning logged; lm_head_weight stays and the pre-A3 LM head runs).
        self._a3_lm_chunks = None
        if self._single_device and tpc.i3_value("LMHEAD") == "A3":
            plan, why = tpc.i3_a3_lm_plan(
                mesh_device, int(self.lm_head_weight.shape[0]), int(self.lm_head_weight.shape[1])
            )
            if plan is None:
                logger.warning(f"[M4] QWEN36_I3_LMHEAD=A3 disabled on this device ({why}); pre-A3 LM head kept")
            else:
                self._a3_lm_chunks = self._a3_build_lm_chunks(mesh_device, plan)

        self.vocab_size = args.vocab_size
        # True: return pre-gather vocab-sharded logits for per-shard argmax + host combine.
        self._ondev_argmax = False
        # True (1x1 greedy only; see set_greedy_token_output): prefill + non-sampling decode return a
        # uint32 greedy token computed on device instead of the logits. Default off.
        self._greedy_token_out = False
        self._paged_kv_caches = None
        # Positions in self.layers of full-attn layers (not checkpoint indices); drives KV cache bind.
        self._attention_layer_indices = [pos for pos, layer in enumerate(self.layers) if layer.is_full_attention]
        self._deltanet_external_states = None  # (recurrent, conv) tuples; set by allocate_kv_caches
        # Shared zero buffers for in-place DN reset between traced replays.
        self._dn_zero_recurrent = None
        self._dn_zero_conv = None
        self._dn_zero_hist = None  # QWEN36_GDN_DECODE_FUSED=2 only
        # Chunk-outer trace: one all-layer chunk captured, replayed per chunk via DMA inputs.
        # Persistent buffers below; addresses baked into trace.
        self._chunked_trace_id = None
        self._chunked_trace_output = None
        self._chunked_chunk_size = None
        self._chunked_trace_prepared = False  # the chunk trace was captured in the prepared order (M3/M5 follow)
        # QWEN36_FLA_SCAN_FID_BY_LEN (tt/gdn/gated_deltanet.py; single device): the fused FLA scan fidelity follows
        # each request's prompt length (scan HiFi2 up to 65536 tokens, HiFi3 above; prep HiFi4).
        # None = flag off or never prepared.
        self._fla_warm_fids = None  # the scan fidelities whose prefill programs prepare compiled
        self._chunked_trace_fla_fid = None  # the scan fidelity baked into the parked chunk trace
        self._fla_hold = False  # True during prepare: the warm-ups pin the fidelity (per-request choice off)
        # (prompt_len, scan fidelity name, chunk trace re-captured, "<prep>/<scan>") of the last request
        self.fla_last_request = None
        # M2 (tp_common M2_FLAG_DEFAULTS), fixed per prepare in _prepare_prefill_trace_chunked_setup so the
        # warm-up forward, the captured trace and prefill_traced_chunked agree. False = current path.
        self._m2_nowhere = False  # NOWHERE: the traced chunk has no vision-splice where (text-only model)
        self._m2_lastrow = False  # LASTROW: the traced chunk returns only row chunk_size - 1 ([1, 1, dim])
        self._m2_repack_pending = False  # REPACK_LATE: a conv-history repack still has to run
        # M3 REPACK_TRACE (tp_common M3 table): the captured GDN conv-history repack trace and the
        # (fused_conv_state, conv_hist) tensor pairs it bakes in. None = eager repack (current path).
        self._m3_repack_trace_id = None
        self._m3_repack_pairs = None
        # P18_PRELUDE (QWEN36_PRELUDE_TRACE): the captured per-request GDN state reset + chunk-0 RoPE trace.
        self._pt_trace_id = None
        self._pt_refs = None
        self._pt_host_cache = {}  # chunk start -> (chunk_start_idx host tensor, last-pos host tensor)
        # M4 (tp_common M4_FLAG_DEFAULTS), fixed per prepare with M2 LASTROW. False / None = current path.
        self._m4_r4a = False  # R4A: tile-aligned [T-32:T] block slices for the last layer's one-row reads
        self._m4_r4b = False  # R4B: the last layer's SDPA = decode SDPA for row chunk_size - 1
        self._chunk_last_pos_tensor = None  # R4B: persistent int32 [1] = cs + chunk_size - 1 (trace input)
        self._m4_last_pos_host = None  # R4B: the value the host last wrote into _chunk_last_pos_tensor
        # M5 (tp_common M5_FLAG_DEFAULTS), fixed per prepare. False / None = current path.
        self._m5_tail = False  # TAIL_TRACE: the exact-multiple tail runs as its own trace (prepared order only)
        self._m5_tail_out = None  # TAIL_TRACE: persistent DRAM output (uint32 token or logits), made in prepare
        self._m5_tail_trace_id = None  # TAIL_TRACE: the captured tail trace
        self._m5_tail_refs = None  # TAIL_TRACE: the tensors the tail trace bakes in (checked at every replay)
        self._m5_addnorm = False  # ADDNORM: fused residual add + RMSNorm in the T == 2048 traced chunk forward
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
        # Persistent B=1 GDN prefill scratch (batched serving): allocated once at warmup, its buffer
        # addresses are baked into the chunk-prefill trace and reused by every prefill_paged_slots
        # replay, so it is never freed/reallocated (only zeroed in place). See _bind_gdn_prefill_scratch.
        self._gdn_prefill_scratch = None

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

    def init_vision_model(self, reference_visual=None, vision_args=None, dtype=ttnn.bfloat8_b, debug=False):
        """Build and attach the TT vision tower (DropInVisionTransformer).

        The vision tower runs on the SAME mesh as the text model. It still needs the HF
        reference visual for the patch embed / positional-interpolation steps that are not
        ported to TT; if ``reference_visual`` is not supplied it is loaded here via
        ``VisionModelArgs.reference_vision_model``. Idempotent — returns the existing tower
        if already built.

        Args:
            reference_visual: HF ``model.model.visual`` to wrap. Loaded internally if None.
            vision_args (VisionModelArgs): vision config on this mesh. Built internally if None.
            dtype (ttnn.dtype): compute dtype for the vision weights.
            debug (bool): run the reference vision path alongside and log PCC.

        Returns:
            DropInVisionTransformer: the attached vision tower.
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

        Mirrors the HF reference's ``get_image_features`` seam: pixel patches in, packed
        image embeddings out — one row per image-placeholder token, ready to be spliced
        into the text embeddings by ``_scatter_vision_tokens``.

        Args:
            pixel_values (torch.Tensor): patchified pixels ``[num_patches, patch_dim]``.
            image_grid_thw (torch.Tensor): per-image grid ``(t, h, w)``, ``[num_images, 3]``.

        Returns:
            ttnn.Tensor: ``[num_image_tokens, H]`` image embeddings, hidden-fractured along
            the last dim on a mesh (same sharding as the text embeddings).
        """
        assert self.vision_model is not None, "init_vision_model() must be called before get_image_features()"
        # Stash the grid (as an IMAGE grid) so the prefill paths can build M-RoPE position ids for
        # this request (the splice positions in input_ids + the (t,h,w) grid are all M-RoPE needs).
        # Clear any stale video grid so the modality (and thus the placeholder token id) is image.
        self._req_image_grid_thw = image_grid_thw
        self._req_video_grid_thw = None
        image_features = self.vision_model.forward(pixel_values, grid_thw=image_grid_thw)
        # The vision tower returns [1, B, S, H]; flatten the leading (batch/seq) dims to the
        # packed [num_image_tokens, H] rows the text-model splice (_scatter_vision_tokens /
        # _set_vision_merge) expects. The hidden dim is unchanged so the mesh hidden-fracture
        # is preserved. B == 1 for now.
        hidden = image_features.shape[-1]
        return ttnn.reshape(image_features, (-1, hidden))

    def get_video_features(self, pixel_values_videos, video_grid_thw):
        """Run the vision tower over a single user's video frames.

        Mirrors the HF reference's ``get_video_features`` seam, which is just ``get_image_features``
        on the video pixels/grid — the vision tower forward is identical for image and video. The
        only differences are downstream: M-RoPE treats the grid as a VIDEO grid (split per frame by
        timestamps, modality==2), and the embeddings splice into ``video_token_id`` placeholders
        rather than ``image_token_id``. Both are selected by stashing the grid here as a video grid.

        Args:
            pixel_values_videos (torch.Tensor): patchified video pixels ``[num_patches, patch_dim]``.
            video_grid_thw (torch.Tensor): per-video grid ``(t, h, w)``, ``[num_videos, 3]``.

        Returns:
            ttnn.Tensor: ``[num_video_tokens, H]`` video embeddings, hidden-fractured along the last
            dim on a mesh (same sharding as the text embeddings).
        """
        assert self.vision_model is not None, "init_vision_model() must be called before get_video_features()"
        # Stash the grid as a VIDEO grid; clear any stale image grid so the modality (and the
        # placeholder token id) is video.
        self._req_video_grid_thw = video_grid_thw
        self._req_image_grid_thw = None
        video_features = self.vision_model.forward(pixel_values_videos, grid_thw=video_grid_thw)
        hidden = video_features.shape[-1]
        return ttnn.reshape(video_features, (-1, hidden))

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

        ``_vis_buf`` holds the image embeddings placed at their token positions (zeros elsewhere);
        ``_vis_mask_buf`` is the 0/1 image mask. Both are zero-initialised, so the ttnn.where baked
        into the captured forward is the identity until a real multimodal request stages them.
        Allocating before warmup means the where compiles in the warmup pass (and is then
        captured), never at request time.

        Shapes/sharding match the activations of the forward that consumes them:
          - single device: vis [1, chunk_size, dim], mask [1, chunk_size, 1] (the 3D embd output);
          - TP: vis [1, 1, chunk_size, dim] HIDDEN-SHARDED across the mesh exactly like embd
            fractures its output (ShardTensor2dMesh dims=(None, -1)), so each device's where sees
            its own [.., dim/TP] vision columns; mask [1, 1, chunk_size, 1] REPLICATED (it
            broadcasts over the sharded hidden dim). DropInVisionTransformer fractures its output
            the same way, so the per-device columns line up.
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

        This is NOT a scatter and is not a substitute for one. The scatter — placing the n
        (variable) vision rows at their n image positions — already happened on host in
        _set_vision_merge, which is why ``_vis_buf`` here is a FULL [1, chunk_size, dim] buffer
        (image rows at their positions, zeros elsewhere), the same length as ``x``, NOT the raw
        [n, dim] vision tensor. So n << seq_len (a few image tokens in a long text prompt) is fine:
        the [.,1] mask broadcasts over the hidden dim and is 1 only at those n positions, so
        out == vision there and out == x (text) everywhere else (mask==0 is the exact identity).

        The placement is forced onto the host because aligning a variable n to fixed positions is
        inherently variable-shape; any on-device form (ttnn.scatter / pad / slice-copy) recompiles
        per request and would clobber the parked trace. ``length`` selects the segment (chunk_size
        for a full chunk, bucket for the masked path); the buffers slice down to it. No-op when the
        buffers are unallocated (text-only / non-traced deployment, which uses the device scatter)."""
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
        """Stage the persistent vision buffers for the next forward (host -> device copy only;
        no program compiles). ``vision_tokens`` None clears the mask (the where becomes identity,
        for text-only requests); otherwise the packed image rows are read back to host, placed at
        their token positions in a zero [1, chunk_size, dim] buffer, and uploaded along with the
        0/1 mask. ``ids_host`` is the segment's token ids (torch), used to locate the image
        placeholders (== hf_config.image_token_id).

        ``vis_row_offset`` is the number of image-placeholder tokens that appear in the prompt
        BEFORE this segment, i.e. the index of the first packed vision row belonging to it. A
        large image whose placeholders span multiple prefill chunks (or spill into the tail) is
        thus spliced correctly: each segment consumes its own slice
        ``vis_host[vis_row_offset : vis_row_offset + n]`` of the packed rows. A segment with no
        image placeholders (text-only chunk / tail) clears the mask (identity merge)."""
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
        """Generator mode-change hook (no prefetcher). Generator.decode_forward calls it first, before
        any decode trace replay or eager decode: M2 REPACK_LATE runs a pending conv-history repack here."""
        self._m2_flush_pending_repack()
        return None

    def _m2_flush_pending_repack(self):
        """M2 REPACK_LATE: enqueue the conv-history repack that prefill_traced_chunked deferred (no-op when
        none is pending). Called by every model entry that uses GDN state after a prefill (switch_mode,
        prepare_decode_inputs_host, decode, GDN reset / save / restore), so the repack runs on CQ 0 before
        any of them; _forward_decode asserts that nothing is pending."""
        if self._m2_repack_pending:
            if os.environ.get("QWEN36_REPACK_AFTER_TTFT") == "1" and self._m3_repack_trace_id is not None:
                # REPACK_AFTER_TTFT: replay the captured repack trace here (first decode step) instead of
                # in the TTFT window; same ops, same buffers as the eager repack (clears _m2_repack_pending).
                self._m3_replay_repack_trace()
            else:
                self._gdn_refresh_conv_hist()  # clears _m2_repack_pending

    def _m3_alloc_zero_biases(self, mesh_device):
        """M3 ZB (QWEN36_M3_ZB=1, single device; tp_common M3 table): allocate the zero bias of each enabled
        M1 item (S2: MLP down projection [1, 2048]; S3: GDN q|k|v in-proj [1, 6144]) ONCE, here at model
        load (before any trace capture), and share it across all layers (Qwen36MLP._m3_zero_bias,
        Qwen36GatedDeltaNet._m3_qkv_zero_bias). The forwards only read it. Returns {item: tensor}; empty
        when the flag is off, on TP, or when neither M1 item is on."""
        out = {}
        if not self._single_device or not tpc.m3_enabled("ZB"):
            return out
        if tpc.m1_enabled("S2"):
            mlps = [l.feed_forward for l in self.layers if isinstance(l.feed_forward, Qwen36MLP)]
            if mlps:
                out["S2"] = tpc.m3_zero_bias("S2", mesh_device)
                for mlp in mlps:
                    mlp._m3_zero_bias = out["S2"]
        if tpc.m1_enabled("S3"):
            gdns = [l.attention for l in self.layers if not l.is_full_attention]
            if gdns:
                out["S3"] = tpc.m3_zero_bias("S3", mesh_device)
                for gdn in gdns:
                    gdn._m3_qkv_zero_bias = out["S3"]
        logger.info(
            f"[M3] ZB: zero biases allocated at load (shared by all layers): "
            f"{ {k: list(v.shape) for k, v in out.items()} }"
        )
        return out

    def _m3_repack_live(self):
        """M3 REPACK_TRACE: [(dn, fused_conv_state, conv_hist)] of every fused-decode GDN layer, when
        _gdn_refresh_conv_hist would run the plain repack on all of them with no allocation (single device,
        every layer bound to a B = 1 conv state, conv_hist already allocated); else None."""
        if self.num_devices > 1:
            return None
        dns = [
            l.attention for l in self.layers if not l.is_full_attention and getattr(l.attention, "_decode_fused", False)
        ]
        if not dns:
            return None
        for dn in dns:
            if dn.fused_conv_state is None or dn.fused_conv_state.shape[0] != 1 or dn.conv_hist is None:
                return None
        return [(dn, dn.fused_conv_state, dn.conv_hist) for dn in dns]

    def _m3_capture_repack_trace(self, device):
        """M3 REPACK_TRACE: capture the GDN conv-history repack (the eager ops of _gdn_refresh_conv_hist) into
        its own trace. capture_prefill_trace_chunked(prepared=True) calls it right after the chunk-trace
        capture (the decode trace is already primed). Every buffer the repack reads or writes exists
        already: the persistent GDN conv state (external buffers bound in prepare) and conv_hist (allocated
        by the prepare warm-ups); intermediates are freed at capture end. Its programs compiled in prepare
        (masked-bucket warm-up + the M3 warm-up there). An eager warm-up runs first; the warm-up and the
        capture must add no program-cache entry (a compile after a trace is parked puts a kernel binary in
        parked-trace scratch). No-op (eager path kept) when there is no plain repack to capture."""
        live = self._m3_repack_live()
        if live is None:
            logger.warning(
                "[M3] REPACK_TRACE: no plain conv-history repack to capture (TP, fused GDN decode off, or a "
                "GDN layer without conv state / conv_hist); prefill_traced_chunked keeps the eager repack"
            )
            return
        n0 = device.num_program_cache_entries()
        self._gdn_refresh_conv_hist()  # eager warm-up (all programs already compiled in prepare)
        ttnn.synchronize_device(device)
        n1 = device.num_program_cache_entries()
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self._gdn_refresh_conv_hist()
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        n2 = device.num_program_cache_entries()
        same = all(dn.fused_conv_state is fcs and dn.conv_hist is hist for dn, fcs, hist in live)
        if n1 != n0 or n2 != n0 or not same:
            ttnn.release_trace(device, trace_id)
            raise RuntimeError(
                f"M3 REPACK_TRACE: program cache entries {n0} -> {n1} (warm-up) -> {n2} (capture), buffers "
                f"unchanged={same}: the repack compiled or rebound state after the chunk trace was parked"
            )
        self._m3_repack_trace_id = trace_id
        self._m3_repack_pairs = live
        from models.demos.blackhole.qwen36.tt.gdn import decode_fused as _df

        logger.info(
            f"[M3] REPACK_TRACE: captured the conv-history repack of {len(live)} GDN layers "
            f"(QWEN36_GDN_CONV_REPACK={_df.repack_variant()}); program cache entries {n0} -> {n1} (warm-up) "
            f"-> {n2} (capture): 0 new compiles"
        )

    def _m3_replay_repack_trace(self):
        """M3 REPACK_TRACE: enqueue the captured conv-history repack (CQ 0, non-blocking), in place of the
        eager _gdn_refresh_conv_hist. The trace bakes the buffer addresses, so every GDN layer must still
        point at the captured conv-state / conv_hist tensors."""
        for dn, fcs, hist in self._m3_repack_pairs:
            assert dn.fused_conv_state is fcs and dn.conv_hist is hist, (
                "M3 REPACK_TRACE: a GDN layer's conv state / conv_hist was rebound after the repack trace "
                "capture; prepare + capture again"
            )
        self._m2_repack_pending = False
        ttnn.execute_trace(self.device, self._m3_repack_trace_id, cq_id=0, blocking=False)

    def _m3_release_repack_trace(self, device=None):
        """M3 REPACK_TRACE: release the repack trace (re-prepare, free_kv_caches). No-op when none."""
        if self._m3_repack_trace_id is not None:
            ttnn.release_trace(device if device is not None else self.device, self._m3_repack_trace_id)
            self._m3_repack_trace_id = None
            self._m3_repack_pairs = None

    # ---- QWEN36_PRELUDE_TRACE (P18_PRELUDE) ---------------------------------------------------------------------
    # The host work before the chunk-0 replay (54 eager state-reset copies, 4 RoPE ops) becomes one trace:
    # per GDN layer copy(zero recurrent -> recurrent_state) and copy(zero conv -> fused_conv_state), then the chunk-0
    # cos/sin slice of the persistent RoPE table copied into the baked _chunk_cos_buf/_chunk_sin_buf. conv_hist is
    # NOT reset in the trace: the repack after the last chunk (gather variant: one embedding per layer over the whole
    # [1, 1, 2048, 32] output, padded rows written as zero) and the masked-bucket tail rebuild it fully before any
    # decode read; the prefill chunk trace never reads it.

    def _pt_live(self):
        """[(dn, recurrent_state, fused_conv_state)] of every GDN layer, or None when the trace does not apply."""
        if self.num_devices > 1 or self._dn_zero_recurrent is None:
            return None
        dns = [l.attention for l in self.layers if not l.is_full_attention]
        if not dns or any(dn.recurrent_state is None or dn.fused_conv_state is None for dn in dns):
            return None
        if not self.rope.rope_device_table_enabled() or self._chunk_cos_buf is None:
            return None
        return [(dn, dn.recurrent_state, dn.fused_conv_state) for dn in dns]

    def _pt_refs_now(self):
        live = self._pt_live()
        if live is None:
            return None
        return (
            self._dn_zero_recurrent,
            self._dn_zero_conv,
            self._chunk_cos_buf,
            self._chunk_sin_buf,
            self.rope.cos_device,
            self.rope.sin_device,
            tuple(x for _, r, c in live for x in (r, c)),
        )

    def _pt_body(self, chunk_size):
        for dn, rec, conv in self._pt_live():
            ttnn.copy(self._dn_zero_recurrent, rec)
            ttnn.copy(self._dn_zero_conv, conv)
        cos_slice, sin_slice = self.rope.get_prefill_rot_mats_table_slice(0, chunk_size)
        ttnn.copy(cos_slice, self._chunk_cos_buf)
        ttnn.copy(sin_slice, self._chunk_sin_buf)
        ttnn.deallocate(cos_slice)
        ttnn.deallocate(sin_slice)

    def _pt_capture_trace(self, device):
        """Capture the prelude trace (after the chunk / repack / tail traces). Eager warm-up first; the warm-up and
        the capture must add no program-cache entry (a compile after a trace is parked is unsafe). The eager
        warm-up is the reset + chunk-0 RoPE the next request would run anyway (state is zero at prepare end)."""
        live = self._pt_live()
        if live is None or self.rope._req_cos is not None:
            logger.warning(
                "[P18] PRELUDE_TRACE: not applicable (TP, no GDN state, or RoPE table off); eager prelude kept"
            )
            return
        chunk_size = self._chunked_chunk_size
        n0 = device.num_program_cache_entries()
        self._pt_body(chunk_size)
        ttnn.synchronize_device(device)
        n1 = device.num_program_cache_entries()
        refs = self._pt_refs_now()
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self._pt_body(chunk_size)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        n2 = device.num_program_cache_entries()
        refs2 = self._pt_refs_now()
        same = all(
            (a is b) if not isinstance(a, tuple) else all(x is y for x, y in zip(a, b)) for a, b in zip(refs, refs2)
        )
        if n2 != n1 or not same:
            ttnn.release_trace(device, trace_id)
            raise RuntimeError(
                f"P18 PRELUDE_TRACE: program cache entries {n0} -> {n1} (warm-up) -> {n2} (capture), buffers "
                f"unchanged={same}: capture compiled or rebound state after the chunk trace was parked"
            )
        self._pt_trace_id = trace_id
        self._pt_refs = refs
        self._pt_chunk_size = chunk_size
        logger.info(
            f"[P18] PRELUDE_TRACE: captured reset of {len(live)} GDN layers (recurrent + conv, no conv_hist) + "
            f"chunk-0 RoPE; program cache entries {n0} -> {n1} (warm-up) -> {n2} (capture)"
        )

    def _pt_replay_ok(self, chunk_size):
        """True when the prelude trace can serve this request (same baked buffers, text-only RoPE)."""
        if self._pt_trace_id is None or self.rope._req_cos is not None or chunk_size != self._pt_chunk_size:
            return False
        now = self._pt_refs_now()
        if now is None:
            return False
        return all(
            (a is b) if not isinstance(a, tuple) else (len(a) == len(b) and all(x is y for x, y in zip(a, b)))
            for a, b in zip(self._pt_refs, now)
        )

    def _pt_replay(self):
        """Enqueue the prelude trace (CQ 0, non-blocking) plus the host-only part of the reset."""
        self._m2_flush_pending_repack()  # same op order as _reset_gdn_state_for_new_sequence
        ttnn.execute_trace(self.device, self._pt_trace_id, cq_id=0, blocking=False)
        for l in self.layers:
            if not l.is_full_attention and l.attention.split_conv_state is not None:
                for buf in l.attention.split_conv_state:
                    ttnn.deallocate(buf)
                l.attention.split_conv_state = None

    def _pt_release_trace(self, device=None):
        if self._pt_trace_id is not None:
            ttnn.release_trace(device if device is not None else self.device, self._pt_trace_id)
            self._pt_trace_id = None
            self._pt_refs = None

    def _m5_tail_trace_refs(self):
        """M5 TAIL_TRACE: the tensors whose buffer addresses the tail trace bakes in: the chunk-trace output it
        reads, its persistent output, the final-norm weight and the LM-head weights of every mode (only the active
        mode's are read; the others are None)."""
        return (
            self._chunked_trace_output,
            self._m5_tail_out,
            self.norm.weight,
            self.lm_head_weight,
            getattr(self, "_lm_head_split_chunks", None),
            getattr(self, "_i3_lm_chunks_cache", None),
            self._a3_lm_chunks,
        )

    def _m5_tail_body(self):
        """M5 TAIL_TRACE: the exact-multiple tail (_exact_multiple_tail_device, the code of the eager tail) on the
        chunk-trace output, then one ttnn.copy of its result into the persistent output. Every intermediate is
        freed before this returns (so none outlives a capture)."""
        out = self._exact_multiple_tail_device(self._chunked_trace_output, self._chunked_chunk_size)
        ttnn.copy(out, self._m5_tail_out)
        ttnn.deallocate(out)

    def _m5_capture_tail_trace(self, device):
        """M5 TAIL_TRACE: capture the exact-multiple tail into its own trace. capture_prefill_trace_chunked(
        prepared=True) calls it right after the chunk-trace capture (and the M3 repack capture). Every buffer the
        tail reads or writes exists already: the chunk-trace output, the norm / LM-head weights and the persistent
        output (allocated in prepare, before the decode trace was primed); intermediates are freed at capture
        end. Its programs (and the copy into the persistent output) compiled in prepare
        (_warm_exact_multiple_tail). An eager warm-up runs first; the warm-up and the capture must add no
        program-cache entry (a compile after a trace is parked puts a kernel binary in parked-trace scratch).
        No-op (eager tail kept) when prepare made no persistent output (max_prompt_len < chunk_size)."""
        if self._m5_tail_out is None:
            logger.warning(
                "[M5] TAIL_TRACE: prepare made no persistent tail output (max_prompt_len < chunk_size, so the "
                "exact-multiple tail was not warmed); prefill_traced_chunked keeps the eager tail"
            )
            return
        n0 = device.num_program_cache_entries()
        self._m5_tail_body()  # eager warm-up (all programs already compiled in prepare)
        ttnn.synchronize_device(device)
        n1 = device.num_program_cache_entries()
        refs = self._m5_tail_trace_refs()
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self._m5_tail_body()
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        n2 = device.num_program_cache_entries()
        same = all(a is b for a, b in zip(refs, self._m5_tail_trace_refs()))
        if n1 != n0 or n2 != n0 or not same:
            ttnn.release_trace(device, trace_id)
            raise RuntimeError(
                f"M5 TAIL_TRACE: program cache entries {n0} -> {n1} (warm-up) -> {n2} (capture), baked tensors "
                f"unchanged={same}: the tail compiled or rebound a tensor after the chunk trace was parked"
            )
        self._m5_tail_trace_id = trace_id
        self._m5_tail_refs = refs
        _mode = (
            tpc.i3_value("LMHEAD") if tpc.i3_enabled("LMHEAD") else f"split{os.environ.get('QWEN36_LMHEAD_SPLIT', '8')}"
        )
        logger.info(
            f"[M5] TAIL_TRACE: captured the exact-multiple tail (LASTROW={int(self._m2_lastrow)}, LM head {_mode}, "
            f"greedy token={int(self._greedy_token_out)}; output {list(self._m5_tail_out.shape)} "
            f"{self._m5_tail_out.dtype} DRAM); program cache entries {n0} -> {n1} (warm-up) -> {n2} (capture): "
            f"0 new compiles"
        )

    def _m5_replay_tail_trace(self):
        """M5 TAIL_TRACE: enqueue the captured tail (CQ 0, non-blocking), in place of the eager tail. The trace
        bakes the buffer addresses, so every baked tensor must still be the captured one."""
        now = self._m5_tail_trace_refs()
        assert all(a is b for a, b in zip(self._m5_tail_refs, now)), (
            "M5 TAIL_TRACE: the chunk-trace output, the persistent tail output or a norm / LM-head weight was "
            "rebound after the tail trace capture; prepare + capture again"
        )
        ttnn.execute_trace(self.device, self._m5_tail_trace_id, cq_id=0, blocking=False)

    def _m5_release_tail_trace(self, device=None, keep_output=False):
        """M5 TAIL_TRACE: release the tail trace and free the persistent output (re-prepare, free_kv_caches).
        keep_output=True (FLA BY_LEN re-capture) keeps the persistent output, which was allocated in prepare
        before the decode trace was primed. No-op when none."""
        if self._m5_tail_trace_id is not None:
            ttnn.release_trace(device if device is not None else self.device, self._m5_tail_trace_id)
            self._m5_tail_trace_id = None
            self._m5_tail_refs = None
        if self._m5_tail_out is not None and not keep_output:
            ttnn.deallocate(self._m5_tail_out)
            self._m5_tail_out = None

    def set_greedy_token_output(self, enabled):
        """Greedy on-device argmax (single device only). When enabled, prefill_masked_bucket,
        prefill_traced_chunked and the non-sampling decode (ttnn_decode_forward) return a uint32
        token tensor (the first max index, same as torch.argmax on the bf16 logits) instead of the
        logits; process_output_decode passes it through as token ids. Default off, so vLLM and all
        other callers are unchanged. Call before prepare_prefill_trace_chunked (or the one-call
        capture_prefill_trace_chunked) and before the first decode_forward / prime_decode_trace: the
        token ops must compile before any trace is parked (#48536). Needs the split LM head
        (QWEN36_LMHEAD_SPLIT > 1, the default); see _lm_head."""
        enabled = bool(enabled)
        if enabled == self._greedy_token_out:
            return
        assert self.num_devices == 1, "greedy token output is single-device (1x1) only"
        assert self._chunked_trace_id is None, "call set_greedy_token_output before capture_prefill_trace_chunked"
        if enabled:
            self._check_greedy_token_lm_head_split()
        self._greedy_token_out = enabled

    @staticmethod
    def _check_greedy_token_lm_head_split():
        """want_token is implemented (and op-tested, plan_0925/T1/op_test.py) on the split LM head
        only. The unsplit LM head (QWEN36_LMHEAD_SPLIT <= 1) + want_token raises instead.
        Every I-3 LM head (QWEN36_I3_LMHEAD = A / B / C, see _lm_head_i3) implements want_token."""
        if tpc.i3_enabled("LMHEAD"):
            return
        split_n = int(os.environ.get("QWEN36_LMHEAD_SPLIT", "8") or "0")
        if split_n <= 1:
            raise NotImplementedError(
                f"greedy on-device token output (set_greedy_token_output / QWEN36_ONDEV_ARGMAX=1) needs the "
                f"split LM head (QWEN36_LMHEAD_SPLIT > 1), got QWEN36_LMHEAD_SPLIT={split_n}; the unsplit "
                "want_token path is not implemented. Set QWEN36_ONDEV_ARGMAX=0 or QWEN36_LMHEAD_SPLIT=8."
            )

    def _lm_head_token(self, x):
        """LM head -> greedy token on device: uint32 [..., 1] (single device, one logical row)."""
        assert self.num_devices == 1, "greedy token output is single-device (1x1) only"
        assert x.shape[-2] == 1, f"greedy token output needs one logical row, got shape {x.shape}"
        return self._lm_head(x, want_token=True)

    def _lm_head(self, x, want_token=False):
        """LM-head matmul. Vocab-sharded mesh: partial logits + all-gather to full replicated.
        Single device: minimal_matmul's default config measured 2.46ms vs ttnn-auto's 3.28ms;
        every swept 1D/DRAM-sharded progcfg overflows L1 at this N (248320), so no explicit
        program_config is passed here (default-config minimal_matmul only).

        QWEN36_LMHEAD_SPLIT (default "8" as of step1-F10c phase 2, 2026-09-21; int N>=0, "0"
        disables): single-device only. Splits the [2048, 248320] lm_head weight into N
        tile-aligned column chunks (248320 = 7760 tiles; N must divide 7760, so N in
        {2,4,5,8,10,...}), sliced once on device with ttnn.slice and cached in
        self._lm_head_split_chunks, then runs N minimal_matmul calls (default config, same as
        the unsplit path below) and ttnn.concat(dim=-1) -- mirroring tt_transformers/tt/lm_head.py's
        column split. N=8 was the best of {4,8} swept in phase 2 (val_b T=2048/4096: -2.0/-2.3ms
        vs N=4's -0.7/-0.8ms; both PCC=1.000000 exact vs logits_after_f10b.pt, math-equivalent
        split); traced_4k demo TTFT 0.164s, no regression (see
        patches/step1_f10c_kvbf8_qknorm_lmhead.patch). Set to "0" to restore the single unsplit
        minimal_matmul below.

        want_token=True (single device, via _lm_head_token): return the greedy token (uint32 [..., 1])
        instead of the logits. Split path: untilize each split to its one logical row, concat the
        rows, one ttnn.argmax. This skips the TILE concat (T1b op test: 149 us vs 171 us for TILE
        concat + untilize + argmax). The unsplit path (QWEN36_LMHEAD_SPLIT <= 1) + want_token raises
        NotImplementedError (untested; see _check_greedy_token_lm_head_split).

        QWEN36_I3_LMHEAD (I-3, single device; tp_common I-3 table): A / A2 / B / C replace the path
        above for a one-tile-row input on the 13x10 grid (see _lm_head_i3); A3 for a one-tile-row input
        on any grid its chunks were built on at load (tpc.i3_a3_lm_plan; 13x10 and 11x10);
        QWEN36_LMHEAD_SPLIT is then ignored. "0" keeps the path above.
        """
        _split_n = int(os.environ.get("QWEN36_LMHEAD_SPLIT", "8") or "0")
        if want_token:
            assert self.num_devices == 1, "want_token (greedy token output) is single-device (1x1) only"
            self._check_greedy_token_lm_head_split()
        _i3 = tpc.i3_value("LMHEAD") if self._single_device else "0"
        # A3 runs wherever its chunks were built at load (any grid i3_a3_lm_plan accepted); on 13x10 it is
        # taken even without them, as before (_lm_head_a3 then asserts the flag was set at load).
        if (
            _i3 != "0"
            and tpc.i3_one_tile_row(x)
            and ((_i3 == "A3" and self._a3_lm_chunks is not None) or tpc.i3_grid_ok(self.mesh_device))
        ):
            return self._lm_head_i3(x, _i3, want_token)
        if want_token and _i3 != "0":
            raise NotImplementedError(
                f"QWEN36_I3_LMHEAD={_i3} + want_token needs a one-tile-row input on the 13x10 grid "
                f"(A3: on a grid its chunks were built on at load) (got shape {list(x.shape)})"
            )
        if self.lm_head_weight is None:
            # I-3 A3 freed the unsplit weight at load (_a3_build_lm_chunks): only its one-tile-row path exists.
            raise NotImplementedError(
                f"QWEN36_I3_LMHEAD=A3 (set at model load) serves one-tile-row inputs only "
                f"(the unsplit LM-head weight was freed); got shape {list(x.shape)} with QWEN36_I3_LMHEAD={_i3}"
            )
        if self._single_device and _split_n > 1:
            cached = getattr(self, "_lm_head_split_chunks", None)
            if cached is None or cached[0] != _split_n:
                rows, vocab = self.lm_head_weight.shape[0], self.lm_head_weight.shape[1]
                tile = 32
                total_tiles = vocab // tile
                assert vocab % tile == 0 and total_tiles % _split_n == 0, (
                    f"QWEN36_LMHEAD_SPLIT={_split_n} must divide the {total_tiles}-tile vocab "
                    f"dim ({vocab}); e.g. 2, 4, 5, 8, or 10 for vocab={vocab}."
                )
                chunk_cols = (total_tiles // _split_n) * tile
                chunks = []
                for i in range(_split_n):
                    start = i * chunk_cols
                    end = start + chunk_cols
                    chunks.append(ttnn.slice(self.lm_head_weight, (0, start), (rows, end)))
                cached = (_split_n, chunks)
                self._lm_head_split_chunks = cached
            _, chunks = cached
            partials = [ttnn.experimental.minimal_matmul(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG) for w in chunks]
            if want_token:
                rows = [ttnn.to_layout(p, ttnn.ROW_MAJOR_LAYOUT) for p in partials]
                for p in partials:
                    ttnn.deallocate(p)
                row = ttnn.concat(rows, dim=-1)
                for r in rows:
                    ttnn.deallocate(r)
                token = ttnn.argmax(row, dim=-1, keepdim=True)
                ttnn.deallocate(row)
                return token
            logits = ttnn.concat(partials, dim=-1)
        elif self._single_device and os.environ.get("QWEN36_LMHEAD_MINIMAL", "1") != "0":
            logits = ttnn.experimental.minimal_matmul(x, self.lm_head_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            logits = ttnn.linear(x, self.lm_head_weight)
        if self._lmhead_vocab_sharded:
            from models.tt_transformers.tt.ccl import tt_all_gather

            logits = tt_all_gather(
                logits,
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=None,
                dim=len(logits.shape) - 1,
                topology=self.args.ccl_topology(),
            )
        return logits

    # I-3 LM head configs (tp_common I-3 table; T4 sweep D8-R006 / D8-O062 / D8-M016 family).
    # mode -> (column splits, DRAM-sharded chunk: shard width tiles per bank (None = interleaved chunk),
    #          DRAM-sharded per_core_N). 7760 vocab tiles: split 4 = 1940 tiles/chunk, split 8 = 970.
    # A2 uses 8 splits: at 4 splits its fp32 CBs (~260 KB/core) clashed with live L1 buffers on the
    # eager prefill_paged path (T=512/1024); 8 splits halve them.
    _I3_LM_CFG = {"A": (4, 244, 31), "A2": (8, 122, 16), "C": (4, None, None)}
    _I3_LM_A_IN0_GRID = (8, 8)  # K = 64 tiles -> 1 tile per core

    def _i3_lm_chunks(self, mode):
        """Column chunks of lm_head_weight for I-3 LM head A / A2 (DRAM width-sharded) / C (interleaved),
        sliced once on device and cached. Built on the first _lm_head call, which is eager (model
        warmup / prefill-trace prepare), before any trace is captured."""
        split, shard_w, _ = self._I3_LM_CFG[mode]
        key = (split, shard_w)
        cached = getattr(self, "_i3_lm_chunks_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        if cached is not None:
            for c in cached[1]:
                ttnn.deallocate(c)
        rows, vocab = self.lm_head_weight.shape[0], self.lm_head_weight.shape[1]
        total_tiles = vocab // 32
        assert vocab % 32 == 0 and total_tiles % split == 0, f"vocab {vocab} not split-{split}"
        cols = (total_tiles // split) * 32
        chunks = []
        for i in range(split):
            c = ttnn.slice(self.lm_head_weight, (0, i * cols), (rows, (i + 1) * cols))
            if shard_w is not None:
                assert cols // 32 <= 8 * shard_w
                cw = ttnn.to_memory_config(c, tpc.i3_dram_width_memcfg(rows, shard_w))
                ttnn.deallocate(c)
                c = cw
            chunks.append(c)
        self._i3_lm_chunks_cache = (key, chunks)
        return chunks

    @staticmethod
    def _dram_allocated_per_bank(device):
        """DRAM bytes allocated per bank (allocator view), or None if the view is unavailable."""
        try:
            return int(ttnn.get_memory_view(device, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank)
        except Exception:  # noqa: BLE001
            return None

    def _a3_build_lm_chunks(self, device, plan):
        """I-3 LM head A3 (M4): build the plan's column chunks of lm_head_weight ([2048, 248320] bfp8; plan =
        tpc.i3_a3_lm_plan for this device), each DRAM WIDTH_SHARDED over the device's DRAM banks (8 banks:
        10 chunks of 776 tiles, shard [2048, 98 * 32]; 8 x 98 = 784 >= 776 tiles, the last bank holds 90
        valid tiles), with ttnn.slice + to_memory_config on device (the same bfp8 tiles as the loaded
        weight), then free the unsplit weight: the model keeps one copy (+1% shard padding).
        Called once from __init__ (eager, before any trace). Logs the build time and the DRAM use."""
        import time

        rows, vocab = int(self.lm_head_weight.shape[0]), int(self.lm_head_weight.shape[1])
        split, nt, cols, shard_w, banks = (plan[k] for k in ("split", "nt", "cols", "shard_w", "banks"))
        assert split * cols == vocab and nt <= banks * shard_w, (vocab, plan)
        mc = tpc.i3_dram_width_memcfg(rows, shard_w, num_banks=banks)
        dram0 = self._dram_allocated_per_bank(device)
        t0 = time.perf_counter()
        chunks = []
        for i in range(split):
            s = ttnn.slice(self.lm_head_weight, (0, i * cols), (rows, (i + 1) * cols))
            chunks.append(ttnn.to_memory_config(s, mc))
            ttnn.deallocate(s)
        ttnn.synchronize_device(device)
        dt = time.perf_counter() - t0
        dram1 = self._dram_allocated_per_bank(device)
        ttnn.deallocate(self.lm_head_weight)
        self.lm_head_weight = None
        dram2 = self._dram_allocated_per_bank(device)
        mb = lambda b: None if b is None else round(b * banks / 2**20, 1)  # noqa: E731
        gx, gy = plan["grid"]
        logger.info(
            f"[M4] QWEN36_I3_LMHEAD=A3 ({gx}x{gy} grid, {banks} DRAM banks): built {split} DRAM width-sharded "
            f"LM-head chunks [{rows}, {cols}] ({nt} tiles, {shard_w} tiles/bank) from the loaded weight in "
            f"{dt:.3f} s; unsplit weight freed. "
            f"DRAM allocated (all banks, MiB): {mb(dram0)} before -> {mb(dram1)} with chunks -> {mb(dram2)} after free"
        )
        return chunks

    def _lm_head_a3(self, x, want_token):
        """I-3 LM head A3 (M4; single device, one tile row, the grid its chunks were planned for at load --
        tpc.i3_a3_lm_plan: 13x10 and 11x10 give the same plan). x: the final-norm output, either
        interleaved (prefill, the non-traced decode paths) or the D3 8-core width-sharded decode norm
        output (_forward_decode). One to_memory_config puts it in the 8x8 in0 layout (I2S or reshard).
        Per chunk: DRAM-sharded linear (tp_common.i3_a3_lm_progcfg, HiFi2 / fp32 dest / packer L1 acc) into
        L1 WIDTH_SHARDED, then at once (so its L1 shard frees early) either the untilize to one RM row in
        L1 interleaved (want_token) or the S2I to DRAM (logits). Then RM concat + argmax (uint32 [..., 1],
        first max index) or TILE concat (logits, DRAM, [..., vocab])."""
        chunks = self._a3_lm_chunks
        assert (
            chunks is not None
        ), "QWEN36_I3_LMHEAD=A3 must be set when the model is built (the A3 chunks are made at load)"
        gx, gy = tpc.I3_A3_LM_CFG["in0_grid"]
        in0_mc = tpc.i3_l1_width_memcfg(int(x.shape[-1]), gx, gy)
        same = x.memory_config() == in0_mc
        xs = x if same else ttnn.to_memory_config(x, in0_mc)
        pc = tpc.i3_a3_lm_progcfg()
        ck = self._i3_lm_ckc_hifi2()
        outs = []
        for c in chunks:
            p = ttnn.linear(
                xs, c, compute_kernel_config=ck, program_config=pc, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
            )
            if want_token:
                outs.append(ttnn.to_layout(p, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG))
            else:
                outs.append(ttnn.sharded_to_interleaved(p, ttnn.DRAM_MEMORY_CONFIG))
            ttnn.deallocate(p)
        if not same:
            ttnn.deallocate(xs)
        out = ttnn.concat(outs, dim=-1)
        for o in outs:
            ttnn.deallocate(o)
        if not want_token:
            return out
        token = ttnn.argmax(out, dim=-1, keepdim=True)
        ttnn.deallocate(out)
        return token

    def _lm_head_i3(self, x, mode, want_token):
        """I-3 LM head (QWEN36_I3_LMHEAD; single device, one tile row, 13x10 grid; A3: the grid its chunks
        were planned for at load). Returns the logits
        (DRAM, TILE, [..., vocab]) or, with want_token, the greedy uint32 token [..., 1] (first max index).
        A: 4 x DRAM-sharded linear (in0 resharded to L1 width 8x8, LoFi, fp32 dest off) -> S2I.
        A2: A with the current path's compute config (HiFi2, fp32 dest), 8 splits (per_core_N 16).
        B: one 1D-mcast linear on 13x10 (bw 2, per_core_N 65, subblock 1x5, LoFi, fp32 dest off).
        C: 4 x minimal_matmul on 13x2 (M/K/N block 1/16/16, subblock 1x4, op-default compute config).
        Token: split outputs -> untilize each (one row) -> RM concat -> argmax (T1 B2b with 4 / 8 splits);
        B: untilize the unsplit logits to one row -> argmax.
        A3 (M4): see _lm_head_a3 (chunks built at load)."""
        if mode == "A3":
            return self._lm_head_a3(x, want_token)
        if self.lm_head_weight is None:
            raise NotImplementedError(
                f"QWEN36_I3_LMHEAD={mode}: the unsplit LM-head weight was freed at load by QWEN36_I3_LMHEAD=A3"
            )
        dram = ttnn.DRAM_MEMORY_CONFIG
        if mode == "B":
            # in0_block_w 2: bw 8 needs 1.41 MB of CBs, which clashed with live L1 buffers on the eager
            # prefill_paged path (T=512/1024); bw 2 is also faster in the op probe (1447 vs 1503 us).
            pc = tpc._i3_1d_progcfg(13, 10, 2, 65, 5)
            logits = ttnn.linear(
                x,
                self.lm_head_weight,
                compute_kernel_config=self._i3_lm_ckc_lofi(),
                program_config=pc,
                memory_config=dram,
            )
            if not want_token:
                return logits
            row = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(logits)
            token = ttnn.argmax(row, dim=-1, keepdim=True)
            ttnn.deallocate(row)
            return token
        chunks = self._i3_lm_chunks(mode)
        if mode in ("A", "A2"):
            xs = ttnn.to_memory_config(x, tpc.i3_l1_width_memcfg(int(x.shape[-1]), *self._I3_LM_A_IN0_GRID))
            pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=1,
                per_core_M=1,
                per_core_N=self._I3_LM_CFG[mode][2],
                fused_activation=None,
                num_workers_per_dram_bank=2,
            )
            parts = [
                ttnn.linear(
                    xs,
                    c,
                    compute_kernel_config=self._i3_lm_ckc_lofi() if mode == "A" else self._i3_lm_ckc_hifi2(),
                    program_config=pc,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                )
                for c in chunks
            ]
            ttnn.deallocate(xs)
            out_mc = ttnn.L1_MEMORY_CONFIG if want_token else dram
            inter = [ttnn.sharded_to_interleaved(p, out_mc) for p in parts]
            for p in parts:
                ttnn.deallocate(p)
            parts = inter
        else:  # "C"
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=1,
                K_block_size=16,
                N_block_size=16,
                subblock_h=1,
                subblock_w=4,
                compute_with_storage_grid_size=ttnn.CoreCoord(13, 2),
            )
            parts = [ttnn.experimental.minimal_matmul(x, c, config=cfg, memory_config=dram) for c in chunks]
        if want_token:
            rows = [ttnn.to_layout(p, ttnn.ROW_MAJOR_LAYOUT) for p in parts]
            for p in parts:
                ttnn.deallocate(p)
            row = ttnn.concat(rows, dim=-1)
            for r in rows:
                ttnn.deallocate(r)
            token = ttnn.argmax(row, dim=-1, keepdim=True)
            ttnn.deallocate(row)
            return token
        logits = ttnn.concat(parts, dim=-1)
        for p in parts:
            ttnn.deallocate(p)
        return logits

    @staticmethod
    def _i3_lm_ckc_hifi2():
        """I-3 LM head A2 compute config = the minimal_matmul op default of the current path (HiFi2,
        fp32 dest, packer L1 acc, approx off)."""
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    @staticmethod
    def _i3_lm_ckc_lofi():
        """I-3 LM head A / B compute config as swept (T4 D8-R006 / D8-O062): LoFi, fp32 dest off."""
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True
        )

    def _final_norm_decode(self, x, lm_in0_sharded=False):
        """Final RMSNorm before the LM head (TP decode).

        The bare `self.norm(x, DECODE)` runs plain ttnn.rms_norm on a DRAM-interleaved [32,dim]
        tensor -> single tile-row -> 1 core (~80us/token). Passing the framework's 'lm_head' norm
        config runs the sharded multi-core norm across lm_head_core_grid instead; output_mem_config
        is forced back to DRAM so the LM-head matmul input is byte-identical (layout-only change).

        lm_in0_sharded (M4, I-3 LM head A3; single device with D3): return the 8-core width-sharded norm
        output itself (no S2I to DRAM); the A3 LM head reshards it to its in0 layout in one op.
        """
        if self.num_devices > 1:
            nc = dict(self.args.get_norm_config("lm_head", Mode.DECODE))
            nc["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
            return self.norm(x, mode=Mode.DECODE, norm_config=nc)
        if self._decode_norm_cfg is not None and decode_norm_sharded_applies(x):
            if lm_in0_sharded:
                return decode_norm_sharded_out(self.norm, x, self._decode_norm_cfg)
            # I-1 D3: 8-core width-sharded norm; S2I back to DRAM (the LM-head input stays DRAM-interleaved).
            return decode_norm_sharded(self.norm, x, self._decode_norm_cfg, ttnn.DRAM_MEMORY_CONFIG)
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
        sequence_parallel: bool = False,
    ):
        # HF_MODEL env var (hub or local path) is canonical; hf_model sets it for back-compat.
        if hf_model is not None:
            import os

            os.environ["HF_MODEL"] = hf_model

        args = Qwen36ModelArgs(
            mesh_device=device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            sequence_parallel=sequence_parallel,
        )

        # layer_indices: run only these checkpoint layers (e.g. [0,3,31]) for profiling.
        # Each keeps its real type via full attention_type_list. Overrides n_layers truncation.
        if layer_indices is not None:
            layer_indices = list(layer_indices)
            assert layer_indices, "layer_indices must be non-empty"
            assert all(
                0 <= i < len(args.attention_type_list) for i in layer_indices
            ), f"layer_indices {layer_indices} out of range [0, {len(args.attention_type_list)})"
            args.layer_indices = layer_indices
            args.n_layers = len(layer_indices)
        elif n_layers is not None:
            args.n_layers = n_layers
            args.attention_type_list = args.attention_type_list[:n_layers]

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
        for i, layer in enumerate(self.layers):
            ckpt_idx = self.layer_indices[i] if hasattr(self, "layer_indices") else i
            kind = "attn" if layer.is_full_attention else "gdn"
            _signpost(f"decode L{ckpt_idx} {kind}")
            x = layer.forward(x, cos=cos, sin=sin, mode="decode", position_tensor=cur_pos_tt)
        _signpost("decode head")
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

        On-device equivalent of the HF reference's
        ``inputs_embeds.masked_scatter(image_mask, image_embeds)``. The embedding is
        flattened to ``[rows, H]``; the packed ``vision_tokens`` are placed into a zero
        buffer at the image-placeholder rows with a dim-0 ``ttnn.scatter``, then merged
        with the text embeddings via
        ``ttnn.where(special_image_mask, vision, text)``. The embeddings/vision never
        leave the device. No-op when ``vision_tokens`` is None or the prompt has no
        image tokens (the text-only path).

        The image-placeholder mask and placement positions are computed on host from the
        token ids (which already live on host at every call site), mirroring the HF
        reference's ``special_image_mask = input_ids == self.config.image_token_id``, then
        uploaded. The two tiny derived tensors uploaded are the ``[n, H]`` scatter index
        (hidden-sharded on a mesh, like the embedding activations) and the ``[rows, 1]``
        where-predicate (replicated on a mesh — it broadcasts over the sharded hidden dim).

        Args:
            x (ttnn.Tensor): text embeddings from ``self.embd`` — ``[B, T, H]`` (the
                raw embedding output), hidden fractured along the last dim on a mesh.
            token_ids (torch.Tensor): the prefill token ids (``input_ids``), ``[B, T]`` on
                host; image placeholders are the entries equal to ``hf_config.image_token_id``.
            vision_tokens (ttnn.Tensor): ``[num_image_tokens, H]`` produced by the
                vision tower (fractured along hidden on a mesh, like ``x``), one row
                per image placeholder token.

        Returns:
            ttnn.Tensor: same logical shape / layout / sharding as ``x`` with the
            vision embeddings scattered in.
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
        # QWEN36_FLA_SCAN_FID_BY_LEN: eager path (no trace replayed), the fidelity of this length. No-op when off.
        self._fla_select_for_request(T, replays_chunk_trace=False, strict=False)

        # Stage the per-request RoPE (M-RoPE for multimodal, 1D for text) before any cos/sin seam.
        self._build_request_rope(token_ids, vision_tokens)

        if T > 1024:
            logits = self.prefill_layer_chunked(token_ids, chunk_size=2048, vision_tokens=vision_tokens)
            self._gdn_refresh_conv_hist()
            return logits

        # Short sequences (<=1024)
        self.reset_state(batch_size=B)

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = self.embd(token_ids_ttnn)
        x = self._scatter_vision_tokens(x, token_ids, vision_tokens)

        cos, sin = self.rope.get_prefill_rot_mats(0, T)

        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="prefill")

        x = self.norm(x, mode=Mode.PREFILL)

        x_last = x[:, -1:, :]
        logits = self._lm_head(x_last)
        self._gdn_refresh_conv_hist()

        return logits

    def prefill_layer_chunked(self, token_ids, chunk_size=2048, page_table=None, vision_tokens=None):
        """Prefill long sequences using layer-at-a-time chunked processing.

        DeltaNet uses larger chunk_size (256 vs 64) to limit Neumann-series error
        (4096 tokens -> 16 sub-chunks, PCC >0.98). page_table enables paged prefill."""
        B, T = token_ids.shape
        # QWEN36_FLA_SCAN_FID_BY_LEN: eager path (no trace replayed), the fidelity of this length. No-op when off.
        self._fla_select_for_request(T, replays_chunk_trace=False, strict=False)
        self.reset_state(batch_size=B)

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
        self._m2_flush_pending_repack()  # M2 REPACK_LATE (no-op unless a repack is pending)
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
        if self._ondev_argmax and self._a3_lm_chunks is None:
            # Pre-gather vocab-sharded logits; caller argmaxes shards, skips all-gather + readback.
            # (I-3 A3, single device: the unsplit weight was freed and the "shard" is the full logits
            # row, so _lm_head runs the A3 chunks below instead.)
            logits = ttnn.linear(x, self.lm_head_weight)
        else:
            logits = self._lm_head(x)
        ttnn.deallocate(x)

        return logits

    def _forward_decode(
        self, token_ids_buf, cos, sin, cur_pos_tensor, page_table, sharded_lm_head=False, want_token=False
    ):
        """Trace-safe paged decode. All inputs are device tensors.

        sharded_lm_head=True: return the pre-gather vocab-sharded logits (no all-gather)
        for the on-device sampler, which does its own cross-device top-k + gather.
        want_token=True (single device): return the greedy uint32 token (_lm_head_token), not logits.
        want_token cannot combine with the pre-gather vocab-sharded logits (sharded_lm_head or the
        mesh _ondev_argmax), which would silently return logits.
        """
        assert not (want_token and (sharded_lm_head or self._ondev_argmax)), (
            "want_token (greedy token output) cannot combine with sharded_lm_head / _ondev_argmax "
            f"(sharded_lm_head={sharded_lm_head}, _ondev_argmax={self._ondev_argmax})"
        )
        # M2 REPACK_LATE: the deferred conv-history repack must run before this forward, and not inside
        # a trace capture (it would be baked into the decode trace). The decode entries (switch_mode,
        # prepare_decode_inputs_host, decode) run it first; any other path fails loudly here.
        assert not self._m2_repack_pending, (
            "M2 REPACK_LATE: a conv-history repack is pending; call switch_mode / prepare_decode_inputs_host "
            "(or _m2_flush_pending_repack) before the decode forward"
        )
        x = self.embd(token_ids_buf)
        if self.num_devices > 1:
            # TP expects [1,1,B,dim_frac]; embd yields [B,1,dim_frac].
            x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1], x.shape[-1]))
        for layer in self.layers:
            if layer.is_full_attention:
                x = layer.forward(x, cos, sin, position_tensor=cur_pos_tensor, page_table=page_table, mode="decode")
            else:
                x = layer.forward(x, mode="decode")
        # M4 (I-3 LM head A3): the final norm hands its 8-core width-sharded output straight to the A3 LM head.
        # A3 is single-device only, where the pre-gather "vocab-sharded" logits (sharded_lm_head /
        # _ondev_argmax) are the full logits: those run the A3 chunks too (the unsplit weight was freed).
        _a3_in0 = self._a3_lm_chunks is not None
        x = self._final_norm_decode(x, lm_in0_sharded=_a3_in0)
        if (sharded_lm_head or self._ondev_argmax) and not _a3_in0:
            # Pre-gather vocab-sharded logits (on-device sampling / greedy argmax).
            logits = ttnn.linear(x, self.lm_head_weight)
        elif want_token:
            logits = self._lm_head_token(x)
        else:
            logits = self._lm_head(x)
        ttnn.deallocate(x)
        return logits

    def _forward_prefill_chunk(
        self, token_buf, cos_buf, sin_buf, chunk_start_idx_tensor, full_page_table, chunk_page_table
    ):
        """Trace-safe single-chunk prefill. Updates paged KV + GDN state in place.
        Returns last-layer hidden [1, chunk_size, hidden_size], or [1, 1, hidden_size] (row
        chunk_size - 1 only) with M2 LASTROW (self._m2_lastrow, fixed at prepare)."""
        x = self.embd(token_buf)
        # F10B (item A): keep the post-embedding tensor -- the inter-layer residual stream -- in L1
        # when T <= 2048, under QWEN36_LAYER_RESID_L1 (same flag/threshold as layer.py's residual-add
        # split; default "0"). Decoupled from QWEN36_LAYER_L1_MAX_T (F10A's norm-output flag, also
        # default 0/off). Measured (step1 F10B): re-testing end to end (this call's own warmup, the
        # first thing capture_prefill_trace_chunked does) with the flag on reproduced the SAME
        # "Statically allocated circular buffers ... clash with L1 buffers" GDN-conv1d error F10A
        # found for the norm case -- driven by layer.py's per-layer residual add keeping this same
        # tensor L1-resident continuously across all 24 layers (see layer.py for detail); isolating
        # just this one L1 hop (post-embedding into layer 0, reverting to DRAM right after layer 0's
        # own add) passed standalone but is not adopted as a separate default given the single-flag
        # framing and its small (one hop of 24) benefit. Default "0" restores DRAM unconditionally
        # (byte-identical to pre-F10A/F10B); "1" reproduces the clash for further investigation.
        _resid_l1 = os.environ.get("QWEN36_LAYER_RESID_L1", "0") == "1" and x.shape[1] <= 2048
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG if _resid_l1 else ttnn.DRAM_MEMORY_CONFIG)
        # Trace-safe vision splice (fixed-shape where over persistent buffers; identity when the
        # mask buffer is zero, which is the case for every text-only chunk and request). The caller
        # stages the buffers before replaying chunk 0 of a multimodal prompt; chunks>0 are cleared.
        # M2 NOWHERE (self._m2_nowhere, fixed at prepare for a text-only model): no where at all.
        if not self._m2_nowhere:
            x = self._apply_vision_merge(x, length=x.shape[1])
        last = len(self.layers) - 1
        # M5 ADDNORM (fixed at prepare; T in M5_ADDNORM_T_SET chunks): every layer but the last defers its MLP residual
        # add (returns (h, mlp_out)); the next layer fuses it with its attention_norm (pending) and owns that x.
        _m5 = self._m5_addnorm and self.num_devices == 1 and x.shape[1] in tpc.M5_ADDNORM_T_SET
        pending = None
        for li, layer in enumerate(self.layers):
            _m5_kw = dict(m5_addnorm=True, pending=pending, defer_out=li < last) if _m5 else {}
            x_in = x if pending is None else None
            if layer.is_full_attention:
                _lastrow = self._m2_lastrow and li == last
                # M4 R4A / R4B (fixed at prepare, only with LASTROW): the last layer's one-row variants.
                _m4 = {}
                if _lastrow and self._m4_r4a:
                    _m4["last_row_tile_slices"] = True
                if _lastrow and self._m4_r4b:
                    _m4["last_row_pos_tensor"] = self._chunk_last_pos_tensor
                x_new = layer.forward(
                    x_in,
                    cos=cos_buf,
                    sin=sin_buf,
                    mode="prefill",
                    page_table=full_page_table,
                    chunk_page_table=chunk_page_table,
                    chunk_start_idx_tensor=chunk_start_idx_tensor,
                    # M2 LASTROW: the last layer computes its output for row chunk_size - 1 only.
                    last_row_only=_lastrow,
                    **_m4,
                    **_m5_kw,
                )
            else:
                x_new = layer.forward(
                    x_in, mode="prefill", chunk_size=layer.attention.long_prefill_chunk_size, **_m5_kw
                )
            if pending is None:
                ttnn.deallocate(x)  # (M5 ADDNORM with pending: the layer made and freed its own x)
            if _m5 and li < last:
                pending, x = x_new, None  # M5 ADDNORM: (h, mlp_out) for the next layer
            else:
                pending, x = None, x_new
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
        self, token_buf, cos_buf, sin_buf, chunk_start_idx_tensor, full_page_table, chunk_page_table
    ):
        """TP trace-safe single-chunk prefill (replicated persistent buffers).
        Full chunk (valid_len==chunk_size); flexible SDPA via device chunk_start_idx.
        Returns hidden [1,1,chunk_size,dim]."""
        chunk_size = self._chunked_chunk_size
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
                x_new = layer.forward(x, mode="prefill", chunk_size=self.args.gdn_chunk_size, valid_len=None)
            ttnn.deallocate(x)
            x = x_new
        return x

    def capture_prefill_trace_chunked(
        self,
        device,
        page_table,
        chunk_size=2048,
        warmup_masked_buckets=True,
        capture_chunk_trace=True,
        prepared=False,
    ):
        """Capture one chunk's all-layer prefill as a trace; replayed per chunk.

        Chunk-outer prefill stays under the 4 GiB trace limit at long context.
        Flexible SDPA (runtime chunk_start) makes one trace serve all chunk positions.

        capture_chunk_trace=False warms the masked-bucket programs but skips parking the chunk trace.
        The batched (B>1) vLLM path passes capture_chunk_trace=True with the PERSISTENT B=1 prefill
        scratch bound (_bind_gdn_prefill_scratch), so the trace bakes that scratch's addresses and
        long prompts replay the traced chunk path per user (prefill_paged_slots rebinds the scratch).

        prepared=True (single device only): the caller already ran prepare_prefill_trace_chunked, so
        only the capture runs here. That is the parked-trace-safe order used by bench_e2e_p150.py and
        text_demo.py: prepare (all prefill programs/buffers) -> prime the decode trace (all decode
        programs/buffers) -> capture this trace. prepared=False (default) keeps the original one-call
        behaviour (setup + warmups + capture), e.g. for the vLLM path."""
        if self.num_devices > 1:
            assert not prepared, "prepared=True is single-device only"
            return self._capture_prefill_trace_chunked_tp(
                device,
                page_table,
                chunk_size=chunk_size,
                warmup_masked_buckets=warmup_masked_buckets,
                capture_chunk_trace=capture_chunk_trace,
            )
        if prepared:
            assert self._chunk_token_buf is not None and self._chunked_chunk_size == chunk_size, (
                "capture_prefill_trace_chunked(prepared=True) needs prepare_prefill_trace_chunked "
                f"(chunk_size={chunk_size}) first"
            )
            assert self._chunked_trace_id is None, "a chunk trace is already parked; prepare again first"
        else:
            self._prepare_prefill_trace_chunked_setup(device, page_table, chunk_size, warmup_masked_buckets)

        self._capture_chunk_traces(device, prepared)

    def _capture_chunk_traces(self, device, prepared):
        """The capture part of capture_prefill_trace_chunked (single device): the chunk trace, then (prepared
        order) the M3 repack trace and the M5 tail trace. Also run by _fla_recapture_chunk_traces
        (QWEN36_FLA_SCAN_FID_BY_LEN) when a request needs the other FLA scan fidelity."""
        # Capture trace.
        # M2 NOWHERE was fixed at prepare for a text-only model; a vision tower attached since then would
        # need the where, which the warm-up did not compile.
        assert not (
            self._m2_nowhere and self.vision_model is not None
        ), "M2 NOWHERE: init_vision_model() ran after prepare; prepare again before capture"
        self._reset_dn_state_inplace()
        self._chunked_trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self._chunked_trace_output = self._forward_prefill_chunk(
            self._chunk_token_buf,
            self._chunk_cos_buf,
            self._chunk_sin_buf,
            self._chunk_start_idx_tensor,
            self._chunk_full_page_table_buf,
            self._chunk_page_table_buf,
        )
        ttnn.end_trace_capture(device, self._chunked_trace_id, cq_id=0)
        self._chunked_trace_prepared = bool(prepared)
        self._chunked_trace_fla_fid = self._fla_current_fid()  # QWEN36_FLA_SCAN_FID_BY_LEN (None when off)
        if prepared:
            # ACKNOWLEDGED CORRUPTIBLE (trace-allocation tracker): in the prepared order the decode
            # trace is already parked, so this trace output lands in memory the decode replay may
            # overwrite. Safe: every prefill replay rewrites it, and prefill_traced_chunked consumes it
            # (last-row slice -> norm -> LM head -> host) before any decode replay runs. Nothing else is
            # allocated here that outlives the capture (intermediates are freed at capture end).
            from ttnn.tools.trace_allocation_tracker import acknowledge_corruptible

            acknowledge_corruptible(self._chunked_trace_output)
        logger.info("Chunked prefill trace captured successfully!")
        if prepared and tpc.m3_enabled("REPACK_TRACE"):
            # M3 REPACK_TRACE: the conv-history repack that prefill_traced_chunked runs after the last
            # chunk replay gets its own trace, captured now (prepare -> prime decode -> chunk capture ->
            # this), after an eager warm-up; no buffer is allocated that outlives the capture.
            self._m3_capture_repack_trace(device)
        if prepared and self._m5_tail:
            # M5 TAIL_TRACE: the exact-multiple tail (final norm + LM head [+ argmax]) gets its own trace, captured
            # now, after an eager warm-up; its persistent output was allocated in prepare.
            self._m5_capture_tail_trace(device)
        if prepared and os.environ.get("QWEN36_PRELUDE_TRACE", "0") == "1":
            self._pt_capture_trace(device)

    # ---- QWEN36_FLA_SCAN_FID_BY_LEN (P11_FLALEN; tt/gdn/gated_deltanet.py) -------------------------------------
    # The fused FLA op's fidelity follows each request's prompt length: scan HiFi2 up to 65536 tokens, scan HiFi3
    # above, prep HiFi4 in both; a choice is named by its scan fidelity (gdn/gated_deltanet.py pairs the prep).
    # The fidelities are hashed ChunkGdnFusedProgramConfig fields, so both programs coexist in the program cache.
    #   prepare: compiles the prefill programs (T = chunk_size chunk + masked buckets) of every fidelity a prompt
    #     of up to max_prompt_len tokens can use, before the decode trace is primed; the chunk trace is then
    #     captured with the fidelity of the longest prompt (fla_scan_fidelity_for_len(max_prompt_len)).
    #   per request (prefill_traced_chunked / prefill_masked_bucket at chunk_start 0): the GDN layers take the
    #     request's fidelity (eager masked bucket / tail); when the request replays the chunk trace and the trace
    #     holds the other fidelity, the chunk trace (and the M3 / M5 traces that bake its buffers) is re-captured
    #     first. The re-capture compiles nothing (checked); a fidelity prepare did not compile raises instead.
    # Flag off (code default): every helper below is a no-op and nothing changes.

    def _fla_gdn_layers(self):
        """The GDN layers whose fused FLA config takes a per-request scan fidelity (single device only)."""
        if self.num_devices > 1:
            return []
        return [
            l.attention
            for l in self.layers
            if not l.is_full_attention and hasattr(l.attention, "set_fla_scan_fidelity")
        ]

    def _fla_set_scan_fidelity(self, scan_fid):
        for dn in self._fla_gdn_layers():
            dn.set_fla_scan_fidelity(scan_fid)

    def _fla_current_fid(self):
        dns = self._fla_gdn_layers()
        return dns[0].fla_scan_fid if dns else None

    def _fla_select_for_request(self, prompt_len, replays_chunk_trace, strict=True):
        """Give the GDN layers the FLA scan fidelity of a prompt_len-token request and, when it replays the chunk
        trace (replays_chunk_trace) and the parked trace holds the other fidelity, re-capture the chunk trace.
        strict (the traced entry points): a fidelity prepare did not compile raises (a compile now would put a
        kernel binary in parked-trace memory). No-op when QWEN36_FLA_SCAN_FID_BY_LEN is off, during prepare, and
        on TP."""
        want = fla_scan_fidelity_for_len(prompt_len)
        if want is None or self._fla_hold or not self._fla_gdn_layers():
            return
        if strict and self._fla_warm_fids is not None and want not in self._fla_warm_fids:
            raise RuntimeError(
                f"QWEN36_FLA_SCAN_FID_BY_LEN: a {prompt_len}-token prompt needs FLA scan fidelity {want}, but prepare "
                f"compiled only {self._fla_warm_fids} (its max_prompt_len is shorter than this prompt). Call "
                f"prepare_prefill_trace_chunked with max_prompt_len >= {prompt_len} first."
            )
        self._fla_set_scan_fidelity(want)
        recaptured = False
        if replays_chunk_trace and self._chunked_trace_id is not None and self._chunked_trace_fla_fid != want:
            self._fla_recapture_chunk_traces()
            recaptured = True
        self.fla_last_request = (int(prompt_len), want.name, recaptured, fla_fidelity_name(want))

    def _fla_recapture_chunk_traces(self):
        """Re-capture the parked chunk trace, and the M3 repack / M5 tail traces that bake its buffers, with the GDN
        layers' current FLA scan fidelity. Runs at the start of a request, before its GDN reset, so the order is
        the same as the first capture: every persistent buffer (chunk inputs, GDN state, conv_hist, the M5 output)
        is kept, the decode trace stays parked, and only the chunk-trace output is allocated again (acknowledged
        corruptible, as at the first capture). Every program was compiled in prepare: a compile here raises."""
        device = self.device
        self._m2_flush_pending_repack()  # M2 REPACK_LATE: the previous request's repack runs first (old op order)
        n0 = device.num_program_cache_entries()
        old_fid = self._chunked_trace_fla_fid
        ttnn.release_trace(device, self._chunked_trace_id)
        self._chunked_trace_id = None
        self._m3_release_repack_trace(device)
        self._pt_release_trace(device)
        self._m5_release_tail_trace(device, keep_output=True)
        if self._chunked_trace_output is not None:
            ttnn.deallocate(self._chunked_trace_output)
            self._chunked_trace_output = None
        self._capture_chunk_traces(device, self._chunked_trace_prepared)
        if not self._chunked_trace_prepared:
            # One-call order: the decode trace may be parked by now, as in the prepared order (see there).
            from ttnn.tools.trace_allocation_tracker import acknowledge_corruptible

            acknowledge_corruptible(self._chunked_trace_output)
        n1 = device.num_program_cache_entries()
        if n1 != n0:
            raise RuntimeError(
                f"QWEN36_FLA_SCAN_FID_BY_LEN: re-capturing the chunk trace ({old_fid} -> {self._chunked_trace_fla_fid}) "
                f"compiled {n1 - n0} programs; prepare must compile both fidelities (max_prompt_len above the "
                f"threshold) before the decode trace is primed"
            )
        logger.info(
            f"[FLA] BY_LEN: chunk trace re-captured with FLA prep/scan {fla_fidelity_name(self._chunked_trace_fla_fid)} "
            f"(was {fla_fidelity_name(old_fid)}); "
            f"program cache entries {n0} -> {n1}: 0 new compiles"
        )

    def prepare_prefill_trace_chunked(
        self, device, page_table, chunk_size=2048, warmup_masked_buckets=True, max_prompt_len=None
    ):
        """Single-device phase 1 of the parked-trace-safe order (T7, 2026-09-25): everything
        capture_prefill_trace_chunked does before begin_trace_capture (persistent chunk buffers, GDN
        external-state binding, zero buffers, chunk + masked-bucket compile warmups), PLUS the eager
        programs prefill_traced_chunked runs around each replay: the rope-table slice at every chunk
        offset up to max_prompt_len, the copy into the persistent cos/sin buffers, and the last-token
        norm + LM-head path of an exact-multiple prompt (the greedy token ops instead when
        set_greedy_token_output(True), T1), and the tail's rope-table slice at every
        (full-chunk offset, masked bucket) pair (with I-1 ROPEWARM on, _warm_rope_table_slices warms
        these slices for every offset up to rope.max_seq_len instead). Then call prime_decode_trace (the decode
        trace compiles and allocates everything it needs while NO trace is parked), and last
        capture_prefill_trace_chunked(..., prepared=True).

        Why: a program compiled after a trace is parked gets its kernel-binary buffer in memory the
        parked trace uses as scratch, and the next replay overwrites it (#48536 class). The old order
        (capture prefill -> prime decode) put ~100 decode kernel binaries, 12 decode constants and
        the decode-trace inputs there; the next prefill replay corrupted them and the decode trace
        hung, depending on layout (TT_METAL_TRACE_ALLOC_TRACKING=1 names such buffers).

        max_prompt_len: longest prompt this process serves (default: the page_table's coverage)."""
        assert self.num_devices == 1, "prepare_prefill_trace_chunked is single-device only"
        if max_prompt_len is None:
            max_prompt_len = int(page_table.shape[1]) * get_block_size(self._paged_kv_caches)
        self._prepare_prefill_trace_chunked_setup(
            device, page_table, chunk_size, warmup_masked_buckets, request_warm_len=max_prompt_len
        )

    def _warm_prefill_request_programs(self, hidden_chunk, chunk_size, max_prompt_len):
        """Compile the per-request eager programs of prefill_traced_chunked's full-chunk path (see
        prepare_prefill_trace_chunked). Run only while no trace is parked."""
        n_full = max_prompt_len // chunk_size
        # I-1 ROPEWARM (default on) runs later in this same prepare step and warms a superset of
        # these slices (every reachable pair up to rope.max_seq_len), so this loop covers ROPEWARM=0.
        if not tpc.i1_enabled("ROPEWARM") and self.rope.rope_device_table_enabled() and self.rope._req_cos is None:
            n_full_rope = min(n_full, self.rope.max_seq_len // chunk_size)
            for c in range(n_full_rope):
                cos_slice, sin_slice = self.rope.get_prefill_rot_mats_table_slice(c * chunk_size, chunk_size)
                ttnn.copy(cos_slice, self._chunk_cos_buf)
                ttnn.copy(sin_slice, self._chunk_sin_buf)
                ttnn.deallocate(cos_slice)
                ttnn.deallocate(sin_slice)
            # Tail of a non-multiple prompt: prefill_masked_bucket slices the table at
            # [cs, cs + bucket) with cs = k * chunk_size (k >= 1). The slice program is keyed on its
            # bounds, so warm every (full-chunk offset, bucket) pair a prompt <= max_prompt_len hits
            # (the masked-bucket warmup only covers chunk_start = 0).
            for c in range(1, n_full + 1):
                cs = c * chunk_size
                for bucket in self._PREFILL_MASK_BUCKETS:
                    if bucket > chunk_size or cs + bucket > self.rope.max_seq_len:
                        continue
                    cos_t, sin_t = self.rope.get_prefill_rot_mats_table_slice(cs, bucket)
                    ttnn.deallocate(cos_t)
                    ttnn.deallocate(sin_t)
        if n_full >= 1:
            self._warm_exact_multiple_tail(hidden_chunk, chunk_size)
        ttnn.synchronize_device(self.device)

    def _exact_multiple_tail_device(self, hidden_chunk, chunk_size):
        """Device ops of prefill_traced_chunked's exact-multiple return path: last row (chunk_size - 1) of the
        last full chunk -> TILE -> DRAM -> final norm -> LM head, or -> the greedy token (_lm_head_token) when
        set_greedy_token_output(True) (T1). Returns the device result (logits or uint32 token). Shared by the
        request path (eager tail), its prepare warm-up (_warm_exact_multiple_tail) and M5 TAIL_TRACE
        (_m5_tail_body), so all of them run the same ops."""
        pos = chunk_size - 1
        if self._m2_lastrow:
            # M2 LASTROW: the chunk forward returns row chunk_size - 1 only (same code as the request path).
            assert hidden_chunk.shape[1] == 1, f"M2 LASTROW: chunk output {tuple(hidden_chunk.shape)} is not one row"
            x_last = hidden_chunk
        else:
            x_last = hidden_chunk[:, pos : pos + 1, :]
        x_last = ttnn.to_layout(x_last, ttnn.TILE_LAYOUT)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = self.norm(x_last, mode=Mode.PREFILL)
        return self._lm_head_token(x_last) if self._greedy_token_out else self._lm_head(x_last)

    def _tail_norm_fast(self, x):
        """Final norm of the last row on the 64 cores of the A3 LM-head in0 layout (L1 WIDTH_SHARDED 8x8, one
        [32, 32] tile per core) instead of L1->DRAM copy + 1-core DRAM norm + I2S: I2S(x) -> sharded rms_norm (same
        eps / gamma / HiFi2 fp32-dest ckc) written in that layout, so the A3 LM head skips its I2S (used by the SP
        vocab-sharded LM head, tt/sp_lmhead_shard.py). 23.5 -> 6.3 us device kernel. Not bit-exact (the cross-core
        sum of squares changes the reduction order)."""
        gx, gy = tpc.I3_A3_LM_CFG["in0_grid"]
        in0_mc = tpc.i3_l1_width_memcfg(int(x.shape[-1]), gx, gy)
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[gx, gy], subblock_w=1, block_h=1, block_w=1, inplace=False
        )
        xs = ttnn.to_memory_config(x, in0_mc)
        n = ttnn.rms_norm(
            xs,
            epsilon=self.norm.eps,
            weight=self.norm.weight,
            program_config=pc,
            memory_config=in0_mc,
            compute_kernel_config=self.norm.compute_kernel_config_hifi2,
        )
        ttnn.deallocate(xs)
        return n

    def _warm_exact_multiple_tail(self, hidden_chunk, chunk_size):
        """Compile prefill_traced_chunked's exact-multiple return path (_exact_multiple_tail_device).
        pos_in_chunk is always chunk_size - 1 there, so this one program set covers every such prompt length.
        Run only while no trace is parked.

        M5 TAIL_TRACE (self._m5_tail, prepared order): also allocate the tail trace's persistent output here
        (DRAM, the result's shape / dtype / layout; before the decode trace is primed and before any trace is
        captured) and compile the ttnn.copy into it, so the tail capture compiles nothing."""
        out = self._exact_multiple_tail_device(hidden_chunk, chunk_size)
        _ = out.cpu()
        if self._m5_tail:
            if self._m5_tail_out is not None:
                ttnn.deallocate(self._m5_tail_out)
            self._m5_tail_out = ttnn.allocate_tensor_on_device(
                ttnn.Shape(list(out.shape)), out.dtype, out.layout, self.device, ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.copy(out, self._m5_tail_out)
        ttnn.deallocate(out)

    def _prepare_prefill_trace_chunked_setup(
        self, device, page_table, chunk_size, warmup_masked_buckets, request_warm_len=0
    ):
        """Setup + compile warmups of capture_prefill_trace_chunked (single device), no capture.

        QWEN36_FLA_SCAN_FID_BY_LEN (see _fla_select_for_request): the warm-ups run with the FLA scan fidelity of
        the longest prompt this process serves (request_warm_len, else the page table's coverage), the fidelity
        the chunk trace is then captured with; when shorter prompts need the other fidelity, its FLA programs (the
        chunk forward and the masked buckets) are compiled here as well, before the decode trace is primed."""
        fla_fids, fla_init, serve_len = [], None, request_warm_len
        if fla_scan_fidelity_for_len(1) is not None and self._fla_gdn_layers():
            serve_len = request_warm_len or int(page_table.shape[1]) * get_block_size(self._paged_kv_caches)
            fla_fids = fla_scan_fidelities_up_to(serve_len)
            fla_init = fla_scan_fidelity_for_len(serve_len)
        self._fla_warm_fids = None
        self._fla_hold = bool(fla_fids)
        try:
            if fla_init is not None:
                self._fla_set_scan_fidelity(fla_init)
            self._prepare_prefill_trace_chunked_setup_body(
                device, page_table, chunk_size, warmup_masked_buckets, request_warm_len=request_warm_len
            )
            for fid in fla_fids:
                if fid != fla_init:
                    self._fla_warm_scan_fidelity(device, page_table, fid, warmup_masked_buckets)
            if fla_init is not None:
                self._fla_set_scan_fidelity(fla_init)
                self._fla_warm_fids = list(fla_fids)
                logger.info(
                    f"[FLA] BY_LEN: prepare (longest prompt {serve_len}; HiFi2 up to {fla_scan_fid_len_max()} tokens) "
                    f"compiled FLA prep/scan fidelities {[fla_fidelity_name(f) for f in fla_fids]}; the chunk trace "
                    f"takes {fla_fidelity_name(fla_init)}"
                )
        finally:
            self._fla_hold = False

    def _fla_warm_scan_fidelity(self, device, page_table, scan_fid, warmup_masked_buckets):
        """QWEN36_FLA_SCAN_FID_BY_LEN (prepare only, no trace parked yet): compile the FLA programs of scan_fid that
        a request can reach -- the T = chunk_size chunk forward (trace capture) and the masked buckets (short
        prompts, long-prompt tails). Every other program of these forwards is a cache hit. The dummy forwards dirty
        the GDN state / KV; the capture and every request reset them."""
        self._fla_set_scan_fidelity(scan_fid)
        self._reset_dn_state_inplace()
        out = self._forward_prefill_chunk(
            self._chunk_token_buf,
            self._chunk_cos_buf,
            self._chunk_sin_buf,
            self._chunk_start_idx_tensor,
            self._chunk_full_page_table_buf,
            self._chunk_page_table_buf,
        )
        ttnn.deallocate(out)
        ttnn.synchronize_device(device)
        if warmup_masked_buckets:
            self.warmup_prefill_masked_buckets(page_table)

    def _prepare_prefill_trace_chunked_setup_body(
        self, device, page_table, chunk_size, warmup_masked_buckets, request_warm_len=0
    ):
        """The body of _prepare_prefill_trace_chunked_setup (every warm-up at the GDN layers' current FLA config)."""
        assert self._deltanet_external_states is not None, "Call allocate_kv_caches first"
        assert chunk_size % 128 == 0, f"chunk_size {chunk_size} must be a multiple of 128"
        B = 1
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_chunk = chunk_size // block_size

        if self._chunked_trace_id is not None:
            ttnn.release_trace(device, self._chunked_trace_id)
            self._chunked_trace_id = None
        self._m3_release_repack_trace(device)  # M3 REPACK_TRACE (no-op when none is captured)
        self._pt_release_trace(device)  # P18_PRELUDE (no-op when none)
        self._m5_release_tail_trace(device)  # M5 TAIL_TRACE: trace + persistent output (no-op when none)

        self._chunked_chunk_size = chunk_size

        # M2 (tp_common M2_FLAG_DEFAULTS): fix the traced-chunk variants once, here, so the warm-up
        # forward below, the capture and prefill_traced_chunked all use the same program list.
        #   NOWHERE: only for a text-only model (no vision tower attached at prepare time).
        #   LASTROW: single device (this function) and a full-attention last layer with the dense MLP.
        self._m2_nowhere = tpc.m2_enabled("NOWHERE") and self.vision_model is None
        self._m2_lastrow = (
            tpc.m2_enabled("LASTROW")
            and self.layers[-1].is_full_attention
            and isinstance(self.layers[-1].feed_forward, Qwen36MLP)
        )
        self._m2_repack_pending = False
        if self._m2_nowhere or self._m2_lastrow or tpc.m2_enabled("REPACK_LATE"):
            logger.info(
                f"[M2] traced chunk: NOWHERE={int(self._m2_nowhere)} LASTROW={int(self._m2_lastrow)} "
                f"REPACK_LATE={int(tpc.m2_enabled('REPACK_LATE'))}"
            )
        # M4 (tp_common M4_FLAG_DEFAULTS): the last-layer LASTROW variants, only with LASTROW active.
        self._m4_r4a = self._m2_lastrow and tpc.m4_enabled("R4A")
        self._m4_r4b = self._m2_lastrow and tpc.m4_enabled("R4B")
        if tpc.m4_enabled("R4A") or tpc.m4_enabled("R4B"):
            logger.info(
                f"[M4] traced chunk last layer: R4A={int(self._m4_r4a)} R4B={int(self._m4_r4b)} "
                f"(LASTROW={int(self._m2_lastrow)}; M4 applies only with LASTROW)"
            )
        # M5 (tp_common M5_FLAG_DEFAULTS): TAIL_TRACE only in the prepared order (request_warm_len > 0: the
        # exact-multiple tail is warmed below and its persistent output made there); ADDNORM for the T == 2048
        # traced chunk forward (warm-up below, capture, replays).
        self._m5_tail = tpc.m5_enabled("TAIL_TRACE") and bool(request_warm_len)
        self._m5_addnorm = tpc.m5_enabled("ADDNORM")
        if tpc.m5_enabled("TAIL_TRACE") or self._m5_addnorm:
            logger.info(
                f"[M5] traced chunk: TAIL_TRACE={int(self._m5_tail)} (prepared order only) "
                f"ADDNORM={int(self._m5_addnorm)} (T in {sorted(tpc.M5_ADDNORM_T_SET)} chunks)"
            )

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
        # M4 R4B: absolute position of the chunk's last row (decode-SDPA cur_pos), next to the chunk start.
        # chunk_size - 1 matches the chunk start 0 above (warm-up / capture); prefill_traced_chunked writes
        # cs + chunk_size - 1 before every replay.
        self._chunk_last_pos_tensor = None
        self._m4_last_pos_host = None
        if self._m4_r4b:
            self._m4_last_pos_host = chunk_size - 1
            self._chunk_last_pos_tensor = ttnn.from_torch(
                torch.tensor([chunk_size - 1], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
        self._chunk_full_page_table_buf = ttnn.from_torch(
            page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self._chunk_page_table_buf = ttnn.from_torch(
            page_table[:, :blocks_per_chunk].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        # TP handoff: add ReplicateTensorToMesh for cos/sin (parity with tt/rope.py).
        # P7_ROPE (QWEN36_ROPE_L1, default "0"): the persistent per-chunk cos/sin buffers are tiny
        # ([1, chunk_size, rope_head_dim] bf16, e.g. 2048x64 = 256 KiB) and every FA layer's RoPE call
        # for the chunk reads the SAME values, so placing them in L1 (interleaved) instead of the
        # default DRAM interleaved turns every one of those reads into an L1 read. Allocated here
        # (before trace capture) and only ever refreshed in place (ttnn.copy /
        # copy_host_to_device_tensor into the same buffer), so the L1 placement holds for both
        # chunks across every trace replay. "1" opts in; default "0" is byte-identical to before.
        _rope_l1_mc = ttnn.L1_MEMORY_CONFIG if os.environ.get("QWEN36_ROPE_L1", "0") == "1" else None
        self._chunk_cos_buf = ttnn.from_torch(
            self.rope.cos_cpu[:chunk_size].unsqueeze(0).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_rope_l1_mc,
        )
        self._chunk_sin_buf = ttnn.from_torch(
            self.rope.sin_cpu[:chunk_size].unsqueeze(0).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_rope_l1_mc,
        )

        # Bind GDN to persistent external state; enable in-place carry across replays.
        for layer, (ext_rec, ext_conv) in zip(
            (l for l in self.layers if not l.is_full_attention), self._deltanet_external_states
        ):
            dn = layer.attention
            dn.recurrent_state = ext_rec
            dn.fused_conv_state = ext_conv
            dn.conv_state_q = None
            dn.conv_state_k = None
            dn.conv_state_v = None
            if dn.split_conv_state is not None:
                for buf in dn.split_conv_state:
                    ttnn.deallocate(buf)
                dn.split_conv_state = None
            dn._chunk_inplace_state = True
        self._init_dn_zero_buffers()

        # Warmup outside trace: compile per-chunk programs.
        self._reset_dn_state_inplace()
        warmup_out = self._forward_prefill_chunk(
            self._chunk_token_buf,
            self._chunk_cos_buf,
            self._chunk_sin_buf,
            self._chunk_start_idx_tensor,
            self._chunk_full_page_table_buf,
            self._chunk_page_table_buf,
        )
        if request_warm_len:
            # Prepared order (T7): every per-request eager program of prefill_traced_chunked,
            # including the exact-multiple tail (T1's greedy token ops when enabled).
            self._warm_prefill_request_programs(warmup_out, chunk_size, request_warm_len)
        elif self._greedy_token_out:
            # One-call order (T1): compile the exact-multiple return path of prefill_traced_chunked
            # (last chunk row -> norm -> _lm_head_token) now: an eager compile after the trace is
            # parked clobbers it (#48536).
            self._warm_exact_multiple_tail(warmup_out, chunk_size)
        ttnn.deallocate(warmup_out)
        ttnn.synchronize_device(device)

        # Warmup masked-bucket programs outside trace (same GDN mode as serving).
        # Dummy prefills dirty state/KV; reset before capture.
        if warmup_masked_buckets:
            self.warmup_prefill_masked_buckets(page_table)

        # I-1 ROPEWARM (QWEN36_I1_ROPEWARM): compile, NOW, the text-only RoPE table slices that
        # prefill_traced_chunked runs after the chunk trace is parked, plus its cos/sin ttnn.copy into
        # the persistent buffers. Otherwise they compile on request 0 after the trace is parked
        # (#48536 class). The slice program hash includes slice_start/slice_end, so every reachable
        # (start, length) pair needs its own warm call: chunk replays (c*chunk_size, chunk_size) and
        # masked tails (c*chunk_size, bucket), bounded by the RoPE table length (max_seq_len).
        if tpc.i1_enabled("ROPEWARM") and self.rope.rope_device_table_enabled() and self.rope._req_cos is None:
            self._warm_rope_table_slices(device, chunk_size)

        # M3 REPACK_TRACE: compile the conv-history repack programs (and allocate conv_hist) now, while no
        # trace is parked; the repack trace is captured after the chunk trace (capture_prefill_trace_chunked,
        # prepared=True). The capture resets the GDN state, so the repacked contents do not matter.
        if tpc.m3_enabled("REPACK_TRACE") and self.num_devices == 1:
            self._gdn_refresh_conv_hist()
            ttnn.synchronize_device(device)

    def _warm_rope_table_slices(self, device, chunk_size):
        """I-1 ROPEWARM helper (single device, text-only RoPE, before the chunk trace capture).

        Runs every rope.get_prefill_rot_mats_table_slice(start, length) that prefill_traced_chunked
        can reach for a prompt that fits the RoPE table (max_seq_len), so none compiles after the
        trace is parked: full-chunk replays (c*chunk_size, chunk_size) and the masked tail at
        chunk_start = c*chunk_size (c >= 1; c == 0 is warmed by warmup_prefill_masked_buckets) for
        each bucket <= chunk_size. Also runs the per-chunk ttnn.copy of the cos/sin slice into
        _chunk_cos_buf/_chunk_sin_buf once (copy programs are keyed by shape only). The copied rows
        are rows [0, chunk_size) of the table the buffers were filled from, so their contents do not
        change. Prompts that start at other offsets (e.g. prefix-cached starts) are not covered."""
        max_len = self.rope.max_seq_len
        buckets = [b for b in self._PREFILL_MASK_BUCKETS if b <= chunk_size]
        pairs = []
        for start in range(0, max_len, chunk_size):
            if start + chunk_size <= max_len:
                pairs.append((start, chunk_size))
            if start > 0:
                pairs.extend((start, b) for b in buckets if start + b <= max_len)
        copied = False
        for start, length in dict.fromkeys(pairs):
            cos_slice, sin_slice = self.rope.get_prefill_rot_mats_table_slice(start, length)
            if not copied and start == 0 and length == chunk_size:
                ttnn.copy(cos_slice, self._chunk_cos_buf)
                ttnn.copy(sin_slice, self._chunk_sin_buf)
                copied = True
            ttnn.deallocate(cos_slice)
            ttnn.deallocate(sin_slice)
        ttnn.synchronize_device(device)
        logger.info(f"ROPEWARM: warmed {len(dict.fromkeys(pairs))} RoPE table slices (max_seq_len={max_len})")

    def _capture_prefill_trace_chunked_tp(
        self, device, page_table, chunk_size=2048, warmup_masked_buckets=True, capture_chunk_trace=True
    ):
        """TP fork of capture_prefill_trace_chunked.

        Replicated persistent buffers; rope_tp cos/sin; GDN uses _stable_state (not external buffers).
        Trace replays _forward_prefill_chunk_tp."""
        assert self._deltanet_external_states is not None, "Call allocate_kv_caches first"
        assert chunk_size % 128 == 0, f"chunk_size {chunk_size} must be a multiple of 128"
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_chunk = chunk_size // block_size

        if self._chunked_trace_id is not None:
            ttnn.release_trace(device, self._chunked_trace_id)
            self._chunked_trace_id = None
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

        if not capture_chunk_trace:
            # Batched (B>1) vLLM path: masked-bucket programs are warmed above; skip parking the
            # chunk trace (it would bake the B=1 prefill scratch that is freed after warmup, and
            # batched serving handles short prompts only). num_full==0 prompts never need it.
            self._chunked_trace_id = None
            self._reset_gdn_state_for_new_sequence()
            logger.info("Masked-bucket prefill programs (TP) warmed; chunk trace skipped (batched path).")
            return

        # Capture trace.
        self._reset_gdn_state_for_new_sequence()
        self._chunked_trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self._chunked_trace_output = self._forward_prefill_chunk_tp(
            self._chunk_token_buf,
            self._chunk_cos_buf,
            self._chunk_sin_buf,
            self._chunk_start_idx_tensor,
            self._chunk_full_page_table_buf,
            self._chunk_page_table_buf,
        )
        ttnn.end_trace_capture(device, self._chunked_trace_id, cq_id=0)
        logger.info("Chunked prefill trace (TP) captured successfully!")

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
                    dn._stable_state,
                )
            )
            # reset_state allocates against self.B, so set B=1 first.
            dn.B = 1
            dn.reset_state()  # builds rec_state [1,Nv,Dk,Dv], conv_states[*] [1,1,D], conv_carry, _zero_conv0
            dn._stable_state = True  # in-place carry so the trace's baked addresses survive replays
        return prev

    def _restore_gdn_batched(self, prev):
        """Restore the batched [B,...] GDN bindings saved by _alloc_gdn_scratch_b1 and free
        the B=1 scratch, so decode reads the assembled batched state."""
        for dn, B_b, rec_b, conv_b, carry_b, zero0_b, stable_b in prev:
            # Free the B=1 scratch allocated for the prefill trace.
            if dn.rec_state is not None:
                ttnn.deallocate(dn.rec_state)
            for cs in dn.conv_states or []:
                ttnn.deallocate(cs)
            if dn.conv_carry is not None:
                ttnn.deallocate(dn.conv_carry)
            if dn._zero_conv0 is not None:
                ttnn.deallocate(dn._zero_conv0)
            # Rebind the batched decode buffers.
            dn.B = B_b
            dn.rec_state = rec_b
            dn.conv_states = conv_b
            dn.conv_carry = carry_b
            dn._zero_conv0 = zero0_b
            dn._stable_state = stable_b

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
            saved = (dn.B, dn.rec_state, dn.conv_states, dn.conv_carry, dn._zero_conv0, dn._stable_state)
            dn.B = 1
            dn.reset_state()  # builds rec_state [1,Nv,Dk,Dv], conv_states[*] [1,1,D], conv_carry, _zero_conv0
            scratch.append((dn, dn.rec_state, dn.conv_states, dn.conv_carry, dn._zero_conv0))
            dn.B, dn.rec_state, dn.conv_states, dn.conv_carry, dn._zero_conv0, dn._stable_state = saved
        self._gdn_prefill_scratch = scratch

    def _bind_gdn_prefill_scratch(self):
        """Bind the persistent B=1 prefill scratch onto every GDN layer (prefill runs B=1); returns the
        saved batched decode bindings for _unbind_gdn_prefill_scratch. Allocates the scratch on first use.
        The drop-in analogue of _alloc_gdn_scratch_b1 that reuses one persistent scratch instead of
        allocating a throwaway per call (so the chunk trace's baked addresses stay valid)."""
        self._ensure_gdn_prefill_scratch()
        prev = []
        for dn, rec, conv, carry, zero0 in self._gdn_prefill_scratch:
            prev.append((dn, dn.B, dn.rec_state, dn.conv_states, dn.conv_carry, dn._zero_conv0, dn._stable_state))
            dn.B = 1
            dn.rec_state = rec
            dn.conv_states = conv
            dn.conv_carry = carry
            dn._zero_conv0 = zero0
            dn._stable_state = True  # in-place carry so the trace's baked addresses survive replays
        return prev

    def _unbind_gdn_prefill_scratch(self, prev):
        """Rebind the batched [B,...] decode buffers saved by _bind_gdn_prefill_scratch, WITHOUT freeing
        the persistent scratch (unlike _restore_gdn_batched, whose scratch is throwaway)."""
        for dn, B_b, rec_b, conv_b, carry_b, zero0_b, stable_b in prev:
            dn.B = B_b
            dn.rec_state = rec_b
            dn.conv_states = conv_b
            dn.conv_carry = carry_b
            dn._zero_conv0 = zero0_b
            dn._stable_state = stable_b

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
        """Capture ONE B=1 full-bucket prefill trace (all-layer forward, valid_len=None) for
        batched serving of prompts whose length is EXACTLY the bucket. GDN points at a B=1
        scratch. Replay once per user via prefill_traced_bucket_batched.

        Only full-bucket prompts are traced: valid_len cannot be masked inside a trace, so a
        short prompt would pad through the GDN recurrence and corrupt the decode state. Callers
        route actual_len < bucket prompts to eager prefill_paged_peruser.

        Args:
          device: mesh device.
          page_table: torch.Tensor [1, bpu] int32 — one user's row (buffer width fixed across
            replays; each replay DMAs a different user's row in).
          bucket: fixed bucket length (128 for ISL==128; must be a multiple of 128).
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
        """Traced batched short-prompt prefill: replay the captured B=1 full-bucket trace once
        per user, stitching each replay's B=1 GDN state into row u of the batched [B,...] decode
        buffer. Attention fills each user's physical blocks via the per-user page-table row
        (batch_idx=0 baked into the trace). Returns a list of B device logits [1, 1, vocab].

        CORRECTNESS CONTRACT: every user's actual_len MUST equal the captured bucket. The trace
        runs valid_len=None (no GDN mask); for a short prompt padded to the bucket that would push
        padding tokens through the GDN recurrence and corrupt the decode state. Short prompts must
        be routed to eager prefill_paged_peruser. Asserts actual_len == bucket and never pads.
        Call capture_prefill_trace_bucket first; release_prefill_trace_bucket before decode.
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

        # Assemble the per-user states into row u in place (_stable_state path).
        self._assemble_per_user_gdn(per_user_rec, per_user_conv)

        # Re-upload the per-user logits as stable device tensors after all allocations.
        return self._reupload_host_logits(host_logits)

    def _assemble_per_user_gdn(self, per_user_rec, per_user_conv):
        """Stitch B per-user B=1 GDN states (host torch) into row u of the batched [B,...] decode
        buffers via assemble_batched_state. The batched GDN bindings MUST already be rebound (writes
        in place under _stable_state). Shared by prefill_traced_bucket_batched/prefill_chunked_peruser.

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

        # Bind the persistent B=1 GDN prefill scratch: prefill runs B=1 and its per-sequence reset
        # would otherwise zero the batched [B,...] decode buffer (clobbering the live rows). This is
        # the SAME scratch whose addresses the chunk-prefill trace baked at warmup, so the traced
        # long-prompt path replays correctly; short prompts use the masked bucket on the same scratch.
        prev = self._bind_gdn_prefill_scratch()
        host_logits = []
        per_user_rec = []
        per_user_conv = []
        try:
            for u in range(N):
                toks = token_ids_list[u]
                assert toks.shape[0] == 1, f"request {u}: token_ids must be [1, T_u]"
                actual = int(valid_lens[u]) if valid_lens is not None else toks.shape[1]
                assert actual >= 1, f"request {u}: empty prompt (actual_len={actual})"
                # Trace-safe prefill into the B=1 scratch: prefill_traced_chunked runs short prompts in
                # one masked-bucket forward and chunks longer ones; GDN state carries + is snapshotted below.
                lg = self.prefill_traced_chunked(toks[:, :actual], pt[u : u + 1], actual_len=actual)
                host_logits.append(
                    ttnn.to_torch(lg, mesh_composer=comp).reshape(-1, self.args.vocab_size)[:1].float().view(1, 1, -1)
                )
                ttnn.deallocate(lg)
                # Snapshot this user's B=1 scratch state (host round trip — the next user's reset
                # overwrites the scratch in place; see prefill_traced_bucket_batched for why not clone).
                per_user_rec.append([ttnn.to_torch(dn.rec_state, mesh_composer=comp) for dn in dn_states])
                per_user_conv.append(
                    [[ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states] for dn in dn_states]
                )
        finally:
            # Always rebind the batched decode buffers (a mid-loop assert must not leave GDN on scratch).
            # Does NOT free the scratch — it persists for the next request and keeps the trace valid.
            self._unbind_gdn_prefill_scratch(prev)

        # Write each user's snapshot into its decode slot, preserving the other live rows.
        for u in range(N):
            self._write_gdn_slot(int(empty_slots[u]), per_user_rec[u], per_user_conv[u])
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

    def _remap_gdn_slots(self, remap):
        """Apply a vLLM batch-condense slot_remap to every GDN layer's batched decode state
        (device-side; slot i takes the state at slot remap[i]). Mirrors seed_manager.apply_slot_remap
        for GDN's per-slot recurrent+conv state, which the plugin's slot_remap does not itself move.
        No-op for an identity remap."""
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.remap_slots(remap)

    def prefill_chunked_peruser(self, token_ids_list, page_table, valid_lens=None):
        """Batched per-user LONG-prefill (TP, eager). Runs the single-user chunk-outer path
        (prefill_traced_chunked) for each user into a B=1 GDN scratch, then stitches each user's
        final GDN state into row u of the batched [B,...] decode buffers.

        Handles ANY prompt length with exact valid_len masking (no last-token padding through the
        GDN recurrence): short -> masked bucket; long -> chunk-outer + masked tail. The long-prompt
        counterpart to prefill_traced_bucket_batched (full-bucket-only) and prefill_paged_peruser
        (single-pass). Call allocate_kv_caches(batch_size=B) first.

        token_ids_list: list of B torch.Tensor [1, T_u] (lengths may differ).
        page_table:      torch.Tensor [B, max_blocks_per_seq] int32 — row u = user u's blocks.
                         IMPORTANT: max_blocks_per_seq MUST be a multiple of 8. The chunked SDPA
                         reads each row as a ROW_MAJOR int32 stick requiring stick_size
                         (width * 4 bytes) % 32 == 0, i.e. width % 8 == 0. A misaligned width
                         makes the SDPA read the wrong KV.
        valid_lens:      optional list of B ints (real token counts); defaults to each T_u.
        Returns:         list of B ttnn logits [1, 1, vocab] (replicated; at valid_len-1).
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
        # F10B item C: text-only masked-bucket prefill slices cos/sin straight from the persistent
        # device table instead of computing on host and uploading (QWEN36_ROPE_DEVICE_TABLE, default
        # "1"); M-RoPE requests (self.rope._req_cos staged) or the flag off keep the host path.
        if self.rope.rope_device_table_enabled() and self.rope._req_cos is None:
            cos, sin = self.rope.get_prefill_rot_mats_table_slice(chunk_start, bucket)
        else:
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
        # The op reads chunk_start on device and has no host check: it must be a multiple of q_chunk.
        assert chunk_start % flexible_sdpa_q_chunk() == 0, f"chunk_start {chunk_start} % SDPA q_chunk != 0"
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
        # Deallocate per-chunk inputs; only hidden survives (leaving these live let the eager
        # tail leak device buffers, which made the NEXT execute_trace hang once a chunk-outer
        # prefill trace was parked -- see the TP twin below for the same pattern).
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(full_pt)
        ttnn.deallocate(chunk_pt)
        if csi_tensor is not None:
            ttnn.deallocate(csi_tensor)
        return x

    def _forward_prefill_chunk_masked_tp(
        self, token_buf, valid_len, chunk_start, page_table, bucket, flex_sdpa=True, vision_tokens=None
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
                    chunk_start_idx_tensor=csi_tensor,
                    valid_len=valid_len,  # unused by full attention
                )
            else:
                x_new = layer.forward(x, mode="prefill", chunk_size=self.args.gdn_chunk_size, valid_len=valid_len)
            ttnn.deallocate(x)
            x = x_new
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
    ):
        """Masked fixed-bucket prefill for a segment of `actual_len` real tokens.

        Pads the segment up to a fixed bucket length, runs all layers ONCE, and masks the GDN
        recurrent + conv state so they reflect exactly `actual_len` real tokens — numerically
        equivalent to the eager exact-length path (prefill_paged) but using one of only a few
        bucket-sized programs instead of compiling a fresh program per prompt length. That
        bounded program set is what makes warmup able to compile every code path before a trace
        is parked, so a short request can never trigger the compile-clobbers-trace hang.

        `chunk_start` is the segment's absolute start position (0 for a from-scratch short
        prompt; num_full*chunk_size for the tail of a long prompt — the carried GDN/KV state
        must already be in place). `vis_row_offset` is the number of image-placeholder tokens
        before this segment (so a tail that holds the bottom of a large image splices the right
        slice of the packed vision rows). Returns ttnn.Tensor (host) [1, 1, vocab_size]: the
        logit after position actual_len-1.
        """
        B_batch, _ = token_ids.shape
        assert B_batch == 1, "masked-bucket prefill is single-sequence"
        if bucket is None:
            bucket = self._mask_bucket_for(actual_len)
        assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"

        if chunk_start == 0:
            # QWEN36_FLA_SCAN_FID_BY_LEN: a new sequence takes the FLA scan fidelity of its length (a long prompt's
            # tail, chunk_start > 0, keeps the one prefill_traced_chunked chose). No-op when the flag is off.
            self._fla_select_for_request(actual_len, replays_chunk_trace=False)
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
        # Fused GDN decode: this forward is the last fused_conv_state writer (whole short prompt or
        # the long-prompt tail). Enqueued before the sync (overlaps the forward); no-op when off.
        self._gdn_refresh_conv_hist()
        ttnn.synchronize_device(self.device)

        if self.num_devices > 1:
            return self._masked_bucket_logits_tp(hidden, actual_len, bucket)

        # One-hot matmul for last row (fixed program per bucket; slice would recompile per length).
        sel = torch.zeros(1, 1, bucket, dtype=torch.float32)
        sel[0, 0, actual_len - 1] = 1.0
        sel_tt = ttnn.from_torch(sel, dtype=hidden.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        x_last = ttnn.matmul(sel_tt, hidden)
        ttnn.deallocate(sel_tt)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = self.norm(x_last, mode=Mode.PREFILL)
        logits = self._lm_head_token(x_last) if self._greedy_token_out else self._lm_head(x_last)
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

        Two program kinds:
          * bucket-keyed (SDPA / GDN mask / norm / MLP-or-MoE): one per (bucket, is_full). Warmed
            by a dummy forward at each bucket, once masked (actual_len < bucket) and once full (==).
          * fill-width-keyed (paged_fill_cache): hashes on the fill shape, so it recompiles per
            fill width. Warmed directly by _warmup_paged_fill_widths (no full forward).

        MUST run in GDN serving state, before any trace is parked (capture_prefill_trace_chunked
        calls this just before begin_trace_capture). page_table must cover the largest bucket."""
        if buckets is None:
            chunk_size = self._chunked_chunk_size
            # Cap at chunk_size: a chunk-outer tail is always < chunk_size, and a whole prompt
            # shorter than a chunk (num_full==0) also routes through a bucket <= chunk_size
            # (prefill_traced_chunked / prefill_masked_bucket) -- buckets above chunk_size are
            # never reached at runtime. Also avoids warming a bucket wider than the vision-merge
            # buffers, which _alloc_vision_merge_buffers sizes to chunk_size (not to the widest
            # fixed bucket).
            buckets = (
                [b for b in self._PREFILL_MASK_BUCKETS if b <= chunk_size] if chunk_size else self._PREFILL_MASK_BUCKETS
            )
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

    def prefill_traced_chunked(self, token_ids, page_table, actual_len, vision_tokens=None):
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
        # QWEN36_FLA_SCAN_FID_BY_LEN: the FLA scan fidelity of this prompt length; re-captures the chunk trace first
        # when it holds the other fidelity (before any per-request staging, as after the first capture). No-op
        # when the flag is off.
        self._fla_select_for_request(actual_len, replays_chunk_trace=num_full > 0)

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
            return self.prefill_masked_bucket(
                token_ids[:, :actual_len], page_table, actual_len=actual_len, chunk_start=0, vision_tokens=vision_tokens
            )

        if self.num_devices > 1:
            # TP long prompt: traced replay preferred; eager masked-bucket fallback if no trace.
            if self._chunked_trace_id is not None:
                return self._prefill_traced_chunked_tp(
                    token_ids, page_table, actual_len, num_full, chunk_size, tail_real, vision_tokens=vision_tokens
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

        # M2 NOWHERE: the parked chunk trace has no vision-splice where (text-only model at capture), so a
        # multimodal request cannot be served by it.
        assert not (self._m2_nowhere and vision_tokens is not None), (
            "M2 NOWHERE: the chunk trace was captured without the vision-splice where (text-only model); "
            "set QWEN36_M2_NOWHERE=0 or call init_vision_model() before prepare/capture"
        )
        # Re-zero GDN once; carries across replays + masked tail (chunk_start>0 skips reset).
        _pt_used = self._pt_replay_ok(chunk_size)
        if _pt_used:
            # P18_PRELUDE: one trace = state reset (no conv_hist) + chunk-0 RoPE cos/sin.
            self._pt_replay()
        else:
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

        # Replay trace for each full chunk. The flexible SDPA reads the chunk start on device (no host
        # check): every c * chunk_size must be a multiple of its q_chunk.
        assert chunk_size % flexible_sdpa_q_chunk() == 0, f"chunk_size {chunk_size} % SDPA q_chunk != 0"
        for c in range(num_full):
            cs = c * chunk_size
            tok_host = ttnn.from_torch(
                token_ids[:, cs : cs + chunk_size].to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            ttnn.copy_host_to_device_tensor(tok_host, self._chunk_token_buf)

            # P18_PRELUDE: these two host tensors depend only on the chunk start, so they are built once per
            # chunk start and reused (same bytes as building them per request).
            _hc = self._pt_host_cache.get((cs, chunk_size))
            if _hc is None:
                _hc = (
                    ttnn.from_torch(
                        torch.tensor([cs], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
                    ),
                    ttnn.from_torch(
                        torch.tensor([cs + chunk_size - 1], dtype=torch.int32),
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    ),
                )
                if _pt_used:
                    self._pt_host_cache[(cs, chunk_size)] = _hc
            csi_host, lp_host = _hc
            ttnn.copy_host_to_device_tensor(csi_host, self._chunk_start_idx_tensor)
            if self._m4_r4b:
                # M4 R4B: absolute position of this chunk's last row = the decode-SDPA cur_pos of the last layer.
                ttnn.copy_host_to_device_tensor(lp_host, self._chunk_last_pos_tensor)
                self._m4_last_pos_host = cs + chunk_size - 1

            blk0 = cs // block_size
            cpt_host = ttnn.from_torch(
                page_table[:, blk0 : blk0 + blocks_per_chunk].contiguous(),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(cpt_host, self._chunk_page_table_buf)

            # M-RoPE-aware per-chunk cos/sin (slices the staged per-request table for multimodal;
            # 1D RoPE for text). Updated into the persistent buffer per chunk so it stays trace-safe
            # (the buffer's device address is baked into the parked trace; only its contents change).
            #
            # F10B item C (QWEN36_ROPE_DEVICE_TABLE, default "1"): text-only requests slice the
            # persistent device table (self.rope.cos_device/sin_device) and ttnn.copy the slice
            # straight into _chunk_cos_buf/_chunk_sin_buf -- device-to-device, no host round trip.
            # M-RoPE requests (self.rope._req_cos staged) or the flag off keep the original host
            # compute (prefill_cos_sin_torch) + upload (copy_host_to_device_tensor) path.
            if _pt_used and c == 0:
                pass  # chunk-0 cos/sin were written by the prelude trace
            elif self.rope.rope_device_table_enabled() and self.rope._req_cos is None:
                cos_slice, sin_slice = self.rope.get_prefill_rot_mats_table_slice(cs, chunk_size)
                ttnn.copy(cos_slice, self._chunk_cos_buf)
                ttnn.copy(sin_slice, self._chunk_sin_buf)
                ttnn.deallocate(cos_slice)
                ttnn.deallocate(sin_slice)
            else:
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
            # M2 NOWHERE: the trace has no where, so there is nothing to stage (text-only request; the
            # masked tail stages its own buffers in prefill_masked_bucket).
            if not self._m2_nowhere:
                self._set_vision_merge(
                    token_ids[:, cs : cs + chunk_size], vision_tokens, self._vis_row_offset_for(token_ids, cs)
                )

            if os.environ.get("QWEN36_PREFILL_DEBUG") == "1":
                print(f"[PTC] before execute_trace chunk={c}", flush=True)
            # M4 R4B: a replay without this chunk's last-row position would attend the wrong rows silently.
            assert (
                not self._m4_r4b or self._m4_last_pos_host == cs + chunk_size - 1
            ), f"M4 R4B: chunk {c}: last-row position {self._m4_last_pos_host} != {cs + chunk_size - 1} (not written)"
            ttnn.execute_trace(self.device, self._chunked_trace_id, cq_id=0, blocking=False)
            if os.environ.get("QWEN36_PREFILL_DEBUG") == "1":
                print(f"[PTC] after execute_trace chunk={c}", flush=True)

        if tail_real == 0:
            # Fused GDN decode (QWEN36_GDN_DECODE_FUSED=2): the last chunk replay is the last
            # fused_conv_state writer. Enqueued before the sync so the host dispatch overlaps the
            # replays; no-op when the flag is off. (A tail rebuilds it in prefill_masked_bucket.)
            # M2 REPACK_LATE: defer it out of the first-token path; the next GDN-state user
            # (switch_mode / prepare_decode_inputs_host / decode / GDN reset, save, restore) runs it
            # first via _m2_flush_pending_repack, so it completes before the first decode step.
            # M3 REPACK_TRACE: replay the repack trace (captured after the chunk trace) at this same point
            # instead of the ~8 eager ops per GDN layer; same ops, same buffers.
            if tpc.m2_enabled("REPACK_LATE"):
                self._m2_repack_pending = True
            elif self._m3_repack_trace_id is not None:
                if os.environ.get("QWEN36_REPACK_AFTER_TTFT") == "1":
                    # Defer the repack-trace replay out of TTFT: _m2_flush_pending_repack replays it before
                    # the first GDN-state user (switch_mode / decode / reset / save / restore).
                    self._m2_repack_pending = True
                else:
                    self._m3_replay_repack_trace()
            else:
                self._gdn_refresh_conv_hist()
            # M5 TAIL_TRACE: replay the tail trace (final norm + LM head [+ argmax] into the persistent output)
            # right after the repack, in place of the eager tail below; same ops as the eager tail.
            if self._m5_tail_trace_id is not None:
                self._m5_replay_tail_trace()
        ttnn.synchronize_device(self.device)
        if os.environ.get("QWEN36_PREFILL_DEBUG") == "1":
            print(f"[PTC] after synchronize chunk={num_full - 1}", flush=True)

        # Tail via masked bucket (or last full chunk hidden if exact multiple of chunk_size).
        if tail_real > 0:
            cs = num_full * chunk_size
            if os.environ.get("QWEN36_PREFILL_DEBUG") == "1":
                print("[PTC] before tail", flush=True)
            result = self.prefill_masked_bucket(
                token_ids[:, cs:actual_len],
                page_table,
                actual_len=tail_real,
                chunk_start=cs,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, cs),
            )
            if os.environ.get("QWEN36_PREFILL_DEBUG") == "1":
                print("[PTC] after tail", flush=True)
            return result
        hidden = self._chunked_trace_output  # last full chunk's hidden state
        pos_in_chunk = (actual_len - 1) - (num_full - 1) * chunk_size
        # Exact multiple: the requested row is always the last row of the last chunk, the row
        # _exact_multiple_tail_device reads (M2 LASTROW: the trace already returns only that row, [1, 1, dim]).
        assert pos_in_chunk == chunk_size - 1, f"exact-multiple tail: last row {pos_in_chunk} != chunk_size - 1"
        if self._m2_lastrow:
            assert hidden.shape[1] == 1, f"M2 LASTROW: trace output {tuple(hidden.shape)} is not one row"
        if self._m5_tail_trace_id is not None:
            # M5 TAIL_TRACE: the tail trace (replayed above, before the sync) wrote the result here.
            return self._m5_tail_out.cpu()
        ttnn.synchronize_device(self.device)
        # Same code as the prepare warm-up (_warm_exact_multiple_tail) and M5 TAIL_TRACE (_m5_tail_body).
        logits = self._exact_multiple_tail_device(hidden, chunk_size)
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

    def _prefill_traced_chunked_tp(
        self, token_ids, page_table, actual_len, num_full, chunk_size, tail_real, vision_tokens=None
    ):
        """TP traced chunk-outer prefill: replay the captured per-chunk trace
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
        # M4 R4B needs the per-chunk last-row position write of the single-device loop; the TP chunk forward
        # (_forward_prefill_chunk_tp) has no LASTROW, so R4B must never be active here.
        assert not self._m4_r4b, "M4 R4B is single-device only (the TP chunk trace has no last-row position input)"
        for c in range(num_full):
            cs = c * chunk_size
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
            cs = num_full * chunk_size
            return self.prefill_masked_bucket(
                token_ids[:, cs:actual_len],
                page_table,
                actual_len=tail_real,
                chunk_start=cs,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, cs),
            )
        return self._masked_bucket_logits_tp(self._chunked_trace_output, chunk_size, chunk_size)

    def reset_state(self, batch_size=None):
        """Reset layer state for a new sequence (eager/pre-trace path; trace uses _reset_dn_state_inplace)."""
        _sp = getattr(self.args, "sequence_parallel", False)
        for layer in self.layers:
            if layer.is_full_attention:
                # TPAttention (sequence_parallel at tp=1) has reset_state(), not reset_cache().
                layer.attention.reset_state() if _sp else layer.attention.reset_cache()
            else:
                # TPGatedDeltaNet.reset_state() takes no batch_size arg (unlike Qwen36GatedDeltaNet).
                layer.attention.reset_state() if _sp else layer.attention.reset_state(batch_size)

    def _reset_gdn_state_for_new_sequence(self):
        """Zero GDN recurrent+conv at sequence start.

        Trace capture runs forward twice; GDN state is non-idempotent. Must re-zero before each
        real sequence. In-place buffers (_chunk_inplace_state) use _reset_dn_state_inplace."""
        self._m2_flush_pending_repack()  # M2 REPACK_LATE: keep the old op order (repack before the reset)
        if self.num_devices > 1:
            # TP: reset_state_inplace preserves decode-trace baked addresses.
            for layer in self.layers:
                if not layer.is_full_attention:
                    layer.attention.reset_state_inplace()
            return
        inplace = any(
            (not l.is_full_attention) and getattr(l.attention, "_chunk_inplace_state", False) for l in self.layers
        )
        if inplace:
            self._reset_dn_state_inplace()
        else:
            self.reset_state(batch_size=1)

    def _reset_dn_state_inplace(self):
        """Zero DN state in place via pre-allocated zero buffers (trace addresses fixed)."""
        assert self._dn_zero_recurrent is not None, "Call _init_dn_zero_buffers first"
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            ttnn.copy(self._dn_zero_recurrent, dn.recurrent_state)
            ttnn.copy(self._dn_zero_conv, dn.fused_conv_state)
            if getattr(dn, "_decode_fused", False) and dn.conv_hist is not None:
                ttnn.copy(self._dn_zero_hist, dn.conv_hist)
            # split_conv_state rebuilt lazily on first decode.
            if dn.split_conv_state is not None:
                for buf in dn.split_conv_state:
                    ttnn.deallocate(buf)
                dn.split_conv_state = None

    def refresh_tp_gdn_conv_hist(self):
        """QWEN36_TP_GDN_DECODE_FUSED (TP mesh): rebuild every TP GDN layer's packed conv history from its
        conv_states (eager device ops, trace safe). Call after writing rec_state / conv_states from outside (SP -> TP
        handoff / inject), before replaying the decode trace. No-op when the flag is off or on single device."""
        for layer in self.layers:
            if not layer.is_full_attention and hasattr(layer.attention, "refresh_fused_conv_hist"):
                layer.attention.refresh_fused_conv_hist()

    def _gdn_refresh_conv_hist(self, variant=None):
        """QWEN36_GDN_DECODE_FUSED=2: rebuild every GDN layer's packed conv_hist from its
        fused_conv_state, on device (no host reads -> trace safe), eager. Call after the last
        fused_conv_state writer (end of prefill), after a state restore and on reset. No-op when
        the flag is off, on TP, or before any fused layer exists. variant: batched | perlayer
        (default: QWEN36_GDN_CONV_REPACK)."""
        self._m2_repack_pending = False  # M2 REPACK_LATE: this call is the (possibly deferred) repack
        if self.num_devices > 1:
            return
        dns = [
            l.attention for l in self.layers if not l.is_full_attention and getattr(l.attention, "_decode_fused", False)
        ]
        if not dns:
            return
        from models.demos.blackhole.qwen36.tt.gdn import decode_fused as _df

        variant = variant or _df.repack_variant()
        live = [dn for dn in dns if dn.fused_conv_state is not None and dn.fused_conv_state.shape[0] == 1]
        for dn in dns:
            if dn not in live:
                dn.refresh_conv_hist()  # no conv state yet -> zero history
        if not live:
            return
        if variant == "batched":
            _df.repack_conv_hist_batched([(dn.fused_conv_state, dn.ensure_conv_hist()) for dn in live], live[0].cfg)
        elif variant == "gather":
            _df.repack_conv_hist_gather(
                [(dn.fused_conv_state, dn.ensure_conv_hist()) for dn in live], live[0].cfg, self.device
            )
        else:
            for dn in live:
                _df.repack_conv_hist(dn.fused_conv_state, dn.ensure_conv_hist(), dn.cfg)

    def _init_dn_zero_buffers(self):
        """Allocate shared zero buffers for DN recurrent and conv shapes."""
        if self._dn_zero_recurrent is not None:
            return
        # First DN layer defines shared zero-buffer shapes.
        first_dn = next(layer.attention for layer in self.layers if not layer.is_full_attention)
        rec_shape = list(first_dn.recurrent_state.shape)
        conv_shape = list(first_dn.fused_conv_state.shape)
        self._dn_zero_recurrent = ttnn.zeros(
            rec_shape,
            dtype=first_dn.recurrent_state.dtype if getattr(first_dn, "_decode_fused", False) else ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._dn_zero_conv = ttnn.zeros(
            conv_shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if getattr(first_dn, "_decode_fused", False):
            self._dn_zero_hist = ttnn.zeros_like(first_dn.ensure_conv_hist())

    def set_paged_kv_caches(self, kv_caches):
        """Attach paged KV caches to the 8 attention layers."""
        self._paged_kv_caches = kv_caches
        for cache_idx, layer_idx in enumerate(self._attention_layer_indices):
            k_cache, v_cache = kv_caches[cache_idx]
            self.layers[layer_idx].attention.set_paged_kv_cache(k_cache, v_cache)

    def allocate_kv_caches(self, kv_cache_shape, dtype, batch_size=1):
        """Allocate caches for all 32 layers. Returns only the attention KV caches (for vLLM)."""
        assert self._deltanet_external_states is None, "allocate_kv_caches already called; deallocate first"
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

        self._deltanet_external_states = []
        _sp = getattr(self.args, "sequence_parallel", False)
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                if _sp:
                    # TPGatedDeltaNet (sequence_parallel at tp=1) self-manages recurrent/conv state
                    # internally -- no num_v_heads/head_k_dim/head_v_dim/conv_kernel_size/.cfg and no
                    # set_external_state (those are Qwen36GatedDeltaNet-only). Same handling as
                    # _allocate_kv_caches_tp's GDN branch; nothing to append (no external buffers).
                    dn.B = batch_size
                    dn.reset_state()
                    dn._stable_state = True
                    continue
                # Fused GDN decode (QWEN36_GDN_DECODE_FUSED=2, B = 1): FP32 state, so prefill writes
                # FP32 directly (no per-chunk state typecasts) and the op updates it in place.
                fused = getattr(dn, "_decode_fused", False) and batch_size == 1
                rec = ttnn.from_torch(
                    torch.zeros(batch_size, dn.num_v_heads, dn.head_k_dim, dn.head_v_dim, dtype=torch.bfloat16),
                    dtype=ttnn.float32 if fused else ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
                if fused:
                    dn.ensure_conv_hist()
                conv = ttnn.from_torch(
                    torch.zeros(
                        batch_size,
                        dn.conv_kernel_size - 1,
                        dn.cfg.q_dim + dn.cfg.k_dim + dn.cfg.v_dim,
                        dtype=torch.bfloat16,
                    ),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
                dn.set_external_state(rec, conv)
                self._deltanet_external_states.append((rec, conv))

        return kv_caches

    def free_kv_caches(self):
        """Release KV caches + GDN state for a fresh generation run."""
        if self._deltanet_external_states is None:
            return
        self._m3_release_repack_trace()  # M3 REPACK_TRACE: it bakes the GDN conv-state buffers freed below
        self._pt_release_trace()  # P18_PRELUDE: it bakes the GDN state buffers freed below
        self._m5_release_tail_trace()  # M5 TAIL_TRACE: it reads the chunk-trace output (released below)
        if getattr(self, "_chunked_trace_id", None) is not None:
            ttnn.release_trace(self.device, self._chunked_trace_id)
            self._chunked_trace_id = None
        self._chunked_trace_fla_fid = None  # QWEN36_FLA_SCAN_FID_BY_LEN: nothing parked, nothing prepared
        self._fla_warm_fids = None
        for rec, conv in self._deltanet_external_states:
            ttnn.deallocate(rec)
            ttnn.deallocate(conv)
        self._deltanet_external_states = None
        if getattr(self, "_paged_kv_caches", None) is not None:
            for k_cache, v_cache in self._paged_kv_caches:
                ttnn.deallocate(k_cache)
                ttnn.deallocate(v_cache)
            self._paged_kv_caches = None

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
                # Fixed-address GDN state for decode trace compatibility.
                layer.attention._stable_state = True
        # Marker for re-entry assert; TP GDN state lives in module, not external buffers.
        self._deltanet_external_states = []
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

        Prefills B users into ONE shared paged KV cache + the batched GDN decode state, one user at
        a time. Each user's full-attention layers fill their own blocks via the per-user page-table
        row; each GDN layer collects that user's from-scratch state, stitched into row u of the
        batched decode buffers by finalize_pending(). Call allocate_kv_caches(batch_size=B) first.

        token_ids_list: list of B torch.Tensor [1, T_u] (lengths may differ).
        page_table:      torch.Tensor [B, max_blocks_per_seq] int32 — row u = user u's blocks.
        valid_lens:      optional list of B ints (real token counts); defaults to each T_u.
        Returns:         list of B ttnn logits [1, 1, vocab_size] (one per user, at valid_len-1).
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
            prev.append((dn, dn.B, dn.rec_state, dn.conv_states, dn.conv_carry, dn._zero_conv0, dn._stable_state))
            dn.B = bg
            dn.reset_state()  # builds rec_state [bg,Nv,Dk,Dv], conv_states[*] [1,bg,D], carry, zero0
            dn._stable_state = True  # forward_prefill_batched writes state in place under this flag
        return prev

    def _assemble_groups_gdn_dev(self, group_rec_dev, group_conv_dev):
        """Assemble per-GROUP GDN states (each already batched [bg,...] on device, from
        forward_prefill_batched) into the full [B,...] batched decode buffers via device-side
        concat — no host round-trip. rec: concat groups along dim 0 -> [B,Nv,Dk,Dv]; conv_states[m]:
        concat groups along dim 1 -> [1,B,D]. The batched GDN bindings MUST already be rebound
        (writes in place under _stable_state). Row u == user u because groups are contiguous
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
        sequential B=1 forwards. Within a group the GDN layers run BATCHED (forward_prefill_batched,
        per-row valid_len masking — bit-exact per user, see test_gdn_tp_batched_prefill) and the
        full-attention layers run PER-USER (attention prefill is B=1 only). Groups of <=4 respect the
        GDN kernel cap BH=B*Nv_tp<=32. Numerically the batched GDN + per-user attention is the same
        math as prefill_paged_peruser, so per-user output (incl. DIFFERENT prompts/lengths) is
        unchanged; it just amortizes the underutilized GDN over the group.

        token_ids_list: list of B torch.Tensor [1, T_u] (lengths may differ).
        page_table:      torch.Tensor [B, blocks_per_user] int32 (row u = user u's blocks).
        valid_lens:      optional list of B ints; defaults to each T_u. Every valid_len MUST be <=
                         the derived bucket (a single GDN chunk-set); callers route longer prompts
                         to the chunked path.
        Returns:         list of B ttnn logits [1, 1, vocab] (prefill_paged_peruser contract).
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
                    # I-3 A3 (single device) freed the unsplit weight: the one-row x_last runs the A3 chunks.
                    if self._a3_lm_chunks is not None:
                        lg = self._lm_head(x_last)
                    else:
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

        For T > 1024: uses paged prefill (paged_fill_cache + chunked_sdpa)
        via prefill_layer_chunked with page_table.
        For T <= 1024: uses direct concat prefill + post-hoc paged cache fill.

        Args:
            token_ids: torch.Tensor [B, T] token IDs
            page_table: torch.Tensor or ttnn.Tensor [B, max_blocks_per_seq] int32
        Returns:
            logits: ttnn.Tensor [B, 1, vocab_size]
        """
        if self.num_devices > 1:
            return self._prefill_paged_tp(token_ids, page_table, valid_len=valid_len, vision_tokens=vision_tokens)

        B, T = token_ids.shape
        # Stage the per-request RoPE (M-RoPE for multimodal, 1D for text) before any cos/sin seam;
        # prefill_layer_chunked (T>1024) inherits the staged table.
        self._build_request_rope(token_ids[:, :valid_len] if valid_len else token_ids, vision_tokens)
        # Keep page_table as torch.Tensor for CPU slicing in prefill_layer_chunked.
        page_table_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        self.reset_state(batch_size=B)

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

        # Fuse DeltaNet conv states for decode.
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                if dn.fused_conv_state is None and dn.conv_state_q is not None:
                    dn.fused_conv_state = ttnn.concat([dn.conv_state_q, dn.conv_state_k, dn.conv_state_v], dim=2)
                    dn.fused_conv_state = ttnn.to_layout(dn.fused_conv_state, ttnn.TILE_LAYOUT)

        # Copy DeltaNet state into external pre-allocated buffers.
        if self._deltanet_external_states is not None:
            dn_idx = 0
            for layer in self.layers:
                if not layer.is_full_attention:
                    dn = layer.attention
                    ext_rec, ext_conv = self._deltanet_external_states[dn_idx]
                    ttnn.copy(dn.recurrent_state, ext_rec)
                    if dn.fused_conv_state is not None:
                        ttnn.copy(dn.fused_conv_state, ext_conv)
                    dn_idx += 1

        self._gdn_refresh_conv_hist()
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
        """Build HOST decode inputs: (tokens_tt, cur_pos_tt, rope_packed, page_table_tt).

        M2 REPACK_LATE: runs a pending conv-history repack first (always outside a trace capture: the
        host inputs built here are copied to the device before the decode forward / trace replay)."""
        from models.demos.blackhole.qwen36.tt.generator_interface import pack_rope_host

        self._m2_flush_pending_repack()

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
            # I-1 D4A (QWEN36_I1_D4A, B=1): row-replicated [1,32,rope_head_dim] cos/sin (packed
            # [2,32,rope_head_dim], same 4 tiles as [2,1,rope_head_dim]) for the head-major decode
            # RoPE in ttnn_gated_attention (rotary_embedding_hf on [1,1,H,64], no transposes).
            rope_rows = 32 if (B == 1 and tpc.i1_enabled("D4A")) else 1
            cos_host, sin_host = self.rope.get_cos_sin_host(
                int(rope_pos_vec[0]), rows=rope_rows
            )  # HOST ttnn [1,rope_rows,rope_head_dim]
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
        # _greedy_token_out: the token ops run inside the captured decode trace; the output is a uint32 token.
        logits = self._forward_decode(tokens, cos, sin, current_pos, page_table, want_token=self._greedy_token_out)
        return logits, None

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False, is_log_probs=False):
        """Convert decode output to host torch. Host-sampling returns logits [B,S,vocab];
        on-device sampling returns sampled token ids or sampled-token log-probs.
        A uint32 tt_out (set_greedy_token_output) holds greedy token ids: returned as [B].
        """
        if self._greedy_token_out and not (is_tokens or is_log_probs) and tt_out.dtype == ttnn.uint32:
            return ttnn.to_torch(tt_out).reshape(-1)[:B]
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
        self._m2_flush_pending_repack()  # M2 REPACK_LATE (no-op unless a repack is pending)
        saved = []
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                saved.append(
                    {
                        "recurrent": ttnn.to_torch(dn.recurrent_state),
                        "conv": ttnn.to_torch(dn.fused_conv_state) if dn.fused_conv_state is not None else None,
                    }
                )
        return saved

    def _restore_deltanet_states(self, saved_states, device):
        """Restore GDN state via ttnn.copy (preserves trace-baked buffer addresses)."""
        self._m2_flush_pending_repack()  # M2 REPACK_LATE (no-op unless a repack is pending)
        idx = 0
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                saved = saved_states[idx]
                restored = ttnn.from_torch(
                    saved["recurrent"],
                    # FP32 state is kept FP32 by the fused GDN decode (QWEN36_GDN_DECODE_FUSED=2); else BF16.
                    dtype=dn.recurrent_state.dtype if getattr(dn, "_decode_fused", False) else ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
                ttnn.copy(restored, dn.recurrent_state)
                ttnn.deallocate(restored)
                if saved["conv"] is not None:
                    restored_conv = ttnn.from_torch(
                        saved["conv"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    )
                    ttnn.copy(restored_conv, dn.fused_conv_state)
                    ttnn.deallocate(restored_conv)
                    dn._restore_split_conv_from_fused()
                idx += 1
        # Fused GDN decode: the decode-trace capture advanced conv_hist; rebuild it from the restored
        # fused_conv_state (the op never writes fused_conv_state). No-op when the flag is off.
        self._gdn_refresh_conv_hist()
