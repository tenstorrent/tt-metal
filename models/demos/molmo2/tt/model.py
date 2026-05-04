# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2-8B full TTNN model assembly.

Assembles:
  - Dual embedding (wte.embedding + wte.new_embedding)
  - 36 text decoder blocks (attention + MLP + RMSNorm residuals)
  - Final RMSNorm (ln_f) + LM head
  - ViT encoder (25 blocks, data-parallel)
  - Image pooling 2D cross-attention adapter
  - Image projector (SwiGLU)
  - Image feature injection (additive, at image_patch_id positions)
  - Bidirectional + causal prefill mask

T3K parallelization:
  - Text decoder: tensor-parallel (ShardTensor2dMesh)
  - ViT, pooling, projector, embedding, lm_head: replicated
"""

import math

import torch

import ttnn

# ---- Prefill sequence-length bucketing ----
# Sequences are padded to the nearest bucket so the same JIT-compiled kernel
# (and optionally the same trace) can be reused across similar-length inputs.
# Extended to 32768 to cover long video prompts (384 frames ≈ S~34k → pads to 65536,
# but typical 105-test videos hit at most ~16k tokens).
PREFILL_BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
# Stops at 32768: model max_seq_len=36864, so anything above pads to 65536
# which is unreachable. 32768 covers all realistic video prompt lengths.
# Trace-capture is capped at 4096: larger traces permanently reserve DRAM that
# OOMs alongside the vision backbone weights.
# JIT compilation (use_trace=False) has no such constraint.
DEFAULT_WARMUP_BUCKETS = [128, 256, 512, 1024, 2048, 4096]
MAX_TRACE_BUCKET = 4096


def get_padded_prefill_len(seq_len: int) -> int:
    """Return the smallest bucket >= seq_len, or next power-of-2 for very long seqs."""
    for b in PREFILL_BUCKETS:
        if seq_len <= b:
            return b
    return 2 ** math.ceil(math.log2(seq_len))


from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.demos.molmo2.tt.attention import TtMolmo2TextAttention
from models.demos.molmo2.tt.image_pooling import TtMolmo2ImagePooling2D
from models.demos.molmo2.tt.image_projector import TtMolmo2ImageProjector
from models.demos.molmo2.tt.mlp import TtMolmo2TextMLP
from models.demos.molmo2.tt.prefill_mask import build_causal_mask, build_molmo2_prefill_mask
from models.demos.molmo2.tt.vision_encoder import TtMolmo2ViTEncoder
from models.tt_transformers.tt.common import Mode, precompute_freqs


class TtMolmo2DecoderBlock(LightweightModule):
    """Single Molmo2 text decoder block: pre-norm attn + pre-norm MLP."""

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
    ):
        super().__init__()

        layer_name = f"model.transformer.blocks.{layer_num}"

        self.attention_norm = RMSNorm(
            device=mesh_device,
            dim=configuration.dim,
            eps=configuration.norm_eps,
            state_dict=state_dict,
            state_dict_prefix=None,
            weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key=f"{layer_name}.attn_norm",
            is_distributed=False,
        )
        self.ff_norm = RMSNorm(
            device=mesh_device,
            dim=configuration.dim,
            eps=configuration.norm_eps,
            state_dict=state_dict,
            state_dict_prefix=None,
            weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key=f"{layer_name}.ff_norm",
            is_distributed=False,
        )
        self.attention = TtMolmo2TextAttention(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=configuration,
        )
        self.feed_forward = TtMolmo2TextMLP(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            configuration=configuration,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos=None,
        rot_mats=None,
        user_id: int = 0,
        mode: str = "prefill",
        attn_mask=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        skip_cfg = ttnn.DRAM_MEMORY_CONFIG

        # Attention sub-block
        attn_in = self.attention_norm(x, mode=Mode.PREFILL if mode == "prefill" else Mode.DECODE)
        attn_out = self.attention.forward(
            attn_in,
            current_pos=current_pos,
            rot_mats=rot_mats,
            user_id=user_id,
            mode=mode,
            attn_mask=attn_mask,
            kv_cache=kv_cache,
        )
        ttnn.deallocate(attn_in)
        h = ttnn.add(x, attn_out, memory_config=skip_cfg, dtype=ttnn.bfloat16)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(x)

        # MLP sub-block
        ff_in = self.ff_norm(h, mode=Mode.PREFILL if mode == "prefill" else Mode.DECODE)
        ff_out = self.feed_forward.forward(ff_in, mode=Mode.PREFILL if mode == "prefill" else Mode.DECODE)
        ttnn.deallocate(ff_in)
        out = ttnn.add(h, ff_out, memory_config=skip_cfg, dtype=ttnn.bfloat16)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        return out


class TtMolmo2Model(LightweightModule):
    """Full Molmo2-8B TTNN model."""

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        weight_cache_path,
        dtype,
        configuration,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.configuration = configuration
        self.dtype = dtype

        # ------------------------------------------------------------------ #
        # Dual embedding: wte.embedding + wte.new_embedding → concat
        # ------------------------------------------------------------------ #
        emb_full = torch.cat(
            [
                state_dict["model.transformer.wte.embedding"],  # [151936, 4096]
                state_dict["model.transformer.wte.new_embedding"],  # [128, 4096]
            ],
            dim=0,
        ).unsqueeze(
            0
        )  # [1, 152064, 4096]

        self.embedding = ttnn.as_tensor(
            emb_full,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # ------------------------------------------------------------------ #
        # RoPE setup: HF-format cos/sin (concatenated halves, rotate_half style)
        # Shape: [1, 1, max_seq_len, head_dim] — used by ttnn.experimental.rotary_embedding
        # ------------------------------------------------------------------ #
        cos_raw, sin_raw = precompute_freqs(
            configuration.head_dim,
            configuration.max_seq_len * 2,
            configuration.rope_theta,
            None,
            None,
        )
        # HF concatenated-halves format: [c0,...,c63, c0,...,c63] per row
        cos_hf = torch.cat(
            [cos_raw[: configuration.max_seq_len], cos_raw[: configuration.max_seq_len]], dim=-1
        )  # [S, head_dim]
        sin_hf = torch.cat([sin_raw[: configuration.max_seq_len], sin_raw[: configuration.max_seq_len]], dim=-1)
        # Store as CPU tensors for slicing at inference time
        self._cos_hf = cos_hf  # [max_seq_len, head_dim]
        self._sin_hf = sin_hf

        # transformation_mats kept for API compatibility (not used with rotary_embedding)
        self.transformation_mats = {"prefill": None, "decode": None}

        # ------------------------------------------------------------------ #
        # Text decoder blocks
        # ------------------------------------------------------------------ #
        self.layers = [
            TtMolmo2DecoderBlock(
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                dtype=dtype,
                transformation_mats=self.transformation_mats,
                configuration=configuration,
            )
            for i in range(configuration.n_layers)
        ]

        # ------------------------------------------------------------------ #
        # Final norm + LM head
        # ------------------------------------------------------------------ #
        self.ln_f = RMSNorm(
            device=mesh_device,
            dim=configuration.dim,
            eps=configuration.norm_eps,
            state_dict=state_dict,
            state_dict_prefix=None,
            weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key="model.transformer.ln_f",
            is_distributed=False,
        )

        lm_head_w = state_dict["lm_head.weight"].T.unsqueeze(0).unsqueeze(0)  # [1, 1, 4096, 152064]
        self.lm_head = ttnn.as_tensor(
            lm_head_w,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # ------------------------------------------------------------------ #
        # Vision backbone
        # ------------------------------------------------------------------ #
        self.vit_encoder = TtMolmo2ViTEncoder(
            mesh_device=mesh_device,
            state_dict=state_dict,
            vit_cfg=configuration,
            weight_cache_path=weight_cache_path,
        )
        self.image_pooling = TtMolmo2ImagePooling2D(
            mesh_device=mesh_device,
            state_dict=state_dict,
            cfg=configuration,
            weight_cache_path=weight_cache_path,
        )
        self.image_projector = TtMolmo2ImageProjector(
            mesh_device=mesh_device,
            state_dict=state_dict,
            cfg=configuration,
            weight_cache_path=weight_cache_path,
        )

        # ------------------------------------------------------------------ #
        # Causal mask cache — built once at init for each bucket ≤ 8192.
        # Avoids the 32 MB H2D upload + ttnn.tril per forward_prefill call.
        # Buckets > 8192 (512 MB+ each) are built dynamically.
        # ------------------------------------------------------------------ #
        _MAX_CACHED_MASK_S = 8192
        self._causal_masks: dict = {}
        for s in [b for b in PREFILL_BUCKETS if b <= _MAX_CACHED_MASK_S]:
            self._causal_masks[s] = build_causal_mask(s, mesh_device)

        # ------------------------------------------------------------------ #
        # Prefill trace state — populated by warmup_all_buckets()
        # Key: bucket_size (int); Value: (trace_id, trace_tensors_dict, trace_logits)
        # ------------------------------------------------------------------ #
        self._prefill_traces = {}

        # ------------------------------------------------------------------ #
        # Decode trace state (populated lazily on first generate() call)
        # ------------------------------------------------------------------ #
        self._decode_trace_id = None
        self._decode_trace_tensors = None  # dict: tok_id, cur_pos, cos, sin
        self._decode_trace_output = None  # stable logits buffer (read after execute_trace)

        # CPU decode cache: block weights pulled from device once and reused.
        # Populated lazily; eliminates the ~20s block-weight pull overhead on
        # subsequent generate() calls.
        self._cpu_block_weights = None
        self._cpu_emb = None
        self._cpu_ln_f_w = None
        self._cpu_lm_head_w = None

    # ------------------------------------------------------------------ #
    # Rope helpers
    # ------------------------------------------------------------------ #

    def _get_rot_mats_prefill(self, seq_len: int):
        """Get HF-format cos/sin for prefill via ttnn.experimental.rotary_embedding.

        Returns the full cos/sin matrix [1, 1, max_seq_len, head_dim].
        rotary_embedding will apply positions 0..seq_len-1 and pad to tile size;
        the attention forward slices the output back to seq_len.
        """
        # Upload full matrix once per call; cache on device for repeated use
        if not hasattr(self, "_cos_hf_tt") or self._cos_hf_tt is None:
            cos_4d = self._cos_hf.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq, head_dim]
            sin_4d = self._sin_hf.unsqueeze(0).unsqueeze(0)

            def _tt(t):
                return ttnn.from_torch(
                    t.to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )

            self._cos_hf_tt = _tt(cos_4d)
            self._sin_hf_tt = _tt(sin_4d)
        return [self._cos_hf_tt, self._sin_hf_tt]

    def _get_rot_mats_decode(self, position: int):
        """Get HF-format cos/sin [1, 1, 1, head_dim] for single-token decode."""
        cos_1 = self._cos_hf[position : position + 1].unsqueeze(0).unsqueeze(0)  # [1, 1, 1, head_dim]
        sin_1 = self._sin_hf[position : position + 1].unsqueeze(0).unsqueeze(0)

        def _tt(t):
            return ttnn.from_torch(
                t.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        return [_tt(cos_1), _tt(sin_1)]

    def reset_kv_cache(self, user_id: int = 0):
        """Zero-fill the KV cache for the given user slot.

        Call this between inference requests to ensure each new input starts
        from a clean state and avoids stale data from previous prefills.
        The decode traces are NOT reset — they are safely reused across calls
        because the stable input buffers (tok_id, cur_pos, cos, sin) are
        always updated before each execute_trace.
        """
        cfg = self.configuration
        zeros_k = torch.zeros(1, cfg.n_local_kv_heads, cfg.max_seq_len, cfg.head_dim, dtype=torch.bfloat16)
        zeros_k_tt = ttnn.from_torch(
            zeros_k,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        for layer in self.layers:
            attn = layer.attention
            ttnn.fill_cache(attn.layer_past[0], zeros_k_tt, user_id)
            ttnn.fill_cache(attn.layer_past[1], zeros_k_tt, user_id)
        ttnn.deallocate(zeros_k_tt)

    # ------------------------------------------------------------------ #
    # Prefill trace helpers
    # ------------------------------------------------------------------ #

    def _allocate_prefill_trace_tensors(self, bucket_size: int) -> dict:
        """Pre-allocate the stable hidden-state input buffer for a prefill trace."""
        cfg = self.configuration
        hidden = ttnn.from_torch(
            torch.zeros(1, 1, bucket_size, cfg.dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return {"hidden": hidden, "bucket_size": bucket_size}

    def _capture_prefill_trace(self, tt: dict) -> tuple:
        """Warm-up + trace capture for a text-only (causal) prefill trace.

        Uses is_causal=True (no explicit mask tensor). Video/image inputs use
        the eager path — is_causal=False with explicit mask SDPA cannot coexist
        with the decode trace: execute_trace hangs when both are active.

        The stable hidden buffer is cloned at the start of _decoder_fwd so
        layer.forward's deallocate(x) frees the clone, not tt["hidden"]. This
        eliminates deallocation suppression during the warmup pass.
        trace_capture_run_begin() is still required during begin/end_trace_capture
        to prevent TT_FATAL from intermediate tensor deallocations.
        """
        from models.demos.molmo2.tt.trace_capture_utils import trace_capture_run_begin, trace_capture_run_end

        bucket_size = tt["bucket_size"]
        rot_mats = self._get_rot_mats_prefill(bucket_size)

        def _decoder_fwd(hidden):
            x = ttnn.clone(hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for layer in self.layers:
                x = layer.forward(x, rot_mats=rot_mats, mode="prefill", attn_mask=None)
            return self.ln_f(x, mode=Mode.PREFILL)  # [1, 1, S, 4096]

        # ---- Warm-up: clone protects hidden, no suppression needed ----
        logits_warmup = _decoder_fwd(tt["hidden"])
        ttnn.synchronize_device(self.mesh_device)
        ttnn.deallocate(logits_warmup)

        # ---- Trace capture: suppression required (TT_FATAL on any deallocate) ----
        tok = trace_capture_run_begin()
        try:
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            trace_logits = _decoder_fwd(tt["hidden"])
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        finally:
            trace_capture_run_end(tok)

        return trace_id, trace_logits

    def warmup_all_buckets(self, bucket_sizes=None, use_trace: bool = False):
        """Pre-warm (and optionally trace-capture) all prefill bucket sizes.

        Two-phase approach (required by TTNN):
          Phase 1 — allocate ALL trace tensors before any trace capture.
          Phase 2 — capture traces for each bucket.

        Args:
            bucket_sizes: list of sequence lengths to pre-warm (default PREFILL_BUCKETS)
            use_trace: if True, capture prefill traces; if False, just warm up kernels
        """
        if bucket_sizes is None:
            bucket_sizes = DEFAULT_WARMUP_BUCKETS

        if use_trace:
            # Trace capture only supports buckets up to MAX_TRACE_BUCKET (larger traces
            # permanently reserve DRAM that OOMs alongside vision backbone weights).
            valid = [b for b in bucket_sizes if b <= self.configuration.max_seq_len and b <= MAX_TRACE_BUCKET]
        else:
            # JIT compilation: no trace-memory constraint — warm all requested buckets.
            valid = [b for b in bucket_sizes if b <= self.configuration.max_seq_len]
        print(f"[prefill warmup] buckets={valid} use_trace={use_trace}", flush=True)

        if use_trace:
            # Phase 1: allocate ALL trace tensors first (before any trace capture)
            all_tt = {b: self._allocate_prefill_trace_tensors(b) for b in valid}
            # Phase 2: capture traces for each bucket
            for b in valid:
                print(f"  bucket {b} ...", flush=True)
                trace_id, trace_logits = self._capture_prefill_trace(all_tt[b])
                self._prefill_traces[b] = (trace_id, all_tt[b], trace_logits)
        else:
            # Run a text-only forward_prefill with dummy ids at each bucket size to
            # trigger JIT kernel compilation and populate the on-disk cache.
            # Subsequent calls at the same S_pad hit the cache and are instant.
            cfg = self.configuration
            for b in valid:
                print(f"  JIT warmup bucket {b} ...", flush=True)
                dummy_ids = torch.zeros(1, b, dtype=torch.long)
                _ = self.forward_prefill(input_ids=dummy_ids, pixel_values=None, user_id=0)
                print(f"  bucket {b} done", flush=True)

        n = len(self._prefill_traces)
        print(f"[prefill warmup] done — {n} traces, {len(valid) - n} JIT-compiled", flush=True)

    def warmup_vision_compile(self) -> None:
        """Pre-compile all vision (ViT + pooling + projector) JIT kernels.

        Runs dummy forward passes for every possible last-chunk size so that no
        cold compilation happens at inference time regardless of the input video.

        Because _run_chunked_ttnn_pooling now pads every chunk to exactly
        _POOL_CHUNK_WINDOWS windows, the pooling TTNN ops always see the same
        tensor shape regardless of video length. One warmup run per k_pool value
        is sufficient to compile all pooling kernels.

        ViT encode chunks (_MAX_VIT_BATCH=8) still vary in the last batch
        (1..8 crops), so we still compile for those.
        """
        N_PATCHES = 729
        _MAX_VIT_BATCH = 8

        # ---- Video: k_pool=9 (3×3) — one run at full chunk size covers all videos ----
        K_POOL_VIDEO = 9
        n_pooled_video = self._POOL_CHUNK_WINDOWS  # always padded to this
        print(f"[vision warmup] video k_pool=9, N_pooled={n_pooled_video} ...", flush=True)
        dummy_pv = torch.zeros(1, _MAX_VIT_BATCH, N_PATCHES, 588)
        dummy_idx = torch.zeros(1, n_pooled_video, K_POOL_VIDEO, dtype=torch.long)
        _ = self.run_vision_backbone(dummy_pv, dummy_idx)
        print("[vision warmup] video done", flush=True)

        # ---- Image: k_pool=4 (2×2) — same, one run at full chunk size ----
        K_POOL_IMAGE = 4
        n_pooled_image = self._POOL_CHUNK_WINDOWS
        print(f"[vision warmup] image k_pool=4, N_pooled={n_pooled_image} ...", flush=True)
        dummy_pv = torch.zeros(1, _MAX_VIT_BATCH, N_PATCHES, 588)
        dummy_idx = torch.zeros(1, n_pooled_image, K_POOL_IMAGE, dtype=torch.long)
        _ = self.run_vision_backbone(dummy_pv, dummy_idx)
        print("[vision warmup] image done", flush=True)

        print("[vision warmup] done", flush=True)

    # ------------------------------------------------------------------ #
    # Decode trace helpers
    # ------------------------------------------------------------------ #

    def _allocate_decode_trace_tensors(self):
        """Pre-allocate stable on-device buffers for trace capture/replay.

        ALL buffers are allocated here, before trace capture, so that no
        device allocation is needed during execute_trace (which would corrupt
        trace-reserved memory regions).
        """
        cfg = self.configuration
        head_dim = cfg.head_dim

        def _tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t,
                dtype=dtype,
                layout=layout,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        # Token ID input for embedding (inside trace) — [1, 1, 1] uint32
        tok_id = _tt(torch.zeros(1, 1, 1, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        # Current decode position — [1] int32
        cur_pos = _tt(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        # RoPE cos/sin for current position — [1, 1, 1, head_dim]
        cos_buf = _tt(torch.zeros(1, 1, 1, head_dim, dtype=torch.bfloat16))
        sin_buf = _tt(torch.zeros(1, 1, 1, head_dim, dtype=torch.bfloat16))
        return {"tok_id": tok_id, "cur_pos": cur_pos, "cos": cos_buf, "sin": sin_buf}

    def _capture_decode_trace(self, tt, prefill_seq_len: int):
        """Warm-up + trace capture for single-token decode.

        Embedding is computed INSIDE the trace so that layer.forward's
        ttnn.deallocate(x) never touches the stable tok_id input buffer.
        The trace reads: tok_id, cur_pos, cos, sin — all pre-allocated stable
        device buffers that are updated before each execute_trace call.

        Args:
            tt: dict from _allocate_decode_trace_tensors()
            prefill_seq_len: position after prefill (first decode position)

        Returns:
            (trace_id, trace_logits_output) — logits is the stable trace output.
        """
        from models.demos.molmo2.tt.trace_capture_utils import trace_capture_run_begin, trace_capture_run_end

        cfg = self.configuration

        def _upload(t_cpu, dtype, layout, device_buf):
            """Upload a CPU tensor into a pre-allocated device buffer in-place."""
            staging = ttnn.from_torch(
                t_cpu,
                dtype=dtype,
                layout=layout,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            ttnn.copy(staging, device_buf)
            ttnn.deallocate(staging)

        # Seed stable buffers with valid initial values (pre-trace — allocation is safe here)
        _upload(torch.zeros(1, 1, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, tt["tok_id"])
        _upload(torch.tensor([prefill_seq_len], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, tt["cur_pos"])
        cos_p = self._cos_hf[prefill_seq_len : prefill_seq_len + 1].unsqueeze(0).unsqueeze(0).bfloat16()
        sin_p = self._sin_hf[prefill_seq_len : prefill_seq_len + 1].unsqueeze(0).unsqueeze(0).bfloat16()
        _upload(cos_p, ttnn.bfloat16, ttnn.TILE_LAYOUT, tt["cos"])
        _upload(sin_p, ttnn.bfloat16, ttnn.TILE_LAYOUT, tt["sin"])

        def _forward_decode(tok_id_t, cur_pos_t, cos_t, sin_t):
            """Single decode step: embedding → 36 layers → ln_f → lm_head.

            Embedding is the FIRST op so layer.forward's ttnn.deallocate(x)
            acts on the embedding output (trace-internal), never on tok_id_t.
            """
            x = ttnn.embedding(tok_id_t, self.embedding)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.reshape(x, [1, 1, 1, cfg.dim])
            rot_mats = [cos_t, sin_t]
            for layer in self.layers:
                x = layer.forward(x, current_pos=cur_pos_t, rot_mats=rot_mats, mode="decode")
            x = self.ln_f(x, mode=Mode.PREFILL)
            return ttnn.linear(
                x,
                self.lm_head,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=cfg.compute_kernel_config_hifi2_fp16,
            )

        # ---- Warm-up (compile) pass — not traced ----
        logits_warmup = _forward_decode(tt["tok_id"], tt["cur_pos"], tt["cos"], tt["sin"])
        ttnn.deallocate(logits_warmup)

        # ---- Trace capture (all ops including embedding inside the trace) ----
        tok = trace_capture_run_begin()
        try:
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            trace_logits = _forward_decode(tt["tok_id"], tt["cur_pos"], tt["cos"], tt["sin"])
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        finally:
            trace_capture_run_end(tok)

        return trace_id, trace_logits

    def _execute_decode_trace(self, token_id: int, position: int) -> torch.Tensor:
        """Update stable input buffers and execute the captured decode trace.

        Each update allocates a small staging tensor, copies into the stable buffer
        in-place (ttnn.copy requires both to be device tensors), then frees staging.
        This follows the same pattern used by the reference demo generator.

        Returns logits [vocab_size] on CPU.
        """
        tt = self._decode_trace_tensors

        def _upload(t_cpu, dtype, layout, device_buf):
            staging = ttnn.from_torch(
                t_cpu,
                dtype=dtype,
                layout=layout,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            ttnn.copy(staging, device_buf)
            ttnn.deallocate(staging)

        # Update token ID ([1, 1, 1] uint32)
        _upload(torch.tensor([[[token_id]]], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, tt["tok_id"])
        # Update position ([1] int32)
        _upload(torch.tensor([position], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, tt["cur_pos"])
        # Update RoPE cos/sin for this position ([1, 1, 1, head_dim])
        cos_p = self._cos_hf[position : position + 1].unsqueeze(0).unsqueeze(0).bfloat16()
        sin_p = self._sin_hf[position : position + 1].unsqueeze(0).unsqueeze(0).bfloat16()
        _upload(cos_p, ttnn.bfloat16, ttnn.TILE_LAYOUT, tt["cos"])
        _upload(sin_p, ttnn.bfloat16, ttnn.TILE_LAYOUT, tt["sin"])

        # Replay the captured trace — reads from the stable buffers updated above
        ttnn.execute_trace(self.mesh_device, self._decode_trace_id, cq_id=0, blocking=True)

        # Read logits from the trace output buffer (written by lm_head inside trace)
        logits_cpu = ttnn.to_torch(ttnn.get_device_tensors(self._decode_trace_output)[0]).float()
        return logits_cpu.squeeze(0).squeeze(0).squeeze(0)  # [vocab_size]

    # ------------------------------------------------------------------ #
    # Vision backbone
    # ------------------------------------------------------------------ #

    # Number of pooling windows to process per TTNN chunk.
    # Peak per chunk (worst case 384-frame video, feat_tt=1.23 GB stays throughout):
    #   feat_tt + 3 × C × 32tile × 2304 × 2 bytes  (to_pool + query_sum + query)
    # C=4096 (≈50 frames × 81 windows): peak ≈ 6.3 GB → safe on T3K (12 GB, ~5 GB weights).
    # 30-frame video (2430 windows): 1 chunk.
    # 384-frame video (31104 windows): 8 chunks.
    _POOL_CHUNK_WINDOWS = 4096

    def _run_chunked_ttnn_pooling(
        self,
        vit_cpu: torch.Tensor,  # [B, n_crops, n_patches, 2304] float32 CPU
        pooled_patches_idx: torch.Tensor,  # [B, N_pooled, k_pool] int64 CPU
    ) -> torch.Tensor:
        """TTNN chunked image pooling. Returns [B, N_pooled, 1152] float32 CPU.

        Uploads the full ViT feature table as a 2D device embedding table once,
        then processes pooling windows in chunks of _POOL_CHUNK_WINDOWS to keep
        DRAM peak manageable at any video length.

        Key correctness properties:
          - uint32 indices: max index for 384 frames = 280k >> bfloat16 safe limit ~256
          - ROW_MAJOR reshape before attn-mask build: avoids tile-padding artefacts
            when k_pool (e.g. 9 or 4) is not a multiple of tile size 32
        """
        B, n_crops, n_patches, feat_dim = vit_cpu.shape
        N_pooled = pooled_patches_idx.shape[1]
        k_pool = pooled_patches_idx.shape[2]
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)

        # H2D: full feature table as 2D ROW_MAJOR embedding lookup table
        feat_2d = vit_cpu.reshape(-1, feat_dim).to(torch.bfloat16)
        feat_tt = ttnn.from_torch(
            feat_2d,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

        all_chunks = []
        for start in range(0, N_pooled, self._POOL_CHUNK_WINDOWS):
            end = min(start + self._POOL_CHUNK_WINDOWS, N_pooled)
            chunk_n_out_real = end - start
            chunk_idx = pooled_patches_idx[0, start:end]  # [chunk_n_out_real, k_pool]

            # Pad the last (possibly short) chunk to _POOL_CHUNK_WINDOWS with -1 indices.
            # This keeps TTNN tensor shapes constant across all chunks → one JIT compile
            # covers every call regardless of video length.  -1 indices are clamped to 0
            # and masked out by valid_flat, so they contribute nothing to the output.
            if chunk_n_out_real < self._POOL_CHUNK_WINDOWS:
                pad = torch.full((self._POOL_CHUNK_WINDOWS - chunk_n_out_real, k_pool), -1, dtype=torch.int64)
                chunk_idx = torch.cat([chunk_idx, pad], dim=0)
            chunk_n_out = self._POOL_CHUNK_WINDOWS

            # Upload indices as uint32 — bfloat16 only represents integers up to ~256
            idx_flat = chunk_idx.reshape(-1).clamp(min=0).to(torch.int32)
            idx_tt = ttnn.from_torch(
                idx_flat.reshape(1, -1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            valid_flat = (chunk_idx.reshape(-1) >= 0).float()
            valid_tt = ttnn.from_torch(
                valid_flat.reshape(1, 1, -1, 1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

            # Gather via ttnn.embedding with global uint32 indices
            gathered = ttnn.embedding(idx_tt, feat_tt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(idx_tt)
            gathered = ttnn.reshape(gathered, [1, 1, chunk_n_out * k_pool, feat_dim])
            gathered = ttnn.mul(gathered, valid_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            to_pool = ttnn.reshape(gathered, [1, chunk_n_out, k_pool, feat_dim])
            ttnn.deallocate(gathered)

            # Masked mean query — fully on device (no D2H)
            query_sum = ttnn.sum(to_pool, dim=2, keepdim=True)
            vm = ttnn.reshape(valid_tt, [1, chunk_n_out, k_pool, 1])
            denom = ttnn.clamp(ttnn.sum(vm, dim=2, keepdim=True), min=1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(vm)
            query = ttnn.div(query_sum, denom, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(query_sum)
            ttnn.deallocate(denom)

            # Attn mask: ROW_MAJOR reshape before [N_out,1,1,k_pool] to avoid tile-padding
            # artefacts when k_pool is not tile-aligned (e.g. k_pool=9 or k_pool=4)
            valid_rm = ttnn.to_layout(valid_tt, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(valid_tt)
            valid_4d = ttnn.to_layout(ttnn.reshape(valid_rm, [chunk_n_out, 1, 1, k_pool]), ttnn.TILE_LAYOUT)
            ttnn.deallocate(valid_rm)
            zeros_buf = ttnn.zeros_like(valid_4d)
            neg_inf_buf = ttnn.from_torch(
                torch.full((1, 1, 1, 1), float("-inf"), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            threshold = ttnn.full_like(valid_4d, 0.5)
            cond = ttnn.gt(valid_4d, threshold, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attn_mask = ttnn.where(cond, zeros_buf, neg_inf_buf, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for t in (valid_4d, zeros_buf, neg_inf_buf, threshold, cond):
                ttnn.deallocate(t)

            # Cross-attention pooling (TP column-parallel QKV, row-parallel wo)
            pooled_tt = self.image_pooling.forward(query, to_pool, attn_mask)
            ttnn.deallocate(query)
            ttnn.deallocate(to_pool)
            ttnn.deallocate(attn_mask)

            chunk_cpu = ttnn.to_torch(ttnn.get_device_tensors(pooled_tt)[0]).float()
            ttnn.deallocate(pooled_tt)
            # Trim padding: keep only the real windows (first chunk_n_out_real rows)
            all_chunks.append(chunk_cpu.squeeze(0).squeeze(1)[:chunk_n_out_real])  # [chunk_n_out_real, 1152]

        ttnn.deallocate(feat_tt)
        pooled_all = torch.cat(all_chunks, dim=0).unsqueeze(0)  # [1, N_pooled, 1152]
        return pooled_all

    def run_vision_backbone(
        self,
        pixel_values: torch.Tensor,  # [B, n_crops, 729, 588] CPU
        pooled_patches_idx: torch.Tensor,  # [B, N_pooled, pool_window] CPU
    ) -> torch.Tensor:
        """Run full vision path and return [N_valid, 4096] CPU image features."""
        B, n_crops, n_patches, px_dim = pixel_values.shape

        # DP ViT: shard crops across devices (1 crop/device), no CCL per block.
        # Chunk size = num_devices so each chunk is evenly sharded. Last chunk is
        # padded to num_devices with zeros and trimmed after forward.
        n_crops_flat = B * n_crops
        pv_4d = pixel_values.reshape(n_crops_flat, 1, n_patches, px_dim).to(torch.bfloat16)

        num_devices = self.mesh_device.get_num_devices()
        _MAX_VIT_BATCH = num_devices  # 1 crop per device per forward
        vit_chunks = []
        for start in range(0, n_crops_flat, _MAX_VIT_BATCH):
            chunk_cpu = pv_4d[start : start + _MAX_VIT_BATCH]
            real_n = chunk_cpu.shape[0]
            # Pad last chunk to num_devices so ShardTensorToMesh(dim=0) works evenly
            if real_n < num_devices:
                pad = torch.zeros(num_devices - real_n, 1, n_patches, px_dim, dtype=torch.bfloat16)
                chunk_cpu = torch.cat([chunk_cpu, pad], dim=0)
            chunk_ttnn = ttnn.from_torch(
                chunk_cpu,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            )
            # n_crops_per_device=1: each device gets its own crop, no pos_emb tiling
            feat_ttnn = self.vit_encoder.forward(chunk_ttnn, n_crops_per_device=1)
            ttnn.deallocate(chunk_ttnn)
            # Collect crop features from all devices (each device has 1 crop)
            feat_parts = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(feat_ttnn)]
            ttnn.deallocate(feat_ttnn)
            feat_chunk = torch.cat(feat_parts, dim=0)  # [num_devices, 1, 729, 2304]
            vit_chunks.append(feat_chunk[:real_n].squeeze(1))  # [real_n, 729, 2304]

        vit_cpu = torch.cat(vit_chunks, dim=0).reshape(B, n_crops, n_patches, 2304)

        # TTNN chunked image pooling — avoids OOM by:
        #   (a) column-parallel TP=8 reduces per-device QKV output 8×
        #   (b) window-based chunking caps peak DRAM per chunk at ~191 MB
        pooled_cpu = self._run_chunked_ttnn_pooling(vit_cpu, pooled_patches_idx)
        # [1, N_pooled, 1152]

        # Apply valid-token filter
        valid_token = (pooled_patches_idx[0] >= 0).any(dim=-1)  # [N_pooled]
        pooled_valid = pooled_cpu[0][valid_token]  # [N_valid, 1152]

        # Image projector (TTNN) → [N_valid, 4096]
        pooled_ttnn = ttnn.from_torch(
            pooled_valid.to(torch.bfloat16).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        proj_out = self.image_projector.forward(pooled_ttnn)
        ttnn.deallocate(pooled_ttnn)

        proj_cpu = ttnn.to_torch(ttnn.get_device_tensors(proj_out)[0]).float().squeeze(0).squeeze(0)
        ttnn.deallocate(proj_out)
        return proj_cpu  # [N_valid, 4096]

    # ------------------------------------------------------------------ #
    # Main forward
    # ------------------------------------------------------------------ #

    def forward_prefill(
        self,
        input_ids: torch.Tensor,  # [B, S] CPU
        pixel_values=None,  # [B, n_crops, 729, 588] CPU or None
        pooled_patches_idx=None,  # [B, N_pooled, pool_window] CPU or None
        token_type_ids=None,  # [B, S] CPU or None
        user_id: int = 0,
    ):
        """Prefill forward pass. Returns logits [B, S, vocab_size] on CPU."""
        B, S = input_ids.shape

        # ---- Embedding ----
        # Dual embedding via ttnn.embedding
        input_ids_ttnn = ttnn.from_torch(
            input_ids.unsqueeze(0),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        x_ttnn = ttnn.embedding(input_ids_ttnn, self.embedding)  # [B, S, 4096]
        ttnn.deallocate(input_ids_ttnn)

        # ---- Vision backbone + image feature injection ----
        # Pre-compute S_pad (power-of-2) so vision path uses the same padded size as
        # the eager path below — ensures ttnn.add and concat use bucket-aligned shapes
        # and don't trigger per-S JIT recompilation at inference time.
        if S <= 8192:
            _S_pad_early = max(256, 1 << math.ceil(math.log2(S)) if S > 1 else 256)
        else:
            _S_pad_early = ((S + 255) // 256) * 256

        if pixel_values is not None:
            image_features = self.run_vision_backbone(pixel_values, pooled_patches_idx)

            # Additive injection via dense delta + ttnn.add — no D2H of embedding.
            # Build a zero tensor, write image features at patch positions (CPU, cheap
            # since we write into zeros not read the embedding), H2D, then add on device.
            # Use _S_pad_early (padded to power-of-2 bucket) so all TTNN ops see the same
            # tensor shape regardless of actual S, avoiding per-S JIT recompilation.
            H = self.configuration.dim
            is_patch_flat = input_ids.view(-1) == self.configuration.image_patch_id
            # Keep delta in float32 to match original precision: old code did
            # float32 add (embedding promoted to f32, features in f32) then
            # truncated to bf16. ttnn.add(bf16, f32) performs in f32 → bf16.
            delta = torch.zeros(1, 1, _S_pad_early, H, dtype=torch.float32)
            delta.view(-1, H)[:S][is_patch_flat] = image_features  # float32, no conversion
            delta_ttnn = ttnn.from_torch(
                delta,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            x_ttnn = ttnn.to_layout(x_ttnn, ttnn.TILE_LAYOUT)
            x_ttnn = ttnn.reshape(x_ttnn, [1, 1, S, H])
            if _S_pad_early > S:
                x_pad = ttnn.from_torch(
                    torch.zeros(1, 1, _S_pad_early - S, H, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                x_ttnn = ttnn.concat([x_ttnn, x_pad], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(x_pad)
            x_ttnn = ttnn.add(x_ttnn, delta_ttnn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(delta_ttnn)
        else:
            # Text-only: layout + reshape here (vision path already did both above)
            x_ttnn = ttnn.to_layout(x_ttnn, ttnn.TILE_LAYOUT)
            x_ttnn = ttnn.reshape(x_ttnn, [1, 1, S, self.configuration.dim])

        # ---- Route to trace or eager ----
        padded_S = get_padded_prefill_len(S)

        # ---- Prefill trace (text-only / causal-only) ----
        # Video/image inputs (token_type_ids≠None) use eager path — is_causal=False
        # SDPA with explicit mask cannot coexist with the decode trace: execute_trace
        # hangs when both are captured. This matches the other branch's design.
        if padded_S in self._prefill_traces and token_type_ids is None:
            trace_id, trace_tt, trace_x_norm = self._prefill_traces[padded_S]

            # ---- Update stable hidden and execute ----
            if padded_S > S:
                pad_t = ttnn.from_torch(
                    torch.zeros(1, 1, padded_S - S, self.configuration.dim, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                x_padded = ttnn.concat([x_ttnn, pad_t], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(pad_t)
                ttnn.deallocate(x_ttnn)
            else:
                x_padded = x_ttnn
            ttnn.copy(x_padded, trace_tt["hidden"])
            ttnn.deallocate(x_padded)
            ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=True)
            # Slice last real token on device — avoids D2H of the full [1,1,padded_S,4096] tensor
            x_last_ttnn = ttnn.slice(
                trace_x_norm,
                (0, 0, S - 1, 0),
                (1, 1, S, self.configuration.dim),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            logits = ttnn.linear(
                x_last_ttnn,
                self.lm_head,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.configuration.compute_kernel_config_hifi2_fp16,
            )
            ttnn.deallocate(x_last_ttnn)
            logits_cpu = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).float().squeeze(0)
            ttnn.deallocate(logits)
            return logits_cpu  # [1, 1, vocab_size]

        # ---- Eager path: pad S to eliminate SDPA partial-tile hangs ----
        # Strategy:
        #   S ≤ 8192: power-of-2 padding → Q-tiles ∈ {8,16,32} — all confirmed safe.
        #   S > 8192: chunk-multiple (×256) padding → minimal DRAM footprint.
        #             Power-of-2 would require S_pad=65536 for S=34395, making the
        #             embedding tensor 536 MB/device which OOMs alongside 12 GB weights.
        #             chunk-multiple gives S_pad=34560 (283 MB/device, fits).
        # In both cases S_pad % 256 == 0 → no partial Q-tiles.
        # Vision path already padded x_ttnn to _S_pad_early (same formula), so skip here.
        S_pad = _S_pad_early
        pad_len = 0 if pixel_values is not None else S_pad - S

        if pad_len > 0:
            # Pad embedding output: [1,1,S,dim] → [1,1,S_pad,dim]
            x_pad = ttnn.from_torch(
                torch.zeros(1, 1, pad_len, self.configuration.dim, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            x_ttnn = ttnn.concat([x_ttnn, x_pad], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(x_pad)

        attn_mask = None
        if token_type_ids is not None:
            # Build mask for S_pad; padding columns/rows are −∞ (attend to nothing)
            if pad_len > 0:
                tti_padded = torch.cat([token_type_ids.long(), torch.zeros(B, pad_len, dtype=torch.long)], dim=1)
            else:
                tti_padded = token_type_ids.long()
            attn_mask = build_molmo2_prefill_mask(
                S_pad,
                tti_padded,
                self.mesh_device,
                dtype=ttnn.bfloat16,
                causal_cache=self._causal_masks.get(S_pad),
            )

        rot_mats = self._get_rot_mats_prefill(S_pad)

        for layer in self.layers:
            x_ttnn = layer.forward(
                x_ttnn,
                rot_mats=rot_mats,
                user_id=user_id,
                mode="prefill",
                attn_mask=attn_mask,
            )

        if attn_mask is not None:
            ttnn.deallocate(attn_mask)

        x_ttnn = self.ln_f(x_ttnn, mode=Mode.PREFILL)

        # Slice last real token on device — avoids D2H of the full [1,1,S_pad,4096] tensor.
        # lm_head on all S_pad tokens would OOM (e.g. 8192×152064×2 = 2.5 GB per device).
        x_last_ttnn = ttnn.slice(
            x_ttnn,
            (0, 0, S - 1, 0),
            (1, 1, S, self.configuration.dim),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x_ttnn)
        logits = ttnn.linear(
            x_last_ttnn,
            self.lm_head,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.configuration.compute_kernel_config_hifi2_fp16,
        )
        ttnn.deallocate(x_last_ttnn)

        logits_cpu = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).float().squeeze(0)
        ttnn.deallocate(logits)
        return logits_cpu  # [1, 1, vocab_size] — last token only

    def forward_decode_step(self, new_token_id: int, current_pos: int) -> torch.Tensor:
        """Single-token decode using the filled KV cache.

        Args:
            new_token_id: integer token id of the newly generated token
            current_pos: position in the sequence to write K/V cache to

        Returns:
            logits [1, vocab_size] CPU tensor for sampling the next token
        """
        # ---- Embedding for the single new token ----
        tok = torch.tensor([[new_token_id]], dtype=torch.int32)
        tok_ttnn = ttnn.from_torch(
            tok.unsqueeze(0),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # [1, 1, 4096]
        x_ttnn = ttnn.embedding(tok_ttnn, self.embedding)
        ttnn.deallocate(tok_ttnn)
        x_ttnn = ttnn.to_layout(x_ttnn, ttnn.TILE_LAYOUT)
        x_ttnn = ttnn.reshape(x_ttnn, [1, 1, 1, self.configuration.dim])

        # ---- Decode RoPE mats for current position ----
        rot_mats = self._get_rot_mats_decode(current_pos)

        # ---- Current position tensor (TTNN INT32) for paged_update_cache ----
        cur_pos_ttnn = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # ---- Decoder blocks (decode mode uses KV cache) ----
        for layer in self.layers:
            x_ttnn = layer.forward(
                x_ttnn,
                current_pos=cur_pos_ttnn,
                rot_mats=rot_mats,
                mode="decode",
            )
        ttnn.deallocate(cur_pos_ttnn)

        # ---- Final norm + LM head ----
        x_ttnn = self.ln_f(x_ttnn, mode=Mode.DECODE)
        logits = ttnn.linear(
            x_ttnn,
            self.lm_head,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.configuration.compute_kernel_config_hifi2_fp16,
        )
        ttnn.deallocate(x_ttnn)

        logits_cpu = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).float().squeeze(0).squeeze(0)
        ttnn.deallocate(logits)
        return logits_cpu  # [1, vocab_size]

    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values=None,
        pooled_patches_idx=None,
        token_type_ids=None,
        max_new_tokens: int = 128,
        eos_token_id: int = None,
        temperature: float = 1.0,
        user_id: int = 0,
    ) -> torch.Tensor:
        """Full autoregressive generate: TTNN prefill + traced TTNN decode.

        On the first call, captures a decode trace (warm-up + begin/end_trace_capture).
        Subsequent calls execute the trace via execute_trace — no host transfers per step.
        """
        B, S = input_ids.shape
        padded_S = get_padded_prefill_len(S)

        # ---- Reset KV cache for clean state before each request ----
        self.reset_kv_cache(user_id=user_id)

        # No lazy warmup here — forward_prefill runs in eager mode for new bucket sizes
        # (JIT compiles on first use). Calling warmup_all_buckets from forward_prefill
        # would cause recursion. Pre-warm explicitly via warmup_all_buckets() before tests.

        # ---- TTNN prefill (uses trace if warmed) ----
        logits = self.forward_prefill(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pooled_patches_idx=pooled_patches_idx,
            token_type_ids=token_type_ids,
            user_id=user_id,
        )
        # forward_prefill now returns [1, 1, vocab] (last real token only)
        next_token = _sample(logits[0, 0], temperature)
        generated_ids = [next_token]

        # ---- Ensure decode trace is ready ----
        if self._decode_trace_id is None:
            print("  [decode] capturing TTNN decode trace (warm-up + capture)...", flush=True)
            self._decode_trace_tensors = self._allocate_decode_trace_tensors()
            self._decode_trace_id, self._decode_trace_output = self._capture_decode_trace(self._decode_trace_tensors, S)
            print("  [decode] trace ready", flush=True)

        # ---- Traced decode loop ----
        current_pos = S
        for step in range(max_new_tokens - 1):
            if eos_token_id is not None and next_token == eos_token_id:
                break
            logits_1 = self._execute_decode_trace(next_token, current_pos)
            next_token = _sample(logits_1, temperature)
            generated_ids.append(next_token)
            current_pos += 1

        return torch.tensor(generated_ids, dtype=torch.long)

    def _build_cpu_block_weights(self):
        """Extract and reconstruct per-block CPU weights from TTNN device tensors.

        Called once; result cached in self._cpu_block_weights.
        """
        cfg = self.configuration
        q_per_dev = cfg.n_heads // cfg.num_devices * cfg.head_dim
        kv_per_dev = cfg.n_kv_heads // cfg.num_devices * cfg.head_dim
        qkv_chunk = q_per_dev + 2 * kv_per_dev

        print("  [decode] pulling block weights to CPU...", flush=True)
        block_weights = []
        for layer in self.layers:
            attn = layer.attention
            ffn = layer.feed_forward

            wqkv_full = torch.cat(
                [ttnn.to_torch(t).float().squeeze(0).squeeze(0) for t in ttnn.get_device_tensors(attn.wqkv)],
                dim=-1,
            )
            wq_cols, wk_cols, wv_cols = [], [], []
            for di in range(cfg.num_devices):
                s = di * qkv_chunk
                wq_cols.append(wqkv_full[:, s : s + q_per_dev])
                wk_cols.append(wqkv_full[:, s + q_per_dev : s + q_per_dev + kv_per_dev])
                wv_cols.append(wqkv_full[:, s + q_per_dev + kv_per_dev : s + qkv_chunk])
            att_proj_weight = torch.cat(
                [torch.cat(wq_cols, dim=-1).T, torch.cat(wk_cols, dim=-1).T, torch.cat(wv_cols, dim=-1).T],
                dim=0,
            )

            wo_cpu = ttnn.to_torch(ttnn.get_device_tensors(attn.wo)[0]).float().squeeze(0).squeeze(0)
            q_norm_w = ttnn.to_torch(ttnn.get_device_tensors(attn.q_norm.weight)[0]).float().reshape(-1)
            k_norm_w = ttnn.to_torch(ttnn.get_device_tensors(attn.k_norm.weight)[0]).float().reshape(-1)

            w1_full = ttnn.to_torch(ttnn.get_device_tensors(ffn.w1)[0]).float().squeeze(0).squeeze(0)
            w3_full = ttnn.to_torch(ttnn.get_device_tensors(ffn.w3)[0]).float().squeeze(0).squeeze(0)
            w2_full = ttnn.to_torch(ttnn.get_device_tensors(ffn.w2)[0]).float().squeeze(0).squeeze(0)

            block_weights.append(
                {
                    "attn_norm": ttnn.to_torch(ttnn.get_device_tensors(layer.attention_norm.weight)[0])
                    .float()
                    .reshape(-1),
                    "ff_norm": ttnn.to_torch(ttnn.get_device_tensors(layer.ff_norm.weight)[0]).float().reshape(-1),
                    "att_proj": att_proj_weight,
                    "attn_out": wo_cpu.T,
                    "q_norm": q_norm_w,
                    "k_norm": k_norm_w,
                    "ff_proj": torch.cat([w3_full.T, w1_full.T], dim=0),
                    "ff_out": w2_full.T,
                }
            )
        print(f"  [decode] {len(block_weights)} block weight sets ready", flush=True)
        return block_weights

    def _generate_cpu_decode(
        self,
        input_ids: torch.Tensor,
        pixel_values=None,
        pooled_patches_idx=None,
        token_type_ids=None,
        max_new_tokens: int = 128,
        eos_token_id: int = None,
        temperature: float = 1.0,
        user_id: int = 0,
    ) -> torch.Tensor:
        """Fallback: TTNN prefill + CPU reference decode. Kept for debugging."""
        import torch.nn.functional as F

        from models.demos.molmo2.reference.functional import rmsnorm

        B, S = input_ids.shape

        logits = self.forward_prefill(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pooled_patches_idx=pooled_patches_idx,
            token_type_ids=token_type_ids,
            user_id=user_id,
        )
        next_token = _sample(logits[0, -1], temperature)
        generated_ids = [next_token]

        print(f"  [cpu_decode] pulling KV cache from device (S={S})...", flush=True)
        cpu_kv = []  # list of (k_cpu, v_cpu) per layer, shape [1, 8, S, 128]

        for layer in self.layers:
            keys_dev = layer.attention.layer_past[0]  # [1, 1, max_seq, 128] per device
            vals_dev = layer.attention.layer_past[1]

            # Collect k from all 8 devices (each has 1 KV head)
            k_parts = [ttnn.to_torch(t).float()[:, :, :S, :] for t in ttnn.get_device_tensors(keys_dev)]
            v_parts = [ttnn.to_torch(t).float()[:, :, :S, :] for t in ttnn.get_device_tensors(vals_dev)]

            k_full = torch.cat(k_parts, dim=1)  # [1, 8, S, 128]  (8 KV heads)
            v_full = torch.cat(v_parts, dim=1)
            cpu_kv.append((k_full, v_full))
        print(f"  [decode] KV cache downloaded ({len(cpu_kv)} layers, " f"k_shape={cpu_kv[0][0].shape})", flush=True)

        # ---- CPU decode loop ----
        # We need the embedding table and block weights on CPU.
        # Reconstruct them from the TTNN tensors stored on the model.
        emb_cpu = ttnn.to_torch(ttnn.get_device_tensors(self.embedding)[0]).float().squeeze(0)
        # emb_cpu: [vocab_size, 4096] or [1, vocab, 4096]
        if emb_cpu.dim() == 3:
            emb_cpu = emb_cpu.squeeze(0)  # [vocab, 4096]

        cfg = self.configuration
        norm_eps = cfg.norm_eps

        # LN final and LM head on CPU
        ln_f_w = ttnn.to_torch(ttnn.get_device_tensors(self.ln_f.weight)[0]).float().reshape(-1)
        lm_head_w = (
            ttnn.to_torch(ttnn.get_device_tensors(self.lm_head)[0]).float().squeeze(0).squeeze(0)
        )  # [4096, vocab] before .T
        # lm_head was stored as lm_head_weight.T → [1,1,4096,vocab]; squeeze gives [4096,vocab]
        # For F.linear out=vocab, in=4096: weight=[vocab,4096], so take .T of [4096,vocab]
        lm_head_w = lm_head_w.T  # [vocab, 4096]

        # RoPE cache — extend to cover decode positions
        from models.tt_transformers.tt.common import precompute_freqs

        cos_raw, sin_raw = precompute_freqs(cfg.head_dim, (S + max_new_tokens + 1) * 2, cfg.rope_theta, None, None)
        from models.demos.molmo2.reference.functional import build_rope_cache

        cos_full, sin_full = build_rope_cache(S + max_new_tokens + 1, cfg.head_dim, cfg.rope_theta, torch.device("cpu"))

        # Pull block weights to CPU (once; cached for subsequent calls)
        if self._cpu_block_weights is None:
            self._cpu_block_weights = self._build_cpu_block_weights()
        block_weights = self._cpu_block_weights

        print("  [decode] block weights ready, starting decode loop...", flush=True)

        import torch.nn.functional as _F

        from models.demos.molmo2.reference.functional import rmsnorm

        def cpu_decode_step(token_id, pos, kv_cache_list):
            """One CPU decode step — single token, attending to positions 0..pos."""
            # Embedding
            x = F.embedding(torch.tensor([token_id]), emb_cpu).float()  # [1, 4096]
            x = x.unsqueeze(0)  # [1, 1, 4096]

            # cos_full: [1, S, 128] — index the sequence dim, not the batch dim
            cos_pos = cos_full[:, pos : pos + 1, :]  # [1, 1, 128]
            sin_pos = sin_full[:, pos : pos + 1, :]

            for i, bw in enumerate(block_weights):
                k_cache, v_cache = kv_cache_list[i]  # [1, 8, pos, 128]

                # Attention
                normed = rmsnorm(x, bw["attn_norm"], eps=norm_eps)
                # Full attention using text_attention reference
                # q,k,v from fused att_proj
                qkv = _F.linear(normed, bw["att_proj"])  # [1, 1, 6144]
                q_dim = cfg.n_heads * cfg.head_dim
                kv_dim = cfg.n_kv_heads * cfg.head_dim
                q_, k_, v_ = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
                q_ = q_.view(1, 1, cfg.n_heads, cfg.head_dim)
                k_ = k_.view(1, 1, cfg.n_kv_heads, cfg.head_dim)
                v_ = v_.view(1, 1, cfg.n_kv_heads, cfg.head_dim)
                q_ = rmsnorm(q_, bw["q_norm"], eps=norm_eps)
                k_ = rmsnorm(k_, bw["k_norm"], eps=norm_eps)
                # Transpose to [B, heads, S, D]
                q_ = q_.transpose(1, 2)
                k_ = k_.transpose(1, 2)
                v_ = v_.transpose(1, 2)
                # Apply RoPE for this position
                cos_b = cos_pos.unsqueeze(2)  # [1, 1, 1, 128]  (broadcast over heads)
                sin_b = sin_pos.unsqueeze(2)

                def _rot(t):
                    h = t.shape[-1] // 2
                    return torch.cat([-t[..., h:], t[..., :h]], dim=-1)

                q_ = q_ * cos_b + _rot(q_) * sin_b
                k_ = k_ * cos_b + _rot(k_) * sin_b
                # Build new K and append to cache
                k_new = k_.squeeze(2)  # [1, 8, 128]
                v_new = v_.squeeze(2)
                k_prev = k_cache  # [1, 8, pos, 128]
                v_prev = v_cache
                k_full = torch.cat([k_prev, k_new.unsqueeze(2)], dim=2)  # [1, 8, pos+1, 128]
                v_full = torch.cat([v_prev, v_new.unsqueeze(2)], dim=2)
                # Update cache
                kv_cache_list[i] = (k_full, v_full)
                # GQA: repeat k/v 4 times
                n_rep = cfg.n_heads // cfg.n_kv_heads
                k_gqa = k_full.repeat_interleave(n_rep, dim=1)  # [1, 32, pos+1, 128]
                v_gqa = v_full.repeat_interleave(n_rep, dim=1)
                # Attention scores
                scale = cfg.head_dim**-0.5
                attn_w = torch.matmul(q_, k_gqa.transpose(-2, -1)) * scale  # [1,32,1,pos+1]
                attn_w = _F.softmax(attn_w.float(), dim=-1).to(q_.dtype)
                attn_o = torch.matmul(attn_w, v_gqa)  # [1, 32, 1, 128]
                attn_o = attn_o.transpose(1, 2).reshape(1, 1, cfg.n_heads * cfg.head_dim)
                attn_o = _F.linear(attn_o, bw["attn_out"])
                x = x + attn_o

                # MLP
                normed_ff = rmsnorm(x, bw["ff_norm"], eps=norm_eps)
                ff = _F.linear(normed_ff, bw["ff_proj"])
                val, gate = ff.chunk(2, dim=-1)
                x = x + _F.linear(_F.silu(gate) * val, bw["ff_out"])

            # Final norm + LM head
            x = rmsnorm(x, ln_f_w, eps=norm_eps)
            logits = _F.linear(x, lm_head_w)  # [1, 1, vocab]: x@lm_head_w.T=[4096]@[4096,vocab]=[vocab]
            return logits[0, 0]  # [vocab]

        # Run decode loop
        current_pos = S
        kv_cache = [(cpu_kv[i][0].clone(), cpu_kv[i][1].clone()) for i in range(len(cpu_kv))]

        for step in range(max_new_tokens - 1):
            if eos_token_id is not None and next_token == eos_token_id:
                break
            logits_1 = cpu_decode_step(next_token, current_pos, kv_cache)
            next_token = _sample(logits_1, temperature)
            generated_ids.append(next_token)
            current_pos += 1
            if (step + 1) % 4 == 0:
                print(f"  [decode] step {step+1}/{max_new_tokens-1}", flush=True)

        return torch.tensor(generated_ids, dtype=torch.long)


def _sample(logits: torch.Tensor, temperature: float) -> int:
    """Sample from logits [vocab_size]. temperature=1.0 → greedy (argmax)."""
    if temperature == 0.0 or temperature == 1.0:
        return int(logits.argmax().item())
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())
