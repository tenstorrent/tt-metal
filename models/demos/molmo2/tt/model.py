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
PREFILL_BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192]
# Default warmup bucket sizes. Capped at 4096 because:
# - 8192 trace reserves ~720 MB permanently, which with 5 GB weights causes OOM
#   when the vision backbone (30 frames) runs during inference.
# - Sequences padded to 8192 compile their kernels lazily on first use and run
#   in eager mode (no trace), which is still fast after the first compile.
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
from models.demos.molmo2.tt.prefill_mask import build_molmo2_prefill_mask
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

        # Keep pooling weights as CPU float32 for the reference CPU pooling path.
        # The TTNN cross-attn path OOMs for video (N_windows~2430 × tile-padded 32 = 720 MB).
        pfx = "model.vision_backbone.image_pooling_2d"
        self._pool_cpu = {
            "wq": state_dict[f"{pfx}.wq.weight"].float(),
            "wq_b": state_dict[f"{pfx}.wq.bias"].float(),
            "wk": state_dict[f"{pfx}.wk.weight"].float(),
            "wk_b": state_dict[f"{pfx}.wk.bias"].float(),
            "wv": state_dict[f"{pfx}.wv.weight"].float(),
            "wv_b": state_dict[f"{pfx}.wv.bias"].float(),
            "wo": state_dict[f"{pfx}.wo.weight"].float(),
            "wo_b": state_dict[f"{pfx}.wo.bias"].float(),
        }
        self.image_projector = TtMolmo2ImageProjector(
            mesh_device=mesh_device,
            state_dict=state_dict,
            cfg=configuration,
            weight_cache_path=weight_cache_path,
        )

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
            zeros_k.to(float),
            dtype=ttnn.bfloat8_b,
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
        """Warm-up + trace capture for a fixed-bucket-size prefill.

        Uses causal-only attention (is_causal=True, no custom mask) so the same
        trace works for text-only, image, and video inputs.  The embedding and
        optional vision injection happen outside this trace.

        Pattern (matching reference demo):
          - Both warm-up and capture use trace_capture_run_begin() to suppress
            ttnn.deallocate() so the stable hidden buffer is never freed.
          - rot_mats are the full [1,1,max_seq,head_dim] matrix — rotary_embedding
            applies the first bucket_size positions automatically.
        """
        from models.demos.molmo2.tt.trace_capture_utils import trace_capture_run_begin, trace_capture_run_end

        cfg = self.configuration
        bucket_size = tt["bucket_size"]
        rot_mats = self._get_rot_mats_prefill(bucket_size)

        def _decoder_fwd(hidden):
            # Trace outputs [1, 1, S, 4096] hidden states (after ln_f but before lm_head).
            # Excluding lm_head from the trace keeps the output tensor at ~8-32 MB
            # instead of ~300 MB-1.2 GB for [S, vocab_size], allowing larger buckets.
            x = hidden
            for layer in self.layers:
                x = layer.forward(x, rot_mats=rot_mats, mode="prefill", attn_mask=None)
            return self.ln_f(x, mode=Mode.PREFILL)  # [1, 1, S, 4096]

        # ---- Warm-up (compile) pass — suppress deallocations to preserve tt["hidden"] ----
        tok = trace_capture_run_begin()
        try:
            logits_warmup = _decoder_fwd(tt["hidden"])
            ttnn.synchronize_device(self.mesh_device)
            ttnn.deallocate(logits_warmup)
        finally:
            trace_capture_run_end(tok)

        # ---- Trace capture ----
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

        valid = [b for b in bucket_sizes if b <= self.configuration.max_seq_len and b <= MAX_TRACE_BUCKET]
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
            # JIT kernels compile lazily on first use — no explicit warmup needed.
            # The first forward_prefill call for each bucket size will compile and
            # cache the TTNN kernels automatically.  Attempting to drive warmup via
            # forward_prefill(dummy_ids) can fail with storage.cpp:169 (typecast
            # bfloat8_b constraint) when the device is freshly initialised.
            print(f"[prefill warmup] use_trace=False — kernels compile on first real use", flush=True)

        n = len(self._prefill_traces)
        print(f"[prefill warmup] done — {n} traces, {len(valid) - n} JIT-compiled", flush=True)

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

    def run_vision_backbone(
        self,
        pixel_values: torch.Tensor,  # [B, n_crops, 729, 588] CPU
        pooled_patches_idx: torch.Tensor,  # [B, N_pooled, pool_window] CPU
    ) -> torch.Tensor:
        """Run full vision path and return [N_valid, 4096] CPU image features."""
        B, n_crops, n_patches, px_dim = pixel_values.shape

        # TP ViT: reshape to [n_crops, 1, 729, 588] on CPU first (avoids tile-volume mismatch),
        # then replicate all crops to all T3K devices.
        # TP weights (ShardTensorToMesh) handle per-device head parallelism;
        # ttnn.all_reduce after each block combines partial head results.
        n_crops_flat = B * n_crops
        pv_4d = pixel_values.reshape(n_crops_flat, 1, n_patches, px_dim).to(torch.bfloat16)

        pv_ttnn = ttnn.from_torch(
            pv_4d,  # [n_crops, 1, 729, 588]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # ViT encode → [1, n_crops, 729, 2304] (same on all devices after all_reduce)
        vit_features_ttnn = self.vit_encoder.forward(pv_ttnn)

        # Pull from device 0 (all devices have identical result after all_reduce)
        vit_cpu = ttnn.to_torch(ttnn.get_device_tensors(vit_features_ttnn)[0]).float()
        ttnn.deallocate(vit_features_ttnn)
        vit_cpu = vit_cpu.reshape(B, n_crops, n_patches, 2304)

        # Image pooling on CPU via reference PyTorch (avoids TTNN DRAM OOM for video).
        # The TTNN cross-attn path allocates [N_windows=2430, pool_window=9→32tile, 4608]
        # ≈720 MB, OOMing alongside 10 GB of model weights.
        from models.demos.molmo2.reference.functional import image_pooling_2d as _pool_ref

        p = self._pool_cpu
        pooled_cpu = _pool_ref(
            vit_cpu,
            pooled_patches_idx,
            p["wq"],
            p["wq_b"],
            p["wk"],
            p["wk_b"],
            p["wv"],
            p["wv_b"],
            p["wo"],
            p["wo_b"],
            num_heads=self.configuration.pool_n_heads,
            head_dim=self.configuration.pool_head_dim,
        )  # [B, N_pooled, 1152]

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

        # Move result to CPU for embedding injection
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
        if pixel_values is not None:
            image_features = self.run_vision_backbone(pixel_values, pooled_patches_idx)

            # Additive injection at image_patch_id positions.
            # x_cpu may be [S, H] or [1, S, H] depending on embedding output shape;
            # use view(-1, H) to handle both (matches the reference pattern).
            x_cpu = ttnn.to_torch(ttnn.get_device_tensors(x_ttnn)[0]).float()
            ttnn.deallocate(x_ttnn)
            H = self.configuration.dim
            is_patch_flat = input_ids.view(-1) == self.configuration.image_patch_id
            x_cpu.view(-1, H)[is_patch_flat] += image_features.to(x_cpu.dtype)

            x_ttnn = ttnn.from_torch(
                x_cpu.view(1, 1, S, H).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            x_ttnn = ttnn.to_layout(x_ttnn, ttnn.TILE_LAYOUT)

        # Reshape to [1, 1, S, dim] for decoder
        x_ttnn = ttnn.reshape(x_ttnn, [1, 1, S, self.configuration.dim])

        # ---- Route to trace or eager ----
        padded_S = get_padded_prefill_len(S)

        # ---- Prefill trace (causal-only, no image mask) ----
        # Padding is only applied when the trace exists for padded_S.
        # Eager path uses the actual sequence length S to avoid concat issues
        # with non-tile-aligned tensors.
        if padded_S in self._prefill_traces:
            trace_id, trace_tt, trace_x_norm = self._prefill_traces[padded_S]
            # Pad x_ttnn to padded_S before copying into the stable trace buffer
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
            # trace_x_norm is [1, 1, padded_S, 4096] — slice to last REAL token, apply lm_head
            x_norm_cpu = ttnn.to_torch(ttnn.get_device_tensors(trace_x_norm)[0]).float()
            # x_norm_cpu: [1, 1, padded_S, 4096] → slice last real token → [1, 1, 1, 4096]
            x_last = x_norm_cpu[:, :, S - 1 : S, :].to(torch.bfloat16)
            x_last_ttnn = ttnn.from_torch(
                x_last,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
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

        # ---- Eager path (no trace — use actual sequence length S, no padding) ----
        # Running with S avoids ttnn.concat on non-tile-aligned tensors.
        # JIT kernels compile for S on first call; subsequent same-S calls reuse them.
        attn_mask = None
        if token_type_ids is not None:
            attn_mask = build_molmo2_prefill_mask(S, token_type_ids.long(), self.mesh_device, dtype=ttnn.bfloat8_b)

        rot_mats = self._get_rot_mats_prefill(S)

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
        logits = ttnn.linear(
            x_ttnn,
            self.lm_head,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.configuration.compute_kernel_config_hifi2_fp16,
        )
        ttnn.deallocate(x_ttnn)

        logits_cpu = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).float().squeeze(0)
        ttnn.deallocate(logits)
        # Eager path uses actual S — return last token's logits [1, 1, vocab]
        return logits_cpu[:, S - 1 : S, :]

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
