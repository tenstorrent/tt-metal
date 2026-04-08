# models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py
"""Full Qwen3.5-9B text model for Blackhole P150.

Assembly: tok_embeddings → 32 × Qwen35TransformerBlock → RMSNorm → LM Head
Manages hybrid state: KV cache (8 attention layers) + recurrent state (24 DeltaNet layers).
"""
import glob

import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_decoder import Qwen35TransformerBlock, rms_norm_ttnn
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_rope import Qwen35RoPESetup
from models.demos.blackhole.qwen3_5_9b.tt.weight_mapping import remap_qwen35_state_dict


class Qwen35Model:
    """Qwen3.5-9B text-only language model on Blackhole P150.

    Usage:
        model = Qwen35Model.from_pretrained(device, checkpoint_dir)
        logits = model.prefill(token_ids)
        logits = model.decode(token_id, position)
    """

    def __init__(self, args, state_dict, device, weight_cache_path=None):
        self.args = args
        self.device = device

        # Embedding
        embed_weight = state_dict["tok_embeddings.weight"]
        self.tok_embeddings = ttnn.as_tensor(
            embed_weight.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=weight_cache_path / "tok_embeddings.weight" if weight_cache_path else None,
        )

        # RoPE setup (for gated attention layers only)
        self.rope = Qwen35RoPESetup(device, args)

        # Transformer layers
        logger.info(f"Loading {args.n_layers} transformer layers...")
        self.layers = []
        for i in tqdm(range(args.n_layers), desc="Loading layers"):
            layer = Qwen35TransformerBlock(args, state_dict, i, device, weight_cache_path)
            self.layers.append(layer)

        # Final norm — pre-offset by +1 for zero-centered RMSNorm
        norm_weight = state_dict["norm.weight"] + 1.0
        self.norm_weight = ttnn.as_tensor(
            norm_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=weight_cache_path / "norm.weight" if weight_cache_path else None,
        )
        self.norm_eps = args.norm_eps

        # LM Head — 2D [in, out] for ttnn.linear
        lm_head_weight = state_dict["output.weight"].T.contiguous()  # [4096, vocab_size]
        self.lm_head_weight = ttnn.as_tensor(
            lm_head_weight,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=weight_cache_path / "output.weight" if weight_cache_path else None,
        )

        self.vocab_size = args.vocab_size
        self._paged_kv_caches = None
        self._attention_layer_indices = [i for i in range(args.n_layers) if args.is_full_attention_layer(i)]

    @classmethod
    def from_pretrained(cls, device, checkpoint_dir, max_batch_size=1, max_seq_len=2048, n_layers=None):
        args = Qwen35ModelArgs(
            mesh_device=device,
            checkpoint_dir=checkpoint_dir,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        if n_layers is not None:
            args.n_layers = n_layers
            args.attention_type_list = args.attention_type_list[:n_layers]

        logger.info("Loading weights from safetensors...")
        raw_state_dict = {}
        safetensor_files = sorted(glob.glob(f"{checkpoint_dir}/model.safetensors-*.safetensors"))
        from safetensors import safe_open

        for path in safetensor_files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    raw_state_dict[key] = f.get_tensor(key)

        logger.info("Remapping weights...")
        state_dict = remap_qwen35_state_dict(raw_state_dict)
        del raw_state_dict

        cache_path = args.weight_cache_path()

        return cls(args, state_dict, device, weight_cache_path=cache_path)

    def prefill(self, token_ids):
        B, T = token_ids.shape

        if T > 1024:
            return self.prefill_layer_chunked(token_ids, chunk_size=1024)

        # Original path for short sequences
        self.reset_state(batch_size=B)

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)

        position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        cos, sin = self.rope.get_rot_mats(position_ids)

        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="prefill")

        x = rms_norm_ttnn(x, self.norm_weight, eps=self.norm_eps)

        x_last = x[:, -1:, :]
        logits = ttnn.linear(x_last, self.lm_head_weight)

        return logits

    def prefill_layer_chunked(self, token_ids, chunk_size=2048):
        """Prefill long sequences using layer-at-a-time chunked processing.

        Unlike prefill_segmented (segment through all layers), this processes
        each layer across the full sequence before moving to the next. DeltaNet
        uses chunk mode with a larger chunk_size (256 vs default 64) to reduce
        error accumulation across sub-chunks. At chunk_size=64, 4096 tokens
        produce 64 sub-chunks where Neumann series errors compound beyond
        tested PCC thresholds. At chunk_size=256, only 16 sub-chunks are needed,
        matching the validated PCC range (>0.98).

        Args:
            token_ids: [B, T] token IDs
            chunk_size: tokens per chunk (default 2048, matches direct prefill limit)
        """
        B, T = token_ids.shape
        self.reset_state(batch_size=B)

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(token_ids_ttnn)

        # Attention layers can use larger chunks than DeltaNet — no Neumann series
        # limitation, and fewer chunks means fewer unique KV cache sizes for SDPA
        # compilation. 4096 = 4x fewer SDPA compilations vs chunk_size=1024.
        attn_chunk_size = max(chunk_size, 4096)

        for layer_idx, layer in enumerate(self.layers):
            layer_chunk_size = attn_chunk_size if layer.is_full_attention else chunk_size

            chunks_out = []
            for chunk_start in range(0, T, layer_chunk_size):
                chunk_end = min(chunk_start + layer_chunk_size, T)

                x_chunk = x[:, chunk_start:chunk_end, :]
                x_chunk = ttnn.to_layout(x_chunk, ttnn.TILE_LAYOUT)

                if layer.is_full_attention:
                    cos = self.rope.cos_device[:, chunk_start:chunk_end, :]
                    sin = self.rope.sin_device[:, chunk_start:chunk_end, :]
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

            if len(chunks_out) == 1:
                x_new = chunks_out[0]
            else:
                x_new = ttnn.concat(chunks_out, dim=1)
                for c in chunks_out:
                    ttnn.deallocate(c)
            x_new = ttnn.to_memory_config(x_new, ttnn.DRAM_MEMORY_CONFIG)

            ttnn.deallocate(x)
            x = x_new

        x_last = x[:, -1:, :]
        x_last = rms_norm_ttnn(x_last, self.norm_weight, eps=self.norm_eps)
        logits = ttnn.linear(x_last, self.lm_head_weight)
        ttnn.deallocate(x)

        return logits

    def decode(self, token_ids, current_pos):
        B = token_ids.shape[0]

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(token_ids_ttnn)

        position_ids = torch.full((B, 1), current_pos, dtype=torch.long)
        cos, sin = self.rope.get_rot_mats(position_ids)

        # Create cur_pos_tensor for SDPA decode + paged_update_cache.
        # paged_update_cache reshapes cache to [B*H_kv, ...] so needs B*H_kv indices.
        n_kv = self.args.n_kv_heads
        cur_pos_tensor = ttnn.from_torch(
            torch.full((B * n_kv,), current_pos, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        for i, layer in enumerate(self.layers):
            x = layer.forward(x, cos=cos, sin=sin, mode="decode", position_tensor=cur_pos_tensor)

        x = rms_norm_ttnn(x, self.norm_weight, eps=self.norm_eps)
        logits = ttnn.linear(x, self.lm_head_weight)
        ttnn.deallocate(x)

        return logits

    def forward_decode(self, x, cos, sin):
        """Trace-compatible decode: fixed shape [B=1, T=1, D=4096], no Python conditionals."""
        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="decode")
        x = rms_norm_ttnn(x, self.norm_weight, eps=self.norm_eps)
        logits = ttnn.linear(x, self.lm_head_weight)
        ttnn.deallocate(x)
        return logits

    def forward_decode_traced(self, token_ids_buf, cos, sin, position_tensor=None):
        """Trace-compatible decode starting from token IDs (embedding inside trace).

        All ops are device-to-device. No host interaction.
        Unlike forward_decode which takes pre-computed embeddings, this takes a
        uint32 token ID buffer and runs embedding as the first traced op.

        Args:
            token_ids_buf: ttnn.Tensor [1, 1] uint32 on device — token ID buffer
            cos: ttnn.Tensor [1, 1, rope_head_dim] bfloat16 TILE on device
            sin: ttnn.Tensor [1, 1, rope_head_dim] bfloat16 TILE on device
            position_tensor: ttnn.Tensor [1] int32 ROW_MAJOR on device (Phase 2, optional)
        """
        x = ttnn.embedding(token_ids_buf, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
        for layer in self.layers:
            if layer.is_full_attention and position_tensor is not None:
                x = layer.forward(x, cos=cos, sin=sin, mode="decode", position_tensor=position_tensor)
            else:
                x = layer.forward(x, cos=cos, sin=sin, mode="decode")
        x = rms_norm_ttnn(x, self.norm_weight, eps=self.norm_eps)
        logits = ttnn.linear(x, self.lm_head_weight)
        ttnn.deallocate(x)
        return logits

    def enable_trace(self, batch_size=1):
        """Prepare all layers for trace-compatible decode."""
        logger.info("Enabling trace mode...")
        for layer in self.layers:
            if layer.is_full_attention:
                layer.attention.enable_preallocated_cache(batch_size)
                layer.attention.enable_trace_mode()
            else:
                layer.attention.enable_inplace_state()

    def _prefill_for_trace(self, token_ids):
        """Prefill that populates pre-allocated KV caches and DeltaNet states."""
        B, T = token_ids.shape

        # Reset DeltaNet states
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.reset_state(batch_size=B)

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(token_ids_ttnn)

        for layer in self.layers:
            if layer.is_full_attention:
                layer.attention.use_trace_mode = False
                layer.attention.use_preallocated_cache = False

        chunk_size = 1024
        attn_chunk_size = max(chunk_size, 4096)
        if T <= chunk_size:
            position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
            cos, sin = self.rope.get_rot_mats(position_ids)
            for layer in self.layers:
                x = layer.forward(x, cos=cos, sin=sin, mode="prefill")
        else:
            for layer_idx, layer in enumerate(self.layers):
                layer_chunk_size = attn_chunk_size if layer.is_full_attention else chunk_size
                chunks_out = []
                for chunk_start in range(0, T, layer_chunk_size):
                    chunk_end = min(chunk_start + layer_chunk_size, T)

                    x_chunk = x[:, chunk_start:chunk_end, :]
                    x_chunk = ttnn.to_layout(x_chunk, ttnn.TILE_LAYOUT)

                    if layer.is_full_attention:
                        cos = self.rope.cos_device[:, chunk_start:chunk_end, :]
                        sin = self.rope.sin_device[:, chunk_start:chunk_end, :]
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

                if len(chunks_out) == 1:
                    x_new = chunks_out[0]
                else:
                    x_new = ttnn.concat(chunks_out, dim=1)
                    for c in chunks_out:
                        ttnn.deallocate(c)
                x_new = ttnn.to_memory_config(x_new, ttnn.DRAM_MEMORY_CONFIG)

                ttnn.deallocate(x)
                x = x_new

                kind = "attention" if layer.is_full_attention else "deltanet"
                logger.info(f"Trace prefill layer {layer_idx} ({kind}) done")

        x = rms_norm_ttnn(x, self.norm_weight, eps=self.norm_eps)
        x_last = x[:, -1:, :]
        logits = ttnn.linear(x_last, self.lm_head_weight)

        # Populate the pre-allocated KV caches from the concat-based prefill cache.
        # Use paged_fill_cache which writes in the same tile format that paged_update_cache
        # (used during decode) and sdpa_decode (used for attention) expect.
        # Following the GPT-OSS pattern (attention/prefill.py:118-121): pass data in
        # original [B, H_kv, S, D] shape with an identity page table.
        for layer in self.layers:
            if layer.is_full_attention:
                attn = layer.attention
                if attn.kv_cache_key is not None and attn.past_key is not None:
                    # past_key: [B, H_kv, S, D], cache: [B, H_kv, max_seq, D]
                    S = attn.past_key.shape[2]
                    block_size = 64

                    # Identity page table: virtual block i → physical block i
                    num_blocks = (S + block_size - 1) // block_size
                    page_table = ttnn.from_torch(
                        torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0),
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=self.device,
                    )

                    # Clip to page-aligned length (paged_fill_cache expects seq_len <= page_table pages * block_size)
                    page_len = num_blocks * block_size
                    k_fill = attn.past_key[:, :, :page_len, :] if page_len < attn.past_key.shape[2] else attn.past_key
                    v_fill = (
                        attn.past_value[:, :, :page_len, :] if page_len < attn.past_value.shape[2] else attn.past_value
                    )

                    ttnn.experimental.paged_fill_cache(attn.kv_cache_key, k_fill, page_table, batch_idx=0)
                    ttnn.experimental.paged_fill_cache(attn.kv_cache_value, v_fill, page_table, batch_idx=0)
                    ttnn.deallocate(attn.past_key)
                    ttnn.deallocate(attn.past_value)
                    ttnn.deallocate(page_table)

                    attn.cache_pos = S
                    attn.update_mask_for_pos(S)
                # Re-enable preallocated cache + trace mode for decode
                attn.use_preallocated_cache = True
                attn.use_trace_mode = True
                attn.past_key = None
                attn.past_value = None

        # Fuse DeltaNet conv states for decode
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                if dn.fused_conv_state is None and dn.conv_state_q is not None:
                    dn.fused_conv_state = ttnn.concat([dn.conv_state_q, dn.conv_state_k, dn.conv_state_v], dim=2)
                    dn.fused_conv_state = ttnn.to_layout(dn.fused_conv_state, ttnn.TILE_LAYOUT)

        return logits

    def capture_decode_trace(self, device):
        """Capture a trace for the decode path after prefill + warmup.

        Pre-allocates input buffers, runs a warmup pass to compile programs,
        then captures the trace. DeltaNet states are saved/restored around
        warmup and capture since both corrupt the in-place state buffers.
        """
        # 1. Pre-allocate buffers (host→device, BEFORE capture)
        self._trace_token_ids = ttnn.from_torch(
            torch.zeros(1, 1, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self._trace_cos = ttnn.from_torch(
            torch.zeros(1, 1, self.args.rope_head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self._trace_sin = ttnn.from_torch(
            torch.zeros(1, 1, self.args.rope_head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # 2. Save DeltaNet states (warmup + capture corrupt them)
        saved_dn_states = self._save_deltanet_states()

        # 3. Warmup (compiles programs, BEFORE capture — host writes OK here)
        # Pass position_tensor=None to use the staging+mask path (not sdpa_decode).
        # sdpa_decode has a known issue reading caches populated by paged_fill_cache.
        _ = self.forward_decode_traced(self._trace_token_ids, self._trace_cos, self._trace_sin)
        ttnn.synchronize_device(device)

        # 4. Restore states after warmup corruption, save again for capture
        self._restore_deltanet_states(saved_dn_states, device)
        saved_dn_states = self._save_deltanet_states()

        # 5. Capture — ONLY device ops between begin/end
        self._trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self._trace_output = self.forward_decode_traced(self._trace_token_ids, self._trace_cos, self._trace_sin)
        ttnn.end_trace_capture(device, self._trace_id, cq_id=0)

        # 6. Restore DeltaNet states after capture corruption
        self._restore_deltanet_states(saved_dn_states, device)

        logger.info("Trace captured successfully (embedding inside trace)!")

    def decode_traced(self, token_ids, current_pos):
        """Execute traced decode: write inputs via host DMA, replay trace.

        Uses the staging+mask approach: K/V are written to the staging position
        (max_seq_len-1) inside the trace. After replay, the caller must copy
        staging to the correct position and update the mask.

        Args:
            token_ids: torch.Tensor [1, 1] — token IDs (int64 or int32)
            current_pos: int — current position in the sequence
        """
        # Host→device copies (fast DMA, before trace replay)
        token_host = ttnn.from_torch(token_ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(token_host, self._trace_token_ids)

        # RoPE from pre-computed CPU table (no device ops)
        cos_host, sin_host = self.rope.get_cos_sin_host(current_pos)
        ttnn.copy_host_to_device_tensor(cos_host, self._trace_cos)
        ttnn.copy_host_to_device_tensor(sin_host, self._trace_sin)

        # Replay the captured trace (staging approach — no position tensor)
        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)

        return self._trace_output

    def reset_state(self, batch_size=None):
        """Reset all layer states for a new sequence."""
        for layer in self.layers:
            if layer.is_full_attention:
                layer.attention.reset_cache()
            else:
                layer.attention.reset_state(batch_size)

    def set_paged_kv_caches(self, kv_caches):
        """Attach paged KV caches to the 8 attention layers."""
        self._paged_kv_caches = kv_caches
        for cache_idx, layer_idx in enumerate(self._attention_layer_indices):
            k_cache, v_cache = kv_caches[cache_idx]
            self.layers[layer_idx].attention.set_paged_kv_cache(k_cache, v_cache)

    def _fill_paged_cache_from_prefill(self, page_table):
        """Transfer concat-based K/V into paged cache after prefill.

        Processes one layer at a time to avoid holding all 8 layers' concat K/V
        simultaneously (~1GB at 64K tokens).
        """
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

    def prefill_paged(self, token_ids, page_table):
        """Prefill using existing concat-based attention, then fill paged cache.

        Args:
            token_ids: torch.Tensor [B, T] token IDs
            page_table: ttnn.Tensor [B, max_blocks_per_seq] int32 on device
        Returns:
            logits: ttnn.Tensor [B, 1, vocab_size]
        """
        B, T = token_ids.shape
        self.reset_state(batch_size=B)

        # Use existing prefill (concat-based K/V for SDPA)
        if T > 1024:
            logits = self.prefill_layer_chunked(token_ids, chunk_size=1024)
        else:
            token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
            x = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
            ttnn.deallocate(token_ids_ttnn)

            position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
            cos, sin = self.rope.get_rot_mats(position_ids)

            for layer in self.layers:
                x = layer.forward(x, cos=cos, sin=sin, mode="prefill")

            x = rms_norm_ttnn(x, self.norm_weight, eps=self.norm_eps)
            x_last = x[:, -1:, :]
            logits = ttnn.linear(x_last, self.lm_head_weight)
            ttnn.deallocate(x)

        # Fill paged cache from accumulated concat K/V
        self._fill_paged_cache_from_prefill(page_table)

        # Prepare DeltaNet for decode (fuse conv states)
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                if dn.fused_conv_state is None and dn.conv_state_q is not None:
                    dn.fused_conv_state = ttnn.concat([dn.conv_state_q, dn.conv_state_k, dn.conv_state_v], dim=2)
                    dn.fused_conv_state = ttnn.to_layout(dn.fused_conv_state, ttnn.TILE_LAYOUT)

        return logits

    def decode_paged(self, token_ids, current_pos, page_table):
        """Single-token decode using paged KV cache.

        Args:
            token_ids: torch.Tensor [B, 1] token IDs
            current_pos: int -- current position in the sequence
            page_table: ttnn.Tensor [B, max_blocks_per_seq] int32 on device
        Returns:
            logits: ttnn.Tensor [B, 1, vocab_size]
        """
        B = token_ids.shape[0]

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(token_ids_ttnn)

        position_ids = torch.full((B, 1), current_pos, dtype=torch.long)
        cos, sin = self.rope.get_rot_mats(position_ids)

        # cur_pos_tensor shape [B] for paged ops (NOT [B*n_kv] like the non-paged path)
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

        x = rms_norm_ttnn(x, self.norm_weight, eps=self.norm_eps)
        logits = ttnn.linear(x, self.lm_head_weight)
        ttnn.deallocate(x)

        return logits

    def _save_deltanet_states(self):
        """Save DeltaNet recurrent + conv states to CPU for restoration after trace capture."""
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
        """Restore DeltaNet states using ttnn.copy into original buffers (preserves addresses)."""
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
                if saved["conv"] is not None:
                    restored_conv = ttnn.from_torch(
                        saved["conv"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    )
                    ttnn.copy(restored_conv, dn.fused_conv_state)
                    ttnn.deallocate(restored_conv)
                idx += 1
