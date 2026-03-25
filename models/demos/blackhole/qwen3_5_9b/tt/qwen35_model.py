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

    @classmethod
    def from_pretrained(cls, device, checkpoint_dir, max_batch_size=1, max_seq_len=2048):
        args = Qwen35ModelArgs(
            mesh_device=device,
            checkpoint_dir=checkpoint_dir,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

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

    def prefill(self, token_ids, segment_size=1024):
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

    def prefill_segmented(self, token_ids, segment_size=1024):
        """Prefill long sequences by processing in segments.

        Each segment runs through all 32 layers. DeltaNet recurrent state and
        conv state carry over between segments automatically (stored as instance
        attributes). Attention KV cache accumulates via concat.

        Args:
            token_ids: [B, T] token IDs, T can be >> segment_size
            segment_size: number of tokens per segment (default 1024, must be <= 1024
                         to avoid L1 OOM on DeltaNet projections)
        """
        B, T = token_ids.shape
        self.reset_state(batch_size=B)

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x_all = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
        # For 65K tokens, x_all is [1, 65536, 4096] = ~512MB — keep in DRAM
        x_all = ttnn.to_memory_config(x_all, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(token_ids_ttnn)

        for seg_start in range(0, T, segment_size):
            seg_end = min(seg_start + segment_size, T)

            # Slice embeddings for this segment
            x_seg = x_all[:, seg_start:seg_end, :]
            x_seg = ttnn.to_layout(x_seg, ttnn.TILE_LAYOUT)

            # RoPE with absolute positions
            position_ids = torch.arange(seg_start, seg_end).unsqueeze(0).expand(B, -1)
            cos, sin = self.rope.get_rot_mats(position_ids)

            # Process through all layers — use "prefill_segmented" mode so DeltaNet
            # layers use recurrent (not chunked) to avoid compound error across segments.
            for layer in self.layers:
                x_seg = layer.forward(x_seg, cos=cos, sin=sin, mode="prefill_segmented")

            logger.info(f"Prefill segment [{seg_start}:{seg_end}] done")

        # Free the full embedding tensor
        ttnn.deallocate(x_all)

        # Final norm + LM head on last token only
        x_last = x_seg[:, -1:, :]
        x_last = rms_norm_ttnn(x_last, self.norm_weight, eps=self.norm_eps)
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
        import time as _time

        B, T = token_ids.shape
        self.reset_state(batch_size=B)

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(token_ids_ttnn)

        for layer_idx, layer in enumerate(self.layers):
            ttnn.synchronize_device(self.device)
            _tl0 = _time.time()

            chunks_out = []
            for chunk_start in range(0, T, chunk_size):
                chunk_end = min(chunk_start + chunk_size, T)

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

            ttnn.synchronize_device(self.device)
            _tl1 = _time.time()
            kind = "attention" if layer.is_full_attention else "deltanet"
            import torch as _torch

            x_diag = ttnn.to_torch(x).float()
            x_last_tok = x_diag[:, -1:, :].float()
            diag_msg = (
                f"Layer {layer_idx} ({kind}) {(_tl1-_tl0)*1000:.0f}ms | "
                f"x: mean={x_diag.abs().mean():.4f} max={x_diag.abs().max():.4f} "
                f"inf={_torch.isinf(x_diag).sum().item()} | "
                f"x_last: mean={x_last_tok.abs().mean():.4f}"
            )
            if not layer.is_full_attention:
                s = ttnn.to_torch(layer.attention.recurrent_state).float()
                diag_msg += f" | state: mean={s.abs().mean():.6f} max={s.abs().max():.4f}"
            logger.info(diag_msg)

        x_last = x[:, -1:, :]
        x_last = rms_norm_ttnn(x_last, self.norm_weight, eps=self.norm_eps)
        logits = ttnn.linear(x_last, self.lm_head_weight)
        ttnn.deallocate(x)

        return logits

    def decode(self, token_ids, current_pos, profile=False):
        import time as _time

        B = token_ids.shape[0]

        if profile:
            ttnn.synchronize_device(self.device)
            _t0 = _time.time()

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(token_ids_ttnn)

        position_ids = torch.full((B, 1), current_pos, dtype=torch.long)
        cos, sin = self.rope.get_rot_mats(position_ids)

        if profile:
            ttnn.synchronize_device(self.device)
            _t1 = _time.time()
            _layer_times = {"deltanet": [], "attention": []}

        for i, layer in enumerate(self.layers):
            if profile:
                ttnn.synchronize_device(self.device)
                _tl0 = _time.time()
            x = layer.forward(x, cos=cos, sin=sin, mode="decode")
            if profile:
                ttnn.synchronize_device(self.device)
                _tl1 = _time.time()
                key = "attention" if layer.is_full_attention else "deltanet"
                _layer_times[key].append(_tl1 - _tl0)

        if profile:
            ttnn.synchronize_device(self.device)
            _t2 = _time.time()

        x = rms_norm_ttnn(x, self.norm_weight, eps=self.norm_eps)
        logits = ttnn.linear(x, self.lm_head_weight)
        ttnn.deallocate(x)

        if profile:
            ttnn.synchronize_device(self.device)
            _t3 = _time.time()
            dn_avg = sum(_layer_times["deltanet"]) / len(_layer_times["deltanet"]) * 1000
            att_avg = sum(_layer_times["attention"]) / len(_layer_times["attention"]) * 1000
            logger.info(
                f"  [PROFILE] embed+rope: {(_t1-_t0)*1000:.1f}ms | "
                f"deltanet(24): {sum(_layer_times['deltanet'])*1000:.1f}ms (avg {dn_avg:.1f}ms) | "
                f"attention(8): {sum(_layer_times['attention'])*1000:.1f}ms (avg {att_avg:.1f}ms) | "
                f"norm+lmhead: {(_t3-_t2)*1000:.1f}ms | "
                f"total: {(_t3-_t0)*1000:.1f}ms"
            )

        return logits

    def forward_decode(self, x, cos, sin):
        """Trace-compatible decode: fixed shape [B=1, T=1, D=4096], no Python conditionals."""
        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="decode")
        x = rms_norm_ttnn(x, self.norm_weight, eps=self.norm_eps)
        logits = ttnn.linear(x, self.lm_head_weight)
        ttnn.deallocate(x)
        return logits

    def enable_trace(self, batch_size=1):
        """Prepare all layers for trace-compatible decode."""
        logger.info("Enabling trace mode for all layers...")
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
        if T <= chunk_size:
            position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
            cos, sin = self.rope.get_rot_mats(position_ids)
            for layer in self.layers:
                x = layer.forward(x, cos=cos, sin=sin, mode="prefill")
        else:
            for layer_idx, layer in enumerate(self.layers):
                chunks_out = []
                for chunk_start in range(0, T, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, T)

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

        # Now populate the pre-allocated KV caches from the concat-based cache
        for layer in self.layers:
            if layer.is_full_attention:
                attn = layer.attention
                if attn.kv_cache_key is not None and attn.past_key is not None:
                    # past_key shape: [B, H_kv, S, D] — copy into pre-allocated [B, H_kv, max_seq, D]
                    past_k_torch = ttnn.to_torch(attn.past_key)
                    past_v_torch = ttnn.to_torch(attn.past_value)
                    S = past_k_torch.shape[2]

                    # Build full cache tensors with prefill data at positions 0..S-1
                    full_k = torch.zeros(B, attn.num_kv_heads, attn.max_seq_len, attn.head_dim, dtype=torch.bfloat16)
                    full_v = torch.zeros(B, attn.num_kv_heads, attn.max_seq_len, attn.head_dim, dtype=torch.bfloat16)
                    full_k[:, :, :S, :] = past_k_torch
                    full_v[:, :, :S, :] = past_v_torch

                    # Overwrite pre-allocated cache (deallocate old, load new to DRAM)
                    ttnn.deallocate(attn.kv_cache_key)
                    ttnn.deallocate(attn.kv_cache_value)
                    attn.kv_cache_key = ttnn.from_torch(
                        full_k,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    attn.kv_cache_value = ttnn.from_torch(
                        full_v,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )

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

        self._prefill_len = T
        return logits

    def capture_decode_trace(self, device):
        """Capture a trace for the decode path after prefill + warmup."""
        # Pre-allocate fixed buffers for trace inputs
        self._trace_input = ttnn.from_torch(
            torch.zeros(1, 1, self.args.dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
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

        # Save DeltaNet states before trace capture. Trace capture executes
        # forward_decode with zero inputs, which corrupts in-place state buffers.
        saved_dn_states = []
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                saved_dn_states.append(
                    {
                        "recurrent": ttnn.to_torch(dn.recurrent_state),
                        "conv": ttnn.to_torch(dn.fused_conv_state) if dn.fused_conv_state is not None else None,
                    }
                )

        # Capture trace
        self._trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self._trace_output = self.forward_decode(self._trace_input, self._trace_cos, self._trace_sin)
        ttnn.end_trace_capture(device, self._trace_id, cq_id=0)

        # Restore DeltaNet states after trace capture. Must use ttnn.copy() into the
        # original buffers (not replace references) since the trace recorded operations
        # against those specific buffer addresses.
        idx = 0
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                saved = saved_dn_states[idx]
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

        logger.info("Trace captured successfully!")

    def decode_traced(self, token_ids, current_pos):
        """Execute traced decode: write inputs, replay trace, update KV cache."""
        # Embed token (outside trace)
        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(token_ids_ttnn)

        # Write embedding into trace input buffer
        ttnn.copy(x, self._trace_input)
        ttnn.deallocate(x)

        # Write position-specific RoPE values into trace buffers
        cos_pos = self.rope.cos_device[:, current_pos : current_pos + 1, :]
        sin_pos = self.rope.sin_device[:, current_pos : current_pos + 1, :]
        ttnn.copy(cos_pos, self._trace_cos)
        ttnn.copy(sin_pos, self._trace_sin)

        # Update attention masks BEFORE replay
        # Expose positions 0..current_pos-1 (already in cache) + staging pos (current token)
        for layer in self.layers:
            if layer.is_full_attention:
                layer.attention.update_mask_for_pos(current_pos)

        # Replay the captured trace
        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=True)

        # After trace: copy staging K/V to the actual position in cache
        for layer in self.layers:
            if layer.is_full_attention:
                layer.attention.update_cache_after_trace(current_pos)

        return self._trace_output

    def reset_state(self, batch_size=None):
        """Reset all layer states for a new sequence."""
        for layer in self.layers:
            if layer.is_full_attention:
                layer.attention.reset_cache()
            else:
                layer.attention.reset_state(batch_size)
