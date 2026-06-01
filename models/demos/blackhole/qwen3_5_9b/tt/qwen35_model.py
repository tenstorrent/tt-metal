# models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py
"""Full Qwen3.5-9B text model for Blackhole P150.

Assembly: tok_embeddings → 32 × Qwen35DecoderLayer → RMSNorm → LM Head
Manages hybrid state: KV cache (8 attention layers) + recurrent state (24 DeltaNet layers).
"""
import math

import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.layer import Qwen35DecoderLayer
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_rope import Qwen35RoPESetup
from models.demos.blackhole.qwen3_5_9b.tt.rms_norm import rms_norm_ttnn


class Qwen35Model:
    """Qwen3.5-9B text-only language model on Blackhole P150.

    Usage:
        # HF_MODEL env var (hub name or local path) is the single source of truth.
        model = Qwen35Model.from_pretrained(device)
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
            layer = Qwen35DecoderLayer(device, args, state_dict, i, weight_cache_path)
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
        self._trace_id = None
        self._prev_page_table = None
        self._deltanet_external_states = None  # list of (recurrent, conv) tuples, set by allocate_kv_caches
        # Trace-prefill: persistent device buffers for inputs that would otherwise
        # be allocated per-call inside prefill_layer_chunked. Populated by
        # capture_prefill_trace_paged. When None, prefill uses the legacy
        # allocate-per-call path.
        self._prefill_trace_inputs = None
        # Trace handle + persistent output buffer for traced prefill replay.
        self._prefill_trace_id = None
        self._prefill_trace_logits = None
        self._prefill_bucket_size = None
        # Shared zero buffers for in-place DN state reset between traced replays.
        self._dn_zero_recurrent = None
        self._dn_zero_conv = None
        # Chunk-outer traced prefill: one chunk's all-layer forward captured as a single
        # trace and replayed per chunk (DMA-advancing per-chunk inputs). Persistent buffers
        # below are reused across replays; their addresses are baked into the trace.
        self._chunked_trace_id = None
        self._chunked_trace_output = None
        self._chunked_chunk_size = None
        self._chunk_token_buf = None
        self._chunk_start_idx_tensor = None
        self._chunk_page_table_buf = None
        self._chunk_full_page_table_buf = None
        self._chunk_cos_buf = None
        self._chunk_sin_buf = None

    @classmethod
    def from_pretrained(cls, device, max_batch_size=1, max_seq_len=2048, n_layers=None, hf_model=None):
        # HF_MODEL (env var) is the single source of truth — a hub name or local path —
        # resolved by Qwen35ModelArgs via the base ModelArgs. `hf_model` is an optional
        # back-compat convenience: if given, it sets HF_MODEL before constructing args.
        if hf_model is not None:
            import os

            os.environ["HF_MODEL"] = hf_model

        args = Qwen35ModelArgs(
            mesh_device=device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        if n_layers is not None:
            args.n_layers = n_layers
            args.attention_type_list = args.attention_type_list[:n_layers]

        logger.info("Loading + remapping weights via Qwen35ModelArgs.load_state_dict()...")
        state_dict = args.load_state_dict()

        cache_path = args.weight_cache_path()
        return cls(args, state_dict, device, weight_cache_path=cache_path)

    def prefill(self, token_ids):
        B, T = token_ids.shape

        if T > 1024:
            return self.prefill_layer_chunked(token_ids, chunk_size=2048)

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

    def prefill_layer_chunked(self, token_ids, chunk_size=2048, page_table=None):
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
            page_table: torch.Tensor [B, max_blocks] or None. When provided,
                        uses paged prefill (paged_fill_cache + chunked_sdpa).
                        When None, uses concat-based KV accumulation.
        """
        B, T = token_ids.shape
        self.reset_state(batch_size=B)

        # Trace-mode: caller pre-loaded persistent device buffers for token_ids,
        # page_table, and the per-chunk page_table sub-buffer (single chunk for
        # full-attn at T==bucket). All host→device transfers must happen *before*
        # this function in trace replay; here we only consume the buffers.
        trace_inputs = self._prefill_trace_inputs

        if trace_inputs is not None:
            token_ids_ttnn = trace_inputs["token_ids"]
            page_table_tt = trace_inputs["page_table"]
        else:
            token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)

        x = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        if trace_inputs is None:
            ttnn.deallocate(token_ids_ttnn)

        # Attention layers can use larger chunks than DeltaNet — no Neumann series
        # limitation, and fewer chunks means fewer unique KV cache sizes for SDPA
        # compilation. 4096 = 4x fewer SDPA compilations vs chunk_size=1024.
        attn_chunk_size = max(chunk_size, 4096)

        if trace_inputs is None:
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
                    # Paged prefill path
                    cos = self.rope.cos_device[:, chunk_start:chunk_end, :]
                    sin = self.rope.sin_device[:, chunk_start:chunk_end, :]

                    block_size = 64
                    chunk_blocks_end = math.ceil(chunk_end / block_size)
                    if trace_inputs is not None:
                        # One pre-allocated buffer per full-attn chunk (sized at capture).
                        chunk_idx = chunk_start // attn_chunk_size
                        chunk_page_table_tt = trace_inputs["chunk_page_tables"][chunk_idx]
                    else:
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
                    # Original concat path (non-paged prefill)
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

            # On the last layer, save the last token from the last chunk BEFORE concat.
            # For long sequences (T > 4096), slicing x[:, -1:, :] on the full
            # [1, T, 4096] concatenated tensor triggers an L1 clash in the slice program.
            # Extracting from the last chunk (at most [1, 4096, 4096]) avoids this.
            # In trace mode we DON'T extract — we return the full hidden state so the
            # caller can slice at `actual_len-1` (a runtime value the trace can't bake in).
            is_last_layer = layer_idx == len(self.layers) - 1
            if is_last_layer and trace_inputs is None:
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

        if trace_inputs is not None:
            # Trace mode: return the full last-layer hidden state. The caller
            # (prefill_traced_paged) will slice at the real prompt length and
            # run rms_norm + lm_head outside the trace.
            return x

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

    def _forward_decode(self, token_ids_buf, cos, sin, cur_pos_tensor, page_table):
        """Device-facing paged decode forward. ALL inputs are device tensors.
        Trace-safe: no host-device transfers inside this function.
        """
        x = ttnn.embedding(token_ids_buf, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
        for layer in self.layers:
            if layer.is_full_attention:
                x = layer.forward(x, cos, sin, position_tensor=cur_pos_tensor, page_table=page_table, mode="decode")
            else:
                x = layer.forward(x, mode="decode")
        x = rms_norm_ttnn(x, self.norm_weight, eps=self.norm_eps)
        logits = ttnn.linear(x, self.lm_head_weight)
        ttnn.deallocate(x)
        return logits

    def capture_decode_trace_paged(self, device, page_table):
        """Capture a trace for paged decode. page_table is a host torch.Tensor."""
        assert self._deltanet_external_states is not None, "Call allocate_kv_caches first"

        if self._trace_id is not None:
            ttnn.release_trace(device, self._trace_id)
            self._trace_id = None
            for buf_name in ["_trace_token_ids", "_trace_cos", "_trace_sin", "_trace_cur_pos", "_trace_page_table"]:
                buf = getattr(self, buf_name, None)
                if buf is not None:
                    ttnn.deallocate(buf)

        self._trace_token_ids = ttnn.from_torch(
            torch.zeros(1, 1, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        cos_host, sin_host = self.rope.get_cos_sin_host(0)
        self._trace_cos = ttnn.to_device(cos_host, device)
        self._trace_sin = ttnn.to_device(sin_host, device)
        self._trace_cur_pos = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self._trace_page_table = ttnn.from_torch(
            page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        saved = self._save_deltanet_states()
        dummy_tokens = torch.zeros(1, 1, dtype=torch.long)
        self.decode_paged(dummy_tokens, 0, page_table)
        ttnn.synchronize_device(device)

        self._restore_deltanet_states(saved, device)
        saved = self._save_deltanet_states()

        self._trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self._trace_output = self._forward_decode(
            self._trace_token_ids,
            self._trace_cos,
            self._trace_sin,
            self._trace_cur_pos,
            self._trace_page_table,
        )
        ttnn.end_trace_capture(device, self._trace_id, cq_id=0)

        self._restore_deltanet_states(saved, device)
        logger.info("Paged trace captured successfully!")

    def decode_traced_paged(self, token_ids, current_pos, page_table):
        """Replay captured paged trace with updated inputs. All params are host types."""
        assert self._trace_id is not None, "Call capture_decode_trace_paged first"
        assert current_pos < self.args.max_seq_len, f"Position {current_pos} >= max_seq_len {self.args.max_seq_len}"

        token_host = ttnn.from_torch(token_ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(token_host, self._trace_token_ids)

        cos_host, sin_host = self.rope.get_cos_sin_host(current_pos)
        ttnn.copy_host_to_device_tensor(cos_host, self._trace_cos)
        ttnn.copy_host_to_device_tensor(sin_host, self._trace_sin)

        cur_pos_host = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        ttnn.copy_host_to_device_tensor(cur_pos_host, self._trace_cur_pos)

        if self._prev_page_table is None or not torch.equal(self._prev_page_table, page_table):
            page_table_host = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(page_table_host, self._trace_page_table)
            self._prev_page_table = page_table.clone()

        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.device)

        return self._trace_output

    def capture_prefill_trace_paged(self, device, page_table, bucket_size=4096, chunk_size=2048):
        """Capture a trace for paged prefill at a fixed bucket size.

        Pre-allocates persistent input buffers (token_ids, page_table,
        chunk_page_table_full) and a shared GDN output buffer, then runs one
        warmup prefill (program-cache prime) followed by a traced prefill.

        Args:
            device: tt-metal device
            page_table: torch.Tensor [B, max_blocks] int32 used during capture.
                Must remain valid (or be replaced via copy_host_to_device) for replay.
            bucket_size: T to capture for. Replay must pad inputs to this length.
            chunk_size: GDN chunk size (must divide bucket_size). Default 2048.
        """
        assert self._deltanet_external_states is not None, "Call allocate_kv_caches first"
        assert bucket_size % chunk_size == 0, f"bucket_size {bucket_size} must be multiple of chunk_size {chunk_size}"

        B = 1
        block_size = 64
        attn_chunk_size = max(chunk_size, 4096)
        num_attn_chunks = math.ceil(bucket_size / attn_chunk_size)

        # Release prior trace if any.
        if self._prefill_trace_id is not None:
            ttnn.release_trace(device, self._prefill_trace_id)
            self._prefill_trace_id = None
            if self._prefill_trace_inputs:
                for key in ("token_ids", "page_table"):
                    if self._prefill_trace_inputs.get(key) is not None:
                        ttnn.deallocate(self._prefill_trace_inputs[key])
                for buf in self._prefill_trace_inputs.get("chunk_page_tables", []):
                    if buf is not None:
                        ttnn.deallocate(buf)
            self._prefill_trace_inputs = None

        # Persistent input buffers. Sized for the bucket; replay copies inputs into these.
        token_ids_dummy = torch.zeros(B, bucket_size, dtype=torch.int32)
        token_ids_buf = ttnn.from_torch(token_ids_dummy, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        page_table_buf = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        # One chunk_page_table buffer per full-attn chunk. For bucket > attn_chunk_size,
        # SDPA prefill operates on a chunk at a time; each chunk needs its own page_table
        # slice. Buffers are sized to their actual chunk's block count (so the SDPA program
        # for the partial last chunk has correct input shape).
        chunk_page_table_bufs = []
        for i in range(num_attn_chunks):
            chunk_start = i * attn_chunk_size
            chunk_end = min(chunk_start + attn_chunk_size, bucket_size)
            blocks_start = chunk_start // block_size
            blocks_end = math.ceil(chunk_end / block_size)
            chunk_page_table_bufs.append(
                ttnn.from_torch(
                    page_table[:, blocks_start:blocks_end].contiguous(),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                )
            )

        self._prefill_bucket_size = bucket_size

        # ---- 1. Warmup OUTSIDE trace mode ----
        # Run a normal prefill to compile every program at the bucket size. We do
        # this BEFORE setting _prefill_trace_inputs so prefill_paged takes its
        # legacy allocate-fresh path. This compiles all kernels and primes the
        # program cache. State and the persistent external DN buffers will be
        # zeroed afterward.
        dummy_ids = torch.zeros(B, bucket_size, dtype=torch.long)
        _ = self.prefill_paged(dummy_ids, page_table)
        ttnn.synchronize_device(device)

        # ---- 2. Allocate the GDN output buffer (shared across all GDN layers) ----
        first_dn = next(layer.attention for layer in self.layers if not layer.is_full_attention)
        Nv = first_dn.num_v_heads
        Dv = first_dn.head_v_dim
        num_pairs = B * Nv
        gdn_output_buf = ttnn.zeros(
            [num_pairs * chunk_size, 1, Dv],
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            dn._trace_prefill_output = gdn_output_buf
            dn.use_inplace_state = True

        # ---- 3. Allocate zero buffers and reset DN state in place ----
        # Note: warmup left dn.fused_conv_state pointing at a NON-persistent buffer
        # (slice of x_padded). Restore it to the external persistent buffer first
        # so subsequent in-place writes always target the same address.
        for layer, (ext_rec, ext_conv) in zip(
            (l for l in self.layers if not l.is_full_attention),
            self._deltanet_external_states,
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
        self._init_dn_zero_buffers()

        # Now activate trace mode (persistent inputs + reset_state no-op)
        self._prefill_trace_inputs = {
            "token_ids": token_ids_buf,
            "page_table": page_table_buf,
            "chunk_page_tables": chunk_page_table_bufs,
            "gdn_output": gdn_output_buf,
        }
        self._reset_dn_state_inplace()

        # ---- 4. Capture the trace ----
        # Inside the trace: prefill_paged returns the full last-layer hidden state
        # [1, bucket, hidden_size]. We then gather a single row at position given
        # by the index buffer, run rms_norm + lm_head, and the trace's output is a
        # tiny [1, 1, vocab_size] logit tensor — independent of bucket size.
        self._prefill_trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self._prefill_trace_logits = self.prefill_paged(dummy_ids, page_table)
        ttnn.end_trace_capture(device, self._prefill_trace_id, cq_id=0)
        logger.info("Prefill trace captured successfully!")

    def prefill_traced_paged(self, token_ids, page_table, actual_len):
        """Replay captured prefill trace.

        `token_ids` must be padded to `bucket_size`. `actual_len` is the number
        of REAL tokens in the prompt (excludes padding) — used to extract the
        next-token logit at the right position.

        Returns: ttnn.Tensor (host) with shape [1, 1, vocab_size] — the logit
        for the token AFTER position `actual_len-1`.

        Note: releases the trace after one replay (sets `_prefill_trace_id` to
        None). For repeated traced prefill, re-capture before each call.
        """
        assert self._prefill_trace_id is not None, "Call capture_prefill_trace_paged first"
        bucket = self._prefill_bucket_size
        T = token_ids.shape[1]
        assert T == bucket, f"token_ids T={T} != bucket {bucket}; pad before calling"
        assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"

        self._reset_dn_state_inplace()

        token_host = ttnn.from_torch(token_ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(token_host, self._prefill_trace_inputs["token_ids"])
        page_table_host = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(page_table_host, self._prefill_trace_inputs["page_table"])
        block_size = 64
        attn_chunk_size = max(2048, 4096)  # mirrors prefill_layer_chunked
        for i, chunk_pt_buf in enumerate(self._prefill_trace_inputs["chunk_page_tables"]):
            chunk_start = i * attn_chunk_size
            chunk_end = min(chunk_start + attn_chunk_size, bucket)
            blocks_start = chunk_start // block_size
            blocks_end = math.ceil(chunk_end / block_size)
            chunk_pt_host = ttnn.from_torch(
                page_table[:, blocks_start:blocks_end].contiguous(),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(chunk_pt_host, chunk_pt_buf)

        ttnn.execute_trace(self.device, self._prefill_trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.device)

        # Trace output is the full last-layer hidden state [1, bucket, hidden_size].
        # Slice at actual_len-1 and run rms_norm + lm_head OUTSIDE the trace. ttnn.gather
        # would let us do the slice inside the trace, but its impl calls ttnn::slice +
        # fill_implicit_tile_padding (both host-allocating) which FATAL during capture.
        # The post-trace slice produces an "Allocating device buffers is unsafe" warning
        # but that's benign (warning, not FATAL).
        hidden = self._prefill_trace_logits  # [1, bucket, hidden_size], TILE
        x_last = hidden[:, actual_len - 1 : actual_len, :]
        x_last = ttnn.to_layout(x_last, ttnn.TILE_LAYOUT)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = rms_norm_ttnn(x_last, self.norm_weight, eps=self.norm_eps)
        logits = ttnn.linear(x_last, self.lm_head_weight)

        ttnn.release_trace(self.device, self._prefill_trace_id)
        self._prefill_trace_id = None
        return logits.cpu()

    def _forward_prefill_chunk(
        self, token_buf, cos_buf, sin_buf, chunk_start_idx_tensor, full_page_table, chunk_page_table
    ):
        """Device-facing single-chunk prefill forward. ALL inputs are persistent device
        buffers (trace-safe: no host->device transfers inside). Processes one chunk through
        all layers, updating the paged KV caches and GDN recurrent/conv state IN PLACE.
        Returns the chunk's last-layer hidden state [1, chunk_size, hidden_size]."""
        x = ttnn.embedding(token_buf, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
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

    def capture_prefill_trace_chunked(self, device, page_table, chunk_size=2048):
        """Capture ONE chunk's all-layer prefill forward as a single trace, replayed per
        chunk by prefill_traced_chunked.

        Chunk-outer prefill (each chunk through all layers, KV + GDN state carried across
        chunks) is mathematically equivalent to the layer-outer whole-sequence prefill, but
        the captured trace holds only ONE chunk's dispatches instead of all of them — keeping
        it under tt-metal's 4 GiB uint32 trace-size ceiling at long context, while every GDN
        call stays at the correct 16-sub-chunk (2048-token) size. Attention uses the flexible
        chunked SDPA (chunk_start_idx as a runtime device tensor) so one trace serves every
        chunk position.

        Args:
            device: tt-metal device.
            page_table: torch.Tensor [1, max_blocks] int32. The full table is used by SDPA;
                per-chunk block slices are used by paged_fill_cache.
            chunk_size: tokens per chunk (must be a multiple of 128, the GDN sub-chunk size).
        """
        assert self._deltanet_external_states is not None, "Call allocate_kv_caches first"
        assert chunk_size % 128 == 0, f"chunk_size {chunk_size} must be a multiple of 128"
        B = 1
        block_size = 64
        blocks_per_chunk = chunk_size // block_size

        if self._chunked_trace_id is not None:
            ttnn.release_trace(device, self._chunked_trace_id)
            self._chunked_trace_id = None

        self._chunked_chunk_size = chunk_size

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

        # ---- Point GDN layers at the persistent external state buffers and enable
        #      in-place state carry across replays. ----
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

        # ---- Warmup OUTSIDE the trace: compile every per-chunk program. ----
        self._reset_dn_state_inplace()
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

        # ---- Capture the trace. ----
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
        logger.info("Chunked prefill trace captured successfully!")

    def _forward_prefill_chunk_eager(self, token_slice, chunk_start, page_table):
        """Eager (non-traced) single-chunk prefill forward for the FINAL partial chunk.

        Processes `token_slice` (the real tail, < chunk_size tokens) through all layers,
        updating the paged KV caches + GDN recurrent/conv state IN PLACE. The chunk-seq GDN
        zero-pads internally to the next multiple of 128, so the post-prefill state matches
        the non-traced path's minimal padding — NOT the bucket padding (repeated last token),
        which would wash out the recurrent state that decode continues from. Returns the
        chunk's last-layer hidden state [1, T_tail_padded, hidden_size]."""
        T_tail = token_slice.shape[1]
        block_size = 64
        tok = ttnn.from_torch(
            token_slice.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        x = ttnn.embedding(tok, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tok)
        cos, sin = self.rope.get_rot_mats(torch.arange(chunk_start, chunk_start + T_tail).unsqueeze(0))
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

    def prefill_traced_chunked(self, token_ids, page_table, actual_len):
        """Prefill by replaying the captured per-chunk trace for each FULL 2048-token chunk,
        then processing the final partial chunk eagerly with minimal padding.

        Only the real prompt (token_ids[:, :actual_len]) is processed; any bucket padding in
        token_ids is ignored. Full chunks (num_full = actual_len // chunk_size) are replayed
        from the trace; the remaining tail (< chunk_size tokens) is run eagerly so the GDN
        kernel zero-pads it to the next multiple of 128 (matching the non-traced path) instead
        of repeating the bucket padding through the recurrence — which corrupts the decode
        state at long context. actual_len is the real prompt length; the next-token logit is
        extracted at actual_len-1. Returns ttnn.Tensor (host) [1, 1, vocab_size].
        """
        assert self._chunked_trace_id is not None, "Call capture_prefill_trace_chunked first"
        chunk_size = self._chunked_chunk_size
        B, T = token_ids.shape
        assert 1 <= actual_len <= T, f"actual_len {actual_len} not in [1, {T}]"
        block_size = 64
        blocks_per_chunk = chunk_size // block_size
        num_full = actual_len // chunk_size
        tail_real = actual_len - num_full * chunk_size

        # Reset GDN state once; it then carries in place across the chunk replays + eager tail.
        self._reset_dn_state_inplace()
        pt_host = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(pt_host, self._chunk_full_page_table_buf)

        # ---- Replay the captured trace for each full 2048-token chunk of real tokens. ----
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

            cos_host = ttnn.from_torch(
                self.rope.cos_cpu[cs : cs + chunk_size].unsqueeze(0).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            sin_host = ttnn.from_torch(
                self.rope.sin_cpu[cs : cs + chunk_size].unsqueeze(0).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(cos_host, self._chunk_cos_buf)
            ttnn.copy_host_to_device_tensor(sin_host, self._chunk_sin_buf)

            ttnn.execute_trace(self.device, self._chunked_trace_id, cq_id=0, blocking=False)

        ttnn.synchronize_device(self.device)

        # ---- Final partial chunk (eager, minimal pad), or extract from the last full chunk
        #      when actual_len is an exact multiple of chunk_size. ----
        if tail_real > 0:
            cs = num_full * chunk_size
            hidden = self._forward_prefill_chunk_eager(token_ids[:, cs:actual_len], cs, page_table)
            pos_in_chunk = (actual_len - 1) - cs
        else:
            hidden = self._chunked_trace_output  # last full chunk's hidden state
            pos_in_chunk = (actual_len - 1) - (num_full - 1) * chunk_size
        ttnn.synchronize_device(self.device)

        x_last = hidden[:, pos_in_chunk : pos_in_chunk + 1, :]
        x_last = ttnn.to_layout(x_last, ttnn.TILE_LAYOUT)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = rms_norm_ttnn(x_last, self.norm_weight, eps=self.norm_eps)
        logits = ttnn.linear(x_last, self.lm_head_weight)
        return logits.cpu()

    def reset_state(self, batch_size=None):
        """Reset all layer states for a new sequence.

        In trace mode (`self._prefill_trace_inputs` set), skip in-place — caller
        must explicitly invoke `_reset_dn_state_inplace` BEFORE trace replay so
        the zeroing happens once outside the captured graph.
        """
        if self._prefill_trace_inputs is not None:
            return
        for layer in self.layers:
            if layer.is_full_attention:
                layer.attention.reset_cache()
            else:
                layer.attention.reset_state(batch_size)

    def _reset_dn_state_inplace(self):
        """Zero DN recurrent + conv state buffers in place (preserves addresses).

        Required between traced-prefill replays: every replay must start from
        zeroed state, but the buffers' addresses are baked into the captured
        trace and cannot change. Uses pre-allocated zero buffers as the source
        for `ttnn.copy`.
        """
        assert self._dn_zero_recurrent is not None, "Call _init_dn_zero_buffers first"
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            ttnn.copy(self._dn_zero_recurrent, dn.recurrent_state)
            ttnn.copy(self._dn_zero_conv, dn.fused_conv_state)
            # split_conv_state is rebuilt lazily on first decode; clear so it
            # gets rebuilt from the freshly-prefilled fused_conv_state.
            if dn.split_conv_state is not None:
                for buf in dn.split_conv_state:
                    ttnn.deallocate(buf)
                dn.split_conv_state = None

    def _init_dn_zero_buffers(self):
        """Allocate one shared zero buffer per DN state shape (recurrent and conv).

        Both shapes are uniform across all DN layers, so a single pair suffices.
        Called by capture_prefill_trace_paged.
        """
        if self._dn_zero_recurrent is not None:
            return
        # Find the first DN layer to get shapes
        first_dn = next(layer.attention for layer in self.layers if not layer.is_full_attention)
        rec_shape = list(first_dn.recurrent_state.shape)
        conv_shape = list(first_dn.fused_conv_state.shape)
        self._dn_zero_recurrent = ttnn.zeros(
            rec_shape,
            dtype=ttnn.bfloat16,
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

    def set_paged_kv_caches(self, kv_caches):
        """Attach paged KV caches to the 8 attention layers."""
        self._paged_kv_caches = kv_caches
        for cache_idx, layer_idx in enumerate(self._attention_layer_indices):
            k_cache, v_cache = kv_caches[cache_idx]
            self.layers[layer_idx].attention.set_paged_kv_cache(k_cache, v_cache)

    def allocate_kv_caches(self, kv_cache_shape, dtype, batch_size=1):
        """Allocate caches for all 32 layers. Returns only 8 attention KV caches (for vLLM)."""
        assert self._deltanet_external_states is None, "allocate_kv_caches already called; deallocate first"

        kv_caches = []
        for idx in self._attention_layer_indices:
            k_cache = ttnn.zeros(kv_cache_shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
            v_cache = ttnn.zeros(kv_cache_shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
            kv_caches.append([k_cache, v_cache])
        self.set_paged_kv_caches(kv_caches)

        self._deltanet_external_states = []
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                rec = ttnn.from_torch(
                    torch.zeros(batch_size, dn.num_v_heads, dn.head_k_dim, dn.head_v_dim, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
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
        B, T = token_ids.shape
        # Keep page_table as torch.Tensor for CPU slicing in prefill_layer_chunked.
        page_table_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        self.reset_state(batch_size=B)

        # Use existing prefill (concat-based K/V for SDPA)
        if T > 1024:
            logits = self.prefill_layer_chunked(token_ids, chunk_size=2048, page_table=page_table_torch)
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

        # Trace mode: skip the post-prefill housekeeping that would re-allocate device
        # tensors during trace capture. The paged-fill is a no-op when prefill_layer_chunked
        # already used paged_sdpa (T > 1024 path), and DN states already live in the external
        # buffers because use_inplace_state=True.
        if self._prefill_trace_inputs is None:
            # Defensive fallback: if paged prefill was used, past_key is None and this is a no-op.
            # If concat path was used (T <= 1024), this copies concat KV into paged cache.
            page_table_device = ttnn.from_torch(
                page_table_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )
            self._fill_paged_cache_from_prefill(page_table_device)

            # Prepare DeltaNet for decode (fuse conv states)
            for layer in self.layers:
                if not layer.is_full_attention:
                    dn = layer.attention
                    if dn.fused_conv_state is None and dn.conv_state_q is not None:
                        dn.fused_conv_state = ttnn.concat([dn.conv_state_q, dn.conv_state_k, dn.conv_state_v], dim=2)
                        dn.fused_conv_state = ttnn.to_layout(dn.fused_conv_state, ttnn.TILE_LAYOUT)

            # Copy DeltaNet states back into external (pre-allocated) buffers so trace can see them
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

        return logits

    def decode_paged(self, token_ids, current_pos, page_table):
        """Single-token decode using paged KV cache.

        Args:
            token_ids: torch.Tensor [B, 1] token IDs
            current_pos: int -- current position in the sequence
            page_table: torch.Tensor or ttnn.Tensor [B, max_blocks_per_seq] int32
        Returns:
            logits: ttnn.Tensor [B, 1, vocab_size]
        """
        B = token_ids.shape[0]
        # Accept host torch.Tensor or device ttnn.Tensor for page_table
        if isinstance(page_table, torch.Tensor):
            page_table = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

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
                    dn._restore_split_conv_from_fused()
                idx += 1
