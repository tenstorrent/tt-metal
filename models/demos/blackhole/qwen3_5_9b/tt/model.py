# models/demos/blackhole/qwen3_5_9b/tt/model.py
"""Full Qwen3.5-9B text model for Blackhole P150.

Assembly: tok_embeddings → 32 × Qwen35DecoderLayer → RMSNorm → LM Head
Manages hybrid state: KV cache (8 attention layers) + recurrent state (24 DeltaNet layers).
"""
import math

import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen3_5_9b.tt.layer import Qwen35DecoderLayer
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.demos.blackhole.qwen3_5_9b.tt.rope import Qwen35RoPESetup
from models.tt_transformers.tt.common import Mode, get_block_size, num_blocks_in_seq


class Qwen35Model:
    """Qwen3.5-9B text-only language model on Blackhole P150.

    Usage:
        # HF_MODEL env var (hub name or local path) is the single source of truth.
        model = Qwen35Model.from_pretrained(device)
        logits = model.prefill(token_ids)
        logits = model.decode(token_id, position)
    """

    def __init__(self, mesh_device, args, state_dict, tensor_cache_path=None):
        self.args = args
        self.device = mesh_device
        self.mesh_device = mesh_device  # Generator reads model.mesh_device
        self.num_devices = mesh_device.get_num_devices()
        # CCL collective for multi-device all-reduce; None on single device (the
        # framework MLP/LMHead/all-reduce ops no-op when there is nothing to reduce).
        if self.num_devices > 1:
            from models.tt_transformers.tt.ccl import TT_CCL

            self.tt_ccl = TT_CCL(mesh_device)
        else:
            self.tt_ccl = None
        self.configuration = args  # Generator reads model.configuration.max_seq_len
        self.sampling = None  # host sampling only (no on-device sampler)
        self.sampling_dp = 1
        self._supports_on_device_sampling = False

        # Embedding — framework Embedding (mesh-aware: ShardTensor2dMesh(dims=(None,3))
        # replicates the table on a 1-device mesh, identical to the old single-device path).
        from models.tt_transformers.tt.embedding import Embedding

        self.embd = Embedding(
            mesh_device=mesh_device,
            args=args,
            weight_cache_path=tensor_cache_path,
            state_dict=state_dict,
            dtype=ttnn.bfloat16,
        )

        # RoPE setup (for gated attention layers only)
        self.rope = Qwen35RoPESetup(mesh_device, args)

        # Transformer layers
        logger.info(f"Loading {args.n_layers} transformer layers...")
        self.layers = []
        for i in tqdm(range(args.n_layers), desc="Loading layers"):
            layer = Qwen35DecoderLayer(mesh_device, args, state_dict, i, tensor_cache_path, tt_ccl=self.tt_ccl)
            self.layers.append(layer)

        # Final norm — framework RMSNorm (mesh-aware; applies the +1 zero-centered
        # offset internally via add_unit_offset=True).
        # NOTE (multi-device/TP handoff): is_distributed=None is correct on single device.
        # For 27B TP, the framework Embedding shards the hidden dim, so the hidden state
        # entering RMSNorm is sharded -> these norms must then pass is_distributed=args.is_distributed_norm
        # + tt_ccl=<the model's self.tt_ccl> (or wrap in tt_transformers DistributedNorm) to all-gather.
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
            # TP: the post-last-layer hidden state is fractured; DistributedNorm
            # gathers it back to a full replicated tensor for the LM head.
            from models.tt_transformers.tt.distributed_norm import DistributedNorm

            self.norm = DistributedNorm(self.norm, args, tt_ccl=self.tt_ccl, TG=args.is_galaxy)

        # LM Head — 2D [in, out] for ttnn.linear. On a single device the weight
        # is placed as-is; on a mesh it is REPLICATED (full vocab on every device)
        # so the full-dim norm output produces full logits without a gather. (A
        # vocab-sharded LM head + ConcatMeshToTensor is a later memory optimization.)
        lm_head_weight = state_dict["output.weight"].T.contiguous()  # [dim, vocab_size]
        self.lm_head_weight = ttnn.as_tensor(
            lm_head_weight,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=tensor_cache_path / "output.weight" if tensor_cache_path else None,
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)) if self.num_devices > 1 else {}),
        )

        self.vocab_size = args.vocab_size
        self._paged_kv_caches = None
        self._attention_layer_indices = [i for i in range(args.n_layers) if args.is_full_attention_layer(i)]
        self._deltanet_external_states = None  # list of (recurrent, conv) tuples, set by allocate_kv_caches
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

    def switch_mode(self, mode):
        """Generator calls this on mode change; Qwen has no prefetcher, so no-op."""
        return None

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
        return cls(device, args, state_dict, tensor_cache_path=cache_path)

    def prefill_tp(self, token_ids, valid_len=None):
        """Tensor-parallel full-model prefill (num_devices>1). Stateless: runs the
        whole sequence from scratch through the fractured-residual TP layers and
        returns the next-token logits at position valid_len-1.

        token_ids: torch [1, T] (pad T to a multiple of 128 for the GDN chunk
        kernel; right-padding does not affect the causal logit at valid_len-1).
        Returns ttnn logits [1, 1, 1, vocab_size] (host).
        """
        from models.demos.blackhole.qwen3_5_9b.tt.attention.rope_tp import rot_mats_prefill

        B, T = token_ids.shape
        assert B == 1, "prefill_tp is single-sequence"
        valid_len = valid_len or T

        tok = ttnn.from_torch(
            token_ids.to(torch.int32),
            dtype=ttnn.uint32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x = self.embd(tok)  # [1, T, dim_frac] (hidden dim sharded across mesh)
        x = ttnn.reshape(x, (1, 1, T, x.shape[-1]))
        cos, sin = rot_mats_prefill(self.device, self.args.rope_head_dim, T, self.args.rope_theta)

        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="prefill", chunk_size=128, valid_len=valid_len)

        x = self.norm(x, mode=Mode.PREFILL)  # DistributedNorm → full [1,1,T,dim]
        x_last = x[:, :, valid_len - 1 : valid_len, :]
        logits = ttnn.linear(x_last, self.lm_head_weight)  # replicated lm_head → full vocab (same on all devices)
        # Logits are replicated across the mesh; take one replica → torch [vocab_size].
        lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
        return lt[0].reshape(-1)[: self.vocab_size]

    def reset_tp(self):
        """Reset every TP layer's KV cache / GDN recurrent+conv state for a new sequence."""
        for layer in self.layers:
            layer.attention.reset_state()

    def decode_tp(self, token_id, pos):
        """Single-token TP decode at absolute position `pos` (B=1). Continues from
        the KV cache + GDN state left by prefill_tp / prior decode steps.
        Returns torch logits [vocab_size]."""
        from models.demos.blackhole.qwen3_5_9b.tt.attention.rope_tp import rot_mats_decode

        tok = ttnn.from_torch(
            torch.tensor([[int(token_id)]], dtype=torch.int32),
            dtype=ttnn.uint32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x = self.embd(tok)  # [1,1,dim_frac]
        x = ttnn.reshape(x, (1, 1, 1, x.shape[-1]))  # [1,1,B=1,dim_frac]
        cos, sin = rot_mats_decode(
            self.device,
            self.args.rope_head_dim,
            self.args.max_seq_len,
            self.args.rope_theta,
            torch.tensor([pos], dtype=torch.int32),
        )
        cur_pos_tt = ttnn.from_torch(
            torch.tensor([pos], dtype=torch.int32),
            dtype=ttnn.int32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="decode", position_tensor=cur_pos_tt)
        x = self.norm(x, mode=Mode.DECODE)
        logits = ttnn.linear(x, self.lm_head_weight)
        lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
        return lt[0].reshape(-1)[: self.vocab_size]

    def generate_tp(self, prompt_ids, max_new_tokens=20):
        """Stateful TP generation (num_devices>1): prefill the prompt (fills KV cache
        + GDN state) then greedily decode one token at a time. Returns list of new ids."""
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

    def prefill(self, token_ids):
        B, T = token_ids.shape

        if T > 1024:
            return self.prefill_layer_chunked(token_ids, chunk_size=2048)

        # Original path for short sequences
        self.reset_state(batch_size=B)

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = self.embd(token_ids_ttnn)

        position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        cos, sin = self.rope.get_rot_mats(position_ids)

        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="prefill")

        x = self.norm(x, mode=Mode.PREFILL)

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

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = self.embd(token_ids_ttnn)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(token_ids_ttnn)

        # Attention layers can use larger chunks than DeltaNet — no Neumann series
        # limitation, and fewer chunks means fewer unique KV cache sizes for SDPA
        # compilation. 4096 = 4x fewer SDPA compilations vs chunk_size=1024.
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
                    # Paged prefill path
                    cos = self.rope.cos_device[:, chunk_start:chunk_end, :]
                    sin = self.rope.sin_device[:, chunk_start:chunk_end, :]

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
        logits = ttnn.linear(x_last, self.lm_head_weight)
        ttnn.deallocate(x)

        return logits

    def decode(self, token_ids, current_pos):
        B = token_ids.shape[0]

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = self.embd(token_ids_ttnn)
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

        x = self.norm(x, mode=Mode.DECODE)
        logits = ttnn.linear(x, self.lm_head_weight)
        ttnn.deallocate(x)

        return logits

    def _forward_decode(self, token_ids_buf, cos, sin, cur_pos_tensor, page_table):
        """Device-facing paged decode forward. ALL inputs are device tensors.
        Trace-safe: no host-device transfers inside this function.
        """
        x = self.embd(token_ids_buf)
        for layer in self.layers:
            if layer.is_full_attention:
                x = layer.forward(x, cos, sin, position_tensor=cur_pos_tensor, page_table=page_table, mode="decode")
            else:
                x = layer.forward(x, mode="decode")
        x = self.norm(x, mode=Mode.DECODE)
        logits = ttnn.linear(x, self.lm_head_weight)
        ttnn.deallocate(x)
        return logits

    def _forward_prefill_chunk(
        self, token_buf, cos_buf, sin_buf, chunk_start_idx_tensor, full_page_table, chunk_page_table
    ):
        """Device-facing single-chunk prefill forward. ALL inputs are persistent device
        buffers (trace-safe: no host->device transfers inside). Processes one chunk through
        all layers, updating the paged KV caches and GDN recurrent/conv state IN PLACE.
        Returns the chunk's last-layer hidden state [1, chunk_size, hidden_size]."""
        x = self.embd(token_buf)
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

    def capture_prefill_trace_chunked(self, device, page_table, chunk_size=2048, warmup_masked_buckets=True):
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
        block_size = get_block_size(self._paged_kv_caches)
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
        # NOTE (multi-device/TP handoff): these cos/sin trace buffers upload without a
        # mesh_mapper (single-device). For 27B TP, add mesh_mapper=ttnn.ReplicateTensorToMesh(device)
        # here for parity with tt/rope.py's replicated cos/sin tables.
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

        # ---- Compile the masked short-prompt bucket programs while still OUTSIDE the trace. ----
        # The GDN is already in in-place mode here, so these compile in the SAME state mode as
        # serving; doing it before begin_trace_capture means a real short prompt later replays
        # an already-compiled bucket instead of compiling (which would clobber the parked trace).
        # The dummy prefills dirty the in-place state + KV cache; the reset below re-zeros state
        # before capture, and real requests overwrite the cache.
        if warmup_masked_buckets:
            self.warmup_prefill_masked_buckets(page_table)

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
        x = self.embd(tok)
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

    # Fixed bucket lengths for the masked tail/short-prompt prefill. Every real length
    # rounds up to one of these, so only this many programs ever compile — small enough to
    # warm up before a trace is parked (vs the unbounded per-length eager tail). All are
    # multiples of the GDN sub-chunk (128) so the chunk kernel adds no internal pad. The
    # masked GDN path runs in DRAM (see gated_deltanet_forward_ttnn), which keeps bucket 512
    # off the L1 circular-buffer clash that the eager L1 path hits at that exact size.
    #
    # NOTE: this is a DELIBERATE divergence from the standard common.get_padded_prefill_len
    # bucket set {128, 1024, 2048, 4096, ...}. The small 256/512 buckets exist for short-prompt
    # TTFT, every entry is a 128-multiple for GDN sub-chunk alignment, and 512 must stay in the
    # masked-DRAM path (see above). The standard pad helper is for token-padding-tolerant softmax
    # attention; GDN needs these specific bucket boundaries plus the exact valid_len mask.
    _PREFILL_MASK_BUCKETS = (128, 256, 512, 1024, 2048)

    @classmethod
    def _mask_bucket_for(cls, length):
        """Smallest fixed bucket >= length (falls back to the next 128-multiple)."""
        for b in cls._PREFILL_MASK_BUCKETS:
            if length <= b:
                return b
        return ((length + 127) // 128) * 128

    def _forward_prefill_chunk_masked(self, token_buf, valid_len, chunk_start, page_table, bucket):
        """Single masked fixed-bucket prefill forward over `bucket` positions.

        token_buf is [1, bucket]: the first valid_len positions are real, the rest are
        right-padding (value irrelevant). Attention runs over the full bucket — padded query
        outputs are discarded and padded K/V land in cache slots past valid_len that decode
        overwrites before reading. GDN layers receive valid_len so the chunk kernel zeroes the
        padded positions out of the recurrent scan (identity updates) and captures the conv
        state at the real boundary, leaving both states decode-correct. Mirrors
        _forward_prefill_chunk_eager but at a fixed bucket length with masking. Returns the
        last-layer hidden state [1, bucket, hidden_size]."""
        block_size = get_block_size(self._paged_kv_caches)
        tok = ttnn.from_torch(
            token_buf.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        x = self.embd(tok)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tok)
        cos, sin = self.rope.get_rot_mats(torch.arange(chunk_start, chunk_start + bucket).unsqueeze(0))
        full_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        blk0 = chunk_start // block_size
        # Fill K/V for ONLY the real blocks (those covering valid_len), NOT the full bucket.
        # The bucket padding would otherwise write K/V into blocks past the request's allocation
        # — the page table is zero-padded there, so those writes land in block 0 and corrupt the
        # real KV. The padded query positions still attend over the full bucket, but their outputs
        # are discarded, so their K/V never needs to be persisted. Fill width = ceil(valid_len/64)
        # (chunk_start is block-aligned); it varies per request, so warmup compiles every width.
        blkN = num_blocks_in_seq(chunk_start + valid_len, block_size)
        chunk_pt = ttnn.from_torch(
            page_table[:, blk0:blkN].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        # Drive the full-attn SDPA via the FLEXIBLE chunked path (chunk_start as a runtime device
        # tensor) so ONE program per bucket serves every chunk_start. The host-int chunk_start path
        # compiles a distinct SDPA program per start position — for a tail at chunk_start>0 that
        # would compile at request time and clobber the parked trace. Warmed at chunk_start=0; the
        # value is runtime so it carries to any tail position.
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

    def prefill_masked_bucket(self, token_ids, page_table, actual_len, chunk_start=0, bucket=None):
        """Masked fixed-bucket prefill for a segment of `actual_len` real tokens.

        Pads the segment up to a fixed bucket length, runs all layers ONCE, and masks the GDN
        recurrent + conv state so they reflect exactly `actual_len` real tokens — numerically
        equivalent to the eager exact-length path (prefill_paged) but using one of only a few
        bucket-sized programs instead of compiling a fresh program per prompt length. That
        bounded program set is what makes warmup able to compile every code path before a trace
        is parked, so a short request can never trigger the compile-clobbers-trace hang.

        `chunk_start` is the segment's absolute start position (0 for a from-scratch short
        prompt; num_full*chunk_size for the tail of a long prompt — the carried GDN/KV state
        must already be in place). Returns ttnn.Tensor (host) [1, 1, vocab_size]: the logit
        after position actual_len-1.
        """
        B_batch, _ = token_ids.shape
        assert B_batch == 1, "masked-bucket prefill is single-sequence"
        if bucket is None:
            bucket = self._mask_bucket_for(actual_len)
        assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"

        if chunk_start == 0:
            # New sequence (from-scratch short prompt, or num_full==0 long prompt). Re-zero the
            # GDN state — the warmup-dirty-state guard. chunk_start>0 is a carried tail, so the
            # in-place GDN/KV state from the full chunks must NOT be reset here.
            self._reset_gdn_state_for_new_sequence()

        real = token_ids[:, :actual_len].to(torch.int32)
        if bucket > actual_len:
            pad = torch.zeros(1, bucket - actual_len, dtype=torch.int32)
            token_buf = torch.cat([real, pad], dim=1)
        else:
            token_buf = real

        hidden = self._forward_prefill_chunk_masked(token_buf, actual_len, chunk_start, page_table, bucket)
        ttnn.synchronize_device(self.device)

        # Select the last real position (actual_len-1) with a one-hot row matmul rather than a
        # static slice: the slice start would vary with the prompt length and compile a fresh
        # program each time, whereas the matmul's program is fixed per bucket (only the one-hot
        # values change). Keeps the whole masked path to one program set per bucket.
        sel = torch.zeros(1, 1, bucket, dtype=torch.float32)
        sel[0, 0, actual_len - 1] = 1.0
        sel_tt = ttnn.from_torch(sel, dtype=hidden.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        x_last = ttnn.matmul(sel_tt, hidden)
        ttnn.deallocate(sel_tt)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = self.norm(x_last, mode=Mode.PREFILL)
        logits = ttnn.linear(x_last, self.lm_head_weight)
        return logits.cpu()

    def warmup_prefill_masked_buckets(self, page_table, buckets=None):
        """Compile every masked-bucket prefill program up front (warmup).

        For each bucket runs two dummy prefills: one at the exact bucket length (the no-mask
        program) and one shorter (the masked program — the GDN mask multiply + the conv-state
        and logit one-hot matmuls). After this, a real short prompt of ANY length rounds up to
        an already-compiled bucket and never compiles at request time — the root cause of the
        trace-clobber hang. MUST run while the GDN is in its serving state mode and BEFORE any
        trace is parked; capture_prefill_trace_chunked calls this just before begin_trace_capture.
        Requires page_table to cover the largest bucket (max 2048 -> 32 blocks of 64)."""
        if buckets is None:
            buckets = self._PREFILL_MASK_BUCKETS
        block_size = get_block_size(self._paged_kv_caches)
        # The masked GDN/SDPA/sel programs key on the bucket; paged_fill_cache keys on the
        # real-block FILL WIDTH = ceil(valid_len/64), which a real prompt/tail can land on at any
        # value in 1..max_width. Warm each width via a vlen=width*block_size: w*64 rounds to its
        # bucket and produces fill width w. This sweep also covers, per bucket, both the no-mask
        # variant (vlen == bucket) and the masked variant (vlen < bucket).
        max_width = max(buckets) // block_size
        for w in range(1, max_width + 1):
            vlen = w * block_size
            b = self._mask_bucket_for(vlen)
            toks = torch.zeros(1, vlen, dtype=torch.int32)
            self.prefill_masked_bucket(toks, page_table, actual_len=vlen, bucket=b)
        ttnn.synchronize_device(self.device)

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
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_chunk = chunk_size // block_size
        num_full = actual_len // chunk_size
        tail_real = actual_len - num_full * chunk_size

        # Short prompt (no full chunks): route the whole prompt through the SAME masked
        # fixed-bucket path the long-prompt tail uses. chunk_start=0 makes prefill_masked_bucket
        # do the sequence-start GDN reset and run one masked forward — there is no trace to replay,
        # so the chunk-input plumbing below is skipped. This is the single bucketed+masked path
        # shared by short prompts and the long-prompt tail; prefill_dispatch routes every traced
        # prefill here so the short/long seam is defined once.
        if num_full == 0:
            return self.prefill_masked_bucket(
                token_ids[:, :actual_len], page_table, actual_len=actual_len, chunk_start=0
            )

        # >= 1 full chunk: re-zero GDN state once (the warmup-dirty-state guard); it then carries
        # in place across the chunk replays + the masked tail (whose chunk_start>0 skips the reset).
        self._reset_gdn_state_for_new_sequence()
        # copy_host_to_device requires an EXACT shape match. The full page-table buffer width
        # was fixed at trace capture, but vLLM pads request page tables to its own
        # max_num_blocks_per_req, which can differ from the captured width (e.g. off-by-one vs
        # the allocated block count). Pad/clip page_table to the buffer width; the trailing
        # entries index blocks beyond the prompt and are never read by SDPA (causal, up to
        # actual_len). No-op when the widths already match (e.g. the demo/tests).
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

        # ---- Final partial chunk: run it through the masked fixed-bucket path. It rounds the
        #      tail up to a warmed bucket and masks the GDN, so (unlike the old eager tail) it
        #      compiles no new program at request time and can't clobber the parked trace, while
        #      the in-place GDN/KV state carried across the replays continues correctly
        #      (chunk_start = cs skips the state reset). When actual_len is an exact multiple of
        #      chunk_size there is no tail — extract the next-token logit from the last full
        #      chunk's hidden state instead. ----
        if tail_real > 0:
            cs = num_full * chunk_size
            return self.prefill_masked_bucket(
                token_ids[:, cs:actual_len], page_table, actual_len=tail_real, chunk_start=cs
            )
        hidden = self._chunked_trace_output  # last full chunk's hidden state
        pos_in_chunk = (actual_len - 1) - (num_full - 1) * chunk_size
        ttnn.synchronize_device(self.device)

        x_last = hidden[:, pos_in_chunk : pos_in_chunk + 1, :]
        x_last = ttnn.to_layout(x_last, ttnn.TILE_LAYOUT)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = self.norm(x_last, mode=Mode.PREFILL)
        logits = ttnn.linear(x_last, self.lm_head_weight)
        return logits.cpu()

    def reset_state(self, batch_size=None):
        """Reset all layer states for a new sequence.

        The chunk-outer prefill trace zeroes DN state out-of-band via
        `_reset_dn_state_inplace`, so this is only the normal (eager / pre-trace) reset.
        """
        for layer in self.layers:
            if layer.is_full_attention:
                layer.attention.reset_cache()
            else:
                layer.attention.reset_state(batch_size)

    def _reset_gdn_state_for_new_sequence(self):
        """Zero the GDN recurrent + conv state at the start of every new sequence.

        This is the guard that makes the standard two-pass trace capture safe. Both the prefill
        and the decode trace captures run the forward TWICE (a compile run + the capture run);
        for softmax + paged KV that is idempotent, but each pass advances GDN's recurrent
        accumulation, so after warmup the bound GDN buffers hold residual dummy state (captured at
        pos 0 on zeroed inputs). Every real sequence MUST re-zero them before consuming any token,
        or that residue would leak into the first request's recurrent state. (The warmup capture
        also dirties paged-KV block 0, but that is benign: the first real prefill overwrites those
        slots via paged_fill_cache before decode reads them.)

        When the GDN runs on externally-bound in-place buffers — the vLLM/traced flow, whose
        addresses the decode trace baked in — zero them in place via _reset_dn_state_inplace so the
        addresses are preserved. Otherwise (eager/non-traced) reassign via reset_state. Detect via
        the per-layer flag set when the trace is parked, so warmup (which runs before
        begin_trace_capture) takes the same path as serving.
        """
        inplace = any(
            (not l.is_full_attention) and getattr(l.attention, "_chunk_inplace_state", False) for l in self.layers
        )
        if inplace:
            self._reset_dn_state_inplace()
        else:
            self.reset_state(batch_size=1)

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
        Called by capture_prefill_trace_chunked.
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
            x = self.embd(token_ids_ttnn)
            ttnn.deallocate(token_ids_ttnn)

            position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
            cos, sin = self.rope.get_rot_mats(position_ids)

            for layer in self.layers:
                x = layer.forward(x, cos=cos, sin=sin, mode="prefill")

            x = self.norm(x, mode=Mode.PREFILL)
            x_last = x[:, -1:, :]
            logits = ttnn.linear(x_last, self.lm_head_weight)
            ttnn.deallocate(x)

        # Post-prefill housekeeping. Paged-fill is a no-op when prefill_layer_chunked already
        # used paged_sdpa (T > 1024 path); for the concat path (T <= 1024) it copies concat KV
        # into the paged cache.
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

        # Copy DeltaNet states back into external (pre-allocated) buffers
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
        x = self.embd(token_ids_ttnn)
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

        x = self.norm(x, mode=Mode.DECODE)
        logits = ttnn.linear(x, self.lm_head_weight)
        ttnn.deallocate(x)

        return logits

    # -------------------------------------------------------------------------
    # Generator contract — decode half
    # -------------------------------------------------------------------------

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """Build HOST ttnn tensors for one decode step.

        Returns a tuple in the order ttnn_decode_forward consumes them:
            (tokens_tt, cur_pos_tt, rope_packed, page_table_tt)
        All tensors are HOST (no device) so copy_host_to_device can move the
        whole tuple in one call.
        """
        from models.demos.blackhole.qwen3_5_9b.tt.generator_interface import pack_rope_host

        B = tokens.shape[0]
        tokens_tt = ttnn.from_torch(tokens.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pos = current_pos[0].item() if isinstance(current_pos, torch.Tensor) else int(current_pos)
        cos_host, sin_host = self.rope.get_cos_sin_host(pos)  # HOST ttnn tensors [1,1,rope_head_dim]
        rope_packed = pack_rope_host(cos_host, sin_host)  # torch-based (host)
        cur_pos_tt = ttnn.from_torch(
            torch.full((B,), pos, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        page_table_tt = (
            ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            if page_table is not None
            else None
        )
        return tokens_tt, cur_pos_tt, rope_packed, page_table_tt

    def prepare_inputs_decode(self, tokens, current_pos, page_table=None):
        """Host-to-device transfer for decode inputs.

        Calls prepare_decode_inputs_host then copy_host_to_device, returning
        device tensors in the same order (tokens, cur_pos, rope_packed, page_table).
        """
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
        sampling_on_device=False,
        capture_sampling_trace=False,
        **kwargs,
    ):
        """Generator-contract decode forward.

        Unpacks the packed rope tensor (rot_mat_idxs) and delegates to the
        trace-safe _forward_decode. GDN and attention KV state are model-bound
        (set by allocate_kv_caches), so kv_cache is accepted but unused.

        Returns: (logits, None)
        """
        from models.demos.blackhole.qwen3_5_9b.tt.generator_interface import unpack_rope

        cos, sin = unpack_rope(rot_mat_idxs)
        logits = self._forward_decode(tokens, cos, sin, current_pos, page_table)
        return logits, None

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False, is_log_probs=False):
        """Convert decode logits from device ttnn to a host float tensor.

        Qwen's _forward_decode returns 3D logits [B, 1, vocab_size]; slice accordingly.
        On-device sampling / log-probs are not supported by this port (host sampling only,
        ``_supports_on_device_sampling=False``), so the is_tokens / is_log_probs branches the
        reference handles never fire here. Assert rather than silently return wrong-shaped data.
        """
        assert not (is_tokens or is_log_probs), "on-device sampling/log-probs unsupported (host sampling only)"
        out = ttnn.to_torch(tt_out).float()
        return out[:B, :S, : self.args.vocab_size].view(B, S, -1)

    def _save_deltanet_states(self):
        """Snapshot DeltaNet recurrent + conv states to host.

        Used to guard the GDN in-place recurrent state across the stock Generator's
        decode-trace capture: the capture runs the forward twice (compile + capture),
        each advancing the recurrent state non-idempotently. Snapshot before capture
        and restore after (see generator_interface.prime_decode_trace) so the replay
        loop starts from the correct post-prefill state.
        """
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
        """Restore DeltaNet states via ttnn.copy into the original buffers (preserves
        addresses, so a captured trace that baked those addresses stays valid)."""
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

    # -------------------------------------------------------------------------
