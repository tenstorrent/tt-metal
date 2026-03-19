# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2.5 greedy-decode generator with Metal trace support.

Supports both paged and non-paged attention modes.

Trace-safe decode: entire decode forward pass is captured as a Metal trace
and replayed each step. Only host↔device data transfers between replays:
  - token ID (uint32)
  - position (int32 + uint32)

Usage:
    gen = TtMiniMaxGenerator(model, device, max_seq_len=2048)
    tokens = gen.generate(prompt_ids, max_new_tokens=128, use_trace=True)
"""

import torch

import ttnn
from models.tt_transformers.tt.common import PagedAttentionConfig

from .model import TtMiniMaxModel


def _mesh_mapper(device):
    if isinstance(device, ttnn.MeshDevice):
        return ttnn.ReplicateTensorToMesh(device)
    return None


def create_page_table(
    batch_size: int,
    paged_attention_config: PagedAttentionConfig,
) -> torch.Tensor:
    """Create a simple sequential page table mapping.

    Args:
        batch_size: Number of users in the batch
        paged_attention_config: Paged attention configuration

    Returns:
        [batch_size, max_blocks_per_user] page table tensor
    """
    max_num_blocks = paged_attention_config.max_num_blocks
    blocks_per_user = max_num_blocks // batch_size

    # Simple sequential assignment: user i gets blocks [i*blocks_per_user, (i+1)*blocks_per_user)
    page_table = torch.zeros(batch_size, blocks_per_user, dtype=torch.int32)
    for b in range(batch_size):
        page_table[b] = torch.arange(
            b * blocks_per_user,
            (b + 1) * blocks_per_user,
            dtype=torch.int32,
        )
    return page_table


class TtMiniMaxGenerator:
    """Greedy autoregressive generator with Metal trace support.

    Sequence of operations:
      1. clear_kv_caches + prefill(prompt_ids)
      2. Compile decode (warmup run)
      3. Capture Metal trace of decode forward
      4. Decode loop: update inputs → replay trace → read logits
    """

    def __init__(
        self,
        model: TtMiniMaxModel,
        device,
        max_seq_len: int = 2048,
        batch: int = 1,
    ):
        self.model = model
        self.device = device
        self.max_seq_len = max_seq_len
        self.batch = batch

        # Decode trace state
        self._trace_id = None
        self._trace_token_input = None
        self._trace_position_int32 = None
        self._trace_position_uint32 = None
        self._trace_logits_output = None

        # Prefill trace state (keyed by sequence length)
        self._prefill_traces = {}  # {seq_len: (trace_id, input_buffer, output_buffer)}

        # Paged attention support
        self._use_paged_attention = model.paged_attention_config is not None
        self._page_table_host = None
        self._page_table_device = None

        if self._use_paged_attention:
            self._init_page_table()

    # ------------------------------------------------------------------
    # Paged attention page table
    # ------------------------------------------------------------------

    def _init_page_table(self):
        """Initialize page table for paged attention."""
        paged_config = self.model.paged_attention_config
        self._page_table_host = create_page_table(self.batch, paged_config)
        rep = _mesh_mapper(self.device)
        self._page_table_device = ttnn.from_torch(
            self._page_table_host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=rep,
        )

    # ------------------------------------------------------------------
    # Persistent device buffers for trace
    # ------------------------------------------------------------------

    def _allocate_trace_buffers(self):
        """Allocate persistent device tensors for trace input/output."""
        B = self.batch
        rep = _mesh_mapper(self.device)

        self._trace_token_input = ttnn.from_torch(
            torch.zeros(1, 1, B, 1, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=rep,
        )

        self._trace_position_int32 = ttnn.from_torch(
            torch.zeros(B, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=rep,
        )

        self._trace_position_uint32 = ttnn.from_torch(
            torch.zeros(B, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=rep,
        )

    def _update_trace_inputs(self, token_id: int, cur_pos: int):
        """Update persistent device buffers with new values for trace replay."""
        B = self.batch

        token_host = ttnn.from_torch(
            torch.tensor([[[[token_id]]]], dtype=torch.int32).expand(1, 1, B, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(token_host, self._trace_token_input)

        pos_host_i32 = ttnn.from_torch(
            torch.tensor([cur_pos], dtype=torch.int32).expand(B),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(pos_host_i32, self._trace_position_int32)

        pos_host_u32 = ttnn.from_torch(
            torch.tensor([cur_pos], dtype=torch.int32).expand(B),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(pos_host_u32, self._trace_position_uint32)

    # ------------------------------------------------------------------
    # Trace capture
    # ------------------------------------------------------------------

    def _capture_trace(self):
        """Capture Metal trace of one decode step."""
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)

        logits = self.model.forward_decode_trace(
            self._trace_token_input,
            self._trace_position_int32,
            self._trace_position_uint32,
            batch=self.batch,
            page_table=self._page_table_device,
        )

        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)

        self._trace_id = trace_id
        self._trace_logits_output = logits

    def _release_trace(self):
        """Release Metal trace resources."""
        if self._trace_id is not None:
            ttnn.release_trace(self.device, self._trace_id)
            self._trace_id = None
            self._trace_logits_output = None

    # ------------------------------------------------------------------
    # Prefill trace capture
    # ------------------------------------------------------------------

    def _get_prefill_trace(self, seq_len: int):
        """Get or create prefill trace for a specific sequence length.

        Args:
            seq_len: Sequence length to trace

        Returns:
            Tuple of (trace_id, input_buffer, output_logits_buffer)
        """
        if seq_len in self._prefill_traces:
            return self._prefill_traces[seq_len]

        # Capture new trace for this sequence length
        trace_data = self._capture_prefill_trace(seq_len)
        self._prefill_traces[seq_len] = trace_data
        return trace_data

    def _capture_prefill_trace(self, seq_len: int):
        """Capture Metal trace for prefill at a specific sequence length.

        Args:
            seq_len: Sequence length to trace

        Returns:
            Tuple of (trace_id, input_buffer, output_logits_buffer)
        """
        B = self.batch
        H = self.model.config.hidden_size
        rep = _mesh_mapper(self.device)

        # Allocate persistent input buffer for prefill [B, S, H] as embeddings
        input_buffer = ttnn.from_torch(
            torch.zeros(B, seq_len, H, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )

        # Get RoPE cos/sin for this sequence length
        cos, sin = self.model.rope.get_cos_sin(seq_len)

        # Warmup run
        warmup_output = self._prefill_forward_from_embeddings(input_buffer, cos, sin)
        warmup_output.deallocate(True)
        ttnn.synchronize_device(self.device)

        # Capture trace
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        output_logits = self._prefill_forward_from_embeddings(input_buffer, cos, sin)
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)

        cos.deallocate(True)
        sin.deallocate(True)

        return (trace_id, input_buffer, output_logits)

    def _prefill_forward_from_embeddings(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor):
        """Run prefill forward from embeddings (for trace capture).

        This is similar to model.forward_prefill but starts from embeddings
        instead of token IDs to allow trace capture.
        """
        B, S, H = x.shape[0], x.shape[1], x.shape[2]

        for layer in self.model.layers:
            x = layer.forward_prefill(x, cos, sin, user_id=0, page_table=self._page_table_device)

        x = self.model.norm(x)
        logits = ttnn.linear(x, self.model.lm_head, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits

    def _execute_prefill_trace(self, input_ids: torch.Tensor, seq_len: int):
        """Execute prefill using cached trace.

        Args:
            input_ids: [B, S] token IDs
            seq_len: Sequence length (must match traced length)

        Returns:
            Output logits tensor
        """
        trace_id, input_buffer, output_logits = self._get_prefill_trace(seq_len)

        # Embed tokens to get input embeddings
        B, S = input_ids.shape
        rep = _mesh_mapper(self.device)
        ids_tt = ttnn.from_torch(
            input_ids.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=rep,
        )
        embeddings = ttnn.embedding(ids_tt, self.model.embed_weight, layout=ttnn.TILE_LAYOUT)
        if len(embeddings.shape) == 3:
            embeddings = ttnn.unsqueeze_to_4D(embeddings)
        embeddings = ttnn.reshape(embeddings, (B, S, self.model.config.hidden_size))
        ids_tt.deallocate(True)

        # Copy embeddings to input buffer
        embeddings_host = ttnn.to_torch(embeddings)
        embeddings.deallocate(True)
        embeddings_host_tt = ttnn.from_torch(
            embeddings_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(embeddings_host_tt, input_buffer)

        # Execute trace
        ttnn.execute_trace(self.device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.device)

        return output_logits

    def _release_prefill_traces(self):
        """Release all prefill traces."""
        for seq_len, (trace_id, input_buffer, output_logits) in self._prefill_traces.items():
            ttnn.release_trace(self.device, trace_id)
            input_buffer.deallocate(True)
            # Note: output_logits may be shared, don't deallocate
        self._prefill_traces.clear()

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        eos_token_id: int | None = None,
        use_trace: bool = False,
        enable_prefill_trace: bool = False,
    ) -> torch.Tensor:
        """Greedy decode with variable ISL support.

        Args:
            prompt_ids:         [B, S] integer token ids.
            max_new_tokens:     maximum tokens to generate.
            eos_token_id:       stop when this token is generated.
            use_trace:          enable Metal trace capture/replay for decode.
            enable_prefill_trace: enable Metal trace capture/replay for prefill.

        Returns:
            [B, S + generated] token ids.
        """
        B, S = prompt_ids.shape

        # --- Prefill ---
        self.model.clear_kv_caches()

        if enable_prefill_trace:
            # Use traced prefill
            logits = self._execute_prefill_trace(prompt_ids, S)
        else:
            # Normal prefill
            logits = self.model.forward_prefill(prompt_ids, page_table=self._page_table_device)

        last_logits = self._extract_logits(logits, S)
        next_token = last_logits.argmax(dim=-1, keepdim=True)

        if not enable_prefill_trace:
            logits.deallocate(True)

        generated = [next_token]
        cur_pos = S

        if use_trace:
            return self._generate_with_trace(prompt_ids, next_token, cur_pos, generated, max_new_tokens, eos_token_id)
        else:
            return self._generate_no_trace(prompt_ids, next_token, cur_pos, generated, max_new_tokens, eos_token_id)

    def _generate_no_trace(self, prompt_ids, next_token, cur_pos, generated, max_new_tokens, eos_token_id):
        """Decode loop without trace (original path)."""
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            logits = self.model.forward_decode(next_token, cur_pos)
            next_token_out = self._extract_logits(logits, 1).argmax(dim=-1, keepdim=True)
            logits.deallocate(True)

            generated.append(next_token_out)
            next_token = next_token_out
            cur_pos += 1

        return torch.cat([prompt_ids, torch.cat(generated, dim=-1)], dim=-1)

    def _generate_with_trace(self, prompt_ids, next_token, cur_pos, generated, max_new_tokens, eos_token_id):
        """Decode loop with Metal trace capture/replay."""
        self._release_trace()

        # Allocate persistent buffers
        self._allocate_trace_buffers()

        # Warmup / compile run (populates program cache)
        self._update_trace_inputs(next_token.item(), cur_pos)
        warmup_logits = self.model.forward_decode_trace(
            self._trace_token_input,
            self._trace_position_int32,
            self._trace_position_uint32,
            batch=self.batch,
            page_table=self._page_table_device,
        )
        warmup_logits.deallocate(True)
        ttnn.synchronize_device(self.device)

        # Capture trace
        self._update_trace_inputs(next_token.item(), cur_pos)
        self._capture_trace()

        # Decode loop: replay trace each step
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # Update inputs for this step
            self._update_trace_inputs(next_token.item(), cur_pos)

            # Replay captured trace
            ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.device)

            # Read logits from persistent output buffer
            next_token_out = self._extract_logits(self._trace_logits_output, 1).argmax(dim=-1, keepdim=True)

            generated.append(next_token_out)
            next_token = next_token_out
            cur_pos += 1

        self._release_trace()

        return torch.cat([prompt_ids, torch.cat(generated, dim=-1)], dim=-1)

    def _extract_logits(self, logits: ttnn.Tensor, seq_len: int) -> torch.Tensor:
        """Pull logits to host. Handles mesh vs single device."""
        if self.model._is_mesh:
            return ttnn.to_torch(ttnn.get_device_tensors(logits)[0])[:, seq_len - 1, :]
        return ttnn.to_torch(logits)[:, seq_len - 1, :]

    def reset_caches(self):
        """Zero out device-resident KV caches for a fresh generation."""
        self._release_trace()
        self._release_prefill_traces()
        self.model.clear_kv_caches()

    def __call__(self, prompt_ids, **kwargs):
        return self.generate(prompt_ids, **kwargs)
