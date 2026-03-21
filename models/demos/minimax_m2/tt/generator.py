# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2.5 generator with Metal trace support and sampling.

Supports both paged and non-paged attention modes.
Supports sampling (temperature, top_k, top_p) and repetition penalty.

Trace-safe decode: entire decode forward pass is captured as a Metal trace
and replayed each step. Only host↔device data transfers between replays:
  - token ID (uint32)
  - position (int32 + uint32)

Usage:
    gen = TtMiniMaxGenerator(model, device, max_seq_len=2048)
    # Greedy decode (default)
    tokens = gen.generate(prompt_ids, max_new_tokens=128, use_trace=True)
    # With sampling
    tokens = gen.generate(prompt_ids, max_new_tokens=128, use_trace=True,
                          temperature=0.7, top_p=0.9, repetition_penalty=1.1)
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import ttnn
from models.tt_transformers.tt.common import PagedAttentionConfig

from .model import TtMiniMaxModel


@dataclass
class SamplingParams:
    """Sampling parameters for text generation."""

    temperature: float = 1.0  # 0.0 = greedy, higher = more random
    top_k: int = 0  # 0 = disabled, >0 = keep top k tokens
    top_p: float = 1.0  # 1.0 = disabled, <1.0 = nucleus sampling
    repetition_penalty: float = 1.0  # 1.0 = disabled, >1.0 = penalize repeated tokens


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: list[torch.Tensor],
    repetition_penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty to logits based on previously generated tokens.

    Args:
        logits: [B, V] logits tensor
        generated_tokens: List of [B, 1] tensors of previously generated token IDs
        repetition_penalty: Penalty factor (1.0 = no penalty, >1.0 = penalize)

    Returns:
        [B, V] logits with penalty applied
    """
    if repetition_penalty == 1.0 or not generated_tokens:
        return logits

    # Collect all generated token IDs
    all_tokens = torch.cat(generated_tokens, dim=-1)  # [B, num_generated]

    # Apply penalty: divide positive logits, multiply negative logits
    for b in range(logits.shape[0]):
        unique_tokens = all_tokens[b].unique()
        for token_id in unique_tokens:
            if logits[b, token_id] > 0:
                logits[b, token_id] /= repetition_penalty
            else:
                logits[b, token_id] *= repetition_penalty

    return logits


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample tokens from logits with temperature, top-k, and top-p filtering.

    Args:
        logits: [B, V] logits tensor
        temperature: Temperature for sampling (0.0 = greedy)
        top_k: Keep only top k tokens (0 = disabled)
        top_p: Keep tokens with cumulative probability <= top_p (1.0 = disabled)

    Returns:
        [B, 1] sampled token IDs
    """
    # Greedy decoding
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # Sample from distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


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
        """Capture Metal trace of one decode step.

        The trace includes only the forward pass. Argmax is done on host after
        trace replay. To add device-resident argmax, use a separate sampling
        trace (see SamplingGenerator in models/common/sampling/generator.py).
        """
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

        NOTE: Currently we recapture the trace for each call because reusing
        a cached prefill trace causes decode to hang. This is a workaround -
        the root cause needs investigation (may be related to KV cache state
        after trace replay vs trace capture).

        Args:
            seq_len: Sequence length to trace

        Returns:
            Tuple of (trace_id, token_buffer, output_logits_buffer)
        """
        # Release old trace if exists (we recapture each time as workaround)
        if seq_len in self._prefill_traces:
            old_trace_id, old_token_buffer, _ = self._prefill_traces[seq_len]
            ttnn.release_trace(self.device, old_trace_id)
            old_token_buffer.deallocate(True)

        # Capture new trace for this sequence length
        trace_data = self._capture_prefill_trace(seq_len)
        self._prefill_traces[seq_len] = trace_data
        return trace_data

    def _capture_prefill_trace(self, seq_len: int):
        """Capture Metal trace for prefill at a specific sequence length.

        Following tt_transformers pattern:
        1. Create host tensor (device=None)
        2. Warmup: copy to device → forward (compiles kernels)
        3. Trace: fresh copy to device → begin trace → forward → end trace
        4. Return the fresh device tensor for trace replay

        Args:
            seq_len: Sequence length to trace

        Returns:
            Tuple of (trace_id, token_buffer, output_logits_buffer)
        """
        B = self.batch
        rep = _mesh_mapper(self.device)

        # Create host tensor (NOT on device yet) - tt_transformers pattern
        token_host = ttnn.from_torch(
            torch.zeros(1, 1, B, seq_len, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None,  # Keep on host
            mesh_mapper=rep,
        )

        # Get RoPE cos/sin for this sequence length
        cos, sin = self.model.rope.get_cos_sin(seq_len)

        # === WARMUP: copy to device → forward (compiles kernels) ===
        warmup_tokens = ttnn.to_device(token_host, device=self.device)
        warmup_out = self._prefill_forward_from_tokens(warmup_tokens, cos, sin)
        warmup_out.deallocate(True)
        ttnn.synchronize_device(self.device)

        # === TRACE: fresh copy to device → begin trace → forward → end trace ===
        token_buffer = ttnn.to_device(token_host, device=self.device)

        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        output_logits = self._prefill_forward_from_tokens(token_buffer, cos, sin)
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)

        ttnn.synchronize_device(self.device)

        return (trace_id, token_buffer, output_logits)

    def _prefill_forward_from_tokens(self, token_ids: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor):
        """Run prefill forward from token IDs (for trace capture).

        Following gpt_oss pattern: embedding happens inside the trace, so the
        token_ids buffer is preserved (not deallocated by layer forward).
        The embedding output gets consumed by layers, but token_ids persists
        for trace replay.

        Args:
            token_ids: [1, 1, B, S] uint32 token IDs (persistent trace buffer)
            cos, sin: RoPE matrices for this sequence length
        """
        B = token_ids.shape[2]
        S = token_ids.shape[3]
        H = self.model.config.hidden_size

        # Embed tokens inside the trace (gpt_oss pattern)
        # token_ids buffer is NOT consumed - only the embedding output is
        x = ttnn.embedding(token_ids, self.model.embed_weight, layout=ttnn.TILE_LAYOUT)
        if len(x.shape) == 3:
            x = ttnn.unsqueeze_to_4D(x)
        x = ttnn.reshape(x, (B, S, H))

        for layer in self.model.layers:
            x = layer.forward_prefill(x, cos, sin, user_id=0, page_table=self._page_table_device)

        x = self.model.norm(x)
        logits = ttnn.linear(x, self.model.lm_head, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits

    def _execute_prefill_trace(self, input_ids: torch.Tensor, seq_len: int):
        """Execute prefill using trace.

        Following gpt_oss pattern: copy token IDs to the trace input buffer,
        then execute trace. Embedding happens inside the trace.

        Args:
            input_ids: [B, S] token IDs
            seq_len: Sequence length (must match traced length)

        Returns:
            Output logits tensor
        """
        trace_id, token_buffer, output_logits = self._get_prefill_trace(seq_len)

        # Prepare token IDs on host in the same format as trace buffer [1, 1, B, S]
        B, S = input_ids.shape
        token_host = ttnn.from_torch(
            input_ids.unsqueeze(0).unsqueeze(0),  # [1, 1, B, S]
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Copy token IDs to persistent device buffer
        ttnn.copy_host_to_device_tensor(token_host, token_buffer)

        # Execute trace
        ttnn.execute_trace(self.device, trace_id, cq_id=0, blocking=True)

        return output_logits

    def _release_prefill_traces(self):
        """Release all prefill traces."""
        for seq_len, (trace_id, token_buffer, output_logits) in self._prefill_traces.items():
            ttnn.release_trace(self.device, trace_id)
            token_buffer.deallocate(True)
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
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Generate tokens with optional sampling support.

        Args:
            prompt_ids:         [B, S] integer token ids.
            max_new_tokens:     maximum tokens to generate.
            eos_token_id:       stop when this token is generated.
            use_trace:          enable Metal trace capture/replay for decode.
            enable_prefill_trace: enable Metal trace capture/replay for prefill.
            temperature:        sampling temperature (0.0 = greedy, higher = more random).
            top_k:              keep only top k tokens (0 = disabled).
            top_p:              nucleus sampling threshold (1.0 = disabled).
            repetition_penalty: penalize repeated tokens (1.0 = disabled, >1.0 = penalize).

        Returns:
            [B, S + generated] token ids.
        """
        B, S = prompt_ids.shape

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # --- Prefill ---
        self.model.clear_kv_caches()

        if enable_prefill_trace:
            # Use traced prefill
            logits = self._execute_prefill_trace(prompt_ids, S)
        else:
            # Normal prefill
            logits = self.model.forward_prefill(prompt_ids, page_table=self._page_table_device)

        last_logits = self._extract_logits(logits, S)

        # Sample first token (no repetition penalty yet since no generated tokens)
        next_token = sample_from_logits(
            last_logits,
            temperature=sampling_params.temperature,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
        )

        if not enable_prefill_trace:
            logits.deallocate(True)

        generated = [next_token]
        cur_pos = S

        if use_trace:
            return self._generate_with_trace(
                prompt_ids, next_token, cur_pos, generated, max_new_tokens, eos_token_id, sampling_params
            )
        else:
            return self._generate_no_trace(
                prompt_ids, next_token, cur_pos, generated, max_new_tokens, eos_token_id, sampling_params
            )

    def _generate_no_trace(
        self, prompt_ids, next_token, cur_pos, generated, max_new_tokens, eos_token_id, sampling_params: SamplingParams
    ):
        """Decode loop without trace (original path)."""
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            logits = self.model.forward_decode(next_token, cur_pos)
            last_logits = self._extract_logits(logits, 1)
            logits.deallocate(True)

            # Apply repetition penalty
            last_logits = apply_repetition_penalty(last_logits, generated, sampling_params.repetition_penalty)

            # Sample next token
            next_token_out = sample_from_logits(
                last_logits,
                temperature=sampling_params.temperature,
                top_k=sampling_params.top_k,
                top_p=sampling_params.top_p,
            )

            generated.append(next_token_out)
            next_token = next_token_out
            cur_pos += 1

        return torch.cat([prompt_ids, torch.cat(generated, dim=-1)], dim=-1)

    def _generate_with_trace(
        self, prompt_ids, next_token, cur_pos, generated, max_new_tokens, eos_token_id, sampling_params: SamplingParams
    ):
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
            last_logits = self._extract_logits(self._trace_logits_output, 1)

            # Apply repetition penalty
            last_logits = apply_repetition_penalty(last_logits, generated, sampling_params.repetition_penalty)

            # Sample next token
            next_token_out = sample_from_logits(
                last_logits,
                temperature=sampling_params.temperature,
                top_k=sampling_params.top_k,
                top_p=sampling_params.top_p,
            )

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
