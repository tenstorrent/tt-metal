# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2.5 greedy-decode generator with Metal trace support.

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

from .model import TtMiniMaxModel


def _mesh_mapper(device):
    if isinstance(device, ttnn.MeshDevice):
        return ttnn.ReplicateTensorToMesh(device)
    return None


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

        self._trace_id = None
        self._trace_token_input = None
        self._trace_position_int32 = None
        self._trace_position_uint32 = None
        self._trace_logits_output = None

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
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        eos_token_id: int | None = None,
        use_trace: bool = False,
    ) -> torch.Tensor:
        """Greedy decode with variable ISL support.

        Args:
            prompt_ids:     [B, S] integer token ids.
            max_new_tokens: maximum tokens to generate.
            eos_token_id:   stop when this token is generated.
            use_trace:      enable Metal trace capture/replay for decode.

        Returns:
            [B, S + generated] token ids.
        """
        B, S = prompt_ids.shape

        # --- Prefill ---
        self.model.clear_kv_caches()
        logits = self.model.forward_prefill(prompt_ids)
        last_logits = self._extract_logits(logits, S)
        next_token = last_logits.argmax(dim=-1, keepdim=True)
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
        self.model.clear_kv_caches()

    def __call__(self, prompt_ids, **kwargs):
        return self.generate(prompt_ids, **kwargs)
