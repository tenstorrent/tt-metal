# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2.5 greedy-decode generator with Metal trace support.

Usage:
    gen = TtMiniMaxGenerator(model, device, max_seq_len=2048)
    gen.setup_trace(prompt_ids)          # warmup + capture trace
    tokens = gen.generate(prompt_ids, max_new_tokens=128)
"""

import torch

import ttnn

from .model import TtMiniMaxModel


class TtMiniMaxGenerator:
    """Greedy autoregressive generator for TtMiniMaxModel.

    Sequence of operations:
      1. prefill(prompt_ids)  — fills KV-cache for all prompt tokens.
      2. decode loop          — generates one token per step using KV-cache.
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

        # Pre-allocate KV caches
        self.kv_caches = model.allocate_kv_cache(batch=batch)

        self._trace_id = None
        self._trace_input = None
        self._trace_output = None
        self._trace_cur_pos = None

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy decode.

        Args:
            prompt_ids:     [B, S] integer token ids.
            max_new_tokens: maximum tokens to generate.
            eos_token_id:   stop when this token is generated.

        Returns:
            [B, S + max_new_tokens] token ids.
        """
        B, S = prompt_ids.shape

        # --- Prefill ---
        logits, self.kv_caches = self.model.forward_prefill(prompt_ids, self.kv_caches)
        # Take last-token logits for first decode step
        last_logits = ttnn.to_torch(logits)[:, S - 1, :]  # [B, V]
        next_token = last_logits.argmax(dim=-1, keepdim=True)  # [B, 1]
        logits.deallocate(True)

        generated = [next_token]
        cur_pos = S  # position of the next token to fill

        # --- Decode loop ---
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            if self._trace_id is not None:
                next_token_out = self._run_trace(next_token, cur_pos)
            else:
                logits, self.kv_caches = self.model.forward_decode(next_token, self.kv_caches, cur_pos)
                next_token_out = ttnn.to_torch(logits)[:, 0, :].argmax(dim=-1, keepdim=True)
                logits.deallocate(True)

            generated.append(next_token_out)
            next_token = next_token_out
            cur_pos += 1

        return torch.cat([prompt_ids, torch.cat(generated, dim=-1)], dim=-1)

    # ------------------------------------------------------------------
    # Metal Trace support
    # ------------------------------------------------------------------

    def setup_trace(self, prompt_ids: torch.Tensor):
        """Warmup all ops then capture decode trace for maximum speed.

        Call once after prefill (or at least after the model is compiled).
        After calling this, `generate()` will use the trace for decode steps.
        """
        B, S = prompt_ids.shape

        # Warmup: compile all decode ops
        dummy_token = torch.zeros((B, 1), dtype=torch.long)
        _logits, _ = self.model.forward_decode(dummy_token, self.kv_caches, 0)
        _logits.deallocate(True)

        # Allocate persistent device tensor for decode input
        dummy_host = ttnn.from_torch(
            dummy_token.unsqueeze(0).unsqueeze(0).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None,  # host tensor
        )
        self._trace_input = ttnn.allocate_tensor_on_device(
            dummy_host.shape,
            dummy_host.dtype,
            dummy_host.layout,
            self.device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.copy_host_to_device_tensor(dummy_host, self._trace_input)

        # Capture trace — forward_decode uses self._trace_input and self.kv_caches
        self._trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._trace_output = self.model.forward_decode(dummy_token, self.kv_caches, 0)[0]
        ttnn.end_trace_capture(self.device, self._trace_id, cq_id=0)

        print("[TtMiniMaxGenerator] Trace captured successfully.")

    def _run_trace(self, next_token: torch.Tensor, cur_pos: int):
        """Execute the captured decode trace."""
        host_in = ttnn.from_torch(
            next_token.unsqueeze(0).unsqueeze(0).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None,
        )
        ttnn.copy_host_to_device_tensor(host_in, self._trace_input)
        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=True)
        return ttnn.to_torch(self._trace_output)[:, 0, :].argmax(dim=-1, keepdim=True)

    def reset_caches(self):
        """Zero out KV caches for a fresh generation."""
        for i, (k, v) in enumerate(self.kv_caches):
            k.deallocate(True)
            v.deallocate(True)
        self.kv_caches = self.model.allocate_kv_cache(batch=self.batch)
