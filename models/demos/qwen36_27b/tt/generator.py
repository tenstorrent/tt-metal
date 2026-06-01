# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Token generation pipeline for Qwen3.6-27B.

Supports:
  1. Prefill: process prompt tokens, populate DeltaNet states + KV caches
  2. Decode: autoregressive token generation
  3. Trace-based decode: captured trace replay for zero-dispatch overhead
     (requires all layers to be device-only — enable with use_trace=True)
"""

import os

import torch
import ttnn

from models.demos.qwen36_27b.tt.model import TtQwen36Model
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig

USE_TRACE = os.environ.get("QWEN_USE_TRACE", "0") == "1"


class Qwen36Generator:
    def __init__(self, model: TtQwen36Model, config: Qwen36ModelConfig, tokenizer=None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.deltanet_state = model.create_deltanet_state()
        self.kv_caches = {}
        self.position = 0

        self.trace_id = None
        self.trace_output = None
        self.trace_input_buf = None

    def prefill(self, token_ids: torch.Tensor):
        B, S = token_ids.shape
        assert B == 1, "Only batch_size=1 supported"

        position_ids = torch.arange(S)
        logits, self.kv_caches = self.model(
            token_ids,
            position_ids,
            self.deltanet_state,
            self.kv_caches,
            mode="prefill",
        )

        self.position = S
        return logits

    def decode_one_token(self, token_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if USE_TRACE and self.trace_id is not None:
            return self._decode_trace(token_id)

        logits, self.kv_caches = self.model(
            token_id,
            self.position,
            self.deltanet_state,
            self.kv_caches,
            mode="decode",
        )
        self.position += 1

        logits_cpu = ttnn.to_torch(logits).float()
        logits_cpu = logits_cpu.reshape(-1, logits_cpu.shape[-1])
        next_token = torch.argmax(logits_cpu, dim=-1, keepdim=True)

        return logits, next_token

    def capture_trace(self):
        """
        Capture a trace for the decode forward pass.
        Requires all model ops to be device-only (no to_torch/from_torch in forward).
        Call after prefill completes and at least one non-trace decode step.
        """
        dummy_token = torch.tensor([[0]], dtype=torch.long)
        self.model(dummy_token, self.position, self.deltanet_state, self.kv_caches, mode="decode")
        self.position += 1
        ttnn.synchronize_device(self.model.device)

        self.trace_input_buf = self.model.embed(dummy_token)

        self.trace_id = ttnn.begin_trace_capture(self.model.device, cq_id=0)
        self.trace_output, self.kv_caches = self.model.forward_from_embedding(
            self.trace_input_buf,
            self.position,
            self.deltanet_state,
            self.kv_caches,
        )
        ttnn.end_trace_capture(self.model.device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.model.device)

    def _decode_trace(self, token_id: torch.Tensor):
        """Execute decode step using captured trace."""
        new_embedding = self.model.embed(token_id)
        ttnn.copy_host_to_device_tensor(new_embedding, self.trace_input_buf)

        ttnn.execute_trace(self.model.device, self.trace_id, cq_id=0, blocking=False)
        self.position += 1

        logits_cpu = ttnn.to_torch(self.trace_output).float()
        logits_cpu = logits_cpu.reshape(-1, logits_cpu.shape[-1])
        next_token = torch.argmax(logits_cpu, dim=-1, keepdim=True)

        return self.trace_output, next_token

    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int = 32, temperature: float = 0.0) -> list[int]:
        last_logits = self.prefill(prompt_tokens)

        logits_cpu = ttnn.to_torch(last_logits).float().reshape(-1)

        if temperature == 0:
            next_token = torch.argmax(logits_cpu[:self.config.vocab_size]).item()
        else:
            probs = torch.softmax(logits_cpu[:self.config.vocab_size] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        generated = [next_token]

        for _ in range(max_new_tokens - 1):
            token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            _, next_token_tensor = self.decode_one_token(token_tensor)
            next_token = next_token_tensor.item()
            generated.append(next_token)

            if self.tokenizer and next_token == self.tokenizer.eos_token_id:
                break

        return generated

    def get_logits_for_sequence(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.reset()
        B, S = token_ids.shape
        all_logits = []

        for t in range(S):
            logits, self.kv_caches = self.model(
                token_ids[:, t:t+1],
                t,
                self.deltanet_state,
                self.kv_caches,
                mode="decode",
            )
            logits_cpu = ttnn.to_torch(logits).float().reshape(-1)
            all_logits.append(logits_cpu)
            self.position = t + 1

        return torch.stack(all_logits, dim=0)

    def reset(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.model.device, self.trace_id)
            self.trace_id = None
            self.trace_output = None
            self.trace_input_buf = None
        self.deltanet_state = self.model.create_deltanet_state()
        self.kv_caches = {}
        self.position = 0
