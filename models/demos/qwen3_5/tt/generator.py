# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generator for Qwen3.5-27B hybrid text model."""

from __future__ import annotations

import time

import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import Mode


class Qwen3_5Generator:
    """Generator for Qwen3.5-27B.
    Manages KV caches, linear-attention states, and the prefill->decode loop.
    """

    def __init__(self, model, model_args, mesh_device, tokenizer=None):
        self.model = model
        self.model_args = model_args
        self.mesh_device = mesh_device
        self.tokenizer = tokenizer
        self.n_layers = model_args.n_layers
        self.kv_caches = [None] * self.n_layers
        self.conv_states = [None] * self.n_layers
        self.recurrent_states = [None] * self.n_layers

    def reset_cache(self):
        self.kv_caches = [None] * self.n_layers
        self.conv_states = [None] * self.n_layers
        self.recurrent_states = [None] * self.n_layers

    def _encode(self, ids):
        t = torch.tensor([ids], dtype=torch.int32)
        return ttnn.from_torch(
            t,
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _forward(self, token_ids, pos, mode):
        m = self.model[0]
        logits_tt, self.kv_caches, self.conv_states, self.recurrent_states = m.forward(
            tokens=self._encode(token_ids),
            current_pos=pos,
            mode=mode,
            kv_caches=self.kv_caches,
            conv_states=self.conv_states,
            recurrent_states=self.recurrent_states,
        )
        logits = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))
        ttnn.deallocate(logits_tt)
        return logits.squeeze(0).squeeze(0)  # (T, vocab)

    @staticmethod
    def _pad_to_tile(tokens: list, tile: int = 32) -> tuple:
        """Pad token list to multiple of tile; return (padded_list, original_len)."""
        orig = len(tokens)
        r = orig % tile
        if r != 0:
            tokens = tokens + [0] * (tile - r)
        return tokens, orig

    def generate(self, prompt_tokens, max_new_tokens=32, temperature=0.0):
        self.reset_cache()
        # Pad prompt to tile (32) for TTNN prefill ops
        padded_tokens, prompt_len = self._pad_to_tile(list(prompt_tokens), tile=32)
        t0 = time.perf_counter()
        logits = self._forward(padded_tokens, 0, Mode.PREFILL)  # (T_padded, vocab)
        ttft_ms = (time.perf_counter() - t0) * 1000

        def sample(lg):
            if temperature == 0.0:
                return int(lg.argmax(-1))
            return int(torch.multinomial(torch.softmax(lg / temperature, dim=-1), 1))

        next_tok = sample(logits[prompt_len - 1])  # logit for last real token
        generated = [next_tok]
        decode_times = []
        pos = len(prompt_tokens)
        eos = getattr(self.model_args, "eos_token_id", None)

        for _ in range(max_new_tokens - 1):
            if eos is not None and next_tok == eos:
                break
            t0 = time.perf_counter()
            logits = self._forward([next_tok], pos, Mode.DECODE)
            decode_times.append((time.perf_counter() - t0) * 1000)
            next_tok = sample(logits[-1])
            generated.append(next_tok)
            pos += 1

        avg_ms = sum(decode_times) / len(decode_times) if decode_times else 0.0
        perf = {
            "ttft_ms": ttft_ms,
            "tokens_generated": len(generated),
            "avg_decode_ms": avg_ms,
            "tok_per_sec": 1000.0 / avg_ms if avg_ms > 0 else 0.0,
        }
        logger.info(
            f"TTFT={ttft_ms:.1f}ms | {len(generated)} tokens | "
            f"decode={avg_ms:.1f}ms/tok ({perf['tok_per_sec']:.1f} tok/s)"
        )
        return generated, perf
