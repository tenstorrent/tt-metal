# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU golden reference for GPT-OSS prefill KV-cache generation.

Runs HuggingFace ``AutoModelForCausalLM`` (trust_remote_code) and captures per-layer
post-RoPE K and raw V before any sliding-window truncation.  GPT-OSS alternates
sliding_attention (window=128) and full_attention layers; ``DynamicCache`` alone would
drop early positions on sliding layers during a long prefill, so ``FullKVCapture``
snapshots K/V in ``update()`` before the parent cache truncates.

Output layout matches the MiniMax-style golden trace:
  key_cache_layer_{i}   [1, num_kv_heads, seq_len, head_dim]
  value_cache_layer_{i} [1, num_kv_heads, seq_len, head_dim]

Weights are mmap'd via ``low_cpu_mem_usage=True`` (same approach as
``tests/accuracy/hf_reference_oracle.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from transformers.cache_utils import DynamicCache


class FullKVCapture(DynamicCache):
    """DynamicCache that snapshots full K/V per layer before sliding-window truncation."""

    def __init__(
        self,
        config,
        *,
        kv_callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
        num_layers: int | None = None,
    ):
        super().__init__(config=config)
        self._kv_callback = kv_callback
        self._num_layers = num_layers
        self.full_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self._num_layers is None or layer_idx < self._num_layers:
            k = key_states.detach()
            v = value_states.detach()
            self.full_kv[layer_idx] = (k, v)
            if self._kv_callback is not None:
                self._kv_callback(layer_idx, k, v)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


@dataclass
class GoldenModelConfig:
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    sliding_window: int | None


class GoldenPrefillModel:
    """HF-reference prefill model for golden KV generation."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        num_layers: int | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        zero_sinks: bool = False,
        disable_sliding_window: bool = False,
    ):
        from transformers import AutoConfig, AutoModelForCausalLM

        model_path = Path(model_path)
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.cfg = GoldenModelConfig(
            num_hidden_layers=num_layers or hf_config.num_hidden_layers,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            hidden_size=hf_config.hidden_size,
            vocab_size=hf_config.vocab_size,
            sliding_window=getattr(hf_config, "sliding_window", None),
        )
        self.compute_dtype = compute_dtype
        self._model_path = model_path

        print(f"[load] loading HF model from {model_path} ({compute_dtype}, cpu, mmap) ...", flush=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
        ).eval()
        print("[load] model loaded", flush=True)

        if zero_sinks:
            n_zeroed = 0
            with torch.no_grad():
                for layer in self._model.model.layers:
                    attn = layer.self_attn
                    if hasattr(attn, "sinks") and attn.sinks is not None:
                        attn.sinks.zero_()
                        n_zeroed += 1
            print(f"[load] zeroed attention sinks on {n_zeroed} layer(s)", flush=True)

        if disable_sliding_window:
            n_disabled = 0
            if hasattr(self._model.config, "layer_types") and self._model.config.layer_types is not None:
                self._model.config.layer_types = ["full_attention"] * len(self._model.config.layer_types)
            for layer in self._model.model.layers:
                attn = layer.self_attn
                if hasattr(attn, "sliding_window"):
                    attn.sliding_window = None
                if hasattr(attn, "is_sliding"):
                    attn.is_sliding = False
                n_disabled += 1
            print(f"[load] disabled sliding window on {n_disabled} layer(s)", flush=True)

    def prefill(
        self,
        input_ids: torch.Tensor,
        *,
        kv_callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Run a single prefill pass. ``input_ids`` is ``[B, S]`` (B must be 1).

        ``kv_callback(layer_idx, key_cache, value_cache)`` is invoked from ``FullKVCapture.update``
        with tensors shaped ``[1, num_kv_heads, seq_len, head_dim]`` (post-RoPE K, raw V).
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.shape[0] != 1:
            raise ValueError(f"golden prefill supports batch size 1, got {input_ids.shape[0]}")

        capture = FullKVCapture(
            config=self._model.config,
            kv_callback=kv_callback,
            num_layers=self.cfg.num_hidden_layers,
        )
        with torch.no_grad():
            self._model(input_ids=input_ids, use_cache=True, past_key_values=capture)

        return [capture.full_kv[i] for i in range(self.cfg.num_hidden_layers) if i in capture.full_kv]


def load_golden_model(
    model_path: str | Path,
    *,
    num_layers: int | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
    zero_sinks: bool = False,
    disable_sliding_window: bool = False,
) -> GoldenPrefillModel:
    """Construct a golden prefill model from an HF checkpoint directory."""
    return GoldenPrefillModel(
        model_path,
        num_layers=num_layers,
        compute_dtype=compute_dtype,
        zero_sinks=zero_sinks,
        disable_sliding_window=disable_sliding_window,
    )
