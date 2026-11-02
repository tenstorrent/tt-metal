# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU golden reference for GPT-OSS prefill KV-cache generation.

Runs HuggingFace ``AutoModelForCausalLM`` (trust_remote_code) and captures per-layer
post-RoPE K and raw V before any sliding-window truncation.  GPT-OSS alternates
sliding_attention (window=128) and full_attention layers; ``DynamicCache`` alone would
drop early positions on sliding layers during a long prefill, so ``FullKVCapture``
snapshots K/V in ``update()`` before the parent cache truncates.

Memory approach:
- Model weights mmap'd via ``low_cpu_mem_usage=True``
- GPT-OSS disables SDPA (attention sinks need concat-softmax). Default path installs
  MiniMax-style query-row + MoE token tiling via ``tiled_ops.install_tiled_ops`` so
  long sequences do not allocate full ``[H, S, S]`` scores (~387 GB at S=55k).
- Pass ``use_tiled_ops=False`` for stock HF eager (matches older short-seq goldens;
  OOMs on long sequences).
- Saves layer-by-layer via streaming callback; releases each layer's full_kv after save

Output layout matches the MiniMax-style golden trace:
  key_cache_layer_{i}   [1, num_kv_heads, seq_len, head_dim]
  value_cache_layer_{i} [1, num_kv_heads, seq_len, head_dim]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from transformers.cache_utils import DynamicCache


class FullKVCapture(DynamicCache):
    """DynamicCache that snapshots full K/V per layer before sliding-window truncation.

    On each ``update()``, HF passes *new* K/V for the current forward (full seq for
    one-shot prefill). ``full_kv`` holds the complete pre-truncation cache.
    ``kv_callback`` fires once the accumulated length reaches ``expected_seq_len``.
    """

    def __init__(
        self,
        config,
        *,
        kv_callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
        num_layers: int | None = None,
        expected_seq_len: int | None = None,
        release_after_callback: bool = True,
    ):
        super().__init__(config=config)
        self._kv_callback = kv_callback
        self._num_layers = num_layers
        self._expected_seq_len = expected_seq_len
        self._release_after_callback = release_after_callback
        self.full_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._callback_done: set[int] = set()

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self._num_layers is None or layer_idx < self._num_layers:
            k_new = key_states.detach()
            v_new = value_states.detach()
            if layer_idx in self.full_kv:
                k_prev, v_prev = self.full_kv[layer_idx]
                k = torch.cat([k_prev, k_new], dim=-2)
                v = torch.cat([v_prev, v_new], dim=-2)
            else:
                k, v = k_new, v_new
            self.full_kv[layer_idx] = (k, v)

            ready = self._expected_seq_len is None or k.shape[-2] >= self._expected_seq_len
            if self._kv_callback is not None and ready and layer_idx not in self._callback_done:
                self._kv_callback(layer_idx, k, v)
                self._callback_done.add(layer_idx)
                if self._release_after_callback:
                    del self.full_kv[layer_idx]

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
    """HF-reference prefill model for golden KV generation (one-shot)."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        num_layers: int | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        zero_sinks: bool = False,
        disable_sliding_window: bool = False,
        use_tiled_ops: bool = True,
    ):
        from transformers import AutoConfig, AutoModelForCausalLM

        model_path = Path(model_path)
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if num_layers is not None and not 1 <= num_layers <= hf_config.num_hidden_layers:
            raise ValueError(f"num_layers must be between 1 and {hf_config.num_hidden_layers}, got {num_layers}")
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
        self.use_tiled_ops = use_tiled_ops
        self.attn_q_chunk: int | None = None
        self.ffn_token_chunk: int | None = None

        if use_tiled_ops:
            from models.demos.gpt_oss_d_p.reference.tiled_ops import install_tiled_ops

            # Install before from_pretrained so any eager dispatch resolves to tiled ops.
            self.attn_q_chunk, self.ffn_token_chunk = install_tiled_ops()
            print(
                f"[load] installed MiniMax-style tiling: "
                f"ATTN_Q_CHUNK={self.attn_q_chunk} FFN_TOKEN_CHUNK={self.ffn_token_chunk} "
                f"(env REF_ATTN_Q_CHUNK / REF_FFN_TOKEN_CHUNK)",
                flush=True,
            )
        else:
            print(
                "[load] using stock HF eager attention/MoE (no tiling) — "
                "may OOM on long sequences due to full [H,S,S] scores",
                flush=True,
            )

        print(f"[load] loading HF model from {model_path} ({compute_dtype}, cpu, mmap) ...", flush=True)
        # GPT-OSS: _supports_sdpa=False (sinks). Force eager (stock or tiled replacement).
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
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
        """Run a **one-shot** prefill pass. ``input_ids`` is ``[B, S]`` (B must be 1).

        ``kv_callback(layer_idx, key_cache, value_cache)`` is invoked from ``FullKVCapture.update``
        once each layer's K/V reaches full sequence length, with tensors shaped
        ``[1, num_kv_heads, seq_len, head_dim]`` (post-RoPE K, raw V).

        Long sequences stay one-shot; bound peak RAM with tiled attention/MoE
        (``use_tiled_ops=True``, ``REF_ATTN_Q_CHUNK`` / ``REF_FFN_TOKEN_CHUNK``).
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.shape[0] != 1:
            raise ValueError(f"golden prefill supports batch size 1, got {input_ids.shape[0]}")

        seq_len = input_ids.shape[1]

        capture = FullKVCapture(
            config=self._model.config,
            kv_callback=kv_callback,
            num_layers=self.cfg.num_hidden_layers,
            expected_seq_len=seq_len,
            release_after_callback=True,
        )

        with torch.no_grad():
            out = self._model(input_ids=input_ids, use_cache=True, past_key_values=capture)
        # top-1 sign-off reference: HF's per-position argmax (the forward above already computes logits)
        self.top1_token_ids = out.logits[0].argmax(-1).to(torch.int32).contiguous().cpu()

        return [capture.full_kv[i] for i in range(self.cfg.num_hidden_layers) if i in capture.full_kv]


def load_golden_model(
    model_path: str | Path,
    *,
    num_layers: int | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
    zero_sinks: bool = False,
    disable_sliding_window: bool = False,
    use_tiled_ops: bool = True,
) -> GoldenPrefillModel:
    """Construct a golden prefill model from an HF checkpoint directory."""
    return GoldenPrefillModel(
        model_path,
        num_layers=num_layers,
        compute_dtype=compute_dtype,
        zero_sinks=zero_sinks,
        disable_sliding_window=disable_sliding_window,
        use_tiled_ops=use_tiled_ops,
    )
