# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Experimental TTNN causal LM bridge for ACE-Step 5 Hz checkpoints (host HF weights + TTNN layers).

This wraps :class:`QwenModelFullDevice` (full TTNN attention / device KV / TTNN RoPE caches via
``TtHfRotaryEmbedding``), which loads ``AutoModelForCausalLM.from_pretrained`` weights and runs the
decoder on TTNN. Token embedding and prefill causal masks use TTNN-only staging
(``qwen_causal_prefix_ttnn.build_prefix_full_device``). It is **not** the
production ``models/tt_transformers`` + vLLM stack; it exists to unblock integration testing and
iterative porting.

**Limitations**

- Device KV cache in ``QwenModelFullDevice`` uses a **fixed batch dimension of 1**. Do not use with LM CFG
  paths that batch cond+uncond (``input_ids.shape[0] > 1``); use ``guidance_scale==1`` for the demo
  CLI when this backend is enabled.
- ``QwenModelFullDevice`` loads ``AutoModelForCausalLM.from_pretrained`` weights once, then frees the HF
  model; it is constructed with ``validate_against_hf=False`` so no duplicate HF weights stay resident.

The wrapper exposes a minimal ``forward`` compatible with :meth:`LocalFiveHzLMHandler._forward_pass`
(``logits``, ``past_key_values``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn


class AceStepFiveHzExperimentalTtnnCausalLM(nn.Module):
    """HF-compatible thin wrapper around experimental ``QwenModel`` (TTNN decode)."""

    def __init__(self, hf_model_dir: str, ttnn_device: Any, *, max_seq_len: int = 16384) -> None:
        super().__init__()
        from models.demos.ace_step_v1_5.ttnn_impl.qwen_model_full_device import QwenModelFullDevice

        self.qwen = QwenModelFullDevice(
            str(hf_model_dir),
            ttnn_device,
            max_seq_len=int(max_seq_len),
            validate_against_hf=False,
        )
        self.config = self.qwen.config
        self.generation_config = SimpleNamespace(use_cache=True)
        self._cursor = 0
        self._ttnn_device = ttnn_device

    def reset_decode_state(self) -> None:
        self.qwen.reset_kv_cache()
        self._cursor = 0

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> SimpleNamespace:
        del attention_mask, kwargs  # API parity with HF; mask handled implicitly in experimental model
        if input_ids is None:
            raise ValueError("AceStepFiveHzExperimentalTtnnCausalLM.forward requires input_ids")
        if int(input_ids.shape[0]) != 1:
            raise RuntimeError(
                "Experimental TTNN 5 Hz LM (ace_step_ds_r1_qwen) only supports batch_size==1. "
                "Use PyTorch HF LM for CFG (cond+uncond) or set guidance_scale=1."
            )

        from models.common.auto_compose import to_torch_auto_compose

        if past_key_values is None:
            logits_tt = self.qwen.forward(input_ids, start_pos=0)
            self._cursor = int(input_ids.shape[1])
        else:
            logits_tt = self.qwen.forward(input_ids, start_pos=self._cursor)
            self._cursor += int(input_ids.shape[1])

        logits_torch = to_torch_auto_compose(logits_tt).to(dtype=torch.float32, device=input_ids.device)
        # TTNN compose can yield extra leading singletons (e.g. [1,1,S,V] prefill, [1,1,1,V] decode).
        # Blind ``while dim>3: squeeze(0)`` can turn decode logits into [1, V], so ``[:, -1, :]`` indexes
        # the vocabulary axis as "sequence" and sampling becomes multilingual garbage.
        vocab = int(getattr(self.config, "vocab_size", logits_torch.shape[-1]))
        if logits_torch.shape[-1] != vocab and logits_torch.shape[-2] == vocab:
            logits_torch = logits_torch.transpose(-1, -2)
        if logits_torch.shape[-1] != vocab:
            raise RuntimeError(f"Experimental LM logits trailing dim {logits_torch.shape[-1]} != vocab_size {vocab}")
        logits_torch = logits_torch.reshape(-1, vocab).view(1, -1, vocab)
        # TTNN TILE can pad the sequence axis; sampling uses ``[:, -1, :]`` — trim to real tokens.
        seq_log = int(input_ids.shape[1])
        if int(logits_torch.shape[1]) > seq_log:
            logits_torch = logits_torch[:, :seq_log, :]
        elif int(logits_torch.shape[1]) < seq_log:
            raise RuntimeError(f"Experimental LM logits seq {logits_torch.shape[1]} < input_ids seq {seq_log}")

        return SimpleNamespace(
            logits=logits_torch,
            past_key_values=True if use_cache else None,
        )

    def train(self, mode: bool = True) -> "AceStepFiveHzExperimentalTtnnCausalLM":
        return self

    def eval(self) -> "AceStepFiveHzExperimentalTtnnCausalLM":
        return self

    def to(self, *args: Any, **kwargs: Any) -> "AceStepFiveHzExperimentalTtnnCausalLM":
        """No-op: weights live on TTNN / host staging; do not move the experimental stack via PyTorch."""
        return self


__all__ = ["AceStepFiveHzExperimentalTtnnCausalLM"]
