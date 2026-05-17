# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Experimental TTNN causal LM bridge for ACE-Step 5 Hz checkpoints (full-device attention).

Wraps :class:`~models.demos.ace_step_v1_5.ttnn_impl.qwen_model_full_device.QwenModelFullDevice`,
which runs **everything on TTNN**:

- Token embedding (``ttnn.embedding``)
- Q / K / V / O / MLP / lm_head matmuls (HF-grade ``HiFi4 / fp32_dest_acc_en=True``, fp32 outputs)
- RMSNorm (input / post-attn / final / q_norm / k_norm) — all fp32 weights, ``ttnn.rms_norm``
- RoPE (``ttnn.experimental.rotary_embedding`` via :class:`TtHfRotaryEmbedding`)
- Causal mask (host-built, tile-aligned, uploaded to device once)
- KV cache (TTNN tensors, grown via ``ttnn.concat(dim=2)``)
- SDPA (TTNN decomposed: matmul → softmax(fp32) → matmul; same kernel config)

The host-attention :class:`~models.demos.ace_step_v1_5.ttnn_impl.ace_step_ds_r1_qwen.QwenModel`
remains available for PCC parity checks (~0.984 vs HF Qwen3 1.7B at L=24) but is no longer the
default — the device-resident path here matches the demo's spirit ("everything on Tensix") and
saves the per-layer host↔device DMA inside attention.

**Precision configuration applied to ``QwenModelFullDevice``** (``qwen_model_full_device.py``):

- HF-grade ``compute_kernel_config`` on every device matmul (``HiFi4 / math_approx_mode=False /
  fp32_dest_acc_en=True / packer_l1_acc=True``).
- Q / K / V / O / lm_head matmul outputs forced to ``ttnn.float32`` so the residual stream does
  not quantize to bf16 between layers (the dominant remaining drift on Qwen3 1.7B before this
  fix).
- All RMSNorm weights (input/post-attn/final + q_norm/k_norm) staged as ``ttnn.float32`` so the
  ``ttnn.rms_norm`` reductions stay in fp32 on the upgraded residual stream.
- Residual stream upgraded to ``fp32`` immediately after the embedding lookup.
- Tile-aligned causal mask + Q/K/V seq-pad in :class:`AttentionFullDevice` so the decomposed
  softmax does not absorb tile-pad keys (the bug that previously dropped prefill PCC to 0.93).

**Limitations**

- Device KV cache uses **batch_size == 1**. The demo's ``--guidance_scale 1`` requirement (DiT
  KV / mesh batch = 1) keeps this consistent.
- Sliding-window attention layers are not supported by ``AttentionFullDevice`` (will raise at
  init); Qwen3 1.7B in ACE-Step is all ``full_attention`` so this does not apply in practice.

The wrapper exposes a minimal ``forward`` compatible with :meth:`LocalFiveHzLMHandler._forward_pass`
(``logits``, ``past_key_values``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn


class AceStepFiveHzExperimentalTtnnCausalLM(nn.Module):
    """HF-compatible thin wrapper around ``QwenModelFullDevice`` (full-device LM body)."""

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
