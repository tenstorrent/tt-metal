# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Experimental TTNN causal LM bridge for ACE-Step 5 Hz checkpoints.

Wraps :class:`~models.demos.ace_step_v1_5.ttnn_impl.ace_step_ds_r1_qwen.QwenModel`, which keeps the
heavy compute on TTNN (token embedding, all Q/K/V/O matmuls, MLP gate/up/down, lm_head matmul) and
runs RMSNorm, RoPE, attention softmax, and the host KV cache on torch CPU using HF's reference
:class:`~transformers.models.qwen3.modeling_qwen3.Qwen3RotaryEmbedding` and
:func:`~transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb` — exactly matching HF
semantics on bf16 weights.

**Precision configuration applied to ``QwenModel``** (``ace_step_ds_r1_qwen.py``):

- HF-grade compute kernel config (``HiFi4``, ``math_approx_mode=False``, ``fp32_dest_acc_en=True``,
  ``packer_l1_acc=True``) on every device matmul: Q / K / V / O, gate / up / down, lm_head.
- Residual stream upgraded to ``fp32`` (``ttnn.typecast`` after embedding) and every device matmul
  output forced to ``fp32`` (``dtype=ttnn.float32``) so the per-layer residual adds do not quantize
  to bf16 — that was the dominant remaining drift in the deep layers (mean_abs > 200, max > 3000)
  per the ``test_llm_per_layer_pcc_debug`` diagnostic.
- ``RMSNorm`` weight stored as ``fp32`` so the manual mul/mean/sqrt/reciprocal/mul chain stays
  uniformly fp32 on the upgraded residual stream.

**Achievable PCC vs HF on bf16 weights:** ~0.984 for Qwen3 1.7B prefill at L=24
(``test_llm_handler_experimental_causal_lm_prefill_decode_pcc_vs_torch``). A bit-exact match with
torch is not possible because TTNN's tile-based bf16 matmul rounds at different boundaries than
torch's BLAS GEMM. If you need ``comp_pcc == 1.0`` (e.g. for plumbing sanity), set
``ACE_STEP_EXPERIMENTAL_LM_PCC=0.98`` (or replace this wrapper with a ``transformers``
pass-through; the HF model API is the same).

**Limitations**

- Host KV cache in :class:`Attention` uses **batch_size == 1**. Use ``guidance_scale==1`` for the
  demo CLI when this backend is enabled.
- ``QwenModel`` is constructed with ``validate_against_hf=False`` so HF weights are released after
  staging.

The wrapper exposes a minimal ``forward`` compatible with :meth:`LocalFiveHzLMHandler._forward_pass`
(``logits``, ``past_key_values``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn


class AceStepFiveHzExperimentalTtnnCausalLM(nn.Module):
    """HF-compatible thin wrapper around the TTNN ``QwenModel`` (host RoPE / KV / softmax + TTNN matmul)."""

    def __init__(self, hf_model_dir: str, ttnn_device: Any, *, max_seq_len: int = 16384) -> None:
        super().__init__()
        from models.demos.ace_step_v1_5.ttnn_impl.ace_step_ds_r1_qwen import QwenModel

        self.qwen = QwenModel(
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
