# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Experimental TTNN causal LM bridge for ACE-Step 5 Hz checkpoints.

This wraps :class:`~models.demos.ace_step_v1_5.ttnn_impl.qwen_tt_transformers_lm.QwenModelTtTransformers`,
the **stock** ``tt_transformers`` driver — all attention / RMSNorm / RoPE / KV-cache / MLP
ops come from ``models/tt_transformers`` (or its dependencies ``models/common/...``):

- Token embedding via :class:`models.tt_transformers.tt.embedding.Embedding`
  (auto-promoted to ``ScaledEmbedding`` when ``ModelArgs.embed_scale`` is set)
- :class:`models.tt_transformers.tt.attention.Attention` for fused QKV (+ optional bias),
  paged KV cache, fused SDPA (Q·Kᵀ, scaled fp32 softmax, ·V), output projection ``o_proj``,
  and Qwen3-style ``q_norm`` / ``k_norm``
- HF RoPE via :class:`models.tt_transformers.tt.rope.HfRotarySetup` + ``get_rot_mats_hf``
- :class:`models.tt_transformers.tt.distributed_norm.DistributedNorm` wrapping
  :class:`models.common.rmsnorm.RMSNorm` for ``input_layernorm`` / ``post_attention_layernorm``
  (a.k.a. ``ffn_norm``) / final norm
- Residual ``add`` inside :class:`models.tt_transformers.tt.decoder.TransformerBlock`
- :class:`models.tt_transformers.tt.mlp.MLP` for ``gate_proj`` / ``up_proj`` / ``down_proj``
- :class:`models.tt_transformers.tt.lm_head.LMHead` for logits

No hand-rolled ``AttentionFullDevice`` / ``QwenModel`` / custom embedding-prefix module
exists anymore — those have been removed.

**Limitations**

- Device KV cache uses **batch_size == 1**. The demo's ``--guidance_scale 1`` requirement
  (DiT KV / mesh batch = 1) keeps this consistent.
- The stock ``tt_transformers`` body relies on ``HF_MODEL`` env var for ``ModelArgs._set_hf_params``.
  :class:`QwenModelTtTransformers` sets it transiently during construction, then restores the
  previous value so other ACE-Step components are unaffected.

The wrapper exposes a minimal ``forward`` compatible with :meth:`LocalFiveHzLMHandler._forward_pass`
(``logits``, ``past_key_values``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn


class AceStepFiveHzExperimentalTtnnCausalLM(nn.Module):
    """HF-compatible thin wrapper around the stock ``tt_transformers`` ACE-Step 5 Hz LM body."""

    def __init__(self, hf_model_dir: str, ttnn_device: Any, *, max_seq_len: int = 16384) -> None:
        super().__init__()
        from models.demos.ace_step_v1_5.ttnn_impl.qwen_tt_transformers_lm import QwenModelTtTransformers

        self.qwen = QwenModelTtTransformers(
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
                "Experimental TTNN 5 Hz LM only supports batch_size==1. "
                "Use PyTorch HF LM for CFG (cond+uncond) or set guidance_scale=1."
            )

        from models.common.auto_compose import to_torch_auto_compose

        is_prefill = past_key_values is None
        if is_prefill:
            logits_tt = self.qwen.forward(input_ids, start_pos=0)
            self._cursor = int(input_ids.shape[1])
        else:
            logits_tt = self.qwen.forward(input_ids, start_pos=self._cursor)
            self._cursor += int(input_ids.shape[1])

        # Stash the prefill tile-row offset *before* composing, since composition flattens away
        # the [1,1,32,V] tile-shape that lets us identify which row is the real last token.
        # ``QwenModelTtTransformers._prefill`` returns ``[1, 1, 32, padded_vocab]`` containing one
        # tile-row that brackets the real last token at row ``offset = (seq_len-1) % 32``;
        # the LMHead can't operate on smaller shards (``shard_height==32`` is mandatory).
        last_token_offset_in_tile = (
            getattr(self.qwen, "_prefill_last_token_offset_in_tile", None) if is_prefill else None
        )

        logits_torch = to_torch_auto_compose(logits_tt).to(dtype=torch.float32, device=input_ids.device)
        # TTNN compose can yield extra leading singletons. Blind ``while dim>3: squeeze(0)`` can
        # turn decode logits into [1, V], so ``[:, -1, :]`` indexes the vocabulary axis as
        # "sequence" and sampling becomes multilingual garbage. Trim the trailing padded-vocab
        # dim first, then normalise to [1, S_out, vocab].
        vocab = int(getattr(self.config, "vocab_size", logits_torch.shape[-1]))
        trail = int(logits_torch.shape[-1])
        if trail != vocab:
            if trail > vocab:
                logits_torch = logits_torch[..., :vocab]
            elif int(logits_torch.shape[-2]) == vocab:
                logits_torch = logits_torch.transpose(-1, -2)
            else:
                raise RuntimeError(
                    f"Experimental LM logits trailing dim {trail} not compatible with vocab_size {vocab}"
                )
        logits_torch = logits_torch.reshape(-1, vocab).view(1, -1, vocab)
        seq_log = int(input_ids.shape[1])

        if is_prefill:
            # Prefill path: [1, S_out, vocab] where S_out is one 32-tile row (or 1 if the
            # underlying ``ttnn.slice`` already returned the logical row). Pull out the real
            # last-token logits using the stashed offset.
            s_out = int(logits_torch.shape[1])
            if last_token_offset_in_tile is not None and last_token_offset_in_tile < s_out:
                last_token_logits = logits_torch[:, last_token_offset_in_tile : last_token_offset_in_tile + 1, :]
            else:
                # Fallback: assume the last row is the real last token (matches decode behaviour
                # when ``ttnn.slice`` already trimmed the tile pad).
                last_token_logits = logits_torch[:, -1:, :]
            # HF's generate loop reads only ``logits[:, -1, :]`` for sampling, so we build a
            # [1, seq_log, vocab] tensor with zeros at positions [0, seq_log-1) and the real
            # last-token logits at position seq_log-1. Per-position prefill logits are not
            # currently exposed by the stock ``tt_transformers`` LMHead (it's a single
            # sharded matmul per tile-row); add chunked LMHead invocation here if a future
            # consumer requires full-sequence logits.
            if seq_log == 1:
                logits_torch = last_token_logits
            else:
                expanded = torch.zeros(
                    1, seq_log, vocab, dtype=last_token_logits.dtype, device=last_token_logits.device
                )
                expanded[:, -1, :] = last_token_logits[:, 0, :]
                logits_torch = expanded
        else:
            # Decode path: stock ``ttnn_decode_forward`` returns logits batch-padded to 32 along
            # the "sequence" axis (one row per user; we only have user 0). Trim to seq_log==1.
            s_out = int(logits_torch.shape[1])
            if s_out > seq_log:
                logits_torch = logits_torch[:, :seq_log, :]
            elif s_out < seq_log:
                raise RuntimeError(f"Experimental LM decode logits seq {s_out} < input_ids seq {seq_log}")

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
