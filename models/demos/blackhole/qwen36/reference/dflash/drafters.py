# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""What DFlash needs from a drafter, and its host implementation.

The mirror of :mod:`.targets`. :class:`SpeculativeDrafter` is two methods — ``reset`` and
``propose`` — and everything else about drafting hides behind them: building the noise embedding,
carrying the KV history, running the LM head, sampling.

* :class:`HostDrafter` wraps the host reference ``DFlashDraftModel``.
* The device implementation, ``TtDrafter``, lives in
  :mod:`models.demos.blackhole.qwen36.tt.dflash.speculative_drafter`.

Both borrow the target's embedding and LM head — the drafter checkpoint ships neither (58 tensors,
no ``lm_head``, no ``embed_tokens``), which is why a drafter is always constructed against a target.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class SpeculativeDrafter(Protocol):
    """The drafter surface :func:`~.generate.dflash_generate` drives."""

    block_size: int
    mask_token_id: int

    def reset(self) -> None:
        """Drop any carried context and start a new sequence."""

    def propose(self, taps, block_ids: torch.Tensor, start: int, *, temperature: float, top_p: float, top_k: int):
        """Fill slots ``1..S-1`` of a block.

        Args:
            taps: the target's residual streams for the newly accepted context tokens, in whatever
                form that target produces (host feature or device tensors) — opaque to the loop.
            block_ids: ``[1, S]``; slot 0 is the confirmed anchor, the rest are ``mask_token_id``.
            start: absolute position of slot 0.

        Returns:
            ``(tokens, probs)`` — ``tokens`` is ``[1, S-1]`` on host; ``probs`` is the drafter's own
            ``[1, S-1, V]`` distribution when sampling (the rejection sampler needs it) and ``None``
            under greedy.
        """


class HostDrafter:
    """The host reference ``DFlashDraftModel``, with its transformers KV cache."""

    def __init__(self, model, target):
        self.model = model
        self.target = target
        self.cache = None

    @property
    def block_size(self) -> int:
        return self.model.block_size

    @property
    def mask_token_id(self) -> int:
        return self.model.mask_token_id

    @property
    def target_layer_ids(self):
        return list(self.model.target_layer_ids)

    def reset(self) -> None:
        from transformers import DynamicCache

        self.cache = DynamicCache(config=self.model.config)

    @torch.inference_mode()
    def propose(self, taps, block_ids, start, *, temperature, top_p, top_k):
        from models.demos.blackhole.qwen36.reference.dflash.generate import _sample_probs, _sampling_probs, truncate_kv

        if self.cache is None:
            self.reset()
        q_len = block_ids.shape[1]
        ctx_len = taps.shape[1]
        # position_ids spans the drafter's K/V axis: the new context rows, then the block.
        position_ids = torch.arange(start - ctx_len, start + q_len).unsqueeze(0)
        hidden = self.model(
            target_hidden=taps.to(self.model.dtype),
            noise_embedding=self.target.embed(block_ids).to(self.model.dtype),
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
        )[:, 1 - q_len :, :]
        # Keep the context rows this step appended, drop the noise block. See truncate_kv.
        truncate_kv(self.cache, start)

        logits = self.model.compute_logits(hidden, self.target.lm_head).float()
        if temperature > 0:
            probs = _sampling_probs(logits, temperature, top_p, top_k)
            return _sample_probs(probs), probs
        return torch.argmax(logits, dim=-1), None
