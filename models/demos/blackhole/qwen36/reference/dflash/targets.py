# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""What DFlash needs from a target model, and its host implementation.

:class:`SpeculativeTarget` is deliberately small — six methods. That list is everything the
speculative loop asks of Qwen3.6-27B, with no HF cache objects or ttnn tensors leaking into
:mod:`.generate`.

* :class:`HFTarget` wraps the host ``Qwen3_5ForCausalLM`` — the golden reference.
* The device implementation, ``TtTarget``, lives in :mod:`models.demos.blackhole.qwen36.tt.dflash.target`.

Running the same loop against both is the point: the device's tokens must match the host's, and
where they do not, the six methods say exactly which one to look at.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class SpeculativeTarget(Protocol):
    """The target-model surface :func:`~.generate.dflash_generate` drives."""

    def reset(self) -> None:
        """Drop all cached state and start a new sequence at position 0."""

    def forward(self, ids: torch.Tensor, start: int, *, all_logits: bool = True):
        """Run ``ids`` (``[1, S]``) at absolute position ``start``, advancing cached state.

        Returns ``(logits, taps)``: logits ``[1, S, V]`` fp32 (or ``[1, 1, V]`` when
        ``all_logits=False`` — prefill only wants the last row), and taps ``[1, S, n*H]`` fp32, the
        target's residual stream at the drafter's tap layers concatenated in the drafter's order.
        """

    def snapshot(self) -> Any:
        """Capture whatever :meth:`restore` needs to undo the next :meth:`forward`."""

    def restore(self, snap: Any, length: int) -> None:
        """Roll cached state back to exactly ``length`` tokens, undoing a rejected block."""

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """The target's raw input embedding of ``ids`` — the drafter's noise input. ``[1, S, H]``."""

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        """The target's output head applied to drafter hidden states. ``[1, S, V]`` fp32."""


class HFTarget:
    """Host ``Qwen3_5ForCausalLM``. Rolls GDN state back by snapshot + replay — see :mod:`.generate`."""

    #: `restore` only rewinds; the accepted prefix must be replayed to re-advance the GDN state.
    replays_after_rollback = True

    def __init__(self, model, tap_layer_ids):
        self.model = model
        self.tap_layer_ids = list(tap_layer_ids)
        self.cache = None

    @property
    def hidden_size(self) -> int:
        return self.model.config.get_text_config().hidden_size

    def reset(self) -> None:
        from transformers import DynamicCache

        self.cache = DynamicCache(config=self.model.config)

    @torch.inference_mode()
    def forward(self, ids, start, *, all_logits=True):
        from models.demos.blackhole.qwen36.reference.dflash.dflash import extract_context_feature

        ids = ids.to(self.model.device)
        position_ids = torch.arange(start, start + ids.shape[1], device=ids.device).unsqueeze(0)
        out = self.model(
            ids,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
            output_hidden_states=True,
            **({} if all_logits else {"logits_to_keep": 1}),
        )
        taps = extract_context_feature(out.hidden_states, self.tap_layer_ids).float()
        return out.logits.float(), taps

    def snapshot(self):
        """Clone every linear-attention layer's conv + recurrent state.

        Only the GDN layers need cloning: attention KV truncates exactly under ``crop``, but
        ``crop`` is a documented no-op for ``linear_attention``.
        """
        snapshot: dict[int, dict] = {}
        for idx, layer in enumerate(self.cache.layers):
            if not (hasattr(layer, "recurrent_states") and hasattr(layer, "conv_states")):
                continue
            conv, rec = layer.conv_states, layer.recurrent_states
            snapshot[idx] = {
                "conv": conv.clone() if isinstance(conv, torch.Tensor) else None,
                "recurrent": rec.clone() if isinstance(rec, torch.Tensor) else None,
                "has_previous_state": getattr(layer, "has_previous_state", False),
            }
        return snapshot

    def restore(self, snap, length):
        for idx, saved in snap.items():
            layer = self.cache.layers[idx]
            for attr, key in (("conv_states", "conv"), ("recurrent_states", "recurrent")):
                current, previous = getattr(layer, attr), saved[key]
                if not isinstance(current, torch.Tensor):
                    continue
                if previous is not None:
                    # copy_ rather than assignment, to keep the (cudagraph-static) address.
                    current.copy_(previous)
                else:
                    # Uninitialised when snapshotted but written since: the pre-block truth is "no
                    # state", so zero it rather than leaving this forward's values behind.
                    current.zero_()
            layer.has_previous_state = saved["has_previous_state"]
        # Qwen3.6-27B has no sliding layers, so this is exact for every attention layer.
        self.cache.crop(length)

    def embed(self, ids):
        weight = self.model.get_input_embeddings().weight
        return torch.nn.functional.embedding(ids.to(weight.device), weight)

    def lm_head(self, hidden):
        head = getattr(self.model, "lm_head", None) or self.model.get_output_embeddings()
        return head(hidden.to(head.weight.dtype)).float()
