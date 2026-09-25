# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The device DFlash drafter behind the speculative loop's drafter protocol.

:class:`TtDrafter` adapts :class:`~.drafter.TtDFlashDrafter` to
:class:`~...reference.dflash.drafters.SpeculativeDrafter`, which is what
:func:`~...reference.dflash.generate.dflash_generate` drives. It keeps the whole draft on the mesh:
the target's taps arrive as device tensors and the draft logits come off the target's own resident
LM head, so only the chosen token ids cross PCIe.
"""

from __future__ import annotations


class TtDrafter:
    """The ttnn drafter. The draft never leaves the mesh except as token ids.

    The target must have been armed with ``set_residual_taps(..., keep_on_device=True)`` so its
    taps arrive as device tensors — :class:`~.target.TtTarget` does that when handed a device
    drafter.
    """

    def __init__(self, tt_drafter, target):
        self.drafter = tt_drafter
        self.target = target

    @property
    def block_size(self) -> int:
        return self.drafter.block_size

    @property
    def mask_token_id(self) -> int:
        return self.drafter.mask_token_id

    @property
    def target_layer_ids(self):
        return self.drafter.target_layer_ids

    def reset(self) -> None:
        self.drafter.reset()

    def propose(self, taps, block_ids, start, *, temperature, top_p, top_k):
        from models.demos.blackhole.qwen36.reference.dflash.generate import _sample_probs, _sampling_probs

        q_len = block_ids.shape[1]
        kv_source = self.drafter.project_taps(taps)
        noise = self.target.embed_device(block_ids)
        hidden = self.drafter.forward(kv_source, noise, start)

        # Slot 0 is the anchor; the drafted slots are 1..q_len-1, i.e. hidden's trailing rows.
        if temperature > 0:
            # Sampling needs the whole distribution, so the logits still come back to host.
            logits = self.target.lm_head_device(hidden, keep_rows=q_len - 1).float()
            probs = _sampling_probs(logits, temperature, top_p, top_k)
            return _sample_probs(probs), probs
        # Greedy: the host only ever argmaxed these, so do it on device and move ids instead.
        return self.target.draft_ids_device(hidden, keep_rows=q_len - 1), None
