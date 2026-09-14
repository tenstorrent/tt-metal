# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""What DFlash needs from a drafter, and the two implementations of it.

The mirror of :mod:`.targets`. :class:`SpeculativeDrafter` is two methods — ``reset`` and
``propose`` — and everything else about drafting hides behind them: building the noise embedding,
carrying the KV history, running the LM head, sampling.

* :class:`HostDrafter` wraps the host reference ``DFlashDraftModel``.
* :class:`TtDrafter` wraps the ttnn :class:`~...tt.dflash.drafter.TtDFlashDrafter`, and keeps the
  whole draft on the mesh: the target's taps arrive as device tensors and the draft logits come off
  the target's own resident LM head, so only the chosen token ids cross PCIe.

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


class TtDrafter:
    """The ttnn drafter. The draft never leaves the mesh except as token ids.

    The target must have been armed with ``set_residual_taps(..., keep_on_device=True)`` so its
    taps arrive as device tensors — :class:`~.targets.TtTarget` does that when handed a device
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
        self._trace_id = None

    def enable_traced_draft(self, q_len=None, ctx_pad=16, warm=2):
        """Capture the drafter's steady-state step, so a replay costs one dispatch instead of ~174.

        The drafter is host-dispatch-bound -- ~9 % device-utilized at 180 dispatches a step -- which
        is exactly the profile a trace erases. What had to happen first (stages 1 and 2) was making
        a step capturable at all: a fixed-capacity KV history so shapes and addresses stop moving,
        and staged inputs so nothing is built or uploaded mid-step.

        Captured: the noise embedding (from the staged token buffer) and the drafter forward. NOT
        captured, deliberately:

        * ``project_taps`` -- its input is whatever tensor the target's ``take_taps`` just produced,
          at a fresh address every step. Staging its OUTPUT is cheap; staging its input would mean
          fixing five more tap buffers for a handful of ops.
        * the KV commit -- ``commit_staged_context``'s offset advances with the accept count, and
          offsets are op attributes that bake at capture.
        * the LM head and its readback -- a device->host read is illegal inside a capture, and it
          reads the trace's OUTPUT tensor, whose address is stable across replays.

        Requires ``ctx_capacity``; steps that do not fit the captured shape fall back to eager (see
        :meth:`propose`).
        """
        import ttnn

        d = self.drafter
        assert d._cap is not None, "tracing the drafter needs TtDFlashDrafter(ctx_capacity=...)"
        q_len = d.block_size if q_len is None else q_len
        sb = d.alloc_step_buffers(q_len=q_len, ctx_pad=ctx_pad)
        dev = d.device

        # Warm with the SAME shapes the capture will record: an un-warmed capture records JIT
        # compilation instead of the step. Contents are irrelevant, only programs are being built.
        d.stage_step(start=ctx_pad, new_ctx=ctx_pad)
        d.stage_tokens(torch.zeros(1, q_len, dtype=torch.long))
        d._staged_new_ctx = ctx_pad
        for _ in range(warm):
            noise = self.target.embed_device_staged(sb["tok"])
            hidden = d.forward(None, noise, ctx_pad, staged=True)
            ttnn.deallocate(noise)
            ttnn.deallocate(hidden)
        ttnn.synchronize_device(dev)

        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        try:
            noise = self.target.embed_device_staged(sb["tok"])
            self._trace_hidden = d.forward(None, noise, ctx_pad, staged=True)
            ttnn.deallocate(noise)
        finally:
            # ALWAYS end the capture. A capture that raises without this leaves the command queue in
            # capture mode, which hangs the process for ~12 minutes in teardown and then blocks the
            # next run on a lock it holds against itself.
            ttnn.end_trace_capture(dev, tid, cq_id=0)
        self._trace_id = tid
        self._trace_q_len, self._trace_ctx_pad = q_len, ctx_pad
        return tid

    def release_draft_trace(self):
        import ttnn

        if getattr(self, "_trace_id", None) is not None:
            ttnn.release_trace(self.drafter.device, self._trace_id)
            self._trace_id = None

    def propose(self, taps, block_ids, start, *, temperature, top_p, top_k):
        from models.demos.blackhole.qwen36.reference.dflash.generate import _sample_probs, _sampling_probs

        q_len = block_ids.shape[1]
        kv_source = self.drafter.project_taps(taps)
        new_ctx = kv_source.shape[-2] if kv_source is not None else 0
        hidden = self._hidden(kv_source, block_ids, start, q_len, new_ctx)

        # Slot 0 is the anchor; the drafted slots are 1..q_len-1, i.e. hidden's trailing rows.
        logits = self.target.lm_head_device(hidden, keep_rows=q_len - 1).float()
        if temperature > 0:
            probs = _sampling_probs(logits, temperature, top_p, top_k)
            return _sample_probs(probs), probs
        return torch.argmax(logits, dim=-1), None

    def _traceable(self, q_len, new_ctx):
        """Whether this step matches the captured shape. A trace serves one shape and no other."""
        return (
            getattr(self, "_trace_id", None) is not None
            and q_len == self._trace_q_len
            # The prompt step hands over the whole prompt's taps at once and is the one step that
            # exceeds ctx_pad. It runs eagerly, once, outside the steady state the capture serves.
            and new_ctx <= self._trace_ctx_pad
        )

    def _hidden(self, kv_source, block_ids, start, q_len, new_ctx):
        """The drafter forward, replayed from the trace when this step fits it."""
        import ttnn

        if not self._traceable(q_len, new_ctx):
            noise = self.target.embed_device(block_ids)
            return self.drafter.forward(kv_source, noise, start)

        d = self.drafter
        d.stage_step(start, new_ctx)
        d.stage_taps(kv_source, new_ctx)
        d.stage_tokens(block_ids)
        # No synchronize_device here, deliberately. An explicit barrier after the replay measured
        # 0.82x -- SLOWER than eager dispatch -- because it makes the host wait for the device every
        # step, which the eager path never does. It is not needed for correctness either: the commit
        # below and the LM head both queue on the same command queue behind the trace, so the device
        # orders them, and the logits readback is the natural sync point.
        ttnn.execute_trace(d.device, self._trace_id, cq_id=0, blocking=False)
        # The append the capture could not record, at the offset that actually varies.
        d.commit_staged_context(new_ctx)
        return self._trace_hidden
