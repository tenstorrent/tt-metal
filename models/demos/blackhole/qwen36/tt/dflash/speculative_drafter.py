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

    #: Context above this many rows is a prompt and is fed in ``INGEST_CHUNK``-row pieces.
    INGEST_ABOVE = 64
    INGEST_CHUNK = 2048

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

    # ---- traced steady-state step ------------------------------------------------------------

    _trace_id = None

    def enable_trace(self) -> None:
        """Capture the steady-state draft step. Needs the target's verify trace already captured
        (the step reads its tap buffers) and every eager program already compiled."""
        import ttnn

        d, t = self.drafter, self.target
        assert d._cap is not None, "the traced step needs a fixed-capacity drafter (ctx_capacity)"
        assert getattr(t.model, "_vfy_trace_id", None) is not None, "capture the verify trace first"
        self.release_trace()
        Q = d.block_size
        d.alloc_trace_io()
        self._tok_buf = ttnn.zeros([1, Q], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=d.device)
        ttnn.deallocate(self._trace_body())  # compile: inputs are benign (no row selected)
        ttnn.synchronize_device(d.device)
        self._trace_id = ttnn.begin_trace_capture(d.device, cq_id=0)
        self._ids_out = self._trace_body()
        ttnn.end_trace_capture(d.device, self._trace_id, cq_id=0)

    def release_trace(self) -> None:
        import ttnn

        if self._trace_id is not None:
            ttnn.release_trace(self.drafter.device, self._trace_id)
            self._trace_id = None

    def _trace_body(self):
        import ttnn

        d, t = self.drafter, self.target
        noise = t._embed_tokens(self._tok_buf, d.block_size, own=False)
        hidden = d.forward_traced_body(noise, t.model.verify_taps())
        ttnn.deallocate(noise)
        ids = t.draft_ids_traced(hidden)
        ttnn.deallocate(hidden)
        return ids

    def _propose_traced(self, block_ids, start, rows):
        import ttnn

        d = self.drafter
        Q = d.block_size
        host = ttnn.Tensor([int(v) for v in block_ids[0].tolist()], [1, Q], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self._tok_buf)
        d.stage_trace_inputs(start, rows)
        ttnn.execute_trace(d.device, self._trace_id, cq_id=0, blocking=False)
        ids = ttnn.to_torch(ttnn.get_device_tensors(self._ids_out)[0]).reshape(-1)[:Q]
        return ids[-(Q - 1) :].long().reshape(1, -1)

    def propose(self, taps, block_ids, start, *, temperature, top_p, top_k):
        import ttnn
        from models.demos.blackhole.qwen36.reference.dflash.generate import _sample_probs, _sampling_probs

        q_len = block_ids.shape[1]
        rows = taps[0].shape[-2]
        if (
            self._trace_id is not None
            and temperature == 0
            and q_len == self.drafter.block_size
            and rows == self.target.last_produced
            and self.drafter.context_len + rows == start
        ):
            return self._propose_traced(block_ids, start, rows), None
        if self.drafter._cap is not None and rows > self.INGEST_ABOVE:
            # A prompt: commit all but the last block-width of rows in bounded chunks, so the step
            # below sees the same shapes as any other and no prompt-sized tensor is ever built.
            head = rows - self.drafter.block_size
            for lo in range(0, head, self.INGEST_CHUNK):
                hi = min(lo + self.INGEST_CHUNK, head)
                part = [ttnn.slice(t, (0, 0, lo, 0), (1, 1, hi, t.shape[-1])) for t in taps]
                self.drafter.ingest_context(part)
                for t in part:
                    ttnn.deallocate(t)
            taps = [ttnn.slice(t, (0, 0, head, 0), (1, 1, rows, t.shape[-1])) for t in taps]
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
