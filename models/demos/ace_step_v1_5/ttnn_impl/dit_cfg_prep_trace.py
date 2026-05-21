# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""One-shot trace for DiT CFG ``enc``/``ctx`` batch setup before the denoise loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

import ttnn

from .dit_sampling_ttnn import concat_duplicate_batch


@dataclass
class _DitCfgPrepTraceState:
    trace_id: Any = None
    enc_one_buf: Any = None
    null_emb_buf: Any = None
    null_rep_buf: Any = None
    enc_pipe_buf: Any = None
    ctx_one_buf: Any = None
    ctx_pipe_buf: Any = None
    s_enc: int = 0
    d_enc: int = 0
    op_event: Any = None
    stage_event: Any = None


def _host_bf16_like(dev_tensor: ttnn.Tensor) -> ttnn.Tensor:
    th = ttnn.to_torch(dev_tensor).contiguous()
    if th.dtype != torch.bfloat16:
        th = th.to(dtype=torch.bfloat16)
    return ttnn.from_torch(th, dtype=ttnn.bfloat16, layout=dev_tensor.layout)


class DitCfgPrepTrace:
    """Trace + 2CQ: ``null_rep`` broadcast, ``concat`` enc, ``concat_duplicate_batch`` ctx."""

    def __init__(self, device: Any) -> None:
        self.device = device
        self._st: Optional[_DitCfgPrepTraceState] = None

    def release(self) -> None:
        if self._st is None:
            return
        if self._st.trace_id is not None:
            try:
                ttnn.release_trace(self.device, self._st.trace_id)
            except Exception:
                pass
        for attr in (
            "enc_one_buf",
            "null_emb_buf",
            "null_rep_buf",
            "enc_pipe_buf",
            "ctx_one_buf",
            "ctx_pipe_buf",
        ):
            t = getattr(self._st, attr, None)
            if t is not None:
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
        self._st = None

    def _body(self, st: _DitCfgPrepTraceState) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        # Use the pre-computed null_rep_buf directly — avoids reshape/repeat inside the
        # trace and the view/alias risk (ttnn.reshape may return a view of null_emb_buf;
        # deallocating the view inside the trace would corrupt null_emb_buf on replay).
        enc_pipe = ttnn.concat([st.enc_one_buf, st.null_rep_buf], dim=0)
        ctx_pipe = concat_duplicate_batch(st.ctx_one_buf)
        return enc_pipe, ctx_pipe

    def _capture(
        self, enc_hs_tt: ttnn.Tensor, ctx_tt: ttnn.Tensor, null_emb_tt: ttnn.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if not hasattr(ttnn, "clone"):
            raise RuntimeError("DitCfgPrepTrace requires ttnn.clone.")
        s_enc = int(enc_hs_tt.shape[1])
        d_enc = int(enc_hs_tt.shape[-1])

        # Pre-compute null_rep once — null_emb is constant for the lifetime of the model.
        # Doing this outside the trace avoids the reshape view/alias issue: if ttnn.reshape
        # returns a view of null_emb_buf and the trace deallocates the view, null_emb_buf
        # is corrupted on every subsequent replay.
        _n4 = ttnn.reshape(ttnn.clone(null_emb_tt), (1, 1, 1, d_enc))
        _nr4 = ttnn.repeat(_n4, (1, 1, s_enc, 1))
        _nr = ttnn.reshape(_nr4, (1, s_enc, d_enc))
        null_rep_buf = ttnn.clone(_nr)
        for _t in (_n4, _nr4, _nr):
            try:
                ttnn.deallocate(_t)
            except Exception:
                pass

        # Warm pass to prime program caches (null_rep_buf clone used so the persistent
        # buffer is not consumed by the warm forward).
        warm_st = _DitCfgPrepTraceState(
            enc_one_buf=ttnn.clone(enc_hs_tt),
            null_rep_buf=ttnn.clone(null_rep_buf),
            ctx_one_buf=ttnn.clone(ctx_tt),
            s_enc=s_enc,
            d_enc=d_enc,
        )
        warm_enc, warm_ctx = self._body(warm_st)
        ttnn.synchronize_device(self.device)
        for _t in (warm_enc, warm_ctx, warm_st.enc_one_buf, warm_st.null_rep_buf, warm_st.ctx_one_buf):
            try:
                ttnn.deallocate(_t)
            except Exception:
                pass

        # Persistent state for trace capture and replay.
        st = _DitCfgPrepTraceState(s_enc=s_enc, d_enc=d_enc)
        st.null_rep_buf = null_rep_buf
        st.enc_one_buf = ttnn.clone(enc_hs_tt)
        st.ctx_one_buf = ttnn.clone(ctx_tt)

        tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        st.enc_pipe_buf, st.ctx_pipe_buf = self._body(st)
        ttnn.end_trace_capture(self.device, tid, cq_id=0)
        ttnn.synchronize_device(self.device)
        st.trace_id = tid
        st.op_event = ttnn.record_event(self.device, 0)
        self._st = st
        return ttnn.clone(st.enc_pipe_buf), ttnn.clone(st.ctx_pipe_buf)

    def build(
        self,
        enc_hs_tt: ttnn.Tensor,
        ctx_tt: ttnn.Tensor,
        null_emb_tt: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if not hasattr(ttnn, "begin_trace_capture"):
            s_enc = int(enc_hs_tt.shape[1])
            d_enc = int(enc_hs_tt.shape[-1])
            null_4d = ttnn.reshape(null_emb_tt, (1, 1, 1, d_enc))
            null_rep_4d = ttnn.repeat(null_4d, (1, 1, s_enc, 1))
            null_rep = ttnn.reshape(null_rep_4d, (1, s_enc, d_enc))
            enc_out = ttnn.concat([enc_hs_tt, null_rep], dim=0)
            ctx_out = concat_duplicate_batch(ctx_tt)
            for t in (null_4d, null_rep_4d, null_rep):
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
            return enc_out, ctx_out

        if self._st is None or int(enc_hs_tt.shape[1]) != self._st.s_enc:
            self.release()
            return self._capture(enc_hs_tt, ctx_tt, null_emb_tt)

        st = self._st
        if st.op_event is not None:
            ttnn.wait_for_event(1, st.op_event)
        ttnn.copy(enc_hs_tt, st.enc_one_buf)
        ttnn.copy(ctx_tt, st.ctx_one_buf)
        # null_rep_buf is constant (null_emb never changes) — no update needed on replay.
        st.stage_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, st.stage_event)
        ttnn.execute_trace(self.device, st.trace_id, cq_id=0, blocking=True)
        st.op_event = ttnn.record_event(self.device, 0)
        ttnn.synchronize_device(self.device)
        return ttnn.clone(st.enc_pipe_buf), ttnn.clone(st.ctx_pipe_buf)


__all__ = ["DitCfgPrepTrace"]
