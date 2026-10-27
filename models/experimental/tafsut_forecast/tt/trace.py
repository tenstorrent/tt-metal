# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""One captured trace per input shape ``(B, bucket context)``: persistent device inputs, refilled per call."""

from __future__ import annotations

import torch
import ttnn

from models.experimental.tafsut_forecast.reference import dataflow as df
from models.experimental.tafsut_forecast.tt.model import TtTafsut


class TracedForward:
    def __init__(self, tt: TtTafsut):
        self.tt = tt
        self._traces: dict = {}  # key -> (trace id, device inputs, device output, constants)

    def _key(self, prep: df.Prepared):
        return (prep.tokens.shape[0], prep.ctx_len)

    def upload(self, prep: df.Prepared):
        """Fill the shape's persistent inputs (captures the trace on first use)."""
        key = self._key(prep)
        host = self.tt.host_inputs(prep)
        if key not in self._traces:
            c = self.tt.constants(*key)
            inp = self.tt.upload_inputs(host)
            out = self.tt.forward_device(inp, c)  # compile run
            ttnn.deallocate(out)
            tid = ttnn.begin_trace_capture(self.tt.device, cq_id=0)
            out = self.tt.forward_device(inp, c)
            ttnn.end_trace_capture(self.tt.device, tid, cq_id=0)
            self._traces[key] = (tid, inp, out, c)
        else:
            self.tt.upload_inputs(host, self._traces[key][1])
        return key

    def execute(self, prep: df.Prepared):
        tid, _, out, _ = self._traces[self._key(prep)]
        ttnn.execute_trace(self.tt.device, tid, cq_id=0, blocking=False)
        return out

    def __call__(self, prep: df.Prepared) -> torch.Tensor:
        self.upload(prep)
        return ttnn.to_torch(self.execute(prep)).float()

    def forecast(self, x, horizon: int, heads: list | None = None) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float32)
        out, remaining = [], horizon
        while remaining > 0:
            take = min(remaining, df.NFUT * df.PATCH)
            prep = self.tt.prepare_inputs(x)
            head = self(prep)
            if heads is not None:
                heads.append(head)
            cur = self.tt.postprocess(head, prep, take)
            out.append(cur)
            remaining -= take
            if remaining > 0:
                x = torch.cat([x, cur[:, :, df.MEDIAN_IDX]], -1)[:, -df.MAX_CTX :]
        return torch.cat(out, 1)

    def release(self):
        for tid, inp, out, _ in self._traces.values():
            ttnn.release_trace(self.tt.device, tid)
            self.tt.deallocate_inputs(inp)
            ttnn.deallocate(out)
        self._traces.clear()
