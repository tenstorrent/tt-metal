# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""One captured trace per input shape ``(B, bucket context)``: persistent device inputs, refilled per call."""

from __future__ import annotations

import torch
import ttnn

from models.experimental.tafsut_forecast.reference import dataflow as df
from models.experimental.tafsut_forecast.tt.model import TtTafsut


class TracedForward:
    """Every shape's persistent buffers (constants, inputs) are allocated before any trace is captured: a buffer
    allocated after a capture can sit in that trace's freed intermediates and is overwritten on replay (it broke
    ``long_horizon`` forward 2 at 512 rows, job 934). A new shape therefore releases and recaptures every trace."""

    def __init__(self, tt: TtTafsut):
        self.tt = tt
        self._inputs: dict = {}  # key -> persistent device inputs
        self._traces: dict = {}  # key -> (trace id, device output)

    def _key(self, prep: df.Prepared):
        return (prep.tokens.shape[0], prep.ctx_len)

    def _release_traces(self):
        for tid, out in self._traces.values():
            ttnn.release_trace(self.tt.device, tid)
            ttnn.deallocate(out)
        self._traces.clear()

    def _capture_all(self):
        dev = self.tt.device
        for key, inp in self._inputs.items():
            ttnn.deallocate(self.tt.forward_device(inp, self.tt.constants(*key)))  # compile run, no trace alive
        for key, inp in self._inputs.items():
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            out = self.tt.forward_device(inp, self.tt.constants(*key))
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            self._traces[key] = (tid, out)

    def upload(self, prep: df.Prepared):
        """Fill the shape's persistent inputs. A new shape first releases every trace, then allocates its constants
        and inputs (no allocation while a trace is alive), then recaptures every shape."""
        key = self._key(prep)
        if key not in self._inputs:
            self._release_traces()
            self._inputs[key] = self.tt.upload_inputs(self.tt.host_inputs(prep))  # host_inputs builds the constants
            self._capture_all()
        else:
            self.tt.upload_inputs(self.tt.host_inputs(prep), self._inputs[key])
        return key

    def execute(self, prep: df.Prepared):
        tid, out = self._traces[self._key(prep)]
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
        self._release_traces()
        for inp in self._inputs.values():
            self.tt.deallocate_inputs(inp)
        self._inputs.clear()
