# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Traced device forward per (s, L, Bp, r0, P): a persistent token buffer, one captured trace, fresh inputs copied in.

PLAN v1.21: every shape's persistent buffers (per-scale constants, input buffer) are allocated before any capture;
add all shapes with ``add`` first, then ``capture``. A buffer allocated while a trace is live can alias its freed
intermediates, so nothing is allocated between capture and the last replay (the host-side tokens are copied into
the existing buffer).

Until stage E solves holding several shape traces at once (D3 item 9), at most one trace is live in the process:
``capture`` asserts that no other trace is live and that it captures exactly one shape; ``release`` frees it.
"""

import ttnn

from models.experimental.flowstate_forecast.tt.model import TILE


_LIVE = [0]  # live traces in this process


class TracedForward:
    def __init__(self, tt):
        self.tt, self.device = tt, tt.device
        self.shapes = {}  # key -> {"inp", "trace", "out"}

    @staticmethod
    def key(meta, s):
        return (float(s), meta["L"], meta["Bp"], meta["r0"], meta["P"])

    def add(self, meta, s):
        k = self.key(meta, s)
        if k not in self.shapes:
            assert not any(v["trace"] is not None for v in self.shapes.values()), "add every shape before capture"
            self.tt.constants(s, meta["L"], meta["P"])
            inp = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, meta["L"] * meta["Bp"], TILE]), ttnn.float32, ttnn.TILE_LAYOUT, self.device
            )
            self.shapes[k] = {"inp": inp, "trace": None, "out": None}
        return k

    def capture(self):
        pending = [(k, v) for k, v in self.shapes.items() if v["trace"] is None]
        assert _LIVE[0] == 0 and len(pending) == 1, f"one live trace at most (D3 item 9): {_LIVE[0]} live, {len(pending)} new"
        for (s, L, Bp, r0, P), v in pending:  # every compile run before the first capture (its outputs freed)
            out, _ = self.tt.forward_device(v["inp"], s, L, Bp, r0, P)
            ttnn.deallocate(out)
        ttnn.synchronize_device(self.device)
        for (s, L, Bp, r0, P), v in pending:
            if v["trace"] is None:
                tid = ttnn.begin_trace_capture(self.device, cq_id=0)
                v["out"], _ = self.tt.forward_device(v["inp"], s, L, Bp, r0, P)
                ttnn.end_trace_capture(self.device, tid, cq_id=0)
                v["trace"] = tid
                _LIVE[0] += 1

    def upload(self, tok_dev_host, k):
        ttnn.copy_host_to_device_tensor(tok_dev_host, self.shapes[k]["inp"])

    def execute(self, k):
        ttnn.execute_trace(self.device, self.shapes[k]["trace"], cq_id=0, blocking=False)
        return self.shapes[k]["out"]

    def release(self):
        for v in self.shapes.values():
            if v["trace"] is not None:
                ttnn.release_trace(self.device, v["trace"])
                _LIVE[0] -= 1
        self.shapes.clear()
