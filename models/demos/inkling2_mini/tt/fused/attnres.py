# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Fused block attention residual: rms(partial) + seal + depth attention in ONE program per call (E2's
residual.AttnRes: seal 4 ops + attend 5-9 ops; FusedRowOps.seal: 4 ops -> 0 or 1).

Drop-in for residual.RowOps.seal / residual.AttnRes (same signatures). Differences the integrator must know:
  * The block-state stack is ONE dense fp32 buffer per FusedRowOps (rows 0..4), updated in place. Every call
    returns that same buffer as "the stack". This is safe for the model's dataflow: rows >= depth are never read,
    and the loop restart (passes 2-3) rewrites rows 1.. with the same or newer values before they are read.
  * AttnRes(stack, partial, depth) leaves rms(partial) in row depth-1, so the boundary-layer seal of the same
    tensor right after is free (FusedRowOps.seal skips it).
  * Use FusedRowOps only for the block stack. MessageAttnRes builds its own 2-row stacks with seal(None, ..):
    keep residual.RowOps for it.
  * Outputs are per-module buffers, valid until that module's next call (the model consumes them in the app).
"""

import struct

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

MAX_DEPTH = 5


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


class FusedRowOps:
    def __init__(self, device, c=2304, eps=1e-6):
        self.device = device
        self.c = c
        self.eps = eps
        self.T = F.pages(c)
        self.stack = F.dense_tensor(device, torch.zeros(MAX_DEPTH, c))
        m = torch.zeros(1, 1, 32, 32)
        m[..., 0, :] = 1.0  # reduce scaler: row 0 of every face
        m[..., 16, :] = 1.0
        self.scaler = ttnn.from_torch(m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        self.fresh = {}  # row -> tensor whose rms is in that row now

    def seal(self, stack, h, k):
        """Stack with rms(h) in row k. Free if the last write of row k was rms(h) (an AttnRes partial)."""
        if self.fresh.get(k) is not h:
            self.run(h, k)
        return self.stack

    def run(self, partial, k, q=None, nq=0, out=None, out_rows=None, out_dense=None):
        F.check_row_input(partial)
        T, ntc, d = self.T, self.c // 32, k + 1
        assert 0 <= k < MAX_DEPTH and self.c % 32 == 0
        named = dict(
            ntc=ntc,
            pages=T,
            nq=nq,
            k=k,
            has_weight=0,
            out_rows=nq if out_rows is None else out_rows,
            split3=0,
            split_out=0,
        )
        reader = F.kernel(
            "attnres_reader.cpp",
            ttnn.ReaderConfigDescriptor(),
            named,
            [
                *F.input_args(partial),
                q.buffer_address() if q is not None else 0,
                self.stack.buffer_address(),
                self.scaler.buffer_address(),
                0,  # no rms weight
            ],
        )
        writer = F.kernel(
            "attnres_writer.cpp",
            ttnn.WriterConfigDescriptor(),
            named,
            [
                self.stack.buffer_address(),
                out.buffer_address() if out is not None else 0,
                out_dense.buffer_address() if out_dense is not None else 0,
            ],
        )
        compute = F.kernel(
            "attnres_compute.cpp",
            F.compute_config([], full_sync=True),
            dict(pages=T, nq=nq, k=k, has_weight=0, inv_c=f32_bits(1.0 / self.c), eps=f32_bits(self.eps)),
        )
        f32 = ttnn.float32
        cbs = [
            F.cb(0, F.input_dtype(partial), T),
            F.cb(1, f32, max(1, nq * T)),
            F.cb(2, f32, max(1, k * T)),
            F.cb(3, f32, 1),
            F.cb(16, f32, T),
            F.cb(17, f32, max(1, nq * T)),
            F.stage_cb(24, ntc),
            F.cb(25, f32, 1),
            F.cb(26, f32, 1),
            F.cb(27, f32, max(1, nq * d)),
            F.cb(28, f32, max(1, nq * d)),
        ]
        tw = F.twin(partial)
        io = [partial] + ([tw[0]] if tw else []) + [self.scaler] + ([q] if q is not None else []) + [self.stack]
        io += ([out_dense] if out_dense is not None else []) + ([out] if out is not None else [])
        F.run(io, [reader, writer, compute], cbs)
        self.fresh[k] = partial
        return out


class FusedAttnRes:
    """softmax_depth(q . rms(S)^T / sqrt(C)) . rms(S), S = [sealed block states..., partial]."""

    def __init__(self, rows: FusedRowOps, weights, name, num_queries=1):
        w = weights[name]
        w = ttnn.to_torch(w) if isinstance(w, ttnn.Tensor) else w
        q = w.float().reshape(-1, w.shape[-1])[:num_queries]
        self.rows = rows
        self.nq = num_queries
        self.q = F.dense_tensor(rows.device, q * q.shape[-1] ** -0.5)  # 1/sqrt(C) folded into q
        self.out = F.OutCache(rows.device)

    def __call__(self, stack, partial, depth: int, split=True, repeat_rows=None, row_out=True):
        """stack: ignored (rows.stack is used; rows 0..depth-2 sealed). partial: row. Returns a list of
        num_queries fp32 rows (split=False: one [1,1,num_queries,C] tensor). repeat_rows=3 with 1 query: the
        output in rows 0..2 (replaces the model's ttnn.repeat before the q/k/v message gates)."""
        r = self.rows
        n_out = repeat_rows or self.nq
        out, out_dense = self.out.get_twin(("out", n_out), n_out, r.c, ttnn.float32, nvec=self.nq)
        # row_out=False: only the dense twin is written (E10c); the returned tensor is then valid only as the input
        # of another fused kernel (which reads the twin), never for a ttnn op.
        out = r.run(partial, depth - 1, self.q, self.nq, out, n_out if row_out else 0, out_dense)
        if not split:
            return out
        return [out] if self.nq == 1 else [ttnn.slice(out, [0, 0, j, 0], [1, 1, j + 1, r.c]) for j in range(self.nq)]
