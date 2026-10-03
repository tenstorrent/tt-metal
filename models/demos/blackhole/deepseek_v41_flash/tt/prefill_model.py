# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash device PREFILL of a batch of prompts: tokens -> embedding -> layers (+Engram) -> last-token logits, leaving
the decode state of every layer on the device (see docs/superpowers/specs/2026-10-02-dsv41-device-prefill-design.md).

Layout: 4 mesh rows x U users per row (U = 4 at batch 16), every user's prompt padded to ``Sp`` (multiple of 64) with its last token;
the mesh row's R = U*Sp tokens are processed in chunks of T = 32 (user-major) by the token-wise blocks (mHC, router, MoE, shared expert,
Engram), attention runs over all R tokens of the row (``DSV41PrefillAttention``).
"""

import time

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_attention import pad_len

T = 32


def to_chunks(x, rows, users, Sp, S, tail):
    """x [rows*users, S, ...] host -> list of chunks [rows*T, 1, *tail] (per row users x Sp tokens user-major, padded with the last real token)."""
    B = x.shape[0]
    xp = torch.cat([x, x[:, -1:].expand(B, Sp - S, *x.shape[2:])], dim=1)
    xr = xp.reshape(rows, users * Sp, *x.shape[2:])
    return [xr[:, c * T : (c + 1) * T].reshape(rows * T, 1, *tail) for c in range(users * Sp // T)]


def from_chunks(ch, rows, users, Sp, S, tail):
    """inverse of ``to_chunks`` for host chunks [rows*T, ...] -> [rows*users, S, *tail]."""
    xr = torch.stack([c.reshape(rows, T, *tail) for c in ch], dim=1).reshape(rows, users * Sp, *tail)
    return xr.reshape(rows * users, Sp, *tail)[:, :S]


class DSV41PrefillModel:
    def __init__(self, md, layers, embedding, head, engram=None, host_rows=None, users_per_row=4):
        """layers: list of (layer_id, DSV41PrefillLayer); engram: {layer_id: DSV41DeviceEngram}; host_rows: HostEngramRows."""
        self.md, self.layers, self.embedding, self.head = md, layers, embedding, head
        self.engram, self.host_rows = engram or {}, host_rows
        self.rows, self.cols = tuple(md.shape)
        self.U = users_per_row
        self.shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(self.rows, self.cols))
        self.pre32 = ttnn.from_torch(
            torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(T, 1, 1, 1),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        self.timing = {}

    def _up(self, t, dtype, layout):
        return ttnn.from_torch(
            t.contiguous(),
            device=self.md,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.shard,
        )

    def sync(self, name, t0):
        ttnn.synchronize_device(self.md)
        self.timing[name] = self.timing.get(name, 0.0) + time.perf_counter() - t0

    # ---- device-input staging (persistent buffers: the whole forward can be captured as ONE trace) ----------------------
    def alloc_inputs(self, S):
        """Persistent device buffers for the token chunks and the Engram rows of a prompt length ``S`` (filled by ``write_inputs``)."""
        Sp = pad_len(S)
        n = self.U * Sp // T
        rows = self.rows
        self.n_chunks, self.S, self.Sp = n, S, Sp
        self.tok_dev = [
            self._up(torch.zeros(rows * T, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) for _ in range(n)
        ]
        self.erows_dev = {
            lid: [
                self._up(torch.zeros(rows * T, 1, 1, e.kin, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                for _ in range(n)
            ]
            for lid, e in self.engram.items()
        }

    def _host(self, t, dtype, layout):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, mesh_mapper=self.shard)

    def write_inputs(self, tokens, hashes=None):
        """Host work of a prefill: pad + chunk the tokens, hash the prompt (the host Engram history must see it once), gather the Engram
        rows and copy everything into the persistent device buffers."""
        B, S = tokens.shape
        assert S == self.S
        rows, U, Sp = self.rows, self.U, self.Sp
        t0 = time.perf_counter()
        for c, dev in zip(to_chunks(tokens, rows, U, Sp, S, (1,)), self.tok_dev):
            ttnn.copy_host_to_device_tensor(self._host(c.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), dev)
        if self.engram:
            if hashes is None:
                hashes = self.host_rows.hashes(tokens, 0)
            self.timing["engram_host_hash"] = time.perf_counter() - t0
            for lid, devs in self.erows_dev.items():
                r = self.host_rows.rows(lid, hashes)  # [B, S, Kin] bf16
                for c, dev in zip(to_chunks(r, rows, U, Sp, S, (1, 1, r.shape[-1])), devs):
                    ttnn.copy_host_to_device_tensor(
                        self._host(c.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT), dev
                    )
        self.timing["host_inputs"] = time.perf_counter() - t0

    def forward_device(self, hook=None, profile=False):
        """Embedding -> layers (+Engram) -> head on the chunks holding the last real token of each user. Only enqueues device ops
        (traceable when ``hook`` / ``profile`` are off). Returns {local user: logits shard [1,1,1,vocab/cols] of its last token}.
        """
        S, Sp = self.S, self.Sp

        def sync(name, t0):
            if profile:
                self.sync(name, t0)

        t0 = time.perf_counter()
        xs = [self.embedding.forward(tt)[0] for tt in self.tok_dev]
        pres = [self.pre32 for _ in xs]
        sync("embedding", t0)
        for lid, pl in self.layers:
            if lid in self.engram:
                t0 = time.perf_counter()
                new = [self.engram[lid].forward(x, rt) for x, rt in zip(xs, self.erows_dev[lid])]
                for x in xs:
                    ttnn.deallocate(x)
                xs = new
                sync("engram_dev", t0)
            t0 = time.perf_counter()
            outs, pouts = pl.forward(xs, pres, S)
            if xs[0] is not outs[0]:
                for x in xs:
                    ttnn.deallocate(x)
            if pres[0] is not self.pre32:
                for p in pres:
                    ttnn.deallocate(p)
            xs, pres = outs, pouts
            sync("layers", t0)
            if hook is not None:
                hook(lid, xs, pres)
        t0 = time.perf_counter()
        full, lg = {}, {}
        for u in range(self.U):
            c, off = divmod(u * Sp + S - 1, T)
            if c not in full:
                full[c] = self.head.forward(xs[c], pres[c])
            lg[u] = ttnn.slice(
                full[c], [0, 0, off, 0], [1, 1, off + 1, full[c].shape[3]]
            )  # only the row of the last real token is read back
        sync("head", t0)
        return lg

    def read_logits(self, lg):
        """Host [B, vocab] logits of the last prompt token of every user from ``forward_device``'s outputs."""
        rows, U, S, Sp = self.rows, self.U, self.S, self.Sp
        out = torch.zeros(rows * U, 129280)
        for u in range(U):
            g = self.head.gather_logits(lg[u]).reshape(rows, -1)  # [rows, vocab]
            for r in range(rows):
                out[r * U + u] = g[r]
        return out

    def run(self, tokens, hook=None, hashes=None):
        """Eager convenience wrapper: tokens [B, S] -> logits [B, vocab] (host fp32) of the LAST prompt token of every user.
        ``hook(layer_id, outs, pouts)`` is called after every layer (diagnostics)."""
        B, S = tokens.shape
        assert B == self.rows * self.U
        self.timing = {}
        if getattr(self, "S", None) != S:
            self.alloc_inputs(S)
        self.write_inputs(tokens, hashes)
        lg = self.forward_device(hook=hook, profile=True)
        t0 = time.perf_counter()
        out = self.read_logits(lg)
        self.timing["readback"] = time.perf_counter() - t0
        return out

    def capture_trace(self):
        """Capture the whole prefill (embedding .. head) as ONE trace; call after an eager ``run`` warmed every program."""
        ttnn.synchronize_device(self.md)
        self.trace_id = ttnn.begin_trace_capture(self.md, cq_id=0)
        self.trace_out = self.forward_device()
        ttnn.end_trace_capture(self.md, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.md)

    def run_traced(self, tokens, hashes=None):
        self.timing = {}
        self.write_inputs(tokens, hashes)
        t0 = time.perf_counter()
        ttnn.execute_trace(self.md, self.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.md)
        self.timing["trace_replay"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = self.read_logits(self.trace_out)
        self.timing["readback"] = time.perf_counter() - t0
        return out

    def last_token_logits(self, xs, pres, S, Sp):
        """Final hc_pre + norm + LM head on the chunks that hold the last real token of each user -> [B, vocab] host fp32."""
        rows, U = self.rows, self.U
        out = torch.zeros(rows * U, 129280)
        cache = {}
        for u in range(U):
            c, off = divmod(u * Sp + S - 1, T)
            if c not in cache:
                lg = self.head.forward(xs[c], pres[c])  # [1,1,T,vocab/cols] per device
                cache[c] = self.head.gather_logits(lg).reshape(rows, T, -1)  # [rows, T, vocab]
                ttnn.deallocate(lg)
            for r in range(rows):
                out[r * U + u] = cache[c][r, off]
        return out
