# SPDX-License-Identifier: Apache-2.0
"""Chunked, traced prefill for MiMo-V2.6-Flash on the 4x8 mesh (asm agent).

Every mesh row processes ONE chunk (CH tokens, default 2048) of ONE of its users per step, rows in lock-step; rows are independent (DP), so each row has its own
queue of users with ragged lengths. All position dependent state (RoPE indices, chunk start, page-table slices) lives in row-sharded device tensors written
before each replay, so the whole model forward of a chunk (embedding -> all layers) is captured once in one trace and replayed. KV is written straight into the paged
pool of tt/kv_pool.py (GA: contiguous blocks per slot; SWA: ring with aliased page table). A row with no work left runs a dummy chunk into the scratch slot.
Pad tokens of a ragged last chunk repeat the user's own prompt tokens cyclically (so MoE routing of pad tokens looks like real routing instead of piling onto
a few experts); their KV rows are written beyond the user's length and are never read (GA: cur_pos masks; SWA: ring slots >= 513 positions older than the window).
"""
import time

import torch

import ttnn
from models.demos.blackhole.mimo_v26.tt.attention import BLOCK, HIDDEN, _pad8

ROWS, COLS = 4, 8


class ChunkedPrefill:
    def __init__(self, model, trace=True, trace_dram_log=None):
        self.m, self.mesh = model, model.mesh
        lay = model.layout
        assert lay is not None, "build MiMoModel(layout=KVLayout(...))"
        self.lay, self.CH, self.use_trace = lay, lay.chunk, trace
        self.attn = {}
        for lyr in model.layers:
            self.attn.setdefault("swa" if lyr.swa else "ga", lyr.attn)
        self.io = {k: a.prefill_io(self.CH) for k, a in self.attn.items()}
        self.rowmap = ttnn.ShardTensor2dMesh(self.mesh, self.mesh.shape, dims=(0, None))
        self.tok = ttnn.from_torch(
            torch.zeros(ROWS, self.CH, dtype=torch.int32),
            device=self.mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.rowmap,
        )
        self.tid = None
        self.out = None
        self.times = []

    # ---------------------------------------------------------------- inputs
    def _host(self, t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.rowmap)

    def set_step(self, tokens, slots, starts):
        """tokens [ROWS, CH] int; slots[r] = user slot of row r (scratch slot for idle rows); starts[r] = chunk start position."""
        CH = self.CH
        ttnn.copy_host_to_device_tensor(self._host(tokens.to(torch.int32), ttnn.uint32), self.tok)
        for kind, a in self.attn.items():
            io = self.io[kind]
            rope, fill, sd = [], [], []
            for r in range(ROWS):
                full = a.logical_page_row(slots[r])
                nb0 = starts[r] // BLOCK
                rope.append(torch.arange(starts[r], starts[r] + CH, dtype=torch.int32))
                fill.append(full[nb0 : nb0 + CH // BLOCK].to(torch.int32))
                sd.append(_pad8(full.reshape(1, -1).to(torch.int32))[0])
            host = dict(
                rope_idx=self._host(torch.stack(rope), ttnn.uint32),
                fill_pt=self._host(torch.stack(fill), ttnn.int32),
                sdpa_pt=self._host(torch.stack(sd), ttnn.int32),
                start=self._host(torch.tensor(starts, dtype=torch.int32), ttnn.int32),
            )
            for k, h in host.items():
                ttnn.copy_host_to_device_tensor(h, io[k])

    # ---------------------------------------------------------------- forward / trace
    def _forward(self):
        m = self.m
        e = ttnn.embedding(self.tok, m.embed, layout=ttnn.TILE_LAYOUT)
        h = ttnn.reshape(e, (1, 1, self.CH, HIDDEN))
        for lyr in m.layers:
            h2 = lyr.chunk(h, self.io["swa" if lyr.swa else "ga"])
            ttnn.deallocate(h)
            h = h2
        return h

    def run_step(self):
        """Run the chunk currently loaded by set_step; returns the persistent output tensor [1,1,CH,4096] (valid until the next call)."""
        t0 = time.perf_counter()
        if not self.use_trace:
            for lyr in self.m.layers:
                lyr.check_cap = False
            out = self._forward()
            ttnn.synchronize_device(self.mesh)
            self.times.append((time.perf_counter() - t0) * 1e3)
            return out
        if self.tid is None:
            out = (
                self._forward()
            )  # eager warm-up: compiles, builds MoE shared buffers, capacity-checked on this real chunk
            ttnn.deallocate(out)
            for lyr in self.m.layers:
                lyr.check_cap = False
            ttnn.synchronize_device(self.mesh)
            self.tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self.out = self._forward()
            ttnn.end_trace_capture(self.mesh, self.tid, cq_id=0)
            ttnn.synchronize_device(self.mesh)
            t0 = time.perf_counter()
        ttnn.execute_trace(self.mesh, self.tid, cq_id=0, blocking=True)
        self.times.append((time.perf_counter() - t0) * 1e3)
        return self.out

    def release(self):
        if self.tid is not None:
            ttnn.release_trace(self.mesh, self.tid)
            self.tid = None

    # ---------------------------------------------------------------- scheduling
    def prefill(self, prompts, collect_full=False, on_step=None):
        """prompts[r] = list of token lists (users of mesh row r, user u -> slot u, <= lay.users). Returns dict:
        last[(r,u)] = hidden [4096] bf16 of the last prompt token (after all layers, before the final norm); full[(r,u)] = [len,4096] if collect_full.
        """
        CH, U = self.CH, self.lay.users
        queues = []
        for r in range(ROWS):
            q = []
            for u, toks in enumerate(prompts[r] if r < len(prompts) else []):
                assert u < U and len(toks) > 0
                n = len(toks)
                nch = -(-n // CH)
                assert nch * CH <= self.lay.max_pos, (n, CH, self.lay.max_pos)
                q += [(u, c * CH, n, c == nch - 1) for c in range(nch)]
            queues.append(q)
        nsteps = max(len(q) for q in queues)
        last, full = {}, {}
        for step in range(nsteps):
            toks = torch.zeros(ROWS, CH, dtype=torch.int64)
            slots, starts, info = [], [], []
            for r in range(ROWS):
                if step < len(queues[r]):
                    u, st, n, lastc = queues[r][step]
                    src = torch.tensor(prompts[r][u], dtype=torch.int64)
                    real = src[st : st + CH]
                    pad = CH - real.numel()
                    if pad:
                        real = torch.cat([real, src.repeat(-(-pad // src.numel()))[:pad]])
                    toks[r] = real
                    slots.append(u)
                    starts.append(st)
                    info.append((u, st, n, lastc))
                else:  # idle row: dummy chunk into the scratch slot
                    src = torch.tensor(prompts[r][0] if len(prompts) > r and prompts[r] else [1], dtype=torch.int64)
                    toks[r] = src.repeat(-(-CH // src.numel()))[:CH]
                    slots.append(U)
                    starts.append(0)
                    info.append(None)
            self.set_step(toks, slots, starts)
            out = self.run_step()
            need = collect_full or any(i is not None and i[3] for i in info)
            if need:
                h = self.m.rows_to_host(out).bfloat16()  # [ROWS, CH, 4096]
                for r, i in enumerate(info):
                    if i is None:
                        continue
                    u, st, n, lastc = i
                    nv = min(CH, n - st)
                    if collect_full:
                        full.setdefault((r, u), []).append(h[r, :nv].clone())
                    if lastc:
                        last[(r, u)] = h[r, nv - 1].clone()
            if on_step:
                on_step(step, info)
        if collect_full:
            full = {k: torch.cat(v, 0) for k, v in full.items()}
        return dict(last=last, full=full, steps=nsteps)

    def first_tokens(self, last):
        """last[(r,u)] -> {(r,u): (token id, logits row)} through the final norm + lm_head (<=32 users per row per call)."""
        x = torch.zeros(ROWS, 32, HIDDEN)
        for (r, u), v in last.items():
            x[r, u] = v.float()
        lg = self.m.final_logits(self.m.upload_rows(x))
        return {k: (int(lg[k[0], k[1]].argmax()), lg[k[0], k[1]]) for k in last}
