# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced greedy TP decode driver for Qwen3.5 on a 1xN mesh.

devloop=True (default): on-device token feedback. One trace = decode forward + a one-op greedy tail (all_gather of
the n per-die untilized logits rows + one argmax on every die: first max index, = torch.argmax over the full vocab ->
a replicated uint32 token) + a copy of that token into the decode token input, so the host never waits between
steps. cur_pos / RoPE for every step are built on host up front and enqueued (non-blocking writes) before each
replay; each step's token is read back non-blocking; one synchronize at the end. Bit-identical logits to the host
loop; -0.41 ms/token host-side.

devloop=False: the traced decode of demo/text_demo.py ``_run_tp_generation``, verbatim: persistent decode inputs, one
trace = decode forward + a per-shard on-device argmax and max value (untilize + argmax + a pad/reshape/max/max chain),
and per step: host input update -> execute_trace -> synchronize -> read the n (index, value) pairs -> host combine.

Reusable by the SP -> TP handoff: build the model with paged KV, construct TPTracedDecoder, capture() (after compiling any
op the handoff runs while the trace is resident), inject the state, then decode(first_token, T, n).
"""
import time

import torch
from loguru import logger

import ttnn


class TPTracedDecoder:
    def __init__(self, model, page_table, devloop=True):
        self.model = model
        self.mesh = model.mesh_device
        self.nd = model.num_devices
        self.vocab = model.args.vocab_size
        self.per_shard = self.vocab // self.nd
        self.page_table = page_table
        self.devloop = bool(devloop)
        self.ag_tail = self.devloop
        self.trace_id = None
        self.dev = None
        self.out = None  # trace outputs: legacy (idx, val, logits) | ag (tok, full_row)
        self.comp = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
        C = 32
        self._C = C
        self._R = (((self.per_shard + C - 1) // C) + 31) // 32 * 32
        logger.info(f"[tp_decode_driver] devloop={int(self.devloop)}")

    # ---------------------------------------------------------------------------------------------------------
    # legacy tail (text_demo _run_tp_generation, verbatim)
    def _maxval_dev(self, sl):
        R, C = self._R, self._C
        padded = ttnn.pad(sl, [(0, 0), (0, 0), (0, 0), (0, R * C - self.per_shard)], value=-1e30)
        grid = ttnn.reshape(padded, (1, 1, R, C))
        part = ttnn.max(grid, dim=-1)
        part_row = ttnn.reshape(part, (1, 1, 1, R))
        val = ttnn.max(part_row, dim=-1)
        for t in (padded, grid, part, part_row):
            ttnn.deallocate(t)
        return val

    def _argmax_dev(self, sl):
        rm = ttnn.to_layout(sl, ttnn.ROW_MAJOR_LAYOUT)
        idx = ttnn.argmax(rm, dim=-1, keepdim=False)
        ttnn.deallocate(rm)
        return idx, self._maxval_dev(sl)

    def _read_tok_legacy(self, idx_t, val_t):
        idxs = ttnn.to_torch(idx_t, mesh_composer=self.comp).reshape(-1)
        vals = ttnn.to_torch(val_t, mesh_composer=self.comp).reshape(-1)
        d = int(torch.argmax(vals).item())
        return d * self.per_shard + int(idxs[d].item())

    # ---------------------------------------------------------------------------------------------------------
    # all_gather + argmax tail
    def _ag_token(self, row):
        """row: this die's logits, row-major [1, 1, 1, vocab/nd] -> (uint32 token [1, 1, 1, 1] replicated, full row)."""
        m = self.model
        full = ttnn.experimental.all_gather_async(
            row,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=m.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=1,
            topology=m.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=m.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        tok = ttnn.argmax(full, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tok, full

    def _step_ops(self):
        """Everything inside the trace: decode forward + the greedy tail (+ the token feedback)."""
        m = self.model
        d = self.dev
        logits = m.ttnn_decode_forward(d[0], d[1], rot_mat_idxs=d[2], page_table=d[3])[0]
        if not self.ag_tail:
            idx, val = self._argmax_dev(logits)
            return (idx, val, logits)
        row = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(logits)
        tok, full = self._ag_token(row)
        ttnn.deallocate(row)
        if self.devloop:
            ttnn.copy(ttnn.reshape(tok, tuple(d[0].shape)), d[0])
        return (tok, full)

    # ---------------------------------------------------------------------------------------------------------
    def _host_inputs(self, token, pos):
        return self.model.prepare_decode_inputs_host(
            torch.tensor([[token]], dtype=torch.int32), torch.tensor([pos], dtype=torch.int32), page_table=None
        )

    def capture(self, token, pos):
        """Persistent inputs + one eager compile step + trace capture (on whatever state the model holds; the caller
        re-injects the real state afterwards)."""
        m = self.model
        m._ondev_argmax = True
        self.dev = m.prepare_inputs_decode(
            torch.tensor([[token]], dtype=torch.int32),
            torch.tensor([pos], dtype=torch.int32),
            page_table=self.page_table,
        )
        t0 = time.perf_counter()
        w = self._step_ops()
        for t in w:
            ttnn.deallocate(t)
        self.trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.out = self._step_ops()
        ttnn.end_trace_capture(self.mesh, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh)
        logger.info(f"[tp_decode_driver] compile+capture {time.perf_counter() - t0:.2f}s")

    def _write(self, host, which):
        for i in which:
            ttnn.copy_host_to_device_tensor(host[i], self.dev[i])

    def _read_tok(self):
        if not self.ag_tail:
            return self._read_tok_legacy(self.out[0], self.out[1])
        return int(ttnn.to_torch(ttnn.get_device_tensors(self.out[0])[0]).reshape(-1)[0])

    def decode(self, first_token, start_pos, n_steps):
        """Greedy: n_steps traced decode steps from first_token at start_pos. Returns (tokens incl. first_token,
        info dict: decode_s, step_s[, phase])."""
        toks = [int(first_token)]
        if not self.devloop:
            step_s, phase = [], {"update": [], "exec_sync": [], "readback": []}
            pos = start_pos
            t_dec0 = time.perf_counter()
            for _ in range(n_steps):
                ta = time.perf_counter()
                self._write(self._host_inputs(toks[-1], pos), (0, 1, 2))
                tb = time.perf_counter()
                ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(self.mesh)
                tc = time.perf_counter()
                toks.append(self._read_tok())
                td = time.perf_counter()
                step_s.append(td - ta)
                phase["update"].append(tb - ta)
                phase["exec_sync"].append(tc - tb)
                phase["readback"].append(td - tc)
                pos += 1
            return toks, {"decode_s": time.perf_counter() - t_dec0, "step_s": step_s, "phase": phase}
        # devloop: all host inputs up front, back-to-back replays, non-blocking token reads, one sync.
        t_dec0 = time.perf_counter()
        hosts = [self._host_inputs(first_token, start_pos + i) for i in range(n_steps)]
        t_host = time.perf_counter()
        reads = []
        for i in range(n_steps):
            self._write(hosts[i], (0, 1, 2) if i == 0 else (1, 2))
            ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
            reads.append(self.out[0].cpu(blocking=False))
        t_enq = time.perf_counter()
        ttnn.synchronize_device(self.mesh)
        for r in reads:
            toks.append(int(ttnn.to_torch(r, mesh_composer=self.comp).reshape(-1)[0]))
        t_end = time.perf_counter()
        return toks, {
            "decode_s": t_end - t_dec0,
            "host_prep_s": t_host - t_dec0,
            "enqueue_s": t_enq - t_host,
            "sync_read_s": t_end - t_enq,
        }

    def decode_teacher_forced(self, tokens, start_pos):
        """Step i feeds tokens[i] at start_pos + i; returns the full-vocab logits of every step (host float [vocab])
        and the greedy token of every step."""
        out_logits, out_toks = [], []
        for i, t in enumerate(tokens):
            self._write(self._host_inputs(int(t), start_pos + i), (0, 1, 2))
            ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh)
            if self.ag_tail:
                full = ttnn.to_torch(ttnn.get_device_tensors(self.out[1])[0]).float().reshape(-1)
            else:
                full = ttnn.to_torch(self.out[2], mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1)).float()
                full = full.reshape(-1, full.shape[-1])[0]
            out_logits.append(full[: self.vocab].clone())
            out_toks.append(self._read_tok())
        return out_logits, out_toks

    def release(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh, self.trace_id)
            self.trace_id = None
