# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bonsai decode on the weight stream (BONSAI_STREAM=1; stream.py).

setup(model) builds the stream once: every decode linear, in the order a decode step runs them, spread over the
receivers (minus the cores of its fused rotation), its pages added to one Plan; the LM head takes the final norm in
its op (NormRotation). activate() / deactivate() switch the stream's sub-devices and core layout on for decode and off
for prefill. While active, every decode step starts the senders (launch()), the linears read the ring, and the other
ops run on the receivers."""

import os

import torch

import ttnn
from models.demos.blackhole.bonsai2.tt import core_ranges
from models.demos.blackhole.bonsai2.tt.stream.stream import Plan, Stream, receiver_count

STATE = None


def enabled():
    return os.environ.get("BONSAI_STREAM") == "1"


def active():
    return STATE is not None and STATE.on


def stream():
    return STATE.st if active() else None


class StreamDecode:
    def __init__(self, model):
        from models.demos.blackhole.bonsai2.tt.ternary.rotate import NormRotation

        device = model.mesh_device
        self.model, self.on = model, False
        n = receiver_count(device)
        # linears (and their rotations) on the first nl receivers (STREAM_LINEAR_CORES; default all)
        nl = int(os.environ.get("STREAM_LINEAR_CORES") or n)
        w = ttnn.to_torch(model.norm.weight).float().reshape(-1)
        self.final_nrot = NormRotation(device, model._ternary_signs[5120], w, model.norm.eps)
        order = []
        for layer in model.layers:
            t = layer.attention.ternary
            if layer.is_full_attention:
                order += [(t.q, layer._bonsai_attn_nrot.B), (t.kv, 0), (t.o, t.rot_out.B)]
            else:
                order += [(t.qkvz, layer._bonsai_attn_nrot.B), (t.out, t.rot_out.B)]
            m = layer.feed_forward.ternary
            order += [(m.gateup, layer._bonsai_ff_nrot.B), (m.down, m.rot_hidden.B)]
        self.head = model.ternary_lm_head.lin
        order.append((self.head, self.final_nrot.B))
        span_ns = set()
        for lin, B in order:
            lin.cores, lin.spread, lin.streamed = nl - B, True, True
            span_ns.add(lin._spans()[0])
        # (before the stream's sub-devices load: a copy op on the whole grid) the decode q / k norm + RoPE consts in
        # L1 while the stream runs (a DRAM read waits behind the senders'); DRAM again for prefill (its CBs need that L1)
        self.qkr = []
        for layer in model.layers:
            for op in (getattr(layer.attention.ternary, "pre_qkr", None) or {}).values():
                self.qkr.append((op, op.consts, ttnn.to_memory_config(op.consts, ttnn.L1_MEMORY_CONFIG)))
        # compact sender groups; each linear's tiles split in a sender-balanced order (Stream span_ns)
        self.st = st = Stream(device, span_ns=sorted(span_ns, reverse=True))
        core_ranges.set_layout(st.rx_order, st.defines())
        pages = [lin.stream_pages(device, st.span_order(lin._spans()[0])) for lin, _ in order]
        self.plan = Plan(st)
        self.takes = [self.plan.add(p) for p in pages]
        self.plan.build()
        if os.environ.get("STREAM_BENCH") == "1":
            self.bench()
        # GDN step input prefetch (gdn_step/kernels/step_prefetch.hpp): one persistent L1 buffer on the step cores,
        # filled for the next GDN layer by an earlier op (the previous down projection, or prefetch_op)
        from models.demos.blackhole.bonsai2.tt.gdn_step.kernels_layout import PF_BYTES

        self.gdns = [
            (layer.attention.ternary, layer.attention) for layer in model.layers if not layer.is_full_attention
        ]
        self.gdn_index = {id(t): i for i, (t, _) in enumerate(self.gdns)}
        nv = self.gdns[0][0].step.nv if self.gdns else 0
        if nv:
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    core_ranges.rects(st.rx_order[:nv]), [1, PF_BYTES // 4], ttnn.ShardOrientation.ROW_MAJOR
                ),
            )
            self.pf = ttnn.from_torch(
                torch.zeros(nv, PF_BYTES // 4, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=mc,
            )
        self.pending, self.ready = None, None
        from models.demos.blackhole.bonsai2.tt.ternary.argmax import prealloc

        prealloc(device)  # the decode argmax's result buffer, before any decode trace
        # the decode embedding's output: persistent, L1, zero (embed() writes row 0 only)
        D = model.embd.weights.shape[-1]
        self.embed_out = ttnn.from_torch(
            torch.zeros(1, 1, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        print(f"BONSAI_STREAM slots per sender {self.plan.slots} segments {len(self.plan.w)}", flush=True)
        self.deactivate()

    def bench(self):
        """STREAM_BENCH=1: the step's stream with drain-only consumers (one per linear), traced: the stream's own pace
        on this plan, without the decode ops."""
        import time

        from models.demos.blackhole.bonsai2.tt.stream.stream import drain

        device = self.st.device

        def step():
            self.st.launch(self.plan)
            for t in self.takes:
                drain(self.st, t)

        step()
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        step()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(3):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        ms = (time.perf_counter() - t0) / 3 * 1e3
        ttnn.release_trace(device, tid)
        nbytes = sum(self.plan.slots) * self.st.slot
        print(f"STREAM_BENCH drain-only step {ms:.2f} ms, {nbytes / ms / 1e6:.0f} GB/s", flush=True)

    def activate(self):
        if os.environ.get("STREAM_OPLOG") == "1" and not getattr(self, "_oplog", False):
            # debug: name every ttnn op as it is called (needs TTNN_CONFIG_OVERRIDES fast runtime mode off)
            import ttnn.decorators as dec

            dec.PRE_OPERATION_HOOKS.append(
                lambda op, args, kwargs: print("OPLOG", getattr(op, "python_fully_qualified_name", op), flush=True)
            )
            self._oplog = True
        self.st.activate()
        core_ranges.set_layout(self.st.rx_order, self.st.defines())
        for op, _, l1 in self.qkr:
            op.consts = l1
        self.on = True

    def deactivate(self):
        self.on = False
        for op, dram, _ in getattr(self, "qkr", []):
            op.consts = dram
        core_ranges.set_layout(None)
        self.st.deactivate()

    def receiver_crs(self):
        """The decode cores as rectangles (for stock ops' sub_core_grids)."""
        return core_ranges.rects(self.st.rx_order)

    def _spec(self, i):
        t, g = self.gdns[i]
        return t.step.prefetch_spec(self.pf, g.fused_conv_state, g.recurrent_state, t.wab)

    def step_pf(self, t):
        """The prefetch buffer for GDN step t, filled now (prefetch_op) unless an earlier op did."""
        i = self.gdn_index[id(t)]
        if self.ready != i:
            t.step.prefetch_op(self._spec(i))
            self.ready = i
        return self.pf

    def after_step(self, t):
        """GDN step t has read the buffer: the next GDN layer's prefetch is pending for the next host op."""
        i = self.gdn_index[id(t)] + 1
        self.ready, self.pending = None, ((i, self._spec(i)) if i < len(self.gdns) else None)

    def take_pending(self):
        if self.pending is None:
            return None
        i, spec = self.pending
        self.pending, self.ready = None, i
        return spec

    def prefetch_first(self):
        """Fill the buffer for the first GDN layer now (e.g. after its state was restored from the host)."""
        if self.gdns:
            self.gdns[0][0].step.prefetch_op(self._spec(0))

    def head_pf(self):
        """The LM head's writer prefetches the first GDN layer's inputs for the next step (its state is final by
        then): the step then starts without a prefetch op (BONSAI_HEAD_PF=0: off)."""
        if not self.gdns or os.environ.get("BONSAI_HEAD_PF", "1") != "1":
            return None
        return self._spec(0)

    def launch(self):
        self.st.launch(self.plan)
        # the previous step's LM head prefetched the first GDN layer (prefetch_first() after a state restore)
        self.pending, self.ready = None, (0 if self.head_pf() is not None else None)


def embed(table, ids):
    """The decode embedding (ttnn.embedding + its tilize) on the first receiver: row ids[0] of table into row 0 of a
    [1, 1, D] bf16 TILE tensor."""
    from models.demos.blackhole.bonsai2.tt.stream.stream import KERNELS, _crs

    D = table.shape[-1]
    device = table.device()
    out = STATE.embed_out
    core = STATE.st.rx_order[0]
    crs = _crs([core])
    rt = ttnn.RuntimeArgs()
    page = list(ids.shape)[-1] * 4  # only page 0 is read: any page size gives its address
    rt[core.x][core.y] = [ids.buffer_address(), page, table.buffer_address(), out.buffer_address()]
    ct = [D]
    for t in (ids, table, out):
        ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    size = 64 + D * 2
    prog = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=KERNELS + "embed_row.cpp",
                core_ranges=crs,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ],
        semaphores=[],
        cbs=[
            ttnn.CBDescriptor(
                total_size=size,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=size)],
            )
        ],
    )
    return ttnn.generic_op([ids, table, out], prog)


def setup(model):
    global STATE
    if STATE is None and enabled():
        STATE = StreamDecode(model)
    return STATE
