# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bonsai decode on the weight stream (BONSAI_STREAM=1; stream.py).

setup(model) builds the stream once: every decode linear, in the order a decode step runs them, spread over the
receivers (minus the cores of its fused rotation), its pages added to one Plan; the LM head takes the final norm in
its op (NormRotation). activate() / deactivate() switch the stream's sub-devices and core layout on for decode and off
for prefill. While active, every decode step starts the senders (launch()), the linears read the ring, and the other
ops run on the receivers."""

import os

import ttnn
from models.demos.blackhole.bonsai2.tt import core_ranges
from models.demos.blackhole.bonsai2.tt.stream.stream import Plan, Stream

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
        self.st = st = Stream(device)
        n = len(st.rx_order)
        core_ranges.set_layout(st.rx_order, st.defines())
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
        self.plan = Plan(st)
        for lin, B in order:
            lin.cores, lin.spread, lin.streamed = n - B, True, True
            self.plan.add(lin.stream_pages(device))
        self.plan.build()
        from models.demos.blackhole.bonsai2.tt.ternary.argmax import prealloc

        prealloc(device)  # the decode argmax's result buffer, before any decode trace
        print(f"BONSAI_STREAM slots per sender {self.plan.slots} segments {len(self.plan.w)}", flush=True)
        self.deactivate()

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
        self.on = True

    def deactivate(self):
        self.on = False
        core_ranges.set_layout(None)
        self.st.deactivate()

    def receiver_crs(self):
        """The decode cores as rectangles (for stock ops' sub_core_grids)."""
        return core_ranges.rects(self.st.rx_order)

    def launch(self):
        self.st.launch(self.plan)


def embed(table, ids):
    """The decode embedding (ttnn.embedding + its tilize) on the first receiver: row ids[0] of table into row 0 of a
    [1, 1, D] bf16 TILE tensor."""
    from models.demos.blackhole.bonsai2.tt.stream.stream import KERNELS, _crs

    D = table.shape[-1]
    device = table.device()
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    core = STATE.st.rx_order[0]
    crs = _crs([core])
    rt = ttnn.RuntimeArgs()
    page = list(ids.shape)[-1] * 4  # only page 0 is read: any page size gives its address
    rt[core.x][core.y] = [ids.buffer_address(), page, table.buffer_address(), out.buffer_address()]
    ct = [D]
    for t in (ids, table, out):
        ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    size = 64 + D * 2 + 2048
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
