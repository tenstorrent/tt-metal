# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-width decode traces for the full model (Stage 6; coordinator 2026-10-19T02:53:49Z, design
`$GLM53_WS/doc/ttnn/06-full-model/per_width_trace_plan.md`).

`Glm53PipelineBackend` maps every decode width to ONE full-width pipeline trace (the KDA decode processes every slot
of its batched state), so on a max_batch-32 server a lone request pays the 32-row step (~125 ms vs ~76 ms) and gets the
32-row programs' numerics. Here each width w in `widths` (< max_batch) gets its own set of per-chip decode traces that
run the stack on w rows:

* positions / page table: the width-w trace slices rows 0..w-1 of the stage's persistent cur_pos [B] and page table
  [B, blocks] (host writes keep fanning out to the persistent tensors; `plus_one` advances the persistent cur_pos, its
  inactive -1 rows untouched), so no per-width position copies and nothing to transfer at a width switch;
* KDA: the slot prefix S[0:w] / C[:, :, 0:w] is copied into w-slot scratch tensors inside the trace, the layer decodes
  them (`fused_recurrent_kda` updates the scratch S in place), and the w slots are written back (per-slot `fill_cache`
  of S, a row-masked blend of C) — slots >= w are never touched;
* MLA: the shared paged latent cache with w users (rows 0..w-1 of the page table);
* streams: one persistent [1, 1, w, 16384] input per stage and output on the last stage; hops send / receive w rows;
* terminal: the chip-0 prologue embeds token rows 0..w-1, the epilogue pads the w head rows to the sampler's 32 rows
  (rows >= w and inactive rows exactly 0), so `io.logits`, `io.tokens` and the two sampler traces are shared by every
  width.

The generator needs no change: `Glm53Generator` already picks the smallest decode width covering the live slots (vLLM
compacts live requests into the lowest slots) and replays `decode_traces[width]`. Widths not in `widths` keep the
full-width trace. At w = 1 the masked-path extras are skipped (the generator only decodes width 1 with slot 0 live),
so a lone request runs the B=1 programs.

Serving switch: `build_model_backend(..., decode_widths=(1, 2, 4))` or env `GLM53_DECODE_WIDTHS=1,2,4` (default off) builds
the standard backend and switches the instances to the subclasses below (they add methods only; state is created in
`allocate_decode_io`); `build_per_width_backend` is the same with explicit widths.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.model import (
    DRAM,
    TOK_ROWS,
    Glm53PipelineBackend,
    PipelineStack,
    _active_mask,
    _row0_ffn,
    _Trace,
    _where_rows,
    build_model_backend,
)
from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import _zeros, decode_attention, forward_unit

PER_WIDTH_DEFAULT = (1, 2, 4)


@dataclass
class _WidthBufs:
    """Persistent per-(stage, group, width) buffers (allocated before any capture)."""

    streams_in: object  # [1, 1, w, 16384] fp32 stage input (stage 0: the prologue's embedding; else the hop target)
    out_buf: object  # [1, 1, w, 16384] fp32 copy of the stage output (last stage / host hand-off), else None
    rowmask: object  # [1, 1, B, 1] bf16: 1 for rows < w (KDA conv-row write-back), None without KDA layers
    trace_id: object = None


class PerWidthStack(PipelineStack):
    """PipelineStack whose `_stage_decode` records / runs width `self.width` (None = the full-width path)."""

    width = None
    pw_widths: tuple = ()

    def allocate_widths(self, widths: Sequence[int], gi: int = 0):
        self.pw_widths = tuple(sorted({int(w) for w in widths if 1 <= int(w) < self.max_batch}))
        for st in self.stages:
            g = st.groups[gi]
            g.wbufs = {}
            has_kda = any(st.decoders[l].is_kda for l in st.state_layers)
            for w in self.pw_widths:
                rowmask = None
                if has_kda:
                    m = torch.zeros(1, 1, self.max_batch, 1)
                    m[0, 0, :w] = 1.0
                    rowmask = ttnn.from_torch(
                        m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=st.mesh, memory_config=DRAM
                    )
                last = st is self.stages[-1] or self.handoff == "host"
                g.wbufs[w] = _WidthBufs(
                    streams_in=_zeros(st.mesh, w), out_buf=_zeros(st.mesh, w) if last else None, rowmask=rowmask
                )

    def free_widths(self):
        for st in self.stages:
            for g in st.groups:
                for wb in getattr(g, "wbufs", {}).values():
                    if wb.trace_id is not None:
                        ttnn.release_trace(st.mesh, wb.trace_id)
                        wb.trace_id = None
                    for t in (wb.streams_in, wb.out_buf, wb.rowmask):
                        if t is not None and t.is_allocated():
                            ttnn.deallocate(t)
                g.wbufs = {}
        self.pw_widths = ()

    def release_widths(self):
        for st in self.stages:
            for g in st.groups:
                for wb in getattr(g, "wbufs", {}).values():
                    if wb.trace_id is not None:
                        ttnn.release_trace(st.mesh, wb.trace_id)
                        wb.trace_id = None

    def free(self):
        self.free_widths()
        super().free()

    def _stage_decode(self, k: int, g_idx: int):
        w = self.width
        if w is None or w >= self.max_batch:
            return super()._stage_decode(k, g_idx)
        st = self.stages[k]
        g = st.groups[g_idx]
        wb = g.wbufs[w]
        if k == 0 and self.decode_prologue is not None:
            self.decode_prologue(g_idx)
        if st.link_in is not None:
            st.link_in.recv(wb.streams_in)
        cur = ttnn.slice(g.cur_pos, (0,), (w,))
        pt = None
        if g.page_table is not None:
            pt = ttnn.slice(g.page_table, (0, 0), (w, g.page_table.shape[-1]))
        views = {}
        for l in st.state_layers:
            c = g.caches[l]
            if st.decoders[l].is_kda:  # copy-in of the slot prefix (contiguous dim-0 / row ranges)
                S, C = c.kda.S, c.kda.C
                s_w = ttnn.slice(S, (0, 0, 0, 0), (w, *tuple(S.shape)[1:]), memory_config=DRAM)
                c_w = ttnn.slice(C, (0, 0, 0, 0), (1, C.shape[1], w, C.shape[-1]), memory_config=DRAM)
                views[l] = replace(c, max_batch=w, kda=replace(c.kda, S=s_w, C=c_w, slots=w))
            else:
                views[l] = c
        masked = self.mask_inactive and w > 1
        m32 = mb = None
        if masked:
            m32 = _active_mask(cur, w)
            mb = ttnn.typecast(m32, ttnn.bfloat16)
            x = ttnn.where(m32, wb.streams_in, 0.0)
        else:
            x = wb.streams_in
        out = self._decode_units_w(st, views, pt, cur, x, m32, mb, keep=wb.streams_in)
        for l, v in views.items():  # write-back of the w slots
            if not st.decoders[l].is_kda:
                continue
            c = g.caches[l]
            for i in range(w):
                s_i = ttnn.slice(v.kda.S, (i, 0, 0, 0), (i + 1, *tuple(v.kda.S.shape)[1:]), memory_config=DRAM)
                ttnn.fill_cache(c.kda.S, s_i, i)
                ttnn.deallocate(s_i)
            slots = c.kda.C.shape[2]
            # rows w..slots-1 lie inside the tile of row w-1 (slots <= 32): the pad is a view of the scratch rows
            cp = ttnn.pad(v.kda.C, ((0, 0), (0, 0), (0, slots - w), (0, 0)), 0.0)
            blended = ttnn.where(wb.rowmask, cp, c.kda.C)
            ttnn.copy(blended, c.kda.C)
            ttnn.deallocate(blended)
            if cp.buffer_address() != v.kda.C.buffer_address():
                ttnn.deallocate(cp)
            ttnn.deallocate(v.kda.S)
            ttnn.deallocate(v.kda.C)
        if st.link_out is not None:
            st.link_out.send(out)
        if wb.out_buf is not None:
            ttnn.copy(out, wb.out_buf)
        if out is not wb.streams_in:
            ttnn.deallocate(out)
        for t in (m32, mb, cur, pt):
            if t is not None:
                ttnn.deallocate(t)
        if st.plus_one:
            ttnn.plus_one(g.cur_pos, skip_negative_entries=True)
        if k == len(self.stages) - 1 and self.decode_epilogue is not None:
            self.decode_epilogue(g_idx)

    @staticmethod
    def _decode_units_w(st, views, pt, cur, x, m32, mb, *, keep):
        """The stage's units on w rows. Unmasked: PipelineStage.decode_units' calls (whole layers through the
        decoder's `decode_forward`); masked (w > 1): PipelineStack._decode_units_masked's (inactive rows zeroed at
        unit and MLA-attention outputs, row-0 copies into the MoE FFN)."""
        masked = m32 is not None
        for u in st.plan.units:
            dec = st.decoders[u.layer]
            cache = views.get(u.layer)
            moe = False
            if u.part == "full" and not masked:
                y = dec.decode_forward(x, cache=cache, page_table=pt, cur_pos=cur)
            else:
                attn = None
                if u.part != "ffn":
                    base = decode_attention(dec, cache, pt, cur)
                    attn = base if (dec.is_kda or not masked) else (lambda h, base=base: _where_rows(mb, base(h)))
                moe = masked and u.part != "attn" and dec.spec.mlp == "moe"
                if moe:
                    dec.layer._ffn = _row0_ffn(type(dec.layer)._ffn.__get__(dec.layer), m32, mb)
                try:
                    if u.part == "full":
                        y = dec.layer.forward(x, attn, decode=True)
                    else:
                        y = forward_unit(dec, x, attn, decode=True, part=u.part)
                finally:
                    if moe:
                        del dec.layer._ffn
            if x is not keep:
                ttnn.deallocate(x)
            if masked:
                x = ttnn.where(m32, y, 0.0)
                ttnn.deallocate(y)
            else:
                x = y
        return x


class PerWidthBackend(Glm53PipelineBackend):
    """Glm53PipelineBackend with one set of per-chip decode traces per width in `pw_request` (module docstring)."""

    pw_request: tuple = PER_WIDTH_DEFAULT

    @property
    def width_bucketed_decode(self) -> tuple:
        """The generator's decode widths are these + max_batch (not every pow2 bucket: 8 / 16 would only replay the
        full-width trace), so serving compacts live requests into the lowest slots and a lone request runs width 1."""
        return tuple(self.pw_request)

    def allocate_decode_io(self, kv_cache):
        io = super().allocate_decode_io(kv_cache)
        self.m.allocate_widths(self.pw_request, kv_cache.group)  # (the previous cache's width buffers: m.free)
        for t in getattr(self, "rowmask_w", {}).values():
            ttnn.deallocate(t)
        self.rowmask_w = {}
        for w in self.m.pw_widths:
            m = torch.zeros(1, 1, TOK_ROWS, 1)
            m[0, 0, :w] = 1.0
            self.rowmask_w[w] = ttnn.from_torch(
                m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.head_mesh, memory_config=DRAM
            )
        return io

    def _width(self, io):
        w = self.m.width
        return None if w is None or w >= io.tokens.rows else w

    def _prologue(self, io, gi: int):
        w = self._width(io)
        if w is None:
            return super()._prologue(io, gi)
        tok = ttnn.reshape(io.tokens.tensors[0], (1, TOK_ROWS))
        x = self.term.embed_streams(tok, w)
        ttnn.copy(x, self.m.stages[0].groups[gi].wbufs[w].streams_in)
        ttnn.deallocate(x)

    def _epilogue(self, io, gi: int):
        """The w head rows padded to the sampler's 32 rows; rows >= w and inactive rows exactly 0 (as the base epilogue;
        the pad of a w-row persistent buffer is a view of it and is never deallocated)."""
        w = self._width(io)
        if w is None:
            return super()._epilogue(io, gi)
        g = self.m.stages[-1].groups[gi]
        xp = ttnn.pad(g.wbufs[w].out_buf, ((0, 0), (0, 0), (0, TOK_ROWS - w), (0, 0)), 0.0)
        if w > 1:
            cur = ttnn.slice(g.cur_pos, (0,), (w,))
            am = _active_mask(cur, w)
            amp = ttnn.pad(am, ((0, 0), (0, 0), (0, TOK_ROWS - w), (0, 0)), 0.0)
            m = ttnn.multiply(amp, self.rowmask_w[w])
            ttnn.deallocate(am)
            ttnn.deallocate(cur)
        else:
            m = self.rowmask_w[w]
        xm = ttnn.where(m, xp, 0.0)
        if m is not self.rowmask_w[w]:
            ttnn.deallocate(m)
        ttnn.copy(xm, self.dec_streams32)
        ttnn.deallocate(xm)
        y = self.term.logits_decode(self.dec_streams32)
        ttnn.copy(y, io.logits.tensors[0])
        ttnn.deallocate(y)

    def prepare_decode(self, io, kv_cache, widths: Sequence[int]) -> None:
        """Warm every per-width program set (one eager step each, state and positions restored), then the full width
        and the samplers (base)."""
        for w in self.m.pw_widths:
            self.m.width = w
            try:
                super().prepare_decode(io, kv_cache, widths)
            finally:
                self.m.width = None
        super().prepare_decode(io, kv_cache, widths)

    def decode(self, io, kv_cache, width: int) -> None:
        self.m.width = width if width in self.m.pw_widths else None
        try:
            self.m.decode_eager(kv_cache.group)
        finally:
            self.m.width = None

    def capture_decode(self, io, kv_cache, width: int):
        if width not in self.m.pw_widths:
            return super().capture_decode(io, kv_cache, width)
        if not self.captured_stack:  # the full-width set + the prefill traces first (base bookkeeping)
            super().capture_decode(io, kv_cache, kv_cache.max_batch)
        m = self.m
        gi = kv_cache.group
        ids = []
        m.width = width
        try:
            for k, st in enumerate(m.stages):
                wb = st.groups[gi].wbufs[width]
                if wb.trace_id is None:
                    wb.trace_id = ttnn.begin_trace_capture(st.mesh, cq_id=0)
                    try:
                        m._stage_decode(k, gi)
                    finally:
                        ttnn.end_trace_capture(st.mesh, wb.trace_id, cq_id=0)
                ids.append((st.mesh, wb.trace_id))
        finally:
            m.width = None
        tr = _Trace("decode", width, ids)
        self.traces.append(tr)
        return tr

    def release(self, trace: _Trace) -> None:
        super().release(trace)
        if not any(t.kind == "decode" for t in self.traces):
            self.m.release_widths()


def build_per_width_backend(model_dir, mesh_device, *, widths: Sequence[int] = PER_WIDTH_DEFAULT, **kw):
    """`build_model_backend` with per-width decode traces for `widths` (each < the generator's max_batch)."""
    return build_model_backend(model_dir, mesh_device, decode_widths=tuple(widths), **kw)
