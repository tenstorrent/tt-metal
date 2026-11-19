# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Packed prefill on the CPU reference (PB.1, the CPU rung of bringup/PACKED_DESIGN.md).

``forward_packed`` runs one packed chunk (tt/packed.ChunkDescriptor: up to 4 segments, each its own slot's
``State``) through ``DS41Reference``'s own components, split exactly as the device design splits them:

    token-wise, ONCE over all chunk rows   embedding, mHC (hc / collapse / residuals), norms, q_a, the window-KV
                                           projection with per-row positions (RoPE from ChunkDescriptor.row_positions),
                                           engram_kv / engram_add, router, routed + shared experts, moe_add
    per segment, on its own rows + state   engram_rows (the n-gram hash with the slot's 3-id tail), compressor (groups
                                           of the segment, the slot's carry), indexer (the slot's keys, top-k), attention
                                           (the slot's window ring, compressed history and top-k)

Pad rows (a segment shorter than its lane, an empty lane) go through the token-wise work like today's pad rows past
actual_end and through none of the per-segment steps: they read and write no slot state, their engram rows are zero
and their attention output is zero. The block graph is the reference's (block_steps): every step runs in graph
order; a per-segment step's tensor output is scattered back to its lane's rows.

Bit-identity with ``DS41Reference.forward_chunk`` of each segment alone holds when every token-wise component is
row-local (kernels.py "GEMM tiling": fixed row tiles; the MoE sums each token's experts in expert-id order): what the
CPU oracle (tests/packed/test_packed_cpu.py) checks.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from models.demos.common.bringup.reference.interface import Ctx
from models.demos.deepseek_v41_flash_d_p.reference import kernels as K
from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import DS41Reference, rms_norm, rope


def _tiles(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [n, K] @ w[N, K].T in fixed EXPERT_ROW_BLOCK-row GEMM calls (a row's result independent of n)."""
    n, b = x.shape[0], K.EXPERT_ROW_BLOCK
    xp = K._pad_rows(x, n + (-n % b))
    return torch.cat([F.linear(xp[r : r + b], w) for r in range(0, xp.shape[0], b)])[:n]


def swiglu_rowwise(self, x, w1, w3, w2, weights, lin=None):
    """DS41Reference._swiglu with every elementwise step on one row at a time and every GEMM in fixed 32-row tiles.
    The grouped original computes clamp / silu / multiply on the whole [group rows, I] tensor; torch's multithreaded
    elementwise kernels then give a row a result that depends on its place in the tensor (1 ulp: h 4.7e-10, expert 60
    of layer 3, 134 rows vs 1; DS41-321), i.e. on which other tokens share the expert."""
    rows = lambda t: [t[i : i + 1] for i in range(t.shape[0])]  # noqa: E731
    lim = self.cfg.swiglu_limit
    xq = torch.cat([self._q8(r) for r in rows(x)]) if x.shape[0] else self._q8(x)
    gate, up = _tiles(xq, w1), _tiles(xq, w3)
    hs = []
    for i in range(x.shape[0]):
        g, u = self._r(gate[i : i + 1]), self._r(up[i : i + 1])
        if lim > 0:
            u = u.clamp(min=-lim, max=lim)
            g = g.clamp(max=lim)
        h = F.silu(g) * u
        if weights is not None:
            h = weights[i : i + 1] * h
        hs.append(self._q8(self._r(h)))
    h = torch.cat(hs) if hs else gate[:0]
    y = _tiles(h, w2)
    return torch.cat([self._r(r) for r in rows(y)]) if y.shape[0] else y


@contextmanager
def fixed_expert_tiles():
    """The CPU oracle's routed experts row by row (every side of a comparison: packed, alone, today): the CPU
    reference's grouped SwiGLU is not composition-independent (DS41-321); the device MoE's row-locality is the device
    oracle's question (PACKED_DESIGN.md, PB.4), not the CPU rung's. The shared expert (all chunk rows, lin=mm) keeps
    the reference's own path."""
    old = DS41Reference._swiglu

    def patched(self, x, w1, w3, w2, weights, lin=K.expert_linear):
        if lin is K.expert_linear:  # (the routed experts' call; the shared expert passes lin=mm)
            return swiglu_rowwise(self, x, w1, w3, w2, weights)
        return old(self, x, w1, w3, w2, weights, lin=lin)

    DS41Reference._swiglu = patched
    try:
        yield
    finally:
        DS41Reference._swiglu = old


# the steps that read or write a slot's state or the chunk's start (everything else is row-local)
PER_SEGMENT = ("engram_rows", "compressor", "indexer", "attention")


def window_kv_rows(ref, layer: int, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """DS41Reference._window_kv with a per-row position vector (the token-wise kv step of a packed chunk)."""
    cfg, w = ref.cfg, ref.w[layer]
    r = ref._r
    kv = r(rms_norm(r(ref._lin8(layer, "attn.wkv.weight", r(x))), w["attn.kv_norm.weight"], cfg.eps))
    kv = r(rope(kv, ref._rot(layer, pos)))
    if ref._dev("window_kv"):
        return kv
    return r(ref._q8(kv))


def forward_packed(ref, desc, ids_per_lane: list, states: dict, rec=None, pad_token: int | None = None):
    """One packed chunk. ``desc``: tt/packed.ChunkDescriptor (prefill end = actual_end: the reference has no cap);
    ``ids_per_lane[i]``: lane i's segment ids [length] (None: empty lane); ``states[slot]``: each live segment's
    slot State (advanced in place, as forward_chunk does). ``rec(name, rows)``: every block boundary over all chunk
    rows ("L{i}.<boundary>", "final_norm"). Returns final_norm [chunk, H]."""
    rec = rec or (lambda name, t: None)
    cfg = ref.cfg
    pad = cfg.vocab_size - 1 if pad_token is None else int(pad_token)  # (the device clamps PAD_ID into the vocab, K-72)
    tokens = desc.pack_ids(ids_per_lane, pad=pad)
    pos = desc.row_positions()
    live = desc.live()
    seg_tokens = {lane: tokens[desc.lane_off(lane) : desc.lane_off(lane) + s.length] for lane, s in live}
    e = ref.embed(tokens)
    rec("embed", e)
    h = ref.first_block_input(e)
    for i in ref.layer_ids:
        steps = ref.block_graph(i)
        env = {"in": h}
        rec(f"L{i}.in", h)
        whole = Ctx(i, 0, desc.chunk, None)  # (token-wise steps read no ctx field)
        for st in steps:
            fn = ref.component(i, st.name)
            args = [env[n] for n in st.inputs]
            if st.name == "kv":
                out = window_kv_rows(ref, i, args[0], pos)
            elif st.name not in PER_SEGMENT:
                out = fn(whole, *args)
            else:
                per = {}
                for lane, s in live:
                    a, b = desc.lane_off(lane), desc.lane_off(lane) + s.length
                    ctx = Ctx(i, s.start, s.length, states[s.slot], extra={"tokens": seg_tokens[lane]})
                    sargs = [_seg_input(n, v, lane, a, b) for n, v in zip(st.inputs, args)]
                    per[lane] = fn(ctx, *sargs)
                out = _scatter(st.name, per, desc, live)
            env[st.output] = out
            if not isinstance(out, _PerLane):
                rec(f"L{i}.{st.output}", out)
        h = env["out"]
    out = ref.final_norm(h)
    rec("final_norm", out)
    return out


class _PerLane(dict):
    """A per-segment step's outputs that are not rows of the chunk (compressed groups, top-k): lane -> tensor."""


# boundaries whose per-segment value is a _PerLane (read by the same segment's later per-segment steps)
_LANE_OUTPUTS = ("compressor", "indexer")


def _seg_input(name: str, v, lane: int, a: int, b: int):
    if isinstance(v, _PerLane):
        return v[lane]
    return v[a:b]


def _scatter(step: str, per: dict, desc, live):
    if step in _LANE_OUTPUTS:
        return _PerLane(per)
    any_t = next(iter(per.values())) if per else None
    if any_t is None:
        raise AssertionError(f"{step}: a packed chunk with no live segment")
    out = any_t.new_zeros((desc.chunk,) + tuple(any_t.shape[1:]))
    for lane, s in live:
        out[desc.lane_off(lane) : desc.lane_off(lane) + s.length] = per[lane]
    return out
