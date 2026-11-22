# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Chunked prefill for Inkling-2-Mini on one p100a: many tokens per pass through all 48 layer applications.

Reuses the weights of a built `Inkling2Model` (tt/model.py) and fills its decode state, so `Generator.decode`
continues at pos = len(prompt):
- attention (per HF cache index c): 1024-slot ring (slot = pos % 1024), global full cache, k/v conv inputs at lag 0-2;
- post-attn / post-mlp ShortConv state rows 0-3 = inputs at lag 0-3 (fp32).

A chunk is T tokens (default 2048; T divides 1024 or is a multiple of it; the last chunk is rounded up to 128). Pad
rows (positions >= len(prompt)) keep the old ring slots, but kv_full gets all T rows: decode writes position P before
any read of it, so those rows are never read. Per application, everything is a [1,1,T,C] sequence tensor:
- block / message attn-res: per-token softmax over depth (no cross-token mixing);
- k=4 short convs: exact unaligned row slices of [3 history rows ++ chunk];
- attention, in-kernel rel-bias path (r3=True; the SDPA needs `rel_logits`): one causal SDPA per layer, banded rel bias
  rel_states @ rel_proj added in the kernel, per-query sink logit; local: ring (position order) ++ chunk, window 1024;
  global: kv_full rows [0, s+T);
- attention, blocked path (older builds): banded rel bias in 128-query blocks (block b, row u, key col c <-> distance
  u + 1024 - c, so the whole additive mask is one batched matmul [rel_states, sink, 1] @ Bu[u]); local: SDPA over the
  blocks, keys = 1152 positions ++ 32 zero rows (sink: zero key and value); global: near part (positions
  [s-1024, s+T)) per block by matmul with its log-sum-exp, far part (positions < s-1024, no bias) by SDPA at head
  dim 160 with anchor keys whose logit is the near LSE, so the far output carries the exact near weight in column 128;
- MoE: router + top-8 on device, token -> expert sort on host, 64-row slots each bound to one expert, indexed
  sparse_matmul (<= 128 slots per call; one call where sparse_matmul takes repeated ids), gathered back and weighted. Needs TILE expert weights:
  the stick-layout layers of a stick cache are refused, so prefill sessions load a TILE cache ($INKLING_TILE_CACHE).

Trace rule: every constant device tensor (and every weight copy) is made in __init__, sized for `chunk`; a prefill
call allocates only per-call tensors. So make the Inkling2Prefill before `Generator.capture()` (a buffer first
allocated after a capture and kept can sit in memory the trace replays write to); prefill calls after the capture
are safe, for any prompt length.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import ttnn
from models.demos.inkling2_mini.convert import cache as wcache
from models.demos.inkling2_mini.tt.attention import (
    D_REL,
    HEAD_DIM,
    N_HEADS,
    N_KV,
    QK_CLIP,
    REL_CLIP,
    TAPS,
    WINDOW,
    log_scale,
)
from models.demos.inkling2_mini.tt.attention_fast import KV_WIDTH, V_WIDTH
from models.demos.inkling2_mini.tt.fused.common import dense_matrix
from models.demos.inkling2_mini.tt.fused.pf_attnres import PfAttnRes
from models.demos.inkling2_mini.tt.fused.pf_conv import PfConv
from models.demos.inkling2_mini.tt.fused.pf_gate import PfGateTail
from models.demos.inkling2_mini.tt.model import APPS, HIDDEN, LOOP_END, LOOP_START, SOFTCAP, VOCAB_REAL
from models.demos.inkling2_mini.tt.moe import INTER, NUM_EXPERTS, RMS_EPS, ROUTE_SCALE, TOP_K, softcapped_swiglu
from models.demos.inkling2_mini.tt.residual import cache_index

EPS = 1e-6
NEG = -1.0e9
GROUPS = N_HEADS // N_KV
SCALE = HEAD_DIM**-0.5
FAR_DH = HEAD_DIM + 32  # far part head dim: q ++ [lse_hi, lse_lo, 0...]
QB = 128  # local attention query block (SDPA batch element)


def _f32(x):
    return x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)


def _bf16(x):
    return x if x.dtype == ttnn.bfloat16 else ttnn.typecast(x, ttnn.bfloat16)


def _rows(x, a, b):
    """Rows [a, b) of a [1, H, R, W] tensor (any alignment)."""
    s = x.shape
    return ttnn.slice(x, [0, 0, a, 0], [s[0], s[1], b, s[3]])


def _cols(x, a, b):
    s = x.shape
    return ttnn.slice(x, [0, 0, 0, a], [s[0], s[1], s[2], b])


def _heads(x, n, d):
    """[1,1,T,n*d] -> [1,n,T,d]."""
    t = x.shape[2]
    return ttnn.permute(ttnn.reshape(x, [1, t, n, d]), (0, 2, 1, 3))


def _unheads(x):
    """[1,n,T,d] -> [1,1,T,n*d]."""
    return ttnn.experimental.nlp_concat_heads(x)


class _Tail:
    """Last 32 input rows of a causal conv stream (rows 29..31 = lags 3..1 before the chunk), bf16."""

    def __init__(self, device, width):
        self.t = ttnn.from_torch(
            torch.zeros(1, 1, 32, width), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def window(self, x):
        """x [1,1,T,W] bf16 -> (ext [1,1,32+T,W], lags[r] = input r steps back, r = 0..3) and advances the tail."""
        x = _bf16(x)
        t = x.shape[2]
        ext = ttnn.concat([self.t, x], dim=2)
        lags = [x] + [_rows(ext, 32 - r, 32 - r + t) for r in range(1, TAPS)]
        self.t = _rows(ext, t, t + 32)
        return ext, lags


def _load(root, layer, names):
    """Host tensors `names` of one layer (None: global) from the weight cache, one file at a time."""
    out = {}
    for e in wcache.manifest(root)["tensors"]:
        if e["layer"] == layer and e["name"] in names:
            out[e["name"]] = ttnn.load_tensor(Path(root) / e["file"])
    return out


def _band_matrix(p, is_global):
    """Blocked-path mask matrix [1, QB, 18, KB] from rel_proj p [16, 1024]: rows 0-15 rel_proj skewed per row offset
    u; row 16: sink column; row 17: 0 in the window / sink, -inf elsewhere (the left operand carries
    [rel_states, sink, 1]): one matmul gives the whole mask."""
    kb = WINDOW + QB + 32
    bu = torch.zeros(QB, D_REL + 2, kb)
    cc = torch.arange(kb)
    for u in range(QB):
        c = torch.arange(u + 1, u + WINDOW + 1)
        bu[u][:D_REL, c] = p[:, u + WINDOW - c]
        lo = 0 if is_global else u + 1  # global: keys at distance >= 1024 stay visible (no bias)
        bu[u][D_REL + 1] = torch.where((cc >= lo) & (cc <= u + WINDOW), 0.0, NEG)
    bu[:, D_REL, WINDOW + QB] = 1.0
    bu[:, D_REL + 1, WINDOW + QB] = 0.0
    return bu.unsqueeze(0)


def _is_sticks(t):
    """Stick-layout expert tensor (convert/stick.py): [1, 1, P*32, 32] instead of [1, E, K, N]."""
    return tuple(t.shape)[:2] == (1, 1) and t.shape[-1] == 32


def _has_r2(dev):
    """Does this build's indexed sparse_matmul take more (sorted, repeated) ids than groups ? Older
    builds reject that in validation."""
    put = lambda t, dt, lay: ttnn.from_torch(t, dtype=dt, layout=lay, device=dev)  # noqa: E731
    a = put(torch.ones(1, 3, 32, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    w = put(torch.ones(1, 2, 32, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
    try:
        out = ttnn.sparse_matmul(
            a,
            w,
            sparsity=put(torch.ones(1, 1, 1, 2), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            indices=put(
                torch.tensor([0, 0, 1], dtype=torch.int32).view(1, 1, 1, 3), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT
            ),
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            program_config=pc,
        )
    except RuntimeError:
        return False
    return bool(ttnn.to_torch(out).float().flatten()[0] == 32.0)


def _host(t):
    return (ttnn.to_torch(t) if isinstance(t, ttnn.Tensor) else t).float()


class _AttnRes:
    """Per-token block attn-res: softmax_d(q . rms(S_d) / sqrt(C)) . rms(S_d), on [1,1,T,C] fp32 states."""

    def __init__(self, dev, q, nq, ckc):
        """q: host [.., 32, C] (row j = query j)."""
        q = _host(q).reshape(-1, HIDDEN)[:32].clone()
        q[nq:] = 0
        self.nq = nq
        qT = (q * HIDDEN**-0.5).T.contiguous().view(1, 1, HIDDEN, 32)  # 1/sqrt(C) folded
        self.qT = ttnn.from_torch(qT, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        self.ckc = ckc

    def __call__(self, normed, partial):
        """normed: list of rms-normed sealed states (fp32). Returns nq fp32 outputs."""
        ns = normed + [ttnn.rms_norm(_f32(partial), epsilon=EPS, compute_kernel_config=self.ckc)]
        lg = [ttnn.matmul(n, self.qT, dtype=ttnn.float32, compute_kernel_config=self.ckc) for n in ns]
        m = lg[0]
        for x in lg[1:]:
            m = ttnn.maximum(m, x)
        e = [ttnn.exp(ttnn.subtract(x, m)) for x in lg]
        z = e[0]
        for x in e[1:]:
            z = ttnn.add(z, x)
        rz = ttnn.reciprocal(z)
        ws = [ttnn.multiply(ex, rz) for ex in e]
        outs = []
        for q in range(self.nq):
            acc = None
            for n, w in zip(ns, ws):
                term = ttnn.multiply(n, _cols(w, q, q + 1))
                acc = term if acc is None else ttnn.add(acc, term)
            outs.append(acc)
        return outs


class _Gate:
    """MessageGate on [1,1,T,C] rows: (1 - tanh(F z)) h + tanh(G z) m, z = down(cat(rms h, rms m)) (bf16 down out)."""

    def __init__(self, dev, w, which, ckc, fused=False):
        p = f"msg_gate.{which}."
        down = _host(w[p + "down"]).reshape(2 * HIDDEN, -1)
        self.down = ttnn.from_torch(
            down.view(1, 1, *down.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
        )
        put = lambda t: ttnn.from_torch(  # noqa: E731
            _host(t).reshape(-1, HIDDEN).view(1, 1, -1, HIDDEN), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev
        )
        self.forget, self.gain = put(w[p + "forget"]), put(w[p + "gain"])
        self.ckc = ckc
        self.tail = PfGateTail(dev) if fused else None

    def __call__(self, h, m):
        h, m = _f32(h), _f32(m)
        nh = ttnn.rms_norm(h, epsilon=EPS, compute_kernel_config=self.ckc)
        nm = ttnn.rms_norm(m, epsilon=EPS, compute_kernel_config=self.ckc)
        x = ttnn.typecast(ttnn.concat([nh, nm], dim=-1), ttnn.bfloat16)
        if self.tail is not None:  # typecast z .. addcmul in one program
            zb = ttnn.matmul(x, self.down, dtype=ttnn.bfloat16, compute_kernel_config=self.ckc)
            return self.tail(zb, self.forget, self.gain, h, m)
        z = ttnn.typecast(ttnn.matmul(x, self.down, dtype=ttnn.bfloat16, compute_kernel_config=self.ckc), ttnn.float32)
        f = ttnn.tanh(ttnn.matmul(z, self.forget, dtype=ttnn.float32, compute_kernel_config=self.ckc))
        g = ttnn.tanh(ttnn.matmul(z, self.gain, dtype=ttnn.float32, compute_kernel_config=self.ckc))
        return ttnn.addcmul(ttnn.addcmul(h, f, h, value=-1.0), g, m)  # h - f*h + g*m, fp32


ATTN_NAMES = ("w_qgrs", "w_k", "w_v", "w_o", "q_clip_scale", "rel_clip_scale", "rel_proj", "k_conv", "v_conv")


class _LayerPF:
    """Prefill-side weights of one decoder layer. The decode modules keep theirs in decode-only forms (DRAM-sharded
    32-row projections, fused-kernel dense pages, bank-local experts), so the small tensors are loaded again from
    the weight cache in plain interleaved form; the experts and router come from the model's MoE when it has them
    in plain form (bank-local gate_up layers load their own copy, ~0.38 GB each)."""

    def __init__(self, pf, lay, root):
        dev = pf.device
        self.lay = lay
        L = lay.L
        names = {f"attn.{n}" for n in ATTN_NAMES} | {"attn_res_q", "mlp_res_q", "post_attn_conv", "post_mlp_conv"}
        names |= {"q_input_norm", "k_input_norm", "v_input_norm"} if lay.is_global else {"attn_norm"}
        if lay.is_loop:
            names |= {f"msg_gate.{g}.{k}" for g in ("q", "k", "v", "mlp") for k in ("down", "forget", "gain")}
        moe = lay.moe
        if not hasattr(moe, "w_gate_up"):
            names.add("experts.w_gate_up")
        w = _load(root, L, names)

        def put(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=dev)

        def dev_of(name):
            return ttnn.to_device(w[name], dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        self.w_qgrs, self.w_k, self.w_v, self.w_o = (dev_of(f"attn.{n}") for n in ("w_qgrs", "w_k", "w_v", "w_o"))
        self.q_scale = put(_host(w["attn.q_clip_scale"]).view(1, N_HEADS, 1, HEAD_DIM))
        self.rel_scale = put(_host(w["attn.rel_clip_scale"]).view(1, N_HEADS, 1, D_REL))
        rel_p = _host(w["attn.rel_proj"]).view(D_REL, WINDOW)
        self.rel_proj = put(rel_p.view(1, 1, D_REL, WINDOW), ttnn.bfloat16)
        self.bu = None if pf.r3 else put(_band_matrix(rel_p, lay.is_global), ttnn.bfloat16)
        # k/v conv taps per lag (fp32). Local: residual and 1/sqrt(K+1) folded in; global: static taps, the
        # residual and 1/sqrt(2K+1) are applied after adding the dynamic taps.
        self.conv_scale = 1.0 / math.sqrt((2 if lay.is_global else 1) * TAPS + 1)
        self.taps = {}
        for kv in ("k", "v"):
            t = _host(w[f"attn.{kv}_conv"]).view(TAPS, N_KV * HEAD_DIM).clone()
            if not lay.is_global:
                t[0] += 1.0
                t *= self.conv_scale
            self.taps[kv] = [put(t[r].view(1, 1, 1, -1)) for r in range(TAPS)]
        if lay.is_global:
            # dyn[h*TAPS + lag] -> lag block of [TAPS * 1024], broadcast over the 128 channels of head h
            expand = torch.zeros(N_KV * TAPS, TAPS * N_KV * HEAD_DIM)
            for h in range(N_KV):
                for lag in range(TAPS):
                    c0 = lag * N_KV * HEAD_DIM + h * HEAD_DIM
                    expand[h * TAPS + lag, c0 : c0 + HEAD_DIM] = 1.0
            self.expand = put(expand, ttnn.bfloat16)
        self.norms = [
            dev_of(n) for n in (("q_input_norm", "k_input_norm", "v_input_norm") if lay.is_global else ("attn_norm",))
        ]
        AR = PfAttnRes if pf.fused_glue else _AttnRes
        self.attn_res = AR(dev, w["attn_res_q"], 3 if lay.is_global else 1, pf.ckc) if L > 0 else None
        self.mlp_res = AR(dev, w["mlp_res_q"], 1, pf.ckc)
        # post-attn / post-mlp conv taps (residual and 1/sqrt(5) folded), one broadcast row per lag
        self.conv_taps = {}
        for k, name in (("attn", "post_attn_conv"), ("mlp", "post_mlp_conv")):
            t = _host(w[name]).reshape(TAPS, -1).clone()
            t[0] += 1
            t /= math.sqrt(5)
            self.conv_taps[k] = [put(t[r].view(1, 1, 1, -1)) for r in range(TAPS)]
        if lay.is_loop:
            self.gates = {g: _Gate(dev, w, g, pf.ckc, pf.fused_glue) for g in ("q", "k", "v", "mlp")}
        rw = ttnn.to_torch(moe.router_w).float().reshape(HIDDEN, NUM_EXPERTS)
        self.router_w = put(rw.view(1, 1, HIDDEN, NUM_EXPERTS))
        self.router_bias_row = ttnn.to_torch(moe.router_bias).float().reshape(1, 1, 1, NUM_EXPERTS)
        self.w_gate_up = moe.w_gate_up if hasattr(moe, "w_gate_up") else dev_of("experts.w_gate_up")
        self.w_down = moe.w_down
        for t in (self.w_gate_up, self.w_down):
            if _is_sticks(t) or len(t.shape) != 4 or t.shape[1] != NUM_EXPERTS or t.layout != ttnn.TILE_LAYOUT:
                raise ValueError(
                    f"layer {L}: prefill needs TILE expert weights [1, {NUM_EXPERTS}, K, N], got {tuple(t.shape)} "
                    f"{t.layout} (stick layout from a stick cache). Load the model from a TILE cache "
                    "(e.g. INKLING_CACHE=/work/inkling-bringup/cache/v1): sparse_matmul cannot read sticks."
                )
        del w


class Inkling2Prefill:
    def __init__(
        self,
        model,
        root=wcache.DEFAULT_ROOT,
        chunk=2048,
        r3=None,
        r2=None,
        slot_rows=64,
        group_grid=(9, 4),
        group_ibw=None,
        r3_chunks=(128, 128),
        weighted_reduce=False,
        fused_glue=True,
        profile=False,
        debug=False,
    ):
        """model: a built Inkling2Model, before any Generator.capture(). chunk: max tokens per pass (a multiple of 128
        that divides 1024 or is a multiple of it); prefill(ids, chunk=...) may use any smaller valid size.
        r3: banded-bias SDPA (needs a build whose SDPA takes `rel_logits`; None: use it when it does).
        r2: one indexed sparse_matmul call with repeated sorted expert ids, weights read once per run of equal
        ids (needs that build; None: probe for it).
        slot_rows: rows per expert slot. group_grid / group_ibw: grouped expert matmul grid and in0 block widths
        (gate_up, down); None: whole K with r2 (the weight reuse needs it), else (12, 18).
        weighted_reduce: MoE combine by attn_res_weighted_reduce_nc. fused_glue: one-program attn-res, short convs
        (+ residual) and message-gate tail (tt/fused/pf_*.py, bit-equal to the op chains).
        profile: synchronized per-section times in self.times. debug: print each application."""
        assert self._chunk_ok(chunk), chunk
        self.m = model
        self.device = dev = model.device
        self.chunk = chunk
        self.gm = slot_rows
        if r2 is None:
            r2 = _has_r2(dev)
        self.r2 = r2
        if r3 is None:
            r3 = "rel_logits" in (ttnn.transformer.scaled_dot_product_attention.__doc__ or "")
        self.r3 = r3
        self.fused_glue = fused_glue
        self._pfconv = PfConv(dev) if fused_glue else None
        self.ckc = model.rows.ckc  # HiFi4, fp32 acc
        self.hifi4 = model.hifi4
        self.mm_cfg = model.mm_cfg
        self.layers = [_LayerPF(self, lay, root) for lay in model.layers]
        g = _load(root, None, {"final_res_q", "msg_q"})
        self.final_res = (PfAttnRes if self.fused_glue else _AttnRes)(dev, g["final_res_q"], 1, self.ckc)
        # cross-pass message queries [C, 64] (rows 4l+j = loop layer l, message j), 1/sqrt(C) folded
        mq = _host(g["msg_q"]).reshape(LOOP_END - LOOP_START, 32, HIDDEN)[:, :4].reshape(-1, HIDDEN)
        mq = torch.nn.functional.pad(mq * HIDDEN**-0.5, (0, 0, 0, 64 - mq.shape[0]))
        self.msg_qT = ttnn.from_torch(
            mq.T.contiguous().view(1, 1, HIDDEN, 64), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev
        )
        del g
        self.sdpa_ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.expert_ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._sdpa_pc = {}
        self.group_grid = tuple(group_grid)  # 9x4: the decode MoE tuning (N = 72 tiles over 36 cores)
        group_ibw = group_ibw or ((HIDDEN // 32, INTER // 32) if r2 else (12, 18))
        self.group_ibw = {HIDDEN: group_ibw[0], INTER: group_ibw[1]}
        self.logits_cb = None
        self.wred = weighted_reduce
        self.r2_step = 1 << 20 if r2 else NUM_EXPERTS  # sparse_matmul ids per call
        self.r3_chunks = tuple(r3_chunks)
        self.debug = debug
        self.profile = profile
        self.times = defaultdict(float)
        self._t = None
        self._consts()

    @staticmethod
    def _chunk_ok(chunk):
        return chunk % QB == 0 and (WINDOW % chunk == 0 or chunk % WINDOW == 0)

    def _consts(self):
        """Every constant device tensor, sized for self.chunk (a call slices what it needs)."""
        T, put, bf = self.chunk, self._put, ttnn.bfloat16
        cdts = (
            {st[0].k_ring.dtype for st in self.m.states.values()}
            if isinstance(self.m.states, dict)
            else {st[0].k_ring.dtype for st in self.m.states}
        )
        self.c_zring = put(torch.zeros(1, N_KV, WINDOW, HEAD_DIM), bf)
        self.c_mass = {d: put(torch.nn.functional.pad(torch.ones(1, N_KV, T, 1), (0, 31)), d) for d in cdts}
        self.c_zero_row = put(torch.zeros(1, 1, 1, HIDDEN), bf)
        self.c_zx = {m: put(torch.zeros(1, 1, 32 - m, HIDDEN), bf) for m in (TAPS, 1 + 2 * TAPS)}
        self.c_row_idx = put(torch.arange(T, dtype=torch.float32).view(1, 1, T, 1))  # MoE valid-row mask
        for lp in self.layers:  # router bias, one row per token
            lp.router_bias_rows = put(lp.router_bias_row.expand(1, 1, T, NUM_EXPERTS).contiguous())
        if self.r3:
            return
        nb, kb = T // QB, WINDOW + QB + 32
        e = torch.zeros(2, 1, 1, 1, D_REL + 2)
        e[0, ..., D_REL] = 1
        e[1, ..., D_REL + 1] = 1
        self.c_e16, self.c_e17 = put(e[0], bf), put(e[1], bf)
        # key positions < 0: absolute query block g < 1024 / QB, key col c <-> position QB*g - 1024 + c
        g = torch.arange(WINDOW // QB).view(-1, 1, 1, 1)
        vm = torch.where(QB * g - WINDOW + torch.arange(kb).view(1, 1, 1, kb) < 0, NEG, 0.0)
        self.c_vmask = put(torch.cat([vm, torch.zeros(nb, 1, 1, kb)]), bf)
        self.c_zkvb = {d: put(torch.zeros(nb, N_KV, 32, HEAD_DIM), d) for d in cdts}
        self.c_zkv = {d: put(torch.zeros(1, N_KV, 32, HEAD_DIM), d) for d in cdts}
        # global near: keys left of the band (col o0 + j is -inf iff o0 + j < 1024, o0 = min(s, 1024)) / right of it
        left = torch.where(torch.arange(WINDOW + T) < WINDOW, NEG, 0.0)
        self.c_gleft = put(left.view(1, 1, 1, -1).expand(1, N_HEADS, QB, -1).contiguous(), bf)
        self.c_gright = put(torch.full((1, N_HEADS, QB, T), NEG), bf)
        self.c_sel = (put(torch.eye(32)[0].view(1, 1, 1, 32), bf), put(torch.eye(32)[1].view(1, 1, 1, 32), bf))
        ka = torch.zeros(1, N_KV, 32, FAR_DH)
        ka[..., HEAD_DIM] = 1
        ka[..., HEAD_DIM + 1] = 1
        va = torch.zeros(1, N_KV, 32, FAR_DH)
        va[..., HEAD_DIM] = 1
        self.c_anchors = (put(ka, bf), put(va, bf))
        self.c_zcol = put(torch.zeros(1, N_KV, self.m.max_ctx, 32), bf)

    def _tick(self, name):
        """PF_PROFILE=1: synchronize and charge the time since the last tick to `name`."""
        if not self.profile:
            return
        ttnn.synchronize_device(self.device)
        now = time.perf_counter()
        if self._t is not None and name:
            self.times[name] += now - self._t
        self._t = now

    # ------------------------------------------------------------------ constants
    def _put(self, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.device)

    def _pos_cols(self, s, t):
        """tau per position [1,1,t,1] fp32 and per-row validity masks."""
        pos = torch.arange(s, s + t, dtype=torch.float64)
        tau = torch.tensor([log_scale(int(p)) for p in pos], dtype=torch.float32).view(1, 1, t, 1)
        return self._put(tau)

    # ------------------------------------------------------------------ attention
    def _linear(self, x, w, dtype=ttnn.bfloat16):
        return ttnn.linear(x, w, dtype=dtype, compute_kernel_config=self.mm_cfg)

    @staticmethod
    def _l2_clip(x32, threshold):
        norm = ttnn.sqrt(ttnn.sum(ttnn.square(x32), dim=-1, keepdim=True))
        factor = ttnn.clamp(ttnn.reciprocal(ttnn.add(norm, 1e-6)), max=1.0 / threshold)
        return ttnn.multiply(x32, ttnn.multiply(factor, threshold))

    def _scaled_clip(self, x, scale, threshold, tau):
        y = ttnn.typecast(self._l2_clip(ttnn.multiply(_f32(x), scale), threshold), ttnn.bfloat16)
        if tau is not None:
            y = ttnn.typecast(ttnn.multiply(_f32(y), tau), ttnn.bfloat16)
        return y

    def _kv(self, lp, x, w, kv, tail):
        """k or v for the chunk: projection, k=4 conv over [history ++ chunk], heads, clip / rms -> [1,8,T,128] bf16."""
        width = N_KV * HEAD_DIM
        proj = self._linear(x, w)
        t = proj.shape[2]
        is_global = lp.lay.is_global
        if is_global:
            cur = _cols(proj, 0, width)
            dyn_all = ttnn.matmul(
                _cols(proj, width, width + N_KV * TAPS), lp.expand, dtype=ttnn.float32, compute_kernel_config=self.hifi4
            )
        else:
            cur = proj
        _, lags = tail.window(cur)
        acc = None
        for lag, xl in enumerate(lags):
            wt = lp.taps[kv][lag]
            if is_global:
                wt = ttnn.add(_cols(dyn_all, lag * width, (lag + 1) * width), wt)
            term = ttnn.multiply(xl, wt, dtype=ttnn.float32)  # bf16 x fp32: exact, no typecast
            acc = term if acc is None else ttnn.add(acc, term)
        if is_global:
            acc = ttnn.multiply(ttnn.add(acc, _f32(lags[0])), lp.conv_scale)
        heads = _f32(_heads(ttnn.typecast(acc, ttnn.bfloat16), N_KV, HEAD_DIM))
        if kv == "k":
            out = self._l2_clip(heads, QK_CLIP)
        else:
            out = ttnn.multiply(heads, ttnn.rsqrt(ttnn.add(ttnn.mean(ttnn.square(heads), dim=-1, keepdim=True), EPS)))
        return ttnn.typecast(out, ttnn.bfloat16)

    def _write_ring(self, ring, k, s, n):
        """Chunk rows [0, n) -> ring slots s%1024 ...; pad rows keep the old slot contents."""
        t = k.shape[2]
        o = s % WINDOW
        if t > WINDOW:  # long chunk (s % 1024 == 0): slot q gets the last real row j < n with j % 1024 == q
            assert o == 0 and WINDOW < n <= t, (s, n, t)
            q = torch.arange(WINDOW)
            src = q + WINDOW * ((n - 1 - q) // WINDOW)  # chunk row per slot
            new = None
            for blk in range(-(-t // WINDOW)):
                lo, hi = blk * WINDOW, min(t, (blk + 1) * WINDOW)
                if not bool(((src >= lo) & (src < hi)).any()):
                    continue
                part = _bf16(_rows(k, lo, hi))
                if hi - lo < WINDOW:
                    part = ttnn.concat([part, _rows(self.c_zring, 0, WINDOW - (hi - lo))], dim=2)
                m = self._put(((src >= lo) & (src < hi)).float().view(1, 1, WINDOW, 1), ttnn.bfloat16)
                term = ttnn.multiply(part, m)
                new = term if new is None else ttnn.add(new, term)
            ttnn.fill_cache(
                ring, ttnn.typecast(new, ring.dtype) if ring.dtype != ttnn.bfloat16 else new, 0, update_idx=0
            )
            return
        kc = ttnn.typecast(k, ring.dtype) if k.dtype != ring.dtype else k
        if n < t:
            mask = self._put(torch.cat([torch.ones(n), torch.zeros(t - n)]).view(1, 1, t, 1), ttnn.bfloat16)
            old = _bf16(_rows(ring, o, o + t))
            new = ttnn.add(old, ttnn.multiply(ttnn.subtract(_bf16(k), old), mask))
            kc = ttnn.typecast(new, ring.dtype) if ring.dtype != ttnn.bfloat16 else new
        ttnn.fill_cache(ring, kc, 0, update_idx=o)

    def _sdpa_cfg(self, qc, kc):
        key = (qc, kc)
        if key not in self._sdpa_pc:
            self._sdpa_pc[key] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                q_chunk_size=qc,
                k_chunk_size=kc,
                exp_approx_mode=False,
            )
        return self._sdpa_pc[key]

    def _record_inputs(self, lp, conv, xq, xk, xv, n, t):
        """Keep what the decode k/v conv window (state.x_hist) needs: the q/k/v inputs at lags 0..3 of the last
        position. Non-last chunks keep their last 3 rows (for a last chunk shorter than 4 rows)."""
        xs = (xq, xk, xv)
        if not self._last:
            conv["xprev"] = [_rows(x, t - 3, t) for x in xs]
            return
        prev = conv.get("xprev")

        def lag(x, i, k):  # input row at position (last - k)
            r = n - 1 - k
            if r >= 0:
                return _rows(x, r, r + 1)
            if prev is None:
                return None
            return _rows(prev[i], 3 + r, 4 + r)

        conv["xwin"] = [[lag(x, i, k) for k in range(TAPS)] for i, x in enumerate(xs)]

    def _attention(self, lp, xq, xk, xv, st, conv, s, n, tau):
        """One attention layer over the chunk. xq/xk/xv [1,1,T,C] bf16 (normed). Returns [1,1,T,C] bf16."""
        lay = lp.lay
        is_global = lay.is_global
        t = xq.shape[2]
        qgrs = self._linear(xq, lp.w_qgrs)
        self._tick("a.qgrs")
        tau_g = tau if is_global else None
        q = self._scaled_clip(_heads(_cols(qgrs, 0, 3072), N_HEADS, HEAD_DIM), lp.q_scale, QK_CLIP, tau_g)
        rel = self._scaled_clip(_heads(_cols(qgrs, 6144, 6528), N_HEADS, D_REL), lp.rel_scale, REL_CLIP, tau_g)
        gate = _cols(qgrs, 3072, 6144)
        sink = ttnn.permute(_cols(qgrs, 6528, 6552), (0, 3, 2, 1))  # [1,24,T,1]

        self._tick("a.q_rel_prep")
        self._record_inputs(lp, conv, xq, xk, xv, n, t)
        k = self._kv(lp, xk, lp.w_k, "k", conv["k"])
        v = self._kv(lp, xv, lp.w_v, "v", conv["v"])
        self._tick("a.kv")

        cdt = st.k_ring.dtype
        k_c = ttnn.typecast(k, cdt) if cdt != ttnn.bfloat16 else k
        v_c = ttnn.typecast(v, cdt) if cdt != ttnn.bfloat16 else v
        qs = ttnn.typecast(ttnn.multiply(_f32(q), SCALE), ttnn.bfloat16)  # kernel adds mask to raw q.k

        if is_global:  # far cache rows [v | 1, 0 x 31 | k] (attention_fast kv_full)
            mass = _rows(self.c_mass[cdt], 0, t)
            ttnn.fill_cache(st.kv_full, ttnn.concat([v_c, mass, k_c], dim=-1), 0, update_idx=s)
        if self.r3:
            attn = self._attention_r3(lp, qs, rel, sink, st, k_c, v_c, s, t)
        elif not is_global:
            attn = self._local_blocked(lp, qs, rel, sink, st, k_c, v_c, s, t)
        else:
            attn = self._global_blocked(lp, qs, rel, sink, st, k_c, v_c, s, t)
        self._tick("a.core")
        # ring writes (after reading the old ring; the global blocked path reads kv_full only below s)
        self._write_ring(st.k_ring, k, s, n)
        self._write_ring(st.v_ring, v, s, n)

        self._tick("a.cache_write")
        attn = _unheads(_bf16(attn))  # [1,1,T,3072]
        gated = ttnn.multiply(_f32(attn), ttnn.sigmoid(_f32(gate)))
        return self._linear(ttnn.typecast(gated, ttnn.bfloat16), lp.w_o)

    def _attention_r3(self, lp, qs, rel, sink, st, k_c, v_c, s, t):
        """Attention by one SDPA with an in-kernel banded rel bias: rel_logits[q, d] = rel_states[q] . proj[:, d]
        for key distance d < 1024, per-query sink logit; local layers also mask d >= 1024. The queries are the
        last t keys: local keys = [ring rotated to position order ++ chunk] (only the s valid slots while s < 1024),
        global keys =
        kv_full rows [0, s + t) (the chunk is written there first)."""
        rl = ttnn.matmul(rel, lp.rel_proj, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi4)  # [1,24,t,1024]
        rl = ttnn.to_layout(rl, ttnn.ROW_MAJOR_LAYOUT)
        if lp.lay.is_global:
            if s == 0:
                kk, vv = k_c, v_c
            else:
                kv = ttnn.slice(st.kv_full, [0, 0, 0, 0], [1, N_KV, s + t, KV_WIDTH])
                kk, vv = _cols(kv, V_WIDTH, KV_WIDTH), _cols(kv, 0, HEAD_DIM)
            window = None
        else:
            if s == 0:
                kk, vv = k_c, v_c
            elif s < WINDOW:  # positions [0, s) sit in slots [0, s); the other slots are not positions yet
                kk = ttnn.concat([_rows(st.k_ring, 0, s), k_c], dim=2)
                vv = ttnn.concat([_rows(st.v_ring, 0, s), v_c], dim=2)
            else:
                o = s % WINDOW
                rot = lambda r: r if o == 0 else ttnn.concat([_rows(r, o, WINDOW), _rows(r, 0, o)], dim=2)  # noqa: E731
                kk = ttnn.concat([rot(st.k_ring), k_c], dim=2)
                vv = ttnn.concat([rot(st.v_ring), v_c], dim=2)
            window = WINDOW
        return ttnn.transformer.scaled_dot_product_attention(
            qs,
            kk,
            vv,
            is_causal=True,
            scale=1.0,
            sliding_window_size=window,
            rel_logits=rl,
            attention_sink=_bf16(sink),
            program_config=self._sdpa_cfg(*self.r3_chunks),
            compute_kernel_config=self.sdpa_ckc,
        )

    def _band(self, lp, rel, sink, s, t, is_global):
        """Additive mask in query-block coordinates [nb, 24, QB, KB]: banded rel bias, window / causal limits, sink
        column (KB-32), key positions < 0. Block b, row u, key col c <-> key position s + QB*b - 1024 + c."""
        nb = t // QB
        kb = WINDOW + QB + 32
        e16, e17 = self.c_e16, self.c_e17
        vmask = None
        if s < WINDOW:  # key positions < 0: absolute query blocks s/QB ..
            g0 = s // QB
            vmask = ttnn.slice(self.c_vmask, [g0, 0, 0, 0], [g0 + nb, 1, 1, kb])

        # A[u, (b, h), :] = [rel[h, QB*b + u, 0:16], sink[h, QB*b + u], 1]
        ra = ttnn.permute(ttnn.reshape(rel, [N_HEADS, nb, QB, D_REL]), (2, 1, 0, 3))
        ra = ttnn.pad(ttnn.reshape(ra, [1, QB, nb * N_HEADS, D_REL]), [(0, 0), (0, 0), (0, 0), (0, 2)], 0.0)
        sk = ttnn.permute(ttnn.reshape(_bf16(sink), [N_HEADS, nb, QB, 1]), (2, 1, 0, 3))
        sk = ttnn.reshape(sk, [1, QB, nb * N_HEADS, 1])
        ra = ttnn.add(ttnn.add(ra, ttnn.multiply(sk, e16)), e17)
        mask = ttnn.matmul(ra, lp.bu, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi4)  # [1, QB, nb*24, KB]
        # u <-> (b, h) swap via row-major (page reorder): 1.4 ms vs 5.1 ms for a TILE permute at nb = 12
        mask = ttnn.to_layout(ttnn.reshape(mask, [QB, nb * N_HEADS, kb]), ttnn.ROW_MAJOR_LAYOUT)
        mask = ttnn.permute(mask, (1, 0, 2))
        mask = ttnn.to_layout(ttnn.reshape(mask, [nb, N_HEADS, QB, kb]), ttnn.TILE_LAYOUT)  # [nb, 24, QB, KB]
        if vmask is not None:
            mask = ttnn.add(mask, vmask)
        return mask

    def _local_blocked(self, lp, qs, rel, sink, st, k_c, v_c, s, t):
        """Local attention in query blocks of QB rows, as an SDPA batch: block b sees keys at positions
        [s + QB*b - 1024, s + QB*b + QB) ++ 32 zero rows. In block coordinates (row u, key col c) the distance is
        d = u + 1024 - c for every block, so the banded bias is rel_states(row) @ Bu[u] with one constant
        [16, KB] matrix per row offset u (Bu[u][:, c] = proj[:, u + 1024 - c] inside the window): one batched
        matmul over u, then one permute into SDPA mask layout [nb, 24, QB, KB]."""
        nb = t // QB
        kb = WINDOW + QB + 32
        o = s % WINDOW
        mask = self._band(lp, rel, sink, s, t, False)
        self._tick("a.mask")

        # keys in position order: ring rotated so that slot (s % 1024) comes first, then the chunk
        ring_k = st.k_ring if o == 0 else ttnn.concat([_rows(st.k_ring, o, WINDOW), _rows(st.k_ring, 0, o)], dim=2)
        ring_v = st.v_ring if o == 0 else ttnn.concat([_rows(st.v_ring, o, WINDOW), _rows(st.v_ring, 0, o)], dim=2)
        k_all = ttnn.concat([ring_k, k_c], dim=2)
        v_all = ttnn.concat([ring_v, v_c], dim=2)
        cdt = k_c.dtype
        zp = ttnn.slice(self.c_zkvb[cdt], [0, 0, 0, 0], [nb, N_KV, 32, HEAD_DIM])
        kbt = ttnn.concat(
            [ttnn.concat([_rows(k_all, QB * b, QB * b + WINDOW + QB) for b in range(nb)], dim=0), zp], dim=2
        )
        vbt = ttnn.concat(
            [ttnn.concat([_rows(v_all, QB * b, QB * b + WINDOW + QB) for b in range(nb)], dim=0), zp], dim=2
        )
        qb = ttnn.permute(ttnn.reshape(qs, [N_HEADS, nb, QB, HEAD_DIM]), (1, 0, 2, 3))  # [nb, 24, QB, 128]
        self._tick("a.kwin")
        out = ttnn.transformer.scaled_dot_product_attention(
            qb,
            kbt,
            vbt,
            attn_mask=mask,
            is_causal=False,
            scale=1.0,
            program_config=self._sdpa_cfg(QB, 128),
            compute_kernel_config=self.sdpa_ckc,
        )
        return ttnn.reshape(ttnn.permute(out, (1, 0, 2, 3)), [1, N_HEADS, t, HEAD_DIM])

    def _global_blocked(self, lp, qs, rel, sink, st, k_c, v_c, s, t):
        """Global attention. Near part (positions [s-1024, s+t) ++ sink rows) per 128-query block by matmul, mask =
        [visible 0 | block band mask | -inf | sink]: gives each query's output and log-sum-exp. Far part (positions
        < s-1024, no bias) by SDPA with an LSE anchor (_global_far)."""
        nb = t // QB
        kb = WINDOW + QB + 32
        o = s % WINDOW
        band = self._band(lp, rel, sink, s, t, True)  # [nb, 24, QB, KB]
        cdt = k_c.dtype
        ring_k = st.k_ring if o == 0 else ttnn.concat([_rows(st.k_ring, o, WINDOW), _rows(st.k_ring, 0, o)], dim=2)
        ring_v = st.v_ring if o == 0 else ttnn.concat([_rows(st.v_ring, o, WINDOW), _rows(st.v_ring, 0, o)], dim=2)
        zpad = self.c_zkv[cdt]
        kn = ttnn.concat([ring_k, k_c, zpad], dim=2)  # key j <-> position s - 1024 + j
        vn = ttnn.concat([ring_v, v_c, zpad], dim=2)
        kk = WINDOW + t + 32
        outs, lses = [], []
        gb = 4  # query blocks per near matmul (fewer, bigger ops)
        for b0 in range(0, nb, gb):
            rowsm = []
            for b in range(b0, min(nb, b0 + gb)):
                bb = ttnn.slice(band, [b, 0, 0, 0], [b + 1, N_HEADS, QB, kb])
                left = QB * b  # keys before the block's band: visible, no bias (invalid if position < 0)
                right = t - QB * b - QB  # keys after the band: future
                parts = []
                if left:
                    o0 = min(s, WINDOW)
                    parts.append(_cols(self.c_gleft, o0, o0 + left))
                parts.append(_cols(bb, 0, WINDOW + QB))
                if right:
                    parts.append(_cols(self.c_gright, 0, right))
                parts.append(_cols(bb, WINDOW + QB, kb))
                rowsm.append(ttnn.concat(parts, dim=-1))
            g = len(rowsm)
            mb = ttnn.reshape(ttnn.concat(rowsm, dim=2) if g > 1 else rowsm[0], [1, N_KV, GROUPS * QB * g, kk])
            qb = ttnn.reshape(_rows(qs, QB * b0, QB * (b0 + g)), [1, N_KV, GROUPS * QB * g, HEAD_DIM])
            lg = ttnn.add(
                ttnn.matmul(qb, kn, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=self.hifi4), _f32(mb)
            )
            mx = ttnn.max(lg, dim=-1, keepdim=True)
            pr = ttnn.exp(ttnn.subtract(lg, mx))
            z = ttnn.sum(pr, dim=-1, keepdim=True)
            ob = ttnn.multiply(
                ttnn.matmul(ttnn.typecast(pr, ttnn.bfloat16), vn, dtype=ttnn.float32, compute_kernel_config=self.hifi4),
                ttnn.reciprocal(z),
            )
            outs.append(ttnn.reshape(ob, [1, N_HEADS, QB * g, HEAD_DIM]))
            lses.append(ttnn.reshape(ttnn.add(mx, ttnn.log(z)), [1, N_HEADS, QB * g, 1]))
        o_near = ttnn.concat(outs, dim=2) if len(outs) > 1 else outs[0]
        far = s - WINDOW
        if far <= 0:
            return o_near
        lse = ttnn.concat(lses, dim=2) if len(lses) > 1 else lses[0]
        return self._global_far(qs, o_near, lse, st, far)

    def _global_far(self, qs, o_near, lse, st, far):
        """Merge the far part (positions < far) into the near output: SDPA at head dim 160 with 32 anchor keys of
        logit lse - log 32, so output column 128 is the exact near weight. o_near [1,24,T,128], lse [1,24,T,1]."""
        t = qs.shape[2]
        qg = ttnn.reshape(qs, [1, N_KV, GROUPS * t, HEAD_DIM])
        o_near = ttnn.reshape(o_near, [1, N_KV, GROUPS * t, HEAD_DIM])
        lse = ttnn.add(ttnn.reshape(lse, [1, N_KV, GROUPS * t, 1]), -math.log(32.0))
        hi = ttnn.typecast(lse, ttnn.bfloat16)
        lo = ttnn.typecast(ttnn.subtract(lse, _f32(hi)), ttnn.bfloat16)
        sel = self.c_sel
        qext = ttnn.add(ttnn.multiply(hi, sel[0]), ttnn.multiply(lo, sel[1]))  # [1,8,3T,32]
        qf = ttnn.concat([qg, qext], dim=-1)  # [1,8,3T,160]
        qf = ttnn.reshape(qf, [1, N_HEADS, t, FAR_DH])

        ka, va = self.c_anchors
        zcol = _rows(self.c_zcol, 0, far)
        kv = ttnn.slice(st.kv_full, [0, 0, 0, 0], [1, N_KV, far, KV_WIDTH])
        kf = ttnn.concat([ttnn.concat([_bf16(_cols(kv, V_WIDTH, KV_WIDTH)), zcol], dim=-1), ka], dim=2)
        vf = ttnn.concat([ttnn.concat([_bf16(_cols(kv, 0, HEAD_DIM)), zcol], dim=-1), va], dim=2)
        of = ttnn.transformer.scaled_dot_product_attention(
            qf,
            kf,
            vf,
            is_causal=False,
            scale=1.0,
            program_config=self._sdpa_cfg(min(128, t), 256),
            compute_kernel_config=self.sdpa_ckc,
        )  # [1,24,T,160]
        of = ttnn.reshape(_f32(of), [1, N_KV, GROUPS * t, FAR_DH])
        w_near = _cols(of, HEAD_DIM, HEAD_DIM + 1)
        out = ttnn.add(_cols(of, 0, HEAD_DIM), ttnn.multiply(o_near, w_near))
        return ttnn.reshape(out, [1, N_HEADS, t, HEAD_DIM])

    # ------------------------------------------------------------------ MoE
    def _grouped_mm(self, a, w, moe, ei, n, k):
        """a [1, G, GM, k] (slot g of one expert; GM = slot_rows) x w[ei[g]] -> [1, G, GM, n]: indexed sparse_matmul
        (only the slot's expert is read), 1D mcast config with per_core_M = GM / 32."""
        g = a.shape[1]
        grid, ibw = self.group_grid, self.group_ibw[k]
        key = ("grpcfg", n, k, grid, ibw, self.gm)
        if key not in self._sdpa_pc:
            mt, pcn = self.gm // 32, -(-(n // 32) // (grid[0] * grid[1]))
            sub_w = max(d for d in range(1, min(pcn, 4) + 1) if pcn % d == 0)
            sub_h = max(d for d in range(1, mt + 1) if mt % d == 0 and d * sub_w <= 4)
            self._sdpa_pc[key] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
                in0_block_w=ibw,
                out_subblock_h=sub_h,
                out_subblock_w=sub_w,
                out_block_h=mt,
                out_block_w=pcn,
                per_core_M=mt,
                per_core_N=pcn,
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            )
        outs = []
        step = self.r2_step
        for i, idx in enumerate(ei):
            g0 = i * step
            ag = a if len(ei) == 1 else ttnn.slice(a, [0, g0, 0, 0], [1, min(g, g0 + step), self.gm, k])
            outs.append(
                ttnn.sparse_matmul(
                    ag,
                    w,
                    sparsity=moe.sparsity,
                    indices=idx,
                    is_input_a_sparse=True,
                    is_input_b_sparse=True,
                    program_config=self._sdpa_pc[key],
                    compute_kernel_config=self.expert_ckc,
                    dtype=ttnn.bfloat16,
                )
            )
        if len(outs) == 1:
            return ttnn.reshape(outs[0], [1, g, self.gm, n])
        sizes = [min(g, i * step + step) - i * step for i in range(len(outs))]
        out = ttnn.concat([ttnn.reshape(o, [1, z, self.gm, n]) for o, z in zip(outs, sizes)], dim=1)
        return ttnn.reshape(out, [1, g, self.gm, n])

    def _moe(self, lp, h, n):
        """h [1,1,T,C] fp32 (gated mlp input) -> [1,1,T,C] bf16. Rows >= n are not routed (output 0)."""
        moe = lp.lay.moe
        t = h.shape[2]
        x = ttnn.rms_norm(h, weight=moe.norm_w, epsilon=RMS_EPS, compute_kernel_config=moe.router_compute)
        x = _bf16(x)
        logits = ttnn.matmul(x, lp.router_w, dtype=ttnn.float32, compute_kernel_config=moe.router_compute)
        bias = _rows(lp.router_bias_rows, 0, t)
        weights, indices = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            logits,
            bias,
            n_groups=1,
            summed_experts_per_group=1,
            topk_groups=1,
            n_activated_experts=TOP_K,
            route_scale=ROUTE_SCALE,
            epsilon=1e-20,
            stable_sort=True,
        )
        self._tick("moe.route")
        x_rm = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [t, HIDDEN])
        rows = self.gm
        ids = ttnn.to_torch(indices).reshape(t, TOP_K)[:n].to(torch.int64).numpy()  # host sync
        self._tick("moe.read_ids")
        flat = ids.reshape(-1)  # pair p = tok * 8 + k
        counts = np.bincount(flat, minlength=NUM_EXPERTS)
        # slots of GM rows, each bound to one expert: ceil(count / GM) slots per expert (load imbalance is
        # large: one expert can take most tokens of a chunk), slot count rounded up to 32 (few program shapes)
        per = -(-counts // rows)
        nslot = max(32, int(-(-per.sum() // 32) * 32))
        slot0 = np.concatenate([[0], np.cumsum(per)[:-1]])
        eid = np.zeros(nslot, dtype=np.int32)
        eid[: per.sum()] = np.repeat(np.arange(NUM_EXPERTS), per)
        order = np.argsort(flat, kind="stable")
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size) - starts[flat[order]]
        dest = slot0[flat] * rows + rank  # buffer row of each pair
        gidx = np.zeros(nslot * rows, dtype=np.int32)
        gidx[dest] = np.arange(flat.size) // TOP_K
        back = np.zeros((TOP_K, t), dtype=np.int32)  # k-major; pad tokens read row 0 with weight 0
        back[:, :n] = dest.reshape(n, TOP_K).T
        gi = self._put(torch.from_numpy(gidx).view(1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        xs = ttnn.reshape(ttnn.embedding(gi, x_rm, layout=ttnn.TILE_LAYOUT), [1, nslot, rows, HIDDEN])
        self._tick("moe.dispatch")
        ei = [
            self._put(
                torch.from_numpy(eid[g0 : g0 + self.r2_step]).view(1, 1, 1, -1), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT
            )
            for g0 in range(0, nslot, self.r2_step)
        ]  # sparse_matmul takes at most 128 (= experts) indices per call; builds with repeated sorted ids lift that (one call)
        bi = self._put(torch.from_numpy(back).view(1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        return self._experts_combine(lp, moe, xs, ei, bi, nslot, weights, n, t)

    def _experts_combine(self, lp, moe, xs, ei, bi, nslot, weights, n, t):
        """Grouped expert FFN over the slots, gathered back per (k, token) and weighted -> [1,1,T,C] bf16."""
        rows = self.gm
        # DRAM is nearly full: free each slot buffer once read (python refs alone keep them past the next one)
        gu = self._grouped_mm(xs, lp.w_gate_up, moe, ei, 2 * INTER, HIDDEN)
        ttnn.deallocate(xs)
        g, u = _cols(gu, 0, INTER), _cols(gu, INTER, 2 * INTER)
        ttnn.deallocate(gu)
        hid = softcapped_swiglu(g, u)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        self._tick("moe.gate_up")
        y = self._grouped_mm(hid, lp.w_down, moe, ei, HIDDEN, INTER)
        ttnn.deallocate(hid)
        self._tick("moe.down")
        cap = rows
        y_rm = ttnn.reshape(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT), [nslot * cap, HIDDEN])
        ttnn.deallocate(y)
        yb = ttnn.reshape(ttnn.embedding(bi, y_rm, layout=ttnn.TILE_LAYOUT), [1, TOP_K, t, HIDDEN])
        ttnn.deallocate(y_rm)
        wk = _bf16(ttnn.permute(weights, (0, 3, 2, 1)))  # [1,8,T,1]
        if n < t:
            wk = ttnn.multiply(wk, ttnn.lt(_rows(self.c_row_idx, 0, t), float(n), dtype=ttnn.bfloat16))
        if self.wred:  # one pass over the 8 planes, no full-size product
            out = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(yb, wk, dim=1)
        else:
            out = ttnn.sum(ttnn.multiply(yb, wk), dim=1, keepdim=True)
        return _bf16(out)

    # ------------------------------------------------------------------ conv / one application
    def _conv(self, taps, x, tail):
        """Static k=4 conv over the chunk (residual + 1/sqrt(5) folded), fp32 math, bf16 out (HF .to(input))."""
        if self.fused_glue:
            return self._pfconv(taps, x, tail)
        _, lags = tail.window(x)
        acc = None
        for r, xl in enumerate(lags):
            term = ttnn.multiply(xl, taps[r], dtype=ttnn.float32)
            acc = term if acc is None else ttnn.add(acc, term)
        return ttnn.typecast(acc, ttnn.bfloat16)

    def _rms(self, x, w):
        """Weighted RMSNorm (fp32 math), bf16 out."""
        return _bf16(ttnn.rms_norm(x, weight=w, epsilon=EPS, compute_kernel_config=self.hifi4))

    def _app(self, L, p, h, stack, msgs, s, n, tau, ctx):
        """h bf16 [1,1,T,C]; stack: list of (sealed fp32, normed fp32). Returns (out bf16, stack)."""
        lp = self.layers[L]
        lay = lp.lay
        c = cache_index(L, p)
        st = self.m.states[c][0]
        tails = ctx["tails"][c]
        normed = [nn for _, nn in stack]
        if self.debug:
            ttnn.synchronize_device(self.device)
            print(f"app L{L} p{p} s{s} t{h.shape[2]}", flush=True)
        self._tick(None)
        if lp.attn_res is None:
            xs = [h]
        else:
            xs = lp.attn_res(normed, h)  # 1 or 3 fp32
        if msgs is not None:
            if len(xs) == 1:
                xs = xs * 3
            xs = [lp.gates[g](xs[i], msgs[i]) for i, g in enumerate(("q", "k", "v"))]
        if lay.is_global:
            qn, kn, vn = (self._rms(xs[i], w) for i, w in enumerate(lp.norms))
        elif len(xs) == 1:
            qn = kn = vn = self._rms(xs[0], lp.norms[0])
        else:
            qn, kn, vn = (self._rms(x, lp.norms[0]) for x in xs)

        self._tick("resid_in")
        tails["triple"] = lay.is_global or msgs is not None  # decode conv-window mode (attention.ROWS)
        a = self._attention(lp, qn, kn, vn, st, tails, s, n, tau)
        self._tick("attn_global" if lay.is_global else "attn_local")
        if self.fused_glue:  # conv in fp32 out; non-boundary layers add the residual h in the same program
            a = self._pfconv(
                lp.conv_taps["attn"], a, tails["attn"], "f32" if lay.boundary else "hres", None if lay.boundary else h
            )
        else:
            a = _f32(self._conv(lp.conv_taps["attn"], a, tails["attn"]))

        if lay.boundary:
            hf = _f32(h)
            stack = stack + [(hf, ttnn.rms_norm(hf, epsilon=EPS, compute_kernel_config=self.ckc))]
            resid = a
        else:
            resid = a if self.fused_glue else ttnn.add(_f32(h), a)
        m = lp.mlp_res([nn for _, nn in stack], resid)[0]
        if msgs is not None:
            m = lp.gates["mlp"](m, msgs[3])
        self._tick("resid_mid")
        y = self._moe(lp, m, n)
        self._tick("moe")
        if self.fused_glue:
            out = self._pfconv(lp.conv_taps["mlp"], y, tails["mlp"], "resid", resid)
        else:
            y = self._conv(lp.conv_taps["mlp"], y, tails["mlp"])
            out = ttnn.typecast(ttnn.add(resid, _f32(y)), ttnn.bfloat16)
        self._tick("conv_out")
        return out, stack

    def _messages(self, h12_in, h17_out):
        """Per-token cross-pass messages: returns fn(l) -> [q, k, v, mlp] fp32 [1,1,T,C]."""
        n0 = ttnn.rms_norm(_f32(h12_in), epsilon=EPS, compute_kernel_config=self.ckc)
        n1 = ttnn.rms_norm(_f32(h17_out), epsilon=EPS, compute_kernel_config=self.ckc)
        l0 = ttnn.matmul(n0, self.msg_qT, dtype=ttnn.float32, compute_kernel_config=self.ckc)
        l1 = ttnn.matmul(n1, self.msg_qT, dtype=ttnn.float32, compute_kernel_config=self.ckc)
        w0 = ttnn.sigmoid(ttnn.subtract(l0, l1))  # softmax over the 2 keys
        diff = ttnn.subtract(n0, n1)

        def get(l):
            return [ttnn.add(n1, ttnn.multiply(diff, _cols(w0, 4 * l + j, 4 * l + j + 1))) for j in range(4)]

        return get

    # ------------------------------------------------------------------ chunk / prompt
    def _new_ctx(self):
        dev = self.device
        tails = {}
        for L, p in APPS:
            c = cache_index(L, p)
            tails[c] = {
                "k": _Tail(dev, N_KV * HEAD_DIM),
                "v": _Tail(dev, N_KV * HEAD_DIM),
                "attn": _Tail(dev, HIDDEN),
                "mlp": _Tail(dev, HIDDEN),
            }
        return {"tails": tails}

    def _chunk(self, ids, s, n, ctx, logits_rows, force=None, collect=None):
        """Run positions [s, s + t) (t = len(ids), first n real). Returns logits [rows, 200058] fp32 (host) for
        logits_rows (chunk-local row indices) or None.

        force(a, s, t) -> torch [t, C] golden input of app a (teacher forcing, also used for the message keys) or None.
        collect: list that gets the host bf16 output [n, C] of each app."""
        m = self.m
        t = len(ids)
        self._s = s
        tok = self._put(torch.tensor(ids, dtype=torch.int32).view(1, t), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        e = ttnn.embedding(tok, m.embed, layout=ttnn.TILE_LAYOUT)
        h = self._rms(ttnn.reshape(e, [1, 1, t, HIDDEN]), m.embed_norm)
        tau = self._pos_cols(s, t)
        stack = []
        pre = None
        h12_in = None
        msgs = None
        for a, (L, p) in enumerate(APPS):
            if L == LOOP_START:
                if p == 1:
                    pre = (h, stack)
                else:
                    msgs = self._messages(h12_in, h)
                    h, stack = pre
            if L == 12:
                h12_in = h
            if force is not None:
                g = force(a, s, t)
                h = self._put(g.reshape(1, 1, t, HIDDEN).to(torch.bfloat16), ttnn.bfloat16)
                if L == 12:
                    h12_in = h
            lm = msgs(L - LOOP_START) if (msgs is not None and LOOP_START <= L < LOOP_END and p > 1) else None
            h, stack = self._app(L, p, h, stack, lm, s, n, tau, ctx)
            if collect is not None:
                collect.append(ttnn.to_torch(h).reshape(t, HIDDEN)[:n].clone())
        ctx["last_h"] = (h, stack)
        if not logits_rows:
            return None
        return self._logits(h, stack, logits_rows)

    def _logits(self, h, stack, rows):
        m = self.m
        out = []
        x = self.final_res([nn for _, nn in stack], h)[0]
        x = self._rms(x, m.final_norm)
        rows = list(rows)
        blk = 32  # the model's DRAM-sharded unembed takes one 32-row tile row
        for b0 in range(0, len(rows), blk):
            rr = rows[b0 : b0 + blk]
            lo = rr[0] - rr[0] % 32
            if rr[-1] >= lo + 32:  # rows spanning two tile rows: split
                rr = [r for r in rr if r < lo + 32]
            xb = _rows(x, lo, lo + 32)
            lg = m.unembed(xb)
            lg = ttnn.multiply(ttnn.tanh(ttnn.multiply(lg, 1.0 / SOFTCAP)), SOFTCAP)
            lt = ttnn.to_torch(lg).float().reshape(-1, lg.shape[-1])
            part = lt[[r - lo for r in rr], :VOCAB_REAL]
            if self.logits_cb is not None:  # stream (host RSS): cb(first absolute position, logits rows)
                self.logits_cb(self._s + rr[0], part)
            else:
                out.append(part)
        return torch.cat(out) if out else torch.zeros(0, VOCAB_REAL)

    def _handoff(self, ctx, n):
        """Decode state of the last position P (tt/model.py reads it):
        - state.x_hist: k/v conv window of q/k/v projection inputs (attention.ROWS: single = rows 0..3 = lags
          0..3 of the shared input; triple = row 0 q, rows 1..4 k lags, rows 5..8 v lags);
        - post-attn / post-mlp FusedShortConv rings: ring[(P - k) % 4] = conv input at lag k, dense fp32 pages."""
        P = ctx["L"] - 1
        zero = self.c_zero_row
        conv_rows = []
        for c, tails in ctx["tails"].items():
            st = self.m.states[c][0]
            q, k, v = (row or [None] * TAPS for row in tails["xwin"])
            fix = lambda r: zero if r is None else _bf16(r)  # noqa: E731
            if tails["triple"]:
                rows = [fix(q[0])] + [fix(r) for r in k] + [fix(r) for r in v]
            else:
                rows = [fix(r) for r in k]
            z = self.c_zx[len(rows)]
            ttnn.copy(ttnn.concat(rows + [z], dim=2), st.x_hist)
            pad = ctx["t"] - n
            for name in ("attn", "mlp"):
                ext = ctx["ext"][c][name]
                e = ext.shape[2] - pad
                conv_rows.append(_bf16(_rows(ext, e - TAPS, e)))  # positions P-3 .. P
        host = ttnn.to_torch(ttnn.concat(conv_rows, dim=2)).float().reshape(-1, TAPS, HIDDEN)  # one read
        i = 0
        for c in ctx["tails"]:
            _, ca, cm = self.m.states[c]
            for ring in (ca, cm):
                for k in range(TAPS):
                    vec = host[i, TAPS - 1 - k]  # lag k
                    ttnn.copy_host_to_device_tensor(
                        ttnn.from_torch(dense_matrix(vec.view(1, -1)), dtype=ring[0].dtype, layout=ttnn.TILE_LAYOUT),
                        ring[(P - k) % TAPS],
                    )
                i += 1

    def prefill(self, ids, logits_all=False, log=None, force=None, collect=None, chunk=None):
        """ids: prompt token list. Fills the model's decode state for positions [0, len). Returns logits fp32 host
        tensor: [len, 200058] if logits_all else [1, 200058] (last position). chunk: tokens per pass (default
        self.chunk, at most self.chunk)."""
        m = self.m
        if getattr(m, "overlap", None) is not None:
            raise RuntimeError(
                "decode weight staging is on (Generator.capture): its L1 tensor clashes with prefill programs; "
                "use Generator(model, overlap=False) in sessions that prefill between capture() and release()"
            )
        L = len(ids)
        assert 0 < L <= m.max_ctx
        chunk = chunk or self.chunk
        assert chunk <= self.chunk and self._chunk_ok(chunk), (chunk, self.chunk)
        m.reset()
        ctx = self._new_ctx()
        ctx["L"] = L
        outs = []
        s = 0
        while s < L:
            n = min(chunk, L - s)
            t = -(-n // QB) * QB
            assert s + t <= m.max_ctx, "max_ctx must cover the padded last chunk"
            chunk_ids = list(ids[s : s + n]) + [0] * (t - n)
            last = s + n == L
            rows = list(range(n)) if logits_all else ([n - 1] if last else [])
            self._last = last
            if last:
                self._keep_ext(ctx)
            col = [] if collect is not None else None
            r = self._chunk(chunk_ids, s, n, ctx, rows, force, col)
            if collect is not None:
                collect.append(col)
            if r is not None:
                outs.append(r)
            if log:
                log(s, n)
            if last:
                ctx["t"] = t
                self._handoff(ctx, n)
            s += n
        return torch.cat(outs)

    def _keep_ext(self, ctx):
        """Make the tails keep [old tail ++ chunk] of the last chunk (for the conv-history handoff)."""
        ctx["ext"] = {}
        for c, tails in ctx["tails"].items():
            ctx["ext"][c] = {}
            for name, tl in tails.items():
                if not isinstance(tl, _Tail):
                    continue
                tl.keep = (ctx["ext"][c], name)
                orig = tl.window

                def window(x, tl=tl, orig=orig):
                    ext, lags = orig(x)
                    tl.keep[0][tl.keep[1]] = ext
                    return ext, lags

                tl.window = window
