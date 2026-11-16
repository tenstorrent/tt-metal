# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""One MoE layer around the lane stream (ws_lane.py). Every phase is single-chip: each chip works on its own T tokens
with replicated router and shared-expert weights, and only the routed stream crosses chips (its weight streaming).
The phases meet in a row-major result buffer [T S, H] bf16, S = K + 1: row t S + k holds token t's k-th routed
expert's output, row t S + K its shared expert's output.

- head (ttnn ops): router logits (fp32), softmax / sigmoid scores, top-K and the renormalized gate weights (w, [TX, 16]
  fp32 row-major, K used); the stream's inputs from the top-K on device: the one-hot routing mask (scatter), its
  running count over the tokens (cumsum: a routed row's position in its expert's list), the per-expert counts, and per
  expert two lists in the stream's token-list layout: its tokens t and their result rows t S + k (scattered to their
  positions; the rows a token does not route to index past the list, which scatter skips);
- routed stream: build_lane with y_rm, each expert's down rows written straight to their result rows;
- end pass (ws_end.py): y[t] = sum over k of w[t, k] row t S + k, plus row t S + K."""

import time
from dataclasses import dataclass

import torch

import ttnn
from models.demos.ws_expert.ws_group import GCfg
from models.demos.ws_expert.ws_lane import LCfg


@dataclass
class LayerCfg:
    T: int = 31250  # tokens a chip
    H: int = 3840
    I: int = 640  # routed expert inner dim (= the lane kernels' slice width)
    nexp: int = 640
    K: int = 12
    IS: int = 2944  # shared expert inner dim
    G: int = 4
    score: str = "softmax"  # or "sigmoid"

    TX = property(lambda s: -(-(s.T + 1) // 32) * 32)  # x rows: the tokens, then zero rows (row T: the lists' pad)
    RS = property(lambda s: -(-s.T // 32) * 32)  # token-list width (GCfg.RMAX 32)
    S = property(lambda s: s.K + 1)  # result rows a token


def gcfg(nexp, G, T):
    return GCfg(nexp=nexp, G=G, T=T, MMAX=T, BK=20, NX=9, LOOK=18)


def proxy_lcfg(G, **kw):
    """The one-chip lane configuration (no fabric, every expert the chip's own), no-hang layout, row-major outputs."""
    return LCfg(
        cycles=[[(0, 0)]] * ((G + 1) // 2),
        nofab=True,
        slots="any,any,any,any",
        keep=0,
        bcast=True,
        noc_split=True,
        y_rm=True,
        **kw,
    )


def result_buffer(mesh, lc: LayerCfg):
    """The row-major result buffer [T S, H] bf16, zeroed (rows nobody writes stay finite)."""
    return ttnn.zeros(
        [lc.T * lc.S, lc.H],
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ---------------------------------------------------------------- head


def head_consts(mesh, lc: LayerCfg):
    """Host-built constants of the head: the valid-token column / K columns (1 for t < T), the routing mask's zeros, the
    lists' initial T, the token lists' scattered values (t), the slot values (k) and t S a token, the w padding."""
    TX = lc.TX
    valid = (torch.arange(TX) < lc.T).float()
    t = lambda x, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(x, dtype=ttnn.float32, layout=lay, device=mesh)
    rm = ttnn.ROW_MAJOR_LAYOUT  # (scatter takes fp32 row-major only)
    return dict(
        validk=t(valid.reshape(1, 1, TX, 1).expand(1, 1, TX, lc.K).contiguous(), rm),
        validc=t(valid.reshape(1, 1, TX, 1)),
        zeros=t(torch.zeros(1, 1, TX, lc.nexp), rm),
        lists0=ttnn.from_torch(
            torch.full((lc.nexp, lc.RS), lc.T, dtype=torch.int32), dtype=ttnn.uint32, layout=rm, device=mesh
        ),
        tok=ttnn.from_torch(
            torch.arange(lc.RS, dtype=torch.int32).expand(lc.nexp, -1).contiguous(),
            dtype=ttnn.uint32,
            layout=rm,
            device=mesh,
        ),
        kslot=t(torch.arange(lc.K).float().expand(1, 1, TX, lc.K).contiguous(), rm),
        trow=t((torch.arange(TX).float() * lc.S).reshape(1, 1, 1, TX)),
        wpad=t(torch.zeros(1, 1, TX, 16 - lc.K)),
    )


def router_weights(mesh, wr):
    return ttnn.from_torch(wr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)


def head(mesh, lc: LayerCfg, x_t, wr_t, hc, marks=None):
    """x_t [TX, H] bf16 TILE (rows >= T zero). Returns the stream's x (row-major), idx_t / dest_t [nexp, RS] (token
    lists, result-row lists) and counts_t [1, nexp] uint32 row-major, the gate weights w [TX, 16] fp32 row-major (0 for
    t >= T), the top-K indices and the logits. marks: a list to append (step, synchronized wall time) to (a breakdown;
    None: no syncs)."""
    TX, E, K = lc.TX, lc.nexp, lc.K

    def mark(name):
        if marks is not None:
            ttnn.synchronize_device(mesh)
            marks.append((name, time.perf_counter()))

    mark("start")
    x_rm = ttnn.to_layout(x_t, ttnn.ROW_MAJOR_LAYOUT)
    ck = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    logits = ttnn.matmul(x_t, wr_t, dtype=ttnn.float32, compute_kernel_config=ck)
    logits = ttnn.reshape(logits, [1, 1, TX, E])
    s = ttnn.softmax(logits, dim=-1) if lc.score == "softmax" else ttnn.sigmoid(logits)
    vals, top = ttnn.topk(s, k=K, dim=-1, largest=True, sorted=True)
    w = ttnn.multiply(ttnn.div(vals, ttnn.sum(vals, dim=-1, keepdim=True)), hc["validc"])
    w = ttnn.reshape(ttnn.to_layout(ttnn.concat([w, hc["wpad"]], dim=3), ttnn.ROW_MAJOR_LAYOUT), [TX, 16])
    mark("router")
    # one-hot routing mask (padding tokens route nowhere), its running count over the tokens
    mask = ttnn.scatter(hc["zeros"], -1, top, hc["validk"])
    mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
    run = ttnn.cumsum(mask, dim=2, dtype=ttnn.float32)
    cnt = ttnn.slice(run, [0, 0, TX - 1, 0], [1, 1, TX, E])
    counts_t = ttnn.reshape(ttnn.to_layout(ttnn.typecast(cnt, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, E])
    mark("counts")
    # lists: entry at the routed row's position, else at RS (past the list: skipped)
    at = ttnn.add(ttnn.multiply(mask, ttnn.subtract(run, float(1 + lc.RS))), float(lc.RS))
    at = ttnn.slice(ttnn.transpose(at, 2, 3), [0, 0, 0, 0], [1, 1, E, lc.RS])
    at = ttnn.reshape(ttnn.to_layout(ttnn.typecast(at, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [E, lc.RS])
    idx_t = ttnn.scatter(hc["lists0"], -1, at, hc["tok"])
    # result rows: t S + the slot k that chose the expert
    ks = ttnn.to_layout(ttnn.scatter(hc["zeros"], -1, top, hc["kslot"]), ttnn.TILE_LAYOUT)
    rr = ttnn.slice(ttnn.add(ttnn.transpose(ks, 2, 3), hc["trow"]), [0, 0, 0, 0], [1, 1, E, lc.RS])
    rr = ttnn.reshape(ttnn.to_layout(ttnn.typecast(rr, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [E, lc.RS])
    dest_t = ttnn.scatter(hc["lists0"], -1, at, rr)
    mark("lists")
    return dict(x_rm=x_rm, idx_t=idx_t, dest_t=dest_t, counts_t=counts_t, w=w, top=top, logits=logits)
