# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""One MoE layer around the lane stream (ws_lane.py). Every phase is single-chip: each chip works on its own T tokens
with replicated router and shared-expert weights, and only the routed stream crosses chips (its weight streaming).

- head (ttnn ops): router logits (fp32), softmax / sigmoid scores, top-K and the renormalized gate weights; the
  stream's inputs from the top-K on device: the one-hot routing mask (scatter), its running count over the tokens
  (cumsum: a routed row's position in its expert's token list), the per-expert counts, each expert's token list (t
  scattered to its position in a list of T, x's zero row; the rows a token does not route to index past the list,
  which scatter skips) and each routed row's output row (the expert's expert-id-ordered base, as the stream's
  WL_OUT_EORD writes it, + the position);
- shared expert through the lane kernels: the shared inner dim in NSL 640-wide slices (the last zero-padded) x G token
  chunks = NSL G units, unit u = i G + g (slice i, chunk g) on group g (LCfg.nofab on every chip, weights replicated);
  the units' down outputs are expert-id ordered, so the shared output is the sum of the NSL slices' blocks;
- routed stream: build_lane with out_eord (expert e's rows from the prefix of ceil(M / 32) over the experts before it);
- combine (ttnn ops): the routed rows untilized, gathered per (token, k) (embedding) in 32-token blocks, each block's
  K 32 rows times its gate matrix (a batched matmul: the weighted sum over k), plus the shared output."""

import time
from dataclasses import dataclass

import torch

import ttnn
from models.demos.ws_expert.ws_fab import slice_layout
from models.demos.ws_expert.ws_group import GCfg
from models.demos.ws_expert.ws_lane import LCfg, make_tables, tables_tensor


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
    chunk: int = 4096  # combine token chunk

    TX = property(lambda s: -(-(s.T + 1) // 32) * 32)  # x rows: the tokens, then zero rows (row T: the lists' pad)
    RS = property(lambda s: -(-s.T // 32) * 32)  # token-list width (GCfg.RMAX 32) = output rows
    NSL = property(lambda s: -(-s.IS // s.I))  # shared slices
    TC = property(lambda s: -(-s.T // (s.G * 32)) * 32)  # shared token chunk (each unit's M)
    ROUT = property(lambda s: -(-(s.T * s.K) // 32) + s.nexp)  # routed output tile rows: >= sum ceil(M / 32)


def gcfg(nexp, G, T):
    return GCfg(nexp=nexp, G=G, T=T, MMAX=T, BK=20, NX=9, LOOK=18)


def proxy_lcfg(G, **kw):
    """The one-chip lane configuration (no fabric, every expert the chip's own), no-hang layout, expert-id order."""
    return LCfg(
        cycles=[[(0, 0)]] * ((G + 1) // 2),
        nofab=True,
        slots="any,any,any,any",
        keep=0,
        bcast=True,
        noc_split=True,
        out_eord=True,
        **kw,
    )


# ---------------------------------------------------------------- shared expert as lane units


def shared_units(lc: LayerCfg, w1, w3, w2):
    """Host weights of the NSL G units (unit u = i G + g: slice i of the gate / up columns and down rows, zero-padded to
    NSL I): ([U, H, I], [U, H, I], [U, I, H])."""
    pad = lc.NSL * lc.I - lc.IS
    w1p = torch.nn.functional.pad(w1.float(), (0, pad))
    w3p = torch.nn.functional.pad(w3.float(), (0, pad))
    w2p = torch.nn.functional.pad(w2.float(), (0, 0, 0, pad))
    sl = [i for i in range(lc.NSL) for _ in range(lc.G)]
    wg = torch.stack([w1p[:, i * lc.I : (i + 1) * lc.I] for i in sl])
    wu = torch.stack([w3p[:, i * lc.I : (i + 1) * lc.I] for i in sl])
    wd = torch.stack([w2p[i * lc.I : (i + 1) * lc.I] for i in sl])
    return wg, wu, wd


def shared_setup(mesh, lc: LayerCfg, w1, w3, w2):
    """The shared expert's lane inputs, the same on every chip: weights, unit token lists (chunk g = tokens g TC ..,
    padded with T) and counts (TC each), the one-chip tables on every chip, the output (NSL G TC rows)."""
    N = mesh.get_num_devices()
    U = lc.NSL * lc.G
    cfg = gcfg(U, lc.G, lc.T)
    lcf = proxy_lcfg(lc.G)
    tb = make_tables((1, 1), cfg, lcf)
    wg, wu, wd = shared_units(lc, w1, w3, w2)
    o = tb["own"][0]
    ws = slice_layout(cfg, lcf.CHT, wg[o], wu[o], wd[o])
    idx = torch.full((U, cfg.RMAX * 32), lc.T, dtype=torch.int32)
    for u in range(U):
        g = u % lc.G
        t = torch.arange(g * lc.TC, (g + 1) * lc.TC, dtype=torch.int32)
        idx[u, : lc.TC] = torch.where(t < lc.T, t, torch.full_like(t, lc.T))
    counts = torch.full((1, U), lc.TC, dtype=torch.int32)
    rm = lambda t: ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    tabs = tb["tabs"].expand(N, -1, -1).contiguous()
    return dict(
        cfg=cfg,
        lc=lcf,
        tb=dict(tb, tabs=tabs),
        tab_t=tables_tensor(mesh, tabs),
        idx_t=rm(idx),
        counts_t=rm(counts),
        ws_t=ttnn.from_torch(
            ws, dtype=cfg.w_dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        ),
        out_t=ttnn.empty(
            [U * lc.TC, lc.H],
            dtype=cfg.out_dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )


def shared_sum(lc: LayerCfg, out_s):
    """The shared expert's output [1, 1, RS, H]: the sum of the NSL slices' blocks (G TC rows each, token order)."""
    s = ttnn.sum(ttnn.reshape(out_s, [1, lc.NSL, lc.G * lc.TC, lc.H]), dim=1, keepdim=True)
    return ttnn.slice(s, [0, 0, 0, 0], [1, 1, lc.RS, lc.H])


# ---------------------------------------------------------------- head


def head_consts(mesh, lc: LayerCfg):
    """Host-built constants of the head: the valid-token column / K columns (1 for t < T), the routing mask's zeros, the
    token lists' initial T and their scattered values (t a token)."""
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
        diag=t(torch.eye(32).repeat(1, lc.K).reshape(1, 32, 32 * lc.K)),
    )


def router_weights(mesh, wr):
    return ttnn.from_torch(wr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)


def head(mesh, lc: LayerCfg, x_t, wr_t, hc, marks=None):
    """x_t [TX, H] bf16 TILE (rows >= T zero). Returns the stream's x (row-major), idx_t [nexp, RS] / counts_t [1, nexp]
    uint32 row-major, the routed rows' output rows R and gate weights w ([1, 1, TX, K]; 0 / 0 for t >= T), the top-K
    indices and the logits. marks: a list to append (step, synchronized wall time) to (a breakdown; None: no syncs)."""
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
    mark("router")
    # one-hot routing mask (padding tokens route nowhere), its running count over the tokens
    mask = ttnn.scatter(hc["zeros"], -1, top, hc["validk"])
    mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
    run = ttnn.cumsum(mask, dim=2, dtype=ttnn.float32)
    cnt = ttnn.slice(run, [0, 0, TX - 1, 0], [1, 1, TX, E])
    rows = ttnn.floor(ttnn.multiply(ttnn.add(cnt, 31.0), 1.0 / 32))
    base = ttnn.subtract(ttnn.cumsum(rows, dim=3, dtype=ttnn.float32), rows)  # expert-id-ordered first tile row
    rfull = ttnn.add(run, ttnn.subtract(ttnn.multiply(base, 32.0), 1.0))
    R = ttnn.multiply(ttnn.gather(rfull, -1, top), hc["validc"])
    mark("counts_rows")
    # token lists: t to its position where routed, else to RS (past the list: skipped)
    at = ttnn.add(ttnn.multiply(mask, ttnn.subtract(run, float(1 + lc.RS))), float(lc.RS))
    at = ttnn.slice(ttnn.transpose(at, 2, 3), [0, 0, 0, 0], [1, 1, E, lc.RS])
    at = ttnn.reshape(ttnn.to_layout(ttnn.typecast(at, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [E, lc.RS])
    idx_t = ttnn.scatter(hc["lists0"], -1, at, hc["tok"])
    counts_t = ttnn.reshape(ttnn.to_layout(ttnn.typecast(cnt, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, E])
    mark("lists")
    return dict(x_rm=x_rm, idx_t=idx_t, counts_t=counts_t, R=R, w=w, top=top, logits=logits)


# ---------------------------------------------------------------- combine


def combine_prep(lc: LayerCfg, hd, hc):
    """The gather indices in (32-token block, k, token) order (uint32 row-major [1, TX K]) and the blocks' gate
    matrices [TX / 32, 32, 32 K] (w (t, k) at row t, column 32 k + t of its block, else 0), bf16."""
    nb, K = lc.TX // 32, lc.K
    Ri = ttnn.to_layout(ttnn.typecast(hd["R"], ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
    Ri = ttnn.reshape(ttnn.permute(ttnn.reshape(Ri, [nb, 32, K]), [0, 2, 1]), [1, nb * K * 32])
    wb = ttnn.reshape(ttnn.repeat_interleave(hd["w"], 32, dim=3), [nb, 32, K * 32])
    return Ri, ttnn.typecast(ttnn.multiply(wb, hc["diag"]), ttnn.bfloat16)


def combine(mesh, lc: LayerCfg, out_r, Ri, Wb, ysh, marks=None):
    """y [1, 1, RS, H] bf16: the routed rows untilized and gathered per (token, k) (embedding), weighted and summed
    over k (per 32-token block its K 32 rows times its gate matrix: a batched matmul; a weigh-and-sum over k with
    eltwise ops was slower), plus the shared output. marks: as head's, per step summed over the chunks."""
    K, H = lc.K, lc.H

    def mark(name):
        if marks is not None:
            ttnn.synchronize_device(mesh)
            marks.append((name, time.perf_counter()))

    ck = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )
    mark("start")
    table = ttnn.to_layout(out_r, ttnn.ROW_MAJOR_LAYOUT)
    mark("untilize")
    ys = []
    for t0 in range(0, lc.RS, lc.chunk):
        n = min(lc.chunk, lc.RS - t0)
        b0, b1 = t0 // 32, (t0 + n) // 32
        ri = ttnn.slice(Ri, [0, b0 * K * 32], [1, b1 * K * 32])
        rows = ttnn.reshape(ttnn.embedding(ri, table, layout=ttnn.TILE_LAYOUT), [b1 - b0, K * 32, H])
        mark("gather")
        part = ttnn.matmul(ttnn.slice(Wb, [b0, 0, 0], [b1, 32, K * 32]), rows, compute_kernel_config=ck)
        part = ttnn.reshape(part, [1, 1, n, H])
        mark("weigh_sum")
        ttnn.deallocate(rows)
        ys.append(ttnn.add(part, ttnn.slice(ysh, [0, 0, t0, 0], [1, 1, t0 + n, H])))
        mark("add_shared")
    ttnn.deallocate(table)
    y = ys[0] if len(ys) == 1 else ttnn.concat(ys, dim=2)
    mark("concat")
    return y
