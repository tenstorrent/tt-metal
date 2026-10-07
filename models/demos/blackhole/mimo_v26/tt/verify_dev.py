# SPDX-License-Identifier: Apache-2.0
"""Device-resident spec-decode VERIFY round (Decoder(block=8, vdev=True)): no host copy, no host compute between rounds.

State (persistent device tensors, float32 tiles, row sharded: every mesh row has its own users; ids/positions < 2^24 are exact):
  blk_in  [1,1,U,8]  block tokens of the NEXT verify: col 0 = anchor (written by the epilogue of the previous round = the bonus token), cols 1..7 = drafts (written by the drafter / write_drafts)
  vpos    [1,1,U,1]  position of block token 0 (number of tokens in the KV cache)
  valive  [1,1,U,1]  1 = user still generating
Prologue (first ops of the traced step): expands the state into the plain decode inputs of the B = 8U virtual rows (tok, cur, idx, the 8 per-token KV update-index tensors with the -1 skip
pattern, live mask). Dead users feed token 0 / position 0 / update index 0 exactly like Decoder.set_inputs.
Epilogue (after the lm_head): target argmax `vpost` [1,1,U,8], accept_prefix (cumulative product of draft == argmax), bonus token, committed count `vcommit`=k+1 (0 for dead users),
`vvalid` [1,1,U,8] (1 for committed rows), `vposrows` [1,1,U,8] (position of each block row), then updates vpos += k+1, valive (eos among the committed tokens / context end), and
blk_in[:, 0] = bonus. Also records post / commit in a ring of N rounds written at a device round counter (host reads it in bulk with Decoder.read_rounds).
"""
import torch

import ttnn

F, RM, TILE = ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
T64 = 64


def init_state(d):
    rep = ttnn.ReplicateTensorToMesh(d.mesh)
    mk = lambda t, mm=rep: ttnn.from_torch(t, device=d.mesh, dtype=F, layout=TILE, mesh_mapper=mm)
    U, BLK, N, rows = d.U, d.block, d.N, d.rows
    d.blk_in = mk(torch.zeros(rows, 1, U, BLK), d.rowmap)
    d.vpos = mk(torch.zeros(rows, 1, U, 1), d.rowmap)
    d.valive = mk(torch.zeros(rows, 1, U, 1), d.rowmap)
    d.vcnt = mk(torch.zeros(1, 1, 1, 1))
    d.v_iota_n = mk(torch.arange(N, dtype=torch.float32).reshape(1, N, 1, 1))
    d.v_iota_b = mk(torch.arange(BLK, dtype=torch.float32).reshape(1, 1, 1, BLK))
    d.rpost = mk(torch.zeros(1, N, U, BLK))
    d.rcommit = mk(torch.zeros(1, N, U, 1))
    # outputs of the last round (for the drafter append): argmax rows, committed mask, row positions
    d.vpost = mk(torch.zeros(1, 1, U, BLK))
    d.vcommit = mk(torch.zeros(1, 1, U, 1))
    d.vvalid = mk(torch.zeros(1, 1, U, BLK))
    d.vposrows = mk(torch.zeros(1, 1, U, BLK))
    d.vk = mk(torch.zeros(1, 1, U, 1))


STATE_NAMES = (
    "blk_in",
    "vpos",
    "valive",
    "vcnt",
    "v_iota_n",
    "v_iota_b",
    "rpost",
    "rcommit",
    "vpost",
    "vcommit",
    "vvalid",
    "vposrows",
    "vk",
)


def _rm(t, dt, shp):
    return ttnn.reshape(ttnn.typecast(ttnn.to_layout(t, RM), dt), shp)


def prologue(d):
    U, BLK, B = d.U, d.block, d.B
    f = lambda t: ttnn.typecast(t, F)
    ar, al = d.v_iota_b, d.valive
    ones = ttnn.add(ttnn.multiply(ar, 0.0), 1.0)
    posb = ttnn.multiply(ttnn.add(d.vpos, ar), al)  # [1,1,U,8] position of every block row (0 for dead users)
    tokf = ttnn.multiply(d.blk_in, al)
    ttnn.copy(_rm(tokf, ttnn.uint32, (1, B)), d.tok)
    ttnn.copy(_rm(posb, ttnn.int32, (B,)), d.pos["cur"])
    ttnn.copy(_rm(posb, ttnn.uint32, (1, B)), d.pos["idx"])
    for t in range(BLK):  # update index of call t: position for the rows of token t, -1 (skip) for all others
        m = f(ttnn.eq(ar, float(t)))
        val = ttnn.add(ttnn.multiply(posb, m), ttnn.subtract(m, 1.0))
        ttnn.copy(_rm(val, ttnn.int32, (B,)), d.upd[t])
    if d.live is not None:
        lv = ttnn.to_layout(_rm(ttnn.multiply(al, ones), F, (1, 1, B, 1)), TILE)
        ttnn.copy(ttnn.typecast(ttnn.pad(lv, [(0, 0), (0, 0), (0, T64 - B), (0, 0)], 0.0), ttnn.bfloat16), d.live.t)


def epilogue(d, pick):
    """pick: float32 tile [1,1,B,1] = target argmax of every virtual row."""
    U, BLK, B = d.U, d.block, d.B
    f = lambda t: ttnn.typecast(t, F)
    ar, al, pos, blk = d.v_iota_b, d.valive, d.vpos, d.blk_in
    post = ttnn.to_layout(ttnn.reshape(ttnn.to_layout(pick, RM), (1, 1, U, BLK)), TILE)  # [1,1,U,8]
    ok = f(ttnn.eq(ttnn.slice(post, [0, 0, 0, 0], [1, 1, U, BLK - 1]), ttnn.slice(blk, [0, 0, 0, 1], [1, 1, U, BLK])))
    c = ttnn.slice(ok, [0, 0, 0, 0], [1, 1, U, 1])
    k = c
    for j in range(1, BLK - 1):  # cumulative product over the draft positions, summed = number of accepted drafts
        c = ttnn.multiply(c, ttnn.slice(ok, [0, 0, 0, j], [1, 1, U, j + 1]))
        k = ttnn.add(k, c)
    oh = f(ttnn.eq(ar, k))  # [1,1,U,8] one-hot at k
    bonus = ttnn.sum(ttnn.multiply(oh, post), dim=3, keepdim=True)
    le = f(ttnn.le(ar, k))  # committed rows 0..k
    valid = ttnn.multiply(le, al)
    commit = ttnn.multiply(ttnn.add(k, 1.0), al)
    hit = None
    for e in d.eos:
        x = f(ttnn.eq(post, float(e)))
        hit = x if hit is None else ttnn.add(hit, x)
    a_new = al
    if hit is not None:
        eos_any = ttnn.sum(ttnn.multiply(hit, le), dim=3, keepdim=True)
        a_new = ttnn.multiply(a_new, f(ttnn.eq(eos_any, 0.0)))
    pos_new = ttnn.add(pos, commit)
    a_new = ttnn.multiply(a_new, f(ttnn.lt(ttnn.add(pos_new, float(BLK + 1)), float(d.max_pos))))
    # outputs for the drafter append / host
    ttnn.copy(ttnn.multiply(post, al), d.vpost)
    ttnn.copy(commit, d.vcommit)
    ttnn.copy(valid, d.vvalid)
    ttnn.copy(ttnn.add(pos, ar), d.vposrows)
    ttnn.copy(ttnn.multiply(k, al), d.vk)
    # rings at the round counter
    ohn = f(ttnn.eq(d.v_iota_n, d.vcnt))  # [1,N,1,1]
    keep = ttnn.subtract(1.0, ohn)
    ttnn.copy(ttnn.add(ttnn.multiply(d.rpost, keep), ttnn.multiply(ohn, ttnn.multiply(post, al))), d.rpost)
    ttnn.copy(ttnn.add(ttnn.multiply(d.rcommit, keep), ttnn.multiply(ohn, commit)), d.rcommit)
    c1 = ttnn.add(d.vcnt, 1.0)
    ttnn.copy(ttnn.multiply(c1, f(ttnn.lt(c1, float(d.N)))), d.vcnt)
    # next round's state: positions, alive, anchor = bonus (0 for dead users)
    e0 = f(ttnn.eq(ar, 0.0))
    ttnn.copy(
        ttnn.add(ttnn.multiply(blk, ttnn.subtract(1.0, e0)), ttnn.multiply(e0, ttnn.multiply(bonus, a_new))), d.blk_in
    )
    ttnn.copy(pos_new, d.vpos)
    ttnn.copy(a_new, d.valive)


def write_drafts(d, drafts):
    """blk_in[..., 1:8] = drafts (device tensor, any numeric dtype/layout, shape [..., U, 7] or [..., U*7]); device ops only, trace friendly."""
    U, BLK = d.U, d.block
    x = drafts
    if x.layout != TILE:
        x = ttnn.to_layout(x, TILE)
    if x.dtype != F:
        x = ttnn.typecast(x, F)
    x = ttnn.reshape(x, (1, 1, U, BLK - 1))
    first = ttnn.slice(d.blk_in, [0, 0, 0, 0], [1, 1, U, 1])
    ttnn.copy(ttnn.concat([first, x], dim=3), d.blk_in)


def set_state(d, anchor, pos, alive):
    """host write (once per generation): anchor/pos long [rows,U], alive bool [rows,U]; resets the round counter."""
    d._vlast = (anchor, pos, alive)
    rows = anchor.shape[0]
    blk = torch.zeros(rows, 1, d.U, d.block)
    blk[:, 0, :, 0] = anchor.float() * alive.float()
    keep = []
    for t_, dev in (
        (blk, d.blk_in),
        (pos.float().reshape(rows, 1, d.U, 1), d.vpos),
        (alive.float().reshape(rows, 1, d.U, 1), d.valive),
    ):
        h = ttnn.from_torch(t_, dtype=F, layout=TILE, mesh_mapper=d.rowmap)
        ttnn.copy_host_to_device_tensor(h, dev)
        keep.append(h)
    hz = ttnn.from_torch(torch.zeros(1, 1, 1, 1), dtype=F, layout=TILE, mesh_mapper=ttnn.ReplicateTensorToMesh(d.mesh))
    ttnn.copy_host_to_device_tensor(hz, d.vcnt)
    keep.append(hz)
    d.c0 = 0
    ttnn.synchronize_device(d.mesh)
    d._keep = keep


def read_rounds(d, n):
    """bulk read of the last n rounds: post long [n,rows,U,8] (0 for dead users), commit long [n,rows,U] (k+1, 0 = user was dead)."""
    cols = d.mesh.shape[1]
    rp, rc = ttnn.get_device_tensors(d.rpost), ttnn.get_device_tensors(d.rcommit)
    post = torch.stack([ttnn.to_torch(rp[r * cols]).reshape(d.N, d.U, d.block) for r in range(d.rows)], 1)
    com = torch.stack([ttnn.to_torch(rc[r * cols]).reshape(d.N, d.U) for r in range(d.rows)], 1)
    idx = [(d.c0 + j) % d.N for j in range(n)]
    d.c0 = (d.c0 + n) % d.N
    return post[idx].round().long(), com[idx].round().long()
