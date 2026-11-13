# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The inputs of a chained MTP verify, built on the device from the unread verify before it.

vllm-tt-plugin's ``supports_chained_spec_verify``: under asynchronous scheduling the runner submits verify n + 1
before it reads verify n, sending placeholders for every live row and verify n's ``VerifyOutput`` as
``spec_predecessor``. When verify n was a one-enqueue step, the device already holds everything verify n + 1 needs:

  * the verify ids of n (``Qwen36Model._vfy_ids_out``, the fused argmax lanes);
  * its accepted drafts m[u] (the drafter pack ``MTPSpecEngine._dr_pack``, whose anchor row the step trace wrote as
    u * T + m[u]);
  * the K drafts its drafter made from that commit (``MTPSpecEngine._dr_outs``).

The chain program (ChainInputs.body, traced) runs before the verify trace of n + 1 and writes:

  * the verify step buffer of n + 1 (spec_step): tokens [ids[u, m], drafts...] (a column past the row's valid drafts
    repeats column 0, as sanitize_verify_block pads), KV positions p0 + m + 1 + j, RoPE table rows, and the acceptance
    rows the step trace of n + 1 reads (STEP_P0 and STEP_ROPE_BASE computed, the others from the host overlay);
  * one GDN verify ctrl page per MambaSpec table group: the start pages of each row's count c = m + 1, picked from
    candidate pages the host stages for every count (GdnBlockState.ctrl_candidates, from the plugin's
    ``MambaGroupState.start_blocks_by_count``).

What the host knows before n is read rides in the overlay (CH_* rows): n's column-0 positions, the rows n + 1 runs,
their M-RoPE deltas, and n + 1's valid drafts and propose decisions (decided for every count n can accept). Every
value is an integer below 2**24 (token ids, positions, table rows, page indices) or -1 (SPEC_STEP_HOLD, a held row's
ctrl word), so float32 holds it exactly and the arithmetic runs in float32 tiles, as spec_accept.graph does.

``chain_graph`` is written against an op set and runs on the device (_ChainDeviceOps) and in torch (ChainTorchOps), so
host tests check the arithmetic against the host staging of the same block.
"""

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.spec_accept import TorchOps, _DeviceOps
from models.demos.blackhole.qwen36.tt.spec_step import draft_anchor, pack_u32_rows, pack_width

# Overlay rows (B words each).
CH_P0_PREV, CH_ACTIVE, CH_DELTA, CH_NUM_VALID, CH_DECLINED, CH_KEEP = range(6)
CHAIN_OVERLAY_ROWS = 6


def chain_overlay(p0_prev, active, deltas, num_valid, declined, keep):
    """int32 [CHAIN_OVERLAY_ROWS, pack_width(B)]: the host rows of a chained step. ``p0_prev`` [B]: the column-0
    position of each row in the predecessor (0 where the chained step has no row), ``active`` [B], ``deltas`` [B] (the
    slots' M-RoPE deltas, any sign), ``num_valid`` [B] (the chained step's valid drafts), ``declined`` and ``keep`` [B]
    (its propose decisions)."""
    rows = [p0_prev, [int(a) for a in active], deltas, num_valid, [int(d) for d in declined], [int(k) for k in keep]]
    B = len(rows[0])
    out = torch.zeros(CHAIN_OVERLAY_ROWS, pack_width(B), dtype=torch.int64)
    for i, r in enumerate(rows):
        out[i, :B] = torch.as_tensor(r, dtype=torch.int64).reshape(-1)
    if bool((out.abs() >= 2**24).any()):
        raise ValueError("a chain overlay word is at or above 2**24 in magnitude")
    return out.to(torch.int32)


def chain_graph(o, B, K, W, groups, nv_heads, ctrl_words):
    """The chain program's arithmetic over the op set ``o``: returns (step buffer [VERIFY_STEP_ROWS, W], [ctrl page
    [1, ctrl_words] per table group]) as float32."""
    T, R = K + 1, B * (K + 1)
    anchor = o.transpose(o.rows(o.pack, draft_anchor(K), draft_anchor(K) + 1, B))  # [B, 1]: u * T + m
    m = o.sub(anchor, o.user_rows)
    ov = o.transpose(o.rows(o.overlay, 0, CHAIN_OVERLAY_ROWS, B))  # [B, rows]

    def column(i):
        return o.slice(ov, (0, i), (B, i + 1))

    p0_prev, act, delta, nv, dec, keep = (column(i) for i in range(CHAIN_OVERLAY_ROWS))
    ids = o.lanes(o.ids, B, T)
    pick = o.eq(o.iota_t, m)  # [B, T]: 1 at column m, the count's one-hot
    tok0 = o.sum(o.mul(pick, ids))  # [B, 1]: the last committed token
    drafts = o.concat([o.draft(k, B) for k in range(K)], 1)  # [B, K]
    valid = o.lt(o.iota_k, nv)
    cols = o.add(o.mul(drafts, valid), o.mul(tok0, o.rsub1(valid)))
    tokens = o.mul(o.concat([tok0, cols], 1), act)
    p0 = o.mul(o.add(p0_prev, o.add(m, o.one)), act)  # column-0 position: p0_prev + count
    pos = o.add(p0, o.iota_t)
    rope = o.minimum(o.relu(o.add(pos, delta)), o.top)
    base = o.minimum(o.relu(o.add(p0, delta)), o.top)
    step = o.concat(
        [
            o.pad_w(o.reshape(tokens, (1, R)), R, W),
            o.pad_w(o.reshape(pos, (1, R)), R, W),
            o.pad_w(o.reshape(rope, (1, R)), R, W),
            o.pad_w(o.transpose(nv), B, W),
            o.pad_w(o.transpose(o.mul(nv, o.zero)), B, W),  # STEP_DRAWN: chained rows are greedy
            o.pad_w(o.transpose(p0), B, W),
            o.pad_w(o.transpose(base), B, W),
            o.pad_w(o.transpose(dec), B, W),
            o.pad_w(o.transpose(keep), B, W),
        ],
        0,
    )
    width = 2 * nv_heads * groups
    sel = None
    for c in range(T):
        term = o.mul(o.slice(pick, (0, c), (B, c + 1)), o.slice(o.cand, (c * B, 0), ((c + 1) * B, width)))
        sel = term if sel is None else o.add(sel, term)
    ctrls = []
    for g in range(groups):
        lo = 2 * nv_heads * g
        rec = o.reshape(o.slice(sel, (0, lo), (B, lo + nv_heads)), (1, B * nv_heads))
        hist = o.reshape(o.slice(sel, (0, lo + nv_heads), (B, lo + 2 * nv_heads)), (1, B * nv_heads))
        parts = [o.head, rec, hist] + ([o.tail] if o.tail is not None else [])
        ctrls.append(o.concat(parts, 1))
    return step, ctrls


class ChainTorchOps(TorchOps):
    """chain_graph's op set in torch over host tensors: ``ids`` [1, 1, 1, >= B*T], ``pack`` the drafter pack,
    ``overlay`` chain_overlay, ``cand`` [T*B, 2*Nv*groups], ``drafts`` K tensors of >= B ids."""

    def __init__(self, ids, pack, overlay, cand, drafts, B, K, table_rows, ctrl_words, nv_heads):
        T = K + 1
        f = lambda t: torch.as_tensor(t).to(torch.float32)
        self.ids, self.pack, self.overlay, self.cand = f(ids), f(pack), f(overlay), f(cand)
        self._drafts = [f(d).reshape(-1) for d in drafts]
        self.iota_t = f(torch.arange(T).reshape(1, T))
        self.iota_k = f(torch.arange(K).reshape(1, K))
        self.user_rows = f((torch.arange(B) * T).reshape(B, 1))
        self.top = f(torch.tensor([[int(table_rows) - 1]]))
        self.one, self.zero = f(torch.ones(1, 1)), f(torch.zeros(1, 1))
        self.head = f(torch.zeros(1, 1 + B))
        rest = int(ctrl_words) - 1 - B - 2 * B * int(nv_heads)
        self.tail = f(torch.zeros(1, rest)) if rest else None

    def draft(self, k, B):
        return self._drafts[k][:B].reshape(B, 1)

    relu = staticmethod(lambda x: torch.relu(x))


def chain_reference(ids, pack, overlay, cand, drafts, B, K, table_rows, ctrl_words, nv_heads, groups):
    """What ChainInputs.body writes, in host integers: (step buffer int32 [VERIFY_STEP_ROWS, W] as uint32 words,
    [ctrl int32 [1, ctrl_words]])."""
    o = ChainTorchOps(ids, pack, overlay, cand, drafts, B, K, table_rows, ctrl_words, nv_heads)
    step, ctrls = chain_graph(o, B, K, pack_width(B * (K + 1)), groups, nv_heads, ctrl_words)
    return step.to(torch.int64).to(torch.int32), [c.to(torch.int64).to(torch.int32) for c in ctrls]


class _ChainDeviceOps(_DeviceOps):
    """chain_graph's op set in ttnn (spec_accept._DeviceOps plus the chain's own inputs)."""

    def __init__(self, chain):
        names = ("ids", "pack", "overlay", "cand", "iota_t", "iota_k", "user_rows", "one", "zero", "head", "tail")
        self.__dict__.update({n: getattr(chain, n) for n in names})
        self.top = chain.c_top
        self._drafts = chain.drafts
        self.inputs = [t for t in (*(getattr(chain, n) for n in names), self.top, *self._drafts) if t is not None]
        self.owned = []

    def draft(self, k, B):
        return self._float(self._keep(ttnn.reshape(self._drafts[k], (B, 1))))

    def relu(self, x):
        return self._keep(ttnn.relu(x))


class ChainInputs:
    """The chain program of a chained MTP verify over B users and K drafts (module doc). Inputs: ``ids`` (the verify's
    argmax lanes [1, 1, 1, 32] uint32), ``pack`` (the drafter pack), ``drafts`` (K [1, 1, B, 1] uint32 drafter
    outputs; stand-ins until bind), and its own ``overlay`` (int32 [CHAIN_OVERLAY_ROWS, W]) and ``cand`` (float32
    [T*B, 2*Nv*groups]), staged per step. Outputs, written in place: ``step`` (the verify step buffer) and ``ctrls``
    (the GDN verify ctrl pages, int32)."""

    def __init__(self, mesh, B, K, table_rows, step, pack, ctrls, nv_heads):
        T = K + 1
        self.mesh, self.B, self.K, self.nv_heads = mesh, int(B), int(K), int(nv_heads)
        self.step, self.pack, self.ctrls = step, pack, list(ctrls)
        self.groups, self.ctrl_words = len(self.ctrls), int(self.ctrls[0].shape[-1])
        f32 = lambda t: self._device(torch.as_tensor(t, dtype=torch.float32), ttnn.float32, ttnn.TILE_LAYOUT)
        self.iota_t = f32(torch.arange(T).reshape(1, T))
        self.iota_k = f32(torch.arange(K).reshape(1, K))
        self.user_rows = f32((torch.arange(B) * T).reshape(B, 1))
        self.c_top = f32([[int(table_rows) - 1]])
        self.one, self.zero = f32([[1.0]]), f32([[0.0]])
        self.head = f32(torch.zeros(1, 1 + B))
        rest = self.ctrl_words - 1 - B - 2 * B * self.nv_heads
        if rest < 0:
            raise ValueError(f"a ctrl page of {self.ctrl_words} words holds no {B} rows of {nv_heads} heads")
        self.tail = f32(torch.zeros(1, rest)) if rest else None
        self.overlay = self._device(torch.zeros(CHAIN_OVERLAY_ROWS, pack_width(B), dtype=torch.int32), ttnn.int32)
        self.cand = f32(torch.zeros(T * B, 2 * self.nv_heads * self.groups))
        self._cand_staged = None
        self._refs = []  # host tensors of the copies of the last two steps
        self.ids = self.standin = self._device(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32)
        self.drafts = self.draft_standins = [
            self._device(torch.zeros(1, 1, B, 1, dtype=torch.int32), ttnn.uint32) for _ in range(K)
        ]
        consts = [self.iota_t, self.iota_k, self.user_rows, self.c_top, self.one, self.zero, self.head, self.tail]
        self.buffers = [t for t in (*consts, self.overlay, self.cand) if t is not None]

    def _device(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def bind(self, ids, drafts):
        """Read the verify trace's ids and the drafter trace's outputs (they exist only after their captures) instead of
        the stand-ins the programs were compiled on."""
        self.ids, self.drafts = ids, list(drafts)

    def stage(self, overlay, cand):
        """Host -> device copies of the overlay (every step) and the ctrl candidates (when they change)."""
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        refs = [ttnn.from_torch(torch.as_tensor(overlay).to(torch.int32), dtype=ttnn.int32, mesh_mapper=rep)]
        ttnn.copy_host_to_device_tensor(refs[0], self.overlay)
        cand = torch.as_tensor(cand).to(torch.float32)
        if self._cand_staged is None or not torch.equal(self._cand_staged, cand):
            refs.append(ttnn.from_torch(cand, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep))
            ttnn.copy_host_to_device_tensor(refs[-1], self.cand)
            self._cand_staged = cand.clone()
        # A copy can read its host tensor after it returns, while the step before still runs.
        self._refs = [*self._refs[-1:], refs]

    def body(self):
        """Trace-capturable: writes ``step`` and every ``ctrls`` page in place."""
        o = _ChainDeviceOps(self)
        W = int(self.step.shape[-1])
        step, ctrls = chain_graph(o, self.B, self.K, W, self.groups, self.nv_heads, self.ctrl_words)
        for out, dtype, dst in (
            (step, ttnn.uint32, self.step),
            *((c, ttnn.int32, d) for c, d in zip(ctrls, self.ctrls)),
        ):
            rm = ttnn.to_layout(o._keep(ttnn.typecast(out, dtype)), ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy(rm, dst)
            o._keep(rm)
        for t in o.owned:
            ttnn.deallocate(t)

    def release(self):
        for t in (*self.buffers, self.standin, *self.draft_standins):
            ttnn.deallocate(t)
        self.buffers, self.standin, self.draft_standins = [], None, []


def step_rows_reference(tokens, positions, rope, accept_rows, B, T):
    """The verify step buffer (spec_step) of B users x T rows that verify_enqueue stages: int32 [VERIFY_STEP_ROWS, W]."""
    from models.demos.blackhole.qwen36.tt.spec_step import (
        VERIFY_POSITIONS,
        VERIFY_ROPE,
        VERIFY_STEP_ROWS,
        VERIFY_TOKENS,
    )

    rows = [[0] * B for _ in range(VERIFY_STEP_ROWS)]
    rows[VERIFY_TOKENS], rows[VERIFY_POSITIONS], rows[VERIFY_ROPE] = tokens, positions, rope
    for r, v in accept_rows.items():
        rows[r] = v
    return pack_u32_rows(rows, width=pack_width(B * T))
