# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The inputs of a chained MTP verify, built on the device from the unread verify before it.

vllm-tt-plugin's ``supports_chained_spec_verify``: under asynchronous scheduling the runner submits verify n + 1
before it reads verify n, sending placeholders for every live row and verify n's ``VerifyOutput`` as
``spec_predecessor``. When verify n was a one-enqueue step, the device already holds everything verify n + 1 needs:

  * the verify ids of n (``Qwen36Model._vfy_ids_out``, the fused argmax lanes), and on a device-sampled step the
    draws of its sampled users (``spec_sample.VerifySampler.draws``), which replace those users' ids;
  * its accepted drafts m[u] (the drafter pack ``MTPSpecEngine._dr_pack``, whose anchor row the step trace wrote as
    u * T + m[u]; over the draws for a drawn user);
  * the K drafts its drafter made from that commit (``MTPSpecEngine._dr_outs``).

The chain program (ChainInputs.body, traced) runs before the verify trace of n + 1 and writes:

  * the verify step buffer of n + 1 (spec_step): tokens [ids[u, m], drafts...] (a column past the row's valid drafts
    repeats column 0, as sanitize_verify_block pads), KV positions p0 + m + 1 + j, RoPE table rows, and the acceptance
    rows the step trace of n + 1 reads (STEP_P0 and STEP_ROPE_BASE computed, the others from the host overlay);
  * one GDN verify ctrl page per MambaSpec table group: the start pages of each row's count c = m + 1, picked from
    candidate pages the host stages for every count (GdnBlockState.ctrl_candidates, from the plugin's
    ``MambaGroupState.start_blocks_by_count``);
  * with a device sampler, the sampler's seeds of n + 1: each drawing user's seeds of count c, picked from the seed
    candidates the host stages for every count (spec_sample.chained_sampling_plan), so a seeded user draws column j
    with the seed of its input position p0 + m + 1 + j, as an unchained verify at that position does.

What the host knows before n is read rides in the overlay (CH_* rows, one word per row u of n + 1): the row s of n
that u continues (CH_SRC: ``slot_remap[u]``, u when no row moved), s's column-0 position in n, the rows n + 1 runs,
their M-RoPE deltas, n + 1's valid drafts and propose decisions (decided for every count n can accept), and whether s's
ids are draws in n and u's in n + 1. Row u reads n's anchor, ids, draws and drafts at row s; every other input (the
overlay, the ctrl and seed candidates) the host stages for row u. Every value is an integer below 2**24 (token ids,
positions, table rows, page indices, device seeds) or -1 (SPEC_STEP_HOLD, a held row's ctrl word).

The chain program is one single-core kernel (kernels/chain_inputs.cpp; its arithmetic is chain_words in
kernels/spec_step_words.hpp). ``chain_graph`` states the same arithmetic as the float32 op graph the kernel replaced,
over an op set: ChainTorchOps runs it in torch, so host tests check it against the host staging of the same block, and
tests/spec_op_graphs.py keeps the ttnn op set, which the device tests run beside the kernel. Float32 is exact on
integers below 2**24, so the kernel's int32 words equal the graph's.
"""

from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.spec_accept import (
    IDS_LANES,
    KERNEL_CB,
    SLOT,
    TorchOps,
    accessor_args,
    check_same_accessor,
    one_core_program,
    slot_bytes,
)
from models.demos.blackhole.qwen36.tt.spec_step import VERIFY_STEP_ROWS, draft_anchor, pack_u32_rows, pack_width

CHAIN_INPUTS_KERNEL = str(Path(__file__).resolve().parent / "kernels" / "chain_inputs.cpp")

# Overlay rows (B words each).
CH_P0_PREV, CH_ACTIVE, CH_DELTA, CH_NUM_VALID, CH_DECLINED, CH_KEEP, CH_DRAWN_PREV, CH_DRAWN, CH_SRC = range(9)
CHAIN_OVERLAY_ROWS = 9


def chain_overlay(p0_prev, active, deltas, num_valid, declined, keep, drawn_prev=None, drawn=None, src=None):
    """int32 [CHAIN_OVERLAY_ROWS, pack_width(B)]: the host rows of a chained step, one word per row u of the chained
    step. ``p0_prev`` [B]: the column-0 position in the predecessor of the row u continues (0 where the chained step has
    no row), ``active`` [B], ``deltas`` [B] (the slots' M-RoPE deltas, any sign), ``num_valid`` [B] (the chained step's
    valid drafts), ``declined`` and ``keep`` [B] (its propose decisions), ``drawn_prev`` and ``drawn`` [B] (whether the
    ids of the predecessor row u continues are draws, and u's in the chained step; None: none), ``src`` [B] (the
    predecessor row each row continues, the runner's slot_remap; None: row u continues row u)."""
    B = len(p0_prev)
    drawn_prev = [0] * B if drawn_prev is None else drawn_prev
    drawn = [0] * B if drawn is None else drawn
    src = list(range(B)) if src is None else src
    rows = [p0_prev, [int(a) for a in active], deltas, num_valid, [int(d) for d in declined], [int(k) for k in keep]]
    rows += [[int(d) for d in drawn_prev], [int(d) for d in drawn], [int(s) for s in src]]
    out = torch.zeros(CHAIN_OVERLAY_ROWS, pack_width(B), dtype=torch.int64)
    for i, r in enumerate(rows):
        out[i, :B] = torch.as_tensor(r, dtype=torch.int64).reshape(-1)
    if bool((out.abs() >= 2**24).any()):
        raise ValueError("a chain overlay word is at or above 2**24 in magnitude")
    return out.to(torch.int32)


def chain_graph(o, B, K, W, groups, nv_heads, ctrl_words):
    """The chain program's arithmetic over the op set ``o``: returns (step buffer [VERIFY_STEP_ROWS, W], [ctrl page
    [1, ctrl_words] per table group], sampler seeds [1, B * T] or None without ``o.seed_cand``) as float32."""
    T, R = K + 1, B * (K + 1)
    ov = o.transpose(o.rows(o.overlay, 0, CHAIN_OVERLAY_ROWS, B))  # [B, rows]

    def column(i):
        return o.slice(ov, (0, i), (B, i + 1))

    p0_prev, act, delta, nv, dec, keep, drawn_prev, drawn, src = (column(i) for i in range(CHAIN_OVERLAY_ROWS))
    # [B, B]: 1 at (u, s) for the predecessor row s that row u continues; a source outside [0, B) has no 1, so the
    # gathered values of that row are 0.
    from_row = o.eq(src, o.iota_b)

    def gather(x):
        """[B, 1] over predecessor rows -> [B, 1] over rows: x[src[u]] (a single nonzero term, exact)."""
        return o.sum(o.mul(from_row, o.transpose(x)))

    def gather_columns(x, n):
        return o.concat([gather(o.slice(x, (0, j), (B, j + 1))) for j in range(n)], 1)

    anchor = gather(o.transpose(o.rows(o.pack, draft_anchor(K), draft_anchor(K) + 1, B)))  # [B, 1]: s * T + m
    m = o.sub(anchor, gather(o.user_rows))
    ids = gather_columns(o.lanes(o.ids, B, T), T)
    if o.draws is not None:
        ids = o.add(ids, o.mul(drawn_prev, o.sub(gather_columns(o.lanes(o.draws, B, T), T), ids)))
    pick = o.eq(o.iota_t, m)  # [B, T]: 1 at column m, the count's one-hot
    tok0 = o.sum(o.mul(pick, ids))  # [B, 1]: the last committed token
    drafts = o.concat([gather(o.draft(k, B)) for k in range(K)], 1)  # [B, K]
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
            o.pad_w(o.transpose(drawn), B, W),
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
    seeds = None
    if o.seed_cand is not None:
        for c in range(T):
            term = o.mul(o.slice(pick, (0, c), (B, c + 1)), o.slice(o.seed_cand, (c * B, 0), ((c + 1) * B, T)))
            seeds = term if seeds is None else o.add(seeds, term)
        seeds = o.reshape(seeds, (1, R))
    return step, ctrls, seeds


class ChainTorchOps(TorchOps):
    """chain_graph's op set in torch over host tensors: ``ids`` [1, 1, 1, >= B*T], ``pack`` the drafter pack,
    ``overlay`` chain_overlay, ``cand`` [T*B, 2*Nv*groups], ``drafts`` K tensors of >= B ids, ``draws`` [1, 1, 1,
    >= B*T] and ``seed_cand`` [T*B, T] (spec_sample.chained_sampling_plan), or None without a device sampler."""

    def __init__(
        self, ids, pack, overlay, cand, drafts, B, K, table_rows, ctrl_words, nv_heads, draws=None, seed_cand=None
    ):
        T = K + 1
        f = lambda t: torch.as_tensor(t).to(torch.float32)
        self.ids, self.pack, self.overlay, self.cand = f(ids), f(pack), f(overlay), f(cand)
        self.draws = None if draws is None else f(draws)
        self.seed_cand = None if seed_cand is None else f(seed_cand)
        self._drafts = [f(d).reshape(-1) for d in drafts]
        self.iota_t = f(torch.arange(T).reshape(1, T))
        self.iota_k = f(torch.arange(K).reshape(1, K))
        self.iota_b = f(torch.arange(B).reshape(1, B))
        self.user_rows = f((torch.arange(B) * T).reshape(B, 1))
        self.top = f(torch.tensor([[int(table_rows) - 1]]))
        self.one = f(torch.ones(1, 1))
        self.head = f(torch.zeros(1, 1 + B))
        rest = int(ctrl_words) - 1 - B - 2 * B * int(nv_heads)
        self.tail = f(torch.zeros(1, rest)) if rest else None

    def draft(self, k, B):
        return self._drafts[k][:B].reshape(B, 1)

    relu = staticmethod(lambda x: torch.relu(x))


def chain_reference(
    ids, pack, overlay, cand, drafts, B, K, table_rows, ctrl_words, nv_heads, groups, draws=None, seed_cand=None
):
    """What ChainInputs.body writes, in host integers: (step buffer int32 [VERIFY_STEP_ROWS, W] as uint32 words,
    [ctrl int32 [1, ctrl_words]], sampler seeds int64 [B * T] or None)."""
    o = ChainTorchOps(ids, pack, overlay, cand, drafts, B, K, table_rows, ctrl_words, nv_heads, draws, seed_cand)
    step, ctrls, seeds = chain_graph(o, B, K, pack_width(B * (K + 1)), groups, nv_heads, ctrl_words)
    return (
        step.to(torch.int64).to(torch.int32),
        [c.to(torch.int64).to(torch.int32) for c in ctrls],
        None if seeds is None else seeds.to(torch.int64).reshape(-1),
    )


def chain_kernel_l1_bytes(B, K, nv_heads, groups, step_words, ctrl_words, sampler):
    """spec_step_words.hpp ChainLayout(B, K, nv_heads, groups, step_words, ctrl_words, sampler).bytes."""
    T, R = K + 1, B * (K + 1)
    reads = 2 * slot_bytes(R) + slot_bytes(B) + CHAIN_OVERLAY_ROWS * slot_bytes(B) + K * B * SLOT
    reads += R * slot_bytes(2 * nv_heads * groups) + (R * slot_bytes(T) if sampler else 0)
    writes = VERIFY_STEP_ROWS * slot_bytes(step_words) + groups * slot_bytes(ctrl_words)
    return reads + writes + (slot_bytes(R) if sampler else 0)


def candidate_words(t):
    """The int32 words the chain kernel reads for host candidates ``t`` (ctrl or seed candidates, float32 integers):
    the value the op graph held in float32, as its float32 -> int32 typecast wrote it."""
    return torch.as_tensor(t).to(torch.float32).to(torch.int64).to(torch.int32)


class ChainInputs:
    """The chain program of a chained MTP verify over B users and K drafts (module doc): one
    kernels/chain_inputs.cpp program. Inputs: ``ids`` (the verify's argmax lanes [1, 1, 1, 32] uint32), ``pack`` (the
    drafter pack), ``drafts`` (K [1, 1, B, 1] uint32 drafter outputs; stand-ins until bind), with a device ``sampler``
    (spec_sample.VerifySampler) its ``draws``, and its own ``overlay`` (int32 [CHAIN_OVERLAY_ROWS, W]), ``cand``
    (int32 [T*B, 2*Nv*groups]) and, with a sampler, ``seed_cand`` (int32 [T*B, T]), staged per step. Outputs,
    written in place: ``step`` (the verify step buffer), ``ctrls`` (the GDN verify ctrl pages, int32) and, with a
    sampler, its ``seeds``."""

    def __init__(self, mesh, B, K, table_rows, step, pack, ctrls, nv_heads, sampler=None):
        T = K + 1
        if B * T > IDS_LANES:
            raise ValueError(f"{B} users x {T} columns exceed the {IDS_LANES} verify id lanes")
        self.mesh, self.B, self.K, self.nv_heads = mesh, int(B), int(K), int(nv_heads)
        self.top = int(table_rows) - 1
        self.step, self.pack, self.ctrls = step, pack, list(ctrls)
        self.draws = None if sampler is None else sampler.draws
        self.seeds = None if sampler is None else sampler.seeds
        self.groups, self.ctrl_words = len(self.ctrls), int(self.ctrls[0].shape[-1])
        self.step_words = int(step.shape[-1])
        if 1 + B + 2 * B * self.nv_heads > self.ctrl_words:
            raise ValueError(f"a ctrl page of {self.ctrl_words} words holds no {B} rows of {nv_heads} heads")
        for k, c in enumerate(self.ctrls[1:], 1):
            check_same_accessor(c, self.ctrls[0], f"ctrl page {k}")
        self.overlay = self._device(torch.zeros(CHAIN_OVERLAY_ROWS, pack_width(B), dtype=torch.int32), ttnn.int32)
        self.cand = self._device(torch.zeros(T * B, 2 * self.nv_heads * self.groups, dtype=torch.int32), ttnn.int32)
        self._cand_staged = None
        self.seed_cand = None if sampler is None else self._device(torch.zeros(T * B, T, dtype=torch.int32), ttnn.int32)
        self._seed_cand_staged = None
        self._refs = []  # host tensors of the copies of the last two steps
        self.ids = self.standin = self._device(torch.zeros(1, 1, 1, IDS_LANES, dtype=torch.int32), ttnn.uint32)
        self.drafts = self.draft_standins = [
            self._device(torch.zeros(1, 1, B, 1, dtype=torch.int32), ttnn.uint32) for _ in range(K)
        ]
        self.cb_bytes = SLOT + chain_kernel_l1_bytes(
            self.B, self.K, self.nv_heads, self.groups, self.step_words, self.ctrl_words, sampler is not None
        )
        self.buffers = [t for t in (self.overlay, self.cand, self.seed_cand) if t is not None]

    def _device(self, t, dtype):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def bind(self, ids, drafts):
        """Read the verify trace's ids and the drafter trace's outputs (they exist only after their captures) instead of
        the stand-ins the program was compiled on."""
        drafts = list(drafts)
        if len(drafts) != self.K:
            raise ValueError(f"{len(drafts)} draft id tensors for K = {self.K}")
        check_same_accessor(ids, self.standin, "the verify ids")
        for k, d in enumerate(drafts):
            check_same_accessor(d, self.draft_standins[0], f"draft ids {k}")
        self.ids, self.drafts = ids, drafts

    def stage(self, overlay, cand, seed_cand=None):
        """Host -> device copies of the overlay (every step), and of the ctrl candidates and, with a sampler, the seed
        candidates (None: zeros) when they change."""
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        refs = [ttnn.from_torch(torch.as_tensor(overlay).to(torch.int32), dtype=ttnn.int32, mesh_mapper=rep)]
        ttnn.copy_host_to_device_tensor(refs[0], self.overlay)
        T = self.K + 1
        staged = [(candidate_words(cand), self.cand, "_cand_staged")]
        if self.seed_cand is not None:
            seeds = torch.zeros(T * self.B, T) if seed_cand is None else seed_cand
            staged.append((candidate_words(seeds), self.seed_cand, "_seed_cand_staged"))
        for host, dev, held in staged:
            if getattr(self, held) is None or not torch.equal(getattr(self, held), host):
                refs.append(ttnn.from_torch(host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep))
                ttnn.copy_host_to_device_tensor(refs[-1], dev)
                setattr(self, held, host.clone())
        # A copy can read its host tensor after it returns, while the step before still runs.
        self._refs = [*self._refs[-1:], refs]

    def program(self):
        """The kernel's program over the bound tensors; without a device sampler the ids stand in for the draws, the
        seed candidates and the seeds, which it then never touches."""
        sampler = self.seeds is not None
        draws, seed_cand, seeds = (self.draws, self.seed_cand, self.seeds) if sampler else (self.ids,) * 3
        draft_page = int(self.drafts[0].buffer_aligned_page_size())
        if not 4 <= draft_page <= SLOT:
            raise ValueError(f"a draft id page of {draft_page} B does not fit one {SLOT} B slot")
        cta = [self.B, self.K, self.nv_heads, self.groups, self.step_words, self.ctrl_words, self.top, int(sampler)]
        cta += [draft_page, KERNEL_CB, self.cb_bytes]
        for t in (self.ids, draws, self.pack, self.overlay, self.cand, seed_cand, self.drafts[0], self.step):
            cta += accessor_args(t)
        cta += accessor_args(self.ctrls[0]) + accessor_args(seeds)
        rt = [t.buffer_address() for t in (self.ids, draws, self.pack, self.overlay, self.cand, seed_cand, self.step)]
        rt += [seeds.buffer_address()] + [d.buffer_address() for d in self.drafts]
        rt += [c.buffer_address() for c in self.ctrls]
        return one_core_program(CHAIN_INPUTS_KERNEL, cta, rt, self.cb_bytes)

    def body(self):
        """Trace-capturable: writes ``step``, every ``ctrls`` page and, with a sampler, its ``seeds`` in place."""
        inputs = (self.ids, self.draws, self.pack, self.overlay, self.cand, self.seed_cand, *self.drafts)
        outputs = (self.step, *self.ctrls, self.seeds)
        ttnn.generic_op([t for t in (*inputs, *outputs) if t is not None], self.program())

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
