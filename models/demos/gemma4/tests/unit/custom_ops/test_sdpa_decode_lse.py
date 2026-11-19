# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device checks of ``custom_ops/sdpa_decode_lse`` (the stock paged decode attention + the row statistics), on the
Gemma 4 global-layer shape (4 local q heads, one KV head, head_dim 512, a bfloat8_b paged cache, the model's
program config):

* O against the stock ``paged_scaled_dot_product_attention_decode`` with the same program and compute configs
  (bitwise: the same kernels on the same core assignment), and against torch (PCC);
* the LSE (``scale * m + ln(l)`` from the stats output) against torch's logsumexp of the scaled scores;
* the pair merge: two calls over the two halves of every row's keys (the second through a page table shifted by
  ``mid / block`` blocks and ``cur_pos - mid``), merged with the LSE formula, against one full call (PCC);
* the program-cache repeat (bitwise) and the out-of-contract refusals;
* the pair split on a (1, 2) mesh (``pair_split=True``, the same inputs replicated on both chips): each chip's
  half (its O and LSE against torch over that half's keys, the empty-half sentinel), the two merged against the
  single full call and torch, the model's device merge (``pair_merge``: the gathered LSEs and the selector) per chip
  and summed, the program-cache repeat;
* the group split (``GEMMA4_SDPA_GROUP_SPLIT``) on a 4-chip line (``-k "group and 1x4"`` / ``"group and 4x1"``):
  ``split_parts=4`` on the (1, 4) row (the 12B: one KV head on 4 chips, 16 q heads) and ``split_parts=2`` over axis 0
  of the (4, 1) column (the 26B-A4B: two KV heads on chip pairs, 8 q heads) at 32K and 86K keys -- each part against
  torch over its keys, the parts merged against torch (PCC >= 0.999 with fp32 dest) and the single full call, and
  the model's device merge (``group_merge`` / ``pair_merge`` over the TP axis) summed per group;
* ``sdpa_prefill_splitk`` (inside the pair-merge cases): the pair-sharded prefill's three-way merge, the served
  causal call against torch and the stock chunked SDPA, and a served-like shape (several Q chunks, the automatic
  split count): the plan covers every (head, q chunk, k chunk) once, the merged statistics' LSE against torch, a
  fixed split count at two chunk starts as one program (a cache hit).

Torch thresholds: the ``fp32_dest`` compute config must beat ``PCC_MIN`` against the fp32 torch golden; the
``stock`` config (bf16 dest, the op default) measures 0.9974-0.9986 against it on these random inputs, so there the
contract is the bitwise match with the stock op and a merge that loses at most 1e-3 PCC against the full call.

A CI subset runs ungated (the 12B unit leg, bh_quietbox_2; ``CI_SUBSET`` below): the single-chip calls at B = 1 and
B = 4 over 4K keys in both compute configs, the out-of-contract refusals, and the 12B's group split on the (1, 4) row
at 32K keys. Everything else is gated (the 64K cases allocate ~0.3 GB of cache per row and the B=4 case a 1 GB torch
mirror):

    GEMMA4_SDPA_LSE_TEST=1 pytest models/demos/gemma4/tests/unit/custom_ops/test_sdpa_decode_lse.py

The pair-split cases (``-k 1x2``) open a 2-chip mesh: run them with a 2-chip system mesh (on a Galaxy
``MESH_DEVICE=P150x2``, ``TT_MESH_GRAPH_DESC_PATH`` a (1, 2) Blackhole descriptor and ``TT_VISIBLE_DEVICES`` two
adjacent chips); the group cases (``-k group``) a 4-chip line (``MESH_DEVICE=P150x4``, ``TT_VISIBLE_DEVICES`` four
chips of one line); the others with a single chip (``-k single-chip``).
"""

import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.custom_ops.sdpa_decode_lse import SdpaDecodeLse
from models.demos.gemma4.custom_ops.sdpa_decode_lse import planner as P
from models.demos.gemma4.custom_ops.sdpa_prefill_splitk.op import SdpaPrefillSplitK
from models.demos.gemma4.tt.attention.prefill import _lse_column, pair_prefill_contribution

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
# the pair: one (1, 2) mesh, fabric as test_factory.parametrize_mesh_with_fabric opens multi-chip meshes (the op
# itself sends nothing between the chips)
PAIR_MESH = [pytest.param((1, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="1x2")]
GATE_ENV = "GEMMA4_SDPA_LSE_TEST"
BLOCK = 64  # the served vLLM block
D = 512
PCC_MIN = 0.999
PAIR_PCC_MIN = 0.9999  # the pair split's merge against the single full call
K_CHUNK = 64  # the model's decode k chunk on a bfp8 cache
Q_STD = 0.125  # scores ~ N(0, (Q_STD * sqrt(D))^2) = N(0, 2.8^2) against unit-variance keys at scale 1
SCALE = 1.0  # the model's scale

gated = pytest.mark.skipif(
    not os.environ.get(GATE_ENV), reason=f"device checks of sdpa_decode_lse: set {GATE_ENV}=1 to run them"
)
# The cases that run without the gate (the CI subset): (B, Hq, pos, cfg) of the single-chip check, and the group split
# (mesh, parts, axis, Hq, B, pos, cfg) of the 12B on the (1, 4) row
CI_SUBSET = {(1, 4, 4095, "stock"), (1, 4, 4095, "fp32_dest"), (4, 4, 4095, "stock"), (4, 4, 4095, "fp32_dest")}
GROUP_CI_SUBSET = {((1, 4), 4, 1, 16, 1, 32767, "fp32_dest")}


def _subset_marks(key, subset):
    """No marks for a CI-subset case, else the gate."""
    return () if key in subset else (gated,)


def _owned_read_tables(page_table, cur_pos, block_size):
    """The ``owned`` mode's per-half read tables over one unsharded cache: block-parity ownership (logical block ``b``
    on pair half ``b % 2``), each half's owned blocks among a row's positions ``0 .. cur_pos // block_size`` in
    position order (padding 0), and the last valid index of that compacted sequence (``-1`` when the half owns none
    of the row's blocks or the row is inactive). Returns ``(tables [2, B, W] int32, cur_pos_h [2, B] int32)``."""
    pt = page_table.to(torch.int64)
    B, W = pt.shape
    bs = int(block_size)
    tables = torch.zeros((2, B, W), dtype=torch.int64)
    pos_h = torch.full((2, B), -1, dtype=torch.int64)
    for b in range(B):
        cp = int(cur_pos[b])
        if cp < 0:
            continue
        last = cp // bs
        tail = cp % bs + 1
        for h in (0, 1):
            owned = [j for j in range(last + 1) if int(pt[b, j]) % 2 == h]
            if not owned:
                continue
            tables[h, b, : len(owned)] = pt[b, owned]
            full = len(owned) - (1 if owned[-1] == last else 0)
            pos_h[h, b] = full * bs + (tail if owned[-1] == last else 0) - 1
    return tables.to(torch.int32), pos_h.to(torch.int32)


def _check_splitk_served_shape(mesh_device, case, row, length, Hq, ckc):
    """SdpaPrefillSplitK on a served-like call over ``row``'s first ``length`` cached keys: several Q chunks
    (Sq 4096 after a 16K prefix when the row is that long, else 2048 after 2048), the automatic split count, so a core
    runs several pieces and the causal diagonal sits past the first Q chunk; the plan covers every (head, q chunk, k
    chunk) once; the output against torch and the stock chunked SDPA, the merged statistics' LSE against torch; then
    a fixed split count at two chunk starts with the same row count, the second a program-cache hit."""
    from models.demos.gemma4.custom_ops.grid import compute_grid
    from models.demos.gemma4.custom_ops.sdpa_prefill_splitk.op import plan_pieces

    chunk = 128
    cs, sq = (16384, 4096) if length >= 16384 + 4096 else (2048, 2048)
    gx, gy = compute_grid(mesh_device)

    def covered(start, splits=None):
        plan = plan_pieces(Hq, sq // chunk, start // chunk, gx * gy, splits=splits)
        seen = {}
        for pieces in plan.cores:
            for h, qi, lo, hi, _ in pieces:
                seen.setdefault((h, qi), []).append((lo, hi))
        for h in range(Hq):
            for qi in range(sq // chunk):
                spans = sorted(seen.pop((h, qi)))
                assert spans[0][0] == 0 and spans[-1][1] == start // chunk + qi + 1, (h, qi, spans)
                assert all(a[1] == b[0] for a, b in zip(spans, spans[1:])), (h, qi, spans)
        assert not seen, f"pieces outside the call: {sorted(seen)[:4]}"
        return plan

    plan = covered(cs)
    fixed = covered(cs, splits=4)
    assert max(len(c) for c in fixed.cores) > 1, "4 splits: expected several pieces on a core"

    pos_k = torch.arange(cs + sq)
    keys = case.k_ref[case.table[row, pos_k // BLOCK].long(), 0, pos_k % BLOCK].float()
    values = case.v_ref[case.table[row, pos_k // BLOCK].long(), 0, pos_k % BLOCK].float()
    q = (torch.randn(1, Hq, sq, D) * Q_STD).to(torch.bfloat16)
    tt_q = ttnn.from_torch(q, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, **case.rep)
    _, tt_row = case.tensors(torch.tensor([length - 1]), case.table[row : row + 1].contiguous())
    kw = dict(num_kv_heads=1, block_size=BLOCK, compute_kernel_config=ckc)

    def reference(start):
        k, v = keys[: start + sq], values[: start + sq]
        s = (q[0].float() @ k.T) * SCALE
        visible = torch.arange(start + sq)[None, :] <= (torch.arange(sq)[:, None] + start)
        s = s.masked_fill(~visible, float("-inf"))
        return torch.softmax(s, dim=-1) @ v, torch.logsumexp(s, dim=-1)

    out, stats = SdpaPrefillSplitK.op(tt_q, case.k_cache, case.v_cache, tt_row, cs, merge_stats=True, **kw)
    ref, lse_ref = reference(cs)
    got = case.host(out)[0].float()
    st = case.host(stats)[0].float()
    lse = SCALE * st[..., 0] + torch.log(st[..., P.TILE])
    stock = ttnn.transformer.chunked_scaled_dot_product_attention(
        tt_q,
        case.k_cache,
        case.v_cache,
        tt_row,
        chunk_start_idx=cs,
        scale=SCALE,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=False,
        ),
        compute_kernel_config=ckc,
    )
    pcc, pcc_stock = _pcc(got, ref), _pcc(case.host(stock)[0].float(), ref)
    lse_err = (lse - lse_ref).abs().max().item()
    logger.info(
        f"split-K served shape: {Hq} heads x {sq} rows after {cs}, {plan.splits} splits, <= "
        f"{max(len(c) for c in plan.cores)} pieces per core: pcc {pcc:.6f} (stock {pcc_stock:.6f}), "
        f"max |lse - torch| {lse_err:.4f}"
    )
    assert pcc >= PCC_MIN and pcc >= pcc_stock - 1e-4, f"split-K {pcc:.6f} vs stock {pcc_stock:.6f}"
    assert lse_err <= 0.05 + 0.01 * lse_ref.abs().max().item(), f"merged lse off by {lse_err:.4f}"
    for t in (out, stats, stock):
        t.deallocate(True)

    # a fixed split count at two chunk starts with the same rows: one program, the second call a cache hit
    entries = None
    for start in (cs, cs - 512):
        res = SdpaPrefillSplitK.op(tt_q, case.k_cache, case.v_cache, tt_row, start, splits=4, **kw)
        pcc_s = _pcc(case.host(res)[0].float(), reference(start)[0])
        res.deallocate(True)
        assert pcc_s >= PCC_MIN, f"split-K splits=4 at {start}: pcc {pcc_s:.6f}"
        n = mesh_device.num_program_cache_entries()
        assert entries is None or n == entries, f"chunk start {start}: {n - entries} new program(s), expected a hit"
        entries = n
    tt_q.deallocate(True)
    tt_row.deallocate(True)


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _assert_merged_vs_torch(cfg, pcc_merged, pcc_full):
    """A pair merge against the fp32 torch golden: above PCC_MIN with fp32 dest; with the stock config no worse than
    the single full call by more than 1e-3 (``pcc_full`` = pcc(full, torch))."""
    if cfg == "fp32_dest":
        assert pcc_merged > PCC_MIN, f"merged vs torch pcc {pcc_merged:.5f}"
    else:  # no absolute floor: the bf16-dest stock op itself measures 0.9974-0.9986 vs fp32 torch on these inputs
        assert pcc_merged >= pcc_full - 1e-3, f"merged vs torch pcc {pcc_merged:.6f}, full call {pcc_full:.6f}"


def _positions(B, pos):
    """Row 0 at ``pos``; with B >= 4 a half-length row, a row just short of ``pos`` and a skipped (-1) row, and past
    four rows (the packed verify's record rows) rows spread below ``pos``, the last mirroring row 0 (a padding row)."""
    if B == 1:
        return torch.tensor([pos])
    head = [pos, pos // 2, max(pos - 33, 0), -1]
    if B == 4:
        return torch.tensor(head)
    rest = [max(pos - 97 * (i + 1), 0) for i in range(B - 5)] + [pos]
    return torch.tensor(head + rest)


def _program_config(mesh_device):
    """The model's decode SDPA program config on a bfp8 cache (tt/attention/decode.py): the device grid, q chunk 32,
    k chunk 64, exact exp, the default max_cores_per_head_batch (16)."""
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        q_chunk_size=32,
        k_chunk_size=K_CHUNK,
        exp_approx_mode=False,
    )


def _compute_config(mesh_device, name):
    """``stock``: None (the op default the model's global decode call runs: HiFi2, approx, fp32 dest off);
    ``fp32_dest``: HiFi2 with fp32 dest accumulation (what the packed verify passes on one chip)."""
    if name == "stock":
        return None
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


class _Case:
    """A paged bfloat8_b K / V cache holding ``B`` rows of ``ceil((max_pos + 1) / BLOCK)`` blocks each (a random
    permutation of the physical blocks), its torch mirror (what the device reads, after the bfp8 rounding) and Q.
    On a multi-chip mesh every input is replicated and the host copies are device 0's."""

    def __init__(self, mesh_device, B, Hq, max_pos, seed=0):
        torch.manual_seed(seed)
        self.device, self.B, self.Hq = mesh_device, B, Hq
        self.n_dev = mesh_device.get_num_devices()
        rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)} if self.n_dev > 1 else {}
        self.rep = rep
        self.width = math.ceil((max_pos + 1) / BLOCK)
        self.num_blocks = B * self.width
        shape = (self.num_blocks, 1, BLOCK, D)
        self.k_cache = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **rep,
        )
        self.v_cache = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **rep,
        )
        self.k_ref = self.host(self.k_cache).float()
        self.v_ref = self.host(self.v_cache).float()
        g = torch.Generator().manual_seed(seed + 1)
        self.table = torch.randperm(self.num_blocks, generator=g).reshape(B, self.width).to(torch.int32)
        self.q = (torch.randn(1, B, Hq, D) * Q_STD).to(torch.bfloat16)
        if B > 4:  # the last row is a padding row of the packed verify: row 0's table and q (_positions: its position)
            self.table[B - 1] = self.table[0]
            self.q[:, B - 1] = self.q[:, 0]
        self.tt_q = ttnn.from_torch(self.q, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, **rep)

    def host(self, t):
        """A torch copy of a device tensor (device 0's on a multi-chip mesh)."""
        return ttnn.to_torch(t) if self.n_dev == 1 else ttnn.to_torch(ttnn.get_device_tensors(t)[0])

    def tensors(self, cur_pos, table):
        tt_pos = ttnn.from_torch(
            cur_pos.to(torch.int32), device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, **self.rep
        )
        tt_tab = ttnn.from_torch(table, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, **self.rep)
        return tt_pos, tt_tab

    def run(self, cur_pos, table, program_config, compute_kernel_config, owned=False):
        tt_pos, tt_tab = self.tensors(cur_pos, table)
        out, stats = SdpaDecodeLse.op(
            self.tt_q,
            self.k_cache,
            self.v_cache,
            tt_pos,
            tt_tab,
            scale=SCALE,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            owned=owned,
        )
        o = self.host(out)[0, :, : self.Hq].float()
        lse = SdpaDecodeLse.lse_from_stats(self.host(stats), self.Hq, SCALE)[0]
        st = self.host(stats).float()
        for t in (out, stats, tt_pos, tt_tab):
            t.deallocate(True)
        return o, lse, st

    def stock(self, cur_pos, table, program_config, compute_kernel_config):
        tt_pos, tt_tab = self.tensors(cur_pos, table)
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            self.tt_q,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=tt_pos,
            page_table_tensor=tt_tab,
            scale=SCALE,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
        o = self.host(out)[0, :, : self.Hq].float()
        for t in (out, tt_pos, tt_tab):
            t.deallocate(True)
        return o

    def golden(self, cur_pos, table):
        out, lse = SdpaDecodeLse.golden(self.q, self.k_ref, self.v_ref, table, cur_pos, scale=SCALE)
        return out[0], lse[0]

    def golden_range(self, b, lo, hi):
        """fp32 torch attention of row b's heads over the keys ``[lo, hi)`` (one pair half): ``(out [Hq, D], lse
        [Hq])``."""
        p = torch.arange(lo, hi)
        blocks, offs = self.table[b, p // BLOCK].long(), p % BLOCK
        kk, vv = self.k_ref[blocks, 0, offs], self.v_ref[blocks, 0, offs]
        s = self.q[0, b].float() @ kk.T * SCALE
        return torch.softmax(s, dim=-1) @ vv, torch.logsumexp(s, dim=-1)

    def raw_max(self, cur_pos, b):
        """Row b's max raw score per head (the stats' m before the device's rounding)."""
        p = torch.arange(int(cur_pos[b]) + 1)
        kk = self.k_ref[self.table[b, p // BLOCK].long(), 0, p % BLOCK]
        return (self.q[0, b].float() @ kk.T).amax(-1)

    def shifted_table(self, shift_blocks):
        """Every row's table advanced by ``shift_blocks`` (the second half's view), padded with block 0."""
        t = torch.zeros_like(self.table)
        t[:, : self.width - shift_blocks] = self.table[:, shift_blocks:]
        return t


def _cases():
    keys = []
    for B in (1, 4):
        for pos in (100, 4095, 65536):
            for cfg in ("stock", "fp32_dest"):
                keys.append((B, 4, pos, cfg))
    # 32 heads: full 32x32 tiles everywhere (the half-tile path is <= 16 heads)
    keys += [(1, 32, 4095, "stock"), (4, 32, 4095, "fp32_dest")]
    # the packed verify's prefix call (GEMMA4_SPEC_VERIFY_PREFIX_LSE): H * P q rows of several record rows, a half
    # tile at 16 rows, a partial full tile at 24
    keys += [(B, Hq, 4095, "stock") for B, Hq in ((5, 16), (5, 24), (16, 16))]
    return [pytest.param(*k, id=f"B{k[0]}-Hq{k[1]}-pos{k[2]}-{k[3]}", marks=_subset_marks(k, CI_SUBSET)) for k in keys]


@pytest.mark.parametrize("B, Hq, pos, cfg", _cases())
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_sdpa_decode_lse_matches_stock_and_torch(mesh_device, device_params, B, Hq, pos, cfg, reset_seeds):
    case = _Case(mesh_device, B, Hq, pos)
    pc, ckc = _program_config(mesh_device), _compute_config(mesh_device, cfg)
    cur_pos = _positions(B, pos)
    live = [b for b in range(B) if cur_pos[b] >= 0]
    tt_pos, tt_tab = case.tensors(cur_pos, case.table)
    args = (case.tt_q, case.k_cache, case.v_cache, tt_pos, tt_tab)
    assert SdpaDecodeLse.unsupported_reason(*args, scale=SCALE, program_config=pc, compute_kernel_config=ckc) is None
    plan = SdpaDecodeLse.plan(*args, scale=SCALE, program_config=pc, compute_kernel_config=ckc)
    logger.info(f"{plan.describe()}")
    tt_pos.deallocate(True)
    tt_tab.deallocate(True)

    out, lse, stats = case.run(cur_pos, case.table, pc, ckc)
    ref_out, ref_lse = case.golden(cur_pos, case.table)
    stock = case.stock(cur_pos, case.table, pc, ckc)

    # O: the stock kernels on the stock core assignment -> the same bits as the stock op
    diff = (out[live] - stock[live]).abs().max().item()
    pcc = _pcc(out[live], ref_out[live])
    logger.info(f"B={B} Hq={Hq} pos={pos} {cfg}: max |O - stock| {diff:.3e}, pcc(O, torch) {pcc:.6f}")
    assert torch.equal(out[live], stock[live]), f"O differs from the stock op (max abs {diff:.3e})"
    if cfg == "fp32_dest":  # stock: no torch floor -- the bf16-dest stock op itself measures 0.9974-0.9986 here
        assert pcc > PCC_MIN, f"pcc(O, torch) {pcc:.5f}"

    # LSE = scale * m + ln(l) against torch's logsumexp of the scaled scores
    err = (lse[live] - ref_lse[live]).abs()
    tol = 0.05 + 0.01 * ref_lse[live].abs()
    m_ref = torch.stack([case.raw_max(cur_pos, b) for b in live])
    m_err = (stats[0, live, :Hq, 0] - m_ref).abs().max().item()
    logger.info(
        f"B={B} Hq={Hq} pos={pos} {cfg}: max |lse - torch| {err.max().item():.4f} (lse range "
        f"{ref_lse[live].min().item():.2f}..{ref_lse[live].max().item():.2f}), max |m - torch max| {m_err:.4f}"
    )
    assert torch.all(err <= tol), f"lse off by {err.max().item():.4f}"

    # the program-cache repeat (fresh output buffers): the same bits
    out2, lse2, _ = case.run(cur_pos, case.table, pc, ckc)
    assert torch.equal(out2[live], out[live]) and torch.equal(lse2[live], lse[live])


@gated
@pytest.mark.parametrize("B, pos", [(1, 100), (1, 4095), (1, 65536), (4, 4095), (4, 65536)])
@pytest.mark.parametrize("cfg", ["stock", "fp32_dest"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_sdpa_decode_lse_pair_merge_matches_full(mesh_device, device_params, B, pos, cfg, reset_seeds):
    """The pair split on one chip: half A attends keys [0, mid) (same table, cur_pos = mid - 1), half B keys
    [mid, cur_pos] (table shifted by mid / BLOCK blocks, cur_pos - mid); the LSE merge of the two equals the full
    call. A row shorter than mid has no B half (cur_pos -1, weight 0). Then the owned split of the pair-sharded
    global KV (``tt/kv_pair_shard.py``): each half attends the blocks whose id has its parity through a compacted
    table and its own cur_pos (``owned=True``); a half owning none of a row's blocks writes the empty-half
    sentinel itself, so this merge has no host-side masking."""
    Hq = 4
    case = _Case(mesh_device, B, Hq, pos, seed=3)
    pc, ckc = _program_config(mesh_device), _compute_config(mesh_device, cfg)
    cur_pos = _positions(B, pos)
    live = [b for b in range(B) if cur_pos[b] >= 0]
    mid = max(BLOCK, ((pos + 1) // 2) // BLOCK * BLOCK)
    pos_a = torch.where(cur_pos >= mid, torch.full_like(cur_pos, mid - 1), cur_pos)
    pos_b = torch.where(cur_pos >= mid, cur_pos - mid, torch.full_like(cur_pos, -1))

    full, lse_full, _ = case.run(cur_pos, case.table, pc, ckc)
    out_a, lse_a, _ = case.run(pos_a, case.table, pc, ckc)
    out_b, lse_b, _ = case.run(pos_b, case.shifted_table(mid // BLOCK), pc, ckc)
    lse_b = torch.where((pos_b >= 0)[:, None], lse_b, torch.full_like(lse_b, float("-inf")))
    merged, lse_m = SdpaDecodeLse.merge_torch(out_a, lse_a, out_b, lse_b)
    ref_out, ref_lse = case.golden(cur_pos, case.table)

    pcc_full = _pcc(merged[live], full[live])
    pcc_ref = _pcc(merged[live], ref_out[live])
    pcc_full_ref = _pcc(full[live], ref_out[live])
    lse_err = (lse_m[live] - lse_full[live]).abs().max().item()
    logger.info(
        f"B={B} pos={pos} mid={mid} {cfg}: pcc(merged, full) {pcc_full:.6f}, pcc(merged, torch) {pcc_ref:.6f}, "
        f"pcc(full, torch) {pcc_full_ref:.6f}, max |lse_merged - lse_full| {lse_err:.4f}"
    )
    assert pcc_full >= PCC_MIN, f"merged vs full pcc {pcc_full:.5f}"
    _assert_merged_vs_torch(cfg, pcc_ref, pcc_full_ref)
    assert lse_err <= 0.05 + 0.01 * lse_full[live].abs().max().item()

    tables, pos_h = _owned_read_tables(case.table, cur_pos, BLOCK)
    out_0, lse_0, _ = case.run(pos_h[0], tables[0], pc, ckc, owned=True)
    out_1, lse_1, _ = case.run(pos_h[1], tables[1], pc, ckc, owned=True)
    owned, lse_o = SdpaDecodeLse.merge_torch(out_0, lse_0, out_1, lse_1)
    pcc_owned = _pcc(owned[live], full[live])
    lse_err_o = (lse_o[live] - lse_full[live]).abs().max().item()
    logger.info(
        f"B={B} pos={pos} {cfg} owned split: pcc(merged, full) {pcc_owned:.6f}, "
        f"max |lse_merged - lse_full| {lse_err_o:.4f}, owned positions {pos_h.tolist()}"
    )
    assert pcc_owned >= PCC_MIN, f"owned merged vs full pcc {pcc_owned:.5f}"
    assert lse_err_o <= 0.05 + 0.01 * lse_full[live].abs().max().item()

    # one-block rows: the non-owner half writes the sentinel on a live row and the merge takes the owner alone
    short = torch.where(cur_pos >= 0, torch.clamp(cur_pos, max=BLOCK // 2), cur_pos)
    full_s, lse_full_s, _ = case.run(short, case.table, pc, ckc)
    tables_s, pos_s = _owned_read_tables(case.table, short, BLOCK)
    assert all((int(pos_s[0, b]) < 0) != (int(pos_s[1, b]) < 0) for b in live), pos_s
    o0, l0, _ = case.run(pos_s[0], tables_s[0], pc, ckc, owned=True)
    o1, l1, _ = case.run(pos_s[1], tables_s[1], pc, ckc, owned=True)
    owned_s, lse_os = SdpaDecodeLse.merge_torch(o0, l0, o1, l1)
    pcc_s = _pcc(owned_s[live], full_s[live])
    logger.info(f"B={B} {cfg} owned split, one-block rows {pos_s.tolist()}: pcc(merged, full) {pcc_s:.6f}")
    assert pcc_s >= PCC_MIN, f"owned one-block merged vs full pcc {pcc_s:.5f}"
    assert (lse_os[live] - lse_full_s[live]).abs().max().item() <= 0.05 + 0.01 * lse_full_s[live].abs().max().item()

    # half-block split (the layout the pair-sharded global KV stores): logical blocks of 2 * BLOCK tokens, half h
    # of block i (= chunk 2 i + h) in the slot half_table[b, i] of half h's own cache; each half runs with the
    # global cur_pos and derives its own, and the merge of the two equals the full call
    hw = (case.width + 1) // 2
    half_table = case.table[:, 0 : 2 * hw : 2].contiguous()
    outs, lses = [], []
    # prefill queries (SdpaPrefillSplitK) over the last row: 128 rows past its complete 2 * BLOCK-token logical blocks
    row = B - 1
    n_blk = (int(cur_pos[row]) + 1) // (2 * BLOCK)
    pf_q = (torch.randn(1, Hq, 128, D) * Q_STD).to(torch.bfloat16)
    pf_tt_q = ttnn.from_torch(pf_q, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, **case.rep)
    pf_ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    pf_halves = []
    for h in (0, 1):
        kh, vh = torch.zeros_like(case.k_ref), torch.zeros_like(case.v_ref)
        for b in range(B):
            for i in range(hw):
                c = 2 * i + h
                if c < case.width:
                    kh[half_table[b, i]] = case.k_ref[case.table[b, c]]
                    vh[half_table[b, i]] = case.v_ref[case.table[b, c]]
        put = dict(
            device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tk, tv = ttnn.from_torch(kh.to(torch.bfloat16), **put), ttnn.from_torch(vh.to(torch.bfloat16), **put)
        tt_pos, tt_tab = case.tensors(cur_pos, half_table)
        out, stats = SdpaDecodeLse.op(
            case.tt_q,
            tk,
            tv,
            tt_pos,
            tt_tab,
            scale=SCALE,
            program_config=pc,
            compute_kernel_config=ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            half_block=True,
            half=h,
        )
        outs.append(case.host(out)[0, :, : case.Hq].float())
        lses.append(SdpaDecodeLse.lse_from_stats(case.host(stats), case.Hq, SCALE)[0])
        # the pair-split prefill (attention.prefill.pair_split_prefill_attention) reads this half the same way:
        # the prefill queries over the half's BLOCK-token pages of the row's complete logical blocks, unmasked,
        # merged over the splits with statistics
        if n_blk >= 2:
            _, tt_row = case.tensors(cur_pos, half_table[row : row + 1].contiguous())
            pf_halves.append(
                SdpaPrefillSplitK.op(
                    pf_tt_q,
                    tk,
                    tv,
                    tt_row,
                    0,
                    num_kv_heads=1,
                    block_size=BLOCK,
                    k_chunk_size=BLOCK,
                    compute_kernel_config=pf_ckc,
                    causal=False,
                    k_chunks=n_blk,
                    merge_stats=True,
                )
            )
            tt_row.deallocate(True)
        for t in (out, stats, tt_pos, tt_tab, tk, tv):
            t.deallocate(True)
    half_merged, lse_hm = SdpaDecodeLse.merge_torch(outs[0], lses[0], outs[1], lses[1])
    pcc_hb = _pcc(half_merged[live], full[live])
    lse_err_hb = (lse_hm[live] - lse_full[live]).abs().max().item()
    logger.info(f"B={B} pos={pos} {cfg} half-block split: pcc(merged, full) {pcc_hb:.6f}, max |dlse| {lse_err_hb:.4f}")
    assert pcc_hb >= PCC_MIN, f"half-block merged vs full pcc {pcc_hb:.5f}"
    assert lse_err_hb <= 0.05 + 0.01 * lse_full[live].abs().max().item()

    if n_blk >= 2:
        prefix = n_blk * 2 * BLOCK
        pos_k = torch.arange(prefix)
        k_pre = case.k_ref[case.table[row, pos_k // BLOCK].long(), 0, pos_k % BLOCK]
        v_pre = case.v_ref[case.table[row, pos_k // BLOCK].long(), 0, pos_k % BLOCK]
        qf = pf_q[0].float()  # [Hq, 128, D]

        def attend(keys, values, visible):
            s = (qf @ keys.T) * SCALE
            s = s.masked_fill(~visible, float("-inf"))
            return torch.softmax(s, dim=-1) @ values

        # the pair split: both halves' prefix results + the chunk's own keys (causal, splits 1); each chip's share of
        # the three-way merge (attention.prefill.pair_prefill_contribution: its half, the partner's by LSE only, half of
        # the chunk term), summed as the pair o_proj's row all-reduce sums them
        fresh_k = torch.randn(1, 1, 128, D).to(torch.bfloat16)
        fresh_v = torch.randn(1, 1, 128, D).to(torch.bfloat16)
        put = dict(device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, **case.rep)
        fk = ttnn.from_torch(fresh_k.reshape(128 // BLOCK, 1, BLOCK, D), **put)
        fv = ttnn.from_torch(fresh_v.reshape(128 // BLOCK, 1, BLOCK, D), **put)
        _, ident = case.tensors(cur_pos, torch.arange(128 // BLOCK, dtype=torch.int32).reshape(1, -1))
        o_c, st_c, _ = SdpaPrefillSplitK.op(
            pf_tt_q,
            fk,
            fv,
            ident,
            0,
            num_kv_heads=1,
            block_size=BLOCK,
            k_chunk_size=BLOCK,
            compute_kernel_config=pf_ckc,
            splits=1,
            merge=False,
        )
        (o_0, st_0), (o_1, st_1) = pf_halves
        lses = [_lse_column(t) for t in (st_0, st_1, st_c)]
        split_out = 0
        for own, other in ((0, 1), (1, 0)):
            share = pair_prefill_contribution(
                ttnn.clone((o_0, o_1)[own]),
                ttnn.clone(lses[own]),
                ttnn.clone(lses[other]),
                ttnn.clone(o_c),
                ttnn.clone(lses[2]),
            )
            split_out = split_out + case.host(share)[0].float()
            share.deallocate(True)
        keys = torch.cat([k_pre, fresh_k[0, 0].float()])
        values = torch.cat([v_pre, fresh_v[0, 0].float()])
        visible = torch.cat([torch.ones(128, prefix, dtype=torch.bool), torch.ones(128, 128).tril().bool()], dim=1)
        pcc_ps = _pcc(split_out, attend(keys, values, visible))
        # the served split-K call: the same rows at positions [prefix - 128, prefix) over the whole cache, causal,
        # q128 / k128 in 2 splits, merged on device
        _, tt_full_row = case.tensors(cur_pos, case.table[row : row + 1].contiguous())
        served = SdpaPrefillSplitK.op(
            pf_tt_q,
            case.k_cache,
            case.v_cache,
            tt_full_row,
            prefix - 128,
            num_kv_heads=1,
            block_size=BLOCK,
            compute_kernel_config=pf_ckc,
            splits=2,
        )
        causal = torch.ones(128, prefix, dtype=torch.bool)
        causal[:, prefix - 128 :] = torch.ones(128, 128).tril().bool()
        ref_sk = attend(k_pre, v_pre, causal)
        pcc_sk = _pcc(case.host(served)[0].float(), ref_sk)
        # the stock chunked SDPA on the same call: the split-K result must be at least as close to torch
        grid = mesh_device.compute_with_storage_grid_size()
        stock = ttnn.transformer.chunked_scaled_dot_product_attention(
            pf_tt_q,
            case.k_cache,
            case.v_cache,
            tt_full_row,
            chunk_start_idx=prefix - 128,
            scale=SCALE,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid, q_chunk_size=128, k_chunk_size=128, exp_approx_mode=False
            ),
            compute_kernel_config=pf_ckc,
        )
        pcc_stock = _pcc(case.host(stock)[0].float(), ref_sk)
        stock.deallocate(True)
        logger.info(
            f"B={B} pos={pos} {cfg} prefill split-K over {prefix} keys: pair split (2 halves + chunk) pcc "
            f"{pcc_ps:.6f}, served causal 2-split pcc {pcc_sk:.6f} (stock chunked SDPA {pcc_stock:.6f})"
        )
        assert pcc_ps >= PAIR_PCC_MIN, f"pair-split prefill vs torch pcc {pcc_ps:.6f}"
        assert pcc_sk >= PCC_MIN and pcc_sk >= pcc_stock - 1e-4, f"split-K {pcc_sk:.6f} vs stock {pcc_stock:.6f}"
        _check_splitk_served_shape(mesh_device, case, row, int(cur_pos[row]) + 1, Hq, pf_ckc)
        for t in (o_0, st_0, o_1, st_1, o_c, st_c, *lses, fk, fv, ident, tt_full_row, served):
            t.deallocate(True)
    pf_tt_q.deallocate(True)


@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_sdpa_decode_lse_refuses_out_of_contract(mesh_device, device_params, reset_seeds):
    """unsupported_reason names what is out of contract; the in-contract call returns None. (CI subset.)"""
    case = _Case(mesh_device, 1, 4, 255)
    tt_pos, tt_tab = case.tensors(torch.tensor([255]), case.table)
    pc = _program_config(mesh_device)
    args = (case.tt_q, case.k_cache, case.v_cache, tt_pos, tt_tab)
    assert SdpaDecodeLse.unsupported_reason(*args, scale=SCALE, program_config=pc) is None
    assert "single device" in SdpaDecodeLse.unsupported_reason(*args, program_config=pc, pair_split=True)
    # the half-block mode: one explicit half runs on one chip; a bounded ring is out of its contract
    assert SdpaDecodeLse.unsupported_reason(*args, program_config=pc, half_block=True, half=0) is None
    assert "single device" in SdpaDecodeLse.unsupported_reason(*args, program_config=pc, half_block=True)
    assert "cache_position_modulo" in SdpaDecodeLse.unsupported_reason(
        *args, program_config=pc, half_block=True, half=0, cache_position_modulo=1024
    )
    assert SdpaDecodeLse.unsupported_reason(*args, program_config=pc, cache_position_modulo=1024) is None
    assert "sliding window" in SdpaDecodeLse.unsupported_reason(*args, sliding_window=128, program_config=pc)
    assert "one KV head" in SdpaDecodeLse.unsupported_reason(
        *args, num_kv_heads=2, block_size=BLOCK // 2, program_config=pc
    )
    pc0 = _program_config(mesh_device)
    pc0.k_chunk_size = 0
    assert "k_chunk_size" in SdpaDecodeLse.unsupported_reason(*args, program_config=pc0)
    grid = mesh_device.compute_with_storage_grid_size()
    one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    sharded_mem = ttnn.create_sharded_memory_config(
        (32, D), one_core, ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
    )
    sharded = ttnn.to_memory_config(case.tt_q, sharded_mem)
    assert "interleaved" in SdpaDecodeLse.unsupported_reason(sharded, *args[1:], program_config=pc)
    big = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        q_chunk_size=32,
        k_chunk_size=64,
        max_cores_per_head_batch=128,
    )
    assert "rounds" in SdpaDecodeLse.unsupported_reason(*args, program_config=big) or grid.x * grid.y < 65


def _shared_view(global_view, heads=2, dim=256):
    """vLLM's hybrid kv-cache-group buffer for a global layer: ``global_view [blocks, block, D]`` (one KV head of D) as
    the tensor the sliding layers allocate, ``[blocks, heads, block * D / (heads * dim), dim]``, with every 32x32 tile at
    the position the geometry override reads it from: a block's tiles in row-major order of the layer's own view
    (row tile r, column tile c -> tile r * (D / 32) + c), laid out in the row-major tile order of the shared shape."""
    blocks, block, d = (int(v) for v in global_view.shape)
    tiles = global_view.reshape(blocks, block // 32, 32, d // 32, 32).permute(0, 1, 3, 2, 4)  # [n, rt, ct, 32, 32]
    tiles = tiles.reshape(blocks, -1, 32, 32)  # the block's tiles in the global view's order
    rows = block * d // (heads * dim)
    shared = tiles.reshape(blocks, heads, rows // 32, dim // 32, 32, 32).permute(0, 1, 2, 4, 3, 5)
    return shared.reshape(blocks, heads, rows, dim)


@gated
@pytest.mark.parametrize("kv_dtype", ["bfloat16", "bfloat8_b"])
@pytest.mark.parametrize("B, Hq", [(5, 16), (5, 24), (8, 32)], ids=lambda v: str(v))
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_sdpa_decode_lse_shared_view_matches_stock(mesh_device, device_params, reset_seeds, B, Hq, kv_dtype):
    """The 31B's global layers under vLLM's hybrid kv-cache groups: their K / V live in the sliding layers' view
    [blocks, 2, 64, 256] and are read as one 512-wide KV head of block 64 through the geometry override
    (GEMMA4_SPEC_VERIFY_PREFIX_LSE passes num_kv_heads=1, block_size=64, as pair_split_sdpa does). O equals the stock
    op's with the same override bit for bit, and O and the LSE match torch over the layer's own view of the keys."""
    torch.manual_seed(11)
    block, width = 64, 64  # 4096 keys per row
    blocks = B * width + 1
    dtype = getattr(ttnn, kv_dtype)
    k_glob = torch.randn(blocks, block, D, dtype=torch.bfloat16)
    v_glob = torch.randn(blocks, block, D, dtype=torch.bfloat16)
    k_cache = ttnn.from_torch(_shared_view(k_glob), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    v_cache = ttnn.from_torch(_shared_view(v_glob), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    assert tuple(k_cache.shape) == (blocks, 2, block, 256)
    # the device's (rounded) values in the layer's own view: undo the view on the host copy
    k_ref = ttnn.to_torch(k_cache).float()
    v_ref = ttnn.to_torch(v_cache).float()
    inv = torch.arange(blocks * block * D, dtype=torch.float64).reshape(blocks, block, D)  # exact past 2**24
    order = _shared_view(inv).reshape(-1).long()  # shared-view element e holds global element order[e]
    k_flat, v_flat = torch.empty(blocks * block * D), torch.empty(blocks * block * D)
    k_flat[order], v_flat[order] = k_ref.reshape(-1), v_ref.reshape(-1)
    k_own, v_own = k_flat.reshape(blocks, 1, block, D), v_flat.reshape(blocks, 1, block, D)
    table = (torch.randperm(blocks - 1) + 1)[: B * width].reshape(B, width).to(torch.int32)
    table[B - 1] = table[0]  # a padding row of the packed verify mirrors row 0
    cur_pos = torch.tensor([4095 - 61 * b for b in range(B - 1)] + [4095], dtype=torch.int32)
    q = (torch.randn(1, B, Hq, D) * Q_STD).to(torch.bfloat16)
    q[:, B - 1] = q[:, 0]
    tt_q = ttnn.from_torch(q, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_pos = ttnn.from_torch(cur_pos, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
    tt_tab = ttnn.from_torch(table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
    pc = _program_config(mesh_device)
    view = dict(num_kv_heads=1, block_size=block)
    args = (tt_q, k_cache, v_cache, tt_pos, tt_tab)
    assert SdpaDecodeLse.unsupported_reason(*args, scale=SCALE, program_config=pc, **view) is None
    out, stats = SdpaDecodeLse.op(*args, scale=SCALE, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG, **view)
    stock = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_q,
        k_cache,
        v_cache,
        cur_pos_tensor=tt_pos,
        page_table_tensor=tt_tab,
        scale=SCALE,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=pc,
        paged_cache_geometry=ttnn.PagedCacheGeometryOverride(block_size=block, num_kv_heads=1),
    )
    o = ttnn.to_torch(out)[0, :, :Hq].float()
    o_stock = ttnn.to_torch(stock)[0, :, :Hq].float()
    lse = SdpaDecodeLse.lse_from_stats(ttnn.to_torch(stats), Hq, SCALE)[0]
    ref_out, ref_lse = SdpaDecodeLse.golden(q, k_own, v_own, table, cur_pos, scale=SCALE)
    pcc = _pcc(o, ref_out[0])
    err = (lse - ref_lse[0]).abs()
    logger.info(
        f"shared view B={B} Hq={Hq} {kv_dtype}: max |O - stock| {(o - o_stock).abs().max().item():.3e}, "
        f"pcc(O, torch) {pcc:.6f}, max |lse - torch| {err.max().item():.4f}"
    )
    assert torch.equal(o, o_stock), "O differs from the stock op reading the same view"
    # a wrong view would read other keys (pcc ~0); the stock bf16-dest op itself measures ~0.998 against fp32 torch
    assert pcc > 0.99, f"pcc(O, torch over the own view) {pcc:.5f}"
    assert torch.all(err <= 0.05 + 0.01 * ref_lse[0].abs()), f"lse off by {err.max().item():.4f}"
    assert torch.equal(o[B - 1], o[0]), "the padding row differs from row 0"
    for t in (out, stats, stock, tt_q, tt_pos, tt_tab, k_cache, v_cache):
        t.deallocate(True)


def _pair_positions(B, pos):
    """Row 0 at ``pos``; with B=4 a row shorter than one K chunk (its half 1 is empty), a half-length row and a
    skipped (-1) row."""
    if B == 1:
        return torch.tensor([pos])
    return torch.tensor([pos, 37, pos // 2, -1])


def _pair_cases():
    out = []
    for cfg in ("stock", "fp32_dest"):
        # pos 100: half 1 holds 37 keys; Hq 8: the 31B integration's q heads per chip (one KV group)
        for B, Hq, pos in ((1, 4, 100), (1, 4, 4095), (1, 4, 65536), (4, 4, 4095), (4, 4, 65536), (1, 8, 65536)):
            out.append(pytest.param(B, Hq, pos, cfg, id=f"B{B}-Hq{Hq}-pos{pos}-{cfg}"))
        out.append(pytest.param(4, 8, 4095, cfg, id=f"B4-Hq8-pos4095-{cfg}"))
    return out


@gated
@pytest.mark.parametrize("B, Hq, pos, cfg", _pair_cases())
@pytest.mark.parametrize("mesh_device, device_params", PAIR_MESH, indirect=True)
def test_sdpa_decode_lse_pair_split_on_two_chips(mesh_device, device_params, B, Hq, pos, cfg, reset_seeds):
    """``pair_split=True`` on a (1, 2) mesh, the same q / caches / page table / cur_pos on both chips: chip (0, c)
    attends half ``c % 2`` of every row's K chunks (``planner.pair_key_range``). Each half's O and LSE match torch
    over that half's keys and an empty half holds the sentinel (O = 0, m = -30000 in bf16, l = 1); the two merged
    (``merge_torch``, and the integration's ``w_self = sigmoid(lse_self - lse_partner)`` form) match the single full
    call and torch; the program-cache repeat gives the same bits; the skipped row does not hang anything."""
    case = _Case(mesh_device, B, Hq, pos, seed=7)
    pc, ckc = _program_config(mesh_device), _compute_config(mesh_device, cfg)
    cur_pos = _pair_positions(B, pos)
    live = [b for b in range(B) if cur_pos[b] >= 0]
    tt_pos, tt_tab = case.tensors(cur_pos, case.table)
    args = (case.tt_q, case.k_cache, case.v_cache, tt_pos, tt_tab)
    kw = dict(scale=SCALE, program_config=pc, compute_kernel_config=ckc)
    assert SdpaDecodeLse.unsupported_reason(*args, pair_split=True, **kw) is None
    logger.info(SdpaDecodeLse.plan(*args, pair_split=True, **kw).describe())

    def pair_call():
        out, stats = SdpaDecodeLse.op(*args, pair_split=True, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
        o = [ttnn.to_torch(t)[0, :, :Hq].float() for t in ttnn.get_device_tensors(out)]
        st = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(stats)]
        out.deallocate(True)
        stats.deallocate(True)
        return o, st

    halves, stats = pair_call()  # mesh order: device (0, 0) = half 0, (0, 1) = half 1
    assert len(halves) == 2
    lses = [SdpaDecodeLse.lse_from_stats(st, Hq, SCALE)[0] for st in stats]
    full, lse_full, _ = case.run(cur_pos, case.table, pc, ckc)  # pair_split=False on the same mesh
    ref_out, _ = case.golden(cur_pos, case.table)

    # each half against torch over its own keys; an empty half holds the sentinel
    m_empty = torch.tensor(P.PAIR_EMPTY_M).to(torch.bfloat16).float().item()
    for h in (0, 1):
        got, want = [], []
        for b in live:
            lo, hi = P.pair_key_range(int(cur_pos[b]), K_CHUNK, h)
            if lo == hi:
                assert torch.all(halves[h][b] == 0), f"half {h} row {b}: an empty half's O is not 0"
                assert torch.all(stats[h][0, b, :Hq, 0] == m_empty), f"half {h} row {b}: m {stats[h][0, b, :Hq, 0]}"
                assert torch.all(stats[h][0, b, :Hq, P.TILE] == P.PAIR_EMPTY_L), f"half {h} row {b}: l"
                logger.info(f"half {h} row {b} (cur_pos {int(cur_pos[b])}): empty, sentinel OK")
                continue
            o_ref, lse_ref = case.golden_range(b, lo, hi)
            err = (lses[h][b] - lse_ref).abs().max().item()
            assert err <= 0.05 + 0.01 * lse_ref.abs().max().item(), f"half {h} row {b} keys [{lo}, {hi}): lse off {err}"
            got.append(halves[h][b])
            want.append(o_ref)
        if got:
            pcc_h = _pcc(torch.stack(got), torch.stack(want))
            logger.info(f"B={B} Hq={Hq} pos={pos} {cfg}: half {h} pcc(O, torch over its keys) {pcc_h:.6f}")
            if cfg == "fp32_dest":  # stock: no torch floor (bf16 dest measures 0.9974-0.9986 vs fp32 torch here)
                assert pcc_h > PCC_MIN, f"half {h}: pcc(O, torch) {pcc_h:.5f}"

    # the merge against the full call and torch
    merged, lse_m = SdpaDecodeLse.merge_torch(halves[0], lses[0], halves[1], lses[1])
    w0 = torch.sigmoid(lses[0] - lses[1])[..., None]
    w1 = torch.sigmoid(lses[1] - lses[0])[..., None]
    by_sigmoid = w0 * halves[0] + w1 * halves[1]
    assert torch.allclose(by_sigmoid[live], merged[live], atol=1e-4), "the sigmoid-form merge differs"
    pcc_full = _pcc(merged[live], full[live])
    pcc_ref = _pcc(merged[live], ref_out[live])
    pcc_full_ref = _pcc(full[live], ref_out[live])
    lse_err = (lse_m[live] - lse_full[live]).abs().max().item()
    logger.info(
        f"B={B} Hq={Hq} pos={pos} {cfg}: pcc(merged, full) {pcc_full:.6f}, pcc(merged, torch) {pcc_ref:.6f}, "
        f"pcc(full, torch) {pcc_full_ref:.6f}, max |lse_merged - lse_full| {lse_err:.4f}"
    )
    assert pcc_full >= PAIR_PCC_MIN, f"merged vs full pcc {pcc_full:.6f}"
    _assert_merged_vs_torch(cfg, pcc_ref, pcc_full_ref)
    assert lse_err <= 0.05 + 0.01 * lse_full[live].abs().max().item(), f"merged lse off by {lse_err:.4f}"

    # the model's device merge (tt/attention/decode.py pair_merge): the LSEs all-gathered over the row and picked by
    # each chip's selector row; each chip's w_self * O_self against the host merge's term for its half (a swapped
    # gather order swaps the weights), and the two summed (the o_proj's TP reduction) against the full call
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.attention.decode import pair_merge
    from models.demos.gemma4.tt.ccl import CCLManager
    from models.demos.gemma4.tt.sdpa_pair_split import pair_selector

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=2), tp_axis=1)
    selector = ttnn.from_torch(
        pair_selector(2),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=mesh_config.shard_mapper(mesh_device, tensor_dim=1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_d, stats_d = SdpaDecodeLse.op(
        *args, pair_split=True, memory_config=ttnn.DRAM_MEMORY_CONFIG, stats_memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw
    )
    w_out = pair_merge(out_d, stats_d, selector, mesh_config, CCLManager(mesh_device, num_links=1), scale=SCALE)
    terms = [ttnn.to_torch(t)[0, :, :Hq].float() for t in ttnn.get_device_tensors(w_out)]
    for h, host_term in ((0, w0 * halves[0]), (1, w1 * halves[1])):
        pcc_t = _pcc(terms[h][live], host_term[live])
        assert pcc_t >= PCC_MIN, f"half {h}: device pair_merge term vs host pcc {pcc_t:.6f}"
    pcc_dev = _pcc((terms[0] + terms[1])[live], full[live])
    logger.info(f"B={B} Hq={Hq} pos={pos} {cfg}: device pair_merge summed vs full pcc {pcc_dev:.6f}")
    assert pcc_dev >= PCC_MIN, f"device pair_merge vs full pcc {pcc_dev:.6f}"
    for t in (w_out, selector):
        t.deallocate(True)

    # the program-cache repeat (fresh output buffers): the same bits on both chips
    halves2, stats2 = pair_call()
    for h in (0, 1):
        assert torch.equal(halves2[h][live], halves[h][live]), f"half {h}: O differs on the repeat"
        ml = [0, P.TILE]  # m and l (the other stats columns are unspecified)
        assert torch.equal(stats2[h][0, live, :Hq][..., ml], stats[h][0, live, :Hq][..., ml]), f"half {h}: stats differ"
    tt_pos.deallocate(True)
    tt_tab.deallocate(True)


def _group_cases():
    """(mesh, parts, axis, Hq) per line orientation, each on the mesh whose split axis is the 4-chip line: the 12B
    (G = 4, 16 q heads) on the (1, 4) row, the 26B-A4B (G = 2 over axis 0, 8 q heads) and the 12B on the (4, 1) column;
    rows at 32K / 86K keys, B = 4 adding a row shorter than one chunk (three of its four parts empty), a half-length
    row and a skipped row."""
    fabric = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}
    out = []
    for cfg in ("stock", "fp32_dest"):
        for B, pos in ((1, 32767), (1, 86015), (4, 86015)):
            for mesh, parts, axis, Hq in (((1, 4), 4, 1, 16), ((4, 1), 2, 0, 8), ((4, 1), 4, 0, 16)):
                key = (mesh, parts, axis, Hq, B, pos, cfg)
                out.append(
                    pytest.param(
                        mesh,
                        fabric,
                        parts,
                        axis,
                        Hq,
                        B,
                        pos,
                        cfg,
                        id=f"{mesh[0]}x{mesh[1]}-G{parts}-axis{axis}-Hq{Hq}-B{B}-pos{pos}-{cfg}",
                        marks=_subset_marks(key, GROUP_CI_SUBSET),
                    )
                )
    return out


@pytest.mark.parametrize(
    "mesh_device, device_params, parts, axis, Hq, B, pos, cfg",
    _group_cases(),
    indirect=["mesh_device", "device_params"],
)
def test_sdpa_decode_lse_group_split_on_four_chips(
    mesh_device, device_params, parts, axis, Hq, B, pos, cfg, reset_seeds
):
    """``pair_split=True, split_parts=G, split_axis=a`` on a 4-chip line, the same q / caches / page table / cur_pos
    on every chip: device ``i`` of axis ``a`` attends part ``i % G`` of every row's K chunks (``planner.pair_key_range``
    with ``parts``). Each part's O and LSE match torch over its keys and an empty part holds the sentinel; the parts
    of each group merged (``merge_torch`` folded) match torch (PCC >= 0.999 with fp32 dest) and the single full call;
    the model's device merge over the TP axis (``group_merge`` for G > 2, ``pair_merge`` for G = 2) gives each chip
    its host term, and each group's terms sum to the full call; the program-cache repeat gives the same bits."""
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.attention.decode import group_merge, pair_merge
    from models.demos.gemma4.tt.ccl import CCLManager
    from models.demos.gemma4.tt.sdpa_pair_split import group_mask, pair_selector

    shape = tuple(int(d) for d in mesh_device.shape)
    tp = shape[axis]
    if tp < 4 or shape[1 - axis] != 1:
        pytest.skip(f"split axis {axis} of mesh {shape} is not the 4-chip line")
    case = _Case(mesh_device, B, Hq, pos, seed=11)
    pc, ckc = _program_config(mesh_device), _compute_config(mesh_device, cfg)
    cur_pos = _pair_positions(B, pos)
    live = [b for b in range(B) if cur_pos[b] >= 0]
    tt_pos, tt_tab = case.tensors(cur_pos, case.table)
    args = (case.tt_q, case.k_cache, case.v_cache, tt_pos, tt_tab)
    kw = dict(scale=SCALE, program_config=pc, compute_kernel_config=ckc)
    split = dict(pair_split=True, split_parts=parts, split_axis=axis)
    assert SdpaDecodeLse.unsupported_reason(*args, **split, **kw) is None
    logger.info(SdpaDecodeLse.plan(*args, pair_split=True, split_parts=parts, **kw).describe())

    def split_call():
        out, stats = SdpaDecodeLse.op(*args, memory_config=ttnn.DRAM_MEMORY_CONFIG, **split, **kw)
        o = [ttnn.to_torch(t)[0, :, :Hq].float() for t in ttnn.get_device_tensors(out)]
        st = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(stats)]
        out.deallocate(True)
        stats.deallocate(True)
        return o, st

    outs, stats = split_call()  # mesh order (row-major) = TP-axis order on a line
    assert len(outs) == tp
    lses = [SdpaDecodeLse.lse_from_stats(st, Hq, SCALE)[0] for st in stats]
    full, lse_full, _ = case.run(cur_pos, case.table, pc, ckc)
    ref_out, _ = case.golden(cur_pos, case.table)

    m_empty = torch.tensor(P.PAIR_EMPTY_M).to(torch.bfloat16).float().item()
    for i in range(tp):
        part = i % parts
        got, want = [], []
        for b in live:
            lo, hi = P.pair_key_range(int(cur_pos[b]), K_CHUNK, part, parts)
            if lo == hi:
                assert torch.all(outs[i][b] == 0), f"device {i} row {b}: an empty part's O is not 0"
                assert torch.all(stats[i][0, b, :Hq, 0] == m_empty), f"device {i} row {b}: m"
                assert torch.all(stats[i][0, b, :Hq, P.TILE] == P.PAIR_EMPTY_L), f"device {i} row {b}: l"
                continue
            o_ref, lse_ref = case.golden_range(b, lo, hi)
            err = (lses[i][b] - lse_ref).abs().max().item()
            assert (
                err <= 0.05 + 0.01 * lse_ref.abs().max().item()
            ), f"device {i} row {b} keys [{lo}, {hi}): lse off {err}"
            got.append(outs[i][b])
            want.append(o_ref)
        if got:
            pcc_p = _pcc(torch.stack(got), torch.stack(want))
            logger.info(
                f"G={parts} axis={axis} B={B} pos={pos} {cfg}: device {i} (part {part}) pcc(O, torch) {pcc_p:.6f}"
            )
            if cfg == "fp32_dest":
                assert pcc_p > PCC_MIN, f"device {i}: pcc(O, torch) {pcc_p:.5f}"

    # each group's parts merged on the host against the full call and torch
    weights = []
    for g in range(tp // parts):
        members = list(range(g * parts, (g + 1) * parts))
        merged, lse_m = outs[members[0]], lses[members[0]]
        for i in members[1:]:
            merged, lse_m = SdpaDecodeLse.merge_torch(merged, lse_m, outs[i], lses[i])
        pcc_full = _pcc(merged[live], full[live])
        pcc_ref = _pcc(merged[live], ref_out[live])
        pcc_full_ref = _pcc(full[live], ref_out[live])
        lse_err = (lse_m[live] - lse_full[live]).abs().max().item()
        logger.info(
            f"G={parts} axis={axis} B={B} pos={pos} {cfg}: group {g} pcc(merged, full) {pcc_full:.6f}, "
            f"pcc(merged, torch) {pcc_ref:.6f}, pcc(full, torch) {pcc_full_ref:.6f}, |lse - lse_full| {lse_err:.4f}"
        )
        # stock (bf16 dest): both arms sit at ~0.997 vs fp32 torch and differ by their bf16 accumulation order (4 parts
        # of 21K keys vs one 86K reduction: 0.99989 at B=1, 86K), so the contract there is the torch check below
        assert pcc_full >= (
            PAIR_PCC_MIN if cfg == "fp32_dest" else PCC_MIN
        ), f"group {g}: merged vs full {pcc_full:.6f}"
        _assert_merged_vs_torch(cfg, pcc_ref, pcc_full_ref)
        assert lse_err <= 0.05 + 0.01 * lse_full[live].abs().max().item(), f"group {g}: merged lse off by {lse_err:.4f}"
        for i in members:  # the exact softmax weight of each part: exp(lse_i - lse_group)
            weights.append(torch.exp(lses[i] - lse_m)[..., None])

    # the model's device merge over the TP axis: each chip's w * O against the host term, the group sums against the
    # full call
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp), tp_axis=axis)
    per_chip = dict(
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=mesh_config.shard_mapper(mesh_device, tensor_dim=1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ccl = CCLManager(mesh_device, num_links=1)
    out_d, stats_d = SdpaDecodeLse.op(
        *args, memory_config=ttnn.DRAM_MEMORY_CONFIG, stats_memory_config=ttnn.DRAM_MEMORY_CONFIG, **split, **kw
    )
    if parts == 2:
        selector = ttnn.from_torch(pair_selector(tp), **per_chip)
        w_out = pair_merge(out_d, stats_d, selector, mesh_config, ccl, scale=SCALE)
    else:
        selector = ttnn.from_torch(group_mask(tp, parts), **per_chip) if parts < tp else None
        w_out = group_merge(out_d, stats_d, selector, mesh_config, ccl, scale=SCALE)
    terms = [ttnn.to_torch(t)[0, :, :Hq].float() for t in ttnn.get_device_tensors(w_out)]
    for i in range(tp):
        pcc_t = _pcc(terms[i][live], (weights[i] * outs[i])[live])
        assert pcc_t >= PCC_MIN, f"device {i}: device merge term vs host pcc {pcc_t:.6f}"
    for g in range(tp // parts):
        summed = sum(terms[i] for i in range(g * parts, (g + 1) * parts))
        pcc_dev = _pcc(summed[live], full[live])
        pcc_dev_ref = _pcc(summed[live], ref_out[live])
        logger.info(
            f"G={parts} axis={axis} B={B} pos={pos} {cfg}: group {g} device merge summed vs full pcc {pcc_dev:.6f}, "
            f"vs torch {pcc_dev_ref:.6f}"
        )
        assert pcc_dev >= PCC_MIN, f"group {g}: device merge vs full pcc {pcc_dev:.6f}"
        if cfg == "fp32_dest":
            assert pcc_dev_ref >= PCC_MIN, f"group {g}: device merge vs torch pcc {pcc_dev_ref:.6f}"
    w_out.deallocate(True)
    if selector is not None:
        selector.deallocate(True)

    outs2, stats2 = split_call()
    for i in range(tp):
        assert torch.equal(outs2[i][live], outs[i][live]), f"device {i}: O differs on the repeat"
        ml = [0, P.TILE]
        assert torch.equal(
            stats2[i][0, live, :Hq][..., ml], stats[i][0, live, :Hq][..., ml]
        ), f"device {i}: stats differ"
    tt_pos.deallocate(True)
    tt_tab.deallocate(True)
