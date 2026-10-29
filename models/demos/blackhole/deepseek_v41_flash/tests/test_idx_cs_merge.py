# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Op-level test of the column-sharded decode selection (tt/indexer_colshard.py ``cs_select``) on SYNTHETIC score rows: sharded (entry e on column e % 8, local top-512, candidate all_gather,
global top-512) vs the replicated ``topk_large_indices`` on the whole row, against a host reference.

No keys, no projection: the score rows are the input, so the ranking (ids, ties, valid lengths, padding rows) is tested exactly and at the 1M-entry sizes of the production slabs.

Per case it prints ``ICM`` lines: how the replicated device top-k resolves bf16 ties at the k-th score (host rule "larger entry id first" = rule A, "smaller id first" = rule B), whether the
sharded ids equal the replicated ones EXACTLY (bit-identical sorted id lists), or equal up to the choice among entries tied at the k-th score, and the ids >= nvalid (masked entries, a consumer skips).
Env: DSV41_ICM_N (comma list of valid entry counts, default 5000,40000,300000,1000000), DSV41_ICM_R (comma list of rows per mesh row, default 1,4,12,24), DSV41_ICM_KINDS.
"""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_indexer_colshard import DEV
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import _cs_consts
from models.demos.blackhole.deepseek_v41_flash.tt.indexer_colshard import cs_select
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

TOPK = 512
NS = [int(x) for x in os.environ.get("DSV41_ICM_N", "5000,40000,300000,1000000").split(",")]
RS = [int(x) for x in os.environ.get("DSV41_ICM_R", "1,4,12,24").split(",")]
KINDS = os.environ.get("DSV41_ICM_KINDS", "random,ties,coarse,flat").split(",")


class _Ix:
    """the attributes ``cs_select`` reads from an indexer"""

    def __init__(self, md, n_local, ccl, mc):
        self.md, self.topk, self.cols, self.n_alloc, self.ccl, self.mc = md, TOPK, int(md.shape[1]), n_local, ccl, mc


def gen_scores(kind, R, N, g):
    """[R, N] float32 scores of bf16-representable values."""
    if kind == "random":  # continuous: ties are rare
        s = torch.randn(R, N, generator=g) * 3.0 + 4.0
    elif kind == "ties":  # many exact bf16 ties around the cutoff (values on a 1/16 grid)
        s = torch.round((torch.randn(R, N, generator=g) * 3.0 + 4.0) * 16) / 16
    elif kind == "coarse":  # a big plateau of equal scores at the k-th rank
        s = torch.round(torch.randn(R, N, generator=g) * 2.0)
    elif kind == "flat":  # all equal
        s = torch.full((R, N), 2.0)
    else:
        raise ValueError(kind)
    return s.to(torch.bfloat16).float()


def host_topk(s, k, rule):
    """ids of the top-k by (score desc, id desc if rule == 'A' else id asc); s [N]"""
    N = s.shape[0]
    idx = torch.arange(N)
    key_id = idx if rule == "A" else -idx
    order = torch.from_numpy(
        __import__("numpy").lexsort((key_id.numpy(), s.numpy()))[::-1].copy()
    )  # last key is primary: score asc -> reversed = desc
    return order[:k]


def upload(md, host, dims, dtype, layout):
    rows, cols = tuple(md.shape)
    mapper = (
        ttnn.ReplicateTensorToMesh(md)
        if dims == (None, None)
        else ttnn.ShardTensor2dMesh(md, dims=dims, mesh_shape=(rows, cols))
    )
    return ttnn.from_torch(
        host.contiguous(),
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEV, indirect=True)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_cs_merge(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    summary = []
    n_bad = 0
    for N in NS:
        n_alloc = -(-(N + 32) // (32 * cols)) * (32 * cols)
        n_local = n_alloc // cols
        ix = _Ix(md, n_local, ccl, mc)
        consts = _cs_consts(md, n_local)
        for R in RS:
            for kind in KINDS:
                g = torch.Generator().manual_seed(N * 31 + R)
                s = gen_scores(kind, R, N, g)
                # per-row valid counts: row 0 = N, the others ragged (some below topk -> masked entries are ranked last), the last row of a block of padding rows = 0
                nvalid = torch.tensor([max(0, N - (37 * i) % max(1, min(N, 300))) for i in range(R)])
                if R >= 4:
                    nvalid[R - 1] = min(N, 300)  # < topk: fewer valid entries than k
                if R >= 12:
                    nvalid[R - 2] = 0  # a padding row
                full = torch.full((R, n_alloc), -1e30)
                full[:, :N] = s
                keep = torch.arange(n_alloc)[None, :] < nvalid[:, None]
                masked = torch.where(keep, full, torch.full_like(full, -1e30)).to(torch.bfloat16)
                # replicated: whole row on every device (rows of the mesh get the same data), valid length = batch maximum (>= k)
                vmax = max(int(nvalid.max()), TOPK)
                rep_in = upload(
                    md, masked.reshape(R, 1, 1, n_alloc), (None, None), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
                )
                vl_full = upload(
                    md,
                    torch.full((1, 1, 1, 1), vmax, dtype=torch.int32),
                    (None, None),
                    ttnn.uint32,
                    ttnn.ROW_MAJOR_LAYOUT,
                )
                rep_ids = ttnn.experimental.topk_large_indices(rep_in, k=TOPK, valid_length_tensor=vl_full)
                rep_ids = ttnn.reshape(rep_ids, [R, 1, 1, TOPK])
                # sharded: local scores of column c = masked[:, c::cols] ([R, n_local]); the global ids are rebuilt by cs_select (cols * slot + c); masking by nvalid on the global ids inside (nv path)
                loc = masked.reshape(R, n_local, cols).permute(2, 0, 1).contiguous()  # [cols, R, n_local]
                loc_in = upload(
                    md, loc.reshape(cols * R, 1, 1, n_local), (None, 0), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
                )
                nv = upload(
                    md,
                    nvalid.float().reshape(1, 1, R, 1),
                    (None, None),
                    ttnn.float32,
                    ttnn.TILE_LAYOUT,
                )
                # (a) global-id masking path (plain decode: nv given)
                os.environ["DSV41_IDX_CS_FAST"] = "0"
                sh_ids_nv, _ = cs_select(ix, consts, loc_in, R, n_local, nv, None)
                # (b) local valid length path (the fused score op masked the scores already: nv None, vl = valid local slots per column of the batch maximum)
                vlc = torch.tensor([max(-(-(vmax - c) // cols), TOPK) for c in range(cols)], dtype=torch.int32)
                vl_loc = upload(md, vlc.reshape(cols, 1, 1, 1), (None, 0), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                sh_ids_vl, _ = cs_select(ix, consts, loc_in, R, n_local, None, vl_loc)
                # (c) the same two with the JIT kernels of the candidate / final stage (tt/idx_cs_kernels, ids ascending)
                os.environ["DSV41_IDX_CS_FAST"] = "1"
                try:
                    f_ids_nv, _ = cs_select(ix, consts, loc_in, R, n_local, nv, None)
                    f_ids_vl, _ = cs_select(ix, consts, loc_in, R, n_local, None, vl_loc)
                    ttnn.synchronize_device(md)
                except Exception as e:  # noqa: BLE001
                    print(f"ICM JIT merge ERR {type(e).__name__}: {str(e)[:400]}", flush=True)
                    f_ids_nv, f_ids_vl = (
                        sh_ids_nv,
                        sh_ids_vl,
                    )  # (the chain results stand in: only the op-chain merge is judged)
                ttnn.synchronize_device(md)
                outs = {}
                for name, t in (
                    ("rep", rep_ids),
                    ("sh_nv", sh_ids_nv),
                    ("sh_vl", sh_ids_vl),
                    ("f_nv", f_ids_nv),
                    ("f_vl", f_ids_vl),
                ):
                    d0 = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(R, -1).long()
                    # the other mesh rows / columns must agree with device 0 (same data)
                    d_last = ttnn.to_torch(ttnn.get_device_tensors(t)[rows * cols - 1]).reshape(R, -1).long()
                    assert bool((d0 == d_last).all()), f"{name}: devices disagree"
                    outs[name] = d0
                nA = nB = nex = nset = nmod = 0
                ninv = {"rep": 0, "sh_nv": 0, "sh_vl": 0, "f_nv": 0, "f_vl": 0}
                nfast = nfast_asc = 0
                for i in range(R):
                    nv_i = int(nvalid[i])
                    sc = masked[i].float()
                    refA = host_topk(sc, TOPK, "A").sort().values
                    refB = host_topk(sc, TOPK, "B").sort().values
                    a = outs["rep"][i].sort().values
                    nA += int(torch.equal(a, refA))
                    nB += int(torch.equal(a, refB))
                    nfast += int(torch.equal(outs["f_vl"][i].sort().values, outs["sh_vl"][i].sort().values)) & int(
                        torch.equal(outs["f_nv"][i].sort().values, outs["sh_nv"][i].sort().values)
                    )
                    nfast_asc += int(bool((outs["f_vl"][i][1:] >= outs["f_vl"][i][:-1]).all()))
                    for nm in ("sh_nv", "sh_vl", "f_nv", "f_vl"):
                        ids = outs[nm][i]
                        ninv[nm] += int((ids >= nv_i).sum()) if nv_i >= TOPK else 0
                    for nm in ("sh_nv", "sh_vl", "f_nv", "f_vl"):
                        b = outs[nm][i].sort().values
                        if nm == "f_vl":
                            nex += int(torch.equal(a, b))
                        # equal up to the choice among entries tied at the k-th score
                        if nv_i >= TOPK:
                            kth = sc[a].min()
                            sa, sb = sc[a.clamp(max=n_alloc - 1)], sc[b.clamp(max=n_alloc - 1)]
                            same_multiset = torch.equal(sa.sort().values, sb.sort().values)
                            gt = lambda ids: set(ids[sc[ids] > kth].tolist())
                            ok = same_multiset and gt(a) == gt(b)
                            if nm == "f_vl":
                                nset += int(set(a.tolist()) == set(b.tolist()))
                                nmod += int(ok)
                            if not ok:
                                n_bad += 1
                        else:  # fewer than k valid entries: every valid entry must be selected
                            va = set(x for x in a.tolist() if x < nv_i)
                            vb = set(x for x in b.tolist() if x < nv_i)
                            if va != vb or len(va) != nv_i:
                                n_bad += 1
                msg = (
                    f"ICM N={N} R={R} {kind}: replicated==ruleA {nA}/{R} ==ruleB {nB}/{R}; sharded(JIT merge, vl) bit-identical to replicated {nex}/{R}, same set {nset}/{R}, equal up to ties at the k-th score {nmod}/{R} "
                    f"(rows with >= k valid entries counted); JIT merge == op-chain merge (same ids) {nfast}/{R}, JIT ids ascending {nfast_asc}/{R}; "
                    f"ids >= nvalid selected (masked entries) chain nv {ninv['sh_nv']} vl {ninv['sh_vl']}, JIT nv {ninv['f_nv']} vl {ninv['f_vl']}"
                )
                print(msg, flush=True)
                summary.append(msg)
                for t in (rep_in, vl_full, rep_ids, loc_in, nv, vl_loc, sh_ids_nv, sh_ids_vl, f_ids_nv, f_ids_vl):
                    ttnn.deallocate(t)
    print("\n".join(summary))
    assert n_bad == 0, f"{n_bad} rows differ beyond ties"


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEV, indirect=True)
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_cs_breakdown(mesh_device):
    """traced us of every stage of the column-sharded selection (JIT merge) on synthetic score rows: local top-k, candidate kernel, the two all_gathers, merge top-k, final kernel; and the replicated top-k of
    the whole row for comparison. Env DSV41_ICM_N (entries), DSV41_ICM_R (rows)."""
    from models.demos.blackhole.deepseek_v41_flash.tests.test_indexer_colshard import traced_us
    from models.demos.blackhole.deepseek_v41_flash.tt.indexer_colshard import cs_cand, cs_final

    md = mesh_device
    rows, cols = tuple(md.shape)
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    for N in NS:
        n_alloc = -(-(N + 32) // (32 * cols)) * (32 * cols)
        n_local = n_alloc // cols
        consts = _cs_consts(md, n_local)
        for R in RS:
            g = torch.Generator().manual_seed(N + R)
            s = gen_scores("random", R, n_alloc, g).to(torch.bfloat16)
            full = upload(md, s.reshape(R, 1, 1, n_alloc), (None, None), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            loc = upload(
                md,
                s.reshape(R, n_local, cols).permute(2, 0, 1).contiguous().reshape(cols * R, 1, 1, n_local),
                (None, 0),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            vmax = max(N, TOPK)
            vl_full = upload(
                md, torch.full((1, 1, 1, 1), vmax, dtype=torch.int32), (None, None), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
            )
            vlc = torch.tensor([max(-(-(vmax - c) // cols), TOPK) for c in range(cols)], dtype=torch.int32)
            vl_loc = upload(md, vlc.reshape(cols, 1, 1, 1), (None, 0), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            sr = ttnn.reshape(loc, [1, 1, R, n_local])
            lids = ttnn.reshape(
                ttnn.experimental.topk_large_indices(sr, k=TOPK, valid_length_tensor=vl_loc), [R, 1, 1, TOPK]
            )
            vals, gids = cs_cand(loc, lids, consts["col_u32"], cols)
            cv = mc.allgather(vals, ccl, axis=1, dim=3)
            ci = mc.allgather(gids, ccl, axis=1, dim=3)
            W = int(cv.shape[-1])
            pos = ttnn.reshape(
                ttnn.experimental.topk_large_indices(ttnn.reshape(cv, [1, 1, R, W]), k=TOPK), [R, 1, 1, TOPK]
            )
            ids = cs_final(pos, ci, n_alloc)
            ttnn.synchronize_device(md)
            st = {
                "replicated topk (whole row)": lambda: ttnn.experimental.topk_large_indices(
                    ttnn.reshape(full, [1, 1, R, n_alloc]), k=TOPK, valid_length_tensor=vl_full
                ),
                "local topk (1/8 row)": lambda: ttnn.experimental.topk_large_indices(
                    sr, k=TOPK, valid_length_tensor=vl_loc
                ),
                "cs_cand kernel": lambda: cs_cand(loc, lids, consts["col_u32"], cols),
                "all_gather vals": lambda: mc.allgather(vals, ccl, axis=1, dim=3),
                "all_gather gids": lambda: mc.allgather(gids, ccl, axis=1, dim=3),
                "merge topk (8 x 512)": lambda: ttnn.experimental.topk_large_indices(
                    ttnn.reshape(cv, [1, 1, R, W]), k=TOPK
                ),
                "cs_final bitmap": lambda: cs_final(pos, ci, n_alloc),
                "cs_final radix": lambda: cs_final(pos, ci),
            }
            out = []
            for name, fn in st.items():
                try:
                    out.append(f"{name} {traced_us(md, fn, inner=10, reps=5):.0f}")
                except Exception as e:  # noqa: BLE001
                    out.append(f"{name} ERR {str(e)[:60]}")
            print(f"ICMB N={N} R={R}: " + "; ".join(out) + " (us)", flush=True)
            for t in (full, loc, vl_full, vl_loc, vals, gids, cv, ci, ids):
                ttnn.deallocate(t)


def _ranges(idx):
    """sorted list of ints -> 'a-b,c-d' compressed ranges"""
    out, s = [], None
    for i, v in enumerate(idx):
        if s is None:
            s = p = v
        elif v == p + 1:
            p = v
        else:
            out.append(f"{s}-{p}" if s != p else f"{s}")
            s = p = v
    if s is not None:
        out.append(f"{s}-{p}" if s != p else f"{s}")
    return ",".join(out[:24]) + (f",...({len(out)} ranges)" if len(out) > 24 else "")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEV, indirect=True)
@pytest.mark.timeout(1800)
@torch.no_grad()
def test_tie_probe(mesh_device):
    """how does the replicated ``topk_large_indices`` resolve ties at the k-th score? One row per case on mesh device 0: the tie class T (entries with the cutoff score, ids ascending) and WHICH of them are selected, as
    ranges of positions in T. Cases: flat row; a plateau (value 5) under 100 high entries (value 10)."""
    md = mesh_device
    for N in (4096, 5000, 16384, 20000, 40000, 100000, 300000):
        n_alloc = -(-(N + 32) // 32) * 32
        for case in ("flat", "plateau"):
            g = torch.Generator().manual_seed(N)
            s = torch.full((1, n_alloc), -1e30)
            s[0, :N] = 5.0
            hi = torch.zeros(0, dtype=torch.long)
            if case == "plateau":
                hi = torch.randperm(N, generator=g)[:100]
                s[0, hi] = 10.0
            t_in = ttnn.from_torch(
                s.to(torch.bfloat16).reshape(1, 1, 1, n_alloc),
                device=md,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            )
            vl = ttnn.from_torch(
                torch.full((1, 1, 1, 1), max(N, TOPK), dtype=torch.int32),
                device=md,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            )
            ids = (
                ttnn.to_torch(
                    ttnn.get_device_tensors(ttnn.experimental.topk_large_indices(t_in, k=TOPK, valid_length_tensor=vl))[
                        0
                    ]
                )
                .reshape(-1)
                .long()
            )
            tie = torch.tensor(sorted(set(range(N)) - set(hi.tolist())))
            chosen = sorted(set(ids.tolist()) - set(hi.tolist()))
            pos = {v: i for i, v in enumerate(tie.tolist())}
            ranks = sorted(pos[c] for c in chosen if c in pos)
            print(
                f"ICMP N={N} {case}: tie class {len(tie)}, selected ties {len(chosen)} at tie-ranks {_ranges(ranks)}; order of the returned ids: first 8 {ids[:8].tolist()}",
                flush=True,
            )
            ttnn.deallocate(t_in)
            ttnn.deallocate(vl)
