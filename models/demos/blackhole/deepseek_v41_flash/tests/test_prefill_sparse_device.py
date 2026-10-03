# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device PREFILL compressed-sparse attention (indexer top-512 + sparse_sdpa, tt/prefill_sparse.py) vs the checkpoint reference on a REAL prefill dump.

Teacher forced: every layer gets the reference's own attention input (``attn_in`` of ``ref_prefill_dump.py``), chunk by chunk; layers run in ascending order inside the
chunk loop (sources before the layers that read them). Reports per layer: attention-output PCC vs the reference (``attn_out``), the device top-512 selection vs the
reference selection (set agreement, attention-mass coverage from ``reference/ref_prefill_sel.py``) and the time per chunk.

Env: DSV41_PS_DIR (dump dir, default /mnt/tt-data/ssinghal/dsv4-prefill-s{S}b1), DSV41_PS_LAYERS ("2,3,8,9,14,15,20"), DSV41_PS_S, DSV41_PS_C (chunk), DSV41_PS_U (users per row),
DSV41_PS_FORCE=1 (sparse path with all-visible ids already while <= 512 entries).
"""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc
from models.demos.blackhole.deepseek_v41_flash.tt.attention import DSV41Attention, DSV41CompressedAttention
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_attention import DSV41PrefillAttention, clear_chunk_caches
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_sparse import attach_prefill_sparse
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

DP = {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}


def coverage(P, pwin, ids, nvis, topk=512):
    """mean over queries with more than ``topk`` visible entries of the dense attention mass share (compressed part, and window + compressed) that ``ids`` covers."""
    q = (nvis > topk).nonzero().flatten()
    if len(q) == 0:
        return float("nan"), float("nan")
    ids = ids[q].long()
    ok = (ids >= 0) & (ids < P.shape[1])
    g = torch.gather(P[q].float(), 1, ids.clamp(min=0)) * ok
    tot = P[q].float().sum(1)
    c1 = (g.sum(1) / tot).mean().item()
    c2 = ((g.sum(1) + pwin[q]) / (tot + pwin[q])).mean().item()
    return c1, c2


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="ring")], indirect=True)
@pytest.mark.timeout(5400)
@torch.no_grad()
def test_prefill_sparse(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    S = int(os.environ.get("DSV41_PS_S", "2048"))
    C = int(os.environ.get("DSV41_PS_C", "512"))
    U = int(os.environ.get("DSV41_PS_U", "1"))
    force = os.environ.get("DSV41_PS_FORCE", "0") == "1"
    layers = [int(x) for x in os.environ.get("DSV41_PS_LAYERS", "2,3,8,9,14,15,20").split(",")]
    d = os.environ.get("DSV41_PS_DIR", f"/mnt/tt-data/ssinghal/dsv4-prefill-s{S}b1")
    mesh_config = mesh_4x8()
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    pas, ws, attns = {}, {}, {}
    for L in layers:
        w = load_layer(L, with_moe=False, max_seq_len=S + 64, with_indexer=True)
        meta = w["meta"]
        ws[L] = w
        if meta["ratio"] == 0:
            a = DSV41Attention(md, mesh_config, ccl, w["attn"], w["freqs_cis"], users_per_row=U, max_seq=256)
        elif meta["is_kv_source"]:
            a = DSV41CompressedAttention(
                md,
                mesh_config,
                ccl,
                w["attn"],
                w["freqs_cis"],
                meta["ratio"],
                w["compressor"],
                users_per_row=U,
                max_comp=128,
            )
        else:
            a = DSV41CompressedAttention(
                md,
                mesh_config,
                ccl,
                w["attn"],
                w["freqs_cis"],
                meta["ratio"],
                None,
                users_per_row=U,
                max_comp=128,
                source=attns[meta["kv_source"]],
            )
        attns[L] = a
        pa = DSV41PrefillAttention(a, w["attn"]["attn_sink"])
        a.prefill = pa
        pa.state_sink = lambda *args: None  # no decode state in this test
        pas[L] = pa
    os.environ["DSV41_PF_SPARSE"] = "1"
    idx_w = {L: ws[L]["indexer"] for L in layers if "indexer" in ws[L]}
    sps = attach_prefill_sparse(
        pas,
        idx_w,
        U,
        S,
        C,
        {L: ws[L]["attn"]["attn_sink"] for L in layers},
        force=force,
        fp4_q=os.environ.get("DSV41_PS_FP4Q", "1") == "1",
    )
    for L in [int(x) for x in os.environ.get("DSV41_PS_DEBUG", "").split(",") if x]:
        sps[L].dbg = True
    refs = {}
    for L in layers:
        dd = torch.load(os.path.join(d, f"layer_{L}.pt"), map_location="cpu")
        refs[L] = {"in": dd["prefill"]["attn_in"][0].float(), "out": dd["prefill"]["attn_out"][0].float()}
        assert refs[L]["in"].shape[0] == S, "dump length != S"
    sel = {}
    for L in layers:
        f = os.path.join(d, f"sel_L{L}.pt")
        if os.path.exists(f):
            sel[L] = torch.load(f, map_location="cpu")
    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    passes = [False, True] if os.environ.get("DSV41_PS_BOTH") == "1" else [force]
    for force_pass in passes:
        for sp_ in sps.values():
            sp_.force = force_pass
        tag = "force" if force_pass else "auto"
        got = {L: torch.zeros(S, 5120) for L in layers}
        ids_dev = {L: torch.full((S, 512), -1, dtype=torch.long) for L in layers}
        times = {L: [] for L in layers}
        for s0 in range(0, S, C):
            for L in layers:
                x = refs[L]["in"][s0 : s0 + C]
                h = ttnn.from_torch(
                    x.unsqueeze(0)
                    .expand(U, C, 5120)
                    .reshape(1, 1, U * C, 5120)
                    .expand(rows, 1, U * C, 5120)
                    .contiguous(),
                    device=md,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=shard,
                )
                ttnn.synchronize_device(md)
                t0 = time.perf_counter()
                out = pas[L].forward(h, S, s0=s0)
                ttnn.synchronize_device(md)
                times[L].append(time.perf_counter() - t0)
                o = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(U * C, 5120)
                got[L][s0 : s0 + C] = o[:C]
                if L in sps:
                    sp = sps[L]
                    ids = sp.ids if sp.indexer is not None else (sp.idx_src.ids if sp.idx_src is not None else None)
                    if ids is not None and sp.active(s0, C) and (s0 + C) // pas[L].ratio > 512:
                        t = ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(U, C, 512)[0].long()
                        t = torch.where(t == 0xFFFFFFFF, torch.full_like(t, -1), t)
                        ids_dev[L][s0 : s0 + C] = t
                ttnn.deallocate(h)
                ttnn.deallocate(out)
            clear_chunk_caches()
        for L in layers:
            r = pas[L].ratio
            p = pcc(got[L], refs[L]["out"])
            per_chunk = [round(pcc(got[L][s : s + C], refs[L]["out"][s : s + C]), 4) for s in range(0, S, C)]
            msg = f"PSPARSE[{tag}] layer {L} ratio {r} S={S} C={C} U={U}: attention-out PCC {p:.5f}; per chunk {per_chunk}; chunk time ms {[round(t * 1e3, 1) for t in times[L]]}"
            if L in sel and "ids" in sel[L]:
                sl = sel[L]
                nvis = sl["nvis"]
                ref_ids = sl["ids"].long()
                dv = ids_dev[L]
                q = (nvis > 512).nonzero().flatten()
                if len(q):
                    agree = []
                    for i in q.tolist():
                        a_ = set(dv[i][dv[i] >= 0].tolist())
                        b_ = set(ref_ids[i][ref_ids[i] >= 0].tolist())
                        agree.append(len(a_ & b_) / max(1, len(b_)))
                    cd = coverage(sl["P"], sl["pwin"], dv, nvis)
                    cr = coverage(sl["P"], sl["pwin"], ref_ids, nvis)
                    msg += (
                        f"\nPSPARSE   selection layer {L}: queries with >512 visible {len(q)}, set agreement mean {sum(agree) / len(agree):.4f} min {min(agree):.4f}; "
                        f"mass coverage (compressed) device {cd[0]:.4f} vs reference {cr[0]:.4f}; total (window + compressed) device {cd[1]:.4f} vs reference {cr[1]:.4f}"
                    )
            cq = torch.nn.functional.cosine_similarity(got[L], refs[L]["out"], dim=-1)
            tail = cq[S // 2 :]
            msg += f"\nPSPARSE   per-query cosine over the last half: min {tail.min():.4f} p5 {tail.quantile(0.05):.4f} median {tail.median():.4f}; worst query {int(S // 2 + tail.argmin())}"
            print(msg, flush=True)
            torch.save(
                {"got": got[L].to(torch.bfloat16), "ids": ids_dev[L]},
                f"/mnt/tt-data/ssinghal/dsv4-logs/h46x_dev_S{S}_C{C}_{tag}_L{L}.pt",
            )
