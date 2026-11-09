# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_SPEC_ADAPTIVE_K_SDPA (with QWEN36_SPEC_ADAPTIVE_K=1): the wide family's T = 8 fused spec-verify SDPA as ONE packed
group per user (Tg = 8: 8 candidates x NH = 3 local q heads in one 32-row q tile, the user's KV read once per layer)
against the r5 plan (TPAttention._SPEC_SDPA_L1_FIT[8]: two groups of Tg = 4 per user, each re-reading the user's KV).

  test_adaptive_k_sdpa_one_group  the op alone at the verify's shapes (B users x T, NH = 3, head_dim 256, bf8 paged KV in
                                  64-row blocks, dynamic k chunk, cores per group = min(cap, grid // groups) as the plan
                                  computes them). T = 8: both calls (the r5 split exists only there) close to a torch
                                  reference on the device's own bf8 cache
                                  (PCC >= 0.99 per user wherever the r5 call is), and close to each other; not asserted
                                  bit-identical (the cores per group differ, so the KV chunk reduction tree differs). At
                                  32K it also times both calls back to back (the ttsim-perf prediction for B = 4: 405 ->
                                  232 us per call on one chip, x16 attention layers). T = 5 / 6 (wide families 4:4 / 4:5,
                                  no r5 table entry: the legacy per-row call otherwise): the one-group call vs torch.

Page-table layouts (``layout``):
  prod         the serving layout: every row spec_table_width(262144, K) = 4097 entries wide, the user's real blocks first,
               then zeros (vLLM's null block 0, never a real block; Qwen36MTPContractForCausalLM._table /
               Qwen36Model._fit_spec_page_table). Every entry the op resolves lies inside the row.
  tight_clamp  rows only ceil((max base + T) / 64) entries wide, every entry a real block, with clamp_kv_tail=True. The
               dynamic k-chunk is 128 rows (two blocks) from position 64 on, so for p_max >= 128 with p_max % 128 < 64 the
               last chunk's second block lies one entry past such a row. Without the clamp the op resolves that entry from
               the row-major page's unwritten DRAM padding (stale host command-queue bytes): after an in-process mesh
               re-open (this file's per-item mesh fixture) it held a wild block id and user 0 at base 127 came out as
               garbage (PCC ~0, |x| ~1e36) on every chip. That was this test's old default layout, not a silicon race.
               The clamp reads the tile holding the group's last position there instead (QWEN36_SPEC_SDPA_KV_CLAMP; see
               test_spec_sdpa_kv_clamp.py for the deterministic poisoned-entry check and the bit-identity check).

Run (inside the tt-metal container, on a Blackhole Galaxy / P150x8 TP mesh):
    source /tmp/penv.sh
    pytest models/demos/blackhole/qwen36/tests/test_spec_adaptive_k_sdpa.py -v -s
"""
import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.spec_engine import spec_table_width

BLOCK, NH, HD = 64, 3, 256
PROD_WIDTH = spec_table_width(262144, 7, BLOCK)  # 4097 for every K <= 63
BASES = {
    "edges": [127, 28, 1000, 0],
    "32k": [32760, 32289, 30000, 31999],
}
CASES = [("edges", 4, 8), ("edges", 2, 8), ("edges", 1, 8), ("32k", 4, 8), ("32k", 2, 8), ("32k", 1, 8)]
CASES += [("edges", 4, 6), ("32k", 4, 6), ("edges", 2, 6), ("32k", 4, 5), ("edges", 1, 5)]
LAYOUTS = ("prod", "tight_clamp")


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def _rep(mesh, t, dtype, layout):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _dev0(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[: t.shape[0]]


def _cfg(mesh, groups, cap):
    grid = mesh.compute_with_storage_grid_size()
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
        max_cores_per_head_batch=min(cap, max(1, (grid.x * grid.y) // groups)),
    )


def _tables(users, nb_real, layout, g):
    """(pool blocks, page table [users, width], real-block table [users, nb_real]) for ``layout``."""
    if layout == "prod":
        nblocks = 1 + users * nb_real  # block 0 = vLLM's null block, never a real block
        real = (1 + torch.randperm(nblocks - 1, generator=g)).reshape(users, nb_real).to(torch.int32)
        pt = torch.zeros(users, PROD_WIDTH, dtype=torch.int32)
        pt[:, :nb_real] = real
        return nblocks, pt, real
    nblocks = users * nb_real
    real = torch.randperm(nblocks, generator=g).reshape(users, nb_real).to(torch.int32)
    return nblocks, real, real


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("pos_set, users, T", CASES, ids=[f"{p}-u{u}-T{t}" for p, u, t in CASES])
def test_adaptive_k_sdpa_one_group(mesh_device, pos_set, users, T, layout, reset_seeds, ensure_gc):
    base = BASES[pos_set][:users]
    nb_real = math.ceil((max(base) + T) / BLOCK)
    g = torch.Generator().manual_seed(23 + users)
    nblocks, pt_u, real = _tables(users, nb_real, layout, g)
    k_init = torch.randn(nblocks, 1, BLOCK, HD, generator=g).to(torch.bfloat16)
    v_init = torch.randn(nblocks, 1, BLOCK, HD, generator=g).to(torch.bfloat16)
    if layout == "prod":
        k_init[0].zero_()
        v_init[0].zero_()
    # candidate (u, j) at position base[u] + j, user-major: the cur_pos list of both plans
    pos = torch.tensor([base[u] + j for u in range(users) for j in range(T)], dtype=torch.int32)
    # Q rows as the verify holds them: [1, B*T, 32, HD] (rows NH..31 of each candidate tile random: must not matter)
    q = torch.randn(1, users * T, 32, HD, generator=g).to(torch.bfloat16)
    clamp = {"clamp_kv_tail": True} if layout == "tight_clamp" else {}

    kc = _rep(mesh_device, k_init, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    vc = _rep(mesh_device, v_init, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    pos_tt = _rep(mesh_device, pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    plans = {
        # QWEN36_SPEC_ADAPTIVE_K_SDPA: 1 group of Tg = T per user (cap 64)
        "one": dict(G=users, tg=T, cap=64, pt=pt_u),
    }
    if T == 8:  # r5 table: 2 groups of Tg = 4 per user (cap 55), each group's page-table row = its user's
        plans["split"] = dict(G=2 * users, tg=4, cap=55, pt=pt_u.repeat_interleave(2, dim=0))
    for p in plans.values():
        p["pt_tt"] = _rep(mesh_device, p["pt"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        p["q_tt"] = _rep(mesh_device, q.reshape(1, p["G"], p["tg"] * 32, HD), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        p["cfg"] = _cfg(mesh_device, p["G"], p["cap"])
    scale = HD**-0.5

    def call(name):
        p = plans[name]
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            p["q_tt"],
            kc,
            vc,
            page_table_tensor=p["pt_tt"],
            cur_pos_tensor=pos_tt,
            scale=scale,
            program_config=p["cfg"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            spec_multi_pos_tiles=p["tg"],
            spec_pack_heads=NH,
            **clamp,
        )

    outs = {}
    for name in plans:
        o = call(name)
        outs[name] = _dev0(mesh_device, o).reshape(users, T, 32, HD)[:, :, :NH, :].float()
        ttnn.deallocate(o)

    kd = _dev0(mesh_device, kc).float()
    vd = _dev0(mesh_device, vc).float()
    qv = q.reshape(users, T, 32, HD)[:, :, :NH, :].float()
    ref = torch.zeros(users, T, NH, HD)
    for u in range(users):
        k_u = kd[real[u].long()].reshape(nb_real * BLOCK, HD)
        v_u = vd[real[u].long()].reshape(nb_real * BLOCK, HD)
        for j in range(T):
            pp = base[u] + j
            s = (qv[u, j] @ k_u[: pp + 1].T) * scale
            ref[u, j] = torch.softmax(s, dim=-1) @ v_u[: pp + 1]

    tag = f"{pos_set}-u{users}-T{T}-{layout}"
    b = torch.nan_to_num(outs["one"])
    if "split" not in outs:
        per_user = [(base[u], round(_pcc(b[u], ref[u]), 6)) for u in range(users)]
        logger.info(f"[ak-sdpa] {tag}: one group (Tg={T}, {users} groups) per user (base, PCC vs torch): {per_user}")
        for bb, pb in per_user:
            assert pb >= 0.99, f"[{tag}] base {bb}: one-group PCC vs torch {pb}"
        for t in (kc, vc, pos_tt, *(p[k] for p in plans.values() for k in ("pt_tt", "q_tt"))):
            ttnn.deallocate(t)
        return
    a = torch.nan_to_num(outs["split"])
    same = bool(torch.equal(a, b))
    pcc_ab = _pcc(a, b)
    maxd = float((a - b).abs().max())
    per_user = [(base[u], round(_pcc(a[u], ref[u]), 6), round(_pcc(b[u], ref[u]), 6)) for u in range(users)]
    logger.info(
        f"[ak-sdpa] {tag}: one group (Tg=8, {plans['one']['G']} groups) vs split (Tg=4, {plans['split']['G']} groups): "
        f"bit-identical={same} PCC={pcc_ab:.7f} max|d|={maxd:.3e}; per user (base, PCC split vs torch, PCC one vs torch): "
        f"{per_user}"
    )
    for bb, pa, pb in per_user:  # both calls are exact attention here: both must match torch
        assert pa >= 0.99, f"[{tag}] base {bb}: split PCC vs torch {pa}"
        assert pb >= 0.99, f"[{tag}] base {bb}: one-group PCC vs torch {pb} (split {pa})"
    assert pcc_ab >= 0.999, f"[{tag}] one-group SDPA vs split PCC {pcc_ab}"

    if pos_set == "32k" and layout == "prod":
        n = int(os.environ.get("AK_SDPA_TIME_ITERS", "40"))
        times = {}
        for name in ("split", "one", "split", "one"):
            for _ in range(3):
                ttnn.deallocate(call(name))
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(n):
                ttnn.deallocate(call(name))
            ttnn.synchronize_device(mesh_device)
            times.setdefault(name, []).append((time.perf_counter() - t0) / n * 1e6)
        logger.info(
            f"[ak-sdpa] {tag} per call: split {[round(t, 1) for t in times['split']]} us, one group "
            f"{[round(t, 1) for t in times['one']]} us; x16 layers: split {16 * min(times['split']) / 1000:.2f} ms, "
            f"one group {16 * min(times['one']) / 1000:.2f} ms"
        )

    for t in (kc, vc, pos_tt, *(p[k] for p in plans.values() for k in ("pt_tt", "q_tt"))):
        ttnn.deallocate(t)
