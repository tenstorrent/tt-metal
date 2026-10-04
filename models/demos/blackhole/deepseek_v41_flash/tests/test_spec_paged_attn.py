# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage 1 of the paged spec plug-in: SpecPaged{Window,Compressed}Attention (paged pool, virtual-user rows, ``paged_kv_step`` nq = n) vs the non-paged
Spec{Window,Compressed}Attention on the same random inputs, same weights, from an EMPTY state, blocks of n rows per user for several consecutive blocks
(accept count forced to n - 1 so the ratio-2 ``prev_cs`` chain advances like a fully accepted block).
Env: DSV41_N (default 4), DSV41_LAYERS ("0,2,3,20,21"), DSV41_BLOCKS (default 12), DSV41_MIN_PCC (0.998)."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as P
from models.demos.blackhole.deepseek_v41_flash.tt.attention import DSV41Attention, DSV41CompressedAttention
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.paged_attention import DSV41PagedStepState
from models.demos.blackhole.deepseek_v41_flash.tt.spec_attention import SpecCompressedAttention, SpecWindowAttention
from models.demos.blackhole.deepseek_v41_flash.tt.spec_paged import (
    RING_SPEC,
    SpecPagedCompressedAttention,
    SpecPagedWindowAttention,
)
from models.demos.blackhole.deepseek_v41_flash.tt.spec_state import SpecStepState
from models.demos.blackhole.deepseek_v41_flash.tt.step_state import DSV41StepState
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_spec_paged_attn(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    n = int(os.environ.get("DSV41_N", "4"))
    layer_ids = [int(x) for x in os.environ.get("DSV41_LAYERS", "0,2,3,20,21").split(",")]
    nblk = int(os.environ.get("DSV41_BLOCKS", "12"))
    ctx = 256
    U = 4
    B = rows * U
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    pool = P.PagedKVPool(md, U, num_pages=U * 2, n_ring_layers=len(layer_ids), max_ctx=ctx, ring_rows=RING_SPEC)
    for b in range(B):
        pool.admit(b, ctx)
    pool.sync_page_table()
    torch.manual_seed(0)
    xs = [torch.randn(B, n, 5120).to(torch.bfloat16) for _ in range(nblk)]
    sources = {}
    built = []
    zero_state = lambda r: (torch.zeros(B, r, 512), torch.full((B, r, 512), -1e9))
    for slot, L in enumerate(layer_ids):
        w = load_layer(L, with_moe=False, max_seq_len=ctx + 64)
        meta = w["meta"]
        kw = dict(users_per_row=U, n=n)
        if meta["ratio"] == 0:
            O = DSV41Attention(md, mc, ccl, w["attn"], w["freqs_cis"], users_per_row=U, max_seq=ctx)
            O.load_window(torch.zeros(B, 0, 512))
            A = SpecWindowAttention(md, mc, ccl, w["attn"], w["freqs_cis"], max_seq=ctx, **kw)
            A.load_window(torch.zeros(B, 0, 512))
            Bp = SpecPagedWindowAttention(md, mc, ccl, w["attn"], w["freqs_cis"], pool, slot, **kw)
        elif meta["is_kv_source"]:
            r = meta["ratio"]
            kvs, scs = zero_state(r)
            O = DSV41CompressedAttention(
                md, mc, ccl, w["attn"], w["freqs_cis"], r, w["compressor"], max_comp=128, users_per_row=U
            )
            O.load_state(torch.zeros(B, 128, 512), torch.zeros(B, 0, 512), kvs, scs, start_pos=0)
            A = SpecCompressedAttention(md, mc, ccl, w["attn"], w["freqs_cis"], r, w["compressor"], max_comp=128, **kw)
            A.load_state(torch.zeros(B, 128, 512), torch.zeros(B, 0, 512), kvs, scs, S=0)
            Bp = SpecPagedCompressedAttention(
                md, mc, ccl, w["attn"], w["freqs_cis"], r, w["compressor"], pool, slot, L, **kw
            )
            Bp.init_state(B)
            sources[L] = (A, Bp, O)
        else:
            sA, sB, sO = sources[meta["kv_source"]]
            r = meta["ratio"]
            O = DSV41CompressedAttention(
                md, mc, ccl, w["attn"], w["freqs_cis"], r, None, max_comp=128, users_per_row=U, source=sO
            )
            O.load_state(torch.zeros(B, 128, 512), None, None, None)
            A = SpecCompressedAttention(md, mc, ccl, w["attn"], w["freqs_cis"], r, None, max_comp=128, source=sA, **kw)
            A.load_state(torch.zeros(B, 128, 512), None, None, None, S=0)
            Bp = SpecPagedCompressedAttention(
                md, mc, ccl, w["attn"], w["freqs_cis"], r, None, pool, slot, meta["kv_source"], source=sB, **kw
            )
        built.append(
            (
                L,
                A,
                Bp,
                SpecStepState(A, max_pos=ctx),
                DSV41PagedStepState(Bp, max_pos=ctx + 64),
                O,
                DSV41StepState(O, max_pos=256),
            )
        )
    toD = lambda x: ttnn.from_torch(
        x.reshape(1, 1, -1, 5120),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols)),
    )
    host = lambda t: torch.cat(
        [ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).reshape(-1, 5120) for r in range(rows)]
    ).float()
    posd = lambda p: ttnn.from_torch(
        p.to(torch.int32),
        device=md,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
    )
    onehot = lambda: ttnn.from_torch(
        torch.nn.functional.one_hot(torch.full((B,), n - 1), n).float().reshape(B, n, 1),
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
    )
    worst = {L: 1.0 for L in layer_ids}
    worst_np = {L: 1.0 for L in layer_ids}
    toO = lambda x: ttnn.from_torch(
        x.reshape(1, 1, -1, 5120),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols)),
    )
    for blk in range(nblk):
        pos = (torch.full((B, 1), blk * n) + torch.arange(n).reshape(1, n)).reshape(-1)
        x = xs[blk].reshape(B * n, 5120)
        for (
            L,
            A,
            Bp,
            ssA,
            ssB,
            O,
            ssO,
        ) in built:  # lockstep over layers so readers see the owner's latent of the same step
            oo = torch.stack(
                [host(O.forward(toO(xs[blk][:, j]), ssO.build(posd(torch.full((B,), blk * n + j))))) for j in range(n)],
                1,
            )
            oa = host(A.forward(toD(x), ssA.build(posd(pos)))).reshape(B, n, 5120)
            ob = host(Bp.forward(toD(x), ssB.build(posd(pos)))).reshape(B, n, 5120)
            A.commit()
            Bp.commit()
            pj = [R.pcc(ob[:, j], oo[:, j]) for j in range(n)]
            pn = [R.pcc(oa[:, j], oo[:, j]) for j in range(n)]
            worst[L] = min(worst[L], min(pj))
            worst_np[L] = min(worst_np[L], min(pn))
            print(
                f"layer {L} n={n} block {blk}: PCC vs ORIGINAL single-token: paged {[round(p, 5) for p in pj]} | non-paged spec {[round(p, 5) for p in pn]}",
                flush=True,
            )
    print("WORST non-paged spec vs original:", {k: round(v, 5) for k, v in worst_np.items()}, flush=True)
    print("WORST per layer:", {k: round(v, 5) for k, v in worst.items()}, flush=True)
    assert min(worst.values()) > float(os.environ.get("DSV41_MIN_PCC", "0.998")), worst
