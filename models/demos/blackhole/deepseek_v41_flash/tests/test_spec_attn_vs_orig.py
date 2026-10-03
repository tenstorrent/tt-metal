# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M1: spec attention (paged, blocks of n rows per user) vs the ORIGINAL single-token attention classes on the device, same weights, same state,
random unit-rms inputs: block outputs vs the original attention run step by step. Env: DSV41_N (default 2), DSV41_LAYERS ("0,2,3")."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.attention import DSV41Attention, DSV41CompressedAttention
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.spec_attention import SpecCompressedAttention, SpecWindowAttention
from models.demos.blackhole.deepseek_v41_flash.tt.spec_state import SpecStepState
from models.demos.blackhole.deepseek_v41_flash.tt.step_state import DSV41StepState
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-m")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@torch.no_grad()
def test_spec_attn_vs_orig(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    n = int(os.environ.get("DSV41_N", "2"))
    layer_ids = [int(x) for x in os.environ.get("DSV41_LAYERS", "0,2,3").split(",")]
    S = torch.load(os.path.join(CHAIN, "tokens.pt"))["prefill_tokens"].shape[1]
    U, B = 4, rows * 4
    nblk = 6 // n
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    torch.manual_seed(0)
    xs = [torch.randn(B, 5120).to(torch.bfloat16) for _ in range(nblk * n)]
    sources = {}
    shard = lambda dim: ttnn.ShardTensor2dMesh(md, dims=(dim, None), mesh_shape=(rows, cols))
    built = []
    for L in layer_ids:
        ref = torch.load(os.path.join(CHAIN, f"layer_{L}.pt"))
        w = load_layer(L, with_moe=False)
        st0, meta = ref["state"], w["meta"]
        sa = {}
        if meta["ratio"] == 0:
            A = DSV41Attention(md, mc, ccl, w["attn"], w["freqs_cis"], users_per_row=U, max_seq=256)
            A.load_window(st0["window"][:, :S])
            Bn = SpecWindowAttention(md, mc, ccl, w["attn"], w["freqs_cis"], users_per_row=U, n=n, max_seq=256)
            Bn.load_window(st0["window"][:, :S])
        else:
            if meta["is_kv_source"]:
                A = DSV41CompressedAttention(
                    md, mc, ccl, w["attn"], w["freqs_cis"], meta["ratio"], w["compressor"], users_per_row=U
                )
                A.load_state(st0["window"], st0["comp"], st0.get("kv_state"), st0.get("score_state"), start_pos=S)
                Bn = SpecCompressedAttention(
                    md, mc, ccl, w["attn"], w["freqs_cis"], meta["ratio"], w["compressor"], users_per_row=U, n=n
                )
                Bn.load_state(st0["window"], st0["comp"], st0.get("kv_state"), st0.get("score_state"), S=S)
                sources[L] = (A, Bn)
            else:
                sA, sB = sources[meta["kv_source"]]
                A = DSV41CompressedAttention(
                    md, mc, ccl, w["attn"], w["freqs_cis"], meta["ratio"], None, users_per_row=U, source=sA
                )
                A.load_state(st0["window"], None, None, None)
                Bn = SpecCompressedAttention(
                    md, mc, ccl, w["attn"], w["freqs_cis"], meta["ratio"], None, users_per_row=U, n=n, source=sB
                )
                Bn.load_state(st0["window"], None, None, None, S=S)
        built.append((L, meta, A, Bn, DSV41StepState(A), SpecStepState(Bn)))
    toA = lambda x: ttnn.from_torch(
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
    outA = {L: [] for L in layer_ids}
    for s in range(nblk * n):  # layers run in lockstep so a reader sees the owner's latent of the SAME step
        for L, meta, A, Bn, ssA, ssB in built:
            outA[L].append(host(A.forward(toA(xs[s]), ssA.build(posd(torch.full((B,), S + s))))))
    worst = 1.0
    for blk in range(nblk):
        x = torch.stack([xs[blk * n + j] for j in range(n)], 1).reshape(B * n, 5120)
        pos = (torch.full((B, 1), S + blk * n) + torch.arange(n).reshape(1, n)).reshape(-1)
        for L, meta, A, Bn, ssA, ssB in built:
            o = host(Bn.forward(toA(x), ssB.build(posd(pos)))).reshape(B, n, 5120)
            Bn.commit()
            pj = [R.pcc(o[:, j], outA[L][blk * n + j]) for j in range(n)]
            worst = min(worst, min(pj))
            print(
                f"layer {L} n={n} block {blk}: PCC spec vs orig per block index {[round(p, 5) for p in pj]}", flush=True
            )
    assert worst > float(os.environ.get("DSV41_MIN_PCC", "0.999")), worst
