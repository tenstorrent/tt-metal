# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Debug: per 32-row block relative error of the prefill attention intermediates (q after RoPE, kv, SDPA out) vs a host float
computation from the same weights, for every user of mesh row 0 / column 0. Finds which stage corrupts which rows."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_prefill_attention_device import build_attn
from models.demos.blackhole.deepseek_v41_flash.tests.test_prefill_sdpa_probe import band_mask, ref_attn
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_attention import DSV41PrefillAttention, pad_len

S = int(os.environ.get("DSV41_S", "128"))
LAYER = int(os.environ.get("DSV41_LAYER", "0"))


def rms(x, w, eps=1e-20):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w


def blocks(got, ref, tag):
    """got/ref [S, ...]: relative error per 32-row block."""
    out = []
    for t in range(0, got.shape[0], 32):
        g, r = got[t : t + 32].float(), ref[t : t + 32].float()
        out.append(float((g - r).norm() / (r.norm() + 1e-9)))
    print(f"DBG {tag}: rel err per 32-row block " + " ".join(f"{v:.3f}" for v in out), flush=True)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(3000)
@torch.no_grad()
def test_prefill_attn_debug(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    d = f"/mnt/tt-data/ssinghal/dsv4-prefill-s{S}"
    ref = torch.load(os.path.join(d, f"layer_{LAYER}.pt"), mmap=True)
    x_all = ref["prefill"]["attn_in"].float()
    w = load_layer(LAYER, with_moe=False, max_seq_len=256)
    attn = build_attn(md, LAYER, w, 256)
    pa = DSV41PrefillAttention(attn, w["attn"]["attn_sink"])
    pa.tap = {}
    Sp = pad_len(S)
    xin = torch.zeros(rows * 4, Sp, 5120)
    xin[:, :S] = x_all
    h = ttnn.from_torch(
        xin.reshape(rows, 1, 4 * Sp, 5120),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
    )
    out = pa.forward(h, S)
    ttnn.synchronize_device(md)
    dev0 = lambda t, r=0, c=0: ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols + c]).float()
    A = w["attn"]
    c_, s_ = attn._rope_inputs(torch.arange(S))
    rope = lambda t: t * c_ + (t @ _pf()) * s_

    def _pf():
        from models.demos.blackhole.deepseek_v41_flash.tt.attention import full_pair_swap

        return full_pair_swap()

    for r, col in [(0, 0), (1, 3)]:
        qd, kvd, od = dev0(pa.tap["qh"], r, col), dev0(pa.tap["kv"], r, col), dev0(pa.tap["o_raw"], r, col)
        for u in range(4):
            x = x_all[r * 4 + u]
            y = x @ A["wq_a"].float().T
            qr = rms(y, A["q_norm"].float())
            q = (qr @ A["wq_b"].float().T).reshape(S, 64, 512)[:, col * 8 : (col + 1) * 8]  # [S,8,512]
            q = rope(q.permute(1, 0, 2))  # [8,S,512]
            kv = rope(rms(x @ A["wkv"].float().T, A["kv_norm"].float()))  # [S,512]
            sink = A["attn_sink"].float()[col * 8 : (col + 1) * 8]
            o = ref_attn(q[None], kv[None, None], sink, band_mask(S))[0]  # [8,S,512]
            print(f"--- row {r} col {col} user {u}", flush=True)
            blocks(qd[u, :, :S].permute(1, 0, 2).reshape(S, -1), q.permute(1, 0, 2).reshape(S, -1), "q (after rope)")
            blocks(kvd[u, 0, :S], kv, "kv")
            blocks(od[u, :, :S].permute(1, 0, 2).reshape(S, -1), o.permute(1, 0, 2).reshape(S, -1), "sdpa out")

    # ---- stage 2: after SDPA (inverse rope, head concat, o-projection, all-reduce) for mesh row 0, column 0 ----
    from models.demos.blackhole.deepseek_v41_flash.tt.attention import full_pair_swap as _fp

    P = _fp()
    wo_a = A["wo_a"].float().reshape(8, 1024, 4096)
    wo_b = A["wo_b"].float()
    fin = dev0(out, 0, 0).reshape(4, Sp, 5120)
    cdev = dev0(pa.tap["c"], 0, 0).reshape(4, Sp, -1)
    pdev = dev0(pa.tap["part"], 0, 0).reshape(4, Sp, 5120)
    for u in range(4):
        x = x_all[u]
        qr = rms(x @ A["wq_a"].float().T, A["q_norm"].float())
        q = (qr @ A["wq_b"].float().T).reshape(S, 64, 512)[:, :8].permute(1, 0, 2)
        q = q * c_ + (q @ P) * s_
        kv = rms(x @ A["wkv"].float().T, A["kv_norm"].float())
        kv = kv * c_ + (kv @ P) * s_
        o = ref_attn(q[None], kv[None, None], A["attn_sink"].float()[:8], band_mask(S))[0]  # [8,S,512]
        o = o * c_ + (o @ P) * (-s_)  # inverse rope
        ch = o.permute(1, 0, 2).reshape(S, 4096)
        c_host = torch.cat([torch.zeros(S, 512), ch], dim=1)
        part_host = (ch @ wo_a[0].T) @ wo_b[:, :1024].T
        print(f"--- stage2 user {u}", flush=True)
        blocks(cdev[u, :S], c_host, "c (concat heads, device col 0)")
        blocks(pdev[u, :S], part_host, "part (o-proj, col 0, before all-reduce)")
        blocks(fin[u, :S], ref["prefill"]["attn_out"][u].float(), "final (after all-reduce) vs reference attn_out")

    # ---- all-reduce variants on the tapped partial sums ----
    ref_a = ref["prefill"]["attn_out"].float()
    R_ = pa.tap["part"].shape[2]

    def check(tag, t):
        ttnn.synchronize_device(md)
        f = dev0(t, 0, 0).reshape(4, Sp, 5120)
        bad = [
            (u, tb)
            for u in range(4)
            for tb in range(0, S, 32)
            if not (float((f[u, tb : tb + 32] - ref_a[u, tb : tb + 32]).norm() / ref_a[u, tb : tb + 32].norm()) < 0.05)
        ]
        print(f"DBG allreduce variant {tag}: bad blocks (user, row) {bad}", flush=True)

    for rep in range(2):
        check(f"full R={R_} (rep {rep})", attn.mesh_config.allreduce(ttnn.clone(pa.tap["part"]), attn.ccl, axis=1))
    for piece in (256, 128, 64, 32):
        parts = [
            attn.mesh_config.allreduce(
                ttnn.slice(pa.tap["part"], [0, 0, i, 0], [1, 1, i + piece, 5120]), attn.ccl, axis=1
            )
            for i in range(0, R_, piece)
        ]
        check(f"pieces of {piece} rows", ttnn.concat(parts, dim=2))
    check(
        "pad_size 32 (mesh_config option)",
        attn.mesh_config.allreduce(ttnn.clone(pa.tap["part"]), attn.ccl, axis=1, pad_size=0)
        if False
        else attn.mesh_config.allreduce(ttnn.clone(pa.tap["part"]), attn.ccl, axis=1),
    )
