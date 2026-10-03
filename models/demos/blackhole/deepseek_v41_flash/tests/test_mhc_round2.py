# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Round 2: expand with native-dtype y / y2, collapse_norm_rm second output, mixes variants (chunking, post fidelity)."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.mhc_ref_impl import DSV41MHC as Ref
from models.demos.blackhole.deepseek_v41_flash.tests.test_mhc_opt import chain_ms, first, pcc
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC as New

D, T = 5120, int(os.environ.get("MHC_T", "4"))
F = ttnn.MathFidelity


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
@torch.no_grad()
def test_mhc_round2(mesh_device):
    md = mesh_device
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt=ttnn.float32, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(
        t, device=md, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    fn, base, scale = torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5])
    ref = Ref(md, fn, base, scale)
    new = New(md, fn, base, scale)
    tt_w = up(torch.rand(1, 1, 1, D) + 0.5)

    def make_x(kind):
        if kind == "random":
            return torch.randn(T, 1, 4, D)
        x = torch.randn(T, 1, 1, D) + 0.3 * torch.randn(T, 1, 4, D)
        x[..., :4] *= 300.0
        x[..., 777] *= 800.0
        return x

    for kind in ("random", "realistic"):
        x = make_x(kind)
        tx = up(x)
        tpre = up(torch.softmax(torch.randn(T, 1, 1, 4), -1) * 2)
        rp, rpo, rc = ref.mixes(tx)
        # --- expand variants
        y_bf = torch.randn(1, 1, T, D).to(torch.bfloat16)
        y2 = torch.randn(1, 1, T, D)
        ty_bf, ty2 = up(y_bf, ttnn.bfloat16), up(y2)
        ty_row32 = up(y_bf.float())
        ty_tok = up(y_bf.float().reshape(T, 1, 1, D))
        exact_y = (y_bf.float() + y2).reshape(T, 1, 1, D)
        r_a = first(ref.expand(up(y_bf.float().reshape(T, 1, 1, D)), tx, rpo, rc))
        r_b = first(ref.expand(up(exact_y), tx, rpo, rc))
        res = {
            "exp_tok": pcc(first(new.expand(ty_tok, tx, rpo, rc)), r_a),
            "exp_row32": pcc(first(new.expand(ty_row32, tx, rpo, rc)), r_a),
            "exp_row_bf16": pcc(first(new.expand(ty_bf, tx, rpo, rc)), r_a),
            "exp_bf16+y2": pcc(first(new.expand(ty_bf, tx, rpo, rc, ty2)), r_b),
        }
        # --- collapse_norm_rm
        cn = first(new.collapse_norm(tx, tpre, tt_w))
        h, htok = new.collapse_norm_rm(tx, tpre, tt_w)
        res["cnrm_h"] = pcc(first(h), cn)
        res["cnrm_h_maxdiff"] = float((first(h) - cn).abs().max())
        res["cnrm_tok"] = pcc(first(htok).reshape(T, D), cn.reshape(T, D))
        res["cnrm_tok_maxdiff"] = float((first(htok).reshape(T, D) - cn.reshape(T, D)).abs().max())
        print(f"PCC[{kind}]", {k: round(v, 6) for k, v in res.items()}, flush=True)
        # --- mixes variants
        for kc, fid in ((640, F.HiFi4),):
            m = New(md, fn, base, scale, kc=kc, post_fidelity=fid)
            npre, npo, nc = m.mixes(tx)
            print(
                f"PCC[{kind}] mixes kc={kc} fid={fid}: pre {pcc(first(npre), first(rp)):.6f} post {pcc(first(npo), first(rpo)):.6f} comb {pcc(first(nc), first(rc)):.6f}",
                flush=True,
            )
            if kind == "random":
                print(f"TIME mixes kc={kc} fid={fid}: {chain_ms(md, lambda: m.mixes(tx)):.3f} ms", flush=True)

    from models.demos.blackhole.deepseek_v41_flash.tt.mhc_mixes import mhc_post, mhc_proj

    tx = up(make_x("realistic"))
    new.mixes(tx)
    wt = list(new._wts.values())[0]
    part = mhc_proj(tx, wt, new.mix_col)
    print(f"TIME proj alone {chain_ms(md, lambda: mhc_proj(tx, wt, new.mix_col)):.3f} ms")
    print(f"TIME post alone {chain_ms(md, lambda: mhc_post(part, new.consts10, T, 20, new._w.eps, new.sq_eps)):.3f} ms")
    tx = up(make_x("realistic"))
    tpre = up(torch.rand(T, 1, 1, 4))
    _, post, comb = new.mixes(tx)
    ty_bf, ty2 = up(torch.randn(1, 1, T, D).to(torch.bfloat16), ttnn.bfloat16), up(torch.randn(1, 1, T, D))
    # old glue: typecast + (add) + reshape + expand
    glue_a = lambda: new.expand(ttnn.reshape(ttnn.typecast(ty_bf, ttnn.float32), [T, 1, 1, D]), tx, post, comb)
    glue_f = lambda: new.expand(
        ttnn.reshape(ttnn.add(ttnn.typecast(ty_bf, ttnn.float32), ty2), [T, 1, 1, D]), tx, post, comb
    )
    print(f"TIME expand attn glue old   {chain_ms(md, glue_a):.3f} ms")
    print(f"TIME expand attn bf16 row   {chain_ms(md, lambda: new.expand(ty_bf, tx, post, comb)):.3f} ms")
    print(f"TIME expand ffn glue old    {chain_ms(md, glue_f):.3f} ms")
    print(f"TIME expand ffn bf16+y2     {chain_ms(md, lambda: new.expand(ty_bf, tx, post, comb, ty2)):.3f} ms")
    old_cn = lambda: (lambda h: ttnn.reshape(ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT), [T, 1, 1, D]))(
        new.collapse_norm(tx, tpre, tt_w)
    )
    print(f"TIME collapse_norm + to_layout/reshape {chain_ms(md, old_cn):.3f} ms")
    print(f"TIME collapse_norm alone               {chain_ms(md, lambda: new.collapse_norm(tx, tpre, tt_w)):.3f} ms")
    print(f"TIME collapse_norm_rm                  {chain_ms(md, lambda: new.collapse_norm_rm(tx, tpre, tt_w)):.3f} ms")
