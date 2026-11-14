"""mtp-device S1: the MTP block (Hy4MTP) on a 1x8 slot vs the oracle golden (spec/mtp_golden.py) on real post-norm hidden states.
Rows of a prompt are fed S at a time (positions p0..p0+S-1, MTP KV appended by the pool decode path) exactly as a verify round does.
Env: GOLDEN (pt), PROMPTS (comma idx), SS (rows per call, e.g. 1,4,8), PAGES, MAXCALLS, ZERO0 (1: zero the position-0 embedding). Prints RES json lines."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import pcc
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt.hy4_attn_opt import Opt
from models.demos.blackhole.hy4_preview.tt.hy4_chain import HID, Hy4Chain, dram_view
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_mtp import Hy4MTP
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def logits_argmax(ch, h):
    lg = ttnn.linear(h, ch.lm_head, compute_kernel_config=ch.ckc_head)
    t = ttnn.to_torch(lg, mesh_composer=ttnn.ConcatMesh2dToTensor(ch.md, dims=(0, 3), mesh_shape=tuple(ch.md.shape)))
    ttnn.deallocate(lg)
    return t.reshape(-1, t.shape[-1]).float().argmax(-1)


def main():
    gold = torch.load(os.environ["GOLDEN"])
    prompts = [int(x) for x in os.environ.get("PROMPTS", ",".join(str(k) for k in gold)).split(",")]
    SS = [int(x) for x in os.environ.get("SS", "1,4").split(",")]
    PAGES = int(os.environ.get("PAGES", "2"))
    maxcalls = int(os.environ.get("MAXCALLS", "1000"))
    full, md = open_row()
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        from models.demos.blackhole.deepseek_v41_flash.tt import ccl_lean

        ccl_lean.make_lean(mc)
        ccl_lean.make_lean_sems(ccl)
        t0 = time.time()
        ch = Hy4Chain(md, mc, ccl, os.environ["HY4_CKPT"])
        ch.build_head()
        ch.add_layers(78, 79)
        ly = ch.mtp_layers[0]
        from models.demos.blackhole.hy4_preview.spec.host_moe import HostRoutedMoE

        hm = HostRoutedMoE(os.environ["HY4_CKPT"], quant=os.environ.get("QUANT") or None)
        sd = ch.rd.layer_sd(-1)
        opt = Opt.from_env()
        res(event="built", sec=round(time.time() - t0, 1), dram=dram_view(md), opt=opt.tag())
        for S in SS:
            ctx = PoolCtx(md, mc, ccl, PAGES, rows=1, S=S, mode="dense", bucket_tokens=PAGES * 512)
            mtp = Hy4MTP(ch, ly, sd, opt, moe_fn=hm)
            for p in prompts:
                g = gold[p]
                ids, H = g["ids"], g["H"]
                n = ids.shape[0] - 1  # rows
                kvp = ttnn.empty(
                    [1, 1, PAGES * 64, 656],
                    dtype=ttnn.fp8_e4m3,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=md,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                kp = ttnn.empty(
                    [PAGES, 1, 64, 128],
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=md,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ly.attn.init_pool(ctx, kvp, kp)
                outs, tapsall, d1 = [], {}, []
                for call, p0 in enumerate(range(0, n - S + 1, S)):
                    if call >= maxcalls:
                        break
                    tok = upload(
                        md, ids[p0 + 1 : p0 + S + 1].reshape(1, S).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                    )
                    h = upload(md, H[p0 : p0 + S].reshape(1, 1, S, HID).bfloat16())
                    ctx.mode = (
                        "dense" if p0 + S <= 2048 else "topk"
                    )  # rows beyond 2048 keys select their top-2048 with the MTP layer's own indexer
                    ctx.set_step([p0])
                    tabs = ly.attn.rope_tabs_opt(p0, S)
                    dense8 = ctx.dense_ids8() if ctx.mode == "dense" else None
                    taps = {}
                    out = mtp.forward(tok, h, tabs, dense8, taps=taps)
                    ttnn.synchronize_device(md)
                    outs.append(ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(S, HID).float())
                    d1.append(logits_argmax(ch, out))
                    for k, v in taps.items():
                        tapsall.setdefault(k, []).append(v)
                    for t in (tok, h, out):
                        ttnn.deallocate(t)
                m = len(outs) * S
                dev = torch.cat(outs)
                rec = dict(
                    event="block", prompt=p, S=S, rows=m, calls=len(outs), pcc_out=round(pcc(dev, g["out"][:m]), 5)
                )
                per_row = [pcc(dev[i], g["out"][i]) for i in range(m)]
                rec["row_min"], rec["row_min_no0"] = round(min(per_row), 4), round(
                    min(per_row[1:]) if m > 1 else per_row[0], 4
                )
                if m > 2048 + S:
                    rec["pcc_out_topk_rows"] = round(pcc(dev[2048 + S :], g["out"][2048 + S : m]), 5)
                for k in ("x0", "a", "x1", "y", "x2"):
                    if k in tapsall:
                        rec["pcc_" + k] = round(pcc(torch.cat(tapsall[k]), g[k][:m]), 5)
                d1 = torch.cat(d1)
                rec["d1_eq_oracle"] = round(float((d1 == g["d1"][:m]).float().mean()), 4)
                if m > 2048 + S:
                    rec["d1_eq_oracle_topk_rows"] = round(
                        float((d1[2048 + S :] == g["d1"][2048 + S : m]).float().mean()), 4
                    )
                if (
                    "main" in g
                ):  # first-draft acceptance proxy: MTP argmax at row t == the main model's argmax at row t+1 (oracle fp32 logits of the dumped hidden states)
                    mm = min(m, g["main"].shape[0] - 1)
                    rec["alpha1_dev"] = round(float((d1[:mm] == g["main"][1 : mm + 1]).float().mean()), 4)
                    rec["alpha1_oracle"] = round(float((g["d1"][:mm] == g["main"][1 : mm + 1]).float().mean()), 4)
                    rec["n_alpha"] = mm
                rec["quant"] = os.environ.get("QUANT") or "bf16"
                if os.environ.get("DIAG_Y") == "1" and "h2" in tapsall:
                    h2 = torch.cat(tapsall["h2"])
                    ex = HostRoutedMoE(os.environ["HY4_CKPT"], exact=True)
                    r_ex = ex(h2)
                    from models.demos.blackhole.hy4_preview.reference.oracle import Conv, shared_expert

                    skey = {k: v.float() for k, v in ex.ck.layer_state(-1).items() if "shared" in k}
                    sh_ex = shared_expert(h2, skey, ex.cfg, Conv())
                    rec["diag_routed_dev_vs_exact"] = round(pcc(torch.cat(tapsall["routed"]), r_ex), 5)
                    rec["diag_shared_dev_vs_exact"] = round(pcc(torch.cat(tapsall["sh"]), sh_ex), 5)
                    rec["diag_y_exact_on_dev_h2_vs_oracle_y"] = round(pcc(r_ex + sh_ex, g["y"][:m]), 5)
                    rec["diag_routed_set_match"] = round(
                        float((ex.last_sel.sort(-1)[0] == g["routes"][:m].sort(-1)[0]).all(-1).float().mean()), 4
                    )
                res(**rec)
                ttnn.deallocate(kvp)
                ttnn.deallocate(kp)
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
