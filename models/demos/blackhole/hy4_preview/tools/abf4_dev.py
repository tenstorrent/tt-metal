# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""attn-bfp4 task, device part on ONE 1x8 slot (TP8 shards, rows = 1). MODE = proj | mlp | block.

  proj : every attention / indexer projection at bf16 | bfp8 | bfp4 weights on the REAL layer-1 golden activations: output PCC vs the fp32 CPU product
         (decode = 32 rows (tile), prefill = 512 rows) and traced matmul latency (default ttnn.linear program config; CCL not included, it is dtype independent).
  mlp  : shared expert (layer-1 acts) and dense layer 0 (layer-0 acts), TP8 DSV41SharedExpertColTP at the three dtypes: PCC vs the oracle and traced latency.
  block: Hy4Attention (layer-attn implementation) layer L prefill chunks vs the golden attention output, with per-tensor weight dtypes (VARIANT env).
Run with the slot env (bin/slot_env.sh) set; python from the main tree python_env, PYTHONPATH = this worktree.
"""
import json
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt.hy4_common import compute_cfg, upload

GOLD = "/mnt/tt-data/ssinghal/hy4-logs/layer-attn/golden"
ATTN_DUMP = "/mnt/tt-data/ssinghal/hy4-logs/program/layer-attn/dump_L1_chunk0.pt"
ACTS = "/mnt/tt-data/ssinghal/hy4-logs/program/layer-moe/acts/"
CKPT = "/mnt/tt-data/Hy4-Preview"
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
MODE = os.environ.get("MODE", "proj")
L = int(os.environ.get("LAYER", "1"))


def P(*a):
    print("ABF4", *a, flush=True)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def traced_us(md, fn, n=7, rep_=10):
    def body():
        for _ in range(rep_):
            o = fn()
            for t in o if isinstance(o, (tuple, list)) else [o]:
                if isinstance(t, ttnn.Tensor):
                    ttnn.deallocate(t)

    body()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    body()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / rep_ * 1e6)
    ttnn.release_trace(md, tid)
    return statistics.median(ts)


def open_md():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)


def close_md(md):
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def host(t, md, dim):
    """gather per-device shards: concat along dim."""
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=dim)).float()


def cfg_1d(K, N, max_cores):
    """DSV4 attention.py _cfg_1d: 1D multicast program config for one tile-row activation, N split over <= max_cores cores."""
    Kt, Nt = K // 32, N // 32
    pcn = -(-Nt // max_cores)
    n = -(-Nt // pcn)
    gx, gy = min(n, 8), -(-n // 8)
    sw = max(d for d in range(1, min(pcn, 4) + 1) if pcn % d == 0)
    ibw = max(d for d in range(1, 9) if Kt % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=sw,
        per_core_M=1,
        per_core_N=pcn,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def run_proj(md):
    from models.demos.blackhole.hy4_preview.config import Hy4Config
    from models.demos.blackhole.hy4_preview.tt.hy4_attn_weights import build_attn_host_weights

    cfg = Hy4Config.from_checkpoint(CKPT)
    from safetensors import safe_open

    wm = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    pre = f"model.layers.{L}."
    sd = {}
    for k, f in wm.items():
        if k.startswith(pre + "self_attn."):
            with safe_open(f"{CKPT}/{f}", "pt") as fh:
                sd[k[len(pre) :]] = fh.get_tensor(k)
    hw = build_attn_host_weights(cfg, sd, 8)
    G = torch.load(f"{GOLD}/L{L}.pt", map_location="cpu")
    ol = torch.load(ATTN_DUMP, map_location="cpu")["o_lat_c0"][0]  # [8,512,512]
    MF = ttnn.MathFidelity
    CK = {
        f: ttnn.init_device_compute_kernel_config(
            md.arch(), math_fidelity=m, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        for f, m in (("hifi4", MF.HiFi4), ("hifi2", MF.HiFi2), ("lofi", MF.LoFi))
    }
    COMBOS = [
        ("bf16", "hifi4"),
        ("bfp8", "hifi4"),
        ("bfp8", "hifi2"),
        ("bfp4", "hifi4"),
        ("bfp4", "hifi2"),
        ("bfp4", "lofi"),
    ]
    COMBOS_DEC = [("bfp8", "hifi4"), ("bfp8", "hifi2"), ("bfp4", "hifi4"), ("bfp4", "lofi")]
    out = {}
    for n_rows, tag in ((32, "decode32"), (512, "prefill512")):
        if os.environ.get("ONLY") and tag != os.environ["ONLY"]:
            continue
        h, qa = G["h"][:n_rows], G["qa"][:n_rows]
        og = (G["o"] * G["gate"])[:n_rows]
        qn = G["q"][:n_rows, :, :192].permute(1, 0, 2)[None]  # [1,64,n,192]
        olat = ol[:, :n_rows].repeat(8, 1, 1)[None]  # [1,64,n,512] (heads 0..7 repeated)
        # name -> (input tensor [1,1,n,K] or [1,H,n,K], input shard dim, weight, out concat dim or 'sum', batched)
        cases = {
            "q_a": (h[None, None], None, "q_a", 3),
            "q_b": (qa[None, None], None, "q_b", 3),
            "kv_a": (h[None, None], None, "kv_a", None),
            "w_uk": (qn, 1, "w_uk", 1),
            "w_uv_t": (olat, 1, "w_uv_t", 1),
            "gate": (h[None, None], None, "gate", 3),
            "o_proj": (og[None, None], 3, "o_proj", "sum"),
            "idx_wq_b": (qa[None, None], None, "idx_wq_b", None),
            "idx_wk": (h[None, None], None, "idx_wk", None),
            "idx_wp": (h[None, None], None, "idx_wp", None),
        }
        for name, (x, xs, wn, od) in cases.items():
            if wn not in hw:
                continue
            W = hw[wn].tensor.to(torch.bfloat16).float()
            Wf = W
            xb = x.to(torch.bfloat16).float()
            ref = torch.matmul(xb, Wf)  # fp32 on the bf16-rounded weights = checkpoint
            wcache = {}
            batched = x.shape[1] > 1
            Kl = hw[wn].tensor.shape[-2] // (8 if hw[wn].shard_dim == 2 else 1)
            Nl = hw[wn].tensor.shape[-1] // (8 if hw[wn].shard_dim == 3 else 1)
            if n_rows == 32 and not batched:
                COMBOS_N = [(f, d, c) for f, d in COMBOS_DEC for c in ("def", "1d64", "1d32")]
            else:
                COMBOS_N = [(f, d, "def") for f, d in COMBOS]
            for fmt, fid, pc in COMBOS_N:
                ckc = CK[fid]
                if fmt not in wcache:
                    wcache[fmt] = upload(md, hw[wn].tensor, DT[fmt], ttnn.TILE_LAYOUT, shard_dim=hw[wn].shard_dim)
                w = wcache[fmt]
                tx = upload(md, x.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, shard_dim=xs)
                kw = {} if pc == "def" else {"program_config": cfg_1d(Kl, Nl, int(pc[2:]))}
                mm = (
                    (lambda: ttnn.matmul(tx, w, compute_kernel_config=ckc))
                    if batched
                    else (lambda: ttnn.linear(tx, w, compute_kernel_config=ckc, **kw))
                )
                y = mm()
                ttnn.synchronize_device(md)
                if od is None:  # replicated: device 0
                    yh = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float()
                elif od == "sum":
                    yh = (
                        host(y, md, 0).reshape(8, *y.shape[1:] if len(y.shape) == 4 else y.shape).sum(0, keepdim=True)
                        if False
                        else ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
                        .float()
                        .sum(0, keepdim=True)
                    )
                else:
                    yh = host(y, md, od)
                ttnn.deallocate(y)
                r = pcc(yh, ref if od != 1 else ref)
                us = traced_us(md, mm)
                out[f"{tag}.{name}.{fmt}.{fid}.{pc}"] = [r, us]
                P(
                    f"proj {tag} {name:9s} {fmt} {fid} {pc}: pcc {r:.6f} traced {us:7.1f} us  W{tuple(hw[wn].tensor.shape)} shard {hw[wn].shard_dim}"
                )
                ttnn.deallocate(tx)
            for w in wcache.values():
                ttnn.deallocate(w)
    json.dump(out, open(os.environ["OUT"], "w"), indent=1)


def run_mlp(md):
    from models.demos.blackhole.hy4_preview.tt import hy4_dense as D

    acts = torch.load(ACTS + "acts_l0_l1.pt")
    ckc = compute_cfg(md, "hifi4", True)
    out = {}
    t = lambda m: m.T.contiguous().to(torch.bfloat16)[None, None]
    wd = torch.load(ACTS + "l0_dense.pt")
    ws = torch.load(ACTS + "l1_gate_shared.pt")
    Xd = torch.cat([h.reshape(-1, 6144) for h, y in acts["dense"]])
    Yd = torch.cat([y.reshape(-1, 6144) for h, y in acts["dense"]])
    longest = max(acts["dense"], key=lambda hy: hy[0].shape[1])
    Xs = torch.cat([h.reshape(-1, 6144) for h, y, r in acts["moe"]])

    def sh_ref(x):
        gg = (x.float() @ ws["mlp.shared_experts.gate_proj.weight"].T).clamp(max=10.0)
        uu = (x.float() @ ws["mlp.shared_experts.up_proj.weight"].T).clamp(-10, 10)
        return (torch.nn.functional.silu(gg) * uu) @ ws["mlp.shared_experts.down_proj.weight"].T

    for which in ("shared", "dense"):
        for decode, T in ((True, 32), (True, 4), (False, 512)):
            for fmt in ("bfp8", "bfp4"):
                if which == "dense":
                    mk = D.make_dense
                    w = (t(wd["mlp.gate_proj.weight"]), t(wd["mlp.up_proj.weight"]), t(wd["mlp.down_proj.weight"]))
                    src = (Xd, Yd) if T <= 32 else (longest[0].reshape(-1, 6144), longest[1].reshape(-1, 6144))
                    x = src[0][:T].to(torch.bfloat16)
                    ref = src[1][:T]
                else:
                    mk = D.make_shared
                    w = (
                        t(ws["mlp.shared_experts.gate_proj.weight"]),
                        t(ws["mlp.shared_experts.up_proj.weight"]),
                        t(ws["mlp.shared_experts.down_proj.weight"]),
                    )
                    x = Xs[:T].to(torch.bfloat16)
                    ref = sh_ref(x)
                mod = mk(md, *w, decode=decode, dtype=DT[fmt])
                tx = upload(md, x.reshape(1, 1, T, 6144))
                part = mod.forward(tx)
                ttnn.synchronize_device(md)
                parts = (
                    ttnn.to_torch(part, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
                    .float()
                    .sum(0)
                    .reshape(T, 6144)
                )
                r = pcc(parts, ref)
                us = traced_us(md, lambda: mod.forward(tx), n=7, rep_=5)
                out[f"{which}.T{T}.{'dec' if decode else 'pf'}.{fmt}"] = [r, us]
                P(f"mlp {which} T={T} decode={decode} {fmt}: pcc {r:.6f} partial-matmuls traced {us:.1f} us")
                ttnn.deallocate(tx)
                del mod
    json.dump(out, open(os.environ["OUT"], "w"), indent=1)


def run_block(md):
    from models.demos.blackhole.hy4_preview.config import Hy4Config
    from models.demos.blackhole.hy4_preview.tests import run_attn_device as R
    from models.demos.blackhole.hy4_preview.tt.hy4_attention import Hy4Attention
    from models.demos.gpt_oss.config import mesh_1x8
    from models.demos.gpt_oss.tt.ccl import CCLManager

    cfg = Hy4Config.from_checkpoint(CKPT)
    R.CKPT, R.L = CKPT, L
    g = torch.load(f"{GOLD}/L{L}.pt")
    conv = g["conv"]
    mc = mesh_1x8()
    ccl = CCLManager(md, num_links=1, topology=ttnn.Topology.Linear)
    sd = R.load_sd(L)
    VAR = os.environ.get("VARIANT", "bfp8_all")
    allf = ["q_a", "q_b", "kv_a", "w_uk", "w_uv_t", "gate", "o_proj", "idx_wq_b", "idx_wk", "idx_wp"]
    spec = {"bf16_all": {}, "bfp8_all": {k: "bfp8" for k in allf}, "bfp4_all": {k: "bfp4" for k in allf}}
    base = spec["bfp8_all"]
    if VAR.startswith("mix:"):  # mix:bfp4=gate,q_b,...;rest bfp8
        d = dict(base)
        for part in VAR[4:].split(";"):
            f, names = part.split("=")
            for n in names.split(","):
                d[n] = f
        spec[VAR] = d
    wdt = {k: DT[v] for k, v in spec[VAR].items()}
    attn = Hy4Attention(
        md,
        mc,
        ccl,
        cfg,
        sd,
        L,
        kv_fmt=os.environ.get("KVFMT", "bf16"),
        rope_interleave=conv["rope_interleave"],
        idx_rope_interleave=conv["idx_rope_interleave"],
        wdtype=ttnn.bfloat16,
        wdtypes=wdt,
    )
    R.WDT, R.KVFMT = ttnn.bfloat16, os.environ.get("KVFMT", "bf16")
    R.NCH = int(os.environ.get("CHUNKS", "6"))
    P("block variant", VAR, {k: str(v) for k, v in spec[VAR].items()})
    R.run_prefill(md, attn, g, cfg)


def run_ccl(md):
    """8-chip-axis collectives on activation-sized payloads (32 tile rows): proxy / upper bound for the 4-chip ROW-axis collectives of the 32-way sharding variant."""
    out = {}
    topo = ttnn.Topology.Linear
    for width in (256, 2048, 6144):
        x = upload(
            md, torch.randn(1, 1, 32, width * 8).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, shard_dim=3
        )  # per device [1,1,32,width]
        ag = lambda: ttnn.all_gather(
            x, dim=3, cluster_axis=1, num_links=1, topology=topo, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        y = ag()
        ttnn.synchronize_device(md)
        ttnn.deallocate(y)
        us = traced_us(md, ag, n=7, rep_=10)
        out[f"all_gather_w{width}"] = us
        P(f"ccl all_gather per-device [32,{width}] x8 -> [32,{width*8}]: {us:.1f} us")
        rs = lambda: ttnn.reduce_scatter(
            x, dim=3, cluster_axis=1, num_links=1, topology=topo, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        y = rs()
        ttnn.synchronize_device(md)
        ttnn.deallocate(y)
        us = traced_us(md, rs, n=7, rep_=10)
        out[f"reduce_scatter_w{width}"] = us
        P(f"ccl reduce_scatter per-device [32,{width}]: {us:.1f} us")
        ttnn.deallocate(x)
    json.dump(out, open(os.environ["OUT"], "w"), indent=1)


def main():
    md = open_md()
    try:
        {"proj": run_proj, "mlp": run_mlp, "block": run_block, "ccl": run_ccl}[MODE](md)
    finally:
        close_md(md)


if __name__ == "__main__":
    main()
