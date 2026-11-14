# prefill-roof X19 follow-up: pieces + config sweep of the shared-expert prefill chunk (layer 1 real weights, T=512, TP8 bfp8, 1x8 slot). Prints SH json lines.
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, dev0, pcc, rep, traced_us
from models.demos.blackhole.hy4_preview.tt import hy4_dense as D
from models.demos.blackhole.hy4_preview.tt.hy4_pf_opt import PfOpt, make_pc

T = 512
acts = torch.load(ACTS + "acts_l0_l1.pt")
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=200_000_000)
P = lambda **k: print("SH " + json.dumps(k), flush=True)
try:
    g = torch.load(ACTS + "l1_gate_shared.pt")
    t = lambda m: m.T.contiguous().to(torch.bfloat16)[None, None]
    s0, s1, s2 = t(g["mlp.shared_experts.gate_proj.weight"]), t(g["mlp.shared_experts.up_proj.weight"]), t(g["mlp.shared_experts.down_proj.weight"])
    X = torch.cat([h.reshape(-1, 6144) for h, y, r in acts["moe"]])[:T]
    xs = rep(md, X.to(torch.bfloat16).reshape(1, 1, T, 6144))
    gg = (X.float() @ g["mlp.shared_experts.gate_proj.weight"].T).clamp(max=10.0)
    uu = (X.float() @ g["mlp.shared_experts.up_proj.weight"].T).clamp(-10, 10)
    ref = (torch.nn.functional.silu(gg) * uu) @ g["mlp.shared_experts.down_proj.weight"].T
    sh = D.make_shared(md, s0, s1, s2, decode=False, pf_opt=PfOpt.parse("mm2d,hifi2"))
    kw = lambda pc: dict(dtype=ttnn.bfloat16, compute_kernel_config=sh.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG, **({"program_config": pc} if pc else {}))
    gu = ttnn.linear(xs, sh.w01, **kw(sh.pc1))
    P(piece="gu_linear", us=round(traced_us(md, lambda: ttnn.linear(xs, sh.w01, **kw(sh.pc1)), n=5, rep_=3), 1))
    def act():
        a_, b_ = gu[:, :, :, : sh.ic], gu[:, :, :, sh.ic :]
        return ttnn.multiply(a_, b_, input_tensor_a_activations=sh.act_a, input_tensor_b_activations=sh.act_b, dtype=sh.mid_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    P(piece="slice+act_mul", us=round(traced_us(md, act, n=5, rep_=3), 1))
    a_ = act()
    P(piece="down_linear", us=round(traced_us(md, lambda: ttnn.linear(a_, sh.w2, **kw(sh.pc2)), n=5, rep_=3), 1))
    mk = lambda gx, gy, bw, sw, M, K, N: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(gx, gy), in0_block_w=bw, out_subblock_h=1, out_subblock_w=sw,
            per_core_M=(M // 32) // gy, per_core_N=(N // 32) // gx, transpose_mcast=False, fused_activation=None, fuse_batch=True)
    for (gx, gy, bw, sw) in ((8, 8, 16, 2), (8, 8, 32, 2), (8, 8, 8, 2), (4, 8, 16, 4), (8, 4, 16, 2), (2, 8, 16, 4), (16 // 2, 8, 24, 1)):
        try:
            pc = mk(gx, gy, bw, sw, T, 6144, 512)
            P(piece="gu_sweep", cfg=[gx, gy, bw, sw], us=round(traced_us(md, lambda: ttnn.linear(xs, sh.w01, **kw(pc)), n=5, rep_=3), 1))
        except Exception as e:
            P(piece="gu_sweep", cfg=[gx, gy, bw, sw], err=str(e).split(chr(10))[0][:80])
    for (gx, gy, bw, sw) in ((12, 8, 8, 4), (12, 4, 8, 4), (8, 8, 8, 4), (6, 8, 8, 4), (12, 8, 4, 4), (12, 8, 2, 4), (4, 8, 8, 4)):
        try:
            pc = mk(gx, gy, bw, sw, T, 256, 6144)
            P(piece="down_sweep", cfg=[gx, gy, bw, sw], us=round(traced_us(md, lambda: ttnn.linear(a_, sh.w2, **kw(pc)), n=5, rep_=3), 1))
        except Exception as e:
            P(piece="down_sweep", cfg=[gx, gy, bw, sw], err=str(e).split(chr(10))[0][:80])
finally:
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
