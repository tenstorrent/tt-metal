# SPDX-License-Identifier: Apache-2.0
"""Run ONLY deltanet_decode_full N times (1x8 line, 6 heads/chip) under the device
profiler, so we can attribute the ~1.06ms kernel to reader(BRISC/NCRISC) vs compute(TRISC)."""
import torch, ttnn
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
TP = 8
cfg = Qwen36ModelConfig()
H = cfg.hidden_size
nv, nk, dk, dv = cfg.linear_num_value_heads, cfg.linear_num_key_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
Kc = cfg.linear_conv_kernel_dim
key_dim, val_dim = dk * nk, dv * nv
nvp, nkp = nv // TP, nk // TP
ratio = nv // nk
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, TP))
SH = lambda d: ttnn.ShardTensor2dMesh(md, dims=(None, d), mesh_shape=(1, TP))
REP = ttnn.ReplicateTensorToMesh(md)
rnd = lambda *s: torch.randn(*s) * 0.02
fS = lambda t, d: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=SH(d))
fR = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=REP)
try:
    x = fR(torch.randn(1, 1, 1, H) * 0.1)
    fW = lambda t, d: ttnn.from_torch(t.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=SH(d))
    Wqkv = rnd(key_dim * 2 + val_dim, H)
    Wq = fW(Wqkv[:key_dim].T, 3); Wk = fW(Wqkv[key_dim:2*key_dim].T, 3); Wv = fW(Wqkv[2*key_dim:].T, 3)
    Wz = fW(rnd(val_dim, H).T, 3); Wb = fW(rnd(nv, H).T, 3); Wa = fW(rnd(nv, H).T, 3)
    cblk = lambda c: ttnn.from_torch(c.unsqueeze(0).unsqueeze(0).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=SH(2))
    conv = ttnn.concat([cblk(rnd(key_dim, Kc)), cblk(rnd(key_dim, Kc)), cblk(rnd(val_dim, Kc))], dim=2)
    A = fS(rnd(nv).view(1,1,1,nv).to(torch.bfloat16), 3); dt = fS(rnd(nv).view(1,1,1,nv).to(torch.bfloat16), 3)
    nw = fR(torch.ones(1,1,1,dv)); rs = fS(torch.zeros(1,nv,dk,dv).to(torch.bfloat16), 1)
    cs = fS(torch.zeros(1,1,key_dim*2+val_dim,32).to(torch.bfloat16), 2)
    cdp = (key_dim*2+val_dim)//TP
    qkv = ttnn.concat([ttnn.linear(x, Wq), ttnn.linear(x, Wk), ttnn.linear(x, Wv)], dim=3)
    z = ttnn.linear(x, Wz); b = ttnn.linear(x, Wb); a = ttnn.linear(x, Wa)
    ttnn.synchronize_device(md)
    for _ in range(50):
        ttnn.experimental.deltanet_decode_full(qkv, z, b, a, cs, rs, conv, A, dt, nw,
            num_heads=nvp, num_k_heads=nkp, k_head_dim=dk, v_head_dim=dv, conv_dim=cdp, conv_kernel_size=Kc, head_expand_ratio=ratio)
    ttnn.synchronize_device(md)
    print("KERNEL_PROF_DONE", flush=True)
finally:
    ttnn.close_mesh_device(md)
