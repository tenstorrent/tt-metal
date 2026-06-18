# SPDX-License-Identifier: Apache-2.0
"""TP gated-GQA attention decode on a 1x8 T3K mesh: shard heads across 8 chips
(q 24->3, kv 4->1 with each kv-head REPLICATED across 2 chips), o_proj row-parallel +
all_reduce. Validated vs host reference. Contiguous q-head sharding keeps GQA groups
aligned: chip i holds q-heads 3i..3i+3 (GQA group i//2) and must use kv-head i//2, so the
4 kv-heads are repeat_interleave'd x2 into 8 slots [0,0,1,1,2,2,3,3] before dim-3 sharding."""
import torch
import ttnn

TP = 8


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def rms(x, w, eps=1e-6):  # per last-dim
    v = x.float().pow(2).mean(-1, keepdim=True)
    return ((x * torch.rsqrt(v + eps)) * (w + 1.0)).to(x.dtype)


def rope(q, k, cos, sin, rdim):
    def rot(x):
        x1, x2 = x[..., :rdim // 2], x[..., rdim // 2:rdim]
        return torch.cat([-x2, x1], dim=-1)
    qr, kr = q[..., :rdim], k[..., :rdim]
    q = torch.cat([qr * cos + rot(qr) * sin, q[..., rdim:]], -1)
    k = torch.cat([kr * cos + rot(kr) * sin, k[..., rdim:]], -1)
    return q, k


def main():
    H, nh, nkv, hd, rdim = 5120, 24, 4, 256, 64
    groups = nh // nkv
    scale = hd ** -0.5
    L = 4
    torch.manual_seed(0)
    Wq = torch.randn(nh * hd * 2, H) * 0.02
    Wk = torch.randn(nkv * hd, H) * 0.02
    Wv = torch.randn(nkv * hd, H) * 0.02
    Wo = torch.randn(H, nh * hd) * 0.02
    qn = torch.randn(hd) * 0.1
    kn = torch.randn(hd) * 0.1
    x = torch.randn(1, 1, 1, H) * 0.3
    pk = torch.randn(1, nkv, L, hd) * 0.3   # past keys (post-rope), per kv-head
    pv = torch.randn(1, nkv, L, hd) * 0.3
    cos = torch.randn(1, 1, 1, rdim) * 0.5  # arbitrary but fixed rope mats
    sin = torch.randn(1, 1, 1, rdim) * 0.5

    # ---- host reference (full 24/4 heads) ----
    q = (x @ Wq.T).view(1, 1, nh, hd * 2)
    query, gate = q[..., :hd], q[..., hd:]
    key = (x @ Wk.T).view(1, 1, nkv, hd)
    val = (x @ Wv.T).view(1, 1, nkv, hd)
    query = rms(query, qn); key = rms(key, kn)
    query, key = rope(query, key, cos, sin, rdim)
    q4 = query.permute(0, 2, 1, 3)                 # [1,nh,1,hd]
    k4 = key.permute(0, 2, 1, 3)   # [1,nkv,1,hd] (no past)
    v4 = val.permute(0, 2, 1, 3)
    k4e = k4.repeat_interleave(groups, dim=1); v4e = v4.repeat_interleave(groups, dim=1)
    aw = torch.softmax((q4.float() @ k4e.float().transpose(-1, -2)) * scale, dim=-1)
    ao = (aw @ v4e.float()).to(x.dtype).permute(0, 2, 1, 3).reshape(1, 1, 1, nh * hd)
    ao = ao * torch.sigmoid(gate.reshape(1, 1, 1, nh * hd).float()).to(ao.dtype)
    ref = ao @ Wo.T

    # ---- TP on mesh ----
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, TP))
    sh = (1, TP)
    SH3 = lambda d: ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=sh)
    SH2 = lambda: ttnn.ShardTensor2dMesh(md, dims=(None, 2), mesh_shape=sh)
    REP = lambda: ttnn.ReplicateTensorToMesh(md)
    f = lambda t, mm, dt=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=mm)
    try:
        nhp = nh // TP                  # 3 query heads/chip
        # GQA with nkv (4) < TP (8): replicate each kv-head across TP//nkv (=2) chips.
        # chip i holds q-heads [3i,3i+3) (GQA group i//2) -> must use kv-head i//2.
        nkvp = max(1, nkv // TP)        # 1 kv-head/chip (replicated)
        rep = TP // nkv                 # 2 chips share each kv-head
        # build 8-slot kv weights [0,0,1,1,2,2,3,3] so dim-3 shard gives chip i kv-head i//2
        Wk_rep = Wk.view(nkv, hd, H).repeat_interleave(rep, dim=0).reshape(nkv * rep * hd, H)
        Wv_rep = Wv.view(nkv, hd, H).repeat_interleave(rep, dim=0).reshape(nkv * rep * hd, H)
        Wq_t = f(Wq.T.unsqueeze(0).unsqueeze(0), SH3(3), ttnn.bfloat8_b)  # col-parallel (heads)
        Wk_t = f(Wk_rep.T.unsqueeze(0).unsqueeze(0), SH3(3), ttnn.bfloat8_b)
        Wv_t = f(Wv_rep.T.unsqueeze(0).unsqueeze(0), SH3(3), ttnn.bfloat8_b)
        Wo_t = f(Wo.T.unsqueeze(0).unsqueeze(0), SH2(), ttnn.bfloat8_b)   # row-parallel
        xt = f(x, REP())
        cos_t = f(cos, REP()); sin_t = f(sin, REP())
        qn_t = f((qn + 1.0).view(1, 1, 1, hd), REP()); kn_t = f((kn + 1.0).view(1, 1, 1, hd), REP())

        qp = ttnn.linear(xt, Wq_t); kp = ttnn.linear(xt, Wk_t); vp = ttnn.linear(xt, Wv_t)
        q2 = ttnn.reshape(qp, [1, 1, nhp, hd * 2])
        qry = ttnn.slice(q2, [0, 0, 0, 0], [1, 1, nhp, hd])
        gt = ttnn.slice(q2, [0, 0, 0, hd], [1, 1, nhp, hd * 2])
        ky = ttnn.reshape(kp, [1, 1, nkvp, hd]); vl = ttnn.reshape(vp, [1, 1, nkvp, hd])
        qry = ttnn.rms_norm(qry, epsilon=1e-6, weight=qn_t)
        ky = ttnn.rms_norm(ky, epsilon=1e-6, weight=kn_t)

        def rope_tt(t, n):
            tr = ttnn.slice(t, [0, 0, 0, 0], [1, 1, n, rdim])
            tp = ttnn.slice(t, [0, 0, 0, rdim], [1, 1, n, hd])
            x1 = ttnn.slice(tr, [0, 0, 0, 0], [1, 1, n, rdim // 2])
            x2 = ttnn.slice(tr, [0, 0, 0, rdim // 2], [1, 1, n, rdim])
            rh = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
            tr = ttnn.add(ttnn.mul(tr, cos_t), ttnn.mul(rh, sin_t))
            return ttnn.concat([tr, tp], dim=-1)
        qry = rope_tt(qry, nhp); ky = rope_tt(ky, nkvp)

        qh = ttnn.permute(qry, [0, 2, 1, 3])   # [1,nhp,1,hd]
        kh = ttnn.permute(ky, [0, 2, 1, 3])
        vh = ttnn.permute(vl, [0, 2, 1, 3])
        kh = ttnn.repeat_interleave(kh, nhp // nkvp, dim=1)   # local 1 kv -> nhp (=3) q
        vh = ttnn.repeat_interleave(vh, nhp // nkvp, dim=1)
        att = ttnn.transformer.scaled_dot_product_attention(qh, kh, vh, is_causal=False, scale=scale)
        att = ttnn.permute(att, [0, 2, 1, 3])
        att = ttnn.reshape(att, [1, 1, 1, nhp * hd])
        gt = ttnn.reshape(gt, [1, 1, 1, nhp * hd])
        att = ttnn.mul(att, ttnn.sigmoid(gt))
        outp = ttnn.linear(att, Wo_t)                    # [1,1,1,H] partial
        out = ttnn.all_reduce(outp, cluster_axis=1, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(md)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))[0:1].float()
        print(f"[shapes] ref={tuple(ref.shape)} out={tuple(out_t.shape)}", flush=True)
        print(f"[PCC] TP attention vs host = {pcc(ref, out_t):.5f}", flush=True)
        print("PASS" if pcc(ref, out_t) > 0.98 else "FAIL", flush=True)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()
