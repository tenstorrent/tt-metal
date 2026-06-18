# SPDX-License-Identifier: Apache-2.0
"""TP DeltaNet decode on a 1x8 T3K mesh. Shard heads (48 v-heads->6, 16 k-heads->2 per chip),
run deltanet_decode_full per-chip on its heads, out_proj row-parallel + all_reduce.
Reference = single-device full 48-head run (sequential open). Heads are independent in
the recurrence, so per-chip-6 + all_reduce(out_proj) == full-48."""
import torch
import ttnn
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.load_weights import create_dummy_state_dict
from models.demos.qwen36_27b.tt.deltanet import TtGatedDeltaNet, TtDeltaNetState

TP = 8


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    cfg = Qwen36ModelConfig()
    H = cfg.hidden_size
    torch.manual_seed(7)
    sd = create_dummy_state_dict(cfg, num_layers=1)
    x_t = torch.randn(1, 1, 1, H) * 0.1
    rs0 = torch.randn(1, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim) * 0.1

    # ---------- reference: single-device full 48-head deltanet ----------
    dev = ttnn.open_device(device_id=0)
    dn = TtGatedDeltaNet(dev, sd, 0, cfg)
    st = TtDeltaNetState(1, cfg.layer_types[:1], dev, cfg)
    st.recurrent_states[0] = ttnn.from_torch(rs0.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    xt = ttnn.from_torch(x_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ref = ttnn.to_torch(dn._decode_step_full_fused(xt, st)).float()
    ttnn.close_device(dev)
    print(f"[ref] full 48-head out shape {tuple(ref.shape)}", flush=True)

    # ---------- TP: 6 v-heads / 2 k-heads per chip ----------
    nk, nv, dk, dv = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    K = cfg.linear_conv_kernel_dim
    key_dim, val_dim = dk * nk, dv * nv
    nkp, nvp = nk // TP, nv // TP  # 2, 6
    ratio = nv // nk               # 3

    # raw torch weights from sd (model uses .T)
    Wqkv = sd["model.layers.0.linear_attn.in_proj_qkv.weight"]   # [conv_dim, H]
    Wz = sd["model.layers.0.linear_attn.in_proj_z.weight"]       # [val_dim, H]
    Wb = sd["model.layers.0.linear_attn.in_proj_b.weight"]       # [nv, H]
    Wa = sd["model.layers.0.linear_attn.in_proj_a.weight"]
    Wo = sd["model.layers.0.linear_attn.out_proj.weight"]        # [H, val_dim]
    conv_w = sd["model.layers.0.linear_attn.conv1d.weight"]      # [conv_dim,1,K]
    A_log = sd["model.layers.0.linear_attn.A_log"]               # [nv]
    dt = sd["model.layers.0.linear_attn.dt_bias"]
    nw = sd["model.layers.0.linear_attn.norm.weight"]            # [dv]
    # split fused qkv rows into q(k-head),k(k-head),v(v-head)
    Wq, Wk, Wv = Wqkv[:key_dim], Wqkv[key_dim:2 * key_dim], Wqkv[2 * key_dim:]
    cq, ck, cv = conv_w[:key_dim], conv_w[key_dim:2 * key_dim], conv_w[2 * key_dim:]

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, TP))
    sh = (1, TP)
    SH = lambda d: ttnn.ShardTensor2dMesh(md, dims=(None, d), mesh_shape=sh)
    REP = ttnn.ReplicateTensorToMesh(md)
    fT = lambda t, mm, dt_=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt_, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=mm)
    try:
        # projection weights: stored .T → [H, out]; shard out-dim (3) into 4 (=heads)
        Wq_t = fT(Wq.T.unsqueeze(0).unsqueeze(0), SH(3), ttnn.bfloat8_b)
        Wk_t = fT(Wk.T.unsqueeze(0).unsqueeze(0), SH(3), ttnn.bfloat8_b)
        Wv_t = fT(Wv.T.unsqueeze(0).unsqueeze(0), SH(3), ttnn.bfloat8_b)
        Wz_t = fT(Wz.T.unsqueeze(0).unsqueeze(0), SH(3), ttnn.bfloat8_b)
        Wb_t = fT(Wb.T.unsqueeze(0).unsqueeze(0), SH(3), ttnn.bfloat8_b)
        Wa_t = fT(Wa.T.unsqueeze(0).unsqueeze(0), SH(3), ttnn.bfloat8_b)
        Wo_t = fT(Wo.T.unsqueeze(0).unsqueeze(0), SH(2), ttnn.bfloat8_b)   # row-parallel (in=val_dim)
        xt = fT(x_t.to(torch.bfloat16), REP)

        # per-chip conv1d weight in [q|k|v] order: shard each block, concat on conv_dim
        def conv_block(c, n_heads_total):
            # c: [block_dim,1,K] -> [1,1,block_dim,K]; shard block_dim (dim2) by TP
            t = c.squeeze(1).unsqueeze(0).unsqueeze(0).permute(0, 1, 2, 3)  # [1,1,block,K]
            return fT(t.to(torch.bfloat16), SH(2))
        # concat per-chip happens on-device after sharding each block consistently
        cq_t = conv_block(cq, nk); ck_t = conv_block(ck, nk); cv_t = conv_block(cv, nv)
        conv_w_t = ttnn.concat([cq_t, ck_t, cv_t], dim=2)  # per chip [1,1,(4+4+12)*128? no, dims]

        A_t = fT(A_log.view(1, 1, 1, nv).to(torch.bfloat16), SH(3))
        dt_t = fT(dt.view(1, 1, 1, nv).to(torch.bfloat16), SH(3))
        nw_t = fT((nw).view(1, 1, 1, dv).to(torch.bfloat16), REP)
        # states (start: recurrent = rs0 sharded by v-head dim1; conv = zeros)
        rs_t = fT(rs0.to(torch.bfloat16), SH(1))
        conv_dim_p = (key_dim * 2 + val_dim) // TP
        cs_t = fT(torch.zeros(1, 1, conv_dim_p * TP, 32).to(torch.bfloat16), SH(2))

        qc = ttnn.linear(xt, Wq_t); kc = ttnn.linear(xt, Wk_t); vc = ttnn.linear(xt, Wv_t)
        qkv_c = ttnn.concat([qc, kc, vc], dim=3)   # per chip [q|k|v]
        zc = ttnn.linear(xt, Wz_t); bc = ttnn.linear(xt, Wb_t); ac = ttnn.linear(xt, Wa_t)

        out_tt, _, _ = ttnn.experimental.deltanet_decode_full(
            qkv_c, zc, bc, ac, cs_t, rs_t, conv_w_t, A_t, dt_t, nw_t,
            num_heads=nvp, num_k_heads=nkp, k_head_dim=dk, v_head_dim=dv,
            conv_dim=conv_dim_p, conv_kernel_size=K, head_expand_ratio=ratio)
        outp = ttnn.linear(out_tt, Wo_t)
        out = ttnn.all_reduce(outp, cluster_axis=1, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(md)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))[0:1].float()
        print(f"[shapes] ref={tuple(ref.shape)} out={tuple(out_t.shape)}", flush=True)
        print(f"[PCC] TP DeltaNet vs full = {pcc(ref, out_t):.5f}", flush=True)
        print("PASS" if pcc(ref, out_t) > 0.97 else "FAIL", flush=True)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()
