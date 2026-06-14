# SPDX-License-Identifier: Apache-2.0
"""Validate the passthrough-reader + host-side conv1d path: compute conv1d+SiLU+l2norm
in Python (torch here, for correctness) and call deltanet_decode_full with the processed
qkv. Must match the original conv-in-reader baseline: OUT sum=25248.68 / STATE sum=33938.80."""
import torch
import ttnn
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.load_weights import create_dummy_state_dict
from models.demos.qwen36_27b.tt.deltanet import TtGatedDeltaNet, TtDeltaNetState

torch.manual_seed(1234)
cfg = Qwen36ModelConfig()
sd = create_dummy_state_dict(cfg, num_layers=1)
dev = ttnn.open_device(device_id=0)
try:
    dn = TtGatedDeltaNet(dev, sd, 0, cfg)
    st = TtDeltaNetState(1, cfg.layer_types[:1], dev, cfg)
    H, Dk, Dv = dn.num_v_heads, dn.head_k_dim, dn.head_v_dim
    nk, key_dim, val_dim, K = dn.num_k_heads, dn.key_dim, dn.value_dim, dn.conv_kernel_size
    rs = torch.randn(1, H, Dk, Dv) * 0.1
    st.recurrent_states[0] = ttnn.from_torch(rs.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    x = ttnn.from_torch((torch.randn(1, 1, 1, cfg.hidden_size) * 0.1).to(torch.bfloat16),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    # projections
    qkv = ttnn.linear(x, dn.in_proj_qkv_w)
    z = ttnn.linear(x, dn.in_proj_z_w); b = ttnn.linear(x, dn.in_proj_b_w); a = ttnn.linear(x, dn.in_proj_a_w)

    # ---- host conv1d + SiLU + l2norm (torch) ----
    qkv_cpu = ttnn.to_torch(qkv).float().reshape(-1)[: key_dim * 2 + val_dim]
    conv_w = dn.conv1d_weight.float()                          # [conv_dim, K]
    conv_state_cpu = st.get_conv_state_cpu(0, K)               # [1, conv_dim, K] or None (zeros)
    if conv_state_cpu is None:
        cs = torch.zeros(key_dim * 2 + val_dim, K)
    else:
        cs = conv_state_cpu.squeeze(0)                          # [conv_dim, K]
    window = torch.cat([cs[:, 1:K], qkv_cpu[:, None]], dim=1)  # [conv_dim, K]
    dot = (window * conv_w).sum(dim=1)                         # [conv_dim]
    qkv_conv = dot * torch.sigmoid(dot)                        # SiLU
    # l2norm: q (×1/sqrt(Dk)), k (no scale), per k-head over Dk; v unchanged
    q = qkv_conv[:key_dim].reshape(nk, Dk)
    k = qkv_conv[key_dim:2 * key_dim].reshape(nk, Dk)
    v = qkv_conv[2 * key_dim:]
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6) * (Dk ** -0.5)
    # NOTE l2norm eps is added to sum_sq in reader: 1/sqrt(sum_sq+eps); approximate with norm
    qn = torch.rsqrt((q.reshape(nk, Dk) ** 2).sum(-1, keepdim=True) + 1e-6) if False else None
    # redo exactly as reader: inv = 1/sqrt(sum_sq+eps)
    qc = qkv_conv[:key_dim].reshape(nk, Dk)
    kc = qkv_conv[key_dim:2 * key_dim].reshape(nk, Dk)
    q = qc * (torch.rsqrt((qc ** 2).sum(-1, keepdim=True) + 1e-6) * (Dk ** -0.5))
    k = kc * torch.rsqrt((kc ** 2).sum(-1, keepdim=True) + 1e-6)
    qkv_proc = torch.cat([q.reshape(-1), k.reshape(-1), v], dim=0)  # [conv_dim]
    qkv_proc_tt = ttnn.from_torch(qkv_proc.view(1, 1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=dev)

    conv_state = st.get_conv_state(0); rec = st.get_recurrent_state(0)
    out, new_state, _ = ttnn.experimental.deltanet_decode_full(
        qkv_proc_tt, z, b, a, conv_state, rec, dn.conv1d_weight_tt, dn.A_log_bf16, dn.dt_bias_bf16, dn.norm_weight,
        num_heads=H, num_k_heads=nk, k_head_dim=Dk, v_head_dim=Dv,
        conv_dim=key_dim * 2 + val_dim, conv_kernel_size=K, head_expand_ratio=dn.head_expand_ratio)
    out = ttnn.linear(out, dn.out_proj_w)
    o = ttnn.to_torch(out).float().reshape(-1)
    ns = ttnn.to_torch(new_state).float().reshape(-1)
    print(f"OUT  sum={o.sum().item():.6f} absmean={o.abs().mean().item():.6f} [:5]={o[:5].tolist()}", flush=True)
    print(f"STATE sum={ns.sum().item():.6f}", flush=True)
    print("--- target: OUT sum=25248.68 / STATE sum=33938.80 ---", flush=True)
finally:
    ttnn.close_device(dev)
