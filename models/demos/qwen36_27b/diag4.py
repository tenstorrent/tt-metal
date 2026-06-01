"""Trace full _decode_step to find divergence."""
import torch, os, glob
SNAP = "/home/yito/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
from safetensors.torch import load_file
state_dict = {}
for f in sorted(glob.glob(os.path.join(SNAP, "*.safetensors"))):
    shard = load_file(f)
    for k, t in shard.items():
        nk = k.replace("model.language_model.", "model.") if k.startswith("model.language_model.") else k
        if any(p in nk for p in ["embed_tokens", "layers.0."]):
            state_dict[nk] = t

token_ids = torch.tensor([[151644]])
embed_w = state_dict["model.embed_tokens.weight"]
hidden = embed_w[token_ids].float()
norm_w = state_dict["model.layers.0.input_layernorm.weight"].float()
var = hidden.pow(2).mean(-1, keepdim=True)
normed = norm_w * (hidden * torch.rsqrt(var + 1e-6))

import ttnn
device = ttnn.open_device(device_id=0)
try:
    from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
    from models.demos.qwen36_27b.tt.deltanet import TtGatedDeltaNet, TtDeltaNetState
    config = Qwen36ModelConfig()
    config.num_hidden_layers = 1
    deltanet = TtGatedDeltaNet(device, state_dict, 0, config, dtype=ttnn.bfloat16)
    ds = TtDeltaNetState(1, ["linear_attention"], device, config)
    
    normed_4d = normed.reshape(1, 1, 1, -1).to(torch.bfloat16)
    normed_tt = ttnn.from_torch(normed_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    
    # --- Manually step through _decode_step ---
    # 1. Projections
    qkv = ttnn.linear(normed_tt, deltanet.in_proj_qkv_w)
    z = ttnn.linear(normed_tt, deltanet.in_proj_z_w)
    b_proj = ttnn.linear(normed_tt, deltanet.in_proj_b_w)
    a_proj = ttnn.linear(normed_tt, deltanet.in_proj_a_w)
    
    qkv_cpu = ttnn.to_torch(qkv).flatten()
    z_cpu_all = ttnn.to_torch(z).flatten()
    b_cpu_all = ttnn.to_torch(b_proj).flatten()
    a_cpu_all = ttnn.to_torch(a_proj).flatten()
    
    # 2. Conv1d (first token → just SiLU)
    qkv_silu = torch.nn.functional.silu(qkv_cpu[:10240].float())
    
    # 3. Split
    key_dim, value_dim = 2048, 6144
    q, k, v = torch.split(qkv_silu, [key_dim, key_dim, value_dim])
    
    # 4. Reshape
    q = q.reshape(1, 16, 128)
    k = k.reshape(1, 16, 128)
    v = v.reshape(1, 48, 128)
    
    # 5. Expand heads
    q = q.repeat_interleave(3, dim=1)  # [1, 48, 128]
    k = k.repeat_interleave(3, dim=1)
    
    # 6. L2 norm + scale
    def l2norm(x, dim=-1, eps=1e-6):
        return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    
    q = l2norm(q.float()) * (128**-0.5)
    k = l2norm(k.float())
    v = v.float()
    
    # 7. Compute g and beta
    b_val = b_cpu_all[:48].float()
    a_val = a_cpu_all[:48].float()
    A_log = ttnn.to_torch(deltanet.A_log).flatten()[:48].float()
    dt_bias = ttnn.to_torch(deltanet.dt_bias).flatten()[:48].float()
    beta = torch.sigmoid(b_val)
    g = -A_log.exp() * torch.nn.functional.softplus(a_val + dt_bias)
    
    print(f"beta: mean={beta.mean():.4f}, min={beta.min():.4f}, max={beta.max():.4f}")
    print(f"g: mean={g.mean():.4f}, exp(g) mean={g.exp().mean():.4f}")
    
    # 8. State update (S is zero for first token)
    S = torch.zeros(48, 128, 128)
    g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
    beta_t = beta.unsqueeze(-1)
    q_t = q[0]  # [48, 128]
    k_t = k[0]
    v_t = v[0]
    
    S = S * g_t  # still zero
    kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)  # zero
    delta = (v_t - kv_mem) * beta_t  # v_t * beta_t
    S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    output_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
    
    print(f"\nRecurrence output_t: norm={output_t.norm():.6f}")
    print(f"  shape: {output_t.shape}")
    print(f"  output_t[0,:5] = {output_t[0, :5]}")
    
    # 9. Z gating
    z_val = z_cpu_all[:6144].reshape(48, 128).float()
    norm_weight = ttnn.to_torch(deltanet.norm_weight).flatten()[:128].float()
    
    variance = output_t.pow(2).mean(-1, keepdim=True)
    out_normed = output_t * torch.rsqrt(variance + 1e-6)
    out_normed = norm_weight * out_normed
    silu_z = torch.nn.functional.silu(z_val)
    out_gated = out_normed * silu_z
    
    print(f"\nNorm weight[:5] = {norm_weight[:5]}")
    print(f"out_normed norm: {out_normed.norm():.6f}")
    print(f"silu(z) norm: {silu_z.norm():.6f}")
    print(f"out_gated norm: {out_gated.norm():.6f}")
    print(f"out_gated[0,:5] = {out_gated[0, :5]}")
    
    # 10. Out projection
    out_w = state_dict["model.layers.0.linear_attn.out_proj.weight"].float()
    out_flat = out_gated.reshape(1, -1) @ out_w.T
    print(f"\nCPU out_proj norm: {out_flat.norm():.6f}")
    print(f"CPU out_proj[:5] = {out_flat[0, :5]}")
    
    # Now run TT _decode_step and compare
    normed_tt2 = ttnn.from_torch(normed_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ds2 = TtDeltaNetState(1, ["linear_attention"], device, config)
    out_tt = deltanet._decode_step(normed_tt2, ds2)
    out_tt_cpu = ttnn.to_torch(out_tt).flatten()[:5120]
    
    print(f"\nTT  out_proj norm: {out_tt_cpu.float().norm():.6f}")
    print(f"TT  out_proj[:5] = {out_tt_cpu[:5]}")
    print(f"Max error: {(out_tt_cpu.float() - out_flat.flatten()).abs().max():.6f}")
    
    # Compare state
    state_tt = ttnn.to_torch(ds2.get_recurrent_state(0))
    S_norm = S.norm()
    state_tt_norm = state_tt.float().norm()
    print(f"\nCPU state norm: {S_norm:.6f}")
    print(f"TT  state norm: {state_tt_norm:.6f}")
    
    # ALSO: check what happens when we run the TT deltanet without conv (to isolate conv bug)
    print(f"\n=== Conv1d investigation ===")
    conv_w = deltanet.conv1d_weight
    print(f"conv1d_weight shape: {conv_w.shape}")
    print(f"conv1d_weight[:3,:] = {conv_w[:3, :]}")
    
    # For first token, conv_state should be all zeros, then qkv fills last column
    # Conv output = sum over kernel dim of (state * weight)
    # For first token with zero initial state, the output should be qkv * weight[:, -1]
    # But wait - in _decode_step, we roll then set [:, -1] = qkv_cpu
    # Then conv = (state * weight).sum(dim=-1)
    
    # Let's trace it manually
    conv_state = torch.zeros(1, 10240, 4)  # [1, conv_dim, kernel_size]
    conv_state_np = conv_state.squeeze(0)  # [10240, 4]
    conv_state_np = torch.roll(conv_state_np, shifts=-1, dims=-1)
    conv_state_np[:, -1] = qkv_cpu[:10240].float()
    print(f"conv_state after update: col sums = {conv_state_np.abs().sum(dim=0)}")
    
    qkv_conv = (conv_state_np * conv_w).sum(dim=-1)
    qkv_conv_silu = torch.nn.functional.silu(qkv_conv)
    
    # Compare with just silu(qkv) (no conv)
    qkv_just_silu = torch.nn.functional.silu(qkv_cpu[:10240].float())
    
    print(f"\nWith conv: qkv_conv[:5] = {qkv_conv[:5]}")
    print(f"SiLU(conv): [:5] = {qkv_conv_silu[:5]}")
    print(f"SiLU(raw):  [:5] = {qkv_just_silu[:5]}")
    print(f"Diff (conv vs raw): max={((qkv_conv_silu - qkv_just_silu).abs().max()):.6f}")
    
finally:
    ttnn.close_device(device)
