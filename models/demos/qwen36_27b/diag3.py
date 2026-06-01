"""Pinpoint divergence between TT and CPU DeltaNet step."""
import torch
import sys, os, glob
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
hidden = embed_w[token_ids].float()  # [1, 1, 5120]

# RMSNorm (CPU)
norm_w = state_dict["model.layers.0.input_layernorm.weight"].float()
var = hidden.pow(2).mean(-1, keepdim=True)
normed = norm_w * (hidden * torch.rsqrt(var + 1e-6))  # [1, 1, 5120]

import ttnn
device = ttnn.open_device(device_id=0)
try:
    normed_4d = normed.reshape(1, 1, 1, -1).to(torch.bfloat16)
    normed_tt = ttnn.from_torch(normed_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    
    # Test what ttnn.to_torch returns for shape
    print("=== Shape check ===")
    rt = ttnn.to_torch(normed_tt)
    print(f"normed_tt → to_torch shape: {rt.shape}")
    
    # Load QKV weight as bfloat16
    qkv_w_raw = state_dict["model.layers.0.linear_attn.in_proj_qkv.weight"]  # [10240, 5120]
    qkv_w_t = qkv_w_raw.T.contiguous()  # [5120, 10240]
    qkv_w_tt = ttnn.from_torch(
        qkv_w_t.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    
    # Projection
    qkv_tt = ttnn.linear(normed_tt, qkv_w_tt)
    qkv_torch = ttnn.to_torch(qkv_tt)
    print(f"\n=== QKV projection ===")
    print(f"qkv output shape: {qkv_torch.shape}")
    qkv_flat = qkv_torch.flatten()
    print(f"qkv flatten length: {qkv_flat.shape[0]}")
    print(f"qkv[:5] = {qkv_flat[:5]}")
    
    # CPU reference
    qkv_cpu = (normed.reshape(1, -1).float() @ qkv_w_raw.float().T)  # [1, 10240]
    print(f"CPU qkv[:5] = {qkv_cpu[0, :5]}")
    print(f"TT-CPU max err (QKV): {(qkv_flat[:10240].float() - qkv_cpu.flatten()).abs().max():.6f}")
    
    # Apply SiLU (CPU ref - no conv state for first token)
    qkv_silu_cpu = torch.nn.functional.silu(qkv_cpu.flatten())
    qkv_silu_tt = torch.nn.functional.silu(qkv_flat[:10240].float())
    print(f"\n=== After SiLU ===")
    print(f"CPU silu[:5] = {qkv_silu_cpu[:5]}")
    print(f"TT  silu[:5] = {qkv_silu_tt[:5]}")
    
    # Split
    key_dim = 128 * 16  # 2048
    value_dim = 128 * 48  # 6144
    q_cpu, k_cpu, v_cpu = torch.split(qkv_silu_cpu, [key_dim, key_dim, value_dim])
    q_tt, k_tt, v_tt = torch.split(qkv_silu_tt, [key_dim, key_dim, value_dim])
    
    # Now test b_proj and a_proj
    b_w = state_dict["model.layers.0.linear_attn.in_proj_b.weight"]  # [48, 5120]
    b_w_t = b_w.T.contiguous()  # [5120, 48]
    b_w_tt = ttnn.from_torch(
        b_w_t.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    b_proj_tt = ttnn.linear(normed_tt, b_w_tt)
    b_torch = ttnn.to_torch(b_proj_tt)
    print(f"\n=== b_proj ===")
    print(f"b_proj output shape: {b_torch.shape}")
    b_flat = b_torch.flatten()
    print(f"b flatten length: {b_flat.shape[0]}")
    print(f"TT b[:10] = {b_flat[:10]}")
    b_cpu = (normed.reshape(1, -1).float() @ b_w.float().T).flatten()
    print(f"CPU b[:10] = {b_cpu[:10]}")
    print(f"TT-CPU max err (b): {(b_flat[:48].float() - b_cpu).abs().max():.6f}")
    
    # Key insight: check if the padded dimensions are causing issues
    # b_proj output should be [1, 1, 1, 48] but TILE_LAYOUT pads to [1, 1, 32, 64]
    print(f"\n=== Padding investigation ===")
    print(f"b_proj full tensor[:64] = {b_flat[:64]}")
    print(f"b_proj [48:64] (should be padding) = {b_flat[48:64]}")

finally:
    ttnn.close_device(device)
