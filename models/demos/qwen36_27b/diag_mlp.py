"""Diagnose MLP output precision with different weight dtypes."""
import torch
import os, glob

SNAP = "/home/yito/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
from safetensors.torch import load_file

# Load layer 0 MLP weights
state_dict = {}
for f in sorted(glob.glob(os.path.join(SNAP, "*.safetensors"))):
    shard = load_file(f)
    for k, t in shard.items():
        nk = k.replace("model.language_model.", "model.") if k.startswith("model.language_model.") else k
        if "layers.0.mlp" in nk:
            state_dict[nk] = t
print(f"Loaded: {list(state_dict.keys())}")

gate_w = state_dict["model.layers.0.mlp.gate_proj.weight"]  # [17408, 5120]
up_w = state_dict["model.layers.0.mlp.up_proj.weight"]      # [17408, 5120]
down_w = state_dict["model.layers.0.mlp.down_proj.weight"]   # [5120, 17408]
print(f"gate_proj: {gate_w.shape}, up_proj: {up_w.shape}, down_proj: {down_w.shape}")

# Create a test input (realistic hidden state)
torch.manual_seed(42)
x = torch.randn(1, 1, 1, 5120, dtype=torch.bfloat16) * 0.1  # Typical hidden state magnitude

# CPU reference
def mlp_cpu(x):
    gate = (x.float() @ gate_w.float().T)
    gate = torch.nn.functional.silu(gate)
    up = (x.float() @ up_w.float().T)
    hidden = gate * up
    out = (hidden @ down_w.float().T)
    return out

ref = mlp_cpu(x.reshape(1, -1))
print(f"\nCPU ref: norm={ref.norm():.4f}, first5={ref[0,:5]}")

import ttnn
device = ttnn.open_device(device_id=0)
try:
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    
    for wdtype_name, wdtype in [("bfloat16", ttnn.bfloat16), ("bfloat8_b", ttnn.bfloat8_b), ("bfloat4_b", ttnn.bfloat4_b)]:
        gate_tt = ttnn.from_torch(gate_w.T.contiguous().unsqueeze(0).unsqueeze(0), dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=device)
        up_tt = ttnn.from_torch(up_w.T.contiguous().unsqueeze(0).unsqueeze(0), dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=device)
        down_tt = ttnn.from_torch(down_w.T.contiguous().unsqueeze(0).unsqueeze(0), dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=device)
        
        g = ttnn.linear(x_tt, gate_tt)
        g = ttnn.silu(g)
        u = ttnn.linear(x_tt, up_tt)
        h = ttnn.mul(g, u)
        out = ttnn.linear(h, down_tt)
        
        out_cpu = ttnn.to_torch(out).flatten()[:5120].float()
        cos_sim = torch.nn.functional.cosine_similarity(ref.flatten().unsqueeze(0), out_cpu.unsqueeze(0))
        max_err = (ref.flatten() - out_cpu).abs().max()
        ratio = out_cpu.norm() / ref.norm()
        
        print(f"\n{wdtype_name}: norm={out_cpu.norm():.4f}, cos_sim={cos_sim.item():.6f}, max_err={max_err:.4f}, norm_ratio={ratio:.4f}")
        print(f"  first5={out_cpu[:5]}")
        
        ttnn.deallocate(gate_tt)
        ttnn.deallocate(up_tt)
        ttnn.deallocate(down_tt)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        ttnn.deallocate(h)
        ttnn.deallocate(out)

finally:
    ttnn.close_device(device)
