"""Diagnostic: compare TT model intermediate outputs with CPU reference."""
import torch
import sys
import os

SNAP = "/home/yito/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

# Load a few weights directly
from safetensors.torch import load_file
import glob

state_dict = {}
for f in sorted(glob.glob(os.path.join(SNAP, "*.safetensors"))):
    shard = load_file(f)
    for k, t in shard.items():
        nk = k
        if k.startswith("model.language_model."):
            nk = "model." + k[len("model.language_model."):]
        # Only load what we need for diagnosis
        if any(p in nk for p in ["embed_tokens", "lm_head", "model.norm", "layers.0.", "layers.3."]):
            state_dict[nk] = t

print(f"Loaded {len(state_dict)} tensors for diagnostic")

# Test 1: Embedding lookup
embed_w = state_dict["model.embed_tokens.weight"]  # [248320, 5120]
token_ids = torch.tensor([[151644, 8948, 198]])  # Some tokens
embeddings = embed_w[token_ids]  # [1, 3, 5120]
print(f"\nTest 1: Embedding")
print(f"  embed_w shape: {embed_w.shape}, dtype: {embed_w.dtype}")
print(f"  embedding[0,0,:5] = {embeddings[0, 0, :5]}")
print(f"  embedding norm: {embeddings[0, 0].float().norm():.4f}")

# Test 2: RMSNorm on embedding
def rms_norm(x, weight, eps=1e-6):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (weight * (x * torch.rsqrt(variance + eps))).to(x.dtype)

norm_w = state_dict["model.layers.0.input_layernorm.weight"]  # [5120]
normed = rms_norm(embeddings[0, 0:1], norm_w)
print(f"\nTest 2: RMSNorm(embedding)")
print(f"  normed[0,:5] = {normed[0, :5]}")
print(f"  normed norm: {normed[0].float().norm():.4f}")

# Test 3: DeltaNet projection
qkv_w = state_dict["model.layers.0.linear_attn.in_proj_qkv.weight"]  # [10240, 5120]
proj = normed.float() @ qkv_w.float().T  # [1, 10240]
print(f"\nTest 3: DeltaNet QKV projection (CPU)")
print(f"  qkv_w shape: {qkv_w.shape}")
print(f"  proj[:5] = {proj[0, :5]}")
print(f"  proj norm: {proj[0].norm():.4f}")
print(f"  proj max: {proj[0].abs().max():.4f}")

# Test 4: Same projection via ttnn (BFP4_B quantized)
import ttnn
device = ttnn.open_device(device_id=0)
try:
    h_tt = ttnn.from_torch(
        normed.unsqueeze(0).unsqueeze(0),  # [1, 1, 1, 5120]
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    # BFP4_B weight (like our model uses)
    w_tt_4b = ttnn.from_torch(
        qkv_w.T.contiguous().unsqueeze(0).unsqueeze(0),  # [1, 1, 5120, 10240]
        dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device,
    )
    # BFP16 weight (for comparison)
    w_tt_16 = ttnn.from_torch(
        qkv_w.T.contiguous().unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    
    result_4b = ttnn.to_torch(ttnn.linear(h_tt, w_tt_4b)).flatten()
    result_16 = ttnn.to_torch(ttnn.linear(h_tt, w_tt_16)).flatten()
    
    print(f"\nTest 4: ttnn.linear comparison")
    print(f"  CPU ref   [:5] = {proj[0, :5]}")
    print(f"  BFP16_TT  [:5] = {result_16[:5]}")
    print(f"  BFP4B_TT  [:5] = {result_4b[:5]}")
    print(f"  BFP16 vs CPU max err: {(result_16.float() - proj[0]).abs().max():.4f}")
    print(f"  BFP4B vs CPU max err: {(result_4b.float() - proj[0]).abs().max():.4f}")
    print(f"  BFP4B vs CPU relative err: {((result_4b.float() - proj[0]).abs() / (proj[0].abs() + 1e-6)).mean():.4f}")
    
    # Test 5: lm_head projection
    lm_w = state_dict["lm_head.weight"]  # [248320, 5120]
    # Use a fake hidden state (just the embedding)
    h_for_lm = embeddings[0, 0:1].unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 5120]
    h_lm_tt = ttnn.from_torch(h_for_lm, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    lm_w_4b = ttnn.from_torch(
        lm_w.T.contiguous().unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device,
    )
    lm_logits = ttnn.to_torch(ttnn.linear(h_lm_tt, lm_w_4b)).flatten()
    lm_ref = (embeddings[0, 0:1].float() @ lm_w.float().T).flatten()
    
    print(f"\nTest 5: lm_head (BFP4_B)")
    print(f"  CPU top-5 tokens: {lm_ref.topk(5)}")
    print(f"  TT  top-5 tokens: {lm_logits[:248320].float().topk(5)}")
    print(f"  Max logit err: {(lm_logits[:248320].float() - lm_ref).abs().max():.4f}")
    
finally:
    ttnn.close_device(device)

print("\nDiagnostic complete.")
