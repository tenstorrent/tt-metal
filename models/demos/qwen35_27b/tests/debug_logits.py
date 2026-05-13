# Quick debug: check logit distribution and per-layer output quality
import os
from pathlib import Path

import torch
from loguru import logger

os.environ.setdefault("HF_MODEL", os.path.expanduser("~/models/Qwen3.5-27B-FP8"))

from transformers import AutoTokenizer

from models.demos.qwen35_27b.tt.model_config import load_qwen35_state_dict

model_path = os.environ["HF_MODEL"]
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Check q_proj weight layout: is it [Q; gate] or [gate; Q]?
logger.info("=== Checking HF weight layout ===")
state_dict = load_qwen35_state_dict(model_path)

# Full attention layer 3 (first full_attention layer)
wqkv = state_dict["layers.3.attention.wqkv.weight"]
logger.info(f"wqkv shape: {wqkv.shape}")  # Should be [12288, 5120]
logger.info(f"wqkv first half stats (Q?): mean={wqkv[:6144].mean():.6f}, std={wqkv[:6144].std():.6f}")
logger.info(f"wqkv second half stats (gate?): mean={wqkv[6144:].mean():.6f}, std={wqkv[6144:].std():.6f}")

# Check if the two halves look similar (both should be weight matrices)
# or if one half is all zeros (indicating wrong mapping)
logger.info(f"wqkv first half abs max: {wqkv[:6144].abs().max():.6f}")
logger.info(f"wqkv second half abs max: {wqkv[6144:].abs().max():.6f}")

# Check K, V weights
wk = state_dict["layers.3.attention.wk.weight"]
wv = state_dict["layers.3.attention.wv.weight"]
logger.info(f"wk shape: {wk.shape}, wv shape: {wv.shape}")

# Check norm weights
attn_norm = state_dict["layers.3.attention_norm.weight"]
ffn_norm = state_dict["layers.3.ffn_norm.weight"]
logger.info(f"attn_norm: mean={attn_norm.mean():.4f}, range=[{attn_norm.min():.4f}, {attn_norm.max():.4f}]")
logger.info(f"ffn_norm: mean={ffn_norm.mean():.4f}, range=[{ffn_norm.min():.4f}, {ffn_norm.max():.4f}]")
logger.info(f"These are OFFSET format (should add 1.0): effective mean ~{1.0 + attn_norm.mean():.4f}")

# Check GDN layer 0
logger.info("\n=== GDN Layer 0 weights ===")
qkv_w = state_dict["layers.0.linear_attn.in_proj_qkv.weight"]
z_w = state_dict["layers.0.linear_attn.in_proj_z.weight"]
a_w = state_dict["layers.0.linear_attn.in_proj_a.weight"]
b_w = state_dict["layers.0.linear_attn.in_proj_b.weight"]
out_w = state_dict["layers.0.linear_attn.out_proj.weight"]
logger.info(f"qkv: {qkv_w.shape}, z: {z_w.shape}, a: {a_w.shape}, b: {b_w.shape}, out: {out_w.shape}")

# Check embedding and LM head
emb = state_dict["tok_embeddings.weight"]
lm = state_dict.get("output.weight")
logger.info(f"\nembed: {emb.shape}, lm_head: {lm.shape if lm is not None else 'None (tied)'}")

# Check if LM head matches embedding (tied weights)
if lm is not None:
    sim = torch.nn.functional.cosine_similarity(emb[:10].flatten().unsqueeze(0), lm[:10].flatten().unsqueeze(0))
    logger.info(f"embed vs lm_head cosine sim (first 10 rows): {sim.item():.4f}")

# Now check the actual HF model to verify q_proj layout
logger.info("\n=== Verifying q_proj = [Q, gate] layout ===")
import json

from safetensors import safe_open

index_path = Path(model_path) / "model.safetensors.index.json"
with open(index_path) as f:
    weight_map = json.load(f)["weight_map"]

# Find q_proj for layer 3 (full attention)
q_proj_key = "model.layers.3.self_attn.q_proj.weight"
q_proj_file = weight_map.get(q_proj_key)
if q_proj_file:
    with safe_open(str(Path(model_path) / q_proj_file), framework="pt") as sf:
        q_proj_raw = sf.get_tensor(q_proj_key)
    logger.info(f"Raw q_proj shape: {q_proj_raw.shape}, dtype: {q_proj_raw.dtype}")
    # Check if there's a scale
    scale_key = q_proj_key + "_scale_inv"
    if scale_key in weight_map:
        logger.info("q_proj has FP8 scale (expected)")

# Also check: does the HF model have a separate g_proj?
g_proj_key = "model.layers.3.self_attn.g_proj.weight"
g_proj_file = weight_map.get(g_proj_key)
if g_proj_file:
    logger.info(f"FOUND separate g_proj! This means q_proj is NOT fused with gate!")
    with safe_open(str(Path(model_path) / g_proj_file), framework="pt") as sf:
        g_proj_raw = sf.get_tensor(g_proj_key)
    logger.info(f"g_proj shape: {g_proj_raw.shape}, dtype: {g_proj_raw.dtype}")
else:
    logger.info("No separate g_proj found - q_proj contains [Q; gate] fused")

# Check all self_attn keys for layer 3
attn_keys = [k for k in weight_map.keys() if "layers.3.self_attn" in k and "scale" not in k]
logger.info(f"Layer 3 self_attn keys: {attn_keys}")

logger.info("\nDone. Check output above for clues.")
