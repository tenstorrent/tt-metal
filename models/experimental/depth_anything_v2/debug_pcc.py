"""Per-layer PCC debug: isolates where accuracy drops in the ViT encoder."""

import torch
import ttnn
import numpy as np
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from PIL import Image
from models.experimental.depth_anything_v2.tt.model_def import (
    custom_preprocessor, vit_embeddings, vit_layer, get_model_config, _dram_tile
)


def pcc(a, b):
    a = a.flatten().astype(float)
    b = b.flatten().astype(float)
    return float(np.corrcoef(a, b)[0, 1])


def move(p, dev):
    if isinstance(p, ttnn.Tensor):
        return ttnn.to_device(p, dev)
    elif isinstance(p, dict):
        return {k: move(v, dev) for k, v in p.items()}
    elif isinstance(p, list):
        return [move(v, dev) for v in p]
    return p


def main():
    torch_model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Large-hf", torch_dtype=torch.float32
    ).eval()
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")

    np.random.seed(42)
    img = Image.fromarray(np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8))
    pv = processor(images=img, return_tensors="pt")["pixel_values"]

    with torch.no_grad():
        pt_embed = torch_model.backbone.embeddings(pv)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    params = custom_preprocessor(torch_model, "test")
    config = get_model_config(1, device)
    emb_params = move(params["backbone"]["embeddings"], device)

    tt_pv = ttnn.from_torch(pv, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_embed = vit_embeddings(config, tt_pv, emb_params, device)
    tt_h = _dram_tile(tt_embed)

    # Embedding PCC
    tt_e = ttnn.to_torch(tt_embed).float()
    p = pcc(pt_embed[0, 1:1370, :].numpy(), tt_e[0, 32:32+1369, :].numpy())
    print(f"Embedding PCC: {p:.6f}")

    # Attention mask
    mask_t = torch.zeros(1, 1, 1, 1408)
    mask_t[0, 0, 0, 1:32] = -1e9
    mask_t[0, 0, 0, 1401:1408] = -1e9
    attn_mask = ttnn.from_torch(mask_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Step through encoder layer 0 manually — isolate MLP
    enc0 = move(params["backbone"]["encoder"]["layer"][0], device)
    pt_layer = torch_model.backbone.encoder.layer[0]

    # --- Attention path ---
    ln1 = ttnn.layer_norm(tt_h, weight=enc0["layernorm_before"]["weight"], bias=enc0["layernorm_before"]["bias"])
    qkv = ttnn.linear(ln1, enc0["attention"]["qkv"]["weight"], bias=enc0["attention"]["qkv"]["bias"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(ln1)
    (query, key, value) = ttnn.transformer.split_query_key_value_and_split_heads(qkv, memory_config=ttnn.DRAM_MEMORY_CONFIG, num_heads=16)
    ttnn.deallocate(qkv)
    attn_scores = ttnn.matmul(query, key, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(query)
    ttnn.deallocate(key)
    attn_scores = ttnn.to_memory_config(attn_scores, ttnn.DRAM_MEMORY_CONFIG)
    attn_scores = ttnn.mul(attn_scores, 1.0/8.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    attn_scores = ttnn.add(attn_scores, attn_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    attn_probs = ttnn.softmax(attn_scores, dim=-1)
    ttnn.deallocate(attn_scores)
    ctx = ttnn.matmul(attn_probs, value, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_probs)
    ttnn.deallocate(value)
    ctx = ttnn.transformer.concatenate_heads(ctx, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    attn_out = ttnn.linear(ctx, enc0["attention"]["output"]["dense"]["weight"], bias=enc0["attention"]["output"]["dense"]["bias"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(ctx)
    resid1 = ttnn.add(_dram_tile(tt_h), _dram_tile(attn_out), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_out)

    with torch.no_grad():
        pt_ln1 = pt_layer.norm1(pt_embed)
        pt_attn = pt_layer.attention(pt_ln1)[0]
        pt_r1 = pt_embed + pt_attn

    tt_r1 = ttnn.to_torch(resid1).float()
    print(f"Residual1 PCC: {pcc(pt_r1[0, 1:1370, :].numpy(), tt_r1[0, 32:32+1369, :].numpy()):.6f}")

    # --- MLP path ---
    ln2 = ttnn.layer_norm(resid1, weight=enc0["layernorm_after"]["weight"], bias=enc0["layernorm_after"]["bias"])
    tt_ln2 = ttnn.to_torch(ln2).float()

    with torch.no_grad():
        pt_ln2 = pt_layer.norm2(pt_r1)
    print(f"LN2 PCC: {pcc(pt_ln2[0, 1:1370, :].numpy(), tt_ln2[0, 32:32+1369, :].numpy()):.6f}")

    # FC1 + GELU
    fc1 = ttnn.linear(ln2, enc0["intermediate"]["dense"]["weight"], bias=enc0["intermediate"]["dense"]["bias"], memory_config=ttnn.DRAM_MEMORY_CONFIG, activation="gelu")
    tt_fc1 = ttnn.to_torch(fc1).float()

    with torch.no_grad():
        pt_fc1_raw = pt_layer.mlp.fc1(pt_ln2)
        pt_gelu = pt_layer.mlp.activation(pt_fc1_raw)
    print(f"FC1+GELU PCC: {pcc(pt_gelu[0, 1:1370, :].numpy(), tt_fc1[0, 32:32+1369, :].numpy()):.6f}")
    print(f"  FC1+GELU shapes: TT={tt_fc1.shape}, PT={pt_gelu.shape}")

    # FC2
    fc2 = ttnn.linear(fc1, enc0["output"]["dense"]["weight"], bias=enc0["output"]["dense"]["bias"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_fc2 = ttnn.to_torch(fc2).float()

    with torch.no_grad():
        pt_fc2 = pt_layer.mlp.fc2(pt_gelu)
    print(f"FC2 PCC: {pcc(pt_fc2[0, 1:1370, :].numpy(), tt_fc2[0, 32:32+1369, :].numpy()):.6f}")

    # Residual2
    resid2 = ttnn.add(_dram_tile(resid1), _dram_tile(fc2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_r2 = ttnn.to_torch(resid2).float()

    with torch.no_grad():
        pt_r2 = pt_r1 + pt_fc2
    print(f"Full layer 0 PCC: {pcc(pt_r2[0, 1:1370, :].numpy(), tt_r2[0, 32:32+1369, :].numpy()):.6f}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
