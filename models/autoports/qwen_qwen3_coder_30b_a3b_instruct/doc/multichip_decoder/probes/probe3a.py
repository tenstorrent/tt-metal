import torch

import ttnn

# --- part 1: single-device SDPA / paged cache with 1 local KV head (TP=4) ---
dev = ttnn.open_device(device_id=0, trace_region_size=50_000_000, l1_small_size=32768)
B, HD, CTX, BLK = 1, 128, 128, 32
for nq, nkv, tag in [(32, 4, "single-die 32Q/4KV"), (8, 1, "TP4 8Q/1KV")]:
    try:
        npages = (CTX // BLK) * B
        kc = ttnn.from_torch(
            torch.randn(npages, nkv, BLK, HD),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        vc = ttnn.from_torch(
            torch.randn(npages, nkv, BLK, HD),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pt = ttnn.from_torch(torch.arange(npages).reshape(B, CTX // BLK).int(), dtype=ttnn.int32, device=dev)
        pos = ttnn.from_torch(torch.tensor([CTX - 1] * B).int(), dtype=ttnn.int32, device=dev)
        q = ttnn.from_torch(
            torch.randn(1, B, nq, HD),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q, kc, vc, page_table_tensor=pt, cur_pos_tensor=pos, scale=HD**-0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        print(f"P|{tag} paged SDPA decode OK out={list(o.shape)}")
        # paged_update_cache with 1 kv head
        upd = ttnn.from_torch(
            torch.randn(1, B, nkv, HD),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.experimental.paged_update_cache(kc, upd, update_idxs_tensor=pos, page_table=pt)
        print(f"P|{tag} paged_update_cache OK")
        for t in (kc, vc, pt, pos, q, o, upd):
            ttnn.deallocate(t)
    except Exception as e:
        print(f"P|{tag} ERR {str(e)[:250]}")

# prefill create-qkv-heads with 1 kv head
for nq, nkv, tag in [(32, 4, "single-die"), (8, 1, "TP4")]:
    try:
        S = 512
        x = ttnn.from_torch(
            torch.randn(1, 1, S, (nq + 2 * nkv) * HD),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            x, num_heads=nq, num_kv_heads=nkv, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        print(f"P|{tag} prefill nlp_create_qkv_heads OK q{list(q.shape)} k{list(k.shape)}")
        o = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, scale=HD**-0.5)
        print(f"P|{tag} prefill SDPA OK out={list(o.shape)}")
        for t in (x, q, k, v, o):
            ttnn.deallocate(t)
    except Exception as e:
        print(f"P|{tag} prefill ERR {str(e)[:250]}")
ttnn.close_device(dev)
