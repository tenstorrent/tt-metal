import torch, ttnn

# Test mask precision with a simple case
B, H, S_q, D = 1, 1, 128, 64
S_kv = S_q

torch.manual_seed(42)
q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
k = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
v = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)

# Create a custom additive mask: 0 = attend, -inf = mask out
mask = torch.zeros(B, 1, S_q, S_kv, dtype=torch.bfloat16)
for i in range(S_q):
    for j in range(S_kv):
        if j > i + (S_kv - S_q):
            mask[:, :, i, j] = float("-inf")

# Also test with a large negative instead of -inf
mask_large_neg = mask.clone()
mask_large_neg[mask == float("-inf")] = -1e4

device_id = 0
device = ttnn.open_device(device_id=device_id)

try:
    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    mask_t = ttnn.from_torch(
        mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    mask_ln_t = ttnn.from_torch(
        mask_large_neg,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Test with -inf mask
    print("Testing with -inf mask...")
    output = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask_t)
    result = ttnn.to_torch(output)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    # Check PCC manually
    ref_f = ref.float().flatten()
    res_f = result.float().flatten()
    # Compute PCC
    ref_mean = ref_f.mean()
    res_mean = res_f.mean()
    ref_centered = ref_f - ref_mean
    res_centered = res_f - res_mean
    numerator = (ref_centered * res_centered).sum()
    denominator = torch.sqrt((ref_centered**2).sum() * (res_centered**2).sum())
    pcc = (numerator / denominator).item()
    print(f"PCC with -inf mask: {pcc}")

    # Check max diff
    max_diff = (ref.float() - result.float()).abs().max().item()
    print(f"Max diff with -inf mask: {max_diff}")

    # Test with large negative mask
    print("\nTesting with -1e4 mask...")
    output2 = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=mask_ln_t
    )
    result2 = ttnn.to_torch(output2)
    ref2 = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask_large_neg)

    ref2_f = ref2.float().flatten()
    res2_f = result2.float().flatten()
    ref2_centered = ref2_f - ref2_f.mean()
    res2_centered = res2_f - res2_f.mean()
    numerator2 = (ref2_centered * res2_centered).sum()
    denominator2 = torch.sqrt((ref2_centered**2).sum() * (res2_centered**2).sum())
    pcc2 = (numerator2 / denominator2).item()
    print(f"PCC with -1e4 mask: {pcc2}")
    max_diff2 = (ref2.float() - result2.float()).abs().max().item()
    print(f"Max diff with -1e4 mask: {max_diff2}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()
finally:
    ttnn.close_device(device)
