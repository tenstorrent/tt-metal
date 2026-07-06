import torch, ttnn

# Simplest multi-block: S=128 (4 tiles), D=64 (2 tiles), no mask
B, H, S_q, D = 1, 1, 128, 64
S_kv = S_q

torch.manual_seed(42)
q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
k = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
v = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)

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

    print("Running SDPA with shape (1,1,128,64)...")
    output = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(q_t, k_t, v_t)
    result = ttnn.to_torch(output)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    from tests.ttnn.utils_for_testing import compute_pcc

    pcc = compute_pcc(ref, result)
    print(f"PCC: {pcc}")
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
finally:
    ttnn.close_device(device)
