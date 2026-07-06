import torch
import ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)

try:
    torch.manual_seed(42)
    B, H, S_q, D = 1, 1, 128, 64
    S_kv = S_q

    q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t)
    output_torch = ttnn.to_torch(output)

    from tests.ttnn.utils_for_testing import assert_with_pcc

    pcc_passed, pcc_msg = assert_with_pcc(ref, output_torch, pcc=0.995)
    print(f"PCC (no mask, 128x64): {pcc_msg}")

    # Now test with mask
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=torch.bfloat16)
    for i in range(S_q):
        for j in range(S_kv):
            if j > i:
                mask[:, :, i, j] = float("-inf")

    ref_masked = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    mask_t = ttnn.from_torch(
        mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    output_masked = scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask_t)
    output_masked_torch = ttnn.to_torch(output_masked)

    pcc_passed, pcc_msg = assert_with_pcc(ref_masked, output_masked_torch, pcc=0.995)
    print(f"PCC (custom mask, 128x64): {pcc_msg}")

    # Check a few values
    print(f"\nNo-mask ref[0,0,0,:4]: {ref[0,0,0,:4]}")
    print(f"No-mask out[0,0,0,:4]: {output_torch[0,0,0,:4]}")
    print(f"Mask ref[0,0,0,:4]: {ref_masked[0,0,0,:4]}")
    print(f"Mask out[0,0,0,:4]: {output_masked_torch[0,0,0,:4]}")

    # Check max diff
    diff = (ref_masked.float() - output_masked_torch.float()).abs()
    print(f"\nMask max diff: {diff.max()}")
    print(f"Mask mean diff: {diff.mean()}")
finally:
    ttnn.close_device(device)
