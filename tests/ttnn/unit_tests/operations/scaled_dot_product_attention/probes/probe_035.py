import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device_id = 0
device = ttnn.open_device(device_id=device_id)

try:
    from tests.ttnn.utils_for_testing import assert_with_pcc

    test_cases = [
        # (B, H, S, D, desc)
        (1, 1, 32, 32, "single_tile"),
        (1, 1, 128, 64, "128x64"),
        (1, 1, 256, 64, "256x64"),
        (1, 1, 128, 128, "128x128"),
        (1, 4, 128, 64, "multi_head"),
        (2, 4, 128, 64, "multi_batch"),
        (1, 1, 512, 64, "long_512"),
        (1, 1, 1024, 64, "long_1024"),
        (1, 1, 2048, 64, "long_2048"),
        (1, 32, 128, 128, "large_model"),
        # GQA with causal
        (1, 8, 128, 64, "gqa_4to1"),  # H_kv=2
        # MQA with causal
        (1, 8, 128, 64, "mqa"),  # H_kv=1
    ]

    for B, H, S, D, desc in test_cases:
        torch.manual_seed(42)
        is_gqa = desc == "gqa_4to1"
        is_mqa = desc == "mqa"
        H_kv = 2 if is_gqa else (1 if is_mqa else H)

        q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
        k = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
        v = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)

        # PyTorch reference with is_causal=True
        if is_gqa or is_mqa:
            repeats = H // H_kv
            k_ref = k.repeat_interleave(repeats, dim=1)
            v_ref = v.repeat_interleave(repeats, dim=1)
            ref = torch.nn.functional.scaled_dot_product_attention(q, k_ref, v_ref, is_causal=True)
        else:
            ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        q_t = ttnn.from_torch(
            q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        k_t = ttnn.from_torch(
            k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        v_t = ttnn.from_torch(
            v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        try:
            output = scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
            output_torch = ttnn.to_torch(output)
            assert_with_pcc(ref, output_torch, pcc=0.995)
            print(f"PASS: causal {desc} ({B},{H},{S},{D})")
        except Exception as e:
            print(f"FAIL: causal {desc} ({B},{H},{S},{D}): {e}")

    # Test that cross-attention + causal raises ExcludedCell
    print("\n--- Testing causal+cross exclusion ---")
    q = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16)
    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    try:
        output = scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        print("FAIL: causal+cross should have raised but didn't")
    except NotImplementedError as e:
        print(f"PASS: causal+cross correctly raised: {type(e).__name__}")
    except Exception as e:
        print(f"FAIL: causal+cross raised wrong exception type: {type(e).__name__}: {e}")

    # Test that is_causal + attn_mask raises ValueError
    print("\n--- Testing is_causal + attn_mask mutual exclusion ---")
    q = torch.randn(1, 1, 128, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 128, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 1, 128, 64, dtype=torch.bfloat16)
    mask = torch.zeros(1, 1, 128, 128, dtype=torch.bfloat16)
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
    try:
        output = scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask_t, is_causal=True)
        print("FAIL: is_causal+attn_mask should have raised ValueError")
    except ValueError as e:
        print(f"PASS: is_causal+attn_mask correctly raised ValueError")
    except Exception as e:
        print(f"FAIL: is_causal+attn_mask raised wrong exception: {type(e).__name__}: {e}")

finally:
    ttnn.close_device(device)
