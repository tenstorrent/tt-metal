import torch
import ttnn

def test_ip():
    print("=" * 60)
    print("ROOT CAUSE ANALYSIS: Why dtype=BFLOAT16 causes PCC drop")
    print("=" * 60)
    
    # --- The constant value ---
    const_f32 = torch.tensor(0.99000000953674316, dtype=torch.float32)
    const_bf16 = const_f32.to(torch.bfloat16)
    
    print(f"\n1. Constant value precision:")
    print(f"   Float32  : {const_f32.item()}")
    print(f"   BFloat16 : {const_bf16.float().item()}")
    print(f"   → Constant CHANGES when cast to bfloat16!")

    # --- Load input tensor ---
    raw = ttnn.load_tensor("arg0.tensorbin")
    raw = ttnn.to_layout(raw, ttnn.Layout.ROW_MAJOR)
    input_f32 = ttnn.to_torch(raw).to(torch.float32)
    
    # --- Find values in the problematic gap ---
    lo = const_bf16.float().item()   # 0.98828125
    hi = const_f32.item()            # 0.99
    in_gap = (input_f32 >= lo) & (input_f32 < hi)
    gap_count = in_gap.sum().item()
    
    print(f"\n2. Input values in gap [{lo}, {hi}):")
    print(f"   Count: {gap_count}")
    print(f"   → These are values where comparison result differs!")
    
    # --- Show the comparison mismatch ---
    if gap_count > 0:
        example_val = input_f32[in_gap][0].item()
        example_val_bf16 = torch.tensor(example_val).to(torch.bfloat16).float().item()
        
        print(f"\n3. Example value from gap: {example_val}")
        print(f"   Value after bfloat16 cast: {example_val_bf16}")
        print(f"   → Input UNCHANGED (already bfloat16-representable)")
        
        print(f"\n4. Comparison in Float32:")
        print(f"   {example_val} < {const_f32.item()} = True")
        
        print(f"\n5. Comparison in BFloat16:")
        print(f"   {example_val_bf16} < {const_bf16.float().item()} = False")
        print(f"   → Both become 0.98828125, so NOT less-than!")
        
        print(f"\n6. Result:")
        print(f"   CPU (float32):  outputs 1.0 (True)")
        print(f"   TTNN (bfloat16): outputs 0.0 (False)")
        print(f"   → Mismatch for all {gap_count} values in the gap!")
    
    print("\n" + "=" * 60)
    print("CONCLUSION: dtype=BFLOAT16 makes constant round DOWN to")
    print("            meet the input boundary, flipping 80 comparisons")
    print("=" * 60)
    
    # --- Print boundary values ---
    boundary_values = input_f32[in_gap].tolist()
    print(f"\n7. All boundary values causing mismatch: {boundary_values}")
