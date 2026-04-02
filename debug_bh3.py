#!/usr/bin/env python3
"""Minimal BH debug: isolate why polygamma returns all-inf on Blackhole."""
import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Test 1: Simple known value - polygamma(1, 5.0) should be ~0.2213
print("=== Test 1: Single known value ===")
torch_input = torch.full((1, 1, 32, 32), 5.0, dtype=torch.bfloat16)
input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
output = ttnn.polygamma(input_tensor, 1)
output_torch = ttnn.to_torch(output)
ttnn.deallocate(output)
print(f"  Input: 5.0, Expected ~0.2213, Got: {output_torch[0,0,0,0].item()}")
print(f"  All inf? {(output_torch == float('inf')).all().item()}")
print(f"  Any finite? {torch.isfinite(output_torch).any().item()}")
print(f"  Output sample: {output_torch[0,0,0,:4].tolist()}")

# Test 2: Check if basic SFPU ops work (reciprocal, pow)
print("\n=== Test 2: Basic SFPU ops ===")
val5 = torch.full((1, 1, 32, 32), 5.0, dtype=torch.bfloat16)
t5 = ttnn.from_torch(val5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

recip_out = ttnn.reciprocal(t5)
recip_torch = ttnn.to_torch(recip_out)
print(f"  reciprocal(5.0) = {recip_torch[0,0,0,0].item()} (expect 0.2)")
ttnn.deallocate(recip_out)

pow_out = ttnn.pow(t5, 2)
pow_torch = ttnn.to_torch(pow_out)
print(f"  pow(5.0, 2) = {pow_torch[0,0,0,0].item()} (expect 25.0)")
ttnn.deallocate(pow_out)

# Test 3: Manual composite polygamma (should work if basic ops work)
print("\n=== Test 3: Manual composite polygamma(1, 5.0) ===")
import math

k = 1
k_der = 1.0 + k
fact_val = math.gamma(k_der)
pos_neg = -1.0 if k in (2, 4, 6, 8, 10) else 1.0
z1 = ttnn.reciprocal(ttnn.pow(t5, k_der))
temp = z1
for idx in range(1, 11):
    z1 = ttnn.reciprocal(ttnn.pow(ttnn.add(t5, idx), k_der))
    temp = ttnn.add(temp, z1)
composite_result = ttnn.multiply(temp, fact_val * pos_neg)
composite_torch = ttnn.to_torch(composite_result)
print(f"  Composite polygamma(1, 5.0) = {composite_torch[0,0,0,0].item()} (expect ~0.2213)")
ttnn.deallocate(composite_result)

# Test 4: Random inputs from seed 213919 (the failing seed)
print("\n=== Test 4: Seed 213919 random inputs ===")
torch.manual_seed(213919)
torch_input = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16) * 9.0 + 1.0
input_tensor2 = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
output2 = ttnn.polygamma(input_tensor2, 1)
output2_torch = ttnn.to_torch(output2)
ttnn.deallocate(output2)
print(f"  First 4 inputs: {torch_input[0,0,0,:4].tolist()}")
print(f"  First 4 outputs: {output2_torch[0,0,0,:4].tolist()}")
num_inf = (output2_torch == float("inf")).sum().item()
num_neginf = (output2_torch == float("-inf")).sum().item()
num_nan = torch.isnan(output2_torch).sum().item()
num_finite = torch.isfinite(output2_torch).sum().item()
print(f"  +inf: {num_inf}, -inf: {num_neginf}, nan: {num_nan}, finite: {num_finite}")

# Test 5: Seed 42 (previously worked)
print("\n=== Test 5: Seed 42 random inputs ===")
torch.manual_seed(42)
torch_input3 = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16) * 9.0 + 1.0
input_tensor3 = ttnn.from_torch(torch_input3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
output3 = ttnn.polygamma(input_tensor3, 1)
output3_torch = ttnn.to_torch(output3)
ttnn.deallocate(output3)
print(f"  First 4 inputs: {torch_input3[0,0,0,:4].tolist()}")
print(f"  First 4 outputs: {output3_torch[0,0,0,:4].tolist()}")
num_inf = (output3_torch == float("inf")).sum().item()
num_finite = torch.isfinite(output3_torch).sum().item()
print(f"  +inf: {num_inf}, finite: {num_finite}")

# Test 6: Check ARCH
print(f"\n=== Device info ===")
print(f"  Device: {device}")

ttnn.deallocate(t5)
ttnn.close_device(device)
