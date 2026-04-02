#!/usr/bin/env python3
"""
Polygamma BH Debug Script — run on Blackhole machine.

Usage:
  1. cd to tt-metal repo on the BH machine (Aswinmcw/polygamma-sfpu-kernel branch)
  2. python debug_bh_polygamma.py

This will:
  - Run polygamma across multiple configs
  - Find exact elements with ULP > 1
  - Print all info needed to fix the kernel
"""

import torch
import ttnn
import struct
import numpy as np


def float_to_bf16_bits(f):
    f32_bits = struct.unpack(">I", struct.pack(">f", f))[0]
    return f32_bits >> 16


def bf16_daz_normalize(bits):
    exp = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    if (exp == 0) and (mantissa != 0):
        return 0x0000
    if bits == 0x8000:
        return 0x0000
    return bits


def bf16_value_order_index_daz(bits):
    bits = bf16_daz_normalize(bits)
    exp = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    if exp == 0xFF and mantissa != 0:
        return -1
    if bits == 0x7F80:
        return 65281
    if bits == 0xFF80:
        return -1
    if bits == 0x0000:
        return 32640
    if bits & 0x8000:
        return 0x7F7F - (bits & 0x7FFF)
    return 32640 + bits - 0x007F


def ulp_distance(a, b):
    a_bits = bf16_daz_normalize(float_to_bf16_bits(a))
    b_bits = bf16_daz_normalize(float_to_bf16_bits(b))
    a_exp = (a_bits >> 7) & 0xFF
    b_exp = (b_bits >> 7) & 0xFF
    if (a_exp == 0xFF and (a_bits & 0x7F) != 0) or (b_exp == 0xFF and (b_bits & 0x7F) != 0):
        return -1
    idx_a = bf16_value_order_index_daz(a_bits)
    idx_b = bf16_value_order_index_daz(b_bits)
    if idx_a < 0 or idx_b < 0:
        return -1
    return abs(idx_a - idx_b)


def analyze_config(device, shape, k, seed=42):
    torch.manual_seed(seed)
    torch_input = torch.rand(shape, dtype=torch.bfloat16) * 9.0 + 1.0

    # High-precision reference
    ref_f64 = torch.special.polygamma(k, torch_input.to(torch.float64))

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.polygamma(input_tensor, k)
    output_torch = ttnn.to_torch(output)
    ttnn.deallocate(output)
    ttnn.deallocate(input_tensor)

    result_flat = output_torch.flatten()
    ref_flat = ref_f64.flatten()
    input_flat = torch_input.flatten()

    worst_ulp = 0
    bad_elements = []
    ulp_counts = {0: 0, 1: 0, 2: 0}

    for i in range(len(result_flat)):
        res_val = result_flat[i].item()
        ref_val = ref_flat[i].item()
        if not np.isfinite(res_val) or not np.isfinite(ref_val):
            continue
        ref_bf16 = torch.tensor(ref_val, dtype=torch.bfloat16).item()
        ulp = ulp_distance(res_val, ref_bf16)
        if ulp < 0:
            continue

        if ulp in ulp_counts:
            ulp_counts[ulp] += 1
        else:
            ulp_counts[ulp] = 1

        if ulp > worst_ulp:
            worst_ulp = ulp

        if ulp > 1:
            bad_elements.append({
                "index": i,
                "input": input_flat[i].item(),
                "got": res_val,
                "expected_f64": ref_val,
                "expected_bf16": ref_bf16,
                "ulp": ulp,
            })

    return worst_ulp, ulp_counts, bad_elements


def main():
    device = ttnn.open_device(device_id=0)

    print("=" * 100)
    print("Polygamma BH Debug — Finding ULP > 1 elements")
    print("=" * 100)

    configs = [
        ((1, 1, 32, 32), 1),
        ((1, 1, 32, 32), 2),
        ((1, 1, 32, 32), 3),
        ((1, 1, 32, 32), 5),
        ((1, 1, 32, 32), 10),
        ((1, 1, 320, 384), 1),
        ((1, 1, 320, 384), 5),
        ((1, 1, 1024, 1024), 1),
    ]

    for shape, k in configs:
        print(f"\n--- shape={shape}, n={k} ---")
        worst_ulp, ulp_counts, bad_elements = analyze_config(device, shape, k)

        print(f"  Max ULP: {worst_ulp}")
        print(f"  ULP distribution: {dict(sorted(ulp_counts.items()))}")

        if bad_elements:
            print(f"  BAD ELEMENTS (ULP > 1): {len(bad_elements)}")
            for e in bad_elements[:10]:  # Show first 10
                print(f"    idx={e['index']}: input={e['input']:.6f}, "
                      f"got={e['got']:.6f}, expected_bf16={e['expected_bf16']:.6f}, "
                      f"expected_f64={e['expected_f64']:.10f}, ULP={e['ulp']}")
        else:
            print("  All elements ULP <= 1 ✅")

    # Also test with different seeds to find more bad cases
    print("\n" + "=" * 100)
    print("Testing with multiple seeds (shape=32x32, n=1 and n=5)")
    print("=" * 100)
    for seed in [0, 42, 123, 456, 789]:
        for k in [1, 5]:
            worst_ulp, _, bad = analyze_config(device, (1, 1, 32, 32), k, seed=seed)
            status = "✅" if worst_ulp <= 1 else f"❌ max={worst_ulp}, {len(bad)} bad"
            print(f"  seed={seed}, n={k}: {status}")

    ttnn.close_device(device)
    print("\nDone!")


if __name__ == "__main__":
    main()
