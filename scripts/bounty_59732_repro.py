"""
Repro for ttnn.sampling low-precision random threshold bias (issue 59732).
Host-only simulation of the BF16 vs 16-bit lattice threshold.
No device required. Demonstrates that BF16 threshold capped at 255/256
causes tail tokens to be under-sampled / never drawn.
"""
import struct
import math
import random

def f32_to_bf16(x: float) -> float:
    """Round to BF16 by truncating mantissa to 7 bits, return as f32."""
    bits = struct.unpack('>I', struct.pack('>f', x))[0]
    # BF16 keeps top 16 bits
    bits_bf16 = bits & 0xFFFF0000
    # round to nearest even would be more accurate but truncation is close
    # For our purpose (threshold near 1.0, step 1/256), truncation vs RNE doesn't change conclusion
    # Use proper RNE:
    lsb = (bits >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    bits_rne = (bits + rounding_bias) & 0xFFFF0000
    # But simpler: use truncation for illustration as in comment; use RNE for final
    return struct.unpack('>f', struct.pack('>I', bits_rne))[0]

def bf16_threshold_distribution(num_samples=100000):
    """Old path: rand in [0, ~0.998] as BF16, actually 0x3F7F7FFF ~0.996... ; near 1.0 step 1/256"""
    # old rand_scale = 0x3F7F7FFF ~ 0.996094? Actually 0x3F7F8000 is 0.996... Let's simulate uniformly then BF16 quant
    samples = [random.random() for _ in range(num_samples)]  # uniform [0,1)
    bf16 = [f32_to_bf16(s) for s in samples]
    return bf16

# Demonstrate 16-bit lattice vs BF16 grid
print("=== ttnn.sampling threshold bias repro (host simulation, issue 59732) ===")
print("Old threshold: packed to BF16 (8-bit mantissa), near 1.0 step = 1/256, capped at ~0.998 (255/256?)")
print("New threshold: (hi*256+lo)/65536, 65536 values uniform in [0, 1-2^-16]\n")

# Probability that a token with p=0.00193 (gap 6.25) is drawn: needs rand in last 0.193% of mass
# With BF16 lattice of 256 values, last bin is 1/256=0.3906% -> token needs to be in last bin but BF16 grid
# misses the tail beyond 255/256 entirely.
# Simulate: draw N thresholds, count fraction > 0.998

N=100000
old_draws = [random.random() for _ in range(N)]
old_bf16 = [f32_to_bf16(x * 0.99609375) for x in old_draws]  # approximate old scale capped at <1.0
# Tail cutoff: tokens needing rand > 1 - p_tail where p_tail ~0.002
p_tail = 0.00193  # gap 6.25 case
cutoff = 1.0 - p_tail

old_never = sum(1 for v in old_bf16 if v > cutoff)
# With BF16 grid, max is ~0.996, but cutoff 0.99807 > max? Actually p_tail small means cutoff 0.998.
# Show that BF16 cannot reach beyond 255/256 = 0.99609375 if truncated, or 0.998 if capped
print(f"Tail cutoff for p={p_tail:.5f}: {cutoff:.5f}")
print(f"BF16 max in old path approx: {max(old_bf16):.6f} (capped at ~0.996-0.998)")
print(f"Fraction of BF16 draws exceeding cutoff (should be ~{p_tail:.5f} if unbiased): {old_never/N:.5f} -> expected {p_tail:.5f}")
print(f"If BF16 max < cutoff, tail token NEVER drawn (bias 0.00x) -- matches issue report: gap-6 at 0.00x\n")

# New path: 16-bit lattice
new_draws = [(random.randint(0,255)*256 + random.randint(0,255))/65536 for _ in range(N)]
new_hits = sum(1 for v in new_draws if v > cutoff)
print(f"16-bit lattice: fraction exceeding cutoff: {new_hits/N:.5f} -> expected {p_tail:.5f}, ratio { (new_hits/N)/p_tail:.2f}x (should be ~1.0)\n")

# Small-prob under-sampling table (gap 5 Expected 0.0067, BF16 sampled 0.20x)
for gap in [3,4,5,6,7]:
    p = 1/(1+math.exp(gap))
    cutoff = 1-p
    bf16_hits = sum(1 for v in (f32_to_bf16(random.random()*0.99609375) for _ in range(50000)) if v > cutoff)
    lattice_hits = sum(1 for _ in range(50000) if (random.randint(0,255)*256 + random.randint(0,255))/65536 > cutoff)
    print(f"gap {gap}: p={p:.4f}  BF16 hits {bf16_hits/50000:.4f} ({bf16_hits/50000/p:.2f}x)  lattice {lattice_hits/50000:.4f} ({lattice_hits/50000/p:.2f}x)")

print("\n=== Verify current kernel fix ===")
# Check current files contain the fix markers
import pathlib
compute = pathlib.Path("tt-metal/ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp").read_text()
writer = pathlib.Path("tt-metal/ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/writer_interleaved.cpp").read_text()
assert "0x43800000U" in compute and "floor_tile" in compute, "compute fix missing"
assert "RAND_HI_ELEMENT" in writer and "RAND_LATTICE_SCALE" in writer, "writer fix missing"
print("Current HEAD contains expected fix markers (0x43800000 floor_tile + RAND_HI/LATTICE_SCALE): PASS")
print("Fix commit: c0b2affd (PR #59759) already merged to origin/main")
