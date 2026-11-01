"""Iteration 29: rmsnorm_bw_rows pass 2 on the FPU (iteration 20's variant A, admissible since the user relaxed the
exact-fp32 rule to "the PCC holds"): gamma * dy by row broadcast, a * (-t) accumulated onto it, the inv scale and
the residual gradient through reuse_dest; 4 tiles per DEST acquire. Factory: the compile arg carries -1/C."""
import shutil
import sys

O = "tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence"
M = "tt-train/sources/ttml/metal/ops/rmsnorm_bw_rows/device"
K = f"{M}/kernels/compute/rmsnorm_bw_rows_kernel.cpp"
F = f"{M}/rmsnorm_bw_rows_program_factory.cpp"


def sub(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        print(f"FAILED {path}: {old[:90]!r} found {n}, expected {count}")
        sys.exit(2)
    open(path, "w").write(s.replace(old, new))


shutil.copy(f"{O}/iter20_rmsnorm_bw_rows_kernel_variantA.cpp.txt", K)
sub(
    F,
    "    const uint32_t inv_c_bits = std::bit_cast<uint32_t>(1.0F / static_cast<float>(input.logical_shape()[-1]));\n",
    "    // -1/C: pass 2 accumulates a * (-t) onto gamma * dy on the FPU (iteration 29).\n"
    "    const uint32_t neg_inv_c_bits = std::bit_cast<uint32_t>(-1.0F / static_cast<float>(input.logical_shape()[-1]));\n",
)
sub(F, "{Wt, S, St, block, inv_c_bits}", "{Wt, S, St, block, neg_inv_c_bits}")
s = open(K).read()
s = s.replace(
    "// Iteration 20: the SFPU per-tile chain (one tile per acquire) moved to the FPU; fp32 operands enter the FPU through\n// 19-bit source registers (~2^-11 relative), below the bf16 output's resolution.",
    '// Iteration 29 (iteration 20\'s variant A, landed once the exact-fp32 rule was relaxed to "the PCC holds"): the\n// SFPU per-tile chain (one tile per acquire) moved to the FPU; fp32 operands enter the FPU through 19-bit source\n// registers (~2^-11 relative): relative error vs fp64 1.7e-3 -> 4.1e-3, PCC 0.999993, 172 -> 145 us per call.',
)
open(K, "w").write(s)
print("iteration 29 patch applied")
