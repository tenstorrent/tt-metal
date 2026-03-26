# Matmul Reference Pointers

## GEMM FLOPS analysis
`tech_reports/GEMM_FLOPS/GEMM_FLOPS.md`
Per-core TFLOPS by math fidelity. How to calculate arithmetic intensity and
determine compute-bound vs memory-bound.

## Matrix engine deep dive
`tech_reports/matrix_engine/matrix_engine.md`
How the FPU executes 8×16 × 16×16 tile operations. Latency, throughput, fidelity.

## Matmul op implementation
`ttnn/cpp/ttnn/operations/matmul/`
Current matmul implementation including 1D/2D/reuse strategies.

## Data formats and accuracy
`tech_reports/data_formats/data_formats.md`
bfloat16/bfloat8_b/bfloat4_b accuracy and memory trade-offs.
