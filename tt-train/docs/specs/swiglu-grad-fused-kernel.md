# Fused SwiGLU Gradient Kernel (`swiglu_grad`)

## Motivation

The SwiGLU backward pass has a block of three memory-bound eltwise operations
that read the same tensors multiple times from DRAM:

```cpp
// Current: 3 kernels, 6 reads + 3 writes = 9 tensor passes
multiply_(gate, dL_dprod);                          // R: gate, dL_dprod  → W: gate (=dL_dswished)
dL_dgate = multiply(linear1, dL_dprod, silu_lhs);  // R: linear1, dL_dprod → W: dL_dgate
dL_dlinear1 = silu_bw(linear1, dL_dswished);       // R: linear1, gate    → W: dL_dlinear1
```

`linear1` is read 3 times, `dL_dprod` 2 times, `gate` read then written then read.
A single fused kernel cuts this to **3 reads + 2 writes = 5 tensor passes** (44% less
DRAM traffic). At Llama-1B B=64 (128 MB per tensor), this saves ~512 MB of DRAM
bandwidth per block per step.

## Interface

```cpp
namespace ttml::metal {

// Fused eltwise kernel for SwiGLU backward gradient computation.
// Reads (linear1, gate, dL_dprod) once, produces (dL_dlinear1, dL_dgate).
//
// Per-element computation:
//   sig          = sigmoid(linear1)
//   swished      = linear1 * sig                       // = silu(linear1)
//   dL_dswished  = gate * dL_dprod
//   dL_dgate     = swished * dL_dprod
//   silu_grad    = sig * (1 + linear1 * (1 - sig))     // = d(silu)/d(linear1)
//   dL_dlinear1  = dL_dswished * silu_grad
//
struct SwiGLUGradResult {
    ttnn::Tensor dL_dlinear1;
    ttnn::Tensor dL_dgate;
};

SwiGLUGradResult swiglu_grad(
    const ttnn::Tensor& linear1,       // [B, N, S, H] — saved from forward
    const ttnn::Tensor& gate,          // [B, N, S, H] — saved from forward
    const ttnn::Tensor& dL_dprod,      // [B, N, S, H] — gradient of silu(linear1)*gate product
    const std::optional<ttnn::Tensor>& preallocated_dL_dlinear1 = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_dL_dgate = std::nullopt);

}  // namespace ttml::metal
```

## Tensor Shapes

All inputs and outputs have the same shape `[B, N, S, H]` with:
- BFLOAT16 dtype
- TILE layout
- DRAM INTERLEAVED memory

## Device Kernel Design

### Circular Buffers

| CB Index | Name | Purpose | Size |
|---|---|---|---|
| c_0 | cb_linear1 | Input: linear1 tiles | 2 × block_size |
| c_1 | cb_gate | Input: gate tiles | 2 × block_size |
| c_2 | cb_dL_dprod | Input: dL_dprod tiles | 2 × block_size |
| c_3 | cb_dL_dlinear1 | Output: dL_dlinear1 tiles | 2 × block_size |
| c_4 | cb_dL_dgate | Output: dL_dgate tiles | 2 × block_size |
| c_5 | cb_sigmoid | Intermediate: sigmoid(linear1) | 2 × block_size |
| c_6 | cb_one_minus_sig | Intermediate: 1 - sigmoid | 2 × block_size |
| c_7 | cb_silu_grad | Intermediate: silu'(linear1) | 2 × block_size |

Total: 8 CBs × 2 × block_size tiles. At block_size=4, that's 8 × 8 × 2KB = 128 KB per core.

### Reader Kernel

Reads 3 input tensors tile-by-tile, row by row:
```
for each row assigned to this core:
    for each block of tiles in the row:
        read block from linear1 → cb_linear1
        read block from gate    → cb_gate
        read block from dL_dprod → cb_dL_dprod
```

### Compute Kernel

Per tile block, computes all outputs from the 3 inputs:

```
// Stage 1: sigmoid(linear1)
sig = sigmoid(cb_linear1)                    → cb_sigmoid

// Stage 2: silu(linear1) * dL_dprod = dL_dgate
swished = cb_linear1 * cb_sigmoid            // in register
dL_dgate = swished * cb_dL_dprod             → cb_dL_dgate

// Stage 3: silu_grad * (gate * dL_dprod) = dL_dlinear1
one_minus_sig = 1.0 - cb_sigmoid             → cb_one_minus_sig
silu_grad = cb_sigmoid * (cb_linear1 * one_minus_sig + 1.0)  → cb_silu_grad
dL_dswished = cb_gate * cb_dL_dprod          // in register
dL_dlinear1 = dL_dswished * silu_grad        → cb_dL_dlinear1

// Pop all inputs
pop(cb_linear1, cb_gate, cb_dL_dprod)
pop(cb_sigmoid, cb_one_minus_sig, cb_silu_grad)
```

### Writer Kernel

Writes 2 output tensors:
```
for each row assigned to this core:
    write row from cb_dL_dlinear1
    write row from cb_dL_dgate
```

## Integration into swiglu_fused Backward

Before (3 kernels + 9 tensor passes):
```cpp
multiply_(gate, dL_dprod);                          // dL_dswished
dL_dgate = multiply(linear1, dL_dprod, silu_lhs);  // silu(linear1) * dL_dprod
dL_dlinear1 = silu_bw(linear1, dL_dswished);       // silu backward
```

After (1 kernel + 5 tensor passes):
```cpp
auto [dL_dlinear1, dL_dgate] = ttml::metal::swiglu_grad(
    linear1, gate, dL_dprod, /*preallocated_dL_dlinear1=*/linear1);
// linear1 buffer reused for dL_dlinear1 output
// gate and dL_dprod can be freed immediately
```

## Expected Impact

| Metric | Before (3 ops) | After (fused) | Improvement |
|---|---|---|---|
| Kernel launches | 3 | 1 | 3× fewer |
| DRAM tensor passes | 9 | 5 | 44% less bandwidth |
| Intermediate allocations | 1 (dL_dgate) | 0 (preallocated) | 1 fewer |
| L1 CB usage | 3 × 7 CBs | 8 CBs | Less total L1 |

At Llama-1B B=64 (128 MB per tensor, 4 blocks):
- Bandwidth saved: 4 × (9-5) × 128 MB = **2 GB less DRAM traffic per step**
- Expected backward speedup: **3-5%** (eltwise is ~15-20% of backward time)

## File Structure

```
tt-train/sources/ttml/metal/ops/swiglu_grad/
├── swiglu_grad.hpp
├── swiglu_grad.cpp
└── device/
    ├── swiglu_grad_device_operation_types.hpp
    ├── swiglu_grad_device_operation.hpp
    ├── swiglu_grad_device_operation.cpp
    ├── swiglu_grad_program_factory.hpp
    ├── swiglu_grad_program_factory.cpp
    └── kernels/
        ├── compute/swiglu_grad_kernel.cpp
        └── dataflow/
            ├── reader_swiglu_grad_interleaved.cpp
            └── writer_swiglu_grad_interleaved.cpp
```

Follows the same structure as `silu_bw`.
