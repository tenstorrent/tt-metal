# TTNN Eltwise Compute Kernel Migration Target Classification

**Survey Date**: 2026-04-30
**Total Kernels Surveyed**: 49 (39 base + 10 kernels_ng)
**Tier 1 Candidates**: 3
**Tier 2 Candidates**: 5
**Tier 3 (Skip)**: 41

---

## Tier 1: Direct Swap (Single LLK Call → One Helper Call)

These kernels implement a single unary SFPU operation or copy-tile pattern. Migration is mechanical: replace the LLK block with a single helper invocation.

| Path | Pattern | Helper Call | Notes |
|------|---------|------------|-------|
| `/eltwise/unary/device/kernels/compute/eltwise_identity_kernel.cpp` | Copy tile only | `tile_move<>()` | `copy_tile_init()` + `copy_tile()` → move |
| `/eltwise/unary_backward/tanh_bw/device/kernels/compute/eltwise_bw_tanh_deriv.cpp` | copy_tile + unary SFPU op + binary op | `eltwise_chain(Load{}, TanhDerivative{}, SfpuMul{})` | Tanh derivative chain with binary mul |
| `/eltwise/unary_backward/gelu_bw/device/kernels/compute/eltwise_bw_gelu_poly.cpp` | copy_tile + unary SFPU op + binary op | `eltwise_chain(Load{}, GeluDerivative{}, SfpuMul{})` | GELU derivative chain with binary mul |

**Total Tier 1**: 3 kernels

---

## Tier 2: Moderate Restructuring (Chainable LLK Sequences)

These kernels use sequential LLK calls that can be composed into an `eltwise_chain(...)`. They may require adopting policies like `WaitUpfrontPopAtEnd` or `BroadcastDim`, and may use `DestReuseOp` for in-place operations.

| Path | Pattern | Proposed Helper | Notes |
|------|---------|-----------------|-------|
| `/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | copy_tile + SFPU_OP_CHAIN_0 | `eltwise_pipeline<Auto, WaitAndPopPerTile, Bulk>(chain, cb_out, num_tiles)` | Uses macro SFPU_OP_CHAIN_0 but runtime-friendly—op selection is compile-time template param |
| `/eltwise/unary/device/kernels/compute/logsigmoid_kernel.cpp` | copy_tile + negative + exp + logsigmoid | `eltwise_chain(Load{}, Negative{}, Exp{}, LogSigmoid{})` | Multi-SFPU op sequence: x → -x → exp(-x) → log(sigmoid) |
| `/eltwise/ternary/device/kernels/compute/ternary_addc_ops_fpu.cpp` | mul_tiles + scalar mul + add with DestReuseOp | `eltwise_binary_chain(Mul{cb_in1, cb_in2}, OptionalScalarMul{}, Add{DestReuse})` | FPU-only: (b*c)*scalar + a, uses binary_dest_reuse_tiles |
| `/eltwise/ternary/device/kernels/compute/ternary_addcmul_int_sfpu.cpp` | copy_tile x3 + fill + mul_int x2 + add_int | `eltwise_chain(Load{cb_in0}, Load{cb_in1}, Load{cb_in2}, Fill{scalar}, MulInt{}, MulInt{}, AddInt{})` | Integer ops: a + scalar·b·c |
| `/eltwise/ternary/device/kernels/compute/ternary_sfpu_no_bcast_ttt.cpp` | copy_tile x3 + ternary op | `ternary_chain(Load{cb_in1}, Load{cb_in2}, Load{cb_in3}, TernaryOp{})` | Generic 3-input SFPU op (no broadcast) |

**Total Tier 2**: 5 kernels

---

## Tier 3: Skip (Macro-Injection / Cross-Iteration State / Mid-Loop Dtype / Multi-DST)

These kernels use patterns that block direct migration:
- **Macro-injection**: Single source compiled once per op via `#define ELTWISE_OP=...`, `#define BINARY_OP`, `#define SFPU_OP_CHAIN_0`, etc.
- **Cross-iteration state**: DEST held across acquire/release cycles
- **Mid-loop dtype swap**: Conditional dispatch on dtype (FP32 vs BF16 vs INT32) using `#ifdef INP_FLOAT32` / `#ifdef INP_FLOAT`
- **Multi-DST orchestration**: Multiple intermediate results held in DEST (e.g., lgamma simultaneously computing log + sin + floor)
- **Intermediate-CB round-trip**: pack_tile to intermediate CB, then wait_front and pop (breaks single-acquire/release pattern)

### Macro-Injection: BINARY_OP / ELTWISE_OP Variants (20 kernels)

| Count | Kernels | Blocker |
|-------|---------|---------|
| 6 | `eltwise_binary_kernel.cpp`, `eltwise_binary_sfpu_kernel.cpp`, `eltwise_binary.cpp`, `eltwise_binary_no_bcast.cpp`, `eltwise_binary_scalar.cpp`, `eltwise_binary_sfpu.cpp` | ELTWISE_OP, BINARY_OP macros; SFPU_OP_CHAIN_0, SFPU_OP_INIT_0, SFPU_OP_FUNC_0 |
| 3 | `eltwise_binary_sfpu_no_bcast.cpp`, `eltwise_binary_sfpu_scalar.cpp`, `eltwise_where_sfpu.cpp` | BINARY_SFPU_OP, PROCESS_POST_ACTIVATIONS macros |
| 2 | `eltwise_where_no_bcast.cpp`, `eltwise_where_sfpu_scalar.cpp` | WHERE_TTS / WHERE_TST, FILL_LLK, BINARY_SFPU_OP macros |
| 1 | `where_tss_kernel.cpp` | SFPU_OP_CHAIN_0, conditional dtype (#ifdef INP_INT32, INP_FLOAT) |
| 4 | `bcast_h.cpp`, `bcast_w.cpp`, `bcast_hw.cpp`, `bcast_h_sharded_optimised.cpp` | BCAST_OP, BCAST_LLKOP, BCAST_DIM macros |
| 10 | All kernels under `binary_ng/device/kernels_ng/compute/*` | BINARY_OP, PROCESS_POST_ACTIVATIONS, unary_bcast, #if SRC_BCAST / SRC_BCAST_B |

**Subtotal Tier 3 (Macro)**: 26 kernels

### Ternary Macro-Injection (8 kernels)

| Kernels | Blocker |
|---------|---------|
| `ternary_addc_ops_sfpu.cpp`, `ternary_addc_ops_sfpu_bcast.cpp`, `ternary_addc_ops_fpu_bcast.cpp`, `ternary_addc_ops_fpu_rowbcast.cpp` | TERNARY_SFPU_OP_INIT/FUNC macros; #if BCAST_A/B/C conditional wait/pop |
| `ternary_addcmul_int_sfpu_bcast.cpp` | #if BCAST_A/B/C conditional waits |
| `ternary_sfpu_no_bcast_tts_tst.cpp`, `ternary_sfpu_col_scalar_bcast_ttt.cpp`, `ternary_sfpu_col_scalar_bcast_tts_tst.cpp`, `ternary_sfpu_row_bcast_ttt.cpp` | TERNARY_SFPU_OP_INIT/FUNC macros |

**Subtotal Tier 3 (Ternary Macro)**: 8 kernels

### Mid-Loop Dtype Swap (3 kernels)

Conditional recompilation path (FP32 vs BF16 logic paths in single kernel):

| Kernels | Blocker | Pattern |
|---------|---------|---------|
| `hardswish_kernel.cpp` | #ifdef INP_FLOAT32 (FPU mul) vs #ifdef INP_FLOAT (binary_dest_reuse mul) | hardsigmoid + conditional mul |
| `mish_kernel.cpp` | #ifdef INP_FLOAT32 (sfpu_helpers) vs #ifdef INP_FLOAT (FPU exp+log1p+tanh+mul) | mish(x) = x·tanh(log1p(exp(x))) with dtype-dependent paths |
| `tanhshrink_kernel.cpp` | #ifdef INP_FLOAT32 (FPU sub) vs #ifdef INP_FLOAT (binary_dest_reuse sub) | tanh + conditional sub |

**Subtotal Tier 3 (Mid-Loop Dtype)**: 3 kernels

### Intermediate-CB Round-Trip (1 kernel)

| Kernel | Blocker | Pattern |
|--------|---------|---------|
| `logit_kernel.cpp` | pack_tile to cb_tmp, wait_front, then pop & reuse | Clamp → pack → pop → div·log chain |

**Subtotal Tier 3 (CB Round-Trip)**: 1 kernel

### Multi-DST Orchestration (2 kernels)

Simultaneous multi-result computation (multiple log, sin, floor values held in DEST):

| Kernel | Blocker | Pattern |
|--------|---------|---------|
| `lgamma_kernel.cpp` | Holds log(z), sin(frac(x)·π), floor(x) in separate DSTs during stirling + adjusted computation | Complex multi-branch factorial approximation |
| `lgamma_fast_kernel.cpp` | Same as above | Faster variant of lgamma with same multi-DST structure |

**Subtotal Tier 3 (Multi-DST)**: 2 kernels

---

## Summary Table

| Tier | Count | Rationale |
|------|-------|-----------|
| **Tier 1** | 3 | Identity, backward tanh, backward GELU: simple copy + unary SFPU + binary op |
| **Tier 2** | 5 | Chainable sequences: eltwise_sfpu, logsigmoid, ternary (add+mul, addcmul_int, sfpu ternary) |
| **Tier 3** | 41 | 26 binary/where/bcast macro-injection, 8 ternary macro-injection, 3 dtype-swap, 1 CB round-trip, 2 multi-DST |

**Total Targets for Phase 1 Migration**: 8 (3 Tier 1 + 5 Tier 2)

---

## Phase 1 Migration Roadmap

### Tier 1 Replacements
1. `eltwise_identity_kernel.cpp` → `tile_move<Source, Destination>()`
2. `eltwise_bw_tanh_deriv.cpp` → `eltwise_pipeline(chain(Load{}, TanhDerivative{}, SfpuMul{}), ...)`
3. `eltwise_bw_gelu_poly.cpp` → `eltwise_pipeline(chain(Load{}, GeluDerivative{}, SfpuMul{}), ...)`

### Tier 2 Migrations (Require Policy Configuration)
1. `eltwise_sfpu.cpp` → Introduce `eltwise_pipeline<SfpuBatching::Auto, SfpuInputPolicy::WaitAndPopPerTile, SfpuOutputPolicy::Bulk>()`
2. `logsigmoid_kernel.cpp` → `eltwise_chain(Load{}, Negative{}, Exp{}, LogSigmoid{})`
3. `ternary_addc_ops_fpu.cpp` → `eltwise_binary_chain(Mul{}, OptionalScalarMul{}, Add{DestReuseOp{}})`
4. `ternary_addcmul_int_sfpu.cpp` → Integer variant: `eltwise_chain(Load{x3}, Fill{}, MulInt{x2}, AddInt{})`
5. `ternary_sfpu_no_bcast_ttt.cpp` → Generic ternary: `ternary_chain(Load{x3}, TernaryOp{})`

### Blockers for Tier 3
- **Macro-Injection (26 kernels)**: Requires either (a) enum-based dispatch at compile-time or (b) instantiation per op. Deferred to Phase 5 (post-macro-factory proposal).
- **Dtype-Swap (3 kernels)**: Requires multi-codegen path or unified template. Deferred.
- **CB Round-Trip & Multi-DST (3 kernels)**: Complex state machines. Leave as-is for now; consider in Phase 6.

---

## Notes

1. **eltwise_sfpu.cpp Classification**: Marked as Tier 2 despite using `SFPU_OP_CHAIN_0` macro because it follows a pure inline macro pattern (not compile-time op selection). The macro is injected as a sequence of ops, not as a function pointer. Migration path: replace macro expansion with `eltwise_chain(...)` template.

2. **Ternary Broadcast Macros**: The `#if BCAST_A/B/C` patterns in ternary_addc_ops_sfpu_bcast.cpp are broadcast-aware synchronization, not op-selection, but they're Tier 3 because the ops (`TERNARY_SFPU_OP_INIT/FUNC`) are still injected.

3. **kernels_ng Scope**: All 10 kernels in `binary_ng/kernels_ng/compute/` inherit the macro-injection pattern from binary_ng and add broadcast specializations (ROW, COL, ROW_COL). These can potentially be unified into a single broadcast-aware helper (Phase 5+).

4. **Normalization & Embedding**: Normalization (layernorm, batch_norm, groupnorm) have their own dedicated helper infra and are out of scope. Embedding contains only tilize_chunked (data movement, not eltwise). Transformer (SDPA) contains attention-specific reductions, not eltwise candidates.

---

## Next Steps

1. **Phase 1 Execution**: Implement Tier 1 and Tier 2 migrations (~1-2 weeks).
2. **Phase 5 Planning**: Design macro factory / enum-based op dispatch for binary_ng and ternary macro-injection kernels.
3. **Phase 6 (Future)**: Revisit dtype-swap, CB round-trip, and multi-DST patterns for unified helper support.
