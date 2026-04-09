# MatmulOp Project Plan

## Goal
Create a `MatmulOp` class in `tt_metal/hw/inc/api/compute/matmul_op.h` that abstracts all uses of
`matmul_tiles` and `matmul_block` LLK calls across the codebase. The class must support ALL 40
active call sites across 28 kernel files, including fused cases where custom control flow
surrounds the matmul call.

## Environment
- Architecture: Blackhole (p100a)
- ARCH_NAME=blackhole
- Single device (no T3000/TG/Galaxy)
- TT_METAL_HOME=/localdev/wransom/tt-metal

## Agent Pipeline (Sequential)

### Agent 1: Design Agent
- **Authority**: Class API, method signatures, configuration struct
- **Input**: This plan + survey data
- **Output**: `design_spec.md`
- **Rules**: Must cover all 40 call sites. Must not write implementation code.

### Agent 2: Test Design Agent
- **Authority**: What to test, expected behaviors, edge cases
- **Input**: `design_spec.md` (NOT the implementation)
- **Output**: `test_spec.md` + test kernel source + Python test harness
- **Rules**: Must write tests against the SPEC only. Must not read implementation.

### Agent 3: Implementation Agent
- **Authority**: Internal implementation of the class
- **Input**: `design_spec.md` + `test_spec.md`
- **Output**: `matmul_op.h`
- **Rules**: Must implement exactly the API from design_spec.md. Must not modify tests.

### Agent 4: Migration Agent
- **Authority**: How each kernel uses MatmulOp
- **Input**: `matmul_op.h` + all 28 kernel files
- **Output**: Migration examples in `migration_examples/` — one file per call site group
- **Rules**: Must cover ALL 40 call sites. Must demonstrate that MatmulOp has the needed features.
  If the API is insufficient, document what's missing in `api_gaps.md`.

### Agent 5: Verification Agent
- **Authority**: Final correctness validation
- **Input**: Everything from prior agents
- **Output**: `verification_report.md`
- **Rules**: Run the representative test subset. Confirm 100% call site coverage.
  If any test hangs for >60 seconds, kill it and report it. Use `tt-smi -r` to recover.

## Call Site Registry (40 active calls across 28 files)

### matmul_tiles call sites (16 calls, 14 files)
| ID | File | Line | Mode | Pattern |
|----|------|------|------|---------|
| T1 | ttnn/.../matmul/.../bmm.cpp | 47 | 3-auto | Single tile, batch*M*N*K |
| T2 | ttnn/.../matmul/.../bmm_large_block_zm.cpp | 72 | 2-semi | h*w*K indexed tiles |
| T3 | ttnn/.../experimental/matmul/attn_matmul/.../transformer_attn_matmul.cpp | 57 | 1-low | Per-row with untilize/retilize |
| T4 | ttnn/.../experimental/matmul/group_attn_matmul/.../transformer_group_attn_matmul.cpp | 101,118 | 1-low | Arch-conditional, pack_untilize |
| T5 | ttnn/.../reduction/.../reduce_w.cpp | 45 | 1-low | Row reduction via matmul |
| T6 | ttnn/.../moreh/moreh_matmul/.../moreh_matmul.cpp | 283 | 1-low | Single tile + transpose/mask |
| T7 | ttnn/.../moreh/moreh_matmul/.../moreh_matmul.cpp | 323 | 1-low | Simple K loop pop-per-tile |
| T8 | ttnn/.../moreh/moreh_mean/.../moreh_mean_w.cpp | 56,105 | 1-low | Width reduce + masking |
| T9 | ttnn/.../moreh/moreh_sum/.../moreh_sum_w.cpp | 50 | 1-low | Width reduce + masking |
| T10 | tt-train/.../sdpa_fw/.../sdpa_fw_compute_kernel.cpp | 101-106 | 1-low | Diagonal QxK^T |
| T11 | tt-train/.../sdpa_fw/.../sdpa_compute_utils.hpp | 149-154 | 1-low | QKxV blocked |
| T12 | tt-train/.../sdpa_bw/.../sdpa_bw_compute_utils.hpp | 269 | 1-low | Reduce + reciprocal |
| T13 | ttnn/kernel/compute/bmm_tilize_untilize.cpp | 183-189 | 1-low | 6-loop, tilize/untilize/bias/SFPU |
| T14 | models/demos/deepseek_v3_b1/.../rope.hpp | 176 | 1-low | RoPE step 1 of 4 |

### matmul_block call sites (24+ calls, 15 files)
| ID | File | Line | Mode | Pattern |
|----|------|------|------|---------|
| B1 | ttnn/.../matmul/.../bmm_large_block_zm_fused_bias_activation.cpp | 308 | 2-semi | K-accum + bias + SFPU + untilize |
| B2 | ttnn/.../matmul/.../bmm_large_block_zm_fused_bias_activation_gathered.cpp | 358 | 2-semi | Same + gathered input |
| B3 | ttnn/.../conv/conv2d/.../conv_bmm_tilize.cpp | 428 | 2-semi | Tilize + K-accum + bias + SFPU |
| B4 | ttnn/.../transformer/sdpa/.../compute_streaming.hpp | 100-103 | 2-semi | Arch-dep (no_mop on BH) |
| B5 | ttnn/.../transformer/sdpa/.../compute_common.hpp | 1229 | 2-semi | matmul_blocks + mask fusion |
| B6 | ttnn/.../transformer/sdpa/.../compute_common.hpp | 1304 | 2-semi | Mx1 reduction |
| B7 | ttnn/.../transformer/sdpa_decode/.../sdpa_flash_decode.cpp | 347,438 | 2-semi | Via matmul_blocks wrapper |
| B8 | ttnn/.../experimental/topk_router_gpt/.../compute.cpp | 103 | 1-low | 1x1x1 tile-by-tile |
| B9 | ttnn/.../experimental/minimal_matmul/.../compute.cpp | 285 | 3-auto | Standard M*N*K |
| B10 | ttnn/.../experimental/conv3d/.../compute.cpp | 55 | 3-auto | Standard subblock+K |
| B11 | ttnn/.../experimental/deepseek/moe/moe_gate_mm/.../compute.cpp | 111,128,177,194 | 1-low | ct=2, ring |
| B12 | ttnn/.../experimental/deepseek/mla/matmul_wo/.../compute.cpp | 79 | 1-low | ct=7 |
| B13 | ttnn/.../experimental/ccl/moe_compute/.../compute.cpp | 196,274 | 1-low | ct=4 |
| B14 | ttnn/.../experimental/ccl/moe_gpt/.../compute.cpp | 210-466 (8 calls) | 1-low | ct=4 + bias-via-ones + SwiGLU |
| B15 | ttnn/.../experimental/ccl/all_gather_minimal_matmul_async/.../compute.cpp | 292 | 3-auto | Standard K-accum |
| B16 | ttnn/.../experimental/ccl/llama_all_gather_matmul_async/.../bmm_large_block_zm_fused_bias_activation_gathered.cpp | 302 | 2-semi | Same as B2 |

## Representative Test Subset (for Blackhole single-device)

These tests cover the major kernel families and run in <60 seconds each:

| Test | Kernel Exercised | Timeout | Command |
|------|-----------------|---------|---------|
| Basic matmul | bmm.cpp, bmm_large_block_zm*.cpp | 60s | `pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_pytorch_2_0_failed_cases -x --timeout=60 -v 2>&1 \| head -50` |
| Linear (with bias) | bmm_large_block_zm_fused_bias_activation.cpp | 60s | `pytest tests/ttnn/unit_tests/operations/matmul/test_linear.py -k "test_linear[batch_sizes0" -x --timeout=60 -v 2>&1 \| head -50` |
| Conv2D | conv_bmm_tilize.cpp | 60s | `pytest tests/ttnn/unit_tests/operations/conv/test_conv2d.py -k "test_conv_features" -x --timeout=60 -v 2>&1 \| head -50` |
| Conv3D | conv3d compute.cpp | 60s | `pytest tests/ttnn/unit_tests/operations/conv/test_conv3d.py::test_conv3d_float32 -x --timeout=60 -v 2>&1 \| head -50` |
| SDPA decode | sdpa_flash_decode.cpp, compute_common.hpp | 60s | `pytest tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py -k "test_sdpa_decode_non_tile_aligned" -x --timeout=60 -v 2>&1 \| head -50` |
| SDPA prefill | sdpa compute_streaming.hpp | 60s | `pytest tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py -k "test_sdpa_prefill" -x --timeout=60 -v 2>&1 \| head -50` |
| Reduction | reduce_w.cpp | 60s | `pytest tests/ttnn/unit_tests/operations/reduce/test_reduction.py -k "test_std[batch_size1-h32-w32" -x --timeout=60 -v 2>&1 \| head -50` |
| Moreh matmul | moreh_matmul.cpp | 60s | `pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_matmul.py -k "test_moreh_matmul[" -x --timeout=60 -v 2>&1 \| head -50` |
| Minimal matmul | minimal_matmul compute.cpp | 60s | `pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_minimal_matmul.py::test_linear_tile_padding -x --timeout=60 -v 2>&1 \| head -50` |

### Tests NOT runnable on single BH (multi-device required):
- moe_gpt, moe_compute, all_gather_matmul, llama_all_gather_matmul, ring_matmul
- topk_router_gpt (skipped on BH: needs 12 DRAM-aligned cores)
- These are covered by migration examples + code review only

## Hang Recovery Protocol
If any test hangs (no output for >60 seconds):
1. Kill the process: `pkill -f pytest`
2. Reset device: `tt-smi -r`
3. Wait 5 seconds
4. Report the hang in verification_report.md
5. Do NOT retry — move to next test

## File Layout
```
.matmul_op_project/
  plan.md                    -- this file
  design_spec.md             -- Agent 1 output
  test_spec.md               -- Agent 2 output
  verification_report.md     -- Agent 5 output
  api_gaps.md                -- Agent 4 output (if API insufficient)
  migration_examples/
    mode3_automatic.cpp      -- Mode 3 examples (T1, B9, B10, B15)
    mode2_fused_bmm.cpp      -- Mode 2 examples (B1, B2, B3, B16)
    mode2_sdpa.cpp           -- Mode 2 examples (B4, B5, B6, B7)
    mode1_tile_simple.cpp    -- Mode 1 examples (T5, T6, T7, T8, T9)
    mode1_tile_complex.cpp   -- Mode 1 examples (T2, T3, T4, T13, T14)
    mode1_block_moe.cpp      -- Mode 1 examples (B8, B11, B12, B13, B14)
    mode1_sdpa_embedded.cpp  -- Mode 1 examples (T10, T11, T12)

tt_metal/hw/inc/api/compute/
  matmul_op.h                -- Agent 3 output (the actual implementation)
```
