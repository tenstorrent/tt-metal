# Kernel Dev: Mamba2SSDScan

## Path
C++ unified kernel — reason: requires explicit BRISC chunk-streaming with variable n_chunks
(override_runtime_arguments), sequential state carry across chunks, and multiple TRISC matmuls
([C,N]@[N,C], [C,C]@[C,D], [C,N]@[N,D], [D,C]@[C,N]) that need coordinated BRISC→TRISC→NCRISC
dataflow. tt-lang can't express a runtime-variable outer loop.

## Math

For each head h in parallel (1 core per head) and each chunk c sequentially (sequential state carry):

```
Inputs per chunk (C=64 tokens per chunk):
  log_decay_c  [C]         =  -exp(A_log) * softplus(dt + dt_bias)   [pre-computed]
  x_dt_c       [C, D]      =  x * softplus(dt + dt_bias)             [pre-computed]
  B_c          [C, N]      =  B from input projection (group g = h//8)
  C_c          [C, N]      =  C from input projection (group g = h//8)
  x_c          [C, D]      =  raw x (for D-skip)
  h_prev       [D, N]      =  SSM state carried from previous chunk (init = zeros)

1. log_cum[i] = cumsum(log_decay_c)[i]          [C]
2. L[i,s] = exp(clamp(log_cum[i] - log_cum[s], max=0)) * causal_mask[i,s]   [C,C]
   (causal_mask = lower triangular, L[i,i]=1, L[i,s]=0 for s>i)
3. Q_K = C_c @ B_c^T                            [C,N] @ [N,C] → [C,C]
4. L_QK = L ⊙ Q_K                               [C,C]  (element-wise)
5. y_intra = L_QK @ x_dt_c                      [C,C] @ [C,D] → [C,D]
6. gamma[i] = exp(log_cum[i])                   [C]   (γ per position)
7. y_cross = (C_c @ h_prev^T) ⊙ gamma_4d        [C,D]  (cross-chunk carry from h_prev)
8. y_c = y_intra + y_cross + D_skip * x_c       [C,D]  (final output)
9. delta[s] = exp(log_cum[C-1] - log_cum[s])    [C]   (per-step weight for state update)
10. h_next = gamma[C-1] * h_prev
            + (x_dt_c * delta_4d)^T @ B_c       [D,N]  (state update)
```

## Optimization Decisions

Results from KernelOptStore — ALL 14 records evaluated:

| Record ID | Display Name | Applicable | Rationale |
|-----------|-------------|------------|-----------|
| program_caching | custom_program_hash | yes | called 23× per M-layer during prefill; cache hit saves O(kernels×CBs) hash |
| cb_double_buffering | Double-buffer streaming inputs | yes | x_dt/B/C/x all stream chunk-by-chunk from DRAM; double-buffer overlaps BRISC prefetch with TRISC compute |
| rt_args | common_runtime_args for addresses | yes | tensor base addresses change per call (new allocation per prefill); common RT args = 1 memcpy all cores |
| matmul_subblock | Matmul subblock from dst_size | yes | [C,C]@[C,D] output = 2×2 tiles; dst_size=8 (fp32_acc=false) → out_sub_w=2, out_sub_h=2 |
| compute_config | MathFidelity/fp32_dest_acc | yes | HiFi2 + fp32_dest_acc=False: BF16 SSM accumulation is fine (not attention-precision-sensitive); dst_size=8 |
| idle_core_marker | sentinel=65 | yes | 64 cores (8×8) for 64 heads — all active; still set sentinel for safety |
| granularity_check | divisibility check | no | C=64 (multiple of 32), D=64, N=128, H=64 — all tile-aligned by construction |
| dispatch_path | generic_op dispatch | yes | prototype phase uses generic_op + unified kernel descriptor |
| output_prealloc | pre-allocate y and h_out | yes | y=[H,n_chunks,C,D] and h_out=[H,D,N] are pre-allocated once; reused per M-layer call |
| tt_lang_sim_gate | sim validation gate | n/a | Path B (C++ unified); no tt-lang sim needed |
| ccl_persistent_buffers | persistent CCL buffers | n/a | no CCL in this kernel |
| ccl_sub_device | CCL sub-device | n/a | no CCL |
| half_tile | half-tile optimization | no | all shapes are multiples of 32 |
| grid_sizing | cores × work tradeoff | yes | 64 cores × 1 head each = maximum head parallelism; sequential over n_chunks is unavoidable due to state carry |

### Code templates applied (from KB records):

**program_caching:**
```python
desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
```

**common_runtime_args:**
```python
d = copy.deepcopy(desc)
d.kernels[BRISC_IDX].common_runtime_args = [x_dt_addr, B_addr, C_addr, x_addr, logd_addr, h_in_addr, D_skip_addr]
d.kernels[NCRISC_IDX].common_runtime_args = [y_addr, h_out_addr]
```

**cb_double_buffering:**
```cpp
// x_dt per chunk: [C=64, D=64] = 2×2 tiles → double-buffer = 8 tiles
uint32_t x_dt_cb_tiles = XDT_TILES_PER_CHUNK * 2;
// B per chunk: [C=64, N=128] = 2×4 tiles → double-buffer = 16 tiles
uint32_t B_cb_tiles = B_TILES_PER_CHUNK * 2;
```

**matmul_subblock:**
```cpp
// dst_size=8 (fp32_dest_acc_en=false)
// QK output [C=64, C=64] = 2×2 tiles → out_sub_w=2, out_sub_h=2
// state_delta output [D=64, N=128] = 2×4 tiles → fits exactly in dst_size=8
```

**idle_core_marker:**
```cpp
if (my_head_idx >= NUM_HEADS) {
    // sentinel: signal idle state with val=65
    volatile tt_l1_ptr uint32_t* sentinel = reinterpret_cast<...>(SENTINEL_L1_ADDR);
    *sentinel = 65;
    return;
}
```

## Deployment Knob Checklist

- [x] Compute config: HiFi2, fp32_dest_acc_en=False, packer_l1_acc=False
- [x] CB double-buffering for x_dt, B, C, x (×2 tile count each)
- [x] Matmul subblock: out_sub_w=2, out_sub_h=2 from dst_size=8
- [x] Idle core marker (sentinel=65) for any non-64-head configurations
- [ ] Granularity check: not needed (all dims tile-aligned by construction)
- [x] custom_program_hash set at construction (not per-call)
- [x] common_runtime_args for tensor addresses (BRISC: inputs, NCRISC: outputs)
- [x] Grid sizing: 8×8=64 cores, 1 head/core documented and justified
- [x] Dispatch path: generic_op (prototype → production C++ op after PCC passes)
- [x] Output tensor pre-allocation (y and h_out reused across M-layer calls)

## CB Layout

| CB ID | Name | Dtype | Tiles | Buffering | Producer | Consumer |
|-------|------|-------|-------|-----------|----------|---------|
| 0 | cb_x_dt | bf16 | 8 (2×2×2) | double | BRISC | TRISC |
| 1 | cb_B | bf16 | 16 (2×4×2) | double | BRISC | TRISC |
| 2 | cb_C | bf16 | 16 (2×4×2) | double | BRISC | TRISC |
| 3 | cb_x | bf16 | 8 (2×2×2) | double | BRISC | TRISC |
| 4 | cb_logd | bf16 | 4 (2×1×2) | double | BRISC | TRISC |
| 5 | cb_h | bf16 | 8 (2×4) | single | TRISC (init) | TRISC |
| 6 | cb_y | bf16 | 4 (2×2) | single | TRISC | NCRISC |
| 7 | cb_L | bf16 | 4 (2×2) | single | TRISC scratch | TRISC |
| 8 | cb_QK | bf16 | 4 (2×2) | single | TRISC scratch | TRISC |
| 9 | cb_h_out | bf16 | 8 (2×4) | single | TRISC | NCRISC |
| 10 | cb_Dskip | bf16 | 2 (2×1) | single | BRISC (once) | TRISC |

## Grid Strategy
- Grid: 8×8 = 64 cores on Blackhole QB
- Core (ry, cx): handles head h = ry * 8 + cx
- Each core loops over all n_chunks sequentially (state carry makes this unavoidable)
- All 64 cores run in parallel → 64× parallelism across heads
- Python cost: O(1) kernel dispatch per M-layer (was O(n_chunks × ~55 TTNN ops))

## Input Pre-processing (Python, before kernel call)
```python
# Original shapes: [B=1, S_pad, H=64, D=64], [B=1, S_pad, N_GROUPS=8, N=128]
# → head-first layout for contiguous per-core reads
x_dt_hcd = ttnn.reshape(ttnn.permute(x_dt_pad, [0, 2, 1, 3]), [H, n_chunks, C, D])
x_hcd    = ttnn.reshape(ttnn.permute(x_pad,    [0, 2, 1, 3]), [H, n_chunks, C, D])
B_gcd    = ttnn.reshape(ttnn.permute(B_pad,    [0, 2, 1, 3]), [N_GROUPS, n_chunks, C, N])
C_gcd    = ttnn.reshape(ttnn.permute(C_pad,    [0, 2, 1, 3]), [N_GROUPS, n_chunks, C, N])
logd_hc  = ttnn.reshape(ttnn.permute(log_decay_pad, [0, 2, 1]), [H, n_chunks * C])
h_in_hdn = ssm_state  # [B=1, H, D, N] → reshape to [H, D, N] for per-head access
```
These 5 permutes + 5 reshapes = 10 TTNN ops total per M-layer (trivial vs. 94K+ chunk-loop ops).

## Expected Performance

At ISL=256K (n_chunks=4096, 23 M-layers):
- Before: ~94K Python iterations × ~55 dispatches = ~5.2M dispatch events
- After: 23 kernel launches (1 per M-layer)
- Python dispatch saved: 99.999%
- Expected wall-clock: 200–400ms for all 23 M-layers (from 10+ minutes)
- Target speedup: 30–100×

## Sim Validation Result (tt-lang path)
sim_pcc: n/a (Path B)

## Hardware Results
hardware_pcc: pending
hardware_ms_iter: pending
reference_ms_iter: ~600ms (10+ minutes / 23 M-layers ÷ 4 outer-65K-chunks / 4096 inner-chunks)
speedup: pending

## Open Questions
1. Do we need per-chunk causal mask as a CB input, or generate it in TRISC from the cumsum?
   → Decision: generate in TRISC (avoid 4096 × [1,H,C,C] DRAM reads)
2. For the state correction at ISL not multiple of C*n_outer_chunks, is the correction loop
   already handled in Python before calling this kernel?
   → Decision: yes, Python handles the padding/correction; kernel always receives S_pad tokens
3. BRISC reading B_c: each core reads group g = head_h // 8. For HEADS_PER_GROUP=8,
   cores 0-7 all read group 0's B data, 8-15 read group 1's, etc.
   → This is correct: B is not head-specific, only group-specific.
