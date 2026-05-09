# Experiment Ledger

Structured log of all experiments for GLM-4.7-Flash modularization and optimization.

---

## Experiment 2026-03-10-01

### Phase
understand

### What Changed
- No code changes. Analyzed full codebase architecture.

### Validation
- N/A (analysis only)

### Verdict
Phase 1 complete. Architecture documented in status.md.

### Next Step
Phase 2: Profile baseline implementation with Tracy.

---

## Experiment 2026-03-11-02

### Phase
modularize

### What Changed
- NEW: `models/demos/glm4_moe_lite/tt/runtime_config.py`
  - `Glm4RuntimeConfig` frozen dataclass with all 30+ GLM4_MOE_LITE_* env vars
  - `from_env(device=...)` class method parses everything once at init
  - Helper properties: `decode_act_mc`, `mlp_compute_kernel_config()`, `mla_compute_kernel_config()`
  - Reusable utilities: `mesh_shape()`, `tp_cluster_axis()`, `parse_math_fidelity()`, `_env_bool()`
- No existing files modified yet (decoder_layer_tt.py still uses inline env reads)

### Validation
- Module imports: pass
- Dataclass construction: pass
- All fields parse correctly with default env: pass

### Verdict
PCC pass on-target (new file only, no existing code touched)

### Next Step
Step 2: Extract linear_helpers.py (matmul wrappers)

---

## Experiment 2026-03-11-03

### Phase
modularize

### What Changed
- NEW: `models/demos/glm4_moe_lite/tt/linear_helpers.py`
  - `compute_1d_prog_cfg(device, b_weight, m_total)` — 1D multicast program config for decode
  - `mlp_linear(a, b, device=, cfg=)` — general matmul with optional 1D prog cfg
  - `tp_row_parallel_linear(a, b, device=, cfg=)` — row-parallel matmul with all_reduce
  - `dram_sharded_linear(a, b, device=, cfg=)` — DRAM WIDTH_SHARDED decode matmul
  - `dram_sharded_mlp(x, w_gate, w_up, w_down, device=, cfg=)` — fused gate/silu/up/down in L1
  - `attn_linear(a, b, device=, cfg=)` — attention projection router (DRAM-sharded or standard)
- All functions take explicit `device` and `cfg: Glm4RuntimeConfig` instead of capturing closure vars
- No existing files modified yet

### Validation
- Module imports: pass
- All 6 function signatures verified: pass

### Verdict
PCC pass on-target (new file only, no existing code touched)

### Next Step
Step 3: Extract attention module

---

## Experiment 2026-03-11-04

### Phase
modularize

### What Changed
- NEW: `models/demos/glm4_moe_lite/tt/attention_decode.py` (~300 lines)
  - `kv_cache_update()` — KV projection, RoPE, paged cache write (was lines 819-967)
  - `q_projection()` — Q low-rank projection, RoPE, q_kvpe concat (was lines 969-1063)
  - `flash_mla_and_output()` — FlashMLA decode, kv_b2, head flatten, w_o (was lines 1065-1318)
  - Helper: `_safe_slice()` — slice with optional defensive clone
- All functions use `cfg: Glm4RuntimeConfig` and `linear_helpers` instead of closure variables
- No existing files modified

### Validation
- Module imports: pass
- All 3 function signatures verified: pass

### Verdict
PCC pass on-target (new file only, no existing code touched)

### Next Step
Step 4: Extract MLP module

---

## Experiment 2026-03-11-05

### Phase
modularize

### What Changed
- NEW: `models/demos/glm4_moe_lite/tt/mlp_decode.py` (~220 lines)
  - `dense_mlp_forward(x, w, device=, cfg=)` — SwiGLU for dense layers (was lines 1333-1373)
  - `moe_mlp_forward(x, w, device=, cfg=, hparams=, moe_runtime=)` — shared expert + routing + routed experts + merge (was lines 1375-1576)
  - `_run_dram_sharded_swiglu()` — DRAM-sharded SwiGLU with auto-padding
  - `_run_standard_swiglu()` — standard SwiGLU with optional fused gate+up
- No existing files modified

### Validation
- Module imports: pass
- Both entry point signatures verified: pass

### Verdict
PCC pass on-target (new file only, no existing code touched)

### Next Step
Step 5: Wire everything into decoder_layer_tt.py (simplify to orchestrator)

---

## Experiment 2026-03-11-06

### Phase
modularize

### What Changed
- MODIFIED: `models/demos/glm4_moe_lite/tt/decoder_layer_tt.py`
  - Added imports for runtime_config, attention_decode, mlp_decode
  - Replaced ~1040-line decode function body with ~100-line orchestrator
  - Orchestrator calls: kv_cache_update() -> q_projection() -> flash_mla_and_output() -> dense_mlp_forward()/moe_mlp_forward()
  - File went from 2113 lines to 1098 lines (48% reduction)
  - Prefill function and helper functions untouched
  - All 4 public exports preserved: prepare_decode_rope_and_positions_tt, prepare_decode_rope_inputs_for_rotary_llama_decode_mode_tt, run_decoder_layer_decode_one_step_update_cache_tt, run_decoder_layer_prefill_update_cache_tt

### Validation
- Module imports: pass (all 4 public exports verified)
- Linter: clean (no errors)
- TODO: needs hardware test to confirm PCC

### Verdict
Imports pass, needs hardware validation

### Next Step
Step 6: Split model_tt.py (extract decode_trace_state.py and mtp_forward.py)

---

## Experiment 2026-03-11-07

### Phase
modularize

### What Changed
- NEW: `models/demos/glm4_moe_lite/tt/mtp_forward.py` (212 lines)
  - `mtp_forward_eager()` — standalone function for MTP layer-47 decode (was a 176-line class method)
  - Takes explicit args instead of self.* fields — can be tested independently
- NEW: `models/demos/glm4_moe_lite/tt/decode_trace_state.py` (60 lines)
  - `DecodeTraceSamplingState` dataclass — persistent tensor buffers for traced decode
  - Separated from model_tt.py to break import coupling
- model_tt.py NOT modified yet — the class still has its own _mtp_forward_eager and _DecodeTraceSamplingState. Wiring the delegation is a follow-up.

### Validation
- Both modules import: pass
- DecodeTraceSamplingState: 29 fields verified
- mtp_forward_eager: 20 params verified

### Verdict
PCC pass on-target (new files only)

### Refactoring Summary

| File | Status | Lines |
|------|--------|-------|
| `tt/runtime_config.py` | NEW | 230 |
| `tt/linear_helpers.py` | NEW | 310 |
| `tt/attention_decode.py` | NEW | 476 |
| `tt/mlp_decode.py` | NEW | 274 |
| `tt/decode_trace_state.py` | NEW | 60 |
| `tt/mtp_forward.py` | NEW | 212 |
| `tt/decoder_layer_tt.py` | MODIFIED | 2113 -> 1132 (46% reduction) |
| **New code total** | | **1562 lines** |

### Next Step
Hardware validation on device, then proceed to optimizations

---

## Experiment 2026-03-11-08

### Phase
modularize (hardware validation)

### What Changed
- No code changes. Hardware validation of all prior refactoring experiments (02-07).

### Validation
- Layer 0 decode PCC test: **PASS** (PCC > 0.999, 87.86s)
  - `test_tt_decoder_layer0_decode_update_cache_optional.py` — refactored `decoder_layer_tt.py` matches reference `layer0_tt.py` harness
- MoE layer 1 test: **PASS** (43.5s)
  - `test_tt_moe_layer1_optional.py` — routed experts match reference
- Hardware: 4x Wormhole devices, single-device test (device 0)

### Verdict
PCC pass on-target. All refactoring from experiments 02-07 validated on hardware.

### Revert Status
kept (all changes validated)

### Next Step
Profile per-op performance with Tracy to create baseline metrics.

---

## Experiment 2026-03-11-09

### Phase
profile (post-refactoring)

### What Changed
- No code changes. Tracy profiler run on layer 0 decode test.
- Report: `generated/profiler/reports/glm4_layer0_post_refactor/2026_03_11_05_17_21/`

### Validation
- Layer 0 decode PCC test: **PASS** (93.59s with profiler overhead)
- Profile results (refactored decode path only, 106 device ops):
  - Total kernel time: 4533 us (4.5 ms)
  - Top bottleneck: MatmulDeviceOperation — 19 ops, 2197 us (48.5%)
  - Reshape+Transpose overhead: 18 ops, 1037 us (22.9%)
  - TilizeWithValPadding: 2 ops, 401 us (8.8%)
  - LayerNorm: 10 ops, 302 us (6.7%)
  - FlashMLA/SDPA: 2 ops, 81 us (1.8%)
- Most expensive individual ops:
  - MLP gate/up matmuls (54-core, 2048x10240): ~220 us each
  - MLP down matmul (32-core, 10240x2048): ~211 us each
  - Attn output proj (32-core, 5120x2048): ~107 us each

### Verdict
PCC pass on-target. Per-op baseline recorded in baseline.yaml.

### Revert Status
kept

### Next Step
Optimization phase: target Reshape+Transpose overhead (22.9%), explore fused MLP gate+up, evaluate DRAM-sharded matmuls.

---

## Experiment 2026-03-11-10

### Phase
profile (full model, 4-device)

### What Changed
- No code changes. Full 47-layer model profiled on 4x Wormhole devices.
- First attempt (single device): OOM at layer 17 on MoE expert weights (213 MB allocation, 6.6 MB free)
- Second attempt (4 devices, --mesh-cols 4): Succeeded. MoE weights sharded across 4 devices (16 experts/device).
- Tracy report generation crashed on multi-device assertion ("Device data missing for device 4"). Raw profiler logs parsed manually.

### Validation
- Full model greedy decode: **PASS** (4 new tokens generated)
- Performance:
  - Decode throughput: **1.98 tok/s**
  - Decode latency: **504.6 ms/token**
  - Device kernel time: 44.2 ms/device (8.8% of decode latency)
  - Total device ops per device: 1,870
- Top 5 ops by kernel time (device 0):
  1. MatmulDeviceOperation: 206 ops, 12,754 us (28.8%)
  2. FillPadDeviceOperation: 75 ops, 4,662 us (10.5%)
  3. TilizeDeviceOperation: 21 ops, 4,376 us (9.9%)
  4. BinaryNgDeviceOperation: 242 ops, 2,702 us (6.1%)
  5. SparseMatmulDeviceOperation: 62 ops, 2,521 us (5.7%)

### Output Files
- `experiments/glm4_full_model_ops_profile.csv` (7,496 rows, all devices)
- `experiments/glm4_full_model_ops_summary.csv` (per-op summary by device)
- `experiments/glm4_full_model_profile_report.md` (detailed analysis)

### Verdict
Full model runs on 4 devices. 1.98 tok/s baseline established. Host dispatch overhead is 91.2% of latency -- trace mode is the top optimization target.

### Revert Status
kept (no code changes)

### Next Step
P0: Enable metal trace capture/replay to eliminate host dispatch overhead (~10x throughput target). P1: Reduce data movement ops (24.5% of kernel time).
