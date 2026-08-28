# Claude gate review — KDA performance-model development specification

Read-only verification against this worktree at cbe7d1b39a7. No spec edits
made.

## Evidence

### Clock units — resolved

- get_clock_rate_mhz() returns int MHz:
  tt_metal/api/tt-metalium/device.hpp:89,
  tt_metal/impl/device/device.cpp:725, and
  tt_metal/distributed/mesh_device.cpp:1852. The spec's section 3.1 source is
  integer by construction.
- The spec now uses integer clock_mhz in both occurrences (spec:55 and
  spec:118) with the identical exact form
  ceil_div(cycles * 1000, clock_mhz). No float/GHz divisor survives anywhere
  in the arithmetic path; section 7 explicitly routes both languages through
  section 3.2.
- Python's source frequency really is GHz (device cycles per ns):
  tests/ttnn/profiling/realtime_profiler_utils.py:50 divides the cycle delta by
  record.frequency to obtain ns, and
  tt_metal/distributed/realtime_profiler_manager.cpp:1433-1437 computes it as
  slope / tracy_ratio = device cycles per nanosecond (GHz), with aiclk/1000
  fallback. So floor(frequency_ghz * 1000 + 0.5) (spec:440) is a well-defined
  nearest-MHz integer, positive in practice (the utility drops records with
  frequency <= 0, line 43). The spec's characterization of it as a regression
  estimate distinct from ARC AICLK is factually correct.

### KDA work formulas — re-derived from the repo oracles, all exact

- Section 5.1 vs
  models/demos/deepseek_v3_d_p/reference/kda/ops.py:106-115: square + normalize
  + weight + gate = 4RV multiplies, RV reduction inputs, R eps adds, R + RV
  SFPU. Matches.
- Section 5.2 vs
  tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_qkv_causal_conv1d_silu.py:114-122:
  4 tap products (4E), 4-term sum (3E), one SiLU per output. Matches.
- Section 5.3 vs test_reduce_affine_transforms.py:146-168: per composition
  A·A (2K³) + A·B (2K²V) + KV adds, over P = H(G-1). Matches.
- Section 5.4 vs test_affine_exclusive_scan.py:151-178: G-1 guarded updates,
  2K²V + KV; the final carry is genuinely never computed. Matches.
- Section 5.5 vs test_prepare_chunk_recurrence.py:81-114: multiplies
  3CK+2CK+CV+2CK+CK+CK+CK = 10CK + CV; adds
  2C + (C-1)K + CK + C²; reductions 2CK; dense
  4C²K + C(C-1)(C+1)/3; SFPU 2C + 3CK + K. Matches term for term,
  including counting k*inverse_decay and kd once as shared subexpressions.
- Sections 5.6/5.7 vs recurrent_chunk_scan_test_utils.py:58-88: recurrent per
  chunk gives exactly 6CKV + 4C²V, KV, 2CV + KV; summary is two _scan_state
  passes (4CKV + 2C²V each — no output terms, which is precisely why it is not
  2× section 5.6) plus the HKV A-subtraction. Matches. The 4C²V term in
  section 5.6 is correct for K != V because both C² matmuls
  (t_inv·(...) and intra·value_new) are C×C · C×V; the op only requires K == V
  in summary mode (recurrent_chunk_scan_device_operation.cpp:97), consistent
  with the spec.

### Profiler adapter contract

Signature and structural detection match
ttnn/api/ttnn/mesh_device_operation_adapter.hpp:1039-1047 (three args, no
device, concrete OpPerformanceModelGeneral<tensor_return_value_t>). Field names
match ttnn/api/ttnn/operation.hpp:95-140;
get_input_bws/get_output_bws carry TT_ASSERT(ideal_ns > 0), which the spec's
clamp-to-1 satisfies. Arch-gated zero-model precedent exists at
indexer_score_device_operation.cpp:518-528.

### DRAM semantics

All three claimed deliberate divergences from the generic model are real:
binary GiB conversion (ttnn/core/operation.cpp:50), largest-tensor time
(lines 60-76), and L1 to 1.0f rather than zero (line 52); the 512 GB/s
Blackhole constant/comment is at lines 36-42.

### Scope

Six device-operation pairs are on disk; recurrent/summary is a mode enum
(recurrent_chunk_scan_device_operation_types.hpp:17); there are seven Python
KDA test files and exactly nine production/regression performance entry points,
all consuming profile_realtime_program and reading duration_ns/runtime_id. All
needed dimensions live in the params structs, so no factory access is required;
initial_state is std::optional, matching the skip-absent-optional-tensor rule.
CMake targets exist as described.

### Fallbacks

Unsupported architecture, non-device input, invalid fidelity, overflow, and
narrowing all map to a zero estimate plus warning, converted to a wrapper
clamped at 1. This is internally consistent with section 4's zero-ideal rule
and with fidelity_multiplier returning 0 for MathFidelity::Invalid.

## Blocking issues

None.

## Non-blocking comments

1. Section 3.1's fallback list does not name clock_mhz <= 0. A failed ARC read
   would divide by zero in C++; Python is already shielded by the profiler's
   frequency <= 0 filter. Recommend folding it into the same
   warning/zero-estimate path.
2. Section 8.2 should state that Tracy expectations must apply the >=1 wrapper
   clamp for zero-work cases such as G = 1, where the estimator reports 0 but
   Tracy will show 1.
3. profile_realtime_program_merged does not carry frequency_ghz through.
   Harmless today because none of the nine KDA entry points use it.
4. Zero bytes for L1 slots means Tracy's per-tensor byte/bandwidth arrays
   under-report L1 traffic relative to the generic model. The spec calls this
   intentional.

## Verdict

PASS

---

## Post-fast-forward gate at current PR6 HEAD

Claude repeated a read-only gate after branch momcilo/kda_perf_model was
fast-forwarded by 21 commits to
ee7353a69eb38a2da9e059b276cf546c0dc46811.

### Evidence

- The spec base metadata matches current HEAD.
- Every section 5.1–5.7 formula was re-derived from the current on-disk
  mathematical oracles and remained exact.
- The six device-operation directories, seven APIs, recurrent/summary mode,
  optional initial state, and logical dimension attributes remain unchanged.
- The concrete three-argument MeshDeviceOperationAdapter performance-model
  hook and OpPerformanceModelGeneral field contract remain unchanged.
- Exactly nine performance entry points remain across seven KDA test files.
- The new PR6 commits change recurrent/summary implementation details, test
  coverage, and calibrated measured durations; none invalidate the
  implementation-independent model or encode a value used by this spec.
- The KDA Common target, realtime-profiler frequency plumbing need, generic
  model divergences, and gtest source registry remain as specified.

### Blocking issues

None.

### Verdict

PASS
