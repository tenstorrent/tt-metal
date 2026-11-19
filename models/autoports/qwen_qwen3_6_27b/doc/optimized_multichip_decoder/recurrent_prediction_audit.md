# Prediction accumulator register candidate

CPU-only preparation, 2026-11-13. No TTNN/torch import, device use, C++ build,
main/default mutation, existing helper change or timing occurred. Parent owns
all hardware validation. This is only the prediction-loop hypothesis; it does
not include row-only Q/K preparation, column ownership or native recurrence.

## Files and control

- `tt/recurrent_prediction_candidate.py`: pure
  `prediction_sources(reader, compute) -> (reader, transformed_compute)`.
- `tests/probe_multichip_prediction.py`: independent focused entrypoint derived
  from `probe_multichip_recurrent.py`, with extended stress/trace lifetime checks.
- `recurrent_prediction_cpu_check.json`: CPU source/geometry/provenance evidence.

The control is **cumulative_exp_scalar**, including its accepted reader/scalar
changes. The candidate is the same source pair with exactly three edits: insert
the new private SFPU helper, insert its call wrapper, replace the four-call
prediction loop. Exact anchor matching rejects drift and repeated application.
The old unused prediction helper remains in source to bound the delta. Reader,
writer, host partition, CB descriptors, DST allocation, formats and runtime
arguments do not change. CPU reversal of those three edits reproduces the
control compute byte-for-byte for grouped on/off and poison on/off.

Frozen original source comes from Git commit
`8502f42bca20dcad1507fbc8cebf7032220b7d8e`; optimized decoder SHA256 remains
`315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22`.
The transformed control compute SHA256 is
`e890570f8dce78c900c7a681c1525618b413830d31ee1a469985dc5212f49671`;
candidate compute SHA256 is
`7bbcad4fb3ba6ececcba4d968cd97169452a8c372ef045646de3cf91cb57484a`.
At handoff, helper SHA256 is
`ab9c1c500ef04942ae75841c55bc6d7e625d5054e28b32478881b2102e3dbd2a`;
probe SHA256 is
`c8c165bdfeed307dbbda706562b463f8bef87cf3ceb974393eb96269b3a41be3`.
JSON contains complete inspected model-source hashes. No post-ready edit is
permitted without parent acknowledgment.

## Arithmetic and ownership

Source anchors are frozen optimized decoder `_gdn_decay_predict_` at5123 and
kernel-main ordered m-loop at5490. State DST0..3 is already loaded before
prediction; prepared K is DST5, accumulator DST6, decay DST7. Existing code
traverses all bands of m0, then all bands of m1, etc.; each segment stores its
partial prediction in DST6, reloaded by the following segment.

The candidate traverses `(half,band)` outside and ordered m0..3 inside. It
retains the four independent lane-vector accumulators in LREG1..4. K occupies
LREG0; LREG5/6 are paired work registers; decay occupies LREG7. The initial
scalar-broadcast helper executes before any accumulator is live. There is no
SFPU initialization, broadcast replay or other helper that clobbers the register
budget between m0 and m3. The pair helper uses explicit compile-time register
parameters and `always_inline`, not C++ arrays of SFPU values.

For every lane, m0 initializes directly from
`SFPMUL(SFPMUL(old_state,decay),K0)`; it does not add a leading zero. Subsequent
segments preserve each separate state-decay multiplication and state DST store,
then the separate K multiplication and native add operand order
`new_product*1 + previous_sum`. Stores of decayed state still occur before the
work register's K multiplication. Each individual lane's m0,m1,m2,m3 order is
unchanged. SFPMUL and SFPADD remain distinct; no FMA, reassociation, lower fidelity
or persistence rounding change is introduced.

After each band, four completed accumulators are stored to the same DST6
addresses. Existing native REDUCE_COL, delta, update/readout, integer BF16 RNE,
state/output pack masks, CB publication and writer remain unchanged. No current
readout is computed from rounded persistent state. No NoC traffic, semaphore,
CB size, head ownership or global tensor is added. Every prior state page and
compact readout byte still has its original sole owner.

The unchanged startup static assertion guarantees eight FP32 DST tiles. The new
call also requires FUSE_DECAY_PREDICT, FUSE_ACCUMULATOR and HOIST_COL_INIT; this
prevents silently using the helper without the incumbent address-mode setup.

## Source register and dependency audit

Blackhole `ckernel_sfpu_binary_bcast.h:98-152` supplies64 destination address
units/tile, face offsets0/16/32/48, band stride4 and eight bands/tile. The CPU
address check covers each even address0..254 exactly once for the four state
tiles and every even address0..62 exactly once for prepared K's four segments.
The four accumulator registers and key/work/decay registers are disjoint and
consume exactly the eight writable LREG0..7. Architectural constants are
unchanged. Compiler register spilling for RISC-V loop/address temporaries is
**not** assessed without compilation; this is an architectural SFPU register
budget, not compiler-resource output.

`llk_math_eltwise_binary_sfpu_params.h:14` starts SFPU at destination base0 and
passes the explicit indices through. `VectorMode::None` executes the custom
helper once; the helper owns its band/segment traversal. The standard wrapper
retains SFPU startup/done and bounds checks, now once instead of four times.

The paired schedule uses `LOAD5,LOAD6,NOP`, two decay multiplies, NOP, two
stores, two K multiplies, NOP, then the two additions. Each read after a load or
arithmetic write has at least the same two intervening instruction slots used
by the native paired-load/add schedule. Independent accumulators' final stores
follow in1..4 order, leaving sufficient separation from the final pair's adds.
The first segment retains four-way independent load/multiply/store scheduling.
This is based on the existing Blackhole two-cycle dependency comments and
native instruction ordering, not disassembly or execution evidence.

Per band, the incumbent issues16 accumulator stores plus12 accumulator loads;
the new schedule issues4 accumulator stores. This removes24 DST accesses/band,
eight bands/column, four columns/head = **768 accumulator DST accesses/head**.
The paired schedule adds12 dependency NOPs/band relative to the old helpers,
so its explicit SFPU-body instruction-count reduction is only12/band or
**384 instructions/head**, before scalar setup, wrappers and compiled loop
control. Do not equate768 with saved cycles or DRAM requests. State read/write
remains24 MiB/device/token at B32, and neither this count nor the earlier125 us
profile establishes a speedup.

A crucial unproven exactness boundary is retaining an FP32 SFPU partial sum
instead of storing/loading it through DEFAULT-format FP32 DST. Source arithmetic
order is preserved, but signed-zero/subnormal/conversion behavior must be
checked on hardware. A mismatch is a failed exact candidate, not grounds to
relax the unchanged quality gate. There is no native-precision waiver here.

## Probe coverage and lifetime

The probe fixes mesh `[1,4]` and same head ownership. `--all-batches` covers
logical B1..32, including awkward lengths. BF16 and FP32 persistent state,
normal, zero-QKV, zero-state and mixed scale inputs are supported. Scale uses
alternating channel scales1/32 and8 plus16x initial state. Every rank/user has
separate deterministic data and at least two changed-input steps. The optional
`--poison-dead-padding` retains the existing V/g/beta NaN-padding control; Q/K
initialization stays unchanged. Poisoning disables timing.

Every eager, rebound, queued, trace and post-timing comparison requires exact
bits **and finite values** for output and persisted state, per rank/user.
Address rebinding allocates independent QKV/g/beta/state/output buffers before
capture, invokes both cached programs on them and compares. At least50 queued
ordinary calls alternate changed inputs. Both programs and input/reset copies
warm before either capture; all retained device allocations precede captures.
Program-cache entry count must not change during capture.

Traces first replay the eager input/state sequence and compare each step to its
saved eager result, then run at least50 further changed-input steps from changed
state without a reset, comparing both paths. Allocation tracking's positive
control allocates one intentionally live buffer after both captures, verifies
rejection naming its unique ID **without replay**, frees it and verifies both
traces clean. No acknowledgement, skip or cache exemption is allowed.

Only an uninstrumented run records timings: warmed traced **complete custom
recurrence** with matched state-update counts, alternating sample order,
preallocated inputs/state/output, and no reset/copies inside timed trace replay.
This is not whole-layer timing. Watcher, profiler, allocation tracking or poison
all disable timing. All device memory retained by a case is released after
trace release. Reports record source hashes and partial failure evidence.

CPU validation passed: four exact source controls, inverse edit proof,
repeat-transform rejection,32 ordered prediction lane vectors,128 state-vector
addresses and32 key-vector addresses,96 head-partition cases (all32 batches at
1/7/110 cores), explicit eight-register budget, Python compilation and Black.
No C++ compiler or hardware equivalence claim follows from these checks.

## Parent commands

Use the current safe device environment from repo root. These are separate
commands; use fresh artifact paths and retain shell/provenance logs. Parent
chooses the appropriate timeout for its queue.

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_prediction --cpu-check

timeout 300s env TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_prediction --batches 1 3 --state-dtypes bf16 fp32 --modes normal --steps 3 --queued 50 --low-cores 1 --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/prediction_smoke_v1.json

timeout 1800s env TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_prediction --all-batches --state-dtypes bf16 fp32 --modes normal zero_qkv zero_state scale --steps 3 --queued 50 --low-cores 1 --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/prediction_wide_v1.json

timeout 600s env TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_prediction --batches 1 3 17 32 --state-dtypes bf16 fp32 --modes normal zero_state scale --steps 3 --queued 50 --low-cores 1 --poison-dead-padding --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/prediction_poison_v1.json

timeout 600s env TT_METAL_WATCHER=0 TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_DEVICE_PROFILER=0 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_prediction --batches 1 3 17 32 --state-dtypes bf16 --modes normal --steps 3 --queued 50 --low-cores 0 --warmup 3 --timing-samples 7 --replays-per-sample 100 --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/prediction_perf_v1.json
```

Pure source override for a later parent-owned integration (no automatic hook or
registration is provided):

```python
from models.autoports.qwen_qwen3_6_27b.tt.recurrent_variants import sources
from models.autoports.qwen_qwen3_6_27b.tt.recurrent_prediction_candidate import prediction_sources
control = sources(local_reader, frozen_compute, "cumulative_exp_scalar", grouped=resolved_grouped)
candidate = prediction_sources(*control)
# Pass source_overrides=candidate to the existing _local12_recurrent.
# resolved_grouped must be the actual _local12_head_partition result, not just
# the requested grouped_qk flag: B1/B2 resolve grouping off.
```

If exactness fails, preserve first output/state/rank/user failure and distinguish
address/lifetime from partial-sum rounding before further adaptation. If it
passes and wins, measure the coherent whole GDN layer before selection. Column
combination, row-only Q/K and prepared native recurrence remain separate work.
