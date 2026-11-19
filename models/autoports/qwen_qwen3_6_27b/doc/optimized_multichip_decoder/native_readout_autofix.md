# AutoFix preparation: full-FP32 readout of prepared native state

CPU-only, 2026-11-13. Applied the AutoFix skill to the completed
[AutoDebug diagnosis](debug_native_precision/AUTODEBUG.md). The parent owns all
hardware. No TTNN/torch import, C++ build, device operation, native source edit,
existing prepared helper/probe edit, main edit or prediction-helper edit occurred.
This report records an **unverified readout-only hypothesis**, not a successful
numerical repair or performance result.

## Failing contract and discriminator

The original public native path failed B1 zero-state step0 at rank1/user0/head6,
PCC0.9892162833519499 against the unchanged0.995 per-head gate. Its FPU readout
re-reads a live FP32 update through narrowed FPU operands. The prepared path
retains this contraction and also marks CB8 direct-FP32 for BF16 persistence
while still feeding CB8 to an FPU readout consumer. The earliest zero-state
case excludes previous persistence rounding and nonzero-state prediction as
necessary causes; it does not prove that readout alone repairs later steps.

First hypothesis: read the same live native state with direct-FP32 copies,
SFPU products and the custom ordered contraction. Keep prediction, residual,
rank-one update and persistence arithmetic unchanged. Decisive outcomes are
per-head/user output and state PCC against dtype-matched FP64/custom controls,
plus **bitwise identity of corrected and original prepared persistent state**.
If output still fails with identical state, this readout-only hypothesis is
insufficient. If state changes, investigate format transitions/ownership before
attributing an improvement to readout precision.

## New files and API

- `tt/prepared_native_precise_readout.py` exports
  `PreparedNativePreciseReadout(initial_state)` and its inherited-style
  `.step(packed_qkv, g, beta) -> compact_fp32_output` interface. It owns independent
  retained state/prepared/output buffers, with no global monkeypatch or registry.
- `precise_readout_sources(reader, compute)` is a pure string adaptation of the
  existing prepared reader and `recurrence_source()` output. Every source anchor
  must match once; caller must also select recurrence direct unpack on CB0/CB8.
- `tests/probe_prepared_native_precise_readout.py` isolates custom, original
  prepared and corrected prepared paths, each with BF16 and FP32 persistence.

The subclass reuses preparation kernels, native helpers and retained allocation
setup from the original prepared helper. Its copied `step` descriptor builder
has exactly two AST-visible changes: use its adapted reader, and configure
recurrence CB0 and CB8 direct-FP32 for **both** persistence dtypes. Preparation
configuration, all other recurrence CB modes, CB capacities, core partition,
runtime arguments, output/state ownership and writer remain unchanged.
No new global tensor, CB, collective, reshard, launch or memory policy is added.

The existing superclass reads native kernel source at construction. That
prototype behavior is retained and source hashes are reported; a maintained
static kernel source/variant is required before production integration. The
pure transform itself performs no file/AST reads. CPU-only AST inspection
checks the source adaptation; device execution does not rewrite Python methods.

## Precision and phase contracts

The prepared reader now materializes only Q as a column rather than a row.
For Q element i, old row-tile address was
`(i//32)*1024 + ((i%32)//16)*256 + i%16`; new column-tile address is
`(i//32)*1024 + ((i%32)//16)*512 + (i%16)*16`.
All128 values remain FP32 bit copies of the same prepared record. Existing
reader zeroing covers every non-column lane. K/V/g/beta mapping is byte-for-byte
unchanged; there is no extra transpose kernel or Q intermediate.

The new contraction uses DST0 as accumulator, DST1 as Q column and DST2 as
later state products, fitting the existing four-tile half-sync contract. For
each value-column n, it loads live matrix tile `4*m+n` from CB8 and Q tile m
from CB0 directly into FP32 Dest, multiplies with SFPU column broadcast, folds
m0,m1,m2,m3 in order with SFPU addition, then invokes the existing SFPU
REDUCE_COL topology. Output packs with the same row-only mask as the custom
`contract()` helper; the mask is cleared before persistence packing. Source
reference: `optimized_decoder.py:5359`, `contract()`.

CB0 Q now has **no FPU consumer**: its old readout matmul was replaced, startup
no longer uses it, and the BF16-expansion restore no longer restores from it.
CB8 likewise has no FPU consumer in this T1 prepared schedule: update packs it,
SFPU readout copies it, BF16 persistence optionally copies it, and it is popped
before the next independent head. Thus CB0/CB8 direct-FP32 is appropriate for
both BF16 and FP32 persistent storage. Remaining FPU operands keep Default
unpack. This contract must not be transplanted unchanged into native T>1, where
live state is consumed by the next token's FPU decay.

Explicit transitions avoid relying on an enabled compute sentinel:

1. Startup uses first-phase FPU-compatible CB5 state and CB3 decay.
2. At each head start, `reconfig_data_format(cb_S,cb_decay)` restores both FPU
   source formats. The BF16 local expansion still reconfigures SrcA to CB14,
   performs the original copy/pack loop and restores SrcA to CB5.
3. Native decay, prediction, residual, K transpose and update helpers are
   unchanged. No new arithmetic or direct-only CB is introduced into them.
4. Each readout matrix/Q copy explicitly selects its direct-FP32 SrcA format
   before `copy_init`/`copy_tile`. An SFPU product after an ordinary narrowed
   copy would not satisfy this hypothesis.
5. BF16 persistence explicitly selects CB8's direct format before the unchanged
   native integer-RNE/pack loop. It occurs after live FP32 readout. FP32
   persistence remains the unchanged rank-one-update pack. Next head restores
   both FPU formats before its arithmetic.

Source basis: AutoDebug's lowering audit; `api/compute/tile_move_copy.h:23`
(copy_init does not itself configure data types),
`api/compute/reconfig_data_format.h:130` (new-only reconfiguration avoids stale
old-format claims), and native update at `fused_recurrent_gated_delta_rule.cpp:63`.
Native helper definitions, including rank-one update and prediction, remain
verbatim. The source proof establishes unchanged arithmetic text and planned
consumers, not bitwise hardware state or passing accuracy.

## Focused probe and failure collection

Six paths are scored: custom BF16/FP32, original prepared FP32/BF16, corrected
prepared FP32/BF16. The public native wrapper is deliberately absent; its known
quality failure cannot prevent scoring this hypothesis. Original prepared
controls remain measured and their failures are saved with `required=false`.
They are reported separately through `original_prepared_control_status`, never
relabelled as numerical passes.

Required gates remain0.995 **per rank, user and head** for corrected/custom
output and state against their own BF16/FP32 persistence-matched FP64 trajectory.
Corrected paths also compare against the dtype-matched custom path with the
same gates. Corrected versus original prepared persistent state requires exact
bits. Cross-dtype trajectory differences are not used as substitutes for these
matched controls. Every check-step saves all20 output/state PCC metric records
and two same-state bit comparisons before a case-level required assertion.
Original prepared numerical failures cannot abort candidate scoring. Runtime,
structural, nonfinite, allocation or compilation errors remain real failures;
there is no fallback or threshold relaxation.

Default first case is B1 zero-state, preserving the original random seed and
input generation. `--all-batches` supports B1..32; explicit boundary batches
include B9/B10 and B17/B31/B32 with multiple heads per worker. Normal, zero-QKV,
zero-state and mixed-scale cases are supported. Multiple changed inputs, changed
initial state, fresh owner/address rebinding, self-eager/trace exact hashes,
prepared-record checks and optional50+ queued changed-input traces remain.
BF16 and FP32 persistence are always both exercised.

All retained owners/inputs/outputs and program warmups exist before first
capture. Rebound owners are tested and freed before capture. Allocation tracking
includes a deliberately live post-capture buffer, requires rejection naming its
ID **without replay**, frees it and verifies all traces clean; no exemptions.
Timings are skipped for watcher, profiler, tracking or `--skip-timing` runs.
Uninstrumented timing covers raw packed QKV → preparation → recurrence →
compact output plus state updates, with matched replay counts, followed by
full trajectory metrics. This is not whole-decoder timing.

If candidate quality fails, all phases of that case are saved before the
case-level assertion; later cases then stop. Therefore the small first command
is the fastest discriminator. The subsequent wide/stress/perf commands are
conditional on the earlier results, and are parent-owned.

## CPU validation and provenance

[CPU evidence](native_readout_cpu_v2.json) passes:

- Inverting every intentional source edit reproduces original prepared reader
  and compute exactly; native helper/update/persistence arithmetic is unchanged.
- Original/corrected host step ASTs differ only in adapted reader binding and
  recurrence direct-unpack `(0,8)`.
- All128 Q column elements have unique, in-range tile addresses; remaining
  prepared geometry/compact-output/head ownership checks pass for all32 batches.
- No CB count/memory increase; three FP32 DST slots suffice for readout.
- Python compilation and formatting pass; CPU check asserts no TTNN/torch import.

No compiler, LLK binary, watcher, exact state or numerical/performance result is
claimed. The helper SHA256 at handoff is
`c74d5b0d15336aa0366066b83dc060c6c4a47710c344de33d2ab91d039d3235b`;
probe SHA256 is
`571e906853d48dce44d6e2fc18256fde899fab45820423490980b9b7fd4c92cd`.
Original prepared helper SHA256 is
`0e5991945bc595b784ece95a7c04db1395aa3abf81cab9f83b5b859b576d96f4`;
native kernel SHA256 is
`08fd7439a9196a2d6b76ae18f505c4dfa1e77035f33f5569834f43d30454647b`.
Generated reader/compute hashes are recorded in CPU JSON. Earlier CPU v1 is
retained as intermediate provenance; v2 describes the final handoff probe.

## Parent launch commands

Run from repository root under the parent's normal hardware environment, using
fresh artifact/log paths. No device command below was executed by this agent.

```bash
timeout 300s env TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_prepared_native_precise_readout --batches 1 --modes zero_state --steps 2 --bank-size 3 --queued 0 --skip-timing --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/native_readout_zero_smoke_v1.json

timeout 1800s env TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_prepared_native_precise_readout --batches 1 3 9 10 17 31 32 --modes zero_state normal zero_qkv scale --steps 4 --queued 50 --skip-timing --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/native_readout_stress_v1.json

timeout 2400s env TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_prepared_native_precise_readout --all-batches --modes zero_state normal zero_qkv scale --steps 4 --queued 50 --skip-timing --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/native_readout_all32_v1.json

timeout 900s env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_prepared_native_precise_readout --batches 1 3 10 32 --modes normal --steps 3 --queued 50 --warmup 2 --samples 7 --replays 100 --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/native_readout_perf_v1.json
```

Opt-in parent integration uses the new owner rather than modifying a live
prepared owner: construct `PreparedNativePreciseReadout(initial_state)` before
warmup/capture, then call `.step()` with the same packed/g/beta interface. The
helper remains isolated. Selection requires a proven fix, later trajectories,
actual decoder quality gates and measured coherent whole-layer benefit. If
readout alone fails, inspect same-state/output evidence before attempting the
next diagnosed native arithmetic phase; a failed first API invocation is not a
numerical rejection.

Parent prelaunch review fixed watcher-presence detection and the performance invocation: watcher must be unset, not set to zero. No device evidence used the erroneous invocation.

## Parent device experiments

`native_readout_zero_smoke_v1` reached compilation and closed allfour drivers normally after missing `sfpu_reduce` declarations. Added the defining `api/compute/compute_kernel_api.h` header; no arithmetic change. Fresh `native_readout_zero_smoke_v2` passes B1 zero-state, two changed eager steps, rebind/changed-state/traces and allocation positive control under watcher10/tracking1/skip0. Worst first-step per-head output PCC changes from0.9887767045 (bothoriginal prepared state dtypes) to0.9999997093 (bothcorrected dtypes). Stored state is bit-identical original-versus-corrected. At secondstep corrected FP32/BF16 outputs reach0.9999972091/0.9999743721. Native prediction/update and BF16 persistence remain unchanged. This confirms readout narrowing caused the earliest observed failure; it does not establish long-trajectory accuracy or a speed gain. No performance timing collected in this instrumented smoke.
