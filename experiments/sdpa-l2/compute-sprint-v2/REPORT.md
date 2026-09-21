# Frontier compute sprint v2

Completed September 18, 2026. Three agents investigated FP32 scheduling/code
generation, shared compensated-state scheduling, and independent ordering/
qualification plus A/B transfers. All device tests shared one exclusive lock.

## Outcome

Retain D's scoped compiler-function optimization for the tested specialization.
Preserve the exact early-identity path for B/E/G as an optional long-context
candidate, not an unconditional default. Keep the v1 implementations for A/C.
No production dispatch changes were made.

Single-core resident KV, fixed Q256/K512/D128; useful QK+PV FLOPs only:

| Variant | v1 TFLOP/s/core | Best qualified path TFLOP/s/core | Additional resident time reduction | Decision |
|---|---:|---:|---:|---|
| D |0.81911|0.83893|2.36%|Retain scoped O2 function attributes|
| C |1.12546|1.12546|—|Keep v1|
| B |1.63081|1.65667|1.56%|Optional early identity guard|
| A |1.99724|1.99724|—|Keep v1|
| E |1.91546|1.97065|2.80%|Optional early identity guard|
| G |1.91469|1.97002|2.81%|Optional early identity guard|

Improved rows use fresh paired controls. Unchanged A/C rates are carried from
the v1 sustained results, not new gains. C/D use eight Q repeats; B/E/G use
sixteen; all retained timing rows use 512 resident K chunks. Compare normalized
throughput, not raw batch times. Repeated resident KV favors identity branches.

Distinct normal 256K gains are smaller for B/E/G: approximately 0.57%, 1.39%,
and 1.36%, respectively; some shorter/fallback-heavy cases regress. D retains
2.5–2.9% gains on distinct long inputs. All retained candidates preserve raw
output bits in the qualified tests, so their L2/PCC behavior is unchanged.
This is a modest additional gain, not the hoped-for broad 5–10% improvement.

## Contract and measurement

The baseline for every comparison is the **retained v1 winner**, not original
main. D/C/B/A/E/G retain their selected fidelity, destination and L1 precision,
exp/subtraction/reciprocal behavior, compensation and input preprocessing.
Q256/K512/D128, noncausal attention, input CB formats/capacities and buffering,
reader and writer are fixed. New work is isolated here; no production dispatch
or v1 source was modified.

Device: Blackhole on bh-lb-08, IRD 223862, logical device 0, grid 12×10,
firmware 19.13.1/KMD 2.9.0. Successful changed kernels were JIT-compiled and run
on the device using the existing host build; this is not a fresh host-library
build. All hardware work was serialized by the existing exclusive lock and
dirty-marker wrapper. A fresh exact BF16 matmul smoke passed before testing.

Compute-only throughput uses resident repeated KV with no recurring input
data movement. Useful FLOPs count only QK and PV, not extra compensation work:
`4 * 256 * 512 * 128 * q_repeats * k_chunks`. Timings are unprofiled blocking
device-trace replay wall times; compilation, input preprocessing and reference
calculation are outside the timer. Long loops amortize host replay overhead.
This is not a full-chip or model throughput claim. Distinct-KV timings include
data movement and are reported separately, especially for input-dependent
identity branches.

## Completed negative experiments

| Implementation hypothesis | Measured result against v1 | Decision |
|---|---|---|
| Retain BF16 correction across two DST halves | E neutral; B 0.11% less time | Do not retain |
| Overlap next-half compensation SFPU with previous-half PACK | E 0.13% slower; B 0.11% slower; shorter splits neutral | Do not retain |
| FP32 block state packing, srcA-only setup, whole-half ZEROACC flag clearing | Neutral or slower | Do not retain |
| FP32 whole-Q identity PV/state batching | Resident D 1.6% / C 2.0% less time, but distinct 256K D only 0.19–0.45% better and C 0.53–0.59% worse | Do not retain |
| QK2×2 internal microtiles with unchanged PV reduction order | Corrected version D 11.1% / C 4.1% slower | Do not retain |
| Scalar BF16 identity guard at correction site | E 6.40% / B 4.89% slower | Do not retain |
| Unrolled BF16 identity guard at correction site | E 0.53% slower; B 0.32% better | Earlier scheduling is more promising |

Details and limitations: [FP32 scheduling](fp32/REPORT.md),
[independent ordering review](review/ORDERING_REVIEW.md),
[sampled compensation phases](review/PHASE_FINDINGS.md).
The FP32 report predates the separate `codegen/` experiment and its statement
of no retained improvement refers to its scheduling candidates.

## Measured candidates

### Exact identity correction scheduled earlier

Compare all 64 first-column BF16 maxima in each two-query-tile group. Only if
old/new bits agree and are finite, use correction exactly 1.0. Exhaustive device
tests of all 65,280 finite BF16 encodings verify the frozen correction path
returns exactly BF16 1.0 for identical maxima under LoFi and HiFi2. Signed zero
and subnormal inputs are included. Mismatches and nonfinite values take the
unchanged fallback.

The original compensated MADs, rounding and stores remain. The correction CB
reserve/push/wait/pop protocol remains even when its payload is unused because
publication is a required PACK-completion fence. Move only the UNPACK scalar
comparison earlier, after already-issued QK/PV work; deliver decisions through
the original mailboxes at the original correction point. This scheduling
distinction matters: simply adding the check at the correction site regresses.

Resident repeated KV strongly favors this branch. A favorable resident number
alone is not evidence for a general distinct-input or model speedup.

Final E/G resident measurements use 16 Q repeats and 512 resident K chunks,
nine alternating timed pairs after five warmups:

| Variant | v1 ms | Earlier guard ms | v1→candidate TFLOP/s/core | Less time |
|---|---:|---:|---:|---:|
| E |287.0091|278.9719|1.91546→1.97065|2.800%|
| G |287.1246|279.0616|1.91469→1.97002|2.808%|

Distinct-data single-core measurements use 11 alternating timed pairs; they
retain recurring input DM and exclude preprocessing:

| Input case | E v1→candidate ms | G v1→candidate ms |
|---|---:|---:|
| Q256/K262144 normal |18.06830→17.81759|18.00756→17.76224|
| Q256/K32768 normal |2.28379→2.29028|2.28019→2.28639|
| Q2048/K8192 growing maxima |4.52720→4.54050|4.49562→4.50592|
| Q2048/K4096 alternating identity/fallback |2.24946→2.22634|2.24590→2.22308|

Thus E/G improve approximately 1.36–1.39% at 256K, but regress 0.23–0.29% in
the shorter normal/growing-max cases. Retain as an optional long-context
specialization for the qualified geometry, not an unconditional default or
a promised model speedup. No universal sequence-length crossover was measured.
Detailed evidence: `compensated/*-identity-early-resident-v1.json` and
`compensated/identity-early-distinct-timings/`.

B independently transfers the same mechanism and confirms **1.561% less
resident time**, 337.1064→331.8441 ms, **1.63081→1.65667 TFLOP/s/core**.
Its distinct-KV timing repeats the same Q job to lengthen the trace, rather
than using the E/G timing shape. These are separate from its distinct multi-Q
correctness tests:

| B distinct-KV input | Repeated Q jobs | v1→candidate ms | Change in time |
|---|---:|---:|---:|
| K32768 normal |32|84.27371→84.92805|+0.776%|
| K32768 growing maxima |32|84.25733→84.69428|+0.519%|
| K8192 alternating identity/fallback |128|83.80849→83.46258|−0.413%|
| K262144 normal |4|84.42858→83.94789|−0.569%|

B's small long-context benefit is even more conditional than E/G's. Keep v1
as the general default. The early guard is an available exact experimental
specialization, not a new numerical recipe. All final timing rows above check
raw eager and final trace bits, including signed zeros.

The final candidate passes 48 B multi-Q qualification records, including
held-out stress, transitions, odd-three-K and short-two-K loops. E/G pass
35 independently audited records covering held-out multi-Q stress, timing
cases, odd-three-K and first-chunk-only boundaries; canonical preparation,
original/prepared input immutability and two mandatory raw-bit replays are
checked. Root audited the B qualification plus final timing: 63 records.

### Compiler function optimization attributes

A private wrapper selects GCC O2 function optimization around the unchanged
v1 C/D kernel. The default compile/link flags and numerical recipe remain;
this is **not** a descriptor/linker optimization-level sweep. Explicit O3 is
the wrapper control; Os is a negative control. Cached ELF sizes/hashes verify
that the compiler actually generated different code. Smaller code alone does
not establish why a particular choice is faster.

This covers functions parsed inside the wrapper's include region. Firmware
outside it and definitions already included earlier retain their original
settings. Changing production `KernelDescriptor.opt_level` to O2 would alter
a different scope and is **not** qualified by this result. Compiler evidence
pins SFPI 7.74/GCC 15.1 and the tested wrapper placement.

D's sustained resident test (eight Q repeats, 512 K chunks, forward/reverse
ordering) gives 335.5801→327.6520 ms, **2.362% less time** and
**0.81911→0.83893 TFLOP/s/core**. Explicit O3 is neutral at 335.5754 ms.
Separate held-out qualification checks eight distinct-KV distributions against
both original and explicit O3, including normal 256K and seven stress 32K cases.
All outputs and trace replays preserve raw BF16 bits.

Multi-Q distinct-data measurements use Q2048, two cores, four Q jobs/core,
the unchanged reader/writer, ABBA order, and a different seed:

| D distinct-data input | v1 ms | O2 function attributes ms | Less time |
|---|---:|---:|---:|
| K8192 normal |5.7544|5.5979|2.72%|
| K8192 changing maxima |5.7495|5.5924|2.73%|
| K262144 normal |171.897|167.571|2.52%|
| K262144 changing maxima |181.623|176.415|2.87%|

The improvement survives distinct inputs and frequent max changes, unlike
the rejected FP32 whole-Q identity batching. Retain this small D improvement.
C's short compiler screen improves only approximately 0.28%; retain v1 C.
Standalone O2 transfer screens regress E 0.84% and G 0.79%; reject those changes.
They preserve raw output/trace bits and canonical input preparation, with
identical numerical defines and CB metadata after removing only implementation
selection and the compiler-experiment flag. A/B standalone O2 also regress,
by 3.11% and 2.19%, respectively, with neutral explicit-O3 controls. Reject
them without expanding qualification for slower candidates. See
[A/B compiler transfer](review/codegen/REPORT.md) and
[D/E/G compiler investigation](codegen/REPORT.md).
Evidence: `codegen/pragma-sustained-D-v1.json`, `pragma-qual-D-v1.json`,
`pragma-dm-D-v1.json`, `pragma-long-D-v1.json`. Root independently audited
30 sustained/qualification records and 24 distinct-input records.

## What this means for further optimization

The evidence so far supports modest incremental gains, not the previously
estimated 5–10% across the improved variants. The rejected experiments are
useful negative evidence: retaining corrections across DST halves, explicit
cross-half SFPU/PACK overlap, state-pack batching, and QK2×2 rearrangement did
not expose the hoped-for free scheduling gain under this contract.

For C/D, sampled issue boundaries place approximately 54–57% of baseline time
in QK plus overlapped exp; the smaller trailing state region cannot by itself
deliver a dramatic improvement. Those measurements are neither exclusive
engine costs nor proof of instruction-cache pressure. A larger gain likely
needs a deliberately redesigned QK/exp producer-consumer schedule, with new
ownership/fence proofs and bitwise qualification. That is a new measured
engineering effort, not a speedup promised by this sprint.

For B/E/G, the early identity path shows that **where the guard executes**
matters as much as removing the correction work. Its performance remains
dependent on max-update frequency. Existing numerical variants remain the
same six choices; optional exact fast paths do not create new accuracy SKUs.

## Verification and safety

Independent parent audits check saved output-bit hashes, preparation and input
immutability gates where recorded, and useful-throughput arithmetic. They
are evidence audits, not extra hardware runs or complete compiler provenance.
Root also checked that frozen v1 source hashes and the entire preexisting
tracked worktree diff were unchanged.

Final checks: all 22 Python files parse successfully; parent re-ran the E/G
35-record and B 63-record audits, D's saved-output/source/geometry audits,
and the rejected compiler-transfer evidence audits. At handoff (16:59 UTC),
all agents reported no remaining device jobs, and root independently acquired
the shared device lock nonblocking and verified the dirty marker was absent.

Three dirty-marker pauses were inspected before root cleared the marker under
the exclusive lock: the rejected FP32 QK2×2 adapter exceeded half-DST capacity
and failed numerical equality; one guard prototype failed JIT compilation due
to pragma macro placement; a host-only reconnaissance script shadowed Python's
`inspect` module. Processes/device closure were checked. No device reset or
hardware-failure claim was warranted; failed logs remain in the evidence.

Qualification is bounded to the tested shape and feature subset. Wider shapes,
masks/causality, ring schedules, different compiler versions and production
dispatch require separate qualification. Bitwise equivalence preserves the
existing accuracy and existing stress failures; it does not newly establish
a universal L2 threshold.
