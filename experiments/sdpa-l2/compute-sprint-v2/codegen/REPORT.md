# Compute code-generation investigation

## Conclusion

Retain the **D-only O2 function-attribute candidate for the tested fixed
Q256/K512/D128 Blackhole specialization**. It preserves the frozen v1 D
algorithm and output bits while reducing sustained resident time by2.36%,
and distinct multi-Q time by2.5–2.9%. C's0.28% screen gain is not compelling;
keep C v1. E/G transfer screens are slower and rejected; A/B transfer is
independently owned by the review agent.

This is separate from the completed FP32 scheduling investigation, whose
no-new-winner conclusion remains correct for those scheduling candidates.
No production files, host libraries, Q/K chunk sizes, input buffering,
readers/writers, preprocessing, numerical flags, or source algorithm changed.

## What was changed, precisely

Compute defaults to O3 for compilation and linking
(`tt_metal/jit_build/build.cpp:426`). The Python KernelDescriptor constructor
accepts `opt_level`, but this checkout/runtime does not export its
KernelBuildOptLevel enum: see the source/runtime evidence in `inspect-v4.log`.
We did not rebuild bindings or force an integer through that interface.

Instead, private `resident.cpp` and `compute.cpp` wrappers use:

```cpp
#pragma GCC push_options
#pragma GCC optimize("O2")
#include "experiments/sdpa-l2/compute-sprint-v1/fp32/resident.cpp"
#pragma GCC pop_options
```

The distinct-DM wrapper includes the corresponding v1 `compute.cpp`.
**This is a function-optimization-attribute experiment, not an O2
descriptor/global/linker build.** The compile and link commands remain O3.
The original fast-math flags, including `-fno-associative-math`, are unchanged.
The pragma covers function definitions parsed under the private include,
including SDPA helpers and kernel_main; firmware code outside that region
and definitions already included before it are not redefined as O2.

An explicit-O3 private wrapper controls for the wrapper/include boundary:
it matches original-v1 performance and output. No lower-opt LLK workaround
was needed for these kernels. This does not establish that O2 is safe across
TT kernels: the existing FP32 softmax source explicitly warns that its LLK
SETC16 immediate fails to fold at O2 (`softmax_device_operation.hpp:34`).

## Measured performance

Hardware: reservation223862, bh-lb-08, one Blackhole compute core for the
resident benchmark; grid12x10 available. Useful attention FLOPs count QK+PV
only. Preprocessing and reference computation are outside timing.

| Repeated-resident screen, q8/k128 | Original v1 ms | O3 wrapper ms | O2 attributes ms | Os attributes ms |
|---|---:|---:|---:|---:|
| D |84.0857|84.0901|82.0865|102.3006|
| C |61.2067|61.2106|61.0380|85.3867|

Two forward/reverse rounds, five warmups/five replays per configuration;
numbers average the two per-round medians. All output bits matched.
Os produces much smaller code but is substantially slower; reject it.

Final D sustained resident test, q_repeats8/k_chunks512, two forward/reverse
rounds, eight warmups/eight replays per configuration:

| D | Original v1 | O3 wrapper | O2 attributes |
|---|---:|---:|---:|
| Round0 ms |335.5843|335.5859|327.6554|
| Round1 ms |335.5758|335.5650|327.6486|
| Mean of round medians ms |335.5801|335.5754|327.6520|
| Useful TFLOP/s/core |0.81911|0.81912|0.83893|

O2 saves2.3625% time. Relative to the nominal HiFi4 ceiling1.3824TF/core at
1350MHz, the new resident throughput is60.69%; this is a nominal derived
ratio, not a newly measured activity counter or proof of an80% ceiling.
Unprofiled timings are decisive; no profiler perturbation enters this table.

Distinct Q2048, two cores (four distinct Q jobs/core), original reader/writer,
seed1239, paired ABBA rounds:

| Workload | v1 ms | O2 ms | Time reduction |
|---|---:|---:|---:|
| K8192 normal |5.75443|5.59786|2.72%|
| K8192 changing max |5.74945|5.59238|2.73%|
| K262144 normal |171.89737|167.57121|2.52%|
| K262144 changing max |181.62288|176.41454|2.87%|

Outlier and common-K K8192 cases also improve about2.7%. Short runs use
eight timed replays, long runs five, each with five warmups. These are
**real-DM two-core checks**, not chip throughput or no-DM measurements.
Changing-max inputs set Q feature0 to8 and increment K feature0 by4 each
512-token chunk. All these paired outputs and trace replays are bit-exact.

## Numerical qualification and limits

- Short odd-K, multiple-Q resident smoke: C/D original/O3/O2/Os all exact.
- Held-out seed1238 D: normal256K; scaled-QK, outliers, common-Q/K/V,
  constant-V, uniform at32K. Original/O3/O2 match raw BF16 uint16 output
  including signed zero, and each independent trace replay matches raw bits.
- Multi-Q D seed1239: four K8192 distributions plus two K262144 distributions,
  four distinct query jobs/core, all raw bits and trace exact.
- Numeric limitations of the v1 recipe are preserved, not repaired.
- C has only smoke/screen coverage for O2 and is not promoted. Arbitrary
  masks, causal/ring modes, other shapes, architectures, compiler versions,
  or mixed wrapper placements are not qualified by this investigation.

Raw evidence: `pragma-paired-v1.json`, `pragma-sustained-D-v1.json`,
`pragma-qual-D-v1.json`, `pragma-dm-D-v1.json`, `pragma-long-D-v1.json`.
Every row carries numerical descriptors, hashes, and the exact arguments.

## E/G transfer screen: reject

Owned `lowp.cpp` wraps the unchanged compensated compute entry and frozen
v1 `combined_fence/compute_streaming.hpp`. `benchmark_lowp.py` is an isolated
adapter copy: original source algorithms, preprocessing, BF16 DST, LoFi,
two K/V slots and Q256/K512/D128 are unchanged. No new identity guard is
combined with this compiler experiment.

Both E/G pass short q_repeats2/k_chunks3 exact output/replay tests. Paired
resident q_repeats8/k_chunks128, five warmups/nine timed AB/BA replays:

| Variant | v1 ms | O2 attributes ms | Result |
|---|---:|---:|---|
| E |35.93252|36.23260|0.84% slower|
| G |35.95805|36.24052|0.79% slower|

Raw output/replay bits match; preprocessing matches the host preparation
oracle, and original/prepared tensors remain unchanged. E uses BF16 Q,
BFP8 K/V; G uses BF16 Q, RNE-prepared BFP4 K/V. Preprocessing is excluded
from compute timing. Source/CB/dataformat provenance and raw timings are in
`pragma-screen-{E,G}-v1.json`. No broader qualification is claimed or
needed to reject these slower candidates. This result reinforces that D's
O2 benefit is recipe-specific, not a global compiler recommendation.

## Code size and compiler evidence

Host-only `inspect_codegen.py` records cached ELF hashes, section sizes,
largest symbols, compiler identity, and compiler hashes. The measured
function-attribute variants produce different ELF code and symbol factoring,
so the pragma is not silently ignored. Example short-kernel text bytes:

| D wrapper | UNPACK | MATH | PACK |
|---|---:|---:|---:|
| O3 |22508|14832|12068|
| O2 |23044|12680|12024|
| Os |9628|8144|6520|

O2 reduces MATH code but makes UNPACK larger. Symbol inspection shows changed
inlining/factoring. This supports a code-generation mechanism, **not a
demonstrated instruction-cache bottleneck**. Os is a useful counterexample
to assuming that smaller code necessarily runs faster.

Compiler: SFPI7.74.0[897], GCC15.1.0. SHA256:

- g++ driver: `2dfa8cb1c02fb350139bdfe886d9dfe2feb687a88757dd8598f96a1832e21a9e`
- cc1plus: `dea06abeea78a7b90b437905e53347e03b00436972cfab6b08dd64db2d9f3e74`
- Resident wrapper: `386111d43a42717bd572f08ebc4b1b0ea6b3bc23053607adbabe929fc834fe7c`
- Distinct-DM wrapper: `0ae277d13bb30a0be1cfa3a89c88178bb9ed06acded3b8fa54ad851d8b9f3b1e`
- Frozen v1 FP32 header: `fef42cc8a8bfb272e4bdd401902c51ca7c880fc2b4f84f25fc930da69e76462c`

The initial host inspection script was unfortunately named `inspect.py` and
shadowed Python's standard library. It failed during import before any
device open. It was renamed, and the coordinator cleared the dirty guard;
no reset or device failure occurred. Subsequent host-only inspections run
outside the device wrapper. All actual device jobs use the global lock.

## Reproduction

Run from the remote repository through the shared v1 `run_locked.sh`:

```text
experiments/sdpa-l2/compute-sprint-v2/codegen/bench_codegen.py --label NEW --variants D --levels original,O3,O2 --q-repeats 8 --k-chunks 512 --warmup 8 --iters 8 --rounds 2
experiments/sdpa-l2/compute-sprint-v2/codegen/bench_codegen.py --label NEW --variants D --levels original,O3,O2 --qualify --seed 1238
experiments/sdpa-l2/compute-sprint-v2/codegen/fullchip_codegen.py --label NEW --variant D --candidate O2 --q-length 2048 --k-length 262144 --distributions normal,changing_max --rounds 2 --iters 5 --seed 1239
```

All successful configurations JIT-compiled on the allocated Blackhole.
All four owned Python drivers also passed local `py_compile` syntax checks.
No host rebuild was required. Source and results remain private experiments;
integration should scope this to the qualified D kernel and retain compiler
and numerical regression tests, rather than globally lowering optimization.
