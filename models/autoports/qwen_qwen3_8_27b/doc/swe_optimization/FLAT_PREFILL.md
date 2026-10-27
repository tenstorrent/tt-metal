# Flat prefill projection geometry

Local correctness, timing, profiles, watcher, advice experiments and independent
review pass. The feature remains default off; CI serving and
SWE promotion are separate gates.

Set `QWEN38_FLAT_PREFILL=1` to fold contiguous `[B,S,K]` rows into `[1,B*S,K]`
for the admitted prefill matmuls, then restore `[B,S,logical_N]`. The guard admits
batches 8, 16 and 32, physical lengths 256/512, tiled interleaved inputs, BFP4
weights, LoFi, grids 11×10/11×8, and the measured K blocks: mix 4/16,
out/down 4, gate/up 20. Other contracts use the previous path. Gate/up use a
block-height cap of 8; the other admitted roles use 16, with legal divisors.

K blocking, activation/output dtype, fidelity, weights, sampling, residual and
collective boundaries are preserved. No persistent tensor or cache-format change
is introduced. Native context remains 262144 tokens. The private 8192-token
allocation below is a test fixture, not a reduced serving capability.

## Measured results

Both arms enable grouped and masked recurrence. Full-model timing uses all 64
layers, B16 with 8 active requests, 4096 input/128 output tokens, physical chunks
of 512, three clean repeats, on-device sampling and buffered traced decode.

| Full model median | Baseline | Flat |
|---|---:|---:|
| TTFT | 8771.253475 ms | 7727.073450 ms |
| Decode interval | 8.082567323 s | 8.083341616 s |
| Decode tokens/s per active user | 15.712829 | 15.711324 |

TTFT falls **11.9046%** (1.135133× inverse-latency ratio). Decode changes by
+0.00958%, effectively unchanged. Tokens match exactly. Clean timing disables
allocation tracking; strict acceptance below uses tracking=1 and
program-cache-skip=0. No inner-loop host sampling, input refresh or readback is
introduced, and all four capture/trace identities remain stable.

The matched 1350 MHz profile uses real layers 0/3 plus embedding, norm, head,
sampler and feedback. It is not an all-layer timing.

| Reduced complete maximum per-ASIC envelope | Baseline | Flat |
|---|---:|---:|
| Prefill | 283297.088889 µs | 248348.825185 µs |
| Prefill device operations, all ASICs | 7404 | 7404 |
| Traced decode | 3540.509630 µs | 3544.221481 µs |
| Decode device operations, all ASICs | 888 | 888 |

Reduced prefill improves 1.140722×; decode differs by +0.1048% in this single
window. Operation-kind counts, source, fixture and precision match. Timing takes
the maximum of complete per-ASIC firmware envelopes, including all operations
and gaps; it never subtracts unsynchronized clocks across chips. Derived advice
tables blank only two outside-predecessor gap fields on four boundary rows per
phase. Raw rows and hashes remain intact. Unused all-active trace 0 is expected:
8-of-16 uses masked trace 1 and buffered sampler 3. Both arms use the same fixed
profiler clock source; the original compiled runtime is untouched.

## Correctness and diagnostics

The full strict pair has five exact 512-entry state maps on all four ASICs:
384 convolution/recurrent entries and 128 decoded paged K/V entries at each
checkpoint. All 12 continuation-logit hashes, fixed/ragged tokens and 24 shared
native-chat completions match. Reduced real-layer controls additionally cover
B8 and B32 sibling projections, five 16-entry cache maps and continuation logits.

Eight extra native-chat cases actually select physical buckets 256/512 at
prompt lengths 226/238/239 and 482/494/495. Every completion matches the control
and reaches EOS within 512 output tokens. Manual review checks the requested
three-sentence English explanation, JSON apple counts 4/3, and order-preserving
Python function/example. Native thinking text is retained in the raw receipts.

A separate B32/M16 run with `TT_METAL_WATCHER=10`, `NOINLINE=1` and all watcher
features enabled matches all five state maps, 12 continuation logits and
fixed/ragged outputs. No watcher error is reported; all four devices detach
normally. Cold watcher preparation took about 17 minutes; its timings are not
performance evidence. Eighteen serving-contract CPU tests also pass.

There are 768 stdlib policy comparisons across all 64 layers, grouped/masked
flags 0/1, and flat absent/0/1. Absent/0 preserve the published policy exactly;
1 adds only the intended opt-in. Invalid values fail.

## Advice experiments and retained choices

Input-L1 placement was screened on ten real layer/role calls at B16/M8/S512,
including the copy and unchanged projection. Seven native-height capacity
failures were retried at the largest smaller legal fitting height. All ten
fitting candidates are exact on all ASICs before/after 90 replays, but slower:
speed ratios 0.74994–0.95995. Retain the selected DRAM path. This does not reject
every possible L1 layout or a future cross-projection reuse strategy.

The suggested 110-core gate/up grid was tested on all four roles at every legal
height 26/13/2/1, preserving K, dtype, fidelity and DRAM input/output. H26 requires
2235008 bytes of static buffers versus 1572864 bytes of physical L1. H13 ends at
1323136 versus the first live allocation at 1293696. H2/H1 fit and are exact,
but slower; each role's best ratio is 0.88732–0.90911. Retain the 88-core grid.
These are measured choices for this target, not a global optimality claim.

Fused-op aggregate HiFi4 labels come from the trailing reduction; embedded
matmul attributes remain LoFi. The head matmul's 26% reduced-fixture share is
about 1.46% of the measured full-model decode step. BF8 head sharding remains a
separate unmeasured opportunity; historical BF16 trials do not reject it.

## Provenance and limits

The first all-H16 prototype failed gate/up L1 allocation before execution; its
raw failure and the adapted controls are retained. The original full strict
receipt omitted two runtime files from its source map. A preserved 27-file
snapshot, exact imported-bytecode/source proof under the runtime Python, and
post-run hash checks close that gap. The collector was then expanded without
rewriting original receipts. All 27 runtime sources still match after the last
hardware run.

An advice-probe descriptor initially included a process address in the default
compute-config representation. The cross-process inventory check failed closed.
Stable layer/role IDs and structured compute fields replaced it; math/timing
function ASTs are unchanged. Original receipts and source are intact and are
attributed only to the actual roles recorded in each receipt.

The portable `tests/probe_prefill_memory.py` reproduces placement and grid
controls with explicit checkpoint, role, memory and geometry arguments. Compact
JSON summaries bind the exact original evidence hashes. Raw CSV, Tracy, logs,
archives and token arrays remain local or in CI.

Local tests use the retained compiled backend. They establish neither CI serving
performance nor SWE accuracy. The published mask-only image contains no flat
implementation. A new immutable image and serving smoke are required next.
