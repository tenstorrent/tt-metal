# Grouped FP32 recurrent update candidate

Status: experimental, default off. Local full-model correctness and warmed timing pass; serving/SWE promotion remains pending.

The prior reduced profile attributes about2037µs to a B32 linear-attention
decoder versus1004µs for a full-attention decoder. There are48 linear layers.
These historical profiled numbers identify a target, not a new speed prediction.

## Operation audit and hypothesis

| Boundary | Existing path | Candidate | Constraints |
|---|---|---|---|
| Q/K/V projection and convolution | Existing packed BF4/LoFi projections and BF16 convolution | Same | Real shapes, policy and TP4 residual layout |
| FP32 recurrent update | Split users into groups of at most9; scale/reformat/gather state for each group; one head per core | Flatten all user/head pairs, process2–4 sequential heads on each core in one invocation | Same FP32 state/scalars, HiFi4 matmuls, per-head order and arithmetic |
| Output/state joining | Concatenate each group's output and state | Already contiguous user/head order | Preserve exact state and all inactive rows |
| Mask, output norm/gate and row projection | Existing where/copy, BF16 norm/gate and native matmul/reduce-scatter | Same | No precision, collective-axis or next-layer residual-contract change |

The recurrence has no inter-chip collective. Each chip retains its12 local value
heads and4 local Q/K heads. B16 uses192 user/head pairs across96 cores at two
heads/core; B32 uses384 pairs across96 cores at four/core. The core-local circular
buffer sizes are unchanged. Between heads, the compute kernel retires the final
local carried-state CB; the writer consumes the separate final-state copy.
Every head reads and writes a disjoint state/output region. This also permits
in-place final state; that path still needs real-layer and serving validation.

Prediction before real-layer measurement: eliminating repeated tensor movement
and primitive invocations could materially reduce the linear decoder time at
B16/B32. Per-core arithmetic is not reduced. A recurrence-only gain must not be
multiplied into a claimed full-model or SWE speedup.

## Small device checks

The initial B32 prototype measured1114→528µs (2.11×), with identical state and
output across three advancing updates. Strict allocation tracking then rejected
the probe's retention of a newly captured output across the other trace. That
harness was corrected to preallocate both outputs before any capture and copy
results into stable buffers. No corruptible-buffer exemption was added; the
initial timings are exploratory, not acceptance evidence.

With the corrected harness and allocation/program-cache tracking enabled:

| Batch / heads per core | Recurrence-only speedup | Correctness |
|---|---:|---|
| 16 /2 | 1.814× | Exact output/state on all four chips |
| 17 /2 | 1.765× | Exact output/state on all four chips |
| 31 /4 | 2.105× | Exact output/state on all four chips |
| 32 /4 | 2.103× | Exact output/state on all four chips |

Each row checks three advancing updates and final equality after another100
timed updates. Inputs use full per-chip Qwen dimensions, normalized FP32 Q/K,
independent per-row initial states and fixed gates; tensors are replicated across
the four devices. This is a primitive regression, not a real-layer/HF accuracy
measurement. Timings include five20-replay windows with tracking overhead;
clean timing remains separate. Raw results are under
`/home/mvasiljev/build/qwen38-swe/grouped-recurrence/*tracked-final.json`.

## Required next checks

The real-weight layer0 comparison uses the selected precision policy and native
TP4 sharded decode. Its experimental subclass changes only the decode split extent
and recurrence function; it retains the generated method source and hash for
review. The production class is unchanged. Check masked/all-inactive rows and
in-place/unmasked state, with strict allocations, then obtain clean timings.
Before any serving promotion, implement the selected path cleanly, profile a
representative non-serving layer, validate full-model output/state/trace lifetime,
and run matched small SWE. Retain native262144 admission and the existing cache
contract. No full-suite forecast is justified yet.


## Full-stack control and production wiring (2026-10-27)

The B16 all64-layer control completed with physical prefill256, cache8192
for this fixed4096/128 workload (native capability remains262144).
Three repeats gave TTFT12144.06/12145.72/12144.93ms and decode
13.80870/13.80882/13.80802 tokens/s/user. Each steady-state loop had127
model and sampling replays, zero captures or host refreshes, one final readback.
All three output token checksums matched; ragged33/65/1 requests repeated
exactly without new traces. Load198.305s and preparation303.753s are startup,
excluded from warmed times. Raw: full-b16-baseline-p256.json in local evidence.
The sky and structured-JSON prompts produced sensible English/correct JSON;
the code prompt was truncated in the example at the fixed256-token cap, so
this is limited qualitative evidence, not a completed instruction-following gate.

An earlier control attempt used default physical4096, unnecessarily warming128
buckets. It was interrupted during uncaptured setup with devices closing
normally, before any scored or timed requests. Retained as setup evidence.

The candidate is now wired through a validated, default-off
`QWEN38_GROUPED_RECURRENCE=1` policy. Only non-prefill length1 decode with
batch greater than cores/12 and at most32 uses the grouped helper; small
batches and prefill retain their original path. No dynamic source rewriting is
used by the runtime or the full-model benchmark. The full-model and matched
repaired-clock results below complete local validation; serving promotion
remains pending.


The initial full64 B16 candidate completed with exactly matching tokens in all
three4096/128 repeats, all three shared qualitative prompts, and the repeated
ragged requests. Median decode17.5110 vs13.8087 tokens/s/user, 1.2681x;
TTFT remains approximately12.14s. See grouped-full-b16-comparison.json.
The final native policy also passes B19 varying nonzero-state strict tracking.
Watcher10 with function inlining disabled passes B32 exact state/output checks,
with no watcher features disabled. Inline watcher fabric programs exceeded the
ACTIVE_ETH config buffer29264>26624 bytes before model creation; both failed
startup attempts and the successful noinline log are retained locally.

Commit hooks reformatted the three inherited compressed runtime Python files
and new C++ during the watcher attempt. Python function ASTs were unchanged
apart from sorting imported names in one adapter-local import; C++ whitespace-
stripped contents matched. The inherited files were restored to the intended
minimal staged edits, while the new kernels retain formatter output. Subsequent
full-stack acceptance runs use these fixed final sources. Black/isort are
excluded for committing inherited compressed Python to avoid unrelated whole-file
reformatting; new Python probes/helpers were checked with both.


## Full64 B32 validation (2026-10-27)

Clean paired runs used all64 layers, fixed4096 input/128 output, physical prefill256,
private cache8192, three repeats, and the original262144 admission capability.
Median decode rose from8.88722 to12.81941 tokens/s/user (**1.44245×**).
TTFT was24.0095 versus24.0184 seconds. Tokens were exact in all repeats;
ragged33/65/1 requests repeated exactly with four stable captures and no host
refreshes/readbacks inside steady-state decode. See full-b32-clean-comparison.json.
This fixed-work speedup is not an end-to-end SWE speedup.

A separate strict allocation-tracking pair compared384 recurrent/convolution
state receipts (48 linear layers ×2 fields ×4 chips) after one fixed-work
request and mixed requests. This strict pair uses one timed repeat per arm; the
separate clean pair above uses three. It then independently advanced alternating row groups by7 tokens from
prefix4097, followed by129 tokens from4104, crossing the120-token decode bridge
and9-token padded eager prefill. Every state receipt and final logits on all four
chips matched exactly. See full-b32-state-comparison-v2.json. The first attempt
failed a harness lifetime check because materialized logits survived into the
next replay; explicitly releasing them fixed the harness, with no exemption.

All32 rows completed three shared prompts within512 tokens: sensible English,
correct JSON counts {alice:4,bob:3}, and complete deduplication functions/examples.
The clean timing pair skips this redundant qualitative pass. No persistent dtype,
state layout, collective, or native context limit changed. These tests use a
smaller cache fixture; peak serving memory/native-context acceptance is pending.

The first grouped reduced profile failed production timestamp validation and
supplies no accepted timing. It is retained as a failed historical attempt.
AutoDebug identified a probable Blackhole clock rollover tear; the focused
natural-rollover regression reproduced the defect, the repair passed, and both
repaired-clock profiles completed as reported below. No profiler assertion is
suppressed.

### Fused-op profiler fidelity metadata

The independent final review traced a baseline report labeling discrepancy:
`MatmulReduceScatter` row9359360 says HiFi4, although its nested raw
`compute_kernel_config` requests LoFi. The retained importer overwrites one
`MATH FIDELITY` field for each compute kernel; this fused op lists the LoFi
matmul kernel first and the HiFi4 ring reduction last. The matching cached
matmul `chlkc_descriptors.h` sets `MATH_FIDELITY=0` (LoFi), and factory propagation
matches the nested attributes. This is a report-label limitation, not a selected
precision fallback. Use per-kernel descriptors/raw attributes for this op;
the aggregate report column alone does not identify its matmul fidelity.


### Signpost boundary gaps

The raw profiler row for each ASIC's first measured operation includes time since
an operation before the signpost. In grouped decode this is about4.56 seconds of
profiler-drain/setup idle time, outside the measured3.398ms firmware envelope.
The full report incorrectly charged this external edge to Embeddings and assigned
it99.9% of decode time. Even a measured-row subset retains that scalar, and host
row sorting does not reliably select the first device operation.

The collector preserves raw CSVs and produces a separate derived report input.
It blanks only the two predecessor-gap fields on the minimum firmware-start row
of each ASIC, records all four affected IDs/original gaps plus raw/derived hashes,
and preserves every row, timestamp, kernel duration, and later gap. The resulting
`*_window_perf_report` is labeled within-window gap accounting. The headline stays
the maximum complete per-ASIC envelope computed from raw endpoints; merged advice
is diagnostic rather than an exact max-chip wall time. Earlier unadjusted reports
are retained locally. No timestamp repair or profiler assertion change is involved.


## Matched repaired-clock device profile

Both reduced real-layer0/3 plus embedding/norm/head/sampler/feedback profiles
completed with all production profiler assertions enabled. Both use the same
fixed Blackhole clock reader, selected policy, real weights, batch16,
4096-input/one-decode work, physical prefill256 and private cache8192.
Source and precision receipts match. All measured operations have complete
firmware endpoints on every ASIC, and all next positions equal4097.

| Phase | Split maximum per-ASIC envelope | Grouped | Device operations across4 ASICs |
|---|---:|---:|---:|
| Decode |3725.584µs|3398.417µs|948→828|
| Prefill |408072.797µs|409591.096µs|17804→17804|

Decode saves327.167µs (1.09627× in this reduced fixture). The unchanged prefill
path varies0.37% in these single windows; no prefill gain is claimed. The reduced
fixture has one linear and one full-attention layer and pays terminal sampling
once; its ratio is not the all64-layer gain or a SWE speedup. Full-model warmed
measurements remain independently1.2681× at B16 and1.44245× at B32.
See grouped-fixed-clock-profile-comparison.json for compact provenance.

The complete captures, raw and derived window CSVs, reports, source-overlay
manifests, and failed earlier attempts remain local. Large CSVs are retained
losslessly compressed. An unused masked trace1 is prepared but not replayed by
this all-active profile; measured decode executes unmasked trace0 and sampler3.
The disabled WASM UI's default-path copy warning does not affect the actual
capture stored in each profile root. Neither warning omits measured operations.
