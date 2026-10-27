# Grouped FP32 recurrent update candidate

Status: experimental, disabled in the model and CI; real-layer validation in progress.

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
used by the runtime or the full-model benchmark. Candidate full-stack evidence
is pending; this is not yet a promoted serving configuration.


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
