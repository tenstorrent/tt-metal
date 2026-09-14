# AutoFix: bounded serving prefill and first-token sampling traces

## Starting evidence

`AUTODEBUG_prefill.md` inspected the source at
`929c84f1afb0296d82d2e39fe1e80a669d621314`. The existing vLLM adapter dispatched
prefill and first-token sampling eagerly. The standalone generator already
owned a safe persistent prefill output and captured prefill after the decode
pair. The earlier stage9 packed-sampling allocation repair establishes why
independently retained public outputs must not be substituted into that path.

The parent verified the source hypothesis and authorized this isolated
implementation. This investigator ran CPU/host tests only, no device/server
execution and no profiler. The parent owns device and serving measurements.

## Implementation and hypothesis checks

**Hypothesis:** a generator-owned bounded token-output path can reuse the
existing prefill trace and remove warmed first-token sampling dispatch without
changing scheduler cache ownership, public output ownership, or decode state.

Implemented `QwenGenerator.serving_prefill_tokens(tokens, *, page_table,
kv_cache, prompt_lens, start_pos, slots)`. Inputs retain the adapter convention:
full token rows, absolute exclusive ends, absolute starts, and physical slots.
The optimized branch requires one scheduled row in slot0, start0 and length
1..4096. It refreshes the bound page table, then calls
`_prefill_for_generate(..., trace_sampling=True)` and samples its persistent
padded output. Other valid shapes use the prior per-row `start:end` public
prefill loop and canonical packed sampler.

Preparation mode is part of the key. Serving records a fourth trace after the
existing decode/model, decode/sampling and prefill traces. Its sampling input is
the preallocated owned prefill output; its token target is the existing
feedback tensor. All four handles share release and partial-capture cleanup.
Standalone's default preparation still records exactly three traces. Warm
prefill and sampling submit separately with `blocking=False`.

The adapter calls the generator helper once for device sampling and preserves
host compatibility's original public path. Request reset, seed binding, token
readback, tensor RoPE deltas, and `_decode_bound=False` remain at the adapter
boundary. Close emits one compact `QWEN_VLLM_COUNTERS` JSON line per close call.
Counters distinguish prefill sampling replays, captures and successful eager
samples, including fallback samples; cold warmup is therefore visible.

**CPU experiment:** the new AST-loaded test initially failed all11 test setups
because the new API was absent. After implementation and adapter coverage it
passes13 tests. These exercise exact slicing and packed order, bounds and wrong
cache rejection, boundary eligibility, warm replay target and nonblocking mode,
preparation-mode changes before allocation, four-handle release, fourth-capture
failure cleanup, and standalone's unchanged three-trace contract.

**Verdict:** host contract verified. The parent reports the initial reduced
device experiment passed, and `prefill_reduced_short.json` records status
`pass` for lengths31/33, batch1, context262144, reduced layers0/3, with exact
logits/tokens and changed-prompt logits. Warm guarded requests each have one
prefill replay, one prefill-sampling replay, three model and three sampling
decode replays, with no eager prefill/sampling, recapture or page-table copy.
The parent's test used allocation tracking, production128MiB trace reservation,
changed prompt/page controls, and stable trace identities. This short result
supports the specific dispatch/lifetime hypothesis; it does not establish
full-model performance or all remaining boundary cases.

## Commands run by this investigator

All commands ran from the tt-metal checkout.

```bash
python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/test_serving_prefill_trace_host.py
# 13 passed

python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/test_prefill_sampling_trace_host.py
# 5 passed

PYTHONPATH=/home/mvasiljevic/qwen38-full-rerun/vllm:/home/mvasiljevic/qwen38-full-rerun/tt-metal \
  python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/test_vllm_prefill_host.py
# 5 passed; actual vLLM prefill consumer with fake device outputs

python_env/bin/python -m py_compile \
  models/autoports/qwen_qwen3_8_27b/tt/generator.py \
  models/autoports/qwen_qwen3_8_27b/tt/generator_vllm.py \
  models/autoports/qwen_qwen3_8_27b/tests/test_serving_prefill_trace_host.py \
  models/autoports/qwen_qwen3_8_27b/tests/test_vllm_prefill_host.py
# passed on final source

python_env/bin/python -m black --check --target-version py312 \
  models/autoports/qwen_qwen3_8_27b/tt/generator.py \
  models/autoports/qwen_qwen3_8_27b/tt/generator_vllm.py \
  models/autoports/qwen_qwen3_8_27b/tests/test_serving_prefill_trace_host.py \
  models/autoports/qwen_qwen3_8_27b/tests/test_vllm_prefill_host.py
# 4 files unchanged
```

The first actual-plugin test invocation lacked the sibling vLLM checkout on
`PYTHONPATH` and failed import before running tests. The corrected command above
passed. No build is required for these Python/documentation changes. No commit
was created by this investigator.

## Remaining gates and precise limitations

* The parent is running expanded reduced lengths32/127/128/129/4095/4096/4097
  and a larger bound batch control. Their results are pending at report time;
  inspect `prefill_reduced_boundaries.json` and the parent's next evidence.
* Full64-layer correctness, native sampled/greedy transitions, qualitative
  serving checks, lifecycle/async controls, and production trace-region capacity
  still need the parent's validation. Short reduced trace capacity is not proof
  of full64-layer capacity.
* Same-workload before/after serving benchmarks must establish TTFT/ITL and
  throughput with identical precision, context, topology, sampling and warmup.
  No performance improvement is claimed by this report.
* Multiple scheduled prompts, nonzero slots, continuation starts and lengths
  beyond4096 retain eager prefill and eager first-token sampling. Cold eligible
  requests also warm eagerly. Only a warmed eligible shape has both prefill
  operations traced. This candidate alone does not satisfy a requirement that
  every fallback shape avoid eager first-token sampling.
* One preparation is retained. A length/mode/cache-binding change releases the
  bundle and must warm/capture again. Ordinary same-shape request reset retains
  stable buffers; changing sampling strategy is outside this implementation.

Status: narrow candidate implemented, host checks pass, initial parent reduced
device check passes; expanded correctness and full-model serving/performance
gates remain open.

## Followup: warmed sampling for every serving fallback shape

The parent subsequently authorized removing warmed eager first-token sampling
from the fallback as well. This section supersedes the earlier statement that
fallback sampling always remains eager. The primary short-prompt fast branch
and public `sample_prefill` are unchanged.

`_prepare_serving_prefill_sampling` now owns the fallback public-output list,
concat and padding temporaries. It preserves the previous cache-scoped packing
signature guard and releases traces before a new packing signature or the
first persistent staging allocation. It clones the padded tensor once into
`gen.prefill_sample_input`, warms its copy, and thereafter copies packed results
into that stable destination. This uses approximately4MiB per device for the
selected BF16 `[1,1,32,62080]` local-vocabulary tensor. Cache replacement through
either `bind_cache` or `_ensure_cache` discards it after trace release; ordinary
request and trace resets retain it.

The helper returns only a host signature. Its public logits and packed tensor
references therefore die before `serving_prefill_tokens` replays sampling.
This boundary is necessary: replaying inside public `sample_prefill(outputs)`
would leave the caller's newly allocated logits alive against an older trace.
The public sampler remains unchanged, including its independent ownership and
existing packing guard tests.

When no owned prefill preparation exists, `_capture` records the fallback
sampling trace over `prefill_sample_input` after the decode pair. Explicit local
handle assignment distinguishes this third sampling trace from standalone's
third model-prefill trace. Existing fourth-trace serving capture, partial
capture cleanup and the nonblocking sampling replay mechanism are retained.
The real first sample warms eagerly when no sampling trace is available;
subsequent same-shape fallback requests replay sampling. The packing signature
is marked warm only after successful sampling.

Followup CPU verification, without TTNN import or devices:

```bash
python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/test_serving_prefill_trace_host.py
# 20 passed
python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/test_prefill_sampling_trace_host.py
# 5 passed
python_env/bin/python -m py_compile \
  models/autoports/qwen_qwen3_8_27b/tt/generator.py \
  models/autoports/qwen_qwen3_8_27b/tests/test_serving_prefill_trace_host.py
# passed
python_env/bin/python -m black --target-version py312 \
  models/autoports/qwen_qwen3_8_27b/tt/generator.py \
  models/autoports/qwen_qwen3_8_27b/tests/test_serving_prefill_trace_host.py
# generator unchanged; test formatting applied successfully
```

New tests use weak references to prove that public/packed tensors are dead at
the instant of sampling replay, verify stable staging identity and first
allocation ordering, distinguish the fallback third trace, exercise failed
capture/sampling and new-count invalidation, and verify both cache-replacement
paths clear the staging buffer. New counters report input allocations and
copies independently of sampling replay/eager counters.

The parent's `fallback_reduced.json` records a passing allocator-enabled probe
using production128MiB trace storage and a four-slot cache. Lengths31/33/4097
match eager logits/tokens exactly, including changed prompts and pages. Warmed
requests of every tested length forbid eager sampling and recapture; only
lengths up to4096 additionally forbid eager model prefill. The nonzero-slot1/3
multirow S31/45 control also matches exactly, with one warmed sampling replay
and no unchanged-page-table copy. This verifies the fallback sampling/lifetime
hypothesis on the reduced layers0/3 path. The parent is running the same B4 and
multirow cases under Watcher10, NOINLINE1 and FABRIC_O3; that check and final
full-model serving evidence are pending at this report update.

Generic prefill computation and packing remain eager. New individual
prefill signatures, new packing signatures, cache/active-slot changes or other
events that release the bundle may still require a cold sample before decode
recaptures it. Consequently, zero eager samples in a measured burst must be
demonstrated by workload warmup and counters, not inferred from this code.

## Final parent validation (supersedes pending experiment statuses above)

Full64 prefill boundary, B4 mixed-slot fallback, Watcher and68-step async checks
pass. Final native primary TTFT improves81.373->62.955ms at S128/G128/N1 with
one warmup; decode remains41.313tokens/s/user. CI S100/G100/N32 completes32/32.
Native shared/extended quality and full64 B32 diverse/seed/lifecycle controls
pass; the separate explicit compatibility suite passes73/73. Final server
cleanup/process audit and four-chip health listing pass. See work_log.md and
README.md for exact artifacts, cold-cost tradeoffs and final independent review.
