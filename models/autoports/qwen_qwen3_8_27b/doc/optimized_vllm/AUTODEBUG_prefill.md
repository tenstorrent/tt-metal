# AutoDebug: serving prefill and sampling trace reuse

## Scope and verdict

Source-only investigation at `929c84f1afb0296d82d2e39fe1e80a669d621314`.
No implementation changes, tests, devices, server actions, or profiler
collection were performed by this investigator. The parent is collecting the
baseline independently. This report proposes hypotheses and discriminating
experiments; it does not claim a measured improvement or hardware pass.

The smallest plausible optimization is a **single scheduled prompt, physical
slot 0, start position 0, length 1..4096, device-sampling** branch owned by the
generator. Reuse `_prefill_for_generate` over the bound scheduler cache, and add a
coordinated prefill sampling trace over its persistent padded logits. The public
prefill and packed sampling paths remain the fallback for multiple prompts,
other slots, continuations, longer prompts, and explicit host compatibility.
Those fallbacks preserve arbitrary valid batches and the advertised 262144
context; they are not evidence of traced performance for those shapes.

The performance hypothesis is verified structurally: the current adapter
executes every prefill model operation and every first-token sampling operation
eagerly. Whether removing this dispatch improves the requested serving metric
still requires the parent's same-workload benchmark.

## Source evidence

* `tt/generator_vllm.py:158-200` resets only fresh recurrent slots, binds packed
  sampling rows, runs public generator prefill once per scheduled row, packs the
  outputs, samples, reads tokens, and clears `_decode_bound`. It does not use the
  owned prefill trace.
* `tt/generator.py:125-188` already has a device-only prefill with persistent
  token IDs, absolute positions and a padded `[1,1,32,vocab_shard]` output. Its
  key includes cache identity and geometry, bound page-table identity and shape,
  fixed slot/start and exact prompt length. A new key releases traces before
  persistent allocation. An existing key refreshes token contents and replays
  nonblocking if the trace exists.
* `tt/generator.py:249-317` intentionally drops owned prefill preparation before
  returning independently owned public logits. It also guards first use of
  individual prefill signatures. The public path supports 4096-token chunks,
  arbitrary fixed slots and prefix positions, and all-logit reads.
* `tt/model.py:245-283` permits preallocated positions, avoiding the otherwise
  hidden `upload` during capture. Slot/start select real operation variants and
  recurrent state slices/scatters; token shape alone is not a sufficient key.
* `tt/generator.py:351-367` guards first use of the packed sampling signature
  before concat/pad/sampling. `doc/vllm_integration/
  AUTOFIX_prefill_sampling_trace.md` records the previous native concat/tilize
  persistent-buffer failure and its fix. Keep this fallback guard.
* `tt/generator.py:412-477` warms decode and sampling, restores recurrent state,
  token/position/RoPE/seed state, then captures decode, decode sampling, and
  finally owned prefill. The prefill trace copies transient logits into a buffer
  allocated before all captures. Its recording does not execute prefill.
* `tt/generator.py:560-570` submits the separate decode/model and sampling traces
  with `blocking=False`. `generator_vllm.py:244-268` retains the one-replica
  128-byte token read and event-based async host-processing split.
* `tt_metal/impl/allocator/trace_allocation_tracker.cpp:117-160` associates later
  allocations with active traces and rejects those still alive at replay. A new
  persistent output or lazy program buffer after capture is a real lifetime
  violation. Synchronization or deleting an unrelated local cannot repair it.

## Recommended implementation boundary

1. **Generator API and selection.** Add
   `serving_prefill_tokens(tokens, *, page_table, kv_cache, prompt_lens,
   start_pos, slots)`. The adapter's device-sampling branch calls this once;
   its explicit-host branch keeps the prior public prefill loop. Define this
   wrapper's `prompt_lens` as the scheduler's absolute exclusive prompt ends,
   matching the adapter input, and document that its fallback converts each to
   the public generator's logical length `end - start`. This preserves current
   adapter slicing exactly. The generator chooses the optimization only when
   there is exactly one row, `slots == [0]`, `starts == [0]`, and
   `1 <= ends[0] <= min(4096, kv_cache.capacity)`. Slice exactly
   `tokens[0:1, :ends[0]]`. Validate ordinary row/length geometry so a malformed
   request cannot accidentally select this narrower helper. The selection
   describes scheduled rows; requiring `kv_cache.batch_size == 1` additionally
   is a reasonable first experiment if the measured server is configured B1.
   The fallback loops over rows using exactly the prior public
   `prefill_forward(tokens[row:row+1, start:end], prompt_lens=[end-start],
   start_pos=[start], slots=[slot], ...)`, then calls `sample_prefill(outputs)`.
   Keep all trace eligibility and ownership decisions inside this generator
   method. A test can replace this one method with the old eager loop while
   retaining the identical adapter's reset, RNG and readback behavior.

2. **Preserve request setup.** Keep `reset_recurrent_slots(fresh)` and
   `_sampling(..., reset=True, output_positions=ends)` before the real prefill.
   Call `generator._refresh_table(table)` before `_prefill_for_generate`, since
   that helper currently assumes its owned page table is already current.
   Pass and retain the exact bound scheduler cache; do not call `_ensure_cache`
   or `reset()`. Clearing the whole cache would destroy scheduler-owned pages.

3. **One owned logits buffer.** Continue using `_prefill_for_generate` and
   `prefill_prepared['output']`. Do not return this tensor from public
   `prefill_forward`, retain a list of aliases to it, clone it after capture, or
   feed it through existing `sample_prefill`. It is already padded to 32 rows;
   `sample_prefill([output])` would add another 31 logical rows. The adapter
   consumes this buffer immediately to produce the existing token result.

4. **Separate prefill sampling trace.** Add `prefill_sample_trace`, initialized
   to `None`, to the same release/reset lifecycle as all existing handles. Add a
   small private helper that samples exactly
   `self.prefill_prepared['output']` through `_sampling_step` on first use or
   `execute_trace(mesh, prefill_sample_trace, cq_id=0, blocking=False)` when
   available, returning `self.tokens`. Reuse the canonical sampler and
   `tt_out_tok` feedback target. Record an eager-prefill-sampling counter and a
   prefill-sampling-replay counter separately.

5. **Capture all dependent traces together.** Extend `_capture` to record the
   fourth trace after the current prefill trace, invoking only
   `_sampling_step(self.prefill_prepared['output'])`. Capture order becomes
   decode, decode sampling, prefill-to-persistent-output, prefill sampling. The
   real cold prefill and first-token sample already warm this exact path and
   advance state once; do not execute a second warm prefill or sample without
   restoring request state. Preserve the existing exception cleanup loop for
   every partially recorded handle. No fourth-trace output may be saved: the
   sampled token is written into the token tensor allocated at construction.

6. **Preserve standalone's three traces.** Add a keyword such as
   `trace_sampling=False` to `_prefill_for_generate`; include its boolean value
   in the preparation key and store it in `prefill_prepared`. The new serving
   wrapper passes `True`. `_capture` records the fourth trace only for this
   preparation mode. A mode change then rebuilds the coordinated bundle before
   any new allocation. A flag set after a three-trace bundle already exists is
   insufficient: there would be no prefill sampling trace. Keep standalone's
   existing sampling path and three-trace tests intact. Do not maintain
   separate unbounded trace caches by prompt length or change sampling strategy.

7. **Keep the output and decode boundary.** Finish both adapter branches through
   the existing token read/host conversion, tensor RoPE deltas, and
   `_decode_bound = False`. Prefill seeds start at request seed plus output
   position `end`; sampling increments them once. First decode rebinds its
   authoritative seed to `position + 1`, so it must not reuse stale request state
   just because a trace bundle survives. Leave `_last_device_sampling`, async
   overlap policy, decode refresh rules and nonblocking submission unchanged.

Do not invoke the existing decode `sample_trace` directly after prefill: it
captures the decode logits address (`self.logits`), not the prefill output.
Copying into `self.logits` is a different design requiring its own warmed copy
and ownership proof, with no advantage over the explicit fourth trace here.

## Lifecycle and allocation checks

* A new prompt length or binding releases all four traces before uploading
  persistent prefill inputs/output. One preparation is retained, bounding trace
  storage; a length change can recapture, so report performance per exact warmed
  shape rather than implying every new length is fast.
* `_release_traces(keep_prefill=True)` preserves prepared buffers only, never
  live trace handles. A later first sample with no handle executes eagerly;
  subsequent decode recaptures the complete bundle. Never leave a stale fourth
  handle after reset-slot warmup, active-slot change, remap, or history-mode
  change invalidates the existing bundle.
* Existing recurrent reset is outside capture and only clears selected fresh
  slots. It retains traces after the same cache's reset programs are warm.
  Prefill capture itself must neither reset state nor advance positions.
* Packed/long/public fallback drops owned preparation before retaining public
  logits; returning later to the optimized branch releases old decode traces
  before allocating fresh private buffers. Preserve the current packing
  signature guard across this transition.
* Sampling parameters, page table and prompt IDs must update stable device
  inputs before replay. Default `sampling_strategy='split'` has
  `allow_force_argmax=False`, so greedy/sample parameter changes keep the graph
  fixed. If argmax mode is extended into this serving branch later, transitions
  must invalidate every dependent sampling trace.
* G=1 on a cold preparation remains eager because there is no decode capture.
  A warmed bundle can serve G=1 with both prefill traces. Do not introduce a
  speculative decode solely to populate a cache for the first-token-only case.
* The additional sampler trace must fit the production 134217728-byte trace
  reservation. The standalone test's 200000000-byte reservation alone cannot
  establish that. Keep allocator and program-cache checks enabled for diagnostic
  device tests, with program-cache exclusion disabled.

## Discriminating verification

The following are proposed tests, not runs performed by this investigator.

**Host branch and lifecycle regressions.** Extend the actual adapter host tests
with a single-row fresh slot-0 case that requires the private path and checks
page refresh before prefill, reset/sampling ordering, exact slicing, packed
token shape, zero tensor RoPE deltas, and `_decode_bound=False`. Existing
`test_vllm_prefill_host.py` must still cover two different rows `[3,1]` with a
continuation, proving fallback outputs remain distinct and ordered. Add long
4097-token, nonzero-start, nonzero-slot and explicit-host cases that forbid the
fast helper. Extract or load the real trace helper methods to assert release of
all four handles, partial-capture cleanup, persistent buffer identity on reuse,
and that the fourth trace reads the prefill output rather than decode logits.
The existing five packed-sampling guard tests remain relevant.

**Reduced device A/B with layers `[0,3]`.** Bind a real externally allocated
cache using the model's actual page size, TP4 mapping, selected dtypes and cache
geometry. Compare the new adapter branch against the previous eager boundary:
`generator.prefill_forward` plus `generator.sample_prefill`, patched into
`serving_prefill_tokens` for the control, using identical
fresh recurrent state, page allocation, sampling parameters and tokens. This
control uses the canonical sampler, not a host argmax substitution. Use lengths
31/32/33, then 4095/4096 and fallback4097. Generate enough decode steps to cross
a 32-token page boundary (e.g. S31/G35). Compare exact token sequences, first
prefill logits, positions, seeds, and recurrent/KV state. Use real text prompts
for separate qualitative evidence; deterministic token fixtures only test the
mechanism.

After one cold request with decode, run a second same-length request while
forbidding `_capture`, eager `_prefill_trace_logits`, and eager `_sampling_step`.
Require two ordered nonblocking prefill replays followed by the normal split
decode replays, zero capture/eager-sampling counters, no full-logit readback on
the measured token-out path, and unchanged trace IDs and persistent addresses.
This guard distinguishes true traced first-token sampling from a traced model
followed by eager sampling. Cold versus warm throughput alone does not.

**Changed-input and state probes.** Alternate equal-length prompts A/B/A and
greedy/sampled/greedy with fixed explicit seeds. Require changed logits for B,
exact recovery for A, no stale first token, and no trace replacement for a fixed
split mode. Change physical page mappings in place between requests and inspect
KV pages: tokens alone cannot detect writing an equivalent stale mapping.
Verify inactive recurrent sentinels if testing a one-row schedule in a larger
bound batch. Alternate fast path, multi-row fallback, continuation/long
fallback, and fast path again. Rebind a new cache and prove all captured
addresses change appropriately. G1 cold/warm must deliver one sample and one
seed increment; capture must not create an extra prefill state update.

**Allocation and async evidence.** Run reduced controls with
`TT_METAL_TRACE_ALLOC_TRACKING=1` and
`TT_METAL_TRACE_ALLOC_TRACEBACKS=1`, leaving
`TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE` unset. Forbid program-cache misses
only while a trace is open, as in `check_prefill_tracing.capture_guards`.
Retain the existing public-output ownership probe and the existing adapter
async test that enqueues read N before replay N+1 and rechecks old host
snapshots. Existing `check_vllm_adapter.py` supplies useful page-growth,
inactive-state and remap controls, but its current `fresh_path` explicitly
releases all traces and therefore does not itself prove cross-request prefill
reuse. Add a repeated-request probe that deliberately retains the bundle.

**Production evidence.** After reduced correctness, run the same full64-layer
server baseline/candidate workload with production trace reservation, native
on-device sampling, identical seed/prompt/shape/warmup policy, and no profiler.
Record TTFT, ITL, throughput and trace/eager counters. Run repeated-request,
sampling and qualitative checks, including return from explicit compatibility
to native sampling. Report cold and warmed behavior separately. A successful
B1 optimization does not establish traced prefill for arbitrary multi-row
schedules; their continued correctness is established by fallback controls.

## Final status

Source inspection supports the narrow four-trace design and identifies the
specific lifetime/ownership conditions. Implementation, numerical equivalence,
production trace-space capacity and performance remain unverified here.
