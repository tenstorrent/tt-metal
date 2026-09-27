# AutoDebug: full-model trace, feedback, and reset contract

Source-only investigation, 2026-09-27. No model implementation edits, target
execution, device calls, or server runs were performed by this investigation.
Scope: Stage06 Gemma4 TP4/EP4 model/generator, common sampling, allocator tracking,
and readiness `Generator` contract. Sources were changing concurrently; this
report distinguishes the initial smoke version from the later reset revision.

## Evidence and limits

- `probe_retry1.log`: reduced layers `(0, 5)`, four output tokens, split traces,
  allocator warning before `SPLIT_TRACE_READY`. Reduced output quality does not
  diagnose the full model or sampler.
- `all_layer_smoke.log`: all 30 layers, prompt length 19, eight output tokens,
  completion `Two plus two is **4**.<turn|>`, seven model and seven sampling
  replays. This is a short functional smoke, not feedback/reset proof.
- Existing `probe_full_model.py` does not inspect persistent buffer identity,
  device positions, cache writes, changed page tables, repeated requests, or
  inactive slots.
- `sampler_comparison.json` records correct changed-logit picks on all four
  chips for canonical split and force-argmax. A standalone sampler has no prior
  model trace; it cannot resolve allocations made while capturing the second
  trace. If allocation tracking was enabled, its host timing includes Python
  `gc.collect()` on every replay and cannot select the faster sampler.

## H1: warning caused by transient sampler-capture scratch

**Verdict: plausible; no unsafe survivor established by source or smoke logs.**

`tt_metal/impl/allocator/allocator.cpp:122` warns once per host thread whenever
an allocation occurs while traces exist. It neither tests buffer lifetime nor
proves address corruption. `MeshDeviceImpl::end_mesh_trace` registers the trace
on exit (`tt_metal/distributed/mesh_device.cpp:1418`). Thus the first top-k or
other scratch allocation inside the subsequent sampler capture is enough to
produce this warning.

Concrete ownership in the default greedy/no-logprobs path:

| Tensor/resource | Allocation and lifetime | Concern |
| --- | --- | --- |
| `tokens`, `positions`, `cache_positions`, `table`, caches | Allocated before model capture; remain live | Correct intended persistent inputs; identity still needs measurement |
| `initial_*` clones and `warmed` logits | Allocated before capture; Python locals survive both captures | Not post-trace allocations; their mere liveness does not explain the warning |
| `trace_logits` | Allocated during first/model capture, retained by generator and sampler | Safe from this tracker's first-trace allocation rule; regenerated before sampling |
| Sampler top-k/gather/typecast/tie-break scratch | Allocated while model trace exists; released before first replay | Most direct warning source; check actual survivors |
| Sampler output | `tt_out_tok=self.tokens`; logprobs `None` | No new retained output in default mode |
| Program-cache buffers | Potentially allocated by an unexpected signature miss | Concrete alternative to inspect if tracker finds survivors |

`SamplingGenerator.precompile` already runs before model capture, and capture
uses `skip_precompile=True`. Do not reintroduce its eager precompile behind the
live model trace. `ttnn.copy(initial, persistent)` uses its destination as the
output tensor (`operations/data_movement/copy/copy.cpp:16`); these restores are
also warmed before capture.

Tracking is conservative per trace: `record_allocation_if_unsafe` registers
post-trace buffer IDs, excludes trace-region buffers, and
`get_unsafe_tracked_ids` retires freed IDs. `ttnn.execute_trace` calls
`UnsafeAllocationTracker.verify_before_replay`, which collects garbage and
raises on surviving tracked IDs. Enable both environment variables before
importing TTNN; keep program-cache accounting enabled. Do not silence the
warning with a blanket `corruptible_allocation_scope`.

Smallest existing combined-path command, **proposed, not run here**:

```bash
OMP_NUM_THREADS=8 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 timeout 300 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_full_model --steps 4 --output models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/trace_allocation_probe.json
```

A pass supports transient-only allocation for that signature. On failure,
record each buffer ID, op context, and Python referrer. Narrowly fix the
identified lifetime or prewarm the exact missing program. A logprobs mode is a
different case: `tt_sampling.py:1126` assigns a newly captured `tt_log_probs`,
and sampler trace metadata retains it behind the model trace. That is a
concrete live-allocation candidate introduced by enabling logprobs, even if
default greedy passes.

## H2: token and position feedback are wired but unproven

**Verdict: intended structural connection present; repeated-state proof absent.**

`_forward` consumes `self.tokens` and both persistent position tensors, then
increments positions on device. `_replay` enqueues the model followed by the
sampler on CQ0, with sampler `tt_out_tok=self.tokens`. The common sampler
validates input/output object identity before replay. High-level autoregressive
generation does not reconstruct feedback on the host. Teacher forcing does
overwrite tokens intentionally, so its accuracy cannot establish feedback.

Capture records commands; it does not execute cache updates. `_capture` first
warms the real next-token position, then restores tokens and positions. The
first model replay rewrites that same cache position. This is consistent with
overwrite semantics, but a cache-row assertion is needed, particularly when
teacher forcing changes the token after warmup.

Smallest discriminating test: reduced `(0,5)` model, prompt ending at position
30, two autoregressive replays crossing positions 31/32. At test boundaries
only, read all four chips' exact `tokens`, `positions`, `cache_positions` and
buffer IDs. Assert:

1. At capture exit, tokens/positions equal the values before warmup.
2. After replay N, token tensor equals N's sampled output and both active
   positions advanced exactly once.
3. Before replay N+1, the same token buffer contains that sample; no token or
   position host copy occurred. Compare N+1 logits against a fresh-cache
   control explicitly fed token N and its position.
4. Repeat with two distinct teacher-forced tokens/positions. Assert inputs
   actually changed and logits match the corresponding controls; merely
   asserting output inequality can fail for coincident predictions.
5. Snapshot the capture-position cache row after warm and after replay with a
   different forced token; compare with a fresh explicit-input control.

Use debug readback only in this probe. Do not add it to the production loop.
Report host-copy counts as well as replay counts; absence of a counter key is
not direct observation that an uninstrumented helper performed no copy.

## H3: reset and request reuse

**Initial verdict: confirmed source contract violation. Later repair unverified.**

The initial `reset` released both traces, dropped cache references without
zeroing, and retained token/position/table/output fields. This contradicts
`models/common/readiness_check/contract.py:239`: zero KV and clear request state
while retaining buffers and compiled traces.

The later version introduces `owned_cache`, in-place cache zero, retained
traces in `reset`, and a separate `teardown`. `_standalone_cache` warms the cache
zero op before capture; this addresses first-use reset program allocation.
Remaining points to verify or resolve:

- `reset` still leaves `table_host` and the device page table unchanged. Define
  and test request-invalid state after reset; do not replay a reset request
  until its token/position/page-table state is explicitly rebound/refreshed.
- `_standalone_cache` and `_bind` still release/recreate traces and input tensors
  for every standalone request. Reset itself retains traces, but cross-request
  trace reuse is not delivered. New prefill signatures justify release;
  identical compatible requests can be a separate reuse optimization.
- External caches are intentionally untouched by new `reset`; document this
  ownership distinction because the base reset contract has no such caveat.
- Direct `prefill_forward` can allocate while a prior decode trace remains
  live. A returned prefill logits tensor retained by its caller is a concrete
  post-trace allocation. Test this low-level request-boundary path separately
  from `generate`, which releases traces before prefill.

Smallest reset test: one reduced request with at least two replays; save cache,
token, position, page-table buffer IDs and model/sampler trace IDs; retain an
alias to every owned cache; call `reset`; assert all cache values zero, IDs and
trace IDs unchanged, request-state semantics explicit, and no tracker failure.
Then A/B/A prompts of equal allocation capacity; compare A token IDs and logits
and log cache/input/trace identities across requests. Separately use larger B
to exercise capacity growth and ensure the old traces are released before
allocation. `teardown` must release both trace IDs before mesh close.

## H4: inactive fixed slots reach RoPE with an invalid index

**Verdict: concrete source defect for callers using position `-1`; device outcome
has not been tested.**

`_bind` and the explicit-refresh branch upload all `start_pos` values directly
to uint32 RoPE positions. Therefore `-1` becomes `0xffffffff`.
`fused_decoder.py:615` embeds that index before paged cache or SDPA can skip the
inactive user. The embedding reader uses it as a weight page ID
(`embeddings_common.hpp:112`). Although cache update and SDPA recognize `-1`,
they cannot make the preceding RoPE access valid. `_forward` also increments
RoPE positions without `skip_negative_entries`, while cache positions retain
their negative sentinel. `batch=len(start_pos)` is slot width, not active count.

Smallest implementation shape for a **static active mask per capture**:

- Bind `active_slots=tuple(i for i,p in enumerate(start_pos) if p>=0)` outside
  capture. Keep dense slot-order token/position/table tensors; never compact
  feedback unless an explicit device scatter restores original slot indices.
- Preserve the existing fast path when every slot is active. Otherwise unroll
  fixed slots in model decode. Active slot `i` calls the existing batch-1 path
  using token, RoPE, cache-position, and page-table slices for `i`. Inactive
  slots contribute a preallocated vocab-sharded logits row that deterministically
  selects harmless token 0. Concatenate rows in original slot order. Allocate
  that dummy logits tensor before either trace exists.
- Increment both position tensors with `skip_negative_entries=True`. The op
  interprets uint32 bits as int32, so `0xffffffff` remains frozen. Inactive
  rows must never reach the RoPE embedding in this implementation.
- Active-mask changes require request-boundary recapture/rebind because the
  static operation sequence changed. Handle the all-inactive case explicitly.
  Enforce batch `1..32`, valid active positions, and valid slot/table dimensions.

Smallest test: B=3, positions `[31,-1,127]`, active slots `[0,2]`, distinct valid
tokens and cache page rows. Prefill slots 0 and 2, snapshot slot 1 cache, replay
twice. Compare active logits/tokens to B1 controls; inactive cache remains
bit-identical; both negative sentinels stay negative; active positions become
33 and 129. Then change active slots to `[1,2]` at an explicit rebind boundary
and repeat. After this focused test, cover B=32 with holes under watcher.

## Page tables and sampling-mode follow-up

Same-shape changed page tables are copied into the captured `self.table`;
unchanged tables are skipped. Test positions 31/32 with two prepopulated,
disjoint physical-page groups: unchanged table gives zero refreshes; switching
to the second valid same-shape mapping gives exactly one refresh and matches
its independent cache oracle. Fill destinations before capture; allocating
replacement cache tensors behind a live trace would confound this test.

New `sampling_params` support needs separate evidence: generation never calls
`reset_prompt_tokens`, so repetition penalties do not include prompt tokens.
Common sampling calls `manual_seed` every draw, while this generator advances
neither `SeedManager` nor device seed state; independence of nongreedy draws is
unproven. Default greedy does not depend on random state. Alternate greedy and
nongreedy-capable requests with logged trace IDs only after defining seed and
penalty semantics; reject unsupported modes explicitly rather than claiming
them from the greedy smoke.

Small unrelated boundary: `generate(..., max_new_tokens=1, next_input=callback)`
never calls the callback. The readiness contract requires collecting that
first prediction; an isolated host-stub callback-count test can establish it.

## Status

Default-path unsafe allocation is **unresolved**, not proven corruption.
Initial reset mismatch is superseded by an **untested in-progress repair**.
Inactive-slot RoPE handling requires correction and focused verification.
Two-step feedback, changed page tables, repeated requests, and reset identities
remain required evidence; the short coherent smoke and teacher-forcing gate
cannot replace them.

## Follow-up: inactive slots preserve the accepted execution strategy

The current model's explicit per-active-slot branch does **not** substitute a
less optimized decoder. `MultichipDecoder` inherits `OptimizedDecoder`
(`tt/multichip_decoder.py:646`), whose accepted `decode_forward` already unrolls
every batch greater than one into per-slot batch-1 calls with sliced RoPE,
cache positions, and page tables (`tt/optimized_decoder.py:737`, slot loop
starting at line 750). The model branch invokes those same layer objects and
TP4 kernels while omitting inactive rows and supplying zero hidden rows in
their original slots. It preserves the accepted precision and collective
policy; no newly vectorized parent batch path is bypassed.

Zeroing only inactive RoPE indices while retaining cache position -1 would
avoid the original invalid embedding read: the selected native SDPA wrapper
is installed at `tt/multichip_decoder.py:843`, and both paged update and native
SDPA recognize cache position -1. That alone is not a complete replacement
contract. Native SDPA's writer returns for inactive users without writing an
output (`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp:134`),
so downstream projection/router work could consume unwritten output unless
separately masked or initialized. Retaining the existing per-slot strategy
avoids this unnecessary decoder-scope change. No hardware experiment was run
for this alternative, and no performance claim is made for it.
