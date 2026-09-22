# AutoDebug: split-trace lifetime and reset contract

Date: 2026-09-12. Source-only investigation; no model/device execution or implementation edits by this agent. Inspected the newly added model/generator, `probe2.log/json`, the allocator tracker, and canonical `models/tt_transformers/tt/generator.py`. The coordinating agent changed the implementation during this review; already-observed repairs are distinguished below from the original probe's behavior.

## What probe2 establishes

`probe2.json` is a reduced two-layer `[0,3]` test, not a full-model correctness result. It runs the same 33-token prompt twice and obtains `[220,220,220,220]` both times. The second run reuses two traces: cumulative counters show six model and six sampling replays, two captures total. This establishes successful replay and same-token repeatability for that one input. A permanently stale logit tensor could also produce this result; token equality does not establish feedback, position refresh, or cache correctness.

At `probe2.log:65`, allocator.cpp reports an allocation made with an active trace. This warning alone does **not** identify an overwritten buffer:

- `allocator.cpp:118-132` emits it at most once per host thread for the process lifetime, regardless of which trace or allocation caused it.
- A trace becomes active at the end of capture (`mesh_device.cpp:1418-1424`). The first allocation inside the subsequent sampling capture therefore qualifies for the warning even when it is ordinary temporary scratch.
- With tracking enabled, `trace_allocation_tracker.cpp:117-135` records allocations made after any active trace; it does not perform an address-overlap proof. Before replay, only still-allocated IDs are returned (lines 138-174). A tracker failure is a concrete lifetime violation to investigate; its error wording is stronger than its conservative accounting algorithm.

## Findings

### 1. Eager prefill outputs survived the second request's replay

**Verified lifetime defect in probe2-era source; repair observed during this review.**

The original `generate()` retained both `logits` from eager prefill and `padded` until the method returned. On the first request these allocations preceded trace capture. On the second request, `_ensure_cache` reused the same cache and traces, so both tensors were allocated after capture and stayed live through the decode loop. Neither tensor is read again once the first token has been sampled. This is exactly a live post-capture allocation that the tracker reports; token repeatability cannot make its lifetime safe.

The coordinator added `del padded, logits` immediately after the first token readback and before decode. That is the smallest appropriate repair. Check the tracker after that point; if another Python alias or persistent operation resource keeps the allocation live, identify that owner rather than adding a broad `mark_corruptible` scope.

`prefill_forward(..., return_all_logits=False)` returns device tensors by design. Other callers that retain those eager results across decode replay need the same ownership decision: consume/release them before replay or copy the needed result to a preallocated persistent buffer. Host-returned logits do not have this device lifetime.

Changing prompt length while reusing live traces was unresolved in the first review. The follow-up `shapes.log` now confirms new persistent program-cache allocations at S31 after S1 (see finding 7). Existing temporary `zeros_like`/prefill activations are not automatically a correctness defect if freed before replay; the new evidence specifically identifies retained resources from program-cache misses.

### 2. The split logits and token identities are wired correctly

**Concern refuted in the inspected source.**

`_capture()` assigns `self.logits = self._model_step()` inside model capture, keeps that tensor alive, and passes that exact object into `_sampling_step(self.logits)` during sampler capture. `_sampling_step` passes the already-allocated `self.tokens` as `tt_out_tok`. The next model replay reads `self.tokens`. `decode_forward()` submits model trace, then sample trace on CQ0, then reads tokens. No replacement logits tensor or intervening eager model invocation occurs in this sequence.

This matches the canonical generator's prepare/record split (`models/tt_transformers/tt/generator.py:2028-2146`): warm model and sampler before either trace, then capture model output and bind sampler to it. The source does not justify adding another logits copy merely because there are two traces.

The exact buffer addresses and feedback contents still need the focused probe below. Python identity is a source-level fact, not proof of device execution.

### 3. Capture backups restore recurrence, but warmup writes K/V

**Verified state mutation with a conditional consequence; not demonstrated corruption of the current generate loop.**

`_capture` backs up convolution, recurrent, token, position and seed state; the updated source also backs up RoPE indices. All backups and the eager warm logits are allocated before the first trace and remain alive through recording. They are not the reported class of post-capture allocations. Keeping them alive can consume extra memory, but no snapshot-lifetime correctness bug is established by the allocator warning.

The eager `_model_step()` writes full-attention K/V via `paged_fused_update_cache` (`optimized_decoder.py:732-734`). K/V is not restored. Consequently `_capture()` is not a transactional “leave the entire cache unchanged” operation: the current position has already been written when it returns.

In the ordinary append-only generation path, the first real model replay uses the restored token/position/recurrence and overwrites that same K/V position before attention reads it. Warmup does not double-advance the restored recurrence or positions, and this K/V write alone is not proof of incorrect output. It becomes hazardous if capture is attempted against a shared prefix, if the caller changes the intended first replay position/token between preparation and replay, or if capture fails and the mutated cache is reused.

Discriminating check: compare CPU snapshots of the current-position K/V page, convolution and recurrent state immediately before and after `_capture`, then compare the first actual replay against one eager decode from an independently initialized equivalent cache. Expect K/V mutation after warmup, restored recurrence before replay, and matched real decode state afterward. Do not use token equality alone. If a fully transactional capture contract is required, restore only the touched K/V rows/pages or warm on a dedicated cache prepared before any trace; cloning all long-context K/V would be an unnecessary expansion.

### 4. Host/device page-table ownership could silently skip a refresh

**Verified in original source; repair observed during this review.**

Original `_refresh_table` accepted the bound device table unchanged but retained `page_host`. Starting from host table T0, a caller could update the bound device table to T1 in place, use it, then request host T0. The equality optimization compared T0 with the stale host shadow and skipped the required copy, leaving T1 bound to the trace.

The coordinator now invalidates `page_host` when a device table is supplied. This is an appropriate narrow repair. Test T0 → in-place device T1 → host T0 and inspect the bound device table; then repeat host T0 and assert that the second equal host submission does not copy again. An unchanged bound device table still needs no per-token upload.

The subsequent bounded audit verified host physical-page range checks and binding geometry checks in the updated source; see finding 8 and `cache_contract_host.json`. The bound table must cover at least `ceil(cache.capacity / 32)` logical pages per slot. An arbitrary changed table does not migrate existing cache contents; tests that expect equivalent logits must move/prefill the corresponding physical pages, or deliberately change only the next unused page and inspect where the next K/V update lands. Device-table contents remain caller-owned; validating their values by host readback is not part of the steady decode path.

### 5. Inactive positions originally reached RoPE before the cache skip

**Verified in original source; repairs observed during this review.**

Probe2-era `model.decode` converted signed positions directly to UINT32 and used them as RoPE embedding indices. An inactive `-1` becomes an invalid large index before full-attention's cache-update skip can apply. The linear-attention `_delta` branch ignored `current_pos` entirely and always wrote convolution/recurrent state. `plus_one(..., skip_negative_entries=True)` at the end of the model does not protect either earlier operation.

The coordinator added a separate nonnegative `rope_indices` tensor, inactive recurrence preservation, and a requirement for authoritative positions when changing `active_slots`. These address the identified mechanisms. Current source also resets RoPE and seeded-sampling tensors at request reset. The old defects should not be reported as still present without testing this new state.

Remaining distinctions to validate:

- An inactive row's signed position must stay -1 and its K/V, convolution and recurrent bytes must remain unchanged for several replays. The updated wrapper clamps the initial inactive RoPE index to zero and limits replay count by the active rows' remaining capacity, which is at most the model context. Thus the unconditional inactive RoPE increment cannot reach an out-of-table read in this bounded request. It need not remain stationary to be safe. Native paged-cache update and SDPA decode explicitly skip a signed -1 position; the fixed-slot tests currently snapshot inactive convolution/recurrent state, not inactive K/V bytes.
- Changing an active mask without authoritative positions is now rejected. Negative values supplied for a supposedly active row should also be rejected or explicitly treated as inactivity; preserving recurrence currently follows the Python `active_slots` configuration, not arbitrary negative device positions.
- The updated `reset()` sets `reset_active_slots=True`; the next decode without an explicit mask releases a previously masked trace and restores the all-active configuration. `generate()` can therefore start an all-active request after a masked low-level caller. The request also resets signed/RoPE positions, seeds and replay capacity.
- Invalid token IDs, out-of-range positions, and malformed host vector lengths should fail on the host before embedding/cache operations. No on-device malformed-index experiment is needed to prove a missing host check. Reserve hardware checks for valid inputs and implemented inactive-row semantics.

### 6. Capture failures can leave a partially published trace

**Verified in original source; repair observed in the subsequent audit. No failure was injected.**

Original `_capture` published `self.trace` immediately after `begin_trace_capture`, and had no exception cleanup around either region. If `_model_step` raised, the command queue could remain in capture mode. If sampler capture failed, a model trace was already published while `sample_trace` was absent or incomplete. `_release_traces` assumes every stored ID can be released normally; allocator active registration occurs at end-capture, so an open region is a distinct state.

The coordinator now uses local capture handles, publishes the pair only after both regions finish, and closes/releases partial regions on failure. This addresses half-published trace state in the normal cleanup-success case. Cleanup failures themselves were not injected; no stronger transactional guarantee is claimed.

### 7. A new prefill signature must compile before the next trace pair

**Hypothesis verified by the coordinator's hardware experiment; repair accepted by source review and its seven-length targeted rerun. Known-length repeat coverage was still pending when this section was updated.**

`full_model_shapes.py` retains one maximum-context cache across logical prompt lengths. `shapes.log:37` reports S1 passed; S31 starts at line 38 and fails immediately before model replay. The tracker reports 63 live buffers. A host-only parse confirmed all 63 carry `program_cache:` contexts. Representative evidence includes embedding, a `[31,1]` reshape, convolution-history slice `[28:31]`, full-attention SDPA, untilize to 31 rows, and the final-token slice `[30:31]`. Fourteen of the entries are `SliceDeviceOperation`. The allocations span the prefill pipeline; deleting the returned logits cannot release the owning cached programs.

`ttnn/api/ttnn/device_operation.hpp:367-385,428-438` applies that allocation context on the program-cache-miss path. The failure therefore verifies new program creation under existing traces, not just an innocuous eager output retained by Python. The tracker stopped the run before the unsafe replay; no resulting numeric corruption was observed or needs to be induced.

The minimal lifecycle repair is:

1. Determine every request's exact prefill signature **before the first device allocation or prefill operation for that call**.
2. If any signature is unknown, release both model and sampling traces. Run the complete requested prefill normally with no old trace pair live.
3. Record signatures only after every requested row/chunk and its LM-head work succeeds.
4. On the next decode, warm/capture one new model+sampling trace pair. Known prefill signatures retain it; ordinary decode does not release or rebuild it.

For the current fixed model, canonical cache and page-table formats, this request-level key covers the Python-selected operation variants:

```python
(cache.batch_size, tuple(page_table.shape), slot,
 absolute_start_pos, logical_prompt_length, return_all_logits)
```

| Key field | Source dependency it covers |
| --- | --- |
| Exact logical length | Embedding/rotary/reshape shapes, 4096-token outer chunks, padded tails, convolution-history slices, final-token slice |
| Exact absolute start | Page-table slice offsets, continuation inside a page, SDPA `chunk_start_idx`, prefix coverage and chunk configuration; start modulo 32 alone is insufficient |
| `return_all_logits` | `model.prefill` full-sequence LM head versus last-token extraction and decode-shaped projection; the LM head is already inside the guarded call |
| Slot plus cache batch | Page-table row selection; convolution/recurrent per-slot slices, before/after concat pieces and writeback into the full batch |
| Full page-table shape | Available prefix window, cache-read geometry and relevant page-table slice shapes |

No extra per-chunk signature is necessary with this conservative whole-request key. For S4097 at start 0 it deterministically covers `(start=0,length=4096)` and `(start=4096,length=1)`. The decoder's further inside-page splitting is also determined by exact start and length. Recording individual chunks could reduce invalidations later, but is unnecessary for the minimal fix. Multiple already-known slot requests in another order add no new device postprocessing shape: outer result concatenation/padding in the all-logits branch is host Torch work.

The normal first-token sampler is outside `prefill_forward`, but its prepared input remains `[1,1,32,62080]` regardless of prompt length. An existing decode trace implies that same model/sampler shape was already warmed. Runtime k/p/temperature/seed updates use stable buffers; a switch to/from force-argmax separately invalidates the trace pair. No missing length-dependent sampler key was found.

Two conditions keep the key sound:

- Clear the signature ledger when replacing or rebinding the cache; the coordinator added this to `_ensure_cache` and `bind_cache` during review. An alternative is including complete cache tensor specifications. A signature compiled for a prior differently shaped physical cache is not evidence for the new binding.
- Require the prefill cache/table to obey that bound canonical specification, or include their actual dtype/layout/memory/padded shapes in the key. `prefill_forward` originally admitted other cache objects and arbitrary device page-table specifications while keying only on B and logical table shape. Enforcing bound cache identity and canonical INT32/ROW_MAJOR/DRAM page tables is smaller than generalizing the key. Keep model precision/policy and the program-cache lifetime fixed; clearing the device program cache must also clear this ledger.

The reviewed implementation already places the unseen-signature guard before host-table upload and token upload, and updates the ledger after the complete model-prefill loop. LM-head and slot-scatter siblings are inside that loop, so no additional guard is needed around them. The repair assumes this generator owns the trace pair for its mesh and that earlier asynchronous use has completed at the request boundary; ordinary `generate` reads tokens before starting the next request. If callers use `read_from_device=False`, drain that work before releasing/reusing its trace storage.

Targeted validation: run S1,31,32,33,4095,4096,4097 under allocation tracking, then repeat S33,31,4097 without rebinding the cache. Starting with an empty ledger and creating both traces for each novel request, cumulative `trace_captures` should reach 14 after the seven unique lengths and remain 14 for the repeats. Require tracker cleanliness, expected position progression, and no program-cache growth on the repeated known signatures. Separately check `return_all_logits=True`, an inside-page continuation, and another slot within B=8 so the sibling-key claims receive hardware coverage. These follow-up jobs were not run by this source-only agent.

`shapes_fixed.json` records all seven unique boundary lengths passing with two decode steps apiece. The coordinator reports that this run used allocation tracking. `full_context.json` additionally records the full 64-layer model at S262143 plus one decode and S262144 with prefill-only token generation. These artifacts establish their recorded execution boundaries; they do not by themselves prove known-signature reuse or general output accuracy.

### 8. Subsequent binding and sampling-parameter audit

**Source findings repaired by the coordinator; focused host-only tests passed.**

The follow-up inspected current `model.py`, `generator.py`, contract tests and saved evidence. It found three concrete wrapper defects beyond the original trace issue:

- Rebinding/replacing a cache allocated new zero signed/RoPE positions but retained `remaining_steps` from the old cache. Decode without authoritative new positions could consequently proceed against an unrelated cache. Both replacement paths now clear the budget to `None`.
- External binding admitted an undersized page table: for example capacity 128 with shape `[B,1]`. The wrapper checked positions against 128 although native paged update indexes the table by `position // 32`. A valid-looking position 33 would address a nonexistent virtual page. Binding now validates cache batch/context limits and sufficient per-slot table width before changing usable state.
- Prefill for an already bound cache used its supplied device table directly, whereas decode required the bound identity. Apart from inconsistent page ownership, a differently specified device table could invalidate the prefill-signature reasoning. Prefill now refreshes the existing bound table after the unseen-signature guard and passes that exact tensor to the model. Host updates use the same shadow/copy logic as decode; another device identity is rejected.

An intermediate ordering of `bind_cache` checked device dtype and host physical-page IDs only after publishing the new cache/positions. The coordinator moved every currently implemented validation ahead of release/publication. Host-only tests confirm that too-narrow tables, wrong batch geometry, negative/out-of-range host IDs and wrong device dtype all raise while preserving the old cache/table/positions/trace handles/budget/signatures.

`cache_contract_host.json` now records 41 passing cases against generator SHA256 `7509ddc079fbadac267bd067660a6a22eb2cea90fb795b41e9c64c7ed59cc411`. The test extracted selected methods through Python AST, supplied CPU Torch and fake TT/model objects, and asserted that no TTNN module was imported. It tested binding state preservation, stale-budget clearing, capacity rounding, exact prefill table identity, one copy for a changed host table, zero extra copy for an equal table, alternate-device-table rejection before model/upload operations, invalid active-mask/position rejection, reset to all-active configuration, and rejection of non-greedy host compatibility requests. The stage-review extension described below added canonical cache-tensor acceptance and rejection coverage. This verifies host control flow, not device memory placement, cache numerics or asynchronous execution. The temporary test command was `python_env/bin/python /tmp/qwen38_host_contract_audit.py`; hardware remained exclusively owned by the coordinator.

Stage review subsequently found missing convolution/layout validation in `bind_cache`. The coordinator added complete validation of these per-device tensor contracts before any trace release, upload or binding publication:

| State | Required shape | Dtype | Layout/memory |
| --- | --- | --- | --- |
| Each full-attention K and V | `[num_pages,1,32,256]` | BFP8 | TILE, interleaved DRAM |
| Linear recurrent | `[B,12,128,128]` | FP32 | TILE, interleaved DRAM |
| Linear convolution | `[B,3,2560]` | BF16 | ROW_MAJOR, interleaved DRAM |

All four tensors must be non-None. The extended host probe accepts a valid externally supplied cache/table without replacing any cache tensor identity. It then separately substitutes None, a malformed shape, wrong dtype, wrong layout, and L1 memory for each K, V, recurrent and convolution tensor: all 20 invalid cases raise `ValueError`, preserve the old cache/table/position/RoPE identities, active mask, ownership, host page shadow, replay budget and signature ledger, and perform zero fake uploads and trace releases. Combined with the original 20 cases and one valid device-binding case, the saved result has 41 passes. No malformed cache was submitted to hardware.

The audit also traced a device-table placement requirement to native paged-cache source. The fused-update program factory aliases a page-table circular buffer only for a sharded tensor, while the reader chooses DRAM uint32 loads versus an L1-sharded uint16 path using `page_table_is_dram`. An arbitrary INT32/RM interleaved-L1 tensor is therefore not interchangeable with the owned DRAM table. Binding now requires `page_table.memory_config() == ttnn.DRAM_MEMORY_CONFIG`; the host test confirms that an L1 device-table input is rejected before altering existing state. No malformed table was submitted to hardware.

Sampling had a separate actual parameter translation error: public temperature T was passed directly to the native multiplier field. Native sampling documents `temp` as `1/T` and multiplies logits by it. The coordinator changed the wrapper to `1.0 / temperature`. `sampling_params_host.json` verifies `.5 -> 2`, `.8 -> 1.25`, and `1 -> 1` for all 32 lanes, with unchanged k/p and seed refresh values. That test used AST plus standard-library fakes only; see `AUTOFIX_sampling_params.md`.

Two additional narrow API concerns were repaired during the audit: an invalid changed `active_slots` tuple was previously published before validation, leaving a bad mask after `ValueError`; and high-level `host_sampling=True` silently used argmax even when non-greedy parameters were supplied. The wrapper now validates a proposed active mask and its authoritative positions before changing the graph configuration, and rejects host compatibility requests with top-k other than one. The host tests cover these branches. The low-level `decode_forward(..., host_sampling=True)` correctly returns full logits to its caller without sampling replay; that caller owns selecting and supplying the next token. The high-level compatibility loop supplies its host-selected greedy token explicitly.

The latest `full_model_contract.py` source adds binding, all-logits repeat/batch equality and explicit greedy host/device equality checks. Original `contract_b3.json` and `contract_b32.json` predate these assertions; `contract_b3_final.json` now records the expanded B3 checks. No unrecorded B32 result is inferred from test source alone.

### 9. Full-stack profiling is rejected before setup

The stage-review follow-up inspected `tests/run_full_model.py` and evaluated only its extracted argument guard using standard-library `argparse`; the script itself was not imported or run. Source order is `parse_args` at line 24, the `a.profile and a.full` guard immediately at line 25, thread setup at line 27, fabric setup at line 28, mesh opening at line 29, and `build_generator` at line 33. Thus the forbidden full-stack profiling combination exits before device/model setup. Module imports precede parsing in the actual script; this check makes no claim that it rejects before imports.

`profile_guard_host.json` records all four full/profile boolean combinations: only `(True, True)` exits with argparse status 2; the three permitted combinations pass the guard. The source hash is `263fbd3c038b1dab764235c1f78a407d9de61978759413c34de04b106758c8f4`. This test imported no Torch, TTNN, model or hardware module and did not execute any setup/profiler function.

## Minimal discriminating probe for the hardware owner

First rerun the reduced existing test with tracking enabled from process startup, without program-cache suppression:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 \
python -m models.autoports.qwen_qwen3_8_27b.tests.run_full_model \
  --length 33 --generate 4 \
  --output models/autoports/qwen_qwen3_8_27b/doc/full_model/probe_trace_alloc.json
```

The coordinating agent must supply its established environment/device wrapper and serialize this job. This command was **not run** by the diagnosis agent. Tracking is captured when TTNN imports; setting the variables later is ineffective. Read `get_unsafe_tracked_ids(mesh, trace_id)` before each replay when isolating a failure. The tracker automatically performs that check in `ttnn.execute_trace` with tracking enabled.

Then extend the exact-contract probe with these four comparisons, reusing the reduced model:

1. **First capture and feedback:** record persistent token, signed-position, RoPE, page-table, logits, convolution and recurrent buffer IDs/addresses before and after capture. Snapshot state to CPU, not newly allocated device backups behind a live trace. Before each sampler replay, compare its output with CPU global argmax of that model replay's logits. Before the next model replay, inspect the exact bound token buffer and positions; require token feedback identity and exactly one position increment.
2. **Changed inputs and request reuse:** force two different valid tokens/positions on equivalent prefilled states and compare logits/state, not merely top-1. Run prompt A, different prompt B of another length within the same allocated capacity, then A again. Require tracker cleanliness at each first replay and compare A's result/state against a fresh equivalent request. This distinguishes consumed temporary buffers from shape-dependent persistent allocation hazards.
3. **Page-table refresh:** keep geometry fixed, redirect the next unused logical page to another allocated physical page, and inspect the physical K/V page written. Also run the T0/device-T1/host-T0 ownership sequence. Ensure the unchanged-host-table branch adds zero uploads after its first submission.
4. **Inactive rows:** on B=8, leave one row inactive with valid safe token/RoPE inputs and preserved signed -1 position. Run several replays while another row advances. Assert byte-exact inactive KV/conv/recurrent preservation and signed position stability, then reactivate with authoritative token/positions and verify only the intended state advances.

The existing probe's cumulative counters should be sampled as before/after deltas for each phase. A same-token reduced-model output such as four spaces is not a sufficient assertion for any of these contracts.

## Scope and source snapshot

No implementation edits, TTNN imports, resets, or device jobs were performed by this agent. The original investigation used source/log/JSON reads and AST/hash inspection. The coordinator explicitly authorized the later AST-extracted host-control-flow tests with fakes; their narrow execution scope and source hashes are recorded above.

During-review snapshot hashes (the coordinator continued editing afterward):

```text
generator.py 10a838919438844da1aef5bd1f6119af8b0071d9d77f41b4aacb327454b9ef9e
model.py     1c738cb74d004b26ca3e75d5c6d1df956ede54fb89863b61d3390885cb1bb2e8
probe2.json  84841bb6158b46fe9e1937e9c156d53250a71a768cbbb8909dea0781d072ba40
```
