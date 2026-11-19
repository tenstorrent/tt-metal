# Runtime fallback and ownership audit

Evidence reconciliation: 2026-11-13T10:23:23.894797+00:00. This is a source audit with scoped existing runtime evidence, not an independent hardware run or a completed-stage verdict. No production source was changed for this audit.

The optimized decoder baseline is **SHA256** `b1d5016dd769ab65fce82e27d483f73a33fa2e746e29e21952d521052948d2d8`, not a Git revision. It is exactly `tt/multichip_decoder.py` from commit `c2254f91` (also the implementation preserved by the optimized-stage evidence). Verified with:

```bash
git show c2254f91:models/autoports/ornith_ai_ornith_1_5_9b/tt/multichip_decoder.py | sha256sum
git diff c2254f91 -- models/autoports/ornith_ai_ornith_1_5_9b/tt/multichip_decoder.py
```

Source hashes at the original reconciliation follow. Each runtime artifact retains its own execution scope and source provenance; these hashes do not retroactively change an artifact’s recorded source hash:

| File | SHA256 |
|---|---|
| `tt/model.py` | `96ccc68b139ac8df11034d6dcf4777e6fb700f356b6adcd8f34d77807b811dbe` |
| `tt/generator.py` | `2358f36b100e3022141ee646f7b9ea09d180827e40f8ed08879ae31aa0f1012e` |
| `tt/multichip_decoder.py` | `39d6f9a424ee0d7abdd7698ab6daaa5a5551050b5939cde1b6d931b9da919f16` |
| `tt/prefill_plan.py` | `2985bbfb67899b46815423a7f82dc867ba47e8c501f8d54cd340932a52ce1426` |
| `tt/prefill_ops.py` | `462b6f8a1a3ae2975f20048a1729f140e9b55cd9ab4b3380570933db1dba012f` |
| `tt/prefill_runtime.py` | `14ac2ede2e63c346deb7ad264557e8da4f22a88f3cc547c75e56d87b7110c4dd` |
| `tt/prefill_contract.py` | `fa2a00bc186c963c688c5ecde96f098382099e1090cdf44cd77cab8f52f066b9` |
| `tt/batch_state.py` | `64efa648ac650d44264d3db643a049a49f47c87dee311b6e36abb56b7b5ce808` |
| `tt/sampling_state.py` | `64fe33127acfc615a753b0fa5b0ed3e2747119a8d03fcb863d893f363f385b8f` |
| `tt/cache_contract.py` | `1fa0b9326da358e06ea31ced88089d7b7a5181d313770eabdc4b4215af1f0ec4` |
| `tt/decode_contract.py` | `334469ffa5f3fac8dde746491a62d14298a53f0e70babd7d711927a24e987fcc` |

## Production execution and precision

`OrnithModel` requires mesh `(1,4)` and loads the checkpoint's text-model weights. Its embedding shards hidden width, gathers at stack entry, and its LM head shards vocabulary. Final RMSNorm applies the checkpoint's weight-plus-one convention. There is no HF model-forward, CPU attention/MLP, single-chip model, or automatically selected replicated-model implementation in this path. CPU checkpoint loading/packing, tokenizer use, request validation, integer maps, sampling parameters and prompt padding are setup/control work; they do not compute hidden activations or replace TTNN layers.

The complete decoder diff from the baseline adds `global_pages` metadata and physical `num_blocks`, bypasses private offsets for global page IDs, adds safe unused SDPA aliases, and adds `masked` plumbing with safe RoPE indices for inactive rows. It does **not** modify the dense projection weights, dtype/fidelity configuration, GDN/conv kernels, CCL implementation, native packed decode, residual chaining, or original chunk-cap selection. `model.decode` calls `decode_batch_packed` directly for each layer and passes its packed result to the next layer.

| Contract | Inspected preservation |
|---|---|
| TP strategy | One native TP4 stack and the original shared `MultichipCollectives`; no model replication or single-chip alternate route. Replicated BF16 residuals across TP ranks are the baseline residual contract, not replicated model execution. |
| Attention/MLP weight precision | Decoder attention and MLP weights remain BF8; attention uses HiFi2 and MLP LoFi according to the existing per-name compute dispatch. Existing FP32 destination/conv/GDN settings remain unchanged. |
| KV and recurrence | Actual paged K and V storage remains BF8; state updates are cast independently to `state.key.dtype` and `state.value.dtype`. BF16 projected/attention inputs and BF8 page storage retain their distinct roles. GDN recurrent state remains FP32 and convolution history BF16. No cache downcast or host reconstruction is added. |
| Prefill collectives | Canonical buckets preserve the original precision branch: linear511 remains below512, linear>=512 retains its BF8 communication policy; full attention follows the original rounded physical height. The existing decoder `linear/reduce/finish` functions own the compute and CCL decisions. |
| Chunking | Original maximum32768 linear and16384 full-attention chunks remain anchored at zero. Physical bucket selection is based on the whole logical request chunk, not a fragment length or cache capacity. |
| Residuals | Prefill staging is BF16 in DRAM with explicit row-major gather/write handling and tiled operation inputs. Decode retains packed BF16 `[1,1,B,4096]` in the original local sharded layout between layers; no boundary host transfer or extra layer-to-layer CCL is inserted. |
| Inactive rows | The masked trace snapshots/restores inactive recurrent/conv rows on device, keeps signed position-1 for paged kernels, and clamps only the RoPE lookup index. Active numerical kernels are retained. The all-active trace omits the mask helpers. |

`PrefillRuntime` is the public generator's prefill implementation. It does not call `model.prefill` as an exception fallback. Its normalized adapters reuse the loaded optimized layer, preserve compute/collective policy and add runtime masks/history selection. Linear padded rows set g=0 and beta=0; runtime gather selects the last3 convolution rows at the actual logical end. Unaligned continuation restores a saved device boundary and at most31 layer-input rows, replays that bounded overlap, and discards overlap output. The actual code does not implement the obsolete one-token-prologue suggestion still present in one validation error string in `make_attention_inputs`.

When a scheduler supplies no final horizon and its growing prompt changes canonical shape/precision compatibility, `PrefillContract` explicitly rebuilds the prefix through the same optimized device runtime. This is recorded as `rebuilt_prefix_rows`; it is extra model work, not a host activation fallback, smaller-chunk substitute, or reuse of incompatible recurrent state. Automatic prefix caching remains excluded under `recurrent_state` and is distinct from within-request resumed prefill.

The original decoder still exposes list/private-state compatibility branches and mixed-pool pack/scatter code. The standard generator constructs a complete ordered pooled cache and calls native packed decode. Its serving registry validates that geometry. These legacy methods are not silently selected in response to a failed optimized execution. An invalid binding, unsupported mode, missing profile or illegal command raises. The optimized rejection history in `../optimized_multichip_decoder/topology_decisions.md` and `work_log.md` remains the policy ledger; none of its rejected candidates is re-enabled here.

## Host boundaries and split sampling

The selected algorithm remains `models/common/sampling/generator.py` / `tt_sampling.py`, called through `_run_sampling`. `SamplingSlotState` handles existing common buffers, slot permutations, prompt/output counts and inactive-state preservation; it does not implement a replacement sampling algorithm.

`capture` records model and sampling traces separately for prepared masked/unmasked and penalty modes. The model trace produces vocabulary-sharded logits and advances active positions on device with `plus_one(skip_negative_entries=True)`. The sampling trace consumes those logits, writes selected IDs into the persistent model token tensor, advances the explicit device seed stream once, and restores inactive seed/count rows. `_execute` only selects and replays the prepared model/sampling traces. It performs no host token computation, logits readback, host argmax, request-time tensor allocation, page-table rebuild, or H2D upload.

The trace wrapper acknowledges the precise producer logits allocation immediately before its sampling consumer. Production generator code does not use a whole-capture `corruptible_allocation_scope`. It avoids the common high-level seeded eager sampling path and uses the common device math inside its own trace. Greedy is common top-k1 selection, not a host argmax implementation. `sampler_probe3.json` compares common top-k greedy against common forced argmax on real reduced-model logits: both selected470; recorded sampling-only times are approximately0.482ms and2.537ms. This is one numerical fixture, not broad tie coverage or full-model performance. The second common implementation in the adjacent tt-transformers checkout was compared by interface/source in `README.md`; this audit does not invent a missing cross-library benchmark.

| Boundary | Permitted host behavior and exclusions |
|---|---|
| Default `generate`, device sampling | Prefill returns device logits and common sampling chooses the first token on device. Decode sampling is traced. Python reads returned IDs to construct the answer and schedule another replay; those reads are observational and do not feed IDs back. In the ordinary path `token_refreshes` stays0 after prefill. |
| `generate(next_input=...)` | Explicit teacher-forcing compatibility uploads the callback-selected next token. This is separately labeled `teacher_forcing`; it is not measured token-out autoregression. |
| `generate(host_sampling=True)` | Explicit greedy host compatibility materializes full logits and uses host argmax/reuploads tokens. It is separately labeled `host_sampling`; it must not be included in on-device token-out throughput. |
| Low-level no `sampling_params` | Returns independent writable full-vocabulary host logits `[B,1,V]` for external host sampling/filters/logprobs. It does not invoke device sampling or advance RNG. Full host input reload is required on every such decode. |
| Low-level device sampling | Returns IDs `[B,1]` or an owning submission handle; no full-logit readback. Inactive row sentinels are applied to the already-read result using immutable submission metadata. |
| Readiness/all-logit prefill | `PrefillRuntime._head` calls `model.host_logits` only when device output was not requested. It slices valid logical output rows on host for diagnostics/host sampling. No read logits become hidden state. |
| Async output | `_submit_read` queues owning asynchronous host transfers before later trace replay can overwrite the source. `read_decode_output` synchronizes/returns events; `process_decode_output_host` converts only that submission and removes its strong pending entry. Neither function samples or advances positions. |
| Setup/admission | Host-to-device copies initialize fixed token/position/table, masks, parameters, histories and seed buffers. Commanded full/page-only reloads copy stable buffers explicitly. Resident decode ignores stale host tokens/positions and does not infer reloads from comparisons. |

No broad exception handler in the production model/generator/runtime substitutes host math or another architecture. Invalid device requests fail rather than silently switching to host sampling. Explicit host compatibility is intentional and observable; it is not a performance fallback.

## Cache ownership and reset

- `OrnithGenerator(...,kv_cache=...,page_table=...)` uses the exact external cache as `self.cache`, requires its explicit CPU integer table, and skips `model.allocate_cache`. The factory forwards both arguments. The registry validates loaded layer topology, slot count, actual pool shape, individual K/V dtype, pooled BF16/FP32 recurrence, TILE layout, allocated storage and `tensor.device()` mesh identity. Reduced-layer diagnostics are permitted internally but cannot advertise the full32-layer cache specification.
- `bind_serving_cache` explicitly identifies the cache used by the one-argument release callback. It rejects closed generators, pending outputs and existing live captures. A later binding does not free or replace the standalone cache; retained allocations stay in the memory budget. Constructor injection avoids that duplicate model-cache allocation.
- Default `reset_slots` and `release_request` target the serving binding. An explicit `kv_cache` scopes reset to that object. Slot release resets only selected recurrent/conv state, runtime prefix history and sampler token counts, and leaves KV backing allocated. New request sampling parameters/seeds are initialized through explicit admission commands. Release is not itself a new sampled request.
- `reset()` is the standalone high-level whole-`self.cache` operation. It zeros owned recurrent state, optionally clears that cache's KV, and invalidates its runtime histories. It is not used as a substitute for slot release. For batch>1 high-level generation now resets slot0 only and preserves peer model/sampler state.
- `prepare_decode` and prefill warmup snapshot the actual passed cache tensors and sampler state, warm the required programs, then restore those snapshots. They do not substitute a fresh private cache or reset unrelated external bindings.
- Full-attention ordinary restart overwrites the valid prefix at position0. Explicit `clear_kv=True` uses the known populated causal page extent, rejects duplicate/invalid causal mappings, skips future padding aliases, and does nothing to unknown unpopulated pages. Automatic cross-request prefix sharing is not supported, so callers must not present another request's live causal pages as owned storage.
- `close` drains pending transfer events, releases trace IDs once and marks the generator closed. It does not free caller KV tensors. Output handles retain transferred data for observational completion; `cache_ownership_full.json` passes read-after-close and repeated close on all32 layers, with six trace releases and caller backing still allocated. Completed outputs leave the pending-submission dictionary.
- Capacity helpers return host metadata and require measured free DRAM, explicit persistent categories and reserve, and an allocation-validated block count. A retained target pool can be credited only by its actual already-measured K/V bytes; this does not authorize allocating a second pool. Arithmetic headroom alone is not an advertised maximum-feasible claim.

## Reconciled evidence and stage boundary

| Artifact | What it establishes at inspection |
|---|---|
| `batch32_full.json` | Pass: all32 layers, B32/context512; mixed prompts33..64, semantic device-greedy prefill, selected-slot/batch consistency, row ordering, inactive guard, resident feedback/position checks and explicit-feedback control. Program count597 retained. |
| `capacity_full_final.json` | Pass: all32 layers, B1/context262144, validated eight-layer allocation triples with8192 physical blocks per layer, nonaligned prefill262143 and full prefill262144, finite terminal decode at position262143; programs820→820, measured allocator memory unchanged across long prefill, and validated total token budget262144 with1GiB/device reserve. Supersedes `capacity_full.json` for final context/allocation/budget boundaries. This is capacity/finite-output evidence, not max-context HF accuracy. |
| `serving_decode_probe_final2.json` | Pass, reduced layers0/3 B3: invalid inputs rejected before mutation, stale host inputs unread, forced changed/unchanged table reload, two pending outputs, observational readback, inactive model/RNG preservation, whole-pool and resident-input remaps, scoped reset and host output shape. Supersedes the failed `serving_decode_probe_final.json` for those cases only. |
| `external_cache.json` | Pass, reduced layers0/3 B3/context256: arbitrary11-block and permuted24-block global pools, external-vs-control logits/state equality, inactive/peer preservation, valid-only clearing, atomic invalid page rejection, stable backing/traces and program count485→485. Constructor-injection/release-routing acceptance is separately supplied by the all32-layer `cache_ownership_full.json` below. |
| `serving_prefill_probe.json` | Pass, reduced layers0/3/context1024: mixed slot orders/starts including68, batched-vs-solo bit equality, device logits/greedy, inactive rows, atomic rejection and implicit-horizon33→68→513 rebuild equality; programs614→614. |
| `sampling_state_probe.json` | Pass: synthetic full32 permutation, scoped reset/snapshot and inactive seed/count preservation. This is buffer protocol evidence, not a complete sampled-request behavior test. |
| `watcher_trace.json`, `logs/watcher_trace.log` | Existing reduced watcher run completes cross-request trace replay (35 model+35 sampling replays, zero token refreshes); log records Watcher startup and clean driver closure. It is not a final-source full32 watcher certificate. Retain the underlying watcher log/command/environment when promoting this gate. |
| `sampling_transition2.json` | Pass, reduced layers0/3 B3/context512: sampled prefill token enters seed history, seeded resume matches uninterrupted continuation, greedy/sampled parameter reload preserves stream position, scoped history reset preserves peers, three penalty-adjusted greedy winners match the host BF16 oracle, eight unseeded admissions vary, and programs520→520. Supersedes failed `sampling_transition.json` for these gates. |
| `prefill_generation.json` | Pass, reduced layers0/3 B3/context256: nine invalid admissions reject atomically; physical32 sampler controls work for logical1/3/32; sampled B3 prefill preserves raw temperatures; initial/warmed B3 native fixed-pool generation preserves inactive peers. Seven model and sampling replays use zero token refreshes; programs472. |
| `cache_ownership_full.json` | Pass, all32 layers B3/context256 with arbitrary29-block pool: injected constructor makes zero model-cache allocations; later binding retains standalone storage; malformed geometry/tables reject atomically; default release targets serving storage and preserves peers; release/close are idempotent; pending output drains and remains readable after close; caller backing survives. Actual single-group eight-layer BF8 K/V specs and configured768-token measured budget are recorded. |
| `runtime_fallback_probe.json` | Pass, reduced layers0/3/context128: host logits/argmax and oversized readbacks are forbidden during token output; direct H2D upload/copy is forbidden inside seven decode replays. Eight device-greedy tokens equal explicit host-greedy compatibility output. Device path records seven sampling replays, zero token refreshes and eight physical-token reads; host control records zero sampling replays and seven token uploads. |
| `tests/full_model_external_ownership_test.py` plus cache host tests | Host-only constructor/metadata/lifetime coverage passed14 tests during the ownership change. These use actual method AST with allocation stubs, not hardware claims. |

The constructor ownership, sampler transition, prefill admission/native generation, runtime host-boundary tripwires and final maximum-context capacity gates listed above now have passing artifacts. No pending claim remains for those probes. `doc/context_contract.json.full_model.status` is `capacity_passed`, preserves262144 supported context without reduction, and records the final measured weight/KV/staging/trace/other-persistent accounting. B32/context512 evidence remains distinct from B1/max-context evidence; B32 simultaneous maximum-length requests are not claimed to fit.

This audit does not decide final HF accuracy, qualitative HF-control verdict, sampler/per-token performance, TTFT/token-out reporting or stage-review status. Those are recorded in their own acceptance artifacts. Passing this scoped runtime audit is not a stage-review pass.

The completed focused tripwire run can be reproduced with:

```bash
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_runtime_fallback_probe --output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/runtime_fallback_probe.json
```

It warms reduced real layers0/3, forbids `model.host_logits`, torch argmax and readbacks larger than the physical32-token buffer during device generation, forbids direct H2D upload/copy inside `_execute`, and compares8 device-greedy tokens with explicit host-compatibility greedy tokens. It records source hashes and counters, not performance. The root agent ran it under serialized device ownership; `runtime_fallback_probe.json` records `passed: true`. It closes this boundary probe only, not full-model quality or throughput.


## Final measured runtime boundary

`performance_primary.json` supplies the accepted full-model measurement at batch 1,
128 input / 128 output tokens: 92.579 ms TTFT and 111.904 tokens/s/user, median
of three warmed serial runs. Every repetition records 127 model/sampling trace
pairs, zero token/page-table refreshes, one request-initial position write, 128
caller token reads, zero captures/releases and unchanged 712 programs. The
teacher-forcing result is separate and includes forced host-token uploads.
The reduced advice-enabled profile (`profile/decode_perf_report.txt`) measures
0.605 ms for sampling/feedback, about 6.8% of full token-out latency. It does
not show dominant sampling cost or a full-vocabulary gather. Profiling-only
warmup drains are removed before its measurement window and are absent from
production code and primary timing.

Final import normalization is recorded in `import_format_provenance.json`. The two cache-helper aliases became qualified calls to the same module functions. After canonicalizing those names and top-level imports, the implementation AST is identical; the runtime and sampling audit above is unchanged. `logs/final_host_contract_tests2.log.gz` passes all 40 host contracts after updating the AST-only fixture namespace.
