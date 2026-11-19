# AutoDebug: concurrent ragged chunked-prefill recall

Inspection date: 2026-11-17. Final evidence read: 11:31 UTC. Inspection only; no reproduction or device activity.

## Findings

**The enabled concurrent chunked-prefill test fails, but its cause is unresolved. No concrete source-level loss of recurrent state, wrong slot ownership, or unsupported resume offset was established.** The two solo recall tests pass. The strongest supported discrepancy is a **coverage gap**: the historical Stage9 full-suite JUnit explicitly skipped all three chunked-prefill tests. Its 68-pass result is not evidence that this path previously worked.

The concurrent schedule really exercises partial prefills, decode interleaving, nontrivial slot remaps, and starts such as 640, 896, 2688, and 2944. Inspection and the existing audit contradict several simple corruption stories: partial requests retain their slots; post-prefill decode reloads authoritative inputs and excludes unfinished requests; remaps include off-batch recurrent state; and this Qwen implementation does not forward an arbitrary unaligned start directly to chunked SDPA.

There are concrete scheduling-dependent arithmetic changes, including different full-attention singleton/block grouping, different MoE row grouping, and recurrent BF16 rounding at public-call boundaries. These are **mechanisms to discriminate**, not a demonstrated explanation of the failed assertion. Existing HF quality/PCC limitations do not cover this exact recall corpus and cannot waive it. No production fix is justified by the current inspection alone.

## Scope, revisions, and direct observations

Paths in this report:

- `M/` = `models/autoports/qwen_qwen3_6_35b_a3b/` in tt-metal.
- `P/` = `../vllm-tt-plugin/`.
- `A/` = `M/readiness_vllm/optimized_acceptance_async_2048/`.
- `T/` = `ttnn/cpp/ttnn/operations/`.

Checkout HEAD is `8fbe38e7`; the most recent model `tt/` source commit is `08384c1007d1a1065955221284f6d2c197d85260`. `git diff 08384c10 -- M/tt` is empty. Plugin HEAD is `5d1785b76d17a8989c9495f6939abea6cef62f09`, with a clean plugin worktree. Existing unrelated tt-metal worktree changes were left untouched.

`A/runtime_identity.json` and `A/server.log:107` identify EngineCore 3726882, API 3726605, all 40 layers, selected `baseline_bfp8`, context 262144, capacity 32, 1x4 Blackhole mesh, page size 32, async scheduling, and decode-only traces. Server log lines 27 and 41 report chunked prefill enabled with budget 2048. Prefix caching is disabled, so this is not a KV-only prefix-cache hit restoring an incomplete recurrent history.

The inspection snapshot of `A/sampling_full/sampling_tests.log` records:

| Test | Observed status |
|---|---|
| `test_a_solo_split_prefill_recalls_its_needle` | PASS |
| `test_a_long_split_prefill_recalls_its_needle` | PASS |
| `test_prefills_sharing_a_step_each_recall_their_own_needle` | FAIL |

**At the final evidence read, pytest had reached 73% and was still running; it had not emitted the failure traceback, expected/actual output, miss count, request indices, or final suite totals.** Do not turn the failed node label into a fabricated wrong-passphrase example or assume a request-to-slot pairing. The final assertion site and exact responses remain necessary evidence. Three concurrent prefill sweeps appear in the server audit; that is compatible with reaching the final accumulated-misses assertion, but is not a substitute for the traceback.

The parent reports passing full40 exact controls, 23 tracked lifecycle checks, 12 zero-difference HTTP-logit cases, and 13 growth checks over 318 submissions. Those are useful contrasts, but none supplies exact outputs/state for this twelve-prompt ragged recall run.

Only source/artifact reads, standard-library log/XML parsing, revision checks, and this report write were performed. Three xhigh inspection subagents separately checked ownership, plugin protocol, and ragged offset/lowered attention contracts. No target code was run, TTNN imported, HTTP request sent, profiler used, process killed, device reset, or server launched.

## What the current and historical tests actually cover

The live pytest command uses `tests/tt -v`, model Qwen/Qwen3.6-35B-A3B, and capacity 32. A read-only check of that known pytest process's command line and **only** its `PYTEST_ADDOPTS` environment field established these effective extra options:

```
-p vllm
-p models.autoports.qwen_qwen3_6_35b_a3b.tests.checkpoint_default_test_requests
--tt-chunked-prefill-budget=2048
--junitxml=models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/optimized_acceptance_async_2048/sampling_full.xml
```

There is no recall reasoning-effort override. `P/tests/tt/test_chunked_prefill.py:112-142` therefore uses legacy `/v1/completions`, temperature 0, and the original 24-generated-token cap. The local pytest plugin affects exactly three unrelated chat test names (`M/tests/checkpoint_default_test_requests.py:23-29`), leaving recall requests unchanged. Server thinking defaults and the Qwen reasoning parser do not transform these raw completion prompts into chat requests.

The concurrent test creates three rounds of four distinct prompts, each with 2048 **filler words**, a nonce, and a distinct `cobalt-heron-{round}-{index}` passphrase planted at 75% of the filler (`P/tests/tt/test_chunked_prefill.py:70-85,224-253`). Those words are not 2048 model tokens. Actual concurrent prompt lengths are 3077, 3103, 3111, and 3137 tokens, in varying admission order. Solo short/long prompts are 3105/9284 tokens according to the audit. `asyncio.gather` preserves client configuration order (`P/tests/tt/utils.py:151-156`); that does not establish that scheduler slot order equals client index order.

The matcher accepts case differences and two Unicode hyphen variants, then requires the complete passphrase with word/hyphen boundaries (`P/tests/tt/utils.py:10-28`). Without exact failure output, one cannot distinguish missing history, an incorrect digit, truncation, or another assertion/error solely from the node name.

The historical `M/readiness_vllm/final64_gate/sampling_full.xml` explicitly contains `<skipped>` for **all three** recall tests, each with the message requiring `--tt-chunked-prefill-budget`. This proves the coverage gap independently of prose describing the older thinking-off server. The installed shared runner's `_run_plugin_sampling_tests` command also does not add that flag (`.../tt-model-bringup/0.1.24/runtime/readiness_check/run_vllm_server.py:450-459`). An older "full profile" label cannot establish enabled recall coverage.

Under the selected qualitative-check rules, raw completion is continuation stress coverage for this chat checkpoint. The canonical stress failure still needs resolution; it is not sufficient evidence for a general correctly formatted chat-quality verdict. Any HF/native control must reproduce the exact endpoint/rendering, prompt, cap, precision, and tokenization before being used causally.

## Observed execution ledger

First concurrent sweep, from `A/server.log`:

| Line | Submitted slot(s) | Absolute prefill interval(s) |
|---:|---|---|
| 7335 | 0 | `[0,2048)` |
| 7341 | 0,1 | `[2048,3111)`, `[0,896)` |
| 7347 | 1 | `[896,2944)` |
| 7354 | 1,2 | `[2944,3103)`, `[0,1792)` |
| 7363 | 3,2 | `[0,640)`, `[1792,3077)` |
| 7369 | 3 | `[640,2688)` |
| 7377 | 3 | `[2688,3137)` |

These are physical slots at submission, not proven client request indices. Slots of completed requests subsequently move; pending slots in this sweep retain identity until completion. The other sweeps repeat these partial-boundary shapes, with different final lengths, at lines 7478-7527 and 7619-7666. The plugin's existing `prefill_chunk_alignment` default is 128 (`P/src/vllm_tt_plugin/config.py:695-718`), consistent with the observed intermediate boundaries. All observed partial boundaries are also divisible by 64; they are not arbitrary element offsets.

The decode transitions immediately following prefill are decisive protocol evidence:

| Decode line | Host positions supplied with authoritative reload | Destination-to-source remap prefix | Active slots after submission |
|---:|---|---|---|
| 7343 | `[3111,-1]` | identity | `[0]` |
| 7358 | `[3103,3112]` | `[1,0,2,3]` | `[0,1]` |
| 7373 | `[3077,3104,3113,-1]` | `[2,0,1,3]` | `[0,1,2]` |
| 7382 | `[3137,3078,3105,3114]` | `[3,0,1,2]` | `[0,1,2,3]` |

Every listed transition has `reload_inputs=True`, sampling-parameter reload/reset true, and `pending_outputs=0` at entry. The same four transition forms occur at 7487/7502/7517/7527 and 7627/7642/7657/7666. Pending slot 1 is marked inactive during a B2 decode with one active request and later resumes prefill; pending slot 3 is marked inactive during a B4 decode with three active requests and later resumes prefill. **The audit proves their intended inactivity, not their tensor bytes.**

## Source findings and causal adjudication

### Retained recurrent state, page ownership, and remapping

1. The runner keeps request-owned state slots when an unscheduled request's input row is removed. Ownership is released on finish/preemption; prefill allocation reuses an existing owner's slot (`P/src/vllm_tt_plugin/model_runner.py:1026-1042,1265-1337`). DP=1 still forwards `empty_slots` (`:4128-4140`). Thus ordinary input-row compaction does not silently reset every continuation into slot 0.
2. The runner sets absolute start to computed-token count and end to start plus scheduled count (`model_runner.py:2445-2475`). Qwen forwards those bounds, binds page tables in submitted row order, slices exactly `tokens[row,start:end]`, and passes absolute start to the executor (`M/tt/generator_vllm.py:168-195`; `M/tt/generator.py:311-327,427-459`). Only `start==0` clears that slot's conv/recurrent state (`generator.py:434-437`).
3. Cache construction allocates separate mutable conv/recurrent/page-table state (`M/tt/cache_contract.py:145-174`) and validation rejects cross-slot/layer mutable aliases (`:237-253`). Shared K/V pools are deliberate; page tables assign logical ownership. Prefill binds the current submitted tables before execution.
4. Slot remapping registers **all capacity slots**, including conv, recurrent, page tables, CPU page shadows, and positions (`M/tt/serving_state.py:263-278`). `SlotState.apply` snapshots every source before writing destinations (`:241-249`). Runner full-capacity ownership maps use the same permutation, and are committed only after the model accepts decode (`model_runner.py:1372-1389`; `P/src/vllm_tt_plugin/async_decode.py:1479-1482`).
5. Any intermediate prefill forces host logits/sampling (`model_runner.py:2666-2670`). Only final rows emit a token or update request output state (`:3900-3957`). Prefill invalidates the resident decode chain (`:4147,4150`; `async_decode.py:409-411`), forcing the subsequent authoritative input reload (`async_decode.py:457-475`). This matches the audit and refutes accidental resident consumption of intermediate-prefill output.
6. Inactive recurrent commits are guarded at the native writer; its CBs are still drained even when stores are skipped (`M/tt/mesh_state_slab_recurrence.py:119-139`). The caller passes activity into the selected native path (`M/tt/multichip_decoder.py:530-546`), enabling the conditional writer define (`mesh_state_slab_recurrence.py:472-477`). The fallback uses `copy_if` (`multichip_decoder.py:550-564`), and convolution carry writes are activity guarded (`M/tt/slot_state.py:257-271`). Request release clears only its slot and does not erase the aggregate K/V pool (`M/tt/cache_contract.py:298-323`).

**Unresolved:** validation checks intra-submission page ownership; off-batch ownership is explicitly delegated to the scheduler/caller (`M/tt/prefill_validation.py:154-156`). No audit here records the complete live request-to-page mapping or all-chip tensor payloads. Therefore inspection does not prove absence of a lower-level masked-write, remap-copy, or page-allocation defect. It also supplies no direct evidence that one occurred.

### Ragged offset and lowered attention contract

The historical test comments about `tt_transformers` Q64/Q256 alignment (`test_chunked_prefill.py:15-20,182-187`) are not this model's implementation.

`M/tt/prefill.py:26-39` emits full-attention blocks only at absolute multiples of 256, with a multiple-of-32 row count up to 256. Otherwise it emits singleton tokens. For observed `[896,2944)`, execution is:

- 128 singleton tokens at absolute 896 through 1023;
- seven 256-row blocks at 1024 through 2560;
- one 128-row block at 2816.

For `[2944,3103)`, all 159 tokens are singleton attention: after reaching alignment at 3072, fewer than 32 remain. No token is skipped or rounded backward. This arithmetic is derived from source, not a device experiment.

Absolute RoPE indices/current position and logical page index `absolute//32` are prepared together (`prefill.py:115-128`). The page-window kernel copies eight page IDs starting at that logical page (`M/tt/prefill_transport.py:64-74`). Singleton attention passes exact `current_pos` to `paged_update_cache`; block attention casts K/V to cache dtype, uses that page window for fill, and passes the absolute runtime tensor to chunked SDPA (`M/tt/multichip_decoder.py:2090-2142`). Both SDPA configurations are Q32/K256.

The lowered SDPA reader/compute/writer divide runtime chunk start by query chunk size (`T/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp:370-371`, `compute/sdpa.cpp:103-104`, `dataflow/writer_interleaved.cpp:199-200`). That is a real alignment contract; **this caller satisfies it** because every block starts on 256. A generic flexible-SDPA alignment concern is therefore not a source explanation for this run. Singleton cache updates take the other path.

Residual rows are suffix-local, while cache positions are absolute. Transport copies `destination[to+i]=source[from+i]`, bounds valid rows, and zeroes only requested padding (`prefill_transport.py:35-44`; `prefill.py:101-113,202-223`). The two coordinate systems are used consistently in the inspected path.

### Numerical execution really changes, but causality is unproved

- Non-256-aligned public starts introduce more full-attention singleton work before aligned blocks. Singleton and block attention differ in head preparation, SDPA path, and output arithmetic. That can alter numerical results without indexing the wrong prefix.
- Linear attention uses call-relative 64-token chunks (`prefill.py:30-32`). The observed intermediate boundaries are all multiples of 64, preserving that grouping. The source retains the last three logical projected rows as conv carry (`multichip_decoder.py:462-471`); chunk64 tails use mathematically neutral zero padding (`M/tt/fused_decoder.py:2053-2062,2092-2094`). This weakens an explanation requiring arbitrary linear chunk misalignment; it does not establish bitwise device-state preservation.
- Every public prefill call rounds recurrent FP32 storage through BF16 once per linear-attention layer (`prefill.py:131-135`). Concurrent partitions add boundaries at 896/1792/640 etc. Solo splitting also rounds state, so this is not automatically a concurrency-specific bug. It is inherited behavior (`multichip_decoder.py:1856-1857`), not an obviously accidental conversion to remove.
- MoE grouping restarts at public-call row zero, with chunks up to 2048 (`prefill.py:217-223`). Selected prefill grouped MoE uses stochastic rounding; expert counts/offsets and `input_m_tiles` depend on launch rows (`multichip_decoder.py:1653-1658,1695-1753`). Thus regrouping the same sequence changes the selected numerical work. This does not prove cross-request leakage: bounded prefill processes submitted requests serially, not as one combined request-row MoE launch.
- Final-prefill sampling can use host logits when another row remains partial, while a lone final row uses device sampling. The first sweep logs host logits for earlier completions and device sampling at line 7377. Equal logits with a unique maximum should select the same greedy token; ties additionally require a matching tie policy. No exact failed response or matched first-token logits here establishes a route discrepancy.

These candidates need identical-token controls. Previous resumed-PCC/HF findings and constrained-thinking quality failures have different prompts, split geometry, caps, or sampling settings; they establish no exact-case inheritance for this failure.

## Asynchronous completion boundaries

Host submission is not device completion. The model's output pool snapshots output with a queued copy (`M/tt/serving_state.py:506-515`), reads with blocking CPU transfer or records a queue-0 event (`:518-526`), synchronizes before conversion when needed, and frees the lease only after an independent CPU result exists (`:528-554`). Plugin decode finalization explicitly synchronizes recorded read events before processing (`P/src/vllm_tt_plugin/async_decode.py:1653-1677`). Prefill and remapping require prior output leases drained (`M/tt/generator.py:420`; `M/tt/serving.py:529`); non-steady transitions drain pending asynchronous steps (`async_decode.py:935-962`).

The observed zero-pending-output reload transitions support this ordering. They do not establish all-chip tensor correctness or rule out a kernel defect merely because a host call returned. No synchronization failure, device assertion, or producer/consumer count contradiction was established in this inspection.

## Candidate ledger and smallest useful discriminator

| Candidate | Direct code contradiction? | Match to observed contrast | Missing evidence / smallest discriminator |
|---|---|---|---|
| Continuation reset/wrong prefill slot | None found; owner reuse and per-slot reset contradict simple story | Concurrent only would fit, but source/audit preserve pending slots | Compare exact owner identity and per-layer state before/after each continuation |
| Decode accidentally activates unfinished prefill | Refuted for logged host activation; device mask behavior remains unverified | Pending slots are `-1` and requested inactive on authoritative reload | Payload check of inactive state before/after B2 and B4 replay, on every chip |
| Unaligned resume directly misindexes chunked SDPA | Refuted for selected caller | Solo/concurrent differ in alignment, but planner handles it | Same-token partition-only control; no offset-rounding fix justified |
| Off-batch KV page alias/remap-copy bug | Not demonstrated | Could explain concurrency dependence | Complete live request/page ledger plus affected KV bytes across transition |
| Partition-sensitive recurrent/attention/MoE numerics | Concrete changed arithmetic; no violated contract established | Explains a possible solo/concurrent differential | Replay each exact observed partition serially, preserving tokens/pages/precision; inspect first differing residual/state/logits |
| Host/device greedy final-prefill route | Route difference observed; no wrong result established | Mixed partial/final admission changes route | Equal-logit first-token host/device comparison with exact request mapping |
| Inherited HF/raw-completion/cap behavior | No exact-corpus control | Solo pass does not guarantee all twelve distinct prompts pass | Exact rendered/raw prompts, 24-token greedy HF and same-code standalone controls |

No candidate currently has both a demonstrated contract violation and the exact failing response correspondence. Ranking one as the root cause would overstate the evidence.

## Follow-up work for the parent; not performed here

1. Preserve the completed pytest traceback/JUnit and all expected/actual responses. Identify whether the failure is final `misses`, request error, truncation, or another assertion. Map client request index to the audited token digest/owner rather than assuming arrival order.
2. Use the exact twelve prompts/token IDs, raw completion format, greedy settings, and 24-token cap for same-code standalone and original-HF controls. Preserve the canonical failed result. Use separately labeled chat-format controls if evaluating checkpoint chat quality; do not silently change the failing workload.
3. For a prompt that fails only under serving, compare identical tokens and page mappings under solo2048 partitioning, the observed ragged partitions executed serially, and those same partitions with the observed B2/B4 decode interleaves. This separates partition arithmetic from concurrent state preservation. Compare initial/final sampled IDs and first divergent logits before reasoning from fluent text.
4. If only interleaving fails, check pending slot1 around the B2 transition and pending slot3 around the B4 transition: all-chip recurrent/conv payloads, positions/activity, page shadows, and live KV ownership. If serial ragged partitions already fail, localize the earliest layer/operation divergence before changing recurrence precision, attention grouping, or MoE rounding.

No new capability, plugin environment variable, additional-config key, hook, default, or model-name gate is proposed. The selected `vllm-integration/references/plugin-changes.md` and `P/AGENTS.md` reserve new shared interface for exact human authorization and require model-owned gaps to be fixed in the model. An existing alignment override, disabling chunked prefill, weakening recall, or extending the cap might be a labeled diagnostic contrast; none is evidence that the required path is repaired.

## Other potential issues / test gaps

Existing bounded-versus-legacy parity evidence uses layers 0 and 3 and compares two implementations of the same partition (`M/doc/full_model/prefill_equivalence/resume_legacy_parity_drain.json`; `M/tests/run_prefill_equivalence_probe.py:195`). It does not prove full40 partition invariance or capacity32 pending-prefill interleaving. Generic flexible-SDPA alignment and selected stochastic MoE concerns merit separate controls, but the selected caller's alignment invariant prevents promoting the former here, and no exact-logit observation promotes the latter.

## Draft claim review

Headline claims were revisited against the actual test entrypoint/options, old JUnit skip elements, observed scheduler intervals, post-prefill reloads, full-capacity remap registration, inactive native state writer, convolution guard, suffix/absolute coordinate split, and lowered attention indexing. Two xhigh subagents additionally challenged the draft's protocol and numerical claims. Wording that could imply verified device-byte preservation was narrowed to source intent and logged host metadata. Plausible but unsupported ownership/alignment stories were not retained as root causes. Exact failure details remain pending at this snapshot; the causal verdict is deliberately unresolved.
