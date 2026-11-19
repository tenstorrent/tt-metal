# AutoDebug: P2 top-k 40 host compatibility overhead

Inspection date: 2026-11-16. No target imports, TTNN imports, device access, HTTP requests, benchmarks, profiler collection, source edits, or server/process operations were performed. Only this report was written. The prior root report was verified byte-identical to `models/autoports/qwen_qwen3_8_flash_next/doc/vllm_integration/AUTODEBUG_before_P2.md` before replacement; that preserved file was not changed. The earlier failed investigator launch is not model/runtime evidence.

**Leading diagnosis:** top-k 40 correctly selects host compatibility sampling, but that path conflates necessary token feedback with repeated metadata preparation, and processes the entire 32-row sampler canvas before selecting the one or two useful rows. These are concrete, avoidable generator-owned costs. Tracked diagnostics add a third route-dependent cost. They fit the failing-versus-passing boundary; their individual latency contributions and whether a particular fix reaches 1.15 remain unmeasured.

Keep `max_device_top_k=32` and the canonical device sampler. There is no supported wider common device sampler in this checkout. The smallest safe first change is to eliminate full 32-row FP32 expansion; the stronger source-level inefficiency is full-context page-table normalization on every host-sampled token. Both can be addressed inside the generator without changing the shared plugin interface or sampling semantics.

## Observations and evidence provenance

Inspected heads:

- tt-metal: `f82efe3784b1e1116eac1412f585dcbb0ed1a368`.
- vllm-tt-plugin: `ef1ec79b9679344d7a6820f8aae661bbb94d5909`.
- Actual plugin source: `/home/ttuser/qwen38-flash/vllm-tt-plugin/src/vllm_tt_plugin`; the prompt's path without `src` is absent. Target model sources and tracked plugin files were clean when investigation began. The wider workspace contains pre-existing artifacts.
- The [saved server log](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/acceptance_tracking/server.log), lines 14, 44, 173, identifies vLLM 0.31.0, all 48 layers, maximum batch 2, context 262144, async scheduling, and device sampling mode `all`. Event payloads show trace diagnostics active. P2 does not pin every runtime Python/binary hash; lowered C++ findings describe the prepared source, not a newly verified runtime build.

**Concurrent edit observed at final verification:** another workspace operation modified `tt/generator.py`, adding an unchanged-page-table shortcut before normalization. This investigation did not make or alter that edit. All generator findings and line numbers below refer to the inspected HEAD baseline, SHA256 `e7ee09fb4979aeeb32a47eacc3f1403732b71f127c5724d5c578a43304e14990`. The observed edited file had SHA256 `6d842f459f310ec984fc87feedf95965474bea2bc944e8826849146d36c42b48`; subsequent work may change it again. The report does not validate the new shortcut, treat it as P2 evidence, or claim its performance. Retrieve the baseline with `git show f82efe3784b1e1116eac1412f585dcbb0ed1a368:models/autoports/qwen_qwen3_8_flash_next/tt/generator.py` when following original line references.

The [raw P2 results](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/doc/eval_readiness/harness_requests/runs.json) and [gate result](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/doc/eval_readiness/harness_requests.json) record:

| Profile | Effective request | 1-user plain / profile TPOT, ms | 2-user plain / profile TPOT, ms | Route |
|---|---|---:|---:|---|
| Reasoning | temperature 1, top-p .95, top-k 40, seed 42 | 103.551 / 133.082 | 198.879 / 239.709 | Host |
| Non-reasoning | temperature 0, seed 42 | 103.515 / 103.514 | 198.773 / 198.749 | Device |
| Model card | temperature 1, top-p .95, top-k 20 | 103.540 / 103.577 | 198.850 / 201.295 | Device |
| Agentic | temperature 1, top-p .95; inherited top-k 20 | 103.560 / 103.571 | 198.883 / 200.791 | Device |

All timed requests returned 128 tokens without errors. Client and server-metric TPOT agree to much less than 0.01 ms. Reasoning ratios are **1.285180** and **1.205300**. Excess over greedy is 29.531 / 40.830 ms; reaching 1.15 requires saving **13.998 / 10.998 ms**, respectively, if paired baselines remain unchanged. These are end-to-end differences, not measurements of conversion, DMA, or sampling alone.

The harness clamps the advertised 32768 output limit to 128 for timing. Existing log events show reasoning prefill length 105 and decode positions 105..231. Actual P2 live context therefore spans only 4..8 pages, while metadata buffers retain the configured **8192-page width**. Carry both dimensions through the diagnosis.

A read-only parse of recorded log events establishes these exact windows. They are producer submission events, not kernel completion timestamps:

| Saved log lines | Role | Decode events | `reload_inputs=true` | Non-null logged page tables |
|---|---|---:|---:|---:|
| 9989–10118 | Reasoning acceptance request before timing | 127 | 127 | 127 |
| 10129–10386 | 1-user timed greedy | 127 | 1 | 5 |
| 10397–10526 | 1-user timed reasoning | 127 | 127 | 127 |
| 10547–10805 | 2-user timed greedy | 127 | 1 | 5 |
| 10825–10956 | 2-user timed reasoning | 127 | 127 | 127 |

Every timed host event's incoming position equals its corresponding generator cache-position shadow. Logged tables have shapes `[1,8192]` / `[2,8192]`. Each timed host window changes tables only at positions 128, 160, 192 and 224; 123 of its 127 decode events stay within the current page. Host submission lines are approximately 16.6 / 33.0 KB each. These explicit windows and fields support the route/reload findings; no correspondence was inferred from unlabelled token samples.

## 1. Required token reload triggers unnecessary metadata preparation

**Proven path.** [Plugin async planning](/home/ttuser/qwen38-flash/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py:464), lines 464–483, reloads inputs on every host step. That is necessary for the host-selected token. [Generator decode](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/generator.py), lines 973–982 and 1037–1046, also treats it as a reason to normalize tables and bind all positions.

The full width survives preparation:

- Plugin `model_runner.py:716–723,2384–2387` requests the context-derived table width; `input_batch.py:783–812` copies rows and zero-fills unused entries without shortening them.
- `model_runner.py:825–860` supplies per-layer group tables at the same width. With one group, the generator repeats the common table (`generator.py:619–622`). Either route supplies 12 full-width tables for this model's 12 attention layers.
- `_table_plan` selects rows and calls `normalize_tables` (`generator.py:625–640`). [Cache normalization](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/cache_pool.py), lines 229–256, calls `.tolist()`, allocates tensors, and iterates **every supplied entry**, including padding, through `_integer` and bounds checks before rematerializing tensors.
- Thus normalization alone performs **12 × 8192 × B = 98,304 / 196,608 Python entry-validation iterations per decode** at B1/B2. Its loop is not restricted to the 4..8 live pages.
- Normalization validates active pages once. `_table_plan:662–681` reconstructs a full retained-slot snapshot and validates again. Both validations turn complete padded tensors into lists, although their ownership loops examine only live prefixes (`cache_pool.py:275–282`).
- `_bind:689–700` compares and clones retained tables. **Unchanged page-table device uploads are already suppressed** by `torch.equal`; repeated host preparation is the issue.

At max_batch 2, the reconstructed 12 × [2,8192] INT32 snapshot alone contains 786,432 bytes. Normalization plus its validation and retained-slot validation materialize 393,216 / 589,824 scalar list entries per B1/B2 step. These counts exclude adapter diagnostics and other metadata copies.

The existing unchanged-within-page shortcut at `generator.py:655–661` requires `tables is None`. Host feedback supplies tables every step and misses it; device steady decode takes it. The validator's own contract says to run on scheduling changes rather than every token (`cache_pool.py:266`). The earliest intervention is before normalization, where new token feedback can be distinguished from changed metadata.

**Positions are a separate avoidable cost.** `_bind:702–707` performs 12 position uploads per active slot, hence 12 / 24 per decode, through `_refresh:265–267`. Device positions already advance in `multichip_decoder.py:1099–1100` → `qsa_chunked.py:75–80`; `generator.py:1050–1051` advances the host shadow. P2 host events show equal incoming/shadow positions. During stable decode these writes resend known values.

**Minimal generator-owned direction:** keep token reloads. Before advanced row copies/normalization, after structural/type checks, compare complete active input rows with retained validated per-slot tables. Reuse the no-table path only for identity remap, exact full-table equality, established cache ownership, and unchanged live-page extents. First use, any mismatch, new page, reset, remap or pool transition must use original validation. Do not rely on object identity; caller tensors may be mutable views. Raw `-1` suffix versus normalized zero may conservatively fail equality; current plugin zero padding covers P2. Do not truncate future reservations: they remain validated and preserved.

Independently omit position writes only when a valid resident-state/shadow invariant proves equality for that slot. Preserve repositioning, capture/preparation effects and transitions. Bare equality after a state reset is insufficient. Do not globally set `reload_inputs=False`, which would reuse stale token IDs. Changing only `reload_page_table` also cannot help: the generator checks `reload_inputs OR reload_page_table`.

This work scales with active users and supplies a cost dimension that fixed 32-row output conversion alone cannot explain. Its milliseconds remain unresolved.

## 2. Full 32-row logits transfer and CPU conversion precede row selection

The sampler canvas is BF16 TILE `[1,1,32,62080]` per vocabulary shard, across four shards, with vocabulary 248320 (`generator.py:119–125`). Host decode submits that entire tensor plus separate slot/active-row metadata (`1062–1066`). `read_decode_output:1220–1223` calls `.cpu()` without selecting rows.

This is a real full-buffer read: `ttnn/core/tensor/tensor_ops.cpp:233–255` lowers to `tt_metal/impl/tensor/tensor_apis.cpp:55–73`, which allocates full TensorSpec host buffers and reads the whole mesh buffer. No active-row descriptor reaches it.

After completion, `generator.py:1234–1241` composes all shards into `[32,248320]`, converts every row to FP32, allocates the submitted-row result, then selects useful physical slots.

| Rows | Aggregate BF16 payload | Corresponding FP32 materialization |
|---|---:|---:|
| Physical 32 | 15,892,480 bytes (15.156 MiB) | 31,784,960 bytes (30.313 MiB) |
| Active 1 | 496,640 bytes (0.474 MiB) | 993,280 bytes (0.947 MiB) |
| Active 2 | 993,280 bytes (0.947 MiB) | 1,986,560 bytes (1.895 MiB) |

These are tensor payload sizes, not measured bandwidth or a full peak-allocation accounting.

**Smallest fix:** remove whole-tensor `.float()` at line 1237. Keep composed values BF16 and copy only `values[output.slots[row]]` into FP32 `result[row,0]`. Assignment already converts selected BF16 values exactly. Preserve output shape, inactive placeholders, mapping and events; explicit `dtype=torch.float32` clarifies the result contract. This avoids converting 31/30 unused rows. It does not reduce DMA or full 32-row composition.

Two further tiers are distinct:

- **Select before vocabulary concatenation:** convert individual shards, select physical slots, and copy into final output vocabulary-column segments. Current composition is C++, not a Python clone/cat chain: `ttnn/ttnn/distributed/distributed.py:736–737` → `ttnn/cpp/ttnn-nanobind/pytensor.cpp:335–353` → `ttnn/core/distributed/distributed_tensor.cpp:592–631`. It transforms complete tiled shards in parallel and concatenates all rows. Per-shard selection avoids full 32-row concatenation, but single-shard `to_torch` still copies/untilizes (`pytensor.cpp:277–282`), and a Python loop may lose existing parallelism. Improvement is unproven.
- **Reduce DMA:** produce compact BF16 ROW_MAJOR output before `.cpu()`. A logical TILE slice of 1/2 rows still occupies 32 physical rows. `untilize_with_unpadding` has a ROW_MAJOR output contract (`ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation.cpp:438–448,574–576`); prefix slots can use inclusive ends `(0,0,B-1,62079)` for local `[1,1,B,62080]`. Sparse/remapped slots need correct extraction, not trimming by active count. Keep the canonical sampler canvas unchanged and preserve mapping through `DecodeSubmission`. Staging must remain valid until its queued read completes and be safe across submissions. Measure extraction plus readback together; compaction is a hardware-validation candidate, not a promised saving.

The composer's internal `.cpu()` returns already-host storage unchanged (`tensor_ops.cpp:234–235`): it is **not a second device transfer**. Its named host all-gather is a single-host no-op (`ttnn/core/distributed/host_ccl.cpp:26–36`), not another fabric collective.

## 3. Tracked diagnostics log unchanged full tables per host token

`generator_vllm.py:161–175` expands `page_table.tolist()` when diagnostics are enabled and either reload flag is true. `_log_event:46–47` JSON-serializes and flushes stdout. P2 host reloads therefore log 8192 / 16384 page IDs per decode despite stable tables; steady device submissions normally log `null`.

Each timed host window contains 127 such lines: approximately 2.11 / 4.19 MB of decode-submission text. This cost is reachable and proportional to users; its TPOT contribution is unknown. The timestamp is constructed after `.tolist()` but before JSON serialization/printing, so marker intervals cannot isolate logging cost.

A generator-adapter change can retain route/position/slot/table-change events and serialize full tables only when changed. Preserve full payloads on changed-table and page-only-refresh events: `models/autoports/qwen_qwen3_8_flash_next/tests/check_vllm_async_history.py:95–99` consumes those payloads. Preserve trace-allocation tracking and downstream evidence requirements. The discriminating comparison must suppress only repeated table payloads, not all tracking or the acceptance gate. No new plugin capability or environment variable is required.

## Async completion and sampler ownership

The dependency chain is:

1. Generator uploads required feedback and queues decode replay with `blocking=False`.
2. Host mode returns logits; device mode runs the common sampler and leaves the next token resident.
3. `read_decode_output(async_read=True)` queues a read and records a CQ0 event (`generator.py:1221–1223`). Wrapper return is submission, not completion.
4. `async_decode.py:1653–1677` waits read events before host conversion. The completion path then invokes plugin sampling (`complete_decode_step:1038–1059`).
5. `model_runner.py:4394–4401,4520–4523` selects active rows and invokes the plugin host sampler. Advanced row indexing makes another active-row copy, not a full 32-row copy.
6. Host submissions cannot use steady overlap (`async_decode.py:674–675`). Pending non-overlap output must finalize before dependent work (`935–965`, caller `model_runner.py:3245–3258`). The selected token is committed and uploaded for the next decode.

The feedback barrier is required by autoregression. It does not demonstrate broken async scheduling. Remove avoidable work around it, not the wait. The generator's `synchronizations` counter does not count plugin event waits; a zero counter would not prove absence of synchronization.

The model's `tt/host_sampling.py` is **not the serving sampler**. Its clones/history processing belong to standalone `generate` (`generator.py:1133–1142`); optimizing them would not affect P2.

The production plugin already skips temperature 1 division and prefetches seeded exponential noise (`host_sampler.py:39–52,103–119,149–156,172–184`). Warmed requests need not pay fresh synchronous noise generation every token. The installed native PyTorch top-p implementation sorts the full vocabulary before masking even for k=40 (`vllm/v1/sample/ops/topk_topp_sampler.py:462–503`), a plausible residual CPU cost. Exact filtering dispatch depends on `HAS_TRITON`; record it in a future measurement rather than infer runtime dispatch solely from that implementation. A generator-private replacement is outside this investigation's intervention boundary. Full-vocabulary logprobs are conditional and P2 requests none.

## Why raising the device limit cannot fix P2

Plugin `model_runner.py:4078–4085` explicitly sends non-greedy k=40 to host. Seed 42 under policy `auto` is not by itself the reason. The generator uses common `SamplingGenerator(max_top_k=32)` and rejects larger non-greedy requests (`generator.py:139–150,324–325`).

No supported wider common device implementation was found:

- Raising only the advertised capability routes k=40 into the model rejection.
- Relaxing that guard still clamps k=40 to 32 in `models/common/sampling/generator.py:854–859`; `tt_sampling.py:221–245` additionally clamps to its configured maximum. That changes the request silently.
- Common sampling calls `ttnn.sampling` (`tt_sampling.py:1470–1478`). Compute fixes `nearest32_K=32`, `logk=5`, and `Kt=1` (`ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp:469–503`).
- The writer consumes one candidate tile and two 16-wide faces per user (`ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/writer_interleaved.cpp:139–188`). For user 0, k=32 scans offsets 0..15 and 256..271. Unclamped k=40 extends the latter scan through 279, into the next row, without the missing eight correct candidates.
- TP4 gathers 128 local candidates (`tt_sampling.py:1355–1375`), but the final global cutoff remains 32. Candidate tensor width is not supported sampling k.

A genuine common k=64 implementation requires coordinated formatter, compute/buffer, writer, validation and test work, including RNG/tie behavior. It is not an available configuration switch or minimal model fix. Sending local top32 to host global-top40 is also unsound: all 40 global winners can lie on one shard. The supported wider path here is the existing host compatibility sampler. No canonical sampler replacement is proposed.

## Discriminating experiments — proposed, not performed

Use a later authorized measurement window after live P3 work. Equal seeds do not promise identical host/device token sequences; use recorded common logits or forced common histories where needed to isolate sampling cost from changing model history. Match prompts, output length, request policy, precision, trace mode and active widths. Host phase timers suffice; no profiler is needed. Measure completed tokens and change one factor at a time.

| Candidate | One discriminating control | Interpretation |
|---|---|---|
| Full 32-row FP32 expansion | Identical completed host tensors, current processing versus only `.float()` elision. Check exact output for prefix, sparse/remapped and inactive rows. | Reduced processing time/allocation isolates conversion; little timing change means it is not dominant. |
| Full 32-row host concatenation | With FP32 elision in both arms, compare parallel composer with selected per-shard copies into the same result. | Improvement supports this tier; regression reveals lost parallelism/extra shard overhead. |
| Padded transfer/untilization | Full TILE read versus compact ROW_MAJOR extraction plus read, identical logits and sampler; include extraction and read-event completion. | Only net improvement with exact mapped logits supports compaction. Smaller DMA alone is insufficient. |
| Repeated page-table preparation | Only validated unchanged-snapshot shortcut; retain position writes and readback. Time `_table_plan`/table binding. Include changed future ID, overlap, page boundary, remap and invalid dtype controls. | Lower stable-step CPU time with full validation on changes isolates metadata cost. |
| Repeated position uploads | Separately omit only proven-equal resident updates; preserve token uploads/tables/sampler. Check reset, reposition, remap, inactive-slot controls. | Counters should lose 12/24 uploads on stable steps with identical outputs/state; unchanged latency limits significance. |
| Diagnostic table payloads | Keep tracking and event fields; suppress only unchanged full-table serialization. | Reduced TPOT isolates this cost. Disabling all tracking confounds the test. |
| Required host feedback/native filtering | Same seeded k=20/.95/temperature 1 request through existing host/device routes in a test harness; also compare host k=20 and k=40. Time conversion, filtering/noise, event wait and feedback separately. | Similar host k=20/k=40 costs support a route overhead explanation; a large filtering span locates residual sampler work without authorizing replacement. |

First implementation: generator-owned FP32 elision, then validated table/position deduplication as separate reviewable changes. Consider compact readback only if remaining measurements justify its extra lifetime/state complexity. This report does not claim FP32 elision alone saves the 11 ms needed for the critical 2-user gate.

## Other Potential Issues and rejected explanations

- **Harness log attribution discrepancy:** raw results assign 128 host lines to the first timed greedy run. `run_harness_requests.py:251–273` performs a host acceptance request but does not drain its log cursor before timing greedy. The saved acceptance window has 128 host prefill/decode markers; the timed greedy window says `device`. Those lines are acceptance-log carryover. TPOT is calculated independently and the failure remains valid. Drain the cursor after acceptance in a separate harness correction.
- **Shared sampler copies:** active-row advanced-index copying and native filtering/RNG work belong to the shared plugin/vLLM boundary. They do not justify shared changes before eliminating generator waste. Standalone model sampler clones are irrelevant.
- **Generic device slowdown, numerical drift or broken batching:** other sampled profiles closely match paired greedy timing, and the 2-user host window explicitly shows two active slots. These alternatives fit the route boundary less directly; no new model correctness defect was established.
- **Compilation as the whole explanation:** acceptance already exercises host sampling before timing. Cache/warmup effects are not fully proven, but 127 timed host decodes repeat the identified work. A one-time compile is insufficient as a complete explanation without further evidence.
- **Changing sampler padding or async flags:** the 32-row canvas and host feedback barrier have real contracts. A logical TILE crop does not demonstrate physically smaller readback, and nonblocking submission does not establish completion.

## Claim review and remaining uncertainty

The initial candidates were checked through callers and lowering. Review confirmed full-buffer reads, full 32-row FP32 expansion, full-width table iteration, 12/24 position uploads and recorded repeated table payloads. It rejected double-DMA/network-gather claims, standalone sampler blame, unconditional page-table-device-upload claims, cap-only k=40 support, and wait removal. The 1-user plain host-line anomaly was traced to cursor timing rather than treated as contradictory runtime routing.

Together, fixed 32-row conversion/transfer, active-user-scaled metadata/position/logging work, and required host sampling/feedback explain the qualitative shape: both host cases are slower, with larger absolute overhead at two users. Two timings cannot allocate that difference among causes. Runtime timing and correctness controls are still required to close P2; this report does not mark the gate passed or waive it.

Checks used source inspection, stdlib JSON/log parsing, arithmetic and hashes. Two independent follow-up reviews checked the completed draft against metadata/async code and readback/lowered conversion code; neither found a mandatory correction. No CPU tensor benchmark ran alongside P3. Selected baseline inspection hashes (see the concurrent-edit qualification above):

```text
generator.py       e7ee09fb4979aeeb32a47eacc3f1403732b71f127c5724d5c578a43304e14990
generator_vllm.py  a3157376fdb19524de014d73da9a37e275494715d874de8cc2fcae7af60d4ac8
cache_pool.py      f2b1946460b6db2bd3acf09753792996c04d72f4fd81dce3227d10a79a76e12a
async_decode.py    083f87d99525e45295e05ebb252f6bdb9085185a9ee1004831e307b316eab4f4
model_runner.py    9867a7c6ca200115a1cc0b2dc5c5747ecbd2cebc8eb4486da9a1e79ffcc01482
host_sampler.py    4d788b35c658cc1efc77d43bec7346ac8174957ac018d3c01f275e7856701dc8
runs.json         78b58cfc20ae84f9a431b7eb1489a62ac653a01311b803c464cfb3e6983fb6ae
```
