**AutoDebug investigation — four BFP8 expert layers versus retained trace capacity**

2026-11-16. CPU inspection only. No TTNN import, device operation, hardware reproduction, or source-code edit was performed by this investigation. This session was already isolated; the AutoDebug launcher was not invoked. Only this report was written by this investigation.

Prepared HEAD and failing artifact HEAD both identify `ccb44bb0cb36efb5eb26718a5520dbdf1ec41dc5`. The checkout contains existing model edits and untracked sweep files, so HEAD alone is not a complete source identity. The inspected generator's traceback line numbers match the failure. File fingerprints appear below.

During the final fingerprint check, another writer changed the harness to add `--trace-region-size` (default 0) and `--trace-diagnostics`. The two focus files remained unchanged. The diagnosis below concerns the supplied v3 artifacts and the original harness snapshot; original harness line references are labeled accordingly. The newer controls can support the proposed experiment, but no result from them was evaluated here.

**Leading diagnosis**

The failure is a **valid rejection of dynamic trace storage entering an earlier retained trace's recorded DRAM envelope**. The four upgrades consume another **1,258,291,200 bytes per rank**, while the harness reserves no trace region and B1 retains additional trace shapes lazily. Successful loading and preparation do not establish that those traces can coexist.

The exact error is not an ordinary allocation-OOM, a numeric accuracy failure, or evidence that `mesh_trace.cpp` should relax its protection. It establishes an address-envelope conflict, not a proven exact replay access collision or proof that every possible placement of four upgrades is impossible.

A fixed trace region is a legitimate, protection-preserving intervention at device construction: it separates trace storage from ordinary DRAM before weights, caches, and temporaries are allocated. **Whether a sufficiently large region fits this candidate remains unproved.** Simply preallocating public prefill inputs/outputs is already implemented. Capturing earlier merely moves the admission failure unless placement or the retained footprint actually improves.

**Direct observations**

Sources: [failed v3 JSON](models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/e8_four_hifi2_v3.json), [failure log](models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/logs/e8_four_hifi2_v3.log#L2138), and [passing baseline v3 JSON](models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/inherited_e4_lofi_v3.json). Memory entries below are the JSON `memory_loaded`, `memory_prepared`, and `memory_final` fields, in **bytes per bank**, with eight banks per rank.

| Observation | inherited_e4_lofi_v3 | e8_four_hifi2_v3 |
|---|---:|---:|
| Capacity / configured batch / layers / mesh | 262144 / 1 / 48 / 1×4 | Same |
| Free after loading | 293,825,728 | 136,539,328 |
| Free after preparation | 288,382,976 | 131,090,432 |
| Largest free block after preparation | 287,523,904 | 130,289,728 |
| Free at final snapshot | 208,076,800 | No final snapshot |
| Completed / passed | true / true | false / false |
| Recorded evaluation runs | 8 | 0 |

The failing JSON has `runs=[]`; it fails inside the first readiness `run_prefill_check`, before that runner returns any result. The log prints `PREPARED` at line 1417 and the fatal at line 2138. There is no earlier completed `entry[...]` line. No candidate accuracy or performance result is established.

The original harness explicitly checks `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, opens `trace_region_size=0`, and preserves the resident generator across runners by deferring teardown (original run_datatype_candidate.py:49–68, 96–105; equivalent current [setup](models/autoports/qwen_qwen3_8_flash_next/tests/run_datatype_candidate.py#L52) and [runner ownership](models/autoports/qwen_qwen3_8_flash_next/tests/run_datatype_candidate.py#L130)). Its two repeats in the passing artifact account for eight runs; the failing command uses the default one repeat.

The comparison is **not solely a weight-dtype change**: [e8_four_hifi2.json:15–53](models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/configs/e8_four_hifi2.json#L15) applies expert HiFi2 to all 48 layers and BFP8 gate/up/down weights to layers 0, 1, 46, 47. The baseline uses expert LoFi. Static weight growth is exact, but equal workspace/trace sizes across the policies must not be assumed. Existing smaller candidate configuration files are not completed fit/accuracy evidence.

**Allocation and execution ledger**

| Phase | Source contract | Consequence |
|---|---|---|
| Device construction | Original harness line 60 selects dynamic tracing; current line 63 retains zero as the default. | No future trace-storage area is withheld from ordinary DRAM in the reported run. |
| Generator construction | [generator.py:92–176](models/autoports/qwen_qwen3_8_flash_next/tt/generator.py#L92) allocates full-capacity cache, all six prefill input/word buckets, sample canvas, and shared persistent prefill output. | The reported 262144 is allocated capacity, not the first reference's prompt length. B1 has no factored-decode owner. |
| Preparation | [generator.py:423–469](models/autoports/qwen_qwen3_8_flash_next/tt/generator.py#L423) warms buckets 1,2,4,8,16,32, decode, checkpoints, and sampling, resets, then synchronizes. Sampler [precompile:476](models/common/sampling/generator.py#L476) does not capture. | B1 preparation establishes warm execution and resident storage, not the complete retained trace set. |
| First reference | Readiness concatenates prompt and generated tokens. CPU-only safe tensor loading found one entry with 203 prompt + 100 generated tokens. | Logical full-prefill length is 303, while cache capacity stays 262144. |
| Chunking | [prefill_spans:692–702](models/autoports/qwen_qwen3_8_flash_next/tt/generator.py#L692) yields nine 32-token chunks, then 8,4,2,1. | Unique trace-capture order, if reached, is 32 → 8 → 4 → 2 → 1. Sixteen is warmed but not used in this entry. |
| Each new shape | [_prefill_chunk:488–495](models/autoports/qwen_qwen3_8_flash_next/tt/generator.py#L488) records once, retains the ID, and enqueues replay with `blocking=False`. | Earlier shapes remain live when later shapes capture. “First prefill check” does not mean “first capture.” |
| Capture end | [mesh_device.cpp:1772–1790](tt_metal/distributed/mesh_device.cpp#L1772) finalizes commands, stops high-water tracking, then allocates trace storage and validates against all live traces. | Temporaries may already be allocator-free but their captured addresses still matter. |
| Completion | This readiness path reads logits after each chunk ([generator.py:813–817](models/autoports/qwen_qwen3_8_flash_next/tt/generator.py#L813)); prepare explicitly synchronizes. | Host capture success is not replay completion. The fatal is before the failed trace's upload/replay; previous chunks have readback boundaries. |

The log omits trace IDs, signatures, and per-trace sizes. Therefore the exact failing tail bucket, the older trace producing the maximum, and the operation creating that maximum remain unresolved. Warning counts are not a reliable mapping.

Capture is not side-effect-free: [FDMeshCommandQueue::record_begin/end](tt_metal/distributed/fd_mesh_command_queue.cpp#L1626) switches command recording/bypass state and builds the command stream, while host allocation and cache activity can still occur. High-water collection includes `record_end()`. The candidate's captured commands are not actually run by merely completing the capture body.

**What the validator proves**

Dynamic trace storage is ordinary `BufferType::DRAM` allocated top-down ([mesh_trace.cpp:50–105](tt_metal/distributed/mesh_trace.cpp#L50)). Allocation and deletion marks record the maximum **address plus per-bank allocated extent**, in the same interleaved address coordinates as the trace buffer ([bank_manager.cpp:481, 516, 620](tt_metal/impl/allocator/bank_manager.cpp#L481)). Freeing a transient buffer does not erase its capture maximum.

Each trace stores its capture maximum. [SubDeviceManager:138](tt_metal/impl/sub_device/sub_device_manager.cpp#L138) and [its tracker:175](tt_metal/impl/sub_device/sub_device_manager_tracker.cpp#L175) compute the maximum across all retained trace pools. Release removes the trace's contribution. Capturing a subsequent trace resets the current capture counters, not the older stored maxima.

The error branch at [mesh_trace.cpp:137–144](tt_metal/distributed/mesh_trace.cpp#L137) establishes:

```text
new_trace_base                         = 4,210,685,952
max_retained_trace_high_water          = 4,218,759,104
current_allocation/deletion_high_water <= new_trace_base
retained_high_water - new_trace_base   =     8,073,152 bytes/bank
                                      =     7.6991577 MiB/bank
```

The 8,073,152-byte gap is an address-envelope deficit; multiplying by eight gives a 64,585,216-byte rank-equivalent margin, **not** an exact missing tensor size or a sufficient reservation recommendation.

The checkout's Blackhole DRAM view ends at **4,278,190,080 = 4 GiB − 16 MiB**, from [blackhole_140_arch.yaml:120](tt_metal/soc_descriptors/blackhole_140_arch.yaml#L120), consumed by [l1_banking_allocator.cpp:226](tt_metal/impl/allocator/l1_banking_allocator.cpp#L226). Using this source boundary:

| Address span | Bytes/bank |
|---|---:|
| Usable top minus retained maximum | 59,430,976 |
| Usable top minus new trace base | 67,504,128 |

The second span is not necessarily all trace bytes: it can include gaps/other allocations. The saved runtime report does not explicitly dump the top boundary; the figure is source-derived and consistent with the report's allocatable capacity. Using 2^32 as the usable top would incorrectly add 16 MiB/bank of headroom.

There is a focused in-tree regression for this exact class of failure: [MultipleLiveTracesValidateAgainstMaxHighWaterMark](tests/tt_metal/distributed/test_mesh_trace.cpp#L874) captures trace A with a high envelope, then rejects trace B below A's envelope even though B has no conflicting DRAM activity of its own. It was read, not executed.

High-water protection is separate from unsafe-buffer-ID tracking. `TT_METAL_TRACE_ALLOC_TRACKING` and the program-cache skip flag govern [trace_allocation_tracker.cpp:183–203](tt_metal/impl/allocator/trace_allocation_tracker.cpp#L183); they do not suppress the BankManager marks. A fixed region keeps unsafe-allocation/replay checks enabled. `trace_storage` is exempted because storage is either separated by the fixed allocator or validated by this dynamic guard.

**Fixed-region remedy and its budget**

The smallest intervention boundary is the harness's mesh-open call, not the guard. A nonzero `trace_region_size=T` means a total budget **across the eight banks of each device**, replicated on each rank, not a budget divided by all four ranks.

[AllocatorImpl::init_one_bank_per_channel](tt_metal/impl/allocator/allocator.cpp#L52) creates disjoint intervals:

```text
N = 8 banks
D = usable bank top
R = round_up(ceil(T / N), 8192)       # T first receives DRAM alignment
ordinary DRAM = [unreserved_base, D - R)
trace storage = [D - R, D)
```

All ordinary warmup/capture allocations must then fit below `D-R`. This removes the overlap by construction rather than disabling protection. It may fail earlier with an ordinary DRAM OOM.

Size the region from the **complete concurrently live trace inventory**, not the failed buffer or one high-water gap. For trace i:

```text
P_i = selected trace page size
S_i = round_up(raw_command_bytes_i, P_i)
E_i = ceil(S_i / (N * P_i)) * P_i     # physical per-bank extent

required: T >= sum(S_i)
          R >= sum(E_i)
sufficient compact-set choice:
          T = N * round_up(max_live_set(sum(E_i)), 8192)
```

The page selector is [dispatch.cpp:211](tt_metal/impl/trace/dispatch.cpp#L211); bank spreading is [bank_manager.cpp:481](tt_metal/impl/allocator/bank_manager.cpp#L481) and [allocator.cpp:784](tt_metal/impl/allocator/allocator.cpp#L784). Logical trace accounting sums S_i, while physical occupancy includes each trace's bank/page tail. Add explicit measured allowance for additional signatures and fragmentation. Do not round the combined logical size only once and assume all per-trace tails disappear.

With the **unchanged prepared resident inventory**, the candidate has only **1,048,723,456 bytes/rank (1000.140625 MiB)** left for a reserved region *and subsequent ordinary allocations*. That is an optimistic upper bound, not a safe region setting. A 1-GiB region already exceeds it by **25,018,368 bytes/rank**; 2 GiB is plainly incompatible. The largest existing ordinary free block is smaller still (130,289,728 bytes/bank), although a construction-time reservation can change placement and fragmentation.

The current inherited baseline's prepared→final net growth is 642,449,408 bytes/rank. Its counters show five prefill captures and two more captures in excluded teacher-forcing warmup, with no releases: a seven-trace subset that does not include the 16-token prefill shape. That snapshot difference is neither the full candidate trace budget nor its transient peak. It does not justify selecting a region size without measurement.

**What “preallocation” can and cannot change**

The ordinary public input/output remedy is already present at [generator.py:111–131](models/autoports/qwen_qwen3_8_flash_next/tt/generator.py#L111). Capacity-specific state and index storage are constructed in [multichip_decoder.py:1010–1082](models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py#L1010), and QSA geometry is prepared in [qsa_chunked.py:44](models/autoports/qwen_qwen3_8_flash_next/tt/qsa_chunked.py#L44). Request [_refresh:263](models/autoports/qwen_qwen3_8_flash_next/tt/generator.py#L263) copies into existing device destinations.

Precapturing all six prefill shapes before serving provides earlier admission and a complete footprint measurement. It does not inherently reduce trace bytes or peak temporaries. A different capture order can change placement; it is an allocation-order experiment until every trace is replayed safely. Arbitrary dummy allocations, deleting Python owners, or releasing traces are not durable address reservations.

Preallocating internal activations into bounded, stable workspaces could alter the footprint, but is a larger lifetime/aliasing refactor and can increase resident memory. No specific missing internal workspace has been identified here. There is no source basis to promise that this will recover the needed margin.

**Minimal discriminating experiment — proposed, not executed**

Keep all 48 layers, TP4, B1, capacity262144, the exact four-layer BFP8/global HiFi2 policy, all other dtype choices, tracking=1, skip-program-cache=0, and the same references. Use a fresh process/device lifetime for each configuration.

1. **Measure the existing failing boundary.** Log each shape/trace ID before capture and, at the C++ populate boundary, raw/padded command bytes, page size, per-bank extent, base/end, current allocation/deletion marks, retained trace maxima, and total trace bytes. Record DRAM allocated/free/largest-block values and program-cache counts before/after capture. This identifies the failing shape and measures net persistent growth after subtracting trace storage. To identify the high-water producer or make a peak-capacity claim, also record allocation ID/context/size when the address maximum advances and track maximum simultaneously allocated ordinary bytes during capture, excluding trace storage. Boundary snapshots and address maxima alone cannot measure that byte peak. Start with the 303-token reference; do not run an expensive accuracy matrix merely to rediscover the same fatal.

2. **Obtain a complete size inventory if the cumulative run cannot.** After identical preparation, calibrate each of six prefill bodies, B1 decode, and the sampling signatures required by this harness with only one live trace at a time. Synchronize any outstanding replay before release, and reset state outside capture. Release is permitted only in this calibration. Keep command/cache signatures comparable; separate processes are cleaner if calibration changes cache state. A single-capture own-high-water failure is useful evidence too. Use exact metadata rather than assuming prepared→final deltas equal trace bytes.

3. **Repeat the unchanged first-prefill sequence with the smallest measured adequate fixed region.** The original harness hardcoded zero; the externally added `--trace-region-size` now exposes the required control. No such source edit was made here. Compute T using the per-bank formula above. A necessary condition for unchanged ordinary allocation requirements is R + W <= prepared_free_per_bank, where W is the measured peak additional ordinary allocation beyond preparation, excluding trace storage. Do not infer W from boundary snapshots or the address maximum. If a measured lower bound violates this condition, reject that proposed reservation for that schedule; otherwise the fixed-region run must still establish placement and fragmentation fit. An arbitrarily large 1–2-GiB region is not a useful control. The new `--trace-diagnostics` records signatures and boundary memory/TRACE occupancy; it does not yet provide allocation-owner/address-high-water events or a transient byte peak.

Interpret the result precisely:

| Outcome | Conclusion |
|---|---|
| Fixed region captures **and replays** the previously failing sequence | Placement/reservation resolves this boundary; proceed to full retention and accuracy. |
| “Creating trace buffers … only … allocated” | Logical trace budget too small; no claim about numerical correctness or ordinary DRAM yet. |
| TRACE allocator OOM | Per-bank region capacity/fragmentation insufficient. |
| Ordinary DRAM OOM during preparation/capture | That reservation leaves insufficient ordinary space for this allocation schedule. |
| Unsafe persistent-allocation/replay rejection | Separate lifetime problem remains; retain all protections and inspect the reported owners. |

After the short control passes, require all six prefill signatures, B1 decode, applicable sampling traces, repeated replay with host readbacks, and both unchanged readiness references while retaining the complete production set. Record every relevant rank's footprint, accuracy, and zero measured captures/releases in the teacher-forcing window. Changing the reservation can be a valid setup fix; capture eviction or recapture inside the measured workload changes the tested regime.

No Python binding for `get_trace_buffers_size()` was found. C++ per-trace metadata is available. Existing Python `get_memory_view(mesh, BufferType.TRACE)` measures fixed-region per-bank occupancy; dynamic traces instead share DRAM ([device bindings:602](ttnn/cpp/ttnn-nanobind/device.cpp#L602)). Memory block-table addresses are allocator-relative: add the DRAM base for dynamic DRAM tables and D−R for fixed TRACE tables. Do not obtain the latter through Python `get_allocator_base_address(..., TRACE)`: this checkout's [binding:642–647](ttnn/cpp/ttnn-nanobind/device.cpp#L642) maps TRACE to L1. That inspection-API discrepancy is not the cause of the reported failure.

**Defensible precision/memory limit if four upgrades remain infeasible**

[static_weight_delta:119–135](models/autoports/qwen_qwen3_8_flash_next/tt/precision.py#L119) gives an exact cost for upgrading all three expert matrices of one layer:

```text
128 local experts × 3 matrices × 1600 tiles × (1088 − 576 bytes/tile)
= 314,572,800 bytes/rank/layer = 300 MiB/rank = 37.5 MiB/bank
```

The observed loaded-memory difference matches four such upgrades exactly. Preparation adds another 49,152 bytes/rank relative to the baseline beyond the weight difference.

There are three distinct limits:

- **Validated limit today:** the inherited zero-upgrade baseline passes; no smaller upgraded policy has a completed full-model result in the inspected sweep. Four is rejected by the current dynamic placement. Do not report two or three as proven to fit.
- **Next experimental cap if the unchanged four-upgrade fixed-region control fails:** at most **three complete layer upgrades, 943,718,400 bytes/rank (900 MiB) of extra weights**, tested as a separately named policy. Removing one upgrade recovers 39,321,600 bytes/bank, greater than the observed 8,073,152-byte envelope gap. Under unchanged other occupancy it leaves 170,412,032 bytes/bank prepared free. This is a rational next admission test, not a mathematical guarantee; trace sizes, allocator placement, full live inventory, and accuracy must be measured again. Preserve global HiFi2 for a clean memory contrast. The existing two-edge HiFi2 config also changes fidelity scope, so it is not that control.
- **If the existing conservative 2-GiB runtime-reserve contract must remain unchanged:** it admits **zero complete upgrades** at full capacity. The shared-pool planner's full B1 budget is 28,610,184,192 resident bytes + 3,222,011,904 BF16 KV bytes for 8194 physical blocks (including its reservations) + 2,147,483,648 reserve against 34,178,731,008 allocatable. Only 199,051,264 bytes remain, less than one 314,572,800-byte upgrade. This planner is not invoked as admission for this standalone private-cache constructor; it is a conservative policy bound, not proof that hardware cannot run a smaller candidate. See [cache_budget.py:18–27, 183–207](models/autoports/qwen_qwen3_8_flash_next/tt/cache_budget.py#L183).

A general candidate budget must include weights, full-capacity state/KV, persistent outputs/program storage, physical live trace storage, **peak additional ordinary workspace**, alignment/fragmentation, and a stated margin. Retained final free bytes alone do not supply the peak. Reducing context, KV dtype, layer count, fidelity, or protection would define another experiment and cannot be credited as fixing this candidate.

**Competing explanations and adjudication**

| Candidate explanation | Evidence / necessary conditions | Disposition |
|---|---|---|
| Dynamic trace placement crosses an older captured envelope | Exact fatal branch; retained-shape lifecycle; exact increased weight occupancy; analogous regression test | Leading diagnosis, high confidence. |
| Four upgrades are impossible under any safe allocation | Needs candidate trace inventory and ordinary peak under a deliberate reservation/placement | Unproved. |
| Public inputs/output were allocated too late | Contradicted by constructor and warmup paths | Rejected for this failure. |
| Later program-cache allocations drive the footprint | Possible because global fidelity differs; current log lacks owner/ID/size mapping | Secondary measurement target; not an established root cause. |
| Trace guard is a false positive because total free memory remains | Free temporaries can still be referenced by replay; high-water is intentionally conservative | No justification to weaken it. Exact access overlap remains unmeasured. |
| Numerical BFP8/HiFi2 bug | No scored output or kernel fault at this boundary | Does not explain this host-side allocation rejection. |

**Other potential issues**

1. **Failed capture ID is not explicitly released by the generator.** [_record_trace:521–531](models/autoports/qwen_qwen3_8_flash_next/tt/generator.py#L521) protects exceptions from the body, but final `end_trace_capture` is outside the try. This failure occurs before the trace ID is inserted into `prefill_traces`. [mesh_device.cpp:1755–1757](tt_metal/distributed/mesh_device.cpp#L1755) intentionally registers the trace even on exception, and [close:1188](models/autoports/qwen_qwen3_8_flash_next/tt/generator.py#L1188) only releases known dictionary entries. Thus the failed trace can remain allocated/registered until broader mesh teardown. This does not cause the first failure, but makes retrying within the same generator a poor experiment. It is a separate cleanup fix opportunity.

2. **Historical memory prose should not become a hard peak limit.** The 1,085,865,984-byte value in [runtime_forecast:68](models/autoports/qwen_qwen3_8_flash_next/tt/cache_budget.py#L68) is older prepared→final net growth with a different trace inventory. Applying it mechanically to this candidate's free space produces a 37,142,528-byte/rank deficit, but does not prove a physical impossibility. The historical [initial precision audit](models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/initial_precision_capacity_audit.md#L25) also overstates post-finiteness free space by 65,536 bytes/rank compared with the actual [acceptance JSON](models/autoports/qwen_qwen3_8_flash_next/doc/full_model/acceptance_max_context_final_v1.json).

**Review and reproducibility**

After drafting, the claims were checked against the exact fatal branch, the begin/end/release lifecycle, per-bank versus per-rank sizing, the B1 preparation path, the source-derived 303-token shape sequence, and the policy's global fidelity scope. No exact failing bucket, exact colliding tensor, or guaranteed fixed-region fit is claimed. Independent xhigh inspections covered the allocator, generator lifecycle, and memory artifacts; only CPU inspection was used.

Cheap local verification included JSON parsing/arithmetic, source searches, and executing only the extracted pure-Python `prefill_spans` AST (not importing the generator). The reference was read on CPU with `torch.load(weights_only=True, map_location="cpu")`. No hardware-dependent tests were run.

Inspected SHA-256:

```text
generator.py:
58d7133c8b8f862f6ac14ebcde92f49eaca36663fb2ea1cbc23c11dd6e995eda
tt_metal/distributed/mesh_trace.cpp:
52ae0d5f171c6d7d9017a0da5c77c522e4bcfa298e4bab28e9feff03854d7c90
tests/run_datatype_candidate.py, original inspected snapshot:
7e37d6f245ca7cea8bf312dd8a135a6c64a754577e3b445520c8e4ac7fdd27ec
tests/run_datatype_candidate.py, externally updated snapshot at final review:
799e390674533c4163a84802cfae32a05b87bae68deb92bd04bf4686cdc6ea9f
```
