# AutoDebug: candidate 1 state-copy traces and async transition mismatch

## Finding

**Both eager and traced scratch copies fail with watcher in the available controls; the candidate passes the same probe without watcher.** The parent-produced eager-copy watcher control also first differs at sequence 8: A consumes resident position 515 and returns `91920` with queued reads versus `163218` with immediate reads. B returns `171092` in both. The new scratch-copy traces are not required to produce this first mismatch. Inspection found no concrete dangling-buffer, capture-time state mutation, copy precision, or async host-output ownership defect on this path. **The underlying cause remains unresolved; the controls suggest watcher-sensitive execution but do not prove a watcher implementation defect.**

This was an inspection-only investigation at checkout `4e5a1ddd3e4ede3a313d293025257c74f4ee8b3f`, comparing candidate source with `86130b35`. No model source was edited, no hardware commands or reproductions were run, and the AutoDebug launcher was not invoked. Three isolated xhigh investigators audited copy lifetime/lowering, async ownership, and watcher/tracker behavior. Only this report was written by this investigation. Paths beginning `tt/`, `tests/`, or `doc/` below are relative to `models/autoports/qwen_qwen3_8_flash_next`; other source paths are relative to checkout root.

## Direct observations and evidence limits

| Artifact | What it establishes | What it does not establish |
| --- | --- | --- |
| `doc/optimized_vllm/copy_trace_async_watcher.json` and corresponding log | C1024, B2, pool80, layers 0/1/3; first `C_singleton` timing comparison fails; both individual histories finish; watcher enabled | First wrong intermediate, matching pre-sequence-8 device state, or cause of corruption |
| `doc/optimized_vllm/eager_copy_async_watcher_control.json` and corresponding log | Parent control with copy preparation disabled also fails at the same first step/value | Exact historical checkout/binary control or proof that traced copies cannot affect later trajectories |
| `doc/optimized_vllm/traced_copy_async_control.json` and corresponding log | Parent's watcher-unset traced-copy control passes both histories, exact tokens/ledger/final state/RNG; all four copy captures present | Repeated stability, eager-without-watcher result, or which watcher effect matters |
| `doc/vllm_integration/adapter_async_transitions_v2.json` | Same expanded probe historically passed both histories with exact tokens, ledger, and final state | A revision- and watcher-matched control for `86130b35` |
| `doc/vllm_integration/adapter_async_table_snapshot_v1.json` | Later Stage 9 C128/pool16 adapter diagnostic passed, including two queued host snapshots | Long resident-feedback execution at positions 515/130 |

The candidate command is `validate_candidate.sh:4-20`: serving environment, devices 0–3, accepted P300x2 descriptor, tracking 1, skip-program-cache 0, OMP8, layers 0/1/3, profiler unset, with `TT_METAL_WATCHER=10` on the transition probe. Its log records watcher startup at line 17 and an ordinary Python assertion at lines 634–642; device closure completes at lines 645–649. It reports no tracker RuntimeError, watcher assertion, or NoC sanitizer error. Absence of those errors is not proof of correct data.

The probe blob is identical at the two requested revisions (`b4b0f3588b7ea700e5719841c6e615b744fcfcdd`). Only `tt/generator.py` and `tt/slot_state.py` change production model code in candidate 1; no TTNN/C++ kernel source changes between `86130b35` and `4e5a1ddd`.

The stronger historical transitions-v2 pass predates the later `e9c3b578` decode table-snapshot optimization. See `doc/vllm_integration/work_log.md:112-124,166-172`. Its log has no watcher startup. Consequently, “same test passed previously” is supported, while “exact `86130b35` plus watcher passed previously” is not. The later table-snapshot artifact uses `tests/probe_vllm_async.py`, context128/pool16, and two authoritative input refreshes in its queued pair (`probe_vllm_async.py:260-318`); it is a narrower contrast.

The parent control appeared during investigation. Its wrapper, `tests/probe_state_copy_control.py:21-28`, replaces `SlotStateMover.prepare_copy_traces` with a no-op and runs the unchanged transition probe. The control log has 15 preparation captures instead of candidate's 19 and no `state_copy` signatures; JSON records `state_copy_control=prior_eager_transfers`. This is evidence supplied by the parent, not a reproduction performed here. It directly removes the candidate's four captures and replay callbacks from that execution.

A second parent control completed while the report was being reviewed: `traced_copy_async_control.json` is `passed:true`, identifies `traced_transfers`, and records exact tokens, ledger, final state and RNG for both `C_singleton` and `A_extra_singleton`. All four runs' tokens match historical transitions-v2. Its log has the four state-copy captures at lines198-201, preparation count19, no watcher startup, and `ADAPTER_ASYNC_TRANSITIONS_PASS` at line971. `copy_control_comparison.json` labels this the without-watcher run; this label is consistent with its process log. The JSON does not contain a complete environment/binary manifest, so provenance remains bounded by those artifacts.

| Current observed control | Watcher enabled | Watcher unset |
| --- | --- | --- |
| Traced scratch copies | FAIL, original candidate | PASS, both histories and exact final-state oracle |
| Eager scratch copies | FAIL, same earliest mismatch | Not present in reviewed evidence |

This is stronger than the historical contrast: the failure is confined to watcher-enabled runs in the current observed cells. It remains a small, unrepeated control set; watcher may expose a shared latent race/state defect rather than own the defect.

### Exact first divergence

| Sequence | Action | Resident positions before decode | Immediate output | Queued output |
| --- | --- | --- | --- | --- |
| 2 | Final A prefill, end513 | — | A `154378` | A `154378` |
| 3 | Initial A singleton, authoritative reload | `[513,0]` | `[72271,-1]` | `[72271,-1]` |
| 4–6 | B prefill chunks, preserving A | — | Final B `69495` | Final B `69495` |
| 7 | Admit B, authoritative reload | `[514,129]` | `[224388,47749]` | `[224388,47749]` |
| 8 | First joint resident replay | `[515,130]` | `[163218,171092]` | `[91920,171092]` |

This table is derived from explicitly sequence-tagged ledger entries, not from pairing unindexed log samples. A's token index 3 is zero-based and includes its final-prefill sample. In the candidate, all 37 immediate A tokens, 41 B tokens, and 4 C tokens match the historical transitions-v2 `C_singleton` result. Queued A differs in 34 of 37 positions; queued B/C match exactly. In the eager watcher control, the first mismatch is identical, but later A trajectories are not identical to the candidate: eager immediate A first differs from the historical reference at token9, and eager queued A differs from candidate queued A starting at token4. Do not describe the two entire failing runs as identical.

Both candidate runs end at host positions `[169,20]`, with final owners B/C. Their `passed:true` fields mean each scripted run reached its end; the enclosing report is `passed:false`. The token assertion at `probe_vllm_async_transitions.py:348` stops execution before exact ledger/final-state comparisons and before `A_extra_singleton`. Thus the latter history is **unrun**, and exact cross-mode final state is **not checked** in this failed artifact. Even a later final-state pass would no longer inspect A's retired state.

Inputs and ownership agree through the first mismatch. Later, sequence39's authoritative A token differs because it is taken from each run's already-diverged output. Therefore the complete ledgers do not remain input-identical after divergence. This is expected autoregressive propagation, not evidence of an independent fixture-input error at sequence8.

## Completion and ownership ledger

`probe_vllm_async_transitions.py:125-211,265-280` constructs this first pair:

1. Before sequence7, `decode(reload=True)` calls `flush()`. The final B prefill has also completed synchronous token readback and survivor checks.
2. Sequence7 reloads authoritative tokens `[72271,69495]`, positions `[514,129]`, and deterministic two-row sampling parameters. Model segments and sampling enqueue on CQ0. Host cache positions advance immediately; these Python integers are scheduler metadata, not completion evidence.
3. Immediate mode uses `.cpu(blocking=True)` and formats sequence7 before submitting sequence8. Queued mode immediately enqueues `.cpu(blocking=False)` plus a CQ0 host event for sequence7, retains that host object, and submits sequence8 without waiting.
4. Sequence8 uses resident feedback and device position. The submitted host values `[1234,1234]` and `[777,777]` are deliberately stale. `reload_inputs=False`, no remap, no page growth, and no state-copy operation occurs on this step (`generator.py:1010-1032,1053-1103`). The unchanged active set is `(0,1)`; both steps use the same width-two decode plan.
5. Queued mode now has two reads, waits for **every** returned event, formats in reverse order, repeats formatting to check stability, then commits IDs in submission order. It does not format incomplete host data.

The observable timing difference is therefore a wait between the first joint step and its first resident successor. This does not prove that the underlying wrong state was created there: equal token IDs at sequence7 do not prove equal logits, cache state, or raw index history. The second run also reuses captures and follows a prior run's reset history.

### Async snapshots: no demonstrated ownership violation

`DecodeSubmission` is frozen and carries tuple slot/active-row metadata (`generator.py:40-45,1112-1119`). The device tensor is intentionally reused; it is **not** itself a snapshot. `read_decode_output` immediately creates a host snapshot with `.cpu(blocking=not async_read)` and records an event on CQ0 (`generator.py:1264-1273`). Formatting reads that host object, not the later live device tensor (`1275-1292`).

Lowering supports the ownership contract:

- `ttnn/core/tensor/tensor_ops.cpp:233-255` routes CPU conversion to `enqueue_read_tensor`.
- `tt_metal/impl/tensor/tensor_apis.cpp:55-73` allocates a fresh distributed host buffer and fresh per-rank storage for each read.
- `tt_metal/distributed/mesh_command_queue_base.cpp:495-529` pins each async destination until the reader completes its memcpy.
- The reader processes read descriptors before subsequent event descriptors (`tt_metal/distributed/fd_mesh_command_queue.cpp:1074-1084,1132-1145`); `tt_metal/distributed/distributed.cpp:141-164` waits on completed events per rank.

The probe retains both hosts through those waits. No late-read alias, freed host destination, mutable submission metadata, or premature formatting was found. This is a source audit, not blanket proof that the lower transport cannot fail. A simple overwrite by the later output also fails to explain why sequence7 remains correct while only sequence8's A value changes. A device-side snapshot tied to sequence8 would still be required to conclusively distinguish model/sampler output from read transport.

## Candidate copy traces: adjudicated findings

The candidate adds two scratch-direction traces per slot after global warmup/reset/synchronization and before factored model capture (`generator.py:425-487`). Replay chooses the trace by source/destination cache identity; ordinary slot-to-slot moves remain eager (`slot_state.py:105-149`). The generator owns and releases IDs, then clears callbacks (`generator.py:1301-1317`).

| Suspect | Code-level adjudication |
| --- | --- |
| Lambda captures all directions as the last slot | False: lambda default arguments freeze each source/destination (`slot_state.py:127-135`). |
| Capture copies stale scratch into live slots | Not on this warmed path: `_record_trace` records without replay; capture enables sysmem bypass and workload recording returns before dispatch (`tt_metal/distributed/fd_mesh_command_queue.cpp:443-456,1626-1640`). |
| Cached copy programs retarget older traces | Runtime arguments are deep-copied into each trace node (`tt_metal/impl/program/dispatch.cpp:3251-3261`). |
| FP32 recurrent state is converted through BF16 | Matching-dtype/memory copy selects raw data movement; a conversion compute kernel is created only for differing dtypes (`ttnn/cpp/ttnn/operations/data_movement/copy/device/copy_device_operation.cpp:17-30,66-74`; `copy_same_memory_config_program_factory.cpp:77,204-222`). |
| New output allocations invalidate fixed addresses | Copy is passed the existing destination; output creation returns it (`copy_device_operation.cpp:192-199`). Caches/scratch precede capture and remain owned. |
| Shared KV is copied between requests | `_state_tensors` explicitly excludes shared K/V; it includes private raw/compressed index history, position, and page table (`slot_state.py:48-63`). |
| Replay uses released trace IDs | Close clears callbacks with trace maps. No close occurs before the observed failure. |
| Nonblocking replay allows ordinary copy/model overlap | All explicit replays use CQ0; the default queue here is also CQ0. Trace entry waits for prior worker completion (`tt_metal/impl/trace/dispatch.cpp:104-142`), and post-trace config state is protected (`186-206`). No queue override was found. |

For equal FP32/BF16 tiled operands the raw copy readers/writers include NoC barriers (`.../copy/device/kernels/reader_unary_start_id.cpp:36-39`, `writer_unary_start_id.cpp:38-41`). Row-major integer position/table copies similarly use raw sticks with barriers (`ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp:32-35`, corresponding writer `37-40`). No copy precision-loss chain was found.

The prefill survivor checks read A's conv/recurrent/PLE/token history, QSA position/table/raw history/compressed keys/owned K/V pages, feedback, and RNG before and after each B prefill (`probe_vllm_async_transitions.py:213-263`; snapshot fields in `probe_vllm_async.py:38-51`). They passed within each run. They do **not** compare A between runs, but do contradict a straightforward “B prefill copied over A” explanation.

Four extra traces do consume trace-storage allocations and shift trace IDs/addresses; this is a real candidate effect, not grounds to claim that they cannot change timing or later trajectories. Nevertheless, the eager watcher control removes both captures and copy replays and preserves the earliest failure. **Do not fix this failure by changing copy lifetime without new contradictory evidence.**

## Watcher and allocation tracking

`TT_METAL_WATCHER=10` means a **10-second** host polling interval, not 10ms (`tt_metal/llrt/rtoptions.cpp:1467-1474`). Enabling watcher also changes compiled device code (`tt_metal/jit_build/build.cpp:419-429`); it is not simply extra host logging. For example, NoC sanitization inserts command-buffer readiness waits before inspecting address-state registers (`tt_metal/hw/inc/internal/debug/sanitize.h:1016-1039`). Mailbox writes, kernel layout, and host polling can perturb timing. None of those facts identifies a defective instruction or proves watcher caused the token mismatch.

The process log only identifies `generated/watcher/watcher.log`; it does not archive that file. Watcher normally opens this shared path with `"w"` (`watcher_server.cpp:293-306`). At inspection it had already been replaced during the later eager control. Do not use its current contents to certify the candidate's watcher status.

Tracking1/skip-program-cache0 **includes** program-cache allocations in tracking; it does not disable program caching. Public trace execution calls allocation verification before replay (`ttnn/ttnn/__init__.py:137-146`, `tools/trace_allocation_tracker.py:130-183`). The positive control confirms that a deliberate live allocation was rejected and replay after deallocation worked in a separate process. Tracking does not validate contents of existing persistent buffers, math, completion protocols, or correctness of caller metadata; trace storage and explicitly exempt allocation contexts have different accounting (`tt_metal/impl/allocator/trace_allocation_tracker.cpp:183-203`). There is no reported tracker rejection in the candidate.

No new model-local corruptible-allocation scope or acknowledgment was introduced. Qwen sampling uses its own `_record_trace` and `_run_sampling` (`generator.py:543-557,572-586`), so the generic sampler's separate capture helper is not evidence of suppression on this path. Capture still performs host/cache/trace-storage bookkeeping; “not executing model kernels during capture” does not mean side-effect-free in every respect.

## Boundary-specific lead, not a diagnosed bug

A's first changed step closes a compressed-index group and activates another packed scorer worker. The actual production path is `MeshQSA._selected_ids` → `causal_indexer_scores(..., packed_decode=True)` → packed Q1 reader/compute/writer → fused selector (`multichip_decoder.py:1160-1165,1225-1299`; `causal_indexer_scores.py:85-111,182-198`). At C1024, compressed keys have 256 rows, eight tiles, and eight scorer cores. Q=1; query/keys are BF16 tiled DRAM and positions FP32 tiled DRAM.

| Consumed position | Compressed write index | Complete groups | Live scorer tiles/cores | Uncompressed tail |
| --- | --- | --- | --- | --- |
| A514, last matching joint step | skip (`-1`) | 128 | 0–3 | 512,513,514 |
| A515, first differing step | 128 | 129 | 0–4 | none |
| B130, same differing submission | skip (`-1`) | 32 | 0 | 128,129,130 |

These are derived from `(position+1)//4` and the actual `write_groups` calculation (`multichip_decoder.py:1231-1235,1272-1285`). They make the group-128 update and activation of scorer core4 within this request history useful measurement points. This is not necessarily that core's first live invocation in the process: queued mode follows the complete immediate run. They do **not** prove that this subsystem diverged first; earlier GDN/PLE/MoE state or an existing tiny numeric difference can remain invisible to token equality until later.

The local queue ledger is coherent. Per live key tile, the packed reader produces four key tiles (CB3) and sixteen broadcast tiles (CB2); compute consumes those, retains the key block across four heads, and produces four output tiles (CB4); writer consumes all four. Every one of eight cores writes its complete output tile, zeroing invalid cells, including non-live cores. At complete128 there are four such live jobs; at129 five; at32 one. No boundary pop/push disagreement or partial output ownership race was found in `tt/kernels/causal_score_packed_q1_{reader,compute,writer}.cpp`. The selector reads four versus five score tiles at128/129 groups, stays below its512-item heap capacity, and writes all2080 IDs (`fused_decoder.py:1262-1314`). All complete groups fit below the top512 cutoff.

The compute path explicitly initializes/undoes row broadcast, calls `copy_init(3)` before key copies, initializes each MUL/ADD, then initializes the SFPU reduction. A fresh low-level audit did not identify a missing format/mode transition on the selected BF16-input/FP32-output path. Accordingly, “newly live core has bad state” remains an unproved hypothesis, not a headline finding.

## Causal assessment

The supported chain ends at a precisely localized observation: matched authoritative tokens and ownership at sequence7 → same width-two resident plan with a different host wait boundary → different completed **host** A output at sequence8. Later reported A IDs and sequence39's authoritative input also diverge; B/C stay equal. The missing link is **which device state, intermediate, sampler result, or read transaction first differs, and why**. The eager control excludes new copy traces as necessary for the earliest failure; it does not supply the missing link.

Two downstream chains must remain separate. If the device sampler actually produces a different A ID, that persistent `decode_tokens` value is consumed by the next resident model step, naturally propagating an autoregressive difference. If only host transport/formatting is wrong, the incorrect host ID is not fed back during resident steps; it enters model execution only at a later authoritative reload such as sequence39. The existing host-output ledger alone does not decide between those chains.

| Candidate explanation | Source contract violation found? | Matches evidence / missing condition | Smallest discriminating intervention |
| --- | --- | --- | --- |
| New scratch-copy lifetime/precision bug | No | Eager control reproduces earliest failure; no copy runs at7→8 | Already remove copy preparation/replay; retain as failed control |
| Watcher-sensitive shared model/runtime behavior | No specific defect found | Both current watcher executions fail; current traced/non-watcher control passes both histories | Complete and repeat current eager/traced × watcher unset/10 |
| Cold/warm capture or cross-run reset history | No | Immediate first creates4 lazy traces; queued reuses them; same asymmetry existed in historical pass | Equal preparation; independent fresh generators; reverse/repeat order |
| Async host snapshot reuse | No | Fresh host buffers, pins/events and correct seq7 oppose simple overwrite | Sequence-tagged device snapshots before reuse, compare to DMA |
| Packed scorer/group update boundary | No | Predicts A515/B130 distinction; not proved earliest divergence | Compare group128/raw history/scores/IDs around514/515 under matched prefix |
| Page-snapshot optimization | No | Present after older transitions-v2; seq8 has no provided table and avoids new branch | Exact86130 runtime control; disable optimization only in a diagnostic if needed |

## Discriminating experiments for the parent

These are follow-ups, not experiments performed by AutoDebug. Keep exact token/state equality and allocation tracking1/skip0.

1. **Finish and repeat the current-source 2×2 matrix:** eager versus traced state copies, watcher unset versus10. Three cells are now available above; eager without watcher remains missing. Run the unchanged transition probe with identical model, prompts, environment and trace setup. Repeat in fresh processes, recording first divergent sequence/value. The single traced/non-watcher pass supports a watcher interaction but does not distinguish host polling from device instrumentation. An exact `86130b35`+watcher run is the cleanest requested revision contrast; the no-op wrapper isolates copy setup without changing model arithmetic but is not a complete binary provenance record.
2. **Remove order/capture history as a variable:** prepare the same lazy prefill/sampling traces before both columns, then fully reinitialize request state, or use fresh identically prepared generators for each mode. Test immediate→immediate and queued→queued repeats, and reverse queued→immediate. Preserve actual sampling-reset, page-table and reload flags. These distinguish read timing from “second execution in one process.”
3. **Separate async DMA from submission overlap at7→8:** after sequence7, compare (a) async read plus event wait before8, (b) blocking read, and (c) submit8 without a sequence7 read, using separate device-owned snapshots to retain both outputs. Change one boundary at a time. A global synchronization that masks the failure localizes a dependency but is not itself a diagnosed fix.
4. **Find the first state difference:** in a separate run, snapshot A and B immediately after final prefill and after sequence7, then compare both columns exactly, with the same added waits. If those match, isolate sequence8 under the same known input prefix and capture intermediate state/output after each real layer. For QSA inspect device position, raw_history, new compressed key128, causal dots/scores, selected IDs/KV, and SDPA output; also retain earlier GDN/PLE/MoE outputs so a QSA coincidence cannot hide an upstream divergence. Snapshot or clone on CQ0 before tensors are overwritten and associate every artifact with submission/layer/rank. Retain the original autoregressive oracle alongside this forced-prefix diagnostic.
5. **If watcher remains discriminating:** leave device instrumentation enabled and greatly lengthen polling interval; then vary specific watcher features one at a time. Preserve the per-run watcher file before another process truncates it, plus source and loaded-binary hashes. This distinguishes polling effects from compiled instrumentation/code-layout effects without declaring either responsible prematurely.
6. **Only if later candidate/eager trajectories still matter:** capture all four copy traces but clear only replay callbacks after preparation, so eager copies run while trace storage remains allocated. Compare to no captures and normal replay. This distinguishes capture/storage effects from executing the copy traces.

Do not infer a watcher pass for its unexecuted `A_extra_singleton` history, replace exact equality with PCC/tolerance, or promote CPU closure mocks to hardware trace-lifetime proof. The non-watcher control did execute and pass that history. No specific production change is justified by the present inspection.

## Evidence fingerprints and report review

SHA256 at inspection:

```text
9cebd1f07757bcdc2bff5e7988f066f6b1fbcfb15740bbe7a9c4fc4e6f2385fc  copy_trace_async_watcher.json
5df24bddb6f489b9ff3d8af85b6bb0e2e4c6a25f7bf20e8b8acd59b4ec1c124b  logs/copy_trace_async_watcher.log
59db4a6384fe2736a1f88a2ed4a831dfcc6b5dead07891f236e4024a4dfaa2bb  eager_copy_async_watcher_control.json
2adb2c552e9a548be27e66f4ed358b326e67535bae5eaed5dec0a126b47ea3bc  logs/eager_copy_async_watcher_control.log
fe4962fe265e7e3a243b949487e87ccdfde0a0d15c6ad74e0e374137125a2b92  traced_copy_async_control.json
d23be2801984e2970d4a2f48f8da5560f40a7eb3b31fcfd94e753426f9762e45  logs/traced_copy_async_control.log
f896c3b1895d0f6b81759daf7abb5154e3a612101bcf19d64dbac214eb1d5f70  ../vllm_integration/adapter_async_transitions_v2.json
dfbabc8463c56f746bbefa3ff0d441bec544f11d3e9a52384c612e08089e75bd  ../vllm_integration/adapter_async_table_snapshot_v1.json
```

The first draft was reviewed against source and evidence again, including an independent xhigh review. Corrections separated host-visible divergence from conditional device-feedback propagation, scoped scorer-core activation to this request history, and incorporated the parent's newly completed non-watcher control. Evidence cutoff includes that control's completion at 2026-11-17 03:32:06 UTC. Review criteria: distinguish recorded outputs from inferred model state; test all headline suspects against the eager control; keep historical source changes separate; account for enqueue versus completion; instantiate the first matching/failing boundary; do not promote the scorer boundary without a violated contract. Hypotheses that did not survive this review are retained above as unproved leads or excluded mechanisms. Checkout-root inherited `AUTODEBUG.md` is outside this deliverable and remains unchanged.
