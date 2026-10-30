# AUTOTRIAGE

## Diagnosis

- **The evidence localizes the reported persistent stall to host→prefetch FetchQ progress, but does not establish the underlying source defect.** The host is sampled polling the fetch-queue consumption fence while the prefetcher is sampled waiting for an empty input slot, with no pending command reads. If those states persist together, they violate the queue's producer/consumer contract. SDPA is not shown stuck in a kernel. The actual cursor values and slot contents needed to distinguish a lost publication, bad fence read, cursor mismatch, or ordinary transient backpressure were not captured; claiming a specific wrap/credit bug or a validated functional fix would exceed the evidence.

Investigation was source/evidence-only. No target execution, hardware access, reset, process termination, launcher invocation, or source modification was performed. `AUTOTRIAGE_INPUT.md` was requested first but is absent from the checkout (also absent from the repository file search). The supplied problem statement and named captures are the investigation inputs. Startup contracts were read.

## Triage Evidence

Evidence directory: `models/autoports/google_gemma_4_31b_it/doc/optimized_decoder/final_context_watcher/`.

- `sliding_long_host_stack.txt`: PID 170651 runs layer 0, batch 1, sequence 262143. Its main thread is in UMD `memcpy_from_device` / `read_block_reconfigure` / `Cluster::read_core`, called by `SystemMemoryManager::fetch_queue_reserve_back`, while `write_program_command_sequence` submits a cache-hit `CopyDeviceOperation` from `to_memory_config` in `prefill_chunk_forward`. This is submission-side waiting, not evidence that the Copy kernel itself is blocked. A single sampled native stack does not prove the read returns repeatedly; persistent spinning is also part of the requester's observation.
- First capture: Inspector identifies op 5073 `SDPAOperation`, preceded by op 5072 `PagedFillCacheDeviceOperation`; its op window includes 5074 `NLPConcatHeadsDeviceOperation`, 5075 `CopyDeviceOperation`, and operations through 5083. That list is not an execution-completion ledger. Q is `[1,32,1024,256]` BF16, K/V are `[8195,16,32,256]` BFP8, and the page table is `[1,8192]` INT32.
- Despite that running-op label, the listed SDPA workers are in firmware launch/notification loops (`brisc.cc:403–408`, `trisc.cc:135–152`, `ncrisc.cc:79–89`), not a reader/compute/writer CB wait or an LLK assertion. Kernel names alone do not establish that their bodies are executing.
- **Both captures** put device 0's prefetch BRISC at physical `16-2`, `cq_prefetch.cpp:944`, polling `prefetch_q_read_slot`. Both put dispatch BRISC at `16-3` in `CBReader::acquire_pages`, `cq_common.hpp:651`, waiting for upstream data. Dispatch subordinate also waits for input. Other mesh devices are unused.
- Second capture: device 0 is idle, no running operation, no worker CB dump, and binary integrity passes. This is consistent with the first capture having mixed operation metadata and later worker state; it is not proof of a persistent SDPA kernel deadlock.
- All six reported dispatch semaphores are identical across captures: prefetch semaphores 0/1/2 are `1445312 / 653 / 5243`; dispatch semaphores 0/1/2 are `1445222 / 5115 / 0`. ARC uptime advances from about `1:47:45.6` to `1:48:38.8`. Unchanged counters over roughly 53 seconds support dispatch quiescence, although they do not by themselves prove a host submission was outstanding throughout that interval. The host capture is later, not simultaneous.
- First-capture binary mismatches and implausible CB geometry are not a coherent SDPA buffer-corruption diagnosis: the workers were sampled in firmware waits, all reported worker CB push/pop counts are zero, and the second capture has no such active-worker findings. Live metadata/program replacement is a plausible explanation, not proven. The first capture also reports cores that triage halted but found no longer halted; that is a capture reliability warning, not proof those cores caused this stall.
- Both captures report ERISC1 hardware/software NoC counter mismatches on `13-1` and `15-1`. Hardware nonposted sent/acked counts are balanced (`484/484`, `335/335`); software counters are zero. No captured causal path connects these to the local worker-based `PREFETCH_HD` queue. Do not promote them to a fabric or hardware root cause. LLK asserts and watcher are disabled, so their empty outputs do not constitute comprehensive assert validation.

## Source Evidence

### Queue ownership and progress ledger

| Resource | Producer / publication | Consumer / release | Required balance |
|---|---|---|---|
| Host issue queue | `write_program_command_sequence`, `tt_metal/impl/program/dispatch.cpp:3448`, writes command fragments and advances issue write pointer | Prefetch `read_from_pcie` reads each advertised byte range | Each FetchQ descriptor advertises the corresponding command bytes exactly once; one descriptor per one-shot program or nonempty split fragment, not necessarily one per TTNN op |
| FetchQ slot | `SystemMemoryManager::fetch_queue_write`, `system_memory_manager.cpp:861`, writes encoded size, then advances host address W by entry size E | `cq_prefetch.cpp:474–618`, reads descriptor, issues PCIe read, clears slot on BH, publishes consumed slot address F, then advances device cursor R | Exactly one descriptor consume and F update per successful issue; no cursor advance on failed reservation |
| Host FetchQ capacity | `fetch_queue_reserve_back`, `system_memory_manager.cpp:697–765`, polls F when W equals its cached fence | Prefetch publishes F to `PREFETCH_Q_RD` at line 612 | A stable full host queue requires unconsumed descriptors ahead of R, not an empty R slot |
| Command-data ring / PCIe transactions | Prefetch issues up to four tagged reads, TRIDs 2–5, reserving `issue_fence` | Retires oldest read with `noc_async_read_barrier_with_trid`, updates committed fence, decrements in-flight count | One retirement per issued read; no outstanding read when the line-944 empty-host loop is entered |
| Prefetch→dispatch pages | Relay inline publishes `ceil(length/page_size)` pages after writes are flushed | Dispatch `CBReader` acquires pages and releases old blocks to prefetch | Semaphore 0 on dispatch counts published pages; prefetch semaphore 0 starts with 128 credits and receives returned pages |
| Prefetch→dispatch-subordinate pages | Prefetch publishes subordinate command pages | Subordinate consumes and returns page credits | Captured subordinate publication count 5115 and return-credit count 5243 differ by the initial 128 |
| Exec-buffer synchronization | Dispatch signals prefetch synchronization semaphore | Prefetch consumes synchronization progress for stall/exec-buffer handling | Prefetch semaphore 1 = 653 is not FetchQ's cursor or a count of queued Copy operations; current PC is not this wait |

The main dispatch page ledger is arithmetically consistent with deferred block release: `1445312 - 128 = 1445184` returned pages, versus `1445222` published, leaving 38 pages not yet returned. With 32-page blocks and deferred release, that difference alone is not proof of a full queue or leaked credits. Crucially, the dispatcher is waiting for new upstream data and the prefetcher is not waiting for downstream space. Subordinate returns equal publications after removing initial credits. These are cumulative counters, not instantaneous occupancy measurements without their local consumed counts.

### Exact stop-site semantics

`cq_prefetch.cpp:867–946` retires any in-flight read before entering the empty-host polling loop. Reaching line 944 implies `!cmd_ready`, `inflight_count == 0`, and an empty current FetchQ slot as seen by the prefetcher. It is not a PCIe acknowledgement barrier, command-data capacity wait, worker-completion wait, or trace completion phase. `cq_common.hpp:639–651` explains dispatch's downstream symptom: its acquired page count has caught up with upstream publication.

`fetch_queue_reserve_back` waits only when W equals the fetched/cached F, and wraps W at the explicit queue limit. F is **the last consumed slot address**, not the next-to-read cursor. Therefore changing the equality test to allow reuse of F or publishing R instead of F would change the queue safety protocol and is not a justified fix.

For base B, entry size E, depth N, use slot indices w and r. After consumption, `r = (f + 1) mod N`. Host blocked at `w == f` consequently means `w = (r - 1) mod N`: the queue has reached its guarded capacity, assuming ordered contiguous publications. Prefetch waiting on zero at r is incompatible with that state persisting in a healthy queue. In a drained queue, `w == r`, whereas `f == r - 1`, so host reservation should succeed. This is the decisive contract to check with actual values.

The prepared non-Galaxy, non-DRAM-backed worker configuration selects N=1534 (`util/dispatch_settings.cpp:128–140`), with 4-byte worker entries. Both sides wrap address cursors explicitly; this path does **not** reconstruct slot indices from a wrapped absolute counter. No close/reopen transition is shown. Non-power-of-two depth alone is therefore not a TRI-005 bug. Initial host/device fence is the queue limit and slots are zeroed (`kernel_config/prefetch.cpp:575–591`). The issue-queue capacity assertion already includes N+1+four in-flight reads (`system_memory_manager.cpp:298–306`); unequal capacities have not been demonstrated.

### Publication checks and model boundary

The source already has a pre-publication `sfence()` in `fetch_queue_write`, explicit consumed-fence publication through `uncached_l1_ptr`, and explicit pointer wrapping on both sides. On BH the alias conversions are identity. Adding these existing behaviors again is not a fix.

The host descriptor window defaults to WC (`tt-umd/api/tt-umd/types/io_window_config.hpp:102`); scalar `write32_to_device` performs a volatile store without an explicit trailing fence. That deserves publication-order inspection if queue values support it, but is **not established as this root cause**: later submissions already execute the pre-publication fence, and a single buffered final descriptor does not alone explain the full-ring equality above. Do not declare that adding an arbitrary fence fixes the observed failure.

The current `optimized_decoder.py:515–609` iterates 1024-token chunks, fills K/V, calls chunked SDPA, finishes the projection/MLP path, and converts output to DRAM. The supplied Python line number predates the reported attention edits; do not mechanically equate it to today's line 592 or assume the sampled Copy is op 5075. The unchanged generic Copy enqueue path is the relevant boundary. A long request submits many more commands than short2049 or decode-batch tests, increasing queue-wrap/backpressure exposure, but that contrast does not prove a particular queue bug. The page-view and 32-aligned continuation changes do not repair this dispatch contract.

## Downstream Effects

Dispatcher and subordinate input waits, worker firmware waits, idle device status, and the blocked model call follow naturally if host descriptors cease to arrive. They are not independent root causes. The retained SDPA name, first-snapshot binary/CB anomalies, and ERISC counter discrepancies are insufficient grounds for rewriting SDPA, changing BFP formats, reducing context, resetting hardware, or adjusting CCL routes.

Timeout configuration is a concrete amplifier: captured `timeout_duration_for_operations=0s` selects the unbounded loop at `system_memory_manager.cpp:108–116`. This explains indefinite polling if queue progress fails, but does not explain why it failed.

## Proposed Fix

**No final functional patch is justified by these captures alone.** The minimally scoped next source change is a bounded, observational FetchQ diagnostic in `fetch_queue_reserve_back`, using the existing timeout/progress mechanism rather than inventing a new queue algorithm. At a prolonged wait, preserve:

- CQ/device/core identity; B, E, N; host W and cached F; repeated raw `PREFETCH_Q_RD` reads; last descriptor address/value and publication sequence count.
- Current device R (from the exact running ELF/state), slot at R, nearby slots across wrap, prefetch in-flight count and committed/reserved command-data fences; dispatch telemetry and counters at the same time.
- Runtime library/dispatch ELF identity and actual IO-window target/mapping. Verify the fence through an independently addressed read if the parent can do so safely.

For a parent-owned follow-up run, enable the existing finite dispatch timeout so a failure captures evidence instead of spinning forever. The timeout path already exists; activating it is containment/diagnosis, not a root-cause repair. Do not alter the live process during this investigation.

Apply the functional fix only at the boundary those values identify:

1. If device F disagrees with host-read F, repair the fence read's address/coordinate/window or visibility contract.
2. If W/F indicate outstanding work but the first expected descriptor is zero, establish where that descriptor was written/cleared; repair publication or cursor ownership there. A release fence or UC doorbell window is appropriate only if a visibility defect is verified.
3. If host/device geometry or R disagrees, repair the mismatched configuration/cursor transition together, preserving the last-consumed fence convention.
4. If repeated samples show progress and W/F become consistent, classify the stack as transient backpressure and measure long-prefill progress; do not patch FetchQ based on an idle snapshot.

Verification after a demonstrated repair should exercise repeated FetchQ wraps with cached small programs and a terminal host wait, then the original maximum-context prefill and the reported passing short/batch cases. No such execution was performed here.

## Uncertainty

- Missing `AUTOTRIAGE_INPUT.md`; no simultaneous host/device cursor/slot capture, no host poll values, and no repeated host stacks in the supplied file. The snapshots strongly narrow the investigation but cannot uniquely identify a source defect.
- The parent reports persistent spinning; the captured stack could also be a slow UMD read or transient reservation. Unchanged dispatch counters support quiescence, not the precise cause.
- Runtime and source identity must be verified before patching: the host stack names UMD 0.9.14 while triage's system-info row reports 0.9.12. These may belong to different processes/environments. Python source changes since launch are explicitly acknowledged; no claim is made that editing current source affects the live process.
- First-capture worker integrity/CB anomalies remain unresolved. Their disappearance and firmware stop sites demote them; they are not proven harmless corruption or proven hardware failure.
- Short2049 and B1..56 passes are requester-reported context, not rerun evidence. Neither a maximum-context hardware limit nor a model accuracy failure is established by these captures.
