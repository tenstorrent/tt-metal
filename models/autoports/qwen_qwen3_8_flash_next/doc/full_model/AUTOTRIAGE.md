# AUTOTRIAGE

## Diagnosis

The supported localization is a host-to-device **dispatch fetch-queue credit stall during an interleaved buffer upload**, before the first layer finishes loading. The evidence does not establish a specific source defect or prove that the omitted mesh descriptor caused it. The leading discriminating hypotheses are stale/mismatched fetch-queue producer/consumer state across device initialization, or incorrect visibility/addressing of the credit/slot writes under sustained uploads. A deterministic PLE size limit is contradicted by the immediately preceding successful load of the same reduced model.

This investigation was read-only apart from this report. No devices, resets, tests, or implementation edits were performed.

## Triage Evidence

- `reduced_smoke_v1.log`: opens at 22:11:26, loads layers 0, 1, and 3 and terminal weights, and fails only on the probe's invalid `mesh.get_device(0)` call. Device/cluster close completes at 22:12:03. Both v1 and v2 explicitly report automatic mesh discovery and the same unknown-motherboard fallback. Thus the missing descriptor is a real departure from earlier stage configuration, but does not distinguish these two runs.
- `reduced_smoke_v2.log`: mesh opens and fabric initializes on all four devices at 22:13:41, then no `LOADED_LAYER` marker appears.
- `triage/reduced_v2_host.txt:556`: main host thread is in `Cluster::read_core` from `SystemMemoryManager::fetch_queue_reserve_back`, called by `issue_buffer_dispatch_command_sequence<InterleavedBufferWriteDispatchParams>` and `write_interleaved_buffer_to_device`. It is not executing Python tensor math, waiting for a model collective, or closing the mesh.
- `triage/reduced_v2.txt:272` and corresponding devices 1–3: all four `cq_prefetch` kernels are in `fetch_q_get_cmds` at the zero-command-slot polling loop (`cq_prefetch.cpp:943`). All four dispatch kernels wait for upstream pages at `cq_common.hpp:664`. The op mesh reports idle. There is no active attention, PLE lookup, LM-head or sampler operation to blame.
- Semaphore snapshots are identical on all four chips: prefetch semaphore 0 is 1,663,872, dispatcher semaphore 0 is 1,663,782. These are cumulative downstream page counters, not the missing fetch ring cursor values. They do not prove fetch-ring fullness or a queue-size mismatch.
- The summary's `pass` labels mean the diagnostic scripts completed, not that every checked subsystem is healthy. The detailed report contains NoC mismatch warnings on Ethernet ERISC1, failures to read dispatcher symbols because BRISC was no longer halted, and `check_broken_components` messages about cores no longer halted after triage. These reduce confidence in a fully coherent stopped snapshot. They are not evidence of a model source bug or of physical component failure.

## Source Evidence

`tt/model.py:QwenModel.__init__` creates workspace/collective helpers, uploads the packed PLE cache, uploads token embeddings, and then loads layer weights. The absence of a layer marker cannot by itself identify which of those writes stalled. `MeshPackedEmbedding.from_cache` is a likely early large upload: the words table represents 320,001,536 rows × 20 uint32 words = 25,600,122,880 bytes globally, or 6,400,030,720 bytes per rank; scales add 1,280,006,144 bytes per rank. v1 passed this setup, so these byte counts are workload pressure, not demonstrated overflow or capacity failure.

Fetch-queue ledger from current source:

| State | Producer/owner | Consumer/action | Contract |
|---|---|---|---|
| Fetch slot command size | host `fetch_queue_write` | prefetch `prefetch_q_read_slot` | Host writes one nonzero size to its current physical ring slot, then advances by the configured entry size. |
| Fetch credit address | prefetch `read_from_pcie` | host `fetch_queue_reserve_back` | Device clears the consumed slot and publishes its consumed physical address to `PREFETCH_Q_RD`; host may proceed when its next write address differs from that fence. |
| Physical ring cursor | host `prefetch_q_dev_ptrs` and device `prefetch_q_rd_ptr` | wrap at identical base/end | Fresh setup initializes both at the queue base. Host's initial cached fence is the queue end. Both advance physical pointers; no absolute-counter-to-modular-index reconstruction appears here. |
| Command-data pages | prefetch → dispatcher | dispatch waits for upstream pages | Dispatcher is downstream of the stalled upload publication/credit loop. |

Relevant source:

- `tt_metal/impl/dispatch/system_memory_manager.cpp:244–312`: creates per-device IO windows and initializes fetch pointers. An explicit issue-queue capacity assertion already enforces max-command-size × (fetch entries + wrap + outstanding reads) ≤ issue queue size. Merely increasing queue capacity is not a supported repair.
- `system_memory_manager.cpp:681–750`: reserve blocks only while host write pointer equals the repeatedly read device credit address, with wrap handling before returning.
- `system_memory_manager.cpp:847–897`: publishes slot size and advances the host ring pointer.
- `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp:598–618`: clears a consumed slot, publishes consumed-entry credit, advances/wraps device cursor. The code already uses uncached credit publication; on Blackhole the alias conversion is identity. Adding that existing behavior is not a fix.
- `cq_prefetch.cpp:930–943`: the observed device stop means no fetch command at the current consumer slot, no pending fetch and no available command. Combined with the host credit wait, this is inconsistent with ordinary sustained backpressure if both snapshots address the same healthy queue: a drained queue should have released space. Exact pointer values and blocked device ID are missing, so cursor mismatch/visibility remains a hypothesis.
- `/home/ttuser/qwen38-flash/lane-env.sh` explicitly unsets `TT_MESH_GRAPH_DESC_PATH`. The accepted descriptor `tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto` declares a physical 2×2 Blackhole mesh with two relaxed channels; automatic discovery also reports a 2×2 physical op mesh. Neither log proves a route difference. All chips are PCIe-local and the blocked path is an interleaved upload, so a CCL route deadlock is not the first explanation.

## Downstream Effects

The four idle dispatch consumers and idle model cores are effects of no new commands reaching them. They do not imply an attention/cache precision issue. Existing Ethernet warnings deserve preservation but do not explain a local fetch-credit mismatch without further routing or addressing evidence. A successful close message in v1 also does not independently prove that every device queue was reinitialized correctly in v2.

## Proposed Fix / Discriminating Next Run

Do not change model precision, sharding, PLE packing, queue depth, or model algorithms based on this capture.

1. After the owner completes the authorized timeout/recovery sequence, restore the accepted descriptor **after sourcing lane-env**, and run the same reduced probe once under its existing bounded timeout. Add flushed setup markers around workspace creation, packed words/scales uploads, embedding upload, and layer upload if localization is still needed. Log descriptor and exact command. This restores the inherited configuration; a pass alone is recovery evidence, not proof that topology was the cause, because reset and descriptor differ together.
2. If that run passes, a second process doing the identical reduced setup after normal close (without another reset) is the narrow lifecycle discriminator. A first-open pass/second-open failure supports queue reinitialization/lifecycle; repeated passes weaken it. No need to load the all-layer model to answer this question.
3. If upload stalls again, preserve state before timeout and capture the blocked host device/CQ, `prefetch_q_dev_ptrs`, `prefetch_q_dev_fences`, actual `PREFETCH_Q_RD` value/address, physical prefetch core, device `prefetch_q_rd_ptr`, ring base/end/entry size, and neighboring slot contents. Compare write IO-window target versus read-core target. A repeated identical ring snapshot establishes a real no-progress condition and distinguishes stale credit, wrong target, cursor disagreement, or lost slot publication. The existing optimized stack trace lacks those values; collecting them requires a diagnostic build/logging or debugger support, not another model rewrite.

## Uncertainty

No Python frame or setup substep marker proves that the stalled buffer is the PLE table. No before/after queue snapshots establish the exact blocked cursor or device. Triage itself reported incomplete halt stability. No source-level violation has been isolated strongly enough to justify a runtime patch. Consequently this is a localized supported diagnosis and experiment plan, not an AutoFix success or a hardware-blocked conclusion.
