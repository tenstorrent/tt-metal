# AUTOTRIAGE

## Diagnosis

- The watcher run exposes an invalid fabric packet-header/route state in the BRISC writer of the minimal-default `all_gather_async` used by Gemma4's persistent decode all-reduce path. The strongest source match for watcher line 119 is `PacketHeaderPool::get_num_headers`, whose assertion requires `route_id < route_id_`; therefore the immediate fault is an invalid or stale route ID, not a decoder numerical error or a host-side trace-capture write. The most likely trigger is packet-header bookkeeping surviving or being reused incorrectly across the dense sequence of persistent async collectives during the eager warm pass before trace capture. The exact producer of the stale route ID remains unproved because the abort ended the process before a live `tt-triage` capture.

## Triage Evidence

- Primary artifact: `watcher_full_path.log`, 2026-08-15 21:04:12. Device 0 worker logical `(0,0)`, virtual `(1,2)`, BRISC intentionally halted on an assert while running `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp`; NCRISC on the same core was in `minimal_default_reader.cpp`.
- The host was blocked at `Gemma4Generator._get_or_capture_decode_trace` line 481, the synchronization after the exact eager warm path and before `begin_trace_capture`. Thus the failed device work belongs to warm decode, not trace recording or replay. The later Python abort and device stop are downstream handling of the BRISC `ebreak`.
- Watcher says the reported line can be from a different file. Line 119 of `minimal_default_writer.cpp` merely reads `chunks_per_sync` and contains no assertion. Line 119 of included `tt_metal/fabric/hw/inc/packet_header_pool.h` is `ASSERT(route_id < route_id_)` in `PacketHeaderPool::get_num_headers`; this is the closest exact source match and is consistent with a fabric-writer BRISC stop.
- The detailed watcher dump shows many unrelated worker cores idle/waiting and active Ethernet RISCs, while the first explicit fault is the single BRISC assert on device 0 core `(0,0)`. Those broad states are victims/fanout, not independent root causes.
- Ordinary fallback-raising execution and profiler execution pass. The watcher build adds assertion instrumentation (`-DWATCHER_ENABLED`), so this contrast is consistent with a latent invalid state that otherwise proceeds without a checked halt. It does not establish that watcher itself caused the bad route state.
- No process remained alive after watcher stopped the device, so no call stacks, semaphore values, NoC counters, or live fabric route table were captured with `tt-triage`. This report consequently cannot prove the precise invalid route value or which invocation first created it.

## Source Evidence

- `Gemma4Generator._get_or_capture_decode_trace` executes a complete eager `model.decode_forward`, then `plus_one` on persistent position tensors, and synchronizes before capture. This places the stop inside the warm model graph.
- `MultichipDecoder._all_reduce_hidden` selects persistent `ttnn.experimental.all_reduce_async` whenever the matrix has at most one tile row. Decode satisfies this condition. Five full-attention layers each execute two TP reductions, so one full decode warm step submits ten persistent async reductions. Sliding-attention layers use the synchronous fallback by policy.
- All full-attention layers share one `persistent_all_reduce_resources` dictionary. It owns three output buffers, three global semaphores, and one shared modulo-three index. The decode ledger for the ten async reductions is therefore resource indices `0,1,2,0,1,2,0,1,2,0`; each reduction decomposes through the experimental async all-gather writer implicated by watcher.
- Each persistent reduction uses a width-sharded L1 input/output, TP cluster axis 1, ring topology, two links, one global semaphore selected by the shared index, and the same persistent output-buffer slot. The all-gather program factory selects worker cores from the device/subdevice grid and builds the minimal-default reader/writer, matching the reported kernel and core.
- The minimal-default writer allocates three packet headers (`pkt_scatter_hdr`, `pkt_unicast_hdr`, `pkt_hdr_sem_inc`) and uses route/connection state for both ring directions. `PacketHeaderPool` owns static `current_offset_`, `route_id_`, and `header_table`; `allocate_header` increments `route_id_`, while `get_num_headers` asserts that a requested route was allocated in the current pool state. A `reset()` implementation exists, but a repository-wide source search finds no caller. This verifies that an explicit reset at this kernel boundary is absent rather than already implemented.
- Producer/consumer ledger:
  - Host/program factory produces topology, forward/backward neighbors, worker routing compile-time arguments, `chunks_per_sync`, and semaphore runtime arguments.
  - BRISC writer produces three local packet headers/routes and fabric data plus atomic semaphore increments.
  - Peer reader consumes the output-ready increments according to the same chunk cadence; downstream all-reduce consumes the gathered persistent buffer.
  - Shared Gemma resources produce only three buffer/semaphore identities for ten ordered decode reductions. Correctness therefore relies on command-queue/fabric completion and per-launch packet-route initialization before an identity is reused.
  - Watcher's route assertion proves that this reliance was violated at packet-header consumption; it does not prove a CB count mismatch. No CB underflow/overflow assert, reader assert, or semaphore wait was reported first.
- The suggested `GEMMA4_MODEL_TRACE_ONLY=1` isolation run is not CCL evidence: that test branch constructs logical-length-one position tensors while `model.decode_forward` slices to 32 slots, and fails on `slice end 32 > shape 1` before reaching the target collective.

## Downstream Effects

- The host synchronization at generator line 481 reports the already-tripped device assert. It is not the cause and moving/removing the synchronization would only delay observation.
- NCRISC remaining in the paired all-gather reader, active Ethernet RISCs, and other worker wait states are expected after the BRISC writer stops producing packets and semaphore increments.
- Trace capture never begins in this failure, so trace allocator rules, replay mutation, split sampling, token feedback, page-table refresh, and position/RoPE advancement are not implicated by this artifact.
- The failure does not justify changing the selected BF16 CCL policy, replicated inter-layer residual contract, topology, or sampling contract.

## Proposed Fix

- First run two focused watcher experiments after device recovery:
  1. Disable only `GEMMA4_MULTICHIP_PERSISTENT_ALL_REDUCE` and run the same full generator probe. A pass localizes the trigger to persistent async CCL; a repeat assert refutes that boundary.
  2. With persistent CCL enabled, run a corrected model-trace-only probe whose token, `current_pos`, and `position_ids` all satisfy the fixed 32-slot decode shape. Sweep one full-attention layer, then enough full-attention layers to cross resource-index reuse (two reductions/layer: one layer uses indices 0/1; two layers reach 2/0). This identifies whether the first modulo-three reuse is necessary.
- Capture watcher DPRINT plus a live `tt-triage` snapshot if a focused run hangs rather than aborts. Instrument, or temporarily strengthen, the packet-header assertion to report requested `route_id`, current `route_id_`, direction, link/worker, and persistent resource index. Also record the three semaphore values before reuse. These measurements distinguish stale packet-header state from premature persistent semaphore/buffer reuse.
- If the invalid route reproduces before any resource-index reuse, the smallest source fix is to establish fresh packet-header-pool state at minimal-default writer entry (call `PacketHeaderPool::reset()` before its three allocations), then rerun watcher, normal correctness, trace replay, and profiler tests. The reset API already exists for re-allocation, and no current call protects this boundary.
- If the failure begins exactly on modulo-three resource reuse and route state is valid at entry, fix resource lifetime instead: expand the persistent ring to the proven maximum in-flight requirement or add a device-side completion dependency before reusing a slot. Do not add a per-token host synchronization; that would violate the optimized token-out contract.
- Do not accept disabling persistent async CCL as the production fix unless hardware/runtime evidence shows a hard unsupported condition. It is a useful localization experiment only and would discard a required optimization.

## Uncertainty

- Watcher cannot report the assertion's originating included file reliably. The exact line-number match to `PacketHeaderPool::get_num_headers` is strong but inferential; line 119 in the named writer itself is not an assert.
- There is no live `tt-triage` capture, route-ID value, semaphore snapshot, or per-device fanout chronology because watcher aborted and stopped the device. Therefore packet-pool reset versus persistent-resource lifetime remains the key unresolved branch.
- Normal and profiler passes show that the workload can complete without watcher checks, but they do not prove the unchecked route ID was valid.
- The malformed model-trace-only probe neither supports nor refutes the CCL diagnosis and must be corrected before it can serve as an isolation test.

## Second Pass: Post-Endpoint-Guard Scatter Assertion

### Revised Diagnosis

- After guarding access to absent endpoint connections, the original device 0 core `(0,0)` line-119 failure is gone. The new artifact, `watcher_full_path_fixed.log`, proves a separate bug: the minimal-default all-gather writer unconditionally initializes a scatter-write header even when program-factory geometry selects exactly one page per fabric packet. A scatter command requires 2--4 chunks, so this one-page case trips `api_common.h:260`. The data loops already select an ordinary unicast write for a one-page tail; only the unconditional scatter-header prepopulation violates the contract.

### New Triage Evidence

- At 2026-08-15 21:16:00 watcher reports device 0 worker logical `(4,0)`, virtual `(5,2)`, BRISC assert line 260 in `minimal_default_writer.cpp`, with paired NCRISC in `minimal_default_reader.cpp`.
- The exact included-source match is `tt_metal/fabric/hw/inc/api_common.h:260`: `populate_unicast_scatter_write_fields` asserts `NOC_SCATTER_WRITE_MIN_CHUNKS <= command_header.chunk_count <= NOC_SCATTER_WRITE_MAX_CHUNKS`. The constants are 2 and 4.
- The Python stack is now in `Sampling1D._sample_topk` at `ttnn.sampling`, called by generator warm sampling at line 476. This is after model decode and before trace capture. It distinguishes the new fault from the earlier persistent decoder all-reduce attribution: this stop belongs to the split sampler's collective path.
- The changed core and assertion, plus disappearance of line 119 after the endpoint guard, are evidence that the guard closed the first invalid-connection fault and exposed the next checked contract violation. The new assert is not downstream fallout from the old one.

### Scatter Call and Geometry Ledger

- The factory computes `num_pages_per_packet = packet_size_bytes / page_size`. It first fatals unless `packet_size_bytes >= page_size`, so the quotient cannot be zero for positive tensor page sizes. It then computes `num_tiles_to_write_per_packet = min(4, num_pages_per_packet)`. Consequently this compile-time value is in `[1,4]`: zero and values above four are excluded by host construction.
- Writer setup calls `fabric_unicast_noc_scatter_write_set_state` unconditionally with `NocUnicastScatterCommandHeader(..., num_tiles_to_write_per_packet)`. This is the only scatter call that can receive one chunk.
- In the local-slice loop, `tiles_to_put_in_current_packet = min(tiles_remaining, num_tiles_to_write_per_packet)`. The scatter call is guarded by `tiles_to_put_in_current_packet > 1`; a one-page packet takes the unicast branch. Since the compile-time maximum is four, its scatter count is always 2--4.
- The forwarding loop has the same `> 1` scatter/unicast branch and the same maximum-four bound. Its scatter count is also always 2--4.
- Therefore neither loop call can supply 0, 1, or more than 4 to the scatter helper. The setup call supplies exactly 1 when tensor page size equals the fabric channel payload size (or otherwise leaves room for only one page). This uniquely explains the line-260 assertion.
- The setup scatter header is not needed in that geometry: every later data packet necessarily selects the existing `pkt_unicast_hdr` path because `num_tiles_to_write_per_packet == 1`.

### Focused Smallest Fix

- Guard only the setup call to `fabric_unicast_noc_scatter_write_set_state` with `if constexpr (num_tiles_to_write_per_packet > 1)`. Keep allocation of the header unchanged initially to minimize route/header lifetime changes; the already-present unicast-header setup remains valid for the one-page case.
- Add a compile-time lower-bound assertion that `num_tiles_to_write_per_packet > 0`, adjacent to the existing `<= 4` assertion, as defensive documentation of the factory/kernel contract. The host factory already enforces this indirectly, so it should not alter valid behavior.
- Verify with a focused watcher test that forces the one-page geometry, then rerun the exact full-generator watcher probe. Also run a 2--4-page geometry to ensure scatter setup and both scatter loops remain checked. Normal, trace-replay, and profiler validation must follow because watcher passing alone does not establish semantic output correctness.
- Do not replace the sampler collective, change sampler shape, disable split sampling, or introduce a host boundary. This is a generic minimal-default writer setup bug and the one-line compile-time guard preserves the model's sampling contract.

### Second-Pass Uncertainty

- The log does not print `page_size`, `packet_size_bytes`, or `chunk_count`; the value one is deduced exhaustively from the factory bounds and all writer call-site guards. A DPRINT-enabled focused run can confirm it directly, but no alternative call site can legally produce the observed out-of-range count under the prepared source.
- No live `tt-triage` capture exists because watcher aborted the device. Unlike the first pass, the exact included assertion and exhaustive call-site bounds make semaphore/route-state inspection unnecessary for choosing the smallest fix.
