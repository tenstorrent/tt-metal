# Hidden-norm fusion follow-up: admission and completion ownership

2026-11-13. CPU/source audit only. This is the only new/edited artifact for this task. No implementation, device access, TTNN/torch import, build or timing. The head-major recurrence candidate/probe are frozen.

## Evidence changes the decision

The isolated two-program result does **not** predict the integrated decoder result. Both must remain in the record:

| Artifact and scope | Native/control | Fused | Difference |
|---|---:|---:|---:|
| `hidden_norm_perf17.json`, complete norm B1 |23.47475us|33.32959us|+9.85484us|
| same, complete norm B32 |22.58496us|32.02560us|+9.44064us|
| `fusions18_l0_regression_{control,fused}_b1_s512.json`, decoder |484.32765us|480.45085us|-3.87680us|
| `fusions18_l0_text_{control,fused}_b1_s512.json`, decoder |484.25822us|480.61306us|-3.64516us|
| `fusions18_l0_regression_{control,fused}_b32_s33.json`, decoder |578.67684us|579.22760us|+0.55076us|
| `fusions18_l0_text_{control,fused}_b32_s33.json`, decoder |578.47316us|578.84837us|+0.37521us|

These JSONs were read directly. Isolated timing has watcher/profiler/allocation tracking disabled. `hidden_norm_wide17.json` passes96 cases with watcher and allocation tracking enabled, without program-cache exemption. Whole-layer PCC is unchanged in each control/fused pair; the parent confirms two norm calls enter the hook. `multichip_decoder.py:511` selects the helper for the exact-tree GDN path.

The approximately constant isolated penalty is compatible with fixed admission/launch/trace or buffer-placement costs. It does not establish which one dominates, and the integrated result actively argues against reporting a roughly20us whole-layer regression. Pending matched whole-layer profiles should settle device-kernel/gap attribution. Current B32 whole-layer differences are below0.1%; treat them as unresolved without dispersion/repetition, rather than a categorical rejection. B1 is a plausible batch-specific keep if repeated final-path timing and quality gates confirm it.

## First bounded variant: dispatch admission, unchanged completion

**The lowest-complexity source-supported candidate removes READY/GO entirely and retains the existing scatter/DONE protocol.** This changes only reader/writer synchronization. It retains the32-core map,26,624B/core CB union, both arithmetic bodies, global4KiB inverse buffer, and unchanged80-core output program. No collective, tensor layout, dtype, or arithmetic tree changes.

Current source (`tt/hidden_norm_fused_candidate.py:READER` and `WRITER_PREFIX`) waits for all32 readers to finish five DRAM reads before releasing any writer. Each reader sends READY and drains it; coordinator0 receives32 credits, then issues32 sequential GO atomics; each writer waits for GO even if its own fold is already ready. This unnecessarily couples the fastest fold producer to the slowest input read and the coordinator's release loop.

The existing runtime already supplies explicit launch admission:

1. `tt_metal/impl/program/dispatch.cpp:1264` emits each program-local semaphore's initial value in the program config writes; at2442 those writes enter the program config sequence; at3122 that sequence is emitted on every invocation, including cached binaries.
2. `dispatch.cpp:2371` emits the dispatch-to-subordinate notification with a write barrier when workers are active. Without a subordinate it emits `CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER` before GO. This is a global configuration-write completion fence, not merely command emission order.
3. `tt_metal/impl/dispatch/kernels/cq_dispatch.cpp:1246` executes that NoC write barrier before notifying the subordinate. `cq_dispatch_subordinate.cpp:667` waits for the notification; `wait_for_workers_and_send_go_signal` at628 additionally waits for prior worker completions before issuing GO.
4. `tt_metal/hw/inc/internal/circular_buffer_init.h:19` sets CB interface metadata: base/read/write pointers, sizes, and received/acknowledged counters. Both its production assembly and explanatory C++ initialize metadata; neither clears the CB data payload. `CircularBuffer::reserve_back` in `api/dataflow/circular_buffer.h:31` waits for capacity; it does not clear payload either.
5. `ProgramImpl::allocate_circular_buffers` in `tt_metal/impl/program/program.cpp:2417` gives each CB descriptor one address across its complete core range. All existing fused CB descriptors cover exactly the same32 workers. Dense CB4 is private, one-shot, statically placed, and never reused/advanced during the program.
6. `tt_metal/distributed/fd_mesh_command_queue.cpp:1844` records the same initialized/config-fenced command sequence in traces. This proof covers queued reuse and replay, not just first launch.

Therefore a fold writer may write its assigned bytes of another role's dense CB **before that receiver executes reserve_back**, provided the receiver does not clear or overwrite the payload. The receiving role can be slower to initialize its local interface without losing those bytes; it later reserves the initial slot, waits for all32 DONE credits, then pushes it. The writer derives the common dense data address from its own initialized interface, not from remote metadata. Remote writes do not mutate CB counters. Dispatch admission, rather than a software round trip, establishes the data allocation's lifetime and semaphore epoch.

Minimal protocol:

- Reader reserves dense, loads/publishes raw input, then (if role<B) waits for32 DONE and publishes dense.
- Writer waits only for its own folded tile, sends the same disjoint64B rows, drains all data writes, sends one DONE credit per inverse receiver, and drains atomics.
- Compute and final inverse/output stores are unchanged. No kernel resets any semaphore. There are no receiver payload clears, pointer wraps, multi-slot reuse, or overlapping CQs in this contract.

At B1 control traffic drops from96 to32 atomic requests; at B32 from1088 to1024. More significantly, the entire READY round trip, coordinator GO loop and all-inputs-finished fence disappear. The retained data write and atomic barriers remain mandatory. Do not remove DONE on the assumption that a NoC write is observed by compute immediately.

This proof is specific to a single private dense slot with identical descriptor addresses and dispatch-serialized invocations. It does not license removing readiness from a stream, ring, global/persistent buffer, differently allocated CB map, or receiver that zeros its payload after launch.

An even smaller diagnostic control moves READY immediately after dense reservation, before DRAM reads, while retaining GO. That preserves explicit software admission and isolates the artificial all-input-read fence, but still pays64 startup atomics and the coordinator loop. It is useful if the parent wants the narrowest experiment first; it should not be the only adaptation attempted before rejecting the dispatch-admitted family.

## Second bounded variant: aggregate completion, multicast release

If completion traffic matters at B32 after the first change, replace32×B DONE atomics with a **single coordinator completion counter**:

1. Each fold writer sends its rows to every inverse receiver, executes a data-write barrier, then sends one atomic increment to coordinator0. Drain the atomic stream before exit.
2. Coordinator0 waits for32 credits. At this point every sender has acknowledged all its data writes to every receiver.
3. Release inverse roles with a single-writer flag multicast over an exact cover of their physical worker coordinates. Each receiver waits for flag1, then publishes its dense tile. B1 can use coordinator0's counter directly, without a release message.

This changes completion from32×B atomics to32 atomics plus a few release writes. A plain B-way unicast release is a valid geometry control, but it recreates a serial coordinator loop similar to the old GO path and may erase the benefit. Prefer exact rectangular multicast where the actual physical/virtual NoC coordinates admit it; derive and validate rectangles from host-resolved coordinates, never infer physical rectangles from logical8×4 or11-column indices. Never include inactive cores simply to fill a rectangle.

Two constraints from the actual APIs matter:

- `noc_semaphore_inc_multicast` (`api/dataflow/dataflow_api.h:2482`) explicitly excludes its sender from the destination group. A rectangle containing the sender needs splitting or a separately accounted self credit. Do not pretend its `num_dests` argument enables atomic loopback.
- `Noc::async_write_multicast` (`api/dataflow/noc.h:500`) supports `NocOptions::MCAST_INCL_SRC` via `noc_async_write_multicast_loopback_src_impl`. A single coordinator can publish a ready value with normal nonposted multicast and a write barrier; matching physical rectangle/node count and source inclusion must be explicit. Keep a dedicated source/release word. Do not reset a live counter or recycle a source scratch word with outstanding transactions. Avoid the stateful inline-L1 write API whose documentation warns about Blackhole hangs.

This adds a global aggregation/release hop versus direct per-receiver DONE. It is promising in request count, not established faster. At B1 it offers essentially no improvement beyond the first variant. Keep first-variant and aggregate controls separate so a poor multicast placement does not reject safe admission removal.

## One-program full chain: aggregate the inverse vector, not B×80 fanout

The earlier per-user inverse-to-every-output-owner proposal requires B×80 data writes and credits; at B32 that is2560 credits in addition to fold completion. A more coherent one-program family assembles the inverse vector once and multicasts it. It preserves80 output owners and the exact arithmetic tree rather than restoring the old slow one-core-per-user norm.

Roles:32 fold workers; firstB also inverse workers;80 workers retain the existing two contiguous output tiles each. Use the same descriptor union on all80 for remote-address invariance, with an exact multicast cover determined from physical worker coordinates. The conservative union is42KiB/core:10KiB raw fold input,4KiB folded,4KiB dense,4KiB scalar inverse,4KiB assembled/output inverse,4KiB output x,8KiB gamma,4KiB final output. This is3,440,640B total local CB allocation, plus program/runtime/semaphore metadata. No extra zero tile is required if nonlogical rows are zeroed only in their disjoint destination ranges.

Dependency graph:

1. All relevant readers reserve their private one-shot slots and prefetch fold x and output x/gamma. Output prefetch is an explicit on/off scheduling control; extra concurrent DRAM traffic can delay fold reads.
2. Fold compute and direct scatter use exactly the first/second variant above. Coordinator releases inverse roles after all dense input is available. Each inverse role retains the native COL then ROW reductions, FP32 reciprocal/epsilon/rsqrt and FP32 pack.
3. Each inverse writer sends its64B **left-face row** to coordinator0's assembled inverse CB and increments an inverse-ready counter after a data-write barrier. Only left faces0/2 are consumed by the unchanged output normalization. Logical rows are disjoint. Coordinator clears only inactive left-face rows B..31; these writes cannot race active producers. Do not clear the complete assembly tile after producers can write.
4. Coordinator waits forB inverse-ready credits and finishes its inactive-row clears. It multicasts the two1024B left faces to all80 output inverse CBs, drains those writes, then multicasts OUTPUT_READY. Output readers publish the inverse CB only after this release; coordinator uses the same assembly/output CB and publishes it once. Every outgoing atomic/write is drained before any role exits.
5. Each output compute role runs the native negative-zero normalization, FP32 gamma multiplication and explicit BF16 typecast, with the same two output tiles. Final writer zeros nonlogical output rows exactly as before. No global inverse tensor or second launch remains.

The coordinator reader can manage releases while its writer scatters/publishes inverse and its compute executes fold/inverse. Preserve that separation: waiting for inverse completion in the same compute thread that must produce it is a deadlock. Non-fold roles still run the output phase; no early return may skip it.

Compared with direct B×80 inverse fanout, this uses B row writes/credits into one assembly plus a handful of multicast data/release requests. Delivered output-inverse payload remains80×2048B=163,840B, matching the native output readers; network injection/request count can fall through multicast replication. Assembly adds64×B bytes. Fold scatter still delivers4096×B bytes. All1,638,400B of baseline DRAM x/gamma/output traffic remain: the fold and output ownership differ, so this proposal does **not** claim to remove the second x read. It removes the remaining4KiB global inverse tensor and one launch, and can overlap output prefetch. None of those savings establishes an end-to-end speedup.

This is more work/risk than dispatch admission removal. Exact physical multicast coverage, source self-inclusion, two completion epochs and per-role CB publication need source-level proofs plus stress. Do not implement it simply because the standalone two-program trace was slower; first inspect integrated profiles and measure the narrow variant.

## Discriminating evidence to request next

1. Finish paired whole-layer profiles with only hidden-norm selection changed. Match batch/context, final policy, output/gamma/input addresses where possible, preceding operations, number of norm invocations, warmed capture, and trace replay cadence. Report norm kernels and gaps separately from whole-layer uninstrumented time. Include standalone and whole-layer source/command hashes; do not compare superseded beta/decay or norm policies.
2. Run the dispatch-admitted variant with unchanged data and completion paths. Retain native3-op and current2-op controls. B1 and B32 are mandatory, with B3/B17 correctness and all32 batches before default selection. The early-READY control can explain whether global DRAM admission was material.
3. Extend the existing queued/traced watcher test with deliberately delayed receiver initialization. A sender must be able to finish dense writes before the receiving reader reserves its slot, with no payload clear. Confirm exact inverse-col0 and final output, counter reset and address rebind. The delay control is correctness-only; timing comes from its uninstrumented counterpart. Test50+ queued invocations and changing input buffers without changing arithmetic/quality gates.
4. Only if integrated evidence still shows a material local protocol bottleneck, test coordinator completion with validated multicast geometry, then the one-program assembled-vector family. Record before/after kernel and whole-layer timing for each coherent family. If a candidate loses only in standalone timing but wins in the final default path, judge the final default path and document the discrepancy instead of extrapolating microbenchmarks.

No source or runtime property found here requires a new collective between decoder layers. Residual layout, context/KV dtype and public sequence-length contracts stay unchanged. The one-program local CB increase must be reflected in per-device allocation evidence if selected; it is program workspace, not persistent KV state.

## Implemented minimal admission control (subsequent CPU-only task)

The parent subsequently authorized implementation in the existing opt-in helper and probe, preserving software admission as the default. Only `tt/hidden_norm_fused_candidate.py`, `tests/probe_hidden_norm_fused.py`, and this appended report section were changed. No model policy, main decoder, other helper/probe, native source, device, TTNN/torch import or native build was touched.

API: `hidden_norm_fused(..., admission="software", defer_receiver_reserve=False)`. `admission="dispatch"` removes the READY increment/drain, coordinator wait/GO loop/drain, and writer GO wait. It retains the exact scatter data requests, their write barrier, all B DONE increments per sender, final atomic barrier, inverse stores, arithmetic and output program. The original three semaphore descriptors are deliberately retained in both variants: IDs0/1 become unused initialized words in dispatch mode, while DONE remains ID2. This keeps semaphore IDs, config footprint, core maps, runtime arguments and CB layout fixed for a narrow comparison; it does not retain any of the64 startup atomic requests.

All three default software kernel strings are pinned to their pre-adaptation SHA-256 values and asserted by the probe:

- Reader: `c012327e3d272059c9b0809fab23d5dd88ad4516d02a9f76f4876f58c7c00d6f`
- Writer: `9c76c4b3fc8829aa8cffed8eb3e6f3442721a244c48072630b20b21da923c7f5`
- Compute: `8327682c24b7c9b623a93fc179324b7d407e4e39534ff7631dc51c14a255a2fd`

The receiver-skew control is deterministic, rather than a busy loop: dispatch reader0 waits for exactly31 peer DONE credits **before** reserving dense/raw. Each of the31 peer writers has already acknowledged its writes into receiver0's dense slot before sending its credit. Writer0 cannot contribute the32nd credit until its own reader publishes raw input, which happens only after this wait. Therefore the wait cannot overshoot31, and the control explicitly exercises data arrival before receiver reservation. Reader0 subsequently publishes raw, writer0 completes the final fold/scatter, and the ordinary DONE32 wait publishes dense. The test is valid at every B1..32; core0 is always an inverse owner. The control is rejected with software admission, because its GO dependency would create a cycle. It disables timing even in an uninstrumented environment.

The probe adds `--admission {software,dispatch}` and `--defer-receiver-reserve`. Existing native three-program versus candidate two-program eager/readout, inverse-col0, address-rebind,50 queued unique-output, changed-input trace groups and alternating-input replay checks remain. MinimumPCC remains .995; `--require-exact` additionally checks zero bit mismatches. When allocation tracking is active, it verifies both captured traces clean, deliberately allocates one live buffer, calls `TraceAllocationTracker.verify_before_replay` to require rejection **without executing an unsafe trace**, frees that buffer, then requires both traces clean again. No acknowledgement, corruption scope, allowlist or program-cache exemption is used. A tracking run explicitly rejects `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=1`.

Completed CPU commands:

```bash
python_env/bin/black --target-version py310 models/autoports/qwen_qwen3_6_27b/tt/hidden_norm_fused_candidate.py models/autoports/qwen_qwen3_6_27b/tests/probe_hidden_norm_fused.py
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_hidden_norm_fused --cpu-check
python -m py_compile models/autoports/qwen_qwen3_6_27b/tt/hidden_norm_fused_candidate.py models/autoports/qwen_qwen3_6_27b/tests/probe_hidden_norm_fused.py
```

Result: PASS96 descriptor cases = all32 batches × software/dispatch/dispatch-deferred. Checks retain the earlier complete dense-byte and inverse-padding ownership proof, add exact unchanged CB/runtime/config/final-program comparisons, assert default source hashes, literal compute and writer-tail preservation, absent READY/GO operations, retained data/atomic ordering, and the deferred-reader dependency graph. Bad admission names and software+deferred controls are rejected. No device library was imported. Python syntax validation is not native compilation.

Final implementation SHA-256 before parent hardware handoff:

- Helper: `29f451eeaeeb602b8ba68fe0279fe4a4d9c60e67059901cd6036104676f7a7ab`
- Probe: `e32c4ac6ef440c4a4cbdc0dcba1b1247eae22a9cf14c3a512e913c3380c343f7`

Parent commands, under its established serialized device reservation/health environment, with fresh output names:

```bash
TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_hidden_norm_fused --admission dispatch --batches 1 3 --modes normal --require-exact --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/hidden_norm_dispatch_smoke_v1.json
TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_hidden_norm_fused --admission dispatch --defer-receiver-reserve --batches 1 3 17 32 --modes normal --require-exact --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/hidden_norm_dispatch_deferred_v1.json
TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_hidden_norm_fused --admission dispatch --all-batches --require-exact --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/hidden_norm_dispatch_wide_v1.json
env -u TT_METAL_WATCHER -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE -u TT_METAL_DEVICE_PROFILER python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_hidden_norm_fused --admission dispatch --batches 1 32 --modes normal --samples 7 --require-exact --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/hidden_norm_dispatch_perf_v1.json
```

Use `--admission software` with another fresh output path for the matched software control. Hardware compilation, watcher/trace behavior, exactness, latency and integrated decoder results remain unverified for dispatch admission at this handoff. The default software path remains unchanged. No further file edits after reporting ready without parent acknowledgment.
