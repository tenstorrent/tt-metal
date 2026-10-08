# Galaxy trace I/O lifetime

Create the model, sampler and generator before recording traces. The initial
warmup prepares both host-logit and device-sampling decode modes, the supported
decode input layouts, and the prefill buckets. Preparation closes at the first
capture. Adding an unprepared variant requires releasing all related traces and
reinitializing the model and generator.

Preparation follows the model's warmup order. With the Blackhole Qwen
prefetcher enabled, prefill must compile before decode so decode's persistent
program buffers do not obstruct prefill's static L1 circular buffers. Both
phases still finish before capture. A decode-first request defers prefill
capture until it is needed; it does not change this preparation order.

Open Blackhole meshes with `l1_small_size=16384` when using the Qwen prefetcher.
Cached collective semaphores then use the separate pool at the top of L1.
Without it, semaphores allocated below the live GCB survive mode switches and
collide with prefill's static CBs. Traced preparation rejects a missing pool
before warming or capturing programs. The Qwen demo configures this pool on
Blackhole automatically.

Persistent inputs and boundary outputs use DRAM. Prefill inputs share storage
only when their tensor specs and mesh topologies match. Decode inputs follow the
same rule; changing the decode variant reloads all host inputs. In particular,
all compatible decode modes share the token buffer used for sampler feedback.
Decode outputs share within one output mode, separately from prefill outputs.
These groups also keep buffers from different sub-device manager domains apart.

Sharded decode inputs have persistent DRAM backing. The trace stages their
contents into transient L1 buffers, runs the model, and copies incremented
positions back to DRAM. Page-table-only updates preserve device-produced tokens
and positions. Prefill can use its static L1 circular buffers between decodes
without overwriting the persistent state.

Warmup and capture run the same boundary copies into explicit output buffers.
The first decode preparation establishes the GCB data and configuration
addresses. The model releases both allocations before it loads the prefill
sub-device manager. After it loads the decode manager, it recreates both
allocations at the saved addresses before any other decode allocation.
If either range is occupied, reconstruction fails before configuration writes
or trace replay. Other allocations can use the released ranges during prefill;
they must release those ranges before the next decode switch.
The model acknowledges only the two GCB allocations as corruptible and reuses
the captured traces. Model capture has no broad allocation-tracker exemption;
unexpected surviving allocations are errors.

Device tensors returned by traced forward calls are **borrowed**. Consume them
before the next call that writes their shared group. Keeping a Python reference
does not retain an earlier value. For retention, copy into separate device
storage allocated before capture, or read to host. Queue an asynchronous host
read on CQ0 before the next write and wait for its event before consuming it.
Completed host reads are independent snapshots. Calls on a generator are
serialized on CQ0; sharing one generator between concurrent requests requires
external ordering.

Use `ttnn.empty_like(output)` for retained storage, and copy with
`ttnn.copy(output, retained, sub_core_grids=model_args.sub_core_grids)` while the
matching model mode is active. Warm that copy before capture. The worker grid
keeps the copy within Galaxy's active sub-device; its wide-row staging is bounded
so vocabulary-sized logits fit alongside the GCB. Warm any eager sampling path
used for comparison before capture too: omitting its optional output selects a
different program-cache entry from traced sampling's explicit feedback output.
