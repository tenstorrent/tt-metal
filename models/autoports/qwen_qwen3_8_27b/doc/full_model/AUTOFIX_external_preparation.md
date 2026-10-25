# AutoFix — external preparation isolation

Contract: preparing traces must preserve caller-owned live cache, page mappings,
request inputs and RNG state, including sparse maps whose unused entries are -1.
The former full-cache snapshot doubled the external pool during warmup and ran
synthetic requests through the caller's map. A32x32768 pool alone is
10,358,358,016bytes/device; doubling it is a capacity risk, not a demonstrated OOM.
Paged fill skips UINT32_MAX, but prefill SDPA's page lookup does not, so synthetic
full-length warmup cannot safely use a map valid only for short live prefixes.

Repair: `tt/generator.py:prepare_decode` allocates one scratch tensor per named
cache field and exact TensorSpec, retaining original physical shapes, dtypes,
layouts and placement. Key/value remain distinct. Every preparation uses a same-shaped temporary map containing only valid physical
page indices, including owned caches with supplied, refreshed or directly mutated
maps. Only caller-owned caches require scratch cache storage. Warmup binds only scratch; scratch
zeroing is deduplicated locally. Public reset and the owned-cache path retain
their semantics. Original cache/map identities and seven small request/RNG/output
buffers restore before capture and again in finally. Scratch remains alive through
capture, then is released after synchronization. Capture records commands without
executing cache writes (`tt_metal/distributed/fd_mesh_command_queue.cpp`, bypass
branch in enqueue_program).

Review also found exception cleanup could call end_trace_capture twice when
finalization had already closed the queue before raising. Cleanup now attempts
known trace releases independently, clears handles, retains the original exception,
and records compact cleanup error types. The reduced probe injects failure after
a real successful record_end and then retries preparation.

`tests/probe_external_prepare.py` is the discriminating device check: sparse map,
nonzero cache sentinels, pre-capture and post-record_end injected failures, successful
preparation, original binding/address preservation, three tracked replays, exact
inactive sampled state and changed active sampled state per layer/field/device.
The full-stack variant uses B32/context33024 and full-domain preparation. Cache
checks sample rows on all devices; they do not copy the entire pool to the host.
Allocator checkpoints are snapshots, not a peak measurement. Request transient
peaks are separately measured by `full_model_memory.py`.

Source review and Python compilation pass. Reduced device validation passes in `external_prepare.json` and
`owned_sparse_{host,device,refreshed,mutated}.json`, with allocation tracking and
program-cache allocations included. `external_prepare_b32.json` passes all64 layers,
B32/context33024 and the complete bucket domain with tracking1/skip-program-cache0.
Injected pre-capture failure and retry preserve sampled cache/map/request/RNG state;
three replays update original active state and preserve inactive samples and unused
pages without recapture. The after-record_end injection remains reduced-only evidence.
Allocator checkpoints are17,815,070,720bytes/device before preparation,
18,546,934,784 with scratch bound,18,910,966,784 warmed before capture, and
18,297,559,040 after scratch release. These are not peak measurements.

Fresh independent review identified and closed independent map ownership: owned
cache plus custom map, pre-prepare refresh, and direct device mutation all now use
the same valid temporary-map preparation. `probe_owned_sparse_map.py` covers these
four routes without changing owned-cache reset semantics; all four reduced tracked device variants pass.

The first full B1 clean attempt exposed a helper type mismatch before measurement:
PyTorch reshape requires tuple(TTNN Shape). This was corrected, and all targeted
map/ownership probes plus mixed-slot and trace regressions passed before resuming.

Active cache samples after each of three replays are compared with the original
sentinels, not the preceding replay. This proves original-buffer mutation and
continued inactive preservation; it does not separately prove every cache field
changed on every consecutive step. Positions are checked after each replay.
