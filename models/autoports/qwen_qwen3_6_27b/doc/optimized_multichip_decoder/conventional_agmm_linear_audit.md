# Conventional AG+matmul on TP4 Linear: source audit

CPU-only, 2026-11-13. No native changes, no device opens, no Ring trial.
This is a repair proposal, not a performance rejection or hardware proof.

## Why removing the first guard is insufficient

1. `all_gather_async/device/all_gather_async_default_program_factory.cpp:220`
   explicitly rejects Linear with a fused signaler. Its ordinary CCL kernels
   already support Linear forwarding, endpoint connections and receive counts.
   Thus the transport itself need not become Ring.
2. `all_gather_matmul_async/device/all_gather_matmul_async_program_factory.cpp:54`
   hardcodes `num_transfers=4`. `ccl/kernel_common/worker_sync_utils.hpp:211`
   derives `num_tensor_slices=num_transfers*2`, and line228 asserts that these
   slices cover the matmul reduction blocks. For this model K5120/localK1280
   that means eight slices of40 tiles versus the actual160 tiles. Even a
   physically available TP4 Ring does not repair this eight-slice assumption.
3. `MatmulOpReceiver` in the same header alternates directions, advances slice
   indexes with wraparound, and waits on balanced per-direction counters.
   Linear endpoints have asymmetric counts: rank0 receives all three remote
   slices from its right; rank3 receives all three from its left. Merely
   changing `num_transfers=ring_size/2` still waits for absent endpoint traffic.
4. `minimal_default_reader.cpp:163–177` computes correct Linear directional
   receive counts, then lines403–410 signal matmul once per complete received
   slice. `minimal_default_writer.cpp:497–505` publishes the local slice on
   semaphore direction1. Receiver scheduling must agree with these producers.
   The first direction1 remote signal waits for the local-write handshake.
   Endpoint rank3 has no such remote receive: the handshake increment must
   be omitted or drained there to keep repeated-launch accounting explicit.

## Bounded repair proposal

Keep the proven FABRIC_1D Linear data transport and native conventional matmul
compute. Add an explicitly selected Linear receiver schedule to the existing
fused-op descriptor ABI; retain Ring's old ABI/behavior outside this branch.
Constrain the initial implementation to tiled interleaved AG output, TP4,
width gather, full local K tile slices, and Kblock dividing40. These are
representable for this model without altering the hidden-sharded residual
contract or padding the public input. Native 1D/2D projection geometry must
still meet physical L1 constraints.

Each receive entry is `(source_rank, semaphore_direction, required_count)`:

| Local rank | Schedule |
| --- | --- |
| 0 | (0,1,1), (1,1,2), (2,1,3), (3,1,4) |
| 1 | (1,1,1), (0,0,1), (2,1,2), (3,1,3) |
| 2 | (2,1,1), (1,0,1), (3,1,2), (0,0,2) |
| 3 | (3,1,1), (2,0,1), (1,0,2), (0,0,3) |

Use the source rank for both A slice offset and B reduction-slice offset;
the existing `use_in1_offset` descriptor route must carry the same schedule.
Assert total consumed K blocks equals160/Kblock, every rank occurs exactly
once, and no semaphore threshold exceeds its producer count. The same
schedule must reach both `update_current_block_start_tile_id` and
`align_to_slice_and_sync` consumers, including sender/receiver 1D configs.
For initial MULTI signal mode all matmul receivers get counters; do not infer
SINGLE mode source-index mapping is compatible without separate checks.

Do not simply remove the Linear assertion. Gate it on the new receiver ABI,
then validate worker synchronization, write visibility before signaling,
counter initialization/retirement, and endpoint handshake lifetime. Preserve
the existing reader/writer atomic barriers and persistent-output startup
ordering. A source change spans the fused factory, signaler serialization,
receiver helper, and affected matmul descriptor/kernel bindings. This is a
bounded native repair suitable for AutoDebug/AutoFix, not a Python layout
flag. Existing minimal fused AGMM remains a legal measured adaptation but
does not establish conventional AGMM performance.

Required hardware sequence after repair: rank-distinct A and B slice oracle,
all four endpoint/interior outputs, fresh-cache watcher, changed-input and
changed-output-buffer replays, allocation tracking, then 1D and2D B1/B32
coherent MLP-family measurements with the same residual contract. Measure
local-read/remote-read overlap and whole-family latency; no immediate
restoration to replicated residual is necessary.

## Physical Ring evidence

The retained `doc/multichip_decoder/device_inventory.log.gz` identifies four
P300c devices and PCI/reset IDs. It contains no Ethernet peer map, link count,
rank-to-fabric-coordinate map, or endpoint wrap link. Existing successful
Linear runs prove their used neighbor routes, not a wraparound route.
Repository UMD example cluster descriptors describe example machines and
cannot certify this host. No host-specific cluster descriptor was identified
in the model artifacts or readable task-local cache search. Consequently
physical Ring availability is **unproven**, not disproven. A host-specific
read-only fabric/ethernet topology capture during the parent's next device
setup would resolve it. Even if Ring is available, the hardcoded eight-slice
receiver issue above must first be repaired for TP4.
