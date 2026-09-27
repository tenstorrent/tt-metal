# AUTODEBUG: full-model sampler all-gather Watcher assertion

## Diagnosis

The native multicast all-gather helper initializes a scatter-write header even
when its compile-time packet geometry selects ordinary unicast writes. For the
failing 4096-byte index tile and 4352-byte fabric payload, the constructor passes
`chunk_count=1` to an API requiring 2–4 chunks. The assertion is
`tt_metal/fabric/hw/inc/api_common.h:279`, reached from the unconditional scatter
state setup in `all_gather/device/kernels/multicast_common.hpp:51`.

This is a concrete source contract violation that explains both the exact
Watcher stop and ordinary non-Watcher success. It is independent of the earlier
minimal-async all-gather endpoint fix. Guard the unused scatter state setup with
the existing `use_scatter_write` compile-time predicate; retain unicast state
setup. Both the reader and writer share this helper.

Investigation is read-only except for this report. No model, kernel, test, or
hardware state was changed by this investigator. Root owns the fix and device
verification.

## Triage evidence and limits

- `../trace_watcher_retry.log` records a four-device Blackhole mesh with
  `FABRIC_1D`. The model reaches `MODEL_READY`, then native all-gather logs a
  4352-byte fabric packet size for 2048-byte pages at 07:23:34 and for 4096-byte
  pages at 07:23:35.
- At 07:23:39 Watcher reports device 0, logical worker `(0,0)`, virtual worker
  `(1,2)`, BRISC assertion **line 279**, while running
  `ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_writer.cpp`.
  NCRISC runs `multicast_reader.cpp`. Waypoints are `NWID,NSMW,W,W,W`.
- The literal writer file has only 248 lines. Watcher explicitly says the line
  may belong to an included file. `api_common.h:279` is the matching reachable
  assertion, not a hypothetical writer line 279.
- `watcher.log` ends at completed dump 10, before the failed dump was flushed.
  That preceding dump identifies top-k reader/writer kernels. The assertion
  evidence is preserved in the process log above; the saved Watcher file does
  not itself provide a complete failed-op snapshot.
- `triage_capture.log`, `triage-summary.txt`, and `tt-triage.txt` do **not**
  contain live operation, stack, semaphore, or CB evidence. Inspector failed
  because the host had already aborted and `/tmp/tt-metal/inspector` was absent.
  The nominal capture exit status must not be interpreted as successful triage.
  This report therefore uses the already-isolated AutoDebug source fallback,
  informed by the actual Watcher stop.
- Root's subsequent explicit all-device capture, `device-triage.txt`, shows
  active Ethernet links up with heartbeats and zero retrains, and ARC rates of
  approximately 9.99 heartbeats/s on all four devices. This supports healthy
  links after the abort; it does not reconstruct lost live execution state.

## Source evidence

### Call path and backend

`tt/generator.py:40–59` constructs the common `SamplingGenerator` with
`max_top_k=32`, `allow_force_argmax=False`, and `get_tt_ccl(mesh_device)`.
`models/common/modules/tt_ccl.py` exposes semaphore services but has no
`line_all_gather` method. Consequently
`models/common/sampling/tt_sampling.py:544–568` uses `ttnn.all_gather`.
For the 1x4 mesh, `_get_sampling_cluster_axis()` returns `None`. The sampler
gathers top-k values first and top-k indices second at lines 1015 and 1041.
This matches the logged 2048-byte and 4096-byte page sequence.

The `_perform_all_gather` branch does not pass the configured `num_links` or
topology to the native call. Changing that policy is unnecessary: neither
link count nor route direction makes a one-chunk scatter command valid.
The model embedding uses `self.layers[0].gather`, a separate path.

Native `AllGatherDeviceOperation::select_program_factory` chooses multicast
for small Blackhole inputs (`all_gather_device_operation.cpp:228–282`).
The observed multicast kernel names independently establish this selection.
The sampler's gathered tensors have a padded 32-row tile height and 32
top-k columns per rank. The index tile is 4096 bytes; values are BF16 with
2048-byte tiles. The exact failed Python operation is inferred from this
source sequence plus the logged page sizes, since Inspector data is absent.

### Packet-state contract

`all_gather_device_operation.cpp:336` obtains the actual fabric maximum
payload size. `all_gather_multicast_factory.cpp:165,256,271,354–384` passes that
payload and the output chunk size into both kernels. `validate_packet_size`
prints these exact values; 4352 is the configured payload, not an estimate
including an unspecified header.

`multicast_common.hpp:187–198` computes:

| Input case | Chunk bytes | Payload bytes | `pages_per_packet` | `use_scatter_write` |
| --- | ---: | ---: | ---: | --- |
| BF16 values | 2048 | 4352 | 2 | true |
| UInt32 indices | 4096 | 4352 | 1 | false |

The header constants are `MIN=2`, `MAX=4`
(`tt_metal/fabric/fabric_edm_packet_header.hpp:196–197`). Nevertheless, the
`FabricWriter` constructor always calls
`fabric_multicast_noc_scatter_write_set_state<ChunkSizes>` with
`NocUnicastScatterCommandHeader(..., pages_per_packet)` at lines 51–58.
The linear API iterates route headers and calls
`populate_unicast_scatter_write_fields` (`linear/api.h:1641–1679`). Its
`ChunkSizes` mask requires a count in [2,4] and asserts at `api_common.h:279`.
The pointer header constructor permits count 1, so this later API assertion
is the expected first count assertion, exactly as observed.

For a larger page than the packet, `pages_per_packet` becomes zero; the same
unconditional header construction would instead violate its own nonzero-count
assertion. Conditional scatter initialization fixes the common contract for
both one-chunk and split-page unicast cases.

The constructor also initializes the alternate scatter route at lines 73–80.
It is disabled for the present linear topology, but requires the same guard.

### Route, semaphore, and CB ledger

For each mesh rank, the reader owns forward routes and the writer owns backward
routes (`all_gather_multicast_factory.cpp:354–384,466–527`). Every other rank
receives exactly one increment for each synchronization phase. Reader waits
for `N-1=3` increments and atomically subtracts 3; it repeats this for completion.
Fresh output enables the initial phase; persistent output omits that phase.

Both kernels construct `FabricWriter` **before** either barrier. At a linear
endpoint, one RISC can have zero routes and skip the per-header count assertion,
while its opposite RISC has a live route and asserts. The zero-route reader
can therefore reach its initial semaphore wait and remain at `NSMW`. The
observed BRISC assertion plus NCRISC `NSMW` is consistent with this endpoint
asymmetry. Physical device-to-mesh-rank mapping was not captured, so no exact
endpoint identity beyond the reported device/core is claimed.

The shared CB0 has depth 3. Its entry size is the input page size multiplied
by `max(1,floor(payload/input_page_size))` and a bounded tile-layout multiplier
(`all_gather_multicast_factory.cpp:273–297`). For each valid input group,
NCRISC reads one CB entry, waits for that read's transaction ID, pushes one,
and sends its forward copies. BRISC waits for one entry, sends backward copies
and the local copy, flushes, then pops one. Partial groups transfer only valid
chunks. The assertion occurs before any of this loop, so there is no observed
CB-count mismatch requiring a capacity change. Reader barrier waits, omitted
remote increments, and later host synchronization are downstream effects.

### Passing contrast and previous fix

When `use_scatter_write=false`, `async_write()` already takes its ordinary
unicast branch (lines 123–140), and `async_writes_flushed()` never sends the
scatter header. With Watcher disabled, the invalid unused header can remain
unobserved while the correct unicast payload path completes. Thus successful
ordinary full-model tests are compatible with this assertion failure.

The Stage05 fix in `minimal_default_writer.cpp:224–230` guards an absent
directional connection accessor. That backend also already seeds an unused
scatter header with a valid count at lines 322 onward. The current native
multicast helper is different and still contains the unguarded initialization;
reapplying the earlier endpoint fix would not repair this stop.

## Proposed fix

In `ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_common.hpp`,
wrap each constructor scatter-state call, including its temporary scatter
command construction, in `if constexpr (use_scatter_write)`. Apply the same
guard inside the alternate-route branch. Keep unicast state initialization
unconditional: it serves both ordinary unicast packets and the single-chunk
tail when the scatter branch flushes a partial packet.

This changes unused initialization only. It preserves the selected native
collective, assertions, topology, payload size, dtype, worker choice, barriers,
and actual transfer logic. It fixes the reader and writer together. Header
pool allocation can remain unchanged for the smallest patch. Do not disable
Watcher, inflate the payload size, or route the sampler through another CCL
backend to avoid this defect.

## Focused verification plan

1. Use a native 1x4 `FABRIC_1D` mesh with default packet configuration and
   `TT_METAL_WATCHER=5 TT_METAL_WATCHER_NOINLINE=1`. Upload distinct rank slices
   with local shape `[1,1,32,32]`, TILE layout, DRAM, dtype UInt32. Call
   `ttnn.all_gather(tensor, dim=3, cluster_axis=None,
   memory_config=ttnn.DRAM_MEMORY_CONFIG)`. Compare every rank exactly with
   the concatenation of the uploaded rank slices. This isolates the failing
   packet setup without loading the model. The preserved original assertion
   supplies pre-fix failure evidence without another intentional device abort.
2. Run the same small case with BF16 as the two-chunk scatter control. Also
   use local width 96 (three tiles) to exercise a partial final packet and
   width 32 to exercise workers assigned zero pages when two links are used.
3. Warm, capture, and replay the same gathers repeatedly; compare every rank
   after each replay. Keep source inputs and outputs alive across capture.
   Use persistent and freshly allocated output cases if the probe supports
   both. A durable CCL regression should require Watcher explicitly, since the
   original defect can return numerically correct output without assertions.
4. Run the original full-model Watcher trace contract command unchanged, then
   the applicable sampling and full-model gates. The focused native probe does
   not by itself certify the full sampler trace path.
5. Record the required C++ build-wrapper result and kernel JIT evidence. A
   shared-header change requires an actual freshly compiled kernel before
   accepting the hardware result.

## Uncertainty

The exact live stack, semaphore values, and CB counters were lost on host
abort. No such measurements are claimed. The deterministic source argument
chain proves the malformed initialization, but successful post-fix Watcher
execution and exact output/trace comparisons remain required. No performance
claim follows from this diagnosis.
