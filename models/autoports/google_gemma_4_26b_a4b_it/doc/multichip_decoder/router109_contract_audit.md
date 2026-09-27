# Router core (10,9): contract audit

**Verdict:** no new shape, routing, batch-slot, or trace-buffer contract defect
was found in the selected TP4 path. The change is a supported model-local
placement workaround on the tested 11×10 Blackhole grid, not a native fix or a
reservation of an otherwise-unused core. The coordinator added an explicit
minimum-grid guard during this audit. Alternate/automatic worker layouts still
need their own validation.

Initially audited and hardware-validated placement runtime SHA256:
`f57234ba01a6a6e20e3188ea7690f8508c5a5ca6c075b1fba56bf1b6df28a44b`.
Final source inspection also covers the new grid guard in runtime
`c96c935847572085ef97b67ebd76c79eaaf31a45b3f6b90277d69d4951e3fc8d`.
Line references below otherwise describe the first hash. The newer source also
contains dormant `sharded_decode_rope` and `moe_ccl_bfp8` controls, selected by
`getattr(..., False)`; they are candidates under measurement, not defaults or
extensions of this audit's hardware evidence.
CPU/read-only source inspection; no TTNN imports, hardware, resets, code edits,
or native changes. This Markdown file is the only audit output.

## Placement and native contract

`tt/multichip_decoder.py:712` replaces the router memory configuration with a
single-core HEIGHT_SHARDED L1 shard at `(10,9)`, shape `(32,32)`, ROW_MAJOR
orientation. It moves **all four** persistent tensors: BF16 bias, UINT16 input
indices, BF16 output values, and UINT16 output indices. Dtypes, layouts, tensor
shapes, score centering, top-8 selection, softmax, expert IDs, and math policy
are unchanged. The four 16-bit 32×32 physical shards still total 8 KiB per
layer/device before allocator overhead; relocation does not add persistent
buffer copies after setup temporaries are released.

`GeneralizedRouter.__call__` in `tt/optimized_decoder.py:917` places each new
input face into `self.memory` before invoking the native gate. The native
validator requires matching input/bias/index shard shapes and grids, and
matching output/value-index grids; all five tensors meet this contract.
`generalized_moe_gate_program_descriptor_builder.cpp:62` takes its worker grid
from the input shard, and places its reader, writer, and compute kernels there.
It does not require origin `(0,0)`.

Both native outputs are copied to interleaved L1 before slicing/viewing the
eight selected values/IDs. Expert consumers therefore do not assume that the
native output remains on core 0 or 1. `enable_indexed_decode` receives the same
router object after relocation, so retained decode indices are not attached
to the discarded original buffers. Setup movement uses `to_memory_config`,
not the unsupported same-core `clone` path seen in diagnosis.

## Actual fixed grids in the selected decode path

All ranges below start at `(0,0)`; `(10,9)` is outside each one.

| Operation | Grid | Source |
| --- | --- | --- |
| QKV, sliding / full | 8×8 / 8×6 | `multichip_decoder.py:94` |
| Native decode SDPA | 8×8 | `optimized_decoder.py:1180`, default retained by TP4 |
| Attention output projection | 11×8 | `multichip_decoder.py:688` |
| Router projection | 4×1 | `optimized_decoder.py:829`, default retained by TP4 |
| Indexed expert gate/up | 6×2 | `multichip_decoder.py:763` |
| Indexed expert down and weighted mix | 11×8 | `multichip_decoder.py:764` |
| Shared geometry 1 gate/up and down | 11×4 | `multichip_decoder.py:250` |
| H2816 sharded normalization | 11×8 | `GemmaRMSNorm._build_sharded_cfg`, `rms_norm.py:52` |

The norm helper maximizes a rectangular divisor of 88 width tiles; on 11×10
it selects 11×8. EP-only expert grids (11×4 and 11×8) and shared geometry 2
(9×2 gate/up, 11×4 down) also exclude this core by source. The explicit DRAM
attention/shared projections use a single worker row, so their declared input
and output grids do not include it either. These source exclusions are not
new hardware acceptance results for those optional policies.

**No global exclusion exists.** Data-movement factories can split work over the
whole device grid. For example, tiled row-invariant permute partitions its
input tiles over the full grid; the expert `(0,2,1,3)` permutation preserves W
and does not select that factory's optional H/W compute transpose. Automatic
elementwise kernels, other permutations, shared geometry 0, and optional
distributed-norm paths are not governed by the fixed-grid table. Fused AGMM
also places collective workers starting at `(0,8)` and creates semaphores over
the full grid. Thus “outside the selected expert/norm compute grids” is the
accurate scope; “globally unused/dedicated hardware core” would be too broad.

## Batch, output sharding, and trace lifetime

- `OptimizedDecoder.decode_forward:737`, inherited by TP4, executes a complete
  B1 forward per request slot in a fixed sequential loop before concatenating
  outputs. Each slot consumes the router's copied values/IDs and finishes its
  expert/tail dispatches before the next slot reuses native output buffers.
  Heterogeneous positions/page-table rows remain per-slot inputs. The move does
  not introduce a new shared-output alias between slots.
- Each decoder/layer owns its own router object and buffers. Mixed-stack layers
  may use the same physical core sequentially but do not share native output
  tensor objects. The existing per-layer 8 KiB footprint still accumulates
  with layer count; a two-layer pass is not a full-stack L1-capacity proof.
- Relocation occurs in `from_state_dict`, before warmup/capture. Decode does
  not mutate the grid or relocate these four persistent buffers. Their device
  writes and subsequent copies are recorded by trace capture; replay does not
  need Python to update `last_decode_indices`. Relocating or replacing buffers
  after capture would require recapture, as for any captured persistent tensor.
- The public output contract remains replicated BF16 `[1,1,S,2816]` for the
  selected path. The optional hidden-width-sharded residual path gathers the
  normalized router input and still receives interleaved values/IDs, so no new
  sharding mismatch was found. Its alternate norms/collectives are a separate
  workload, however, and their success is not established by the selected-path
  results. Concurrent executions of one decoder on independent queues already
  share mutable router/cache state; this audit covers the existing serialized
  CQ0 contract, not reentrancy.

## Precise remaining limits and evidence

1. **Device requirement, now guarded:** `(10,9)` requires available logical
   compute dimensions at least 11×10. The first hash only checked mesh shape
   `(1,4)` and existing projections required 11×8, which did not imply ten
   rows. The coordinator has now added, immediately after the mesh-shape check,
   `workers = mesh_device.compute_with_storage_grid_size()` and rejection when
   `workers.x < 11 or workers.y < 10`, with an explicit router-placement error.
   Source inspection confirms the guard. This resolves delayed failure on an
   undersized worker grid; it does not reserve any cores or generalize the
   workload to other architectures. No code edit was made by this audit agent.
2. **Geometry dependence:** there is no allocator/sub-device reservation
   preventing a future math grid from including this core. New geometry,
   fallback backends, larger batched kernels replacing the sequential loop, or
   overlapping queues must not inherit a blanket determinism claim. Recheck
   actual worker placement and the original replica/replay gate for each.
3. **Adjacent validation:** integrated heterogeneous B32/cache-slot tests,
   Watcher, and final chosen geometry remain acceptance work for the coordinator. Old B32
   artifacts with core 0/1 do not prove the new placement. No new batch test was
   run here.

Read integrated artifacts `sliding_router109_integrated_bf16.json`,
`sliding_router109_integrated_bfp8.json`, and
`full_router109_integrated_bfp8.json`: all report the `f57234ba` runtime hash and passing
ordinary paired 4096/128 trace/cache checks. `stack_router109_bfp8.json` passes
layers `[0,5]` in one trace with direct handoff; its input length is **33**, not
4096. Diagnostic core109 output/physical-I/O/1024-replay passes, restored-core1
failure, and core1 allocation-roundtrip failure are recorded in
`AUTOFIX_full_router1_bfp8.md`. Together these support placement rather than
fresh allocation as the effective intervention. They do not identify a native
state register, kernel race, or silicon mechanism.

Audit checks: parsed the current Python source without importing it, verified
the runtime hash and artifact provenance, and inspected native gate validation
and factory placement. Documentation-only output needs no build.
