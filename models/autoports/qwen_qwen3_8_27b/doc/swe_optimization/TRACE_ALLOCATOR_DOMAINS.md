# Separate regular DRAM from reserved TRACE allocation

The retained backend's graph processor emits coarse placement `type=DRAM`
for both regular DRAM and TRACE buffers. It also emits numeric `buffer_type`:
DRAM0, L1=1, SYSTEM_MEMORY2, L1_SMALL3, TRACE4. The allocator creates separate
DRAM and TRACE managers; reserved TRACE capacity reduces regular DRAM capacity,
and `get_memory_view(DRAM)` excludes allocations from the TRACE manager.

`physical_memory_inventory.py` now keys events by the numeric allocator domain.
It preserves original graph parameters, reports TRACE usage separately, and
requires exact reconciliation against each explicitly supplied allocator view.
Missing/unknown numeric metadata fails. Only listed independently reconciled
domains are claimed as measured. Configured reserved capacity is distinct from
used TRACE allocation bytes and logical trace payload.

An actual B1 full64 diagnostic first failed under the old coarse-type parser.
Offline replay of its preserved constructor/preparation graphs proves:

| Allocation-event bytes per device | Constructor | After preparation |
|---|---:|---:|
| Regular DRAM |8,695,110,656|8,865,768,448|
| Fixed TRACE arena live allocations |0|25,624,576|

Regular DRAM matches the observed allocator exactly at both checkpoints. Three
TRACE buffers account for the entire25,624,576-byte former excess. Preserve
the original failed receipt; this is a corrected accounting interpretation,
not a changed model run or an unexplained-residual exemption. TRACE occupancy
from this replay is not independently verified against a TRACE allocator view;
the next diagnostic supplies that view explicitly.

Evidence outside Git:
`/home/mvasiljev/build/qwen38-swe/memory-prefix/B1_TRACE_DOMAIN_REPLAY.json`.
The two original graph SHA256 values are
`42d467096636e06447e68584df491a1e38766a3883d7c4cae33713e3064efb9c`
and `d6ce68709f48c30b760cee797856083841baa23687d036dd5a7096594d48cddb`.
They belong to `full64-b1-final-v4-strict-control.memory.{construction,preparation}.graph.json.gz`
under sibling `local-tsu-ttft` evidence, with128MiB configured TRACE capacity.

The measured B32 memory pair used trace_region_size0: its trace buffers use
regular DRAM. Its exact allocator deltas and measured DRAM trace-release bytes
remain valid. Do not subtract those buffers as though they occupied a fixed
arena, or infer B32 TRACE occupancy from the B1 diagnostic.

Source proof in retained backend3fbf: `ttnn/core/graph/graph_processor.cpp`
allocation/deallocation metadata, `tt_metal/api/tt-metalium/buffer_types.hpp`,
and `tt_metal/impl/allocator/allocator.cpp` DRAM/TRACE manager initialization.
Eight new pure CPU regressions exercise domain separation and strict failure;
the four prior owner/alias accounting tests continue to pass.
