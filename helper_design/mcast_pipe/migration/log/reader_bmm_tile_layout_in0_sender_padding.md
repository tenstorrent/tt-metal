# reader_bmm_tile_layout_in0_sender_padding — v8 (remigration)

`ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp`
Role: sender. Tag: clean. Tier 0a (matmul). Commit `2dc2b7e56175e21eacfa4bea67ced8a7f23dcd38`. migrated v8 — PASS.

## Transform (v7 -> v8)
v8 SenderPipe dropped the 3rd template param `NUM_ACTIVE_RECEIVER_CORES`; fan-out is now derived from rect `area()`, and a divergent consumer-ack count moves to a runtime ctor arg.

**NOT a pure deletion — this is a DIVERGENT site.** The original v7 comment itself flagged `num_dests < area`. The mcast rect passed to the kernel is the full receiver bounding box (`start_core_noc`=top-left, `end_core_noc`=bottom-right; 1D factory ~line 924), so `area()` == `in0_mcast_receiver_num_cores` (the whole box). But the consumer-ack count is `in0_mcast_num_dests` (CTA 17) = `num_cores - 1` (active receivers, sender excluded). For the 1D interleaved mcast_in0 path, `num_cores = num_blocks_total = num_blocks_x` (factory lines 228/241), which is **less than the bounding-box area when num_blocks_x does not fill the grid** — the `uneven_width=2` cases.

Edit:
- Deleted the 3rd template arg (`in0_mcast_num_dests,`).
- Added `in0_mcast_num_dests` (CTA 17) as the ctor's 3rd arg `consumer_ack_count`.

The default `ACK_EQUALS_FANOUT` (= `area()-1`) over-counts: inactive cores in the box receive the data mcast but never ack, so the PRE_HANDSHAKE `consumer_ready.wait()` hangs.

## Symptom that exposed this
Pure deletion compiled and passed every `uneven_width=0` case but **HUNG** on `uneven_width=2 + in_sharded=False` (device timeout, cores 19-26/18-26; `system_memory_manager.cpp:757`), deterministically. After adding `consumer_ack_count` the isolated nodeid and the full suite pass.

## Validation
`test_matmul.py::test_matmul_1d_multiple_output_blocks_per_core`
- smoke (`--dev`, mcast_in0=True, grid (8,2), n=2048-k=1024-m=256, in_sharded=True): PASS
- full suite (`--run-all`): **48 passed, 16 skipped, 0 failed** (23s).
JIT-built: confirmed (`generated/inspector/programs_log.yaml`).

## Lines
Template arg removed: 1. Ctor arg added: 1 (`in0_mcast_num_dests`). diff_lines_removed (template arg): 1.

## Note
This is the matmul analogue of the dram_sharded / conv-WS "split mcast-dest vs consumer-ack" gap — but here it is EXPRESSIBLE under v8 (the point of Round 10's D2 count split). v7 could not separate the two with a single `NUM_ACTIVE_RECEIVER_CORES`; v8's runtime `consumer_ack_count` ctor arg does. The v7 entry for this kernel set `count = in0_mcast_num_dests` as the template arg, which (incorrectly) drove the data-mcast fan-out to num_cores-1 too; v8 lets the data mcast use the full rect area while keeping the smaller ack count.
