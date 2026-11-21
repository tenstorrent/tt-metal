# DFB imbalance audit — sub-family: sharding / reshard / to_memory_config / move / reallocate / copy / clone

Paths relative to ttnn/cpp/ttnn/operations/ unless noted.

## Dispatch notes (Quasar)
- `ttnn.to_memory_config` (mainline, core/to_memory_config/to_memory_config_op.cpp) -> mainline prim::interleaved_to_sharded / prim::sharded_to_interleaved / ttnn::reshard / prim::copy. No Quasar reroute.
- `ttnn.experimental.quasar.to_memory_config` (experimental/quasar/to_memory_config/to_memory_config_op.cpp) -> prim::qsr::sharded_to_interleaved, quasar::interleaved_to_sharded, quasar::reshard, and the fallback is mainline `ttnn::prim::copy` (with or without dtype).
- `ttnn.interleaved_to_sharded` / `ttnn.sharded_to_interleaved` / `ttnn.reshard` -> mainline data_movement/sharded/** Metal2 factories (they have explicit is_quasar branches). The `experimental.quasar.*` variants -> experimental/quasar/{interleaved_to_sharded,sharded_to_interleaved,reshard}.
- `ttnn.experimental.quasar.reallocate` -> quasar::move -> MoveDeviceOperation: MULTI_CORE_SHARDED (sharded input) -> MoveShardedProgramFactory (no DFB); MULTI_CORE_OVERLAP -> MoveOverlapProgramFactory (Metal2); MULTI_CORE -> reuses mainline `CopyDeviceOperation::SameMemoryConfig::create_descriptor` (legacy ProgramDescriptor + CBDescriptor, NOT ported to Metal2).
- `ttnn.reallocate` (yolo test_reallocate) -> mainline data_movement/move (all three factories are legacy ProgramDescriptor; whether they build/run on Quasar is outside this audit's scope; counted anyway).
- `prim::copy` select: SameMemoryConfig (legacy descriptor) if in/out mem configs equal and not ND; else DefaultRowMajor (RM) / DefaultTilized (TILE), both Metal2.
- `ttnn.clone` -> data_movement/clone (Metal2).
- `experimental.quasar.to_device` -> host `tensor.to_device()`; no device kernels.

## Findings

### F1 CONFIRMED (cross-kernel, cat. 3): mainline sharded_to_interleaved, ROW_MAJOR, partial last height shard
- Factory: data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp
  - reader args: line ~244-248 `num_tiles_per_core = num_units_per_shard` (= shard_spec.shape[0] for RM) on every used core.
  - writer RM args: line ~268-282 `block_height = shard_height = min(shard_h, tensor_h - h0)`.
- Kernels:
  - eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp:25 `dfb.push_back(num_tiles_per_core)` (DFB `in`, borrowed from INPUT, producer).
  - data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2.cpp:38/51 `wait_front(block_height)` / `pop_front(block_height)` (consumer).
- Trigger: ROW_MAJOR input, HEIGHT_SHARDED or BLOCK_SHARDED, and tensor_h % shard_h != 0 → the last-height-shard core(s): pushed = shard_h, popped = tensor_h - h0 < shard_h. finish() on the reader waits forever for (shard_h - shard_height) acks.
- Same imbalance on OUT DFB if convert_df (compute eltwise_copy_metal2 pushes per_core_tile_cnt = num_units_per_shard CTA; writer pops shard_height) — RM+convert is unusual.
- TILE path is balanced (writer uses block_num_tiles = num_units_per_shard for wait/pop and only iterates the unpadded tiles).
- Exercised? Not by the captured tests I found: llama/qwen graph_ops s2i cases are TILE WIDTH_SHARDED; yolo clone path uses TILE. SUSPECT for model runs with non-divisible RM height shards.
- Minimal fix: writer RM `wait_front/pop_front(num_units_per_shard)` (pass a separate `block_num_units` runtime arg = num_units_per_shard, keep the loop over `block_height`) — mirror of the TILE writer. Alternatively make reader push `shard_height` for that core (requires per-core reader args).

### F2 CONFIRMED (cross-kernel, cat. 3): experimental/quasar sharded_to_interleaved, ROW_MAJOR, partial last shard
- Factory: experimental/quasar/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp
  - reader: line ~222 `reader_run.runtime_arg_values["num_units"][core] = num_units_per_shard` for all used cores.
  - writer RM: lines ~282-305: `shard_height = num_units_per_shard_height_last` for end core (HEIGHT_SHARDED) / end row (BLOCK_SHARDED), passed as `block_height`.
  - compute (convert_df): `num_units = num_units_per_shard` (line ~333).
- Kernels: experimental/quasar/sharded_to_interleaved/device/kernels/dataflow/reader_unary_sharded.cpp:17 `cb_in0.push_back(num_units)`; writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:29/42 `wait_front/pop_front(block_height)`.
- Trigger: RM, HEIGHT/BLOCK sharded, num_units_height % shard_h != 0 → end core pushes shard_h, pops shard_h_last. Hang in reader finish().
- Exercised? resnet tests/ops/test_sharded_to_interleaved.py (shard_h 255/256/1568/6272 × 1-2 cores) and test_to_memory_config sharded→interleaved use exact multiples → NOT triggered there. Real path: resnet conv2d_DRAM uses quasar.to_memory_config(RM HEIGHT_SHARDED → DRAM). If a conv activation's shard height was rounded up (e.g. NHW not divisible by num_cores*32-ish), it WILL hang under #57646. SUSPECT for model runs.
- Minimal fix: same as F1 — writer RM wait/pop the full `num_units_per_shard` (new RT arg) while writing only `block_height` sticks; or set reader `num_units` to shard_height for the last core.

### F3 SUSPECT/legacy (cat. 3): copy SameMemoryConfig, RM sharded + dtype conversion
- data_movement/copy/device/copy_same_memory_config_program_factory.cpp:~180-198 + 205-220: RM sharded reader (`reader_unary_stick_start_id.cpp`) pushes `num_units_per_core * num_shards` (`full_input_row/input_unit_size`), but compute eltwise_copy CTA = `num_units_per_core` → if num_shards>1 and convert_dtype, compute pops fewer and writer waits more than compute pushes. Already a hang pre-#57646; legacy ProgramDescriptor path (reached on Quasar via quasar move MULTI_CORE only without dtype change), so not exercised. No action needed for #57646.

### Notes (cat. 4, benign for finish())
- Push-without-reserve on borrowed shard DFBs: reader_unary_sharded_metal2.cpp:25 (mainline) and experimental/quasar/sharded_to_interleaved/.../reader_unary_sharded.cpp:17. Push count matches consumer pops (except F1/F2), so finish() is satisfied; no reserve needed since the borrowed DFB is sized to the shard.
- Reshard (generic, generic diff_width, same_width, same_height; mainline + quasar) kernels use only `get_write_ptr/get_read_ptr` on the shard DFB and scratch DFB — zero reserve/push/wait/pop → posted==acked==0, finish() trivially satisfied.
- nd_reshard_copy_local_shards.cpp (mainline + quasar): no DFB at all.
- copy/device/kernels/writer_unary_start_id.cpp:19-20 `#ifdef OUT_SHARDED wait_front(num_tiles)` with no pop — dangling wait, but OUT_SHARDED is never defined by any factory that uses this kernel (SameMemoryConfig only adds BACKWARDS). Dead code; would be a cat. 2 imbalance if ever enabled. Fix (hygiene): add `pop_front(num_tiles)`.

## Balanced (checked)
- mainline interleaved_to_sharded factory: reader_unary_sharded_blocks_interleaved_start_id_metal2 (reserve/push block_num_tiles=curr_num_units_per_shard) ↔ writer_unary_sharded_metal2 / writer_unary_sharded_blocks_start_id_metal2 (wait/pop num_units / block_width_padded_num_tiles = same value) ↔ eltwise_copy_metal2 (per_core_tile_cnt same). RM: reader_unary_stick_layout_..._metal2 (reserve/push block_height) ↔ writer num_units / block_height (curr_num_units_per_shard updated together with shard_height). Balanced.
- experimental/quasar interleaved_to_sharded: same structure (reader_unary_sharded_blocks_interleaved_start_id, reader_unary_stick_layout_..., writer_unary_sharded, writer_unary_sharded_blocks_start_id, writer_unary_sharded_stick_layout_start_id, compute eltwise_copy) — balanced.
- mainline sharded_to_interleaved TILE path (reader_unary_sharded_metal2 ↔ writer_unary_sharded_blocks_interleaved_start_id_metal2, block_num_tiles = num_units_per_shard; ttnn/kernel/compute/eltwise_copy_metal2 CTA same) — balanced.
- experimental/quasar sharded_to_interleaved TILE path (reader_unary_sharded ↔ writer_unary_sharded_blocks_interleaved_start_id, block_num_tiles = num_units_per_shard; compute eltwise_copy num_units same) — balanced.
- nd_reshard_copy_pages_reader/_writer (mainline + quasar): reserve/push and wait/pop 1 per page over identical [start_page,end_page) — balanced.
- reshard generic / same_width / same_height (mainline + quasar): no counter ops — balanced (0/0).
- quasar move overlap: move_interleaved_with_overlap(.cpp/_writer.cpp), move_stick_layout_interleaved_with_overlap(.cpp/_writer.cpp): reserve/push num_pages ↔ wait/pop num_pages, same RT args per core — balanced.
- quasar move sharded: reader_unary_local_l1_copy_backwards.cpp — no DFB.
- mainline move (legacy): overlap kernels self-loop reserve/push then wait/pop num_tiles — balanced; sharded backwards copy — no DFB.
- copy DefaultTilized: reader_unary_interleaved_start_id_metal2 ↔ writer_unary_interleaved_start_id_metal2 (non-OUT_SHARDED branch) ↔ sharded/.../compute/eltwise_copy_metal2 (per_core_tile_cnt) — same num_pages per core — balanced.
- copy DefaultRowMajor parallel: redistribute_pages_row_major_parallel_reader/_writer, 1 per unit over identical ranges — balanced.
- copy DefaultRowMajor non-parallel: redistribute_pages_row_major_reader (one reserve at output-subblock start, one push at output-subblock end; traced both branches) ↔ redistribute_pages_row_major_writer (one wait/pop per output subblock) — balanced.
- copy SameMemoryConfig (legacy): reader_unary_start_id ↔ writer_unary_start_id (tile), reader_unary_stick_start_id ↔ writer_unary_stick_start_id (RM sharded, num_shards equal for same mem config), ttnn/kernel/dataflow/reader/writer_unary_stick_layout_interleaved_start_id (RM interleaved), ttnn/kernel/compute/eltwise_copy (CTA per group) — balanced except F3.
- clone: read_kernel/write_kernel (+_rm, _sharded, _rm_sharded) per-unit reserve/push ↔ wait/pop with identical per-core counts; compute_kernel.cpp uses compute_kernel_lib::copy PerTile wait/pop + reserve/push for num_tiles CTA = same per-group count (convert only on TILE, validated) — balanced.

## Coverage
Factories checked: 17 (i2s mainline+quasar, s2i mainline+quasar, reshard 5 mainline + 5 quasar, quasar move overlap/sharded/multicore→copy, mainline move 3, copy 3, clone 1). Kernels checked: ~45. Findings: 2 CONFIRMED (F1, F2 — same RM partial-last-shard bug in mainline and quasar s2i; not hit by current per-op test shapes), 1 SUSPECT legacy (F3), 1 dead-code dangling wait (OUT_SHARDED).
