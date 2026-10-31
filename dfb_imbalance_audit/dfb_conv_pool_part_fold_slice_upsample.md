## Part: fold / padded_slice / slice_write / upsample (mainline)

Q = ttnn/cpp/ttnn/operations/experimental/quasar ; M = ttnn/cpp/ttnn/operations

### FINDINGS

#### F1 CONFIRMED (latent; NOT exercised by the current tests): slice_write RM sharded-input, last/tail cores pop fewer than the reader pushes
- Factory: Q/slice_write/device/slice_write_rm_sharded_input_program_factory.cpp:135-139, 182-202
- Kernels: Q/slice_write/device/kernels/dataflow/slice_write_reader_sharded.cpp:17-18 (producer), slice_write_writer_interleaved.cpp:51-75 (consumer)
- DFB: SW in_dfb (in0), borrowed from the sharded input (num_entries = shard height S).
- Reader: reserve_back(S)+push_back(S). The factory passes `reader_kernel_args = {num_sticks_per_core}`, which is the FULL shard height S, unclamped.
- Writer: n = num_sticks_this_core = min(S, max_num_sticks_this_core+1), which can be < S and can be 0. B = num_read_per_barrier = S / merge_num_sticks_to_read(...), so B divides S. The loop runs ceil(n/B) iterations, and each does wait_front(B)/pop_front(B). Pops = ceil(n/B)*B <= S.
- Imbalance: on any core where n < S (a partial tail core, or a core past the end of the slice region with n=0), pushes = S but pops = ceil(n/B)*B. The S - ceil(n/B)*B posted entries are never acked, so the producer hangs in finish() under #57646.
- Exercised? test_slice_write.py uses `_fit_cores`, which picks exact, unpadded shards, so n == S on every core and the test is balanced. The resnet conv DRAM-slicing path uses output_layout=TILE, so it routes to the TiledSharded factory (balanced). It is reached only by RM, HEIGHT- or BLOCK-sharded slice_write inputs whose shard height is padded relative to the slice (e.g. RM-output conv/pool through op_slicing).
- Fix options (any one):
  - In the factory, set the reader push to the writer's pop count: `reader_kernel_args = {n == 0 ? 0 : div_up(n, B) * B}`.
  - In the writer, after the loop, drain the remainder by passing `num_pushed` and doing wait_front/pop_front of (num_pushed - popped).

### BALANCED / OK (traced)
- fold MultiCore (sharded), Q/fold/.../writer_cb2s_row_major.cpp. The one source is used as both reader and writer kernel. SRC0/DST0 are borrowed and reached only through get_read_ptr/get_write_ptr, with no reserve/push/wait/pop: 0/0 on each side, so finish() is trivial.
- fold MultiCoreDRAMFold tiled: reader_dram2cb_tiled.cpp pushes tc per (sb, local_h, w_tile), i.e. nb*sh*tw*tc. compute/untilize.cpp (kernel_lib untilize, WaitBlock) has per_core_block_cnt = nb*sh*tw (main and cliff CTAs match the reader/writer num_blocks RTAs) with block = tc tiles; all paths in untilize_helpers.inl:227-270 do wait/pop/reserve/push exactly block_width per block. writer_cb2dram_for_tiled_input.cpp does wait/pop tc per (sb, local_h, w_tile). SRC2 self-loop: the writer is bound as producer and consumer but only uses get_write_ptr as scratch, with no DFB ops (0/0).
- fold MultiCoreDRAMFold RM: reader_dram2cb_for_rm_input.cpp pushes 1 per patch and writer_cb2dram_for_rm_input.cpp pops 1 per patch, both with work_per_core = the same CTA. The SRC1 scratch (USE_SCRATCH_SRC1, self-loop) uses get_write_ptr only (0/0). Separate correctness issue, not a DFB-count problem: work_per_core = div_up(total, ncores) is applied to every core, so cores past total_patches reuse stale src/dst indices and the last partial core walks past the end of the output (fold_multi_core_dram_program_factory.cpp:~300, ~425).
- padded_slice RM: padded_slice_reader_rm_interleaved_start_id.cpp (aligned and non-aligned TRID paths) with Q/interleaved_to_sharded/.../writer_unary_sharded.cpp. The factory sets num_sticks_per_core = num_sticks_per_core_read = num_read_per_barrier = this_core_num_sticks (clamped; 0 on empty cores). That gives exactly 1 iteration, which pushes n; the writer does wait_front(n)/pop_front(n), with num_units = the same n. For n=0, both sides do nothing.
- padded_slice Tile factory (padded_slice_tile_program_factory.cpp:480-516): legacy CreateKernel/CB (Metal-1), not a DFB program and rejected on Quasar. conv2d forces RM input, so this factory is not selected. Not analyzed.
- slice_write TiledSharded: reader pushes num_tiles_this_core; the writer has B=1 and iterations = num_tiles_this_core, so it pops the same count. Balanced (factory lines 179-201).
- slice_write RMInterleaved (slice_write_reader_interleaved.cpp + slice_write_writer_interleaved_strided.cpp): legacy CreateKernel/CreateCircularBuffer, not reachable as a DFB program on Quasar (op_slicing passes sharded slice outputs). Skipped.
- upsample (mainline, yolo test_upsample.py): the input is RM DRAM-interleaved with integer scale 2, so the path is INTEGER_OPTIMIZED and MultiCoreInterleavedProgramFactory (RM, no compute). reader_upsample_unary_stick_layout_interleaved_start_id.cpp pushes 1 per page with num_pages = blocks*1. writer_upsample_interleaved.cpp does wait/pop num_tiles_per_block_row=1 per block with num_blocks = blocks. Balanced. The tiled variant (reader + untilize_metal2.cpp + writer: blocks*tiles_in_row each, g1/g2 CTAs match the RTAs) is also balanced but not exercised.
- upsample sharded (yolo test_upsample_sharded, HEIGHT_SHARDED RM): MultiCoreShardedProgramFactory with writer_upsample_multi_core_sharded.cpp run as 2 kernels. in0/out0/config are all borrowed and accessed through pointers plus async_read; there are no reserve/push/wait/pop calls (0/0).
- upsample bilinear / nearest_float: not selected by the yolo tests (mode=nearest, integer scale, no ND shard). Not analyzed.

### Reachability note (legacy DataMovementKernel => TT_FATAL on Quasar, kernel.hpp:531)
- upsample: interleaved, sharded and nearest_float factories are Metal2 ProgramSpec/KernelSpec (DFB). Bilinear factory is legacy CreateKernel/CB => UNREACHABLE on Quasar.
- padded_slice Tile factory and slice_write RMInterleaved factory: legacy CreateKernel/CB => UNREACHABLE on Quasar.
