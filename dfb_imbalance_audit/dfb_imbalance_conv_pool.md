# DFB imbalance audit — family conv_pool

## (a) COLLECT

Q = ttnn/cpp/ttnn/operations/experimental/quasar ; M = ttnn/cpp/ttnn/operations

Test usage: resnet50/quasar -> ttnn.experimental.quasar.{conv2d,max_pool2d,avg_pool2d,fold,padded_slice,slice_write}
(ttnn.conv2d/ttnn.max_pool2d only in WH-control tests or `if is_quasar() else` fallbacks).
llama32_1b_quasar -> quasar.max_pool2d (1 site). qwen3_vl_ops -> none.
yolo_ops -> MAINLINE ttnn.conv2d, ttnn.max_pool2d, ttnn.upsample (no quasar redirect found).

### Quasar conv2d (Q/conv2d) — sharded factory conv2d_op_sharded_program_factory.cpp:1624-1653, 2358
- reader: reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2_metal2.cpp (block sharded)
          reader_conv_activations_padded_with_halo_3x3_weights_v2_metal2.cpp (height sharded)
          reader_depthwise_conv1d_metal2.cpp (1D depthwise)
- compute: conv_bmm_tilize_metal2.cpp (fused) | conv_tilize_only_metal2.cpp (TT_METAL_QSR_CONV_SPLIT_PROGRAM) |
           conv_unpack_tilize_probe_metal2.cpp (TT_METAL_QSR_CONV_UNPACK_TILIZE) | compute_depthwise_conv1d_metal2.cpp
- writer: writer_tiled_out_2d_mcast_{sender,receiver}_..._metal2.cpp (BS) | reader_writer_tiled_out_1d_mcast_{sender,receiver}_..._metal2.cpp (HS)
- drain: drain_out_metal2.cpp (split program A)
- width-sharded factory: conv_bmm_tilize_metal2.cpp + activation_reader_width_sharded_metal2.cpp + weights_reader_width_sharded_metal2.cpp
- sub-ops called by quasar conv2d: quasar.halo, quasar.tilize/to_layout/move (other family), quasar.matmul::linear (other family: 1x1 + split path), quasar op_slicing -> padded_slice + slice_write (DRAM slicing)

### Halo
- Q/halo: halo_gather.cpp + compute/pack_untilize.cpp
- M/sliding_window/halo (mainline ttnn.conv2d / ttnn.max_pool2d): halo_gather.cpp + pack_untilize.cpp

### Pool
- Q/pool_generic: reader_pool_2d.cpp, reader_mpwi.cpp, compute_pool_2d.cpp, compute_mpwi.cpp
- M/pool/generic (yolo ttnn.max_pool2d): same 4 names

### Fold (Q/fold)
- fold_multi_core (sharded): writer_cb2s_row_major.cpp (both reader+writer kernel use it)
- fold_multi_core_dram tiled: reader_dram2cb_tiled.cpp, compute/untilize.cpp, writer_cb2dram_for_tiled_input.cpp
- fold_multi_core_dram RM: reader_dram2cb_for_rm_input.cpp, writer_cb2dram_for_rm_input.cpp

### padded_slice (Q)
- RM factory: padded_slice_reader_rm_interleaved_start_id.cpp + Q/interleaved_to_sharded/.../writer_unary_sharded.cpp
- Tile factory: LEGACY CreateKernel (M sliding_window pack_untilize + experimental/padded_slice kernels) — conv2d forces RM input so not selected on resnet path

### slice_write (Q)
- RMSharded: slice_write_reader_sharded.cpp + slice_write_writer_interleaved.cpp
- TiledSharded: slice_write_reader_sharded.cpp + slice_write_writer_interleaved.cpp
- RMInterleaved: slice_write_reader_interleaved.cpp + slice_write_writer_interleaved_strided.cpp

### op_slicing (Q/op_slicing) — host-only orchestration, no kernels.

### Mainline conv2d (yolo ttnn.conv2d) M/conv/conv2d/device
- conv_bmm_tilize.cpp, compute_depthwise_conv1d.cpp, reader_conv_activations_{2d_mcast_,}padded_with_halo_3x3_weights_v2.cpp,
  reader_depthwise_conv1d.cpp, writer_tiled_out_2d_mcast_{sender,receiver}..., reader_writer_tiled_out_1d_mcast_{sender,receiver}...,
  activation_reader_width_sharded.cpp, weights_reader_width_sharded.cpp, conv_reader_common.hpp

### Mainline upsample (yolo ttnn.upsample) M/pool/upsample/device
- sharded: writer_upsample_multi_core_sharded.cpp ; interleaved: reader_upsample_unary_stick_layout_interleaved_start_id.cpp + untilize_metal2.cpp + writer_upsample_interleaved.cpp
- nearest_float: reader/writer_upsample_nearest_float.cpp ; bilinear: reader_bilinear_multi_core_sharded.cpp + bilinear.cpp

## (b) ANALYSIS
(sections appended below)

### Note on finish() semantics (local tree, tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:274-320)
finish() only checks posted vs acked: DM: read_posted==read_acked on every TC; TRISC unpack/pack: TC occupancy (posted&0xFFFF)==0.
reserve_back() never posts (reserve_back_impl:143 only spins on free space), so a *dangling reserve_back by itself does not hang finish()* in this tree.
What hangs: (i) entries pushed (or implicit-sync NoC-posted) but never popped; (ii) a consumer that wait_front()s a persistent tile and never pops it.
Implicit-sync NoC reads/writes into a DFB DO post (commit_implicit_read) — relevant only where disable_dfb_implicit_sync_for_all is false.

## Pool — Quasar pool_generic (resnet stem max_pool2d, global avg_pool2d TILE, llama max_pool2d)
Factory: Q/pool_generic/device/pool_multi_core_program_factory.cpp (readers: disable_dfb_implicit_sync_for_all=true, :1176/:1192).

P1 CONFIRMED HANG — in_scalar_cb (DFB_IN_SCALAR_0), one_scalar_per_core==true
  - reader_pool_2d.cpp:330-341 reader pushes 1 (per reader thread/lane).
  - compute_pool_2d.cpp:~190 `if constexpr (one_scalar_per_core) in_scalar_cb_0.wait_front(1);` — NO pop_front anywhere (pop only in !one_scalar_per_core branch at end of per-stick loop).
  - counts: pushed 1 / popped 0 per lane -> reader DM finish spins (acked!=posted) and compute UNPACK finish spins (occupancy 1).
  - Exercised: YES — every max_pool2d (one_scalar_per_core always true for MAX) incl. resnet stem maxpool (ttnn_functional_resnet50_large.py:597/875, test_stem_maxpool, test_max_pool2d*, test_maxpool_hang), llama max_pool2d; also avg pool w/o padding (global avgpool).
  - Fix: at end of compute kernel_main: `if constexpr (one_scalar_per_core) { in_scalar_cb_0.pop_front(1); }`.

P2 CONFIRMED HANG — out_cb (DFB_OUT) compute self-loop, TILE output (OUTPUT_TILED / is_output_tiled)
  - compute_pool_2d.cpp: out_cb.reserve_back(in_ntiles_c) (per 32-stick group) and out_cb.push_back(in_ntiles_c) after fast_tilize_block; factory binds compute as PRODUCER+CONSUMER of DFB_OUT (factory :1263-1273, comment "no kernel-side pop is needed").
  - counts: pushed ceil(sticks/32)*in_ntiles_c, popped 0 -> PACK finish (occupancy!=0) + UNPACK finish on the consumer face hang.
  - Exercised: YES — resnet global avg_pool2d (output_layout=TILE, ttnn_functional_resnet50_large.py:768/945), test_global_avgpool.py, test_avg_pool2d.py (TILE).
  - Fix: either drop the out_cb push (borrowed resident output; no consumer) and the self-loop consumer binding, or add `out_cb.wait_front(n); out_cb.pop_front(n);` after each push (UNPACK side) / once at end.
  - Row-major output: out_cb is census-only (PACK_TO_SCRATCH=1) -> 0/0, balanced.

P3 BENIGN-under-current-finish (user's known example) — in_cb (DFB_IN_0), is_large_kernel
  - reader_pool_2d.cpp:68 reserve_back(1) per c_i, :145-146 push_back(1)+reserve_back(1) at every chunk boundary -> per c_i: reserves = interm_reduction_chunks+1, pushes = interm_reduction_chunks. Last reserve dangling.
  - Cross-kernel pushes vs compute pops (compute_pool_2d.cpp wait/pop 1 per chunk, interm_reduction_chunks per c_i): EQUAL (boundary condition `%max==0 || ==total` fires exactly ceil(total/max) times since contiguous reads never straddle a chunk boundary).
  - So no unacked post; with the finish_impl in this tree it should NOT hang. If upstream #57646 also asserts reserve/push balance, fix: move the trailing reserve into `if (processed_sticks != total_elems_to_reduce)`.
  - Exercised: global avgpool 7x7 (49 > max_sticks_for_reduction) — resnet TILE avgpool path.
  - (mainline M/pool/generic reader_pool_2d.cpp:66/114-115 identical pattern.)

Balanced (Quasar pool, non-indices):
  - in_cb !large: 1 reserve/1 push per c_i; compute 1 wait/pop per chunk(=1). OK.
  - in_scalar_cb !one_scalar_per_core: fill_scalar (pool_kernels_common.hpp:96-128) reserve/push 1 per lane stick; compute wait/pop 1 per stick. OK (assumes reader lane stick count == compute num_out_sticks_per_thread; both derived from same core stick count, SUSPECT only if reader_indices count != out_nhw_this_core).
  - scratch_cb (DFB_SCRATCH_0, RM output): compute reserve on first c-block, push 1 on last c-block per stick; reader wait/pop 1 per lane stick (#ifndef OUTPUT_TILED). OK.
  - pre_tilize_cb / fast_tilize_cb aliased self-loop (TILE): per 32-stick group pre: pushes 32*in_ntiles_c (incl. filler on last partial tile) / pops 32*in_ntiles_c; fast: push/wait/pop in_ntiles_c. Trailing reserve_back(...) dangling but no post. OK.
  - census DFBs (in_shard, reader_indices, clear_value, out_shard, config): reader producer face, compute consumer face, zero push/pop; DRAM-config async_read into reader_indices_cb/config_cb has implicit sync disabled -> no post. OK.

Quasar mpwi (return_indices) — NOT exercised by any test in scope (no return_indices usage):
  P4 CONFIRMED (unexercised) clear_value_cb: reader_mpwi.cpp:496 pushes 1 always (READER_ID==0); compute_mpwi.cpp:136/295 wait/pop only if is_large_kernel -> !large: 1 posted, 0 acked. Fix: gate push on is_large_kernel or pop unconditionally.
  P5 CONFIRMED (unexercised) out_cb / out_idx_cb self-loop on reader1 (reader_mpwi.cpp:346-370 reserve+push output_faces, never popped; factory binds reader1 P+C) -> DM finish hang. Fix: drop push (resident borrowed output) or pop after push.
  P6 CONFIRMED (unexercised) reader_indices_cb config_in_dram: reader0 push 1 (:553), reader1 wait_front(1) never pops (:556). Fix: reader1 pop_front(1) at end.
  Other mpwi DFBs (in_cb, in_idx/compute_tmp_idx, *_inc_cb, pack_tmp/pack_idx_tmp, in_scalar) look balanced (inc/scalar waited at start, popped at compute end :288-295).

## Pool — mainline M/pool/generic (yolo ttnn.max_pool2d)
UNREACHABLE on Quasar: Pool2D::MultiCore builds a ProgramDescriptor (pool_multi_core_program_factory.cpp:265) -> program.cpp:444-496 maps to legacy DataMovementConfig -> kernel.hpp:531 TT_FATAL "DataMovementKernel is not supported on Quasar". yolo test_max_pool2d.py would FATAL before finish() matters.
Latent (if ever ported): same P1 (compute_pool_2d.cpp:123 wait no pop), P2-analog (RM out_dfb push_back(output_faces) :250 consumed by nothing in-kernel — out is borrowed), P3 pattern (reader :66/:114-115), plus reader_id==1 wait_front without pop on clear_value_dfb (:249), reader_indices_dfb (:277), config_dfb (:305).

## Halo — Quasar Q/halo (used by quasar conv2d and quasar pool)
Kernels: halo_gather.cpp (reader0/reader1, implicit sync disabled factory :403-408), compute/pack_untilize.cpp (kernel_lib untilize, 1 tile-row block wait/pop src + reserve/push out per block).

H1 CONFIRMED (structural; exercised only on cores with unreferenced trailing input blocks) — src DFB (borrowed input), TILE input (!skip_untilize)
  - halo_gather.cpp:296-303 reader0 (SRC_PRODUCER) reserve/push input_npages = ntiles_per_block * input_nblocks_per_core (factory :150-152; SAME value for every core, from the remapped max shard).
  - pack_untilize.cpp: compute pops total_blocks * block_size(=1) * tiles_per_row, total_blocks = number_of_blocks_per_core[core] (factory :595) = (max referenced block id)+1 (sliding_window.cpp:756-767 divide_blocks_between_cores).
  - If a core's highest referenced tile-row < input_nblocks_per_core-1 (tail core of an uneven NHW split / padded remapped shard, or trailing rows unused by stride), pushed > popped -> reader0 DM finish + compute UNPACK finish hang.
  - Exercised: depends on shape/core count; resnet halo inputs that are TILE and whose NHW isn't evenly split into 32-row blocks per core (e.g. small spatial dims like layer4 7x7xN on multi-core). SUSPECT-for-tests, CONFIRMED in code.
  - Fix: reader0 push total_blocks*block_size*tiles_per_row (pass total_blocks to reader0 RTA) — or compute after its loop waits/pops the remaining input_npages - total_blocks*tiles_per_row tiles.
H2 SUSPECT — untilize_out0/1 consumed by reader0/1: compute pushes alternate blocks; each reader pops blocks lazily up to its last referenced block then one final pop (halo_gather.cpp:363-402). Balanced iff every block in [0,total_blocks) of the reader's parity is referenced by that reader's config (true for halo since every input stick is copied locally/remotely). If a reader has 0 segments but compute pushed blocks to it (cannot happen since blocks are assigned by referenced src id) -> fine.
Balanced: pad_fill/pad_read cross-reader pair (push 1 / wait+pop 1 each, :320-342); gather/padding_config_scratch (DRAM path, implicit sync disabled, no push/pop); skip_untilize path has no src/untilize DFB traffic.


---
<!-- merged from dfb_conv_pool_part_convsharded.md -->
# Quasar conv2d SHARDED factory — DFB balance audit (part: convsharded)

K = ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels
F = ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/conv2d_op_sharded_program_factory.cpp

finish() semantics used (tt-2xx/dataflow_buffer.inl:274-320): DM side spins until read_acked == read_posted
on every TC; TRISC UNPACK/PACK side spins until tile_counters[tc].posted (live occupancy) == 0. So ANY entry
that is pushed and never popped hangs both the producer's and the (self-)consumer's finish().

## CONFIRMED

### C1. DFB_OUT never popped — fused conv_bmm_tilize_metal2 (HS + BS), ALL configs  [HIGH, exercised]
- F:1946-1952, F:2214-2217: OUT is bound as compute PRODUCER + "degenerate" compute CONSUMER (self-loop) on the
  borrowed OUTPUT shard. The kernel never calls cb_out.wait_front/pop_front on it.
- K/conv_bmm_tilize_metal2.cpp pushes to OUT:
  - !untilize_out && !fuse_bias: :616/:632 last K-block packs to cb_mm_out (= out), out_subblock_num_tiles per subblock
  - fuse_bias && !untilize_out: :747/:753 (cb_untilize_mode_out = out)
  - untilize_out: :273/:286 reblock_and_untilize or kernel_lib untilize (:785) push out
- Counts per core: pushes = in1_num_blocks_w * in0_num_blocks_h * out_block_num_tiles (= per-core M*N tiles);
  pops = 0. After finish(): PACK and UNPACK TRISC spin on OUT's TC posted != 0 forever.
- Exercised: every fused conv (resnet test_conv2d.py stem, test_conv2d_block_sharded.py, test_conv_hang.py,
  e2e layer3/4 BS convs, any HS conv not routed to split).
- Fix (minimal): drain OUT at kernel end in compute: after the outer loops,
  `cb_out.wait_front(total_out_tiles); dummy_unpack(out_cb_id); cb_out.pop_front(total_out_tiles);` (Quasar
  needs the dummy_unpack between wait/pop — TEN-4746 trap; and wait_front of the full count requires
  capacity >= total, which holds because OUT is the borrowed full shard; but note Quasar's borrowed ring holds
  num_entries-1 — see drain comment F:2142-2146 — so popping per block inside the loop is safer), or add a
  credit-only drain DM kernel exactly like drain_out_metal2.cpp (as already done for Program A, F:2349-2387).
  The drain-DM route is the proven one; it needs a free DM (RISCV_0 is the weights writer here, so on Quasar
  it would need a third DM slot or fold the drain into the writer kernel: writer pops out_block per (bw,bh)).
  Alternatively, if the upstream PR exempts borrowed producer-only DFBs, rebind OUT as producer-only (no
  self-consumer) — check how finish() treats a DFB bound only as producer with no consumer.

### C2. DFB_BIAS waited every block, never popped — fused conv (HS + BS) with bias  [HIGH, exercised]
- Producer (writer DMs), pushes bias_ntiles ONCE (load_bias latch):
  - K/reader_writer_tiled_out_1d_mcast_sender_..._metal2.cpp:314 reserve / :365 push (bias_ntiles)
  - K/reader_writer_tiled_out_1d_mcast_receiver_..._metal2.cpp:220 / :231
  - K/writer_tiled_out_2d_mcast_sender_..._metal2.cpp:289 / :340
  - K/writer_tiled_out_2d_mcast_receiver_..._metal2.cpp:193 / :204
- Consumer K/conv_bmm_tilize_metal2.cpp:723 `cb_bias.wait_front(bias_ntiles_w)` per (bw,bh) block, indexed by
  bias_block_offset; NO cb_bias.pop_front anywhere.
- Counts: pushed = bias_ntiles_per_core (F:1888/1920 == compute bias_ntiles_w F:2293); popped = 0.
  Writer DM finish(): read_acked(0) != read_posted(bias_ntiles) -> hang; compute UNPACK finish: posted != 0 -> hang.
- Exercised: all resnet fused convs pass bias (test_conv2d.py:119, test_conv2d_block_sharded.py:106,
  test_conv_hang.py:146, e2e).
- Fix: after the in1_num_blocks_w loop (end of kernel_main, under `#ifdef FUSE_BIAS if constexpr (fuse_bias)`):
  `dummy_unpack(bias_cb_id); cb_bias.pop_front(bias_ntiles_w);` (the wait already happened; dummy_unpack orders
  POP after WAIT on Quasar). Skip-compute cores (C/S3 below) must also do the wait+pop.

### C3. DFB_READER_INDICES push_back with no reserve_back and no consumer pop — CONFIG_TENSOR_IN_DRAM  [not exercised]
- K/reader_conv_activations_padded_with_halo_3x3_weights_v2_metal2.cpp:49-56 (HS) and
  K/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2_metal2.cpp:213-223 (BS):
  get_write_ptr + async_read + push_back(1) with no reserve_back(1) (type 1), and the reader (bound both
  PRODUCER and CONSUMER, F:1700-1709) never wait_front/pop_front (type 4). Writers only wait_front(1) under
  split_reader (forced off) and never pop either (1d sender :122, 1d receiver :102, 2d sender :112, 2d receiver :83).
- Counts: reserve 0 / push 1 / pop 0 -> DM finish hang.
- Exercised: only if Conv2dConfig.config_tensors_in_dram=True; resnet/llama tests never set it.
- Fix: `cb.reserve_back(1)` before the read; at kernel end `cb.wait_front(1); cb.pop_front(1);` in the reader.

### C4. Unpack-tilize probe (TT_METAL_QSR_CONV_UNPACK_TILIZE) — diagnostic only  [exercised only by test_conv2d_unpack_tilize_probe.py / test_conv_hang.py with the env var]
- K/conv_unpack_tilize_probe_metal2.cpp:75-78: in_scalar self-loop reserve/push/wait 1, never popped (1 left).
- :100-102: act_tilized self-loop (F:2125-2131 producer+consumer) pushes M*K tiles, never popped.
- Fix: end of kernel: `dummy_unpack; in_scalar_cb.pop_front(1);` and pop act_tilized per chunk (wait/pop after
  push) or at end. Or delete the probe.

## SUSPECT (latent / needs runtime-config knowledge)

### S1. CHECK_SKIP_COMPUTE skips weights (and bias) pops — K/conv_bmm_tilize_metal2.cpp:530-539, :703-706
- skip_compute path pops mm_in0 then `continue`s before `cb_in1.wait_front/pop_front` (:542/:698); later
  `continue`s before the bias stage. If a weights producer runs on that node, weights pushed
  (out_num_blocks_h * in0_num_blocks_w * weight_block_num_tiles) are never popped.
- HS: receiver writer placed on input_cores \ sender (F:1358-1366) and compute on input_cores; skip_compute
  set when input_cores != output_cores and core.x > out bbox end_x (F:568-583, F:1305). So a skip core still
  gets weight/bias pushes from the receiver -> finish hang (and pre-PR capacity stall if >capacity).
- BS: 2D sender returns early on skip_work (2d sender :104) — no pushes, balanced there; 2D receivers on a
  skip column would also have no mcast to receive (pre-existing semantic issue, not a count issue).
- Fix: on the skip path do `cb_in1.wait_front(in1_block_num_tiles); dummy_unpack; cb_in1.pop_front(...)` before
  `continue`, and the bias wait+pop at the end. Exercised only when HS input grid != output grid (stride-2 HS
  conv where output shard grid shrinks); I could not confirm any resnet test hits it.

### S2. Block-sharded SKIP_MCAST reader (num_cores_c_in == num_cores_c_out == 1) — 2d reader :283-354
- With SKIP_MCAST the reader pushes ACT_ROW_MAJOR but never pushes ACT nor pops ACT_TILIZED. Compute (BS)
  tilizes ACT_RM -> pushes ACT_TILIZED (no consumer) and waits ACT (mm_in0 = dfb::act, :527) which nobody
  pushes -> pre-existing deadlock regardless of finish(). Counts: ACT push 0 vs pop
  nbh*in0_num_blocks_w*act_block_num_tiles; ACT_TILIZED push N vs pop 0. Legacy WH relied on the ACT/
  ACT_TILIZED CB overlap; the DFB port has them separate (F:1500-1502). Only on a 1-column BS grid; not seen
  in tests. Fix: under SKIP_MCAST do a local copy (or alias) ACT_TILIZED -> ACT: reserve ACT, wait ACT_TILIZED,
  local L1 copy, push ACT, pop ACT_TILIZED.

### S3. Block-sharded split tilize-only (TT_METAL_QSR_CONV_SPLIT_PROGRAM + BS) — F:1093-1095 allows BS
- Compute conv_tilize_only consumes ACT, pushes OUT; but the 2D reader still pushes ACT_RM (no consumer
  bound) and pops ACT_TILIZED (no producer) — imbalance + hang. conv2d.cpp:1012-1014 only activates the split
  for HEIGHT_SHARDED and output-spec gating likely TT_FATALs (F:1570-1583) first, so not reached today.
  Fix: either restrict factory gate to height_sharded, or give the BS tilize-only compute the ACT_RM input.

### S4. 1D depthwise (compute_depthwise_conv1d_metal2.cpp) — OUT self-loop residue (same class as C1)
- Non-coalesced, no scratch (:184-186): taps push B to OUT each, taps>0 pop B -> B left per (h) block.
  Coalesced (:173-175): push B, pop 0. Scratch (:176-183): scratch balanced ((taps-1)*B each way), OUT push B
  pop 0. OUT bound producer+consumer (F:2160-2171). Residue = per-core output tiles -> finish hang.
- Not exercised (resnet has no 1D depthwise). Fix: same as C1.

## Balanced (checked)
- conv_bmm_tilize_metal2 ACT (HS): reader pushes act_block_num_tiles per (bh,outer) (HS reader :147/:187);
  compute tilize pops in0_num_subblocks_read*in0_block_w = act_block_num_tiles per K-block (needs
  in1_num_blocks_w == 1, which HS enforces via weight_block_w = per-core width; otherwise pre-existing hang).
- conv_bmm_tilize_metal2 ACT_TILIZED (HS self-loop): tilize pushes act_block_num_tiles, :527/:697 pops same.
- conv_bmm_tilize_metal2 ACT_ROW_MAJOR / ACT_TILIZED / ACT (BS, mcast on): reader ACT_RM push per (nbh,outer)
  = compute tilize pops (once per in0_nblocks_w_tilize = conv_act_c_blocks K-blocks); ACT_TILIZED pushed by
  compute == popped by reader :353; ACT pushed act_w_num_outer per outer == popped per K-block
  (in0_num_blocks_w = conv_act_c_blocks * window_outer).
- conv_bmm_tilize_metal2 WEIGHTS: writers push weight_block_num_tiles per (bw,bh,weight_h[,outer]) =
  num_blocks_act_w * conv_act_c_blocks per (bw,bh) (F:1874/1881) == compute pops per K-block.
- conv_bmm_tilize_metal2 MATMUL_PARTIALS self-loop: all 6 combos (l1_acc x fuse_bias x spill, +untilize_out)
  have pushes == pops per output block (RESTORE_PARTIALS_* only rewinds ring entry_idx, not counters; the
  finish() TRISC check is counter-only so it is unaffected). Note: correctness of ring rewind vs finish
  ordering not affected since counts net to zero at end of each (bh,bw).
- kernel_lib tilize / untilize helpers (tilize_helpers.inl:178-191, untilize_helpers.inl:220-268): per-block
  wait/reserve/push/pop balanced.
- conv_tilize_only_metal2.cpp + drain_out_metal2.cpp (split Program A, HS): ACT reader push nbh*act_block_num_tiles
  == compute pops num_blocks*in0_block_w (window_outer==1, full_K==act_block_w under the in0_num_blocks_w==1
  gate); OUT compute push num_blocks*in0_block_w == drain pop (identical CTAs F:2278-2281 vs F:2363-2369).
- reader_conv_activations_padded_with_halo_3x3_weights_v2_metal2.cpp (HS): reserve==push per block
  (activation_reuse forced off F:629, so push_remaining_tiles/evil_set paths dead).
- reader_conv_activations_2d_mcast_..._metal2.cpp (BS): ACT_RM reserve==push; ACT reserve==push per outer_i;
  mcast_block_chunked waits ACT_TILIZED, pop at :353 matches.
- reader_depthwise_conv1d_metal2.cpp: reserve==push per (bh,outer).
- 1d sender / 1d receiver / 2d sender / 2d receiver writers: WEIGHTS reserve==push; BIAS reserve==push
  (imbalance is cross-kernel, C2); ACT_SECOND_READER paths dead (split_reader forced off F:827).


---
<!-- merged from dfb_conv_pool_part_convws_mainline.md -->
## Part: Quasar conv2d WIDTH-sharded factory + mainline ttnn.conv2d / mainline halo

finish() semantics checked (tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:274-320). On TRISC unpack/pack it spins
until the tile counter's posted (tiles available) is 0, so every push has to be popped. On DM it spins until
read_acked == read_posted. Note: finish() is not auto-called on this branch (#57646 is not merged here).

### Reachability
- MAINLINE ttnn.conv2d (M/conv/conv2d/device/*) and mainline halo (M/sliding_window/halo) are UNREACHABLE on Quasar.
  They build ProgramDescriptors with CBDescriptors and KernelDescriptors. program.cpp:444-496 maps these to legacy
  Reader/Writer DataMovementConfig and ComputeConfig, then calls CreateKernel. tt_metal.cpp:778 creates a
  DataMovementKernel, which hits TT_FATAL "DataMovementKernel is not supported on Quasar" (kernel.hpp:530-532;
  ComputeKernel has the same check at kernel.hpp:738). Nothing in the conv, sliding_window or pool trees redirects
  to experimental/quasar. So yolo_ops/test_conv2d.py (ttnn.conv2d) FATALs at program creation, during halo or
  conv, whatever the shard layout. Its kernels are latent only.
- Quasar WIDTH-sharded factory (Q/conv2d/device/conv2d_op_width_sharded_program_factory.cpp) is selected when the
  input is WIDTH_SHARDED (conv2d_device_operation.cpp:36). No test in the 4 scoped dirs passes WIDTH_SHARDED to
  quasar.conv2d: resnet uses HEIGHT/BLOCK explicitly, and its WIDTH_SHARDED configs are only for avgpool/fc.
  So it is reachable but NOT exercised.

### Findings — Quasar width-sharded trio
DFBs: act (mcast result), act_row_major, act_tilized, weights, bias, matmul_partials (compute self-loop),
out (compute self-loop, borrowed OUTPUT), act_sharded (act-reader self-loop, borrowed INPUT), reader_indices
(act-reader self-loop).

W1. CONFIRMED — `bias` is waited on but never popped (compute consumer).
    conv_bmm_tilize_metal2.cpp:723 does `cb_bias.wait_front(bias_ntiles_w)` once per in0 h-block. No pop_front(bias)
    exists anywhere in the kernel. Producer: weights_reader_width_sharded_metal2.cpp:99-112 reserves and pushes
    weight_block_width_ntiles once.
    Counts: pushed = bias_ntiles, popped = 0. UNPACK finish() spins (posted != 0), and the weights reader's DM
    finish() spins (acked != posted).
    This kernel is SHARED: the sharded factory (HS/BS fused path, the resnet path with folded-BN bias) binds the same
    compute kernel. Its senders push bias once (writer_tiled_out_2d_mcast_sender_..._metal2.cpp:289/340 and
    reader_writer_tiled_out_1d_mcast_sender_..._metal2.cpp:314/365; receivers presumably mirror this, not checked
    here), so this is EXERCISED by every resnet fused conv with bias.
    Fix: at the end of kernel_main, after the in1_num_blocks_w loop, add
    `#ifdef FUSE_BIAS if constexpr (fuse_bias) { cb_bias.pop_front(bias_ntiles_w); } #endif`.
    UNPACK tiles were consumed after the wait, so the TEN-4746 bare wait->pop trap does not apply. Mainline
    conv_bmm_tilize.cpp:529 has the same bug (latent).

W2. CONFIRMED — `out` self-loop is pushed and never popped (compute).
    The factory (lines 518-525, 569-572) binds compute as both PRODUCER and CONSUMER of OUT, but there is no DM drain.
    In width-sharded, compute pushes out_block_num_tiles per h-block: mm_out to out at lines 616/632 on the last
    K-block when !fuse_bias; line 747/753 when fuse_bias && !untilize_out; reblock_and_untilize line 286 when
    untilize_out. Nothing pops.
    Counts: pushed = in0_num_blocks_h * out_block_num_tiles (= per_core_out_ntiles), popped = 0. PACK finish()
    spins forever.
    Fix: on Quasar, either add a credit-only drain DM kernel (like drain_out_metal2.cpp on the sharded split path)
    or have compute pop out after packing (`cb_out.wait_front(N); cb_out.pop_front(N)` at kernel end, which needs
    a TDMA op or dummy_unpack between them per TEN-4746). The sharded factory's OUT binding should be checked
    for the same "fake producer-only" pattern.

W3. CONFIRMED (CONFIG_TENSOR_IN_DRAM only) — `reader_indices` gets a push with no reserve and no pop
    (act-reader self-loop). activation_reader_width_sharded_metal2.cpp:110-112 does async_read into reader_indices,
    then push_back(1) with no reserve_back(1). Nothing pops it, and line 117 then calls get_write_ptr() to read it.
    Counts: reserve=0 / push=1 / wait=0 / pop=0. DM finish() spins (posted 1, acked 0).
    Path: config_tensors_in_dram=true. Not exercised.
    Fix: reserve_back(1) before the read; capture get_write_ptr() before push (or use get_read_ptr() after a
    wait_front(1)); then wait_front(1) + pop_front(1) at kernel end.
    The L1 path (borrowed, no push/pop) is balanced 0/0.

W4. CONFIRMED (SKIP_MCAST only) — `act` and `act_tilized` are both unbalanced.
    With SKIP_MCAST (width-sharded, num_cores_c_in == num_cores_c_out == 1, conv2d_utils.cpp:587-588), the
    reader's whole mcast block (activation_reader_width_sharded_metal2.cpp:194-244) is compiled out:
    - act: compute does wait/pop H*W*in0_block_num_tiles (lines 527/697), reader pushes 0. It hangs before finish().
    - act_tilized: compute pushes H*per_core*act_block_num_tiles, reader pops 0.
    Also, get_cb_info gives ACT num_pages=0 with overlapped_by_cb=ACT_TILIZED (conv2d_op_program_factory_common.cpp:256-270),
    but make_dfb (factory:484-492) ignores overlapped_by_cb, so this would create a 0-entry act DFB.
    Path: single-core width-sharded only. Not exercised.
    Fix: under SKIP_MCAST, the reader should forward act_tilized to act (wait_front/pop_front act_tilized and
    reserve/push act; or a local L1 copy), or compute should use act_tilized as mm_in0. Alternatively TT_FATAL
    on Quasar.

W5. SUSPECT (ordering, not count) — `act_tilized` is popped without a wait on non-sender cores.
    activation_reader_width_sharded_metal2.cpp:243 pops act_block_num_tiles every block. wait_front (line 203)
    only runs when act_w_outer_i == this_core_id, so cores with this_core_id >= num_input_cores
    (output-only cores, num_output_cores > num_input_cores) never wait. Totals are balanced (compute pushes
    H*per_core blocks on every core), but the DM may ack credits before compute posts them (underflow risk).
    Fix: `tilized_in0_cb.wait_front(act_block_num_tiles)` before the pop when this_core_id >= num_input_cores.
    Minor: line 159 calls get_write_ptr() before reserve_back() (order swap). This is not an imbalance.

Balanced (traced):
- act_row_major: reader reserves exactly once per started tile-row and pushes once per tile-row, including the
  partial trailing row at lines 180-183. Dummy cores do reserve+push ntile_height*ntile_width per block. Compute
  tilize consumes act_block_h_ntiles*in0_block_w per tilize, H*per_core times. Assumes reader indices cover
  act_block_h_datums rows per block, which is the same assumption as legacy WH.
- act (non-SKIP): reader pushes H*per_core*num_input_cores*act_block_num_tiles = H*W*blk; compute pops the same.
- weights: reader pushes H*local*remote = H*W blocks of weight_block_num_tiles; compute waits/pops in1_block_num_tiles
  H*W*in1_num_blocks_w times. in1_num_blocks_w = 1 because weight_block_w_ntiles = per_core_out_matrix_width_ntile
  (factory:131/352).
- matmul_partials self-loop: balanced in all 4 combinations of packer_l1_acc x fuse_bias, spill and non-spill.
  - non-l1acc spill !bias: push (W-1)B, reload-pop (W-1)B.
  - non-l1acc bias: push WB, pop (W-1)B reload + B bias.
  - l1acc !bias: (W-2) push/pop pairs + 1 push popped by last-block reload.
  - l1acc bias: (W-1) push/pop + B popped by bias.
  - untilize_out: the partials block is popped by untilize/reblock.
  RESTORE_* rewinds only ring position, not credit.
- act_sharded self-loop (borrowed INPUT): address source only, 0/0.
- reader_indices L1 path: 0/0.
- CHECK_SKIP_COMPUTE / SPLIT_READER / activation_reuse branches: not defined/enabled by the width-sharded factory.

### Mainline (latent, UNREACHABLE on Quasar)
- M/conv/conv2d/device/kernels/conv_bmm_tilize.cpp:529: same bias wait-without-pop as W1.
- M/.../activation_reader_width_sharded.cpp: same structure as the Quasar fork (lines 166-198 act_rm, 206-278 act/act_tilized).
  W3/W4/W5 apply equally.
- compute_depthwise_conv1d.cpp, the other readers/writers, halo_gather.cpp and pack_untilize.cpp were not deeply analysed
  because they FATAL at CreateKernel on Quasar.


---
<!-- merged from dfb_conv_pool_part_fold_slice_upsample.md -->
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
