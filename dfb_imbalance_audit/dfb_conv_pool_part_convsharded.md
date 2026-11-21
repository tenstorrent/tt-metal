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
