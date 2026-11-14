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
