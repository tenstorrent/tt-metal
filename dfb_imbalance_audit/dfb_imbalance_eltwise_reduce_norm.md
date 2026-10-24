# DFB imbalance audit for #57646 (forced DataflowBuffer::finish()) — family: eltwise_reduce_norm

Read-only code audit; nothing built or run. Per-sub-family detail (file:line counts, fixes, coverage) in scratchpad/ern/*.md, appended in full below.

## Collected ops (call counts across 4 test dirs + model code)
add 29, multiply 17, mul 10, subtract 12, div 5, gt 4, where 6, eq_ 1, experimental.quasar.add 2 / add_ 5 / multiply 3,
sigmoid 3, log 2, exp 1, remainder 1, plus_one 14, manual_seed 7,
sum 5, mean 2, max 2, argmax 9, topk 10, sampling 12,
softmax 3, rms_norm 7, rms_norm_pre_all_gather 7, rms_norm_post_all_gather 5, layer_norm (qwen3_vl),
scatter (qwen3_vl), scatter_add 2, gather 2. clamp: unused. moe_routing_remap: only gpt_oss (outside scope).

## Important dispatch note
Mainline legacy (KernelDescriptor/Gen1) factories TT_FATAL on Quasar at tt_metal/impl/kernels/kernel.hpp:530
("DataMovementKernel is not supported on Quasar"), so they never reach finish(). This covers mainline binary_ng
(ttnn.add/mul/sub/div/gt/eq_), ternary where, unary (sigmoid/exp/log/remainder), gather, moe_routing_remap.
Note: yolo_ops/qwen3_vl add/mul/sub/div/sigmoid tests therefore fail at host on Quasar unless rerouted (llama e2e
monkeypatches add/mul/sub to experimental.quasar.* at test_llama_e2e.py:1380).

## HIT BY TESTS (CONFIRMED)
1. sampling writer — reduction/sampling/device/kernels/dataflow/writer_interleaved.cpp:72/77 (dfb k) and :83/88 (dfb p):
   writer self-loop reserve+push 1, never popped -> finish() hangs every core, every ttnn.sampling call.
   Fix: dfb_k.pop_front(1) / dfb_p.pop_front(1) after values read into locals (~:80, ~:91).
   (Separate pre-existing deadlock: W=32 (Wt=1) passes validation but compute top_k waits 2 tiles — test_sampling K=32.)
2. sharded layernorm/rmsnorm eps — writer_unary_sharded_ln.cpp:68-70 / writer_unary_sharded_ln_rm_gb.cpp:72-74 push 1 eps tile;
   compute/layernorm_sharded.cpp never wait/pop (uses it by index at :520-521 w/o wait = also a race).
   Tests: llama graph_ops/test_rms_norm.py (8x4,8x8), debug_ops/test_quasar_sharded_rmsnorm_mcast_fault.py, qwen test_rms_norm.py (8x5).
   Fix: wait_front(1) early (inside #ifndef IDLE_CORE), pop_front(1) at end.
3. sharded layernorm/rmsnorm out — layernorm_sharded.cpp reserve/push num_tiles_per_block (:669/:692 and gamma/beta siblings)
   into borrowed output DFB that is a compute self-loop when no write-back (layernorm_op_multi_core_sharded.cpp:116); never popped.
   Same tests. Fix: wait_front+pop_front(num_tiles_per_block) at kernel end (or skip reserve/push on borrowed out).
4. interleaved layernorm.cpp gamma/beta padding — readers push round_up(Wt,blk) (reader_unary_interleaved_ln_rm_gb.cpp:164,197,203,236;
   reader_unary_interleaved_ln.cpp:178-183), compute pops only Wt (layernorm.cpp:431,434).
   Tests: qwen test_rms_norm.py [1,1,32,128],[1,1,8,128],[1,32,4096,128],[1,8,4096,128] (Wt=4, blk=8 -> 8 pushed / 4 popped).
   Fix: pop total_buffer_size (layernorm.cpp:172) instead of Wt.

## REACHABLE, NOT EXERCISED BY THESE TESTS (CONFIRMED)
- experimental/quasar binary_ng Quasar-native (TTNN_QSR_NATIVE=1, all-sharded bf16 add): kernels_qsr/dataflow/writer_no_bcast_dfb.cpp:33-37
  never drains main out ring (compute pushes c_main_tiles, writer pops 0; comment says "do not call finish()"). Only nightly
  test_binary_ng_quasar_native.py sharded cases. Fix: per-TC wait/pop drain like kernels_dfb writer.
- reduction/generic reader_unary_reduce_input_rows_partitioned_sharded.cpp:41/61 (in1 self-loop reserve/push num_tiles, no pop; HS W-reduce).
- reduction/generic reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp:54/83 (same; WS H-reduce).
- reduction/generic reader_unary_reduce_rm.cpp:77-81 clear_value push_back(1) with no reserve and no pop (row-major sum/mean;
  also H-split stage 2 when Ht>=20). Fix: reserve before fill, pop at end.
- normalization layernorm_large_tensor.cpp:421-423 scaler popped only #ifndef RMSNORM.
- softmax_sharded.cpp: pops borrowed in0 never pushed, pushes borrowed out0 never popped, pops causal mask never pushed.
- attention softmax.cpp: mask_padded pushed by writer (:39-51), pop at :396-400 guarded by !FUSED_SCALE_MASK.
- layernorm_post_allgather.cpp (LayerNorm, Wt==dfb_length): gamma/beta waited never popped.
- sharded post_all_gather: writer scaler self-loop never drained; eps popped only on all-to-all+enable_sqrt; out not popped
  w/o write-back; (SUSPECT) scaler_global popped only under enable_sqrt.
- sharded pre_all_gather: self-loop out pushed (layernorm_sharded_pre_allgather.cpp:304-330) never popped; (SUSPECT) receiver
  unconditionally waits/pops ex2 that compute pushes only on some cores.
- Legacy/unreachable on Quasar today (fix when ported): moe_routing_remap writer waits c_0/c_1 without pop, c_2 scratch push never popped.

## SUSPECT / benign
- manual_seed readers read_all_data.cpp:38,45 / read_user_id.cpp:34: scratch reserve_back(1) never pushed (same shape as
  reader_pool_2d). No counter effect under current finish_impl; only matters if reserve==push is also checked. Fix: use get_write_ptr() w/o reserve.
- binary_ng kernels_dfb reader_row_col_mixed_bcast_dfb.cpp: sharded branch reserves without push (unreachable: bcast never borrows).
- quasar reduction writer_unary_interleaved_start_id[_metal2].cpp OUT_SHARDED wait w/o pop (define never set).
- reshard_writer.hpp:39-71 over-wait/pop when block_ht>1; topk Wt==1 (validation blocks).

## Balanced (summary; per-kernel lists in sub-reports)
binary_ng kernels_dfb (MetalV2 default Quasar path: no-bcast, scalar, row/col/scalar/mixed bcast, post_lhs/post_rhs self-loops),
kernels_qsr interleaved/tail paths; plus_one; manual_seed kernel_communication; unary kernels (unreachable anyway);
reduce.cpp/reduce_metal2.cpp (scaler popped), reduce_rm, reduce_*_neg, prepare_reduce_scaler & generate_* helpers;
argmax (no DFB sync), argmax_nc; topk single/multi-core; sampling other DFBs; scatter + bf16 scatter-reduce;
yolo softmax (moreh_softmax_h), attention softmax non-fused, softmax_large_tensor; interleaved layernorm (except gamma/beta),
layernorm_large_tensor (except RMS scaler), layernorm_sharded other DFBs, rmsnorm_pre_allgather/layernorm_pre_allgather,
rmsnorm_post_allgather_metal2.
Not audited: 2D pre-allgather kernels, Welford variants (rejected on Quasar), moreh softmax w/w_large/h_large/c_large.



---
# Sub-report: binary

# DFB finish() balance audit: binary / binary_ng / bcast / ternary where (Quasar)

Context: PR #57646 forces DataflowBuffer::finish() at kernel end for every DFB. The DM side spins until
read_acked == read_posted per TC. The TRISC unpack/pack side spins until TC occupancy (posted) == 0
(tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:274-321).

## 1. What actually runs on Quasar (dispatch)

| Python op | Host path | Quasar device kernels |
|---|---|---|
| ttnn.add / mul / multiply / subtract / div / gt / eq_ (mainline) | eltwise/binary_ng ProgramFactory (KernelDescriptor only; `program_factory_t = variant<ProgramFactory>`) | NONE: host TT_FATAL "DataMovementKernel is not supported on Quasar" (tt_metal/impl/kernels/kernel.hpp:530-532). No device DFB, so the finish rule doesn't apply |
| ttnn.where (mainline ternary) | eltwise/ternary ternary_program_factory.cpp (KernelDescriptor) | NONE: same host FATAL |
| ttnn.experimental.quasar.{add,add_,multiply,subtract,divide,...} | experimental/quasar/binary_ng BinaryNgDeviceOperation::select_program_factory (binary_ng_device_operation.cpp:729) | (a) ProgramFactoryQuasarNative: only if env TTNN_QSR_NATIVE=1 and bf16 ADD, no bcast, no acts (kernels_qsr/). (b) ProgramFactoryMetalV2: the default; kernels_dfb/. (c) ProgramFactory (descriptor; kernels/ + kernels_ng/) is a host FATAL on Quasar |
| experimental/quasar/binary/device/kernels/* | not referenced by any factory (grep) | dead |

Test routing:
- llama e2e: `_install_quasar_eltwise` (test_llama_e2e.py:1380) monkeypatches ttnn.add/mul/multiply/subtract/sub to ttnn.experimental.quasar.*, and forces sharded output configs to DRAM.
- llama graph_ops/prototype_ops and resnet call the quasar ops directly.
- yolo_ops (ttnn.add/div/multiply/subtract) and qwen3_vl_ops (ttnn.add/multiply) call the MAINLINE ops, so they FATAL at host on Quasar. No device kernel runs, so no finish risk. They also have no Quasar coverage.
- where/gt/div/eq_ (llama sampling/penalties, tt_log_probs) are mainline only, so they are host FATAL on Quasar.

Configs the tests use on the live MetalV2 path:
- resnet residual `quasar.add_(out, ds_out, activations=[RELU])`: [1,1,NHW,C] HEIGHT_SHARDED, identical memory configs, in place. Takes the borrow path, FPU add, PACK_RELU fast path (no post DFB), num_tiles_per_cycle=min(8, shard tiles).
- llama graph_ops test_add: [1,1,32,2048] / [1,1,1024,2048], DRAM interleaved plus a width-sharded variant. NONE bcast.
- llama graph_ops test_multiply: [1,1,32,8192] width-sharded [32,128] on 8x8 for a/b/out with input_tensor_a_activations=[SILU]. Borrow path, SFPU, POST_LHS self-loop DFB, num_tiles_per_cycle=2. Also [1,1,1024,8192] DRAM interleaved + SILU.
- llama prototype test_multiply: `quasar.multiply(x, _EMBED_SCALE)` takes the tensor-scalar path (reader_scalar_op_dfb + writer_scalar_dfb + *_scalar_dfb compute).
- llama e2e: same-shape add/mul (residual, MLP gate with SILU) and the scalar embed scale.

## 2. Findings

### F1 (CONFIRMED, by design): Quasar-native factory, borrowed output main ring is never drained
- File: ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/dataflow/writer_no_bcast_dfb.cpp:33-37. Under `#if DST_SHARDED`, the writer only handles `#if TAIL_TILES`. There is no wait_front/pop_front on `dfb::out` for the main ring. The comment at lines 34-36 says it on purpose: "Its credits stay posted ... Do not call finish() on it: nothing acks, so it hangs."
- DFB: OUT ("binary_ng_out_dfb", c_2). Producer: compute (kernels_qsr/compute/eltwise_binary_no_bcast_dfb.cpp process_tiles push_back). The writer is bound ConsumerOf(OUT) (binary_ng_quasar_native_factory.cpp:1005-1006) whenever has_main_ring.
- Counts per core: compute pushes c_main_tiles (= shard tiles minus tail_tiles). Writer pops 0.
- Effect under #57646: the compute PACK finish waits for OUT TC occupancy to reach 0, and the writer's DM finish on OUT waits for acked==posted. Both hang forever.
- Trigger: TTNN_QSR_NATIVE=1, bf16 ADD, NONE bcast, no activations, all of a/b/out L1 sharded with one memory config (borrow_shards), shard volume > tail_tiles (has_main_ring).
- Test coverage: none of the model tests. TTNN_QSR_NATIVE is off by default and is set only by tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_binary_ng_quasar_native.py (_BORROWED_SHARDS / _UNEVEN_SHARDS / _SMALL_SHARDS arms). The resnet residual add_ (HS sharded, in place, bf16 add) would hit this only if TTNN_QSR_NATIVE were set, and its RELU activation fails the native gate anyway.
- Minimal fix: drain the main ring in the writer per tile counter, mirroring the reader's SRC_SHARDED publish loop (reader_no_bcast_dfb.cpp:43-59). Then drop the "do not finish" comment:
  ```cpp
  #if HAS_MAIN_RING
      {
          DataflowBuffer dfb_out(dfb::out);
          const uint32_t dst_num_tiles = get_arg(args::dst_num_tiles);
          constexpr uint32_t num_tcs = get_arg(args::num_tcs);
          const uint32_t num_threads = get_num_threads();
          const uint32_t tile_step = num_tcs * num_threads;
          uint32_t first = get_my_thread_id();
          for (uint32_t c = 0; c < num_tcs && first < dst_num_tiles; ++c, first += num_threads) {
              const uint32_t n = (dst_num_tiles - first + tile_step - 1) / tile_step;
              dfb_out.wait_front(n);
              dfb_out.pop_front(n);
          }
      }
  #endif
  ```
  In the borrow path writer_threads=1 and writer_num_tcs=compute_threads, so this matches the strided rotation. kernels_dfb/dataflow/writer_no_bcast_dfb.cpp (DST_SHARDED: wait_front(dst)/pop_front(dst)) is the already-working precedent.

### F2 (SUSPECT, latent, unreachable): mixed ROW/COL reader, sharded ROW operand reserves without pushing
- File: kernels_dfb/dataflow/reader_row_col_mixed_bcast_dfb.cpp. In the tw loop, the ROW operand's `dfb_in1.reserve_back(onetile)` / `dfb_in0.reserve_back(onetile)` runs unconditionally, but the matching push_back sits inside `#if !SRC_SHARDED_B` / `#if !SRC_SHARDED`. So a sharded ROW operand reserves per tile and never pushes. Compute (eltwise_binary_row_col_bcast_dfb.cpp:83/104) would wait forever.
- Unreachable: the MetalV2 factory sets SRC_SHARDED = borrow_shards, and borrow_shards needs a, b and c all borrowed. is_native_L1_sharding (binary_ng_utils.cpp:781-858) only reports both inputs sharded for identical shapes, which means NONE bcast. On top of that, bcast + borrow TT_FATALs unless num_tiles_per_cycle==1 (metal_v2_factory.cpp:575-578). Fix only if borrow is ever enabled for bcast: move push_back out of the `#if !SRC_SHARDED*` guard (or drop the reserve too).
- Related dead code, balanced: reader_col_bcast_dfb.cpp / reader_scalar_bcast_dfb.cpp do reserve+push without data for a sharded bcast operand (counts match compute, but the data would be wrong). Also unreachable.

### No other imbalance found on the Quasar-live path.

## 3. Kernels checked and found balanced (MetalV2 default path, kernels_dfb/)
- reader_no_bcast_dfb.cpp: in0/in1 push dst_num_tiles each (NoC walk always reaches dst_num_tiles, since split is over c.physical_volume and the walk is bounded by c dims) or src_num_tiles (borrowed) = compute num_tiles.
- writer_no_bcast_dfb.cpp: out pops dst_num_tiles (NoC) or wait/pop dst_num_tiles (DST_SHARDED) = compute pushes.
- eltwise_binary_no_bcast_dfb.cpp / eltwise_binary_sfpu_no_bcast_dfb.cpp: full chunks + remainder pop/push exactly num_tiles. post_lhs/post_rhs self-loop PREPROCESS push n / binary pop n.
- eltwise_utils_dfb.hpp / eltwise_utils_sfpu_dfb.hpp (PREPROCESS): pre wait/pop n, post reserve/push n.
- reader_scalar_op_dfb.cpp: in0 push dst_num_tiles (or src_num_tiles sharded; scalar never borrows).
- writer_scalar_dfb.cpp: in1 reserve/push 1 (scalar fill, once) + out drain dst_num_tiles.
- eltwise_binary_scalar_dfb.cpp / eltwise_binary_sfpu_scalar_dfb.cpp: RHS wait 1 once, pop 1 at end (dummy_unpack only when num_tiles==0). RHS PREPROCESS 1-in/1-out. LHS per chunk.
- reader_bcast_dfb.cpp (ROW): in0 and in1 each push 1 per output tile = compute.
- eltwise_binary_row_bcast_dfb.cpp / eltwise_binary_sfpu_row_bcast_dfb.cpp: per tile, bcast pop 1, llk_post push 1 / pop 1, other pop 1, out push 1.
- reader_col_bcast_dfb.cpp (COL): bcast operand push 1 per tile-row entered = ceil((tile_start+num_tiles)/Wt), the same as the compute's complete+remaining iterations. Other operand 1 per tile.
- eltwise_binary_col_bcast_dfb.cpp / eltwise_binary_sfpu_col_bcast_dfb.cpp: per process_tile, bcast pop 1, llk_post push 1 + post_bcast pop 1, other pop freq-tile_start. num_tiles==0 early return matches reader 0.
- reader_scalar_bcast_dfb.cpp (SCALAR): bcast push 1 per (C) slab entered = ceil((start_t+num_tiles)/HtWt) = compute iterations.
- eltwise_binary_scalar_bcast_dfb.cpp / eltwise_binary_sfpu_scalar_bcast_dfb.cpp: same structure as COL with freq=Ht*Wt, so balanced.
- reader_row_col_mixed_bcast_dfb.cpp (non-sharded, the only live config): COL operand push 1 per row, ROW operand push 1 per tile. Compute: COL post_bcast pop 1 per row, raw_row pop 1 per tile, llk_post push/pop 1 per tile.
- eltwise_binary_row_col_bcast_dfb.cpp / eltwise_binary_sfpu_row_col_bcast_dfb.cpp: balanced as above.
- Factory self-loops (binary_ng_metal_v2_factory.cpp): POST_LHS/POST_RHS/LLK_POST_* sized num_tiles_per_cycle. Producer and consumer are both compute, with push/pop paired in the same iteration.

## 4. Quasar-native (kernels_qsr/), off by default
- reader_no_bcast_dfb.cpp: interleaved per-thread strided loops push the same totals as compute/writer and call finish() explicitly. SRC_SHARDED main ring: per-TC publish sums to dst_num_tiles. Tail rings: num_tcs pushes (with padding) = compute_threads entries.
- eltwise_binary_no_bcast_dfb.cpp (qsr): main my_tiles per thread sums to num_tiles. Tail: each compute thread wait/pop 1 of each input, push 1 out_tail, which totals compute_threads.
- writer_no_bcast_dfb.cpp (qsr): interleaved balanced. Tail out drained num_tcs = compute_threads (balanced). Main ring under DST_SHARDED is **F1 (imbalanced)**.

## 5. Not reachable on Quasar (not audited for finish)
- mainline ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels{,_ng}/**, eltwise/ternary/device/kernels/** (where), and experimental/quasar/binary_ng/device/kernels{,_ng}/** (descriptor ProgramFactory): all built through KernelDescriptor, so DataMovementKernel/ComputeKernel throws on Quasar at host.
- experimental/quasar/binary/device/kernels/** (eltwise_binary_kernel.cpp, eltwise_binary_sfpu_kernel.cpp, reader_binary_interleaved_start_id.cpp): unreferenced.
- kernel_lib / compute_kernel_api binary/bcast helpers: the kernels_dfb/kernels_qsr compute uses only binary_tiles_init / BINARY_OP / unary_bcast / copy_tile / pack_tile (no DFB sync inside). eltwise_utils_common.hpp has no wait/pop.


---
# Sub-report: unary

# DFB finish() balance audit — unary/SFPU + plus_one + manual_seed + moe_routing_remap

Repo: /localdev/bbradel/tt-metal (branch bbradel-59396_qsr_3mb_llama). Read-only.
finish() semantics assumed (tt-2xx/dataflow_buffer.inl:274): DM waits read_acked==read_posted per TC;
TRISC unpack/pack waits TC occupancy==0. NOTE: DM reserve_back() does NOT touch HW counters (only
push_back -> llk_intf_inc_posted, pop_front -> inc_acked; implicit-sync only via NocOptions::TXN_ID
overloads), so a dangling reserve_back with no push is invisible to finish() as written.

## Test usage (grep of the 4 test trees + modules they import)
| op | call site | config |
|---|---|---|
| ttnn.sigmoid | yolo_ops/test_sigmoid.py:51,74 | [1,80,8400] / [1,80,33600] bf16 TILE, DRAM-interleaved and L1-interleaved |
| ttnn.exp | llama32_1b_quasar/sampling/tt_log_probs.py:367 | only on log-probs path |
| ttnn.log | tt_log_probs.py:477,533 | only on log-probs path |
| ttnn.remainder | tt_log_probs.py:423 (fp32 tensor, int scalar, DRAM) -> binary_composite_op.cpp:711 -> ttnn::unary_remainder (unary factory) | only on log-probs path |
| ttnn.plus_one | models/llama32_1b/model.py:1078-1079, executor.py:2308-2342, tests/{ops,prototype_ops}/test_plus_one.py, qwen3_vl_ops/test_plus_one.py | int32 pos/rot idx |
| ttnn.manual_seed | modules/sampling/sampling_1d.py:398, sampling/tt_sampling.py:712, tests/ops/test_sampling.py:99, test_sampling_1d.py:1226 | always seeds=Tensor,user_ids=Tensor -> Case 4 SetSeedsSetCores |
| ttnn.clamp | NOT used | |
| ttnn.moe_routing_remap | NOT used in the 4 test trees (only models/experimental/ops/quasar/gpt_oss/tt/experts/decode.py:61) | |

Log-probs path (exp/log/remainder): LogProbsCalculator only computes when num_devices in {8,32}
(test_sampling_1d.py:1498-1528); Quasar runs 1x1/1x2 -> returns None -> these unaries are NOT exercised
on Quasar.

## Dispatch on Quasar

### Unary (sigmoid / exp / log / unary_remainder)
ttnn::sigmoid (unary.cpp:570) / exp / log / unary_remainder -> unary_impl -> UnaryDeviceOperation,
single factory `ProgramFactory::create_descriptor` (unary_program_factory.cpp) — Gen1 ProgramDescriptor
with CBDescriptor + ReaderConfigDescriptor/WriterConfigDescriptor/ComputeConfigDescriptor. No
is_quasar/ARCH_QUASAR selection, no Metal2 variant (the *_metal2.cpp readers/writers in the kernels dir are
bound by other ops, not by unary). Program(ProgramDescriptor) -> CreateKernel -> DataMovementKernel ctor
TT_FATAL "DataMovementKernel is not supported on Quasar" (tt_metal/impl/kernels/kernel.hpp:530-532;
ComputeKernel likewise kernel.hpp:738). => On Quasar unary ops FATAL at program build; kernels never run,
so finish() can't trip. (Implication: yolo test_sigmoid cannot currently pass on Quasar regardless.)

Kernels: reader_unary.cpp, writer_unary.cpp, compute eltwise_sfpu.cpp (sigmoid/exp/log/remainder all
map to default eltwise_sfpu.cpp via get_compute_kernel_path, unary_op_utils.cpp:1196).

Balance analysis (would matter once ported):
- Interleaved TILE: reader pushes in_units=npc (1/iter, reserve==push), compute npc (wait/pop 1,
  reserve/push 1 per tile), writer npc (wait/pop 1). group1/group2 split uses same npc for all three
  (enumerate_core_rt_args). Noop cores: all args zero -> 0/0/0. BALANCED.
- RM interleaved: reader pushes npc*chunks_per_row, compute_units = npc*chunks_per_row, writer pops
  npc*chunks_per_row (same k.chunks_per_row for reader/writer). BALANCED.
- Sharded: reader reserve(N)/push(N) with N=in_shard_pages(core); compute o_tiles; writer
  wait(M)/pop(M) M=o_tiles. Balanced iff in_shard_pages==out_shard_pages; get_shard_specs requires native
  L1 sharding and !is_uneven(output), RM pages are shard_elems/tile_hw for both dtypes -> equal in all
  realistic configs. SUSPECT-only for exotic mismatched in/out shard specs (not tested).
- Other compute kernels in dir (identity, hardswish, lgamma[_fast], logit (c_1 self-loop push1/pop1 per
  tile), logsigmoid, mac_tss, where_tss): each tile exactly one wait+one pop on c_0, one reserve+push on
  c_2; hardswish's pop is in exactly one of 3 mutually exclusive Optional branches. BALANCED.
- Side note (not DFB-balance): reader_unary.cpp:55 / writer_unary.cpp uses
  get_local_cb_interface(cb).fifo_page_size, which is stale on Quasar (use get_entry_size()) — needed when porting.

### plus_one
ttnn/cpp/ttnn/operations/experimental/plusone/device/plusone_program_factory.cpp — Metal2 ProgramSpec.
DFB "in0" (1 entry, self-loop PRODUCER+CONSUMER on reader; borrowed from input when sharded).
Kernel reader_plusone_interleaved.cpp: only get_write_ptr(), no reserve/push/wait/pop -> counters 0/0.
BALANCED (finish trivially passes; borrowed DFBs start empty).

### manual_seed
ttnn/cpp/ttnn/operations/reduction/manual_seed/device/manual_seed_program_factory.cpp — Metal2.
Factories 1/2 (SingleSeedToAllCores / SingleSeedSingleCore): compute-only manual_seed_set_seed.cpp, no DFBs.
Factory 3 SingleSeedSetCores (reader_manual_seed_read_user_id.cpp + manual_seed_single_seed_receive_user_id.cpp)
Factory 4 SetSeedsSetCores (reader_manual_seed_read_all_data.cpp + manual_seed_receive_all_data.cpp) <- the one tests use.
- kernel_communication DFB (reader PRODUCER -> compute CONSUMER): reader reserve(1)/push(1)
  (read_all_data.cpp:64,70; read_user_id.cpp:51,56); compute wait(1)/pop(1) (receive_all_data.cpp:22,40;
  single_seed_receive_user_id.cpp:25,38). Both executed unconditionally. BALANCED.
- user_ids DFB (self-loop on reader): reserve_back(1) at read_all_data.cpp:38 / read_user_id.cpp:34,
  never push/pop. seeds DFB (self-loop): reserve_back(1) at read_all_data.cpp:45, never push/pop.
  Class (4) dangling reserve. Under the stated finish() semantics this is HARMLESS (DM reserve_back does
  not modify posted/acked; the noc.async_read used is the non-TXN_ID overload, so no implicit post) ->
  read_posted==read_acked==0. SUSPECT only if #57646 also asserts reserve==push (the pool_2d M+1 vs M
  example is the same pattern: an extra reserve). If so, minimal fix: add `user_ids_dfb.push_back(1);
  user_ids_dfb.wait_front(1); user_ids_dfb.pop_front(1);` (same for seeds) after reading, or simply drop
  reserve_back and use get_write_ptr() as scratch (like plus_one). Exercised: Case 4 on every
  ttnn.manual_seed call in tests (test_sampling.py:99, test_sampling_1d.py:1226, sampling_1d.py:398).

### moe_routing_remap (not exercised by the 4 test trees)
ttnn/cpp/ttnn/operations/data_movement/moe_routing_remap/device/moe_routing_remap_program_factory.cpp —
Gen1 ProgramDescriptor (CBDescriptor + Reader/WriterConfigDescriptor) -> FATALs on Quasar like unary.
Kernel-level imbalances (CONFIRMED in code, unreachable on Quasar today):
- c_0 routing_weights: reader reserve/push 1 (reader_moe_routing_remap.cpp:34,63); writer wait_front(1)
  (writer_moe_routing_remap.cpp:42) with NO pop_front -> 1 entry left (class 2/3).
- c_1 local_weights_idxs: reader reserve/push 1 (reader:43,64); writer wait_front(1) (writer:43) no pop
  -> 1 left.
- c_2 local_weights: writer reserve_back(1)+push_back(1) (writer:36,38) as scratch, no consumer pops ->
  1 posted never acked (class 4/5).
Fix when ported: add routing_weights_dfb.pop_front(1); local_weights_idxs_dfb.pop_front(1) at end of
writer; for c_2 drop reserve/push and use get_write_ptr() as scratch (or add wait_front/pop_front after
the final write barrier). Only user: models/experimental/ops/quasar/gpt_oss/tt/experts/decode.py:61.

## Summary
- No finish()-relevant imbalance reachable on Quasar in this sub-family.
- unary (sigmoid/exp/log/remainder) and moe_routing_remap are Gen1 -> FATAL on Quasar before kernels run.
- manual_seed has dangling reserve_back on two self-loop scratch DFBs (benign under counter semantics; SUSPECT).
- moe_routing_remap kernels have 3 genuine imbalances (CONFIRMED in code; off-path).


---
# Sub-report: reduce

# DFB push/pop balance audit — reductions / argmax / topk (Quasar, PR #57646 finish() semantics)

Read-only audit. Semantics assumed (dataflow_buffer.inl finish_impl ~L274): DM side spins until
read_acked == read_posted on every TC of the DFB (incl. self-loop DFBs where the same DM is both
producer and consumer); TRISC unpack/pack spins until TC occupancy (posted) == 0.
Implicit-sync NoC overloads (credit-bumping) only fire with NocOptions::TXN_ID; none of the kernels
below pass it, so plain noc.async_read/write into/out of a DFB do NOT move counters.

## (a) op -> factory -> kernels (Quasar selection)

| Python op (tests) | C++ path on Quasar | Factory | Kernels |
|---|---|---|---|
| ttnn.sum/mean dim=2 (H) — resnet test_reduce_sum_mean.py H32/64/96 x C64/128, TILE, DRAM-interleaved; llama tt_log_probs sum dim=2; gpt_oss test_sum dim=1 | mainline reduction/generic (no quasar reroute) | ReduceMultiCoreH (metal2), interleaved, num_h_slices=1 (split needs Ht>=20) | reader_unary_transpose_wh_universal_input_cols_partitioned.cpp, compute/reduce.cpp, eltwise writer_unary_interleaved_start_id_metal2.cpp |
| ttnn.sum/mean/max dim=3/-1 (W) — test_reduce_sum_mean.py W variants; tt_log_probs max/sum dim=-1 | mainline | ReduceMultiCoreW (metal2), interleaved | reader_unary_reduce_universal_start_id.cpp, compute/reduce.cpp, writer_unary_interleaved_start_id_metal2.cpp |
| ttnn.max dim=2 (tt_log_probs) | mainline | MultiCoreH (or split if Ht>=20) | as H above |
| avg_pool2d global fast path (resnet GAP) -> experimental::quasar::pool_sum(dim=2) | experimental/quasar/reduction | qsr ReduceMultiCoreH (metal2, interleaved only; rm/width-sharded/negate TT_FATAL) | reader_unary_transpose_wh_universal_input_cols_partitioned_metal2.cpp, compute/reduce_metal2.cpp, writer_unary_interleaved_start_id_metal2.cpp (quasar fork) |
| ttnn.argmax(RM, dim=-1) — llama tests/ops/test_argmax.py (1,1,{1,32},{1024,2048}), qwen3_vl test_argmax (1,1,32,151936) | mainline argmax | ArgMaxMultiCore (ROW_MAJOR + last dim) | reader_argmax_interleaved_multicore.cpp only (DM-only op) |
| ttnn.topk — llama tests/ops/test_topk.py (1,1,{1,32}->32 rows, W 1024/2048, k 1/10/32, uint16 indices_tensor); sampling_1d full-vocab halves | mainline topk (BH-only large-indices route skipped on Quasar; stable rejected on Quasar) | Ht<=2 & pow2 W>=1024 -> MultiCore (if cost check passes) else SingleCore | single: reader_create_index_tensor.cpp, compute/topk.cpp, writer_binary_interleaved.cpp; multi: reader_create_index_local_topk.cpp, topk_local.cpp, writer_local_topk.cpp, reader_final_topk.cpp, topk_final.cpp, writer_final_topk.cpp |

## (b) Findings

### F1 — CONFIRMED (under #57646 semantics): borrowed-shard self-loop DFB pushed, never popped (mainline HEIGHT_SHARDED W reduce)
- File: ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_input_rows_partitioned_sharded.cpp:41 (reserve_back(num_tiles)), :61 (push_back(num_tiles)); no pop.
- DFB: dfb::in1 = SRC1_DFB (borrowed_from INPUT_TENSOR), bound PRODUCER+CONSUMER to the reader only (reduce_op_multi_core_w_program_factory.cpp ~L377-387 "Self-loop").
- Counts per core: posted = num_tiles (shard_Ht*Wt), acked = 0 -> DM finish() spins forever.
- Trigger: ttnn.sum/mean/max dim=-1 with input AND output HEIGHT_SHARDED (use_height_sharding). Comment in kernel says "push ... to leave the buffer balanced" — that was correct for the old CB contract but not for finish().
- Exercised by listed tests: NO (all reduce tests use DRAM/L1 interleaved).
- Fix: after push_back add `dfb_in1.wait_front(num_tiles); dfb_in1.pop_front(num_tiles);` (after the last async_read_barrier), or drop the reserve/push and use the borrowed base address directly.

### F2 — CONFIRMED (same pattern): mainline WIDTH_SHARDED H reduce reader
- File: ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp:54 reserve_back(num_tiles), :83 push_back(num_tiles); no pop.
- DFB: dfb::in1 = SRC1_DFB, self-loop PRODUCER+CONSUMER on reader (reduce_op_multi_core_h_program_factory.cpp L494-503).
- Counts: posted num_tiles, acked 0 -> hang.
- Trigger: H reduce with in+out WIDTH_SHARDED. Not exercised by listed tests. (The quasar fork qsr MultiCoreH TT_FATALs width-sharded, so pool_sum cannot hit it.)
- Fix: same as F1.

### F3 — CONFIRMED (code-level): dense row-major reduce reader `clear_value` self-loop pushed once, never popped (and no reserve)
- File: ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_rm.cpp:77-81 — `rm_fill_buffer_with_identity_pattern(dfb_clear_value.get_write_ptr(), ...)` then `dfb_clear_value.push_back(onepage)`; no reserve_back before, no pop_front ever (used as NoC source via get_read_ptr for the whole kernel).
- DFB: dfb::clear_value (CLEAR_VALUE_DFB), bound PRODUCER+CONSUMER to the reader: W factory L349/354, H factory L459/464.
- Counts: reserves 0 / pushes 1 / pops 0 -> DM finish(): posted 1, acked 0 -> hang. (Missing reserve is a second, pre-existing contract violation.)
- Trigger (mainline, not opt-in on mainline): ttnn.sum/ttnn.mean on a ROW_MAJOR 4D BF16/FP32 interleaved tensor over W or H (rm_base_eligible, reduce_op.cpp L161-194); ALSO stage 2 of the TILE H-split (reduce_op.cpp ~L470-505, row_major_h_dense_path=true) for SUM/MEAN/MAX/MIN H-reduce when Ht >= 20 and NC*Wt < grid cores.
- Exercised by listed tests: NO (all sum/mean/max inputs are TILE; H in tests <= 96 so Ht<=3 < 20). Llama tt_log_probs reductions are TILE too. Would be hit by any H>=640 tall H-reduce with few columns.
- Fix: `dfb_clear_value.reserve_back(onepage);` before the fill, and `dfb_clear_value.pop_front(onepage);` at the very end of reduce_rm_reader() (after the last stage_slab's async_read_barrier). Quasar fork copy (experimental/quasar/.../reader_unary_reduce_rm.cpp:79, legacy CB c_4) has the identical defect but is unreachable (qsr H factory FATALs rm_path; qsr W factory rm_path opt-in default-off).

### Latent / SUSPECT (not reachable today)
- L1: experimental/quasar/reduction/generic/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp `#ifdef OUT_SHARDED cb.wait_front(num_pages);` with NO pop (the mainline eltwise fork pops). Quasar H/HW factories never define OUT_SHARDED -> unreachable. Fix: add `cb.pop_front(num_pages)`.
- L2: same in legacy experimental/quasar/.../dataflow/writer_unary_interleaved_start_id.cpp:27 (OUT_SHARDED wait without pop). Used by qsr W (legacy) and qsr welford factories; OUT_SHARDED never set. Unreachable.
- L3: legacy experimental/quasar/.../dataflow/reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp:49 `cb_in1.reserve_back(num_tiles)` with no push/pop (dangling reserve). No qsr factory references this file (qsr H FATALs width-sharded) — dead code.
- L4: topk single-core compute/topk.cpp with Wt==1: count loop never runs, result_prep gets T pushed but transpose_and_pack waits 2T -> hangs before finish (pre-existing). Validation requires padded width >= 64 (Wt>=2), so unreachable unless the front-end passes W=32 (gpt_oss test_topk has W=32 inputs; assume front-end pads to 64 — not verified, outside sub-family test dirs).

## Kernels checked and found balanced

Reduction (mainline, metal2):
- reader_unary_reduce_universal_start_id.cpp — scaler: prepare_reduce_scaler reserve1/push1; in0 reserve/push batch == num_tiles.
- compute/reduce.cpp — reduce<> WaitAndPopPerTile pops Ht*Wt*NC; scaler waited in helper, popped once at end (L63); out push = outputs.
- writer_unary_interleaved_start_id_metal2.cpp (eltwise fork) — wait/pop 1 per page; OUT_SHARDED wait+pop.
- reader_unary_transpose_wh_universal_input_cols_partitioned.cpp — un-split, split(num_h_slices>1 incl. trailing partial push), welford variants all reserve==push; H compute pops Ht*num_cols (slice_Ht*num_cols when split).
- Cross-kernel H (interleaved): reader Ht*num_cols == compute Ht*Wt(num_cols)*NC(1); out num_cols == writer num_pages. W: reader rows*Wt == compute Ht(rows)*Wt; out rows == writer.
- Single-core HW: reader NC*Ht*Wt == compute REDUCE_SCALAR pops; out NC == writer num_tiles/(Ht*Wt).
- compute/reduce_rm.cpp — acc self-loop: non-last chunks push, non-first chunks reload-pop (balanced for any chunk count); scaler popped at end (L171); tile_in tilize push / reduce pop matched.
- compute/reduce_w_neg.cpp, reduce_h_neg.cpp, reduce_hw_neg.cpp — acc/ineg self-loops balanced (push every step, pop on non-first + final pop); scaler popped at end. (Negate is FATAL'd on qsr path; mainline reachable for ttnn.min only.)
- writer_unary_sharded_metal2.cpp (data_movement) — wait+pop num_units.
- writer_reduce_rm_scalar.cpp — wait/pop paired per tile / per wt chunk.
- welford kernels / reader_unary_reduce_universal_twopass_start_id.cpp — not audited in depth: mainline WelfordReduce TT_FATALs on Quasar (welford_reduce_device_operation.cpp:33).
Reduction (experimental quasar, pool_sum / GAP path):
- reader_unary_transpose_wh_universal_input_cols_partitioned_metal2.cpp — scaler 1/1; in push Ht*num_cols.
- reader_unary_reduce_universal_start_id_metal2.cpp — scaler 1/1; in push num_tiles.
- compute/reduce_metal2.cpp — pops all input, pops scaler at end (L53).
- writer_unary_interleaved_start_id_metal2.cpp (qsr fork) — balanced on the non-OUT_SHARDED path (see L1).
- legacy compute/reduce.cpp (qsr W factory) — pops scaler c_2 at end (L48).
Helpers:
- kernel_lib reduce_helpers_dataflow.inl prepare_reduce_scaler — reserve_back(1)/push_back(1) (L163/L203); calculate_and_prepare_* delegates.
- kernel_lib reduce_helpers_compute.inl reduce<> — waits scaler(1) never pops (BY DESIGN; every caller above pops); input pop per policy (WaitAndPopPerTile/BulkWaitBulkPop pop; Preloaded/Persistent pop nothing -> caller's job; no in-scope caller uses them); accumulator reload pops.
- kernel_lib index_tile_dataflow.hpp generate_index_tile — reserve1, push1 on both return paths.
- kernel/dataflow generate_reduce_scaler.hpp / generate_bcast_scalar(.hpp/_metal2.hpp) — all reserve1+push1 (consumers outside this sub-family must pop).
Argmax:
- reader_argmax_interleaved_multicore.cpp (the path tests use) — no reserve/push/wait/pop at all; DFBs only used for addresses; NoC calls without TXN_ID -> 0/0, balanced.
- reader_argmax_interleaved_metal2.cpp, reader_argmax_tile_layout.cpp, reader_argmax_tile_layout_h.cpp (single-core) — no DFB sync calls.
- argmax_nc: reader_argmax_nc.cpp push num_output_tiles*num_reduce_tiles == argmax_nc_compute.cpp pops; out push/pop num_output_tiles with writer_argmax_nc.cpp.
TopK:
- single-core: reader_create_index_tensor.cpp pushes Wt input + Wt index per row; compute/topk.cpp pops 2+(Wt-2)=Wt each; transposed_* self-loop net 0 per count iter (Case A: +2-1-1+1-1; B/C: +1-1+1-1); result_prep_* self-loop holds T after every count iter, +T at end, transpose_and_pack pops 2T -> 0; outputs T per row popped by writer_binary_interleaved.cpp (Kt==T).
- multi-core (legacy ProgramDescriptor): reader_create_index_local_topk.cpp push Wt_local/row; topk_local.cpp pops Wt_local, transposed self-loop push Wt / (pop Wt,push Wt) x logWt / pop Wt; values/ind Kt per row popped by writer_local_topk.cpp; reader_final_topk.cpp reserve/push Wt_final per row on gathered CB == topk_final.cpp wait/pop Wt; final transposed self-loop balanced; writer_final_topk.cpp pops Kt per row. Fused-keys mode (stable only) is rejected on Quasar.


---
# Sub-report: norm

# DFB finish() balance audit — softmax / rms_norm / layer_norm / distributed norm (Quasar)

Scope: READ-ONLY. Semantics assumed (dataflow_buffer.inl finish_impl:274): DM side spins until read_acked==read_posted on every TC;
TRISC unpack/pack spins until occupancy==0. reserve_back alone does not move counters. So per DFB: total pushed == total popped.
Note: commit 42d50b74afb (#59492, tt-emule CB sanitizer) already fixed most "waited-but-not-popped" sites in these kernels.
What remains is mostly what that sanitizer cannot see: tiles that are pushed but never waited (padding tiles, unread scalars),
and compute self-loop or borrowed outputs that are pushed and never popped.

Paths below are relative to ttnn/cpp/ttnn/operations/normalization/ unless they are absolute.

## Op -> factory -> kernels on Quasar
- rms_norm / layer_norm, interleaved -> LayerNormMultiCoreProgramFactory (layernorm/device/layernorm_op_multi_core.cpp).
  On Quasar, select_interleaved_statistics_backend always returns TILE_REDUCTION, so use_welford is false.
  - Compute: layernorm.cpp (small) or layernorm_large_tensor.cpp (large). Large is not allowed for a row-major affine with tiled input,
    so the llama/qwen rms tests are always small. The qwen layer_norm cases W=1024 and W=4096 are expected small when L1 is empty.
  - Reader: reader_unary_interleaved_ln_rm_gb.cpp (row-major gamma: all rms tests), reader_unary_interleaved_ln.cpp (TILE gamma/beta:
    qwen layer_norm), or reader_unary_interleaved_ln_large_tensor.cpp.
  - Writer: writer_unary_interleaved_start_id_blocked.cpp.
  - Default blk: rms_norm uses fp32_dest_acc=false, so blk=8. layer_norm uses fp32_dest_acc=true, so blk=4.
- rms_norm / layer_norm, sharded -> LayerNormShardedProgramFactory (sharded_layernorm_factory_helpers.cpp).
  - Kernels: reader_mcast_sender/receiver_unary_sharded_ln.cpp, writer_unary_sharded_ln_rm_gb.cpp (row-major gamma) or
    writer_unary_sharded_ln.cpp, and compute layernorm_sharded.cpp (Welford is never selected on Quasar).
  - IN0 is borrowed from the input and OUT is borrowed from the output. For non-distributed runs writes_back is always false
    (layernorm_op_multi_core_sharded.cpp:116), so OUT is a compute self-loop.
  - Tests: llama graph_ops 8x4 [32,64] bw=2 and 8x8 [32,32] bw=1; debug_ops 8x4; qwen 8x5 [32,64]. All are RMSNORM, use a row-major
    gamma, have no beta and no residual, and use legacy_reduction=0.
- rms_norm_pre/post_all_gather with interleaved input -> layernorm_distributed factories.
  - Pre: reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp, rmsnorm_pre_allgather.cpp (layernorm_pre_allgather.cpp for LN),
    writer_unary_interleaved_start_id_blocked.cpp.
  - Post: reader_unary_interleaved_ln_rm_gb_post_allgather.cpp, rmsnorm_post_allgather_metal2.cpp (layernorm_post_allgather.cpp for LN).
  - With sharded input these ops route instead to the sharded LayerNorm factory with PRE/POST_ALL_GATHER (no tests).
- softmax:
  - yolo dim=2 on [1,4,16,A] -> GeneralHSmall (Ht=1): moreh_softmax/device/kernels/{reader_moreh_softmax_h, moreh_softmax_h, writer_moreh_softmax_h}.cpp.
  - dim=-1 (gpt_oss [1,4] and [128,4]) -> AttentionOptimized: attention/{reader_unary_interleaved_sm, softmax, writer_unary_interleaved_start_id_blocked_sm}.cpp
    and the large variants.
  - The sharded attention softmax has no tests.

## Findings

### F1 CONFIRMED, hit by tests: sharded layernorm_sharded.cpp leaves EPS pushed but never popped
- Producer: writer_unary_sharded_ln_rm_gb.cpp:72-74 and writer_unary_sharded_ln.cpp:68-70 call generate_bcast_col_scalar, which does
  reserve(1)+push(1) on every non-idle core.
- Consumer: compute/layernorm_sharded.cpp never calls wait_front or pop_front on dfb::eps. It only indexes it at :520-521 (add_tiles) on
  all-to-all cores with enable_sqrt. That read is also an unsynchronized read with no wait.
- Counts: 1 pushed, 0 popped on every core. The writer's DM finish spins forever.
- Tests: all sharded rms_norm tests (llama graph_ops 8x4 and 8x8, debug_ops test_quasar_sharded_rmsnorm_mcast_fault.py, qwen test_rms_norm.py 8x5).
- Fix: in layernorm_sharded.cpp, under #ifndef IDLE_CORE (the kernel is non-Welford only), add `DataflowBuffer dfb_eps_obj(dfb_eps);
  dfb_eps_obj.wait_front(1);` near the top and `dfb_eps_obj.pop_front(1);` at the end. Every non-idle compute binds EPS as CONSUMER
  (factory :1131/:1161).

### F2 CONFIRMED, hit by tests: sharded OUT, a borrowed self-loop, is pushed and never popped
- Compute pushes num_tiles_per_block into dfb::out. Which line does it depends on the affine:
  - no gamma and no beta: dfb_im is out (:594);
  - gamma only: dfb_outgamma is out (:648), which is the test case;
  - beta: :692.
- Nothing pops it. writes_back is false for non-distributed runs, so the writer runs SKIP_WRITE_BACK and the factory binds OUT as a
  compute self-loop (:1127). Compute pack and unpack both see occupancy num_tiles_per_block.
- Tests: the same as F1.
- Fix: at the end of layernorm_sharded.cpp, `dfb_out.wait_front(num_tiles_per_block); dfb_out.pop_front(num_tiles_per_block);`.
  This is safe because the data is already in the output tensor.

### F3 CONFIRMED, hit by qwen tests: interleaved layernorm.cpp GAMMA/BETA push round_up(Wt,blk) but pop Wt
- Producer: the reader pushes block.full_block_size() for each block. See reader_unary_interleaved_ln_rm_gb.cpp:164,197 (gamma) and
  :203,236 (beta), and reader_unary_interleaved_ln.cpp:178-183 via layernorm_dataflow_utils.h:183,190.
- Consumer: compute/layernorm.cpp:341-348 and :380-387 wait with Upfront/PopNone. The end pop at :431/:434 is pop_front(Wt).
- Leftover: round_up(Wt,blk)-Wt tiles. The sanitizer missed this because the padding tiles are never waited.
- Tests: qwen3 test_rms_norm.py cases [1,1,32,128], [1,1,8,128], [1,32,4096,128] and [1,8,4096,128]. W=128 gives Wt=4 with blk=8,
  so 8 are pushed and 4 popped.
- Not hit: llama W=2048, qwen rms W=2560, and qwen layer_norm W=1024/4096 with blk=4.
- Fix: replace pop_front(Wt) with wait_front(total_buffer_size)+pop_front(total_buffer_size). total_buffer_size is defined at :172.

### F4 CONFIRMED, reachable, not in tests: layernorm_large_tensor.cpp leaves SCALER unpopped for RMSNORM
- Producer: reader_unary_interleaved_ln_large_tensor.cpp:103-119 pushes 1, or 2 when W%32!=0.
- Consumer: compute :421-423 pops only under #ifndef RMSNORM, even though the variance reduce (:267-271) uses the scaler. The
  RMSNORM path also never waits for the scaler (a race).
- Trigger: rms_norm with a large tensor and a TILE or absent weight.
- Fix: wait once before the loop and pop unconditionally.

### F5 CONFIRMED, not in tests: softmax_sharded.cpp
- (a) IN0 is borrowed from SRC and is a compute self-loop. Compute pops it (PopPolicy::AtEnd: :47 via calc_numeric_stable, :92, :126)
  but nobody pushes it, so block_h*block_w are popped against 0 pushed. This also trips the pop-without-post HW tile-counter fault
  noted in layernorm_sharded.cpp:370.
- (b) OUT0 is borrowed from DST and is a compute self-loop. It is pushed block_w per row (:159-161) and never popped.
- (c) FUSED_ATTN with mask_sharded_resident (borrowed, causal, sharded mask) pops block_w per row with mask_pop=AtEnd and mask_wait=None,
  with no push.
- Fix:
  - drop the in0 pops (PopPolicy::None) or add a matching compute reserve+push of the shard up front;
  - pop out0 at the end;
  - for the resident mask, use PopPolicy::None.
- Factory: softmax_program_factory_attention_optimized_sharded.cpp:153-196 and :299-306.

### F6 CONFIRMED, not in tests: attention softmax.cpp with FUSED_SCALE_MASK and MASK_PADDED_DATA leaves MASK_PADDED unpopped
- Producer: writer_unary_interleaved_start_id_blocked_sm.cpp:39-51 pushes 1 when mask_padded_data.
- Consumer: compute/softmax.cpp waits it with PopPolicy::None at :174-179, but the end-of-kernel pop at :396-400 is guarded by
  `MASK_PADDED_DATA && !FUSED_SCALE_MASK`. So the fused case pushes 1 and pops 0.
- Trigger: ttnn.scale_mask_softmax with a non-tile-aligned W on the small kernel. The large kernel (:577) pops correctly.
- Fix: change the guard at :396 to `#if defined(MASK_PADDED_DATA)` and construct dfb_mask_padded_obj under the same guard (:127).

### F7 CONFIRMED, not in tests: interleaved layernorm_post_allgather.cpp (LayerNorm, not RMS) leaves GAMMA/BETA unpopped when Wt == dfb_length
- Producer: the reader pushes gamma/beta once, Wt tiles at ncht==0 (reader_unary_interleaved_ln_rm_gb_post_allgather.cpp:148-172).
- Consumer: compute :59-62 waits Cumulative and pops None, and there is no end-of-kernel pop (:207-208 pop only eps and reduce).
- Counts: Wt pushed, 0 popped.
- Trigger: ttnn.layer_norm_post_all_gather with gamma or beta. The rms metal2 kernel pops correctly (:189-194).
- Fix: at the end, add `#ifdef FUSE_GAMMA if (Wt==dfb_length) wait_front(Wt); pop_front(Wt)`, and the same for beta.

### F8 CONFIRMED, not in tests: sharded POST_ALL_GATHER (layernorm_sharded_post_allgather.cpp + writer_unary_sharded_ln*.cpp)
- (a) SCALER: the writer pushes 1 and the factory binds it as a writer self-loop (sharded_layernorm_factory_helpers.cpp:983-986, "never
  drained"). Nobody pops it, so the writer DM finish hangs. Fix: the writer pops it (wait+pop) right after generating it, or stop
  generating it on the post path.
- (b) EPS: the writer pushes 1 on every core. Compute pops it only on IS_ALLGATHER_WORKER && enable_sqrt (:236/:250), so it is left
  over on non-all-to-all cores and on two-stage non-second-stage cores. Fix: an unconditional wait+pop at the end on the other cores.
- (c) SCALER_GLOBAL: pushed on every all-to-all core and popped only if enable_sqrt (:413-418). Left over on two-stage non-second-stage
  cores. SUSPECT, since it only matters if the post path uses a two-stage reduce.
- (d) OUT: when !writes_back (skip_write_back), it is a self-loop that is pushed and never popped. Same as F2.

### F9 CONFIRMED (out) / SUSPECT (deadlock), not in tests: sharded PRE_ALL_GATHER
- OUT is a self-loop (factory :1145). layernorm_sharded_pre_allgather.cpp:304-330 pushes into dfb_reduction_out, which is OUT when the
  reduce is single-stage or on the second-stage reader. It is never popped, so the end-of-kernel drain hangs.
- Separately, reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp:217-219 and :228-231 wait and pop ex2 on every all-to-all
  receiver, unconditionally. Compute pushes ex2 only for two-stage non-second-stage cores, so single-stage receivers with more than one
  all-to-all worker deadlock before finish is ever reached. This is pre-existing.

### F10 SUSPECT, not in tests
- reshard_writer.hpp:39-71. num_tiles_to_write_in_current_segment already multiplies by block_ht, and is then iterated inside a
  block_ht loop. If block_ht>1, the writer waits and pops block_ht times more OUT tiles than compute pushed. This only applies to sharded
  post-allgather with writes_back.
- Interleaved post-allgather (both kernels) when cb_length < Wt (huge rows): the reader pushes gamma every row (dfb_iterations!=1),
  but rmsnorm_post_allgather_metal2.cpp waits gamma(Wt), which exceeds capacity, and pops it once. When dfb_iterations==1 with
  leftovers, the reader pushes gamma only at ncht==0 while LN post pops per block every row. Both are pre-existing and functional.
- layernorm_large_tensor.cpp with ROW_MAJOR input: the E[x] pass waits on dfb_in, which only the later tilize fills, so it deadlocks.
  IN_RM sees 3 pushes against 2 pops. This is a pre-existing functional bug.
- reader_unary_interleaved_ln_rm_gb.cpp with row-major input and a row-major gamma: there is no TILIZE_IN support, so expect a JIT
  failure. This is not finish-related.

## Checked and balanced
- moreh_softmax_h.cpp + reader_moreh_softmax_h.cpp + writer_moreh_softmax_h.cpp (yolo test): in Ht, x_minus_max, exps, max, tmp,
  recip_sum_exps, mask, max_scaler and sum_scaler are all balanced. The end pops are at :216-218.
- attention softmax.cpp, non-fused (gpt_oss tests):
  - in0 pad: the reader pushes and drain_dfb_pad consumes.
  - out0 pad: compute pushes and the writer drains.
  - exps and x are cycled; max, recip, the scalers and mask_padded (non-fused) are balanced.
- softmax_large_tensor.cpp + reader_unary_interleaved_sm_large_tensor.cpp: 2 or 3 passes, with stream_pad drained per pass. mask_padded,
  the scalers, recip and max_final are balanced.
- layernorm.cpp small, interleaved: EPS, SCALER, IN, INB, X, XMM, XMM2, EX, EX2, EX2PE, FUSION, OUT and the row-major IN_RM/OUT_RM are
  balanced (see F3 for gamma/beta).
- layernorm_large_tensor.cpp: IN, INB, GAMMA, BETA, XMM, XMM2, FUSION, ACCUMULATE, EX, EX2, EX2PE and EPS are balanced. SCALER is
  balanced for LN only (F4).
- layernorm_sharded.cpp and the sharded mcast sender/receiver readers (non-distributed):
  - IN0 (borrowed, not pushed or popped), IN1 and in-place pre-add, XMM (aliased to in0 for RMS), X/XMM2, EX_PARTIAL(2), EX_EXTERNAL(2)
    including the two-stage dummy push/pop, EX, EX2, EX2PE, EX_GLOBAL (block_h per pass), SCALER, SCALER_GLOBAL (all-to-all only, with
    matching writer and compute placement), GAMMA, BETA, COL_MASK and MASK_SCRATCH are balanced.
  - The open problems are EPS (F1) and OUT (F2).
- rmsnorm_pre_allgather.cpp, layernorm_pre_allgather.cpp, reader_..._pre_allgather.cpp and the writer (block_size=1): inp, res, fused,
  x2, out and reduce are balanced.
- rmsnorm_post_allgather_metal2.cpp + reader_..._post_allgather.cpp (llama post test, Wt fits): stats, inp, var, recip_sqrt_var,
  x_normed, times_gamma_out, out, eps, reduce and gamma/beta (Wt pushed once, popped once) are balanced.
- layernorm_post_allgather.cpp: stats, stats_reduced, mean_squared, var, recip, x_minus_mean, x_normed, eps and reduce are balanced
  (gamma/beta in F7).

## Not audited
- layernorm_pre_allgather_2d.cpp, rmsnorm_pre_allgather_2d.cpp and reader_layernorm_preallgather_2d.cpp (only with use_2d_core_grid; no tests).
- The Welford variants (rejected on Quasar).
- moreh softmax w/w_large/h_large/c_large (not reached by tests).
- reader_unary_sharded_sm_rm_mask / causal_mask_hw_dims (only skimmed).


---
# Sub-report: sampling

# DFB finish() balance audit — sampling / scatter / scatter_add / gather

Semantics used (tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:274 finish_impl):
- DM side: spin until read_acked == read_posted on every TC of the DFB (i.e. total push_back == total pop_front).
- TRISC UNPACK/PACK: spin until TC occupancy (posted) == 0.
- reserve_back alone does not touch posted/acked on DM or PACK, so a dangling reserve (no push) is benign for finish;
  pushes-without-pops are what hang. Implicit-sync credits (ptiles_read_/ctiles_written_) only move for
  Noc::async_read/async_write<TXN_ID> overloads; none of the kernels below use them.

## Op -> factory -> kernels (Quasar)

| op | factory (Quasar) | kernels |
|---|---|---|
| ttnn.sampling | SamplingProgramFactory (Metal2 ProgramSpec, one per-core writer) — reduction/sampling/device/sampling_program_factory.cpp | reader_values_indices_tensor.cpp, compute/sampling.cpp, writer_interleaved.cpp |
| ttnn.scatter (no reduce, any dtype) / scatter_add int32 (llama penalties) | ScatterProgramFactory (Metal2) | reader_scatter.cpp, writer_scatter.cpp, ../common.hpp (load_to_dfb / write_to_output) |
| ttnn.scatter_add / scatter reduce, BF16 | ScatterReduceBfloat16ProgramFactory (Metal2) | reader_bf16_reduction_scatter.cpp, writer_bf16_reduction_scatter.cpp |
| ttnn.gather | native: SingleRowSingleCore/SingleRowMultiCore/RmSingleRowSingleCore/RmSingleRowMultiCore; codegen (bf16 TILE interleaved, default route): Interleaved/Tiled/Streaming | ALL are Gen1 ProgramDescriptor + CBDescriptor + Reader/WriterConfigDescriptor -> CreateKernel -> DataMovementKernel ctor TT_FATAL "DataMovementKernel is not supported on Quasar" (tt_metal/impl/kernels/kernel.hpp:530). Never runs on Quasar => no finish() exposure. |

sampling does not invoke the topk op; it uses the topk LLKs inline in compute/sampling.cpp (covered below).

## Findings

### F1 — CONFIRMED (finish hang): sampling writer DM self-loop on `k` and `p` DFBs pushed, never popped
- File: ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/writer_interleaved.cpp
  - `k`: lines 72 reserve_back(1), 77 push_back(1); no pop anywhere.
  - `p`: lines 83 reserve_back(1), 88 push_back(1); no pop anywhere.
- Role: writer is bound as BOTH PRODUCER and CONSUMER of SAMPLING_K / SAMPLING_P (factory lines ~424-445); no other kernel touches them.
- Counts: posted=1, acked=0 per DFB per core -> DM finish() spins forever (unconditional, every config, every core).
- Exercised: every ttnn.sampling call — tests/ops/test_sampling.py, tests/prototype_ops/test_sampling.py
  (both currently module-skipped), modules/sampling/sampling_1d.py:405, sampling/tt_sampling.py:718,
  tests/modules/sampling/test_sampling_1d.py:1253.
- Fix (minimal): after reading the per-core value, release the entry:
  `const uint32_t k = k_ptr[core_id]; dfb_k.pop_front(1);` (after line 80) and
  `const uint32_t p = p_ptr[core_id]; dfb_p.pop_front(1);` (after line 91).
  (Alternative: drop reserve/push and treat as scratch like `temp`/`out`, but pop is the smallest change.)
- Note: compute/sampling.cpp:491-503 already got the "pop leftover waited tiles" treatment (cur_max, topk_mask,
  temp, scaler_max, scaler_sum) — writer k/p were missed.

### F2 — CONFIRMED (pre-existing deadlock, cross-kernel count mismatch, independent of finish): Wt == 1 (W = 32)
- Validation (sampling_device_operation.cpp:87-93) accepts Wt=1 because `(1 & 0) == 0`.
- Reader (reader_values_indices_tensor.cpp:38-52) pushes Wt=1 tile to `input_values` and 1 to `index`.
- Compute top_k (compute/sampling.cpp:245-277) loops `wt=0; wt<Wt; wt+=2` and does `wait_front(2)` / `pop_front(2)`
  on both -> waits for a 2nd tile that is never produced -> hang. Also packs 2 tiles into
  `input_transposed`/`index_transposed` whose capacity is Wt=1 (factory: num_entries=Wt) -> overflow if it ever got there.
- Symptom match: test_sampling.py skip reason "writer at CWFW, TRISC1 at MWDD, for both k=1 and k=32" — writer
  waits local_vals forever, math idles because unpack is stuck at wait_front(2).
- Exercised: tests/ops/test_sampling.py and tests/prototype_ops/test_sampling.py use K=_MAX_TOP_K=32 ->
  topk_values [1,1,batch,32] -> Wt=1 (skipped). sampling_1d.py on a 1x1 mesh: width = max_top_k*num_devices = 32 -> Wt=1.
  Upstream WH/BH tests only use W>=64 (tests/ttnn/unit_tests/operations/reduce/test_sampling.py:186-187,303).
- Fix options: (a) host: reject Wt<2 in validate and pad W to 64 in the caller (values -inf, dummy indices);
  (b) factory/kernel: Wt_eff=max(Wt,2), size transposed DFBs to Wt_eff, reader pushes a -inf padding values tile
  + index tile j=1 when Wt==1, compute uses Wt_eff/logWt_eff=1. (b) keeps the op API unchanged.

## Balanced (checked) — sampling DFB ledger (Ht=1, Kt=1, per core)
- input_values: reader +Wt; compute -2 per pair => -Wt (Wt even). OK (except F2).
- index: reader generate_index_tile (kernel_lib/index_tile_dataflow.hpp:41, 1 push/call) +Wt; compute -Wt. OK.
- final_indices_rm: reader +num_users; writer wait/pop num_users. OK.
- scaler_max / scaler_sum: writer prepare_reduce_scaler (reduce_helpers_dataflow.inl:163/203) +1 each; reduce helper waits never pops; compute pops 1 each (sampling.cpp:502-503). OK.
- topk_mask: writer generate_mask<mask,PNHt=1>(Sk_chunk_t=ids_per_batch/32=1) +1; compute wait Kt, pop Ht*Kt=1. OK.
- temp: writer raw NOC read (no FIFO) then generate_bcast_unary_scalar +1; compute mul_bcast_scalar waits, final pop 1. OK.
- rand_tile: compute +1; writer -1. OK.
- input_transposed / index_transposed (compute self-loop): +Wt, per merge iter -Wt/+Wt, final -Wt. OK.
- values (compute self-loop): top_k +1; add/mul/sub_exp inplace -1/+1; reduce WaitUpfrontNoPop 0; mul_block_bcast_cols -1. OK.
- cur_max: reduce +1; sub_exp waits; final pop 1. OK.
- cur_sum: reduce +1; recip inplace -1/+1; mul_block_bcast_cols -1. OK.
- local_vals: compute +1; writer -1. OK.  output_ind: compute +Kt=1; writer -1. OK (Ht=1 always).
- output: no FIFO ops on either side (plain SRAM window). OK.
- k, p: NOT OK (F1).

## Other kernels checked and found balanced
- scatter/reader_scatter.cpp + writer_scatter.cpp: input/index/source are reader self-loops (load_to_dfb +1, wait/pop -1 per chunk); output reader +1 per (stick, io-chunk), writer -1 per (stick, io-chunk) with identical chunk count (output stick == input stick). Balanced. Exercised by qwen3_vl_ops/test_scatter.py ([2766,2560] bf16, idx/src [2752,2560], no reduce) and llama scatter_add (int32 [32, vocab/ndev] RM -> ScatterProgramFactory with ADD; tests/modules/sampling/test_penalties_1d.py).
- scatter/reader_bf16_reduction_scatter.cpp + writer_bf16_reduction_scatter.cpp: same structure; fp32 accumulator is a Scratchpad (not a DFB); output reserve/push after input pop. Balanced. (bf16 scatter_add/reduce only; not hit by current Quasar tests.)
- scatter/kernels/common.hpp load_to_dfb (+1) / write_to_output (wait/-1): balanced per call.
- gather native kernels (gather_{reader,writer}_{single_row,rm_single_row}_{single,multi}_core.cpp) and codegen kernels (gather_{reader,writer}{,_tiled,_streaming}.cpp): reader/writer counts match by inspection (incl. matching `break` on current_index_tile_id >= Wt_index in multi-core pair; streaming n_chunks*chunk_tiles both sides; tiled per-row Wt_input). Irrelevant on Quasar today: every gather factory is Gen1 and TT_FATALs at kernel creation. Used by sampling/tt_log_probs.py (only for multi-device 8/32) — not on the Quasar e2e path.
