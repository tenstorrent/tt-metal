# DFB imbalance audit — family: matmul

## (a) COLLECTED: Quasar-reachable factories and kernels

M = ttnn/cpp/ttnn/operations/matmul/device/kernels
Q = ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels
MM = ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels

### Mainline ttnn.matmul / ttnn.linear (all factories use create_program_artifacts = Metal 2.0 on every arch)
Callers: llama32_1b_quasar (attention_1d, mlp_1d, lm_head, tests/ops|graph_ops|debug_ops test_linear*, test_quasar_matmul_*), qwen3_vl_ops/test_linear.py (DRAMSharded 100%, Mcast2D 98%).
- MultiCore            : M/dataflow/reader_bmm_8bank_output_tiles_partitioned_metal2.cpp, M/dataflow/writer_unary_interleaved_start_id.cpp, M/compute/bmm_metal2.cpp
- ReuseOptimized       : M/dataflow/reader_bmm_tile_layout_in0.cpp, M/dataflow/reader_writer_bmm_tile_layout_in1.cpp, M/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp
- Mcast2D              : M/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp, ..._in0_sender_receiver_padding_block_sharded_metal2.cpp, ..._in0_receiver_metal2.cpp, ..._in1_sender_writer_padding_metal2.cpp, ..._in1_receiver_writer_padding_metal2.cpp, compute metal2
- Mcast1D (mcast_in0 / mcast_in1) : same metal2 set as 2D
- Mcast1D gather_in0 / global_cb : legacy MeshWorkload (CreateKernel) — not reachable on Quasar test configs
- DRAMSharded          : M/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp, M/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp, compute metal2
- BatchedHSDRAMSharded : M/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp, ..._in1_sender_dram_sharded_height.cpp, compute metal2

### ttnn.experimental.quasar.matmul / linear (resnet50 quasar fc layer, test_linear.py, test_matmul_dram_weights_kspill.py, llama tests/graph_ops/test_linear_small_grid.py, prototype_ops/test_linear.py)
- MultiCore            : Q/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp, Q/dataflow/writer_unary_interleaved_start_id.cpp, Q/compute/bmm.cpp
- ReuseOptimized       : Q/dataflow/reader_bmm_tile_layout_in0.cpp, Q/dataflow/reader_writer_bmm_tile_layout_in1.cpp, Q/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp
- Mcast1D/2D (create_program_artifacts): Q/.../*_metal2.cpp (same 5 dataflow + compute metal2), Q/dataflow/blank.cpp, Q/compute/blank.cpp (noop pad cores)
- DRAMSharded (create_descriptor): Q/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp, Q/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp, Q/compute/bmm_large_block_zm_fused_bias_activation.cpp (NON-metal2)
- BatchedHSDRAMSharded (create_descriptor): Q/..._height.cpp x2, Q/compute/bmm_large_block_zm_fused_bias_activation.cpp
- Unified              : Q/dataflow/unified_matmul_reader.cpp, Q/dataflow/unified_matmul_writer.cpp, Q/compute/unified_matmul_compute.cpp
- gather_in0           : legacy MeshWorkload — not reachable

### ttnn.experimental.minimal_matmul (llama tests/ops|graph_ops|prototype_ops test_minimal_matmul.py, qwen3_vl_ops/test_minimal_matmul.py)
- ProgramFactory::create_program_artifacts: MM/dm_in0_sender_metal2.cpp, MM/dm_in1_sender_out_metal2.cpp, MM/compute_metal2.cpp, MM/matmul_dataflow_common_metal2.hpp

### ttnn.experimental.all_gather_matmul_async
- Uses legacy matmul_multi_core_reuse_mcast_{1d,2d}_optimized_helper (CreateKernel, non-metal2 kernels). Tests are multi-device only (pytest.skip "multi-device op" on single device) -> NOT reachable on Quasar.

### ttnn.sparse_matmul
- Not called by any test in scope (only gpt_oss_ops) -> out of scope.

## Note on what finish() actually checks (tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:274 finish_impl)
- DM (producer or consumer): spins until overlay read_acked == read_posted for every TC of the DFB.
- TRISC unpack/pack: spins until tile_counters[tc].posted (tiles currently available) == 0.
- reserve_back_impl (line 143) is stateless (waits for free space; nothing is recorded), so a dangling
  reserve_back with NO matching push does not change posted/acked and, per the code in this tree, cannot
  by itself trip finish(). What trips finish() is: (a) a consumer that wait_front()s but never pops (or pops
  fewer than the producer pushed), (b) a producer self-loop that pushes without popping, (c) cross-kernel
  push total > pop total. Dangling reserves are still listed below (class 1) since the caller flagged the
  pool reserve-ahead pattern, but are marked BENIGN-per-current-finish_impl.

## (b) FINDINGS

### F1 — CONFIRMED, EXERCISED. Quasar metal2 in1 writers: OUT_SHARDED wait_front with NO pop_front
- Files:
  - ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp:497-500
  - ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding_metal2.cpp:252-255
  ```
  #if OUT_SHARDED
      cb_out.wait_front(batch * out_num_nonzero_subblocks_h * out_num_nonzero_subblocks_w * out_subblock_w * out_subblock_h);
  #endif
  ```
- DFB: dfb::cb_out (borrowed from the output tensor, quasar 1D factory :5753 / :6768, 2D likewise). Role: consumer (DM writer).
- Trigger: any output that is L1-sharded -> factory sets OUT_SHARDED (quasar 1D mcast_in0 :5669, mcast_in1 :6684-6685; 2D :3660-3662).
- Counts per core: compute pushes batch*per_core_M*per_core_N tiles; writer waits for the same count, pops 0 -> posted != acked -> DM finish() spins forever.
- Exercised: YES
  - resnet50 quasar fc: ttnn_functional_resnet50.py:818 ResnetLinear(output_mem_config=L1_WIDTH_SHARDED) -> 1D mcast_in0 -> in1_sender_writer. (test_resnet50_e2e.py, test_resnet50_layers_1_3.py, ...)
  - resnet tests/ops/test_linear.py (L1_WIDTH_SHARDED out, mcast_in0=True)
  - resnet tests/ops/test_matmul_dram_weights_kspill.py (L1_HEIGHT_SHARDED out, mcast_in0=False -> in1_sender_writer AND in1_receiver_writer)
  - llama tests/graph_ops/test_linear_small_grid.py if output sharded (via quasar.matmul 1D).
- Fix: add `cb_out.pop_front(<same count>);` after each wait_front (exactly what the mainline twin does at
  matmul/.../reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp:839-841 and ..._receiver_writer_padding_metal2.cpp:250-252).
  Also see F3 for the count formula caveat (fc config uses out_block_w == per_core_N, so the count is right there).

### F2 — CONFIRMED (latent / not exercised by in-scope tests). Quasar ReuseOptimized reader_writer: OUT_SHARDED wait_front with NO pop
- File: ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_writer_bmm_tile_layout_in1.cpp:162-164
  `cb_out.wait_front(batch * out_num_subblocks_h * out_num_subblocks_w * out_subblock_w * out_subblock_h);` — no pop.
- DFB: dfb::out; role consumer (DM). Trigger: quasar MatmulMultiCoreReuseProgramConfig with sharded output (factory :354 sets OUT_SHARDED).
- Counts: compute pushes batch*S*subblock_tiles == waited count; pops 0.
- Exercised: not seen in scope (the quasar ReuseOptimized path needs MatmulMultiCoreReuseProgramConfig + sharded out; prototype_ops/test_linear.py uses default interleaved). SUSPECT-reachable through quasar default-config selection with sharded output.
- Fix: add matching pop_front (mainline twin does: matmul/.../reader_writer_bmm_tile_layout_in1.cpp:184-189).

### F3 — CONFIRMED (latent). OUT_SHARDED writer drain count uses out_block, not per_core shard (mainline AND quasar 1D/2D writers)
- Files: matmul/.../reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp:836-842,
  matmul/.../reader_bmm_tile_layout_in1_receiver_writer_padding_metal2.cpp:247-253 (+ the quasar twins in F1).
- Count drained = batch * out_num_nonzero_subblocks_h * out_num_nonzero_subblocks_w * sbh * sbw
  = batch * out_block_h * out_block_w (RTA out_num_nonzero_subblocks_{h,w} = out_block_{h,w}/out_subblock_{h,w}; mainline 2D factory :155-158/:192-197).
- Compute pushes batch * num_blocks_h_dim * num_blocks_w_dim * out_block tiles = batch * per_core_M * per_core_N.
- The factories explicitly allow sharded output with per_core_M != out_block_h (mainline 2D :406, 1D :3337 `do_not_inplace_interm0_out_*`), and out_block_w < per_core_N.
  In that case pops < pushes by batch*(per_core_M*per_core_N - out_block_h*out_block_w) -> finish() hang on the writer.
- Exercised: no in-scope config found (llama/qwen 2D outputs are DRAM interleaved; resnet fc uses out_block_w == per_core_N, per_core_M == out_block_h == 1; test_quasar_qkv_matmul_dfb 1D uses default out_block = per_core).
- Fix: drain `batch * num_blocks_h_dim * num_blocks_w_dim * out_block_h * out_block_w` (or simply batch*per_core_M*per_core_N) in the OUT_SHARDED branch.

### F4 — CONFIRMED (latent). Mainline block-sharded in0 sender: self-loop dfb::in0_sharded push without pop
- File: ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded_metal2.cpp:130 (`dfb_in2.reserve_back(in0_num_tiles)`) and :388 (`dfb_in2.push_back(in0_num_tiles)`); nothing ever wait/pops it.
- DFB: in0_sharded (borrowed from IN0). Bindings: the in0 sender kernel is BOTH producer and consumer (mainline 1D factory :3961-3972, 2D :1281-1288) -> self-loop.
- Counts: pushes = batch*in0_block_num_tiles, pops = 0 -> DM finish() waits read_acked==read_posted forever.
- Trigger: mainline ttnn.matmul/linear with L1-sharded in0 on 1D mcast_in0 (WIDTH_SHARDED in0) or 2D (BLOCK_SHARDED in0).
- Exercised: not by in-scope tests found (qwen3 2D cases and llama 2D/1D cases all have DRAM-interleaved in0; DRAM-sharded configs use the dram_sharded kernel). Reachable by any mainline sharded-in0 1D/2D linear.
- Fix: since the shard is only addressed via get_read_ptr, drop the reserve/push entirely (the comment at :385 says it was added "to leave the buffer balanced" — on Quasar it does the opposite), or add `dfb_in2.wait_front(in0_num_tiles); dfb_in2.pop_front(in0_num_tiles);` right after the push. Quasar twin kernel already avoids this (no in2 DFB; uses TensorAccessor base address).

### F5 — CONFIRMED (latent / unreachable). writer_unary_interleaved_start_id OUT_SHARDED: wait_front with no pop
- Files: matmul/.../dataflow/writer_unary_interleaved_start_id.cpp:29-30; experimental/quasar/matmul/.../dataflow/writer_unary_interleaved_start_id.cpp:24-25.
- Trigger: OUT_SHARDED — never set by either MultiCore factory (no OUT_SHARDED define for this writer), so unreachable today.
- Fix: add pop_front(num_pages) for hygiene.

### F6 — PATTERN MATCH to pool reserve-ahead (class 1), BENIGN per current finish_impl. DRAM-sharded in1 reader reserve-ahead
- Files: matmul/.../dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp:131 (reserve 2 blocks up front), :170-174 (push 1 + reserve 2 per steady-state iteration), :190 (final push 1);
  quasar twin: experimental/quasar/.../reader_bmm_tile_layout_in1_sender_dram_sharded.cpp:114, :133-137, :153.
- DFB in1, producer DM. Pushes = num_blocks (balanced vs compute pops = num_blocks; B==1 enforced by factory :1311). After the loop one extra block (in1_block_num_tiles) is still "reserved" when num_blocks >= 2 (reserves cover num_blocks+1 blocks).
- Exercised: YES — every DRAM-sharded matmul with num_blocks >= 2 (qwen3_vl_ops/test_linear.py DRAMSharded cases = 100% of qwen matmuls; llama decode QKV/WO/FF1/FF2/FF3/lm_head; llama graph_ops/test_linear.py; debug_ops/test_quasar_matmul_dram_sharded.py).
- Impact: none with the finish_impl in this tree (reserve is not recorded); flag only if PR #57646's finish adds a reserved-vs-pushed check. Fix if needed: skip the reserve on the last iteration (`if (block + 1 < num_blocks) reserve_back(2*...)`).

### F7 — SUSPECT (pre-existing; would hang the consumer, not finish). IN1_SHARDED one-shot push vs per-batch pops
- matmul/.../reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp:259-261 and quasar twin :178-180 push in1_block_num_tiles*num_blocks_inner_dim ONCE; compute pops batch*num_blocks_h_dim*num_blocks_w_dim*num_blocks_inner_dim blocks. Balanced only if batch*nbh*nbw == 1. Pop>push would deadlock compute today (not a finish() issue). Not exercised (in1 is DRAM in all in-scope tests).

### Benign/odd but balanced (noted for auditability)
- Compute MM_PARTIALS_RELOAD_ALIAS (mainline compute metal2 :121-127): alias DFB dfb::intermed0_reload_alias is read via evil_set_read_ptr + copy_block, never waited/popped/pushed -> its TC stays 0 -> finish OK.
- Compute PACKER_L1_ACC "bare" wait/pop drain of mm_partials (mainline :526-566, quasar metal2 :472-499, quasar legacy :457-490): counts traced — non-last blocks push (nb-1)*S, pops (nb-2)*S drain + S reload (no bias) / (nb-1)*S drain + S bias-stage (bias). Balanced.
- Compute bias: wait at (b==0&&bh==0)||nbw>1, pop per block if nbw>1 else ONE pop at kernel end (mainline :724-731, quasar metal2 :622-630, quasar legacy :626-634). Matches every in1 writer's push condition. Balanced.
- in0 block-sharded sender on cores without output work: push then self-pop (mainline :375-377, quasar :371-373); bound as self-loop (factory :3946-3959). Balanced.
- Quasar 1D noop/phantom cores (blank.cpp kernels, factory :7180-7218): DFBs bound but zero traffic. Balanced.
- DRAM-sharded in0 worker_core_type 1 (storage sender w/o compute) uses only get_write_ptr; no reserve/push; compute returns early via is_worker_core=0. Balanced (zero).
- minimal_matmul in0 reuse: reader skips push at k=0 after each N stride (dm_in0_sender_metal2.cpp ~:314-317, reuse_block=true per n-block, reset per m-block); compute skips pop of last K block when n < N_blocks-1 (compute_metal2.cpp:518-530). Per m-row both = N*K-(N-1). Balanced.
- minimal_matmul deferred output write: each block written once (deferred to next block's k==defer_write_k_block, clamped to K_blocks-1 in factory :943; last block written immediately). Balanced.
- minimal_matmul FUSE_TERNARY/FUSE_BIAS intermediate self-loop (compute_metal2.cpp:191-361): net 0 per block. Balanced.
- unified matmul: borrowed A/B pushed once, popped per K chunk; planner only borrows when batch==1 && one C slice/core (and A only when num_K_chunks==1; K_tiles % K_chunk_tiles == 0 enforced :241/:400). Balanced. Not reachable from in-scope tests (needs explicit MatmulUnifiedProgramConfig).
- sparsity DFB self-loop (in0 sender / in1 writer, SPARSITY only): reserve...push/wait/pop at end. Balanced; SPARSITY unreachable on Quasar (static_assert get_batch_from_reader).

## Coverage: kernels checked (B = balanced for Quasar-reachable configs, X = finding)
Mainline (M = ttnn/cpp/ttnn/operations/matmul/device/kernels):
- B  M/compute/bmm_metal2.cpp (+ reader_bmm_8bank_output_tiles_partitioned_metal2.cpp, writer_unary_interleaved_start_id.cpp non-sharded) — MultiCore: Kt*N in0/in1, N out
- X  M/dataflow/writer_unary_interleaved_start_id.cpp — F5 (OUT_SHARDED unreachable)
- B  M/dataflow/reader_bmm_tile_layout_in0.cpp — ReuseOptimized, batch RTA == compute batch CTA
- B  M/dataflow/reader_writer_bmm_tile_layout_in1.cpp — incl. bias full block once / pop once at end, OUT_SHARDED pops
- B* M/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp — internal balanced (partials, bias, transpose, untilize)
- B  M/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp — in0 per block, fake-batch reuse pushes, sparsity self-loop
- X  M/dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded_metal2.cpp — F4
- B  M/dataflow/reader_bmm_tile_layout_in0_receiver_metal2.cpp
- X  M/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp — F3 (count), F7 (IN1_SHARDED suspect); non-sharded out padding skips traced balanced
- X  M/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding_metal2.cpp — F3 (count); padding w/h skip traced: nzh*(nzw*sbt + w_skip) + h_skip == out_block tiles
- B  M/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp — types 1/2/3 traced
- X  M/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp — F6 (benign reserve-ahead); out wait/pop + bias once balanced
- B  M/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp
- B  M/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp
Quasar (Q = ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels):
- B  Q/compute/bmm.cpp + Q/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp
- X  Q/dataflow/writer_unary_interleaved_start_id.cpp — F5 (unreachable)
- B  Q/dataflow/reader_bmm_tile_layout_in0.cpp
- X  Q/dataflow/reader_writer_bmm_tile_layout_in1.cpp — F2
- B  Q/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp
- B  Q/compute/bmm_large_block_zm_fused_bias_activation.cpp (DRAM-sharded / HS descriptor paths)
- B  Q/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp (incl. TEN-4746 loopback block; push per block)
- B  Q/dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded_metal2.cpp (no in2 DFB)
- B  Q/dataflow/reader_bmm_tile_layout_in0_receiver_metal2.cpp
- X  Q/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp — F1 (+F3, F7)
- X  Q/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding_metal2.cpp — F1 (+F3)
- B  Q/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp
- X  Q/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp — F6 (benign)
- B  Q/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp, Q/..._in1_sender_dram_sharded_height.cpp
- B  Q/dataflow/blank.cpp, Q/compute/blank.cpp (noop cores, zero traffic)
- B  Q/dataflow/unified_matmul_reader.cpp, unified_matmul_writer.cpp, Q/compute/unified_matmul_compute.cpp (not reachable from tests)
minimal_matmul (ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels):
- B  dm_in0_sender_metal2.cpp, dm_in1_sender_out_metal2.cpp, compute_metal2.cpp, matmul_dataflow_common_metal2.hpp (no DFB ops of its own beyond those traced)
Not analyzed (not Quasar-reachable in scope): legacy non-metal2 mainline kernels used only by gather_in0 / all_gather_matmul_async (multi-device, tests skip), sparse_matmul (only gpt_oss_ops), minimal_matmul fabric_bound_* (legacy create, not selected), mainline PrefetcherPipe / GLOBAL_CB in1 paths.
