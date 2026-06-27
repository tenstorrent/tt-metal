# Conv op-family kernel→test map (Phase 2, STATIC analysis only — no device runs)

Repo root: `/localdev/sjovic/tt-metal`

## Dispatch overview (conv2d sharded factory)

File: `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp`
(`multi_core_optimized_conv_sharded_v2_impl`)

- Strategy flags (lines 247-248): `block_sharded = memory_layout == BLOCK_SHARDED`, `height_sharded = memory_layout == HEIGHT_SHARDED`.
- Kernel string defaults (lines 770-777): the **1d** weights-mcast sender/receiver pair is the default.
- Override (lines 779-799):
  - `if (!is_conv_1d_depthwise_conv && block_sharded)` → swap reader to the 2d halo reader and the weights-mcast pair to the **2d** sender/receiver.
  - `else if (is_conv_1d_depthwise_conv)` → depthwise kernels (not in scope).
  - `else` (height-sharded, non-depthwise) → keeps the **1d** weights-mcast pair + height-sharded halo reader.
- CreateKernel sites: sender at line 1114 on `mcast_sender_cores`; receiver at line 1126 on `mcast_receiver_cores` (only if `!skip_weights_mcast`); reader at line 1137. All weights-mcast kernels run on RISCV_0 (writer/BRISC). Reader runs on RISCV_1 (NCRISC).

The width-sharded path is a SEPARATE factory selected one level up:
`conv2d_device_operation.cpp:35` — `if (memory_layout == WIDTH_SHARDED)` → `conv2d_op_width_sharded_program_factory.cpp`, which always uses `activation_reader_width_sharded.cpp` (line 322-323).

Op entry: `ttnn.conv2d` for all conv2d kernels; `ttnn.experimental.conv3d` for the conv3d writer.

Test shard-layout aliases (defined in `tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py` and re-imported by the unit file):
`HS = HEIGHT_SHARDED`, `BS = BLOCK_SHARDED`, `WS = WIDTH_SHARDED`.
The unit `tests/ttnn/unit_tests/operations/conv/test_conv2d.py` imports `run_conv, HS, WS, BS` from the nightly file and defines small fast cases:
`test_conv_features` / `test_conv_dram_config` shape table (lines 16-18, 102-104):
- `(353,384,8,8, WS, None)` — width-sharded, 8x8
- `(128,128,32,32, BS, None)` — block-sharded, 32x32
- `(16,16,256,256, HS, {"act_block_h":32})` — height-sharded

NOTE on `-k`: pytest `-k` cannot use `name=value`. Filter on a value substring (e.g. a shape/dtype id) only.
"Built?" check: after a run, grep the kernel basename in the JIT artifact tree:
`ls generated/jit_cache 2>/dev/null; grep -rl "<basename>" generated/ 2>/dev/null` (kernel .cpp gets copied/compiled per-program under generated/).

---

### ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp
- role / tag: weights-mcast SENDER (1d) / clean, sender. Canonical clean two-sided pair.
- factory: `multi_core_optimized_conv_sharded_v2_impl` in `conv2d_op_sharded_program_factory.cpp:1114` (string set at :772-774).
  Selected when: NOT block_sharded AND NOT depthwise (i.e. **height-sharded** conv), AND `!skip_weights_mcast`. This is the default string; the `if (block_sharded)` branch at :779 does NOT fire. CreateKernel'd on `mcast_sender_cores`.
- op: `ttnn.conv2d`
- candidate_validation_set: `test_conv_features` param `(16,16,256,256, HS, {"act_block_h":32})` — smallest height-sharded case. (file: unit `test_conv2d.py:18`)
- candidate_regression_set: nightly `test_conv2d.py` height-sharded rows across `test_conv_features`/`test_conv_dram_config`; resnet50/unet HS cases (`test_resnet50_conv_wh`, `test_unet_conv_wh`), `test_halo_reshard_conv[HS]`, `test_conv_core_nondivis[HS]`.
- verification_command:
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/conv/test_conv2d.py -k "256 and bfloat16"`
  Confirm built: `grep -rl reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks generated/`
- coverage_confidence: high
- gaps: sender always exists on the sender core; receiver only if `!skip_weights_mcast`. A single-core / skip-mcast HS config would build the sender but exercise no actual mcast — pick a multi-core HS shape (256x256 spans many cores) to guarantee real send.

### ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp
- role / tag: weights-mcast RECEIVER (1d) / clean, receiver.
- factory: `conv2d_op_sharded_program_factory.cpp:1126` (string at :775-777). Selected when: same height-sharded condition as the 1d sender AND `!skip_weights_mcast` (the `if (!skip_weights_mcast)` guard at :1125). On `mcast_receiver_cores = all_cores.subtract(mcast_sender_cores)` (:754).
- op: `ttnn.conv2d`
- candidate_validation_set: same as 1d sender — `test_conv_features` `(16,16,256,256, HS, {"act_block_h":32})`. Multi-core HS guarantees receiver cores exist.
- candidate_regression_set: same HS regression set as the 1d sender.
- verification_command:
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/conv/test_conv2d.py -k "256 and bfloat16"`
  Confirm built: `grep -rl reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks generated/`
- coverage_confidence: high
- gaps: only instantiated when `skip_weights_mcast` is false; a degenerate single-mcast-core HS shape would skip it. The 256x256 HS shape is multi-core so receiver is present.

### ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp
- role / tag: weights-mcast SENDER (2d) / clean, sender.
- factory: `conv2d_op_sharded_program_factory.cpp:1114` (string set in the block-sharded branch at :784-786). Selected when: `!is_conv_1d_depthwise_conv && block_sharded` (line :779). Sender cores are the top row/col of the block-shard grid (:725-733, transpose_mcast dependent).
- op: `ttnn.conv2d`
- candidate_validation_set: `test_conv_features` param `(128,128,32,32, BS, None)` — smallest block-sharded case (unit `test_conv2d.py:17`).
- candidate_regression_set: nightly block-sharded rows; `test_conv_for_segformer_512x512` (BLOCK_SHARDED, :1049), `test_halo_reshard_conv[BS]`, `test_conv_core_nondivis[BS]`, SD/unet BS cases.
- verification_command:
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/conv/test_conv2d.py -k "32 and bfloat16"`  (32x32 BS row; widen filter if it collides with WS 8x8)
  Confirm built: `grep -rl writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks generated/`
- coverage_confidence: high
- gaps: `-k "32"` may also match the 256 HS / other "32"-bearing ids; if ambiguous use a BS-unique substring (e.g. run the full small table and confirm via the generated/ grep that the 2d sender basename appears). transpose_mcast (COL_MAJOR shard) picks a different sender-core geometry but same kernel — both covered by any BS shape.

### ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp
- role / tag: weights-mcast RECEIVER (2d) / clean, receiver.
- factory: `conv2d_op_sharded_program_factory.cpp:1126` (string at :787-789), under the same `block_sharded` branch, guarded by `!skip_weights_mcast` (:1125). On `mcast_receiver_cores`.
- op: `ttnn.conv2d`
- candidate_validation_set: same as 2d sender — `(128,128,32,32, BS, None)`. 32x32 block-sharded spans a >1-core grid so receivers exist.
- candidate_regression_set: same BS regression set as the 2d sender.
- verification_command:
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/conv/test_conv2d.py -k "32 and bfloat16"`
  Confirm built: `grep -rl writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks generated/`
- coverage_confidence: high
- gaps: same `skip_weights_mcast` caveat as the 1d receiver; need a genuinely multi-core block-sharded grid (32x32 BS is fine).

### ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp
- role / tag: activation reader, WIDTH-sharded / refactor, hybrid. INCLUDE_SRC loopback + `noc.async_write_barrier()` fence; mixed R->S atomic-increment counter (sender sem) / S->R `set_multicast` VALID flag (receiver sem). Lines: counter inc :267, `wait(VALID)` :271, INCLUDE_SRC mcast + barrier :227/:253.
- factory: `multi_core_optimized_conv_width_sharded_v2_impl` in `conv2d_op_width_sharded_program_factory.cpp:323` (CreateKernel near :347+). Selected when: input `memory_layout == WIDTH_SHARDED` (dispatched at `conv2d_device_operation.cpp:35`). Always this kernel in the width-sharded factory (no sub-branch).
- op: `ttnn.conv2d`
- candidate_validation_set:
  1. `test_conv_features` param `(353,384,8,8, WS, None)` — smallest WS case in the shared small table (unit `test_conv2d.py:16`).
  2. `test_conv_ws` smallest row `(2,128,256,9,9,3,3,1,1,1)` (nightly `test_conv2d.py:871`) — dedicated WS test, but needs 8x8 grid (skips on 7-row WH / N300).
- candidate_regression_set: full `test_conv_ws` shape table (nightly :870-883) across `auto_shard`/`tilized_input`/`enable_act_double_buffer`/`enable_weights_double_buffer`; the WS rows of `test_conv_features`/`test_conv_dram_config`.
- verification_command:
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/conv/test_conv2d.py -k "8 and bfloat16"`  (the 353,384,8,8 WS row; cross-check generated/ grep since "8" is broad)
  Confirm built: `grep -rl activation_reader_width_sharded generated/`
- coverage_confidence: high
- gaps: hybrid sync (counter + flag + loopback barrier) only fully exercised when the width-shard spans multiple cores so a real mcast fires; the 8x8 / 384-channel WS shape is multi-core. `test_conv_ws` is grid-gated (skips unless core_grid.y==8) — on a 7-row machine fall back to the `(353,384,8,8,WS)` row in `test_conv_features` which has no grid guard.

### ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp
- role / tag: activation halo reader, BLOCK-sharded (2d) / refactor, hybrid. All 3 F3 sub-cases via `mcast_block_chunked<...>` (defined :66, called :281): CHUNKED send when block > `NOC_MAX_BURST_SIZE` (R4 streaming, DEFERRED part). Sender/receiver/non-sender column-mcast roles (:271 sender, INCLUDE_SRC VALID flag :296, receiver path :305+).
- factory: `conv2d_op_sharded_program_factory.cpp:1137` (string at :781-783), in the `!is_conv_1d_depthwise_conv && block_sharded` branch (:779). On `all_cores` (block-sharded uses all cores; `height_sharded ? input_cores : all_cores` at :1140).
- op: `ttnn.conv2d`
- candidate_validation_set: `test_conv_features` param `(128,128,32,32, BS, None)` with `filter=3, padding=(1,2,2,3)` (the 3x3 path) — unit `test_conv2d.py:17`. This is the same case that builds the 2d weights-mcast pair, so one BS run covers reader + both weight kernels.
- candidate_regression_set: `test_conv_for_segformer_512x512` (3x3 BS, :1049); `test_halo_reshard_conv[BS]`, `test_conv_core_nondivis[BS]`; SD/unet BS 3x3 cases. To force the CHUNKED (>NOC_MAX_BURST_SIZE) sub-case, use a BS shape with a large `act_block` (high channel count / large act_block_h), e.g. segformer 640-channel BS row.
- verification_command:
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/conv/test_conv2d.py -k "32 and bfloat16"`  (BS 32x32, filter 3)
  Confirm built: `grep -rl reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2 generated/`
- coverage_confidence: med
- gaps: the small 32x32/128-ch BS case likely exercises only the single-shot (non-chunked) mcast path — block fits under `NOC_MAX_BURST_SIZE`. The CHUNKED R4-streaming/DEFERRED sub-case needs a large per-core act block (verify by an act-block-size estimate vs NOC_MAX_BURST_SIZE before claiming coverage). Despite the `3x3` name the kernel is used for the general block-sharded reader; confirm 3x3 vs other filters do not pick a different reader (they do not — factory only branches on shard layout/depthwise).

### ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/writer.cpp
- role / tag: conv3d writer / refactor, hybrid. 3 weight-share modes (compile arg 22 `WeightShareMode`: Disabled / Chain / Mcast) + `enable_streaming_output` (compile arg 25). Per-role runtime branch on `WeightShareRole` (Local / ChainInjector/Middle/Tail / McastSender/Receiver/Passive). Ack-only receive on McastReceiver/Passive (:139, :247).
- factory: `conv3d_program_factory.cpp:747` (CreateKernel; string :749). Mode chosen at :514-525:
  - `WeightShareMode::Disabled` when `group_size == 1` (parallelism gives one core per weight group).
  - `WeightShareMode::Mcast` when `group_size > 1` AND the row-strip rectangles fit the grid (`num_groups * rows_per_group <= grid.y`, :518-520).
  - `WeightShareMode::Chain` when `group_size > 1` but strips don't fit (:522-523).
  `enable_streaming_output` (:127): `C_in_num_blocks == 1 && matmul_M_t > 1 && small_output_write_transactions`.
- op: `ttnn.experimental.conv3d` (`ttnn::experimental::conv3d::conv3d`)
- candidate_validation_set:
  1. `test_conv3d_no_config` `[(1,32,4,8,8),32,(3,3,3),(2,2,2),1,(0,1,1),"zeros"]` (unit `test_conv3d.py:400`) — smallest conv3d; default parallelism → likely `Disabled` (Local-role write path).
  2. `test_conv3d_sweep_shapes` smallest combo `B=1,C_in=12,C_out=64,T=8,H=10,W=9, kernel_111, stride_111, groups_1` (:189-198) — sweeps groups 1/2/4 which changes group_size → exercises Chain/Mcast modes across the matrix.
- candidate_regression_set: full `test_conv3d_sweep_shapes` matrix (covers all 3 modes via groups × parallelism); `test_conv3d_qwen_shapes` (C_in blocking, skip_for_blackhole); nightly `tests/ttnn/nightly/unit_tests/operations/conv/test_conv3d.py`.
- verification_command:
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/conv/test_conv3d.py -k "kernel_333 and groups_4"`  (groups_4 maximizes chance of Mcast/Chain; for Disabled use `-k "no_config"`)
  Confirm built: `grep -rl "conv3d/device/kernels/writer.cpp\|experimental.*writer" generated/` — basename `writer.cpp` is generic, so grep the full path or a unique string from the kernel (e.g. `WeightShareRole::McastSender`) in the generated copy.
- coverage_confidence: med
- gaps: which of the 3 modes a given test hits is NOT directly a test param — it is derived from `group_size`/`num_groups` (parallelism factors) inside the factory (:512-525). Cannot statically prove Mcast vs Chain selection per param without the parallelism-factor computation; needs the later device phase (read the factory `log_debug "Weight share: mode=..."` at :526) to confirm each mode is covered. `enable_streaming_output` likewise depends on `matmul_M_t`/transaction-size heuristics, not a test knob. `writer.cpp` basename is non-unique (collides with other ops' writer.cpp) — must grep by full path.
