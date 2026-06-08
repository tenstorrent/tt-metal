# Kernel→Test Map: Normalization op-family (Phase 2, STATIC)

Static analysis only (grep/read). No device runs. Verification commands are
candidates to be confirmed by JIT build-cache later (Phase 4), NOT yet run.

## Factory routing summary (dispatch conditions)

### LayerNorm (`ttnn.layer_norm` / `ttnn.rms_norm` and distributed variants)
`LayerNormDeviceOperation::select_program_factory`
(`layernorm/device/layernorm_device_operation.cpp:17`):
- `input.is_sharded()` → `LayerNormShardedProgramFactory`
  (impl `layernorm/device/layernorm_op_multi_core_sharded.cpp`)
- else → `LayerNormMultiCoreProgramFactory` (interleaved; not in scope).

Inside the sharded factory, kernel paths chosen by
`KernelPaths::get(is_pre_all_gather, is_post_all_gather, use_row_major_kernel, use_welford)`
(`layernorm/device/sharded_layernorm_factory_helpers.cpp:452`), where the stage
comes from `operation_attributes.distributed_norm_stage`
(`layernorm_op_multi_core_sharded.cpp:67-68`):
- `PRE_ALL_GATHER`  → `reader_mcast_*_sharded_ln_pre_allgather.cpp`   (set by `ttnn.layer_norm_pre_all_gather` / `ttnn.rms_norm_pre_all_gather`)
- `POST_ALL_GATHER` → `reader_mcast_*_sharded_ln_post_allgather.cpp`  (set by `ttnn.layer_norm_post_all_gather` / `ttnn.rms_norm_post_all_gather`, with `stats` tensor)
- `NOT_DISTRIBUTED` → `reader_mcast_*_sharded_ln.cpp`                  (plain sharded `ttnn.layer_norm` / `ttnn.rms_norm`)

Distributed-stage entry points:
`layernorm_distributed/layernorm_pre_all_gather.cpp:37` sets `PRE_ALL_GATHER`;
`layernorm_distributed/layernorm_post_all_gather.cpp:40` sets `POST_ALL_GATHER`;
both require `input_tensor.is_sharded()` for the sharded kernels (else interleaved factory).
`use_welford` (sharded program-config flag, `layernorm_op_multi_core_sharded.cpp:90`)
only swaps the COMPUTE kernel (`layernorm_sharded_welford.cpp` vs `layernorm_sharded.cpp`)
for the NOT_DISTRIBUTED path — it does NOT change any of the six reader/writer kernels here.

Both reader_sender and reader_receiver are always created for the sharded multi-core grid
(`layernorm_op_multi_core_sharded.cpp:353-359`), so any multi-core sharded case builds both.

### GroupNorm (`ttnn.group_norm`)
`GroupNormDeviceOperation::select_program_factory`
(`groupnorm/device/groupnorm_device_operation.cpp:15`):
- `input.is_sharded()` → `GroupNormShardedProgramFactory`
  (`groupnorm/device/groupnorm_sharded_program_factory.cpp`) → the `*_gn_v2.cpp` kernels.
  - `use_mcast = (num_cores_per_batch > 1 || num_cores_per_group > 1)`
    (`groupnorm_sharded_program_factory.cpp:357`); receiver kernel only created when
    `use_mcast` (`:542`). Sender always created.
  - `use_welford = operation_attributes.use_welford` (`:47`, from the `use_welford`
    pybind arg, default False, `groupnorm/groupnorm_nanobind.cpp:138`):
    True → `welford_reader_mcast_*_sharded_gn_v2.cpp`;
    False → `reader_mcast_*_sharded_gn_v2.cpp`.
- non-sharded (interleaved): `batch >= num_virtual_rows` → `GroupNormNoMcastProgramFactory`
  (`groupnorm_no_mcast_program_factory.cpp`; sender only, no receiver);
  else → `GroupNormMcastProgramFactory` (`groupnorm_mcast_program_factory.cpp`; sender + receiver).
  These use the non-v2 `*_unary_gn.cpp` kernels, welford-swapped the same way
  (`groupnorm_mcast_program_factory.cpp:445-454`, `groupnorm_no_mcast_program_factory.cpp:572-574`).

---

## Per-kernel map

### ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln_post_allgather.cpp
- role / tag: sender — C1 (pure-flag, no-pre-handshake, fresh-slot) — clean
- factory: `LayerNormShardedProgramFactory` via `KernelPaths::get` `layernorm/device/sharded_layernorm_factory_helpers.cpp:467` (selected when: input sharded AND `distributed_norm_stage == POST_ALL_GATHER`)
- op: `ttnn.layer_norm_post_all_gather` / `ttnn.rms_norm_post_all_gather` (sharded input, with `stats`)
- candidate_validation_set: `test_distributed_layernorm_sharded.py::test_post_allgather_layernorm` [num_devices=4, input_df=bfloat16, core_grid=(8,2), is_rmsnorm=False]
- candidate_regression_set: full `test_post_allgather_layernorm` cross-product (num_devices 4/8 × bf8/bf16 × is_rmsnorm T/F); plus `test_simulated_distributed_layernorm` (num_devices=1, runs pre+post)
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py -k "post_allgather and bfloat16 and not rmsnorm"` then confirm `grep -rl reader_mcast_sender_unary_sharded_ln_post_allgather generated/ | grep -i kernel`
- coverage_confidence: med
- gaps: test simulates multi-device by chunking on one chip — single-device run is valid. `num_devices=4` is the smallest parametrized; confirm it fits the available grid.

### ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp
- role / tag: receiver — C1 — clean
- factory: `LayerNormShardedProgramFactory` via `KernelPaths::get` `sharded_layernorm_factory_helpers.cpp:469` (selected when: sharded AND `POST_ALL_GATHER`)
- op: `ttnn.layer_norm_post_all_gather` / `ttnn.rms_norm_post_all_gather`
- candidate_validation_set: same as sender — `test_post_allgather_layernorm` [num_devices=4, bfloat16, core_grid=(8,2), is_rmsnorm=False] (receiver builds on the multi-core grid)
- candidate_regression_set: full `test_post_allgather_layernorm`
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py -k "post_allgather and bfloat16 and not rmsnorm"` then `grep -rl reader_mcast_receiver_unary_sharded_ln_post_allgather generated/`
- coverage_confidence: med
- gaps: receiver only present when grid is multi-core in the reduce group; (8,2) grid satisfies this.

### ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_gn_v2.cpp
- role / tag: receiver — C2 — clean
- factory: `GroupNormShardedProgramFactory` `groupnorm_sharded_program_factory.cpp:548` (selected when: sharded input, `use_mcast` true (`:542`), `use_welford == False`)
- op: `ttnn.group_norm` (sharded, block-sharded multi-core, use_welford=False)
- candidate_validation_set: `test_group_norm.py::test_group_norm_with_block_sharded_v2_8x4_grid` [shape (1,1280,16,16,32), welford_mode="legacy"]
- candidate_regression_set: `test_group_norm_with_block_sharded_v2_8x4_grid` and `_8x8_grid` with `legacy` id (all shapes)
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/fused/test_group_norm.py -k "block_sharded_v2_8x4 and legacy"` then `grep -rl reader_mcast_receiver_unary_sharded_gn_v2 generated/`
- coverage_confidence: high
- gaps: height-sharded case (1,320,32,32,16) may be single-core-per-group → no receiver; use the 8x4 block-sharded case to guarantee mcast.

### ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_gn_v2.cpp
- role / tag: sender — C2 (multi-rect; raw-L1 src) — refactor
- factory: `GroupNormShardedProgramFactory` `groupnorm_sharded_program_factory.cpp:529` (selected when: sharded input, `use_welford == False`; sender always created)
- op: `ttnn.group_norm` (sharded, use_welford=False)
- candidate_validation_set: `test_group_norm.py::test_group_norm_with_block_sharded_v2_8x4_grid` [shape (1,1280,16,16,32), welford_mode="legacy"]
- candidate_regression_set: 8x4 + 8x8 block-sharded `legacy` + `test_group_norm_with_height_sharded` [legacy]
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/fused/test_group_norm.py -k "block_sharded_v2_8x4 and legacy"` then `grep -rl reader_mcast_sender_unary_sharded_gn_v2 generated/`
- coverage_confidence: high
- gaps: none material — sender builds on any sharded legacy GN.

### ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp
- role / tag: sender — C2 (flag-only; atomic-barrier) — refactor
- factory: `LayerNormShardedProgramFactory` via `KernelPaths::get` `sharded_layernorm_factory_helpers.cpp:460` (selected when: sharded AND `PRE_ALL_GATHER`)
- op: `ttnn.layer_norm_pre_all_gather` / `ttnn.rms_norm_pre_all_gather` (sharded input)
- candidate_validation_set: `test_distributed_layernorm_sharded.py::test_pre_allgather_layernorm` [num_devices=4, bfloat16, core_grid=(8,4), is_rmsnorm=False]
- candidate_regression_set: full `test_pre_allgather_layernorm` (+ `test_pre_allgather_layernorm_1d_reduce`, core_grid (1,4)) + `test_simulated_distributed_layernorm`
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py -k "pre_allgather_layernorm and bfloat16 and not rmsnorm and not 1d"` then `grep -rl reader_mcast_sender_unary_sharded_ln_pre_allgather generated/`
- coverage_confidence: med
- gaps: multi-device simulated on one chip; confirm (8,4) grid available.

### ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp
- role / tag: receiver — C2 (atomic-barrier) — refactor
- factory: `LayerNormShardedProgramFactory` via `KernelPaths::get` `sharded_layernorm_factory_helpers.cpp:462` (selected when: sharded AND `PRE_ALL_GATHER`)
- op: `ttnn.layer_norm_pre_all_gather` / `ttnn.rms_norm_pre_all_gather`
- candidate_validation_set: same as sender — `test_pre_allgather_layernorm` [num_devices=4, bfloat16, core_grid=(8,4), is_rmsnorm=False]
- candidate_regression_set: full `test_pre_allgather_layernorm` + `_1d_reduce`
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py -k "pre_allgather_layernorm and bfloat16 and not rmsnorm and not 1d"` then `grep -rl reader_mcast_receiver_unary_sharded_ln_pre_allgather generated/`
- coverage_confidence: med
- gaps: receiver needs multi-core reduce group; (8,4) ok.

### ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_sender_unary_gn.cpp
- role / tag: sender — C2 (multi-rect) — refactor — INTERLEAVED (non-v2)
- factory: `GroupNormMcastProgramFactory` `groupnorm_mcast_program_factory.cpp:447` OR `GroupNormNoMcastProgramFactory` `groupnorm_no_mcast_program_factory.cpp:572` (selected when: NON-sharded input AND `use_welford == True`; Mcast vs NoMcast by `batch >= num_virtual_rows`)
- op: `ttnn.group_norm` with INTERLEAVED (non-sharded) input + use_welford=True
- candidate_validation_set: NONE in unit tests — all `test_group_norm.py` cases shard the input before the op.
- candidate_regression_set: sweep `tests/ttnn/python_api_testing/sweep_tests/ttnn_ops.py:2281` (interleaved group_norm, num_groups=1) — but does not set use_welford=True.
- verification_command: (no direct unit test) — would need a custom probe passing interleaved input + use_welford=True; e.g. `scripts/tt-probe.sh group_norm` constructing a non-sharded DRAM input and calling `ttnn.group_norm(..., use_welford=True)`, then `grep -rl welford_reader_mcast_sender_unary_gn generated/`
- coverage_confidence: low
- gaps: GAP — no unit test exercises the interleaved (non-v2) GroupNorm path; needs a new probe/test.

### ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_sender_unary_sharded_gn_v2.cpp
- role / tag: sender — C2 (multi-rect) — refactor
- factory: `GroupNormShardedProgramFactory` `groupnorm_sharded_program_factory.cpp:527` (selected when: sharded input AND `use_welford == True`; sender always created)
- op: `ttnn.group_norm` (sharded, use_welford=True)
- candidate_validation_set: `test_group_norm.py::test_group_norm_with_block_sharded_v2_8x4_grid` [shape (1,1280,16,16,32), welford_mode="welford"]
- candidate_regression_set: 8x4 + 8x8 block-sharded `welford` + height-sharded `welford`
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/fused/test_group_norm.py -k "block_sharded_v2_8x4 and welford"` then `grep -rl welford_reader_mcast_sender_unary_sharded_gn_v2 generated/`
- coverage_confidence: high
- gaps: none material.

### ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_receiver_unary_gn.cpp
- role / tag: receiver — C2 — refactor — INTERLEAVED (non-v2)
- factory: `GroupNormMcastProgramFactory` `groupnorm_mcast_program_factory.cpp:452` (selected when: NON-sharded input, `batch < num_virtual_rows` (mcast), `use_welford == True`)
- op: `ttnn.group_norm` INTERLEAVED input + use_welford=True (mcast case only)
- candidate_validation_set: NONE in unit tests.
- candidate_regression_set: none.
- verification_command: (no direct test) custom probe with interleaved input small batch + use_welford=True; `grep -rl welford_reader_mcast_receiver_unary_gn generated/`
- coverage_confidence: low
- gaps: GAP — interleaved mcast welford GN path untested; receiver additionally needs `batch < num_virtual_rows`.

### ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp
- role / tag: receiver — C2 — refactor
- factory: `GroupNormShardedProgramFactory` `groupnorm_sharded_program_factory.cpp:546` (selected when: sharded input, `use_mcast` true (`:542`), `use_welford == True`)
- op: `ttnn.group_norm` (sharded, multi-core mcast, use_welford=True)
- candidate_validation_set: `test_group_norm.py::test_group_norm_with_block_sharded_v2_8x4_grid` [shape (1,1280,16,16,32), welford_mode="welford"]
- candidate_regression_set: 8x4 + 8x8 block-sharded `welford`
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/fused/test_group_norm.py -k "block_sharded_v2_8x4 and welford"` then `grep -rl welford_reader_mcast_receiver_unary_sharded_gn_v2 generated/`
- coverage_confidence: high
- gaps: receiver needs use_mcast; 8x4 block-sharded guarantees it (height-sharded may not).

### ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln.cpp
- role / tag: sender — C3 (TWO-PHASE flag→monotone-counter streaming) — refactor
- factory: `LayerNormShardedProgramFactory` via `KernelPaths::get` `sharded_layernorm_factory_helpers.cpp:474` (selected when: sharded AND `NOT_DISTRIBUTED`)
- op: `ttnn.layer_norm` / `ttnn.rms_norm` (plain sharded; not pre/post-allgather)
- candidate_validation_set: `test_layer_norm_sharded.py::test_layer_norm_sharded_single_stage` [smallest single_stage_param_set h=64 w=64, dtype=bfloat16, tensor_type="random", use_welford=False]
- candidate_regression_set: `test_layer_norm_sharded_single_stage` + `_two_stage` + `_with_residual` + `_with_weight_and_bias` (use_welford T/F; dtype bf16/f32)
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/fused/test_layer_norm_sharded.py -k "single_stage and random and not welford and bfloat16"` then `grep -rl reader_mcast_sender_unary_sharded_ln.cpp generated/` (use full basename — substring collides with the allgather variants)
- coverage_confidence: high
- gaps: grep must use exact basename so it does not match `_post_allgather`/`_pre_allgather`. `use_welford` only changes the compute kernel, not this reader.

### ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln.cpp
- role / tag: receiver — C3 — refactor
- factory: `LayerNormShardedProgramFactory` via `KernelPaths::get` `sharded_layernorm_factory_helpers.cpp:475` (selected when: sharded AND `NOT_DISTRIBUTED`)
- op: `ttnn.layer_norm` / `ttnn.rms_norm` (plain sharded)
- candidate_validation_set: `test_layer_norm_sharded.py::test_layer_norm_sharded_single_stage` [smallest single_stage param, bfloat16, random, use_welford=False]
- candidate_regression_set: `test_layer_norm_sharded_single_stage` + `_two_stage`
- verification_command: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/fused/test_layer_norm_sharded.py -k "single_stage and random and not welford and bfloat16"` then `grep -rl reader_mcast_receiver_unary_sharded_ln.cpp generated/`
- coverage_confidence: high
- gaps: same basename-collision caveat as the sender.

### ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_sender_unary_gn.cpp
- role / tag: sender — C4 (double-flag-per-exchange) — refactor — INTERLEAVED (non-v2)
- factory: `GroupNormMcastProgramFactory` `groupnorm_mcast_program_factory.cpp:449` OR `GroupNormNoMcastProgramFactory` `groupnorm_no_mcast_program_factory.cpp:574` (selected when: NON-sharded input AND `use_welford == False`; Mcast vs NoMcast by `batch >= num_virtual_rows`)
- op: `ttnn.group_norm` with INTERLEAVED (non-sharded) input, use_welford=False
- candidate_validation_set: NONE in unit tests (all unit tests shard input first).
- candidate_regression_set: sweep `tests/ttnn/python_api_testing/sweep_tests/ttnn_ops.py:2281` (interleaved, num_groups=1, no welford).
- verification_command: (no direct unit test) custom `scripts/tt-probe.sh group_norm` with interleaved DRAM input + `ttnn.group_norm(...)` (use_welford default False), then `grep -rl reader_mcast_sender_unary_gn.cpp generated/`
- coverage_confidence: low
- gaps: GAP — interleaved GroupNorm path has no unit-test coverage; sweep is the only known caller.

### ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_mcast_receiver_unary_gn.cpp
- role / tag: receiver — C4 — refactor — INTERLEAVED (non-v2)
- factory: `GroupNormMcastProgramFactory` `groupnorm_mcast_program_factory.cpp:454` (selected when: NON-sharded input, `batch < num_virtual_rows` (mcast), `use_welford == False`)
- op: `ttnn.group_norm` INTERLEAVED input, use_welford=False, mcast case
- candidate_validation_set: NONE.
- candidate_regression_set: none known.
- verification_command: (no direct test) custom probe, interleaved input with small batch (batch < num_virtual_rows), `grep -rl reader_mcast_receiver_unary_gn.cpp generated/`
- coverage_confidence: low
- gaps: GAP — needs interleaved mcast case (`batch < num_virtual_rows`); no test constructs it.

---

## Coverage gaps (rollup)
- The four INTERLEAVED (non-v2) GroupNorm kernels — `reader_mcast_sender_unary_gn`,
  `reader_mcast_receiver_unary_gn`, `welford_reader_mcast_sender_unary_gn`,
  `welford_reader_mcast_receiver_unary_gn` — have NO unit-test coverage. Every
  `test_group_norm*.py` test shards the input before calling the op, which routes to
  the `GroupNormShardedProgramFactory` (gn_v2 kernels). The interleaved factories
  (`GroupNormMcastProgramFactory` / `GroupNormNoMcastProgramFactory`) are reachable only
  via the sweep harness (`ttnn_ops.py`) or a custom probe. Before migrating these four,
  a probe/test passing interleaved input (and use_welford True/False, plus a small batch
  to force `batch < num_virtual_rows` for the receivers) must be added.
- Distributed LN kernels (pre/post-allgather, C1+C2) are tested only via single-chip
  multi-device simulation (chunk-on-one-device); valid for compile+build verification.
