# AutoDebug Report: multichip decoder timeout/read hang

## Verdict

The stack captured at about 13 minutes is a **downstream diagnostic hang**, not evidence that the model or CCL originally deadlocked. The 300-second signal timeout raises a pytest failure; pytest's default long traceback then formats function arguments. Formatting a live TTNN tensor calls its device-reading `__repr__`, which can wait indefinitely on the command queue it interrupted. The smallest command-line mitigation is `--tb=short` (or `--tb=no`): short tracebacks explicitly skip function-argument repr. `--no-showlocals` alone is insufficient because pytest formats function arguments separately.

There are nevertheless real, earlier failures and one additional likely harness race:

1. BFP8 batch-32 prefill exceeds L1 circular-buffer capacity before multichip execution.
2. The older BFP4 path passed a decode-shaped width-sharded memory config to a prefill all-reduce.
3. After those issues were bypassed/fixed, a run reached the first baseline-versus-multichip host read and waited with unsynchronized work outstanding on a 1x1 submesh and its 1x2 parent, which alias physical device 0. This is a likely command-queue ordering bug in the comparison harness, but requires a rerun to prove.

There is **no positive evidence of a fabric/CCL deadlock** in the original capture. There is positive evidence of a CCL *validation/configuration* bug in the later BFP4 run.

This investigation was inspection-only; no TT hardware was used.

## Direct observations

- `/tmp/llama_multichip_bfp8.log` selected one test with pytest-timeout 2.4.0 configured for `300.0s`, signal mode. It stops during first-use kernel compilation and has no completed pytest failure footer.
- Supplied GDB evidence at about 13 minutes put the main thread in `ttnn::to_string -> Tensor::cpu -> enqueue_read -> FDMeshCommandQueue::finish_nolock`, reached via Python builtin `repr`. A runtime reader thread was in `completion_queue_wait_front`.
- `pytest_timeout.timeout_sigalrm` dumps thread stacks and calls `pytest.fail`; it does not itself repr test values. The repr occurs later while pytest constructs the failure report.
- `_pytest._code.FormattedExcinfo.repr_args()` calls `saferepr()` on every visible frame argument when `funcargs=True`; pytest's normal node failure path requests `funcargs=True`. For traceback style `short`, pytest explicitly sets `reprargs=None`.
- `ttnn/cpp/ttnn-nanobind/pytensor.cpp` binds `Tensor.__repr__` to `ttnn::to_string`. For an allocated single-device tensor, `ttnn/core/tensor/to_string.cpp` calls `tensor.cpu()`, and `ttnn/core/tensor/tensor_ops.cpp` enqueues a blocking host read.
- Triage showed ARC heartbeats near 10/s on both devices, healthy DDR status, and normal ARC uptime. It did not show a device crash.
- Many low-level `tt-triage` reads, including running-op detail, callstacks, and lightweight asserts, were unavailable because the installed `tt_umd.noc_read` binding accepts `bytearray` but the triage caller passed `memoryview`. The summary's `pass` labels therefore overstate the completeness of those checks.

## Finding 1: failure rendering explains the captured repr/CQ wait

The observed chain is complete:

1. The 300-second signal alarm interrupts a live Python/TTNN stack and raises a pytest failure.
2. Default/long pytest traceback construction formats frame arguments even though `--showlocals` is false.
3. At least one argument is a live TTNN tensor.
4. TTNN tensor repr reads a single-device tensor to host.
5. That read enters `FDMeshCommandQueue::finish_nolock`, matching GDB exactly.

This proves that the process was blocked in diagnostic formatting when captured. It does not prove why the original work had not completed by 300 seconds, nor that every observed read wait is repr-driven.

Smallest harness mitigations, in priority order:

- For diagnosis, add `--tb=short` to the pytest command. It preserves a useful traceback while preventing pytest's per-frame argument repr. This is the change that exposed the fast underlying failures. `--tb=line` or `--tb=no` are also safe but less informative.
- Keep the current test-local `pytest.mark.timeout(1800)` if first-use full-shape packing/JIT legitimately exceeds the repo-wide 300 seconds. This avoids a premature signal, but it only delays failure formatting and does not protect an immediate model exception; it is therefore complementary to `--tb=short`, not a substitute.
- A broader TTNN change making device-tensor repr metadata-only would eliminate this class globally, but is not the smallest harness fix and is outside this report's scope.

## Finding 2: the baseline/parent mesh comparison has a likely ordering race

The BFP4 r3 run, after the DRAM-prefill-CCL and tuple-weight fixes, progressed to the first baseline-versus-multichip prefill comparison. Supplied GDB evidence then showed a normal explicit device-read wait in the same FD mesh CQ/read machinery, rather than a Python `repr` path.

The pre-patch harness enqueued `baseline.prefill_forward` on `single_mesh` and immediately enqueued `multichip.prefill_forward` on the parent `mesh_device`. The 1x1 baseline submesh aliases device 0 of the parent 1x2 mesh, so both mesh views could have CQ 0 commands outstanding on the same physical device. The repeated teardown message, `MeshDevice cq ID 0 is in use by parent mesh ID 0 during close of mesh ID 1`, is downstream of earlier exceptions in the short logs but reinforces this ownership/lifetime hazard.

The current harness contains the smallest ordering fix:

- synchronize the 1x1 submesh;
- materialize the baseline output (and, for decode, cache slices) on the host;
- only then submit parent-mesh work;
- synchronize the parent before reading/comparing its result.

This is a likely independent contributor to read waits, not a proven root cause until the patched harness completes on hardware. It also means the original 13-minute observation should not be reduced to “pytest repr was the only bug”: repr explains the captured downstream stack, while aliased-mesh overlap is a plausible upstream harness defect.

## Finding 3: BFP8 has a real prefill resource failure before CCL

`/tmp/llama_multichip_contract.log` fails in the single-chip `OptimizedDecoder.prefill_forward`, before `MultiChipDecoder.prefill_forward`:

- path: `_mlp_prefill -> _prefill_linear -> ttnn.linear`;
- error: statically allocated circular buffers require `1676032 B`, exceeding the `1572864 B` L1 limit.

This is a real matmul/program-config/dtype capacity failure. It is neither failure-rendering noise nor a multichip/fabric failure. Moving the correctness fixture to BFP4 avoids this specific BFP8 path but does not establish BFP8 batch-32 support.

## Finding 4: BFP4 exposed a real prefill CCL memory-config bug

`/tmp/llama_multichip_contract_bfp4.log` gets through the baseline and fails at the first multichip prefill `all_reduce_async`:

- error: `Shard height 32 must match physical height 1024 for width sharded`;
- lowered path: `all_reduce_async -> reduce_scatter_minimal_async -> TensorSpec`.

For logical prefill shape `[1, 32, 7, 4096]`, TILE padding makes the physical flattened height `1 * 32 * 32 = 1024`. The default residual config was created for decode and has width-sharded height 32, so it cannot describe this prefill result.

The current code applies the narrow intended split:

- both prefill attention and MLP all-reduces request `ttnn.DRAM_MEMORY_CONFIG`;
- decode all-reduces retain the width-sharded residual config.

The r3 run progressing to the output comparison is evidence that the old prefill shard-height validation failure was passed after this change. It is not yet evidence of end-to-end correctness because that run then waited at the cross-mesh host-read boundary.

## Remaining uncertainty and next check

- Run the patched BFP4 test with `--tb=short` and the test-local long timeout. A pass through prefill comparison, decode, cache checks, and trace replay is needed to clear both the CCL split and aliased-mesh ordering fix.
- Validate decode separately: static inspection shows the width-sharded policy is shape-appropriate for batch-32 decode, but does not prove fabric completion or trace replay correctness.
- Treat BFP8 batch-32 prefill as unsupported until its L1 program configuration is resized and rerun.
- Repair the triage `noc_read(..., memoryview)` compatibility issue before relying on future worker callstack/running-op captures.
