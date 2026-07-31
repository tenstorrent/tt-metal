# AUTOTRIAGE

## Diagnosis

- The leading diagnosis is an incomplete layout contract at the first
  advisor-seeded 1D matmul boundary, most likely the QKV projection or the
  first dense gate projection: the hand-transcribed program configuration is
  paired with generic `L1_MEMORY_CONFIG` /
  `L1_WIDTH_SHARDED_MEMORY_CONFIG` conversions rather than the advisor IR's
  concrete per-value shard specs. The earlier
  `Sharded inputs require sharded outputs` fatal directly establishes that
  the seed crossed an incompatible sharded/interleaved residual boundary.
  Adding boundary reverts removed that validation failure but did not prove
  that the producer, conversion, and consumer agree on grid, shard shape, and
  core ordering.
- This is a ranked diagnosis, not a proven device stop-site. The retry was
  externally terminated after 6:46 with no post-setup output, and the installed
  `tt-triage`/`tt_umd` combination could not read either call stacks or running
  operations. The first repair action must therefore localize the exact
  operation before changing more than one boundary.

## Triage Evidence

- Retry command:

  ```text
  GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZATION_POLICY=advisor_seed \
    pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_real_weights_prefill_decode[blackhole-sliding_attention-device_params0-mesh_device0] -q
  ```

- The pytest process remained alive for 6:46 and emitted no output after
  setup. It was externally sent `SIGTERM`; pytest's nominal 300-second timeout
  did not terminate it. No second TT fatal was observed.
- All four devices listed healthy immediately after termination, and no reset
  was required. This argues against persistent ARC/PCIe corruption, but does
  not distinguish a host wait from a recoverable device-kernel wait.
- The raw triage capture contained no usable device state and was not retained
  after the diagnosis because it exceeded the repository's compact-artifact
  limit. `dump_callstacks.py` and running-operation aggregation both failed on
  every attempted worker with:

  ```text
  noc_read(): incompatible function arguments
  ... Invoked with ... int, int, int, int, memoryview
  ```

  The installed binding accepts a returned byte count or a `bytearray`, while
  the triage helper passed the old `memoryview` signature. Consequently the
  artifact proves only a tooling API mismatch; it proves no RISC stop-site,
  CB wait, NoC state, active operation, or all-pass condition.
- The previous run's fatal, recorded in `work_log.md`, is the only direct
  kernel-contract evidence:
  `TT_FATAL: Sharded inputs require sharded outputs`.
  The broad later silence must not be classified as the same failure without
  localization.

## Source Evidence

### Ranked hypothesis 1: implicit shard specs do not faithfully reproduce the IR

- `optimized_decoder.py:_attention_decode` converts normalized residual input
  using generic `ttnn.L1_MEMORY_CONFIG`, invokes QKV with an advisor 1D program
  config, and requests generic
  `ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG`.
- It repeats the pattern at output projection. `_dense_mlp` repeats it for
  gate/up/down, including a width-sharded-to-generic-L1 conversion before the
  down projection and a generic DRAM conversion afterward.
- By contrast, `shard_advise/final_ir.mlir` assigns distinct concrete layouts
  to these values (`#ttnn_layout12`, `13`, `22`, `10`, `19`, `23`, `24`).
  A layout enum alone does not document the IR's shard shape, core range,
  orientation, or physical padding.
- Producer/consumer ledger:

  | Boundary | Producer | Expected consumer | Current transition |
  | --- | --- | --- | --- |
  | residual -> QKV | decoder RMSNorm/add in DRAM contract | 11x8 QKV 1D mcast | generic L1 interleaved conversion |
  | QKV -> head split | 11x8 width-sharded matmul output | decode QKV head split with height-sharded output | implicit width shard, then head split |
  | concat -> O-proj | concat then explicit DRAM interleave | 11x8 O-proj 1D mcast | DRAM -> generic L1 -> implicit width shard |
  | residual -> gate/up | decoder DRAM residual/norm | two 11x6 1D mcasts | generic L1 shared input |
  | gate/up -> GELU/mul | two implicit width shards | elementwise same-shard consumers | generic width-sharded outputs |
  | mul -> down | width-sharded elementwise result | 11x8 down 1D mcast | generic L1 interleaved conversion |
  | down -> residual add | implicit width-sharded output | existing DRAM residual contract | explicit generic DRAM conversion |

  The initial fatal fits the last row before its revert. The retry could stop
  at any earlier row; no evidence yet identifies which.

### Ranked hypothesis 2: the failure is confined to one 1D role, not the whole seed

- The real-weight test runs functional prefill first; the advisor policy only
  changes batch-1 decode attention and dense MLP. Therefore a stall after
  setup is consistent with the first decode QKV, O-proj, gate/up, or down
  operation, plus their conversions.
- Four geometries are introduced simultaneously:
  QKV `(11x8, in0_block_w=2, per_core_N=3)`,
  O-proj `(11x8, 8, 1)`,
  gate/up `(11x6, 8, 1)`, and
  down `(11x8, 2, 1)`.
  The retry provides no evidence that any individual role works.
- A binary role-by-role A/B is therefore higher-value than another full-layer
  retry. It will also separate compilation/dispatch latency from a device
  wait.

### Ranked hypothesis 3: `num_global_cb_receivers` transcription mismatch

- The advisor IR explicitly emits `gather_in0=false` and
  `num_global_cb_receivers=0` for all five relevant linear ops.
- `_advisor_1d_program_config` omits both fields. Runtime inspection prints
  `gather_in0=0` (matching) but `num_global_cb_receivers=1` (different), because
  the nanobind constructor defaults the latter to 1.
- This discrepancy should be corrected for exact IR reproduction, but it is
  not currently the best hang explanation. In
  `matmul_device_operation.cpp`, `num_global_cb_receivers` validation and the
  global-CB receiver protocol are under `if (program_config.gather_in0)`;
  factory code selects the remote-CB path only when an actual `global_cb` is
  present. With `gather_in0=false` and no global CB, receiver count appears
  inert. An isolated A/B must verify this rather than assuming either value is
  harmless.

### Lower-ranked alternatives

- A one-time compile exceeding five minutes is possible but does not explain
  the preceding layout fatal, and the run remained past the pytest timeout.
  Record host stack and compile-cache timestamps to distinguish compilation
  from dispatch.
- Paged sliding SDPA and cache update are lower-ranked because the advisor
  candidate does not change their program config or cache contract. They
  remain possible downstream victims if malformed Q/K/V layout reaches them.
- Persistent device damage is demoted by the clean post-termination device
  list and lack of required reset.

## Downstream Effects

- A host wait in synchronization, pytest teardown, or mesh close would be
  downstream unless a host stack shows the run never entered a seeded op.
- Paged cache, SDPA, and residual addition can wait on or consume a malformed
  upstream tensor; they should not be blamed merely because they occur after
  a seeded matmul.
- The healthy device list after `SIGTERM` is recovery evidence only. It is not
  correctness evidence for the advisor configuration.
- The previous residual-boundary fatal must not be treated as rejection
  evidence for all advisor recommendations; it proves only that the original
  mixed layout chain was incoherent.

## Proposed Fix

No implementation fix is authorized in this triage task. Use these focused
verify/refute experiments, in order:

1. Add temporary test-only operation breadcrumbs with an explicit
   `ttnn.synchronize_device()` after each decode boundary: pre-QKV conversion,
   QKV, head split/RoPE/cache updates/SDPA/concat, O-proj, residual revert,
   gate, up, GELU/mul, pre-down conversion, down, and final DRAM revert.
   Run with an outer shell timeout and preserve the last completed marker.
   This identifies the first stuck producer rather than a downstream waiter.
2. Run four isolated real-shape single-linear probes, one per distinct
   geometry/weight role. For each, compare:
   functional DRAM; advisor program plus generic layouts; and advisor program
   plus the exact concrete memory config derived from the corresponding IR
   layout. Synchronize immediately after the op. A hang only in the generic
   layout variant confirms hypothesis 1.
3. For the first failing isolated role, A/B only
   `num_global_cb_receivers=0` versus `1`, keeping
   `gather_in0=false`. If both complete identically, refute hypothesis 3 and
   still set the production helper explicitly to the advisor value for
   transcription fidelity. If only `1` hangs, capture it as the minimal repro.
4. Capture a host Python stack while hung (`py-spy dump` or `gdb -p` where
   permitted). A stack in program compilation/cache locking refutes a device
   CB diagnosis; a stack in command-queue finish/synchronize supports an
   in-flight device operation.
5. Repair the triage helper/binding mismatch or use the supported
   `noc_read(..., size) -> bytes` form, then rerun focused running-op and
   call-stack capture against the minimal isolated probe. Keep the workload
   alive until evidence is collected. Do not report the current failed
   capture as all-pass.
6. Once the exact role and stop-site are known, make the smallest layout
   change and rerun the isolated probe twice before the full real-weight
   layer. A first API error or first successful invocation is insufficient:
   require repeated completion, PCC, and watcher-clean evidence.

The likely source-level repair, if experiment 2 confirms hypothesis 1, is to
construct and use explicit shard specs matching each advisor IR value (or to
reject that role with isolated evidence and retain the proven DRAM boundary),
instead of combining the advisor program geometry with generic memory-config
constants.

## Uncertainty

- There is no valid device stop-site, kernel name, CB/semaphore ledger, or
  running-op record for the retry because the triage reader itself failed.
- The test emitted no operation-level breadcrumbs, so the exact seeded role is
  unknown.
- It is unknown whether the process was waiting in compilation, dispatch,
  device synchronization, or teardown when terminated.
- The meaning of the advisor IR's concrete layout aliases must be extracted
  from the IR layout declarations before claiming that generic TTNN memory
  configs are equivalent.
- `num_global_cb_receivers=0` is a proven transcription difference but only an
  unverified causal candidate while `gather_in0=false`.
