# AutoFix Report: watcher async CCL assertion

## Starting evidence

- Source report: `AUTOTRIAGE.md`; original artifact: `watcher_full_path.log`.
- Original symptom: the reduced real-weight generator aborts at its eager decode
  synchronization. Watcher reports BRISC line 119 in
  `experimental/ccl/all_gather_async/.../minimal_default_writer.cpp`.
- All runs below used TP4 `FABRIC_1D_RING`, watcher interval 10,
  `TT_METAL_WATCHER_DISABLE_ETH=1`, and fallback exceptions.

## Hypothesis experiments

### Persistent modulo-three resource reuse is required

- Experiment: rerun the original generator probe with only
  `GEMMA4_MULTICHIP_PERSISTENT_ALL_REDUCE=0`.
- Command: `GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_MULTICHIP_PERSISTENT_ALL_REDUCE=0
  TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' pytest -q -s
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_real_weight_full_model_probe`
- Result: exit 134, with the same device-0 `(0,0)` BRISC line-119 assert in
  `minimal_default_writer.cpp` at the same eager generator synchronization.
- Verdict: **refuted**. Persistent Gemma buffers/semaphores and their modulo-three
  reuse are not necessary. The non-persistent reduction still lowers through the
  same async all-gather writer.
- Evidence: `autofix_watcher_persistent_disabled.log`.

### Failure occurs before the first resource-slot reuse

- Experiment: temporary, subsequently removed test-harness changes made the
  model-only branch use the canonical 32 decode slots (one active position 32,
  31 inactive rows), select arbitrary layer indices, and pass logical
  `batch_size=1`. Layer 5 alone executes two full-attention reductions (slots
  0/1); layers 5 and 11 execute four (slots 0/1/2/0).
- Commands: the same watcher/fallback command above plus
  `GEMMA4_MODEL_TRACE_ONLY=1 GEMMA4_PROBE_LAYERS=5`, then
  `GEMMA4_PROBE_LAYERS=5,11`.
- Result: both passed (exit 0), including eager synchronization and trace
  capture. No production or test implementation change remains.
- Verdict: **refuted** for both pre-reuse and first-reuse model-only cases. The
  original failure requires additional full-generator work or its particular
  collective sequence; it is not explained by the shared modulo-three ring.
- Evidence: `autofix_watcher_model_layer5.log` and
  `autofix_watcher_model_layers5_11.log`.

### Invalid endpoint-direction connection access

- Source experiment: line 119 has a stronger exact included-header match than
  the packet-header-pool theory:
  `fabric_connection_manager.hpp:119` is
  `ASSERT(has_forward_connection())` in `get_forward_connection()`.
  In the non-worker-mux path, `minimal_default_writer.cpp:216-217`
  unconditionally calls `get_backward_connection()` or
  `get_forward_connection()`. Only later send/use sites are guarded by
  `detail::valid_targets(direction)`. An endpoint writer can therefore assert
  while acquiring a direction it will never use.
- Verdict: **verified at the source boundary and consistent with the exact
  watcher line**. The focused passes show it is invocation/topology-argument
  dependent, which is also consistent with an endpoint direction rather than
  global packet-header state.

### One-chunk scatter header initialization

- After guarding endpoint connection acquisition, the original repro advanced
  past line 119 and stopped on line 260 at a different core.
- Exact source contract: `api_common.h:260` requires scatter chunk count 2..4.
  Both data-loop scatter calls are already guarded by `tiles_to_put > 1`; only
  unconditional header prepopulation could pass the compile-time value one.
- Fix: compile scatter-state setup only when
  `num_tiles_to_write_per_packet > 1`, while preserving the unicast state/path.
- Verdict: **verified** by the next exact watcher rerun.

## Final status

- **Fixed.** The shared direct-fabric writer now avoids nonexistent endpoint
  connections and invalid one-chunk scatter setup.
- The exact original reduced generator watcher command, including greedy then
  sampled trace transition and fallback exceptions, passes: one test in 11.56
  seconds (`watcher_full_path_fixed_v2.log`).
- Nearby validation passes: ordinary reduced profile, mixed prompt/inactive
  slot probe, all-30-layer batch-32 probe, AIME24 prefill/teacher forcing, and
  host-visible plus no-readback token-out measurements.
- Persistent CCL, nonblocking replay, device token feedback/position advance,
  dtype/fidelity policy, topology, and split sampling remain enabled. No host
  synchronization or speculative packet-pool reset was retained.
