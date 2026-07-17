# AutoFix Report

## Starting Evidence

- Starting report: `doc/multichip_decoder/AUTOTRIAGE.md`.
- Original failing command: `FALCON3_RUN_MULTICHIP_FUSED_BOUNDARY=1 pytest -q -s tests/test_multichip_decoder.py::test_fused_reduce_scatter_distributed_norm_all_gather_matmul_boundary` (path shown relative to the model autoport directory).
- Original symptom: the ordinary O/all-reduce/residual/RMSNorm/QKV boundary completed, while the first fused warm invocation failed to complete in more than 60 seconds. The pre-reset capture is in `logs/fused_boundary_hang_triage.txt` and `logs/fused_boundary_hang_summary.md`.

## Hypothesis Experiments

### H1: oversized reduce-scatter workspace causes the fused hang

- Hypothesis: the fused O matmul produces `[1,1,32,3072]`, so changing the persistent intermediate from `[4,1,32,3072]` to the reference `[1,1,32,3072]` should make fused RS complete.
- Focused experiment: `test_fused_rs_workspace_probe` first runs ordinary matmul plus `reduce_scatter_minimal_async`, synchronizes and prints `STANDALONE_RS_PASS`, then runs only `matmul_reduce_scatter_async` with no residual, norm, statistics AG, or AGMM.
- Command:

  ```bash
  FALCON3_RUN_MULTICHIP_RS_PROBE=1 \
  FALCON3_RS_PROBE_LINKS=2 \
  FALCON3_RS_PROBE_WORKSPACE_BATCH=1 \
  scripts/run_safe_pytest.sh --dev \
    models/autoports/tiiuae_falcon3_10b_base/tests/test_multichip_decoder.py::test_fused_rs_workspace_probe -s -vv
  ```

- Result: standalone two-link Ring RS completed; corrected-workspace fused RS hit the five-second dispatch timeout on all four devices. Watcher identified the active fused matmul kernels plus `ring_reduce_scatter_minimal_async_{reader,writer}` and `ring_reduction`. The safe runner reset all four devices.
- Verdict: **refuted as the cause**. The workspace was corrected because it matches the reference contract, but the fused hang remains.

### H2: the fused primitive fails only with two Ring links

- Hypothesis: TP4 fused matmul-RS may be correct with one link and broken only in the two-link route.
- Command:

  ```bash
  FALCON3_RUN_MULTICHIP_RS_PROBE=1 \
  FALCON3_RS_PROBE_LINKS=1 \
  FALCON3_RS_PROBE_WORKSPACE_BATCH=1 \
  scripts/run_safe_pytest.sh --dev \
    models/autoports/tiiuae_falcon3_10b_base/tests/test_multichip_decoder.py::test_fused_rs_workspace_probe -s -vv
  ```

- Result: standalone one-link Ring RS completed and printed `STANDALONE_RS_PASS`; fused RS again timed out with the same fused matmul/RS kernel family active. The safe runner reset all four devices.
- Verdict: **refuted**. The failure is not specific to two-link routing.

## Final Status

- Status: **legitimate primitive limitation; graph rewrite rejected**.
- The isolated controls prove that the TP4 Ring data geometry and standalone reduce-scatter work with both one and two links. `matmul_reduce_scatter_async` itself does not complete for this Blackhole `[1,1,32,768] @ [1,1,768,3072]` geometry, even before any distributed RMSNorm or all-gather-matmul is enqueued.
- No speculative production change was kept. The decoder retains separate local row matmul plus BF16 Ring all-reduce, which is correct, trace-safe, watcher-clean in its model test, and faster than the measured topology/dtype alternatives.
- The manual isolated probe remains as a durable reproducer. Repairing the fused TTNN primitive is outside this model-stage scope; after such a repair, the exact boundary test must be rerun before a sharded-residual graph can be reconsidered.

### Follow-up exact-boundary experiments

- Standalone reduce-scatter, local residual add, distributed RMSNorm, standalone
  all-gather, and ordinary matmul completed and matched the replicated boundary
  at PCC `0.99979438`. With one trace replay per synchronization, its isolated
  median was `1.370169 ms` versus `1.381010 ms` (`1.007912x`).
- The latency result is not accepted: two attempts passed inside pytest and
  then failed device teardown on active Ethernet core `29-25`. Each required
  the bounded list/reset/list sequence and passed a subsequent 1x4 mesh smoke.
  `results/graph_rewrite_final_decision.json` records this clean-exit rejection.
- AutoFix source review also found that fused
  `all_gather_matmul_async_program_factory.cpp` hardcodes
  `num_transfers = 4`, matching its located 1x8 reference. The receiver expects
  `num_transfers * 2 * 3 = 24` blocks, while Falcon TP4 supplies 12. A TP4
  implementation would need `ring_size / 2 = 2`; that core-source repair is
  outside this model-stage scope.
- Final decision: retain the replicated BF16 two-link Ring all-reduce path.
  The small isolated latency win does not override the clean-exit/watcher gate.

### H3: complete sub-device cleanup makes the standalone rewrite safe

- Hypothesis: the standalone boundary poisoned Ethernet teardown because it
  reset the stall group but did not clear the loaded sub-device manager.
- Patch: the test now mirrors the fused-RS cleanup with
  `reset_sub_device_stall_group()` followed by
  `clear_loaded_sub_device_manager()` after synchronized trace replay and
  persistent-buffer deallocation.
- Result: the boundary passed PCC `0.99979438`, measured `1.370985 ms` versus
  `1.381470 ms` (`1.007648x`), completed pytest teardown in `0.15 s`, and the
  process exited normally. The immediately following 1x4 decoder smoke failed
  during mesh open because active Ethernet core `29-25` had no heartbeat. A
  bounded four-board reset was still required; a post-reset mesh open/close
  smoke passed.
- A production integration prototype was also rejected before retention: merely
  constructing the required shared async-CCL semaphore state made the real
  batch-32 watcher run exceed the TENSIX kernel-config buffer
  (`72,640 > 70,656` bytes), and its failure teardown again left core `29-25`
  without a heartbeat.
- AutoFix verdict: **failed at the decoder boundary**. Every trace sample was
  synchronized; host readback completed; worker dispatch state was cleared;
  mesh-command-queue destruction performs a blocking drain; the directly
  constructed `TT_CCL` was not in a process-persistent cache. Global semaphore
  wrappers cannot restore ERISC firmware state. The poisoned heartbeat survived
  normal mesh close and process exit, and only a board reset recovered it.
  Additional decoder-local `del`, cache clearing, garbage collection, or sync
  calls cannot repair this fabric-runtime defect.

The stable all-reduce graph is therefore the final selected path. Reconsider
the sharded-residual candidate only after the async CCL/fabric termination bug
and TP4 fused-AGMM transfer-count limitation are fixed below this decoder.
