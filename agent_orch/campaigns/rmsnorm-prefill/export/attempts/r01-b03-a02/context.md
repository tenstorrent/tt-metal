## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml: the job, shapes (20 tile-rows x 28/32/48/56 local tile-cols), gate, allowed paths.
- tests/ttnn/nightly/unit_tests/operations/fused/test_fused_rms_norm_prefill.py: no trace. The host enqueues each call to 4 chips (~0.7 µs apart) at ~13.5-18 µs per call.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp: the parent's col_splits choice, the SliceGroup (1-2 kernel groups)
  machinery, derive_worker_cap comments (64 / 48 knee is a multi-round contention effect; this is 1 round), grid 11x10 = 110 cores.
- git diff root..r01-b03-a01 (whole op): the reader col_start/row stride, the writer peer push / go relay, the compute peer add + full_row_cols.

## Nodes consulted
- r01-b03-a01 (parent): the column-split protocol works (PCC unchanged). Its loss comes from shapes with uneven slices.
  I found the root cause in its ops CSV: OP TO OP LATENCY is 35-52 µs on the 2-kernel-group shapes vs 550 ns on glm (1 group).
- r01-b01-a01, r01-b04-a01: the gamma-reorder wins (1.109 / 1.076). After them, the drain is the critical path on wide shapes.
  Not on this lineage.
- r01-b02-a01: the column split never engaged (34-stick fabric packet cap). That is why the leader-combine exists.

## Profiling (own analysis, scripts in /tmp/*_r01b03a02.py)
- ops_perf_results per shape/chip: o2o, kernel, host us/call (baseline 13.5-14.1, parent 15.2-18.2).
- profile_log_device.csv for the parent's glm, per core: the drain end rises with core x (17.3 -> 23 µs), TRISC end is uniform ~17-18 µs.
  So the post-AG tail is NoC/DRAM write congestion. The NCRISC finishes ~7.5-9 µs and is idle afterwards (an idea for a later node).
