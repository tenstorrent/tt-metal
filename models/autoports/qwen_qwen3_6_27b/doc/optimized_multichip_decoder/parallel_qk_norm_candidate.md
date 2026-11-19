# Parallel Q/K norm adaptation (not hardware validated)

The serial candidate has correct TP4 output but cannot yet beat two native norms. It executes Q and K on each user core sequentially. This adaptation gives Q and K distinct workers in one GenericOp. Q workers 0..B-1 read local shards. K workers B..2B-1 remotely read immutable K shards from the original owner 0..B-1. The source owner virtual coordinates come from `worker_core_from_logical_core`, not a hard-coded Blackhole coordinate translation; `UnicastEndpoint` performs the active NoC coordinate conversion. Output page ownership remains `(branch, user*8+tile)` with no collisions.

Only new files were created. Existing candidate/main/probe files were not edited. The FP32 reduction, FP32 norm weights, epsilon and single BF16 output rounding are identical to the serial candidate. The maximum footprint is 64 workers (8x8), 69,632 bytes of static CB storage per worker for FP32 weights. Outputs retain the existing DRAM contract for partial RoPE. There is no tensor transfer to the host inside the candidate and no setup-only weight copy.

`parallel_qk_norm_cpu_audit.json` covers all batches 1..32, disjoint workers and output pages, complete coverage and local-source shard address bounds. Python syntax compilation passed. This proves ownership arithmetic only, not hardware compilation, watcher cleanliness or performance.

Parent-serialized first hardware command:

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
TT_METAL_WATCHER=10 "$D/run_job.sh" parallel_qk_norm_first env TT_METAL_TRACE_ALLOC_TRACKING=1 python -m models.autoports.qwen_qwen3_6_27b.tests.probe_parallel_qk_norm --batches 1 3 17 32 --weight-dtype float32 --output "$D/parallel_qk_norm_first.json"
```

After passing, repeat without watcher, tracking disabled, 5 samples x 100 replays using a fresh output name. The probe compares the entire native reshard-plus-norm path to the parallel candidate, checks unchanged .995 native and torch PCC per branch/rank, zero head/epsilon, exact changed-input trace replay and input immutability. Broaden watcher coverage to every batch 1..32 before integration. Full-layer real-activation and final-default gates remain required. No policy hook is installed by this new helper.
