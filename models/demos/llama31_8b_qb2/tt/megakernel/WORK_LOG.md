# Experiment 2 work log

## 2026-09-23 09:30 UTC — preparation

- Isolated checkout on `codex/llama31-qb2-megakernel`, initial SHA `f776a26ce77921cc84331cafb1434ba20c6ec46b`; original base `b8915544692d8f9feb2c890afbc2f22791560cd2`. Experiment 1 remains reproducible at its SHA; original progress copied to EXPERIMENT1_PROGRESS.md.
- Read reservation, prior README/progress, supplied REPORT and benchmark summary. Prior measured complete kernel is slower; HF diagnostic 0.99 target did not pass either implementation.
- Local `scontrol show job 114624` verifies RUNNING, moconnor, exact host qb2-120-p01t01, OverSubscribe=NO and expiry 17:12:44 UTC. `scontrol show jobs -o` exact-node filter contains only this job. SSH login failed public-key authentication; local `squeue` lacks select/graph plugin, so used working job queries. Evidence: artifacts/reservation-live.txt and exact-node-jobs.txt.
- No hardware touched yet. Local Clang 20.1.8, CMake 4.0.2, Python 3.12.3 match. Initializing pinned submodules, configuring local build-current and venv-current; pinned Torch 2.11 CPU and Transformers 5.12.1. Commands adapted from read-only prior mirror; both tar and tt_pybinds install components will be installed and build IDs checked.
- Sandbox namespaces unavailable (bwrap); authorized commands use reviewed execution outside it.
- Serial device runner adapted with allocation/host/expiry guard, unique logs, source SHA/patch capture and bounded timeout/triage. Work deadline 17:02:44, preservation begins 16:52:44.

## Initial model and discriminating experiments (hypotheses, unmeasured)

Per chip/layer selected payload is 6,684,672 B QKV + 4,456,448 B O + 16,515,072 B gate/up + 15,597,568 B down = 43,253,760 B, excluding norm/KV/metadata; 32 layers read 1.384 GB/chip. These are tensor bytes, not DRAM bus counters. The 8-worker GU body already reuses its workers for O and down; it double buffers weight blocks but starts phase reads only after the activation join. Native GU has 16 compute workers. All 86 layer workers synchronize twice per layer and reset CB state, while 24 terminal workers wait until the head.

First reproduce native and original resident paths at all requested cases. Then use controlled component and full-model interventions to choose among: wider shared projection work; weight prefetch before activation joins/deeper buffering; reducing global reset/barrier overhead with bounded counters. Preserve quantization and arithmetic order initially. A shared pool is a candidate, not a goal by itself. Charge initial/refill prefetch to token latency. Separate profiler/watcher from latency.

Artifacts live at /home/moconnor/llama-minlat-114624/artifacts and durable /data/moconnor/llama-minlat-114624. Raw tensors/build/cache stay out of Git.
