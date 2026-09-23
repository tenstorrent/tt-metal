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

## 09:35 UTC — source findings and candidate order

- Native DRAM reader is triple buffered with per-block transaction IDs; original megakernel has two buffers and a full read barrier per block. GU8/O/down/QKV bank rows are contiguous across K, allowing whole-block reads; GU16/head halves require strided rows. First interventions: coalesced reads; then triple-buffered, two-block-in-flight reads. Keep arithmetic unchanged and measure L1 capacity.
- Projection output subblocks are GU8=4, GU16=2, O/down=4 tiles; available destination capacity permits testing GU=7 and O/down=8 without changing K accumulation order or rounding locations. This is a separate inexpensive compute intervention.
- A concrete shared-core extension can colocate QKV on eight later SwiGLU workers (disjoint CB indices, LoFi is irrelevant to SFPU-only SiLU/multiply), freeing the existing QKV row for 16 GU workers. This changes worker function across dependent phases; no attention/MLP overlap is assumed. It will be pursued if focused GU geometry evidence warrants it.
- Initial tt-smi enumeration sees all four chips. No device files held afterward. Full matching-runtime connectivity/ring mesh still pending build completion.
- Diagnostic install conflict (tt-smi 5.2 requires tt-umd 0.9.5 vs triage 0.9.9) resolved by separate venv-smi, copied local tt-smi source before installation; main runtime env uses tt-exalens 0.3.32/tt-umd 0.9.9. No system package changes.

## 09:37 UTC — healthy mesh and baseline suite

1280-action build passed; tar, tt_pybinds and runtime libraries installed. Actual mapped libtt_metal agrees with source/installed BuildID 467a3c083d79401b9b46f83db7a46430ca790654. Full connectivity and ring mesh [1,4], 11x10 worker grid, eight DRAM banks passed. No reset required. Baseline suite started under serialized runner; first native then original decode_token for 128/2048/8192/long with five warmed generations. Historical references fix teacher stream; HF diagnostics retained at 128/2048. Harness adds setup/warmup timing metadata only.

## 09:50 UTC — refreshed matched baselines complete

All eight runs pass exact teacher logits, all64 request-touched KV tensors, all greedy outputs, and five repeat generations. Initial-benchmark-summary.json preserves trials and result hashes. Unprofiled median ms/token (native / original resident):

- Context128/32 outputs (31 decode steps): 7.635744386 / 9.243840999.
- Context2048/32 outputs (31 decode steps): 8.067759355 / 9.665413420.
- Context8192/32 outputs (31 decode steps): 8.677945743 / 10.334084839.
- Context128/256 outputs (255 decode steps): 7.678513819 / 9.287472176.

Native is ~1.15 ms faster than historical results at every context, while original resident remains close to history. Cause unisolated; use these refreshed pairs, not historical timings, for candidate score. HF diagnostic target remains failed at 128/2048. Kernel sources were unchanged throughout initial suite.

## 10:04 UTC — initial interventions measured

All nine focused variants pass bitwise real layer0/31 GU/SwiGLU/down and replay. Component medians (us; 20 trials of100 replays, four layer/input cases): original GU8 144.258; wide143.379; coalesced148.086; coalesced+wide147.714; pipelined2+wide141.339; pipelined3+wide133.956. GU16 original153.595, wide152.284, pipeline2+wide147.338. Wider GU work is not currently justified.

Full context128 pipeline2+wide+bounded-barrier passes exact teacher logits, all64 touched KV tensors and32 greedy outputs; five-trial median9.132832ms versus original9.243841 and native7.635744. A measured component/model improvement, still no native-baseline win.

Packer configuration hoisting and per-bank request VC are separately implemented/tested, but have only small component effects: original+hoist144.269, wide+hoist143.615, pipeline2+wide+hoist140.890, pipeline3+wide+hoist133.671, row-pipeline3+wide+hoist135.676, pipeline3+wide+hoist+bankVC133.691. GU16+wide+hoist152.254; pipeline3+wide+hoist+bankVC144.810. All exact. Native reader placement queried: bank-order logical cores [(0,9),(0,0),(0,7),(0,3),(7,9),(7,1),(7,6),(7,4)], unlike row3 resident projection workers. Physical mapping saved in reader-placement-mesh.log; healthy open/close passed.

Next: separate full-model profiles; phase-ahead L1 weight staging on idle norm readers, initially six GU/down K blocks, tested separately to expose contention. Needed reads/refills stay inside token time. Existing arithmetic and first-experiment defaults remain available.
