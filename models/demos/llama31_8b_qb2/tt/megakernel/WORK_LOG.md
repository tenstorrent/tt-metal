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

## 10:16 UTC — profiler failure, recovery and hourly checkpoint

Baseline profiler run profile-initial-baseline (source43db96c6, op-support16000) hit a host profiler assertion: end marker without start for setup Tilize on chip0 TRISC streams. Its three signposts do not qualify an incomplete capture. Child/core reporter exited on their own; guarded cleanup sent no signals. Installed missing pycapnp for triage, whose next attempt failed internally in configuration_provider; this is not successful triage. Failure logs/process metadata retained in artifacts/profile-initial-baseline-failure. Authorized reset and enumeration plus full Cluster health/connectivity and ring mesh passed at10:11:56. No firmware changes/reboot. Retry profiling later with smaller capacity/setup draining if evidence supports it.

Prepared (not yet applied) phase-ahead GU/down prefix staging on idle normalization readers. Six GU blocks require774144 B/helper, down731136 B/helper, plus~260KB existing norm CB storage. No additional live DRAM tensors. Helpers publish ready pointers only after read completion; layer barriers prevent overwrite and flags reset each layer. Consumer fetches identical bytes from helper L1. Initial/refill reads are inside token latency; actual overlap and full-model benefit remain hypotheses. Next focused real layer/replay/page/inactive gate before full model.

## 10:25 UTC — prefetch bring-up stall and repair

First six-block GU+down two-layer test (prefetch-loop-six, dirty patch on edddf61d) timed out after300s. Live triage succeeded; all projection NCRISCs waited for GU mailbox while norm helpers were at the end barrier. Found a four-byte publication write with source state[302] and destination state[300/301], violating matching NoC low-address alignment. Changed helper source to the matching state[300+role]. No numeric/performance result from failed run. Saved runner patch plus original overlay/new files, triage/callstacks; runner terminated only its child group. Authorized reset + full health/connectivity/mesh passed10:24:50. Retrying corrected test under Watcher only. Runner now disables child core dumps to avoid unrelated minutes-long crash reporters. Earlier profiler triage failed because its live process was already gone; the live kernel-hang triage worked.

## 10:27 UTC — prefetch focused gate passes

Corrected six-block GU+down stage passes exact real layers0/31 output and all KV contents, inactive(-1) warmups, page migration, page boundaries and repeated trace replay. Run prefetch-loop-six-watcher-workers uses Watcher on all worker cores with Ethernet instrumentation disabled: full Watcher first failed at mesh initialization because fabric ACTIVE_ETH program28800B exceeds26624B kernel configuration capacity (no model ran). Worker-only Watcher run passed with no reported access violation. Prefetch capacity therefore fits this focused composition; full-model/long-context allocation remains to qualify. Next unprofiled GU-only/down-only/both full128 trials, then profile retry4000.

## 10:42 UTC — prefetch measurements and static CB reuse

Full128/32, five trials each, source04ff6b2e: GU6 alone median9.063192839ms, down6 alone11.338431580ms, both6=11.384580224ms. All exact teacher/all64KV/greedy checks. GU6 improves~70us against pipeline2+wide+bounded9.132832; down staging is rejected as currently placed/scheduled. Attribution to contention is a hypothesis, not a measured bandwidth counter.

Static O/GU/down CB aliasing implemented behind alias_projection_cbs: common physical storage, original per-phase ring capacities installed before first layer, then existing loop reconfiguration restores them. O->GU waits for residual/gather/norm; GU->down waits for every SFPU consumer, so scratch lifetimes do not overlap. Three buffers now use~535040B static projection scratch rather than1205248B. Head output/partial rings alias because each final partial subblock is loaded before its output overwrite; saves131072B, making triple-buffer head static991232B. No new persistent tensor storage across prefill. Focused two-layer/page/replay/inactive test with alias+three buffers passed under worker-only Watcher; full head/model qualification next.

Profile4000 native has all999ops on each of4chips in all3windows with endpoints. Deeper timing sanity FAILED for chip2: about21.6ms offset in cross-core operation spans; chips0/1/3 agree at~8.367ms firmware span. Do not qualify the full aggregate timing or tt-perf-report merged output. Chip3 matmul medians (instrumented): QKV17.141us,O13.085,GU44.834,down35.982,head368.593. These suggest projection inefficiency and motivate DRAM-near placement. Resident profile failed source-marker hash collision (42508: ATTENTION-CONCAT line27 and MLP-GU-MATH line258). No kernel hang; child closed cleanly. Raw logs retained. Profiler issues do not invalidate separate unprofiled results.

## 10:51 UTC — full triple buffering and DRAM-near placement gate

Full128/32 source6cddb258: alias+two buffers9.121142872ms, alias+three8.921141804ms, alias+three+GU6 prefetch8.855448322ms, all five-trial medians and exact teacher/all64KV/greedy checks. First experiment9.243841/native7.635744, so still no native win.

Implemented bank-ordered DRAM-near placement for eight shared O/GU/down workers. A bijection moves the displaced SFPU, attention and one pre-attention norm worker into vacated row3 slots; attention retains32 workers/native arithmetic grouping. GU writers scatter logical output onto SFPU storage to avoid conflating bank order with row-major tensor sharding. Original-reader+wide component median138.549510us versus row-wide143.379245, exact. Standalone triple-buffer component fails allocation before launch (static ends1140864 vs live buffer1136384 on core0,9); full loop uses the validated static aliases instead. Full two-layer+alias3 placement gate passes exact page/replay/inactive tests under worker-only Watcher. Full model measurement next. Current-source static profiler hash precheck finds no collision after line changes; earlier raw collision still preserved.

## 11:06 UTC — strongest candidate and rejected GU16/input-read variants

DRAM-near GU8, aliased three-buffer projections, pipelined reader, wide subblocks and bounded layer barrier: context128/32 median8.504254871ms (five trials), exact teacher/all64KV/greedy. Same configuration plus GU6 prefix prefetch regresses to9.795657997ms; prefetch stays disabled. Original resident9.243840999 and native7.635744386 on this reservation. No native win.

Extended component-only GU16 placement to primary bank-near workers plus distinct secondary workers selected by directed NoC distance, preserving half-bank column order. Original reader158.249639us and pipeline3=153.074995us; both exact, both slower than best GU8 component133.955940us. Complete resident still uses GU8. One host API lookup failed before launch; corrected binding recorded in placement.py.

Contiguous activation-block reads preserve K bytes/order for the fixed divisible shard geometries. GU8 row pipeline3=134.202575us (no gain), GU16 DRAM-near150.809020us (small component improvement but still slower). Both exact four-case layer/input/replay checks. Flag remains disabled for strongest full-model candidate; full QKV/head coalesced-input behavior not yet qualified. Next: strongest candidate2048/8192/256-output qualification, then further critical-path experiments.

## 11:11 UTC — strongest candidate qualifies all requested cases

Sourcec96b2d90, five trials each:2048/32=8.930579936ms (8.925874–8.942436),8192/32=9.596915292ms (9.589387–9.604281),128/256=8.544241596ms (8.543873–8.544314). All exact teacher logits, all64 touched KV tensors and greedy outputs; long replay/page-boundary checks passed. Together with128/32=8.504254871 this saves~0.74ms versus originalresident at every case, but still loses~0.87–0.92ms versus refreshednative. Qualification summary has exact trials/hashes. New separate profile-placement3 capture running; no Watcher/headline timing mixed. Hourly durable checkpoint refreshed.

## 11:17 UTC — terminal-head look-ahead bring-up passes

Implemented opt-in prefetch_head_workers: sixteen idle terminal head NCRISCs reuse existingCB1 (835584B each). Eight helpers stage full QKV835584B/bank, eight full O557056B/bank; consumers acknowledge only after all helper-to-local reads complete, allowing next-layer DRAM refill during later phases. Ready/consumed sense bits persist across replays and cannot wrap. First-layer reads and every refill are inside token latency; no cross-token weight residency assumed. Adds4096B static table scratch/head worker and three small semaphore fields on helper/consumer cores, no persistent full-weight tensors. It adds~11MB/chip/layer remoteL1 traffic, so latency benefit remains a hypothesis.

Full128/32 one-repeat worker-only Watcher run head-prefetch-watcher128 passed exact teacher logits/all64 touched KV/greedy via new explicit --require-exact gate, including inactive capture/warmup. Watcher inspectedall4chips; Ethernet remains excluded for known code-size limit. Separate unprofiled on/off trials next. Trimmed optional MLP wait/write markers only in layer-loop compilation so reader phases fit512-word marker buffer; unprofiled kernel behavior unchanged.

Profile-placement3 all4chips/all3windows operationcoverage passed, but O/GU/down reader markers truncate at20–21layers. QKV/O/GU/down compute phases coverall32layers and include dependency waits, not isolated compute cost. Chip2 cross-core clock offset remains. phase-coverage-audit.json explicitly marks timing/phase limitations. Compressed validated raw profiles now durable in profile-archives (~1.4GB); no truncated reader timing used as full-token evidence.
