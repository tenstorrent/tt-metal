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

## 11:22 UTC — prefetch loses; remaining counters explicitly bounded

Terminal-head full QKV+O staging source86dd2b89 context128/32 five-trial median10.589174352ms; matched staging-off8.506229290ms. Both pass strict exact teacher/all64KV/greedy gate. Extra traffic/placement contention is a hypothesis; isolate QKV and O next before discarding all helper staging.

Added host reservations sharing collective semaphore lifetime: bounded loop<=2^26 collective phases (1,048,576 full invocations), original monotonic loop<=2^25 (524,288). Largest native gather counter is8*phases; old barrier at most110*phases. Both stay below uint32 wrap, explicitly including warmup/capture/direct calls and generator trace replay. Raw execute_trace users must reserve their replay count; documented bounded API, no indefinite-service claim. Exact source boundary check accepted the last64 phases and rejected the next64 without mutating count for both limits. Subsequent full benchmarks exercise the generator hook.

## 11:27 UTC — head staging rejected by controlled isolation

Context128/32, five trials, dirty33b3c0fc selector patch: QKV-only staging10.020776742ms, O-only9.069761453ms; both strict exact. Compared withoff8.506229290 andboth10.589174352, each staging role independently loses, approximately additive. Keep helper staging disabled. DRAM issue/NoC contention remains an attribution hypothesis; no overlap speedup claimed. Counter-budget generator hook exercised by these full runs without changing numerical outputs.

Prepared direct QKV/O/GU/down worker reuse: move QKV math onto the existing eight bank-near projection workers before attention, alias its dependent A/B/partial buffers, scatter six output tiles/bank into original packed layout. Removes eight fixed QKV workers from layer barrier participation (78 active layer workers); no claim that fewer workers/programs is itself faster. Same LoFi/BF16 arithmetic/order and weights. Focused Watcher real layer/page/replay gate next.

## 11:30 UTC — shared QKV pool focused gate passes

Direct QKV reuse adds CB8(A),9(B),26(partial),27(output) to the shared projection workers; A/B/partial alias dependent O/GU/down storage with original logical capacities. Six QKV tiles/bank scatter into original packed storage before RoPE/cache notification. Layer workers drop86->78; terminal head/norm placement is explicitly held fixed for this controlled test. Estimated extra static projection storage~61KB (larger shared A ring plus12KB QKV output), no new persistent activation tensor.

First focused attempt shared-qkv-loop-watcher was rejected before model launch: hardware semaphore IDs must be<=15. Replaced proposed16/17/18 with projection-local unused8/9/12; local norm8/9 are separate addresses on separate cores. No device hang/reset. Corrected shared-qkv-loop-watcher-v2 passed exact real layer0/31 output/KV, remapped pages and boundaries, inactive(-1), and repeated trace replay under worker-only Watcher onall4chips. Full128 latency/exact validation next.

## 11:33 UTC — shared QKV loses first full measurement

Source6356d4d8 full128/32 five-trial median8.626025546ms, strict exact teacher/all64KV/greedy. Compared with8.506229290 fixed-function QKV, direct reuse loses~120us despite bank-near placement. Packed-output scatter and scheduling changes are candidates for attribution, not yet isolated. Keep the separately placed QKV for strongest result. Source remains opt-in for further controlled testing. Reduced shared-QKV profiler detail to three selected read/math/barrier zones/layer so all32 fit optional marker capacity; raw-replay focused harness now reserves counter budget explicitly.

Next controlled experiment increases aliased layer-projection buffering from3 to4/5 slots, and outstanding DMA blocks from2 to3/4; terminal head stays capped at3 buffers by L1 capacity. This tests whether a deeper reader window helps without adding helper-to-consumer traffic. Same weights/activation bytes/K order, no numerical reassociation.

## 11:34 UTC — deeper read-window focused gate passes

Aliased5-buffer/4-outstanding-block composition passes focused exact real layers/pages/inactive/replay under worker-only Watcher (buffer5-lookahead4-loop-watcher). Slot TRIDs1..5 fit hardware0..15; each individual block remains below half255 transaction credits, prior slot completions are waited before reuse, and all final blocks drain before resetting TRID0. Terminal head independently caps at3 buffers/lookahead<=3 to fit L1. Next full128 matrix:4/2,5/2,4/3,5/4 (buffers/outstanding). No helper prefetch or shared-QKV changes in these timings.

## 11:41 UTC — deeper windows give only a small screening gain

Source6992ac33 context128/32, five trials all strict exact: buffers/outstanding4/2=8.497506035ms,5/2=8.501549064,4/3=8.490740419,5/4=8.556033034. Prior3/2=8.504255–8.506229. The~15us advantage of4/3 needs paired/interleaved confirmation;5/4 regresses. No native win.

Next design preloads a2/3-block prefix into each projection's own existing weight ring before the activation wait. It adds no weight bytes or helper-to-consumer copy. O and QKV can start at layer entry; GU waits for local O writer completion, down for local GU writer completion, then reads during norm/SwiGLU dependency windows. Local semaphore8/9 marks aliased-storage release; helper prefetch/shared-QKV combinations are rejected for this initial experiment. Synchronous prefix completion before activation wait makes the refill charged and readiness explicit. Focused exact/Watcher gate precedes per-phase/all full measurements.

## 11:43 UTC — local early-weight prefix focused gate passes

Three-block early prefixes for QKV/O/GU/down passed exact focused real-layer/page/inactive/replay gate under all4chip worker Watcher (early-local3-loop-watcher). Each prefix completes before its activation wait; normal block streaming publishes the same bytes and skips its original DRAM read. Existing weight rings hold the prefix, so no L1-capacity increase and no duplicated weight traffic. GU/down joins wait only for the preceding local output writer to establish aliased-weight storage is free. Prefix staging is within every layer/token and is not amortized. Full128 per-phase/depth isolation next. Extra profiling zones need selective capture to avoid optional-marker truncation; no profile timing claimed for this candidate yet.

## 11:53 UTC — early local-prefetch full isolation

Sourcecca5f7e8 context128/32 five-trial medians, all strict exact: all phases/3blocks8.439294355ms; all/2blocks8.407293838; QKV3 alone8.472632966; O3 alone8.480771775; GU3 alone8.472652550; down3 alone8.574672224. Versusoff8.504–8.506, local prefetch helps modestly; three-block down prefetch independently loses. Add QKV+O+GU-only selection to test that interaction. Strongest128 is provisional8.407294; other contexts not yet tested for this feature.

Prepared scratch-init-once flag: selected norm constants/statistic padding and RoPE/V-cache/attention-concat padding are initialized at layer zero of EVERY invocation, then only their defined data lanes are overwritten during the remaining layers. The selected buffers have stable, disjoint storage and no other writer to their padding/constant lanes. Layer ordinal is published through existing start synchronization in state[483]. No inter-invocation/prefill residency assumed. Test norm and padding separately as well as together; full original clear path remains default.

## 11:54 UTC — once-per-token scratch initialization gate passes

The combined padding/norm scratch-init-once configuration passed exact focused layer0/31 output/allKV, inactive(-1), page migration/boundaries and replay under worker-only Watcher onall4chips. No extra scratch storage is allocated. Norm constant rings are republished after metadata reset; selected statistic padding and RoPE/V-cache/concat padding keep initialized lanes within one invocation. Full128 all/padding/norm isolation next, plus early-prefix2/3 excluding down.

## 12:03 UTC — scratch/full-prefix isolation and multicast candidate

Source269afedd context128/32 five-trial strict-exact medians: scratch all8.363791099ms, padding-only8.417085707, norm-only8.449468093. Their savings are approximately additive versus8.506229. Early-prefix without down:2blocks8.425361807 and3blocks8.414819127; all/2 remains slightly stronger8.407293838. Combine scratch-all plus early2 next, and finally test contiguous activation reads across full QKV/head (prior GU8 component showed no gain but did not qualify that full combination).

Prepared multicast release replaces86 individual semaphore writes with two worker-only NoC rectangles on this mesh. Release field is allocated onall110 recipients, including idle head workers; arrival/epoch remain on actual layer workers. Rectangles are derived from the checked logical-to-physical worker map and split across missing NoC columns/rows. Sender-containing rectangle uses loopback with inclusive destination count; others use ordinary multicast. Arrival reset/sense ordering and both layer boundaries remain unchanged. Focused combined scratch/early/coalesced-input Watcher gate next.

## 12:05 UTC — multicast combined focused gate passes; checkpoint

Multicast release + scratch-all + early-prefix2 + contiguous activation reads passed focused strict exact layer/page/inactive/replay validation under worker-only Watcher onall4chips (multicast-combined-loop-watcher). Release broadcasts target two reserved worker rectangles without crossing DRAM/nonworker columns. Full-model separate/composite latency screening next. New screening-summary JSON contains every unprofiled full run's source/tuning/trials/setup and exact comparison metadata, excluding Watcher/profile timings. Hourly durable bundle and evidence refreshed.

## 2026-09-23 12:19 UTC — combined screening

Source bcd77aa8, five warmed unprofiled full128/32 trials per variant, exact teacher/all64 KV/greedy checks passed. Foundation+multicast8.438444129; +coalesce8.501564291; scratch+early2all8.267074999; +coalesce8.260359871; +multicast8.205477419; +both8.194119260 ms/token. Best trial range8.193365227–8.197416225. Best remains0.558375ms slower than fresh native7.635744. Qualification now running: `bash ../artifacts/commands/combined-qualification.sh`.

## 2026-09-23 12:24 UTC — combined qualification and new component candidates

Combined candidate passed exact teacher/all64 touchedKV/greedy at2048/32=8.625901774,8192/32=9.296411840,128/256=8.238191223 ms/token, five warmed unprofiled trials each (31/255 denominator). Native remains faster. Added opt-in raw GU bank-half layout for16workers and batched seven-tile SwiGLU preserving BF16 intermediate. Neither new change is qualified yet. Raw permutation retains originalweights, adds16,515,072B/chip/layer DRAM and copiesonce duringconstruction. Added explicit experimental_setup_seconds to fullbenchmark (synchronized, outside warmedscore). Nextcommand `bash ../artifacts/commands/gu-swiglu-components.sh`: separate WorkerWatcher gates then paired componenttimings.

## 2026-09-23 12:27 UTC — component losses, selective profiling

75ff6b81 rawGU16 passed WorkerWatcher and twelve exact real-layer/output checks. Unprofiled20-trial component median153.063235339us control versus152.360334760us repacked: insufficient to beatGU8, no full16 conversion pursued. BatchedSwiGLU component WorkerWatcher exact; buffers2 control144.442014862us versusbatch145.257669501us, slight loss. First buffers3 standalone batch attempt failed host staticL1 check beforelaunch (CBend1140864 > firstbuffer1136384), nohang/reset. Re-ran paired buffers2. Initial component resultfilename collided with runner metadata; printed originalWatcher result remainsinlog; v2 result filenames fixed. Added selective O/GU/down reader-prefix markers to fit all32layers; profile-only CLI.

## 2026-09-23 12:37 UTC — complete phase evidence and compact transport gate

Source42b84308 selective main/O/GU/down profiles eachcovered35ops x4chips x3decodewindows. Prefix/reader audits verify32layers x8readers x4chips x3windows (3072 intervals perselectedzone), nounmatchedmarkers. Same-core median prefixcompletion leads activationread: QKV10.02us,O60.99us,GU21.21us,down10.84us. Prefixdurationmedians5.92/4.65/4.39/4.14us respectively. No sumofworkerintervals orNoCpayload-as-DRAM-counterclaim. MaininstrumentedFWspanchips0/1/3~7.74ms; chip2~29.4ms invalidcrosscoreclockoffset remains. Raw/evidence artifacts/profile-combined-{main,o,gu,down}, audits/prefix-overlap-audit.json.

Implemented opt-in compact BF16 active-row transport for norm-only orallnorm/O/SwiGLU inputs. Restoreszero-padded tiles in existinginputrings, scalarcopies onlyrawBF16words, noarithmeticchange. Paddinginitialized layerzero EVERY invocation, no persistentprefill assumption. Uses existingCB31scratch (prefix128B table preserved), extra4KCB31onlyhead; requires triple aliasedpipeline/scratchall/separateQKV. Compact-all focusedreal0/31 page/inactive/replayWorkerWatcher passed exact. Terminalhead row/order/select bankmapping alsoimplemented; unqualifieduntilfullscreen. Next: artifacts/commands/compact-head-screen128.sh.

## 2026-09-23 12:45 UTC — transport/placement screens and next candidates

Source8440b6c1 five-trial unprofiled128/32 exactscreens: freshcombinedcontrol8.194862324; compactnorm8.252282710 (loss); compactall8.172176805 (22.7us gain); headorder8.248042064 (loss); headselect8.148219804 (46.6usgain). Headselect chooses16of24terminalworkers by minimum summed native NoC0 hopmetric, remaining8normalize. It doesnot move anylayerworker. Allteacher/all64KV/greedyexact. These lastsmallwins needpairedconfirmation andallcontexts.

New unqualified candidates: independentlydeeperQKV buffers/prefix3..8, leavingMLP/head ringsunchanged; and16-row projection tilegeometry withunchanged2048-bytepage stride/full32-rowtensor layouts. Activebatchone row retainsKorder/LoFi orheadHiFi2/partialBF16 arithmetic. Lower16outputrows arezeroedonceperinvocation before anyinputpublish to preserveexternalpadding. Hardwareexactgates required beforefullmeasurement. Commands/scripts artifacts/commands/add-{qkv-prefetch,tiny-projections}.py andnextcomponentgates.

## 2026-09-23 12:50 UTC — QKV8 L1 failure and bounded recovery

Tiny16 MLP passed12exactchecks underWorkerWatcher. IndependentQKV8/prefix8 loop warmup ran, but tracecapture laterallocated morepersistentbuffers and failedhoststaticL1 check: CBregionends1,225,856B, firstlivebuffer1,201,408B (24,448Boverlap). Thisisnot a DRAM/NoCdeadlock; failedcaptureleftteardownstalled. Runner180s timeoutcapturedlivett-triage(all23checks passed) andterminatedonlyitschildgroup at12:49:31. Authorized recovery-qkv8-l1 reset/health/mesh nowrunning. BoundQKVoptions to<=6buffers; fixed focusedharnessfinally-end-capture beforetrace release onhosterrors. Added explicitwatcher/headlineeligibility metadata. Noheadline8-bufferresult.

## 2026-09-23 12:59 UTC — sixteen-row projections improve full decode

Sourcebb4a9cf9 full128/32 withheadselect: tiny16=7.971907549; normal32+QKV4/prefix4=8.143589742; QKV6/prefix6=8.161759612; tiny16+QKV4=7.966220905; tiny16+compactall=7.974274614. Fivewarmunprofiledtrials andexactteacher/all64KV/greedy each. Native7.635744 stillfaster; noend-to-endwin. IndependentQKV6loop andtiny16loop WorkerWatcherbothpassedreal0/31,inactive,replay,pageedges. TinyMLPcomponent141.535025323us vscontrol144.442014862 atbuffers2, twelveexactchecks. Compact transportdoesnotaddtoTiny16 gain; keepoff. QKV4difference~5.7us needsmatchedconfirmation.

Next unqualifiedsource experiments: fullDST synchronization allows14-wideGU and16-wideO/down/head subblocks atsameKorder/rounding; comparehead/mlp independently. Norm16-row geometry andoptional8-tile fullDST subblocks preservecolumnreduction order. Full-loop plusfullheadWatcher gates first.

## 2026-09-23 13:02 UTC — wide projection correctness failure

Source77a81d2f fullDST MLP14/16 subblocks withprojectionTile16 producednonfiniteoutput inreal0/31 loopWatcher atposition127. NoWatcherboundsfault/hang; processclosednormally. ThisvariantisNOTqualified. Savedoriginal /tmp focusedfailuredata underartifacts/full-dst-loop-watcher-evidence beforeothertests. Runnernowassignsunique QB2_MEGAKERNEL_ARTIFACT_DIR forfocusedtests. Norm-only gatescontinueindependently. NeedisolatefullDST32geometry versus16geometrybeforeacceptinganywideprojectionmode.

## 2026-09-23 13:11 UTC — tiny norm win and broader qualification

Source03e3807a (samekernels77a81d2f) full128five-trial medians: Tiny16projections+headselect+normTile16=7.939418936; +normfullDST8wide=7.957743162 (loss); norm32+bankVC=7.973184422 (noimprovement over7.971908control). Both tiny/widenormloopWatcherpassedexactreal0/31,page/inactive/replay. FullDST MLP32-row componentalso FAILED (gate/upfinitebutPCC0, hugeincorrectvalues); geometry16isnotsolecause. Evidenceuniqueartifacts/full-dst-mlp32-watcher-evidence. No wideprojectionheadaccepted. Qualifyingtiny16projections+tiny16norm+headselect (QKVdefault3/prefix2,bankVCoff,wideDSToff,compactoff) at2048/8192/long viaartifacts/commands/tiny-qualification.sh.

## 13:24 UTC — attention reassociation gate
Tiny16 projection+norm, headselect qualified exact at2048=8.363490485,8192=9.032629291,128/256=7.976935169ms (five warmed trials each), versus128=7.939418936. Still~0.30–0.35ms behind native. Explicit attention worker/chunk knobs preserve native HiFi4/FP32 and full causal/KV work. Secondary policy declared BEFORE tests: per-step/aggregate teacher and all64 KV PCC>=0.9999, relativeL2<0.01, exact teacher top1 and greedy, stable replay.32workers/chunk128 FAILED focused PCC0.999891698; reject under unchanged policy.16workers/chunk256 passed focused WorkerWatcher, full128/8192 screening nowrunning via attention16-screen.sh. Partial/failure evidence retained.

## 13:33 UTC — worker-count bounds and repaired fullDST
Attention16/chunk256 exact128=7.808543482 (five trials);8192 FAILED numericalpolicy PCC0.990714669,relL2 0.136107370,9.085803323ms. Attention8 exact128=7.894093903,long=7.934810333;2048 FAILED PCC0.999091864,relL2 0.034718525,8.575485163ms. These are context-limited options, not generalreplacements. Headselect alsochangeswhenattentionfreescores, so full-delta iscombineduntilcontrolled.
Applied chunk-wide-projections.py: FullDST14/16 math/copy/pack operations nowissue twoAPIgroups of7/8tiles withinonefullDSTownership interval. Standalone32-row GU8 component WorkerWatcher passes12EXACTchecks, repairingprior numericalfailure. FocusedTiny16full-loopWatcher next; speednotyetestablished. No tolerancechange.

## 13:36 UTC — head own-prefix candidate
RepairedfullDST completeTiny16loopWatcher exact; full1288.082513097ms exact, slowerthan7.939419; keepdisabled. Implemented --head-early-blocks2/3: lastlayerQKVwriter publishesonce to16terminalheadworkers afterRoPE notification; eachheadreader locallyclearsitsdedicatedglobalflag thenfillsitsownexistingweightBring beforewaitingforfinalnorm. Boundedflag0/1 withserialinvocationcontract, noextraweights/copies/L1CBstorage;2blocks8,912,896B/chip prefix,3blocks13,369,344B/chip. Entire refillinsidecurrenttokenlatency. Triggerguardusesabsolutelayerindex, includesinactivewarmup. FullmodelWorkerWatcher exactgate next.

## 13:43 UTC — head results and specialized QKV candidate
Headownprefix fullWorkerWatcher exactpassed. Five-trial unprofiled128 control7.937778647;prefix2=7.931728517;prefix3=7.933573452, allstrict exact. Nominal~6us gainneedspairedconfirmation; no nativewin.
Prepared qkv_custom_mm opt-in usesexistingBlackholecustom_mm LoFi MVMUL, sameKblock16/sixcolumns/no splitaccumulation/BF16 L1partials andsameweights.8-row CBformats retainoriginal2048-byte ring/page extents; readercompactsunpublishedAblock into contiguous512-byte8-rowtiles (customunpacker walksfacescontiguously), writerrestoresexternalnative32-rowlayout beforeRoPE/cachepublication. Non-row-zero activationlanes areknownzero underbatch1normcontract. Explicitnumericqualificationrequired, notassumedexact. FocusedWorkerWatcher next.

## 13:45 UTC — custom QKV failure; local reset experiment
CustomQKV focusedWorkerWatcher FAILED outputPCC0.987372875, no timeout/boundsfault. Keepdisabled; componentisolationneeded toseparateimplementation/layoutfailurefromnumericalassociation. No performanceclaimortolerancerelaxation.
InlineCBreset opt-in reusesexistinglocalendbarrier: allRISCsfinishanddrainengines; otherRISCsresetonlylocalinterfaces whilewaiting, NCRISC resetsstreamregistersonlyafterunchangedcross-coreendbarrier, thenreleaseslocalpeers. Nextcross-corestartbarrier remainsandpreventsnewproducertrafficuntilresetscomplete. RemovesredundantCBresetlocalsync/tensixdrain; bothglobalbarriersretained. No bufferdatawrittenbyreset. Addedhostdescriptorplan exportedoutsidewarmedtrace timing, countsstaticCBaliasesonce andlabels pinnedtensorviewsaspossiblyoverlapping. FocusedWatcher nowrunning. Newplan.py iscommittedimmediatelyafterlaunch; runmetadataoldSHA+patchdoesnotcontainuntrackedplanfile, use thiscommit'splan.py forreproduction.

## 13:49 UTC — inline reset result and plan audit
InitialinlinegatefailedHOSTmetadataquerybecauseCBFormatDescriptor.data_format getterreturnsunboundtt::DataFormat; noresidentbodylaunched. RemovednonessentialdtypequeryandhandledemptycomputeRTargs. RerunfocusedWorkerWatcher exactpassed. Full128inlineCBreset=7.936444870ms, strict exact, versusfresh7.937778647control:~1.3us is notmaterialwithoutpairedconfirmation. Keepoptional. Descriptorplan nowreportsmaxstaticCB991232B/core. audit-kernel-code.py onTiny8192runtime: maxkerneltextBR12000,NC7672,TR0 14788,TR1 13652,TR2 11180B,60uniqueELFs in4residentprograminstances; thesearetextsections, notsummedpeakL1.

## 13:52 UTC — custom QKV layout failure isolated and fixed
Isolatedone-layerQKVsave showedEVERYfirst16columnsofevery32-columntile EXACT, whileallrighthalvesmissingaccumulation(PCC0.817fullQKV). Rootcauseconsistentwith8-rowcopy_blockplacingpartialsecondfaceatDSTrow8whilecustomMVMULexpectsrow16. ChangedONLYAto8-rowformat;Partial/Outretain16-rowgeometryandnativeexternalfaceoffset512, removedoutputconversion. Complete2-layerfocusedWorkerWatcher nowPASSESEXACToutput/KV/inactive/page/replay. Custominputblockcompactionunchanged. Thiswasimplementationlayoutbug, notacceptednumericaldrift. Full128timingnext.
