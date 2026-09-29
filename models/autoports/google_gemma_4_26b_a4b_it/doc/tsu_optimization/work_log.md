# Gemma 4 TSU optimization

## Contract and starting provenance

Primary workload: ISL4096, OSL128, concurrency1, four requests. TSU is
`1000 / mean_tpot_ms`. Preserve supported context262144 and concurrency32;
guard short-input, long-context and high-concurrency benchmark rows. Evaluation
repair is out of scope; SWE run36530661132 is left untouched.

Starting TT-Metal: `557b527e096cba271ea270d922cf6f22eb598b54`, branch
`mvasiljevic/gemma4-ttft-opt`. Starting inference-server:
`318c40a5d6f5636d58e11ebd9bbe258c02cf72ce`, branch
`mvasiljevic/gemma4-ttft-monorepo-compat`. Preserve unrelated untracked
`doc/PIPELINE_INTERVENTIONS.md` and root `vllm/`.

Pinned remote image: `ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.22.0-919c110d3d4331b7753c1db78618e879905ae46d-c9cfebcf0490066ff85e1e3fba2c7d456ce5ce42-ttmetal-7f72b1c6e905-109033860866`.
Runtime TT-Metal919c110d, upstream vLLM0.26, plugin snapshot7f72b1c6.

Required optimization, profiler, device-usage, tracing and vLLM integration
skills read. Enabled AutoDebug dependency verified in the host plugin inventory.
Only reduced non-serving paths may collect device profiles. Serving measurement
uses benchmark JSON and opt-in host phase diagnostics.

## Baseline reconciliation and hypotheses

Historical remote4K/C1:23.232ms TPOT,43.04TSU,2095.2ms mean TTFT.
Historical direct local:19.75ms TPOT,50.6TSU. These are not matched baselines:
the local runtime uses compiled TTNN7aee192, and its launcher defaults to async
scheduling; the remote deployment selects synchronous scheduling.

The remote matrix's8K/128 TPOT23.2ms versus8K/1024 TPOT21.1ms indicates a
fixed per-request decode transition cost, in addition to context-dependent work.
Source audit found that `configure_sampling` releases decode traces unless the
short-prefill cache can be reused. That cache supports only1..1024 prompt tokens.
`can_reuse_serving_decode` also requires `prefill_prepared`. Thus long prompts
recapture decode per request. Hypothesis: removing unnecessary compatible-graph
recapture can recover approximately2–3ms amortized TPOT at OSL128 while keeping
the steady-state decode graph unchanged. This is not yet a measured win.

## Environment checks

Four local device nodes exist; no process held them at initial `fuser` check.
Two unrelated TT-XLA containers are running and remain untouched. Docker access
works through `sudo -n`; root's default registry credentials cannot pull the
pinned image, while using the existing user Docker config starts the pull.

## Candidate ledger

| Candidate | Exact config / command | Repeats | TPOT / TSU / TTFT | Correctness | Decision |
|---|---|---:|---|---|---|
| Historical remote | run36490378756,4K/128/C1,4requests | 1 cohort |23.232ms /43.04 /2095.2ms | zero failed requests | baseline only |
| Historical local | prior handoff, older compiled runtime | historical |19.75ms /50.6 /not reconciled | historical | comparison only |
| Compatible decode trace retention | `GEMMA4_EAGER_PREFILL_DECODE_REUSE=1`; exact eager-prefill signature, same live decode trace |31 reduced safety rows; serving pending|unmeasured serving|Watcher, allocation tracker, changed inputs/pages and shape transitions pass|opt-in experiment |
| Exact-image sync baseline | `tsu_benchmark.py --shape 4096 128 1 4 --repeats 2 --warmups 2` |2 cohorts|23.28075/23.23975ms;42.95394/43.02972TSU;2094.968/2093.647ms TTFT|all requests complete|baseline|
| Prefill trace max8192 | same harness; `GEMMA4_PREFILL_TRACE_MAX_LENGTH=8192` |2 cohorts|20.65857/20.86472ms;48.40606/47.92780TSU;2132.063/2094.614ms TTFT|4K texts match; reduced24-case Watcher pass|rejected as default: full8K trace capacity overflow; cold4K capture6111ms|
| Short guard under max8192 | `--shape 128 128 1 4`, otherwise same |2 cohorts|20.21387/20.30079ms;49.47099/49.25916TSU;99.526/98.193ms TTFT|all requests complete|no short-input regression apparent|
| BFP4 DRAM head | `probe_full_lm_head`, commands below |5x20 replays per legal case|device-path host medians: full-width416.18us; chunk32K454.02us; chunk16K462.83us; control389.81/389.94us; no serving claim|all legal cases PCC>=0.999 and recorded exact local top1|rejected: slower or L1-illegal|
| Decode-only reuse, sync | same baseline chat harness, `GEMMA4_EAGER_PREFILL_DECODE_REUSE=1` |2 cohorts|20.84107/20.92596ms;47.98218/47.78754TSU;2143.940/2095.958ms TTFT;4790.756/4753.555ms E2EL|all8 primary texts exactly match baseline|provisional win; guards/async/CI pending|
| Decode-only reuse,8K guard | `--shape 8192 128 1 2` |2 cohorts|20.91322/20.88477ms;47.81663/47.88177TSU;4172.725/4172.235ms TTFT|all4 texts exactly match baseline; full-context1GB trace works|pass|
| Decode-only reuse,short guard | `--shape 128 128 1 4` |2 cohorts|20.30297/20.31132ms;49.25388/49.23362TSU;100.325/98.748ms TTFT|all requests complete|no material regression|
| Sync output/context guards | `--shape 128 1024 1 4 --shape 32768 128 1 2` |1 cohort each|20.56603/21.16456ms;48.62386/47.24879TSU;98.591/16751.022ms TTFT|all6 requests complete|capacity/output guards; matched async comparison pending|
| Sync concurrency guard | `--shape 128 128 32 32` |1 cohort|784.54487ms;1.27462TSU;6088.252ms TTFT;105725.450ms E2EL|all32 requests complete|reduced-N guard, not the final N128 matrix row|
| Decode-only reuse, async | same chat harness, scheduler async |2 primary cohorts|19.54711/19.72699ms;51.15845/50.69196TSU;2146.467/2099.315ms TTFT;4628.951/4604.643ms E2EL|all8 primary texts exactly match original baseline|provisional throughput profile; guards/CI pending|
| Async short guard | `--shape 128 128 1 4` |2 cohorts|19.10598/19.11874ms;52.33963/52.30470TSU;109.241/106.817ms TTFT|all requests complete|8–10ms TTFT tradeoff versus sync, lower E2EL2535.700/2534.897ms|
| Async output/context guards | same guard commands as sync |1 each|19.42082/20.06467ms;51.49113/49.83884TSU;103.712/16765.165ms TTFT|all6 texts and lengths exactly match sync|lower E2EL19971.213/19313.378ms|
| Async concurrency guard | `--shape 128 128 32 32` |1 cohort|680.37688ms;1.46977TSU;14421.257ms TTFT;100829.121ms E2EL|all32 texts/lengths exactly match sync|TTFT/TPOT phase tradeoff; cold capture histories differ|
| Full-model head K4/K2/K1/K4 | `measure_tsu_paths --head-blocks 4 2 1 4 --repeats 5` |5 steady per block,1 warmup excluded|19.641300/19.629814/19.628722/19.639584ms TPOT; generator only|all128 tokens exactly match control|K1 provisional; tiny0.06% full-model win, final serving/qualitative pending|
| Selected default, async, K1 | same primary chat harness; no reuse environment override |2 cohorts|19.53948/19.76070ms;51.17844/50.60549TSU;2144.887/2101.889ms TTFT;4626.400/4611.498ms E2EL|all20 primary/short/8K texts and38 guard texts exactly match controls|local performance reproduced; qualitative/review/CI pending|

### Same-image local baseline

Owned container `gemma4-tsu-experiment` uses the exact remote image, digest
`sha256:686c60b59bb8dc0da80e869981590522fa9920e4078f321a5ffadaa9088beb21`.
Image source is imported unchanged; the local checkout is mounted at
`/workspace/tt-metal` for tools and output only. `USER`/`LOGNAME` are supplied
for the host UID6002. Mesh open/close passed for1x4 Blackhole, firmware19.9.0,
KMD2.10.0. Initial smoke needed writable container-local JIT-cache directories;
the permission failure closed all devices cleanly and the retry passed.

Launch: image Python runs local `tools/ttft_server.py --no-async-scheduling
--output .../readiness_vllm/tsu_optimization/baseline_sync`, with
`GEMMA4_AUTOPORT_TTFT_DIAGNOSTICS=1`. Full context262144/max sequences32,
1GB trace region, FABRIC_1D, on-device sampling, chunked/prefix caching disabled.
No profiler is enabled.

Native comparison command: `tools/ttft_benchmark.py --lengths 4096
--output-len 128 --requests 4 --warmups 2 --output .../baseline_sync/native`.
4K/C1 measured TPOT23.190671677ms=43.12TSU; TTFT median2097.9676ms;
ITL median20.7298ms. Short128/128/C1 measured TPOT20.26887ms and TTFT98.0699ms.
Raw command maps, JSON, logs and full host phase diagnostics are in that folder.
Warmed4K requests each recapture decode for293.35–302.49ms. This fixed cost
adds approximately2.35ms to TPOT across127 intervals, explaining most of the gap.

Remote TTI uses a vLLM0.13 benchmark client, whose request backend explicitly
defaults temperature0. The0.26 image client does not; local comparisons explicitly
set temperature0. The new `tools/tsu_benchmark.py` matches the TTI chat endpoint,
random seed0, exact truncation, headers, lengths and counts; explicit warmups and
client version are recorded separately. It saves command arrays and server info.

First two chat4K cohorts: TPOT23.28075/23.23975ms, TSU42.95394/43.02972,
mean TTFT2094.968/2093.647ms, mean E2EL5051.623/5045.096ms. Native detailed
ITLs identify first-decode gaps317.155/314.640/313.905/316.395ms; subsequent
mean intervals20.801/20.870/20.931/20.881ms. Removing setup must reduce total
request latency too; merely moving this delay into TTFT will not count as a win.

Prepared experiment (not selected): `GEMMA4_PREFILL_TRACE_MAX_LENGTH` overrides
the existing1024-token eligibility bound, with original default preserved.
`tests/check_tsu_prefill_trace.py` compares reduced real layers0/5 against eager
controls, with full262144 cache/page-table shapes, changed token IDs, changed
live page mappings, and repeated exact-length requests. It checks generated
token equality, zero recaptures on reuse and prefill replay counters. Hardware
execution follows baseline-server shutdown; no profiling overlaps serving.

Reduced initial probe passes all8 traced cases at4096/4097. Each sequence of
six generated tokens equals its eager control, including changed tokens and
reversed live page mappings. Repeated eligible requests show0 decode captures
and1 prefill replay. Full262144 context shapes and allocation tracking with
program-cache exclusions disabled were used. Tracking adds substantial host
overhead, so these synchronized probe times are correctness diagnostics only.
Source is the mounted current Python model with the unchanged image's compiled
runtime. Exact command is `check_tsu_prefill_trace.py --lengths 4096 4097
--trace-max-length 8192 --output .../prefill_trace_probe.json`, under
`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`.

Next Watcher probe adds1025/4095/8192/8193, retaining the8193 eager fallback.
`TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1` preserves all worker
assertions; Ethernet instrumentation is excluded due to the documented inherited
ACTIVE_ETH config-buffer limit. No profiler flags are set.

Every selected optimization must be reproduced with exact commands, output
artifacts, trace/input counters, correctness controls and final runtime provenance.

Watcher probe completed successfully: all24 rows at1025/4095/4096/4097/8192/8193
match eager controls. Eligible repeated requests capture neither decode nor
prefill again;8193 preserves the eager fallback. The run exited0, no Watcher
fault was reported, and devices were released before starting the candidate
server. Full JSON and console/Watcher logs are under `readiness_vllm/tsu_optimization`.

The first full-model candidate server uses the identical baseline launch plus
`PYTHONPATH=/workspace/tt-metal:/home/container_app_user/tt-metal` and
`GEMMA4_PREFILL_TRACE_MAX_LENGTH=8192`; compiled runtime, model precision,
context capacity and scheduler are unchanged. This bind-mounted Python run
is exploratory, not final immutable-image evidence.

Launch correction: the first attempt kept the image checkout as working
directory, so Python `-m` resolved its models namespace before PYTHONPATH.
The log correctly exposed `max_length=1024`; no candidate benchmarks were run.
Stopped that owned server, confirmed devices released, and relaunched with
`docker exec --workdir /tmp`. A read-only import probe now resolves the mounted
generator while retaining image TTNN and installed vLLM0.26. Corrected artifacts
use `prefill8192_cwd_sync`, and the failed-attribution launch is retained.

CPU-only trace contracts pass18 tests (including five new limit/capacity cases).
The first pytest invocation could not write its inherited JUnit path; rerun
with `-o addopts=` passed and retained the console log.

Device-audit follow-up: previous LM-head DRAM-sharded trials were BF16/HiFi4;
the later selected BFP4/LoFi geometry sweep covered interleaved weights only.
Extend the existing isolated probe to preserve the selected BFP4 dtype in DRAM
shards and compare against production K-block4. This may change CB feasibility;
it is a new candidate, not a claimed win. No precision change is proposed.

### 8K trace eligibility: exploratory serving result

Identical chat harness, two warmups and four measured4K/128/C1 requests per
cohort, same synchronous scheduler. Candidate cohorts:20.65857/20.86472ms
TPOT =48.40606/47.92780TSU; mean TTFT2132.063/2094.614ms;
mean E2EL4755.702/4744.433ms. Baseline cohorts were23.28075/23.23975ms
TPOT =42.95394/43.02972TSU and5051.623/5045.096ms E2EL. The first candidate
cohort's four exact generated texts equal baseline; remaining comparisons are
pending. Logs confirm one decode capture followed by compatible reuse.

Important cold-shape tradeoff: first4K `_capture` costs6111ms because it also
records the long prefill graph. This cost is excluded by the explicitly recorded
warmups, not hidden in steady TPOT. Arbitrary changing prompt lengths still
invalidate the one-entry exact-length cache. Final qualification must retain
cold/setup evidence and distinguish warmed fixed-shape wins from cold-serving
behavior. No default change has been selected yet.

**8192 candidate rejected:** the full-model8K guard cannot capture within the
existing1GB trace region: `mesh_trace.cpp:82` requests1,032,519,680bytes.
This is a host-side trace-capacity exception, not a device hang. Engine exited,
all device handles were released, and no reset was needed. Preserve full console
and failed benchmark JSON; no8K performance claim is made. The reduced model
fits, demonstrating why full-model capacity qualification is necessary.

Isolated AutoDebug now audits retaining only decode across an already-warmed
exact-shape eager prefill, as standalone `generate` already does. The initial
child sandbox preflight failed because unprivileged namespaces are unavailable;
the inspection-only launcher was retried with its explicit child-sandbox skip
under the existing unrestricted execution authority. Report is pending.

After the serving engine exited, started reduced0/5-layer profiling at4096 input
and full262144 cache/page-table capacity, keeping terminal head, sampling and
token recorder. No serving process or Watcher overlaps this run. Installed
`tt-perf-report==1.3.0` in the owned container Python environment; all dependencies
were already satisfied and unchanged.

The first reduced profile was deliberately interrupted during warmup after
spotting a harness-only recorder-index issue: its extra profiled token would
write beyond the exhausted three-token recording buffer. Reset the existing
index to zero before the signposted replay. The Python child reports
KeyboardInterrupt and closed its devices; its partial report is not performance
evidence even though the Tracy wrapper itself exits0. No invalid recorder replay
was measured. The isolated BFP4 head sweep now runs with profiling disabled;
the corrected reduced profile will follow after it releases the mesh.

Full-width BFP4/LoFi DRAM-sharded head sweep completed: production interleaved
K4 baseline389.813us; only K1/three-reader DRAM candidate fit,416.176us
(6.8% slower). All four devices retain exact local top1 and PCC>0.999999.
Eight other candidates fail exact L1/CB legality checks; no execution fault was
suppressed. Five rounds of20 warmed nonblocking replays per legal case, measured
including input reshard, output conversion, padding slice and concatenation.
Commands use `probe_full_lm_head --weight-dtype bfloat4_b --fidelity LoFi
--baseline-block 4 --chunks 65536 --blocks 1 2 4 --readers 1 2 3 --rounds 5
--replays 20`; JSON records complete geometry, byte estimates and raw repeats.
The first launch lacked a writable runtime log path and exited cleanly before
model execution; the corrected run uses owned `head_runtime`. A follow-up tests
32768/16384 chunks with the same precision and complete-path accounting.

### Decode-only retention and fresh timing reconciliation

AutoDebug's inspection-only report is `AUTODEBUG_decode_reuse.md`; the bounded
AutoFix hypothesis and gates are in `AUTOFIX.md`. Keep only a warmed exact-shape
B1 greedy decode trace across eager prefill, without capturing the long prefill
graph. Cache object/address/spec, logical prompt length, token/table metadata,
active slot and live trace identity all participate in guards. Different shapes,
replaced caches, resumed requests and unsupported sampling invalidate reuse.
Request-local prefill outputs die after the blocking first-token read and before
decode replay. The original1024 prefill trace bound is unchanged by default.

Reduced safety command: `check_tsu_prefill_trace --reuse-eager --lengths 1025
4095 4096 4097 8192 8193 16384`, with Watcher10, worker checks enabled,
`TT_METAL_TRACE_ALLOC_TRACKING=1`, tracebacks enabled and program-cache skipping
disabled. All28 main rows plus three4096→4097→4096 transitions pass: exact
eager-control tokens, changed token values and physical page IDs; stable shapes
retain their trace with zero captures/program-cache growth. Shape transitions
recapture exactly once. No allocation-tracker or Watcher failure; devices close
normally. Times under these diagnostics are not performance measurements.

Full30-layer unprofiled direct paths, same image/full262144 cache/4K input:
queued model+sampler19.625935/19.625688ms =50.95299/50.95363TSU;
buffered generator50.91188/50.91205TSU; per-token host-read generator
50.80728/50.82301TSU. All128 tokens match. These use synthetic repeated input,
not the serving random prompt, so only geometry/configuration is matched.
The direct path performs no per-token scheduler input/page refresh; serving
also handles page growth and response bookkeeping.

Corrected reduced profiler includes layers0/5, terminal head, sampler and token
recorder. Measured device window2548.31us: model trace2016.56us, sampler514.46us,
recorder7.23us, plus gaps. Head381.836us, local TopkLargeIndices279.054us;
SDPA57.556/35.864us. These are reduced-path times, not the full model. Raw CSV
SHA256 `67c35db44628fce8ff3c9f908511996079cca18ad29cfd515f3611b3755fcef1`
and compact tt-perf-report evidence live under `profile_retry`. Roofline advice
is a model, not proof of a bandwidth bottleneck. No profiler ran with serving.

The32K/16K chunked DRAM-head follow-up also loses: best454.022/462.829us versus
389.942us interleaved control. All27 proposed reader/chunk/block combinations
are now measured or rejected by exact allocation legality; no head change kept.

Full serving decode-only candidate launched with synchronous scheduling,
`GEMMA4_EAGER_PREFILL_DECODE_REUSE=1`, diagnostics enabled, and the original
1GB trace region. Mounted Python source takes precedence via working directory
`/tmp` and explicit `PYTHONPATH`; compiled runtime remains the exact image.
Candidate is still opt-in pending full-model performance/correctness qualification.

Matched serving results above now confirm the decode-only hypothesis, with
exact primary and8K texts. The rejected long-prefill environment override was
removed from production; probe-only method substitution preserves reduced
reproduction. Production eligibility remains hard-bounded to1024 tokens.
The post-cleanup CPU contracts pass119 tests. The running sync server predates
this cleanup but used exactly the same1024 bound; its source hashes remain in
`eager_sync/launch.json`. A final restart must reproduce selected source.

Adapter-only changes after sync launch are formatting: cache-spec tuple line
breaks and the `can_reuse_serving_decode` call line break. Reversing these two
changes in memory exactly reproduces the recorded launch adapter SHA256
`366a5440804d33d0f462ab8d9941687b4e83fdf9556df61412ff0fec77306d3c`.
All lifecycle guards were present before the sync launch. The async launch
records the cleaned files' hashes for the next measured cohort.

Qualitative gate: pinned official chat template/revision, six accepted suite
prompts, three greedy repeats each, and six seed71 sampled controls. All18
greedy texts exactly match the previous accepted suite. Prompt0 produces
“Data flows through nodes”; prompt4 retains the correct French translation
“Bonjour, comment allez-vous aujourd'hui ?”; the story and Python responses
remain coherent. Four long sampled controls stop at the configured256-token
limit; this is recorded, not classified as corruption. Full texts, rendered
prompts, token IDs, sampling parameters and server metadata are preserved in
`eager_sync/qualitative.json`. These are correctness checks, not timing cohorts.

Sync guard run finishes normally and the owned API server exits0 after SIGINT;
device-holder checks are empty before the next launch. Async comparison uses
the same source/configuration, only omitting `--no-async-scheduling`, with
the identical chat commands/repeats/warmups. Its startup is not a measured win.
Independent stage-review is checking the bounded decode-retention checkpoint
read-only while these experiments continue; it is not overall task closure.

Async4K/short/8K cohorts and1024-output/32K/C32 guards all complete and match
the corresponding original or sync candidate text/length controls. C32 is not
a clean steady-state speedup claim: sync capture12870.843ms versus async
10744.550ms, with different preceding process histories (sync had qualitative
replays and an earlier4075ms capture). Median ITL681.371→677.349ms is a much
smaller change than mean TPOT784.545→680.377ms; much of the latter moves setup
from ITL into TTFT. Preserve this distinction in final matrix reporting.

Owned async server exits0 and releases all devices before isolated head tests.
The remaining interleaved head sweep uses BFP4/LoFi, full11x10 physical grid,
per-core N19/20/22/24/32 and K blocks2/4/8/11/22/44/88, five rounds of20 replays.
N20 permits subblock4 without shrinking the grid; conversion-inclusive L1-input
controls follow. No production head configuration has changed.

Full-grid tile sweep:35 candidates plus control;32 legal passes and four exact
L1 rejections. Control389.852us, K2/N19 best380.785us; N20/subblock4 does not
beat N19/subblock1. Conversion-inclusive L1 sweep loses to matching DRAM inputs
(K1/N19 L1 best383.041us). Repeated four-grid small-K sweep finds native11x10
K1/N19 best375.951us, K2/N19=381.255us; smaller grids lose. Every legal case
retains all four local top1 winners with PCC extremely close to1. Full-generator
paired K4→K2→K1→K4 controls are still required: a14us isolated saving is only
about0.07% of the complete20ms decode, not a material serving win by itself.

The verified decode-retention behavior is now the source default, with explicit
`GEMMA4_EAGER_PREFILL_DECODE_REUSE=0` preserving fallback. This changes only the
default switch; all measured candidates explicitly enabled identical behavior.
New CPU cases check default/disable selection and the existing disabled-prefill
trace gate. Final default reproduction and exact-image remote qualification
remain outstanding; no final default performance number is claimed yet.

Adapted small-chunk DRAM controls also lose under selected BFP4/LoFi:30 cases
at4096/8192 chunk widths, K8/11/22/44/88 and1/2/3 readers;20 legal passes and
10 exact L1 rejections. Best8192/K11/two-readers483.230us versus390.019us
control. This includes reshard, per-chunk conversions, slice and concatenation;
larger K blocks were not dismissed on the original full-width allocation error.
All device jobs close normally, and no server overlaps these runs.

Started the full30-layer, full262144-cache, unprofiled generator control with
`measure_tsu_paths --head-blocks 4 2 1 4 --repeats 5`. It rebuilds prompt KV,
uses128 advancing decode tokens at4K input, releases before each policy change,
excludes the first run of each block from steady selection, and checks exact
generated token identity. Return-to-K4 controls time-order drift. No default
head change is selected until this run is assessed.

Full-generator paired controls complete: mean TPOT19.641300ms(K4 initial),
19.629814ms(K2),19.628722ms(K1),19.639584ms(K4 return), five steady repeats
each. Every128-token output matches. The11–13us K1 improvement exceeds the
~1.7us control drift but is only0.06% full-model TSU; select K1 provisionally
for final qualitative/serving checks, with no material serving-speed claim.
Raw per-repeat/source-hash evidence is `head_full_paths.json`.

Default-enabled CPU check initially found seven incomplete existing mocks
missing generator constructor state (`host_sampling`/prefill eligibility).
The fixtures now supply their intended eager/short-trace state; production
guards are unchanged. Preserve both failed and retry logs.

Final default/K1 safety reruns:125 CPU contracts pass;31 reduced Watcher
rows pass with source hashes and model/sampler trace identities, including
zero warmed captures/program-cache growth. A separate11-row4K/tile-boundary
Watcher run explicitly enables allocation tracking and tracebacks with program
cache exclusions disabled; it also passes. All devices close normally.
The first31-row final rerun enabled Watcher but not allocation tracking;
do not conflate it with the explicitly tracked11-row rerun or earlier31-row
K4 tracked run. TTI serving/catalog/image CPU suite passes111 tests.
Large completed console logs are committed losslessly as `.log.gz`; original
uncompressed copies remain local and ignored. No evidence is removed.

Short-context reduced capture completes and closes devices before final server
launch. Its offline CSV analysis finishes during server process startup, before
model readiness or requests; no serving request is profiled.128-token SDPA
sliding/global18.271/17.539us versus4K35.864/57.556us. A25/5-layer extrapolation
suggests~0.64ms context cost, consistent with remaining~0.5ms serving gap;
not an all-layer measurement. TopK278.955/279.054us is context-independent.
Final selected server starts with no explicit reuse switch (default1), K1 head,
async scheduling, same full capacity and diagnostics. `selected_async/launch.json`
records exact source hashes; final performance/qualitative checks follow.

Selected-default primary reproduction:51.17844/50.60549TSU, TPOT19.539480/
19.760701ms, TTFT2144.887/2101.889ms, E2EL4626.400/4611.498ms. Short guard
52.40541/52.34933TSU and8K50.66086/50.65048TSU. All20 texts and output lengths
exactly match original image baseline. This reproduces the main gain; the tiny
K1 contribution is not distinguishable within serving noise. Long-output,
32K, C32 and shared qualitative checks are still running/pending.

A new high-concurrency audit identifies explicit row serialization in inherited
`OptimizedDecoder.decode_forward` for batch>1. This is compatible with~680ms
C32 intertoken despite~20ms B1. Native batched attention/router/expert layouts
may provide a separate optimization, but removing the loop is not assumed safe:
indexed TP experts currently assume one logical row. Investigate reduced real
layers after checkpointing verified C1 changes. Overall avenues are not yet
declared exhausted; remote full matrix and this candidate remain outstanding.

Final selected-default guards complete:128/1024 TPOT19.406878ms,51.52812TSU,
TTFT105.425ms,E2EL19958.662ms;32K/128 TPOT20.015878ms,49.96034TSU,
TTFT16781.758ms,E2EL19323.774ms. C32/N32 TPOT680.538670ms,1.46942TSU,
TTFT14577.533ms,E2EL101005.944ms,medianITL677.348ms. All38 texts and output
lengths exactly match the earlier K4 async guards. C32 steady ITL is unchanged;
no high-concurrency device speedup is claimed from K1/default switching.
Official six-prompt qualitative replay is now running separately from timing.

Final selected-default qualitative passes:18/18 greedy outputs exactly match
the pinned official suite and all six seed71 sampled texts/finish reasons
exactly match prior K4 synchronous controls. Four sampled long responses still
hit the configured256-token limit, identical to controls. No new language,
repetition or formatting regression. The owned final server is stopped with
SIGINT only after these completed requests; no device reset is used.

Fresh xhigh independent selected-local review returns `clean-pass` with no
required local work (`review_selected_local.md`). It independently recomputes
50.89035TSU/+18.37%, confirms all58 benchmark/guard outputs and18+6 qualitative
controls, and checks async page-growth draining in the actual installed plugin.
It explicitly does not waive exact-image CI/full23-row matrix or unresolved
batched decode. Stage-owned checkpoint commits/pushes follow under the user's
explicit request; unrelated untracked paths remain excluded.

Publication hygiene: the host pre-commit interpreter path is unavailable, so
hooks run normally in the owned container, without bypassing them. The500KB
per-file rule excludes full warmup-heavy CSV archives; full originals remain
local. Committed decode-window CSVs retain every raw field/signpost and produce
bit-identical whole-window summary objects; manifests include both hashes.
Hooks add final newlines to benchmark JSON and normalize table whitespace.
The new host test adopts the repo's `expect_error` fixture. No runtime source
changes result from these checks; final launch implementation hashes remain valid.
