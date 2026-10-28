> Historical experiment ledger. The authoritative final-source result is opt-in
> **36.0909 → 40.6731 TSU (+12.70%)**, full64 native262144 B1 4096/128, all three
> repeats above40. See [the final report](../optimized_full_model/README.md).
> Earlier statements below about unmeasured/open candidates are superseded by
> the dated final closures at the end; they are not current stage blockers.

# Local B1 TSU and TTFT investigation

Current full64 native result: median36.0894292695→37.9550493669 TSU (+5.16944%)
for the default-off compact B1 recurrence candidate. All three fixed-work token
trajectories match exactly, fresh full-model AIME accuracy and shared chat quality
pass, and separate watcher10/profile evidence exists. **Above40 is not achieved.**
Full strict cache/state comparison, remaining optimization, and independent stage
review are still required; this is not a completed or accepted stage.

Primary user objective: full64-layer warmed batch1 decode above40 tokens/s/user.
Use the optimized-full-model stage07 contract, with4096 input and128 output tokens,
three clean repeats, native262144 allocation for headline B1 evidence, on-device
sampling and buffered traced decode. TTFT is measured at caller-visible first token.
No reduced fixture or arithmetic forecast can establish this target.

Base source:206dd3a0ec6f6e1494d4408e6590e47d4bc97ae0. Isolated checkout
`/data/mvasiljev/qwen38-local-tsu-ttft`, branch `mvasiljevic/qwen38-local-tsu-ttft`.
Runtime retained at3fbf3dbf1e88112aae1ca39cff9d1bc8a1da66e8; the local launcher
imports this isolated checkout before runtime TTNN paths. The parent's
`/workspace` and original Granite checkout are excluded from model imports.
Weights are verified local Qwen/Qwen3.8-27B revision
1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0. Raw evidence stays in
`/home/mvasiljev/build/qwen38-swe/local-tsu-ttft`.

Enabled tt-autodebug/model-bringup installations and startup environment validated.
Read optimize, qualitative-check, tracing, tt-device-usage, target-workload and the
actual07 goal. Fresh ownership check found only container sleep infinity, no device
owners. Bounded tt-smi listed all four p300c; isolated1x4 mesh smoke passed. No reset.

## Initial source topology audit

| Path | Material sequence | Candidate and constraints | Status |
|---|---|---|---|
| B1 linear attention | norm, mix, convolution, gate preparation, q/k normalization and expansion, FP32 recurrence, output norm/gate, out projection/all-reduce | Compact head-row I/O around unchanged FP32 compute | Full64 candidate gain measured; exact reduced state/output and full AIME/quality pass |
| B1 full attention | norm, mix, split/RoPE, paged cache update, SDPA, gate, out projection/all-reduce | Retain selected precision and native cache/page-table shapes | Unchanged; reduced layer envelope about374us |
| Both MLPs | norm, packed gate-up, activation/product, down projection/all-reduce | Preserve BFP4/LoFi and residual layout | Coherent same-K2/C80 tiny16 window later passes exact checks but is slightly slower; not selected |
| Terminal | replicated hidden, final norm, generic BF8/HiFi2 head, Sampling1D, device feedback/collector | Same-policy sharded head and exact split greedy controls | Two legal head configurations and pad128 local-topk1 are slower; selected paths retained |
| Generator | eager prefill, fixed inputs, nonblocking model/sampler trace pair,128-token output window | Measure actual host gap before kernel tuning; retain no inner-loop input refresh/readback | Canonical counter checks retained |

B1 already uses in-place FP32 recurrence. Grouped/masked/flat flags do not create
new B1 math paths. The existing grouped benchmark's mixed check requested three
prompts even at batch1; the harness now limits that case to its configured batch.
The reduced profile harness accepts an explicit context allocation for native-B1
shape matching. No runtime source changes have been made.

## Runs

- Initial all64 B1 clean diagnostic:4096in/128out,chunk4096,3repeats,context8192.
  This inherited fixture allocation is **not headline native-context evidence**.
  Canonical native262144 baseline follows before candidate or target claims.
- Full baseline/candidate acceptance must retain AIME prefill and traced teacher
  forcing top5>=98%/top100=100%, shared native-chat suite/control, watcher and strict
  allocation tracking with program-cache allocations included. Use reduced
  one-real-layer-per-kind profiles only, with actionable advice attempted and
  numerical evidence. Independent stage review is required before completion.

Status: work in progress; compact candidate measured below target, with stage
acceptance still pending.

## Measured native B1 baseline

The completed clean baseline uses all64 layers, native262144 allocation,
4096-input/128-output tokens, chunk4096, and three warmed repeats. The same
isolated source and byte-identical NVMe device-source root are required for
matched candidates. Raw receipt: `full64-b1-native-baseline-clean.json`.

| Repeat | TSU | TTFT ms |
|---|---:|---:|
| 1 | 36.0898759688 | 436.9047140 |
| 2 | 36.0894292695 | 438.4893328 |
| 3 | 36.0868408215 | 440.4431870 |

Baseline median36.0894292695 TSU and438.4893328ms TTFT. Each record has127 model and127
sampler trace replays, zero inner-loop readbacks/input refreshes/captures, and one
boundary readback. Trace IDs0/1/2 and mixed replay remain stable/exact. Median
decode step27.7089447ms must fall below25ms to exceed40 TSU. The target remains
unmet; no completed stage is claimed.

## Native-shape reduced profile and decisions

The valid fixed-profiler run uses real layers0/3 plus embedding, final norm,
selected head, sampler and feedback, with native262144 allocation and4096 prefill.
It is diagnostic evidence, not full64 performance. `profile-b1-baseline` retains
per-ASIC complete-envelope CSVs and actual advice; `profile-b1-full-source-proof.json`
closes source attribution including the recurrence kernel. Decode envelope is
2394.742222us,568 operations across four ASICs. Maximum same-ASIC segments:

| Segment | us | Interpretation |
|---|---:|---|
| Embed/RoPE prefix | 48.388889 | Once per step |
| Linear-attention layer | 466.830370 | 48 real model layers |
| Full-attention layer | 374.851852 | 16 real model layers |
| Final norm/head/sampler/feedback | 1509.999259 | Once per step |

Extrapolating those reduced segments gives29.963876ms, already slower than the
actual full64 step27.708945ms; this does not identify an orchestration gap.
The head is931us, approximately3.36% of the actual full64 step, rather than its
misleading38.6% share of the reduced fixture. Sampler top-k is about282us.

Real advice and bounded experiments:

| Hypothesis | Evidence | Decision |
|---|---|---|
| Gate/up DRAM K2→20,80 storage cores | Two paired medians:93.345/93.421us baseline vs97.332/97.194us candidate; per-ASIC PCC0.999848–0.999873 | Slower; reject this configuration |
| Gate/up8 storage cores,K20 |93.839/93.420us vs93.577/93.557us | Flat; no gain established |
| Down K34,one reader | Static L1 end1538688 exceeds first live1068672 | Initial capacity failure; legal two-reader adaptation later measured slower |
| BF8/HiFi2 sharded head32K chunks,K5,two readers | Static L1 end1569920 exceeds first live1121920 | Adapt chunk/K before rejecting family |
| Tiny-height16 gate/up compatibility retile | Reverse wide-retile static allocation exceeds live L1 | Data movement/capacity failure; no precision or full-family rejection |
| Greedy local top-k1 | Width4 rejected alignment; pad32 then hung in final sampling before capture | Root-caused one-tile/two-tile mismatch; pad128 prepared, unverified |

The actual backend permits K blocks that are whole multiples of shard width and
one to three DRAM readers on this Blackhole path. Tileheight16 is supported;
heights below16 are explicitly rejected for known hangs and will not be forced.
New B1 prefill M4096 input-L1/gate-up grid advice is a distinct shape from earlier
B16 M8192 probes and remains actionable. Aggregate fused-core/FLOP estimates
include CCL/reduction and cannot justify a110-compute-core shape by themselves.

## Sampler hang and lease handoff

Raw host stacks and four-device triage preceded terminating only the owned probe.
AutoDebug found width32 passes validation but reader publishes one tile while
compute waits for two; captured CB occupancy agrees on all four ASICs. The
independent inspection agrees. The experimental caller workaround pads to the
known-working final width128 and retains exact token-parity gates; hardware
confirmation is pending. No backend/model change is accepted from this diagnosis.

After evidence preservation: owned process exit143; fresh owners empty; reset0;
post-reset list0; isolated1x4 open/close smoke0 with `MESH_SMOKE_OK`; UMD detached
12:07:04.732UTC. Fresh12:07:44UTC process/device evidence was idle. Parent then
explicitly transferred the exclusive local lease to the memory agent. This
investigation remains CPU-only until an explicit return grant. No live process
from this investigation remains. Quality/tracker/watcher/advice and independent
stage review gates still apply to any eventual selected full64 change.

## Compact recurrence: measured candidate and quality

`QWEN38_COMPACT_B1_RECURRENCE=1` selects compact head-row reader/writer I/O only
for B1 decode with no prepared metadata or active mask. Prefill, masked/prepared
rows and larger batches retain their existing paths. The FP32 HiFi4 compute
kernel, public FP32 q scale/exp, state layout, selected precision and context
remain unchanged. Reader zeroes the matrix-padding rows; writer commits disjoint
head rows and full state. The scalar read uses Blackhole's64-byte DRAM alignment.

The real layer0 probe passes all24 landed-operand checks and all eight complete
output/state checks across four ASICs, before and after replay. Three alternating
100-replay samples per arm give93.511450→60.707894us medians. Watcher10/NOINLINE1
passes the same exact checks with all features enabled and a periodic watcher
scan; all devices detach normally. The earlier watcher120 run is supplementary.

The all64 clean candidate receipt is `full64-b1-native-compact-v1-clean.json`:

| Repeat | TSU | TTFT ms |
|---|---:|---:|
| 1 | 37.9586533019 | 434.0062100 |
| 2 | 37.9550493669 | 433.8579401 |
| 3 | 37.9537084404 | 438.6735968 |

Median37.9550493669 TSU is5.16944% higher than baseline. Median434.0062100ms TTFT
is1.0224% lower, but this decode-only change does not establish a prefill gain.
All three128-token trajectories and mixed outputs exactly match baseline; trace
IDs, three captures and the127+127 replay/no-inner-loop-host-boundary contracts
remain intact. Source hashes match before/after the live run. Another5.38782%
throughput improvement is needed to exceed40; no arithmetic forecast counts as
achievement.

Fresh full64 readiness uses the exact BF16 HF reference described below and
native262144 allocation. Prefill top1/top5/top100 are99/100/100%; teacher-forcing
decode98/100/100%. The separate traced teacher-forcing diagnostic is38.2610 TSU;
it is not the native token-out number. The100-token AIME autoregressive HF/TT
texts are coherent partial reasoning under the same cap; neither reaches its
final math answer in that budget. Shared native-chat512-token controls complete
at EOS: explain133 tokens with three correct English sentences; structured114
with exact4/3 JSON and no fence; code158 with correct order-preserving dedup and
an example. HF controls149/129/239 tokens also finish correctly at EOS. Raw
quality/readiness artifacts are under `full64-b1-compact-quality-v1/`.

## Candidate profiler and source resolution

The valid `profile-b1-compact-v2/` reduced profile and real tt-perf-report advice
use the fixed profiler header,100000 program-support count and disabled live
Tracy pushes. All source/header hashes are bound by
`profile-b1-compact-v2-full-source-proof.json`. Decode has528 device ops across
four ASICs versus568 baseline, removing ten structural ops per ASIC. Complete
maximum per-ASIC decode envelope2394.742222→2367.967407us is a reduced-fixture
measurement, not the full64 gain. Linear-layer envelope466.830370→436.860741us;
full-attention374.851852→374.239259us; terminal1509.999259→1514.092593us.
The derived full64 layer-plus-terminal estimate28.519822ms remains slower than
the actual26.346956ms unprofiled step, so it exposes no avoidable host gap.

The head advice was adapted after capacity failure:16K/R2/K5 and32K/R3/K5 are
legal with selected BF8/HiFi2 but slower, respectively966.918→1150.622us and
966.645→1221.318us including placement/concat/copy. Down K34/R2 is legal but slower
on both real layer kinds, approximately53.9→59.4us. Tiny16 exact DMA conversion
made the previously capacity-limited gate/up probe legal and bit-exact, but its
individual boundary form is slower, about93.6→96.4us. No entire topology family
is rejected from those configurations. Pad128 split-local-topk1 is exact for real
logits and all13 tested boundary/tie controls on all ASICs, but450.628→492.389us
is slower; retain Sampling1D. B1 prefill M4096 direct-L1/grid advice remains open.

Profiler v1 hardware closed normally, but postprocessing rejected missing device
op962560 because its launch omitted the explicit profiler-support options. That
is retained failed setup evidence, not performance/advice evidence. Corrected v2
completed and extracted without that failure. Watcher and profiler were separate.

The compiled backend maps to retained `_ttnncpp.so` SHA256
`afd47e3a3337851b2938ca27e3438ece90d3eb5aa99d78261ef1d9b45b7019d0`.
Relative SDK kernel sources can resolve from CWD before TT_METAL_HOME; native
baseline/candidate both used `/workspace`, while the quality runner's isolated
CWD caused cold JIT identities. Both checkouts have identical clean tracked C++
trees: ttnn `d2f1a480df2967d613a26bba10b3197bfd516237`, tt_metal
`c1cc3ba6605a47f5a3b9095b93e68fb40bc42bc6`. The NVMe byte-copy proof describes
TT_METAL_HOME's copied files and headers, not every CWD-resolved op source.
`kernel-cwd-and-backend-identity.json` records this distinction.

Remaining gates include the full strict pair's five512-entry state maps and12
continuation-logit hashes, final selected-source timing, actionable advice, and
independent review. The B1 strict harness now tests its all-inactive parity with
seven legal inactive decode calls; zero-active prefill continuation is outside
that API contract. It retains all five state checkpoints/logit sets and adds an
explicit unchanged-state assertion. This harness adaptation is not yet run.

The healthy lease returned to the memory agent at generation3, with fresh
13:23:33UTC empty-fuser/only-sleep/all-four-device proof in `LEASE2_HANDOFF.json`.
Further optimization drafts remain outside the active model: four-way independent
V32 recurrence and a coherent tiny16 MLP activation path. They have no hardware
result and count toward no target. No SWE-task speedup is claimed.

## Startup I/O recovery and reference controls

The initial NFS8192-cache diagnostic was intentionally interrupted during bucket2912
preparation, with zero timing records. Python saved its interruption receipt and
UMD detached normally; no reset. Runtime source hashes remained unchanged.
A fresh ownership/list check was clean. This is startup cancellation, not model
correctness or performance evidence.

For future runs,14,013 compiler/device-source files totaling764,413,771 bytes
were copied to NVMe and SHA256-compared against retained3fbf runtime. Symbolic
links match. The compiled TTNN library and Python imports remain original.
Separate caches and a separate fixed-profiler overlay preserve both prior roots.
NVMe mesh smoke passed; compiler workers became CPU-active rather than mostly
NFS-waiting. Native262144 canonical baseline now runs under this verified copy;
all candidate comparisons must use the same root. Byte-proof receipt SHA256:
`f5b257420ad69fe0369a62ffb01473c3ae81673a856271453ee2d2d48165408c`.

A fresh exact-revision BF16 CPU HF AIME24 native-chat reference contains203 prompt
tokens and100 greedy continuation/top100 positions. Its raw-local SHA256 is
`e16f1192147d682eaa91d1957e5d5e39a981dcde106b6364b2255f8cc7d8aa1f`.
The existing text-only loader uses exact text weights from the multimodal model;
no reference comparator is changed. Shared3-prompt HF controls use the native
chat template, default thinking, greedy sampling and512-token cap. Explain149
and structured129 tokens reach EOS with correct three-sentence and4/3 JSON
finals. Code239 tokens also reaches EOS with a correct order-preserving deduplication function and example. All CPU HF control processes are terminal before native timing.

## Combined candidate supersedes serial intermediate

The historical serial-only sections above are retained as experiment history.
The later full64 B1 combined candidate measures TSU 40.676908/40.669768/40.679877
(median 40.676908), versus baseline median 36.089429, on the same native262144
4096/128 workload and timing boundaries. TTFT median 439.126455 ms establishes
no gain against 438.489333 ms baseline. All token trajectories are exact.
See `../optimized_full_model/work_log.md` for current scope and exact component
receipts. Quality/profile evidence for the serial implementation is not reused
as the combined implementation's final acceptance gate. Those gates are pending.
The old statement that V-column parallelism has no hardware result is superseded
by `parallel-v-b1-recurrence-{watcher-v1,clean-v1}.json`; exact head repack and
gate fusion receipts are also retained. No stage pass, local commit or push yet.

## Final coherent tiny16 MLP closure

The complete real0/3 B1 window includes out all-reduce, residual/norm, packed
gate/up, activation/product, down all-reduce and final residual, preserving
BF4/LoFi, BF16, BF8 CCL, the original persistent owner and L1 output contract.
The source-matched C80/K2 gate/up candidate uses16-row tiles through activation
and down, with exact32↔16 NoC conversions only at its external boundaries.
This is a reduced512-context selection fixture, not full64 TSU or native-context
quality evidence. Other-layer/comparison-only tensors reside in DRAM outside
the window; each actual input TensorSpec is restored before measurement.

The first prototype hit a Tiny16 reshape that reset physical storage sizing;
direct producer placement removes that redundant reshape. Subsequent old LLK
init/entry names were adapted to retained startup/mul/kernel_main APIs. Debug
reference tensors initially caused an artificial L1 shortfall; those lifetimes
were reduced before drawing any capacity conclusion. No family is rejected from
these compatibility failures. All devices closed normally after each attempt.

The first legal candidate had exact gate/up but inexact activation. Source
inspection established that selected BinaryNg actually uses SFPU multiply and
UnpackToDestFp32 for BF16 operands; the prototype used FPU/default unpack. The
corrected candidate retains BF16 SiLU staging and the exact selected SFPU/RNE
recipe. `coherent-tiny-mlp-watcher-v8.json` passes full all-device output and
component equality with watcher10/allfeatures/NOINLINE.

`coherent-tiny-mlp-clean-v8.json` then passes before/after300 replays per arm on
all ASICs. Median complete-window latency is193.797350→194.179593us for layer0
and191.883817→192.193408us for layer3. It is not selected because the measured
coherent configuration provides no latency gain. This does not reject every
tiny-tile geometry or a future direct-DST fusion. No unvalidated reduced result
is promoted to a model/TSU claim. The last clean UMD close is16:40:44.963 UTC.

All formal v4 gates, source identities and final accounting are in the canonical
optimized-full-model report. The validated runtime remains unchanged through
this external experiment. Final independent review/local commit are next.
