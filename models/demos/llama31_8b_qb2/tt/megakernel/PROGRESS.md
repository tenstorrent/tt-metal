# Resumable experiment checkpoint

Updated 2026-09-22 18:46 UTC. **Complete four-chip, batch-one token-to-logits
prototype works at contexts 128, 2048 and 8192; it is slower than baseline.**
Embedding, one decoder body repeated through 32 weight/KV table rows, and final
all-gather/norm/head execute in one mesh program. Prefill and native sampling
remain separate. This is an experimental implementation, not serving support.

## Source and transfer

Base origin/main: `b8915544692d8f9feb2c890afbc2f22791560cd2`.
Branch: `codex/llama31-qb2-megakernel`.
Latest functional source: `dec2ca2f` (complete synchronization counters), preceded
by `52740ac9` (optimized clearing) and `9a65f533` (NoC profiler coverage fix).
Parent verified `03a2955ac1df78dd0388e085e437dc7fba18b327` on GitHub at18:18 UTC.
`artifacts/PARENT_CHECKPOINT.md` and verified `checkpoint-current.bundle` identify
the newest transfer source. Mark authorized this branch push only, no PR/posts.
Parent uses existing laptop authentication; do not copy credentials or start OAuth.

Run root `/home/moconnor/llama-megakernel-113796`; evidence in `artifacts/`,
mirrored to `/data/moconnor/llama-megakernel-113796`. Artifact `REPORT.md` contains
full methods, provenance, measured results and limitations. `WORK_LOG.md` retains
chronology; each bounded hardware command has a JSON record and log. Keep failed
trials, raw profiles and numerical tensors outside Git.

## Hardware and build

Only node `qb2-120-p03t06`, exclusive Slurm113796, expires19:22:10 UTC.
Final checkpoint/stop own sideband workloads by19:15 for parent transfer; no
experiment workload after19:20. Do not extend/release/change allocations.
Mark permits serialized device resets during this reservation. Do not stop
another user's processes. The earlier unrelated owner exited by16:57.

Hardware healthy. Latest authorized reset followed a profiler-only marker
assertion: triage preserved; the child exited itself before cleanup, no process
was signalled. Four-device enumeration, full connectivity and ring mesh opening
passed18:07:35. Normal long-generation and context8192 checks passed afterward.
No active ownership/recovery guard. Device operations must run serially under
the runner/lock; preserve tt-triage before terminating an own hung process.

Matching current-main runtime built with Clang20.1.8, CMake4.0.2, SFPI7.80.0[956],
Python3.12.3, Torch2.11 CPU and Transformers5.12.1. Submodules updated. Separate
venv-report has tt-perf-report1.3.0. System toolchain preserved. Initial1280-action
host/runtime build, native SDPA descriptor binding and profiler C++ changes pass;
new kernels have real SFPI JIT and hardware validation. Latest host build and
both install components passed18:13. Install both components, not just bindings:

```bash
source ../artifacts/run-env.sh
export CCACHE_DIR=/home/moconnor/llama-megakernel-113796/cache/ccache
cmake --build build-current --target ttnn test_system_health tracy_profiler_cli_tools --parallel 16
cmake --install build-current --component tar
cmake --install build-current --component tt_pybinds
python ../artifacts/run_device.py --name UNIQUE --timeout 300 -- python -m \
  models.demos.llama31_8b_qb2.tests.benchmark_megakernel --mode decode_token \
  --context 128 --tokens 32 --repeats 3 \
  --reference ../artifacts/model-baseline-128-complete \
  --hf-reference ../artifacts/hf-reference-128/reference.pt --output /outside/repo/result
```

## Qualified results

Real checkpoint `0e9e39f249a16976918f6564b8830bc894c89659`, unchanged selected
`gu4_head8_lm_head_hifi2`: GU BFP4; QKV/O/down/head/KV BFP8; BF16 activations,
residuals and collectives; LoFi projections, HiFi4/FP32 norm/RoPE/SDPA, HiFi2 head.

Every teacher logit and all64 KV tensors are bitwise exact to the matched TT
baseline. All greedy outputs agree and three warmed generations repeat exactly:

| Context / output tokens | Baseline ms/decode token | Complete prototype |
|---|---:|---:|
|128 /32|8.788292|9.221568|
|2048 /32|9.220867|9.639827|
|8192 /32|9.834495|10.299342|
|128 /256|8.829523|9.263688|

Medians of three unprofiled generations; k1/p0/T1/seed42, EOS stopping disabled.
The first complete128 prototype was11.153716ms; NoC scratch clearing improved
it, but no variant beats baseline. Long generation verifies positions128→383.
Artifacts: `model-baseline{128,2048}-final`, `model-decode-token{128,2048}-zero-all`,
`model-{baseline,token}8192`, `model-{baseline,token}128-long`.

Focused real layer0/31 and one-/two-layer loops verify distinct weight/KV rows,
paged writes, physical-page migration, in-place captured page-table remapping,
positions127→129 and255→257, inactive warmup and repeated replay. Final head
matches three real embeddings/eight replays. Full-layer tests have12checks.
Full-model KV comparisons cover all request-touched pages in all64 caches,
including padded rows; they do not read the entire reserved cache arena. Focused
layer/loop checks also compare every allocated page and unused sentinels.

HF BF16 diagnostics at128/2048 are separate: logit PCC.977533/.984128 and teacher
top1 agreement100%/93.75%, identical for baseline/prototype. The provisional .99
HF threshold fails both; the repeated fixed prompt is not task accuracy evidence.
No HF reference was used for the additional8192/long-generation TT comparisons.

## Measured resources and profiling

Context128 profiles: three complete windows/all four chips/no missing durations.
Baseline999 operations/chip/token; complete prototype35, including one110-core
model GenericOp and34native sampling-boundary operations. Device3/1350MHz:

| Profile | Kernel-duration sum | First-to-last firmware span |
|---|---:|---:|
|baseline-current|8.525875ms|9.394897ms|
|token128-optimized|9.091968ms|9.170225ms|

Profiler span improves while kernel sum and unprofiled generation worsen;
that does not establish a speedup. Summed firmware durations overlap. Raw Tracy,
CSV, coverage and concise tt-perf-report outputs retained. Context8192 matched profiles pass three windows/all four chips: baseline kernel
sum9.539092ms/FWspan10.412064ms; prototype10.155805/10.235210ms. Use16000-op
support at8192; the first4000-support capture lost warmup records and is invalid. No serving benchmark was run.

Physical JIT max kernel text50880B, total code/config54720B, RTA/CRTA3072B,
semaphore240B and CB descriptors528B. Descriptor max static local CB845824B/core;
head835584B; state4096B on86loop workers. Actual grid11x10,64CB indices.
Measured context128 allocator L1: baseline84480B/bank versus prototype223744;
largest contiguous free1345920 versus1206656. These tensor snapshots exclude
static CB/code and are not execution peaks. Full block tables in result.json.

Separate Sum profiling exports all66048 global-barrier intervals:86workers ×
64barriers ×4chips ×3replays. Each interval sum equals its accumulated counter.
Median3.143us, p95215.952us includes worker waiting. Summing minimum worker
intervals per barrier gives median106.597us/token; this is a lower-envelope
statistic, not isolated overhead. Cross-RISC waits/CB reset are outside it.
Artifact `profile-token128-barriers-v4/synchronization-summary.json`; no marker
flush is used. Earlier incomplete/failed profiler experiments are retained.

Focused real layer0 MLP NoC capturesv3: three windows/all four chips, exact
numerics and independent payload coverage. DRAM reads32,112,640B baseline versus
32,113,664 fused including1024B table; local reads1,146,880→4,390,912B, local writes
1,867,776→262,144B. No DRAM weight-byte reduction. These are issued API payloads
at32B resolution, not DRAM bus/full-model counters. Fixed payload encoding and
four-word event reservation; host tests/SFPI/hardware pass. Old stale-decoder and
incomplete captures are invalidated; the expected-payload gate rejects the latter.

## Remaining work and limits

Device/model/build work is finished. Final full connectivity and four-chip
ring mesh open/close passed18:43:59. Complete the final source checkpoint,
REPORT/manifest audit and durable mirror; no more hardware runs are needed.
Keep parent checkpoint current. B1 only; contexts128/2048/8192 and greedy sampling
qualified. No serving/vLLM, non-greedy, broad accuracy or persistent multi-token
qualification. KV allocation addresses stay fixed while bound; page mappings
remain mutable device inputs. No measured whole-model DRAM bus or peak-memory
claim. No performance improvement over the traced baseline is claimed.


Long-running replay limit: the current global barrier accumulates32-bit arrivals
(86 increments per barrier,64 barriers per full token). It can wrap after about
780,000 full-token invocations of one loop instance, including warmup. This was
identified by source review, not stress-tested. A bounded or safely resettable
barrier is required before indefinite serving; this prototype makes no such claim.
