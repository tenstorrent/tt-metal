# Resumable experiment checkpoint

Updated 2026-09-22 18:07 UTC. **A complete batch-one, four-chip token-to-logits
program is working at contexts 128 and 2048. It remains slower than baseline.**
Prefill and native traced sampling stay separate. No serving qualification.

Base origin/main: `b8915544692d8f9feb2c890afbc2f22791560cd2`.
Branch: `codex/llama31-qb2-megakernel`. Full-program milestones:
`ead3b217ddb1c0b469569098c204b44b0e41c292` (context128),
`b5d317ddc3d7e419eba6a73e919d89dacd56b1d5` (native scratch reuse/context2048).
The current checkpoint additionally replaces scalar scratch clearing with
NoC zero-seed/doubling copies. Exact transfer SHA is in artifact
`PARENT_CHECKPOINT.md`; parent verified ead3b217 on GitHub at17:21 UTC.
Mark authorized branch pushes only, no PR/posts. Parent transfers bundles and
pushes using existing laptop credentials; do not copy credentials/start OAuth.

## Hardware and environment

Node/job `qb2-120-p03t06` / exclusive Slurm113796, expires19:22:10 UTC.
Stop experiment workloads before19:20; do not extend/release allocations.
Mark permits unlimited serialized device resets during this allocation.
Do not stop another user's workloads. The earlier unrelated device owner exited
by16:57; ownership guard archived. Current-runtime connectivity and ring mesh
passed16:58. A full-program packet-header-pool assertion at17:10 was captured
with tt-triage before stopping only our process. Reset17:12, enumeration of all
four chips, full connectivity and ring mesh opening passed. Fixed pool reuse
before each drained/closed fabric phase. Hardware now healthy, no active guard.

Run root `/home/moconnor/llama-megakernel-113796`, artifacts in `artifacts/`,
mirrored to `/data/moconnor/llama-megakernel-113796`. Matching runtime built with
Clang20.1.8/CMake4.0.2/SFPI7.80.0[956], Python3.12.3, Torch2.11 CPU,
Transformers5.12.1, NumPy<2. Separate `venv-report`: tt-perf-report1.3.0.
System toolchain unchanged. Initial host/runtime build1280actions passed;
native SDPA descriptor binding and profiler metadata changes built successfully.

```bash
source ../artifacts/run-env.sh
export CCACHE_DIR=/home/moconnor/llama-megakernel-113796/cache/ccache
cmake --build build-current --target ttnn test_system_health tracy_profiler_cli_tools --parallel 16
cmake --install build-current --component tar
cmake --install build-current --component tt_pybinds
python ../artifacts/run_device.py --name UNIQUE --timeout 600 -- python -m pytest \
  models/demos/llama31_8b_qb2/tests/test_megakernel_loop.py --timeout=0 -x -s
```

Install BOTH components: `tar` refreshes loaded build-current/lib/libtt_metal.so;
`tt_pybinds` refreshes the repository extension. RPATH patching changes file
hashes, so verify source/installed ELF Build IDs. Device commands run serially
under the bounded runner/lock. On hangs it preserves triage before terminating
only its own child. Kernel changes undergo real SFPI JIT and hardware testing.

## Implementation and qualification

`decode_token` contains embedding, one decoder body repeated32times via a
128-byte-per-layer DRAM weight/KV address table, final all-gather/RMSNorm/head.
Reusable native workspace avoids context2048 prefill L1 conflict. There are86
loop workers plus16 head and8 final-norm workers on the actual11x10 grid.
Current Blackhole supports64 CB indices; attention uses CB32. Current DeepSeek
cross-RISC/64-CB reset helpers and chip barriers protect reusable state. Native
fabric initialization coordinates chips. Native paged SDPA retains32-worker
arithmetic; relocated workers explicitly read query from its producing core.
Sampling owns token/position feedback outside the model program.

Real checkpoint revision `0e9e39f249a16976918f6564b8830bc894c89659`;
selected `gu4_head8_lm_head_hifi2`: GU BFP4; QKV/O/down/head/KV BFP8;
BF16 activations/collectives/residuals, LoFi projections, HiFi4/FP32
norm/RoPE/SDPA, HiFi2 head. No precision relaxation.

Real layer checks cover positions127→129 and255→257, KV writes, physical-page
migration, in-place page-table remapping after trace capture, repeated replay.
One-/two-layer loops select actual layers0/31 with distinct cache allocations;
three inactive position−1 warmups preserve all cache pages. Head alone is exact
for three real embeddings/eight replays. Full32-layer teacher logits and all64
KV tensors are bitwise exact at both tested contexts; all32 greedy tokens match,
and three repeated warmed generations are stable.

| B1,32 outputs/31 decode steps, median of3 trials | Context128 ms/token | Context2048 ms/token |
|---|---:|---:|
| Original traced baseline |8.785510|9.219413|
| First complete prototype |11.153716|11.567906|
| Current NoC scratch clearing |9.221568|9.639827|
| Refreshed installed-runtime baseline |8.788292|9.220867|

Greedy k1/p0/T1/seed42, EOS stopping disabled, matched checkpoint/precision/HF
teacher stream. These are unprofiled host generation timings, not serving.
Artifacts: `model-baseline-128-complete`, `model-baseline-2048-hfmatched`,
`model-decode-token128-zero-all`, `model-decode-token2048-zero-all`.
Earlier partial prototypes were also exact but slower (9.13–9.38ms/context128).

BF16 HF diagnostics are separate: context128 aggregate logitPCC.977533,
relativeL2.208836, teacher top1 agreement100%; context2048 PCC.984128,
relativeL2.172836, top1 agreement93.75%. Same values for TT baseline/prototype.
The provisional .99 HF criterion fails both; these prompts are not broad task
accuracy qualification. Saved HF references and actual outputs remain outside Git.

## Measurements and remaining work

Separate drained profiles have3 complete windows/all4 chips/no missing duration:
refreshed baseline `profile-baseline-current`999ops/device/window, median
kernel sum8.525875ms, first-to-last firmware span9.394897ms on device3/1350MHz.
Initial full `profile-token128-drained`35ops/device/window, kernel sum11.009188ms,
firmware span11.089068ms. One110-core GenericOp is the model; the second1-core
GenericOp belongs to native token recording. Optimized profile pending.
Raw Tracy/device CSV, merged CSV and concise tt-perf-report outputs retained.
Summed firmware durations overlap and are not latency.

Physical JIT telemetry before the last clearing sites: max kernel text50880B,
max total code/config54720B; runtime arguments3072B, semaphore config240B,
CB config528B. Descriptor max local CB845824B/core, head835584B/core;
loop state4096B on86cores. `allocator_after_warmup` in new benchmark results
records allocator-owned L1/DRAM tensors; this excludes static CB/code and is not
peak memory. Context128 L1 allocated223744B/bank, largest free1206656B.

Optimized physical profile `profile-token128-optimized` has3 complete windows
on all four chips:35ops/device/window, median kernel sum9.091968ms and firmware
span9.170225ms. Full384attention-concat samples fall39.836→4.716us. The shorter
profiled firmware span does not establish a speedup: unprofiled generation and
kernel sum remain slower. Optional per-phase markers truncate some late layers.
A separate SumN1 run exported78of86worker totals; an attempted end flush caused
a profiler host marker assertion. Failure/triage retained, process exited on
its own; authorized recovery reset/checks underway at18:07. Direct export to
unused loop-state words480/481 is pending hardware validation and uncommitted.

Qualified focused traffic `traffic-{baseline,mlp}-v3` uses real layer0 weights:
three windows/all four chips, exact numerics, complete operation/request coverage
and independent payload check. DRAM-addressed reads32,112,640B baseline versus
32,113,664B fused (1,024B address-table overhead). Local reads1,146,880→4,390,912B;
local writes1,867,776→262,144B; multicast491,520→0B. No weight-byte reduction.
These are issued API payloads at32B resolution, not DRAM bus/full-model counters.
Host metadata boundary tests and real SFPI/hardware validation pass. The first
pair used a stale installed decoder and is invalid; v2 fused silently dropped
requests. Commit9a65f533 reserves all four event words before flushing and adds
an independent expected-payload gate, which rejects that incomplete v2 capture.
See artifact traffic-comparison.json. Both install components are now refreshed.

Current hardware-qualified optimized code52740ac9 plus profiler fix9a65f533;
parent's last GitHub receipt is ead3b217. Fresh bundle/checkpoint at18:07 contains
these results and refreshed README. Next: finish complete synchronization
counter export, longer full-model position-growth qualification, final source/
report review and durable backup. Keep checkpoint fresh for18:10/19:10 retrieval.
Limits: B1 only, contexts128/2048 tested; no serving/vLLM, non-greedy sampling,
broad accuracy or persistent multi-token loop qualification. Cache allocation
addresses stay fixed while their loop/trace exists; page mappings remain mutable
device inputs. Do not call the prototype a performance improvement.
