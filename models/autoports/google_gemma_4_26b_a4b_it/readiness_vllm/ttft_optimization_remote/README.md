# Remote benchmark and evaluation evidence

## Handoff status

Remote qualification produced a complete benchmark sweep, a complete standard
eval, and valid Terminal-Bench and timeout-limited SWE-Bench results. Per user
direction, no further eval repair is planned. Remaining work is
performance-only: close the 4K/C1 decode gap from 43.04 tokens/s/user remotely
to the 50.6 tokens/s/user direct local baseline.

## Published source revisions

- Durable TT-Metal code revision: `mvasiljevic/gemma4-ttft-opt` at
  `f3bfd3c03af2a64cf86fb853dc93c7fa6cfbb073` (the reusable image and
  performance runs remain pinned to `919c110d3d4331b7753c1db78618e879905ae46d`)
- Durable tt-inference-server code revision:
  `mvasiljevic/gemma4-ttft-monorepo-compat` at
  `bfc2ee9bd5bda291f7b58f2f8d978e151b70b3a4` (individual runs below remain
  pinned to their recorded revisions)
- vLLM plugin transport base: `tenstorrent/vllm-tt-plugin` at
  `c9cfebcf0490066ff85e1e3fba2c7d456ce5ce42`
- Embedded plugin source: exact runtime snapshot of `tenstorrent/vllm` at
  `7f72b1c6e905f5137fe3377f2e7b42738d3f271d`

The embedded source is necessary because the reviewed vLLM commit is not
remotely fetchable with the available repository permissions. See
`doc/ttft_optimization/remote_ci_workaround.md` for the transport design and
provenance checks.

## Pre-dispatch verification

- The 23-file snapshot was compared byte-for-byte with the original local
  vLLM commit.
- All 20 Python files in the snapshot compiled successfully.
- tt-inference-server source-build tests passed: 9 focused image-build tests
  and 64 API-server tests (73 total).
- The implementation-scoped benchmark-profile fix passed 47 benchmark-config
  tests plus the focused Gemma autoport serving-contract test.
- The QB2 dispatcher is used from `main`; the runtime implementation comes
  only from the two source revisions above.

## Runs

### Initial benchmark and reusable image build

- Run: https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36453468226
- Dispatcher ref: `main`
- Workflow: `benchmarks`
- Model / implementation: `gemma-4-26B-A4B-it` / `gemma4-autoport`
- Runner / device: `bh-qb-ge` / `p300x2`
- Docker image input: blank, intentionally building the exact source once
- Build job: `109033860866` (successful in 1h 04m 49s)
- Published image:
  `ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.22.0-919c110d3d4331b7753c1db78618e879905ae46d-c9cfebcf0490066ff85e1e3fba2c7d456ce5ce42-ttmetal-7f72b1c6e905-109033860866`
- Runtime manifest digest:
  `sha256:2bec6195411ca5df2088c686507cabd6ea3e58fa45f527ede994ab32bde864b4`
- Verified OCI labels:
  - engine version: `0.26.0+empty`
  - plugin repository / revision: `tenstorrent/vllm-tt-plugin` /
    `c9cfebcf0490066ff85e1e3fba2c7d456ce5ce42`
  - snapshot repository / revision: `tenstorrent/tt-metal` /
    `7f72b1c6e905f5137fe3377f2e7b42738d3f271d`
- Benchmark status: canceled by the operator after 13/23 points completed.

This run was not hung. GitHub did not expose live logs while the hardware job
was running, and the generic benchmark sweep was mistaken for a stall. Once
the canceled job released its buffered logs, they showed 13 successful points
with no request failures. Point 14 spent about 15 minutes preparing/prefilling
the first request, reached `SPLIT_TRACE_READY`, expanded to 28 live requests,
and was generating steadily at 42--45 output tokens/s when cancellation took
effect. The cancellation was therefore an operator error.

The completed results were:

| Point | ISL | OSL | Concurrency | Requests | Duration (s) | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 128 | 1 | 8 | 110.58 | 9.26 | 148.39 | 107.66 |
| 2 | 128 | 128 | 32 | 256 | 1202.10 | 27.26 | 36290.68 | 896.20 |
| 3 | 128 | 1024 | 1 | 4 | 83.81 | 48.87 | 96.72 | 20.39 |
| 4 | 128 | 1024 | 32 | 128 | 2883.42 | 45.46 | 3770.66 | 700.96 |
| 5 | 1024 | 128 | 1 | 4 | 12.63 | 40.55 | 530.15 | 20.68 |
| 6 | 1024 | 128 | 32 | 128 | 464.22 | 35.29 | 16144.95 | 786.68 |
| 7 | 2048 | 128 | 1 | 4 | 15.95 | 32.11 | 1055.90 | 23.08 |
| 8 | 2048 | 128 | 32 | 128 | 530.91 | 30.86 | 32087.85 | 792.42 |
| 9 | 4096 | 128 | 1 | 4 | 20.11 | 25.47 | 2094.25 | 23.09 |
| 10 | 4096 | 128 | 32 | 128 | 663.21 | 24.70 | 63979.51 | 801.73 |
| 11 | 8192 | 128 | 1 | 2 | 14.31 | 17.89 | 4187.73 | 23.35 |
| 12 | 8192 | 128 | 31 | 62 | 459.54 | 17.27 | 124216.72 | 831.09 |
| 13 | 8192 | 1024 | 1 | 2 | 51.71 | 39.60 | 4208.28 | 21.16 |

The required fixed 4K-input results are points 9 and 10 above. They were
successful, but the canceled run could not publish its checkpointed partial
report because artifact upload was skipped after cancellation.

### Benchmark-selection correction and rerun

The initial job selected the repository-wide sweep because `ci-nightly` does
not narrow benchmark configurations and the autoport had no performance
reference rows. The full source diagnosis and fix evidence are in
tt-inference-server `AUTODEBUG.md` and `AUTOFIX.md` at `b27ce82e`.

The fix selects the existing `(4096, 128)` profile only for the exact
Gemma-autoport/P300X2 tuple; normal expansion yields the required concurrency
1 and 32 points. The sibling canonical Gemma implementation retains its
normal sweep.

- Preflight-only failed run:
  https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36480889258
  - Cause: an abbreviated inference-server SHA was passed to `actions/checkout`.
  - No image pull, server launch, or hardware benchmark occurred.
- Corrected rerun:
  https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36481065792
  - Full inference-server SHA supplied.
  - Exact published image above supplied as `docker-image`.
  - All three image-build jobs skipped as intended.
  - Status: workflow success, two cases selected, zero request failures.
  - C1: mean TTFT 2102.07 ms, mean TPOT 82.81 ms, output throughput
    10.14 tok/s.
  - C32: mean TTFT 94660.33 ms, mean TPOT 931.98 ms, output throughput
    16.64 tok/s.

The successful status proves the selection and reporting paths, but these are
not accepted performance numbers. The Docker log shows that `/health` became
ready before the launcher's background trace process finished. That process
continued submitting progressively longer captures, including the 130944-token
case, throughout both benchmark points and created `/tmp/ready` only as C32
ended. The benchmark therefore contended with warmup on the same device. For
comparison, the local C1 baseline is 19.75 ms TPOT (50.6 decode tok/s), and the
canceled run's 4K C1 point was 23.09 ms TPOT (43.3 decode tok/s).

The inference-server follow-up changes Docker workflow readiness to require
both HTTP health and the existing in-container `/tmp/ready` trace-completion
marker. It is host orchestration only and reuses the same immutable image. A
fresh two-point run is required before accepting remote performance.

### Warmup-gated diagnostic and Qwen-style full matrix

- Diagnostic run:
  https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36485728646
  - Inference-server revision:
    `338b8309a1e1cf45ff82e3035bd02ac719a4cfa4`
  - Exact reusable image supplied; all image-build jobs skipped.
  - Status: success, two cases, zero request failures.
  - Server log created `/tmp/ready` at `21:52:18`; the orchestrator released
    server bring-up at `21:52:22`; the first benchmark command began at
    `21:52:22`. No benchmark request overlapped background trace capture.
  - C1: mean TTFT 2094.45 ms, mean/median/P99 TPOT
    23.125/23.12/23.23 ms, output throughput 25.44 tok/s. Mean TPOT is
    43.24 decode tok/s.
  - C32: mean TTFT 64652.85 ms, mean TPOT 808.98 ms, output throughput
    24.47 tok/s.

The C1 result reproduces the canceled run's valid 4K point (43.3 decode tok/s)
and removes the contaminated rerun's 12.1 tok/s regression. It is stable across
all four requests but remains 14.6% below the direct local 50.6 tok/s baseline;
the remote evidence therefore reports both values rather than claiming they are
identical.

At the user's request, the final benchmark coverage now uses the same standard
12 ISL/OSL shapes as the Qwen 3.8 tt-inference-server release. Qwen's explicit
all-user token budget expands those shapes to 24 C1/C16 rows; Gemma's capacity
rules expand them to 23 rows with long-context concurrency caps.

The shapes are `(128,128)`, `(128,1024)`, `(1024,128)`, `(2048,128)`,
`(4096,128)`, `(8192,128)`, `(8192,1024)`, `(10000,1024)`, `(16384,128)`,
`(32768,128)`, `(65536,128)`, and `(131072,128)`, where each pair is
`(input tokens, output tokens)`.

| ISL | OSL | Concurrency / requests |
|---:|---:|:---|
| 128 | 128 | 1 / 8; 32 / 256 |
| 128 | 1024 | 1 / 4; 32 / 128 |
| 1024 | 128 | 1 / 4; 32 / 128 |
| 2048 | 128 | 1 / 4; 32 / 128 |
| 4096 | 128 | 1 / 4; 32 / 128 |
| 8192 | 128 | 1 / 2; 31 / 62 |
| 8192 | 1024 | 1 / 2; 28 / 56 |
| 10000 | 1024 | 1 / 2; 23 / 46 |
| 16384 | 128 | 1 / 2; 15 / 30 |
| 32768 | 128 | 1 / 1; 7 / 7 |
| 65536 | 128 | 1 / 1; 3 / 3 |
| 131072 | 128 | 1 / 1 |

- Full 23-row benchmark:
  https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36490378756
- Separate standard eval:
  https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36490421571
- Separate agentic eval through the branch-only `evals` transport shim:
  https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36490958151
- Benchmark and standard-eval inference-server revision:
  `33b6b94b2d2926fbf38997d244cb86ae7432bd5c`
- Initial agentic shim revision:
  `5ae83388d66b73410d45c27efb67a06d6688b16e`
- Status: the full benchmark completed successfully on the separate QB2 board,
  using the same immutable image. All 23 cases completed with zero failed
  requests. The standard-eval and agentic outcomes are tracked separately
  below.

The server created its trace-readiness marker at `22:26:23`; the first
benchmark command started at `22:26:26`, so none of these measurements overlap
background trace capture.

The completed full-sweep measurements are:

| ISL | OSL | Concurrency | Requests | Mean TTFT (ms) | Mean TPOT (ms) | Output tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 1 | 8 | 97.7 | 20.1 | 48.3 |
| 128 | 128 | 32 | 256 | 4152.0 | 769.9 | 40.2 |
| 128 | 1024 | 1 | 4 | 96.2 | 20.3 | 49.1 |
| 128 | 1024 | 32 | 128 | 3837.9 | 698.6 | 45.6 |
| 1024 | 128 | 1 | 4 | 530.2 | 20.7 | 40.6 |
| 1024 | 128 | 32 | 128 | 16166.5 | 785.9 | 35.3 |
| 2048 | 128 | 1 | 4 | 1053.6 | 22.9 | 32.3 |
| 2048 | 128 | 32 | 128 | 32110.2 | 792.0 | 30.9 |
| 4096 | 128 | 1 | 4 | 2095.2 | 23.2 | 25.4 |
| 4096 | 128 | 32 | 128 | 63993.4 | 800.6 | 24.7 |
| 8192 | 128 | 1 | 2 | 4186.0 | 23.2 | 18.0 |
| 8192 | 128 | 31 | 62 | 124855.0 | 831.7 | 17.2 |
| 8192 | 1024 | 1 | 2 | 4184.9 | 21.1 | 39.7 |
| 8192 | 1024 | 28 | 56 | 112548.7 | 655.9 | 36.6 |
| 10000 | 1024 | 1 | 2 | 5207.6 | 21.1 | 38.2 |
| 10000 | 1024 | 23 | 46 | 114258.2 | 541.5 | 35.2 |
| 16384 | 128 | 1 | 2 | 8414.5 | 23.2 | 11.3 |
| 16384 | 128 | 15 | 30 | 117877.2 | 458.4 | 10.9 |
| 32768 | 128 | 1 | 1 | 16467.2 | 23.5 | 6.6 |
| 32768 | 128 | 7 | 7 | 105686.6 | 318.4 | 6.1 |
| 65536 | 128 | 1 | 1 | 34253.1 | 24.0 | 3.4 |
| 65536 | 128 | 3 | 3 | 82487.0 | 288.9 | 3.2 |
| 131072 | 128 | 1 | 1 | 74153.0 | 24.9 | 1.7 |

At the required 4K/C1 point, `1000 / 23.232 = 43.04` decode tokens/s.
This agrees with the warmup-gated diagnostic (43.24 decode tokens/s) and is
14.9% below the direct local 50.6 decode-tokens/s baseline. The report marks
the rows `NA`, rather than graded pass/fail, because no release performance
targets exist for this experimental implementation; completion and request
failure counts were therefore checked directly in the raw run log.

The QB2 `main` and tt-shield caller workflow enums reject a direct `agentic`
input with HTTP 422. The shim maps the allowed `evals` transport to the
inference server's existing `AgenticWorkflow` only in the pinned revision
above. The metadata flag was removed from the following branch-head commit so
ordinary future `evals` runs keep their normal meaning. No QB2 model-bringup
branch is used.

The initial agentic job waited correctly for background trace completion, then
failed before sending any model request. The `evals` transport had provisioned
the standard-eval environment, while the effective agentic workflow required
`EVALS_AGENTIC/bin/harbor`. Inference-server revision
`14c07a7a7bdd6439285ecb8d42cdebd6b08529f3` resolves the metadata override
during dependency selection and provisions the Harbor environment. The focused
workflow-dispatch suite passes (`120 passed`). The retry is:

- https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36494376361

It is pinned to the same TT-Metal and embedded vLLM revisions and exact image;
image building is skipped again.

That retry completed with a nominal CI success but is not accepted as an
agentic result. Harbor launched both five-task subsets, yet Terminal-Bench was
0/5 with five setup errors because its containers requested 32 CPUs on the
16-CPU runner. SWE-Bench was 0/5 with five connection errors and zero tokens;
its task containers used host loopback from their own Docker namespaces, and
the vLLM log confirms that no requests arrived during the agent phase.

The next inference-server revision ports the exact mini-swe-agent host-gateway
translation already proven by the successful Qwen 3.8 branch, adds its two
regression tests, and caps this Gemma Terminal-Bench config to 16 CPUs. The
focused agentic and routing suites pass (`207 passed`). Its retry is:

- https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36497403867
- Inference-server revision:
  `5c3b2261f6b266d4e7b7b8856de7f0291f92ecc6`

The retry again uses QB2 `main`, the exact reusable image, and the same pinned
TT-Metal/vLLM sources. The temporary metadata flag is removed in branch-head
revision `8903dcc66136d2281673c89af3923ed4676252db`.

This run produced a valid Terminal-Bench 2 result: 2/5 tasks passed (40%),
versus the published 14% reference. `hf-model-inference` and
`portfolio-optimization` passed; `caffe-cifar-10` and `password-recovery`
ended at their agent timeout, and `financial-document-processor` completed
without reward. Across the five trials Harbor recorded 6,836,813 input tokens
and 90,518 output tokens, so this is genuine model traffic rather than an
infrastructure-only score.

Its SWE-Bench report is rejected: all 5/5 trials ended with
`NonZeroAgentExitCodeError`, and Harbor recorded zero input and output tokens.
The earlier host-gateway fix worked—the requests reached vLLM—but vLLM returned
HTTP 400 because `tool_choice=auto` requires both automatic tool choice and a
tool-call parser. Inference-server revision `cbd7b59c` adds
`--enable-auto-tool-choice --tool-call-parser gemma4` to the autoport launch
contract and a regression assertion. A pinned, SWE-only transport commit avoids
re-running the already-valid three-hour Terminal-Bench result:

- https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36515755285
- Pinned inference-server revision:
  `003efcacaa512f8a719eabd2dfb30c97b72e5d46`

The retry uses QB2 `main`, the same immutable image, and skips every image-build
job. Branch head `40d99f1f` restores the normal eval catalog and transport after
dispatch while retaining the durable tool-call launch fix.

This nominally successful run is also rejected as infrastructure-only. The
tool-choice flags activated the intended parser, but the embedded parser was
built against an older vLLM constructor contract: all five clients received
HTTP 400 `Gemma4ToolParser.__init__() takes 2 positional arguments but 3 were
given`. Harbor again recorded zero model tokens.

TT-Metal revision `f3bfd3c03af2a64cf86fb853dc93c7fa6cfbb073`
makes the snapshot parser accept both `(tokenizer)` and
`(tokenizer, model_config)` factories and updates its source manifest. To reuse
the already-built image, inference-server revision `6fdd793a` applies the same
narrow compatibility wrapper from `vllm-tt-metal/src`, which dev-mode CI
bind-mounts over the image launcher. Its regression test recreates the legacy
constructor and instantiates it with the newer two-argument factory contract.
The next SWE-only retry is:

- https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36518073007
- Pinned inference-server revision:
  `997a1268e46556ecc2c12384806acb6455e6412c`

This run deliberately keeps TT-Metal input `919c110d...` because that is the
exact revision inside the immutable reused image; the bind-mounted
inference-server bridge supplies the parser compatibility fix. Branch head
`1dadb353ce1d53e94ccc7c47622b0d66e642bd12` restores the normal eval catalog
after dispatch.

### Evals

The first independent GPQA-Diamond run `36490421571` completed with a nominal
CI success and a reported 57.5% (23/40), but it is not accepted as final: six
prompts exceeded lm-eval's default 30-minute timeout and were recorded as
`__INFERENCE_ERROR__`. Completed-prompt accuracy was 23/34 (67.65%). The raw
outputs otherwise use the intended chat template and contain coherent
thinking/final-answer responses.

Inference-server revision `b7e1491956aa7ab5df54f29576f6f660a80e88a3`
serializes this long-thinking task and raises its client request timeout to two
hours, following the sibling Gemma policy. Its same-image retry is:

- https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36501826660

The retry completed successfully with all 40 samples present, no
`__INFERENCE_ERROR__` sentinels, no partial outputs, and no empty responses.
It scored 62.5% (25/40; reported standard error 7.75 percentage points), versus
the model-card full-set reference of 82.3%. Thirty responses ended in the
requested boxed-answer form. Qualitative inspection found coherent reasoning
and answers in most samples, but also several long repetitive generations that
ran to the 32K output cap; the lower score is therefore reported as a real
quality limitation rather than being hidden by the workflow's `EXPERIMENTAL`
accuracy waiver. Mean wall time was 139.31 seconds per sample.

### Agentic evals

The direct `agentic` dispatch was rejected before creating a run because QB2
`main` does not expose that workflow value. The first `evals` transport attempt
failed at missing Harbor provisioning. Retry `36494376361` provisioned and ran
Harbor but produced only infrastructure errors, as documented above. Corrected
retry `36497403867` produced a valid 40% Terminal-Bench result but an invalid
zero-token SWE-Bench result due to missing server tool-choice flags. SWE-only
retry `36515755285` reached the parser but exposed its constructor mismatch;
SWE-only retry `36518073007` completed successfully with the no-rebuild
compatibility bridge and the same serving image.

The final SWE-Bench Verified result is 1/5 resolved (20%). The reported
1452.04-second value was the concurrent suite wall duration divided by five,
not an actual per-trial mean. This establishes that the full agent, tool-call
parser, vLLM serving, and scorer path is functional: the preceding constructor
failure returned HTTP 400 and zero model tokens, whereas this run generated a
scorable patch that earned reward. It is nevertheless timeout-limited rather
than a clean completion result. Harbor reports `AgentTimeoutError` for all five
trials after a 2h01m wall-clock run; four trials scored zero and one scored one.
The 20% score is therefore retained as observed evidence, with the timeout
condition reported explicitly rather than treating the nominal workflow PASS
or the experimental-model acceptance waiver as proof that every trial
completed normally.

Together, the agentic evidence is 2/5 (40%) on Terminal-Bench 2 and 1/5 (20%)
on SWE-Bench Verified. The former has 6,836,813 input and 90,518 output tokens;
the latter's CI report preserves the scored-trial counts but does not upload
Harbor's per-trial token counters. Neither score is substituted for the Qwen
reference: these are Gemma4 measurements on the same five-task subsets.

A final retry serializes the five SWE trials, writes compact per-trial JSON
summaries into the uploaded workflow logs, and computes mean trial duration
from child start/finish timestamps when available:

- https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36530661132
- Pinned inference-server revision:
  `7babaac11b8a7c9bf17dabf287dac2f4a9264be5`

All image-build jobs were skipped and the immutable image was reused. The run
was left in progress by explicit user direction; its eventual result does not
trigger further eval fixes. Revision
`bfc2ee9bd5bda291f7b58f2f8d978e151b70b3a4` restores normal eval routing.
