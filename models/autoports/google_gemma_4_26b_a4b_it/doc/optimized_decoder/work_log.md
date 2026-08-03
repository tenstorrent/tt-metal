# Optimized decoder work log

Date: 2026-07-29 UTC

Functional starting point: `bae72d8aa452c8bcdf2d8d70216de58a1fb32c25`

Scope is restricted to `tt/optimized_decoder.py`, its optimized tests, and
`doc/optimized_decoder`. No multichip decoder, full model, or vLLM work is part
of this stage.

## Baseline and provenance

The clean functional baseline was regenerated from the current checkout in a
detached worktree. The parked WIP `1196bb0e214` supplied only a code/test seed;
none of its mixed-source measurements are stage evidence.

```bash
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
GEMMA4_FUNCTIONAL_DECODER_PERF=1 GEMMA4_RANGE_DOWNLOAD=1 \
pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py::test_functional_decoder_perf_profile \
-k batch1
```

At batch 1 and sequence/current position 1024:

| Layer | Prefill | Traced decode |
| --- | ---: | ---: |
| sliding | 680.885 ms | 3.034 ms |
| full | 681.819 ms | 3.210 ms |

Functional source SHA:
`6fdec7f771d77e14d99aa201a49099c0f00a0abc600ae489e27afccf3df59e2e`.
TTNN extension SHA:
`004ca3464fc6aa2925633bd6b50d6b910ecac170a2c7330587760f42bb0593c0`.

## Operation-topology audit

This audit was recorded before optimized implementation.

| Region | Current topology | Candidates | Action and evidence |
| --- | --- | --- | --- |
| attention input | RMSNorm, packed QKV, head creation, rotary | fidelity/dtype; QKV DRAM sharding | retained packed QKV; role-isolated DRAM trials recorded |
| paged attention | paged update + SDPA decode; composite prefill SDPA | cache dtype; SDPA configs | preserved caller-owned pages and composite ops |
| attention output | concat, O projection, residual/norm boundary | precision-locked O DRAM sharding; coherent residual chain | full BFP8/LoFi O selected; sliding O and R11/R22 rejected |
| dense MLP | repeated same-input gate/up, activation/multiply, down | packed gate/up; precision; per-role DRAM sharding | packed MLP DRAM block 11 selected for both kinds |
| router | router matmul, top-k, softmax, scatter | router fidelity/dtype | kept FP32/BF16 contract; expert-fidelity trials are not mislabeled as router trials |
| decode experts | 3 sparse matmuls/user, GeGLU, weighting, reduction | BFP8/BFP4, block widths, L1 | BFP8/LoFi block 11 and L1 input selected |
| prefill experts | sparse all-expert execution in chunks | dtype, chunk and large configs | BFP8 32-token chunks selected; arbitrary tail supported |
| crossings | head/cache/sparse contract boundaries | coherent sharding; remove host/layout round trips | no host fallback; residual sharding rejected on PCC |

## Implemented actions

- Packed dense gate/up reduces two same-input matmuls to one uploaded weight
  and one matmul, with on-device split.
- Routed expert prefill uses logical 32-token chunks, a large full-chunk sparse
  program, and a distinct tail configuration.
- Expert decode uses block width 11, `per_core_M=1`, `per_core_N=2`, BFP8/LoFi,
  and an L1 input.
- Sliding attention stays BF16/HiFi4; full attention selects BFP8/LoFi.
- Dense MLP selects BFP8/LoFi for both layer kinds.
- Per-role DRAM-sharded decode controls cover `qkv`, `o_proj`, `mlp_gate`,
  `mlp_up`, `packed_mlp_gate_up`, and `mlp_down`, plus independent legal block
  widths. Both layer kinds auto-select packed gate/up + down with block width
  11; full attention also auto-selects its precision-locked O projection.
- Coherent R11/R22 residual/norm candidates keep the chain resident in L1 and
  expose explicit boundary counters, but remain opt-in because PCC misses.
- Low-precision paged prefill bulk fill correctly casts to cache dtype; public
  cache allocation remains caller-owned BF16.

## Precision, layout, and program sweep

Immutable JSONs under `candidate_runs/` contain source/test hashes, resolved
constructor policy, environment overrides, and PCC/timing evidence.

| Family | Evidence | Resolution |
| --- | --- | --- |
| dense MLP BFP4/LoFi, DRAM block 11 | sliding 0.979019/0.988925; full 0.989108/0.991148 | reject |
| dense MLP BFP8/LoFi, DRAM block 11 | passes; sliding 1.259/15.489, full 1.391/15.249 ms | select |
| attention BFP8 | sliding boundary min 0.993506; full passes | full only |
| sliding attention BF16/HiFi2 or LoFi | trace min-user 0.993700/0.948959 | reject |
| full attention BFP8/LoFi | all robustness gates pass; 1.389/15.257 ms | select |
| attention BFP4/LoFi | current cumulative real-weight run misses | reject |
| expert BFP4/LoFi | sliding batch-32 min 0.993908; block-22 retry 0.993876 | reject |
| prefill expert BFP4 | sliding/full boundary min 0.993363/0.993806 | reject |
| KV BFP8 | sliding 0.998631/0.999396; full 0.998337/0.999807 | passes, but not default because caller owns BF16 cache |
| mixed gate/up BFP4, down BFP8 | all real-weight cases miss 0.995 | reject |
| down-only BFP4 | prefill misses; full decode reaches 0.995113 | reject |
| residual R11/R22 on final O+MLP policy | sliding 0.994795/0.994694; full 0.999854/0.999857 | reject: sliding PCC, slower B1, B32 L1 clash |

DRAM-sharded role/geometry matrix:

| Roles | Sliding PCC | Full PCC | Batch-1 / batch-32 decode | Resolution |
| --- | ---: | ---: | --- | --- |
| all dense | not selected | 0.985799 | 1.347/15.546 sliding; 1.392/15.254 full | reject PCC |
| QKV + O | 0.993140 | 0.985759 | isolated correctness | reject |
| QKV | 0.993052 | 0.999816 | full 1.487/15.345 | reject sliding PCC and slower full |
| full QKV + final MLP | — | 0.968413 | current precision-locked cumulative run | reject PCC |
| O, default | 0.999544 | 0.990663 | sliding 1.297/15.505 | reject full PCC; slower than selected sliding MLP |
| O, HiFi4 | — | 0.990762 | retry after failure | reject |
| O, HiFi4 block 1 | — | 0.990705 | second legal retry | reject |
| full O BFP8/LoFi + final MLP | — | 0.999754 | 1.266/15.124 candidate; 1.270/15.148 final | select full |
| packed MLP + down, block 1 | 0.999624 | 0.999812 | sliding 1.312/15.538; full 1.421/15.293 | full candidate |
| packed MLP + down, block 11, HiFi2 | 0.999560 | 0.999783 | sliding 1.289/15.510; full 1.406/15.280 | superseded |
| packed MLP + down, block 11, LoFi | 0.999617 | 0.999746 | sliding 1.259/15.489; full 1.391/15.249 | selected both |
| packed block 22 + down block 11 | passes | passes | sliding 1.261/15.508; full 1.387/15.251 | reject serving regression |
| packed block 22 + down block 3 | passes | passes | sliding 1.265/15.470; full 1.420/15.253 | reject cross-batch result |
| packed block 22 + down block 33 | device L1 clash | device L1 clash | static CB end 1,455,936 overlaps allocation at 1,370,112 | reject hard limit |
| O + packed MLP/down, block 11 | 0.999456 | — | sliding 1.307/15.534 | reject combination as slower |
| O + final LoFi packed MLP/down | 0.999520 | — | sliding 1.274/15.464 | reject; 0.75% primary batch-1 regression |

The attempted packed/down block-4/3 geometry received a precise validation
error because the 22-tile local shard width is not divisible by block 4.
Block 11 was then tested successfully rather than rejecting the family on the
first API error. The later legal block-22/11, block-22/3, and block-22/33
review sweep established that block 11 is the best non-regressing B1/B32
policy; block 33 is API-valid but exceeds device L1. The program-config API
does not expose an independent output-subblock field.

## Correctness and capacity commands

Normal suite:

```bash
GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/final_results.xml
```

Result: `18 passed, 12 skipped in 169.00s`. The gated context, capacity, and
performance probes below were run separately.

Final real-weight PCC is 0.998545/0.999617 sliding and
0.996870/0.999754 full (prefill/decode), for both natural and shared full-cache
views. Batch-32 trace aggregate/min-user is 0.999465/0.998390 sliding and
0.999754/0.999754 full; eager/replay and repeat/replay are 1.0.

Advertised context:

```bash
GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k test_optimized_advertised_context_traced_decode \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/final_context_results.xml
```

Result: `2 passed in 26.35s` at position 262,143.

Non-aligned capacity:

```bash
GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_PREFILL_CAPACITY_LENGTH=262143 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k test_optimized_prefill_capacity_probe \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/final_capacity_results.xml
```

Result: `2 passed in 183.46s`. The 262,144-token context contract is unchanged.

## Performance commands and results

Final batch-1 and serving-batch decode:

```bash
GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUNCTIONAL_DECODER_PERF=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k test_optimized_decoder_perf_profile
```

Serving-batch prefill:

```bash
GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPT_SERVING_PREFILL_PERF=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k test_serving_batch32_prefill_perf
```

| Workload | Functional | Optimized |
| --- | ---: | ---: |
| sliding prefill B1 | 680.885 ms | 95.194 ms |
| full prefill B1 | 681.819 ms | 106.328 ms |
| sliding trace decode B1 | 3.034 ms | 1.272 ms |
| full trace decode B1 | 3.210 ms | 1.270 ms |
| sliding prefill B32 | 21781.406 ms | 2995.823 ms |
| full prefill B32 | 21818.454 ms | 3386.442 ms |
| sliding trace decode B32 | 68.825 ms | 15.474 ms |
| full trace decode B32 | 68.703 ms | 15.148 ms |

Batch 1 uses a sub-tile activation with `per_core_M=1`; serving batch 32 was
measured independently and was not used as a proxy for batch-1 tuning.

## Tracy and same-run accounting

Successful final commands used the exact parametrized nodes; replace
`LAYER_NODE` with either node shown:

```bash
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
GEMMA4_FUNCTIONAL_DECODER_PERF=1 GEMMA4_RANGE_DOWNLOAD=1 \
GEMMA4_OPT_DECODE_DEVICE_PROFILE=1 \
python -m tracy -r -p -v --check-exit-code --dump-device-data-mid-run \
--op-support-count=5000 -o /tmp/gemma4-tracy-final \
-m pytest -q LAYER_NODE
```

- `models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_decoder_perf_profile[blackhole-batch1-sliding_attention_1024-device_params0-mesh_device0]`
- `models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_decoder_perf_profile[blackhole-batch1-full_attention_1024-device_params0-mesh_device0]`

`tt-perf-report` used `OPTIMIZED_DECODE_DEVICE` to
`OPTIMIZED_DECODE_DEVICE_END` for decode and the exact `PERF_PREFILL_*`
signposts for prefill, with `--active-experts 8`.

| Layer | Device-op sum | Same-run traced host | Modeled DRAM roofline |
| --- | ---: | ---: | ---: |
| sliding | 1.196 ms | 1.326 ms | 22.5%, 115 GB/s |
| full | 1.203 ms | 1.324 ms | 19.5%, 100 GB/s |

The device-op sum is the signposted device evidence. Same-run trace replay
adds dispatch/trace/synchronization overhead.

Sliding decode is 23.96% dense matmul, 29.86% norms, 18.44% sparse matmul, and
3.54% SDPA. Full is 18.34% dense matmul (selected width-sharded O plus MLP),
30.43% norms, 18.21% sparse matmul, and 4.44% SDPA. Prefill is 78.31%/79.83%
sparse matmul. Raw Tracy is in
`/tmp/gemma4-tracy-sliding-final-v2-20260729` and
`/tmp/gemma4-tracy-full-final-v2-20260729`; compact CSV/PNG evidence is
stage-owned.

## AutoFix and review remediation

The first independent review returned more-work-needed. The `autofix` skill
ran a fresh AutoDebug investigation (`AUTODEBUG.md`) and implemented/tested
the coherent R11/R22 residual chain. The first R11 QKV attempt exceeded L1 by
25,408 bytes; retrying the other legal K divisor ran end to end. High
accumulation and R22 did not recover the required PCC, so the change remains
an opt-in candidate.

Further remediation added current cumulative precision evidence, batch-32
minimum-user checks before candidate acceptance, per-role DRAM controls,
QKV/O/MLP isolation, multiple legal O retries, legal MLP block-11 geometry,
serving-batch prefill baselines, and same-run device/host accounting.

The stable rereview then requested current-source mixed-precision, larger
geometry, residual, and precision-locked projection trials. Those trials
rejected BFP4 gate/down policies, block-22 variants, QKV, and residual
sharding, while discovering that full-attention BFP8/LoFi O sharding is both
correct and the strongest batch-1 result. It is now part of the selected
runtime and every final gate was regenerated at source hash
`c3097938ef5162426d3f8684a9de9fdc3bccdbe0db51aa583aaeb4a9067fc37c`.
The mixed/geometry/projection artifacts use preceding hash `56781044…`; the
only subsequent runtime change made the already-tested explicit full-O roles
the default (plus its explanatory comment). Exact-source final PCC, B1/B32,
context, capacity, watcher, and Tracy runs therefore revalidate the selected
cumulative path. R11/R22 were rerun after that integration at `c3097938…`.

## Watcher and health

Profiler and watcher were never enabled together.

```bash
TT_METAL_WATCHER=10 GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k 'real_weights_prefill_decode or (traced_decode_batch_contract and batch32) or trace_mutable_stable_buffers' \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/watcher_results.xml
```

Result: `7 passed, 23 deselected in 102.83s`. The 2,171-line device log is
preserved as `final_watcher_device_log.txt` and has no error, assert, hang,
timeout, illegal-NoC, or kill report. Post-run
`/home/mvasiljevic/.ttsmi-venv/bin/tt-smi -s` showed four healthy P300C boards,
DRAM status true, and zero corrected/uncorrected GDDR errors. The exact derived
fields and timestamp are in `post_watcher_health_summary.json`; the earlier
full raw health dump remains in `final_device_health.txt`.

## Optimize checklist

- [x] Fresh functional PCC/performance baseline with source provenance.
- [x] Operation-topology and movement audit before implementation.
- [x] Optimized-path identity/counters; no functional fallback.
- [x] Precision and fidelity sweep with real-weight and robustness gates.
- [x] Layout/program sweeps at batch 1 and serving batch 32.
- [x] Legal batch-1 DRAM-sharded matmuls, including per-role and geometry retries.
- [x] Packed same-input projections.
- [x] Large prefill sparse programs and distinct tail configs.
- [x] Composite SDPA and paged-cache operations with explicit compute policy.
- [x] Arbitrary non-aligned logical sequence lengths.
- [x] Paged-cache semantics, determinism, trace mutability, and layer-kind coverage.
- [x] Full advertised context and non-aligned capacity.
- [x] No Torch conversion, host fallback, redundant reshard, or layout round trip.
- [x] Final Tracy and advice-enabled `tt-perf-report` for both layer kinds.
- [x] Stress/repeated replay and watcher-clean optimized correctness.
- [x] Independent final `stage-review` clean-pass.
- [x] Local stage commits and recorded SHAs.

## Local commits

Implementation, tests, and complete evidence checkpoint:
`b79bcad4c17` (`Add optimized Gemma 4 decoder stage`).

Documentation/checklist finalization:
`cce3dfc2746` (`Record optimized decoder checkpoint`).

The bookkeeping commit that records this second SHA is the final local HEAD
and is reported at handoff.

The persistent `/dev/shm` warning did not abort any gate. All evidence above
was regenerated from base `bae72d8aa452c8bcdf2d8d70216de58a1fb32c25`.
