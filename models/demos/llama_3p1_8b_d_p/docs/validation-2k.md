<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama-3.1-8B-Instruct: 2K prefill validation

Hardware verification completed on 17 September 2026. Independent stage review returned clean-pass with no required work.
This report covers the direct eager prefill model on one Blackhole Galaxy: SP4/TP8, all 32 layers,
two slots, 2,048 input tokens per slot and 1,024-token chunks. BFP8_B is the target cache type;
BF16 is a diagnostic control. Weights and activations use BF16.

## What changed

The model now composes checkpoint loading, embedding, all decoder layers, final RMSNorm and the
vocabulary-sharded head. Token and cache tensors remain caller-owned; numerical forward operations
stay on device. The optional layer observer is a diagnostic enqueue hook, not a migration-ready signal.

The original late-layer 2K failure compared accumulated low-precision states with a separate FP32
trajectory. Independent stock-HF BF16 controls also exceeded intermediate limits while preserving
sampled predictions. The canonical tests therefore enforce each independent reference layer against
the exact native input, plus full-model FP32 logits/token agreement and exact structural checks.
The existing numerical constants were not widened. Accumulated raw hidden/KV misses remain
recorded, including the known BFP8 final-hidden stripe miss. This is a test-contract correction,
not a claim that the raw FP32 mismatch disappeared.

See [the full accuracy contract](../tests/full_model/NATIVE_INPUT_ACCURACY.md) for thresholds,
references, mutation checks, prompt construction and limitations.

## Canonical device results

Each dtype ran boundary/restart, baseline native-input and held-out native-input cases.
The boundary case contains eight calls and retains 1,033/1,537-token tails plus a restart at 1,536.
Each native-input case contains 12 forwards across baseline, capture and replay phases,
with 9,792 local rows covering whole tensors and every 256-token SP stripe.

| Cache | Case | Local rows / misses | Min final-logit PCC | Max final-logit NL2 | Min top-1 / top-5 |
|---|---|---:|---:|---:|---:|
| BF16 | boundary | companion case | 0.992047644 | 0.116787955 | 98.484850% / 100.000000% |
| BF16 | baseline | 9,792 / 0 | 0.995247690 | 0.091010582 | 98.484850% / 100.000000% |
| BF16 | held_out | 9,792 / 0 | 0.995247690 | 0.091010582 | 98.484850% / 100.000000% |
| BFP8 | boundary | companion case | 0.992667528 | 0.114480532 | 100.000000% / 100.000000% |
| BFP8 | baseline | 9,792 / 0 | 0.992667528 | 0.114480532 | 100.000000% / 100.000000% |
| BFP8 | held_out | 9,792 / 0 | 0.992667528 | 0.114480532 | 97.810221% / 100.000000% |

All six cases passed. Both suite actual and verified exits are zero; each JUnit file contains
exactly three passing cases and no skips/errors. All 72 pinned inputs per suite stayed unchanged.
Both device meshes closed cleanly. Exact cache isolation, embedding/weight identity, input retention,
layer order and baseline/captured/replay decoded-value equality passed.

The held-out fixture was registered before execution. Its 2,048-token prefix has 668 unique IDs
and one BOS token. Both runs match the registered source and token IDs. Only slot0 changes;
slot1 retains the original tea fixture. Repeated replay samples are not independent prompts.

## K/V accuracy over all 32 layers

[Per-layer CSV](validation-2k-layers.csv) contains 256 rows: two cache types, two prompt cases,
32 layers and two reference types. It reports hidden/K/V minimum PCC, maximum NL2, comparison
counts and misses. Local extrema cover whole tensors and stripes. Raw-global extrema cover
whole 2K tensors only. Extrema can come from different heads/slots; PCCs are never averaged.

| Cache | Prompt | Tensor | Local min PCC | Local max NL2 |
|---|---|---|---:|---:|
| BF16 | baseline | hidden | 0.999908066 | 0.014348851 |
| BF16 | baseline | k | 0.999993449 | 0.003784667 |
| BF16 | baseline | v | 0.999994110 | 0.003531447 |
| BF16 | held_out | hidden | 0.999908066 | 0.014348851 |
| BF16 | held_out | k | 0.999993520 | 0.003757660 |
| BF16 | held_out | v | 0.999994110 | 0.003531447 |
| BFP8 | baseline | hidden | 0.999856716 | 0.017884472 |
| BFP8 | baseline | k | 0.999917984 | 0.013360051 |
| BFP8 | baseline | v | 0.999934625 | 0.011646022 |
| BFP8 | held_out | hidden | 0.999856716 | 0.017884472 |
| BFP8 | held_out | k | 0.999917984 | 0.013360051 |
| BFP8 | held_out | v | 0.999935394 | 0.011607068 |

### Preserved accumulated-reference observations

| Cache | Prompt | Whole-2K raw hidden/K/V misses | Final-hidden stripe misses across phases |
|---|---|---:|---:|
| BF16 | baseline | 180 | 0 |
| BF16 | held_out | 74 | 0 |
| BFP8 | baseline | 289 | 3 |
| BFP8 | held_out | 157 | 3 |

The known BFP8 stripe is slot1, first chunk, final layer, tokens 0–255:
PCC 0.987577134 and NL2 0.157918870. It misses the old raw-global 0.99/0.15 envelope
and is visible before, during and after capture. Its local layer/cache checks and final logits
pass separately. The original diagnostic failure remains archived, with its original exit status.
Stock-HF BF16 also had 435 V-cache misses in its first-1K control, with all 132 sampled top-1
choices preserved. Neither observation alone proves arbitrary future prompts correct.

CPU integration passed 14 contract/mapping/mutation tests. Earlier staged checks passed 34 host
packing/configuration tests, 32 checkpoint-layer identity cases, and device embedding, head,
one-layer, resource and short 32-layer cases. Tests have explanatory comments before each method.

## Saved performance baseline — no remeasurement

Source: `performance-preparation-002/attempts/attempt-002-bfp8-baseline/captures/report.json`. Verification records actual/dispatch/verified exits 0, no reasons, completion 2026-09-16 22:19:40 UTC. Its historical accuracy status remains provisional/unaccepted.

This is **synchronized eager host-wall timing** on bh-glx-120-b09u02, job107973/step6, all 32 chips, BF16 weights/activations and BFP8_B cache, including final norm/head. It is untraced. No device-profiler or serving-duration claim is made.

One warmup ran per slot, then three measured requests per slot alternating slot0/slot1. Each request completed [0,1024) and [1024,2048) before the next slot started. Thus users and chunks did not execute concurrently. Both outputs stayed allocated until the request finished; readback/hash occurred afterward.

| Slot | Forward median (range), ms | Prompt median (range), ms | Tokens/s/user median (range) |
|---|---:|---:|---:|
| 0 | 430.608 (425.203–432.149) | 432.562 (427.171–434.197) | 4,734.58 (4,716.75–4,794.34) |
| 1 | 429.661 (427.595–430.864) | 431.713 (429.613–434.998) | 4,743.89 (4,708.07–4,767.08) |

| Slot | Chunk | Forward median, ms | With upload/sync median, ms |
|---|---|---:|---:|
| 0 | [0,1024) | 222.458 | 223.418 |
| 0 | [1024,2048) | 208.149 | 209.137 |
| 1 | [0,1024) | 221.425 | 222.455 |
| 1 | [1024,2048) | 207.491 | 209.250 |

Forward wall sums two intervals from completed upload synchronization to completed model-call synchronization. Prompt wall starts before the first upload and ends after the second forward synchronization. Throughput is **2,048 / each request's prompt wall**. Independently calculated medians need not add exactly.

Intervals include host dispatch/synchronization. They exclude tokenization, model/cache load, golden work, logits readback/hash, cleanup and report writes. Tokenization took 1.733 s; model/cache load plus synchronization took 337.405 s. Slot0/slot1 warmups were 177.832/.426 s. The first includes fresh-JIT effects, not isolated compile time. Program-cache entries grew 10→116 in the first warmup, then stayed 116 in measured requests. All 512 output checks were finite and equal to their warmup logits.

Snapshot HEAD was `734a6c29463be5e33a681354a2b039855f094bda` plus pinned Task8 overlay in `repos/tt-metal-perf-b09u02`. All 13 model production-file hashes match canonical frozen production hashes. Standard SDPA header SHA256: `c67e91ad533a1a0ebca805a7e4df4fdd6d3d8f1760291d70c17276f3c6b2e0ec`. The attempt holds the complete 4,670-file pin map. Report SHA256: `27642d239ea1d5b0fecde5d5046eee863dbfb0358ef40d3d8ec6300b941a2f94`.

Six warmed requests support only this fixed 2K workload. They do not establish simultaneous-user throughput, decode speed, startup/network TTFT, kernel speedup, free-generation quality or long-context performance. Policy/comment/test changes do not alter these production bytes. Reconsider measurement for material precision, communication, chunking or kernel changes.

## Reproduction and evidence

Run from the repository root inside a current user-assigned Galaxy allocation, after activating
the matching TTNN/native build environment.
Set LLAMA31_8B_CHECKPOINT to the local Instruct checkpoint. Use a fresh artifact directory.
The test parameters request the 4x8 mesh and FABRIC_1D_RING directly. The site plugin only
sets CPU threads and records collection metadata; it does not change numerical assertions.

~~~bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export LLAMA31_8B_CHECKPOINT=/mnt/models/meta-llama/Llama-3.1-8B-Instruct
export LLAMA_PREFILL_EVIDENCE_DIR=/path/to/fresh/artifacts
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ARCH_NAME=blackhole
export TT_MESH_GRAPH_DESC_PATH="$PWD/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto"
python -m pytest --rootdir=. -c /dev/null -q \
  models/demos/llama_3p1_8b_d_p/tests/full_model/test_prefill_model_vs_ref.py::test_full_prefill_context_boundaries \
  models/demos/llama_3p1_8b_d_p/tests/full_model/test_prefill_native_input_accuracy.py::test_prefill_all_layers_from_native_inputs \
  --junitxml=/path/to/fresh/artifacts/pytest.xml
~~~

Independent review: /data/divanovic/llama31-8b-disagg/notes/task-8-2k-stage-review.md.

The exact site launches, interpreter/native identity, collected IDs, CPU thread settings,
checkpoint/token hashes, JUnit, per-row reports and source pins are archived below:

- Root: /data/divanovic/llama31-8b-disagg/evidence/task-8-full-prefill
- Canonical: canonical-2k-validation-002/{bf16,bfp8}
- CPU integration: canonical-native-input-integration-001
- Registered prompt: held-out-2k-preregistration-001
- Original local/accumulated comparison: full-context-local-correctness-proposal-001
- Stock HF control: hf-bf16-baseline-001
- Performance: performance-preparation-002/attempts/attempt-002-bfp8-baseline

Canonical attempt001 failed during collection because the launcher loaded conftest twice.
Attempt002 removed that redundant flag; model/test/fixture source stayed unchanged.
This launcher failure remains archived and did not execute a numerical device test.

## Scope still pending

This milestone establishes 2K direct-call prefill under the documented test contract.
It does not establish free-running text generation, packed-byte migration, long-context accuracy,
common-runner readiness or SC4 Blaze decode. The common adapter allocation/build-runtime
methods remain scaffolds and need separate integration tests. Layer callbacks do not signal
completed device writes. Longer-context milestones precede migration as requested.
