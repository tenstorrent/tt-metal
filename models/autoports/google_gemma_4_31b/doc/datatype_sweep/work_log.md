# Gemma 4 31B datatype sweep work log

Stage: 08 datatype-sweep
Model: `google/gemma-4-31B` revision `d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3`
Target: four Blackhole P150b devices, `MeshShape(1,4)`, TP4, `FABRIC_1D`
Accuracy gates: top-1 >= 90%, top-5 >= 98%, retained top-100 expectation = 100%
Selection metric: full-60-layer, batch-1, trace-verified teacher-forcing decode t/s/u.

## Starting state

- Stage 07 checkpoint: `727b333b7bf` (`Record Gemma 4 31B Stage 07 checkpoint`).
- Stage 07 selected policy: BFP8/LoFi attention, BFP4/LoFi gate-up and down, BF16 residuals, BF16 prefill CCL, BFP8 decode CCL, BFP8 KV cache, and BF16/HiFi2 LM head/logits.
- Stage 07 readiness: top-1 91/100, top-5 100/100, top-100 100/100; traced teacher forcing 23.15 t/s/u; matched warmed token-out steady 34.182 t/s/u.
- Unrelated dirty state was present before Stage 08: deleted `tt_metal/python_env/requirements-dev.txt`, `.exp_run/`, `fusion_tests/`, and older untracked `doc/full_model/` profiler/triage outputs. These are preserved and excluded from the Stage 08 checkpoint.

## Prompt-format and reference decision

The required readiness workload is the main AIME24 prompt with 100 generated tokens. The exact pinned `GemmaTokenizer` has `chat_template=None`; a direct local probe on 2026-07-15 showed `apply_chat_template` raises `ValueError` because no template is configured. Per `$qualitative-check`, Stage 08 does not invent a chat wrapper or substitute another tokenizer. It uses the exact-checkpoint AIME24 reference:

`models/autoports/google_gemma_4_31b/doc/full_model/readiness_aime24_plain.refpt`

Metadata records prompt length 149, continuation length 100, top-k 100, exact revision, and the absent-template blocker. The Stage 08 baseline is refreshed against this prompt-correct reference.

## Hardware health

Commands were serialized. Watcher and profiler are not combined.

```text
timeout 60 tt-smi -ls --local
```

Result: all four local P150b devices visible.

```text
timeout 180 env LD_LIBRARY_PATH=$PWD/build/lib python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print('MESH_SMOKE_OK')
PY
```

Result: `MESH_SMOKE_OK`. The first import-only attempt omitted `LD_LIBRARY_PATH` and failed before opening a device; the corrected repo-local command passed.

## Runtime precision plumbing

Stage 08 adds a required JSON precision policy path consumed by `build_generator`. Candidate paths can be supplied with `--precision-config` through the Stage 08 harness or `GEMMA4_31B_PRECISION_CONFIG`; after selection, `doc/datatype_sweep/selected_precision_config.json` is loaded automatically whenever callers do not explicitly supply a model config.

Consumed fields:

- attention QKV and output weight dtypes plus separate compute fidelities;
- MLP gate/up and down weight dtypes plus separate compute fidelities;
- layer exceptions;
- embedding-to-stack activation and per-layer residual dtype;
- phase-specific prefill/decode CCL dtype;
- KV-cache dtype;
- LM-head weight dtype and compute fidelity;
- logits dtype and sampling gather-value dtype.

`Gemma4FullModel.precision_runtime_summary()` derives evidence from constructed layers and terminal/sampling config, rather than echoing JSON fields. Candidate JSON retains this runtime summary.

Static verification:

```text
env LD_LIBRARY_PATH=$PWD/build/lib MPLCONFIGDIR=/tmp/mplconfig \
  pytest -q models/autoports/google_gemma_4_31b/tests/test_precision_config.py \
            models/autoports/google_gemma_4_31b/tests/test_full_model_contract.py
```

Result after the two scoped AutoFix repairs: 49 passed with three benign
warnings. All initial policy artifacts resolve; the existing full-model contracts
remain clean. The suite includes runtime-policy propagation, fixed BF16 cache
update/QKV-split boundaries, and exact BFP8 packed-MLP circular-buffer
accounting.

## Baseline refresh

Command:

```text
env LD_LIBRARY_PATH=$PWD/build/lib MPLCONFIGDIR=/tmp/mplconfig timeout 7200 \
  python -m models.autoports.google_gemma_4_31b.tests.run_datatype_sweep_candidate \
  --model-dir /home/odjuricic/.cache/huggingface/hub/models--google--gemma-4-31B/snapshots/d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3 \
  --precision-config models/autoports/google_gemma_4_31b/doc/datatype_sweep/configs/baseline_bfp8attn_bfp4mlp_lofi_bf16lm.json \
  --reference models/autoports/google_gemma_4_31b/doc/full_model/readiness_aime24_plain.refpt \
  --output models/autoports/google_gemma_4_31b/doc/datatype_sweep/candidates/baseline_bfp8attn_bfp4mlp_lofi_bf16lm.json \
  --tensor-cache /tmp/gemma4_31b_full_model_tensor_cache --run-prefill
```

The harness runs full-prefill accuracy, resets state, then runs exactly 100 teacher-forced tokens with `enable_trace=True`. It rejects performance evidence unless model trace replays equal the 99 decode tokens.

Refreshed baseline result: top-1 91%, top-5 100%, top-100 100%, TTFT
749.986 ms, trace-verified teacher-forcing decode 22.740 t/s/u, and
teacher-forcing end-to-end 19.594 t/s/u. The later same-weight group reran the
teacher-forcing portion after the AutoFix changes and recorded 23.139 t/s/u;
the original refreshed prefill result remains attached to the candidate JSON.

## Reduced runtime smokes and AutoFix

Before the expensive 60-layer matrix, each distinct physical weight/cache
family ran a real two-layer (representative sliding layer 0 and full-attention
layer 5), 100-token traced smoke. Reduced scores are not full-model accuracy
evidence and are excluded from Pareto ranking.

The BFP8 activation/residual smoke exposed two hard TTNN input contracts:
`paged_update_cache` and `nlp_create_qkv_heads_decode` accept BF16/FP32, not
packed BFP8 input. `$autofix` preserved BFP8 inter-layer residual storage while
making the QKV matmul emit BF16 directly and defensively normalizing cache
update operands. The repaired smoke completed with 99 trace replays. Exact
diagnosis and repair evidence is under `autofix_activation_dtype/`.

The BFP8 MLP smoke then reproduced an exact packed gate/up L1 overflow:
1,937,280 bytes required versus 1,572,864 available at block width 12.
`$autofix` derived a BFP8-only block width of 6 (1,090,176 bytes, 482,688 bytes
headroom); the passing BFP4 width-12 path is unchanged. Both BFP8+LoFi and
BFP8+HiFi2 smokes then completed. The exact failure and repaired logs are
`smokes/group_f.log`, `smokes/mlp_bfp8_lofi_autofix.log`, and
`smokes/mlp_bfp8_hifi2_autofix.log`.

## Full-model sweep progress

The serialized full-model physical-weight groups completed with trace
verification for every numeric candidate:

| Config | Top-1 | Top-5 | Top-100 | traced decode t/s/u | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `baseline_bfp8attn_bfp4mlp_lofi_bf16lm` | 91% | 100% | 100% | 23.139 | pass |
| `attention_bfp8_hifi2` | 90% | 100% | 100% | 22.728 | pass |
| `mlp_bfp4_hifi2` | 91% | 100% | 100% | 20.313 | pass |
| `decode_ccl_bf16` | 91% | 100% | 100% | 23.988 | pass |
| `residual_activation_bfp8` | 1% | 1% | 5% | 24.112 | reject: accuracy |
| `lm_head_bfp8_lofi` | 91% | 100% | 100% | 23.471 | pass |
| `lm_head_bfp8_hifi2` | 92% | 100% | 100% | 24.561 | pass; current leader |
| `lm_head_bfp4_lofi` | 90% | 100% | 100% | 23.428 | pass |
| `lm_head_bfp4_hifi2` | 90% | 100% | 100% | 24.488 | pass |
| `attention_qkv_bfp4_lofi` | 89% | 100% | 100% | 23.062 | reject: top-1 |
| `attention_qkv_bfp4_hifi2` | 89% | 100% | 100% | 22.907 | reject: top-1 |
| `attention_output_bfp4_lofi` | 91% | 100% | 100% | 23.176 | pass |
| `attention_output_bfp4_hifi2` | 91% | 100% | 100% | 23.789 | pass |
| `mlp_bfp8_lofi` | 92% | 100% | 100% | 21.868 | pass |
| `mlp_bfp8_hifi2` | 92% | 100% | 100% | 20.064 | pass |
| `kv_cache_bf16` | 91% | 100% | 100% | 22.832 | pass |
| `canonical_accuracy_bfp8_hifi2_bf16commcache` | 93% | 100% | 100% | 17.799 | pass; reject on performance |
| `activation_bfp8_residual_bf16` | 89% | 100% | 100% | 22.983 | reject: top-1 |
| `lm_head_bfp8_hifi2_decode_ccl_bf16` | 91% | 100% | 100% | 22.512 | pass; refined combination regressed |
| `attention_output_bfp8_hifi2` | 90% | 100% | 100% | 22.464 | pass; review-added isolated row regressed |
| `lm_head_bfp8_hifi2_attention_output_bfp4_hifi2` | 91% | 99% | 100% | 22.289 | pass; review-added combination regressed |

The BFP8 residual point is faster than the original passing frontier but fails
decisively; it cannot be selected. BFP8/HiFi2 LM head leads the passing
frontier at 24.561 t/s/u.

BFP4 LM head is gate-safe at exactly the minimum top-1 threshold, but its best
HiFi2 point is still slower and less accurate than BFP8/HiFi2. It therefore
does not trigger a refined combination candidate.

BFP4 QKV is an accuracy rejection at 89% top-1 under both LoFi and HiFi2;
raising compute fidelity does not recover the lost match. The matched LoFi
artifact closes the material BFP4-QKV coverage requirement without a runtime
blocker.

BFP4 attention output passes at 91%/100% under both fidelities. HiFi2 is the
faster of the pair. Independent review correctly rejected the original
component-screening rationale and required the compatible combination with the
selected LM head. That full combination passes at 91%/99%/100% but regresses to
22.289 t/s/u. The review also required isolated BFP8 output/HiFi2; it passes at
90%/100%/100% but regresses to 22.464 t/s/u versus the 23.139 baseline. Its
conditional selected-LM-head combination was therefore not triggered.

The repaired BFP8 MLP candidates both pass at 92%/100%, and their runtime
summaries record the BFP8-only packed block width 6. They are materially slower
than the BFP4 baseline (21.868 LoFi and 20.064 HiFi2), so higher precision does
not trigger a refined combination.

BF16 KV cache constructs successfully at the full advertised context and
passes at 91%/100%, but traced decode falls to 22.832 t/s/u. The capacity row
is valid and no capability reduction is required, but BFP8 remains the faster
eligible cache storage dtype.

The canonical BFP8-MLP/HiFi2/BF16-communication-and-cache control passes with
the best observed top-1 score, 93%, but its 17.799 t/s/u is the slowest numeric
point in the sweep. It is accuracy-safe and full-context runtime-safe, yet
strictly unsuitable for performance selection.

Isolating only the embedding-to-stack activation as BFP8 while preserving BF16
inter-layer residuals avoids the catastrophic all-BFP8 residual failure, but
still reaches only 89% top-1. Because it fails the declared gate and is slower
than the passing frontier, it does not trigger a refined combination.

## KV capacity recomputation

`context_capacity.md` recomputes every evaluated storage choice together with
its active weight policy. The baseline BFP8 row retains 6,552,705,656
bytes/device margin and the physical full-context batch-3 upper bound. The
isolated BF16-KV candidate retains baseline weights; its 5,578,424,320-byte KV
allocation leaves 3,763,493,496 bytes/device at the full 262,144-token context.
Its full-context batch 2 is short 1,814,930,824 bytes/device, but advertised
production capability is batch 1 and does not need reduction.

The canonical control also changes both retained MLP weight placements from
BFP4 to BFP8. Exact payload scaling adds 4,624,220,160 bytes/device, so combining
canonical weights with BF16 KV reaches 35,086,247,304 bytes/device under Stage
07's conservative accounting, 860,726,664 bytes/device above its envelope.
The full Stage 08 run nevertheless constructed the advertised-context cache,
completed 100 tokens with 99 trace replays, and closed normally. This proves
that the retained 12 GiB general reserve is not fully concurrent with this
workload; the accounting result is not a hard physical limit. The policy
therefore retains the 262,144-token batch-1 contract and is rejected on its
measured 17.800 t/s/u performance, not on capacity.

## Candidate matrix

The coarse matrix includes:

- Stage 07 baseline/performance policy;
- canonical accuracy policy and isolated BF16 KV/decode-CCL controls;
- BFP8+LoFi versus BFP8+HiFi2 attention fidelity;
- BFP4+LoFi candidates for QKV and attention output independently;
- BFP4+LoFi baseline MLP plus BFP4+HiFi2 and BFP8+LoFi/HiFi2 controls;
- LM-head BFP8+LoFi, BFP8+HiFi2, and BFP4+LoFi;
- BFP8 activation/residual boundary.
- isolated BFP8 embedding activation with the inter-layer residual retained in BF16.
- refined BFP8/HiFi2 LM-head plus BF16 decode-CCL combination after both
  one-axis candidates independently passed and improved traced decode.
- isolated BFP8 attention-output/HiFi2 with QKV retained at LoFi;
- selected BFP8/HiFi2 LM head combined with BFP4 attention-output/HiFi2.

Every material BFP4 group considered has an explicit LoFi candidate. The
conditional selected-LM-head plus BFP8-output/HiFi2 config remains in `configs/`
for provenance but was not evaluated because its isolated change regressed the
baseline. All evaluated results and rejection reasons are recorded above.

## AutoFix: BFP8 activation/residual cache-update inputs

The first reduced `residual_activation_bfp8` smoke failed during the
non-aligned bounded-sliding prefill tail, before traced decode. The exact log is
`doc/datatype_sweep/smokes/group_a.log`; `paged_update_cache` rejected packed
K/V because its update-input contract is FLOAT32/BFLOAT16 even when the cache
itself is BFP8.

`$autofix` produced
`doc/datatype_sweep/autofix_activation_dtype/AUTODEBUG.md` and `AUTOFIX.md`.
The repair preserves BFP8 embedding/residual storage and BFP8 KV allocation,
but normalizes only bounded-prefill tail and decode K/V update operands to
BF16 before either paged update kernel. The constructed runtime summary now
records this derived update format. Ordinary bulk prefill continues to fill in
the configured cache dtype.

Static verification covered the helper, bounded-prefill ordering, both decode
branches, policy plumbing, and existing full-model contracts:

```text
env LD_LIBRARY_PATH=$PWD/build/lib MPLCONFIGDIR=/tmp/mplconfig pytest -q \
  models/autoports/google_gemma_4_31b/tests/test_cache_update_dtype_contract.py \
  models/autoports/google_gemma_4_31b/tests/test_precision_config.py \
  models/autoports/google_gemma_4_31b/tests/test_full_model_contract.py
```

Result: 46 passed; three unrelated import/deprecation warnings. The Stage 08
device owner reruns the original reduced smoke serially for hardware proof.

The first hardware rerun progressed through the repaired prefill cache update,
then exposed a separate decode prewarm contract failure in
`nlp_create_qkv_heads_decode`; exact evidence is
`doc/datatype_sweep/smokes/group_a_autofix.log`. Decode QKV matmul inherited the
BFP8 inter-layer residual dtype, while the head-split kernel accepts only
FLOAT32/BFLOAT16 tile input. The second isolated repair requests BF16 directly
as the QKV matmul output dtype in both optimized and multichip decode, avoiding
a separate conversion copy and retaining BFP8 only at the inter-layer storage
boundary. The runtime summary records `qkv_split_input_dtype`; downstream K/V
cache normalization is then an identity during decode.

The same static suite passed 47 tests after the second repair, with the same
three unrelated warnings. The serialized hardware owner reruns the reduced
smoke again for end-to-end proof.

The independent `mlp_bfp8_lofi` smoke then failed at packed gate/up decode
matmul compilation; exact evidence is `doc/datatype_sweep/smokes/group_f.log`.
Its BFP8 weight CB at the production 14-core/block-width-12 geometry is
1,645,056 bytes before other buffers, exceeding Blackhole's 1,572,864-byte L1.
The observed complete static CB requirement was 1,937,280 bytes, exactly
reproduced by the factory's BF16 input/intermediate, BFP8 weight/output, and
unreserved-base terms. Width 6 is the largest smaller divisor of the actual 12
K tiles/core and predicts 1,090,176 bytes, leaving 482,688 bytes of headroom.

The third isolated repair dynamically uses packed gate/up block width 6 only
when the active MLP gate/up weight is BFP8. The passing BFP4 path remains width
12; separate gate/up and down paths are unchanged. Constructed runtime evidence
records the effective block width. The focused static suite, now including
`test_mlp_dtype_geometry.py`, passed 49 tests with three unrelated warnings.
The Stage 08 device owner reruns group F serially for hardware proof.

## AutoFix: precision-policy consumption audit

A source audit found that the shared single-device `OptimizedDecoder` loader
used `attention_weight_dtype` for decode QKV and output tensors and one shared
attention fidelity, ignoring the separate resolved policy fields. The repair
keeps the intentionally shared prefill QKV/output dtype, but loads single-device
decode QKV and output tensors from their respective resolved dtypes and uses
separate resolved compute configurations.

The measured four-chip `MultichipDecoder` path did not share this defect: its
prefill loader intentionally consumes `attention_prefill`, while its decode
QKV/output tensors and compute configs already consume the separate resolved
fields. Thus no Stage 08 TP4 numeric row is invalidated. The single-device
policies that would require rerun if such evidence existed are exactly:
`attention_bfp8_hifi2`, `attention_qkv_bfp4_lofi`,
`attention_qkv_bfp4_hifi2`, `attention_output_bfp4_lofi`,
`attention_output_bfp4_hifi2`, and
`canonical_accuracy_bfp8_hifi2_bf16commcache`.

The audit also fixed non-default-batch eager sampling to use the selected
sampling dtype. This does not affect current batch-1 rows because they use the
primary sampler and all current policies select FP32. TP4 runtime summaries now
include actual constructed tensor dtypes for prefill/decode attention,
prefill/decode MLP, and LM-head shards, making policy-versus-physical evidence
explicit. Seven standalone AST/source tests passed without importing TTNN; the
exact command is recorded in `autofix_activation_dtype/AUTOFIX.md`.

## Final matrix and selection

All 21 full-model policies completed on the same 149-token non-aligned AIME24
prompt and 100-token reference. Every JSON records 99 model trace replays for
99 decode tokens. The final refined `lm_head_bfp8_hifi2_decode_ccl_bf16`
combination passed at 91%/100%/100% but regressed to 22.512 t/s/u, slower than
both components. Review-added isolated BFP8 output/HiFi2 and selected-LM-head
plus BFP4-output/HiFi2 rows also passed but regressed to 22.464 and 22.289
t/s/u. The exact winner is unchanged.

`lm_head_bfp8_hifi2` is selected because it is the exact maximum traced decode
rate among accuracy-passing rows: 92% top-1, 100% top-5, 100% top-100, and
24.561083170322924 t/s/u. The next passing policy is
`lm_head_bfp4_hifi2` at the exact 90% top-1 floor and 24.48828054650073
t/s/u. The Stage 07 baseline refresh is 23.139082344511007 t/s/u, so selection
improves the ranking metric by 6.15% while improving top-1 from 91% to 92%.

`selected_precision_config.json` expands the winner without inheritance and
includes every weight group, layer exceptions, compute fidelity,
activation/residual, CCL, KV-cache, logits, and sampling assumption. Normal
`build_generator` construction loads this artifact when no explicit model or
precision config is supplied. The selected BFP8 LM head saves 330,301,440
bytes/device versus Stage 07; `context_capacity.md` and
`../context_contract.json` retain 262,144 tokens with a selected-policy margin
of 6,883,007,096 bytes/device.

Aggregation command:

```text
env MPLCONFIGDIR=/tmp/mplconfig python -m \
  models.autoports.google_gemma_4_31b.tests.generate_datatype_sweep_artifacts \
  --candidates models/autoports/google_gemma_4_31b/doc/datatype_sweep/candidates \
  --output-dir models/autoports/google_gemma_4_31b/doc/datatype_sweep \
  --selected lm_head_bfp8_hifi2
```

Result: 21 rows in both `sweep_results.json` and `sweep_results.csv`; selected
row matches the fastest passing row. Both pyplot charts are 2,540 x 1,455 PNGs
and include all evaluated points, a global Pareto frontier, a red selected
star, and the vertical dotted minimum-accuracy line. Both final PNGs were
visually inspected; the decision zoom and explicit outlier overview are
legible.

## Final static verification

The first expanded run found one test-fixture regression after eager sampling
began consuming `model.config.sampling_dtype`: the dummy model lacked config.
`$autofix` added a non-default BF16 dtype and an assertion that it reaches
`Sampling1DConfig`. The repaired complete command is:

```text
env LD_LIBRARY_PATH=$PWD/build/lib MPLCONFIGDIR=/tmp/mplconfig pytest -q \
  models/autoports/google_gemma_4_31b/tests/test_cache_update_dtype_contract.py \
  models/autoports/google_gemma_4_31b/tests/test_mlp_dtype_geometry.py \
  models/autoports/google_gemma_4_31b/tests/test_precision_config.py \
  models/autoports/google_gemma_4_31b/tests/test_full_model_contract.py \
  models/autoports/google_gemma_4_31b/tests/test_datatype_sweep_artifacts.py
python models/autoports/google_gemma_4_31b/tests/test_precision_loader_source_contract.py
```

Final result: `57 passed, 3 warnings in 15.54s`, followed by `7 tests ... OK`.
All 22 config inputs resolve, including the documented conditional config; the
21 evaluated result JSONs aggregate cleanly. The
warnings are the existing Pydantic migration and SWIG deprecation warnings.

## Post-selection normal-path token-out

This command intentionally supplies no `GEMMA4_31B_PRECISION_CONFIG` or
explicit precision path. The harness-default loader resolves the selected
artifact into the model config passed to `build_generator`:

```text
env LD_LIBRARY_PATH=$PWD/build/lib MPLCONFIGDIR=/tmp/mplconfig timeout 10800 \
  python -m models.autoports.google_gemma_4_31b.tests.run_full_model_qualitative \
  --model-dir models/autoports/google_gemma_4_31b \
  --hf-model /home/odjuricic/.cache/huggingface/hub/models--google--gemma-4-31B/snapshots/d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3 \
  --prompt-source models/common/readiness_check/vllm_prompts.txt \
  --output-dir models/autoports/google_gemma_4_31b/doc/datatype_sweep/qualitative \
  --benchmark-reference models/autoports/google_gemma_4_31b/doc/full_model/readiness_aime24_plain.refpt \
  --benchmark-tokens 100 \
  --benchmark-output models/autoports/google_gemma_4_31b/doc/datatype_sweep/post_selection_token_out.json \
  --benchmark-warmups 1 --benchmark-repeats 5 --benchmark-only
```

The harness-default path loaded `selected_precision_config.json`. Constructed
runtime evidence reports physical BFP8 attention, physical BFP4 MLP, and physical
BFP8/HiFi2 LM-head tensors, plus the selected BF16 activation/residual,
BF16/BFP8 prefill/decode CCL, BFP8 cache, BF16 logits, and FP32 sampling gather
policy. Five post-warmup medians are: TTFT 479.707 ms, overall token-out decode
24.787 t/s/u, and steady decode 34.256 t/s/u. Every sample has 99 model trace
replays, one final sampled-token readback, and zero full-logit readbacks. This
is recorded separately from teacher forcing and is the number later serving
comparisons should use.

Separately, the source/static construction tests prove that a direct
`build_generator` caller which supplies neither `model_config` nor a precision
path discovers and consumes the same selected artifact.

The sampling fix exposed one static-test fixture regression: its dummy model
had no `config`. The fixture now provides a non-default BF16 `sampling_dtype`
and asserts `_get_eager_sampler` propagates it into `Sampling1DConfig`, which
strengthens rather than weakens the consumption contract. The focused
non-device test passed (`1 passed, 3 warnings in 1.86s`) with
`LD_LIBRARY_PATH=$PWD/build/lib`; the warnings are pre-existing import/migration
deprecations. This test-only repair does not invalidate TP4 numeric results.

## Selected qualitative validation

The normal-path qualitative command likewise supplies no precision override:

```text
env LD_LIBRARY_PATH=$PWD/build/lib MPLCONFIGDIR=/tmp/mplconfig timeout 10800 \
  python -m models.autoports.google_gemma_4_31b.tests.run_full_model_qualitative \
  --model-dir models/autoports/google_gemma_4_31b \
  --hf-model /home/odjuricic/.cache/huggingface/hub/models--google--gemma-4-31B/snapshots/d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3 \
  --prompt-source models/common/readiness_check/vllm_prompts.txt \
  --output-dir models/autoports/google_gemma_4_31b/doc/datatype_sweep/qualitative \
  --max-new-tokens 64
```

Result: six HF/TT records, normal selected-config path, exit zero, and a normal
four-device close. `qualitative_prompt_format.json` proves the physical BFP8/
HiFi2 LM head and exact selected artifact path. The exact tokenizer has no chat
template, so the run correctly records completion mode.

Degeneracy command:

```text
python models/common/readiness_check/check_degenerate_output.py \
  --root models/autoports/google_gemma_4_31b/doc/datatype_sweep/qualitative \
  --scope vllm --missing-artifacts critical \
  --json models/autoports/google_gemma_4_31b/doc/datatype_sweep/qualitative/degenerate_output_check.json
```

Result: exit zero, `No degenerate output detected`. Manual `$qualitative-check`
review also passes. Two prompts match HF exactly, four differ but remain
coherent base-model continuations, and four TT outputs match the passing Stage
07 controls token-for-token. Repetition on prompt 1 is identical in HF; prompt
2's corpus-phrase repetition is unchanged from Stage 07. Exact assessment and
limitations are in `qualitative/verdict.md`.

## AutoFix: Stage-review P2 and generator packaging

The review verified that the original Pareto plots technically contained all
required data, but unconditional labels overlapped and the 1% residual-
activation outlier compressed the other policies into an unreadable strip.
The aggregation generator now computes the global frontier over every row,
then renders a threshold-centered decision zoom beside an explicit overview
containing every exact point. Both panels retain the red selected star and
vertical dotted threshold. Passing/rejected markers differ, and deterministic
collision-spaced callouts cover the selected policy, three closest passing
alternatives, frontier alternatives, and key accuracy/combined-gate failures.
No top-5 point is jittered away from its exact 100% value.

Temporary current-data generation and visual inspection used:

```text
env MPLCONFIGDIR=/tmp/mplconfig python -m \
  models.autoports.google_gemma_4_31b.tests.generate_datatype_sweep_artifacts \
  --candidates models/autoports/google_gemma_4_31b/doc/datatype_sweep/candidates \
  --output-dir /tmp/gemma4_dtype_plot_autofix_20260715 \
  --selected lm_head_bfp8_hifi2 \
  --min-top1 0.90 --min-top5 0.98 --min-top100 1.0
```

Both temporary PNGs were visually inspected. After the review-required rows
completed, the same generator produced the final 21-row 2,540 x 1,455 PNGs.
Those final files were also visually inspected: decision clusters, selected
point, global frontier, thresholds, alternatives, and failures are legible,
and the overview explicitly identifies the excluded 1% outlier.

Focused source-only verification:

```text
env LD_LIBRARY_PATH=$PWD/build/lib MPLCONFIGDIR=/tmp/mplconfig \
  pytest -q models/autoports/google_gemma_4_31b/tests/test_datatype_sweep_artifacts.py
```

Result: `2 passed in 0.93s`. An initial invocation without
`LD_LIBRARY_PATH=$PWD/build/lib` stopped during repository `conftest` import
because `_ttnncpp.so` was not discoverable; it did not collect or execute a
test. No TT device was opened.

The generator was also renamed from the ignored untracked
`build_datatype_sweep_artifacts.py` to the non-ignored
`generate_datatype_sweep_artifacts.py` using `apply_patch`; README/work-log
commands were updated. `git check-ignore -v` returns no ignore rule for the new
path, and `git status --short` lists it normally. The old name remains only in
`stage_review.md` as historical review evidence.

## Review-required candidate closure and device health

The first independent review returned `more-work-needed` for missing compatible
attention-output coverage and unreadable plots. The plotting/packaging repair is
recorded above. The two required full-model runs used the same 60-layer,
149+100-token, batch-1, 99-replay command family and exact hardware/reference:

- `attention_output_bfp8_hifi2`: 90%/100%/100%, 856.934 ms TTFT,
  22.463623753965983 traced decode t/s/u, pass;
- `lm_head_bfp8_hifi2_attention_output_bfp4_hifi2`: 91%/99%/100%,
  941.583 ms TTFT, 22.289408789244337 traced decode t/s/u, pass.

Both candidate runtime summaries contain physical prefill/decode tensor dtypes,
including BFP4 for the second row's decode output projection and BFP8/HiFi2 for
its LM head. Both logs record 99 trace replays and normal four-device close.
Because isolated BFP8-output/HiFi2 did not improve the 23.139082344511007
baseline, the review's conditional BFP8-output/HiFi2 plus selected-LM-head row
was not run. The direct compatible BFP4-output combination also regressed, so
`lm_head_bfp8_hifi2` remains the exact fastest passing row.

Post-run health command:

```text
timeout 60 tt-smi -ls --local
```

Result: exit zero; all four local Blackhole P150b devices were available and
resettable. No reset or recovery was needed.

## Source provenance and checkpoint packaging

Each candidate JSON records `git_commit=727b333b7bf0a62cebcd01afcc9ff64c796deffa`,
the immutable Stage 07 parent beneath the live Stage 08 worktree. That field is
not presented as the complete measured Stage 08 source revision. All numeric
groups used the Stage 08 precision/runtime sources documented in this log; the
final Stage 08 checkpoint SHA is appended after the required clean review.

Repository-wide ignore rules cover `*.csv` and `*.log`. The local checkpoint
therefore force-adds only `doc/datatype_sweep/sweep_results.csv` and Stage 08
`doc/datatype_sweep/**/*.log`, together with the normally tracked Stage 08
sources/artifacts. Pre-existing `.exp_run/`, `fusion_tests/`, the deleted
requirements file, and unrelated `doc/full_model/` profiler/triage outputs are
excluded. No push is performed.

## Independent rereview

Fresh `$stage-review` verdict: `clean-pass`; required work: none. The reviewer
independently verified the 21-row aggregates, K/L coverage and conditional-M
rationale, both final Pareto PNGs, selected/default physical consumption,
post-selection token-out, qualitative/context/non-aligned evidence, AutoFix
closure, static tests, packaging plan, and absence of vLLM work. The full report
is `stage_review.md`.

After the verdict, `doc/context_contract.json` was advanced from
`complete_pending_stage_review` to `complete_clean_pass`. Checkpoint branch:
`odjuricic/agentic-research/graph-rewrite-skill`. The Stage 08 source checkpoint
SHA is appended below after the isolated local commit.
