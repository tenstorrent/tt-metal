# Gemma 4 31B Stage 08 datatype sweep

Stage 08 selects `lm_head_bfp8_hifi2` as the default precision policy for the
repo-local `google/gemma-4-31B` full model on four Blackhole P150b devices. It
is the fastest of 21 evaluated full-60-layer policies that satisfies top-1 >=
90%, top-5 >= 98%, and top-100 = 100% on the pinned 100-token AIME24 readiness
reference. Its ranking result is 92% / 100% / 100% at 24.561 trace-verified
teacher-forcing decode tokens/s/user with 99 model trace replays.

`selected_precision_config.json` is a complete, non-inherited policy artifact.
`build_generator` loads it automatically when callers supply neither an
explicit model config nor a precision path; the qualitative/token-out harness
resolves the same artifact through its default config loader. The selected
policy is BFP8 attention, BFP4 MLP, BFP8/HiFi2 LM head, BF16
activation/residual and prefill CCL, BFP8 decode CCL and KV cache, BF16 logits,
and FP32 sampler gather values, with no layer exceptions.

## Measurement contract

All Pareto ranking uses full-model, batch-1, traced teacher forcing on the same
149-token non-aligned AIME24 prompt and exactly 100 reference tokens. A result
is eligible only when model trace replays equal the 99 decode tokens. Eager or
untraced decode results are excluded from ranking. TTFT is captured for every
row but is not the selection metric because same-process candidates can reuse
compiled programs.

The exact pinned tokenizer has no chat template and raises `ValueError` from
`apply_chat_template`. Per the qualitative-check contract, this stage does not
invent a wrapper. It refreshes and uses the exact-checkpoint completion-mode
reference `../full_model/readiness_aime24_plain.refpt`; the missing template is
a recorded model-artifact limitation.

## Results and Pareto decision

The closest passing alternatives and key rejections are:

| Policy | Top-1 | Top-5 | Top-100 | traced decode t/s/u | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `lm_head_bfp8_hifi2` | 92% | 100% | 100% | 24.561 | selected |
| `lm_head_bfp4_hifi2` | 90% | 100% | 100% | 24.488 | passes, slower and lower top-1 |
| `decode_ccl_bf16` | 91% | 100% | 100% | 23.988 | passes, slower |
| Stage 07 baseline | 91% | 100% | 100% | 23.139 | passes, 6.15% slower than selected |
| `attention_qkv_bfp4_lofi` | 89% | 100% | 100% | 23.062 | reject: top-1 |
| `activation_bfp8_residual_bf16` | 89% | 100% | 100% | 22.983 | reject: top-1 |
| `residual_activation_bfp8` | 1% | 1% | 5% | 24.112 | reject: accuracy collapse |
| `canonical_accuracy_bfp8_hifi2_bf16commcache` | 93% | 100% | 100% | 17.799 | passes, slowest measured row |
| `lm_head_bfp8_hifi2_decode_ccl_bf16` | 91% | 100% | 100% | 22.512 | refined combination regressed |
| `attention_output_bfp8_hifi2` | 90% | 100% | 100% | 22.464 | review-added isolated fidelity row regressed |
| `lm_head_bfp8_hifi2_attention_output_bfp4_hifi2` | 91% | 99% | 100% | 22.289 | review-added compatible combination regressed |

The selected point is the fastest passing point, not merely a preferred
accuracy/performance compromise. `top1_perf_pareto.png` and
`top5_perf_pareto.png` plot all 21 full-model policies, draw their Pareto
frontiers, mark the selection in red, and show the vertical minimum-accuracy
line. Each chart uses a decision-region zoom plus an explicit all-policy
overview so the 1% residual-activation outlier remains visible without
compressing the 89--93%/100% cluster. Passing and rejected policies use
different markers, while collision-spaced callouts identify the selected
policy, closest passing alternatives, global-frontier points, and key
failures. `sweep_results.json` and `sweep_results.csv` retain every row's
complete policy, fidelities, accuracy, TTFT, traced decode rate, regime,
command, hardware, mesh, trace counters, and pass/fail status.

Every material BFP4 matmul group considered has a matched LoFi result: the
Stage 07 MLP baseline, QKV, attention output, and LM head. QKV BFP4 fails top-1
under both LoFi and HiFi2. Attention-output and LM-head BFP4 pass, but neither
beats the selected BFP8 LM head. No material BFP4 claim relies on a missing
candidate or an untested fidelity assumption.

The independent review requested direct coverage of two compatible
attention-output refinements. Isolated BFP8-output/HiFi2 passes but falls to
22.464 t/s/u. Combining the selected BFP8/HiFi2 LM head with the passing
BFP4-output/HiFi2 policy also passes, but falls to 22.289 t/s/u. Because the
isolated BFP8-output refinement did not improve the 23.139 t/s/u baseline, its
conditional LM-head combination was not run. These direct results preserve the
selected winner without relying on a component-screening assumption.

## Runtime consumption and repairs

The precision loader propagates weight groups, compute fidelities, layer
exceptions, activation/residual, phase-specific CCL, KV cache, LM head,
logits, and sampling fields into construction. Runtime summaries record actual
constructed tensor dtypes, fixed BF16 QKV-split/cache-update boundaries, and
effective MLP geometry; they do not only echo JSON. The selected/post-selection
normal-path artifacts provide the measured physical proof.

The 19 original candidate JSONs predate physical-tensor fields in the summary,
so their aggregate `physical_weight_dtypes` values are null. Their resolved
policy summaries and source/static consumption audit remain intact. Both
review-added rows record actual prefill/decode tensor dtypes, and the selected
normal-path artifact independently records every selected physical group.

Reduced hardware smokes exposed two real packed-activation kernel contracts
and a BFP8 MLP L1 overflow. `$autofix` kept cache-update and decode-head-split
operands BF16, and selected a BFP8-only packed gate/up block width 6 whose
1,090,176-byte circular-buffer estimate leaves 482,688 bytes of L1 headroom.
BFP4 retains block width 12. A later source audit repaired separate decode
attention dtype/fidelity consumption in the single-device optimized loader;
the measured TP4 loader already consumed those fields, so no TP4 numeric row
was invalidated. Exact evidence is under `autofix_activation_dtype/`.

## Context and non-aligned support

The selected KV cache remains BFP8 and does not change page layout or chunking.
Its selected 149-token run itself is non-aligned and trace-verified. Replacing
the BF16 LM head with BFP8 saves 330,301,440 bytes/device, yielding
10,577,814,016 bytes/device of physical weights, 2,789,212,160 bytes/device of
batch-1 advertised-context KV, 27,342,513,544 total accounted bytes, and
6,883,007,096 bytes/device margin. The 262,144-token context and physical
batch-3 upper bound remain unchanged.

The isolated BF16 KV candidate also constructs at full advertised batch-1
context. The canonical BFP8-MLP/BF16-KV control exceeds the retained Stage 07
conservative reserve envelope but nevertheless constructs, traces, executes,
and closes normally at the advertised context; this proves the general reserve
is not fully concurrent with that workload. No evaluated or selected policy
requires a context capability reduction. See `context_capacity.md` and
`../context_contract.json`.

## Post-selection token-out

The benchmark command was run with no precision environment override and no
explicit precision path. Its harness-default loader resolved
`selected_precision_config.json` into the model config passed to
`build_generator`, and the runtime summary reported physical BFP8 prefill and
decode attention, physical BFP4 prefill/decode MLP, and a physical BFP8/HiFi2
LM head. On the same 149+100 workload, one warmup followed by five no-readback
samples measured median TTFT 479.707 ms, overall decode 24.787 t/s/u, and
steady decode 34.256 t/s/u. Every sample recorded 99 model trace replays, one
final sampled-token readback, and zero full-logits readbacks. This token-out
number is the serving-comparison baseline for later stages; the 24.561 t/s/u
teacher-forcing number remains the Stage 08 Pareto ranking metric.

The separate source/static construction tests prove that callers invoking
`build_generator` with neither a model config nor precision path also resolve
the same selected artifact by default.

## Qualitative check

The selected normal path generated six 64-token completion-mode controls. The
scoped degeneracy checker exits zero, two prompts match HF token-for-token, and
manual review finds no TT-only loop, wrong language, incoherence, or runtime
fallback signature. Four of six TT continuations are identical to the passing
Stage 07 controls; the two changed trajectories remain coherent. Repetition on
the supervised/unsupervised prompt also occurs identically in HF. The recorded
limitation is prompt format: the exact tokenizer has no chat template, so
instruction-like inputs behave as base-model corpus autocomplete. See
`qualitative/verdict.md` for the prompt-by-prompt assessment.

## Reproduction commands

The baseline/full candidate command family is:

```text
env LD_LIBRARY_PATH=$PWD/build/lib MPLCONFIGDIR=/tmp/mplconfig timeout 10800 \
  python -m models.autoports.google_gemma_4_31b.tests.run_datatype_same_weights_group \
  --model-dir /home/odjuricic/.cache/huggingface/hub/models--google--gemma-4-31B/snapshots/d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3 \
  --precision-config <config.json> \
  --reference models/autoports/google_gemma_4_31b/doc/full_model/readiness_aime24_plain.refpt \
  --output-dir models/autoports/google_gemma_4_31b/doc/datatype_sweep/candidates \
  --tensor-cache /tmp/gemma4_31b_full_model_tensor_cache
```

Aggregate tables and plots:

```text
env MPLCONFIGDIR=/tmp/mplconfig python -m \
  models.autoports.google_gemma_4_31b.tests.generate_datatype_sweep_artifacts \
  --candidates models/autoports/google_gemma_4_31b/doc/datatype_sweep/candidates \
  --output-dir models/autoports/google_gemma_4_31b/doc/datatype_sweep \
  --selected lm_head_bfp8_hifi2
```

The exact post-selection token-out, qualitative, static-test, review, and
commit commands and results are recorded in `work_log.md`.

## Artifacts

- `selected_precision_config.json`: normal default precision policy.
- `sweep_results.json`, `sweep_results.csv`: complete 21-policy result set.
- `top1_perf_pareto.png`, `top5_perf_pareto.png`: pyplot Pareto charts.
- `candidates/`: per-policy full-model JSON and serialized raw logs.
- `configs/`: inherited candidate inputs used during exploration, including the
  documented conditional BFP8-output combination that was not triggered.
- `smokes/`, `autofix_activation_dtype/`: reduced failures and repair proof.
- `context_capacity.md`, `../context_contract.json`: capacity recomputation.
- `post_selection_token_out.json`: warmed normal-path token-out measurement.
- `qualitative/`, `qualitative.log`: prompt-format and output review evidence.
- `stage_review.md`: independent Stage 08 review verdict.
- `work_log.md`: chronological commands, limitations, results, and commit SHAs.
